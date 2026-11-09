# 수정(2026-10-15): 교사 VLM ② LoRA SFT — VAL split 채점(탐욕 생성 → 파서 → 지표) + 지름길 정량화 대조군.
"""교사 VLM 평가 (val 전용 — 학습·체크포인트 선택에 쓰지 않음).

모드(--modes, 쉼표):
  full       : 6 영상 + 명령 + 자차상태(학습과 같은 프롬프트)
  text_only  : 영상 제거(DriveBench 식 대조군) — 같은 모델, 같은 텍스트. full 과의 차이 = 영상이 기여한 몫
  no_cmd / no_ego / no_cmd_ego : 해당 프롬프트 줄 제거(입력 강건성 탐침; 학습 dropout 과 같은 조립 규칙)
기준선(모델 불필요, 항상 계산):
  copy       : 의도 = (keep speed, 명령→횡방향), 행동 = 현재 속도 직진 등속(레코드에 copy_baseline 있으면 그것)
  codebook_oracle : GT action 토큰 복호 궤적(코드북 양자화 하한)
지표: 인지(객체 재현율/정밀도 — 같은 클래스·방위·거리 허용오차 일대일 매칭, 거리·속도 MAE, 깊이순서 정확도),
      지도 줄(교차보도 상태·개수), 의도(unavailable 제외), 행동(L2 1/2/3 s, 충돌 — tools/eval_nusc_planning.py 와 같은 식,
      future_complete 표본만; val 5,119 기대).
산출: <out>/gen_<mode>.jsonl(원문 생성, 재개 가능), <out>/metrics.json, <out>/summary.md
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_common import (CODEBOOK, NUM_CAMS, REPO, build_messages, build_user_text, future_complete,  # noqa: E402
                        load_images, load_meta, load_records, make_processor, require_gpu_env, resolve_data_dir,
                        system_text)
import answer_grammar as AG  # noqa: E402  # 수정(2026-10-16): 설계 검토 반영
from teacher_metrics import (IntentAcc, MapAcc, PerceptionAcc, PlanAcc, copy_baseline, fut_boxes,  # noqa: E402
                             load_planning_eval, select_parser)

MODE_DROPS = {"full": (False, False, False), "text_only": (False, False, True), "no_cmd": (True, False, False),
              "no_ego": (False, True, False), "no_cmd_ego": (True, True, False)}


def gen_mode(model, proc, recs, mode, meta, out_file, batch, max_new_tokens, device, grammar=None):
    import torch
    from transformers import LogitsProcessorList  # 수정(2026-10-16): 설계 검토 반영
    cname = grammar.info["name"] if grammar is not None else None
    done = set()
    if os.path.exists(out_file):
        for line in open(out_file):
            e = json.loads(line)
            if e.get("constrained") != cname:   # 수정(2026-10-16): 설계 검토 반영 — 제약/비제약 생성을 한 파일에 섞지 않는다
                raise SystemExit(f"{out_file}: 기존 줄 constrained={e.get('constrained')} != 이번 {cname} — 다른 출력 폴더 사용")
            done.add(e["token"])
    todo = [r for r in recs if r["token"] not in done]
    print(f"[gen:{mode}] {len(todo)} 남음 (완료 {len(done)})", flush=True)
    dc, de, text_only = MODE_DROPS[mode]
    tok = proc.tokenizer
    tok.padding_side = "left"
    f = open(out_file, "a")
    t0 = time.time()
    for s in range(0, len(todo), batch):
        rs = todo[s:s + batch]
        texts, imgs = [], []
        for r in rs:
            user = build_user_text(r, dc, de)
            use_prev = bool(meta.get("use_prev", False))
            n_img = 0 if text_only else NUM_CAMS * (2 if use_prev else 1)
            msgs = build_messages(system_text(r, meta["system"]), user, n_img)
            texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            if not text_only:
                imgs.extend(load_images(r, use_prev, meta.get("frame_order", "prev_first")))
        enc = proc(text=texts, images=imgs if imgs else None, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        if "pixel_values" in enc:
            enc["pixel_values"] = enc["pixel_values"].to(model.dtype)
        lp = None
        if grammar is not None:   # 수정(2026-10-16): 설계 검토 반영 — 문법 제약 생성(캐시 추출·GRPO 와 같은 제약)
            lp = LogitsProcessorList([AG.make_logits_processor(grammar, enc["input_ids"].shape[1], len(rs))])
        with torch.no_grad():
            o = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens, logits_processor=lp,
                               pad_token_id=tok.pad_token_id, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
        new = o[:, enc["input_ids"].shape[1]:]
        for r, ids in zip(rs, new):
            txt = tok.decode(ids, skip_special_tokens=False)
            hit_end = "<|im_end|>" in txt
            txt = txt.split("<|im_end|>")[0].replace("<|endoftext|>", "")
            f.write(json.dumps(dict(token=r["token"], text=txt, ended=hit_end, constrained=cname), ensure_ascii=False) + "\n")
        f.flush()
        n = s + len(rs)
        el = time.time() - t0
        print(f"[gen:{mode}] {n}/{len(todo)}  {el / n:.2f}s/표본  ETA {(len(todo) - n) * el / n / 60:.1f}분", flush=True)
    f.close()


def load_gen(path):
    return {json.loads(l)["token"]: json.loads(l) for l in open(path)}


def get_fut(cache, infos, infos_index, token, rule):
    k = ("fut", token)
    if k not in cache:
        fut = fut_boxes(infos, infos_index[token], rule)
        if len(fut) != 6:
            raise RuntimeError(f"미래 박스 {len(fut)} != 6 (future_complete 표본) {token}")
        cache[k] = fut
    return cache[k]


def score(recs, gens, parser, infos_index, infos, pe, codebook, args, gt_col_cache):
    perc, mp, it, pl, pl_all = PerceptionAcc(args.dist_tol_abs, args.dist_tol_rel), MapAcc(), IntentAcc(), PlanAcc(pe), PlanAcc(pe)
    n_fmt = n_end = 0
    for r in recs:
        g = gens.get(r["token"])
        if g is None:
            raise KeyError(f"생성 결과 없음: {r['token']} (생성이 끝나지 않았다)")
        n_end += int(g.get("ended", True))
        try:
            p = parser(g["text"]); n_fmt += 1
        except Exception:
            p = None              # 형식 오류 = 그 표본의 모든 항목 오답/누락으로 채점(아래), 비율은 format_valid 로 보고
        t = r["targets"]
        perc.add(p["objects"] if p else [], t["objects"], (p or {}).get("ego_speed"), t.get("ego_speed"))
        mp.add(p["map"] if p else None, t.get("map"))
        if future_complete(r):
            gi = t.get("intent")
            gi = None if (gi is None or gi == "unavailable") else dict(longitudinal=gi["longitudinal"], lateral=gi["lateral"])
            unc = bool(r.get("intent_uncertain", (t.get("intent") or {}).get("uncertain", False)))
            it.add(p["intent"] if p else None, gi, unc)
            traj = None
            if p and p.get("action") and all(0 <= k < codebook.K for k in p["action"]):
                traj = codebook.decode(p["action"])
            fut = get_fut(gt_col_cache, infos, infos_index, r["token"], args.fut_box_rule)
            key = ("gtcol", r["token"])
            gc = pl.add(traj, t["fut_traj"], fut, gt_col_cache.get(key))
            if gc is not None:
                gt_col_cache[key] = gc
            # 전 표본 판(형식 오류 행동 = 정지 궤적(원점 6점)으로 벌점)
            pl_all.add(traj if traj is not None else [[0.0, 0.0]] * 6, t["fut_traj"], fut, gt_col_cache.get(key))
    n = len(recs)
    return dict(n=n, format_valid=n_fmt / max(n, 1), ended_with_im_end=n_end / max(n, 1),
                perception=perc.result(), map=mp.result(), intent=it.result(),
                planning_valid_only=pl.result(), planning_all_invalid_as_stop=pl_all.result())


def score_baselines(recs, infos_index, infos, pe, codebook, args, gt_col_cache):
    it_c, pl_c, pl_o = IntentAcc(), PlanAcc(pe), PlanAcc(pe)
    src = {}
    for r in recs:
        if not future_complete(r):
            continue
        t = r["targets"]
        ci, ctraj, s = copy_baseline(r, codebook)
        src[s] = src.get(s, 0) + 1
        gi = dict(longitudinal=t["intent"]["longitudinal"], lateral=t["intent"]["lateral"])
        unc = bool(r.get("intent_uncertain", t["intent"].get("uncertain", False)))
        it_c.add(ci, gi, unc)
        fut = get_fut(gt_col_cache, infos, infos_index, r["token"], args.fut_box_rule)
        key = ("gtcol", r["token"])
        gt_col_cache[key] = pl_c.add(ctraj, t["fut_traj"], fut, gt_col_cache.get(key))
        pl_o.add(codebook.decode(t["action_tokens"]), t["fut_traj"], fut, gt_col_cache[key])
    return dict(copy=dict(intent=it_c.result(), planning=pl_c.result(), source=src),
                codebook_oracle=dict(planning=pl_o.result()))


def check_gt_alignment(recs, infos, infos_index):
    """targets.fut_traj == infos gt_ego_fut_trajs 누적합 (정답 파일과 평가 pkl 정합 검사)."""
    import numpy as np
    bad = 0
    for r in recs:
        if not future_complete(r):
            continue
        i = infos[infos_index[r["token"]]]
        g = np.asarray(i["gt_ego_fut_trajs"], np.float64).cumsum(0)[:6, :2]
        if not np.allclose(g, np.asarray(r["targets"]["fut_traj"]), atol=1e-3):
            bad += 1
    if bad:
        raise RuntimeError(f"정답 fut_traj 와 {len(recs)} 중 {bad} 표본이 infos 와 불일치 — 평가 pkl 확인")


def fmt(x, p=3, pct=False):
    if x is None:
        return "-"
    return f"{100 * x:.{max(p - 2, 1)}f}%" if pct else f"{x:.{p}f}"


def write_md(res, path):
    L = ["# 교사 VLM 평가 (val)", "", f"모델: `{res['model']}`  ·  파서: `{res['parser']}`  ·  표본 {res['n_records']}"
         f" (future_complete {res['n_future_complete']})", ""]
    modes = [m for m in res["modes"]]
    L += ["## 인지·지도", "", "| 모드 | 형식유효 | 재현율 | 정밀도 | 거리 MAE(m) | 속도 MAE(m/s) | 깊이순서 | 지도 줄 정확 | 교차보도 상태 | divider | boundary |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in modes:
        r = res["modes"][m]; p, mp = r["perception"], r["map"]
        L.append(f"| {m} | {fmt(r['format_valid'], pct=True)} | {fmt(p['recall'], pct=True)} | {fmt(p['precision'], pct=True)} | "
                 f"{fmt(p['dist_mae'], 2)} | {fmt(p['speed_mae'], 2)} | {fmt(p['depth_order_acc'], pct=True)} | "
                 f"{fmt(mp['line_exact'], pct=True)} | {fmt(mp['ped_state_acc'], pct=True)} | {fmt(mp['divider_acc'], pct=True)} | "
                 f"{fmt(mp['boundary_acc'], pct=True)} |")
    L += ["", "## 의도(unavailable 제외)·행동(future_complete, SparseDrive 식)", "",
          "| 모드 | 종방향 acc | 횡방향 acc | 결합 acc | 횡 macro-F1 | L2 1s | L2 2s | L2 3s | L2 avg | Col 1s | Col 2s | Col 3s | Col avg | 유효 행동 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(name, it, pl):
        l2, c = pl.get("L2", {}), pl.get("obj_box_col", {})
        return (f"| {name} | {fmt(it.get('lon_acc'), pct=True)} | {fmt(it.get('lat_acc'), pct=True)} | {fmt(it.get('joint_acc'), pct=True)} | "
                f"{fmt(it.get('lat_macro_f1'))} | {fmt(l2.get('1s'))} | {fmt(l2.get('2s'))} | {fmt(l2.get('3s'))} | {fmt(l2.get('avg'))} | "
                f"{fmt(c.get('1s'), pct=True)} | {fmt(c.get('2s'), pct=True)} | {fmt(c.get('3s'), pct=True)} | {fmt(c.get('avg'), pct=True)} | "
                f"{pl.get('n_eval', 0)} |")
    for m in modes:
        r = res["modes"][m]
        L.append(row(m, r["intent"], r["planning_valid_only"]))
    b = res["baselines"]
    L.append(row("copy(명령+등속)", b["copy"]["intent"], b["copy"]["planning"]))
    L.append(row("codebook oracle", {}, b["codebook_oracle"]["planning"]))
    L += ["", "HiP-AD Table 3 (참고, 같은 계획 프로토콜): L2 0.28/0.53/0.87 avg 0.56 · Col 0.01/0.05/0.15 avg 0.07 (%)", "",
          "- 지름길 판독: text_only·copy 가 full 에 가까우면 의도/행동이 영상이 아닌 명령·자차상태에서 온다는 뜻.",
          "- planning_all_invalid_as_stop(형식 오류 행동을 정지 궤적으로 채점)은 metrics.json 참조."]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=False, help="병합 모델 디렉터리(merge_teacher_lora.py 산출). --score_only 면 생략 가능")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--infos", default="data/infos/nuscenes_infos_val.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="full,text_only")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=768)   # 수정(2026-10-15): v2-sensor 정답 547(+종료) 토큰
    ap.add_argument("--parser", default=None, help="module:function (기본 자동 선택)")
    ap.add_argument("--dist_tol_abs", type=float, default=2.0)
    ap.add_argument("--dist_tol_rel", type=float, default=0.1)
    ap.add_argument("--fut_box_rule", choices=["dataset_default", "num_lidar_pts"], default="dataset_default")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--grammar", choices=["on", "off"], default="on",  # 수정(2026-10-16): 설계 검토 반영
                    help="on(기본) = 문법 제약 생성(answer_grammar.py; 캐시 추출·GRPO 와 같은 분포). off = 제약 없음(v1 형식·진단 전용)")
    ap.add_argument("--max_samples", type=int, default=0, help="스모크 전용(0 = 전체 val)")
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    if args.split != "val":
        raise SystemExit("평가는 val 전용(train hold-out 은 train_sft.py 가 체크포인트 선택에 사용)")

    out = Path(args.out) if os.path.isabs(args.out) else REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    meta = load_meta(args.model) if args.model else json.load(open(out / "teacher_meta.json"))
    data_dir = resolve_data_dir(args.data_dir or meta.get("data_dir"))
    recs = load_records(data_dir, "val", max_n=args.max_samples or None)
    modes = [m for m in args.modes.split(",") if m]
    for m in modes:
        if m not in MODE_DROPS:
            raise SystemExit(f"모드 {m} 미지원: {list(MODE_DROPS)}")

    if not args.score_only:
        dev = require_gpu_env(allow_cpu=(args.device == "cpu"))
        import torch
        from transformers import Qwen2VLForConditionalGeneration
        json.dump(meta, open(out / "teacher_meta.json", "w"), indent=1)
        dt = torch.bfloat16 if meta.get("merged_dtype", "bf16") == "bf16" else torch.float32
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=dt, attn_implementation=args.attn)
        model.to("cpu" if dev == "cpu" else "cuda").eval()
        if dev != "cpu" and args.attn == "sdpa":  # 수정(2026-10-16): SFT 발산 원인 진단 — fp32 이면 정확한 math 커널(학습과 같은 수치)
            from sft_common import exact_fp32_attention
            print(f"[attn] {exact_fp32_attention(dt)}", flush=True)
        if meta.get("rope", "mrope") != "mrope":
            print("[주의] 학습 rope=1d — generate() 는 M-RoPE 를 쓴다(학습과 위치 불일치). 결과 해석 주의.", flush=True)
        proc = make_processor(args.model, meta["pixels"])
        grammar = None
        if args.grammar == "on":   # 수정(2026-10-16): 설계 검토 반영
            grammar = AG.load_grammar(proc.tokenizer, data_dir=data_dir)
            if args.max_new_tokens < grammar.n_pos:
                raise SystemExit(f"--max_new_tokens {args.max_new_tokens} < 문법 고정 길이 {grammar.n_pos}")
            print(f"[grammar] {grammar.info}", flush=True)
        for m in modes:
            gen_mode(model, proc, recs, m, meta, str(out / f"gen_{m}.jsonl"), args.batch, args.max_new_tokens,
                     model.device, grammar)
        del model

    pname, parser, rates = select_parser(args.parser, recs)
    print(f"[parser] {pname} (후보 GT 파싱률 {rates})", flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import ActionCodebook
    codebook = ActionCodebook(str(REPO / CODEBOOK))
    infos = pickle.load(open(REPO / args.infos if not os.path.isabs(args.infos) else args.infos, "rb"))["infos"]
    infos_index = {i["token"]: k for k, i in enumerate(infos)}
    check_gt_alignment(recs, infos, infos_index)
    pe = load_planning_eval()
    cache = {}
    res = dict(model=args.model, parser=pname, data_dir=str(data_dir), n_records=len(recs),
               n_future_complete=sum(future_complete(r) for r in recs), modes={},
               fut_box_rule=args.fut_box_rule, dist_tol=[args.dist_tol_abs, args.dist_tol_rel],
               constrained_decoding=(AG.CONSTRAINT_NAME if args.grammar == "on" else None))  # 수정(2026-10-16): 설계 검토 반영
    for m in modes:
        res["modes"][m] = score(recs, load_gen(out / f"gen_{m}.jsonl"), parser, infos_index, infos, pe, codebook, args, cache)
    res["baselines"] = score_baselines(recs, infos_index, infos, pe, codebook, args, cache)
    json.dump(res, open(out / "metrics.json", "w"), indent=1, ensure_ascii=False)
    write_md(res, out / "summary.md")
    print(open(out / "summary.md").read(), flush=True)


if __name__ == "__main__":
    main()
