# 수정(2026-10-20): 사용자 승인 — 설명가능성 4가지(학습 변경 없음)
"""추론 시 교사 토큰 가리기(반사실) — "이 정보를 빼면 계획이 어떻게 바뀌나". GPU1 추론, 같은 체크포인트, 학습 변경 없음.

조건(표본마다, 같은 시간 뱅크 상태에서 다시 추론 — 개입은 이 프레임 하나에만):
  base         실제 스트림 추론(평가와 같음). 이 실행의 뱅크 갱신만 다음 프레임으로 이어진다.
  noop         base 직전 상태로 되돌려 가리지 않고 다시 추론 → GPU 비결정(deformable atomicAdd) + 상태 복원 오차의 잡음 바닥
  noop_pad     모두 유효인 key padding mask 를 넘겨 flash unpad 경로로 추론 → 가리기 경로 자체의 차이(경로 동등성)
  gate_off     언어 항 제거(A=0 → 출력 = query, 모든 계층·모든 쿼리) — '언어 없음' 반사실
  mask_<폴더>  폴더 7 개 각각 key padding mask 로 가림(모든 계층·모든 쿼리 공통)
  mask_ctx_img / mask_ctx_text   context 의 영상 1,080 토큰 / 지시문·명령·자차 상태 줄 토큰만 가림
  mask_slot_top1 / mask_slot_top3 base 에서 선택 계획 쿼리가 가장 많이 본(계층 평균) **채워진** 객체 칸 1·3 개의 57 토큰을 가림
                 (교사가 객체를 하나도 적지 않은 표본은 해당 없음으로 NaN 기록·개수 보고)
가리기 = 모델이 실제 쓰는 경로(flash, key_padding_mask; 유효 = True). --impl explicit 이면 모든 조건을 fp32 명시적 계산으로.
지표(조건별): 최종 계획(plan_temp_2hz)의 GT 대비 L2(시점별)·충돌(평가기 tools/eval_nusc_planning.py 규칙: 미래 GT 상자, GT 궤적 충돌 시점 제외,
  미래 GT 불완전 표본은 NaN), 선택 모드 번호·바뀜 여부, base 계획 대비 변위(6 점 평균·3 s 점), 모드 확률 총변동거리(TV).
출력(--out): manifest.json, shard_XXXX.npz, shard_XXXX.state.pt(실제 스트림 뱅크 — 재개), summary.json/md(--summarize)

계획 점수기(Hydra-MDP 식, tools/plan_scorer) 결과가 있으면 '왜 이 후보인가' 패널용 후보별 예측 하위 점수를 내보낸다:
  python tools/xai/xai_counterfactual.py scorer --val_dump <dump_dir> --scorer <scorer.pt> --out <file.npz>
  (후보 16 개마다 imitation softmax·no-collision·drivable·progress 시그모이드·선택 점수·선택 번호; 원 모델 확률·선택도 함께)

사용(GPU1 전용, GPU0 금지):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. OPENBLAS_CORETYPE=SkylakeX python tools/xai/xai_counterfactual.py run \
      configs/nuscenes/v2_teacher/stage2/test.py results/hipad_nusc_laq_v2_teacher_stage2/latest.pth --out results/<run>_xai/counterfactual
  스모크: ... --smoke_samples 3 --max_gpu_mib 12000 --need_mib 12000 · 요약만: summarize --out <dir>
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xai_common as X  # noqa: E402

FOLDER_CONDS = [f"mask_{f}" for f in X.FOLDERS]
DEFAULT_CONDS = ["base", "noop", "noop_pad", "gate_off"] + FOLDER_CONDS + ["mask_ctx_img", "mask_ctx_text", "mask_slot_top1", "mask_slot_top3"]
PER_SAMPLE = ("idx", "token", "scene_token", "cmd", "complete", "n_teacher_obj", "slot_rank", "base_modes", "gt_traj",
              "plan", "sel", "probs", "l2", "col", "disp_mean", "disp_3s", "tv", "det_dscore", "map_dscore", "applicable")
TOPK_SCORE = 100   # 인지 쪽 영향(참고): 검출·지도 상위 100 점수(내림차순 정렬)의 평균 절대 변화


def parse():
    from mmcv import DictAction
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("config"); r.add_argument("checkpoint")
    r.add_argument("--out", required=True)
    r.add_argument("--conditions", nargs="+", default=DEFAULT_CONDS)
    r.add_argument("--impl", choices=["flash", "explicit"], default="flash")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--scenes_per_shard", type=int, default=25)
    r.add_argument("--smoke_samples", type=int, default=0)
    r.add_argument("--max_gpu_mib", type=int, default=0)
    r.add_argument("--need_mib", type=int, default=12000)
    r.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    s = sp.add_parser("summarize"); s.add_argument("--out", required=True)
    c = sp.add_parser("scorer")
    c.add_argument("--val_dump", required=True); c.add_argument("--scorer", required=True); c.add_argument("--out", required=True)
    return ap.parse_args()


def cond_setup(hook, layout, name, slot_rank, n_obj):
    """hook 설정. 반환 False = 이 표본에 해당 없음(객체 칸 조건인데 채워진 칸 없음)."""
    hook.keep, hook.force_pad, hook.zero_lang = None, False, False
    if name in ("base", "noop"):
        return True
    if name == "noop_pad":
        hook.force_pad = True
        return True
    if name == "gate_off":
        hook.zero_lang = True
        return True
    if name.startswith("mask_slot_top"):
        k = int(name[len("mask_slot_top"):])
        filled = [s for s in slot_rank if s < n_obj]
        if not filled:
            return False
        hook.keep = layout.keep_mask([f"slot{s}" for s in filled[:k]])
        return True
    if name == "mask_ctx_img":
        hook.keep = layout.keep_mask([f"ctx_img_{c}" for c in X.CAM_SHORT])
        return True
    if name == "mask_ctx_text":
        hook.keep = layout.keep_mask(["ctx_text"])
        return True
    if name.startswith("mask_") and name[5:] in X.FOLDERS:
        hook.keep = layout.keep_mask([name[5:]])
        return True
    raise ValueError(f"알 수 없는 조건 {name}")


def run(a):
    X.require_gpu1(a.need_mib)
    import torch
    from mmdet.datasets import build_dataloader
    from torch.utils.data import Subset
    import scorer_io
    import common_v2 as V
    if a.conditions[0] != "base" or len(set(a.conditions)) != len(a.conditions):
        raise SystemExit("조건 목록은 'base' 로 시작하고 중복이 없어야 한다")
    if a.max_gpu_mib > 0:
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gpu_mib * 2 ** 20 / tot), 0)
    cfg = X.load_cfg(a.config, a.cfg_options)
    ds, ds_cfg = X.build_test_dataset(cfg)
    infos = ds.data_infos
    shards = X.scene_shards(infos, a.scenes_per_shard)
    if a.smoke_samples > 0:
        n_sm = min(a.smoke_samples, len(infos))
        shards = [(s0, min(s1, n_sm)) for s0, s1 in shards if s0 < n_sm]
    man = dict(config=a.config, checkpoint=scorer_io.file_sig(a.checkpoint), ann_file=ds_cfg.ann_file, impl=a.impl,
               conditions=a.conditions, num_samples=shards[-1][1], shards=[list(s) for s in shards], smoke=bool(a.smoke_samples),
               cfg_options=dict(a.cfg_options or {}), created_by=X.TAG,
               collision_rule="tools/eval_nusc_planning.py single_coll (ego box vs future GT boxes, GT-colliding steps excluded)")
    X.check_manifest(a.out, man, ("config", "checkpoint", "ann_file", "impl", "conditions", "num_samples", "shards", "smoke", "cfg_options"))
    done = [os.path.exists(os.path.join(a.out, f"shard_{k:04d}.npz")) for k in range(len(shards))]
    if all(done):
        print(f"[완료] {a.out}")
        return
    first = done.index(False)
    if any(done[first:]):
        raise SystemExit(f"샤드 {first} 이후에 이미 있는 샤드가 있다 — 순차 가정 위반")

    model, det = X.build_model(cfg, a.checkpoint)
    dec = det.head.onedecoder_head
    qta = dec.qwen_token_attn
    if qta is None:
        raise SystemExit("이 config 에는 교사 토큰 교차 어텐션이 없다")
    bank = X.BankState(det)
    rng = X.decoder_rows(dec)
    C, M = dec.ego_fut_cmd, dec.ego_fut_mode
    hook = X.XaiHook(qta, list(range(*rng["plan"])), [rng["ego"][0]], None, None)
    hook.impl = a.impl
    hook.install()
    pe = X.planning_eval_module()
    idx_val = X.load_val_index()
    if first > 0:
        bank.load(torch.load(os.path.join(a.out, f"shard_{first - 1:04d}.state.pt"), map_location="cuda:0", weights_only=False))
        print(f"[재개] 샤드 {first} 부터")
    K = len(a.conditions)
    layout = None
    t0, n_done, t_fwd, n_fwd = time.time(), 0, 0.0, 0
    for k in range(first, len(shards)):
        s0, s1 = shards[k]
        loader = build_dataloader(Subset(ds, list(range(s0, s1))), samples_per_gpu=1, workers_per_gpu=a.workers, dist=False, shuffle=False)
        rec = {key: [] for key in PER_SAMPLE}
        for j, data in enumerate(loader):
            i = s0 + j
            info = infos[i]
            qt = X.qwen_type_from_data(data)
            if layout is None:
                layout = X.TokenLayout(qt)
                hook.seg_mat = layout.matrix(torch, "cuda")
            elif not np.array_equal(qt, layout.qwen_type):
                raise RuntimeError(f"색인 {i}: 토큰 유형 배치가 첫 표본과 다르다")
            cmd = int(np.argmax(np.asarray(info["gt_ego_fut_cmd"])))
            e = idx_val.get(info["token"])
            if e is None:
                raise RuntimeError(f"{info['token']}: 교사 색인 없음")
            n_obj = len(V.parse_answer(e["gen_text"])["objects"])
            pre = bank.save()
            outs, appl, post, slot_rank = [], [], None, None
            for c, name in enumerate(a.conditions):
                if c > 0:
                    bank.load(pre)
                ok = cond_setup(hook, layout, name, slot_rank, n_obj)
                hook.record = (name == "base")
                hook.clear()
                if not ok:
                    outs.append(None); appl.append(False)
                    continue
                torch.cuda.synchronize(); ts = time.time()
                with torch.no_grad():
                    r = model(return_loss=False, rescale=True, **data)[0]["img_bbox"]
                torch.cuda.synchronize(); t_fwd += time.time() - ts; n_fwd += 1
                outs.append(dict(plan=np.asarray(r["plan_temp_2hz"], np.float32)[:6, :2], probs=np.asarray(r["plan_mode_probs"], np.float32),
                                 modes=np.asarray(r["plan_modes"], np.float32),
                                 det_s=np.sort(np.asarray(r["scores_3d"], np.float32))[::-1][:TOPK_SCORE],
                                 map_s=np.sort(np.asarray(r["scores"], np.float32))[::-1][:TOPK_SCORE]))
                appl.append(True)
                if name == "base":
                    post = bank.save()
                    sel_b = int(np.argmax(outs[0]["probs"]))
                    nl = len(hook.rec["layers"])
                    sl = np.stack([hook.rec["agg"][l].numpy() for l in range(nl)]).mean(0)[cmd * M + sel_b]  # (n_seg,) 계층 평균
                    slot_rank = list(np.argsort([-sl[layout.names.index(f"slot{s}")] for s in range(X.N_SLOTS)]))
            bank.load(post)   # 다음 프레임은 실제 스트림(base) 상태에서
            hook.keep, hook.force_pad, hook.zero_lang, hook.record = None, False, False, False
            gt = np.asarray(info["gt_ego_fut_trajs"], np.float32).cumsum(0)[:6, :2]
            gm = np.asarray(info["gt_ego_fut_masks"], np.float32)[:6]
            fut = ds._get_fut_boxes(i, info)
            complete = bool((gm == 1).all()) and len(fut) == 6
            base = outs[0]
            P = np.full((K, 6, 2), np.nan, np.float32); SEL = np.full(K, -1, np.int64); PR = np.full((K, M), np.nan, np.float32)
            L2 = np.full((K, 6), np.nan, np.float32); COL = np.full((K, 6), np.nan, np.float32)
            DM = np.full(K, np.nan, np.float32); D3 = np.full(K, np.nan, np.float32); TV = np.full(K, np.nan, np.float32)
            DS = np.full(K, np.nan, np.float32); MS = np.full(K, np.nan, np.float32)
            gcol = None
            for c, o in enumerate(outs):
                if o is None:
                    continue
                P[c] = o["plan"]; SEL[c] = int(np.argmax(o["probs"])); PR[c] = o["probs"]
                d = np.linalg.norm(o["plan"] - base["plan"], axis=-1)
                DM[c], D3[c] = d.mean(), d[-1]
                TV[c] = 0.5 * np.abs(o["probs"] - base["probs"]).sum()
                DS[c] = np.abs(o["det_s"] - base["det_s"]).mean(); MS[c] = np.abs(o["map_s"] - base["map_s"]).mean()
                if complete:
                    l2, col, gcol = X.plan_metrics(pe, o["plan"], gt, gm, fut, gcol)
                    L2[c], COL[c] = l2, col.astype(np.float32)
            rec["idx"].append(i); rec["token"].append(info["token"]); rec["scene_token"].append(info["scene_token"])
            rec["cmd"].append(cmd); rec["complete"].append(complete); rec["n_teacher_obj"].append(n_obj)
            rec["slot_rank"].append(np.asarray(slot_rank, np.int64)); rec["base_modes"].append(base["modes"]); rec["gt_traj"].append(gt)
            rec["plan"].append(P); rec["sel"].append(SEL); rec["probs"].append(PR); rec["l2"].append(L2); rec["col"].append(COL)
            rec["disp_mean"].append(DM); rec["disp_3s"].append(D3); rec["tv"].append(TV); rec["det_dscore"].append(DS); rec["map_dscore"].append(MS); rec["applicable"].append(np.asarray(appl))
            n_done += 1
        out = {key: np.asarray(v) for key, v in rec.items()}
        for key in ("token", "scene_token"):
            out[key] = out[key].astype("U64")
        X.atomic_savez(os.path.join(a.out, f"shard_{k:04d}.npz"), **out)
        sp_ = os.path.join(a.out, f"shard_{k:04d}.state.pt")
        torch.save(bank.save(), sp_ + ".tmp"); os.replace(sp_ + ".tmp", sp_)
        el = time.time() - t0
        print(f"[샤드 {k + 1}/{len(shards)}] 색인 {s0}..{s1 - 1} | 누적 {n_done} 표본 {el:.0f}s ({el / max(n_done, 1):.2f}s/표본, "
              f"순수 추론 {t_fwd / max(n_fwd, 1):.3f}s/회 × {K} 조건) | GPU 최대 {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB", flush=True)
    hook.uninstall()
    m = json.load(open(os.path.join(a.out, "manifest.json")))
    m["timing"] = dict(sec_per_forward=t_fwd / max(n_fwd, 1), sec_per_sample_wall=(time.time() - t0) / max(n_done, 1), n_samples=n_done)
    json.dump(m, open(os.path.join(a.out, "manifest.json"), "w"), indent=1, ensure_ascii=False)
    summarize(a.out)


def load_shards(out):
    man = json.load(open(os.path.join(out, "manifest.json")))
    parts = {}
    for k in range(len(man["shards"])):
        p = os.path.join(out, f"shard_{k:04d}.npz")
        if not os.path.exists(p):
            raise RuntimeError(f"{p} 없음 — 추론을 끝까지 재개한 뒤 요약한다")
        z = np.load(p)
        for key in z.files:
            parts.setdefault(key, []).append(z[key])
    return man, {k: np.concatenate(v, 0) for k, v in parts.items()}


def summarize(out):
    man, D = load_shards(out)
    conds = man["conditions"]
    n = len(D["idx"])
    comp = D["complete"].astype(bool)
    res = dict(n=n, n_future_complete=int(comp.sum()), smoke=man["smoke"], checkpoint=man["checkpoint"], conditions={},
               note="L2·충돌은 평가기와 같은 누적 평균(1s/2s/3s = 2/4/6 번째 점까지 평균) — 미래 완비 표본만")
    def cum(v):  # (n, 6) → 1s/2s/3s/avg 누적 평균(평가기와 같음)
        m = np.nanmean(v, 0)
        c = [float(m[:j + 1].mean()) for j in range(6)]
        return {"1s": c[1], "2s": c[3], "3s": c[5], "avg": (c[1] + c[3] + c[5]) / 3}
    for c, name in enumerate(conds):
        ap_ = D["applicable"][:, c].astype(bool)
        mk = comp & ap_
        r = dict(n_applicable=int(ap_.sum()))
        if mk.any():
            r["L2"] = cum(D["l2"][mk, c]); r["obj_box_col"] = cum(D["col"][mk, c])
            r["dL2_vs_base_avg"] = r["L2"]["avg"] - cum(D["l2"][mk, 0])["avg"]
        if ap_.any():
            r["sel_changed_rate"] = float((D["sel"][ap_, c] != D["sel"][ap_, 0]).mean())
            r["disp_mean_m"] = float(np.nanmean(D["disp_mean"][ap_, c])); r["disp_3s_m"] = float(np.nanmean(D["disp_3s"][ap_, c]))
            r["disp_3s_p90_m"] = float(np.nanpercentile(D["disp_3s"][ap_, c], 90))
            r["prob_tv_mean"] = float(np.nanmean(D["tv"][ap_, c]))
            r["det_score_change"] = float(np.nanmean(D["det_dscore"][ap_, c])); r["map_score_change"] = float(np.nanmean(D["map_dscore"][ap_, c]))
        res["conditions"][name] = r
    if "noop" in res["conditions"]:
        res["noise_floor"] = {k: res["conditions"]["noop"].get(k) for k in ("disp_mean_m", "disp_3s_m", "sel_changed_rate", "prob_tv_mean",
                                                                             "det_score_change", "map_score_change")}
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    f = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
    L = [f"# 교사 토큰 가리기 반사실 요약 (표본 {n}, 미래 완비 {int(comp.sum())}{', 스모크' if man['smoke'] else ''})", "",
         "| 조건 | 적용 | L2 avg | ΔL2 vs base | Col avg | 선택 모드 바뀜 | 변위 평균(m) | 변위 3s(m) | 확률 TV | 검출 점수 변화 | 지도 점수 변화 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res["conditions"].items():
        col = "-" if "obj_box_col" not in r else "%.2f%%" % (100 * r["obj_box_col"]["avg"])
        chg = "-" if "sel_changed_rate" not in r else "%.1f%%" % (100 * r["sel_changed_rate"])
        L.append(f"| {name} | {r['n_applicable']} | {f(r.get('L2', {}).get('avg'))} | {f(r.get('dL2_vs_base_avg'))} | {col} | {chg} | "
                 f"{f(r.get('disp_mean_m'))} | {f(r.get('disp_3s_m'))} | {f(r.get('prob_tv_mean'))} | {f(r.get('det_score_change'), 4)} | "
                 f"{f(r.get('map_score_change'), 4)} |")
    L += ["", "noop = 같은 상태에서 다시 추론(GPU 비결정 잡음 바닥). 다른 조건의 변위가 noop 보다 커야 의미가 있다."]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def scorer_export(a):
    """계획 점수기(Hydra-MDP 식)의 후보별 예측 하위 점수 → npz (그림 '왜 이 후보인가' 패널)."""
    import torch
    import scorer_io
    from apply_and_eval import load_scorer
    from train_scorer import build_inputs, predict, selection_score
    dump = scorer_io.load_dump(a.val_dump)
    model, ck = load_scorer(a.scorer)
    X_, _ = build_inputs(dump, ck["norm"])
    o = predict(model, X_)
    score = selection_score(o, ck["weights"])
    out = dict(idx=dump["idx"], token=dump["token"], cmd=dump["cmd"], modes=dump["modes"], probs=dump["probs"], sel_model=dump["sel"],
               im=torch.softmax(o["im"], -1).numpy(), nc=torch.sigmoid(o["nc"]).numpy(), dac=torch.sigmoid(o["dac"]).numpy(),
               ep=torch.sigmoid(o["ep"]).numpy(), score=score.numpy(), sel_scorer=score.argmax(-1).numpy(),
               weights=json.dumps(ck["weights"]), scorer=a.scorer, val_dump=a.val_dump)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)
    print(f"[완료] {a.out}: 표본 {len(out['idx'])}, 점수기 선택이 원 모델과 다른 비율 {(out['sel_scorer'] != out['sel_model']).mean():.3f}")


def main():
    a = parse()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "summarize":
        summarize(a.out)
    else:
        scorer_export(a)


if __name__ == "__main__":
    main()
