# 수정(2026-10-20): 사용자 승인 — 설명가능성 4가지(학습 변경 없음)
"""교사 토큰 교차 어텐션 기록(무엇을 읽었나) — GPU1 추론, 평가와 같은 순서·설정, 장면 단위 샤드로 재개 가능.

무엇을 기록하나(표본마다, 디코더 6 계층 각각):
  - 계획 쿼리(현재 명령 묶음의 16 모드)와 자차(ego) 쿼리의 교사 토큰 어텐션(머리 8 개 평균)을 해석 단위(segment)로 합한 값
      segment = 폴더 7(context·detection·tracking·motion·map·plan·action) + context 부분 7(카메라 6·지시문 줄) + 객체 칸 8 + 기타 3
      (xai_common.TokenLayout — 폴더 _meta.json 의 answer_positions 로 칸 번호를 정한다)
  - 최종 선택 모드(argmax plan_mode_probs) 쿼리와 ego 쿼리의 토큰별 가중치(계층 평균·마지막 계층, fp16) — 상위 토큰·그림용
  - 엔트로피(가중치가 얼마나 퍼졌나), 검출 900·지도 100 쿼리 평균 segment 가중치(선택)
  - 게이트 주입량: 쿼리 구간별 mean ‖g⊙A‖/‖query‖ (언어 항이 쿼리를 실제로 몇 % 바꿨나) + 학습된 |g| 표(manifest)
가중치 계산(flash 커널은 가중치를 돌려주지 않는다): xai_common.explicit_cross_attn 이 같은 모듈의 in_proj(q/k/v)·out_proj 로
  softmax(QKᵀ/√d) 를 fp32 로 직접 계산한다. 모델 출력 자체는 flash 경로 그대로 두고(--impl flash, 기본) 같은 호출에서 명시적 A 와
  flash A 의 차이를 매 계층 기록·검사한다(상대차 > --max_rel_diff 면 중단 — 수식이 다르다는 뜻). 학습·가중치·파일은 바꾸지 않는다.

출력(--out): manifest.json, shard_XXXX.npz, shard_XXXX.state.pt(시간 뱅크 — 정확한 재개), summary.json/summary.md(--summarize)
사용(GPU1 전용, GPU0 금지):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. OPENBLAS_CORETYPE=SkylakeX python tools/xai/xai_attention.py \
      configs/nuscenes/v2_teacher/stage2/test.py results/hipad_nusc_laq_v2_teacher_stage2/latest.pth --out results/<run>_xai/attention
  스모크: --smoke_samples 3 --max_gpu_mib 12000 --need_mib 12000   · 요약만: --summarize
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xai_common as X  # noqa: E402

PER_SAMPLE = ("idx", "token", "scene_token", "cmd", "sel", "probs", "plan", "n_teacher_obj", "agg_modes", "agg_ego", "tok_sel",
              "tok_ego", "ent_modes", "ent_ego", "det_agg", "map_agg", "inj", "a_diff")


def parse():
    from mmcv import DictAction
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--impl", choices=["flash", "explicit"], default="flash",
                    help="flash(기본) = 모델 출력은 원래 flash 커널, 가중치는 같은 호출에서 명시적 계산 / explicit = 출력도 fp32 명시적 계산")
    ap.add_argument("--max_rel_diff", type=float, default=0.05, help="명시적 A vs flash A 상대 노름차 상한(bf16 커널 오차 ≈ 1e-2)")
    ap.add_argument("--no_det_map", action="store_true", help="검출·지도 쿼리 평균 가중치 생략")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scenes_per_shard", type=int, default=25)
    ap.add_argument("--smoke_samples", type=int, default=0)
    ap.add_argument("--max_gpu_mib", type=int, default=0, help=">0 이면 이 프로세스 GPU 메모리 상한(공유 GPU 보호)")
    ap.add_argument("--need_mib", type=int, default=12000, help="시작 전 GPU1 사용량 + 이 값 ≤ 88000 MiB 검사")
    ap.add_argument("--summarize", action="store_true", help="추론 없이 샤드 요약만")
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    return ap.parse_args()


def run(a):
    X.require_gpu1(a.need_mib)
    import torch
    from mmdet.datasets import build_dataloader
    from torch.utils.data import Subset
    import scorer_io
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
               num_samples=shards[-1][1], shards=[list(s) for s in shards], smoke=bool(a.smoke_samples),
               cfg_options=dict(a.cfg_options or {}), created_by=X.TAG)
    X.check_manifest(a.out, man, ("config", "checkpoint", "ann_file", "impl", "num_samples", "shards", "smoke", "cfg_options"))
    done = [os.path.exists(os.path.join(a.out, f"shard_{k:04d}.npz")) for k in range(len(shards))]
    if all(done):
        print(f"[완료] {a.out}: 샤드 {len(shards)} 개 모두 있다")
        return
    first = done.index(False)
    if any(done[first:]):
        raise SystemExit(f"샤드 {first} 이후에 이미 있는 샤드가 있다 — 순차 가정 위반")

    model, det = X.build_model(cfg, a.checkpoint)
    dec = det.head.onedecoder_head
    qta = dec.qwen_token_attn
    if qta is None:
        raise SystemExit("이 config 에는 교사 토큰 교차 어텐션(qwen_token_attn)이 없다")
    bank = X.BankState(det)
    rng = X.decoder_rows(dec)
    C, M = dec.ego_fut_cmd, dec.ego_fut_mode
    plan_rows = list(range(*rng["plan"]))
    hook = X.XaiHook(qta, plan_rows, [rng["ego"][0]], None if a.no_det_map else rng["det"], None if a.no_det_map else rng["map"])
    hook.impl, hook.record = a.impl, True
    hook.install()
    gt = X.static_gate_table(qta)
    mp = os.path.join(a.out, "manifest.json")
    m = json.load(open(mp)); m.update(static_gate=gt, rows={k: list(v) for k, v in rng.items()}, C=C, M=M); json.dump(m, open(mp, "w"), indent=1)

    idx_val = X.load_val_index()
    import common_v2 as V
    if first > 0:
        st = torch.load(os.path.join(a.out, f"shard_{first - 1:04d}.state.pt"), map_location="cuda:0", weights_only=False)
        bank.load(st)
        print(f"[재개] 샤드 {first} 부터")
    layout, seg_names = None, None
    t0, n_done, max_rel = time.time(), 0, 0.0
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
                seg_names = layout.names
                hook.seg_mat = layout.matrix(torch, "cuda")
                m = json.load(open(mp)); m.update(segments=seg_names, L=layout.L, segment_sizes={n_: int(len(layout.seg[n_])) for n_ in seg_names})
                json.dump(m, open(mp, "w"), indent=1)
            elif not np.array_equal(qt, layout.qwen_type):
                raise RuntimeError(f"색인 {i}: 토큰 유형 배치가 첫 표본과 다르다(폴더 고정 길이 가정 위반)")
            hook.clear()
            with torch.no_grad():
                r = model(return_loss=False, rescale=True, **data)[0]["img_bbox"]
            R = hook.rec
            nl = len(R["layers"])
            if R["layers"] != list(range(nl)) or nl == 0:
                raise RuntimeError(f"색인 {i}: 기록 계층 순서 {R['layers']}")
            for l, (mx, rel) in R["a_diff"].items():
                max_rel = max(max_rel, rel)
                if rel > a.max_rel_diff:
                    raise RuntimeError(f"색인 {i} 계층 {l}: 명시적 A 와 flash A 상대차 {rel:.3g} > {a.max_rel_diff} — 가중치 계산 수식 재확인")
            cmd = int(np.argmax(np.asarray(info["gt_ego_fut_cmd"])))
            probs = np.asarray(r["plan_mode_probs"], np.float32)
            sel = int(np.argmax(probs))
            g0 = cmd * M
            agg = np.stack([R["agg"][l].numpy() for l in range(nl)])          # (nl, C*M+1, n_seg)
            tok = np.stack([R["tok"][l].float().numpy() for l in range(nl)])   # (nl, C*M+1, L)
            ent = np.stack([R["ent"][l].numpy() for l in range(nl)])
            rec["idx"].append(i); rec["token"].append(info["token"]); rec["scene_token"].append(info["scene_token"])
            rec["cmd"].append(cmd); rec["sel"].append(sel); rec["probs"].append(probs)
            rec["plan"].append(np.asarray(r["plan_temp_2hz"], np.float32)[:6, :2])
            e = idx_val.get(info["token"])
            rec["n_teacher_obj"].append(-1 if e is None else len(V.parse_answer(e["gen_text"])["objects"]))
            rec["agg_modes"].append(agg[:, g0:g0 + M].astype(np.float32)); rec["agg_ego"].append(agg[:, -1].astype(np.float32))
            rec["tok_sel"].append(np.stack([tok[:, g0 + sel].mean(0), tok[-1, g0 + sel]]).astype(np.float16))
            rec["tok_ego"].append(np.stack([tok[:, -1].mean(0), tok[-1, -1]]).astype(np.float16))
            rec["ent_modes"].append(ent[:, g0:g0 + M].astype(np.float32)); rec["ent_ego"].append(ent[:, -1].astype(np.float32))
            nseg = len(seg_names)
            rec["det_agg"].append(np.stack([R["det_agg"][l].numpy() for l in range(nl)]) if R["det_agg"] else np.zeros((nl, nseg), np.float32))
            rec["map_agg"].append(np.stack([R["map_agg"][l].numpy() for l in range(nl)]) if R["map_agg"] else np.zeros((nl, nseg), np.float32))
            rec["inj"].append(np.asarray([[R["inj"][l].get(s, np.nan) for s in ("det", "map", "plan", "ego")] for l in range(nl)], np.float32))
            rec["a_diff"].append(np.asarray([R["a_diff"].get(l, (np.nan, np.nan)) for l in range(nl)], np.float32))
            n_done += 1
        out = {key: np.asarray(v) for key, v in rec.items()}
        for key in ("token", "scene_token"):
            out[key] = out[key].astype("U64")
        X.atomic_savez(os.path.join(a.out, f"shard_{k:04d}.npz"), **out)
        sp = os.path.join(a.out, f"shard_{k:04d}.state.pt")
        torch.save(bank.save(), sp + ".tmp"); os.replace(sp + ".tmp", sp)
        el = time.time() - t0
        print(f"[샤드 {k + 1}/{len(shards)}] 색인 {s0}..{s1 - 1} | 누적 {n_done} 표본 {el:.0f}s ({el / max(n_done, 1):.3f}s/표본) "
              f"| 명시적-vs-flash 최대 상대차 {max_rel:.2e} | GPU 최대 {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB", flush=True)
    hook.uninstall()
    m = json.load(open(mp)); m["timing"] = dict(sec_per_sample_wall=(time.time() - t0) / max(n_done, 1), n_samples=n_done)
    json.dump(m, open(mp, "w"), indent=1, ensure_ascii=False)
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
    seg = man["segments"]
    S = {n: j for j, n in enumerate(seg)}
    folders = list(X.FOLDERS)
    n = len(D["idx"])
    sel_agg = D["agg_modes"][np.arange(n), :, D["sel"]]                     # (n, nl, n_seg) 선택 모드
    res = dict(n=n, smoke=man["smoke"], checkpoint=man["checkpoint"], static_gate=man.get("static_gate"),
               note="attention share = 머리 평균 softmax 가중치의 segment 합(폴더 7 개 합 = 1)")
    def share(A):  # (n, nl, n_seg) → 계층별 폴더 평균
        return {f: A[:, :, S[f]].mean(0).tolist() for f in folders}
    res["plan_selected_by_layer"] = share(sel_agg)
    res["plan_all16_by_layer"] = share(D["agg_modes"].mean(2))
    res["ego_by_layer"] = share(D["agg_ego"])
    res["det_by_layer"] = share(D["det_agg"])
    res["map_by_layer"] = share(D["map_agg"])
    res["plan_selected_layer_mean"] = {f: float(sel_agg[:, :, S[f]].mean()) for f in folders}
    # 균등 기준(모든 토큰에 같은 가중치일 때의 비율 = 토큰 수 비율)과 그 대비 배수(lift) — 해석 기준선
    cnt = man.get("segment_sizes") or {}
    res["uniform_share"] = {f: cnt[f] / man["L"] for f in folders} if cnt else None
    if cnt:
        res["plan_selected_lift"] = {f: res["plan_selected_layer_mean"][f] / res["uniform_share"][f] for f in folders}
    res["plan_selected_context_parts"] = {p: float(sel_agg[:, :, S[p]].mean()) for p in seg if p.startswith("ctx_")}
    # 객체 칸: 교사가 실제로 적은 객체(채워진 칸) vs 빈 칸, 칸당 평균
    nobj = D["n_teacher_obj"]
    slot = np.stack([sel_agg[:, :, S[f"slot{k}"]].mean(1) for k in range(X.N_SLOTS)], 1)   # (n, 8) 계층 평균
    filled = np.arange(X.N_SLOTS)[None] < nobj[:, None]
    res["plan_selected_slot_filled_mean"] = float(slot[filled].mean()) if filled.any() else None
    res["plan_selected_slot_empty_mean"] = float(slot[~filled].mean()) if (~filled).any() else None
    res["plan_selected_slot_by_index"] = slot.mean(0).tolist()
    res["entropy_selected_by_layer"] = D["ent_modes"][np.arange(n), :, D["sel"]].mean(0).tolist()
    res["entropy_uniform_nats"] = float(np.log(man["L"]))
    res["injection_ratio_by_layer"] = {s: np.nanmean(D["inj"][:, :, j], 0).tolist() for j, s in enumerate(("det", "map", "plan", "ego"))}
    res["explicit_vs_flash_rel_diff_max"] = float(np.nanmax(D["a_diff"][:, :, 1])) if np.isfinite(D["a_diff"][:, :, 1]).any() else None
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    L = [f"# 교사 토큰 어텐션 요약 (표본 {n}{', 스모크' if man['smoke'] else ''})", "",
         "선택된 계획 모드 쿼리의 폴더별 어텐션 비율(계층 평균):", "",
         "| " + " | ".join(folders) + " |", "|" + "---|" * len(folders),
         "| " + " | ".join(f"{100 * res['plan_selected_layer_mean'][f]:.1f}%" for f in folders) + " |",
         ("| " + " | ".join(f"균등 {100 * res['uniform_share'][f]:.1f}% (×{res['plan_selected_lift'][f]:.2f})" for f in folders) + " |")
         if res.get("uniform_share") else "", "",
         "context 부분: " + ", ".join(f"{p} {100 * v:.1f}%" for p, v in res["plan_selected_context_parts"].items()), "",
         f"객체 칸 평균(칸 하나): 채워진 칸 {res['plan_selected_slot_filled_mean']} · 빈 칸 {res['plan_selected_slot_empty_mean']}", "",
         "계층별 언어 주입량 mean ‖g⊙A‖/‖q‖: " + "; ".join(f"{s} " + "/".join(f"{v:.3f}" for v in vals)
                                                    for s, vals in res["injection_ratio_by_layer"].items()), "",
         f"명시적 계산 vs flash 출력 최대 상대차: {res['explicit_vs_flash_rel_diff_max']}"]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    a = parse()
    if a.summarize:
        summarize(a.out)
    else:
        run(a)


if __name__ == "__main__":
    main()
