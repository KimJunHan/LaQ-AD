# 수정(2026-11-09): C3 GRPO 설계 v4.1 §12-⑤ 덤프 스크립트 — 신규 파일(기존 파일 무수정).
"""GRPO 롤아웃 경로 덤프(GPU, 추론만, no_grad). G1 1k 덤프 · navtrain 전체 μ_ref/B0 덤프 · 내부 보류 집합 덤프를 같은 코드로 만든다.

경로(§8 "μ_ref 는 GRPO 롤아웃 경로로", §9.1): rollout.build_rl_model(플래그 3 개) + RL 파이프라인 + GroupInBatchSampler(시드 0 기본,
  micro-batch 8, 스트림 그룹 6 = 학습과 같은 스트림 문맥) + extract_feat/head 직접 호출 + 명령 묶음 16 모드 + rescore + 상위 M.
기록: RL 프레임(평가 프레임 ∧ 지표 캐시 있음)을 **처음 방문할 때만** 기록한다(샘플러가 장면을 두 번 줄 수 있음 — 두 번째는 순전파만).
  대상 RL 프레임을 모두 방문하면(또는 --max-rl-frames 도달) 끝낸다.
장면 선택(--scene-select): all | hash_first:N(G1: 장면 해시 순 첫 N 장면) | hash_mod:m:r(내부 보류 집합, 제안 규칙 hash_mod:20:0).
  부분집합은 고른 장면의 시퀀스만 샘플러에 넣는다(rollout.SeqSubset) — 다른 장면의 뱅크 문맥은 없다.
산출(--out 폴더)
  dump.npz : tokens(navsim) (n,), frame_tokens, cmd, mu16 (n,16,8,2) ego 프레임, raw16, resc16 (n,16), top (n,M), sel (n,)
  sens.npz : sensitivity.py --dump 형식 — tokens, mu (n,M,8,2) = 상위 M 모드 μ(G0 학생 기준 궤적)
  mu_ref.npz : tokens, mu16 (학습 루프 KL 감시용 — navtrain 전체 덤프에서 쓴다)
  manifest.json, (진행 중) state.pt — 재개: 처리한 micro-batch 수·스트림 뱅크 상태·기록·torch/cuda/numpy/python RNG. 샘플러 fast_forward 로 이어 감.
  DONE : 완료 표식. 모든 산출 파일(dump·sens·mu_ref·manifest)을 쓴 **뒤 맨 마지막에** 쓴다. 체인·재실행은 DONE 으로만 완료를 판단한다.
GPU: CUDA_VISIBLE_DEVICES=3(NAVSIM 전용). 0 은 HIPAD_ALLOW_GPU0=1 이고 --max-rl-frames ≤ 200(스모크)일 때만. 2 금지.
사용(체인이 부른다):
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python tools/navsim/grpo/dump_rollout.py --model-config $MODEL_CFG --ckpt $GRPO_INIT_CKPT \
      --out results/grpo_navsim/gates/dump_g1 --scene-select hash_first:1000
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(os.path.abspath(__file__)).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import rollout as RO  # noqa: E402


def rng_state():
    import torch
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                numpy=np.random.get_state(), py=random.getstate())


def set_rng_state(st):
    import torch
    torch.set_rng_state(st["torch"].cpu() if hasattr(st["torch"], "cpu") else st["torch"])
    if st.get("cuda") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda"]])
    np.random.set_state(st["numpy"])
    random.setstate(st["py"])


def _save_npz(path, **z):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **z)
    os.replace(tmp, path)


def finalize(out, rec, man, M, n_mb, t0):
    """산출 파일을 원자적으로 쓰고 DONE 을 맨 마지막에 쓴다. 중간에 죽으면 DONE 이 없어 재실행이 state.pt 에서 다시 마무리한다."""
    z = {k: np.asarray(v) for k, v in rec.items()}
    for k in ("tokens", "frame_tokens"):
        z[k] = z[k].astype("U64")
    _save_npz(out / "dump.npz", **z)
    mu_top = np.stack([m[t] for m, t in zip(z["mu16"], z["top"])]) if len(z["top"]) else np.zeros((0, M, 8, 2))
    _save_npz(out / "sens.npz", tokens=z["tokens"], mu=mu_top.astype(np.float64))
    _save_npz(out / "mu_ref.npz", tokens=z["tokens"], mu16=z["mu16"])
    man = dict(man, n_recorded=len(z["tokens"]), n_micro_batches=n_mb, seconds=time.time() - t0,
               rescored_frames=int((z["resc16"] < z["raw16"] - 1.0).any(-1).sum()) if len(z["raw16"]) else 0)
    mp = out / "manifest.json"
    mp.with_name("manifest.json.tmp").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    os.replace(mp.with_name("manifest.json.tmp"), mp)
    (out / "DONE.tmp").write_text(json.dumps(dict(n_recorded=man["n_recorded"], n_micro_batches=n_mb, t=time.strftime("%F %T"))))
    os.replace(out / "DONE.tmp", out / "DONE")
    sp = out / "state.pt"
    if sp.exists():
        sp.unlink()


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene-select", default="all")
    ap.add_argument("--max-rl-frames", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--groups", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--metric-cache", default=RO.DEFAULT_METRIC_CACHE)
    ap.add_argument("--save-every", type=int, default=500, help="micro-batch 마다 재개 상태 저장")
    ap.add_argument("--max-gpu-mib", type=int, default=0)
    ap.add_argument("--max-mb", type=int, default=0, help="안전 상한(0 = 대상 프레임 수 × 4 / batch)")
    return ap.parse_args()


def main():
    a = parse()
    dev = os.environ.get("CUDA_VISIBLE_DEVICES")
    if dev == "2" or dev is None or "," in dev:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={dev} — 한 장, GPU2 금지")
    if dev == "0" and not (os.environ.get("HIPAD_ALLOW_GPU0") == "1" and 0 < a.max_rl_frames <= 200):
        raise SystemExit("GPU0 은 HIPAD_ALLOW_GPU0=1 + --max-rl-frames ≤ 200 스모크 전용")
    if dev not in ("0", "3") and os.environ.get("GRPO_ALLOW_GPU") != dev:
        raise SystemExit(f"GPU{dev} — NAVSIM 은 GPU3 전용(병렬은 GRPO_ALLOW_GPU={dev}, 사용자 승인 사항)")
    os.chdir(ROOT)
    import torch
    if a.max_gpu_mib > 0:
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gpu_mib * 2 ** 20 / tot), 0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = RO.load_model_cfg(a.model_config, bc_target="gt")
    from mmdet.apis import set_random_seed
    set_random_seed(a.seed, deterministic=False)
    det = RO.build_rl_model(cfg, a.ckpt, "cuda:0")
    dec = det.head.onedecoder_head
    if dec.num_stream_groups != a.groups:
        raise SystemExit(f"--groups {a.groups} ≠ 디코더 스트림 그룹 {dec.num_stream_groups}")
    ds = RO.build_rl_dataset(cfg, select=a.scene_select)
    table = RO.frame_table(ds.data_infos)
    cache = RO.metric_cache_tokens(a.metric_cache)
    target = {v[0] for v in table.values() if v[0] is not None and v[0] in cache}
    n_eval_nocache = sum(1 for v in table.values() if v[1] and v[0] not in cache)
    goal = min(len(target), a.max_rl_frames) if a.max_rl_frames else len(target)
    man = dict(model_config=a.model_config, ckpt=os.path.abspath(a.ckpt), scene_select=a.scene_select, seed=a.seed,
               batch=a.batch, groups=a.groups, M=a.M, target_rl_frames=len(target), goal=goal, eval_frames_no_cache=n_eval_nocache,
               metric_cache=a.metric_cache, frame="ego = lidar (NAVSIM lidar2ego identity)", tool="tools/navsim/grpo/dump_rollout.py",
               rescore_ego_offset=float(dec.plan_decoder.rescore_ego_offset), ego_start_yaw=float(dec.plan_decoder.ego_start_yaw))
    mp = out / "manifest.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        diff = [k for k in man if old.get(k) != man[k]]
        if diff:
            raise SystemExit(f"{out} 기존 manifest 와 다르다({diff}) — 다른 폴더를 쓸 것")
        if (out / "DONE").exists():
            print(f"[완료] {out}"); return
    else:
        mp.write_text(json.dumps(man, ensure_ascii=False, indent=1))
    rec = {k: [] for k in ("tokens", "frame_tokens", "cmd", "mu16", "raw16", "resc16", "top", "sel")}
    seen, n_mb = set(), 0
    sp = out / "state.pt"
    if sp.exists():
        st = torch.load(sp, map_location="cuda:0", weights_only=False)
        rec, seen, n_mb = st["rec"], set(st["seen"]), st["n_mb"]
        dec._stream_states, dec.run_step = st["stream_states"], st["run_step"]
        set_rng_state(st["rng"])
        print(f"[재개] micro-batch {n_mb}, 기록 {len(seen)}", flush=True)
    loader = RO.build_rl_loader(ds, a.batch, a.groups, a.workers, a.seed, fast_forward=n_mb)
    max_mb = a.max_mb or (4 * len(ds) // a.batch + 10 * a.groups)
    t0 = time.time()
    it = iter(loader)
    with torch.no_grad():
        while len(seen) < goal:
            if n_mb >= max_mb:
                raise RuntimeError(f"micro-batch {n_mb} 에서도 대상 {goal} 중 {len(seen)} 만 방문 — 샘플러·대상 정의 확인")
            batch = next(it)
            data = RO.to_device(batch, "cuda:0")
            outs, data = RO.forward_heads(det, data, grad=False)
            raw16, mu16, cmd = RO.plan_bundle(det, outs, data)
            resc = RO.rescore16(det, outs, raw16, mu16)
            metas = RO.frame_metas(data)
            mu_np, raw_np, resc_np = (x.float().cpu().numpy() for x in (mu16, raw16, resc))
            for b, m in enumerate(metas):
                nav = table[m["token"]][0]
                if nav is None or nav not in target or nav in seen or len(seen) >= goal:
                    continue
                seen.add(nav)
                rec["tokens"].append(nav); rec["frame_tokens"].append(m["token"]); rec["cmd"].append(int(cmd[b]))
                rec["mu16"].append(mu_np[b]); rec["raw16"].append(raw_np[b]); rec["resc16"].append(resc_np[b])
                rec["top"].append(RO.top_modes(resc_np[b], raw_np[b], a.M)); rec["sel"].append(int(np.argmax(resc_np[b])))
            n_mb += 1
            if n_mb % a.save_every == 0:
                torch.save(dict(rec=rec, seen=sorted(seen), n_mb=n_mb, stream_states=dec._stream_states, run_step=dec.run_step,
                                rng=rng_state()),
                           str(sp) + ".tmp")
                os.replace(str(sp) + ".tmp", sp)
                el = time.time() - t0
                print(f"[덤프] micro-batch {n_mb}, RL 프레임 {len(seen)}/{goal} ({el:.0f}s)", flush=True)
    finalize(out, rec, man, a.M, n_mb, t0)
    print(f"[완료] {out}: RL 프레임 {len(seen)}, micro-batch {n_mb}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
