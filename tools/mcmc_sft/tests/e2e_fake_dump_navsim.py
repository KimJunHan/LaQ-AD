# 수정(2026-11-06): 사용자 결정 "NAVSIM MCMC SFT" — 새 파일. 가짜 덤프 → gen_mcmc_targets_navsim.py 끝까지(CPU, navsim env).
"""navtrain info 앞 --frames 프레임(평가·이력 섞임)으로 덤프 형식(dump_plan_modes_navsim.py)의 가짜 덤프를 만든다
(모드 = GT + 시점 비례 가우시안, logit 무작위; smoke=True). 생성기를 --allow_smoke_dump --limit_chain 50 으로 돌리고 출력을 검사한다:
  상태 수 합 = 프레임 수, 이력·불완전·체인 밖 프레임 증분 = info 비트 동일, 체인 프레임 x* ∈ C(L2·후진·7 하위 점수 ≥ GT),
  공식 pdm_score 재검사 수 ≥ 체인 수, smoke 표시. 끝으로 데이터셋 켬 경로 시험용 전체 크기(152,495 token, smoke=False) 정답 파일을
  scratch 에 쓴다(체인으로 바뀐 행만 다름) — hipad env 의 check_sft_configs_navsim.py samples/compare 로 이어서 확인.
  source scripts/navsim/navsim_env.sh && PYTHONPATH=. $NAVSIM_PY tools/mcmc_sft/tests/e2e_fake_dump_navsim.py --work <scratch 폴더>
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "tools", "mcmc_sft"))
import gen_mcmc_targets_navsim as G  # noqa: E402

INFO = "data/infos/navsim_20261019/navsim_navtrain.pkl"
NAV2CMD3 = {0: 0, 1: 1, 2: 2, 3: 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--chains", type=int, default=50)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    dump = os.path.join(a.work, "dump"); out = os.path.join(a.work, "gen")
    os.makedirs(dump, exist_ok=True)
    with open(os.path.join(ROOT, INFO), "rb") as f:   # 프로젝트 info
        all_infos = pickle.load(f)["infos"]
    infos = all_infos[:a.frames]
    rng = np.random.default_rng(0)
    rec = {k: [] for k in ("idx", "token", "scene_token", "timestamp", "is_eval", "navsim_token", "cmd", "modes", "probs", "raw_logits",
                           "post_logits", "rescored", "sel", "plan", "gt_traj", "gt_disp", "gt_mask", "ego_status")}
    scale = np.linspace(0.1, 1.0, 8)[:, None]
    for i, x in enumerate(infos):
        disp = np.asarray(x["gt_ego_fut_trajs"], np.float32); cum = disp.cumsum(0)
        modes = (cum[None] + rng.normal(0, 0.8, (16, 8, 2)) * scale[None]).astype(np.float32)
        raw = rng.normal(0, 1, 16).astype(np.float32)
        p = np.exp(raw - raw.max()); p = (p / p.sum()).astype(np.float32); sel = int(np.argmax(p))
        es = np.asarray(x["ego_status"], np.float32)
        rec["idx"].append(i); rec["token"].append(x["token"]); rec["scene_token"].append(x["scene_token"])
        rec["timestamp"].append(x["timestamp"] / 1e6); rec["is_eval"].append(bool(x.get("is_eval_frame")))
        rec["navsim_token"].append(x.get("navsim_token") or ""); rec["cmd"].append(NAV2CMD3[int(np.argmax(x["driving_command"]))])
        rec["modes"].append(modes); rec["probs"].append(p); rec["raw_logits"].append(raw); rec["post_logits"].append(raw)
        rec["rescored"].append(False); rec["sel"].append(sel); rec["plan"].append(modes[sel]); rec["gt_traj"].append(cum)
        rec["gt_disp"].append(disp); rec["gt_mask"].append(np.asarray(x["gt_ego_fut_masks"], np.float32))
        rec["ego_status"].append(np.where(np.isfinite(es), es, 0.0).astype(np.float32))
    arr = {k: np.asarray(v) for k, v in rec.items()}
    for k in ("token", "scene_token", "navsim_token"):
        arr[k] = arr[k].astype("U64")
    np.savez(os.path.join(dump, "shard_0000.npz"), **arr)
    json.dump({"config": "FAKE", "checkpoint": {"path": "FAKE"}, "split": "navtrain", "ann_file": INFO, "num_samples": a.frames,
               "shards": [[0, a.frames]], "smoke": True, "tool": "e2e_fake_dump_navsim.py"}, open(os.path.join(dump, "manifest.json"), "w"))
    cmd = [sys.executable, os.path.join(ROOT, "tools/mcmc_sft/gen_mcmc_targets_navsim.py"), "--dump", dump, "--info", os.path.join(ROOT, INFO),
           "--metric_cache", os.path.join(os.environ["NAVSIM_EXP_ROOT"], "metric_cache_navtrain"), "--out", out, "--arms", "mcmc,randc",
           "--allow_smoke_dump", "--limit_chain", str(a.chains), "--workers", str(a.workers)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)

    st = json.load(open(os.path.join(out, "stats.json")))
    n_ev = sum(bool(x.get("is_eval_frame")) for x in infos)
    assert sum(st["status_counts"].values()) == a.frames and st["status_counts"]["history_no_cache"] == a.frames - n_ev, st["status_counts"]
    assert st["n_run"] == a.chains and st["smoke"] is True
    info_disp = np.stack([np.asarray(x["gt_ego_fut_trajs"], np.float32) for x in infos])
    tg = np.load(os.path.join(out, "targets_gt.npz"))
    assert bool(tg["smoke"]) and np.array_equal(tg["gt_ego_plan_sft_2hz"], info_disp)
    changed_rows = {}
    for arm in ("mcmc", "randc"):
        z = np.load(os.path.join(out, f"targets_{arm}.npz"))
        assert bool(z["smoke"]) and list(z["token"]) == [x["token"] for x in infos]
        ch = np.nonzero(z["n_accept"] > 0)[0]
        keep = np.setdiff1d(np.arange(a.frames), ch)
        assert np.array_equal(z["gt_ego_plan_sft_2hz"][keep], info_disp[keep]), f"{arm}: 바뀌지 않은 행이 info 와 다르다"
        assert set(ch) <= set(np.nonzero(z["status"] == 0)[0])
        run = np.nonzero(~np.isnan(z["sub_gt"][:, 0]))[0]
        assert len(run) == a.chains
        for i in run:
            x = z["traj"][i]; g = info_disp[i].cumsum(0)
            assert G.mean_l2(x, g) <= 0.5 + 1e-9 and G.reverse_ok(x, g, 0.05) and G.sub_in_C(z["sub_star"][i], z["sub_gt"][i])
        changed_rows[arm] = ch
        print(f"[{arm}] 수용률 {st[arm]['accept_rate']:.3f}, 체인 불변 {st[arm]['frac_chains_unchanged']:.3f}, "
              f"L2 평균 {st[arm]['l2_star_gt_8pt']['mean']:.3f}, PDM 호출/표본 {st[arm]['pdm_calls_per_sample']:.2f}, 바뀐 행 {len(ch)}")
    assert st["n_verified_official"] >= a.chains
    print(f"[통과] 상태 {st['status_counts']}, GT 하위 점수 실패 {st['gt_subscore_fail_counts']}, 공식 재검사 {st['n_verified_official']}, "
          f"워커 초/표본 {st['worker_seconds_per_sample']}")
    # 데이터셋 켬 경로 시험용 전체 크기 정답 파일(시험 전용, scratch)
    z = np.load(os.path.join(out, "targets_mcmc.npz"))
    full = np.stack([np.asarray(x["gt_ego_fut_trajs"], np.float32) for x in all_infos])
    full[:a.frames] = z["gt_ego_plan_sft_2hz"]
    p = os.path.join(a.work, "targets_e2e_full_nonsmoke_TESTONLY.npz")
    np.savez(p, token=np.array([x["token"] for x in all_infos]), gt_ego_plan_sft_2hz=full, smoke=np.array(False))
    ch = changed_rows["mcmc"]
    hist = [i for i in range(a.frames) if not infos[i].get("is_eval_frame")][:1]
    print(f"[데이터셋 시험] {p} — 바뀐 색인 예 {ch[:3].tolist()}, 이력 색인 {hist}")
    json.dump({"file": p, "changed": ch[:3].tolist(), "history": hist}, open(os.path.join(a.work, "dataset_test_idx.json"), "w"))


if __name__ == "__main__":
    main()
