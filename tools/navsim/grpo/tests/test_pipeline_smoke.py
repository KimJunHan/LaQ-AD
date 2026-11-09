# 수정(2026-11-06): C3 M1–M3 연결 스모크(실 채점기 출력 → advantage) — 신규 파일.
"""navsim env 에서 실행(채점기 필요):
  source scripts/navsim/navsim_env.sh && $NAVSIM_PY tools/navsim/grpo/tests/test_pipeline_smoke.py --tokens 8
장면마다 16 앵커(ξ = 0 대용) + 앵커 4 개 주위 AR(1) 차선축 표본 8 개(ar1, 실 PDM 중심선 기저)를 채점(guide=True)하고
advantage.scene_advantage 를 모든 변형으로 돌려 형식·합 조건(Σw = 8, Σu = 2, 단계 하나)을 확인한다."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools/navsim"))
sys.path.insert(0, str(ROOT / "tools/navsim/grpo"))
import advantage as AD  # noqa: E402
import ar1  # noqa: E402
import grpo_scorer as GS  # noqa: E402
from extract_pdm_centerlines import S_GRID, extract_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric-cache", default=os.path.join(os.environ.get("NAVSIM_EXP_ROOT", ""), "metric_cache_navtrain"))
    ap.add_argument("--anchors", default=str(ROOT / "data/kmeans/navsim_v2_plan_temp_48.npy"))
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=2.0)
    a = ap.parse_args()
    GS._init_worker(True)
    anchors = np.load(a.anchors).astype(np.float64)[:16]               # lidar2ego = 단위행렬(NAVSIM)
    paths = GS.cache_paths(a.metric_cache)
    toks = sorted(paths)[::997][:a.tokens]
    sig = ar1.sigma_table(a.kappa)
    res = []
    for ti, tok in enumerate(toks):
        mc = GS.load_cache(paths[tok])
        cl = extract_one(mc)
        top = np.arange(4)
        E, info = ar1.axis_basis(anchors[top], cl["xyh"][:, :2].astype(np.float64), s_grid=S_GRID, valid=cl["valid"])
        pol = []
        for q in range(4):
            eta = ar1.sample_eta(0, 0, tok, int(top[q]), 8)
            pol.append(ar1.sample_tau(anchors[top[q]], E[q], sig, eta))
        cands = np.concatenate([anchors, np.concatenate(pol)])
        out = GS.score_token(mc, GS._as_poses(cands), True, guide=True)
        sc = AD.build_scene(tok, out, mu16=anchors)
        for v in AD.VARIANTS:
            r = AD.scene_advantage(sc, v)
            assert r["A"].shape == (4, 8, 8, 2)
            np.testing.assert_allclose(r["w"].sum(-1), 8.0)
            np.testing.assert_allclose(r["u"].sum(-1), 2.0)
            assert set(r["stage"].tolist()) <= {1, 2, 3, 4}
            if v == "Full":
                res.append(r)
        print(f"{tok}: 단계 {res[-1]['stage'].tolist()}, 주 실패 {[f for f in res[-1]['ftype'].ravel() if f]!s:.80}, "
              f"기저 out_of_range {int(info['out_of_range'].sum())}/32, r* {out['ref_epdms']:.3f}", flush=True)
    print("[스모크 통과]", AD.batch_log(res))


if __name__ == "__main__":
    main()
