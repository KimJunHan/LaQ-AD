# 수정(2026-11-06): 사용자 결정 "NAVSIM MCMC SFT" — 새 파일. tools/mcmc_sft/gen_mcmc_targets_navsim.py 단위 시험(CPU, navsim env).
"""  source scripts/navsim/navsim_env.sh && PYTHONPATH=. $NAVSIM_PY -m pytest -q tools/mcmc_sft/tests/test_gen_mcmc_navsim.py
  A (순수, 채점기 불필요): heading 규칙 = 채점기·데이터셋 규칙, 후진 조건(직진·회전·정차·GT 후진), 제안 대칭·공분산,
     토이 혼합에서 MH 정상성(정확한 표본에서 시작해 10 단계 뒤 분포 불변; 틀린 커널(언덕 오르기)은 같은 시험에서 실패 = 검정력),
     지연 판정과 전체 판정의 체인 비트 동일.
  B (채점기·navtrain metric cache 필요, NAVSIM_EXP_ROOT): C 판정 = 공식 pdm_score 하위 점수 정의, 실제 장면 체인 재현.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "tools", "mcmc_sft"))
sys.path.insert(0, os.path.join(ROOT, "tools", "navsim"))
import gen_mcmc_targets_navsim as G  # noqa: E402

T = G.T


def straight(v=10.0):
    return np.stack([v * 0.5 * np.arange(1, T + 1), np.zeros(T)], -1)


# ------------------------------------------------------------------------------------------------ A. 순수
def test_heading_rule_matches_scorer_and_dataset():
    from navsim_score_vocab_navtrain import heading_from_xy
    rng = np.random.default_rng(0)
    for _ in range(200):
        xy = np.cumsum(rng.normal(0, rng.choice([0.01, 0.5, 3.0]), (T, 2)), 0)
        assert np.array_equal(G.heading_xy(xy), heading_from_xy(xy)[:, 2])


def test_reverse_straight_and_threshold():
    gt = straight(10.0)
    x = gt.copy(); x[3:, 0] -= 6.0            # 3 번 시점에서 1 m 후진(10 m/s × 0.5 s = 5 m 전진 → −1 m)
    assert not G.reverse_ok(x, gt, 0.05)
    st = np.zeros((T, 2))                     # 정차 GT
    x = st.copy(); x[2:, 0] = -0.06           # 6 cm 후진 → 금지
    assert not G.reverse_ok(x, st, 0.05)
    x = st.copy(); x[2:, 0] = -0.04           # 4 cm → 허용(문턱 5 cm)
    assert G.reverse_ok(x, st, 0.05)
    assert G.reverse_ok(gt, gt, 0.05) and G.reverse_ok(st, st, 0.05)
    assert G.reverse_ok(x, st, -1.0)          # 조건 끔


def test_reverse_allowed_when_gt_reverses():
    gt = np.zeros((T, 2)); gt[:, 0] = -0.3 * np.arange(1, T + 1)   # GT 가 계속 후진
    x = gt * 1.2
    assert G.reverse_ok(x, gt, 0.05)


def test_reverse_tangent_on_sharp_turn():
    """U 턴 GT: 끝 무렵 Δx < 0 이지만 진행 방향으로는 전진. Δx 기준이면 GT 를 조금만 흔들어도 '후진' 오판 — 접선 기준은 허용."""
    ang = np.linspace(np.pi / 8, np.pi, T)
    gt = np.stack([6 * np.sin(ang), 6 * (1 - np.cos(ang))], -1)
    dx = np.diff(gt[:, 0], prepend=0.0)
    assert (dx < -0.05).any()                                       # Δx 로는 GT 자체가 후진처럼 보인다
    assert G.reverse_ok(gt * 1.01, gt, 0.05)
    x = gt.copy(); x[5:] -= 3.5 * np.stack([np.cos(G.heading_xy(gt)[4]), np.sin(G.heading_xy(gt)[4])])   # 진행 방향 반대로 3.5 m(GT 한 칸 ≈ 2.4 m 보다 크게 → 그 칸이 후진)
    assert not G.reverse_ok(x, gt, 0.05)


def test_reverse_tangent_equals_dx_when_straight():
    gt = straight(5.0)
    rng = np.random.default_rng(1)
    for _ in range(200):
        x = gt + rng.normal(0, 0.5, gt.shape)
        dx = np.diff(x[:, 0], prepend=0.0); dg = np.diff(gt[:, 0], prepend=0.0)
        assert G.reverse_ok(x, gt, 0.05) == (not ((dx < -0.05) & (dg >= -0.05)).any())


def test_proposal_cov_and_symmetry():
    sig = np.stack([np.linspace(0.05, 0.6, T), np.linspace(0.1, 0.4, T)], -1)
    ch = G.proposal_chol(sig, 1.0)
    t = 0.5 * np.arange(1, T + 1)
    rng = np.random.default_rng(2)
    for m in (0, 3, 7):
        z = rng.standard_normal((200000, T - m))
        d = z @ ch[m][0].T
        K = np.outer(sig[m:, 0], sig[m:, 0]) * np.exp(-(t[m:, None] - t[None, m:]) ** 2 / 2)
        assert np.abs(np.cov(d.T).reshape(T - m, T - m) - K).max() < 0.01
        assert np.abs(d.mean(0)).max() < 0.01     # 영평균 → δ 와 −δ 같은 밀도(대칭)


def toy_problem(seed=3, M=4):
    rng = np.random.default_rng(seed)
    gt = straight(4.0).astype(np.float32)
    mu = gt[None] + rng.normal(0, 0.25, (M, T, 2)) * np.linspace(0.2, 1, T)[None, :, None]
    raw = rng.normal(0, 1, M); log_pi = raw - np.log(np.exp(raw).sum())
    var = np.stack([np.linspace(0.02, 0.3, T), np.linspace(0.05, 0.25, T)], -1) ** 2
    inv_var = 1 / var; log_norm = float(-0.5 * np.log(2 * np.pi * var).sum())
    return gt, mu, log_pi, var, inv_var, log_norm


def fake_score(x):
    """가짜 PDM: 끝점 횡 y 가 0.15 m 넘으면 DAC 0, 3 번 시점 전방 x 가 GT(6 m)보다 0.2 m 넘게 앞서면 TTC 0. (subscores 8, epdms)."""
    s = np.ones(8)
    if x[-1, 1] > 0.15:
        s[1] = 0.0
    if x[3, 0] > 6.2:
        s[5] = 0.0
    return s, float(s[1] * (5 + 5 * s[5] + 4) / 14)


def in_C_full(x, gt, eps, sub_gt):
    return bool(G.mean_l2(x, gt) <= eps) and G.reverse_ok(x, gt, 0.05) and G.sub_in_C(fake_score(x)[0], sub_gt)


def exact_samples(n, gt, mu, log_pi, var, eps, sub_gt, rng):
    """p_C 정확 표본: 혼합에서 뽑고 C 밖이면 버리는 거절 표집(정의상 정확)."""
    out = []
    pi = np.exp(log_pi)
    while len(out) < n:
        j = rng.choice(len(pi), p=pi)
        x = (mu[j] + rng.standard_normal((T, 2)) * np.sqrt(var)).astype(np.float32)
        if in_C_full(x, gt, eps, sub_gt):
            out.append(x)
    return np.stack(out)


def chain_from(x0, gt, mu, log_pi, inv_var, log_norm, chol, eps, sub_gt, key, n_steps, hill=False):
    """run_chain 과 같은 커널을 x0 에서 시작(시험용: run_chain 은 GT 에서 시작하므로 초기 상태만 바꾼 복제).
    hill=True 면 틀린 커널(밀도가 오를 때만 수용 = 언덕 오르기)."""
    rng = np.random.default_rng(key)
    x = x0.copy(); lp = G.log_density(x, mu, log_pi, inv_var, log_norm)
    for _ in range(n_steps):
        m = int(rng.integers(0, T)); z = rng.standard_normal((T - m, 2))
        d = np.zeros((T, 2))
        for a in range(2):
            d[m:, a] = chol[m][a] @ z[:, a]
        xn = (x.astype(np.float64) + d).astype(np.float32); u = rng.random()
        lpn = G.log_density(xn, mu, log_pi, inv_var, log_norm)
        dens = (lpn > lp) if hill else (np.log(u) < lpn - lp)
        if dens and in_C_full(xn, gt, eps, sub_gt):
            x, lp = xn, lpn
    return x


def test_mh_stationarity_toy_mixture():
    from scipy.stats import ks_2samp
    gt, mu, log_pi, var, inv_var, log_norm = toy_problem()
    eps = 0.5
    sub_gt = fake_score(gt)[0]
    chol = G.proposal_chol(np.sqrt(var), 1.0)
    rng = np.random.default_rng(4)
    n = 3000
    x0 = exact_samples(n, gt, mu, log_pi, var, eps, sub_gt, rng)
    x10 = np.stack([chain_from(x0[k], gt, mu, log_pi, inv_var, log_norm, chol, eps, sub_gt, [9, k], 10) for k in range(n)])
    ref = exact_samples(n, gt, mu, log_pi, var, eps, sub_gt, rng)            # 독립 기준 표본
    proj = {"end_x": lambda a: a[:, -1, 0], "end_y": lambda a: a[:, -1, 1], "mid_y": lambda a: a[:, 3, 1],
            "l2": lambda a: G.mean_l2(a, gt)}
    moved = np.mean([not np.array_equal(a, b) for a, b in zip(x0, x10)])
    assert moved > 0.3, f"체인이 거의 움직이지 않아 시험이 무의미({moved})"
    for nm, f in proj.items():
        p = ks_2samp(f(x10), f(ref)).pvalue
        assert p > 1e-3, f"정상성 위반 의심 {nm}: KS p={p:.2e}"
    # 검정력: 틀린 커널(언덕 오르기)은 같은 시험에서 걸려야 한다
    xh = np.stack([chain_from(x0[k], gt, mu, log_pi, inv_var, log_norm, chol, eps, sub_gt, [9, k], 10, hill=True) for k in range(n)])
    ps = [ks_2samp(f(xh), f(ref)).pvalue for f in proj.values()]
    assert min(ps) < 1e-6, f"검정력 부족: 언덕 오르기 커널도 통과(p {ps})"


def test_lazy_equals_full_and_C_holds():
    gt, mu, log_pi, var, inv_var, log_norm = toy_problem(5)
    chol = G.proposal_chol(np.sqrt(var), 1.0)
    sub_gt = fake_score(gt)[0]
    n_calls = {True: 0, False: 0}
    for k in range(300):
        for arm in ("mcmc", "randc"):
            outs = {}
            for lazy in (True, False):
                cnt = [0]

                def sf(x):
                    cnt[0] += 1
                    return fake_score(x)
                outs[lazy] = G.run_chain(arm, [1, G.ARMS[arm], k], gt, mu, log_pi, inv_var, log_norm, chol, 0.5, 10, 0.05, sf,
                                         sub_gt, lazy=lazy, ep_gt=fake_score(gt)[1])
                n_calls[lazy] += cnt[0]
            (x1, lp1, c1, s1, e1), (x2, lp2, c2, s2, e2) = outs[True], outs[False]
            assert np.array_equal(x1, x2) and lp1 == lp2 and c1["acc"] == c2["acc"] and np.array_equal(s1, s2) and e1 == e2
            assert in_C_full(x1, gt, 0.5, sub_gt)
            assert np.array_equal(s1, fake_score(x1)[0])
    assert n_calls[True] < n_calls[False]


def test_sub_in_C_excludes_ep_and_is_relative():
    g = np.array([1, 0, 1, 1, 0.9, 1, 1, 1.0])         # GT 가 DAC 실패
    x = np.array([1, 0, 1, 1, 0.1, 1, 1, 1.0])         # EP 낮아도 됨, DAC 0 도 GT 이상
    assert G.sub_in_C(x, g)
    x2 = x.copy(); x2[0] = 0.5                          # NC 악화 → 밖
    assert not G.sub_in_C(x2, g)
    x3 = x.copy(); x3[7] = 0.0                          # HC 악화 → 밖
    assert not G.sub_in_C(x3, g)


# ------------------------------------------------------------------------------------------------ B. 실제 채점기
def _cache_env():
    root = os.environ.get("NAVSIM_EXP_ROOT")
    if not root or os.environ.get("OPENBLAS_CORETYPE") != "SkylakeX":
        pytest.skip("navsim_env.sh 미적용")
    return os.path.join(root, "metric_cache_navtrain")


def test_acceptance_vs_official_scorer():
    import pickle
    import grpo_scorer as GS
    cache = _cache_env()
    paths = GS.cache_paths(cache)
    with open(os.path.join(ROOT, "data/infos/navsim_20261019/navsim_navtrain.pkl"), "rb") as f:   # 프로젝트 info
        infos = pickle.load(f)["infos"]
    ev = [x for x in infos if x.get("is_eval_frame") and (np.asarray(x["gt_ego_fut_masks"]) == 1).all()]
    rng = np.random.default_rng(6)
    pick = [ev[i] for i in rng.choice(len(ev), 6, replace=False)]
    GS._init_worker(False)
    n_out = 0
    for info in pick:
        mc = GS.load_cache(paths[info["navsim_token"]])
        gt = np.asarray(info["gt_ego_fut_trajs"], np.float32).cumsum(0)
        sg, _, eg = GS._official(mc, GS._as_poses(gt.astype(np.float64)[None])[0])
        r = GS.score_token(mc, GS._as_poses(gt.astype(np.float64)[None]), margins=False)
        assert np.array_equal(r["subscores"][0], sg) and float(r["epdms"][0]) == eg
        for k in range(10):
            x = (gt + rng.normal(0, 0.6, gt.shape) * np.linspace(0.1, 1, T)[:, None]).astype(np.float32)
            so, _, _ = GS._official(mc, GS._as_poses(x.astype(np.float64)[None])[0])
            rx = GS.score_token(mc, GS._as_poses(x.astype(np.float64)[None]), margins=False)["subscores"][0]
            assert np.array_equal(rx, so)
            want = all(so[j] >= sg[j] for j in (0, 1, 2, 3, 5, 6, 7))      # 정의 그대로
            assert G.sub_in_C(rx, sg) == want
            n_out += not want
    print(f"[B] 6 장면 × 10 섭동, C 밖(하위 점수) {n_out}")
