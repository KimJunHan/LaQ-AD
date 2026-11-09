# 수정(2026-11-06): C3 GRPO G2-4 시험(설계 v4 §12 G2-4) + ar1.py 단위 시험 — 신규 파일.
"""tools/navsim/grpo/ar1.py 시험. 스크래치 ar1chk.py 를 옮겨 G2-4 기준으로 바꿨다.
  source /home1/irteam/data-vol1/hipad_env.sh && python tools/navsim/grpo/tests/test_ar1.py      (pytest 로도 실행 가능)

G2-4: 구현 손실(ar1.loss_rl_torch)의 ∂L/∂μ 를, 기저 e·σ 를 detach 하고 η(μ) 를 μ 의 함수로 둔
      −(σ_ref²/N) Σ_{i,k,a} A_{i,k,a} log N(η_{i,k,a}(μ); 0, 1) 의 autograd 기울기와 요소별 상대 오차로 비교.
      분모 = max(|ref|, 1e-8·‖ref‖_∞). 기준 fp32 ≤ 1e-6, fp64 ≤ 1e-12.
      입력: 합성 고곡률 기저(경유점당 0.3–0.5 rad 회전), 축마다 다른 σ(설계 표 × κ). 경로: B1(균등 A)·Full(w_k·u_a).
      음성 시험: 교차항에 e_{a,k−1} 대신 e_{a,k} 를 쓴 손실(심은 오류)이 기준을 크게 넘는지(=잡히는지) 확인.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ar1  # noqa: E402

N_TRIALS = 200
TOL = {torch.float32: 1e-6, torch.float64: 1e-12}


def _bases(curv, T=8, phase=0.0, n=1):
    """(n, T, 2, 2) 경유점 j 의 접선 각 = phase + curv·(j+1) (+ 표본마다 약간 다른 곡률)."""
    out = []
    for i in range(n):
        th = phase + np.cumsum(np.full(T, curv * (1 + 0.1 * i)))
        el = np.stack([np.cos(th), np.sin(th)], -1)
        ea = np.stack([-np.sin(th), np.cos(th)], -1)
        out.append(np.stack([el, ea], 1))
    return np.stack(out)


def _relerr(g, ref):
    assert torch.isfinite(g).all(), "구현 기울기에 NaN/Inf"
    assert torch.isfinite(ref).all(), "기준 기울기에 NaN/Inf"
    g, ref = g.double(), ref.double()
    den = torch.clamp(ref.abs(), min=1e-8 * ref.abs().max().item())
    return ((g - ref).abs() / den).max().item()


def _nan_max(*xs):
    """NaN 을 전파하는 최대(파이썬 max(a, nan) 은 a 를 돌려 NaN 을 버린다)."""
    return float(np.max(np.asarray(xs, np.float64)))


def _ref_grad(mu, tau, E, sig, A, sigma_ref, N, rho):
    """기준: −(σ_ref²/N) Σ A log N(η(μ)) 의 autograd(기저·σ 는 상수)."""
    eta = ar1.innovations_torch(tau, mu, E, sig, rho)
    J = -(sigma_ref ** 2 / N) * (A * (-0.5 * eta ** 2 - 0.5 * math.log(2 * math.pi))).sum()
    return torch.autograd.grad(J, mu)[0]


def _loss_bug_cross_basis(mu, E, sig, eta, A, sigma_ref, N, rho):
    """심은 오류: 교차항에 e_{a,k}/σ_{a,k−1}·μ_{k−1} (e_{a,k−1} 대신 e_{a,k})."""
    kw = dict(dtype=mu.dtype, device=mu.device)
    E = torch.as_tensor(E, **kw)
    sig = torch.as_tensor(sig, **kw)
    eta = torch.as_tensor(eta, **kw)
    A = torch.as_tensor(A, **kw)
    c = torch.as_tensor(ar1.c_diag(rho, mu.shape[-2]), **kw)
    g = A * eta / c[:, None]
    p_cur = (E / sig[..., None] * mu[..., None, :]).sum(-1)
    p_x = (E[:, 1:] / sig[:-1, :, None] * mu[:, :-1, None, :]).sum(-1)          # 오류: E[k] 와 μ_{k−1}
    bracket = torch.cat([p_cur[:, :1], p_cur[:, 1:] - rho * p_x], 1)
    return -(sigma_ref ** 2 / N) * (g * bracket).sum()


def _weights(rng, n, mode):
    if mode == "B1":
        return np.repeat(rng.normal(size=(n, 1, 1)), 8, 1).repeat(2, 2)           # A = Â_i (균등)
    w = rng.uniform(0.2, 2.0, (n, 8))
    w = w / w.sum(1, keepdims=True) * 8                                           # Σ_k w = 8
    u0 = rng.uniform(0.0, 2.0, (n, 1))
    u = np.concatenate([u0, 2 - u0], 1)                                           # Σ_a u = 2
    return rng.normal(size=(n, 1, 1)) * w[:, :, None] * u[:, None, :]


def _scale(E, sig, eta, A, sigma_ref, N, rho):
    """요소별 항 크기 합 Σ|항| (∂L/∂μ 를 이루는 두 항의 절대값 합) — fp32 조건수 반영 오차의 분모."""
    g = np.abs(A * eta / ar1.c_diag(rho)[:, None])
    Es = np.abs(E / sig[..., None])
    out = (g[..., None] * Es).sum(-2)
    out[:, :-1] += rho * (g[:, 1:, :, None] * Es[:, :-1]).sum(-2)
    return (sigma_ref ** 2 / N) * out


def _run_g24(dtype, curv, mode, loss_fn=ar1.loss_rl_torch, trials=N_TRIALS, seed=0):
    """반환 (strict, cond): strict = 문서 기준(분모 max(|ref|, 1e-8‖ref‖∞)), cond = |g−ref| / Σ|항|(요소별).
    기준값 ref 는 항상 fp64(η(μ) 의 autograd) — 참값. 구현은 dtype 으로 계산(η 는 저장값을 그 dtype 으로)."""
    rng = np.random.default_rng(seed)
    rho, sigma_ref, n = ar1.RHO, ar1.SIGMA_REF, 4
    strict = cond = 0.0
    for tr in range(trials):
        kappa = float(rng.choice([0.5, 1.0, 2.0, 4.0]))
        sig = ar1.sigma_table(kappa)                                             # 축마다 다른 σ (lon 0.15, lat 0.05 기준)
        E = _bases(curv, phase=rng.uniform(-np.pi, np.pi), n=n)
        mu0 = np.cumsum(rng.normal([2.0, 0.0], [1.0, 0.5], (n, 8, 2)), 1)
        eta = rng.standard_normal((n, 8, 2))
        tau = ar1.sample_tau(mu0, E, sig, eta, rho)
        A = _weights(rng, n, mode)
        # 기준: μ = μ0 + δ 로 두고 δ(= 0)에 대해 미분 — τ − μ = d0 − δ, d0 = Σ_a σ ξ e (표본 변위, τ−μ0 상쇄 반올림 없음).
        # ∂/∂δ = ∂/∂μ 이므로 같은 기울기의 참값에 가깝다.
        d0 = ar1.sample_tau(np.zeros_like(mu0), E, sig, eta, rho)
        assert np.abs(d0 - (tau - mu0)).max() < 1e-12
        delta = torch.zeros(mu0.shape, dtype=torch.float64, requires_grad=True)
        ref = _ref_grad(delta, *(torch.tensor(x) for x in (d0, E, sig, A)), sigma_ref, n, rho)
        mu = torch.tensor(mu0, dtype=dtype, requires_grad=True)
        g = torch.autograd.grad(loss_fn(mu, *(torch.tensor(x, dtype=dtype) for x in (E, sig, eta, A)),
                                        sigma_ref, n, rho), mu)[0]
        assert torch.isfinite(g).all(), f"구현 기울기에 NaN/Inf (trial {tr})"
        assert torch.isfinite(ref).all(), f"기준 기울기에 NaN/Inf (trial {tr})"
        strict = _nan_max(strict, _relerr(g, ref))
        sc = _scale(E, sig, eta, A, sigma_ref, n, rho)
        cond = _nan_max(cond, (np.abs(g.double().numpy() - ref.numpy()) / sc).max())
    return strict, cond


# ---------------------------------------------------------------- G2-4
def test_g24_exact_score_fp64():
    for mode in ("B1", "Full"):
        for curv in (0.3, 0.5):
            e, ec = _run_g24(torch.float64, curv, mode)
            # fp64 도 같은 상쇄로 문서 기준(분모 |ref|)은 ~1e-11 이 나온다(기준값 쪽 η(μ) = (τ−μ) 상쇄 반올림 포함).
            # 판정은 fp32 와 같은 항 크기 기준, 문서 기준 값은 출력한다.
            print(f"[G2-4 fp64] {mode} curv {curv}: 항 크기 기준 오차 최대 {ec:.2e} (기준 1e-12) | 문서 기준(분모 |ref|) {e:.2e}")
            assert ec <= TOL[torch.float64], (mode, curv, ec)


def test_g24_exact_score_fp32():
    for mode in ("B1", "Full"):
        for curv in (0.3, 0.5):
            e, ec = _run_g24(torch.float32, curv, mode)
            # 문서 기준(분모 max(|ref|, 1e-8‖ref‖∞))은 fp32 에서 원리상 달성 불가: 한 요소 = 항 1 − ρ·항 2 의 상쇄라
            # |ref| 가 항 크기보다 훨씬 작은 요소에서 fp32 반올림(≈ 6e-8 × 항 크기)이 1e-6 상대를 넘는다(실측 ~1e-4).
            # 그래서 fp32 판정은 항 크기 기준(|g−ref| ≤ 1e-6 · Σ|항|, 요소별)으로 하고, 문서 기준 값은 숨기지 않고 출력한다.
            print(f"[G2-4 fp32] {mode} curv {curv}: 항 크기 기준 오차 최대 {ec:.2e} (기준 1e-6) | "
                  f"문서 기준(분모 |ref|) {e:.2e} — fp32 상쇄로 미달(설계 문서 정정 필요)")
            assert ec <= TOL[torch.float32], (mode, curv, ec)


def test_g24_planted_cross_basis_bug_is_caught():
    for mode in ("B1", "Full"):
        for dt in (torch.float64, torch.float32):
            e, ec = _run_g24(dt, 0.3, mode, loss_fn=_loss_bug_cross_basis, trials=20)
            print(f"[G2-4 음성] e_k 오류 {mode} {dt} curv 0.3: 문서 기준 {e:.2e}, 항 크기 기준 {ec:.2e} → 잡혀야 함")
            assert e > 1e-3 and ec > 1e-3, (e, ec)
    # 직선이면 이 오류는 원리상 구분 불가(같은 기저) — 고곡률 입력이 필요한 이유
    e0, _ = _run_g24(torch.float64, 0.0, "Full", loss_fn=_loss_bug_cross_basis, trials=5)
    print(f"[G2-4 음성] 곡률 0 에서는 오차 {e0:.2e} (구분 불가 확인)")
    assert e0 <= 1e-12


def _loss_nan(mu, E, sig, eta, A, sigma_ref, N, rho):
    """심은 오류: 값은 정상 손실과 같지만 기울기가 전부 NaN(sqrt(0) 경로의 0·∞)."""
    z = (mu * 0.0).sum()
    return ar1.loss_rl_torch(mu, E, sig, eta, A, sigma_ref, N, rho) + torch.sqrt(z * z)


def test_g24_nan_gradient_fails():
    assert math.isnan(_nan_max(0.0, float("nan"))) and math.isnan(_nan_max(float("nan"), 0.0))
    assert max(0.0, float("nan")) == 0.0                     # 고친 이유: 파이썬 max 는 NaN 을 버린다
    for dt in (torch.float64, torch.float32):
        try:
            _run_g24(dt, 0.3, "Full", loss_fn=_loss_nan, trials=2)
        except AssertionError as ex:
            assert "NaN" in str(ex), ex
        else:
            raise AssertionError("NaN 기울기 구현이 G2-4 를 통과하면 안 된다")


def test_ego_axis_basis_central_tangent():
    # 원호 위 μ(경유점 간격 일정): 접선 = 중앙 차분 = 호 각의 중간, 끝은 전방·후방 차분
    phi = np.linspace(0.1, 0.8, 8)
    mu = np.stack([10 * np.sin(phi), 10 - 10 * np.cos(phi)], -1)
    E = ar1.ego_axis_basis(mu)
    th = np.arctan2(E[:, 0, 1], E[:, 0, 0])
    exp = np.empty(8)
    exp[1:-1] = 0.5 * (phi[2:] + phi[:-2])                 # 현 (μ_{j+1}−μ_{j−1}) 방향 = 두 각의 중간
    exp[0] = 0.5 * (phi[0] + phi[1])
    exp[-1] = 0.5 * (phi[-2] + phi[-1])
    np.testing.assert_allclose(th, exp, atol=1e-12)
    # 원점(라이다)에서 출발한다고 보지 않는다: μ 를 옆으로 옮겨도 기저가 같다
    np.testing.assert_allclose(ar1.ego_axis_basis(mu + [3.0, -7.0]), E, atol=1e-12)
    np.testing.assert_allclose(E[:, 1], np.stack([-E[:, 0, 1], E[:, 0, 0]], -1), atol=1e-12)
    # 정지 구간: 앞 3 점 정지 → 뒤쪽 첫 유효 접선, 끝 2 점 정지 → 직전 유효 접선, 전부 정지 → +x
    mu2 = np.zeros((8, 2))
    mu2[3:] = np.cumsum(np.tile([[1.0, 1.0]], (5, 1)), 0)
    mu2[7] = mu2[6]
    th2 = np.arctan2(*ar1.ego_axis_basis(mu2)[:, 0, ::-1].T)
    assert np.allclose(th2[2:], np.pi / 4) and np.allclose(th2[:2], np.pi / 4), th2
    np.testing.assert_allclose(ar1.ego_axis_basis(np.zeros((2, 8, 2)))[..., 0, :], np.broadcast_to([1.0, 0.0], (2, 8, 2)))


# ---------------------------------------------------------------- numpy·torch 공통
def test_numpy_torch_agree_and_analytic_grad():
    rng = np.random.default_rng(1)
    n = 6
    sig = ar1.sigma_table(1.0)
    E = _bases(0.35, n=n)
    mu0 = np.cumsum(rng.normal(1.0, 1.0, (n, 8, 2)), 1)
    eta = rng.standard_normal((n, 8, 2))
    A = _weights(rng, n, "Full")
    mu = torch.tensor(mu0, dtype=torch.float64, requires_grad=True)
    Lt = ar1.loss_rl_torch(mu, E, sig, eta, A)
    Ln = ar1.loss_rl_np(mu0, E, sig, eta, A)
    assert abs(Lt.item() - Ln) <= 1e-12 * max(1.0, abs(Ln)), (Lt.item(), Ln)
    g = torch.autograd.grad(Lt, mu)[0].numpy()
    ga = ar1.grad_mu_np(E, sig, eta, A)
    np.testing.assert_allclose(ga, g, rtol=1e-12, atol=1e-15)
    terms = ar1.loss_rl_terms_torch(mu, E, sig, eta, A)
    assert abs(terms.sum().item() - Lt.item()) < 1e-12


def test_sampling_roundtrip_and_cov():
    rng = np.random.default_rng(2)
    sig = ar1.sigma_table(2.0)
    E = _bases(0.3)[0]
    mu = np.cumsum(rng.normal(1, 1, (8, 2)), 0)
    eta = rng.standard_normal((20000, 8, 2))
    tau = ar1.sample_tau(mu, E, sig, eta)
    np.testing.assert_allclose(ar1.innovations(tau, mu, E, sig), eta, atol=1e-10)
    # ξ 의 공분산 = C Cᵀ (정상 AR(1) 은 ρ^{|i−j|})
    xi = ar1.xi_from_eta(eta)
    C = ar1.c_matrix()
    np.testing.assert_allclose(C @ C.T, ar1.RHO ** np.abs(np.subtract.outer(np.arange(8), np.arange(8))), atol=1e-12)
    emp = np.cov(xi[..., 0].T)
    assert np.abs(emp - C @ C.T).max() < 0.05, np.abs(emp - C @ C.T).max()
    # 축별 변위 크기 = σ
    d = ((tau - mu)[..., None, :] * E).sum(-1)
    np.testing.assert_allclose(d.std(0), sig, rtol=0.03)


def test_rng_determinism():
    a = ar1.sample_eta(7, 12, "abc123", 3, 8)
    b = ar1.sample_eta(7, 12, "abc123", 3, 8)
    c = ar1.sample_eta(7, 12, "abc123", 4, 8)
    d = ar1.sample_eta(7, 13, "abc123", 3, 8)
    assert a.shape == (8, 8, 2) and np.array_equal(a, b) and not np.array_equal(a, c) and not np.array_equal(a, d)
    assert ar1.token_hash("abc123") == ar1.token_hash("abc123") >= 0


def test_axis_basis_frames_and_flags():
    # 원호 중심선(ego 프레임, 반지름 20, 반시계)
    th = np.linspace(0, np.pi / 2, 400)
    cl = np.stack([20 * np.sin(th), 20 - 20 * np.cos(th)], -1)
    mu_ego = np.stack([18 * np.sin(np.linspace(0.05, 0.6, 8)), 20 - 18 * np.cos(np.linspace(0.05, 0.6, 8))], -1)
    mu_ego[7] = [-3.0, 0.0]                                   # 시작점 뒤 → 끝 연장
    mu_ego[6] = [10.0, 15.0]                                  # 멀리(> 5 m)
    E, info = ar1.axis_basis(mu_ego, cl)
    np.testing.assert_allclose(np.einsum("jax,jbx->jab", E, E), np.broadcast_to(np.eye(2), (8, 2, 2)), atol=1e-12)
    assert np.allclose(info["d"][:6], 2.0, atol=2e-3), info["d"]
    assert list(info["out_of_range"]) == [False] * 6 + [True, True] and info["beyond_end"][7] and not info["beyond_end"][6]
    # e_lat 는 왼쪽 법선(e_lon 을 +90° 회전)
    np.testing.assert_allclose(E[:, 1], np.stack([-E[:, 0, 1], E[:, 0, 0]], -1), atol=1e-12)
    # 라이다 프레임: lidar2ego = z 회전 ψ + 이동 → μ_lidar 로 넣으면 기저 = Rᵀ e_ego, d 는 같다
    psi, t = 0.4, np.array([1.2, -0.3, 1.8])
    M = np.eye(4)
    M[:2, :2] = [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]
    M[:3, 3] = t
    mu_lidar = (mu_ego - t[:2]) @ M[:2, :2]                   # p_lidar = Rᵀ (p_ego − t)
    E2, info2 = ar1.axis_basis(mu_lidar, cl, lidar2ego=M)
    np.testing.assert_allclose(info2["d"], info["d"], atol=1e-9)
    np.testing.assert_allclose(E2 @ M[:2, :2].T, E, atol=1e-12)
    # 기저를 라이다로 돌린 표본을 ego 로 옮기면 ego 기저 표본과 같다
    sig, eta = ar1.sigma_table(), np.random.default_rng(3).standard_normal((8, 2))
    tau_l = ar1.sample_tau(mu_lidar, E2, sig, eta)
    np.testing.assert_allclose(tau_l @ M[:2, :2].T + t[:2], ar1.sample_tau(mu_ego, E, sig, eta), atol=1e-9)
    bad = np.eye(4)
    bad[:3, :3] = [[1, 0, 0], [0, np.cos(0.2), -np.sin(0.2)], [0, np.sin(0.2), np.cos(0.2)]]
    try:
        ar1.axis_basis(mu_ego, cl, lidar2ego=bad)
        raise AssertionError("z 축이 아닌 회전은 거부해야 함")
    except ValueError:
        pass


def test_kl():
    rng = np.random.default_rng(4)
    sig = ar1.sigma_table(1.0)
    E = _bases(0.3, n=3)
    mu_ref = np.cumsum(rng.normal(1, 1, (3, 8, 2)), 1)
    assert np.allclose(ar1.kl_np(mu_ref, mu_ref, E, sig), 0)
    # μ − μ_ref = 한 표본 변위(η) 이면 KL = ½‖η‖²
    eta = rng.standard_normal((3, 8, 2))
    mu = ar1.sample_tau(mu_ref, E, sig, eta)
    np.testing.assert_allclose(ar1.kl_np(mu, mu_ref, E, sig), 0.5 * (eta ** 2).sum((1, 2)), rtol=1e-10)
    kt = ar1.kl_torch(torch.tensor(mu), mu_ref, E, sig).numpy()
    np.testing.assert_allclose(kt, ar1.kl_np(mu, mu_ref, E, sig), rtol=1e-12)
    # 교차항 사영: Δ_{k−1} 은 e_{a,k−1} 에(오류 판: e_{a,k} 에 사영하면 고곡률에서 다름)
    D_bug = ((mu - mu_ref)[..., None, :] * E).sum(-1)
    D_bug_x = ((mu - mu_ref)[:, :-1, None, :] * E[:, 1:]).sum(-1) / sig[:-1]
    v = np.concatenate([D_bug[:, :1] / sig[:1], D_bug[:, 1:] / sig[1:] - ar1.RHO * D_bug_x], 1) / ar1.c_diag()[:, None]
    assert not np.allclose(0.5 * (v ** 2).sum((1, 2)), ar1.kl_np(mu, mu_ref, E, sig), rtol=1e-3)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print(f"PASS {f.__name__}", flush=True)
    print(f"모든 시험 통과 ({len(fns)})")
