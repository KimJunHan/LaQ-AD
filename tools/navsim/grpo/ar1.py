# 수정(2026-11-06): C3 GRPO 설계 v4 §2(정책·손실)·§8(KL) 구현 M1 — 신규 파일.
"""AR(1) 차선축 가우시안 정책: 축 기저, σ, η 표본, 혁신 분해, 정확 점수 손실 L_RL, 감시용 KL (docs/c3_grpo_design.md §2, §8).

기호(설계 v4 머리말과 같다)
  j: 경유점 0..7, k: 혁신 0..7, a ∈ {0 = lon, 1 = lat}. 경유점 0.5 s × 8.
  E[..., j, a, :]  경유점 j 의 축 a 단위벡터(2,), sig[j, a] = σ_{a,j}, c_kk[k] = (k = 0 → 1, 아니면 √(1−ρ²)).
  ξ_{0,a} = η_{0,a},  ξ_{j,a} = ρ ξ_{j−1,a} + √(1−ρ²) η_{j,a},  τ_j = μ_j + Σ_a σ_{a,j} ξ_{j,a} e_{a,j}.

좌표계(구현 기본값)
  중심선은 ego 프레임(extract_pdm_centerlines 의 CenterlineStore, +x 전진, +y 왼쪽)이다.
  μ 는 모델 계획 좌표(라이다 프레임)다. axis_basis 는 μ 를 lidar2ego 로 ego 에 옮겨 사영하고, 얻은 축 벡터를 lidar2ego 회전의
  역(전치)으로 라이다 프레임에 되돌린다. 손실·표본·KL 은 모두 라이다 프레임에서 계산한다(NAVSIM 은 lidar2ego = 단위행렬로 확인됨).
  기저·σ 는 상수(detach)다. torch 함수는 E·sig·eta 를 detach 해서 쓴다.

사영 규칙(구현 기본값)
  경유점을 중심선 폴리라인의 최근접 선분에 사영하고 그 선분 방향을 e_lon, 왼쪽 법선을 e_lat 로 쓴다.
  폴리라인 끝을 지나면 끝 접선 연장선 기준(접선 = 끝 선분 방향). CenterlineStore 격자(−10..+120 m)는 원 경로 끝 밖이 이미 끝 접선
  직선 연장이므로 격자 전체를 폴리라인으로 쓰고, valid·s_grid 를 주면 원 경로 밖(연장 구간)에 사영된 점을 beyond_end 로 표시한다.
  |d| > 5 m 이어도 같은 최근접 선분 접선을 쓴다.
  두 경우 모두 out_of_range 와 횡거리 d 를 돌려준다(버리지 않는다 — 호출자가 기록).

손실(§2.3, 문서 식 그대로)
  L_RL = −(σ_ref²/N) Σ_i Σ_{k,a} A_{i,k,a} (η_{i,k,a}/c_kk) [⟨e_{a,k}/σ_{a,k}, μ_k⟩ − ρ ⟨e_{a,k−1}/σ_{a,k−1}, μ_{k−1}⟩]
  (k = 0 이면 둘째 항 없음). 둘째 항은 경유점 k−1 의 기저·σ·μ 다.
  ∂L/∂μ = −(σ_ref²/N) Σ A ∇_μ log N(η(μ)) (기저·σ 고정) — tests/test_ar1.py 가 autograd 와 요소별로 대조한다.

KL(§8, 감시만)
  KL ≈ ½ Σ_{k,a} ((Δ_{k,a}/σ_{a,k} − ρ Δ_{k−1,a}/σ_{a,k−1}) / c_kk)²,  Δ_{k,a} = ⟨μθ,k − μ_ref,k, e_{a,k}⟩ (θ 측 기저, 각 경유점 자기 기저).

η RNG(구현 기본값): np.random.default_rng([run_seed, step, token_hash(token), mode]) → (G, 8, 2) 표준 정규.
"""
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_pdm_centerlines import project  # noqa: E402  (numpy 전용 사영 도우미, 다른 에이전트 소유 — 읽기만)

T = 8
LON, LAT = 0, 1
RHO = 0.8
SIG_BASE = (0.15, 0.05)        # [m] σ_lon,j = κ·0.15·√(j+1), σ_lat,j = κ·0.05·√(j+1)
SIGMA_REF = 0.15               # [m] §2.3 기본값(G2-5 규칙으로 한 번 고칠 수 있음)
MAX_LAT = 5.0                  # [m] 사영 신뢰 한계(out_of_range 표시용)


# ---------------------------------------------------------------- 상수
def sigma_table(kappa=1.0, T_=T):
    """(T, 2) σ[j, a]."""
    r = np.sqrt(np.arange(T_) + 1.0)
    return np.stack([kappa * SIG_BASE[0] * r, kappa * SIG_BASE[1] * r], -1)


def c_diag(rho=RHO, T_=T):
    c = np.full(T_, math.sqrt(1.0 - rho ** 2))
    c[0] = 1.0
    return c


def c_matrix(rho=RHO, T_=T):
    """(T, T) 하삼각 C[j, k] = ρ^{j−k} c_kk (j ≥ k): ξ_j = Σ_k C[j,k] η_k."""
    c = c_diag(rho, T_)
    j, k = np.meshgrid(np.arange(T_), np.arange(T_), indexing="ij")
    return np.where(j >= k, rho ** np.clip(j - k, 0, None) * c[None, :], 0.0)


def token_hash(token):
    """토큰 문자열 → 63-bit 비음수 정수(프로세스·파이썬 해시 시드와 무관)."""
    return int.from_bytes(hashlib.sha256(str(token).encode()).digest()[:8], "little") >> 1


def sample_eta(run_seed, step, token, mode, G, T_=T):
    """η (G, T, 2) ~ N(0,1), RNG = default_rng([run_seed, step, token_hash, mode])."""
    for v in (run_seed, step, mode):
        if int(v) < 0:
            raise ValueError("run_seed·step·mode 는 비음수 정수")
    rng = np.random.default_rng([int(run_seed), int(step), token_hash(token), int(mode)])
    return rng.standard_normal((G, T_, 2))


# ---------------------------------------------------------------- 기저
def _rot2(lidar2ego):
    """lidar2ego (4,4)|(3,3)|(2,2) → (R2 (2,2), t2 (2,)). 3D 회전은 z 축 회전이어야 한다(아니면 오류 — 2D 기저가 정의 안 됨)."""
    M = np.asarray(lidar2ego, np.float64)
    if M.shape == (4, 4):
        R, t = M[:3, :3], M[:3, 3]
    elif M.shape == (3, 3):
        R, t = M, np.zeros(3)
    elif M.shape == (2, 2):
        return M.copy(), np.zeros(2)
    else:
        raise ValueError(f"lidar2ego 형식 {M.shape}")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise ValueError("lidar2ego 회전이 직교가 아니다")
    if abs(R[2, 2] - 1.0) > 1e-6:
        raise ValueError(f"lidar2ego 가 z 축 회전이 아니다(R[2,2]={R[2, 2]:.6f}) — 2D 기저 회전 정의 불가")
    return R[:2, :2].copy(), t[:2].copy()


def axis_basis(mu_lidar, cl_xy_ego, lidar2ego=None, s_grid=None, valid=None, max_lat=MAX_LAT):
    """μ (..., T, 2) 라이다 프레임(detach 한 값) → E (..., T, 2(axis), 2(xy)) 라이다 프레임, info dict.
    info: d (..., T) 부호 횡거리(+왼쪽, ego 기준 m), s (..., T), out_of_range (..., T) bool(끝 밖 또는 |d| > max_lat),
          beyond_end (..., T) bool(끝 밖), tangent_ego (..., T)."""
    if valid is not None and s_grid is None:
        raise ValueError("valid 를 주면 s_grid 도 줘야 한다(CenterlineStore 격자: 끝 밖 연장 점 구간 판정)")
    mu = np.asarray(mu_lidar, np.float64)
    R, t = (np.eye(2), np.zeros(2)) if lidar2ego is None else _rot2(lidar2ego)
    p_ego = mu @ R.T + t
    pr = project(p_ego, cl_xy_ego, s_grid=s_grid, valid=valid, max_lat=np.inf)
    th = pr["tangent"]
    e_lon = np.stack([np.cos(th), np.sin(th)], -1)
    e_lat = np.stack([-np.sin(th), np.cos(th)], -1)
    E_ego = np.stack([e_lon, e_lat], -2)                      # (..., T, 2, 2)
    E = E_ego @ R                                             # 행벡터 v_ego → v_lidar = Rᵀ v_ego  ⇔ v_ego @ R
    beyond = pr["out_of_range"]                               # max_lat = inf 이므로 끝 밖만
    oor = beyond | (np.abs(pr["d"]) > max_lat)
    return E, dict(d=pr["d"], s=pr["s"], out_of_range=oor, beyond_end=beyond, tangent_ego=th)


def ego_axis_basis(mu_lidar, min_move=0.05):
    """대조 R(RAD 식 ego 축, §12.1(7) "경유점별 μ 접선"): 경유점 j 의 heading = detach μ 의 접선.
    접선: 내부 j 는 중앙 차분 (μ_{j+1} − μ_{j−1})/2, 처음은 전방 차분 μ_1 − μ_0, 끝은 후방 차분 μ_{T−1} − μ_{T−2}.
    정지(차분 길이 < min_move [m/경유점])인 경유점은 가장 가까운 앞쪽 유효 접선(없으면 뒤쪽 첫 유효 접선)을 쓰고,
    전부 정지면 +x(라이다 프레임). 반환 (..., T, 2(axis), 2(xy))."""
    mu = np.asarray(mu_lidar, np.float64)
    T_ = mu.shape[-2]
    if T_ < 2:
        raise ValueError("경유점이 2 개 이상이어야 접선을 정의할 수 있다")
    dv = np.empty_like(mu)
    dv[..., 0, :] = mu[..., 1, :] - mu[..., 0, :]
    dv[..., -1, :] = mu[..., -1, :] - mu[..., -2, :]
    if T_ > 2:
        dv[..., 1:-1, :] = 0.5 * (mu[..., 2:, :] - mu[..., :-2, :])
    n = np.linalg.norm(dv, axis=-1)
    th = np.arctan2(dv[..., 1], dv[..., 0])
    flat_th, flat_ok = th.reshape(-1, T_), (n >= min_move).reshape(-1, T_)
    flat_o = np.zeros_like(flat_th)
    for b in range(flat_th.shape[0]):
        ok = np.flatnonzero(flat_ok[b])
        if len(ok) == 0:
            continue                                              # 전부 정지 → +x
        cur = flat_th[b, ok[0]]                                   # 앞쪽 유효 없음 → 뒤쪽 첫 유효
        for j in range(T_):
            if flat_ok[b, j]:
                cur = flat_th[b, j]
            flat_o[b, j] = cur
    out = flat_o.reshape(mu.shape[:-1])
    return np.stack([np.stack([np.cos(out), np.sin(out)], -1), np.stack([-np.sin(out), np.cos(out)], -1)], -2)


# ---------------------------------------------------------------- 표본·혁신 (numpy)
def xi_from_eta(eta, rho=RHO):
    eta = np.asarray(eta, np.float64)
    xi = np.empty_like(eta)
    xi[..., 0, :] = eta[..., 0, :]
    s = math.sqrt(1.0 - rho ** 2)
    for j in range(1, eta.shape[-2]):
        xi[..., j, :] = rho * xi[..., j - 1, :] + s * eta[..., j, :]
    return xi


def sample_tau(mu, E, sig, eta, rho=RHO):
    """μ (..., T, 2), E (..., T, 2, 2), sig (T, 2), η (G, ..., T, 2) 또는 (..., T, 2) → τ 같은 앞 모양."""
    xi = xi_from_eta(eta, rho)
    return np.asarray(mu) + ((sig * xi)[..., None] * E).sum(-2)


def innovations(tau, mu, E, sig, rho=RHO):
    """τ, μ → η (…, T, 2). E 는 각 경유점에서 정규직교여야 한다."""
    xi = ((np.asarray(tau) - np.asarray(mu))[..., None, :] * E).sum(-1) / sig
    v = xi.copy()
    v[..., 1:, :] = xi[..., 1:, :] - rho * xi[..., :-1, :]
    return v / c_diag(rho, xi.shape[-2])[:, None]


def score_weights_np(A, eta, rho=RHO):
    """정확 점수 손실의 혁신 계수 A η / c_kk (…, T, 2)."""
    return np.asarray(A) * np.asarray(eta) / c_diag(rho, np.asarray(eta).shape[-2])[:, None]


def loss_rl_np(mu, E, sig, eta, A, sigma_ref=SIGMA_REF, N=None, rho=RHO):
    """numpy L_RL 값(배치 앞축 = 표본 i). 모양: mu/E/eta/A 가 (N, T, …) 로 브로드캐스트."""
    mu = np.asarray(mu, np.float64)
    g = score_weights_np(A, eta, rho)                           # (N, T, 2)
    p = (E / sig[..., None] * mu[..., None, :]).sum(-1)         # (N, T, 2) ⟨e_{a,j}/σ_{a,j}, μ_j⟩
    bracket = p.copy()
    bracket[..., 1:, :] = p[..., 1:, :] - rho * p[..., :-1, :]
    Nn = g.shape[0] if N is None else N
    return -(sigma_ref ** 2 / Nn) * (g * bracket).sum()


def grad_mu_np(E, sig, eta, A, sigma_ref=SIGMA_REF, N=None, rho=RHO):
    """해석적 ∂L_RL/∂μ (N, T, 2)."""
    g = score_weights_np(A, eta, rho)                           # (N, T, 2)
    Es = E / sig[..., None]                                     # (N, T, 2, 2)
    out = (g[..., None] * Es).sum(-2)                           # 항 1: 경유점 k
    out[..., :-1, :] -= rho * (g[..., 1:, :, None] * Es[..., :-1, :, :]).sum(-2)   # 항 2: 혁신 k 가 경유점 k−1 에
    Nn = g.shape[0] if N is None else N
    return -(sigma_ref ** 2 / Nn) * out


def kl_np(mu, mu_ref, E, sig, rho=RHO):
    """감시용 근사 KL(§8), 앞축마다 (…,)."""
    D = ((np.asarray(mu) - np.asarray(mu_ref))[..., None, :] * E).sum(-1) / sig     # Δ_{j,a}/σ_{a,j}
    v = D.copy()
    v[..., 1:, :] = D[..., 1:, :] - rho * D[..., :-1, :]
    return 0.5 * ((v / c_diag(rho, D.shape[-2])[:, None]) ** 2).sum((-1, -2))


# ---------------------------------------------------------------- torch 판
def _torch():
    import torch
    return torch


def loss_rl_torch(mu, E, sig, eta, A, sigma_ref=SIGMA_REF, N=None, rho=RHO):
    """torch L_RL. mu (N, T, 2) 는 그래프를 가진 텐서, 나머지는 텐서/배열(모두 detach, mu 의 dtype·device 로 옮김)."""
    torch = _torch()
    kw = dict(dtype=mu.dtype, device=mu.device)
    E = torch.as_tensor(E, **kw).detach()
    sig = torch.as_tensor(sig, **kw).detach()
    eta = torch.as_tensor(eta, **kw).detach()
    A = torch.as_tensor(A, **kw).detach()
    c = torch.as_tensor(c_diag(rho, mu.shape[-2]), **kw)
    g = A * eta / c[:, None]                                    # (N, T, 2)
    p = (E / sig[..., None] * mu[..., None, :]).sum(-1)         # (N, T, 2)
    bracket = torch.cat([p[..., :1, :], p[..., 1:, :] - rho * p[..., :-1, :]], -2)
    Nn = g.shape[0] if N is None else N
    return -(sigma_ref ** 2 / Nn) * (g * bracket).sum()


def loss_rl_terms_torch(mu, E, sig, eta, A, sigma_ref=SIGMA_REF, N=None, rho=RHO):
    """기록용: (N, T, 2) 항별 손실 기여(합 = loss_rl_torch). 축별·혁신별 기울기 노름 계측에 쓴다."""
    torch = _torch()
    kw = dict(dtype=mu.dtype, device=mu.device)
    E = torch.as_tensor(E, **kw).detach()
    sig = torch.as_tensor(sig, **kw).detach()
    eta = torch.as_tensor(eta, **kw).detach()
    A = torch.as_tensor(A, **kw).detach()
    c = torch.as_tensor(c_diag(rho, mu.shape[-2]), **kw)
    p = (E / sig[..., None] * mu[..., None, :]).sum(-1)
    bracket = torch.cat([p[..., :1, :], p[..., 1:, :] - rho * p[..., :-1, :]], -2)
    Nn = mu.shape[0] if N is None else N
    return -(sigma_ref ** 2 / Nn) * (A * eta / c[:, None]) * bracket


def innovations_torch(tau, mu, E, sig, rho=RHO):
    """η(μ) — μ 의 함수로(G2-4 기준값 계산용)."""
    torch = _torch()
    kw = dict(dtype=mu.dtype, device=mu.device)
    E = torch.as_tensor(E, **kw)
    sig = torch.as_tensor(sig, **kw)
    tau = torch.as_tensor(tau, **kw)
    xi = ((tau - mu)[..., None, :] * E).sum(-1) / sig
    v = torch.cat([xi[..., :1, :], xi[..., 1:, :] - rho * xi[..., :-1, :]], -2)
    return v / torch.as_tensor(c_diag(rho, mu.shape[-2]), **kw)[:, None]


def kl_torch(mu, mu_ref, E, sig, rho=RHO):
    torch = _torch()
    kw = dict(dtype=mu.dtype, device=mu.device)
    E = torch.as_tensor(E, **kw).detach()
    sig = torch.as_tensor(sig, **kw)
    mu_ref = torch.as_tensor(mu_ref, **kw).detach()
    D = ((mu - mu_ref)[..., None, :] * E).sum(-1) / sig
    v = torch.cat([D[..., :1, :], D[..., 1:, :] - rho * D[..., :-1, :]], -2)
    c = torch.as_tensor(c_diag(rho, mu.shape[-2]), **kw)
    return 0.5 * ((v / c[:, None]) ** 2).sum((-1, -2))
