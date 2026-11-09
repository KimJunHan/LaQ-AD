# 수정(2026-11-05): 사용자 지시 "MCMC SFT 확인한대로 진행" — 새 파일. 설계 docs/mcmc_sft_plan_design.md (v2, 사용자 승인) 구현.
"""MCMC SFT 정답 생성(CPU 전용) — 기반 모델 train 덤프 위의 명시적 밀도 + 정확한 Metropolis–Hastings.

설계 v2 와 한 줄씩 대응:
  p(x | 장면) = Σ_j π_j N(x; μ_j, Σ)           μ_j = 덤프 modes(명령 그룹 16 모드, 누적 궤적), π = softmax(덤프 raw_logits; rescore 전)
  Σ (대각, 시점·축별)                          train 덤프 완비 표본에서 r = GT − 최근접 모드(6 시점 평균 L2 최소) 의 분산(np.var, ddof=0).
                                               (실행 전에 고정되는 train 전용 규칙; val 은 보지 않는다)
  # 수정(2026-11-05): 사용자 결정 "속도 구간별 Σ" — v2.1 의 단일 Σ 는 정차·저속 표본에도 주행 속도 수준의 퍼짐을 줘
    정답이 뒤로 가는 궤적이 됐다(검토 1: 끝점 0.5 m 미만 표본에서 5 cm 넘는 후진 GT 0.35 % → MCMC 67.5 %, 무작위 96.5 %).
    v2.2(--speed_bins, 기본 0.5,3,6,9 m/s): **현재** 자차 속도 = info ego_status[6:8](CAN 속도 vx,vy — 모델 입력 ego_status 와 같은 값,
    미래 GT 아님)의 노름으로 구간 b 를 정하고, Σ_b = 그 구간 완비 표본만의 잔차 분산(구간 × 시점 × 축). 밀도 p 와 제안 핵의 주변 std
    모두 그 표본 구간의 Σ_b 를 쓴다. 구간 경계는 잔차를 보기 전에 정한 고정값: 정차 근처 < 0.5 m/s, 그 위 3 m/s 폭, ≥ 9 m/s.
    (CAN 결측 표본은 ego_status 가 전부 0 이라 모델 입력과 같게 첫 구간에 든다.) 표본마다 Σ_b 가 고정이라 체인 안에서 제안은
    여전히 x 와 무관한 영평균 가우시안 → 대칭, 정확한 MH 그대로. --speed_bins "" 이면 v2.1(단일 Σ)과 비트 동일.
  C = {x : 6 시점 평균 L2(x, GT) ≤ ε  그리고  평가기 충돌 규칙상 (x 충돌 & ~GT 충돌) 이 어느 시점에도 없음}
  # 수정(2026-11-05): 사용자 결정 "C 에 후진 금지" (v2.3, --no_reverse_m 0.05; 음수면 끔 = v2.2):
    좌표 = nuScenes lidar 좌표(+x 오른쪽, +y 전방 — 덤프 modes/gt_traj 와 평가기 같은 좌표). 시점별 전방 변위
    Δy_t(x) = y_t − y_{t−1} (t = 0..5, y_{−1} = 0 = 현재 자차 위치). C 에 다음 조건을 더한다:
    "어떤 t 에서도 Δy_t(x) < −0.05 m 이면서 Δy_t(GT) ≥ −0.05 m 인 일은 없다" (GT 에 없는 5 cm 넘는 후진 금지).
    C 를 좁히기만 하므로 목표 p_C 가 바뀔 뿐 MH 는 그대로 정확하고, x = GT 이면 조건이 자명하게 성립(GT ∈ C, 매 표본 단언).
      충돌 판정 = tools/plan_scorer/subscores.py collision_steps(평가기 single_coll 과 같은 판정 — 같은 파일의 시험·전수 대조로 확인됨).
  초기 상태 = GT (GT ∈ C 를 매 표본 단언).
  제안 κ: m ~ U{0..5}, 시점 m..5 에 영평균 가우시안 변위 δ 를 더한다. 대칭 → 수용 A = min(1, p(x')/p(x))·1[x' ∈ C].
  # 수정(2026-11-05): 사용자 결정 "lr 1/10, 부드러운 잡음" — v2 의 시점별 독립 δ_t ~ N(0, σ_t²) 는 x* 를 GT 보다 덜 매끄럽게 했다
    (2차 차분 평균 0.43 vs GT 0.20). v2.1(--proposal smooth, 기본): 축별로 δ_{m..5} ~ N(0, K_m),
    K_ij = σ_i σ_j exp(−(t_i − t_j)² / (2ℓ²)), t = 0.5 s 간격, ℓ = --corr_len_s(기본 1.0 s).
    주변 std 는 그대로 σ_t(Σ 의 시점 std), 이웃 시점 상관 0.88·두 칸 0.61 → 저주파(부드러운) 변위.
    가우시안 제곱지수 핵은 양의 정부호라 공분산으로 유효하고, δ 분포가 x 와 무관한 영평균 가우시안이며 m 도 x 와 무관 →
    q(x'|x) = q(x|x') (대칭)이므로 수용식은 v2 와 같은 정확한 MH. 군 (d) 도 같은 제안을 쓴다.
    --proposal indep 는 v2 제안(기록·재현용).
  N = 10 단계, 표본당 체인 1 개, 최종 상태 x* = 정답.
  군 (d) 'randc': 같은 제안·같은 단계 수, 수용 = 1[x' ∈ C] 만(밀도 없음 = C 위 균등 분포를 목표로 하는 MH).
  미래 GT 불완전 표본(gt_mask 미완) = GT 그대로(충돌 판정 불가 — 평가기도 제외).

난수: 표본마다 독립 생성기 np.random.default_rng([seed, arm_id, idx]) — 작업자 수·순서와 무관하게 재현된다.
  표본당 소비 순서 고정: 단계마다 m(integers) → δ(standard_normal (6−m,2)) → u(random). 두 군은 arm_id 로 다른 흐름.

출력(--out 아래):
  sigma.json                    Σ std/var/평균(시점·축), 표본 수
  targets_<arm>.npz             token, idx, valid, gt_ego_plan_sft_2hz (N,6,2) 시점별 변위(학습 키), traj (N,6,2) 누적,
                                n_accept, logp_gt, logp_star, l2_star(6 시점 평균) — arm ∈ {mcmc, randc}
  stats.json                    C 통계(모드·표본), 군별 수용률·밀도·L2 통계, 평가기 shapely 재검사 결과
  meta.json                     입력 서명(덤프 manifest·체크포인트·info pkl), 인자, 시드, 이 파일 sha256
학습 키 형식: info gt_ego_fut_trajs 와 같은 시점별 변위. disp = info 변위 + diff(Δ) (Δ = x* − GT 누적, Δ_{-1}=0) —
  바뀌지 않은 시점(Δ_t = Δ_{t−1} = 0)은 info 값과 비트 동일, 체인이 한 번도 수용하지 않은 표본·불완전 표본은 표본 전체가 비트 동일.

사용: PYTHONPATH=. OPENBLAS_CORETYPE=SkylakeX python tools/mcmc_sft/gen_mcmc_targets.py \
        --dump results/hipad_nusc_laq_v2_teacher_stage2_e3_nuim_ema_plan_scorer/dump_train \
        --info data/infos/nuscenes_infos_train.pkl --out results/mcmc_sft/nuim_e3_ema [--workers 16]
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "tools", "plan_scorer"))
import scorer_io  # noqa: E402
import subscores  # noqa: E402

ARMS = {"mcmc": 0, "randc": 1}
T = 6


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--info", required=True, help="train info pkl (덤프 manifest ann_file 과 같아야 함)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--proposal", choices=["smooth", "indep"], default="smooth")  # 수정(2026-11-05): 사용자 결정 "lr 1/10, 부드러운 잡음"
    ap.add_argument("--corr_len_s", type=float, default=1.0)  # 수정(2026-11-05): 사용자 결정 "lr 1/10, 부드러운 잡음" — 제곱지수 핵 길이 ℓ(초)
    ap.add_argument("--speed_bins", default="0.5,3,6,9", help="현재 속도 구간 경계(m/s, 쉼표). 빈 문자열 = 단일 Σ(v2.1)")  # 수정(2026-11-05): 사용자 결정 "속도 구간별 Σ"
    ap.add_argument("--no_reverse_m", type=float, default=0.05)  # 수정(2026-11-05): 사용자 결정 "C 에 후진 금지" — 음수면 후진 조건 끔(v2.2)
    ap.add_argument("--limit", type=int, default=0, help=">0 이면 처음 N 표본만(스모크; 출력에 smoke 표시)")
    return ap.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mean_l2(x, gt):
    """x (...,T,2), gt (T,2) → 6 시점 평균 유클리드 거리(float64)."""
    return np.linalg.norm(np.asarray(x, np.float64) - np.asarray(gt, np.float64), axis=-1).mean(-1)


def log_density(x, mu, log_pi, inv_var, log_norm):
    """log Σ_j π_j N(x; μ_j, diag(var)). x (T,2), mu (M,T,2) float64."""
    d = np.asarray(x, np.float64)[None] - mu
    lj = log_pi - 0.5 * (d * d * inv_var[None]).sum((1, 2)) + log_norm
    mx = lj.max()
    return float(mx + np.log(np.exp(lj - mx).sum()))


# ------------------------------------------------------------------------------------------------ 작업자
_G = {}


def _init(g):
    import torch
    torch.set_num_threads(1)
    _G.update(g)


def fwd_step(x):
    """수정(2026-11-05): 사용자 결정 "C 에 후진 금지" — (…,T,2) 누적 궤적 → (…,T) 시점별 전방(+y, lidar) 변위, 기준 y_{-1}=0."""
    y = np.asarray(x, np.float64)[..., 1]
    return np.diff(y, axis=-1, prepend=0.0)


def reverse_ok(x, gt, thr):
    """수정(2026-11-05): 사용자 결정 "C 에 후진 금지" — GT 에 없는 thr 넘는 후진이 없으면 True. thr < 0 이면 항상 True(조건 끔)."""
    if thr < 0:
        return np.ones(np.asarray(x).shape[:-2], bool) if np.asarray(x).ndim > 2 else True
    bad = (fwd_step(x) < -thr) & (fwd_step(gt) >= -thr)
    return ~bad.any(-1) if bad.ndim > 1 else not bool(bad.any())


def in_C(x, gt, fut, gt_col, eps, rev_thr=-1.0):
    """반환 (in_C, l2_ok, col_ok, rev_ok)."""
    l2_ok = bool(mean_l2(x, gt) <= eps)
    col = subscores.collision_steps(np.asarray(x, np.float32)[None], fut)[0] & ~gt_col
    col_ok = not bool(col.any())
    rev_ok = bool(reverse_ok(x, gt, rev_thr))  # 수정(2026-11-05): 사용자 결정 "C 에 후진 금지"
    return l2_ok and col_ok and rev_ok, l2_ok, col_ok, rev_ok


def proposal_chol(sigma, proposal, corr_len_s):
    """# 수정(2026-11-05): 사용자 결정 "lr 1/10, 부드러운 잡음" — m 별·축별 접미 공분산의 촐레스키 인자 L[m][a] (T−m, T−m). δ[m:, a] = L[m][a] @ z."""
    t = 0.5 * np.arange(1, T + 1)
    out = []
    for m in range(T):
        row = []
        for a_ in range(2):
            s_ = sigma[m:, a_]
            if proposal == "indep":
                K = np.diag(s_ ** 2)
            else:
                d = t[m:, None] - t[None, m:]
                K = np.outer(s_, s_) * np.exp(-d ** 2 / (2 * corr_len_s ** 2))
            row.append(np.linalg.cholesky(K))  # 양의 정부호가 아니면 여기서 실패(대체 없음)
        out.append(row)
    return out


def run_chain(arm, i, gt, fut, gt_col, mu, log_pi, sigma, inv_var, log_norm, eps, n_steps, seed, chol, rev_thr=-1.0):
    rng = np.random.default_rng([seed, ARMS[arm], int(i)])
    x = gt.astype(np.float32).copy()
    lp = log_density(x, mu, log_pi, inv_var, log_norm)
    c = dict(prop=0, acc=0, prop_in_C=0, prop_l2_ok=0, prop_col_ok=0, prop_rev_ok=0, dens_pass=0)
    for _ in range(n_steps):
        m = int(rng.integers(0, T))
        delta = np.zeros((T, 2), np.float64)
        z = rng.standard_normal((T - m, 2))  # 난수 소비 순서는 v2 와 같다
        for a_ in range(2):  # 수정(2026-11-05): 사용자 결정 "lr 1/10, 부드러운 잡음" — 축별 상관 가우시안(indep 이면 대각 = v2 와 같은 값)
            delta[m:, a_] = chol[m][a_] @ z[:, a_]
        xn = (x.astype(np.float64) + delta).astype(np.float32)  # 체인 상태 = float32(저장·충돌 판정과 같은 값)
        u = rng.random()
        lpn = log_density(xn, mu, log_pi, inv_var, log_norm)
        ok, l2_ok, col_ok, rev_ok = in_C(xn, gt, fut, gt_col, eps, rev_thr)  # 수정(2026-11-05): 사용자 결정 "C 에 후진 금지"
        c["prop"] += 1; c["prop_in_C"] += ok; c["prop_l2_ok"] += l2_ok; c["prop_col_ok"] += col_ok
        c["prop_rev_ok"] += rev_ok
        log_ratio = (lpn - lp) if arm == "mcmc" else 0.0
        dens = np.log(u) < log_ratio if u > 0 else True  # u ∈ [0,1): A = min(1, ratio) ⇔ u < ratio
        c["dens_pass"] += bool(dens)
        if dens and ok:
            x, lp = xn, lpn
            c["acc"] += 1
    return x, lp, c


def _work(i):
    g = _G
    gt = g["gt"][i]
    fut = g["fut"][i]
    mu = g["modes"][i].astype(np.float64)
    log_pi = g["log_pi"][i]
    eps = g["eps"]
    # 모드의 C 통계(같은 판정)
    gt_col = subscores.collision_steps(gt[None], fut)[0]
    ok_gt = in_C(gt, gt, fut, gt_col, eps, g["rev_thr"])[0]  # 수정(2026-11-05): 사용자 결정 "C 에 후진 금지" — GT ∈ C 단언(후진 조건 포함)
    if not ok_gt:
        raise AssertionError(f"색인 {i}: GT 가 C 밖(설계상 불가능)")
    pc = subscores.collision_steps(g["modes"][i], fut) & ~gt_col[None]
    mode_l2 = mean_l2(g["modes"][i], gt)
    mode_in_C = (mode_l2 <= eps) & ~pc.any(1) & reverse_ok(g["modes"][i], gt, g["rev_thr"])  # 수정(2026-11-05): 사용자 결정 "C 에 후진 금지"
    bi = int(g["bin"][i])  # 수정(2026-11-05): 사용자 결정 "속도 구간별 Σ" — 이 표본의 속도 구간
    inv_var, log_norm, sig, chol = g["inv_var"][bi], g["log_norm"][bi], g["sigma"][bi], g["chol"][bi]
    lp_gt = log_density(gt, mu, log_pi, inv_var, log_norm)
    out = dict(i=i, mode_in_C=mode_in_C, mode_l2_ok=mode_l2 <= eps, lp_gt=lp_gt,
               gt_min_mode_l2=float(mode_l2.min()), arms={})
    for arm in ARMS:
        x, lp, c = run_chain(arm, i, gt, fut, gt_col, mu, log_pi, sig, inv_var, log_norm, eps,
                             g["n_steps"], g["seed"], chol, g["rev_thr"])
        # 평가기 원 함수(shapely)로 최종 상태 재검사 — 벡터화 판정과 다르면 중단
        ev_x = subscores.evaluator_collision_steps(x[None], fut)[0]
        ev_g = subscores.evaluator_collision_steps(gt[None], fut)[0]
        ev_col = bool((ev_x & ~ev_g).any())
        if not reverse_ok(x, gt, g["rev_thr"]):  # 수정(2026-11-05): 사용자 결정 "C 에 후진 금지"
            raise AssertionError(f"색인 {i} {arm}: 최종 상태에 GT 에 없는 후진")
        if ev_col or mean_l2(x, gt) > eps:
            raise AssertionError(f"색인 {i} {arm}: 최종 상태가 C 밖(평가기 재검사 충돌 {ev_col}, L2 {mean_l2(x, gt):.4f})")
        out["arms"][arm] = dict(x=x, lp=lp, c=c, l2=float(mean_l2(x, gt)),
                                min_mode_l2=float(mean_l2(g["modes"][i], x).min()))
    return out


# ------------------------------------------------------------------------------------------------ 본체
def main():
    a = parse()
    t0 = time.time()
    dump = scorer_io.load_dump(a.dump)
    man = dump["manifest"]
    if man.get("split") != "train" or man.get("smoke"):
        raise SystemExit(f"덤프 분할 {man.get('split')} smoke={man.get('smoke')} — train 전체 덤프만 쓴다(val 금지)")
    if os.path.realpath(a.info) != os.path.realpath(os.path.join(_ROOT, man["ann_file"])) and \
            os.path.basename(a.info) != os.path.basename(man["ann_file"]):
        raise SystemExit(f"--info {a.info} 가 덤프 ann_file {man['ann_file']} 와 다르다")
    with open(a.info, "rb") as f:  # 프로젝트 자체 info pkl(데이터셋과 같은 파일) — 외부 입력 아님
        d = pickle.load(f)
    infos = d["infos"] if isinstance(d, dict) else d
    N = len(dump["idx"])
    if len(infos) != N:
        raise SystemExit(f"info {len(infos)} != 덤프 {N}")
    info_disp = np.stack([np.asarray(x["gt_ego_fut_trajs"], np.float32) for x in infos])  # (N,6,2)
    info_tok = np.array([x["token"] for x in infos])
    if not np.array_equal(info_tok, dump["token"]):
        raise SystemExit("info 토큰 순서가 덤프와 다르다")
    if not np.array_equal(info_disp.cumsum(1)[:, :T, :2], dump["gt_traj"]):
        raise SystemExit("덤프 gt_traj != cumsum(info gt_ego_fut_trajs) — 입력 불일치")
    valid = (dump["gt_mask"] == 1).all(1)
    info_valid = np.stack([(np.asarray(x["gt_ego_fut_masks"]) == 1).all() for x in infos])
    if not np.array_equal(valid, info_valid):
        raise SystemExit("덤프 gt_mask 와 info 마스크가 다르다")

    # ---- Σ (train 덤프 완비 표본 전체; --limit 와 무관하게 전체로 고정) ----
    # 수정(2026-11-05): 사용자 결정 "속도 구간별 Σ" — 현재 속도 구간별 Σ_b (구간 하나면 v2.1 과 같은 계산)
    edges = [float(e) for e in a.speed_bins.split(",") if e.strip()]
    if edges != sorted(edges) or any(e <= 0 for e in edges):
        raise SystemExit(f"--speed_bins {a.speed_bins}: 양수·오름차순이어야 한다")
    ego = np.stack([np.asarray(x["ego_status"], np.float64) for x in infos])
    if not np.isfinite(ego).all():
        raise SystemExit("ego_status 에 비유한 값 — 속도 구간을 정할 수 없다")
    speed = np.linalg.norm(ego[:, 6:8], axis=1)  # CAN 속도(자차 좌표 vx, vy) — 모델 입력 ego_status[6:8]
    sbin = np.digitize(speed, edges)  # 0..B-1
    B = len(edges) + 1
    gt_all = dump["gt_traj"].astype(np.float64)
    md_all = dump["modes"].astype(np.float64)
    sigma, inv_var, log_norm, bins_info = [], [], [], []
    os.makedirs(a.out, exist_ok=True)
    for b in range(B):
        sel = valid & (sbin == b)
        if sel.sum() < 100:
            raise SystemExit(f"속도 구간 {b} 완비 표본 {int(sel.sum())} < 100 — 구간 경계를 다시 정할 것")
        gtv = gt_all[sel]
        mv = md_all[sel]
        near = np.linalg.norm(mv - gtv[:, None], axis=-1).mean(-1)  # (Nb,M) 6 시점 평균 L2
        j = near.argmin(1)
        r = gtv - mv[np.arange(len(mv)), j]
        var = r.var(0)  # (T,2)
        if not (var > 0).all():
            raise SystemExit(f"속도 구간 {b}: 분산 0 인 시점·축")
        sigma.append(np.sqrt(var)); inv_var.append(1.0 / var)
        log_norm.append(float(-0.5 * np.log(2 * np.pi * var).sum()))
        lo = 0.0 if b == 0 else edges[b - 1]; hi = edges[b] if b < len(edges) else float("inf")
        bins_info.append({"bin": b, "speed_mps": [lo, hi], "n_valid": int(sel.sum()), "n_total": int((sbin == b).sum()),
                          "std_x": sigma[-1][:, 0].tolist(), "std_y": sigma[-1][:, 1].tolist(),
                          "mean_x": r.mean(0)[:, 0].tolist(), "mean_y": r.mean(0)[:, 1].tolist(), "var": var.tolist()})
        print(f"[Σ 구간 {b} {lo}-{hi} m/s, n={int(sel.sum())}] std x", np.round(sigma[-1][:, 0], 3),
              "y", np.round(sigma[-1][:, 1], 3), flush=True)
    sig_info = {"n_valid": int(valid.sum()), "n_total": int(N),
                "rule": "var(GT - nearest mode by mean 6-pt L2) within current-speed bin, train dump, ddof=0",
                "speed": "||info ego_status[6:8]|| (CAN vel, model input)", "edges_mps": edges,
                "n_can_all_zero": int((ego == 0).all(1).sum()), "bins": bins_info}
    json.dump(sig_info, open(os.path.join(a.out, "sigma.json"), "w"), indent=1)

    raw = dump["raw_logits"].astype(np.float64)
    log_pi = raw - raw.max(1, keepdims=True)
    log_pi = log_pi - np.log(np.exp(log_pi).sum(1, keepdims=True))

    run_idx = np.nonzero(valid)[0]
    if a.limit > 0:
        run_idx = run_idx[run_idx < a.limit]
    g = dict(gt=dump["gt_traj"].astype(np.float32), fut=dump["fut_boxes"], modes=dump["modes"].astype(np.float32),
             log_pi=log_pi, sigma=sigma, inv_var=inv_var, log_norm=log_norm, eps=a.eps, n_steps=a.n_steps, seed=a.seed,
             chol=[proposal_chol(sg, a.proposal, a.corr_len_s) for sg in sigma], bin=sbin,
             rev_thr=a.no_reverse_m)  # 수정(2026-11-05): 사용자 결정 "C 에 후진 금지"  # 수정(2026-11-05): 사용자 결정 "속도 구간별 Σ"
    res = {}
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(g,)) as pool:
            for k, o in enumerate(pool.imap_unordered(_work, run_idx.tolist(), chunksize=32)):
                res[o["i"]] = o
                if (k + 1) % 2000 == 0:
                    print(f"  {k + 1}/{len(run_idx)} {time.time() - t0:.0f}s", flush=True)
    else:
        _init(g)
        for i in run_idx.tolist():
            res[i] = _work(i)
    print(f"[체인] {len(res)} 표본 {time.time() - t0:.0f}s", flush=True)

    # ---- 저장·통계 ----
    stats = {"eps": a.eps, "n_steps": a.n_steps, "seed": a.seed, "n_total": int(N), "n_valid": int(valid.sum()),
             "n_run": int(len(run_idx)), "n_incomplete_keep_gt": int((~valid).sum()), "smoke": bool(a.limit > 0)}
    ri = np.array(sorted(res))
    mic = np.stack([res[i]["mode_in_C"] for i in ri])
    ml2 = np.stack([res[i]["mode_l2_ok"] for i in ri])
    stats["C_modes"] = {"frac_modes_in_C": float(mic.mean()), "frac_samples_no_mode_in_C": float((~mic.any(1)).mean()),
                        "frac_modes_l2_ok": float(ml2.mean()), "frac_samples_no_mode_l2_ok": float((~ml2.any(1)).mean()),
                        "mean_modes_in_C_per_sample": float(mic.sum(1).mean())}
    lp_gt = np.array([res[i]["lp_gt"] for i in ri])
    gt_min_mode = np.array([res[i]["gt_min_mode_l2"] for i in ri])
    stats["gt"] = {"logp_mean": float(lp_gt.mean()), "min_mode_l2_mean": float(gt_min_mode.mean())}
    for arm in ARMS:
        disp = info_disp.copy()
        traj = dump["gt_traj"].astype(np.float32).copy()
        n_acc = np.zeros(N, np.int64)
        lp_star = np.full(N, np.nan); lp_g = np.full(N, np.nan); l2s = np.zeros(N)
        cs = {k: 0 for k in ("prop", "acc", "prop_in_C", "prop_l2_ok", "prop_col_ok", "prop_rev_ok", "dens_pass")}
        for i in ri:
            o = res[i]["arms"][arm]
            x = o["x"]
            delta = x.astype(np.float64) - dump["gt_traj"][i].astype(np.float64)
            if o["c"]["acc"] > 0:
                dd = np.diff(np.concatenate([np.zeros((1, 2)), delta], 0), axis=0).astype(np.float32)
                disp[i, :T, :2] = info_disp[i, :T, :2] + dd
            traj[i] = x
            n_acc[i] = o["c"]["acc"]; lp_star[i] = o["lp"]; lp_g[i] = res[i]["lp_gt"]; l2s[i] = o["l2"]
            for k in cs:
                cs[k] += o["c"][k]
        recon = disp.cumsum(1)[:, :T, :2]
        err = float(np.abs(recon[ri] - traj[ri]).max())
        if err > 1e-4:
            raise AssertionError(f"{arm}: cumsum(disp) 와 x* 차 {err}")
        unchanged = n_acc == 0
        if not np.array_equal(disp[unchanged], info_disp[unchanged]):
            raise AssertionError(f"{arm}: 바뀌지 않은 표본의 변위가 info 와 비트 동일하지 않다")
        dlp = lp_star[ri] - lp_g[ri]
        l2r = l2s[ri]
        min_mode = np.array([res[i]["arms"][arm]["min_mode_l2"] for i in ri])
        l2_step = np.linalg.norm(traj[ri].astype(np.float64) - dump["gt_traj"][ri].astype(np.float64), axis=-1)
        ev = subscores.evaluator_metrics(l2_step, np.zeros_like(l2_step, bool))["L2"]
        stats[arm] = {
            "accept_rate": cs["acc"] / cs["prop"], "proposals": cs["prop"], "accepted": cs["acc"],
            "frac_prop_in_C": cs["prop_in_C"] / cs["prop"], "frac_prop_l2_ok": cs["prop_l2_ok"] / cs["prop"],
            "frac_prop_col_ok": cs["prop_col_ok"] / cs["prop"], "frac_prop_rev_ok": cs["prop_rev_ok"] / cs["prop"], "frac_density_pass": cs["dens_pass"] / cs["prop"],
            "frac_chains_unchanged": float(unchanged[ri].mean()), "mean_accepts_per_chain": float(n_acc[ri].mean()),
            "logp_star_minus_logp_gt": {"mean": float(dlp.mean()), "median": float(np.median(dlp)),
                                        "frac_gt0": float((dlp > 0).mean()), "frac_lt0": float((dlp < 0).mean())},
            "l2_star_gt_6pt": {"mean": float(l2r.mean()), "median": float(np.median(l2r)),
                               "p90": float(np.quantile(l2r, 0.9)), "max": float(l2r.max())},
            "l2_star_gt_evaluator_style": ev,
            "min_mode_l2_star_mean": float(min_mode.mean()), "min_mode_l2_gt_mean": float(gt_min_mode.mean()),
            "recon_max_err": err,
            # 수정(2026-11-05): 사용자 결정 "lr 1/10, 부드러운 잡음" — 매끄러움(누적 궤적 2차 차분 노름 평균; GT 와 비교)과 시점별 편차
            "second_diff_mean_star": float(np.linalg.norm(np.diff(traj[ri].astype(np.float64), 2, axis=1), axis=-1).mean()),
            "second_diff_mean_gt": float(np.linalg.norm(np.diff(dump["gt_traj"][ri].astype(np.float64), 2, axis=1), axis=-1).mean()),
            "step_dev_mean": l2_step.mean(0).tolist(), "step_dev_p99": np.quantile(l2_step, 0.99, axis=0).tolist(),
            "step_dev_max": l2_step.max(0).tolist(), "max_single_step_dev": float(l2_step.max()),
        }
        # 수정(2026-11-05): 사용자 결정 "속도 구간별 Σ" — 정차 표본 검사(검토 1 정의): 정차 = GT 끝점 |GT_5| < 0.5 m(완비 표본),
        #   후진 = 어떤 시점 변위의 전방(y) 성분 < −5 cm, 끝점 이동 = |x*_5 − GT_5|,
        #   yaw 규칙 뒤집힘 = 평가기 "|traj_5 − traj_0| < 0.5 m 이면 yaw = π/2" 판정이 x* 와 GT 에서 다른 비율(완비 전체·정차 안)
        gt_r = dump["gt_traj"][ri].astype(np.float64); xs_r = traj[ri].astype(np.float64)
        stat = np.linalg.norm(gt_r[:, -1], axis=1) < 0.5
        def _rev(t):
            dd = np.diff(np.concatenate([np.zeros((len(t), 1, 2)), t], 1), axis=1)
            return (dd[..., 1] < -0.05).any(1)
        still = lambda t: np.linalg.norm(t[:, -1] - t[:, 0], axis=1) < 0.5  # noqa: E731
        flip = still(xs_r) != still(gt_r)
        stats[arm]["stationary"] = {
            "frac_stationary": float(stat.mean()), "n_stationary": int(stat.sum()),
            "reverse_gt5cm_gt": float(_rev(gt_r)[stat].mean()), "reverse_gt5cm_star": float(_rev(xs_r)[stat].mean()),
            "reverse_gt5cm_star_all": float(_rev(xs_r).mean()), "reverse_gt5cm_gt_all": float(_rev(gt_r).mean()),
            # 수정(2026-11-05): 사용자 결정 "C 에 후진 금지" — GT 에 없는 후진(C 의 새 조건 위반) 비율: 조건을 켜면 0 이어야 한다
            "new_reverse_vs_gt_star_all": float((~reverse_ok(xs_r, gt_r, 0.05)).mean()),
            "new_reverse_vs_gt_star_stationary": float((~reverse_ok(xs_r, gt_r, 0.05))[stat].mean()),
            "endpoint_shift_mean": float(np.linalg.norm(xs_r[stat, -1] - gt_r[stat, -1], axis=1).mean()),
            "yaw_rule_flip_all": float(flip.mean()), "yaw_rule_flip_stationary": float(flip[stat].mean()),
        }
        stats[arm]["per_speed_bin"] = [{"bin": b, "n": int((sbin[ri] == b).sum()),
                                        "l2_star_mean": float(l2r[sbin[ri] == b].mean()),
                                        "reverse_gt5cm_star": float(_rev(xs_r)[sbin[ri] == b].mean())}
                                       for b in range(len(sigma))]
        np.savez(os.path.join(a.out, f"targets_{arm}.npz"), token=dump["token"], idx=dump["idx"], valid=valid,
                 gt_ego_plan_sft_2hz=disp, traj=traj, n_accept=n_acc, logp_gt=lp_g, logp_star=lp_star, l2_star=l2s,
                 smoke=np.array(bool(a.limit > 0)))
    stats["seconds"] = time.time() - t0
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    meta = {"args": vars(a), "dump_manifest": man, "info": scorer_io.file_sig(a.info), "script_sha256": sha256(__file__),
            "rng": "np.random.default_rng([seed, arm_id, idx]); arm_id mcmc=0 randc=1; per step m, z, u",
            "proposal": a.proposal, "corr_len_s": a.corr_len_s, "speed_bins": a.speed_bins, "no_reverse_m": a.no_reverse_m,
            "design": "docs/mcmc_sft_plan_design.md v2.3"}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()
