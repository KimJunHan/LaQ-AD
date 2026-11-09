# 수정(2026-11-06): 사용자 결정 "NAVSIM MCMC SFT" — 새 파일(기존 파일 무수정). 설계 docs/mcmc_sft_navsim_plan.md (확정판).
"""NAVSIM MCMC SFT 정답 생성(CPU 전용, navsim env) — 2단계 모델 navtrain 덤프 위의 명시적 밀도 + 정확한 Metropolis–Hastings.
nuScenes 판 tools/mcmc_sft/gen_mcmc_targets.py(설계 v2.3)와 한 줄씩 대응하고 바뀐 것만 적는다.

  궤적       8 점 @0.5 s(4 s), 라이다 = ego(+x 전방, +y 왼쪽, 후륜축 원점). 덤프 modes/gt_traj 와 채점기 입력이 같은 좌표.
  p(x)       Σ_j π_j N(x; μ_j, Σ_b) — μ = 덤프 modes(명령 묶음 16 모드, 누적), π = softmax(덤프 raw_logits = rescore 전 logit).
  Σ_b        현재 속도 구간 b 별 "GT − 최근접 모드(8 점 평균 L2)" 잔차 분산(np.var, ddof 0, 체인 대상 프레임 전체; --limit_chain 무관).
             속도 = ‖info ego_status[0:2]‖(NAVSIM ego_status = vx, vy, ax, ay; 비유한값 0 — 데이터셋 입력과 같은 규칙).
             경계 0.5/3/6/9 m/s(nuScenes 와 같은 사전 고정값). 구간당 표본 ≥ --min_bin(기본 100) 단언.
  C          {x : 8 점 평균 L2(x, GT) ≤ ε(0.5 m)
                  ∧ GT 에 없는 후진 없음: ∄t: f_t(x) < −0.05 m ∧ f_t(GT) ≥ −0.05 m,
                    f_t(y) = (y_t − y_{t−1}) · (cos h_{t−1}, sin h_{t−1}),  y_{−1} = 0,  h = GT 의 heading 규칙(이동 < 0.05 m 면 직전 heading,
                    h_{−1} = 0 = 현재 자차 +x). 직진 GT 에서는 f_t = Δx_t(사용자 표현 "+x 전방 기준 후진")와 같고, 회전 구간에서는
                    GT 진행 방향 기준이라 90° 넘게 도는 GT(navtrain 평가 프레임 0.33 %)의 정상 전진을 후진으로 오판하지 않는다.
                  ∧ s_k(x) ≥ s_k(GT), k ∈ {NC, DAC, DDC, TLC, TTC, LK, HC} (EP 제외 7 개) }
             s_k = tools/navsim/grpo_scorer.py score_token 의 subscores(공식 pdm_score 와 비트 동일, 사람 궤적 벌점 필터 적용 후 값).
             x 와 GT 모두 (8,2) → heading_from_xy 로 포즈를 만들어 채점(평가 plan_lidar_to_ego_poses 와 같은 규칙).
             GT 가 어떤 하위 점수에 실패해도(예 DAC 0) 정의는 그대로 상대 비교 — 그 하위 점수 조건은 "GT 보다 나쁘지 않음" 이라 자명해지고,
             표본은 버리지 않는다. 그런 표본 수를 하위 점수별로 stats 에 기록.
  초기 상태  GT(GT ∈ C 매 표본 단언). N = 10 단계, 표본당 체인 1 개, 최종 x* = 정답(10 단계 뒤 x* 는 p_C 표본이 아님 — v2.2 주의).
  제안       m ~ U{0..7}, 축별 δ_{m..7} ~ N(0, K_m), K_ij = σ_i σ_j exp(−(t_i − t_j)²/(2ℓ²)), t = 0.5·(1..8) s, ℓ = 1.0 s,
             σ = Σ_b 의 시점·축 std. x 와 무관한 영평균 가우시안 → 대칭, A = min(1, p(x')/p(x))·1[x' ∈ C].
  판정 순서  난수 소비 m → z → u (nuScenes 와 같음). 군 mcmc: u 와 밀도비를 먼저 보고, 통과분만 값싼 조건(L2·후진) → 통과분만 PDM 채점.
             수용 = 밀도 ∧ C 이므로 체인 값은 C 를 매번 계산할 때와 비트 동일(--no_lazy 로 확인 가능; 단위 시험이 확인).
             그래서 "제안 중 C 안 비율" 은 밀도 통과분 기준(stats 에 명시). 군 randc 는 밀도 없음(C 위 균등 목표) → 값싼 조건 통과분 전부 채점.
  프레임 상태(status, 정답 파일에 저장; 조용히 빠지는 프레임 없음):
     0 chain       평가 프레임 ∧ metric cache 있음 ∧ 미래 GT 8 점 완비 → 체인
     1 history     이력 전용 프레임(is_eval_frame False, navsim token 없음 → metric cache 가 정의상 없다) → 정답 = GT 그대로(비트 동일).
                   학습 손실에는 GT 로 들어간다(데이터셋은 152,495 프레임 전부를 쓴다).
     2 incomplete  평가 프레임이지만 미래 GT 마스크 미완(채점 불가) → GT 그대로
     평가 프레임에 metric cache 가 없으면 = 데이터 오류로 즉시 중단(navtrain 캐시 103,288/103,288 확인됨; 대체·건너뛰기 없음).
  최종 재검사 x*(바뀐 체인)와 GT 를 공식 pdm_score 단독 호출(grpo_scorer._official)로 다시 채점해 score_token 값과 비트 동일한지 단언 → C 재확인.

난수: 표본마다 np.random.default_rng([seed, arm_id, idx]) — 작업자 수·순서·재개와 무관하게 재현.
출력(--out): sigma.json, targets_{gt,mcmc[,randc]}.npz, stats.json, meta.json, (진행 중) _progress.pkl
  targets_*.npz: token (N) = info 전체 순서, idx, status, valid, gt_ego_plan_sft_2hz (N,8,2) float32 시점별 증분(학습 키),
                 traj (N,8,2) 누적, n_accept, logp_gt, logp_star, l2_star, sub_gt/sub_star (N,8)(체인 밖 NaN), epdms_gt/epdms_star, smoke.
  학습 키 = info 증분 + diff(x* − GT) — 바뀌지 않은 표본(체인 불변·status 1·2)은 info 와 비트 동일. targets_gt = info 그대로(기준·GT 점수 기록용).
  # 수정(2026-11-06): 사용자 결정 "GT-SFT 군 제거" — targets_gt 는 학습 군에 쓰지 않는다. #이전 "(GT-SFT 군)"
사용(navsim env, CPU 16):
  source scripts/navsim/navsim_env.sh
  PYTHONPATH=. $NAVSIM_PY tools/mcmc_sft/gen_mcmc_targets_navsim.py --dump results/mcmc_sft_navsim/dump_navtrain_s2e3nuim_gtfilt_ema \
      --info data/infos/navsim_20261019/navsim_navtrain.pkl --metric_cache $NAVSIM_EXP_ROOT/metric_cache_navtrain \
      --out results/mcmc_sft_navsim/s2e3nuim_gtfilt_ema_v1 --workers 16  # 수정(2026-11-09): GT 필터 2단계 이름(사용 예만)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools", "navsim"))

ARMS = {"mcmc": 0, "randc": 1}
T = 8
DT = 0.5
SUBSCORES = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
             "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
             "history_comfort"]                    # = grpo_scorer.SUBSCORES (main 에서 같음을 단언)
C_IDX = [0, 1, 2, 3, 5, 6, 7]                       # EP(4) 제외 7 개
STATUS = {0: "chain", 1: "history_no_cache", 2: "eval_incomplete_gt"}
HEADING_MIN_STEP = 0.05


# ------------------------------------------------------------------------------------------------ 순수 함수(단위 시험 대상)
def heading_xy(xy, min_step=HEADING_MIN_STEP):
    """(T,2) 누적 → (T,) heading. navsim_score_vocab_navtrain.heading_from_xy·NavsimDataset.plan_lidar_to_ego_poses 와 같은 규칙."""
    out = np.zeros(len(xy), np.float64)
    prev_xy, prev_h = np.zeros(2), 0.0
    for t in range(len(xy)):
        d = np.asarray(xy[t], np.float64) - prev_xy
        h = float(np.arctan2(d[1], d[0])) if np.hypot(*d) >= min_step else prev_h
        out[t] = h
        prev_xy, prev_h = np.asarray(xy[t], np.float64), h
    return out


def ref_dirs(gt):
    """GT 기준 진행 방향 단위벡터 u_t = (cos h_{t−1}, sin h_{t−1}), h_{−1} = 0. (T,2)"""
    h = heading_xy(gt)
    hp = np.concatenate([[0.0], h[:-1]])
    return np.stack([np.cos(hp), np.sin(hp)], -1)


def fwd_steps(x, u):
    """x (…,T,2) 누적 → (…,T) 시점별 전방 변위 (x_t − x_{t−1})·u_t, x_{−1} = 0."""
    x = np.asarray(x, np.float64)
    d = np.diff(x, axis=-2, prepend=np.zeros(x.shape[:-2] + (1, 2)))
    return (d * u).sum(-1)


def reverse_ok(x, gt, thr, u=None):
    """GT 에 없는 thr 넘는 후진이 없으면 True. x (T,2) 또는 (K,T,2). thr < 0 이면 조건 끔."""
    x = np.asarray(x)
    if thr < 0:
        return np.ones(x.shape[:-2], bool) if x.ndim > 2 else True
    u = ref_dirs(gt) if u is None else u
    bad = (fwd_steps(x, u) < -thr) & (fwd_steps(gt, u) >= -thr)
    return ~bad.any(-1) if bad.ndim > 1 else not bool(bad.any())


def mean_l2(x, gt):
    return np.linalg.norm(np.asarray(x, np.float64) - np.asarray(gt, np.float64), axis=-1).mean(-1)


def log_density(x, mu, log_pi, inv_var, log_norm):
    """log Σ_j π_j N(x; μ_j, diag(var)). x (T,2), mu (M,T,2) float64."""
    d = np.asarray(x, np.float64)[None] - mu
    lj = log_pi - 0.5 * (d * d * inv_var[None]).sum((1, 2)) + log_norm
    mx = lj.max()
    return float(mx + np.log(np.exp(lj - mx).sum()))


def proposal_chol(sigma, corr_len_s, n_t=T, dt=DT):
    """m 별·축별 접미 공분산 K_m 의 촐레스키 L[m][a]. δ[m:, a] = L[m][a] @ z. 양의 정부호가 아니면 실패(대체 없음)."""
    t = dt * np.arange(1, n_t + 1)
    out = []
    for m in range(n_t):
        row = []
        for a_ in range(2):
            s_ = sigma[m:, a_]
            d = t[m:, None] - t[None, m:]
            row.append(np.linalg.cholesky(np.outer(s_, s_) * np.exp(-d ** 2 / (2 * corr_len_s ** 2))))
        out.append(row)
    return out


def sub_in_C(sub_x, sub_gt):
    """7 하위 점수(EP 제외) 모두 GT 이상."""
    return bool(np.all(np.asarray(sub_x)[C_IDX] >= np.asarray(sub_gt)[C_IDX]))


def run_chain(arm, key, gt, mu, log_pi, inv_var, log_norm, chol, eps, n_steps, rev_thr, score_fn, sub_gt, lazy=True, ep_gt=float("nan")):
    """한 체인. score_fn(x (T,2) float32) → (subscores (8,), epdms). 반환 (x*, logp*, 카운터, x* 의 subscores, x* 의 epdms).
    lazy=False 면 매 제안에서 C 전체를 계산(시험용 — 체인 값은 같아야 한다)."""
    n_t = len(gt)
    rng = np.random.default_rng(key)
    u_dir = ref_dirs(gt)
    x = gt.astype(np.float32).copy()
    lp = log_density(x, mu, log_pi, inv_var, log_norm)
    sub_x = np.asarray(sub_gt, np.float64).copy()
    ep_x = ep_gt
    c = dict(prop=0, acc=0, dens_pass=0, cheap_eval=0, l2_ok=0, rev_ok=0, cheap_ok=0, scored=0, sub_ok=0,
             sub_fail=np.zeros(len(SUBSCORES), np.int64))
    for _ in range(n_steps):
        m = int(rng.integers(0, n_t))
        z = rng.standard_normal((n_t - m, 2))
        delta = np.zeros((n_t, 2), np.float64)
        for a_ in range(2):
            delta[m:, a_] = chol[m][a_] @ z[:, a_]
        xn = (x.astype(np.float64) + delta).astype(np.float32)   # 체인 상태 = float32(저장·채점과 같은 값)
        u = rng.random()
        c["prop"] += 1
        if arm == "mcmc":
            lpn = log_density(xn, mu, log_pi, inv_var, log_norm)
            dens = (np.log(u) < lpn - lp) if u > 0 else True
        else:
            lpn = log_density(xn, mu, log_pi, inv_var, log_norm)
            dens = True
        c["dens_pass"] += bool(dens)
        if not dens and lazy:
            continue
        c["cheap_eval"] += 1
        l2_ok = bool(mean_l2(xn, gt) <= eps)
        rv_ok = bool(reverse_ok(xn, gt, rev_thr, u_dir))
        c["l2_ok"] += l2_ok; c["rev_ok"] += rv_ok
        ok = l2_ok and rv_ok
        c["cheap_ok"] += ok
        sub_n, ep_n = None, None
        if ok or not lazy:
            sub_n, ep_n = score_fn(xn)
            sub_n = np.asarray(sub_n, np.float64)
            c["scored"] += 1
            s_ok = sub_in_C(sub_n, sub_gt)
            c["sub_ok"] += s_ok
            c["sub_fail"] += (sub_n < np.asarray(sub_gt)).astype(np.int64)
            ok = ok and s_ok
        if dens and ok:
            x, lp, sub_x, ep_x = xn, lpn, sub_n, ep_n
            c["acc"] += 1
    return x, lp, c, sub_x, ep_x


def sigma_by_bin(gt, modes, sel, sbin, n_bins, min_bin):
    """구간별 Σ_b: sel 표본에서 r = GT − 최근접 모드(8 점 평균 L2) 의 분산."""
    sigma, inv_var, log_norm, info = [], [], [], []
    for b in range(n_bins):
        s = sel & (sbin == b)
        if s.sum() < min_bin:
            raise SystemExit(f"속도 구간 {b} 표본 {int(s.sum())} < {min_bin} — 구간 경계를 다시 정할 것")
        g = gt[s].astype(np.float64); mv = modes[s].astype(np.float64)
        j = np.linalg.norm(mv - g[:, None], axis=-1).mean(-1).argmin(1)
        r = g - mv[np.arange(len(mv)), j]
        var = r.var(0)
        if not (var > 0).all():
            raise SystemExit(f"속도 구간 {b}: 분산 0 인 시점·축")
        sigma.append(np.sqrt(var)); inv_var.append(1.0 / var); log_norm.append(float(-0.5 * np.log(2 * np.pi * var).sum()))
        info.append({"bin": b, "n": int(s.sum()), "std_x_fwd": np.sqrt(var)[:, 0].tolist(), "std_y_lat": np.sqrt(var)[:, 1].tolist(),
                     "mean_x": r.mean(0)[:, 0].tolist(), "mean_y": r.mean(0)[:, 1].tolist(), "var": var.tolist()})
    return sigma, inv_var, log_norm, info


# ------------------------------------------------------------------------------------------------ 작업자
_G = {}


def _init(g):
    import grpo_scorer as GS
    GS._init_worker(False)   # margins=False: 하위 점수에는 영향 없음(cont 여유 거리만 끔)
    _G.update(g)
    _G["GS"] = GS


def _score_fn(mc):
    GS = _G["GS"]

    def f(x):
        r = GS.score_token(mc, GS._as_poses(np.asarray(x, np.float64)[None]), margins=False)
        return r["subscores"][0], float(r["epdms"][0])
    return f


def _work(i):
    g = _G; GS = g["GS"]
    t0 = time.time()
    mc = GS.load_cache(g["paths"][g["navsim_token"][i]])
    t_load = time.time() - t0
    gt = g["gt"][i].astype(np.float32)
    score = _score_fn(mc)
    r_gt = GS.score_token(mc, GS._as_poses(gt.astype(np.float64)[None]), margins=False)
    sub_gt = r_gt["subscores"][0].astype(np.float64); ep_gt = float(r_gt["epdms"][0])
    if not reverse_ok(gt, gt, g["rev_thr"]) or not sub_in_C(sub_gt, sub_gt) or mean_l2(gt, gt) > g["eps"]:
        raise AssertionError(f"색인 {i}: GT 가 C 밖(설계상 불가능)")
    mu = g["modes"][i].astype(np.float64); log_pi = g["log_pi"][i]
    bi = int(g["bin"][i])
    inv_var, log_norm, chol = g["inv_var"][bi], g["log_norm"][bi], g["chol"][bi]
    lp_gt = log_density(gt, mu, log_pi, inv_var, log_norm)
    mode_l2 = mean_l2(g["modes"][i], gt)
    mode_cheap = (mode_l2 <= g["eps"]) & reverse_ok(g["modes"][i], gt, g["rev_thr"])
    out = dict(i=i, sub_gt=sub_gt, ep_gt=ep_gt, lp_gt=lp_gt, filtered=r_gt["filtered"].copy(),
               mode_cheap=mode_cheap, gt_min_mode_l2=float(mode_l2.min()), arms={})
    if g["mode_pdm"]:
        rm = GS.score_token(mc, GS._as_poses(g["modes"][i].astype(np.float64)), margins=False)["subscores"]
        out["mode_in_C"] = mode_cheap & np.array([sub_in_C(s, sub_gt) for s in rm])
    for arm in g["arms"]:
        x, lp, c, sub_x, ep_x = run_chain(arm, [g["seed"], ARMS[arm], int(i)], gt, mu, log_pi, inv_var, log_norm, chol, g["eps"],
                                          g["n_steps"], g["rev_thr"], score, sub_gt, lazy=g["lazy"], ep_gt=ep_gt)
        if not reverse_ok(x, gt, g["rev_thr"]) or mean_l2(x, gt) > g["eps"] or not sub_in_C(sub_x, sub_gt):
            raise AssertionError(f"색인 {i} {arm}: 최종 상태가 C 밖")
        out["arms"][arm] = dict(x=x, lp=lp, c=c, sub=sub_x, ep=ep_x, l2=float(mean_l2(x, gt)),
                                min_mode_l2=float(mean_l2(g["modes"][i], x).min()))
    n_ver = 0
    if g["verify_final"]:   # 공식 pdm_score 단독 채점과 비트 동일(= C 판정이 평가 채점기와 같다)
        cands = [("gt", gt, sub_gt, ep_gt)] + [(a_, o["x"], o["sub"], o["ep"]) for a_, o in out["arms"].items() if o["c"]["acc"]]
        for nm, xx, ss, ee in cands:
            so, _, eo = GS._official(mc, GS._as_poses(xx.astype(np.float64)[None])[0])
            if not (np.array_equal(so, ss) and eo == ee):
                raise AssertionError(f"색인 {i} {nm}: score_token {ss}/{ee} != 공식 pdm_score {so}/{eo}")
            n_ver += 1
    out["n_verified"] = n_ver
    out["t_load"] = t_load; out["t_total"] = time.time() - t0
    return out


# ------------------------------------------------------------------------------------------------ 본체
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--info", required=True)
    ap.add_argument("--metric_cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="mcmc", help="쉼표: mcmc,randc (targets_gt 는 항상 만든다)")
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--corr_len_s", type=float, default=1.0)
    ap.add_argument("--speed_bins", default="0.5,3,6,9")
    ap.add_argument("--no_reverse_m", type=float, default=0.05)
    ap.add_argument("--min_bin", type=int, default=100)
    ap.add_argument("--limit_chain", type=int, default=0, help=">0 이면 체인 대상 처음 N 개만(스모크 — 출력 smoke)")
    ap.add_argument("--allow_smoke_dump", action="store_true", help="스모크 덤프 허용(시험 전용 — 출력 smoke)")
    ap.add_argument("--mode_pdm", action="store_true", help="모드 16 개도 PDM 채점해 'C 안 모드' 통계(+≈0.7 h)")
    ap.add_argument("--no_lazy", action="store_true", help="매 제안 C 전체 계산(시험용; 체인 값 동일)")
    ap.add_argument("--no_verify_final", action="store_true")
    ap.add_argument("--ckpt_every", type=int, default=5000)
    return ap.parse_args()


def main():
    a = parse()
    t0 = time.time()
    import dump_plan_modes_navsim as DM
    import grpo_scorer as GS
    if list(GS.SUBSCORES) != SUBSCORES:
        raise SystemExit("grpo_scorer.SUBSCORES 순서가 바뀌었다")
    if a.workers > GS.MAX_WORKERS:
        raise SystemExit(f"워커 상한 {GS.MAX_WORKERS}")
    if os.environ.get("OPENBLAS_CORETYPE") != "SkylakeX":
        raise SystemExit("OPENBLAS_CORETYPE=SkylakeX 필요(scripts/navsim/navsim_env.sh)")
    mc_real = os.path.realpath(a.metric_cache)
    if os.path.basename(mc_real.rstrip("/")) != "metric_cache_navtrain" or any(s in mc_real for s in ("navtest", "navhard")):
        raise SystemExit(f"--metric_cache {a.metric_cache} — navtrain 캐시만(navtest·navhard 는 정답 생성에서 읽지 않는다)")
    arms = [x for x in a.arms.split(",") if x]
    if not arms or any(x not in ARMS for x in arms):
        raise SystemExit(f"--arms {a.arms}")

    dump = DM.load_dump(a.dump)
    man = dump["manifest"]
    smoke = bool(man.get("smoke")) or a.limit_chain > 0
    if man.get("split") != "navtrain":
        raise SystemExit(f"덤프 분할 {man.get('split')} != navtrain")
    if man.get("smoke") and not a.allow_smoke_dump:
        raise SystemExit("스모크 덤프 — 학습 정답에 쓰지 않는다(시험이면 --allow_smoke_dump)")
    if os.path.basename(a.info) != os.path.basename(man["ann_file"]):
        raise SystemExit(f"--info {a.info} 가 덤프 ann_file {man['ann_file']} 와 다르다")
    with open(a.info, "rb") as f:   # 프로젝트 자체 info pkl(데이터셋과 같은 파일)
        d = pickle.load(f)
    infos = d["infos"]
    N = len(dump["idx"])
    if not man.get("smoke") and len(infos) != N:
        raise SystemExit(f"info {len(infos)} != 덤프 {N}")
    infos = infos[:N]
    info_tok = np.array([x["token"] for x in infos])
    info_disp = np.stack([np.asarray(x["gt_ego_fut_trajs"], np.float32) for x in infos])
    info_mask = np.stack([np.asarray(x["gt_ego_fut_masks"], np.float32) for x in infos])
    info_eval = np.array([bool(x.get("is_eval_frame")) for x in infos])
    info_nav = np.array([x.get("navsim_token") or "" for x in infos])
    es = np.stack([np.asarray(x["ego_status"], np.float32) for x in infos])
    info_es = np.where(np.isfinite(es), es, 0.0).astype(np.float32)
    del d, infos
    for nm, a_, b_ in (("token", info_tok, dump["token"]), ("gt_disp", info_disp, dump["gt_disp"]), ("gt_mask", info_mask, dump["gt_mask"]),
                       ("is_eval", info_eval, dump["is_eval"]), ("navsim_token", info_nav, dump["navsim_token"]),
                       ("ego_status", info_es, dump["ego_status"])):
        if not np.array_equal(a_, b_):
            raise SystemExit(f"덤프 {nm} 가 info 와 다르다 — 입력 불일치")
    if not np.array_equal(info_disp.cumsum(1), dump["gt_traj"]):
        raise SystemExit("덤프 gt_traj != cumsum(info gt_ego_fut_trajs)")
    if dump["modes"].shape[1:] != (16, T, 2):
        raise SystemExit(f"덤프 modes 모양 {dump['modes'].shape}")

    paths = GS.cache_paths(a.metric_cache)
    complete = (info_mask == 1).all(1)
    status = np.full(N, -1, np.int64)
    status[~info_eval] = 1
    status[info_eval & ~complete] = 2
    status[info_eval & complete] = 0
    if (info_nav[~info_eval] != "").any():
        raise SystemExit("이력 프레임에 navsim_token 이 있다 — 상태 정의 재확인")
    miss = [t for t in info_nav[info_eval] if t not in paths]
    if miss:
        raise SystemExit(f"평가 프레임 {len(miss)} 개의 metric cache 가 없다(예 {miss[:3]}) — 데이터 오류, 캐시를 먼저 만든다(건너뛰지 않음)")
    counts = {STATUS[s]: int((status == s).sum()) for s in STATUS}
    print("[상태]", counts, flush=True)

    # ---- Σ_b (체인 대상 전체; --limit_chain 무관) ----
    edges = [float(e) for e in a.speed_bins.split(",") if e.strip()]
    if edges != sorted(edges) or any(e <= 0 for e in edges):
        raise SystemExit(f"--speed_bins {a.speed_bins}")
    speed = np.linalg.norm(info_es[:, :2].astype(np.float64), axis=1)
    sbin = np.digitize(speed, edges)
    B = len(edges) + 1
    sel = status == 0
    sigma, inv_var, log_norm, binfo = sigma_by_bin(dump["gt_traj"], dump["modes"], sel, sbin, B, a.min_bin)
    for b, bi in enumerate(binfo):
        print(f"[Σ 구간 {b} n={bi['n']}] std x(전방) {np.round(bi['std_x_fwd'], 3)} y(횡) {np.round(bi['std_y_lat'], 3)}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"rule": "var(GT - nearest mode by mean 8-pt L2) within current-speed bin, status==0 frames, ddof=0",
               "speed": "||info ego_status[0:2]|| (vx, vy; non-finite -> 0)", "edges_mps": edges, "bins": binfo},
              open(os.path.join(a.out, "sigma.json"), "w"), indent=1)

    raw = dump["raw_logits"].astype(np.float64)
    log_pi = raw - raw.max(1, keepdims=True)
    log_pi = log_pi - np.log(np.exp(log_pi).sum(1, keepdims=True))
    run_idx = np.nonzero(sel)[0]
    if a.limit_chain > 0:
        run_idx = run_idx[:a.limit_chain]

    prog_p = os.path.join(a.out, "_progress.pkl")
    sig_args = dict(vars(a)); sig_args.pop("workers"); sig_args.pop("ckpt_every")
    res = {}
    if os.path.exists(prog_p):
        with open(prog_p, "rb") as f:   # 이 스크립트가 같은 --out 에 쓴 진행 파일
            pr = pickle.load(f)
        if pr["args"] != sig_args or pr["dump_manifest"] != man:
            raise SystemExit(f"{prog_p} 의 인자·덤프가 지금과 다르다 — 다른 --out 을 쓰거나 파일을 확인할 것")
        res = pr["res"]
        print(f"[재개] 완료 {len(res)} 표본", flush=True)
    todo = [int(i) for i in run_idx if int(i) not in res]

    g = dict(gt=dump["gt_traj"].astype(np.float32), modes=dump["modes"].astype(np.float32), log_pi=log_pi, navsim_token=info_nav,
             paths=paths, inv_var=inv_var, log_norm=log_norm, chol=[proposal_chol(s, a.corr_len_s) for s in sigma], bin=sbin,
             eps=a.eps, n_steps=a.n_steps, seed=a.seed, rev_thr=a.no_reverse_m, arms=arms, lazy=not a.no_lazy,
             verify_final=not a.no_verify_final, mode_pdm=a.mode_pdm)

    def save_prog():
        with open(prog_p + ".tmp", "wb") as f:
            pickle.dump({"args": sig_args, "dump_manifest": man, "res": res}, f)
        os.replace(prog_p + ".tmp", prog_p)

    if todo:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(g,)) as pool:
            for k, o in enumerate(pool.imap_unordered(_work, todo, chunksize=4)):
                res[o["i"]] = o
                if (k + 1) % a.ckpt_every == 0:
                    save_prog()
                if (k + 1) % 2000 == 0 or k + 1 == len(todo):
                    el = time.time() - t0
                    print(f"  {k + 1}/{len(todo)} {el:.0f}s ({(k + 1) / el:.2f} 표본/s)", flush=True)
        save_prog()
    print(f"[체인] {len(res)} 표본 {time.time() - t0:.0f}s", flush=True)

    # ---- 저장·통계 ----
    run_set = set(run_idx.tolist())
    ri = np.array(sorted(i for i in res if i in run_set))
    stats = {"eps": a.eps, "n_steps": a.n_steps, "seed": a.seed, "n_total": int(N), "status_counts": counts,
             "n_run": int(len(ri)), "smoke": smoke, "C": "L2<=eps & no-reverse(GT-heading tangent, 5 cm) & 7 subscores >= GT (EP excluded)",
             "note_prop_stats": "mcmc: cheap/PDM 통계는 밀도 통과 제안 기준(지연 판정); randc: 전 제안 기준"}
    sub_gt_all = np.stack([res[i]["sub_gt"] for i in ri])
    stats["gt_subscore_fail_counts"] = {SUBSCORES[k]: int((sub_gt_all[:, k] < 1).sum()) for k in range(len(SUBSCORES))}
    stats["gt_any_C_subscore_fail"] = int((sub_gt_all[:, C_IDX] < 1).any(1).sum())
    stats["gt_epdms_mean"] = float(np.mean([res[i]["ep_gt"] for i in ri]))
    stats["n_verified_official"] = int(sum(res[i]["n_verified"] for i in ri))
    mc_ = np.stack([res[i]["mode_cheap"] for i in ri])
    stats["modes_cheap"] = {"frac_modes": float(mc_.mean()), "frac_samples_none": float((~mc_.any(1)).mean())}
    if a.mode_pdm:
        mi = np.stack([res[i]["mode_in_C"] for i in ri])
        stats["modes_in_C"] = {"frac_modes": float(mi.mean()), "frac_samples_none": float((~mi.any(1)).mean())}
    tt = np.array([res[i]["t_total"] for i in ri])
    stats["worker_seconds_per_sample"] = {"mean": float(tt.mean()), "median": float(np.median(tt))}

    def nan8():
        return np.full((N, len(SUBSCORES)), np.nan)
    sub_gt_arr = nan8(); ep_gt_arr = np.full(N, np.nan)
    sub_gt_arr[ri] = sub_gt_all; ep_gt_arr[ri] = [res[i]["ep_gt"] for i in ri]
    common = dict(token=dump["token"], idx=dump["idx"], status=status, valid=complete, smoke=np.array(smoke))
    np.savez(os.path.join(a.out, "targets_gt.npz"), gt_ego_plan_sft_2hz=info_disp.copy(), traj=dump["gt_traj"].astype(np.float32),
             n_accept=np.zeros(N, np.int64), sub_gt=sub_gt_arr, epdms_gt=ep_gt_arr, **common)
    gt_r = dump["gt_traj"][ri].astype(np.float64)
    for arm in arms:
        disp = info_disp.copy(); traj = dump["gt_traj"].astype(np.float32).copy()
        n_acc = np.zeros(N, np.int64); lp_s = np.full(N, np.nan); lp_g = np.full(N, np.nan); l2s = np.zeros(N)
        sub_s = nan8(); ep_s = np.full(N, np.nan)
        cs = {k: 0 for k in ("prop", "acc", "dens_pass", "cheap_eval", "l2_ok", "rev_ok", "cheap_ok", "scored", "sub_ok")}
        sf = np.zeros(len(SUBSCORES), np.int64)
        for i in ri:
            o = res[i]["arms"][arm]
            x = o["x"]
            if o["c"]["acc"] > 0:
                delta = x.astype(np.float64) - dump["gt_traj"][i].astype(np.float64)
                dd = np.diff(np.concatenate([np.zeros((1, 2)), delta], 0), axis=0).astype(np.float32)
                disp[i] = info_disp[i] + dd
            traj[i] = x; n_acc[i] = o["c"]["acc"]; lp_s[i] = o["lp"]; lp_g[i] = res[i]["lp_gt"]; l2s[i] = o["l2"]
            sub_s[i] = o["sub"]; ep_s[i] = o["ep"]
            for k in cs:
                cs[k] += o["c"][k]
            sf += o["c"]["sub_fail"]
        err = float(np.abs(disp.cumsum(1)[ri] - traj[ri]).max()) if len(ri) else 0.0
        if err > 1e-4:
            raise AssertionError(f"{arm}: cumsum(disp) 와 x* 차 {err}")
        unchanged = n_acc == 0
        if not np.array_equal(disp[unchanged], info_disp[unchanged]):
            raise AssertionError(f"{arm}: 바뀌지 않은 표본(체인 불변·status 1·2)의 증분이 info 와 비트 동일하지 않다")
        xs_r = traj[ri].astype(np.float64)
        l2_step = np.linalg.norm(xs_r - gt_r, axis=-1)
        dlp = lp_s[ri] - lp_g[ri]
        stat_ = np.linalg.norm(gt_r[:, -1], axis=1) < 0.5
        dsub = sub_s[ri] - sub_gt_all
        stats[arm] = {
            "accept_rate": cs["acc"] / max(cs["prop"], 1), "counts": cs, "density_pass_rate": cs["dens_pass"] / max(cs["prop"], 1),
            "cheap_ok_rate_of_evaluated": cs["cheap_ok"] / max(cs["cheap_eval"], 1),
            "C_subscore_ok_rate_of_scored": cs["sub_ok"] / max(cs["scored"], 1),
            "subscore_fail_vs_gt_of_scored": {SUBSCORES[k]: int(sf[k]) for k in range(len(SUBSCORES))},
            "pdm_calls_per_sample": cs["scored"] / max(len(ri), 1),
            "frac_chains_unchanged": float(unchanged[ri].mean()) if len(ri) else 0.0,
            "logp_star_minus_gt": {"mean": float(dlp.mean()), "median": float(np.median(dlp)), "frac_gt0": float((dlp > 0).mean())},
            "l2_star_gt_8pt": {"mean": float(l2s[ri].mean()), "median": float(np.median(l2s[ri])),
                               "p90": float(np.quantile(l2s[ri], 0.9)), "max": float(l2s[ri].max())},
            "step_dev_mean": l2_step.mean(0).tolist(), "step_dev_max": l2_step.max(0).tolist(),
            "second_diff_mean_star": float(np.linalg.norm(np.diff(xs_r, 2, axis=1), axis=-1).mean()),
            "second_diff_mean_gt": float(np.linalg.norm(np.diff(gt_r, 2, axis=1), axis=-1).mean()),
            "epdms_star_mean": float(np.nanmean(ep_s[ri])), "epdms_gt_mean": stats["gt_epdms_mean"],
            "ep_star_minus_gt_mean": float(dsub[:, 4].mean()),
            "subscore_improved_vs_gt": {SUBSCORES[k]: int((dsub[:, k] > 0).sum()) for k in range(len(SUBSCORES))},
            "stationary": {"n": int(stat_.sum()),
                           "endpoint_shift_mean": float(np.linalg.norm(xs_r[stat_, -1] - gt_r[stat_, -1], axis=1).mean()) if stat_.any() else 0.0},
            "new_reverse_vs_gt": float(np.mean([not reverse_ok(xs_r[k], gt_r[k], 0.05) for k in range(len(ri))])) if len(ri) else 0.0,
            "per_speed_bin": [{"bin": b, "n": int((sbin[ri] == b).sum()),
                               "l2_star_mean": float(l2s[ri][sbin[ri] == b].mean()) if (sbin[ri] == b).any() else None,
                               "accept_mean": float(n_acc[ri][sbin[ri] == b].mean()) if (sbin[ri] == b).any() else None}
                              for b in range(B)],
        }
        np.savez(os.path.join(a.out, f"targets_{arm}.npz"), gt_ego_plan_sft_2hz=disp, traj=traj, n_accept=n_acc, logp_gt=lp_g,
                 logp_star=lp_s, l2_star=l2s, sub_gt=sub_gt_arr, sub_star=sub_s, epdms_gt=ep_gt_arr, epdms_star=ep_s, **common)
    stats["seconds"] = time.time() - t0
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1, ensure_ascii=False)
    json.dump({"args": vars(a), "dump_manifest": man, "info": {"path": a.info, "size": os.path.getsize(a.info)},
               "metric_cache": mc_real, "script_sha256": sha256(__file__),
               "rng": "np.random.default_rng([seed, arm_id, idx]); arm_id mcmc=0 randc=1; per step m, z, u",
               "design": "docs/mcmc_sft_navsim_plan.md"}, open(os.path.join(a.out, "meta.json"), "w"), indent=1, ensure_ascii=False)
    if os.path.exists(prog_p) and not smoke:
        os.replace(prog_p, os.path.join(a.out, "_progress_final.pkl"))
    print(json.dumps(stats, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
