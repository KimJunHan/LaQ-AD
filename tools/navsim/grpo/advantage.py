# 수정(2026-11-06): C3 GRPO 설계 v4 §4·§5·§6·§7·§8(L_g·L_cls)·§10(변형) 구현 M2 — 신규 파일.
"""채점 결과 → 단계 ①–④, advantage A_{i,k,a}, L_g·L_cls 목표 (docs/c3_grpo_design.md).

입력(장면 하나, build_scene 으로 채점기 출력에서 만든다)
  policy: 상위 M = 4 모드 × G = 8 정책 표본의 채점 결과(배열 앞 두 축 (M, G)):
          epdms, fail_idx (·,·,7), fail_wp (·,·,7), coll_front, min_obj_dist_fs_m, lat_dev_wp_m  (+ filtered (7,) 장면 공통)
  det:    16 모드 ξ = 0 의 epdms (16,)  — L_cls 목표, 감시. 기준선·붕괴 판정에는 쓰지 않는다(§2.1, §7).
  mu16:   16 모드 μ (16, 8, 2) 라이다 프레임(detach) — 단계 ③ q* 선택용
  guide:  채점기 guide=True 출력(ref_traj, ref_epdms, human_traj, human_epdms; ego 프레임)

단계(§7, 장면 정책 32 표본으로만 판정, 같다 = 표준편차 ≤ 1e-6)
  앵커 g 의 8 표본 EPDMS 에 분산 있음 → ①.  아니면(앵커 내 붕괴)
    장면 32 표본에 분산 있음 → ② (μ 에 RL 항 없음)
    장면 32 표본이 모두 같고 0 → ③ (L_g, 장면당 한 번, q* 하나)
    장면 32 표본이 모두 같고 > 0 → ④ (c 성분 RLOO, 앵커 내 8)
  ③·④ 는 장면 전체가 붕괴일 때만 생기므로 그 장면의 앵커는 모두 같은 단계다. 한 앵커는 정확히 한 단계에 속한다.

advantage(§4·§5, 선택지 (i)): A_{i,k,a} = Â_i · w_{i,k} · u_a(f*_i),  Σ_k w = 8, Σ_a u = 2(실패 없는 표본은 w = 1, u = 1).
  Â_i = r_i − mean_{j≠i} r_j (앵커 내 RLOO, std 로 나누지 않음).
  주 실패 f*: 곱 지표(collision·offroad·direction·red_light) 중 가장 이른 시뮬레이션 시점(같으면 이 순서), 없으면 ttc. lk·comfort 제외.
  collision 은 채점기 coll_front 로 정면/측면을 나눈다. j_f = 그 유형의 fail_wp.
  커널: G0 json(sensitivity.py 출력)이 주어지고 그 유형이 measured 면 w_k = max(w0, shape_f[k − j_f]),
        아니면 사전 커널 w_k = w0 + (1−w0) ρ^{max(0, j_f−k)} (k ≤ j_f + L), w0 (k > j_f + L). 표본마다 Σ_k w = 8 재정규화.
  u: G0 json 의 축 비율(measured), 아니면 §5 기본 표.

④ 연속 성분(§6): c_i = ½ tanh(min_obj_dist_fs/2) − ½ tanh(lat_dev_wp/0.5),  Â_i = λ_c (c_i − mean_{j≠i} c_j), λ_c = 0.1, w = u = 1.
  min_obj_dist_fs = NaN(전방·측방 객체 없음)은 +∞ 로 본다(tanh → 1). 이 처리 건수는 log 에 센다.
③ L_g(§7): τ* = PDM-Closed(r* > 0) → 사람 궤적. 가중 = λ_g · min(1, r*), λ_g = 0.1. q* = argmin_q ‖μ_q − τ*‖₁ (16 모드, 라이다 프레임).
  손실 L_g = 가중 · ‖μθ,q* − τ*‖₁ / 8 은 학습 쪽에서 계산한다(여기는 목표만).
L_cls(§8): 16 모드 ξ = 0 EPDMS 의 softmax(τ_c = 0.1) 목표, rescore 전 원시 logit 에 건다.
  §12.1(5): 명령 묶음 16 모드 중 하나라도 rescore 가 −999 를 준 프레임은 L_cls 에서 뺀다(cls_valid False).
  그 판정은 rescore 후 logit(rescored_logits, (16,)) 또는 −999 마스크(sentinel_mask, (16,) bool)로 한다.
  그 프레임도 순전파·뱅크 갱신·L_BC 는 그대로 한다(학습 쪽 — 여기는 L_cls 목표·mask 만 낸다).
  epdms16 에 NaN 또는 −999(채점 안 된 후보 표시)가 있어도 cls_valid False. 무효 사유는 cls_invalid_reason 에 남긴다.

변형(§10): Full, B1, B2, R, P, A-a, A-b, A-d
  B1  : w = u = 1, 단계 ① 만(②③④ 는 RL·L_g 없음)
  B2  : w_k ∝ 1[k ≤ j_f](실패 없음 = 1), Σ = 8, + (b)(d)
  R   : Full 과 같은 advantage(축만 ego 축 — ar1.ego_axis_basis, 학습 쪽)
  P   : w = u = 1, ① RLOO, ③ 안전 참조 SFT 가중 × P_ALLFAIL_WEIGHT(0.5, 전부 실패 그룹 가중 하향), ②④ 없음  [구현 해석, §10 문구 기준]
  A-a : w = 1.  A-b : u = (1, 1).  A-d : ③④ 없음(① + L_cls)
"""
import json
from pathlib import Path

import numpy as np

T = 8
FAIL_TYPES = ["collision", "ttc", "offroad", "direction", "red_light", "lane_keeping", "comfort"]   # 채점기 순서
PRODUCT = [0, 2, 3, 4]          # collision, offroad, direction, red_light (같은 시점이면 이 순서)
KTYPES = ["collision_front", "collision_side", "ttc", "offroad", "direction", "red_light"]
U_DEFAULT = {"collision_front": (1.4, 0.6), "collision_side": (0.6, 1.4), "ttc": (1.4, 0.6), "offroad": (0.6, 1.4),
             "direction": (0.6, 1.4), "red_light": (2.0, 0.0), None: (1.0, 1.0)}
RHO, W0, L_LEAD = 0.8, 0.2, 2
LAMBDA_C, LAMBDA_G, TAU_C = 0.1, 0.1, 0.1
P_ALLFAIL_WEIGHT = 0.5
EQ_TOL = 1e-6
SENTINEL = -999.0
VARIANTS = ("Full", "B1", "B2", "R", "P", "A-a", "A-b", "A-d")


# ---------------------------------------------------------------- 커널
def load_kernel(path):
    """G0 json → {type: dict(status, shape (15,) Δ=−7..7, u (2,))}. 파일이 없으면 예외(조용히 사전으로 넘어가지 않음)."""
    d = json.loads(Path(path).read_text())
    out = {"label": d.get("label", ""), "w0": float(d.get("w0", W0)), "types": {}}
    for f in KTYPES:
        e = d["types"].get(f)
        if e is None or e.get("status") != "measured":
            out["types"][f] = dict(status="prior", n=0 if e is None else e.get("n", 0))
            continue
        shape = np.asarray(e["shape"], np.float64)
        if shape.shape != (15,) or not np.isfinite(shape).all():
            raise ValueError(f"G0 커널 {f}: shape 는 Δ=−7..7 의 유한값 15 개여야 한다(빈 Δ 는 sensitivity.py 가 채우고 기록)")
        u = np.array([e["u"]["lon"], e["u"]["lat"]], np.float64)
        if abs(u.sum() - 2.0) > 1e-6:
            raise ValueError(f"G0 커널 {f}: Σu = {u.sum()} ≠ 2")
        out["types"][f] = dict(status="measured", shape=shape, u=u, n=int(e["n"]))
    return out


def prior_w(jf, w0=W0, rho=RHO, L=L_LEAD):
    k = np.arange(T)
    w = np.where(k <= jf + L, w0 + (1 - w0) * rho ** np.maximum(0, jf - k), w0)
    return w * (T / w.sum())


def g0_w(jf, shape, w0):
    w = np.maximum(w0, shape[np.arange(T) - jf + 7])
    return w * (T / w.sum())


def b2_w(jf):
    w = (np.arange(T) <= jf).astype(np.float64)
    return w * (T / w.sum())


def primary_failure(fail_idx, fail_wp, coll_front):
    """한 표본 → (유형 KTYPES 이름 또는 None, j_f 또는 -1). fail_idx 는 필터 반영 값(-1 없음, -2 이력 기원 comfort)."""
    best = None
    for i in PRODUCT:
        t = fail_idx[i]
        if t >= 0 and (best is None or t < fail_idx[best]):
            best = i
    if best is None and fail_idx[1] >= 0:
        best = 1
    if best is None:
        return None, -1
    name = {0: "collision_front" if coll_front else "collision_side", 1: "ttc", 2: "offroad", 3: "direction",
            4: "red_light"}[best]
    jf = int(fail_wp[best])
    if not 0 <= jf < T:
        raise ValueError(f"주 실패 {name} 의 fail_wp {jf} 가 0..7 밖")
    return name, jf


def weights(ftype, jf, variant="Full", kernel=None):
    """(w (8,), u (2,), 출처 문자열)."""
    if ftype is None or variant in ("B1", "P"):
        return np.ones(T), np.ones(2), "none"
    kt = None if kernel is None else kernel["types"].get(ftype)
    w0 = W0 if kernel is None else kernel["w0"]
    if variant == "A-a":
        w, src_w = np.ones(T), "flat"
    elif variant == "B2":
        w, src_w = b2_w(jf), "b2"
    elif kt is not None and kt["status"] == "measured":
        w, src_w = g0_w(jf, kt["shape"], w0), "g0"
    else:
        w, src_w = prior_w(jf, w0), "prior"
    if variant == "A-b":
        u, src_u = np.ones(2), "flat"
    elif kt is not None and kt["status"] == "measured":
        u, src_u = kt["u"].copy(), "g0"
    else:
        u, src_u = np.array(U_DEFAULT[ftype], np.float64), "table"
    return w, u, f"{src_w}/{src_u}"


# ---------------------------------------------------------------- 보조
def rloo(x):
    """(…, G) → x_i − mean_{j≠i} x_j."""
    x = np.asarray(x, np.float64)
    G = x.shape[-1]
    return x - (x.sum(-1, keepdims=True) - x) / (G - 1)


def c_component(min_obj_fs, lat_dev):
    d = np.where(np.isnan(min_obj_fs), np.inf, min_obj_fs)
    return 0.5 * np.tanh(d / 2.0) - 0.5 * np.tanh(np.asarray(lat_dev) / 0.5)


def cls_target(epdms16, rescored_logits=None, sentinel_mask=None, tau_c=TAU_C):
    """(target (16,), valid bool, 사유 str). 무효 프레임은 target 이 0 벡터, valid False.

    rescored_logits: rescore 후 16 logit(−999 = rescore 가 배제한 모드). sentinel_mask: 같은 정보의 bool (16,).
    둘 다 None 이면 rescore −999 판정을 하지 않는다(사유 "no_rescore_info" 는 valid 를 바꾸지 않고 호출자가 셀 수 있게 남긴다).
    목표는 rescore 전 원시 logit 에 걸 것(학습 쪽). rescored_logits 는 −999 마스크 출처로만 쓴다.
    """
    e = np.asarray(epdms16, np.float64)
    if e.shape != (16,):
        raise ValueError(f"epdms16 모양 {e.shape}")
    masks = []
    if rescored_logits is not None:
        lg = np.asarray(rescored_logits, np.float64)
        if lg.shape != (16,):
            raise ValueError(f"rescored_logits 모양 {lg.shape}")
        masks.append(lg <= SENTINEL + 1e-6)
    if sentinel_mask is not None:
        sm = np.asarray(sentinel_mask, bool)
        if sm.shape != (16,):
            raise ValueError(f"sentinel_mask 모양 {sm.shape}")
        masks.append(sm)
    if any(m.any() for m in masks):
        return np.zeros(16), False, "rescore_sentinel"
    if np.isnan(e).any() or np.any(e <= SENTINEL + 1e-6):
        return np.zeros(16), False, "epdms_invalid"
    z = e / tau_c
    z = np.exp(z - z.max())
    return z / z.sum(), True, ("ok" if masks else "no_rescore_info")


def ego_to_lidar(xy_ego, lidar2ego=None):
    if lidar2ego is None:
        return np.asarray(xy_ego, np.float64)[..., :2]
    M = np.asarray(lidar2ego, np.float64)
    if M.shape == (4, 4):
        R, t = M[:2, :2], M[:2, 3]                                   # 3D 동차(z 회전·이동만 평면에 쓴다)
    elif M.shape == (3, 3):
        R, t = M[:2, :2], M[:2, 2]                                   # 2D 동차
    else:
        raise ValueError(f"lidar2ego 는 3×3 또는 4×4 — 받은 모양 {M.shape}")
    return (np.asarray(xy_ego, np.float64)[..., :2] - t) @ R        # p_lidar = Rᵀ (p_ego − t)


def build_scene(token, out, M=4, G=8, n_det=16, mu16=None, top_modes=None, lidar2ego=None):
    """채점기 출력(후보 순서 = [ξ=0 16 모드, 정책 M×G(모드 순, 표본 순)]) → advantage 입력 dict."""
    K = len(out["epdms"])
    if K != n_det + M * G:
        raise ValueError(f"{token}: 후보 {K} ≠ {n_det} + {M}×{G}")
    sl = slice(n_det, None)
    pol = {k: np.asarray(out[k])[sl].reshape((M, G) + np.asarray(out[k]).shape[1:])
           for k in ("epdms", "fail_idx", "fail_wp", "coll_front", "min_obj_dist_fs_m", "lat_dev_wp_m")}
    guide = {k: out[k] for k in ("ref_traj", "ref_epdms", "human_traj", "human_epdms") if k in out}
    return dict(token=token, policy=pol, det_epdms=np.asarray(out["epdms"])[:n_det], mu16=mu16, top_modes=top_modes,
                guide=guide, lidar2ego=lidar2ego)


# ---------------------------------------------------------------- 본체
def scene_advantage(scene, variant="Full", kernel=None, rescored_logits=None, sentinel_mask=None):
    """rescored_logits / sentinel_mask: 명령 묶음 16 모드의 rescore 후 logit 또는 −999 마스크(§12.1(5), cls_target 참고)."""
    if variant not in VARIANTS:
        raise ValueError(f"변형 {variant}")
    pol = scene["policy"]
    r = np.asarray(pol["epdms"], np.float64)
    M, G = r.shape
    log = dict(c_nan_as_inf=0)
    std_a = r.std(-1)
    allsame = r.std() <= EQ_TOL
    stage = np.zeros(M, np.int64)
    for g in range(M):
        if std_a[g] > EQ_TOL:
            stage[g] = 1
        elif not allsame:
            stage[g] = 2
        elif r.mean() <= EQ_TOL:
            stage[g] = 3
        else:
            stage[g] = 4
    Ahat = np.zeros((M, G))
    W = np.ones((M, G, T))
    U = np.ones((M, G, 2))
    ftypes = np.full((M, G), None, dtype=object)
    jfs = np.full((M, G), -1, np.int64)
    srcs = np.full((M, G), "", dtype=object)
    c_val = None
    use = dict(Full=(1, 3, 4), B1=(1,), B2=(1, 3, 4), R=(1, 3, 4), P=(1, 3), **{"A-a": (1, 3, 4), "A-b": (1, 3, 4)},
               **{"A-d": (1,)})[variant]
    for g in range(M):
        if stage[g] == 1:
            Ahat[g] = rloo(r[g])
            for i in range(G):
                f, jf = primary_failure(pol["fail_idx"][g, i], pol["fail_wp"][g, i], bool(pol["coll_front"][g, i]))
                w, u, src = weights(f, jf, variant, kernel)
                W[g, i], U[g, i], ftypes[g, i], jfs[g, i], srcs[g, i] = w, u, f, jf, src
        elif stage[g] == 4 and 4 in use:
            if c_val is None:
                log["c_nan_as_inf"] = int(np.isnan(pol["min_obj_dist_fs_m"]).sum())
                c_val = c_component(pol["min_obj_dist_fs_m"], pol["lat_dev_wp_m"])
            Ahat[g] = LAMBDA_C * rloo(c_val[g])
    A = Ahat[..., None, None] * W[..., :, None] * U[..., None, :]           # (M, G, T, 2)
    Lg = None
    if stage[0] == 3 and 3 in use:
        Lg = guide_target(scene, P_ALLFAIL_WEIGHT if variant == "P" else 1.0)
    tgt, valid, why = cls_target(scene["det_epdms"], rescored_logits, sentinel_mask)
    log["cls_reason"] = why
    return dict(token=scene["token"], stage=stage, A=A, Ahat=Ahat, w=W, u=U, ftype=ftypes, jf=jfs, src=srcs, c=c_val,
                Lg=Lg, cls_target=tgt, cls_valid=valid, variant=variant, log=log)


def guide_target(scene, extra_weight=1.0):
    gd = scene["guide"]
    if "ref_traj" not in gd:
        raise ValueError(f"{scene['token']}: 단계 ③ 인데 안내 궤적 없음 — 채점기를 guide=True 로 돌려야 한다")
    if gd["ref_epdms"] > 0:
        src, tau, rs = "pdm", gd["ref_traj"], float(gd["ref_epdms"])
    elif np.isfinite(gd.get("human_epdms", np.nan)):
        src, tau, rs = "human", gd["human_traj"], float(gd["human_epdms"])
    else:
        src, tau, rs = "none", None, 0.0                                  # 사람 궤적 없음(합성 장면) → 가중 0, 기록
    out = dict(source=src, r_star=rs, weight=LAMBDA_G * min(1.0, rs) * extra_weight, q_star=-1, tau_star=None)
    if tau is None:
        return out
    tau_l = ego_to_lidar(tau, scene.get("lidar2ego"))
    mu16 = scene.get("mu16")
    if mu16 is None:
        raise ValueError(f"{scene['token']}: q* 선택에 16 모드 μ 필요")
    d = np.abs(np.asarray(mu16) - tau_l[None]).sum((-1, -2))
    out.update(q_star=int(d.argmin()), tau_star=tau_l, l1_to_qstar=float(d.min()))
    return out


def batch_log(results):
    """iter 기록(§7): 단계별 그룹 비율, ③ 안내 원천 비율, 유형별 j_f 분포, 축별 |A| 합."""
    st = np.concatenate([x["stage"] for x in results])
    out = {f"stage{s}_frac": float((st == s).mean()) for s in (1, 2, 3, 4)}
    srcs = [x["Lg"]["source"] for x in results if x["Lg"] is not None]
    for s in ("pdm", "human", "none"):
        out[f"guide_{s}"] = srcs.count(s)
    for f in KTYPES:
        jf = np.concatenate([x["jf"][x["ftype"] == f] for x in results])
        out[f"jf_hist_{f}"] = np.bincount(jf, minlength=T).tolist() if len(jf) else [0] * T
    A = np.concatenate([x["A"].reshape(-1, T, 2) for x in results])
    out["absA_lon"], out["absA_lat"] = float(np.abs(A[..., 0]).sum()), float(np.abs(A[..., 1]).sum())
    out["cls_invalid"] = int(sum(not x["cls_valid"] for x in results))
    for why in ("rescore_sentinel", "epdms_invalid", "no_rescore_info"):
        out[f"cls_{why}"] = int(sum(x["log"].get("cls_reason") == why for x in results))
    return out
