# 수정(2026-10-16): 교사 GRPO — 공용 모듈(보상·그룹 상대 이점·k3 KL·표집·토큰 로그확률·참조 정책 교체·SFT 체크포인트 적재).
#   train_grpo.py 와 smoke_grpo_cpu.py 가 쓴다. sft_common.py / train_sft.py 는 읽기 전용으로 import 만 한다(수정 금지).
#
# 알고리즘(DeepSeekMath GRPO arXiv 2402.03300 §4.1 + AutoVLA RFT):
#   * 프롬프트마다 G 개 정답 표집 o_1..o_G ~ π_θ(·|q). 보상 r_i 는 GT 에서 계산(의도 문장 보상 없음).
#   * 이점 A_i = (r_i − mean(r)) / (std(r) + eps) — 그룹(같은 프롬프트) 안에서. 모든 토큰이 같은 A_i 를 받는다(결과 감독).
#   * step 마다 정책 갱신은 한 번(표집한 정책 = 갱신 직전 정책) → 중요도 비 π_θ/π_old ≡ 1, 잘라내기(clip) 불필요.
#     손실 = −(1/N) Σ_i (1/|o_i|) Σ_t [ exp(logπ_θ − sg(logπ_θ)) · A_i − β · KL_t ],  sg = stop-gradient(값 1, 기울기 ∇logπ_θ).
#   * KL_t = k3 추정자 π_ref/π_θ − log(π_ref/π_θ) − 1 ≥ 0 (표집 토큰에서; E_{π_θ}[k3] = KL(π_θ‖π_ref), 불편).
#   * π_ref = 같은 base + SFT LoRA(동결). 메모리 절약: 어댑터를 둘 두지 않고, SFT 학습 파라미터 값(LoRA A/B + 새 토큰 행,
#     ≈22M fp32 ≈ 90 MB)만 사본으로 들고 참조 순전파 동안 학습 파라미터에 잠깐 덮어썼다가 되돌린다(RefSnapshot).
#   * LoRA·새 토큰 행만 학습, 비전 인코더 동결. fp32. LoRA dropout = 0(표집 분포와 학습 때 log π 가 같은 함수여야 한다).
#   * 수정(2026-10-16): 설계 검토 반영 — 문법 제약 표집(answer_grammar.py: 위치별 TRAIN 허용 집합 ∩ 문법 상태).
#     실제로 답을 만드는 정책은 제약된 분포 q_θ(y_t|·) = π_θ(y_t|·)·1[y_t∈A_t] / Z_t,  Z_t = Σ_{a∈A_t} π_θ(a|·)
#     (A_t 는 앞서 고른 토큰에 따라 달라진다). 목적 E_{q_θ}[R] 의 기울기는 E_q[R ∇log q_θ] = E_q[R (∇log π_θ − ∇log Z_t)].
#     제약 없는 log π_θ 를 쓰면 −∇log Z_t 항이 빠진다: E_{y~q}[∇log π_θ(y)] = ∇log Z_t ≠ 0 이라 이점 A_i 와 곱해진
#     이 항은 평균 0 이 아니고(같은 그룹에서도 접두마다 Z_t 가 다르다) 기울기가 편향된다. KL 도 실제 두 분포
#     q_θ·q_ref 사이여야 한다. 그래서 표집 로그확률·학습 로그확률·참조 로그확률을 모두 허용 집합으로 다시 정규화한
#     log q 로 계산한다(answer_logprobs(grammar=…)). 허용 토큰이 하나뿐인 위치(머리말·강제 채움·<|im_end|>)는
#     log q = 0 → 기울기·KL 0(실제로 고른 것이 없으므로 맞는 값). 캐시 추출·val 평가도 같은 제약으로 생성한다.
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sft_common as S  # noqa: E402  (읽기 전용 import — 수정 금지 파일)
import common_v2 as V2  # noqa: E402
import teacher_metrics as TM  # noqa: E402

TAG = "# 수정(2026-10-16): 교사 GRPO"
ANSWER_TOKENS = 547          # README: v2-sensor 정답 토큰 수(전 표본 고정, <|im_end|> 제외)
END_TOKEN = "<|im_end|>"


# ================================================================ 보상
@dataclass
class RewardConfig:
    """기본값 근거는 각 줄 주석. 모든 하위 점수 ∈ [0,1], 최종 보상 ∈ [0,1]. 형식 오류 = 0(최소).
    유효 정답: R = w_fmt + (1 − w_fmt) · Σ_c w_c r_c / Σ_{c 가용} w_c  (c ∈ {perc, map, act}).
    가용성: map = GT 지도 있음, act = GT 미래 완전(future_complete). 불가용 성분은 가중 재정규화(GT 미상 → 보상 없음,
    SFT 의 손실 제외 규칙과 같다)."""
    w_fmt: float = 0.1           # 문법 통과 기본 점수 → 유효 정답 > 형식 오류 가 항상 성립(형식 유지 압력)
    w_perc: float = 0.4
    w_map: float = 0.2
    w_act: float = 0.4
    # 인지(객체) — 매칭은 teacher_metrics.match_objects(같은 클래스·방위·|Δd| ≤ max(2 m, 0.1·d), 헝가리안)와 같다
    match_tol_abs: float = 2.0
    match_tol_rel: float = 0.1
    of_pos: float = 0.3          # 객체 칸 안 필드 가중(가용 필드로 재정규화)
    of_size: float = 0.1
    of_heading: float = 0.1
    of_motion: float = 0.2
    of_fut: float = 0.2
    of_ttc: float = 0.1          # TTC 일관성: 예측 위험 구간 == GT 구간(GT 미래 기반 라벨)
    s_pos: float = 1.0           # m, exp(−err/σ). 1 m 오차 → 0.37
    s_size: float = 0.5          # m, |Δl|+|Δw|
    s_speed: float = 1.0         # m/s
    s_fut: tuple = (0.5, 1.0, 1.5)   # m, 1/2/3 s 변위 오차(시간에 비례해 허용)
    # 지도
    mf_pc: float = 0.2
    mf_count: float = 0.2
    mf_lanes: float = 0.4
    mf_free: float = 0.2
    s_pc: float = 1.0            # m, 횡단보도 거리
    s_count: float = 1.0         # 개수 차 1 → 0.37
    s_lane: float = 0.5          # m, 단면 횡거리(차로 폭 ~3.5 m 의 1/7)
    s_free: float = 2.0          # m, 자유거리
    # 행동(코드북 복호 궤적)
    af_l2: float = 0.6
    af_col: float = 0.3
    af_feas: float = 0.1
    s_l2: float = 1.0            # m, exp(−max(0, ADE − ADE_oracle)/σ). ADE_oracle = GT 행동 토큰 복호의 양자화 오차(p50 0.02 m)
    feas_vmax: float = 25.0      # m/s — train GT(코드북 복호) 최대 20.2, p99.9 18.2
    feas_amax: float = 8.0       # m/s² — |Δv|/0.5 s(첫 칸은 라벨 v0 기준). GT p99.9 7.5
    feas_wmax: float = 1.0       # rad/s — 진행방향 변화율(두 구간 모두 >1 m/s 일 때). GT 최대 0.73
    feas_latmax: float = 4.0     # m/s² — v·ω. GT 최대 3.2
    feas_vmin_yaw: float = 1.0   # m/s — 이보다 느리면 방향 변화율 검사 제외(정지 근처 방향 잡음)

    @classmethod
    def from_json(cls, s):
        d = {} if not s else (json.load(open(s)) if os.path.exists(s) else json.loads(s))
        bad = [k for k in d if k not in cls.__dataclass_fields__]
        if bad:
            raise ValueError(f"RewardConfig 에 없는 키 {bad}")
        c = cls(**d)
        if isinstance(c.s_fut, list):
            c.s_fut = tuple(c.s_fut)
        for k in ("w_fmt", "w_perc", "w_map", "w_act"):
            if getattr(c, k) < 0:
                raise ValueError(f"{k} < 0")
        if not (0.0 <= c.w_fmt < 1.0):
            raise ValueError("w_fmt ∈ [0,1)")
        return c

    def to_dict(self):
        return asdict(self)


def _ex(err, s):
    return float(math.exp(-abs(float(err)) / float(s)))


def _wmean(pairs):
    """[(가중, 점수|None)] → 가중 평균(None 은 불가용 → 제외). 전부 불가용이면 None."""
    num = den = 0.0
    for w, v in pairs:
        if v is None or w <= 0:
            continue
        num += w * v; den += w
    return None if den == 0 else num / den


def object_score(p, g, cfg, speed_known=True):
    """매칭된 예측 p · GT g(둘 다 common_v2 형 dict) 의 칸 점수 ∈ [0,1]. GT 미상 필드(<na>)는 제외."""
    pos = _ex(math.hypot(p["lat"] - g["lat"], p["lon"] - g["lon"]), cfg.s_pos)
    size = _ex(abs(p["length"] - g["length"]) + abs(p["width"] - g["width"]), cfg.s_size)
    heading = None if g.get("heading") is None else float(p.get("heading") == g["heading"])
    if g.get("motion") == "na":
        motion = None
    elif p.get("motion") == "na":
        motion = 0.0
    else:
        st = float(p["motion"] == g["motion"] and p.get("vel_sector") == g.get("vel_sector"))
        if speed_known and g.get("speed") is not None and p.get("speed") is not None:
            motion = 0.5 * st + 0.5 * _ex(p["speed"] - g["speed"], cfg.s_speed)
        else:
            motion = st          # 이전 프레임 없음 → 속력 숫자는 GT 감독 제외(SFT loss_mask 와 같은 규칙)
    fs = []
    for h, gf in enumerate(g.get("fut") or []):
        if gf is None:
            continue
        pf = (p.get("fut") or [None] * 3)[h]
        fs.append(0.0 if pf is None else _ex(math.hypot(pf[0] - gf[0], pf[1] - gf[1]), cfg.s_fut[h]))
    fut = float(np.mean(fs)) if fs else None
    ttc = None if g.get("ttc") is None else float(p.get("ttc") == g["ttc"])
    return _wmean([(cfg.of_pos, pos), (cfg.of_size, size), (cfg.of_heading, heading), (cfg.of_motion, motion),
                   (cfg.of_fut, fut), (cfg.of_ttc, ttc)])


def perception_reward(pred_objs, gt_objs, cfg, speed_known=True):
    """Σ_매칭 칸 점수 / max(n_pred, n_gt) — 놓친 GT·거짓 예측 둘 다 벌점(F1 형). 둘 다 0 개면 1."""
    if not pred_objs and not gt_objs:
        return 1.0, dict(n_pred=0, n_gt=0, n_match=0)
    P = [TM.norm_obj(o) for o in pred_objs]
    dxy = bool(pred_objs) and all("dist" not in o for o in pred_objs)
    G = [TM.norm_gt_obj(o, dist_from_xy=dxy) for o in gt_objs]
    M = TM.match_objects(P, G, cfg.match_tol_abs, cfg.match_tol_rel)
    tot = sum(object_score(pred_objs[i], gt_objs[j], cfg, speed_known) for i, j in M)
    return tot / max(len(P), len(G)), dict(n_pred=len(P), n_gt=len(G), n_match=len(M))


def map_reward(pm, gm, cfg):
    """GT 지도 없음 → None(불가용). 예측 '<na>'(지도 미상) → 0."""
    if gm is None:
        return None
    if pm is None:
        return 0.0
    sp, sg = TM._pc_state(pm.get("ped_crossing")), TM._pc_state(gm.get("ped_crossing"))
    if sp != sg:
        pc = 0.0
    elif sg in ("none", "here"):
        pc = 1.0
    else:
        pc = _ex(pm["ped_crossing"]["dist"] - gm["ped_crossing"]["dist"], cfg.s_pc)
    cnt = 0.5 * _ex(pm["n_divider"] - gm["n_divider"], cfg.s_count) + 0.5 * _ex(pm["n_boundary"] - gm["n_boundary"], cfg.s_count)
    ls = []
    for y in V2.LANE_Y:
        dp, dg = pm["lanes"][str(y)], gm["lanes"][str(y)]
        for k in ("div_l", "div_r", "bnd_l", "bnd_r"):
            a, b = dp[k], dg[k]
            ls.append(1.0 if (a is None and b is None) else 0.0 if (a is None or b is None) else _ex(a - b, cfg.s_lane))
    free = _ex(pm["free_dist"] - gm["free_dist"], cfg.s_free)
    return _wmean([(cfg.mf_pc, pc), (cfg.mf_count, cnt), (cfg.mf_lanes, float(np.mean(ls))), (cfg.mf_free, free)])


def feasibility_checks(traj, v0, cfg):
    """(6,2) 궤적(lidar: +x 오른쪽, +y 전방, 0.5 s 간격) → 검사 통과 bool 배열(고정 길이 22: 속력 6·가속 6·방향 5·횡가속 5).
    느린 구간(방향 변화율 정의 불안정)은 통과로 둔다."""
    p = np.concatenate([np.zeros((1, 2)), np.asarray(traj, np.float64)], 0)
    d = np.diff(p, axis=0)
    v = np.linalg.norm(d, axis=1) / V2.DT
    vv = np.concatenate([[float(v0)], v])
    a = np.diff(vv) / V2.DT
    h = np.arctan2(d[:, 1], d[:, 0])
    dh = (np.diff(h) + np.pi) % (2 * np.pi) - np.pi
    w = np.abs(dh) / V2.DT
    slow = (v[1:] <= cfg.feas_vmin_yaw) | (v[:-1] <= cfg.feas_vmin_yaw)
    return np.concatenate([v <= cfg.feas_vmax, np.abs(a) <= cfg.feas_amax,
                           slow | (w <= cfg.feas_wmax), slow | (w * v[1:] <= cfg.feas_latmax)])


class ActionContext:
    """future_complete 표본 하나의 행동 보상 문맥(GT 미래 박스·GT/oracle 충돌·oracle ADE·oracle 실현가능 검사)."""

    def __init__(self, rec, codebook, fut_boxes, coll_fn, cfg):
        t = rec["targets"]
        self.gt = np.asarray(t["fut_traj"], np.float64)[:6, :2]
        self.v0 = float(t.get("label_v0", 0.0) or 0.0)
        self.fut = fut_boxes
        self.coll = coll_fn
        self.oracle = codebook.decode(t["action_tokens"])
        self.ade_oracle = float(np.linalg.norm(self.oracle - self.gt, axis=1).mean())
        self.gt_col = coll_fn(self.gt, fut_boxes)
        self.or_col = coll_fn(self.oracle, fut_boxes)
        self.or_feas = feasibility_checks(self.oracle, self.v0, cfg)

    def reward(self, traj, cfg):
        """반환 (r_act, 세부). 라벨 상대 규칙: 라벨(GT 궤적 또는 그 코드북 복호)이 스스로 어기는 시점·검사는 벌점 없음 →
        GT 행동 토큰은 정확히 1 점."""
        ade = float(np.linalg.norm(np.asarray(traj, np.float64) - self.gt, axis=1).mean())
        r_l2 = _ex(max(0.0, ade - self.ade_oracle), cfg.s_l2)
        pc = self.coll(traj, self.fut)
        new_col = pc & ~(self.gt_col | self.or_col)
        r_col = 0.0 if new_col.any() else 1.0
        fc = feasibility_checks(traj, self.v0, cfg)
        charged = self.or_feas                      # oracle 이 통과하는 검사만 채점
        r_feas = float(fc[charged].mean()) if charged.any() else 1.0
        r = _wmean([(cfg.af_l2, r_l2), (cfg.af_col, r_col), (cfg.af_feas, r_feas)])
        return r, dict(ade=ade, l2=r_l2, col=r_col, feas=r_feas, collide=bool(new_col.any()))


def make_collision_fn():
    """tools/eval_nusc_planning.single_coll(SparseDrive 평가기와 같은 박스·yaw 규약) → numpy bool(6)."""
    import torch
    pe = TM.load_planning_eval()

    def coll(traj, fut):
        return pe.single_coll(torch.as_tensor(np.asarray(traj, np.float32)), fut).numpy().astype(bool)
    return coll


def build_action_contexts(records, infos_path, codebook, cfg, valid_rule="dataset_default"):
    """future_complete 레코드 → {token: ActionContext}. infos 는 한 번 읽고 버린다(미래 박스 = teacher_metrics.fut_boxes,
    eval_teacher 와 같은 규칙). 정답 파일의 fut_traj 와 infos 의 gt_ego_fut_trajs 가 다르면 즉시 실패."""
    import pickle
    coll = make_collision_fn()
    infos = pickle.load(open(infos_path, "rb"))["infos"]
    index = {i["token"]: k for k, i in enumerate(infos)}
    out = {}
    for r in records:
        if not S.future_complete(r):
            continue
        k = index.get(r["token"])
        if k is None:
            raise KeyError(f"{r['token']} 가 {infos_path} 에 없다(정답 파일·infos 분할 불일치)")
        g = np.asarray(infos[k]["gt_ego_fut_trajs"], np.float64).cumsum(0)[:6, :2]
        if not np.allclose(g, np.asarray(r["targets"]["fut_traj"]), atol=1e-3):
            raise RuntimeError(f"{r['token']}: 정답 fut_traj ≠ infos gt_ego_fut_trajs")
        fut = TM.fut_boxes(infos, k, valid_rule)
        if len(fut) != 6:
            raise RuntimeError(f"{r['token']}: 미래 박스 {len(fut)} != 6 (future_complete 표본)")
        out[r["token"]] = ActionContext(r, codebook, fut, coll, cfg)
    del infos
    return out


def score_answer(text, ended, rec, codebook, act_ctx, cfg, parser=V2.parse_answer):
    """생성 정답 1개 → (보상 ∈ [0,1], 세부 dict). 형식 오류(파서 실패·종료 토큰 없음·행동 토큰 범위 밖) = 0."""
    d = dict(fmt=0.0, perc=None, map=None, act=None, total=0.0)
    if not ended:
        d["err"] = "no_end"
        return 0.0, d
    try:
        p = parser(text)
    except Exception as e:    # 형식 오류는 '건너뛰기'가 아니라 최소 보상으로 채점된다(정책이 그 결과를 학습)
        d["err"] = f"parse: {str(e)[:80]}"
        return 0.0, d
    t = rec["targets"]
    d["fmt"] = 1.0
    d["perc"], d["perc_info"] = perception_reward(p["objects"], t["objects"], cfg, S.prev_available(rec))
    d["map"] = map_reward(p["map"], t.get("map"), cfg)
    if S.future_complete(rec):
        if act_ctx is None:
            raise KeyError(f"{rec['token']}: future_complete 표본인데 행동 문맥이 없다")
        if p["action"] is None:
            d["act"] = 0.0                      # GT 미래가 있는데 '<na>' 행동 → 0
        else:
            d["act"], d["act_info"] = act_ctx.reward(codebook.decode(p["action"]), cfg)
    comp = _wmean([(cfg.w_perc, d["perc"]), (cfg.w_map, d["map"]), (cfg.w_act, d["act"])])
    total = cfg.w_fmt + (1.0 - cfg.w_fmt) * (comp if comp is not None else 1.0)
    if not (0.0 <= total <= 1.0 + 1e-12) or not math.isfinite(total):
        raise FloatingPointError(f"{rec['token']}: 보상 {total} ∉ [0,1] — 보상 식 점검(세부 {d})")
    d["total"] = total
    return total, d


# ================================================================ 이점·KL·손실
def group_advantages(rewards, eps=1e-4):
    """rewards (P,G) → A (P,G) = (r − mean_g) / (std_g + eps). std = 표본 표준편차(unbiased, DeepSeekMath/TRL 과 같음).
    그룹 보상이 모두 같으면 A = 0(학습 신호 없음 — 수식 그대로의 결과이고 표본을 버리는 것이 아님; 비율을 로그)."""
    import torch
    r = torch.as_tensor(rewards, dtype=torch.float32)
    if r.dim() != 2 or r.shape[1] < 2:
        raise ValueError(f"rewards 모양 {tuple(r.shape)} — (P, G≥2) 필요")
    if not torch.isfinite(r).all():
        raise FloatingPointError(f"비유한 보상 {r}")
    mu = r.mean(1, keepdim=True)
    sd = r.std(1, keepdim=True, unbiased=True)
    return (r - mu) / (sd + eps)


def kl_k3(logp, logp_ref):
    """토큰별 k3 = π_ref/π_θ − log(π_ref/π_θ) − 1 (Schulman 2020 'Approximating KL Divergence'). 항상 ≥ 0, π_θ=π_ref 면 0."""
    import torch
    d = logp_ref - logp
    return torch.exp(d) - d - 1.0


def grpo_loss(logp, logp_ref, adv, mask, beta, denom):
    """logp (R,T) 기울기 있음, logp_ref (R,T) 상수, adv (R,), mask (R,T) bool, denom = 정규화 상수.
    seq 정규화: denom = 이 step 전체 행 수 N 이고 행마다 감독 토큰 수로 나눈다(DeepSeekMath 식).
    token 정규화: denom = 이 step 전체 감독 토큰 수(이 경우 행별 나눗셈 없음) — 호출자가 row_norm 을 고른다.
    반환 (손실 합 / denom, 통계). 마이크로배치 손실을 그대로 backward 하면 step 전체 평균의 기울기가 된다."""
    import torch
    m = mask.to(logp.dtype)
    pg = -torch.exp(logp - logp.detach()) * adv[:, None]
    kl = kl_k3(logp, logp_ref.detach())
    per = torch.where(mask, pg + beta * kl, torch.zeros_like(logp))
    if denom["kind"] == "seq":
        row = per.sum(1) / m.sum(1).clamp(min=1.0)
        loss = row.sum() / denom["value"]
    elif denom["kind"] == "token":
        loss = per.sum() / denom["value"]
    else:
        raise ValueError(denom)
    with torch.no_grad():
        n = m.sum().clamp(min=1.0)
        st = dict(kl_sum=float((kl * m).sum()), n_tok=float(m.sum()), kl_max=float((kl * m).max()) if m.any() else 0.0,
                  pg_sum=float((pg * m).sum()), kl_mean=float((kl * m).sum() / n))
    return loss, st


# ================================================================ 모델 적재·참조 정책
def load_sft_meta(sft_ckpt):
    return S.load_meta(sft_ckpt)


def set_dropout_zero(model):
    """LoRA dropout 등 모든 Dropout p→0. 표집(eval) 과 학습 순전파(train, gradient checkpointing 위해 train 모드)가 같은
    함수여야 on-policy 로그확률이 맞는다(Qwen2 의 attention dropout 은 원래 0)."""
    import torch
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) and m.p != 0.0:
            m.p = 0.0; n += 1
    return n


def build_policy_from_base(model, tok, meta, adapter_dir, device):
    """base(Qwen2VLForConditionalGeneration, fp32) + SFT 어댑터 → 학습 가능 PeftModel. train_sft.build_model 과 같은 순서
    (특수토큰 추가 → resize → 새 행 초기화 → 동결 → GC → peft) 후 SFT 어댑터 값을 덮어쓴다(새 행 초기값은 무관해진다)."""
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    toks = meta["special_tokens"]
    new_ids, n_base = S.add_special_tokens(tok, toks)
    if n_base != meta["n_base_vocab"] or len(tok) != meta["vocab_after"]:
        raise RuntimeError(f"토크나이저 불일치: base {n_base}/{meta['n_base_vocab']}, 전체 {len(tok)}/{meta['vocab_after']}")
    S.resize_for_tokens(model, len(tok))
    S.init_new_rows(model, tok, toks, new_ids, n_base, mode=meta.get("new_token_init", "desc"), seed=0)
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    lo = meta["lora"]
    pm = S.build_peft(model, new_ids, lo["r"], lo["alpha"], 0.0, lo["targets"])   # dropout 0 (위 set_dropout_zero 주석)
    sd = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    res = set_peft_model_state_dict(pm, sd)
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    # tie_word_embeddings=True: peft 가 lm_head 의 tied 토큰 어댑터 값을 embed 값의 사본으로 함께 저장하는 경우가 있다
    #   (from_pretrained 로 만든 모델에서 실측). 메모리에서는 lm_head 어댑터가 embed 의 delta 파라미터를 공유하므로 그 키는
    #   '모델에 없는 키'로 보고된다 — 파일 안 두 값이 비트 단위로 같을 때만 중복 사본으로 인정하고, 다르면 실패.
    ek = "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta"
    lk = "base_model.model.lm_head.token_adapter.trainable_tokens_delta"
    if lk in unexpected:
        if ek not in sd or not torch.equal(sd[ek], sd[lk]):
            raise RuntimeError("SFT 어댑터의 lm_head 토큰 행이 embed 토큰 행과 다르다(묶음 가중치 가정 위반)")
        unexpected.remove(lk)
    base = S.base_of(pm)
    lt = getattr(getattr(base, "lm_head", None), "token_adapter", None)
    if lt is not None and lt.trainable_tokens_delta["default"].data_ptr() != \
            base.model.embed_tokens.token_adapter.trainable_tokens_delta["default"].data_ptr():
        raise RuntimeError("lm_head 토큰 어댑터가 embed 토큰 delta 를 공유하지 않는다")
    if unexpected:
        raise RuntimeError(f"SFT 어댑터에 모델에 없는 키 {len(unexpected)}개: {unexpected[:4]}")
    # 같은 이유로 lm_head 의 공유 delta 이름이 '빠진 키'로도 보고된다 — 위에서 공유(data_ptr 동일)를 확인했으므로 제외
    missing = [k for k in (getattr(res, "missing_keys", []) or []) if ("lora_" in k or "trainable_tokens" in k)
               and not (lt is not None and k.startswith(lk))]
    if missing:
        raise RuntimeError(f"SFT 어댑터에 빠진 학습 키 {len(missing)}개: {missing[:4]}")
    for p in pm.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    for n, p in pm.named_parameters():
        if p.requires_grad and ".visual." in n:
            raise RuntimeError(f"비전 인코더 파라미터가 학습 대상이다: {n}")
    set_dropout_zero(pm)
    pm.to(device)
    return pm, new_ids, n_base


def load_policy(sft_ckpt, device, attn="sdpa"):
    """SFT 체크포인트 디렉터리(<sft_out>/best|final|latest: adapter_model.safetensors + teacher_meta.json) → (pm, proc, tok, meta, new_ids, n_base)."""
    import torch
    from transformers import Qwen2VLForConditionalGeneration
    meta = load_sft_meta(sft_ckpt)
    if meta.get("dtype", "fp32") != "fp32":
        print(f"[주의] SFT dtype={meta.get('dtype')} — GRPO 는 프로젝트 규칙대로 fp32 로 적재한다", flush=True)
    if attn not in ("sdpa", "eager"):   # 수정(2026-10-16): 설계 검토 반영 — fp32 정확 경로만(flash 계열은 fp32 미지원)
        raise ValueError(f"attn={attn}: GRPO 는 fp32 — 'sdpa'(math 커널 고정) 또는 'eager' 만 허용")
    model = Qwen2VLForConditionalGeneration.from_pretrained(meta["base"], torch_dtype=torch.float32, attn_implementation=attn)
    proc = S.make_processor(meta["base"], meta["pixels"])
    tok = proc.tokenizer
    pm, new_ids, n_base = build_policy_from_base(model, tok, meta, sft_ckpt, device)
    # 수정(2026-10-16): 설계 검토 반영 — fp32 SDPA 를 정확한 math 커널로 고정·자기검사(SFT 발산 원인과 같은 문제를 막는다).
    #   CUDA 가 아니면 'unchanged'(CPU SDPA 는 math 경로). 실패하면 예외(건너뛰지 않음).
    print(f"[attn] 정책 적재 후: {S.exact_fp32_attention(torch.float32)}", flush=True)
    return pm, proc, tok, meta, new_ids, n_base


def trainable_params(pm):
    return [(n, p) for n, p in pm.named_parameters() if p.requires_grad]


class RefSnapshot:
    """π_ref = 적재 시점(SFT) 학습 파라미터 값의 사본. with snap.active(pm): 동안 학습 파라미터에 SFT 값을 넣고 끝나면 정책 값으로
    되돌린다(no_grad, .data 복사 → autograd 그래프 없음). 학습 순전파 전에만 호출한다(계산 그래프가 파라미터를 잡고 있지 않을 때)."""

    def __init__(self, pm):
        self.names = [n for n, _ in trainable_params(pm)]
        self.ref = [p.detach().clone() for _, p in trainable_params(pm)]

    def active(self, pm):
        import contextlib
        import torch

        @contextlib.contextmanager
        def ctx():
            ps = [p for _, p in trainable_params(pm)]
            if [n for n, _ in trainable_params(pm)] != self.names:
                raise RuntimeError("학습 파라미터 목록이 스냅샷과 다르다")
            backup = [p.detach().clone() for p in ps]
            with torch.no_grad():
                for p, r in zip(ps, self.ref):
                    p.data.copy_(r)
            try:
                yield
            finally:
                with torch.no_grad():
                    for p, b in zip(ps, backup):
                        p.data.copy_(b)
        return ctx()

    def max_abs_diff(self, pm):
        return max(float((p.detach() - r).abs().max()) for (_, p), r in zip(trainable_params(pm), self.ref))


# ================================================================ 프롬프트·영상 특징
def encode_prompt(proc, rec, meta, drop_cmd=False, drop_ego=False):
    """학습(SFT)과 같은 프롬프트(system 모드·이전 프레임·순서는 SFT meta). 반환 dict(ids (L,), pixel_values, grid)."""
    use_prev = bool(meta.get("use_prev", False))
    n_img = S.NUM_CAMS * (2 if use_prev else 1)
    msgs = S.build_messages(S.system_text(rec, meta["system"]), S.build_user_text(rec, drop_cmd, drop_ego), n_img)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs = S.load_images(rec, use_prev, meta.get("frame_order", "prev_first"))
    enc = proc(text=[text], images=imgs, return_tensors="pt")
    return dict(ids=enc["input_ids"][0], pixel_values=enc["pixel_values"], grid=enc["image_grid_thw"], token=rec["token"])


def image_features(pm, pixel_values, grid):
    """이미지 단위 시각 인코딩(sft_common.embed_inputs 와 같은 계산·같은 OOM 회피). 비전 동결 → no_grad, 프롬프트당 한 번
    계산해 표집·참조·정책 순전파 모두에 재사용(결정적이라 결과 동일)."""
    import torch
    base = S.base_of(pm)
    with torch.no_grad():
        pv = pixel_values.to(next(base.visual.parameters()).device).type(base.visual.get_dtype())
        sizes = [int(t * h * w) for t, h, w in grid.tolist()]
        if sum(sizes) != pv.shape[0]:
            raise ValueError(f"패치 수 {sum(sizes)} != pixel_values {pv.shape[0]}")
        return torch.cat([base.visual(p, grid_thw=g[None].to(pv.device)) for p, g in zip(pv.split(sizes), grid)], 0)


def embed_with_feats(pm, ids, feats):
    """ids (B,L), feats = 이 배치의 영상 토큰 특징을 행 순서대로 이은 텐서 → 입력 임베딩(새 토큰 행 = peft 학습 행 포함)."""
    base = S.base_of(pm)
    emb = base.model.embed_tokens(ids)
    m = ids == base.config.image_token_id
    if int(m.sum()) != feats.shape[0]:
        raise ValueError(f"영상 토큰 {int(m.sum())} != 영상 특징 {feats.shape[0]}")
    return emb.masked_scatter(m.unsqueeze(-1).expand_as(emb), feats.to(emb.dtype))


def _top_p_filter(logits, top_p):
    import torch
    if top_p >= 1.0:
        return logits
    sl, si = torch.sort(logits, descending=True, dim=-1)
    pr = torch.softmax(sl, -1)
    cum = pr.cumsum(-1)
    rm = (cum - pr) > top_p
    sl = sl.masked_fill(rm, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, si, sl)


def sample_answers(pm, prompts, feats, G, max_new, eos_id, pad_id, temperature=1.0, top_p=1.0, greedy=False,
                   generator=None, grammar=None):
    """프롬프트 P 개 × G 표집(KV 캐시, M-RoPE). 접두(prefill)는 프롬프트당 한 번 계산 후 캐시를 G 배 복제 — 각 행을 따로
    생성한 것과 같은 계산. 반환 dict(ids (P·G,T) [끝 뒤 pad], n_gen (P·G,) [끝 토큰 포함 길이], ended (P·G,) bool,
    logp (P·G,T) 표집 시점 로그확률(진단: 학습 순전파와 일치해야 함)). 행 순서 = 프롬프트 p 의 g 번째 → p·G+g.
    grammar(answer_grammar.AnswerGrammar) 를 주면 행마다 문법 상태를 따라 허용 집합 안에서만 고르고, logp 는 허용 집합으로
    다시 정규화한 값(log q). # 수정(2026-10-16): 설계 검토 반영"""
    import torch
    from transformers.cache_utils import DynamicCache
    base = S.base_of(pm)
    dev = next(base.model.parameters()).device
    if grammar is not None and max_new > grammar.n_pos:   # 수정(2026-10-16): 설계 검토 반영
        raise ValueError(f"문법 제약 생성: max_new {max_new} > 고정 길이 {grammar.n_pos}(정답 547 + <|im_end|>)")
    P = len(prompts)
    L = max(len(x["ids"]) for x in prompts)
    ids = torch.full((P, L), pad_id, dtype=torch.long)
    am = torch.zeros((P, L), dtype=torch.long)
    for i, x in enumerate(prompts):                      # 왼쪽 패딩(생성)
        n = len(x["ids"]); ids[i, L - n:] = x["ids"]; am[i, L - n:] = 1
    ids, am = ids.to(dev), am.to(dev)
    grid = torch.cat([x["grid"] for x in prompts], 0).to(dev)
    was_training = pm.training
    pm.eval()
    with torch.no_grad():
        emb = embed_with_feats(pm, ids, torch.cat(feats, 0))
        pos, deltas = base.get_rope_index(ids, grid, None, am)
        cache = DynamicCache()
        out = base.model(inputs_embeds=emb, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True,
                         return_dict=True)
        cache = out.past_key_values
        h = out.last_hidden_state[:, -1]
        if G > 1:
            cache.batch_repeat_interleave(G)
            h = h.repeat_interleave(G, 0); am = am.repeat_interleave(G, 0); deltas = deltas.repeat_interleave(G, 0)
        B = P * G
        toks = torch.full((B, max_new), pad_id, dtype=torch.long, device=dev)
        lps = torch.zeros((B, max_new), dtype=torch.float32, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        n_gen = torch.full((B,), max_new, dtype=torch.long, device=dev)
        states = [grammar.start() for _ in range(B)] if grammar is not None else None
        for t in range(max_new):
            logits = base.lm_head(h).float()
            if grammar is not None:   # 수정(2026-10-16): 설계 검토 반영 — 허용 집합(K 개)만 모아 그 안에서 표집·정규화
                idx, valid = grammar.gather_index([grammar.key(st) for st in states], dev)
                sub = logits.gather(1, idx).masked_fill(~valid, float("-inf"))
                if greedy:
                    j = sub.argmax(-1)
                    lp = torch.log_softmax(sub, -1).gather(-1, j[:, None])[:, 0]
                else:
                    lg = _top_p_filter(sub / temperature, top_p)
                    j = torch.multinomial(torch.softmax(lg, -1), 1, generator=generator)[:, 0]
                    lp = torch.log_softmax(sub / temperature, -1).gather(-1, j[:, None])[:, 0]
                nxt = idx.gather(1, j[:, None])[:, 0]
                for b in range(B):
                    if not bool(done[b]):
                        grammar.step(states[b], int(nxt[b]))
            elif greedy:
                nxt = logits.argmax(-1)
                lp = torch.log_softmax(logits, -1).gather(-1, nxt[:, None])[:, 0]
            else:
                lg = _top_p_filter(logits / temperature, top_p)
                prob = torch.softmax(lg, -1)
                nxt = torch.multinomial(prob, 1, generator=generator)[:, 0]
                lp = torch.log_softmax(logits / temperature, -1).gather(-1, nxt[:, None])[:, 0]
            nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
            toks[:, t] = nxt
            lps[:, t] = torch.where(done, torch.zeros_like(lp), lp)
            hit = (~done) & (nxt == eos_id)
            n_gen = torch.where(hit, torch.full_like(n_gen, t + 1), n_gen)
            done = done | hit
            if bool(done.all()) or t == max_new - 1:
                break
            am = torch.cat([am, torch.ones((B, 1), dtype=am.dtype, device=dev)], 1)
            p_t = (L + t + deltas).view(1, B, 1).expand(3, B, 1)        # HF: cache_position + rope_deltas
            e_t = base.model.embed_tokens(nxt[:, None])
            out = base.model(inputs_embeds=e_t, attention_mask=am, position_ids=p_t, past_key_values=cache,
                             use_cache=True, return_dict=True)
            cache = out.past_key_values
            h = out.last_hidden_state[:, -1]
    if was_training:
        pm.train()
    T = int(n_gen.max())
    return dict(ids=toks[:, :T].cpu(), n_gen=n_gen.cpu(), ended=done.cpu(), logp=lps[:, :T].cpu())


def answer_logprobs(pm, prompts, feats, gen_ids, n_gen, temperature=1.0, rope="mrope", grammar=None):
    """학습 순전파: 행마다 [프롬프트 + 생성] (오른쪽 패딩), 생성 토큰 위치에서만 lm_head(전 위치 로짓 미생성).
    prompts/feats = 행별 리스트(같은 프롬프트를 G 번 반복 가능), gen_ids (R,T), n_gen (R,).
    반환 logp (R,T) — 생성 길이 밖 위치 값은 0(마스크로 제외).
    grammar 를 주면 생성 토큰으로 문법 상태를 재현해 위치마다 허용 집합으로 정규화한 log q(표집과 같은 분포).
    # 수정(2026-10-16): 설계 검토 반영"""
    import torch
    base = S.base_of(pm)
    dev = next(base.model.parameters()).device
    R, T = gen_ids.shape
    lens = [len(x["ids"]) + int(n) for x, n in zip(prompts, n_gen)]
    Lm = max(len(x["ids"]) for x in prompts) + T
    ids = torch.zeros((R, Lm), dtype=torch.long)          # 패딩 자리(attention 0)의 id 값은 결과에 영향 없음
    am = torch.zeros((R, Lm), dtype=torch.long)
    idx = torch.zeros((R, T), dtype=torch.long)
    for r, x in enumerate(prompts):
        lp = len(x["ids"])
        ids[r, :lp] = x["ids"]; ids[r, lp:lp + T] = gen_ids[r]
        am[r, :lens[r]] = 1
        idx[r] = torch.arange(lp - 1, lp - 1 + T)          # 위치 lp−1+t 의 은닉 → 생성 토큰 t 예측
    ids, am, idx = ids.to(dev), am.to(dev), idx.to(dev)
    grid = torch.cat([x["grid"] for x in prompts], 0).to(dev)
    emb = embed_with_feats(pm, ids, torch.cat(feats, 0))
    pos = base.get_rope_index(ids, grid, None, am)[0] if rope == "mrope" else None
    if rope not in ("mrope", "1d"):
        raise ValueError(rope)
    out = base.model(input_ids=None, inputs_embeds=emb, attention_mask=am, position_ids=pos, use_cache=False,
                     return_dict=True)
    h = out.last_hidden_state.gather(1, idx[..., None].expand(R, T, out.last_hidden_state.shape[-1]))
    logits = base.lm_head(h).float() / temperature
    tgt = gen_ids.to(dev)
    if grammar is None:
        lse = torch.logsumexp(logits, -1)
    else:   # 수정(2026-10-16): 설계 검토 반영 — 허용 집합 위 logsumexp (R,T,K) 모음(전 어휘 마스크 텐서 없이)
        lists = []
        for r in range(R):
            n = int(n_gen[r])
            keys = grammar.replay_keys(gen_ids[r, :n].tolist())   # 허용 밖 토큰이면 ValueError(표집 경로 불일치)
            lists.append([grammar.allowed_list(k) for k in keys])
        K = max(len(l) for row in lists for l in row)
        gi = torch.empty((R, T, K), dtype=torch.long)
        gv = torch.zeros((R, T, K), dtype=torch.bool)
        for r, row in enumerate(lists):
            for t, l in enumerate(row):
                gi[r, t, :len(l)] = torch.as_tensor(l); gi[r, t, len(l):] = l[0]; gv[r, t, :len(l)] = True
            for t in range(len(row), T):             # 생성 길이 밖: 자기 자신 하나(값 0, 아래에서 0 으로 덮음)
                gi[r, t, :] = gen_ids[r, t]; gv[r, t, 0] = True
        gi, gv = gi.to(dev), gv.to(dev)
        lse = torch.logsumexp(logits.gather(-1, gi).masked_fill(~gv, float("-inf")), -1)
    lp = logits.gather(-1, tgt[..., None])[..., 0] - lse
    valid = torch.arange(T, device=dev)[None] < n_gen.to(dev)[:, None]
    return torch.where(valid, lp, torch.zeros_like(lp))


def gt_position_mask(tok, rec, T, exclude_intent=False):
    """GT 정답 토큰화(sft_common.tokenize_answer: SFT 와 같은 손실 마스크 규칙)의 감독 위치(label≠−100) → 길이 T bool.
    v2-sensor 는 고정 길이(547+종료) 문법이라 위치 k = 모든 유효 정답에서 같은 필드 → GT 값 미상(<na>)·미래 불완전 Intent/Action
    ·이전 프레임 없음 속력 위치를 정책 기울기·KL 에서 뺀다. GT 길이 밖(T > 548) 위치는 감독(형식 이탈 벌점이 닿게)."""
    a_ids, a_lab, _ = S.tokenize_answer(tok, rec)
    if len(a_ids) != ANSWER_TOKENS + 1:
        raise RuntimeError(f"{rec['token']}: GT 정답 {len(a_ids)} 토큰 != {ANSWER_TOKENS}+1 (고정 길이 가정 위반)")
    keep = [l != -100 for l in a_lab]
    if exclude_intent:
        seg = S.answer_segments(tok, rec)
        ii = S.SEG_NAMES.index("intent")
        keep = [k and s != ii for k, s in zip(keep, seg)]
    out = np.ones(T, dtype=bool)
    n = min(T, len(keep))
    out[:n] = keep[:n]
    return out


def grad_norm(params):
    import torch
    gs = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return float(torch.stack(gs).norm()) if gs else 0.0
