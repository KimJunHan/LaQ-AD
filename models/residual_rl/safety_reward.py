# 수정: 안전제어 residual RL 의 보상 함수(Bench2Drive Driving Score 분해 + dense shaping).
#   docs/residual_ppo_design.md §2 Reward 와 정합. env(폐쇄루프)에서 매 tick 호출.
#
#수정(2026-08-11): 가중치 정책 3종을 설계해 등록한다.
#
#   [왜 3종인가]
#   안전 보상에는 자명한 퇴화 해가 있다 — **정지하면 충돌하지 않는다**. 단일 가중치 집합으로
#   낮은 충돌률을 보고하면 그 결과가 실제 회피 능력에서 온 것인지 주행 포기에서 온 것인지
#   구별할 수 없다. 서로 다른 안전/진행 교환비를 갖는 세 정책을 함께 학습·보고하면 결과가
#   한 점이 아니라 **경계면(frontier)** 으로 제시되어, 안전 개선이 진행량을 얼마나 대가로
#   치렀는지가 드러난다.
#
#   [설계 기준 — 임의값이 아니다]
#   세 정책은 `collision / progress` 비(=충돌 1회가 몇 m 의 전진과 등가인가)로 규정한다.
#   본 env 의 한 에피소드는 horizon 12 스텝 x dt 0.5s = 6초이고, 도심 주행 속도 ~10 m/s
#   기준 에피소드당 전진량은 대략 60 m 이다. 이 60 m 를 기준선으로 삼는다.
#
#     safety_first : 비 200  -> 충돌 1회 = 200 m. 에피소드 전체 전진량(60 m)으로도 상쇄
#                              불가. 안전 항이 구조적으로 지배한다(nuPlan 의 곱셈 항과 같은 취지).
#     balanced     : 비  50  -> 충돌 1회 ~= 에피소드 전진량과 같은 크기. 실질적 교환이 발생하는
#                              지점. Bench2Drive Driving Score 의 완주율x위반벌점 구성과 정합.
#     progress_first: 비  15 -> 충돌 1회 = 15 m, 에피소드의 1/4 미만. 진행이 안전을 이길 수
#                              있도록 의도적으로 허용한다. 퇴화 해 검증용 대조군이다.
#
#   나머지 항은 이 비를 유지하면서 함께 조정한다. comfort/residual 은 정책이 base 제어에서
#   과도하게 이탈하는 것을 막는 정규화 항이므로, 안전 우선일수록 이탈 허용폭을 넓게 둔다
#   (회피 기동에는 큰 보정이 필요하기 때문).
import numpy as np


# 정책별 가중치. 위 설계 기준 참조.
REWARD_POLICIES = {
    # 안전 우선 — 충돌/안전마진 지배. 회피를 위한 큰 보정을 허용(residual/comfort 완화).
    "safety_first": dict(
        progress=0.5, collision=100.0, safety=10.0, redlight=20.0, offroad=10.0,
        comfort=0.05, tracking=0.1, residual=0.02, completion=10.0,
    ),
    # 균형 — Bench2Drive Driving Score 구성에 맞춘 기준 정책(기존 기본값).
    "balanced": dict(
        progress=1.0, collision=50.0, safety=5.0, redlight=10.0, offroad=5.0,
        comfort=0.1, tracking=0.2, residual=0.05, completion=20.0,
    ),
    # 진행 우선 — 퇴화 해(정지) 대조군. 안전 개선이 주행 포기가 아님을 보이는 데 쓴다.
    "progress_first": dict(
        progress=2.0, collision=30.0, safety=3.0, redlight=10.0, offroad=5.0,
        comfort=0.2, tracking=0.3, residual=0.1, completion=40.0,
    ),
}


class SafetyReward:
    """안전 중심 보상. 충돌·신호위반·이탈은 큰 페널티, 전진은 dense 보상,
    급조작(comfort)·residual 크기는 정규화 페널티.

    policy: REWARD_POLICIES 의 키. w 를 직접 주면 policy 보다 우선한다.
    """

    def __init__(self, w=None, policy="balanced", safe_margin=1.0):
        self.safe_margin = safe_margin
        if w is not None:
            self.policy = "custom"
            self.w = dict(w)
        else:
            if policy not in REWARD_POLICIES:
                raise KeyError(
                    f"알 수 없는 보상 정책 '{policy}'. 가능한 값: {sorted(REWARD_POLICIES)}"
                )
            self.policy = policy
            self.w = dict(REWARD_POLICIES[policy])

    def trade_off_ratio(self):
        """충돌 1회가 몇 m 의 전진과 등가인지. 세 정책을 구분하는 설계 지표."""
        return self.w["collision"] / self.w["progress"]

    def step(self, info):
        """info: env 제공 dict.
        keys: progress, collision, safety_violation, min_surface, offroad,
              jerk, lat_accel, lat_dev, residual, done_completed (+ optional redlight)."""
        w = self.w
        r = w["progress"] * info.get("progress", 0.0)
        r -= w["collision"] * float(info.get("collision", False))
        # 안전마진 shaping: 마진 이내에서 표면거리가 작을수록 큰 페널티(충돌 직전 회피 유도)
        surf = info.get("min_surface", np.inf)
        if np.isfinite(surf) and surf < self.safe_margin:
            r -= w["safety"] * (self.safe_margin - max(surf, 0.0)) / self.safe_margin
        r -= w["redlight"] * float(info.get("redlight", False))
        r -= w["offroad"] * float(info.get("offroad", False))
        comfort = abs(info.get("jerk", 0.0)) + abs(info.get("lat_accel", 0.0))
        r -= w["comfort"] * comfort
        r -= w["tracking"] * abs(info.get("lat_dev", 0.0))
        res = np.asarray(info.get("residual", 0.0), dtype=np.float32)
        r -= w["residual"] * float(np.linalg.norm(res))
        if info.get("done_completed", False):
            r += w["completion"]
        return float(r)
