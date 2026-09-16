# 수정: IL base 출력에 bounded residual 을 주입하는 안전제어 wrapper.
#   a_final = base(s) + clip(Δ_RL, -bound, +bound).  base 안정성 보존(파국 발산 불가).
#   주입 지점(설계 docs/residual_ppo_design.md):
#     - control-level: PID 출력(steer/throttle/brake) 직후 → VehicleControl 직전.
#     - planning-level: plan_temp/plan_spat 궤적 직후 → PID 호출 전.
import numpy as np


class ResidualControlWrapper:
    """control-level residual 주입(Phase 1).

    base 제어 [steer, throttle, brake] 에 Δ 를 더하고 물리 범위로 clip.
    """

    def __init__(self, bound=(0.1, 0.2, 0.2)):
        self.bound = np.asarray(bound, dtype=np.float32)  # [Δsteer, Δthrottle, Δbrake] 상한

    def apply(self, base_control, residual):
        base = np.asarray(base_control, dtype=np.float32)
        delta = np.clip(np.asarray(residual, dtype=np.float32), -self.bound, self.bound)
        out = base + delta
        # 물리 범위: steer[-1,1], throttle[0,1], brake[0,1]
        out[0] = np.clip(out[0], -1.0, 1.0)
        out[1] = np.clip(out[1], 0.0, 1.0)
        out[2] = np.clip(out[2], 0.0, 1.0)
        return out


class ResidualPlanWrapper:
    """planning-level residual 주입(Phase 2).

    base 궤적 (N,2) waypoint 에 Δtraj 를 더하고 ±max_shift 로 clip.
    """

    def __init__(self, max_shift=2.0):
        self.max_shift = float(max_shift)

    def apply(self, base_traj, residual):
        base = np.asarray(base_traj, dtype=np.float32)
        delta = np.clip(np.asarray(residual, dtype=np.float32).reshape(base.shape),
                        -self.max_shift, self.max_shift)
        return base + delta
