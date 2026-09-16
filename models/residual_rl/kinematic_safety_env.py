# 수정: nuScenes 장면을 재생하는 경량 kinematic(bicycle-model) 폐쇄루프 안전 env.
#   목적: 차량 제어부 Residual RL(Safety Control) 의 학습·평가 환경(CARLA 불필요).
#   - ego: base 컨트롤러(reference 추종)가 기본 제어를 내고, residual 이 안전 보정.
#   - 다른 에이전트: nuScenes gt_agent_fut_trajs(로그 미래궤적) 으로 전진.
#   - 안전: ego-에이전트 최소거리/충돌, comfort(jerk/lat) 측정 → SafetyReward.
#   설계 docs/residual_ppo_design.md. RL 라이브러리 불필요(자체 step API).
import numpy as np


def obb_separation(ca, ya, hla, hwa, cb, yb, hlb, hwb):
    """두 2D OBB(방향성 사각형)의 부호있는 분리거리. >0=틈(gap), <0=침투(충돌).
    SAT(분리축 정리): 각 박스의 길이축/폭축 4개에 투영 → max gap.
    ca/cb: 중심(2,), ya/yb: yaw(rad), hl/hw: 길이/폭 half-extent.
    (외접원 근사의 횡방향 과대평가를 제거 — 충돌모델 정교화)."""
    cya, sya = np.cos(ya), np.sin(ya)
    cyb, syb = np.cos(yb), np.sin(yb)
    ax = np.array([cya, sya]); ay = np.array([-sya, cya])   # A 길이축/폭축
    bx = np.array([cyb, syb]); by = np.array([-syb, cyb])
    T = np.asarray(cb, np.float64) - np.asarray(ca, np.float64)
    sep = -np.inf
    for n in (ax, ay, bx, by):
        projT = abs(T @ n)
        rA = hla * abs(ax @ n) + hwa * abs(ay @ n)
        rB = hlb * abs(bx @ n) + hwb * abs(by @ n)
        gap = projT - rA - rB
        if gap > sep:
            sep = gap
    return float(sep)


class KinematicBicycle:
    """간단 bicycle model. state=[x,y,yaw,v]."""
    def __init__(self, dt=0.5, wheelbase=2.8, v_max=20.0, a_max=4.0, steer_max=0.6):
        self.dt, self.L = dt, wheelbase
        self.v_max, self.a_max, self.steer_max = v_max, a_max, steer_max

    def step(self, s, accel, steer):
        x, y, yaw, v = s
        accel = np.clip(accel, -self.a_max, self.a_max)
        steer = np.clip(steer, -self.steer_max, self.steer_max)
        x += v * np.cos(yaw) * self.dt
        y += v * np.sin(yaw) * self.dt
        yaw += v / self.L * np.tan(steer) * self.dt
        v = np.clip(v + accel * self.dt, 0.0, self.v_max)
        return np.array([x, y, yaw, v], dtype=np.float32)


class KinematicSafetyEnv:
    """nuScenes 장면 재생 폐쇄루프 안전 env.

    한 에피소드 = 한 nuScenes 키프레임의 미래(fut_ts 스텝).
      reset(scene_idx) → obs
      step(residual) → obs, reward, done, info
    residual action = [Δaccel, Δsteer] (bounded). base = reference 추종(pure-pursuit + 속도제어).

    충돌은 박스 extent 기반: dist < ego_radius + agent_radius (하드 충돌),
    + 안전마진(SAFE_MARGIN) 이내면 '안전마진 위반'(graded). 관련성 없는(후방·먼) 에이전트는 제외.
    """
    EGO_RADIUS = 2.46      # m, obs 용 자차 외접원(충돌판정엔 미사용)
    EGO_L = 4.5            # m, 자차 길이 (OBB 충돌)
    EGO_W = 2.0            # m, 자차 폭 (OBB 충돌)
    SAFE_MARGIN = 1.0      # m, 추가 안전마진
    OFFROAD_LAT = 10.0     # m, reference 횡편차 임계

    def __init__(self, infos, dt=0.5, horizon=12, res_bound=(2.0, 0.2)):
        self.infos = infos
        self.dt, self.horizon = dt, horizon
        self.res_bound = np.asarray(res_bound, dtype=np.float32)  # [Δaccel, Δsteer]
        self.model = KinematicBicycle(dt=dt)
        self._scene = None

    # ---- scene 구성 ----
    def reset(self, scene_idx):
        info = self.infos[scene_idx]
        # reference: 자차 미래궤적(expert) 누적위치. ego_fut 가 델타면 cumsum.
        ego_fut = np.asarray(info.get('gt_ego_fut_trajs', np.zeros((self.horizon, 2))), np.float32)
        if ego_fut.ndim == 2 and ego_fut.shape[0] >= self.horizon:
            ref = np.cumsum(ego_fut[:self.horizon], axis=0)
        else:  # 직진 fallback (4m/step)
            ref = np.stack([np.arange(1, self.horizon + 1) * 4.0, np.zeros(self.horizon)], axis=1)
        self.ref = ref.astype(np.float32)
        # 다른 에이전트: gt_boxes 중심 + gt_agent_fut_trajs(델타→누적)
        boxes = np.asarray(info.get('gt_boxes', np.zeros((0, 7))), np.float32)
        fut = np.asarray(info.get('gt_agent_fut_trajs', np.zeros((0, self.horizon, 2))), np.float32)
        masks = np.asarray(info.get('gt_agent_fut_masks', np.zeros((len(boxes), self.horizon))), np.float32)
        self.agents = []
        for i in range(len(boxes)):
            start = boxes[i, :2]
            if i < len(fut) and fut[i].shape == (self.horizon, 2):
                traj = start[None] + np.cumsum(fut[i], axis=0)
            else:
                traj = np.tile(start, (self.horizon, 1))
            m = masks[i] if i < len(masks) else np.ones(self.horizon)
            w, l = (boxes[i, 3], boxes[i, 4]) if boxes.shape[1] >= 5 else (1.8, 4.0)
            yaw0 = float(boxes[i, 6]) if boxes.shape[1] >= 7 else 0.0
            # 헤딩: 궤적 진행방향(이동≥0.3m), 정지 시 초기 yaw (OBB 정렬용)
            full = np.concatenate([start[None], traj], axis=0)  # (H+1,2)
            d = np.diff(full, axis=0)                            # (H,2)
            head = np.where(np.hypot(d[:, 0], d[:, 1]) > 0.3,
                            np.arctan2(d[:, 1], d[:, 0]), yaw0).astype(np.float32)
            self.agents.append(dict(
                traj=traj.astype(np.float32), m=m.astype(np.float32),
                hl=0.5 * float(l), hw=0.5 * float(w), head=head))
        # ego 초기상태: 원점, +x 방향, 초기속도(ego_status 있으면 사용)
        es = np.asarray(info.get('ego_status', np.zeros(10)), np.float32)
        v0 = float(np.linalg.norm(es[:2])) if es.size >= 2 and np.isfinite(es[:2]).all() else 5.0
        self.ego = np.array([0.0, 0.0, 0.0, v0], dtype=np.float32)
        self.t = 0
        self.prev_accel = 0.0
        return self._obs()

    # ---- base 컨트롤러(reference 추종): pure-pursuit + 속도매칭 ----
    def _base_control(self):
        x, y, yaw, v = self.ego
        look = min(self.t + 1, self.horizon - 1)
        tgt = self.ref[look]
        dx, dy = tgt[0] - x, tgt[1] - y
        # 목표 heading
        alpha = np.arctan2(dy, dx) - yaw
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        Ld = max(np.hypot(dx, dy), 1e-2)
        steer = np.arctan2(2 * self.model.L * np.sin(alpha), Ld)
        # 목표 속도: reference 진행 속도
        v_ref = np.linalg.norm(self.ref[look] - self.ref[max(look - 1, 0)]) / self.dt
        accel = np.clip((v_ref - v) / self.dt, -self.model.a_max, self.model.a_max)
        return np.array([accel, steer], dtype=np.float32)

    def _active_agents(self):
        """현재 t 의 활성 에이전트 [(pos(2,), yaw, hl, hw), ...]."""
        ti = min(self.t, self.horizon - 1)
        out = []
        for a in self.agents:
            if a['m'][ti] > 0:
                out.append((a['traj'][ti], float(a['head'][ti]), a['hl'], a['hw']))
        return out

    def _agent_positions(self):
        act = self._active_agents()
        if not act:
            return np.zeros((0, 2), np.float32)
        return np.asarray([p for p, _, _, _ in act], np.float32)

    def _min_surface_dist(self):
        """ego-에이전트 OBB 표면거리 최소값(부호있음). <0=겹침=충돌.
        외접원 근사 대신 SAT 로 차량 형상·yaw 반영 → phantom 충돌 제거."""
        act = self._active_agents()
        if not act:
            return np.inf
        ec, eyaw = self.ego[:2], float(self.ego[2])
        ehl, ehw = 0.5 * self.EGO_L, 0.5 * self.EGO_W
        best = np.inf
        for pos, ayaw, ahl, ahw in act:
            sep = obb_separation(ec, eyaw, ehl, ehw, pos, ayaw, ahl, ahw)
            if sep < best:
                best = sep
        return float(best)

    def _obs(self):
        x, y, yaw, v = self.ego
        ap = self._agent_positions()
        # 가장 가까운 K=5 에이전트 상대위치(ego 프레임)
        K = 5
        rel = np.zeros((K, 2), np.float32)
        if len(ap) > 0:
            d = ap - self.ego[:2]
            order = np.argsort(np.linalg.norm(d, axis=1))[:K]
            c, s = np.cos(-yaw), np.sin(-yaw)
            for j, idx in enumerate(order):
                rel[j] = [c * d[idx, 0] - s * d[idx, 1], s * d[idx, 0] + c * d[idx, 1]]
        look = min(self.t + 1, self.horizon - 1)
        ref_rel = self.ref[look] - self.ego[:2]
        base = self._base_control()
        return np.concatenate([[v, yaw, self.prev_accel], ref_rel, base, rel.reshape(-1)]).astype(np.float32)

    def step(self, residual):
        residual = np.clip(np.asarray(residual, np.float32), -self.res_bound, self.res_bound)
        base = self._base_control()
        accel = base[0] + residual[0]
        steer = base[1] + residual[1]
        prev_v = self.ego[3]
        self.ego = self.model.step(self.ego, accel, steer)
        self.t += 1
        # 안전·comfort 측정 (박스 extent 기반 surface distance)
        surf = self._min_surface_dist()
        collision = surf < 0.0
        safety_violation = surf < self.SAFE_MARGIN
        lat_dev = abs(self.ego[1] - self.ref[min(self.t, self.horizon - 1)][1])
        offroad = lat_dev > self.OFFROAD_LAT
        jerk = abs(accel - self.prev_accel) / self.dt
        lat_accel = self.ego[3] * self.ego[3] * np.tan(np.clip(steer, -1, 1)) / self.model.L
        self.prev_accel = accel
        done = self.t >= self.horizon or collision or offroad
        info = dict(progress=float(self.ego[3] * self.dt), collision=bool(collision),
                    safety_violation=bool(safety_violation), offroad=bool(offroad),
                    jerk=float(jerk), lat_accel=float(lat_accel), residual=residual.copy(),
                    min_surface=float(surf), lat_dev=float(lat_dev),
                    done_completed=bool(self.t >= self.horizon and not collision and not offroad))
        return self._obs(), info, done

    @property
    def obs_dim(self):
        return 3 + 2 + 2 + 5 * 2  # v,yaw,prev_accel + ref_rel + base + 5 agents
