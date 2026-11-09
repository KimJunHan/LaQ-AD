# 수정(2026-11-06): 사용자 결정 "B2D 폐쇄루프 E2 + 나침반 NaN 틱 처리(권장안)" — 새 파일(원본 없음)
#   폐쇄루프 자차 상태 4 값 [속도, 종방향 가속도, yaw rate, available] 과 교사 프롬프트 자차 상태 줄을 한 곳에서 만든다.
#   CARLA·torch 를 import 하지 않는 순수 파이썬 모듈(teacher_refresh.py 와 같은 방식) — CPU 단위 시험
#   tools/verify_b2d/test_closedloop_e2_cpu.py 로 검사한다.
#
#   [식] 학습 쪽과 같은 식(코드로 확인):
#     tools/data_converter/bench2drive_converter.py: ego_vel = [anno.speed, 0, 0], ego_accel = [acc[0], −acc[1], acc[2]],
#       ego_rotation_rate = −anno.angular_velocity
#     tools/vla_teacher/gen_teacher_gt_b2d.ego_inputs_b2d: (max(ego_vel[0], 0), ego_accel[0], sgn × ego_rotation_rate[2]),
#       sgn = +1 (b2d_conventions_train.json yaw_rate_regress_slope = +1.3937 > 0)
#     → 폐쇄루프: speed = max(speedometer, 0), accel = IMU accelerometer[0], yaw = YAW_SIGN × (−IMU gyroscope[2]).
#     (데이터의 anno.acceleration 은 IMU 값: 정지 표본 ego_accel z ≈ 9.77 = 중력 포함, 기반 에이전트 custom_status 도 같은 대응.)
#   [교사·학생 단일 출처] 프롬프트 줄 = tools/vla_teacher/common_b2d.ego_line(speed, accel, yaw) 그대로.
#     학생 4 값 = 그 줄을 tools/b2d/build_b2d_ego_input.PAT 로 다시 읽은 10 진 값(float32) + available 1 — 학습 표(pkl)를 만든
#     식과 같은 함수·같은 정규식이라 같은 줄이면 비트 단위로 같다.
#   [나침반 NaN 틱(사용자 결정 2026-11-06 권장안)] 기반 에이전트(hipad_b2d_agent.tick)는 compass 가 NaN 이면 가속도·각속도를 0 으로
#     채운다. 여기서는 0 대신 직전 정상 틱의 IMU 값(가속도 3, 각속도 3)을 유지한다. 속도는 속도계 값 그대로.
#     경로 시작 등 정상값이 아직 없을 때만 가속도·yaw rate 0(available 은 1 그대로).
#     비정상 판정 = compass NaN(기반 에이전트와 같은 조건) 또는 가속도·각속도에 유한하지 않은 값(별도 계수).
#     속도가 유한하지 않으면 즉시 실패(대체값 없음).
#   [형식 범위 밖(사용자 결정 2026-11-09)] |가속도| ≥ 99.995, |yaw| ≥ 9.9995, 속도 ≥ 99.95 는 예외 대신 형식 범위 안 최댓값
#     (속도 99.9, 가속도 ±99.99, yaw ±9.999; 부호 유지)으로 자르고 기록한다. 교사 프롬프트 줄·학생 입력은 자른 같은 값(단일 출처).
#     속도 NaN 은 지금처럼 실패. (수정 2026-11-09)
#   [기록] 경로별 NaN 틱 수·유지 나이(틱)·시작 0 채움 횟수·연속 구간(시작 틱, 길이)·교사 갱신 틱에서의 상태를 summary() 로 낸다.
import importlib.util
import math
import os
import sys

import numpy as np

YAW_SIGN = 1.0   # b2d_conventions_train.json yaw_rate_regress_slope(+1.3937) > 0 → gen_teacher_gt_b2d.ego_inputs_b2d sgn = +1
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_VLA_DIR = os.path.join(_REPO, "tools", "vla_teacher")
_EGO_TOOL = os.path.join(_REPO, "tools", "b2d", "build_b2d_ego_input.py")
STATE_OK, STATE_HELD, STATE_ZERO = "ok", "held", "zero_fill"
# 수정(2026-11-09): 사용자 결정 "형식 범위 밖 = 범위 안 최댓값으로 자르고 기록" — (이 값 이상이면 자름, 자른 값) = common_b2d.ego_line 범위 조건
CLIP_LIMITS = {"speed": (99.95, 99.9), "accel": (99.995, 99.99), "yaw": (9.9995, 9.999)}  # 수정(2026-11-09)

__all__ = ["EgoStateHold", "ego_line_and_input", "parse_ego_line", "clip_to_format", "YAW_SIGN", "CLIP_LIMITS"]  # 수정(2026-11-09): clip_to_format·CLIP_LIMITS 추가

_cache = {}


def _common_b2d():
    if "b2d" not in _cache:
        if _VLA_DIR not in sys.path:
            sys.path.insert(0, _VLA_DIR)
        import common_b2d  # noqa: E402  (교사 정답·프롬프트 생성과 같은 모듈)
        if os.path.realpath(os.path.dirname(common_b2d.__file__)) != os.path.realpath(_VLA_DIR):
            raise ImportError(f"common_b2d 가 다른 경로에서 적재됐다: {common_b2d.__file__} (기대 {_VLA_DIR})")
        _cache["b2d"] = common_b2d
    return _cache["b2d"]


def _ego_tool():
    if "tool" not in _cache:
        spec = importlib.util.spec_from_file_location("_build_b2d_ego_input", _EGO_TOOL)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)   # 최상위는 상수·함수 정의뿐(__main__ 보호)
        _cache["tool"] = mod
    return _cache["tool"]


def parse_ego_line(line):
    """프롬프트 자차 상태 줄 → float32 (4,) — tools/b2d/build_b2d_ego_input.build 와 같은 정규식·같은 변환."""
    tool = _ego_tool()
    m = tool.PAT.fullmatch(line)
    if m is None:
        raise ValueError(f"자차 상태 줄 형식 불일치(결측 줄은 폐쇄루프에서 쓰지 않는다): {line!r}")
    return np.asarray([float(m.group(1)), float(m.group(2)), float(m.group(3)), 1.0], np.float32)


def ego_line_and_input(vals):
    """(speed, accel, yaw) → (common_b2d.ego_line 문자열, 그 줄을 다시 읽은 float32 (4,)).
    범위 밖(속도 ≥ 99.95, |가속| ≥ 99.995, |yaw| ≥ 9.9995)은 ego_line 이 ValueError — 여기서는 자르지 않는다.
    수정(2026-11-09): 폐쇄루프 값은 EgoStateHold.update 가 clip_to_format 으로 먼저 잘라 넘긴다(사용자 결정)."""
    line = _common_b2d().ego_line(*vals)
    return line, parse_ego_line(line)


def clip_to_format(vals):  # 수정(2026-11-09): 형식 범위 밖 값 → 범위 안 최댓값(부호 유지)
    """(speed, accel, yaw) → (자른 값 튜플, [(신호, 원값, 자른 값)]). 범위 안이면 값 그대로(비트 불변)."""  # 수정(2026-11-09)
    out, ev = [], []  # 수정(2026-11-09)
    for name, v in zip(("speed", "accel", "yaw"), vals):  # 수정(2026-11-09)
        lim, mx = CLIP_LIMITS[name]  # 수정(2026-11-09)
        if abs(v) >= lim:  # 수정(2026-11-09): ego_line 범위 조건(speed < 99.95, |a| < 99.995, |w| < 9.9995)의 반대
            c = math.copysign(mx, v)  # 수정(2026-11-09): 부호 유지(속도는 max(v,0) 뒤라 항상 +)
            ev.append((name, float(v), c))  # 수정(2026-11-09)
            v = c  # 수정(2026-11-09)
        out.append(v)  # 수정(2026-11-09)
    return tuple(out), ev  # 수정(2026-11-09)


def _finite_all(x):
    return all(math.isfinite(float(v)) for v in x)


class EgoStateHold:
    """틱마다 update(step, speed, compass, accel3, gyro3) → dict(vals, accel3, gyro3, state, hold_age)."""

    def __init__(self, route_id=None):
        self.route_id = route_id
        self.last_accel3 = None     # 직전 정상 틱 가속도(3)
        self.last_gyro3 = None      # 직전 정상 틱 각속도(3)
        self.last_ok_step = None
        self.n_ticks = 0
        self.n_invalid = 0          # NaN(비정상) 틱 수 = n_held + n_zero_fill
        self.n_invalid_compass = 0  # compass NaN
        self.n_invalid_imu_only = 0  # compass 정상인데 가속도·각속도 비유한
        self.n_held = 0
        self.n_zero_fill = 0
        self.hold_age_hist = {}     # 유지 나이(틱) → 횟수
        self.runs = []              # [시작 step, 길이, 'held'|'zero_fill'] 연속 비정상 구간
        self.first_step = None
        self.n_refresh = 0
        self.refresh_states = {STATE_OK: 0, STATE_HELD: 0, STATE_ZERO: 0}
        self.refresh_hold_age_hist = {}
        self.clip_events = []  # 수정(2026-11-09): [step, 신호, 원값, 자른 값, 교사 갱신 틱 여부(note_refresh 에서 채움)]
        self.n_clip_ticks = 0  # 수정(2026-11-09): 하나라도 자른 틱 수
        self.n_clip_by_signal = {"speed": 0, "accel": 0, "yaw": 0}  # 수정(2026-11-09)
        self.n_refresh_clipped = 0  # 수정(2026-11-09): 교사 갱신 틱(프롬프트·학생 값에 실제로 쓰인 틱)에서 자른 횟수

    def update(self, step, speed, compass, accel3, gyro3):
        speed = float(speed)
        if not math.isfinite(speed):
            raise ValueError(f"속도계 값이 유한하지 않다: {speed} (step {step})")
        accel3 = [float(v) for v in accel3]
        gyro3 = [float(v) for v in gyro3]
        if len(accel3) != 3 or len(gyro3) != 3:
            raise ValueError(f"IMU 가속도·각속도 길이 {len(accel3)}/{len(gyro3)} != 3")
        if self.first_step is None:
            self.first_step = step
        self.n_ticks += 1
        compass_nan = math.isnan(float(compass))
        imu_ok = _finite_all(accel3) and _finite_all(gyro3)
        if not compass_nan and imu_ok:
            self.last_accel3, self.last_gyro3, self.last_ok_step = accel3, gyro3, step
            state, age, a3, g3 = STATE_OK, 0, accel3, gyro3
        else:
            self.n_invalid += 1
            if compass_nan:
                self.n_invalid_compass += 1
            else:
                self.n_invalid_imu_only += 1
            if self.last_ok_step is not None:
                state, age, a3, g3 = STATE_HELD, step - self.last_ok_step, self.last_accel3, self.last_gyro3
                self.n_held += 1
                self.hold_age_hist[age] = self.hold_age_hist.get(age, 0) + 1
            else:
                state, age, a3, g3 = STATE_ZERO, None, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
                self.n_zero_fill += 1
            if self.runs and self.runs[-1][0] + self.runs[-1][1] == step and self.runs[-1][2] == state:
                self.runs[-1][1] += 1
            else:
                self.runs.append([step, 1, state])
        vals = (max(speed, 0.0) + 0.0, float(a3[0]), YAW_SIGN * (-float(g3[2])))  # 수정(2026-11-09): 검토 — 속도 −0.0 이면 "speed -0.0" 이 되어 학습 형식 정규식에 안 맞음 → +0.0 으로 정규화(양수 비트 불변)
        vals_raw = vals  # 수정(2026-11-09): 자르기 전 값(기록용)
        vals, clipped = clip_to_format(vals)  # 수정(2026-11-09): 형식 범위 밖 → 범위 안 최댓값
        if clipped:  # 수정(2026-11-09): 조용히 넘기지 않기 — 경로별 기록
            self.n_clip_ticks += 1  # 수정(2026-11-09)
            for name, raw, c in clipped:  # 수정(2026-11-09)
                self.n_clip_by_signal[name] += 1  # 수정(2026-11-09)
                self.clip_events.append([step, name, raw, c, False])  # 수정(2026-11-09)
        return dict(vals=vals, vals_raw=vals_raw, clipped=clipped,  # 수정(2026-11-09): vals_raw·clipped 추가
                    accel3=list(a3), gyro3=list(g3), state=state, hold_age=age, step=step)

    def note_refresh(self, res):
        """교사 갱신 틱(학생 E2 값이 정해지는 틱)의 상태를 따로 센다."""
        self.n_refresh += 1
        self.refresh_states[res["state"]] += 1
        if res["state"] == STATE_HELD:
            a = res["hold_age"]
            self.refresh_hold_age_hist[a] = self.refresh_hold_age_hist.get(a, 0) + 1
        if res["clipped"]:  # 수정(2026-11-09): 갱신 틱에서 자른 경우 표시
            self.n_refresh_clipped += 1  # 수정(2026-11-09)
            for e in self.clip_events:  # 수정(2026-11-09)
                if e[0] == res["step"]:  # 수정(2026-11-09)
                    e[4] = True  # 수정(2026-11-09)

    def summary(self):
        return dict(route_id=self.route_id, n_ticks=self.n_ticks, first_step=self.first_step,
                    n_nan_ticks=self.n_invalid, n_nan_compass=self.n_invalid_compass,
                    n_nonfinite_imu_only=self.n_invalid_imu_only, n_held=self.n_held, n_zero_fill_start=self.n_zero_fill,
                    hold_age_ticks_max=max(self.hold_age_hist) if self.hold_age_hist else 0,
                    hold_age_ticks_hist={str(k): v for k, v in sorted(self.hold_age_hist.items())},
                    nan_runs=[list(r) for r in self.runs],
                    n_teacher_refresh=self.n_refresh, refresh_states=dict(self.refresh_states),
                    refresh_hold_age_ticks_hist={str(k): v for k, v in sorted(self.refresh_hold_age_hist.items())},
                    n_clip_ticks=self.n_clip_ticks, n_clip_by_signal=dict(self.n_clip_by_signal),  # 수정(2026-11-09): 자르기 기록
                    n_refresh_clipped=self.n_refresh_clipped,  # 수정(2026-11-09)
                    clip_events=[dict(step=e[0], signal=e[1], raw=e[2], clipped=e[3], refresh=e[4]) for e in self.clip_events])  # 수정(2026-11-09)
