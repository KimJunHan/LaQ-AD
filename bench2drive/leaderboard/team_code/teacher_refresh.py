# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   폐쇄루프 교사(Qwen2-VL-2B LoRA SFT) 출력의 2 Hz 갱신 + 직전 출력 재사용(사용자 결정 2026-10-16, 메모 project_b2d_teacher_2hz).
#   CARLA·torch 를 import 하지 않는 순수 파이썬 모듈 — CPU 단위 시험(tools/verify_b2d/test_teacher_refresh_cpu.py)으로 검사한다.
#
#   [시간 기준] 갱신은 '틱 수'가 아니라 시뮬레이션 시각(초)으로 판단한다. leaderboard 는 20 Hz(fixed_delta 0.05 s)이고
#     기반 에이전트(hipad_b2d_agent.py) 는 timestamp = step / frame_rate(20) 를 모델에 넘긴다. 같은 값을 여기에 넣는다.
#     규칙: 출력이 아직 없거나, 장면(route)이 바뀌었거나, t − t_last ≥ REFRESH_PERIOD − EPS 이면 교사를 다시 부른다.
#     20 Hz 에서는 10 틱마다(0, 0.5, 1.0 … s) 갱신되고, 사이 9 틱은 직전 출력을 그대로 쓴다(나이 0.00–0.45 s).
#     학습 캐시 규칙(10 Hz 데이터, clip 별 f0 = infos 의 첫 frame_idx, keyframe = f0, f0+5, …, 나이 0.0–0.4 s;
#     models/encoder/vl_tokens_cache/loading_b2d.py b2d_keyframes)과 갱신 주기·시작점이 같다(경로 첫 틱 = 첫 keyframe).
#     수정(2026-10-19): keyframe 기준을 frame_idx%5 → clip 첫 프레임 f0 기준으로 변경(일부 clip 은 frame 1 부터 시작).
#     차이: 20 Hz 폐쇄루프의 마지막 틱 나이 0.45 s 는 학습에 없는 값(10 개 중 1 개 틱) — 보고서의 열린 질문.
#   [실패 처리] 교사 호출이 예외를 내면 그대로 올린다(직전 출력으로 대신하지 않는다 — 조용한 skip 금지).
#     교사가 None 을 돌려줘도 즉시 실패.
#   [지연 모드] mode="delayed" 는 오프라인 비교 전용(에이전트에서는 생성자에서 거부): 갱신 시점에 계산한 출력을 다음 갱신 때부터
#     쓴다(= 한 주기 늦은 출력). 폐쇄루프 에이전트는 mode="fresh" 만 허용.
import math

REFRESH_PERIOD = 0.5   # s (2 Hz)
EPS = 1e-6             # 부동소수 비교 여유(0.05 × 10 = 0.5000000000000001 등)

__all__ = ["TeacherRefreshScheduler", "REFRESH_PERIOD", "keyframe_of", "teacher_age_s"]


def keyframe_of(frame_idx, f0=0, stride=5):
    """학습 캐시 규칙(10 Hz B2D, loading_b2d.b2d_keyframes 와 같음): clip 첫 프레임 f0 기준
    kf = f0 + stride·((f − f0) // stride)."""
    f, f0 = int(frame_idx), int(f0)
    if f < f0:
        raise ValueError(f"frame_idx {f} < clip 첫 프레임 {f0}")
    return f0 + stride * ((f - f0) // stride)


def teacher_age_s(frame_idx, f0=0, stride=5, hz=10.0):
    """학습 캐시 규칙에서 프레임 f 의 교사 출력 나이(초) = (f − kf) / 10."""
    f = int(frame_idx)
    return (f - keyframe_of(f, f0, stride)) / hz


class TeacherRefreshScheduler:
    """teacher_fn(sim_t, **inputs) → 교사 출력(임의 객체, 보통 dict(qwen_hidden, qwen_mask, qwen_type)).

    step(sim_t, scene_id, **inputs) → (output, info). info = dict(refreshed, age_s, t_teacher, n_calls).
    """

    def __init__(self, teacher_fn, period=REFRESH_PERIOD, mode="fresh", allow_delayed=False):
        if mode not in ("fresh", "delayed"):
            raise ValueError(f"mode {mode!r} (허용 fresh|delayed)")
        if mode == "delayed" and not allow_delayed:
            raise ValueError("mode='delayed' 는 오프라인 지연 비교 전용 — allow_delayed=True 를 명시해야 한다(폐쇄루프 에이전트 금지)")
        if not (period > 0):
            raise ValueError(f"period {period}")
        self.teacher_fn = teacher_fn
        self.period = float(period)
        self.mode = mode
        self.reset(None)
        self.n_calls = 0
        self.age_log = []          # (scene_id, sim_t, age_s, refreshed)

    def reset(self, scene_id):
        self.scene_id = scene_id
        self.t_last = None         # 마지막 갱신 시각
        self.out = None            # 지금 쓰는 출력
        self.t_out = None          # 지금 쓰는 출력이 계산된 입력 시각
        self.pending = None        # delayed 모드: 다음 갱신 때 쓸 출력
        self.t_pending = None
        self.t_prev_step = None

    def _due(self, t):
        return self.t_last is None or (t - self.t_last) >= self.period - EPS

    def step(self, sim_t, scene_id=None, **inputs):
        t = float(sim_t)
        if not math.isfinite(t):
            raise ValueError(f"sim_t 가 유한하지 않다: {sim_t}")
        if scene_id != self.scene_id:
            self.reset(scene_id)
        if self.t_prev_step is not None and t < self.t_prev_step - EPS:
            raise RuntimeError(f"시각이 거꾸로 갔다: {self.t_prev_step} → {t} (같은 장면 안). 장면 전환이면 scene_id 를 바꿔야 한다.")
        self.t_prev_step = t
        refreshed = False
        if self._due(t):
            res = self.teacher_fn(t, **inputs)        # 예외는 그대로 올린다
            if res is None:
                raise RuntimeError(f"교사 출력이 None (t={t})")
            self.n_calls += 1
            refreshed = True
            self.t_last = t
            if self.mode == "fresh":
                self.out, self.t_out = res, t
            else:  # delayed: 첫 갱신은 이전 출력이 없으므로 현재 것을 쓴다(장면 첫 프레임 복제 규칙과 같음) — 기록에 표시
                if self.pending is None:
                    self.out, self.t_out = res, t
                else:
                    self.out, self.t_out = self.pending, self.t_pending
                self.pending, self.t_pending = res, t
        if self.out is None:
            raise RuntimeError("교사 출력이 없다(갱신 규칙 오류)")
        age = round(t - self.t_out, 6)
        self.age_log.append((self.scene_id, t, age, refreshed))
        return self.out, dict(refreshed=refreshed, age_s=age, t_teacher=self.t_out, n_calls=self.n_calls)

    def age_histogram(self):
        h = {}
        for _s, _t, a, _r in self.age_log:
            k = round(a, 2)
            h[k] = h.get(k, 0) + 1
        return dict(sorted(h.items()))
