# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   LaQ v2(교사 캐시 7 폴더) Bench2Drive 폐쇄루프 에이전트 — 교사 출력 2 Hz 갱신·직전 출력 재사용(사용자 결정 2026-10-16).
#   기존 hipad_b2d_laq_agent.py(원본 Qwen 18층 토큰을 매 틱 인코딩, v1 계열)는 그대로 두고 새 파일로 분리한다.
#
#   차이점(hipad_b2d_laq_agent.LaQSparseAgent 대비):
#     (1) 매 틱 Qwen 을 돌리지 않는다. teacher_refresh.TeacherRefreshScheduler 가 시뮬레이션 시각(step/20 s) 기준
#         0.5 s 마다(20 Hz 에서 10 틱마다) 교사를 부르고, 사이 틱에는 직전 교사 출력(qwen_hidden/qwen_mask/qwen_type)을 그대로 쓴다.
#         → 학습 캐시 규칙(10 Hz 프레임 f 는 keyframe f − f%5 의 교사 출력 사용)과 같은 갱신 주기.
#     (2) 모델 호출 뒤 토큰을 비우지 않는다(재사용이 의도). 대신 매 틱 스케줄러가 돌려준 출력을 넣고, 나이(age_s)를 기록한다.
#     (3) qwen_type(토큰별 폴더 번호 0..6)도 넘긴다(LoadTeacherCacheNusc 와 같은 3 개 키, (1, L) 모양).
#     (4) 교사 입력: 현재 6 캠 + 0.5 s 전 6 캠(cams_prev; 교사 정답·SFT 와 같은 12 영상 입력) + 명령 + 자차 상태 줄.
#         0.5 s 전 영상이 아직 없으면(장면 첫 갱신) 현재 영상을 복제한다(교사 정답 생성 규칙 '장면 첫 샘플: 현재 프레임 복제'와 같음).
#   교사 본체(문법 제약 생성 + 7 폴더 은닉 상태 자르기)는 GPU 작업이라 아직 온라인 함수가 없다:
#     환경변수 LAQ_TEACHER_BUILDER="<모듈>:<함수>" 가 가리키는 함수(build(device) → callable(sim_t, cams, cams_prev, command, ego)
#     → dict(qwen_hidden=(1,L,1536) bf16, qwen_mask=(1,L) uint8, qwen_type=(1,L) int64)) 를 쓴다. 지정이 없으면 setup 에서 즉시 실패.
#     그 함수는 tools/vla_teacher/extract_teacher_cache_v2.py 와 같은 프롬프트(B안)·문법 제약(answer_grammar_v1)·자르기(laq_cache_layout)
#     를 써야 한다 — 학습 캐시와 다르면 학습/평가 불일치(남은 일, 보고서 참조).
#   CARLA 없이 시험할 수 있는 부분(갱신 규칙)은 teacher_refresh.py + tools/verify_b2d/test_teacher_refresh_cpu.py 에 있다.
# 수정(2026-11-06): 사용자 결정 "B2D 폐쇄루프 E2 + 나침반 NaN 틱 처리(권장안)" — 아래 '# 수정(2026-11-06)' 줄들.
#   (5) E2(자차 상태 4 값 [속도, 종가속, yaw rate, available]) 모델이면 metas["ego_input"] = (1, 4) float32 를 넘긴다.
#       값은 교사가 새로 불린 틱(info["refreshed"])에 정하고 사이 틱은 재사용(학습: 10 Hz 프레임 → 직전 2 Hz 키프레임 값, 나이 0~0.4 s).
#       단일 출처: 교사 호출에 ego=(speed, accel, yaw)·ego_line=common_b2d.ego_line(...) 를 넘기고, 교사 출력 dict 의
#       "ego_line"(프롬프트에 실제로 넣은 줄)을 build_b2d_ego_input.PAT 로 다시 읽어 학생 값으로 쓴다(ego_state_hold.py).
#       교사 빌더 계약 추가: 출력 dict 에 ego_line 을 넣어야 한다(E2 모델이면 필수, 있으면 넘겨준 줄과 같아야 한다 — 다르면 즉시 실패).
#   (6) 나침반 NaN 틱: 기반 tick 의 0 채움 대신 직전 정상 틱 IMU 값 유지(경로 시작에 정상값이 없을 때만 0). 교사에 넘기는 tick_data
#       사본의 acceleration/angular_velocity 도 같은 값으로 바꾼다(교사·학생 같은 규칙). 경로별 기록 → destroy 에서 JSON·로그.
import collections
import datetime  # 수정(2026-11-06): 자차 상태 기록 파일 이름(덮어쓰기 방지)
import importlib
import json  # 수정(2026-11-06): 경로별 자차 상태(NaN 틱) 기록 JSON
import os
import pathlib  # 수정(2026-11-06): 기록 파일 경로

import numpy as np
import torch
from PIL import Image

from teacher_refresh import TeacherRefreshScheduler, REFRESH_PERIOD
from ego_state_hold import EgoStateHold, ego_line_and_input, parse_ego_line  # 수정(2026-11-06): E2 자차 상태·NaN 틱 유지(순수 파이썬)

try:
    from hipad_b2d_agent import SparseAgent, CAMERA
except ImportError as e:  # carla 모듈 부재 등 — 조용히 넘기지 않는다
    raise ImportError(
        f"hipad_b2d_agent 적재 실패({e}). CARLA 0.9.15 python 'carla' 모듈과 bench2drive/leaderboard, scenario_runner 가 "
        "PYTHONPATH 에 있어야 한다(run_evaluation.sh 참조).") from e

SIM_HZ = 20
TRAIN_HZ = 2
# 교사 영상 순서(교사 정답·QA 와 같은 qwen_common.CAM_ORDER: front, front_left, front_right, back_left, back, back_right)
TEACHER_CAM_ORDER = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")


def get_entry_point():
    return 'LaQTeacherSparseAgent'


def _load_builder():
    spec = os.environ.get("LAQ_TEACHER_BUILDER", "")
    if ":" not in spec:
        raise RuntimeError("LAQ_TEACHER_BUILDER='<모듈>:<함수>' 가 없다 — 온라인 교사(문법 제약 생성 + 7 폴더 자르기) 함수를 지정해야 한다. "
                           "원본 Qwen 18층 토큰으로 대신하지 않는다(학습 캐시와 다른 입력 → 학습/평가 불일치).")
    mod, fn = spec.split(":", 1)
    return getattr(importlib.import_module(mod), fn)


class _TeacherInjectingModel:
    """self.model 호출에 이번 틱 스케줄러 출력(qwen_hidden/qwen_mask/qwen_type)을 덧붙이는 래퍼."""

    def __init__(self, model, agent):
        self._model = model
        self._agent = agent

    def __call__(self, *args, **kwargs):
        cur = self._agent._teacher_cur
        if cur is None:
            raise RuntimeError("이번 틱의 교사 출력이 없다 — tick() 에서 스케줄러가 호출되지 않았다.")
        for k in ("qwen_hidden", "qwen_mask", "qwen_type"):
            if k not in cur:
                raise KeyError(f"교사 출력에 {k} 가 없다")
            kwargs[k] = cur[k]
        if self._agent._e2_on:  # 수정(2026-11-06): E2 모델만 ego_input 추가(E2 꺼진 모델은 kwargs 종전과 동일)
            x = self._agent._ego_input_cur  # 수정(2026-11-06): 교사 갱신 틱에 정한 값(사이 틱 재사용)
            if x is None:  # 수정(2026-11-06): 조용히 넘기지 않는다
                raise RuntimeError("E2 모델인데 이번 틱 ego_input 이 없다 — tick() 에서 교사 갱신 값이 정해지지 않았다.")  # 수정(2026-11-06)
            if "ego_input" in kwargs:  # 수정(2026-11-06): 기반 에이전트가 넘기지 않는 키 — 겹치면 출처가 둘
                raise RuntimeError("ego_input 이 이미 kwargs 에 있다(출처 중복)")  # 수정(2026-11-06)
            kwargs["ego_input"] = torch.from_numpy(x.copy()).view(1, 4).to(self._agent.device)  # 수정(2026-11-06): (1,4) float32 → metas["ego_input"]
        self._agent._teacher_cur = None   # 다음 틱은 스케줄러를 다시 거쳐야 한다(재사용 여부는 스케줄러가 정한다)
        return self._model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


class LaQTeacherSparseAgent(SparseAgent):

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        dec = self.model.head.onedecoder_head
        if not dec.with_close_loop:
            raise RuntimeError("with_close_loop=False — 폐쇄루프 다중 뱅크가 꺼져 있다.")
        if (dec.close_loop_hz, dec.open_loop_hz, self.frame_rate) != (SIM_HZ, TRAIN_HZ, SIM_HZ):
            raise RuntimeError(f"시간 정렬 불일치: {dec.close_loop_hz}/{dec.open_loop_hz}/{self.frame_rate} (기대 20/2/20)")
        self.model.qwen_cache_dir = None   # 폐쇄루프에는 token 이 없다(캐시 파일 경로 끄기)
        teacher = _load_builder()(device="cuda")
        self.teacher_sched = TeacherRefreshScheduler(teacher, period=REFRESH_PERIOD, mode="fresh")
        self._teacher_cur = None
        self._cam_hist = collections.deque()      # (sim_t, [PIL]*6) — 0.5 s 전 영상 공급용
        self._route_id = os.environ.get("ROUTES_SUBSET", "") + "|" + str(id(self))
        self.teacher_age_log = []
        self._e2_on = getattr(dec, "ego_input_embed_cfg", None) is not None  # 수정(2026-11-06): E2(ego_input_embed) 켜짐 여부
        if self._e2_on and int(dec.ego_input_embed_cfg["in_dims"]) != 4:  # 수정(2026-11-06): 4 값 형식만 지원
            raise RuntimeError(f"ego_input_embed in_dims {dec.ego_input_embed_cfg['in_dims']} != 4")  # 수정(2026-11-06)
        self._ego_input_cur = None  # 수정(2026-11-06): 지금 쓰는 학생 E2 값(float32 (4,)) — 교사 갱신 틱에만 바뀐다
        self._ego_line_cur = None  # 수정(2026-11-06): 그 값의 출처 줄(교사 프롬프트 줄)
        self.ego_hold = EgoStateHold(self._route_id)  # 수정(2026-11-06): 나침반 NaN 틱 IMU 유지·기록
        self.ego_input_log = []  # 수정(2026-11-06): (step, 상태, 유지 나이) — 교사 갱신 틱만
        self.model = _TeacherInjectingModel(self.model, self)

    def _prev_cams(self, t, cams):
        """t − 0.5 s 의 영상(없으면 현재 복제). 1 초보다 오래된 기록은 버린다(메모리)."""
        self._cam_hist.append((t, cams))
        while self._cam_hist and self._cam_hist[0][0] < t - 1.0 - 1e-6:
            self._cam_hist.popleft()
        for tt, cc in self._cam_hist:
            if abs(tt - (t - REFRESH_PERIOD)) < 1e-6:
                return cc, True
        return cams, False

    def tick(self, input_data):
        tick_data = super().tick(input_data)          # self.step 이 여기서 1 증가
        t = self.step / self.frame_rate                # 기반 에이전트 run_step 의 inputs['timestamp'] 와 같은 값
        by_name = dict(zip(CAMERA, (tick_data['imgs'][c] for c in CAMERA)))
        cams = [Image.fromarray(np.ascontiguousarray(by_name[c])) for c in TEACHER_CAM_ORDER]
        cams_prev, prev_ok = self._prev_cams(t, cams)
        imu = input_data['IMU'][1]  # 수정(2026-11-06): 기반 tick 이 0 으로 바꾸기 전 원 IMU 값(compass = 마지막 값)
        ego = self.ego_hold.update(self.step, tick_data['speed'], imu[-1], imu[:3], imu[3:6])  # 수정(2026-11-06): NaN 틱 직전 정상값 유지
        will_refresh = self.teacher_sched.scene_id != self._route_id or self.teacher_sched._due(t)  # 수정(2026-11-06): 줄은 갱신 틱에만 만든다(사이 틱 값은 쓰이지 않음)
        ego_line, ego_in = ego_line_and_input(ego["vals"]) if will_refresh else (None, None)  # 수정(2026-11-06): 교사 프롬프트 줄·학생 값 단일 출처
        td_teacher = dict(tick_data)  # 수정(2026-11-06): 교사용 사본 — 기반 run_step 의 tick_data 는 그대로
        td_teacher['acceleration'] = np.asarray(ego["accel3"], np.float64)  # 수정(2026-11-06): 0 채움 대신 유지값
        td_teacher['angular_velocity'] = np.asarray(ego["gyro3"], np.float64)  # 수정(2026-11-06): 0 채움 대신 유지값
        td_teacher['ego_state'] = ego["state"]  # 수정(2026-11-06): ok|held|zero_fill
        out, info = self.teacher_sched.step(t, scene_id=self._route_id, cams=cams, cams_prev=cams_prev,
                                            prev_available=prev_ok, tick_data=td_teacher,  # 수정(2026-11-06): 사본 전달
                                            ego=ego["vals"], ego_line=ego_line)  # 수정(2026-11-06): 교사 프롬프트 자차 줄 입력
        if info["refreshed"] != will_refresh:  # 수정(2026-11-06): 갱신 예측과 스케줄러 판단이 어긋나면 즉시 실패
            raise RuntimeError(f"교사 갱신 예측 {will_refresh} != 스케줄러 {info['refreshed']} (t={t})")  # 수정(2026-11-06)
        if info["refreshed"]:  # 수정(2026-11-06): 교사 갱신 틱에만 E2 값을 정한다(학습 키프레임 재사용 규칙)
            used = out.get("ego_line") if isinstance(out, dict) else None  # 수정(2026-11-06): 교사가 프롬프트에 실제로 쓴 줄
            if used is not None and used != ego_line:  # 수정(2026-11-06): 교사·학생 숫자 불일치는 즉시 실패
                raise RuntimeError(f"교사 프롬프트 자차 줄 {used!r} != 넘긴 줄 {ego_line!r}")  # 수정(2026-11-06)
            if used is None and self._e2_on:  # 수정(2026-11-06): E2 모델은 교사 빌더가 쓴 줄을 돌려줘야 한다(단일 출처 확인)
                raise RuntimeError("E2 모델인데 교사 출력에 ego_line 이 없다 — LAQ_TEACHER_BUILDER 가 프롬프트 자차 줄을 돌려줘야 한다.")  # 수정(2026-11-06)
            self._ego_line_cur = ego_line if used is None else used  # 수정(2026-11-06)
            self._ego_input_cur = ego_in if used is None else parse_ego_line(used)  # 수정(2026-11-06): 같은 줄 → 같은 비트
            self.ego_hold.note_refresh(ego)  # 수정(2026-11-06): 갱신 틱 상태 기록
            self.ego_input_log.append((self.step, ego["state"], ego["hold_age"]))  # 수정(2026-11-06)
            if ego["state"] != "ok":  # 수정(2026-11-06): 조용히 넘기지 않기 — 갱신 틱의 유지/0 채움은 즉시 로그
                print(f"[LaQ-teacher] step {self.step}: 나침반 NaN 갱신 틱 — 상태 {ego['state']}, 유지 나이 {ego['hold_age']} 틱, "
                      f"줄 {ego_line}", flush=True)  # 수정(2026-11-06)
            if ego["clipped"]:  # 수정(2026-11-09): 사용자 결정 — 갱신 틱에서 형식 범위 밖 값을 자르면 즉시 로그
                print(f"[LaQ-teacher] step {self.step}: 자차 상태 형식 범위 밖 → 자름 {ego['clipped']} (원값 {ego['vals_raw']}), "
                      f"줄 {ego_line}", flush=True)  # 수정(2026-11-09)
        self._teacher_cur = out
        self.teacher_age_log.append((self.step, t, info["age_s"], info["refreshed"]))
        return tick_data

    def destroy(self):
        try:  # 수정(2026-11-06): 기록 쓰기가 실패해도 기반 destroy(모델 해제)는 하고, 예외는 그대로 올린다
            self._destroy_logs()  # 수정(2026-11-06)
        finally:  # 수정(2026-11-06)
            super().destroy()  # 수정(2026-11-06)

    def _destroy_logs(self):  # 수정(2026-11-06): 종전 destroy 의 기록 부분 + 자차 상태 기록
        if hasattr(self, "teacher_sched"):
            print(f"[LaQ-teacher] 교사 호출 {self.teacher_sched.n_calls} 회, 나이 분포 {self.teacher_sched.age_histogram()}", flush=True)
        if hasattr(self, "ego_hold"):  # 수정(2026-11-06): 경로별 자차 상태(NaN 틱) 기록 — 로그 + JSON
            summ = self.ego_hold.summary()  # 수정(2026-11-06)
            summ.update(routes=os.environ.get("ROUTES", ""), e2_on=self._e2_on)  # 수정(2026-11-06)
            print(f"[LaQ-teacher] 자차 상태 기록 {json.dumps(summ, ensure_ascii=False)}", flush=True)  # 수정(2026-11-06)
            out_dir = pathlib.Path(getattr(self, "save_path", None) or os.environ.get("SAVE_PATH", "."))  # 수정(2026-11-06)
            stem = pathlib.Path(os.environ.get("ROUTES", "route")).stem  # 수정(2026-11-06)
            p = out_dir / f"ego_state_stats_{stem}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"  # 수정(2026-11-06): 덮어쓰기 없음
            with open(p, "x") as fh:  # 수정(2026-11-06)
                json.dump(summ, fh, indent=1, ensure_ascii=False)  # 수정(2026-11-06)
            print(f"[LaQ-teacher] 자차 상태 기록 파일 {p}", flush=True)  # 수정(2026-11-06)
