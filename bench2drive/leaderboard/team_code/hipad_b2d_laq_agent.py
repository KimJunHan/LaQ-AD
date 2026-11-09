# 수정(2026-10-09): LaQ-AD Bench2Drive 폐쇄루프 에이전트 — hipad_b2d_agent.SparseAgent 에 온라인 Qwen 주입만 더한다.
#
#   왜 별도 파일인가: 학습은 오프라인 캐시(qwen_cache_dir)를 token 으로 읽지만, CARLA 폐쇄루프에서는 매 틱 영상이
#   새로 생성되어 캐시가 존재할 수 없다. 따라서
#     (1) 모델의 qwen_cache_dir 을 끄고(그대로 두면 token 이 없어 SparseDetector._maybe_load_qwen_cache 가 즉시 실패),
#     (2) 매 틱 같은 6방향 영상을 models/encoder/vlm_qwen2vl/qwen_online.QwenOnlineEncoder(캐시 생성과 동일 지시문·화소예산·18층)로
#         인코딩해 model(..., qwen_hidden=, qwen_mask=) 로 넘긴다. simple_test 는 **data 를 head 의 metas 로
#         전달하므로 통합 디코더(models/decoder/unified_decoder.py)가 metas['qwen_hidden'] 을 읽는다(없으면 디코더가 즉시 실패 — 조용한 무융합 없음).
#   기반 에이전트의 제어·PID·좌표 변환·영상 전처리(JPEG q20 재압축 포함)는 한 줄도 바꾸지 않는다.
#
#   [폐쇄루프 시간 정렬 — 논문 부록 B "Closed-loop Inference"]
#     논문: 2Hz 로 학습, 시뮬레이션은 더 높은 주파수 → 여러 memory 를 번갈아 써서 각 memory 가 2Hz 간격 프레임만 보게 함.
#     공개 코드의 구현(확인 2026-10-09): 기반 에이전트가 cfg.model.head.onedecoder_head.with_close_loop=True 로 빌드하면
#     SparseOneDecoder.init_instance_bank_list 가 det/map/ego/plan 인스턴스 뱅크를 bank_length = close_loop_hz //
#     open_loop_hz = 20 // 2 = 10 벌 deepcopy 하고(models/decoder/unified_decoder.py init_instance_bank_list), 매 forward 에서
#     bank_idx = run_step % bank_length 로 고른 뒤 같은 뱅크에 cache 하고 run_step += 1 (SparseOneDecoder.forward 시작·끝).
#     leaderboard 는 20Hz(leaderboard_evaluator.frame_rate=20, fixed_delta 0.05s)이므로 각 뱅크는 0.5s 간격 프레임만
#     보며, timestamp=step/20 이라 뱅크 내부 dt=0.5s 가 학습(10Hz 데이터를 5 그룹으로 나눈 2Hz 시퀀스)과 일치한다.
#     논문의 "10Hz·memory 5개"와 간격이 같은 구성이다. 아래 setup 에서 이 조건이 깨지면 즉시 실패시킨다.
#   [Qwen 토큰과 뱅크의 일치] 매 틱 tick() 에서 그 틱의 영상으로 인코딩하고, 모델 호출 직후 비운다 — 다른 틱(=다른 뱅크)의
#     토큰이 재사용될 수 없다.
#
#   사용: bench2drive/leaderboard/scripts/run_evaluation_b2d_laq.sh. TEAM_CONFIG 는 "<config>+<ckpt>+<save_name>".
#   Qwen 경로는 환경변수 LAQ_QWEN_PATH (기본 ckpts/qwen2-vl-2b). VRAM(주행 모델 + Qwen2-VL-2B bf16)은 LAQ_VRAM_LOG_EVERY
#   틱마다 로그로 남긴다(사용자 상한 24GB 실측용).
import os
import pathlib
import importlib.util as _ilu

import numpy as np
import torch
from PIL import Image

try:
    from hipad_b2d_agent import SparseAgent, CAMERA
except ImportError as e:  # carla 모듈 부재 등 — 조용히 넘기지 않고 원인을 드러낸다
    raise ImportError(
        f"hipad_b2d_agent 적재 실패({e}). CARLA 0.9.15 의 python 'carla' 모듈(py3.10 wheel)과 "
        "bench2drive/leaderboard, scenario_runner 가 PYTHONPATH 에 있어야 한다(run_evaluation.sh 참조).") from e

_QO_PATH = pathlib.Path(__file__).resolve().parents[3] / "models" / "encoder" / "vlm_qwen2vl" / "qwen_online.py"  # 수정(2026-10-15): 경로 이동
_spec = _ilu.spec_from_file_location("qwen_online", str(_QO_PATH))
_qo = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_qo)

SIM_HZ = 20      # leaderboard_evaluator.frame_rate (fixed_delta_seconds = 1/20)
TRAIN_HZ = 2     # 학습 시퀀스 주파수(10Hz 데이터의 5 그룹 분할)


def get_entry_point():
    return 'LaQSparseAgent'


class _QwenInjectingModel:
    """self.model 호출에 현재 틱의 qwen_hidden/qwen_mask 를 덧붙이는 얇은 래퍼(그 외 속성은 위임)."""

    def __init__(self, model, agent):
        self._model = model
        self._agent = agent
        self._calls = 0
        self._log_every = int(os.environ.get("LAQ_VRAM_LOG_EVERY", "200"))

    def __call__(self, *args, **kwargs):
        a = self._agent
        if a._qwen_hidden is None:
            raise RuntimeError("이번 틱의 Qwen 토큰이 없다 — tick() 에서 인코딩이 수행되지 않았다.")
        kwargs["qwen_hidden"] = a._qwen_hidden
        kwargs["qwen_mask"] = a._qwen_mask
        out = self._model(*args, **kwargs)
        a._qwen_hidden = a._qwen_mask = None   # 다음 틱(다른 뱅크)에 재사용 금지
        if self._log_every > 0 and self._calls % self._log_every == 0 and torch.cuda.is_available():
            print(f"[LaQ] tick {self._calls} peak VRAM alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GB "
                  f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB", flush=True)
        self._calls += 1
        return out

    def __getattr__(self, name):
        return getattr(self._model, name)


class LaQSparseAgent(SparseAgent):

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        # (0) 폐쇄루프 시간 정렬 조건 확인(위 머리말). 하나라도 어긋나면 뱅크 간격이 학습과 달라진다.
        dec = self.model.head.onedecoder_head
        if not dec.with_close_loop:
            raise RuntimeError("with_close_loop=False — 폐쇄루프 다중 뱅크가 꺼져 있다(기반 에이전트 설정 확인).")
        if (dec.close_loop_hz, dec.open_loop_hz, self.frame_rate) != (SIM_HZ, TRAIN_HZ, SIM_HZ):
            raise RuntimeError(
                f"시간 정렬 불일치: close_loop_hz={dec.close_loop_hz}, open_loop_hz={dec.open_loop_hz}, "
                f"agent frame_rate={self.frame_rate} (기대 {SIM_HZ}/{TRAIN_HZ}/{SIM_HZ}).")
        if dec.close_loop_bank_length not in (None, SIM_HZ // TRAIN_HZ):
            raise RuntimeError(f"close_loop_bank_length={dec.close_loop_bank_length} != {SIM_HZ // TRAIN_HZ}")
        # (1) 오프라인 캐시 경로 끄기 — 폐쇄루프에는 token 이 없다.
        self.model.qwen_cache_dir = None
        # 수정(2026-10-15): 보조 캐시 혼합(qwen_cache_dir_aux)은 SparseDetector 에서 제거되어(켜면 생성 시 즉시 실패)
        #   여기서 따로 확인할 필요가 없다.
        qwen_path = os.environ.get("LAQ_QWEN_PATH", "/home1/irteam/data-vol1/src/HiP-AD/ckpts/qwen2-vl-2b")
        if not os.path.isdir(qwen_path):
            raise FileNotFoundError(f"Qwen2-VL-2B 경로 없음: {qwen_path} (LAQ_QWEN_PATH)")
        self.qwen = _qo.QwenOnlineEncoder(qwen_path, device="cuda")
        self._qwen_hidden = None
        self._qwen_mask = None
        self.model = _QwenInjectingModel(self.model, self)

    def tick(self, input_data):
        tick_data = super().tick(input_data)
        # (2) 모델이 보는 것과 같은 틱 영상(RGB, JPEG q20 재압축 후)을 지시문 순서로 인코딩.
        #     B2D 학습 캐시도 같은 해상도(1600x900) JPEG 원본을 RGB 로 읽어 만든다(tools/qwen_cache_b2d.py).
        by_name = dict(zip(CAMERA, (tick_data['imgs'][c] for c in CAMERA)))
        imgs = [Image.fromarray(np.ascontiguousarray(by_name[c])) for c in _qo.B2D_CAM_ORDER]
        self._qwen_hidden, self._qwen_mask = self.qwen.encode(imgs)
        return tick_data
