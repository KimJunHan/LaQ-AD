# 수정(2026-10-19): 사용자 결정 — 지도 정렬 선택적 Gaussian 잡음(지도 밖 잡음 제거)
#   configs/nuscenes/v2_teacher/stage2/train.py(와 이를 상속하는 token_lidar·token_radar 2단계)의 custom_imports 가 부르는 등록 모듈.
#   새 모듈(MapAlignedPlanTruncatedDiffusion, PlanMapMemoryInstanceBank)은 기존 __init__.py 를 고치지 않으려고 패키지 자동 import
#   밖에 두었다. 같은 플러그인 소스가 두 이름("projects.mmdet3d_plugin" / "mmdet3d_plugin")으로 import 될 수 있으므로
#   tools/sensor_tokens_register.py 와 같은 규칙으로 이미 올라온 쪽 이름을 따른다(두 이름 모두 등록하면 레지스트리 KeyError).
import importlib
import sys

_base = "mmdet3d_plugin" if ("mmdet3d_plugin" in sys.modules and "projects.mmdet3d_plugin" not in sys.modules) \
    else "projects.mmdet3d_plugin"
importlib.import_module(_base + ".models.decoder.temporal_bank.plan_map_memory_bank")
importlib.import_module(_base + ".models.decoder.noise_t.plan_diffusion_map")
