# 수정(2026-10-19): 사용자 결정 — 언어 토큰 자리에 LiDAR/레이더 토큰만 바꿔 넣는 비교(인코더 함께 학습)
#   configs/nuscenes/token_{lidar,radar}/ 의 custom_imports 가 부르는 등록 모듈.
#   새 모듈(SensorTokenSparseDetector, SensorBEVTokenizer)은 기존 __init__.py 를 고치지 않으려고 패키지 자동 import 밖에 두었다.
#   같은 플러그인 소스가 두 이름으로 import 될 수 있다:
#     tools/train.py·tools/test.py           → "projects.mmdet3d_plugin" (config 를 읽을 때 이 모듈이 먼저 불린다)
#     tools/eval_nusc_planning.py            → "mmdet3d_plugin" 을 먼저 import 한 뒤 Config.fromfile
#   두 이름으로 모두 올리면 mmcv 레지스트리에 같은 이름이 두 번 등록돼 실패한다(KeyError). 그래서 이미 올라온 쪽 이름을 따라
#   새 모듈을 import 한다(아무것도 없으면 projects.mmdet3d_plugin — train/test 의 plugin_dir 와 같은 이름).
import importlib
import sys

_base = "mmdet3d_plugin" if ("mmdet3d_plugin" in sys.modules and "projects.mmdet3d_plugin" not in sys.modules) \
    else "projects.mmdet3d_plugin"
importlib.import_module(_base + ".models.encoder.sensor_tokens")
importlib.import_module(_base + ".models.sensor_token_detector")
