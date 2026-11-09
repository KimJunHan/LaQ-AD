# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
#   configs/navsim/v2_teacher/stage{1,2}/*.py 의 custom_imports 가 부르는 등록 모듈.
#   NavsimTeacherDataset·LoadPointsFromPcdNavsim(models/loader/navsim/dataset_v2_teacher.py)은 패키지 __init__ 을 고치지 않으려고
#   자동 import 밖에 두었다(nuScenes 학습의 import 경로 무변경). tools/plan_diffusion_map_register.py 와 같은 규칙으로
#   이미 올라온 플러그인 이름을 따른다(두 이름으로 모두 등록하면 레지스트리 KeyError).
import importlib
import sys

_base = "mmdet3d_plugin" if ("mmdet3d_plugin" in sys.modules and "projects.mmdet3d_plugin" not in sys.modules) \
    else "projects.mmdet3d_plugin"
importlib.import_module(_base + ".models.loader.navsim.dataset_v2_teacher")
