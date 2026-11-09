# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   configs/b2d/v2_teacher/*/train.py 의 custom_imports 가 부르는 등록 모듈. LoadTeacherCacheB2D 를 기존 __init__.py 를
#   고치지 않고 레지스트리에 올린다. tools/plan_diffusion_map_register.py 와 같은 규칙: 같은 플러그인 소스가 두 이름
#   ("projects.mmdet3d_plugin" / "mmdet3d_plugin")으로 import 될 수 있으므로 이미 올라온 쪽 이름을 따른다(둘 다 등록하면 KeyError).
import importlib
import sys

_base = "mmdet3d_plugin" if ("mmdet3d_plugin" in sys.modules and "projects.mmdet3d_plugin" not in sys.modules) \
    else "projects.mmdet3d_plugin"
importlib.import_module(_base + ".models.encoder.vl_tokens_cache.loading_b2d")
