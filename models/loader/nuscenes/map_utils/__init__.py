# 수정(2026-10-15): 경로 이동 mmdet3d_plugin/datasets/map_utils → models/loader/nuscenes/map_utils.
#수정: Path B — nuScenes map GT 추출(map_utils) 패키지 (SparseDrive 포팅).
from .nuscmap_extractor import NuscMapExtractor

__all__ = ["NuscMapExtractor"]
