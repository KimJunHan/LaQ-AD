# 수정(2026-10-15, 2차): ResNet-50 + FPN (multi-scale features). 백본(mmdet ResNet)과 FPN(mmdet FPN)은 config 로 빌드하고,
#   SparseDetector.extract_feat 가 6캠 영상을 통과시켜 4단계 특징을 만든다. 여기에는 학습용 GridMask 증강만 있다.
from .grid_mask import GridMask

__all__ = ["GridMask"]
