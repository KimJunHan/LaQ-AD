# 수정(2026-10-15, 2차): 넥 — FPN(mmdet, config) 뒤 multi-scale 특징에 붙는 보조 깊이 감독(multi_scale_depth/).
from .multi_scale_depth import DenseDepthNet

__all__ = ["DenseDepthNet"]
