# 수정(2026-10-15, 2차): multi-scale features 에 대한 보조 깊이 감독(DenseDepthNet, loss_dense_depth).
from .depth import DenseDepthNet

__all__ = ["DenseDepthNet"]
