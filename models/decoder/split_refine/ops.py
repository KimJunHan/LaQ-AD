# 수정(2026-10-15, 2차): 아키텍처 그림 기준 재배치 models/decoder/deformable_aggregation/aggregation.py(CustomOperation) → models/decoder/split_refine/ops.py
#   operation_order 의 concat / split / deformable / refine 자리에 놓이는 항등 연산(실제 처리는 SparseOneDecoder.forward 가 한다).
import torch.nn as nn
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

__all__ = ["CustomOperation"]


@PLUGIN_LAYERS.register_module()
class CustomOperation(BaseModule):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, feature):
        return self.identity(feature)
