# 수정(2026-10-15): 경로 이동 models/map/blocks.py(SparsePoint3DEncoder) → models/decoder/query_anchors/point3d_encoder.py (소스 정리, 동작 동일).
#   점열 앵커(지도 polyline 20점, 계획 궤적 6점) 위치 임베딩. 미사용 KeyPoint3DEncoder 는 제거.
import torch
import torch.nn as nn

from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import POSITIONAL_ENCODING

from ...common.layers import linear_relu_ln

__all__ = ["SparsePoint3DEncoder"]


@POSITIONAL_ENCODING.register_module()
class SparsePoint3DEncoder(BaseModule):
    def __init__(
        self, 
        embed_dims: int = 256,
        num_sample: int = 20,
        coords_dim: int = 2,
        return_points_embed: bool = False,
    ):
        super(SparsePoint3DEncoder, self).__init__()
        self.embed_dims = embed_dims
        self.input_dims = num_sample * coords_dim
        self.return_points_embed = return_points_embed
        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.pos_fc = embedding_layer(self.input_dims)

    def forward(self, anchor: torch.Tensor):
        pos_feat = self.pos_fc(anchor)
        if self.return_points_embed:
            return pos_feat, None
        else:
            return pos_feat
