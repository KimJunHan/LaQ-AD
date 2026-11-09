# 수정(2026-10-15): 경로 이동 models/map/blocks.py → models/decoder/split_refine/map_refine.py (소스 정리, 동작 동일).
#   SparsePoint3DEncoder 는 models/decoder/query_anchors/point3d_encoder.py, SparsePoint3DKeyPointsGenerator 는
#   models/decoder/deformable_aggregation/aggregation.py 로 옮겼고, 미사용 KeyPoint3DEncoder 는 제거했다.
import torch
import torch.nn as nn

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

from ...common.layers import linear_relu_ln

__all__ = ["SparsePoint3DRefinementModule"]


@PLUGIN_LAYERS.register_module()
class SparsePoint3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        coords_dim: int = 2,
        num_cls: int = 3,
        with_cls_branch: bool = True,
        with_line_key_points = False,
    ):
        super(SparsePoint3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.with_line_key_points = with_line_key_points
        if self.with_line_key_points:
            self.output_dim = coords_dim
        else:
            self.output_dim = num_sample * coords_dim
        self.num_cls = num_cls

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )

        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, self.num_cls),
            )

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        output = self.layers(instance_feature + anchor_embed)
        output = output + anchor
        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature)  ## NOTE anchor embed?
        else:
            cls = None
        qt = None
        return output, cls, qt
