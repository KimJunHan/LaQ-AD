# 수정(2026-10-15): 경로 이동 models/det/blocks.py → models/decoder/split_refine/det_refine.py (소스 정리, 동작 동일).
#   SparseBox3DEncoder 는 models/decoder/query_anchors/box3d_encoder.py, SparseBox3DKeyPointsGenerator 는 models/decoder/deformable_aggregation/aggregation.py 로 옮겼다.
import torch
import torch.nn as nn

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

from ...common.box3d import *
from ...common.layers import linear_relu_ln

__all__ = ["SparseBox3DRefinementModule"]


@PLUGIN_LAYERS.register_module()
class SparseBox3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        output_dim=11,
        num_cls=10,
        normalize_yaw=False,
        refine_yaw=False,
        with_cls_branch=True,
        with_quality_estimation=False,
    ):
        super(SparseBox3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.normalize_yaw = normalize_yaw
        self.refine_yaw = refine_yaw

        self.refine_state = [X, Y, Z, W, L, H]
        if self.refine_yaw:
            self.refine_state += [SIN_YAW, COS_YAW]

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
        self.with_quality_estimation = with_quality_estimation
        if with_quality_estimation:
            self.quality_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, 2),
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
        feature = instance_feature + anchor_embed
        output = self.layers(feature)
        # 수정(2026-10-10): refine_state 는 0 부터 연속([X..H] 또는 [X..COS_YAW]) → 같은 위치의 슬라이스(동기화 제거)
        rs = slice(self.refine_state[0], self.refine_state[-1] + 1)
        assert list(range(rs.start, rs.stop)) == list(self.refine_state)
        output[..., rs] = (
            output[..., rs] + anchor[..., rs]
        )
        if self.normalize_yaw:
            # 수정(2026-10-10): 슬라이스(동기화 제거). 리스트 인덱스는 복사본을 만들었으므로 같은 의미로 clone
            #   (뷰를 그대로 넘기면 제자리 대입이 normalize 가 저장한 입력을 바꿔 autograd 오류).
            output[..., SIN_YAW:COS_YAW + 1] = torch.nn.functional.normalize(
                output[..., SIN_YAW:COS_YAW + 1].clone(), dim=-1
            )
        if self.output_dim > 8:
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            translation = torch.transpose(output[..., VX:], 0, -1)
            velocity = torch.transpose(translation / time_interval, 0, -1)
            output[..., VX:] = velocity + anchor[..., VX:]

        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature)
        else:
            cls = None
        if return_cls and self.with_quality_estimation:
            quality = self.quality_layers(feature)
        else:
            quality = None
        return output, cls, quality
