# 수정(2026-10-15): 경로 이동 models/det/blocks.py(SparseBox3DEncoder) → models/decoder/query_anchors/box3d_encoder.py (소스 정리, 동작 동일).
#   검출 앵커(11차원 3D 박스) 위치 임베딩. 자차(ego) 쿼리도 같은 인코더를 공유한다(SparseOneDecoder 참조).
import torch
import torch.nn as nn

from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import POSITIONAL_ENCODING

from ...common.box3d import *
from ...common.layers import linear_relu_ln

__all__ = ["SparseBox3DEncoder"]


@POSITIONAL_ENCODING.register_module()
class SparseBox3DEncoder(BaseModule):
    def __init__(
        self,
        embed_dims,
        vel_dims=3,
        mode="add",
        output_fc=True,
        in_loops=1,
        out_loops=2,
    ):
        super().__init__()
        assert mode in ["add", "cat"]
        self.embed_dims = embed_dims
        self.vel_dims = vel_dims
        self.mode = mode

        def embedding_layer(input_dims, output_dims):
            return nn.Sequential(
                *linear_relu_ln(output_dims, in_loops, out_loops, input_dims)
            )

        if not isinstance(embed_dims, (list, tuple)):
            embed_dims = [embed_dims] * 5
        self.pos_fc = embedding_layer(3, embed_dims[0])
        self.size_fc = embedding_layer(3, embed_dims[1])
        self.yaw_fc = embedding_layer(2, embed_dims[2])
        if vel_dims > 0:
            self.vel_fc = embedding_layer(self.vel_dims, embed_dims[3])
        if output_fc:
            self.output_fc = embedding_layer(embed_dims[-1], embed_dims[-1])
        else:
            self.output_fc = None

    def forward(self, box_3d: torch.Tensor):
        # 수정(2026-10-10): 리스트 인덱스→슬라이스(같은 값, 매 호출 H2D 동기화 제거)
        pos_feat = self.pos_fc(box_3d[..., X:Z + 1])
        size_feat = self.size_fc(box_3d[..., W:H + 1])
        yaw_feat = self.yaw_fc(box_3d[..., SIN_YAW:COS_YAW + 1])
        if self.mode == "add":
            output = pos_feat + size_feat + yaw_feat
        elif self.mode == "cat":
            output = torch.cat([pos_feat, size_feat, yaw_feat], dim=-1)

        if self.vel_dims > 0:
            vel_feat = self.vel_fc(box_3d[..., VX : VX + self.vel_dims])
            if self.mode == "add":
                output = output + vel_feat
            elif self.mode == "cat":
                output = torch.cat([output, vel_feat], dim=-1)
        if self.output_fc is not None:
            output = self.output_fc(output)
        return output
