# 수정(2026-10-15): 경로 이동 models/ego/blocks.py → models/decoder/split_refine/ego_refine.py. 미사용 SparseEgoRefinementModule
#   (with_supervise_ego_status=False 경로 전용) 제거. 모든 config 가 자차 상태 감독(EgoStatusRefinementModule)을 쓴다.
import torch.nn as nn

from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

__all__ = ["EgoStatusRefinementModule"]


@PLUGIN_LAYERS.register_module()
class EgoStatusRefinementModule(BaseModule):
    def __init__(self, embed_dims=256, status_dims=6):
        super(EgoStatusRefinementModule, self).__init__()
        self.embed_dims = embed_dims

        self.plan_status_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, status_dims),
        )

    def forward(self, ego_feature, ego_anchor_embed):
        planning_status = self.plan_status_branch(ego_feature + ego_anchor_embed)

        return planning_status
