# 수정(2026-10-15): 경로 이동 models/plan/instance_bank.py → models/decoder/temporal_bank/plan_bank.py 및 정리.
#   제거(어떤 config 도 켜지 않음): with_instance_feat / with_all_front_views / with_custom_status_embed /
#   num_temp_instances / feat_grad 인자와 그 분기, 디코더가 호출하지 않는 update()(DN 분리 포함), get() 의 dn_metas 인자.
#   계획 쿼리 특징은 항상 전방 카메라 최상위 FPN 특징을 plan_feature_encoder 로 줄여 만든다(동작 동일).
import torch
from torch import nn
import numpy as np

from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from ....ops import feature_maps_format
from .topk import topk
from ..query_anchors.plan_anchor import load_plan_anchors

__all__ = ["PlanningInstanceBank"]


@PLUGIN_LAYERS.register_module()
class PlanningInstanceBank(nn.Module):
    def __init__(
            self,
            embed_dims,
            anchor_paths,
            anchor_types=None,
            anchor_scales=None,
            num_temp_mode=0,
            confidence_decay=0.6,
            feature_map_scale=None,
            max_time_interval=2,
            anchor_grad=True,
            ego_fut_ts=6,
            ego_fut_cmd=3,
            ego_fut_mode=6,
            front_view_idx=0,  # 수정(2026-11-06): NAVSIM 전방 카메라 색인 — 계획 쿼리 초기 특징을 뽑을 카메라 축 색인(nuScenes·B2D 0 = CAM_FRONT, NAVSIM 1 = CAM_F0)
    ):
        super(PlanningInstanceBank, self).__init__()
        self.front_view_idx = int(front_view_idx)  # 수정(2026-11-06): NAVSIM 전방 카메라 색인
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_cmd = ego_fut_cmd
        self.ego_fut_mode = ego_fut_mode
        self.num_temp_mode = num_temp_mode

        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval

        self.anchor_paths = anchor_paths
        self.anchor_types = anchor_types
        self.anchor_group = len(anchor_types)
        self.ego_mode_group = len(anchor_types)

        kernel_size = tuple([int(x / 2) for x in feature_map_scale])

        # 수정(2026-10-15, 2차): 앵커 파일 적재는 decoder/query_anchors/plan_anchor.py (cmd anchors 3×16)
        anchor = load_plan_anchors(anchor_paths, anchor_types, anchor_scales)

        self.anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float32), requires_grad=anchor_grad)  # [N, 12]
        self.num_anchor = len(self.anchor)

        self.plan_feature_encoder = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.Conv2d(embed_dims, embed_dims, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size),
        )

        self.reset()

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.confidence = None
        self.metas = None
        self.mask = None

    def get(self, batch_size, metas, feature_maps):
        instance_feature, anchor = self.prepare_planning(batch_size, feature_maps, metas)

        if self.cached_anchor is not None:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            bs, _, _ = anchor.shape
            temp_ego_feature = self.cached_feature.reshape(bs, -1, self.embed_dims)
            temp_ego_anchor = self.cached_anchor.reshape(bs, -1, self.ego_fut_ts * 2)
        else:
            temp_ego_feature, temp_ego_anchor = None, None

        return instance_feature, anchor, temp_ego_feature, temp_ego_anchor

    def prepare_planning(self, batch_size, feature_maps, metas=None):
        feature_maps_inv = feature_maps_format(feature_maps, inverse=True)
        # only center front-view feature
        feature_map = feature_maps_inv[0][-1][:, self.front_view_idx]  # 수정(2026-11-06): NAVSIM 전방 카메라 색인 #이전 feature_maps_inv[0][-1][:, 0] (NAVSIM 에선 CAM_L0)
        instance_feature = self.plan_feature_encoder(feature_map)
        instance_feature = instance_feature.unsqueeze(1).squeeze(-1).squeeze(-1)
        instance_feature = torch.tile(instance_feature, (1, self.num_anchor, 1))

        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        return instance_feature, anchor

    def cache(
            self,
            instance_feature,
            anchor,
            confidence,
            metas=None,
            feature_maps=None,
    ):
        if self.num_temp_mode <= 0:
            return

        bs, nj, _ = anchor.shape
        num_cmd = self.ego_fut_cmd * self.anchor_group
        num_mode = self.ego_fut_mode
        num_temp_mode = self.num_temp_mode

        # [bs * num_cmd, num_mode]
        _instance_feature = instance_feature.detach().reshape(bs * num_cmd, -1, self.embed_dims)
        _anchor = anchor.detach().reshape(bs * num_cmd, -1, self.ego_fut_ts * 2)
        _confidence = confidence.detach().reshape(bs * num_cmd, -1, 1)

        self.metas = metas
        _confidence = _confidence.squeeze(-1).sigmoid()
        if self.confidence is not None:
            _confidence[:, : num_temp_mode] = torch.maximum(
                self.confidence.reshape(bs * num_cmd, -1) * self.confidence_decay, _confidence[:, : num_temp_mode])

        (confidence, (cached_feature, cached_anchor)) = topk(
            _confidence, num_temp_mode, _instance_feature, _anchor)

        # [bs, num_cmd, num_mode]
        self.confidence = confidence.view(bs, num_cmd, num_temp_mode)
        self.cached_feature = cached_feature.view(bs, num_cmd, num_temp_mode, self.embed_dims)
        self.cached_anchor = cached_anchor.view(bs, num_cmd, num_temp_mode, self.ego_fut_ts * 2)