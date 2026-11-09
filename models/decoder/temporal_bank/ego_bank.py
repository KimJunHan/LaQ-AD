# 수정(2026-10-15): 경로 이동 models/ego/instance_bank.py → models/decoder/temporal_bank/ego_bank.py 및 정리.
#   제거(어떤 config 도 켜지 않음): anchor_handler(시간 투영)·with_instance_feat·plan_anchor(with_supervise_ego_status=False
#   전용)·num_anchor·anchor_grad·feat_grad 인자와 그 분기, 디코더가 호출하지 않는 update()(존재하지 않는 속성을
#   참조하던 죽은 코드), get() 의 dn_metas 인자. 자차 쿼리 특징은 전방 카메라 최상위 FPN 특징으로 만든다(동작 동일).
import torch
import numpy as np

from torch import nn
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from ....ops import feature_maps_format
from ..query_anchors.ego_anchor import ego_anchor


__all__ = ["EgoInstanceBank"]


@PLUGIN_LAYERS.register_module()
class EgoInstanceBank(nn.Module):
    def __init__(
            self,
            embed_dims,
            anchor_type='nus',
            feature_map_scale=None,
            num_temp_instances=0,
            max_time_interval=2,
            custom_anchor=None,  #수정(2026-10-09): NAVSIM — anchor_type='custom' 일 때 자차 앵커 11차원 직접 지정
            front_view_idx=0,  # 수정(2026-11-06): NAVSIM 전방 카메라 색인 — 자차 쿼리 초기 특징을 뽑을 카메라 축 색인(nuScenes·B2D 0 = CAM_FRONT, NAVSIM 1 = CAM_F0)
    ):
        super(EgoInstanceBank, self).__init__()
        self.front_view_idx = int(front_view_idx)  # 수정(2026-11-06): NAVSIM 전방 카메라 색인
        self.embed_dims = embed_dims
        self.max_time_interval = max_time_interval
        self.num_temp_instances = num_temp_instances
        kernel_size = tuple([int(x / 2) for x in feature_map_scale])

        # 수정(2026-10-15, 2차): 자차 앵커 값은 decoder/query_anchors/ego_anchor.py
        self.anchor = nn.Parameter(torch.tensor(ego_anchor(anchor_type, custom_anchor), dtype=torch.float32), requires_grad=False)

        self.num_anchor = len(self.anchor)

        self.ego_feature_encoder = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dims),
            nn.Conv2d(self.embed_dims, self.embed_dims, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dims),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size),
        )

        self.reset()

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None

    def get(self, batch_size, metas, feature_maps):
        instance_feature, anchor = self.prepare_ego(batch_size, feature_maps)

        if self.cached_anchor is not None and batch_size == self.cached_anchor.shape[0]:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=anchor.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            temp_ego_feature = self.cached_feature
            temp_ego_anchor = self.cached_anchor
        else:
            temp_ego_feature, temp_ego_anchor = None, None

        return instance_feature, anchor, temp_ego_feature, temp_ego_anchor

    def prepare_ego(self, batch_size, feature_maps,):
        feature_maps_inv = feature_maps_format(feature_maps, inverse=True)
        feature_map = feature_maps_inv[0][-1][:, self.front_view_idx]  # 수정(2026-11-06): NAVSIM 전방 카메라 색인 #이전 feature_maps_inv[0][-1][:, 0]  # 'front-view feature' (NAVSIM 에선 CAM_L0)

        instance_feature = self.ego_feature_encoder(feature_map)
        instance_feature = instance_feature.unsqueeze(1).squeeze(-1).squeeze(-1)

        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        return instance_feature, anchor

    def cache(
            self,
            instance_feature,
            anchor,
            metas=None,
            feature_maps=None
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()

        self.metas = metas
        self.cached_feature = instance_feature.detach()
        self.cached_anchor = anchor.detach()