# 수정(2026-10-15, 2차): 아키텍처 그림 기준 재배치 models/decoder/deformable_aggregation/aggregation.py → models/decoder/deformable_aggregation/aggregation.py
#   CustomOperation(concat/split 자리 표시 연산)은 decoder/split_refine/ops.py 로 옮겼다.
# 수정(2026-10-15): 경로 이동·정리(소스 정리, 동작 동일).
#   models/blocks.py            → DeformableFeatureAggregation, AsymmetricFFN, CustomOperation
#   models/det/blocks.py        → SparseBox3DKeyPointsGenerator
#   models/map/blocks.py        → SparsePoint3DKeyPointsGenerator
#   DenseDepthNet 은 models/neck/multi_scale_depth/depth.py, linear_relu_ln 은 models/common/layers.py 로 옮겼다.
#   제거한 미사용 분기(어떤 config 도 켜지 않음): DeformableFeatureAggregation 의 use_points_embed /
#   use_anchor_embed / temporal_fusion_module / use_temporal_anchor_embed / num_sample, 키포인트 생성기의
#   with_points_embed / with_anchor_embed / 시간축(T_cur2temp_list·timestamp) 분기.
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    FEEDFORWARD_NETWORK,
)

from ...common.box3d import *
from ...common.layers import linear_relu_ln
from ...common.sync_free import const_like  # 수정(2026-10-10): 동기화 제거

try:
    from ....ops import deformable_aggregation_function as DAF
except:
    DAF = None

__all__ = [
    "DeformableFeatureAggregation",
    "AsymmetricFFN",
    "SparseBox3DKeyPointsGenerator",
    "SparsePoint3DKeyPointsGenerator",
]


@ATTENTION.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
        bev_num_levels=0,  # 수정(2026-10-15): B 융합 — BEV(LiDAR/Radar) 표본 수준 수. 0(기본) = 종전과 완전히 같음
        bev_range=(-51.2, -51.2, 51.2, 51.2),  # 수정(2026-10-15): B 융합 — BEV 격자 x0,y0,x1,y1 (LiDAR 키프레임 좌표, m)
        bev_feats_key="fusion_bev_feats",  # 수정(2026-10-15): B 융합 — metas 에서 BEV 특징 리스트를 읽는 키
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        if embed_dims % num_groups != 0:
            raise ValueError(f"embed_dims must be divisible by num_groups, but got {embed_dims} and {num_groups}")
        self.group_dims = int(embed_dims / num_groups)
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.embed_dims = embed_dims

        self.use_camera_embed = use_camera_embed
        self.use_deformable_func = use_deformable_func

        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)

        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        self.num_pts = self.kps_generator.num_pts

        self.output_proj = Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = Sequential(*linear_relu_ln(embed_dims, 1, 2, 12))
            self.weights_fc = Linear(embed_dims, num_groups * num_levels * self.num_pts)
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(embed_dims, num_groups * num_cams * num_levels * self.num_pts)
        # 수정(2026-10-15): B 융합 — FUTR3D 식 모달리티 공통 쿼리 표본추출. 각 키포인트를 BEV 격자 (x,y) 에 놓고
        #   bilinear 표본추출한 특징을 카메라 표본과 **같은 softmax** 로 가중합한다(카메라 로짓 cams×levels×pts 에
        #   BEV 로짓 bev_levels×pts 를 이어 붙여 그룹별 softmax). bev_num_levels=0 이면 모듈을 만들지 않는다
        #   → 기존 config 의 state_dict·난수열·순전파가 종전과 같다.
        self.bev_num_levels = int(bev_num_levels)
        self.bev_range = tuple(float(v) for v in bev_range)
        self.bev_feats_key = bev_feats_key
        if self.bev_num_levels > 0:
            self.bev_weights_fc = Linear(embed_dims, num_groups * self.bev_num_levels * self.num_pts)

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        if self.bev_num_levels > 0:  # 수정(2026-10-15): B 융합 — 카메라 가중치와 같이 0 초기화(초기 softmax 균등)
            constant_init(self.bev_weights_fc, val=0.0, bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, anchor_embed, instance_feature)
        if self.bev_num_levels > 0:  # 수정(2026-10-15): B 융합 — 카메라·BEV 공동 softmax 가중치
            weights, bev_weights = self._get_weights_with_bev(instance_feature, anchor_embed, metas)
        else:
            weights = self._get_weights(instance_feature, anchor_embed, metas)

        if self.use_deformable_func:
            points_2d = (
                self.project_points(
                    key_points,
                    metas["projection_mat"],
                    metas.get("image_wh"),
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )
        else:
            features = self.feature_sampling(
                feature_maps,
                key_points,
                metas["projection_mat"],
                metas.get("image_wh"),
            )
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(dim=2)  # fuse multi-point features
        if self.bev_num_levels > 0:  # 수정(2026-10-15): B 융합 — BEV 표본을 같은 가중 합에 더한다(합 = 공동 softmax 가중합)
            features = features + self.bev_sampling(key_points, metas[self.bev_feats_key], bev_weights)
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(metas["projection_mat"][:, :, :3].reshape(bs, self.num_cams, -1))
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(bs, num_anchor, self.num_cams, 1, self.num_pts, 1)
            # 수정(2026-10-10): 같은 CPU 난수를 pinned 비동기 복사(값·난수열 동일, 동기화 제거)
            mask = mask.pin_memory().to(device=weights.device, non_blocking=True).to(dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (1 - self.attn_drop)

        return weights

    def _get_weights_with_bev(self, instance_feature, anchor_embed, metas):
        """수정(2026-10-15): B 융합 — _get_weights 와 같은 카메라 로짓에 BEV 로짓을 이어 붙여 그룹별 공동 softmax.
        반환: weights (bs,A,cams,levels,pts,G) — _get_weights 와 같은 모양, bev_weights (bs,A,bev_levels,pts,G)."""
        bs, num_anchor = instance_feature.shape[:2]
        base = instance_feature + anchor_embed
        feature = base
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(metas["projection_mat"][:, :, :3].reshape(bs, self.num_cams, -1))
            feature = feature[:, :, None] + camera_embed[:, None]
        cam_logits = self.weights_fc(feature).reshape(bs, num_anchor, -1, self.num_groups)
        bev_logits = self.bev_weights_fc(base).reshape(bs, num_anchor, -1, self.num_groups)
        n_cam = cam_logits.shape[2]
        w = torch.cat([cam_logits, bev_logits], dim=2).softmax(dim=2)
        weights = w[:, :, :n_cam].reshape(
            bs, num_anchor, self.num_cams, self.num_levels, self.num_pts, self.num_groups)
        bev_weights = w[:, :, n_cam:].reshape(bs, num_anchor, self.bev_num_levels, self.num_pts, self.num_groups)
        if self.training and self.attn_drop > 0:
            # 카메라 경로와 같은 방식(시점·점 단위 drop). BEV 는 하나의 '시점'으로 본다.
            mask = torch.rand(bs, num_anchor, self.num_cams, 1, self.num_pts, 1)
            mask = mask.pin_memory().to(device=weights.device, non_blocking=True).to(dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (1 - self.attn_drop)
            bmask = torch.rand(bs, num_anchor, 1, self.num_pts, 1)
            bmask = bmask.pin_memory().to(device=weights.device, non_blocking=True).to(dtype=weights.dtype)
            bev_weights = ((bmask > self.attn_drop) * bev_weights) / (1 - self.attn_drop)
        return weights, bev_weights

    def bev_sampling(self, key_points, bev_feats, bev_weights):
        """수정(2026-10-15): B 융합 — 키포인트 (x,y) 를 BEV 격자 정규화 좌표로 바꿔 수준별 bilinear 표본추출 후 가중합.
        key_points (bs,A,P,3) LiDAR 키프레임 좌표, bev_feats [(bs,C,H_l,W_l)], 캔버스[b,:,iy,ix] 규약
        (픽셀 (iy,ix) 중심 = x0+(ix+0.5)·(x1-x0)/W …) → grid_sample(align_corners=False) 와 정확히 대응. 격자 밖은 0."""
        if isinstance(bev_feats, torch.Tensor):
            bev_feats = [bev_feats]
        assert len(bev_feats) == self.bev_num_levels, (len(bev_feats), self.bev_num_levels)
        bs, num_anchor, num_pts = key_points.shape[:3]
        x0, y0, x1, y1 = self.bev_range
        gx = (key_points[..., 0] - x0) / (x1 - x0) * 2 - 1
        gy = (key_points[..., 1] - y0) / (y1 - y0) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).to(dtype=bev_feats[0].dtype)  # (bs,A,P,2)
        out = 0
        for l, fm in enumerate(bev_feats):
            s = torch.nn.functional.grid_sample(fm, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            s = s.permute(0, 2, 3, 1).reshape(bs, num_anchor, num_pts, self.num_groups, self.group_dims)
            out = out + (bev_weights[:, :, l, :, :, None] * s).sum(dim=2)
        return out.reshape(bs, num_anchor, self.embed_dims)

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(projection_mat[:, :, None, None], pts_extend[:, None, ..., None]).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(key_points, projection_mat, image_wh)
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(fm.flatten(end_dim=1), points_2d)
            )
        features = torch.stack(features, dim=1)
        # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims
        features = features.reshape(bs, num_cams, num_levels, -1, num_anchor, num_pts).permute(0, 4, 1, 2, 5, 3)

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(features.shape[:-1] + (self.num_groups, self.group_dims))
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features


@FEEDFORWARD_NETWORK.register_module()
class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)


@PLUGIN_LAYERS.register_module()
class SparseBox3DKeyPointsGenerator(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        fix_scale=None,
    ):
        super(SparseBox3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = nn.Parameter(
            torch.tensor(fix_scale), requires_grad=False
        )
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, num_learnable_pts * 3)

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(self, anchor, anchor_embed=None, instance_feature=None):
        # 수정(2026-10-15): 시그니처를 DeformableFeatureAggregation 의 호출 순서(anchor, anchor_embed, instance_feature)에
        #   맞춰 명시했다. 종전 시그니처 (anchor, instance_feature=None, T_cur2temp_list=None, ...) 에서는 DFA 가 넘긴
        #   anchor_embed 가 instance_feature 자리로 들어가 학습형 키포인트 오프셋을 계산했고, instance_feature 는
        #   T_cur2temp_list 자리로 들어갔으나 timestamp 가 없어 시간축 분기는 타지 않았다.
        #   → 오프셋은 종전과 똑같이 anchor_embed 로 계산한다(동작 동일). 미사용 시간축 분기는 제거.
        bs, num_anchor = anchor.shape[:2]
        size = anchor[..., None, W:H + 1].clamp(max=5.0).exp()  # 수정(2026-10-10): 슬라이스(동기화 제거)  # 수정: exp overflow(inf) 원천차단
        key_points = self.fix_scale * size
        if self.num_learnable_pts > 0 and anchor_embed is not None:
            learnable_scale = (
                self.learnable_fc(anchor_embed)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )
            key_points = torch.cat(
                [key_points, learnable_scale * size], dim=-2
            )

        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])

        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 2, 2] = 1

        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1)
        key_points = key_points + anchor[..., None, X:Z + 1]  # 수정(2026-10-10): 슬라이스(동기화 제거)
        return key_points

    @staticmethod
    def anchor_projection(
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            vel = anchor[..., VX:]
            vel_dim = vel.shape[-1]
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            )

            center = anchor[..., X:Z + 1]  # 수정(2026-10-10): 슬라이스(동기화 제거)
            if time_intervals is not None:
                time_interval = time_intervals[i]
            elif src_timestamp is not None and dst_timestamps is not None:
                time_interval = (src_timestamp - dst_timestamps[i]).to(
                    dtype=vel.dtype
                )
            else:
                time_interval = None
            if time_interval is not None:
                translation = vel.transpose(0, -1) * time_interval
                translation = translation.transpose(0, -1)
                center = center - translation
            center = (
                torch.matmul(
                    T_src2dst[..., :3, :3], center[..., None]
                ).squeeze(dim=-1)
                + T_src2dst[..., :3, 3]
            )
            size = anchor[..., W:H + 1]  # 수정(2026-10-10): 슬라이스(동기화 제거)
            yaw = torch.matmul(
                T_src2dst[..., :2, :2],
                anchor[..., SIN_YAW:COS_YAW + 1].flip(-1)[..., None],  # 수정(2026-10-10): [COS,SIN] 순서 = 슬라이스 뒤집기(동기화 제거)
            ).squeeze(-1)
            yaw = yaw.flip(-1)  # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — 마지막 축 길이 2 → [1,0] 색인과 같은 원소(기울기도 같은 자리로), 리스트 색인 H2D 제거
            vel = torch.matmul(
                T_src2dst[..., :vel_dim, :vel_dim], vel[..., None]
            ).squeeze(-1)
            dst_anchor = torch.cat([center, size, yaw, vel], dim=-1)
            dst_anchors.append(dst_anchor)
        return dst_anchors


@PLUGIN_LAYERS.register_module()
class SparsePoint3DKeyPointsGenerator(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        num_learnable_pts: int = 0,
        fix_height: Tuple = (0,),
        ground_height: int = 0,
    ):
        super(SparsePoint3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.num_learnable_pts = num_learnable_pts
        self.num_pts = num_sample * len(fix_height) * num_learnable_pts

        if self.num_learnable_pts > 0:
            self.learnable_fc = Linear(self.embed_dims, self.num_pts * 2)

        self.fix_height = np.array(fix_height)
        self.ground_height = ground_height

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(self, anchor, anchor_embed=None, instance_feature=None):
        # 수정(2026-10-15): 미사용 분기(with_points_embed / with_anchor_embed / 시간축 키포인트) 제거.
        #   오프셋은 종전 기본 경로와 같이 instance_feature 로 계산한다(동작 동일).
        assert self.num_learnable_pts > 0, 'No learnable pts'
        bs, num_anchor, _ = anchor.shape

        key_points = anchor.view(bs, num_anchor, self.num_sample, -1)

        ### learn offset
        offset = (
            self.learnable_fc(instance_feature)
            .reshape(bs, num_anchor, self.num_sample, len(self.fix_height), self.num_learnable_pts, 2)
        )

        key_points = offset + key_points[..., None, None, :]
        key_points = torch.cat(
            [
                key_points,
                key_points.new_full(key_points.shape[:-1]+(1,), fill_value=self.ground_height),
            ],
            dim=-1,
        )
        fix_height = const_like(key_points, self.fix_height)  # 수정(2026-10-10): 장치별 캐시 상수(동기화 제거)
        height_offset = key_points.new_zeros([len(fix_height), 2])
        height_offset = torch.cat([height_offset, fix_height[:,None]], dim=-1)
        key_points = key_points + height_offset[None, None, None, :, None]
        key_points = key_points.flatten(2, 4)
        return key_points

    def anchor_projection(
        self,
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            dst_anchor = anchor.clone()
            bs, num_anchor, _ = anchor.shape

            dst_anchor = dst_anchor.reshape(bs, num_anchor, self.num_sample, -1).flatten(1, 2)
            T_src2dst = torch.unsqueeze(T_src2dst_list[i].to(dtype=anchor.dtype), dim=1)

            dst_anchor = torch.matmul(T_src2dst[..., :2, :2], dst_anchor[..., None]).squeeze(dim=-1) + T_src2dst[..., :2, 3]
            dst_anchor = dst_anchor.reshape(bs, num_anchor, self.num_sample, -1).flatten(2, 3)
            dst_anchors.append(dst_anchor)

        return dst_anchors
