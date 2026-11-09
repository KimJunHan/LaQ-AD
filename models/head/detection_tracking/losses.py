# 수정(2026-10-15): 경로 이동 models/det/losses.py → models/head/detection_tracking/losses.py (소스 정리, 동작 동일).
import torch
from ...common.sync_free import const_like  # 수정(2026-10-10): 동기화 제거
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmdet.models.builder import LOSSES

from ...common.box3d import *


@LOSSES.register_module()
class SparseBox3DLoss(nn.Module):
    def __init__(
        self,
        loss_box,
        loss_centerness=None,
        loss_yawness=None,
        cls_allow_reverse=None,
        cns_detach=True,  # 수정(2026-10-15): 실험 C — centerness 타깃의 예측 중심 detach 여부(기본 True = 10-08 이후 동작 유지, False = SparseDrive 원본)
    ):
        super().__init__()

        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.loss_box = build(loss_box, LOSSES)
        self.loss_cns = build(loss_centerness, LOSSES)
        self.loss_yns = build(loss_yawness, LOSSES)
        self.cls_allow_reverse = cls_allow_reverse
        self.cns_detach = bool(cns_detach)  # 수정(2026-10-15): 실험 C — config 로 제어

    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        prefix="",
        suffix="",
        quality=None,
        cls_target=None,
        **kwargs,
    ):
        # 수정(2026-10-10): 동기화 제거 — 리스트 인덱스→슬라이스, 상수는 장치별 캐시(값 동일, models/common/sync_free.py)
        # Some categories do not distinguish between positive and negative
        # directions. For example, barrier in nuScenes dataset.
        if self.cls_allow_reverse is not None and cls_target is not None:
            if_reverse = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., SIN_YAW:COS_YAW + 1],
                    box[..., SIN_YAW:COS_YAW + 1],
                    dim=-1,
                )
                < 0
            )
            if_reverse = (
                torch.isin(
                    cls_target, const_like(cls_target, self.cls_allow_reverse)
                )
                & if_reverse
            )
            box_target[..., SIN_YAW:COS_YAW + 1] = torch.where(
                if_reverse[..., None],
                -box_target[..., SIN_YAW:COS_YAW + 1],
                box_target[..., SIN_YAW:COS_YAW + 1],
            )

        output = {}
        box_loss = self.loss_box(
            box, box_target, weight=weight, avg_factor=avg_factor
        )
        output[f"{prefix}loss_box{suffix}"] = box_loss

        if quality is not None:
            cns = quality[..., CNS]
            yns = quality[..., YNS].sigmoid()
            #수정(2026-10-08): 예측 중심을 detach. BCEWithLogits 는 타깃에 대해서도 미분 가능해서
            #  detach 가 없으면 cns 손실이 박스 중심에 가짜 기울기를 흘린다
            #  (cns logit -2.0 에서 dL/dbox_x = -0.7358 로 중심을 GT 에서 밀어냄, 정당한 회귀 기울기 0.5 초과).
            # 수정(2026-10-15): 실험 C — cns_detach=False 면 SparseDrive 원본(detach 없음: cns 손실이 타깃을 통해 박스 중심에도
            #   기울기를 보낸다). 기본 True 는 위 10-08 동작 그대로(기존 config·체크포인트 재현성 유지).
            box_center = box[..., X:Z + 1].detach() if self.cns_detach else box[..., X:Z + 1]
            cns_target = torch.norm(
                box_target[..., X:Z + 1] - box_center, p=2, dim=-1
            )
            cns_target = torch.exp(-cns_target)
            cns_loss = self.loss_cns(cns, cns_target, avg_factor=avg_factor)
            output[f"{prefix}loss_cns{suffix}"] = cns_loss

            yns_target = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., SIN_YAW:COS_YAW + 1],
                    box[..., SIN_YAW:COS_YAW + 1],
                    dim=-1,
                )
                > 0
            )
            yns_target = yns_target.float()
            yns_loss = self.loss_yns(yns, yns_target, avg_factor=avg_factor)
            output[f"{prefix}loss_yns{suffix}"] = yns_loss
        return output
