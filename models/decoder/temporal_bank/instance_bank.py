# 수정(2026-10-15): 경로 이동 models/instance_bank.py → models/decoder/temporal_bank/instance_bank.py.
#   Sparse4DHead 전용 노이즈 쿼리(DN) 분기(get 의 dn_metas 투영, update 의 DN 분리·재결합)를 제거했다.
#   통합 디코더는 DN 쿼리를 만들지 않아 해당 분기는 항상 건너뛰어졌다(동작 동일). topk 는 models/common/layers.py.
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from ...common.box3d import *
from .topk import topk
from ...common.sync_free import to_device_async  # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — T_temp2cur 비동기 H2D


__all__ = ["InstanceBank"]


@PLUGIN_LAYERS.register_module()
class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
        class_names=None,
        zero_velocity_classes=None,
    ):
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval
        self.class_names = class_names
        self.zero_velocity_classes = zero_velocity_classes

        if anchor_handler is not None:
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)
        if len(anchor.shape) == 3: # for map
            anchor = anchor.reshape(anchor.shape[0], -1)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.reset()

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0

    def get(self, batch_size, metas=None):
        instance_feature = torch.tile(self.instance_feature[None], (batch_size, 1, 1))
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        if self.cached_anchor is not None and batch_size == self.cached_anchor.shape[0]:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            if self.anchor_handler is not None:
                # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — new_tensor(pageable H2D = 호스트 동기화) → 고정 메모리 비동기 복사.
                #   같은 numpy(float64) 를 같은 dtype(cached_anchor.dtype)으로 CPU 에서 변환 → 같은 값(스트림 순서로 결과 동일).
                T_temp2cur = to_device_async(
                    np.stack(
                        [x["T_global_inv"] @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])]
                    ),
                    self.cached_anchor.device,
                    dtype=self.cached_anchor.dtype,
                )
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor, [T_temp2cur], time_intervals=[-time_interval])[0]

            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                self.default_time_interval,  # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — 파이썬 스칼라(같은 dtype 으로 변환, 값 동일) — new_tensor H2D 제거
            )
        else:
            self.reset()
            time_interval = instance_feature.new_full((batch_size,), self.default_time_interval)  # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — H2D 대신 fill(값·dtype·장치 동일)

        return instance_feature, anchor, self.cached_feature, self.cached_anchor, time_interval

    def update(self, instance_feature, anchor, confidence):
        if self.cached_feature is None:
            return instance_feature, anchor

        N = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk(
            confidence, N, instance_feature, anchor
        )
        selected_feature = torch.cat([self.cached_feature, selected_feature], dim=1)
        selected_anchor = torch.cat([self.cached_anchor, selected_anchor], dim=1)
        instance_feature = torch.where(self.mask[:, None, None], selected_feature, instance_feature)
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        self.confidence = torch.where(
            self.mask[:, None],
            self.confidence,
            0,  # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — 스칼라(confidence dtype 유지, 값 동일) — new_tensor H2D 제거
        )
        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                -1,  # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — 스칼라(int64 유지, 값 동일) — new_tensor H2D 제거
            )

        return instance_feature, anchor

    def cache(
        self,
        instance_feature,
        anchor,
        confidence,
        metas=None,
        feature_maps=None,
    ):
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        confidence, indices = confidence.max(dim=-1)
        confidence = confidence.sigmoid()
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        self.temp_confidence = confidence

        (self.confidence, (self.cached_feature, self.cached_anchor)) = topk(
            confidence, self.num_temp_instances, instance_feature, anchor)

        if self.class_names and self.zero_velocity_classes is not None:
            (cached_confidence, (cached_indices,)) = topk(confidence, self.num_temp_instances, indices)
            cached_indices = cached_indices[..., 0]
            for class_name in self.zero_velocity_classes:
                class_indice = self.class_names.index(class_name)
                self.cached_anchor[..., VX:][cached_indices==class_indice] = 0.0


    def get_instance_id(self, confidence, anchor=None, threshold=None):
        confidence = confidence.max(dim=-1).values.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1).long()

        if self.instance_id is not None and self.instance_id.shape[0] == instance_id.shape[0]:
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)

        # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — arange(텐서)(.item 동기화) + where(mask) 색인(nonzero 동기화) 제거.
        #   torch.where(mask) 는 행 우선 순서이므로 k 번째(0 부터) 새 인스턴스 id = prev_id + k = 행 우선 누적합 − 1 + prev_id (같은 정수).
        flat = mask.reshape(-1).long()
        new_ids = (torch.cumsum(flat, 0) - 1 + self.prev_id).reshape(mask.shape)
        instance_id = torch.where(mask, new_ids, instance_id)
        self.prev_id = self.prev_id + flat.sum()  # 종전 += 와 같은 값·dtype(int64 0 차원 텐서)
        self.update_instance_id(instance_id, confidence)

        return instance_id

    def update_instance_id(self, instance_id=None, confidence=None):
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        instance_id = topk(temp_conf, self.num_temp_instances, instance_id)[1][0]
        instance_id = instance_id.squeeze(dim=-1)
        self.instance_id = F.pad(instance_id, (0, self.num_anchor - self.num_temp_instances), value=-1)

