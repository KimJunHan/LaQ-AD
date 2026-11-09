# 수정(2026-10-15): 경로 이동 models/det/target.py → models/head/detection_tracking/target.py. Sparse4DHead 전용 DN 메서드
#   (get_dn_anchors / update_dn / cache_dn) 제거 — 헝가리안 정합(sample) 동작 동일.
import torch
from ...common.sync_free import const_like, to_device_async, gather_to_host  # 수정(2026-11-09): gather_to_host 추가(정합 비용 일괄 전송). 수정(2026-10-10): 동기화 제거(속도) — 값 동일, models/common/sync_free.py 참조
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmdet.core.bbox.builder import BBOX_SAMPLERS

from ...common.box3d import *
from ...common.base_target import BaseTargetWithDenoising


__all__ = ["SparseBox3DTarget"]


@BBOX_SAMPLERS.register_module()
class SparseBox3DTarget(BaseTargetWithDenoising):
    def __init__(
        self,
        cls_weight=2.0,
        alpha=0.25,
        gamma=2,
        eps=1e-12,
        box_weight=0.25,
        reg_weights=None,
        cls_wise_reg_weights=None,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
    ):
        super(SparseBox3DTarget, self).__init__(
            num_dn_groups, num_temp_dn_groups
        )
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reg_weights = reg_weights
        if self.reg_weights is None:
            self.reg_weights = [1.0] * 8 + [0.0] * 2
        self.cls_wise_reg_weights = cls_wise_reg_weights
        # 수정(2026-10-15): dn_noise_scale / max_dn_gt / add_neg_dn 은 DN 경로 제거로 쓰이지 않는다(config 호환용으로만 받음).

    def encode_reg_target(self, box_target, device=None):
        outputs = []
        for box in box_target:
            output = torch.cat(
                [
                    box[..., X:Z + 1],  # 수정(2026-10-10): 리스트 인덱스→슬라이스(같은 값, H2D 동기화 제거)
                    box[..., W:H + 1].log(),
                    torch.sin(box[..., YAW]).unsqueeze(-1),
                    torch.cos(box[..., YAW]).unsqueeze(-1),
                    box[..., YAW + 1 :],
                ],
                dim=-1,
            )
            if device is not None:
                output = output.to(device=device)
            outputs.append(output)
        return outputs

    def sample(
        self,
        cls_pred,
        box_pred,
        cls_target,
        box_target,
    ):
        # 수정(2026-11-09): 학습 속도 — 비용 계산(prepare)과 정합·목표 생성(finish)을 둘로 나눴다. 이 함수의 동작·결과는 종전과 같다
        #   (prepare → 이 층 비용만 GPU→CPU 1 회 → finish). unified_decoder.loss 는 det·map 전 층의 prepare 를 먼저 부르고
        #   비용을 한 번에 옮긴 뒤(gather_to_host) 층별 finish 를 부른다 → 층마다 있던 GPU→CPU 대기 6 회가 1 회로 준다.
        st = self.prepare(cls_pred, box_pred, cls_target, box_target)
        return self.finish(st, gather_to_host(st["costs"]))

    def prepare(self, cls_pred, box_pred, cls_target, box_target):
        """# 수정(2026-11-09): 종전 sample() 의 비용 계산 부분(코드·연산 순서 그대로). 반환 st["costs"] = 표본별 비용(GPU, detach) 또는 None."""
        bs, num_pred, num_cls = cls_pred.shape

        cls_cost = self._cls_cost(cls_pred, cls_target)

        box_target = self.encode_reg_target(box_target, box_pred.device)

        instance_reg_weights = []
        for i in range(len(box_target)):
            weights = torch.logical_not(box_target[i].isnan()).to(
                dtype=box_target[i].dtype
            )
            if self.cls_wise_reg_weights is not None:
                for cls, weight in self.cls_wise_reg_weights.items():
                    weights = torch.where(
                        (cls_target[i] == cls)[:, None],
                        const_like(weights, weight),  # 수정(2026-10-10): 장치별 캐시 상수
                        weights,
                    )
            instance_reg_weights.append(weights)
        box_cost = self._box_cost(box_pred, box_target, instance_reg_weights)

        # 수정(2026-10-10): 동기화 제거 — 표본별 비용을 모아 GPU→CPU 한 번, 정합 결과는 pinned 비동기 복사(값 동일)
        costs = [(cls_cost[i] + box_cost[i]).detach() if (cls_cost[i] is not None and box_cost[i] is not None) else None
                 for i in range(bs)]
        return dict(costs=costs, cls_pred=cls_pred, box_pred=box_pred, cls_target=cls_target,
                    box_target=box_target, instance_reg_weights=instance_reg_weights)

    def finish(self, st, host_costs):
        """# 수정(2026-11-09): 종전 sample() 의 정합·목표 생성 부분(코드 그대로). host_costs = st["costs"] 의 CPU numpy(같은 순서, None 유지)."""
        cls_pred, box_pred, cls_target = st["cls_pred"], st["box_pred"], st["cls_target"]
        box_target, instance_reg_weights = st["box_target"], st["instance_reg_weights"]
        bs, num_pred, num_cls = cls_pred.shape
        indices = []
        for cost in host_costs:
            if cost is None:
                indices.append([None, None]); continue
            cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
            assign = linear_sum_assignment(cost)
            indices.append([to_device_async(x, cls_pred.device, torch.int64) for x in assign])

        output_cls_target = (
            cls_target[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        )
        output_box_target = box_pred.new_zeros(box_pred.shape)
        output_reg_weights = box_pred.new_zeros(box_pred.shape)
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(cls_target[i]) == 0:
                continue
            output_cls_target[i, pred_idx] = cls_target[i][target_idx]
            output_box_target[i, pred_idx] = box_target[i][target_idx]
            output_reg_weights[i, pred_idx] = instance_reg_weights[i][
                target_idx
            ]
        self.indices = indices
        return output_cls_target, output_box_target, output_reg_weights

    def _cls_cost(self, cls_pred, cls_target):
        bs = cls_pred.shape[0]
        cls_pred = cls_pred.sigmoid()
        cost = []
        for i in range(bs):
            if len(cls_target[i]) > 0:
                neg_cost = (
                    -(1 - cls_pred[i] + self.eps).log()
                    * (1 - self.alpha)
                    * cls_pred[i].pow(self.gamma)
                )
                pos_cost = (
                    -(cls_pred[i] + self.eps).log()
                    * self.alpha
                    * (1 - cls_pred[i]).pow(self.gamma)
                )
                cost.append(
                    (pos_cost[:, cls_target[i]] - neg_cost[:, cls_target[i]])
                    * self.cls_weight
                )
            else:
                cost.append(None)
        return cost

    def _box_cost(self, box_pred, box_target, instance_reg_weights):
        bs = box_pred.shape[0]
        cost = []
        for i in range(bs):
            if len(box_target[i]) > 0:
                cost.append(
                    torch.sum(
                        torch.abs(box_pred[i, :, None] - box_target[i][None])
                        * instance_reg_weights[i][None]
                        * const_like(box_pred, self.reg_weights),  # 수정(2026-10-10): 장치별 캐시 상수
                        dim=-1,
                    )
                    * self.box_weight
                )
            else:
                cost.append(None)
        return cost
