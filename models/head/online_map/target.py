# 수정(2026-10-15): 경로 이동 models/map/target.py → models/head/online_map/target.py (소스 정리, 동작 동일).
import torch
from ...common.sync_free import dev_const, to_device_async, gather_to_host  # 수정(2026-11-09): gather_to_host 추가(정합 비용 일괄 전송). 수정(2026-10-10): 동기화 제거(속도) — 값 동일, models/common/sync_free.py 참조
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmdet.core.bbox.builder import (BBOX_SAMPLERS, BBOX_ASSIGNERS)
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core import (build_assigner, build_sampler)
from mmdet.core.bbox.assigners import (AssignResult, BaseAssigner)

from ...common.base_target import BaseTargetWithDenoising


@BBOX_SAMPLERS.register_module()
class SparsePoint3DTarget(BaseTargetWithDenoising):
    def __init__(
        self,
        assigner=None,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
        num_cls=3,
        num_sample=20,
        roi_size=(30, 60),
    ):
        super(SparsePoint3DTarget, self).__init__(num_dn_groups, num_temp_dn_groups)
        self.assigner = build_assigner(assigner)
        # 수정(2026-10-15): dn_noise_scale / max_dn_gt / add_neg_dn 은 DN 경로 제거로 쓰이지 않는다(config 호환용으로만 받음).

        self.num_cls = num_cls
        self.num_sample = num_sample
        self.roi_size = roi_size

    def sample(self, cls_preds, pts_preds, cls_targets, pts_targets):
        # 수정(2026-11-09): 학습 속도 — 비용 계산(prepare)과 정합·목표 생성(finish)을 둘로 나눴다. 이 함수의 결과는 종전과 같다.
        #   unified_decoder.loss 는 det·map 전 층의 prepare 를 먼저 부르고 비용을 한 번에 옮긴 뒤(gather_to_host) 층별 finish 를 부른다.
        st = self.prepare(cls_preds, pts_preds, cls_targets, pts_targets)
        return self.finish(st, gather_to_host(st["costs"]))

    def prepare(self, cls_preds, pts_preds, cls_targets, pts_targets):
        """# 수정(2026-11-09): 종전 sample() 의 비용 계산 부분(코드·연산 순서 그대로). st["costs"] = 표본별 비용(GPU) 또는 None."""
        pts_targets = [x.flatten(2, 3) if len(x.shape)==4 else x for x in pts_targets]
        # 수정(2026-10-10): 동기화 제거 — 표본별 비용 행렬을 모아 GPU→CPU 전송을 한 번으로 묶는다(정합 결과 동일).
        costs, permutes = [], []
        for(cls_pred, pts_pred, cls_target, pts_target) in zip(cls_preds, pts_preds, cls_targets, pts_targets):
            # normalize to (0, 1)
            pts_pred = self.normalize_line(pts_pred)
            pts_target = self.normalize_line(pts_target)
            preds = dict(lines=pts_pred, scores=cls_pred)
            gts = dict(lines=pts_target, labels=cls_target)
            cost, perm = self.assigner.cost_matrix(preds, gts)
            costs.append(None if cost is None else cost.detach()); permutes.append(perm)  # 수정(2026-11-09): 검토 A — 6층 비용 그래프를 정합까지 붙잡아 손실 단계 메모리 4~5배 → detach(정합·손실 비트 동일 확인)
        return dict(costs=costs, permutes=permutes, cls_preds=cls_preds, pts_preds=pts_preds,
                    cls_targets=cls_targets, pts_targets=pts_targets)

    def finish(self, st, host_costs):
        """# 수정(2026-11-09): 종전 sample() 의 정합·목표 생성 부분. host_costs = st["costs"] 의 CPU numpy(같은 순서, None 유지)."""
        cls_preds, pts_preds, cls_targets, pts_targets = st["cls_preds"], st["pts_preds"], st["cls_targets"], st["pts_targets"]
        indices = []
        for cost, c, perm in zip(st["costs"], host_costs, st["permutes"]):
            if cost is None:
                indices.append((None, None, None)); continue
            r, cidx = linear_sum_assignment(c)
            indices.append((to_device_async(r, cost.device, torch.int64), to_device_async(cidx, cost.device, torch.int64), perm))
        
        bs, num_pred, num_cls = cls_preds.shape
        output_cls_target = cls_targets[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        output_box_target = pts_preds.new_zeros(pts_preds.shape)
        output_reg_weights = pts_preds.new_zeros(pts_preds.shape)
        for i, (pred_idx, target_idx, gt_permute_index) in enumerate(indices):
            if len(cls_targets[i]) == 0:
                continue
            permute_idx = gt_permute_index[pred_idx, target_idx]
            output_cls_target[i, pred_idx] = cls_targets[i][target_idx]
            output_box_target[i, pred_idx] = pts_targets[i][target_idx, permute_idx]
            # 수정(2026-11-09): 동기화 제거 — 종전 `output_reg_weights[i, pred_idx] = 1` 은 파이썬 스칼라를 index_put 하면서
            #   표본×층마다 작은 H2D 복사(=동기화, 96 회/iter)를 만들었다. 행 i 의 뷰에 pred_idx 행들을 1.0 으로 채우는 index_fill_ 은
            #   스칼라를 커널 인자로 넘겨 복사가 없다. 같은 원소(행 i, 행 pred_idx 의 모든 열)를 같은 값 1 로 쓰므로 결과가 같다.
            output_reg_weights[i].index_fill_(0, pred_idx, 1.0)

        return output_cls_target, output_box_target, output_reg_weights

    def normalize_line(self, line):
        if line.shape[0] == 0:
            return line
        
        line = line.view(line.shape[:-1] + (self.num_sample, -1))
        
        # 수정(2026-10-10): 동기화 제거 — 같은 식(float32)으로 만든 상수를 장치별 캐시
        r0, r1 = self.roi_size
        origin = dev_const(line, ("map_origin", r0, r1), lambda: -torch.tensor([r0/2, r1/2], dtype=line.dtype))
        line = line - origin

        # transform from range [0, 1] to (0, 1)
        eps = 1e-5
        norm = dev_const(line, ("map_norm", r0, r1, eps), lambda: torch.tensor([r0, r1], dtype=line.dtype) + eps)
        line = line / norm
        line = line.flatten(-2, -1)

        return line


@BBOX_ASSIGNERS.register_module()
class HungarianLinesAssigner(BaseAssigner):
    """
        Computes one-to-one matching between predictions and ground truth.
        This class computes an assignment between the targets and the predictions
        based on the costs. The costs are weighted sum of three components:
        classification cost and regression L1 cost. The
        targets don't include the no_object, so generally there are more
        predictions than targets. After the one-to-one matching, the un-matched
        are treated as backgrounds. Thus each query prediction will be assigned
        with `0` or a positive integer indicating the ground truth index:
        - 0: negative sample, no assigned gt
        - positive integer: positive sample, index (1-based) of assigned gt
        Args:
            cls_weight (int | float, optional): The scale factor for classification
                cost. Default 1.0.
            bbox_weight (int | float, optional): The scale factor for regression
                L1 cost. Default 1.0.
    """

    def __init__(self, cost=dict, **kwargs):
        self.cost = build_match_cost(cost)

    def assign(self,
               preds: dict,
               gts: dict,
               ignore_cls_cost=False,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """
            Computes one-to-one matching based on the weighted costs.
            This method assign each query prediction to a ground truth or
            background. The `assigned_gt_inds` with -1 means don't care,
            0 means negative sample, and positive number is the index (1-based)
            of assigned gt.
            The assignment is done in the following steps, the order matters.
            1. assign every prediction to -1
            2. compute the weighted costs
            3. do Hungarian matching on CPU based on the costs
            4. assign all to 0 (background) first, then for each matched pair
            between predictions and gts, treat this prediction as foreground
            and assign the corresponding gt index (plus 1) to it.
            Args:
                lines_pred (Tensor): predicted normalized lines:
                    [num_query, num_points, 2]
                cls_pred (Tensor): Predicted classification logits, shape
                    [num_query, num_class].

                lines_gt (Tensor): Ground truth lines
                    [num_gt, num_points, 2].
                labels_gt (Tensor): Label of `gt_bboxes`, shape (num_gt,).
                gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                    labelled as `ignored`. Default None.
                eps (int | float, optional): A value added to the denominator for
                    numerical stability. Default 1e-7.
            Returns:
                :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        
        num_gts, num_lines = gts['lines'].size(0), preds['lines'].size(0)
        if num_gts == 0 or num_lines == 0:
            return None, None, None

        # compute the weighted costs
        gt_permute_idx = None # (num_preds, num_gts)
        if self.cost.reg_cost.permute:
            cost, gt_permute_idx = self.cost(preds, gts, ignore_cls_cost)
        else:
            cost = self.cost(preds, gts, ignore_cls_cost)

        # do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu().numpy()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        return matched_row_inds, matched_col_inds, gt_permute_idx

    def cost_matrix(self, preds: dict, gts: dict, ignore_cls_cost=False):
        """수정(2026-10-10): assign() 의 비용 계산 부분만(GPU 텐서 반환). 표본을 모아 한 번에 CPU 로 옮기기 위함."""
        num_gts, num_lines = gts['lines'].size(0), preds['lines'].size(0)
        if num_gts == 0 or num_lines == 0:
            return None, None
        if self.cost.reg_cost.permute:
            return self.cost(preds, gts, ignore_cls_cost)
        return self.cost(preds, gts, ignore_cls_cost), None