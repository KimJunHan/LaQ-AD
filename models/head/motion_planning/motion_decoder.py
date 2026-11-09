# 수정(2026-10-15): 경로 이동 models/motion/decoder.py → models/head/motion_planning/motion_decoder.py.
#   미사용 SparseBox3DMotionDecoder·HierarchicalPlanningDecoder 와 그것들만 쓰던 충돌 판정 도우미
#   (check_collision, corners_in_box, box3d_to_corners*) 를 제거했다. SparseMotionDecoder 동작 동일.
import torch
import numpy as np
from typing import Optional
from mmdet.core.bbox.builder import BBOX_CODERS

from ...common.box3d import *
from ..detection_tracking.decoder import SparseBox3DDecoder, decode_box

__all__ = ["SparseMotionDecoder"]


@BBOX_CODERS.register_module()
class SparseMotionDecoder(SparseBox3DDecoder):
    def __init__(self):
        super(SparseMotionDecoder, self).__init__()

    def decode(
            self,
            cls_scores,
            box_preds,
            instance_id=None,
            quality=None,
            motion_output=None,
            output_idx=-1,
    ):
        squeeze_cls = instance_id is not None

        cls_scores = cls_scores[output_idx].sigmoid()

        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        if not squeeze_cls:
            cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if quality[output_idx] is None:
            quality = None
        if quality is not None:
            centerness = quality[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        # anchor_queue = motion_output["anchor_queue"]
        # anchor_queue = torch.stack(anchor_queue, dim=2)
        # period = motion_output["period"]

        for i in range(bs):
            category_ids = cls_ids[i]
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            if quality is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            box = decode_box(box)
            trajs = motion_output["prediction"][-1]
            traj_cls = motion_output["classification"][-1].sigmoid()
            traj = trajs[i, indices[i] // num_cls]
            traj_cls = traj_cls[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                traj = traj[mask[i]]
                traj_cls = traj_cls[mask[i]]
            traj = traj.cumsum(dim=-2) + box[:, None, None, :2]
            output.append(
                {
                    "trajs_3d": traj.cpu(),
                    "trajs_score": traj_cls.cpu()
                }
            )

            # temp_anchor = anchor_queue[i, indices[i] // num_cls]
            # temp_period = period[i, indices[i] // num_cls]
            # if self.score_threshold is not None:
            #     temp_anchor = temp_anchor[mask[i]]
            #     temp_period = temp_period[mask[i]]
            # num_pred, queue_len = temp_anchor.shape[:2]
            # temp_anchor = temp_anchor.flatten(0, 1)
            # temp_anchor = decode_box(temp_anchor)
            # temp_anchor = temp_anchor.reshape([num_pred, queue_len, box.shape[-1]])
            # output[-1]['anchor_queue'] = temp_anchor.cpu()
            # output[-1]['period'] = temp_period.cpu()

        return output
