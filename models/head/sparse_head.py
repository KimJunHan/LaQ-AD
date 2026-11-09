# 수정(2026-10-15): 경로 이동 models/sparse_head.py → models/head/sparse_head.py 및 정리.
#   원본 Sparse4D 분기(task_config 의 with_det / with_map / with_motion_plan 으로 개별 헤드를 쓰는 경로)와
#   미사용 compute_planner_metric_stp3_ori 를 제거했다. 모든 config 가 task_config=dict(with_onedecoder=True) 다.
#   통합 디코더의 scenes(장면 토큰) 출력도 제거되어 model_outs 는 5-튜플이다.
import copy
import numpy as np
import torch

from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmdet.models import build_head

from ..loader.evaluation import PlanningMetric

__all__ = ["SparseHead"]


@HEADS.register_module()
class SparseHead(BaseModule):
    def __init__(
        self,
        task_config: dict,
        init_cfg=None,
        onedecoder_head=None,
        evaluate_bench2dive=False,
        **kwargs,
    ):
        super(SparseHead, self).__init__(init_cfg)
        self.task_config = task_config
        if not self.task_config.get('with_onedecoder', False):
            raise ValueError("task_config.with_onedecoder=True 만 지원한다(원본 Sparse4D 개별 헤드 경로는 2026-10-15 제거).")
        self.onedecoder_head = build_head(onedecoder_head)

        self.evaluate_bench2dive = evaluate_bench2dive

        if self.evaluate_bench2dive:
            self.planning_metric = PlanningMetric()

    def init_weights(self):
        self.onedecoder_head.init_weights()

    def forward(self, img, feature_maps, metas: dict):
        return self.onedecoder_head(img, feature_maps, metas)

    def loss(self, model_outs, data):
        losses = dict()
        det_output, map_output, ego_output, plan_output, motion_output = model_outs
        loss_onedecoder = self.onedecoder_head.loss(
            det_output, map_output, ego_output, plan_output, motion_output, data)
        losses.update(loss_onedecoder)
        return losses

    def post_process(self, model_outs, data):
        det_output, map_output, ego_output, plan_output, motion_output = model_outs

        det_result, map_result, ego_result, plan_result, motion_result = self.onedecoder_head.post_process(
            det_output, map_output, ego_output, plan_output, motion_output, data)

        task = self.onedecoder_head.task_select[0]
        if task == "det": batch_size = len(det_result)
        if task == "map": batch_size = len(map_result)
        if task == "plan": batch_size = len(plan_result)
        if task == "motion": batch_size = len(motion_result)

        results = [dict()] * batch_size
        for i in range(batch_size):
            for task in self.onedecoder_head.task_select:
                if task == "det": results[i].update(det_result[i])
                if task == "map": results[i].update(map_result[i])
                if task == "plan": results[i].update(plan_result[i])
                if task == "motion": results[i].update(motion_result[i])

        if self.evaluate_bench2dive:
            results = self.evaluate_metric(results, data)

        return results

    def evaluate_metric(self, results, data):
        for bs in range(len(results)):
            metric_dict = self.compute_planner_metric_stp3(bs, results, data)
            results[bs]['metric_results'] = metric_dict
        return results

    def compute_planner_metric_stp3(self, bs, results, data, remap_box=True):
        pred_ego_fut_trajs = results[bs]['plan_temp_2hz'][None]
        gt_ego_fut_trajs = data["gt_ego_fut_trajs_2hz"].cumsum(dim=-2)
        gt_agent_boxes = data["gt_bboxes_3d"][bs]
        if remap_box:
            temp = copy.deepcopy(gt_agent_boxes[:, 3])
            gt_agent_boxes[:, 3] = gt_agent_boxes[:, 4]
            gt_agent_boxes[:, 4] = temp
            gt_agent_boxes[:, 6] = - gt_agent_boxes[:, 6] - np.pi / 2
        gt_agent_feats = data["gt_attr_labels"]
        fut_valid_flag = (data["gt_ego_fut_masks_2hz"] == 1).all().item()

        metric_dict = dict()
        metric_dict['fut_valid_flag'] = fut_valid_flag

        segmentation, pedestrian = self.planning_metric.get_label(gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)

        future_second = 3
        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i + 1) * 2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy
                )
                metric_dict['plan_L2_{}s'.format(i + 1)] = traj_L2
                metric_dict['plan_obj_col_{}s'.format(i + 1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i + 1)] = obj_box_coll.mean().item()
            else:
                metric_dict['plan_L2_{}s'.format(i + 1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i + 1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i + 1)] = 0.0

        return metric_dict
