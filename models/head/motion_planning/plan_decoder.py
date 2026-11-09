# 수정(2026-10-15): 경로 이동 models/plan/decoder.py → models/head/motion_planning/plan_decoder.py (소스 정리, 동작 동일).
import torch
import numpy as np
from typing import Optional
from mmdet.core.bbox.builder import BBOX_CODERS
from ...common.box3d import *
from ..detection_tracking.decoder import *
from ...common.box_ops import box3d_to_corners_gpu


def check_collision(boxes1, boxes2):
    '''
        A rough check for collision detection:
            check if any corner point of boxes1 is inside boxes2 and vice versa.

        boxes1: tensor with shape [N, 7], [x, y, z, w, l, h, yaw]
        boxes2: tensor with shape [N, 7]
    '''
    col_1 = corners_in_box(boxes1.clone(), boxes2.clone())
    col_2 = corners_in_box(boxes2.clone(), boxes1.clone())
    collision = torch.logical_or(col_1, col_2)

    return collision


def corners_in_box(boxes1, boxes2):
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return False

    boxes1_yaw = boxes1[:, 6].clone()
    boxes1_loc = boxes1[:, :3].clone()
    cos_yaw = torch.cos(-boxes1_yaw)
    sin_yaw = torch.sin(-boxes1_yaw)
    rot_mat_T = torch.stack(
        [
            torch.stack([cos_yaw, sin_yaw]),
            torch.stack([-sin_yaw, cos_yaw]),
        ]
    )
    # translate and rotate boxes
    boxes1[:, :3] = boxes1[:, :3] - boxes1_loc
    boxes1[:, :2] = torch.einsum('ij,jki->ik', boxes1[:, :2], rot_mat_T)
    boxes1[:, 6] = boxes1[:, 6] - boxes1_yaw

    boxes2[:, :3] = boxes2[:, :3] - boxes1_loc
    boxes2[:, :2] = torch.einsum('ij,jki->ik', boxes2[:, :2], rot_mat_T)
    boxes2[:, 6] = boxes2[:, 6] - boxes1_yaw

    corners_box2 = box3d_to_corners_gpu(boxes2)[:, [0, 3, 7, 4], :2]
    # corners_box2 = torch.from_numpy(corners_box2).to(boxes2.device)
    H = boxes1[:, [3]]
    W = boxes1[:, [4]]

    collision = torch.logical_and(
        torch.logical_and(corners_box2[..., 0] <= H / 2, corners_box2[..., 0] >= -H / 2),
        torch.logical_and(corners_box2[..., 1] <= W / 2, corners_box2[..., 1] >= -W / 2),
    )
    collision = collision.any(dim=-1)

    return collision


@BBOX_CODERS.register_module()
class SparsePlanDecoder(object):
    def __init__(
            self,
            ego_fut_ts=6,
            ego_fut_cmd=3,
            ego_fut_mode=3,
            ego_vehicle='nus',
            ego_size=None,  #수정(2026-10-09): NAVSIM — ego_vehicle='custom' 일 때 [l, w, h] 직접 지정
            anchor_types=None,
            anchor_refer=None,
            speed_refer=None,
            with_rescore=False,
            adapt_status=False,
            ego_start_yaw=None,  # 수정(2026-11-06): rescore 자차 박스 정지 모드·첫 시점 yaw(라디안). None=π/2(y 전진, nuScenes·B2D 종전값), NAVSIM(+x 전진)=0.0
            rescore_ego_offset=0.5,  # 수정(2026-11-06): rescore 자차 박스 중심의 궤적 원점 대비 전방 오프셋(m). 0.5=SparseDrive nuScenes 라이다 원점값, NAVSIM 후륜축 원점=1.461
    ):
        super(SparsePlanDecoder, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_cmd = ego_fut_cmd
        self.ego_fut_mode = ego_fut_mode
        self.ego_vehicle = ego_vehicle
        self.with_rescore = with_rescore
        self.adapt_status = adapt_status
        self.ego_start_yaw = float(np.pi / 2 if ego_start_yaw is None else ego_start_yaw)  # 수정(2026-11-06): 전진축 규약(위 인자 설명)
        self.rescore_ego_offset = float(rescore_ego_offset)  # 수정(2026-11-06): 자차 박스 전방 오프셋(위 인자 설명)

        self.anchor_types = anchor_types
        self.anchor_refer = anchor_refer

        self.speed_refer = speed_refer

        assert anchor_types is not None
        self.num_group = len(anchor_types)

        if self.ego_vehicle == 'nus':
            self.ego_size = [4.08, 1.73, 1.56]
        elif self.ego_vehicle == 'b2d':
            self.ego_size = [4.89, 1.84, 1.49]
        elif self.ego_vehicle == 'custom':
            #수정(2026-10-09): NAVSIM 자차 제원(변환기 egometa 출력값)
            if ego_size is None or len(ego_size) != 3:
                raise ValueError("ego_vehicle='custom' 은 ego_size=[l, w, h] 가 필요하다.")
            self.ego_size = list(ego_size)
        else:
            raise NotImplementedError

    def decode(self, ego_output, det_output, motion_output, planning_output, data):

        prediction = planning_output['prediction'][-1]
        classification = planning_output['classification'][-1]

        reg_preds = list(prediction.chunk(chunks=self.num_group, dim=2))
        cls_preds = list(classification.chunk(chunks=self.num_group, dim=2))

        batch_size = classification.shape[0]
        for i in range(self.num_group):
            cls_preds[i] = cls_preds[i].reshape(batch_size, self.ego_fut_cmd, -1)
            reg_preds[i] = reg_preds[i].reshape(batch_size, self.ego_fut_cmd, -1, self.ego_fut_ts, 2).cumsum(dim=-2)

        cls_preds, reg_preds = self.select(ego_output, det_output, motion_output, cls_preds, reg_preds, data)

        outputs = []
        for i in range(batch_size):
            output = dict()
            # 수정(2026-10-09): 계획 모드별 확률 시각화용 출력(평가 지표에는 쓰지 않음)
            _mreg, _mcls = self._vis_modes
            output["plan_modes"] = _mreg[i].cpu()                       # (모드, T, 2) 누적 궤적
            output["plan_mode_probs"] = _mcls[i].float().softmax(-1).cpu()  # (모드,)
            speed_dict = dict()
            for idx, anchor_type in enumerate(self.anchor_types):
                if anchor_type[0] in ["temp", "spat"]:
                    output["plan_{}_{}".format(anchor_type[0], anchor_type[1])] = reg_preds[idx][i].cpu()
                elif anchor_type[0] == "speed":
                    if anchor_type[1] not in speed_dict:
                        speed_dict[anchor_type[1]] = {
                            "cls_preds": [cls_preds[idx]],
                            "reg_preds": [reg_preds[idx][i]],
                            "speed_areas": [anchor_type[2]]
                        }
                    else:
                        speed_dict[anchor_type[1]]["cls_preds"].append(cls_preds[idx])
                        speed_dict[anchor_type[1]]["reg_preds"].append(reg_preds[idx][i])
                        speed_dict[anchor_type[1]]["speed_areas"].append(anchor_type[2])
                else:
                    raise NotImplementedError


            if len(speed_dict):
                for speed_type in speed_dict:
                    speed_dict[speed_type]["cls_preds"] = torch.stack(speed_dict[speed_type]["cls_preds"])
                    speed_dict[speed_type]["reg_preds"] = torch.stack(speed_dict[speed_type]["reg_preds"])

                if self.with_rescore and len(det_output["prediction"]) > 0 and len(motion_output["prediction"]) > 0:
                    speed_dict = self.rescore_speed(speed_dict, det_output, motion_output)

                for speed_type in speed_dict:
                    speed_cls_preds = speed_dict[speed_type]["cls_preds"]
                    speed_reg_preds = speed_dict[speed_type]["reg_preds"]

                    speed_idx = torch.argmax(speed_cls_preds)
                    if self.adapt_status:
                        status_idx = -1
                        ego_speed = ego_output['status'][-1][i, 0, 0].item()
                        for idx, (start, end) in enumerate(speed_dict[speed_type]["speed_areas"]):
                            if ego_speed >= start and ego_speed < end:
                                status_idx = idx
                        if status_idx != -1:
                            speed_idx = status_idx

                    output["plan_speed_{}".format(speed_type)] = speed_reg_preds[speed_idx].cpu()

            outputs.append(output)
        return outputs

    def select(self, ego_output, det_output, motion_output, cls_preds, reg_preds, data):
        if len(det_output["prediction"]) > 0:
            det_classification = det_output["classification"][-1].sigmoid()
            det_anchors = det_output["prediction"][-1]
            det_confidence = det_classification.max(dim=-1).values

        if len(motion_output["prediction"]) > 0:
            motion_cls = motion_output["classification"][-1].sigmoid()
            motion_reg = motion_output["prediction"][-1].cumsum(-2)

        # cmd select
        bs, fut_cmd, fut_mode = cls_preds[0].shape
        bs_indices = torch.arange(bs, device=cls_preds[0].device)

        # step1. select command
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1) if fut_cmd > 1 else 0
        for i in range(self.num_group):
            cls_preds[i] = cls_preds[i][bs_indices, cmd]
            reg_preds[i] = reg_preds[i][bs_indices, cmd]

        # rescore
        if self.with_rescore and len(det_output["prediction"]) > 0 and len(motion_output["prediction"]) > 0:
            for anchor_type in self.anchor_types:
                if anchor_type == ("temp", "2hz"):
                    idx = self.anchor_types.index(anchor_type)
                    ego_fut_mode = reg_preds[idx].shape[1]
                    cls_preds[idx], _ = self.rescore(
                        cls_preds[idx], reg_preds[idx], motion_cls, motion_reg, det_anchors, det_confidence, ego_fut_mode=ego_fut_mode)

        # todo notice: temp & spat plan share the same pred cls score
        # todo while speed plan share the same cls score in the same speed areas
        # todo which means that they share the rescored cls score (including self.anchor_refer)

        # 수정(2026-10-09): 정성 시각화(계획 모드별 확률)용 — 명령으로 고른 묶음의 전 모드 궤적과 점수를 보관.
        #   선택·추론 결과는 바꾸지 않는다(복사본만 저장). 확률 = 그 명령 묶음 안 logit 의 softmax(rescore 반영 후).
        _ref = self.anchor_types.index(self.anchor_refer)
        self._vis_modes = (reg_preds[_ref].detach(), cls_preds[_ref].detach())

        # step2. select modality
        mode_idx = self.anchor_types.index(self.anchor_refer)
        mode_idx = cls_preds[mode_idx].argmax(dim=-1)
        for i, anchor_type in enumerate(self.anchor_types):
            cls_preds[i] = cls_preds[i][bs_indices, mode_idx]
            reg_preds[i] = reg_preds[i][bs_indices, mode_idx]

        return cls_preds, reg_preds

    def rescore(self,
                plan_cls,
                plan_reg,
                motion_cls,
                motion_reg,
                det_anchors,
                det_confidence,
                score_thresh=0.15,
                static_dis_thresh=0.5,
                dim_scale=1.1,
                num_motion_mode=1,
                offset=None,  # 수정(2026-11-06): None → self.rescore_ego_offset(생성자 인자). #이전 offset=0.5
                ego_fut_ts=None,
                ego_fut_mode=None
    ):
        if offset is None:  # 수정(2026-11-06): 오프셋 기본값을 config 로 받는다
            offset = self.rescore_ego_offset  # 수정(2026-11-06): 생성자 인자 rescore_ego_offset 값을 쓴다
        if ego_fut_ts is None:
            ego_fut_ts = self.ego_fut_ts
        if ego_fut_mode is None:
            ego_fut_mode = self.ego_fut_mode
        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            traj_cat = torch.cat([zeros, traj], dim=-2)
            return traj_cat

        def get_yaw(traj, start_yaw=np.pi / 2):
            yaw = traj.new_zeros(traj.shape[:-1])
            yaw[..., 1:-1] = torch.atan2(
                traj[..., 2:, 1] - traj[..., :-2, 1],
                traj[..., 2:, 0] - traj[..., :-2, 0],
            )
            yaw[..., -1] = torch.atan2(
                traj[..., -1, 1] - traj[..., -2, 1],
                traj[..., -1, 0] - traj[..., -2, 0],
            )
            yaw[..., 0] = start_yaw
            # for static object, estimated future yaw would be unstable
            start = traj[..., 0, :]
            end = traj[..., -1, :]
            dist = torch.linalg.norm(end - start, dim=-1)
            mask = dist < static_dis_thresh
            start_yaw = yaw[..., 0].unsqueeze(-1)
            yaw = torch.where(
                mask.unsqueeze(-1),
                start_yaw,
                yaw,
            )
            return yaw.unsqueeze(-1)

        ## ego
        bs = plan_reg.shape[0]
        plan_reg_cat = cat_with_zero(plan_reg)
        ego_box = det_anchors.new_zeros(bs, ego_fut_mode, ego_fut_ts + 1, 7)
        ego_box[..., [X, Y]] = plan_reg_cat
        ego_box[..., [W, L, H]] = ego_box.new_tensor(self.ego_size) * dim_scale
        ego_box[..., [YAW]] = get_yaw(plan_reg_cat, self.ego_start_yaw)  # 수정(2026-11-06): 전진축 규약을 config 로(NAVSIM +x 전진 → 0.0). #이전 get_yaw(plan_reg_cat) = π/2 고정

        ## motion
        motion_reg = motion_reg[..., :ego_fut_ts, :]
        motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None, None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, 1, ego_fut_ts + 1, 2)
        motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)

        motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
        motion_box[..., [X, Y]] = motion_reg
        motion_box[..., [W, L, H]] = det_anchors[..., None, None, [W, L, H]].clamp(max=5.0).exp()  # 수정: exp overflow(inf) 원천차단
        box_yaw = torch.atan2(
            det_anchors[..., SIN_YAW],
            det_anchors[..., COS_YAW],
        )
        motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))

        filter_mask = det_confidence < score_thresh
        motion_box[filter_mask] = 1e6

        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]

        bs, num_ego_mode, ts, _ = ego_box.shape
        bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape
        ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode, 1, 1, 1).flatten(0, -2)
        motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1, 1).flatten(0, -2)

        ego_box[:, X] += offset * torch.cos(ego_box[:, YAW])  # 수정(2026-11-06): flatten 뒤 (N,7) 이라 열 색인. #이전 ego_box[0] += ...(0 번 행 전체를 바꿔 오프셋 미적용·행 0 파손)
        ego_box[:, Y] += offset * torch.sin(ego_box[:, YAW])  # 수정(2026-11-06): 같은 이유. #이전 ego_box[1] += ...(1 번 행 파손)
        col = check_collision(ego_box, motion_box)
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4)
        col = col.flatten(2, -1).any(dim=-1)
        all_col = col.all(dim=-1)
        col[all_col] = False  # for case that all modes collide, no need to rescore
        score_offset = col.float() * -999
        plan_cls = plan_cls + score_offset

        return plan_cls, all_col

    def rescore_speed(self, speed_dict, det_output, motion_output):
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values

        motion_cls = motion_output["classification"][-1].sigmoid()
        motion_reg = motion_output["prediction"][-1].cumsum(-2)

        if "2hz" in speed_dict and self.speed_refer[1] == '2hz':
            speed_cls = speed_dict["2hz"]["cls_preds"].permute(1, 0)
            speed_reg = speed_dict["2hz"]["reg_preds"].unsqueeze(0)
            rescore_plan_reg = speed_reg
            rescore_motion_reg = motion_reg
        elif "5hz" in speed_dict and self.speed_refer[1] == '5hz':
            speed_cls = speed_dict["5hz"]["cls_preds"].permute(1, 0)
            speed_reg = speed_dict["5hz"]["reg_preds"].unsqueeze(0)
            rescore_plan_reg = speed_reg[:, :, [2, 5]]  # approxmite
            rescore_motion_reg = motion_reg[:, :, :, [0, 1]]
        else:
            raise NotImplementedError

        speed_cls, all_col = self.rescore(speed_cls, rescore_plan_reg, motion_cls, rescore_motion_reg,
                                          det_anchors, det_confidence, \
                                          ego_fut_ts=rescore_plan_reg.shape[2],
                                          ego_fut_mode=rescore_plan_reg.shape[1])

        for speed_type in speed_dict:
            speed_dict[speed_type]["cls_preds"] = speed_cls.permute(1, 0)
            speed_dict[speed_type]["reg_preds"] = speed_dict[speed_type]["reg_preds"] * (1 - all_col.float())

        return speed_dict