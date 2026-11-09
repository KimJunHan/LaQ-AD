# 수정(2026-10-15): 경로 이동 mmdet3d_plugin/datasets/navsim_dataset.py → models/loader/navsim/dataset.py.
"""수정(2026-10-09): LaQ-AD NAVSIM 데이터셋 (navtrain 학습 / navtest 추론).

입력: tools/navsim/build_navsim_infos.py 산출 info pkl (+ navsim_devkit_steps.py 의 egometa/lidar/map 덧붙임).
nuScenes 용 NuScenes3DDataset(구 NuScenesMiniDataset) 과 같은 키를 내도록 상속한다(파이프라인·헤드 공유). 다른 점:
  - 카메라 8대(CAM_F0, L0, L1, L2, R0, R1, R2, B0), 투영 = K @ inv(sensor2lidar) — 라이다 프레임 모델.
  - ego_status = ego_dynamic_state 4차원(vx, vy, ax, ay), driving_command(4, 좌/직/우/미상 one-hot)를
    gt_ego_fut_cmd 로 넘긴다 — NAVSIM 의 명령은 경로만 반영한 "입력"이다(docs/agents.md:79). nuScenes 처럼 미래 궤적에서
    만든 명령이 아니므로 명령 임베딩 입력으로 써도 정답 누설이 아니다. '미상' 명령 표본도 버리지 않는다(no-skip).
  - 계획 정답 8 포즈(4 s @ 2 Hz) 증분, 라이다 프레임(rear-axle 경로).
  - evaluate(): 평가 프레임(장면 현재 프레임)의 계획을 ego rear-axle 국소좌표 (8,3) 로 바꿔 {navsim_token: poses} pkl 저장.
    EPDMS 채점은 devkit env 에서 tools/navsim/run_navsim_eval.sh 가 한다.
조용한 대체 없음: 필수 키·메타데이터·영상 크기가 config 와 다르면 즉시 실패한다.
"""
import os
import os.path as osp
import pickle

import numpy as np
from mmdet.datasets import DATASETS

from ..nuscenes.dataset import NuScenes3DDataset

NAVSIM_CAMS = ("CAM_L0", "CAM_F0", "CAM_R0")  # 수정(2026-10-09): SparseDriveV2 NAVSIM 설정(사용자 결정) — 전방 3캠 l0·f0·r0, 1프레임 (swc-17/SparseDriveV2 sparsedrive_config.py:63)
# 이전(8캠): NAVSIM_CAMS = ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0")
# 물체 이름: navsim/planning/scenario_builder/navsim_scenario_utils.py:22-31 (ego 제외)
NAVSIM_DET_CLASSES = ("vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier", "czone_sign", "generic_object")


def plan_lidar_to_ego_poses(plan_xy_lidar, lidar2ego, min_step=0.05):
    """계획 (T,2) 라이다 프레임 누적 변위(현재 rear-axle 기준) → NAVSIM Trajectory (T,3) ego 국소좌표.

    학습 정답이 rear-axle 위치들의 라이다 프레임 증분이므로(build_navsim_infos.frame_info), cumsum 은 현재
    rear-axle 로부터의 변위다. ego 프레임으로는 회전만 적용하면 된다(원점 = 현재 rear-axle).
    heading: 인접 위치 차분의 방향. 이동이 min_step 미만이면 직전 heading 을 유지한다(정지 시 각도 불안정 방지,
    첫 스텝 직전 heading = 0 = 현재 자차 방향). 이 규칙은 결정적이며 표본을 버리지 않는다.
    """
    p = np.asarray(plan_xy_lidar, dtype=np.float64)
    R = np.asarray(lidar2ego, dtype=np.float64)[:3, :3]
    xyz = np.concatenate([p, np.zeros((len(p), 1))], 1) @ R.T
    xy = xyz[:, :2]
    out = np.zeros((len(xy), 3), np.float32)
    prev_xy, prev_h = np.zeros(2), 0.0
    for t in range(len(xy)):
        d = xy[t] - prev_xy
        h = float(np.arctan2(d[1], d[0])) if np.hypot(*d) >= min_step else prev_h
        out[t] = (xy[t, 0], xy[t, 1], h)
        prev_xy, prev_h = xy[t], h
    return out


def test_mode_of(kwargs):
    return bool(kwargs.get("test_mode", False))


@DATASETS.register_module()
class NavsimDataset(NuScenes3DDataset):
    def __init__(self, *args, det_classes=NAVSIM_DET_CLASSES, map_classes=None,
                 ego_anchor=None, ego_size=None, image_hw=None, plan_key="plan_temp_2hz", map_roi=None,
                 num_frames=None,
                 require_map=True, require_lidar_bin=True, **kwargs):
        super().__init__(*args, det_classes=det_classes, map_classes=map_classes, with_velocity=True, **kwargs)
        self.plan_key = plan_key
        self.image_hw = tuple(image_hw) if image_hw is not None else None
        self._image_checked = False
        meta = self.metadata
        if not meta or "kind" not in meta:
            raise RuntimeError(f"{self.ann_file}: build_navsim_infos.py 산출물이 아니다(metadata 없음).")
        # config ↔ 변환 산출물 일치 검사(egometa 가 계산한 자차 앵커·크기, map 클래스)
        if ego_anchor is not None:
            if "ego_anchor" not in meta:
                raise RuntimeError("info 에 ego_anchor 가 없다 — navsim_devkit_steps.py egometa 를 먼저 실행할 것.")
            if not np.allclose(meta["ego_anchor"], ego_anchor, atol=1e-4):
                raise ValueError(f"config ego_anchor {ego_anchor} != 변환기 값 {meta['ego_anchor']}")
        if ego_size is not None and not np.allclose(meta.get("ego_size", [np.nan] * 3), ego_size, atol=1e-4):
            raise ValueError(f"config ego_size {ego_size} != 변환기 값 {meta.get('ego_size')}")
        if num_frames is not None and not test_mode_of(kwargs) and len(self.data_infos) != num_frames:
            raise ValueError(f"info 프레임 수 {len(self.data_infos)} != config navsim_train_frames {num_frames} "
                             "(epoch 길이 계산이 틀린다 — config 값을 변환기 출력으로 교체)")
        if require_map:
            if map_roi is not None and tuple(meta.get("map_roi", ())) != tuple(float(x) for x in map_roi):
                raise ValueError(f"config map_roi_size {map_roi} != 지도 GT 생성 ROI {meta.get('map_roi')}")
            if tuple(meta.get("map_classes", ())) != tuple(map_classes or ()):
                raise ValueError(f"config map_classes {map_classes} != 변환기 {meta.get('map_classes')} "
                                 "(navsim_devkit_steps.py map 미실행이면 먼저 실행)")
        self.require_map = require_map
        self.require_lidar_bin = require_lidar_bin

    # ------------------------------------------------------------------ data
    def _check_image_size(self, path):
        if self._image_checked or self.image_hw is None:
            return
        from PIL import Image
        W, H = Image.open(path).size
        if (H, W) != self.image_hw:
            raise ValueError(f"NAVSIM 영상 크기 {(H, W)} != config image_hw {self.image_hw} (data_aug_conf H/W 수정 필요)")
        self._image_checked = True

    def get_data_info(self, index):
        info = self.data_infos[index]
        l2e = np.asarray(info["lidar2ego"], np.float64)
        e2g = np.asarray(info["ego2global"], np.float64)
        lidar2global = e2g @ l2e
        input_dict = dict(
            token=info["token"], frame_token=info["token"], scene_token=info["scene_token"],
            timestamp=info["timestamp"] / 1e6, lidar2global=lidar2global, lidar2ego=l2e,
            pts_filename=info["lidar_path"],
        )
        if self.require_lidar_bin and not str(info["lidar_path"]).endswith(".bin"):
            raise RuntimeError(f"{info['lidar_path']}: .pcd 그대로다 — navsim_devkit_steps.py lidar 를 먼저 실행할 것"
                               "(보조 깊이 손실의 LoadPointsFromFile 은 (n,5) float32 .bin 을 읽는다).")
        if self.require_map:
            if "map_annos" not in info:
                raise RuntimeError(f"{info['token']}: map_annos 없음 — navsim_devkit_steps.py map 미실행")
            input_dict["map_infos"] = info["map_annos"]
            input_dict["map_geoms"] = self.anno2geom(info["map_annos"])
            input_dict["map_location"] = info["map_location"]
        es = np.asarray(info["ego_status"], np.float32)
        valid = np.isfinite(es)
        input_dict["ego_status"] = np.where(valid, es, 0.0).astype(np.float32)
        input_dict["ego_status_mask"] = valid.astype(np.float32)

        imgs, l2i, l2c, e2i, Ks = [], [], [], [], []
        for cam in NAVSIM_CAMS:
            c = info["cams"][cam]
            s2l = np.eye(4)
            s2l[:3, :3] = c["sensor2lidar_rotation"]
            s2l[:3, 3] = c["sensor2lidar_translation"]
            lidar2cam = np.linalg.inv(s2l)
            K = np.eye(4)
            K[:3, :3] = c["cam_intrinsic"]
            imgs.append(c["data_path"])
            l2i.append(K @ lidar2cam)
            l2c.append(lidar2cam.T)                       # NuScenes3DDataset 과 같은 transpose 규약
            e2i.append(K @ np.linalg.inv(l2e @ s2l))
            Ks.append(K)
        self._check_image_size(imgs[0])
        input_dict.update(img_filename=imgs, camera_names=list(NAVSIM_CAMS), lidar2img=l2i,
                          lidar2cam=l2c, ego2img=e2i, cam_intrinsic=Ks)
        input_dict.update(self.get_ann_info(index))
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        boxes = np.asarray(info["gt_boxes"], np.float32).reshape(-1, 7)
        vel = np.asarray(info["gt_velocity"], np.float32).reshape(-1, 2)
        if not np.isfinite(vel).all():
            raise ValueError(f"{info['token']}: gt_velocity 비유한값 — 변환 단계에서 원인 확인 필요")
        names = np.asarray(info["gt_names"])
        labels = np.array([self.det_classes.index(n) if n in self.det_classes else -1 for n in names], np.int64)
        res = dict(
            gt_names=names, gt_labels_3d=labels,
            gt_bboxes_3d=np.concatenate([boxes, vel], 1).astype(np.float32),
            instance_inds=np.asarray(info["instance_inds"], np.int32),
            gt_agent_fut_trajs=np.asarray(info["gt_agent_fut_trajs"], np.float32),
            gt_agent_fut_masks=np.asarray(info["gt_agent_fut_masks"], np.float32),
            gt_ego_fut_trajs=np.asarray(info["gt_ego_fut_trajs"], np.float32),
            gt_ego_fut_masks=np.asarray(info["gt_ego_fut_masks"], np.float32),
            gt_ego_fut_cmd=np.asarray(info["driving_command"], np.float32),
        )
        # 계획 손실은 앵커 유형 주파수 접미사 키를 읽는다(decoder/unified_decoder.get_gt_trajs) — ("temp","2hz") 대응.
        res["gt_ego_fut_trajs_2hz"] = res["gt_ego_fut_trajs"]
        res["gt_ego_fut_masks_2hz"] = res["gt_ego_fut_masks"]
        if self._plan_sft is not None:  # 수정(2026-11-06): 사용자 결정 "NAVSIM MCMC SFT" — 학습 전용 SFT 정답(부모 NuScenes3DDataset.get_ann_info 를 부르지 않아 여기서 넣는다; 마스크는 위 2hz 마스크 그대로)
            res["gt_ego_plan_sft_2hz"] = self._plan_sft[info["token"]].copy()
        return res

    # ------------------------------------------------------------------ eval
    def collect_navsim_predictions(self, results):
        """평가 프레임의 계획 → {navsim_token: (8,3) ego 국소 포즈}."""
        if len(results) != len(self.data_infos):
            raise RuntimeError(f"결과 수 {len(results)} != 표본 수 {len(self.data_infos)} (추론 누락)")
        preds = {}
        for info, res in zip(self.data_infos, results):
            if not info.get("is_eval_frame"):
                continue
            r = res["img_bbox"] if "img_bbox" in res else res
            if self.plan_key not in r:
                raise KeyError(f"결과에 {self.plan_key} 가 없다(키: {sorted(r.keys())}) — config plan_anchor_types 확인")
            xy = np.asarray(r[self.plan_key], np.float32)
            if xy.shape != (8, 2) or not np.isfinite(xy).all():
                raise ValueError(f"{info['navsim_token']}: 계획 형식 오류 {xy.shape}")
            preds[info["navsim_token"]] = plan_lidar_to_ego_poses(xy, info["lidar2ego"])
        expected = self.metadata.get("eval_tokens")
        if expected is not None and set(preds) != set(expected):
            raise RuntimeError(f"평가 token 불일치: 누락 {len(set(expected) - set(preds))}, "
                               f"초과 {len(set(preds) - set(expected))}")
        return preds

    def evaluate(self, results, logger=None, jsonfile_prefix=None, **kwargs):
        out_dir = jsonfile_prefix or "evaluation/navsim"
        os.makedirs(out_dir, exist_ok=True)
        preds = self.collect_navsim_predictions(results)
        path = osp.join(out_dir, "navsim_pred.pkl")
        with open(path + ".tmp", "wb") as f:
            pickle.dump(preds, f)
        os.replace(path + ".tmp", path)
        print(f"[navsim] {len(preds)} 장면 계획 저장 -> {path}\n"
              f"  EPDMS 채점(devkit env): bash tools/navsim/run_navsim_eval.sh {path} <out_dir>")
        return {"navsim/num_scenes": float(len(preds))}
