#수정: VLA-v0 — nuScenes-mini 경량 데이터셋. det 만 (map/motion/plan/track 비활성).
#  기존 data/nuscenes/nuscenes_infos_{train,val}.pkl (mmdet3d preprocessing 출력) 을 직접 소비.
#  스키마: {token, cams:{CAM_X:{data_path,sensor2lidar_*,cam_intrinsic,...}}, gt_boxes(Nx7), gt_names(N,),
#          lidar2ego_*, ego2global_*, num_lidar_pts/num_radar_pts, valid_flag}.
import copy
import math
import os.path as osp
import pickle

import numpy as np
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from pyquaternion import Quaternion
from shapely.geometry import LineString  #수정: Path B — map anno2geom
from torch.utils.data import Dataset


# nuScenes mmdet3d 표준 10-class
NUSC_DET_CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)


def _compose_rt(R, t):
    """3x3 R + 3-vec t → 4x4 homogeneous."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _quat_to_rot(q_wxyz):
    return Quaternion(q_wxyz).rotation_matrix


@DATASETS.register_module()
class NuScenesMiniDataset(Dataset):
    """nuScenes-mini 용 minimal Dataset.
    Loaders pipeline (LoadMultiViewImageFromFiles, ResizeCropFlipImage 등) 와 호환되도록
    img_filename, lidar2img, cam_intrinsic, gt_bboxes_3d, gt_labels_3d 등을 제공.
    """

    def __init__(
        self,
        data_root,
        ann_file,
        pipeline=None,
        modality=None,
        test_mode=False,
        det_classes=NUSC_DET_CLASSES,
        with_velocity=True,
        filter_empty_gt=True,
        point_cloud_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
        data_aug_conf=None,
        split_group=0,
        # 미사용 키들 — config 호환용 placeholder
        map_file=None, map_classes=None, eval_cfg=None,
        name_mapping=None, sample_rate=1, past_frames=2,
        future_frames=6, spatial_points=6, plan_anchor_types=None,
        with_next_target_point=False, with_seq_flag=False,
        sequences_split_num=1, keep_consistent_seq_aug=True,
        remap_box=False, align_static_yaw=False,
        map_num_pts=20, with_connect_lane=False,
        # mmdet build_dataset 가 자동 주입하는 kwargs 흡수
        work_dir=None,
        **kwargs,
    ):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.modality = modality or dict(use_camera=True, use_lidar=False)
        self.test_mode = test_mode
        self.det_classes = list(det_classes)
        self.with_velocity = with_velocity
        self.filter_empty_gt = filter_empty_gt
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.data_aug_conf = data_aug_conf
        #수정(2026-08-03): map_classes 를 실제로 보관한다. 기존엔 "미사용 placeholder" 로 받기만 하고
        #  버려서 map 평가기(VectorEvaluate)가 요구하는 dataset.MAP_CLASSES 가 존재하지 않았다.
        #  HiP-AD Table 4 의 'map mAP' 산출에 필수.
        self.MAP_CLASSES = list(map_classes) if map_classes is not None else None
        self.map_classes = self.MAP_CLASSES

        # sampler 가 읽는 attr (B2D 와 호환)
        self.split_group = split_group
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        self.with_seq_flag = with_seq_flag

        with open(ann_file, "rb") as f:
            data = pickle.load(f)
        self.data_infos = data["infos"] if isinstance(data, dict) else data
        self.metadata = data.get("metadata", {}) if isinstance(data, dict) else {}

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = None

        #수정: VLA-FULL — flag 를 all-zeros(=1 group) 대신 scene 단위 sequence group 으로 설정.
        #  GroupInBatchSampler 가 groups_num >= batch_size×world_size 를 요구 → all-zeros 면 batch>1 불가.
        #  (mini 는 batch=1 이라 all-zeros 로도 돌았음. full 학습은 batch>1 필요 → scene 그룹핑.
        #   ON/OFF 동일 적용되어 ablation 비교성 유지.) B2D 의 _set_sequence_group_flag 포팅(folder→scene_token).
        if self.with_seq_flag:
            self._set_sequence_group_flag()
        else:
            self.flag = np.zeros(len(self.data_infos), dtype=np.int64)

    def _set_sequence_group_flag(self):
        """scene 단위로 sequence 그룹, 각 sequence 를 sequences_split_num 으로 분할."""
        if self.sequences_split_num == -1:
            self.flag = np.arange(len(self.data_infos))
            return

        res = []
        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and self.data_infos[idx]["scene_token"] != self.data_infos[idx - 1]["scene_token"]:
                curr_sequence += 1
            res.append(curr_sequence)
        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, bin_counts[curr_flag],
                                   math.ceil(bin_counts[curr_flag] / self.sequences_split_num)))
                        + [bin_counts[curr_flag]]
                    )
                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1
                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.sequences_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    def __len__(self):
        return len(self.data_infos)

    # ---------------- core data flow ----------------
    def get_data_info(self, index):
        info = self.data_infos[index]
        token = info["token"]

        # lidar → ego
        lidar2ego = _compose_rt(_quat_to_rot(info["lidar2ego_rotation"]),
                                np.asarray(info["lidar2ego_translation"]))
        ego2global = _compose_rt(_quat_to_rot(info["ego2global_rotation"]),
                                 np.asarray(info["ego2global_translation"]))
        lidar2global = ego2global @ lidar2ego

        input_dict = dict(
            token=token,
            frame_token=token,
            scene_token=info.get("scene_token", token),
            timestamp=info["timestamp"] / 1e6,  # nuScenes timestamp 는 microsecond
            pts_filename=info.get("lidar_path"),
            lidar2global=lidar2global,
        )
        #수정: Path B — map/ego GT 첨부. map_geoms(anno2geom)는 VectorizeMap transform 이 vector 로 변환.
        if "map_annos" in info:
            input_dict["map_infos"] = info["map_annos"]
            input_dict["map_geoms"] = self.anno2geom(info["map_annos"])
            input_dict["map_location"] = info.get("map_location")
        if "ego_status" in info:
            es = np.asarray(info["ego_status"], np.float32)
            input_dict["ego_status"] = es
            #수정: Path B — ego_status_mask (유효성 마스크, nan/0 아닌 항목=1). ego head 가 요구.
            input_dict["ego_status_mask"] = (np.isfinite(es) & (es != 0)).astype(np.float32)

        if self.modality.get("use_camera", True):
            image_paths, camera_names = [], []
            lidar2img_rts, lidar2cam_rts, ego2img_rts, cam_intrinsics = [], [], [], []
            for cam_name, cam in info["cams"].items():
                # pkl 의 data_path 가 다른 mmdet3d setup 기준 (예: 'mmdetection3d/data/nuscenes/samples/...').
                # nuScenes 표준 sub-tree 인 'samples/...' 또는 'sweeps/...' 만 추출해 data_root 와 join.
                p = cam["data_path"]
                if "samples/" in p:
                    p = "samples/" + p.split("samples/", 1)[1]
                elif "sweeps/" in p:
                    p = "sweeps/" + p.split("sweeps/", 1)[1]
                if not osp.isabs(p) and self.data_root:
                    p = osp.join(self.data_root, p)
                image_paths.append(p)
                camera_names.append(cam_name)

                # cam → lidar (mmdet3d preprocessor 출력. R 은 R.T 형태로 저장됨에 주의)
                # nuScenes converter 의 sensor2lidar 는: lidar_pt = R @ cam_pt + t.
                cam2lidar = _compose_rt(np.asarray(cam["sensor2lidar_rotation"]),
                                        np.asarray(cam["sensor2lidar_translation"]))
                lidar2cam = np.linalg.inv(cam2lidar)

                intrinsic = np.asarray(cam["cam_intrinsic"])
                intrinsic_pad = np.eye(4)
                intrinsic_pad[:3, :3] = intrinsic

                lidar2img = intrinsic_pad @ lidar2cam
                cam2ego = lidar2ego @ cam2lidar     # ego ← lidar ← cam
                ego2cam = np.linalg.inv(cam2ego)
                ego2img = intrinsic_pad @ ego2cam

                lidar2cam_rts.append(lidar2cam.T)  # B2D pipeline 호환 위해 transpose
                lidar2img_rts.append(lidar2img)
                ego2img_rts.append(ego2img)
                cam_intrinsics.append(intrinsic_pad)

            input_dict.update(dict(
                img_filename=image_paths,
                camera_names=camera_names,
                lidar2img=lidar2img_rts,
                lidar2cam=lidar2cam_rts,
                ego2img=ego2img_rts,
                cam_intrinsic=cam_intrinsics,
            ))

        annos = self.get_ann_info(index)
        input_dict.update(annos)
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        gt_boxes = info["gt_boxes"].copy() if hasattr(info["gt_boxes"], "copy") else np.array(info["gt_boxes"])
        gt_names = info["gt_names"]
        # velocity append: nuScenes 표준은 (cx,cy,cz,w,l,h,yaw) + (vx,vy) → 9D
        if self.with_velocity and gt_boxes.shape[1] == 7:
            vel = np.asarray(info.get("gt_velocity", np.zeros((gt_boxes.shape[0], 2), np.float32)),
                             dtype=np.float32).copy()
            #수정: Path B — velocity nan → 0 sanitize (nuScenes 속도 미정의 객체). mAVE 개선 + 손실 안정.
            vel[np.isnan(vel).any(axis=1)] = 0.0
            gt_boxes = np.concatenate([gt_boxes, vel], axis=1)
        # valid_flag: 적어도 1 lidar/radar pt 이상
        if "valid_flag" in info:
            mask = info["valid_flag"]
        elif "num_lidar_pts" in info:
            mask = info["num_lidar_pts"] > 0
        else:
            mask = np.ones(len(gt_names), dtype=bool)
        gt_boxes = gt_boxes[mask]
        gt_names = np.asarray(gt_names)[mask]

        gt_labels = []
        for n in gt_names:
            gt_labels.append(self.det_classes.index(n) if n in self.det_classes else -1)
        gt_labels = np.array(gt_labels, dtype=np.int64)

        #수정: Path B — instance_inds 는 info 의 실제 tracking id 사용(없으면 arange). motion/track 용.
        if "instance_inds" in info:
            inst = np.asarray(info["instance_inds"], dtype=np.int32)[mask]
        else:
            inst = np.arange(int(mask.sum()), dtype=np.int32)

        result = dict(
            gt_names=gt_names,
            gt_labels_3d=gt_labels,
            gt_bboxes_3d=gt_boxes.astype(np.float32),
            instance_inds=inst,
        )
        #수정: Path B — motion GT (agent 미래 궤적)
        if "gt_agent_fut_trajs" in info:
            result["gt_agent_fut_trajs"] = np.asarray(info["gt_agent_fut_trajs"], np.float32)[mask]
            result["gt_agent_fut_masks"] = np.asarray(info["gt_agent_fut_masks"], np.float32)[mask]
        #수정: Path B — plan/ego GT (+ planning collision eval 용 fut_boxes)
        if "gt_ego_fut_trajs" in info:
            result["gt_ego_fut_trajs"] = np.asarray(info["gt_ego_fut_trajs"], np.float32)
            result["gt_ego_fut_masks"] = np.asarray(info["gt_ego_fut_masks"], np.float32)
            result["gt_ego_fut_cmd"] = np.asarray(info["gt_ego_fut_cmd"], np.float32)
            result["fut_boxes"] = self._get_fut_boxes(index, info)
            #수정(2026-08-11): 계획 손실은 앵커 유형에서 파생된 주파수 접미사 키를 읽는다
            #   (sparse_onedecoder.get_gt_trajs: 'gt_ego_fut_trajs_{freq}').
            #   B2D dataset 은 여러 주파수를 만들며 그 중 하나를 접미사 없는 키로도 둔다.
            #   nuScenes 자차 GT 는 2Hz x 6스텝(3초) 하나뿐이므로 2hz 키로 별칭을 만든다.
            #   plan_anchor_types = [("temp","2hz")] 와 대응한다.
            result["gt_ego_fut_trajs_2hz"] = result["gt_ego_fut_trajs"]
            result["gt_ego_fut_masks_2hz"] = result["gt_ego_fut_masks"]
        return result

    @staticmethod
    def anno2geom(annos):
        """map_annos({label: [np.array(line)]}) → {label: [LineString]} (VectorizeMap 입력)."""
        out = {}
        for label, anno_list in annos.items():
            out[label] = [LineString(a) for a in anno_list]
        return out

    @staticmethod
    def _T_global(info):
        """lidar2global 4x4."""
        l2e = _compose_rt(_quat_to_rot(info["lidar2ego_rotation"]),
                          np.asarray(info["lidar2ego_translation"]))
        e2g = _compose_rt(_quat_to_rot(info["ego2global_rotation"]),
                          np.asarray(info["ego2global_translation"]))
        return e2g @ l2e

    def _get_fut_boxes(self, index, info):
        """planning collision eval 용 — 미래 frame 박스를 현재 lidar frame 으로 변환 (SparseDrive 동일)."""
        fut_ts = int(np.asarray(info["gt_ego_fut_masks"]).sum())
        fut_boxes = []
        cur_tok = info["scene_token"]
        cur_T = self._T_global(info)
        for i in range(1, fut_ts + 1):
            if index + i >= len(self.data_infos):
                break
            fi = self.data_infos[index + i]
            if fi["scene_token"] != cur_tok:
                break
            m = fi["valid_flag"] if "valid_flag" in fi else (fi["num_lidar_pts"] > 0)
            fb = np.asarray(fi["gt_boxes"], np.float32)[m].copy()
            if len(fb) == 0:
                fut_boxes.append(fb); continue
            T = np.linalg.inv(cur_T) @ self._T_global(fi)
            fb[:, :3] = fb[:, :3] @ T[:3, :3].T + T[:3, 3]
            yaw = np.stack([np.cos(fb[:, 6]), np.sin(fb[:, 6])], -1) @ T[:2, :2].T
            fb[:, 6] = np.arctan2(yaw[..., 1], yaw[..., 0])
            fut_boxes.append(fb)
        return fut_boxes

    # ---------------- augmentation (B2D 동일 패턴) ----------------
    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_aug_conf.get("rand_flip", False) and bool(np.random.choice([0, 1]))
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self.data_aug_conf.get("rot3d_range", [0, 0]))
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0.0
            rotate_3d = 0.0
        return dict(resize=resize, resize_dims=resize_dims, crop=crop, flip=flip,
                    rotate=rotate, rotate_3d=rotate_3d)

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        data = self.get_data_info(idx)
        data["aug_config"] = aug_config
        if self.pipeline is not None:
            data = self.pipeline(data)
        return data

    # ---------------- evaluate (nuScenes-devkit 정식) ----------------
    # nuScenes detection 클래스별 default attribute (DefaultAttribute)
    DEFAULT_ATTR = {
        "car": "vehicle.parked", "truck": "vehicle.parked",
        "construction_vehicle": "vehicle.parked", "bus": "vehicle.parked",
        "trailer": "vehicle.parked", "barrier": "",
        "motorcycle": "cycle.without_rider", "bicycle": "cycle.without_rider",
        "pedestrian": "pedestrian.standing", "traffic_cone": "",
    }

    def _attr_from_velocity(self, cls_name, velocity):
        """예측 속도로 nuScenes attribute 를 결정한다.

        수정(2026-08-10): 기존엔 DEFAULT_ATTR 고정값만 넣어 **모든 차량을 항상 'vehicle.parked'**
          로 제출했다. 그 결과 mAAE 0.4343(통상 ~0.19)으로 NDS 가 부당하게 깎였다.
          우리 mAVE 는 0.266 으로 속도 예측 자체는 멀쩡한데 그 정보를 속성에 반영하지 않고
          버리고 있었던 것 — 모델이 아니라 제출 변환의 문제다.
          SparseDrive nuscenes_3d_dataset 의 속성 결정 규칙을 그대로 따른다(임계 0.2 m/s).
        """
        import numpy as np

        speed = float(np.sqrt(velocity[0] ** 2 + velocity[1] ** 2))
        if speed > 0.2:
            if cls_name in ("car", "construction_vehicle", "bus", "truck", "trailer"):
                return "vehicle.moving"
            if cls_name in ("bicycle", "motorcycle"):
                return "cycle.with_rider"
            return self.DEFAULT_ATTR.get(cls_name, "")
        if cls_name == "pedestrian":
            return "pedestrian.standing"
        if cls_name == "bus":
            return "vehicle.stopped"
        return self.DEFAULT_ATTR.get(cls_name, "")

    def _ego_box_to_global(self, box, info):
        """box: [x,y,z,w,l,h,yaw,vx,vy] in lidar frame → nuScenes global [translation, size, rotation_quat, velocity]."""
        # lidar → ego
        lidar2ego_R = _quat_to_rot(info["lidar2ego_rotation"])
        lidar2ego_t = np.asarray(info["lidar2ego_translation"])
        # ego → global
        ego2global_R = _quat_to_rot(info["ego2global_rotation"])
        ego2global_t = np.asarray(info["ego2global_translation"])

        center_lidar = np.asarray(box[:3])
        # composition: global = ego2global_R @ (lidar2ego_R @ center_lidar + lidar2ego_t) + ego2global_t
        center_ego = lidar2ego_R @ center_lidar + lidar2ego_t
        center_global = ego2global_R @ center_ego + ego2global_t

        # rotation: yaw in lidar → rotation quat in global
        #수정(2026-08-10): l 과 w 가 뒤바뀌어 나가고 있었다.
        #  컨버터(tools/data_converter/nuscenes_converter.py:311)가
        #  `gt_boxes = concat([locs, dims[:, [1,0,2]], rots])` 로 GT 를 **(l, w, h)** 순서로 저장한다
        #  (dims = nuScenes box.wlh 이므로 [1,0,2] 재배열 = w,l 교환). 모델도 그 순서로 학습된다.
        #  반면 nuScenes 제출 규격의 'size' 는 **(w, l, h)** 다. 기존 코드는 모델 출력 box[3:6]
        #  = (l, w, h) 를 순서 변경 없이 size 로 넘겨 l/w 가 교환된 채 제출됐다.
        #  실측 확인: info pkl gt_boxes[3:6]=[4.26,1.73,1.49] vs nuScenes ann size=[1.73,4.26,1.49].
        #  증상: mASE 0.704(정상 ~0.27). 길쭉한 클래스만 무너지는 지문 —
        #        barrier 0.889 / bus 0.851 / bicycle 0.813 / car 0.748 vs cone 0.314 / ped 0.345.
        #  (mAP 은 center-distance 매칭이라 영향 없음 → 0.3996 은 유효. NDS 만 과소평가였다.)
        wlh = [float(box[4]), float(box[3]), float(box[5])]   # (l,w,h) → (w,l,h)
        yaw_lidar = float(box[6])
        q_lidar = Quaternion(axis=[0, 0, 1], radians=yaw_lidar)
        q_global = Quaternion(matrix=ego2global_R) * Quaternion(matrix=lidar2ego_R) * q_lidar

        # velocity (vx, vy) in lidar/ego → rotate by global rotation (z-only)
        if len(box) >= 9:
            vel_lidar = np.array([float(box[7]), float(box[8]), 0.0])
            vel_global = ego2global_R @ (lidar2ego_R @ vel_lidar)
            velocity = [float(vel_global[0]), float(vel_global[1])]
        else:
            velocity = [0.0, 0.0]

        return dict(
            translation=[float(c) for c in center_global],
            size=wlh,
            rotation=list(q_global),  # [w, x, y, z]
            velocity=velocity,
        )

    def _to_nusc_submission(self, results, out_path):
        import json, torch
        nusc_results = {}
        for idx, r in enumerate(results):
            preds = r.get("img_bbox") if isinstance(r, dict) else None
            if preds is None:
                continue
            info = self.data_infos[idx]
            token = info["token"]
            boxes = preds.get("boxes_3d")
            scores = preds.get("scores_3d")
            labels = preds.get("labels_3d")
            if torch.is_tensor(boxes):
                boxes = boxes.cpu().numpy()
            if torch.is_tensor(scores):
                scores = scores.cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            sample_list = []
            for b, s, lab in zip(boxes, scores, labels):
                lab = int(lab)
                if not (0 <= lab < len(self.det_classes)):
                    continue
                cls_name = self.det_classes[lab]
                conv = self._ego_box_to_global(b, info)
                sample_list.append(dict(
                    sample_token=token,
                    detection_name=cls_name,
                    detection_score=float(s),
                    attribute_name=self._attr_from_velocity(cls_name, conv["velocity"]),
                    **conv,
                ))
            nusc_results[token] = sample_list
        meta = dict(use_camera=True, use_lidar=False, use_radar=False, use_map=False, use_external=False)
        out = dict(meta=meta, results=nusc_results)
        with open(out_path, "w") as f:
            json.dump(out, f)
        return out_path

    # nuScenes tracking 벤치마크가 제외하는 클래스(정적/비이동 객체) — devkit TRACKING_NAMES 기준
    TRACKING_EXCLUDE = ("barrier", "traffic_cone", "construction_vehicle")

    def format_tracking_results(self, results, out_path, thresh=0.2):
        """detection 출력을 nuScenes tracking 제출 포맷으로 변환한다.

        수정(2026-08-03): SparseDrive _format_bbox(tracking=True) 이식.
          tracking 은 별도 학습 태스크가 아니라, temporal instance bank 가 프레임 간에
          유지하는 instance_ids 를 그대로 track ID 로 쓰는 후처리 평가다
          (sparse_onedecoder.with_instance_id=True → det decoder 가 instance_ids 출력).
          따라서 stage1/stage2 어느 체크포인트로도 산출할 수 있다.
        """
        import json
        import torch

        nusc_results = {}
        missing_ids = 0
        for idx, r in enumerate(results):
            preds = r.get("img_bbox") if isinstance(r, dict) else None
            if preds is None:
                continue
            info = self.data_infos[idx]
            token = info["token"]
            boxes = preds.get("boxes_3d")
            scores = preds.get("scores_3d")
            labels = preds.get("labels_3d")
            ids = preds.get("instance_ids")
            if ids is None:
                missing_ids += 1
                continue
            if torch.is_tensor(boxes):
                boxes = boxes.cpu().numpy()
            if torch.is_tensor(scores):
                scores = scores.cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            if torch.is_tensor(ids):
                ids = ids.cpu().numpy()

            sample_list = []
            for b, s, lab, tid in zip(boxes, scores, labels, ids):
                if thresh is not None and float(s) < thresh:
                    continue
                lab = int(lab)
                if not (0 <= lab < len(self.det_classes)):
                    continue
                cls_name = self.det_classes[lab]
                if cls_name in self.TRACKING_EXCLUDE:
                    continue
                conv = self._ego_box_to_global(b, info)
                sample_list.append(dict(
                    sample_token=token,
                    tracking_name=cls_name,
                    tracking_score=float(s),
                    tracking_id=str(int(tid)),
                    **conv,
                ))
            nusc_results[token] = sample_list

        if missing_ids:
            raise ValueError(
                f"instance_ids 가 없는 샘플 {missing_ids}개 — tracking 평가 불가. "
                "config 의 with_instance_id(sparse_onedecoder)와 task_select 의 'det' 를 확인하라."
            )

        meta = dict(use_camera=True, use_lidar=False, use_radar=False,
                    use_map=False, use_external=False)
        with open(out_path, "w") as f:
            json.dump(dict(meta=meta, results=nusc_results), f)
        return out_path

    def _evaluate_single_tracking(self, result_path, out_dir, logger=None):
        """nuScenes tracking 평가 → AMOTA / AMOTP 등.

        수정(2026-08-03): SparseDrive _evaluate_single(tracking=True) 이식.
        """
        import collections
        import collections.abc
        import os.path as osp
        import mmcv

        #수정(2026-08-03): motmetrics 1.1.3(metrics.py:8)이 `from collections import Iterable` 을
        #  쓰는데 Python 3.10 에서 이 별칭이 제거돼 import 가 죽는다.
        #  nuscenes-devkit 의 tracking 코드가 motmetrics<=1.1.3 API 에 맞춰 작성돼 있어
        #  상위 버전으로 올리면 지표가 조용히 달라질 위험이 있다 → 패키지를 바꾸지 않고,
        #  Python 이 제거한 별칭(=collections.abc.Iterable 과 동일 객체)만 복원한다.
        #  값을 가리거나 계산을 건너뛰는 우회가 아니라 이름 복원이다.
        for _name in ("Iterable", "Mapping", "Sequence", "Callable"):
            if not hasattr(collections, _name):
                setattr(collections, _name, getattr(collections.abc, _name))

        from nuscenes.eval.tracking.evaluate import TrackingEval
        from nuscenes.eval.common.config import config_factory as track_config_factory

        version = (self.metadata or {}).get("version", "v1.0-mini")
        eval_set = "val" if version == "v1.0-trainval" else "mini_val"
        nusc_eval = TrackingEval(
            config=track_config_factory("tracking_nips_2019"),
            result_path=result_path,
            eval_set=eval_set,
            output_dir=out_dir,
            verbose=False,
            nusc_version=version,
            nusc_dataroot=self.data_root,
        )
        nusc_eval.main()
        metrics = mmcv.load(osp.join(out_dir, "metrics_summary.json"))

        keys = ["amota", "amotp", "recall", "motar", "gt", "mota", "motp",
                "mt", "ml", "faf", "tp", "fp", "fn", "ids", "frag", "tid", "lgd"]
        detail = {f"tracking/{k}": metrics[k] for k in keys if k in metrics}
        print("-------------- tracking --------------")
        for k in ("amota", "amotp", "recall", "mota"):
            if k in metrics:
                print(f"  {k.upper()}: {metrics[k]:.4f}")
        return detail

    def format_motion_results(self, results, thresh=None):
        """motion 예측을 UniAD 방식 motion 평가 제출 포맷으로 변환한다.

        수정(2026-08-03): SparseDrive nuscenes_3d_dataset.format_motion_results 이식.
          좌표계 주의 — 원본과 동일하게 **박스 중심만 global 로 변환하고 궤적(trajs)은
          모델 출력 그대로 둔다.** motion_utils.load_gt 가 GT 궤적을 만들 때
          convert_local_coords_to_global(fut_traj_local, box.center, box.rotation) 을
          LIDAR_TOP 프레임의 박스 기준으로 적용하므로 GT 궤적도 LIDAR 프레임이다.
          매칭은 global 중심(center_distance)으로, 궤적 오차(minADE/minFDE)는 LIDAR
          프레임에서 계산된다. 여기서 trajs 를 global 로 바꾸면 GT 와 프레임이 어긋난다.
        """
        import torch

        nusc_annos = {}
        for idx, r in enumerate(results):
            preds = r.get("img_bbox") if isinstance(r, dict) else None
            if preds is None or "trajs_3d" not in preds:
                continue
            info = self.data_infos[idx]
            token = info["token"]
            boxes = preds.get("boxes_3d")
            scores = preds.get("scores_3d")
            labels = preds.get("labels_3d")
            trajs = preds.get("trajs_3d")
            if torch.is_tensor(boxes):
                boxes = boxes.cpu().numpy()
            if torch.is_tensor(scores):
                scores = scores.cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            if torch.is_tensor(trajs):
                trajs = trajs.cpu().numpy()

            annos = []
            for i, (b, s, lab) in enumerate(zip(boxes, scores, labels)):
                if thresh is not None and float(s) < thresh:
                    continue
                lab = int(lab)
                if not (0 <= lab < len(self.det_classes)):
                    continue
                cls_name = self.det_classes[lab]
                conv = self._ego_box_to_global(b, info)
                anno = dict(
                    sample_token=token,
                    detection_name=cls_name,
                    detection_score=float(s),
                    attribute_name=self._attr_from_velocity(cls_name, conv["velocity"]),
                    **conv,
                )
                traj_i = trajs[i]
                anno["trajs"] = traj_i.tolist() if hasattr(traj_i, "tolist") else traj_i
                annos.append(anno)
            nusc_annos[token] = annos

        return {"meta": dict(self.modality), "results": nusc_annos}

    def _evaluate_single_motion(self, submission, out_dir, logger=None, seconds=6):
        """UniAD 프로토콜 motion 평가 → EPA / minADE / minFDE / MissRate.

        수정(2026-08-03): SparseDrive _evaluate_single_motion 이식.
        """
        import prettytable
        from mmcv.utils import print_log
        from nuscenes import NuScenes
        from nuscenes.eval.detection.config import config_factory

        from .evaluation.motion_nusc.motion_eval_uniad import NuScenesEval as NuScenesEvalMotion

        version = (self.metadata or {}).get("version", "v1.0-mini")
        eval_set = "val" if version == "v1.0-trainval" else "mini_val"
        nusc = NuScenes(version=version, dataroot=self.data_root, verbose=False)
        nusc_eval = NuScenesEvalMotion(
            nusc,
            #수정(2026-08-03): 원본은 copy.deepcopy(config) 였으나 이 devkit 버전의
            #  DetectionConfig 는 내부에 dict_keys 를 들고 있어 deepcopy 가 TypeError 로 죽는다.
            #  config_factory 는 호출마다 새 인스턴스를 만들므로 deepcopy 자체가 불필요하다.
            config=config_factory("detection_cvpr_2019"),
            result_path=submission,
            eval_set=eval_set,
            output_dir=out_dir,
            verbose=False,
            seconds=seconds,
        )
        metrics = nusc_eval.main(render_curves=False)

        MOTION_METRICS = ["EPA", "min_ade_err", "min_fde_err", "miss_rate_err"]
        class_names = ["car", "pedestrian"]
        table = prettytable.PrettyTable()
        table.field_names = ["class names"] + MOTION_METRICS
        for class_name in class_names:
            row = [class_name]
            for m in MOTION_METRICS:
                row.append("%.4f" % metrics[f"{class_name}_{m}"])
            table.add_row(row)
        print_log("\n-------------- motion --------------\n" + str(table), logger=logger)
        return metrics

    def format_map_results(self, results, prefix=None):
        """map 예측을 VectorEvaluate 제출 포맷으로 변환한다.

        수정(2026-08-03): SparseDrive nuscenes_3d_dataset.format_map_results 이식.
          HiP-AD Table 4 의 'map mAP' 산출용. 우리 map decoder(models/map/decoder.py)가
          이미 SparseDrive 와 동일한 vectors/scores/labels 키로 내보내므로 변환은 1:1이다.
        """
        import os
        import os.path as osp
        import mmcv

        submissions = {"results": {}}
        for j, pred in enumerate(results):
            if pred is None:  # 빈 예측
                continue
            pred = pred["img_bbox"] if isinstance(pred, dict) and "img_bbox" in pred else pred
            if "vectors" not in pred:
                continue
            single_case = {"vectors": [], "scores": [], "labels": []}
            token = self.data_infos[j]["token"]
            for i in range(len(pred["scores"])):
                vector = pred["vectors"][i]
                # 선(line)은 점이 2개 이상이어야 한다 — 원본과 동일 규칙
                if len(vector) < 2:
                    continue
                single_case["vectors"].append(vector)
                single_case["scores"].append(pred["scores"][i])
                single_case["labels"].append(pred["labels"][i])
            submissions["results"][token] = single_case

        out_path = osp.join(prefix or "./", "submission_vector.json")
        os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
        print(f"[map] submission 저장: {out_path}")
        mmcv.dump(submissions, out_path)
        return out_path

    def evaluate(self, results, logger=None, jsonfile_prefix=None,
                 eval_mode=None, eval_config=None, **kwargs):
        """HiP-AD Table 4 지표 산출.

        수정(2026-08-03): detection 전용이던 것을 eval_mode 분기 구조로 확장.
          Table 4 = detection(mAP/NDS) · map(mAP) · tracking(AMOTA) · motion(minADE).
          eval_mode 미지정 시 detection 만 수행 → 기존 호출부(회귀) 보존.
          각 태스크는 모델이 해당 출력을 실제로 산출할 때만 평가한다. 값이 없는 지표를
          0 으로 채우면 '성능 0' 으로 오독되므로 미산출임을 명시한다.
        """
        import os, json
        from nuscenes.nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import DetectionEval
        from nuscenes.eval.detection.config import config_factory

        eval_mode = eval_mode or dict(with_det=True)
        out_dir = jsonfile_prefix or os.path.join("./work_dirs", "nusc_eval")
        os.makedirs(out_dir, exist_ok=True)
        results_dict = {}

        if not eval_mode.get("with_det", True):
            # detection 을 끄면 아래 DetectionEval 블록 전체가 무의미하다.
            return self._evaluate_aux(results, out_dir, eval_mode, eval_config, logger, results_dict)

        result_path = os.path.join(out_dir, "results_nusc.json")
        self._to_nusc_submission(results, result_path)
        print(f"[NuScenesMiniDataset.evaluate] submission saved: {result_path}")

        #수정: VLA-FULL — v1.0-mini 하드코딩 제거. pkl metadata 의 version 으로 split 자동 선택
        #  (mini → mini_val, trainval → val). version 미기재 시 mini 로 fallback.
        version = (self.metadata or {}).get("version", "v1.0-mini")
        eval_set = "val" if version == "v1.0-trainval" else "mini_val"
        print(f"[NuScenesMiniDataset.evaluate] version={version}, eval_set={eval_set}")
        nusc = NuScenes(version=version, dataroot=self.data_root, verbose=False)
        eval_cfg = config_factory("detection_cvpr_2019")
        nusc_eval = DetectionEval(
            nusc, config=eval_cfg, result_path=result_path,
            eval_set=eval_set, output_dir=out_dir, verbose=False,
        )
        metrics_summary = nusc_eval.main(render_curves=False, plot_examples=0)
        # 요약 출력
        nice = {
            "mAP": metrics_summary["mean_ap"],
            "NDS": metrics_summary["nd_score"],
            "mATE": metrics_summary["tp_errors"]["trans_err"],
            "mASE": metrics_summary["tp_errors"]["scale_err"],
            "mAOE": metrics_summary["tp_errors"]["orient_err"],
            "mAVE": metrics_summary["tp_errors"]["vel_err"],
            "mAAE": metrics_summary["tp_errors"]["attr_err"],
        }
        for cls_name, ap in metrics_summary["mean_dist_aps"].items():
            nice[f"AP_{cls_name}"] = ap
        print("-------------- detection --------------")
        for k, v in nice.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        results_dict.update(nice)

        #수정(2026-08-03): detection 이후 map/tracking/motion 을 이어서 평가한다.
        return self._evaluate_aux(results, out_dir, eval_mode, eval_config, logger, results_dict)

    def _evaluate_aux(self, results, out_dir, eval_mode, eval_config, logger, results_dict):
        """map / tracking / motion 평가 (HiP-AD Table 4 나머지 열).

        수정(2026-08-03): 태스크별로 분리해 두어, 모델이 산출하지 않는 지표는
          0 으로 채우지 않고 '미산출'로 남긴다.
        """
        if eval_mode.get("with_map"):
            if eval_config is None:
                raise ValueError(
                    "with_map=True 인데 eval_config 가 없다. VectorEvaluate 는 GT map vector 를 "
                    "별도 dataset(eval_pipeline)으로 다시 로드하므로 config 의 eval_config 가 필요하다."
                )
            if not self.MAP_CLASSES:
                raise ValueError("with_map=True 인데 dataset 의 map_classes 가 비어 있다.")
            from .evaluation.map.vector_eval import VectorEvaluate

            result_path = self.format_map_results(results, prefix=out_dir)
            #수정(2026-08-10): 워커 수를 명시한다. 기본값(N_WORKERS=16)으로 full val(6019)을 돌렸더니
            #  Pool(16) 의 각 워커가 부모의 gts(6019 샘플 map vector)를 통째로 물고 fork 되어
            #  워커당 RSS 2.9GB + swap 13.5GB → swap 33GB 전량 소진 → 전 워커가 futex_wait 에서
            #  CPU 0% 로 교착(2026-08-08 16:55 이후 43시간 무진행). 40샘플 스모크 때는 n_workers=2
            #  로 넘겨서 드러나지 않았던 문제.
            #  워커를 줄이는 것은 문제를 가리는 우회가 아니라 자원 초과라는 원인 자체를 없애는 것이다.
            #  (map 클래스가 3개뿐이라 병렬도 4면 충분 — 클래스 단위로 병렬화되기 때문.)
            map_evaluator = VectorEvaluate(eval_config, n_workers=int(
                (eval_mode or {}).get("map_eval_workers", 3)))
            map_dict = map_evaluator.evaluate(result_path, logger=logger)
            results_dict.update(map_dict)

        if eval_mode.get("with_tracking"):
            import os
            track_path = os.path.join(out_dir, "results_nusc_tracking.json")
            self.format_tracking_results(
                results, track_path, thresh=eval_mode.get("tracking_threshold", 0.2))
            track_dir = os.path.join(out_dir, "tracking")
            os.makedirs(track_dir, exist_ok=True)
            results_dict.update(
                self._evaluate_single_tracking(track_path, track_dir, logger=logger))

        if eval_mode.get("with_motion"):
            #수정(2026-08-03): motion 은 stage2(task_select 에 'motion')에서만 산출된다.
            #  stage1 체크포인트로 평가하면 trajs_3d 가 없으므로 0 으로 채우지 않고 미산출로 남긴다.
            first = results[0] if results else None
            sample = first.get("img_bbox", first) if isinstance(first, dict) else first
            if not (isinstance(sample, dict) and "trajs_3d" in sample):
                print("[eval] 이 체크포인트는 motion 예측(trajs_3d)을 산출하지 않는다 "
                      "(config 의 task_select 에 motion 미포함) → motion 지표 미산출.")
            else:
                submission = self.format_motion_results(
                    results, thresh=eval_mode.get("motion_threshhold"))
                results_dict.update(
                    self._evaluate_single_motion(submission, out_dir, logger=logger))

        return results_dict
