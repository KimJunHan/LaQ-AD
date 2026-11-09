# 수정(2026-10-15): 경로 이동 mmdet3d_plugin/datasets/pipelines/transform.py → models/loader/pipelines/transform.py. LoadQwenCacheNusc 는 models/encoder/vl_tokens_cache/loading.py 로 이동.
import numpy as np
import mmcv
import cv2, copy
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor

@PIPELINES.register_module()
class MultiScaleDepthMapGenerator(object):
    def __init__(self, downsample=1, max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth

    def __call__(self, input_dict):
        points = input_dict["points"][..., :3, None]
        gt_depth = []
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = input_dict["img_shape"][i][:2]

            pts_2d = (
                np.squeeze(lidar2img[:3, :3] @ points, axis=-1)
                + lidar2img[:3, 3]
            )
            pts_2d[:, :2] /= pts_2d[:, 2:3]
            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                    # depths <= self.max_depth,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, 0.1, self.max_depth)
            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)
                depth_map = np.ones([h, w], dtype=np.float32) * -1
                depth_map[v, u] = depths
                gt_depth[j].append(depth_map)

        input_dict["gt_depth"] = [np.stack(x) for x in gt_depth]
        return input_dict

@PIPELINES.register_module()
class B2DMultiScaleDepthMapGenerator(object):
    def __init__(self, downsample=1, max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth

    def __call__(self, input_dict):
        points = input_dict["points"][..., :3]
        points = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)

        gt_depth = []
        for i, ego2img in enumerate(input_dict["ego2img"]):
            H, W = input_dict["img_shape"][i][:2]

            pts_2d = (ego2img @ points.T).T
            pts_2d[:, :2] /= np.clip(pts_2d[:, 2:3], 1e-5, points[:, 2:3].max())

            U = np.round(pts_2d[:, 0]).astype(np.int32)
            V = np.round(pts_2d[:, 1]).astype(np.int32)
            depths = pts_2d[:, 2]
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                    # depths <= self.max_depth,
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
            depths = np.clip(depths, 1e-5, self.max_depth)
            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)
                depth_map = np.ones([h, w], dtype=np.float32) * -1
                depth_map[v, u] = depths
                gt_depth[j].append(depth_map)

        input_dict["gt_depth"] = [np.stack(x) for x in gt_depth]
        return input_dict

@PIPELINES.register_module()
class NuScenesSparse4DAdaptor(object):
    def __init(self):
        pass

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32(np.stack(input_dict["lidar2img"]))
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        input_dict["T_global_inv"] = np.linalg.inv(input_dict["lidar2global"])
        input_dict["T_global"] = input_dict["lidar2global"]
        if "cam_intrinsic" in input_dict:
            input_dict["cam_intrinsic"] = np.float32(np.stack(input_dict["cam_intrinsic"]))
            input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]
        if "instance_inds" in input_dict:
            input_dict["instance_id"] = input_dict["instance_inds"]
        if "gt_bboxes_3d" in input_dict:
            input_dict["gt_bboxes_3d"][:, 6] = self.limit_period(
                input_dict["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )
            input_dict["gt_bboxes_3d"] = DC(
                to_tensor(input_dict["gt_bboxes_3d"]).float()
            )
        if "gt_labels_3d" in input_dict:
            input_dict["gt_labels_3d"] = DC(
                to_tensor(input_dict["gt_labels_3d"]).long()
            )

        imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        input_dict["img"] = DC(to_tensor(imgs), stack=True)

        for key in [
            'gt_map_labels', 
            'gt_map_pts',
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
        ]:
            if key not in input_dict:
                continue
            input_dict[key] = DC(to_tensor(input_dict[key]), stack=False, cpu_only=False) 

        for key in [
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            "command_near_xy",
            'ego_status',
            'ego_input',  # 수정(2026-11-05): 사용자 결정 "E2 자차 상태 직접 입력" — 없으면 건너뜀(종전과 동일)
        ]:
            if key not in input_dict:
                continue
            input_dict[key] = DC(to_tensor(input_dict[key]), stack=True, cpu_only=False, pad_dims=None)
        
        return input_dict

    def limit_period(
        self, val: np.ndarray, offset: float = 0.5, period: float = np.pi
    ) -> np.ndarray:
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val

@PIPELINES.register_module()
class InstanceNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][gt_bboxes_mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][gt_bboxes_mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][gt_bboxes_mask]
        if 'gt_attr_labels' in input_dict:
            input_dict['gt_attr_labels'] = input_dict['gt_attr_labels'][gt_bboxes_mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(classes={self.classes})"
        return repr_str

@PIPELINES.register_module()
class BEVObjectRangeFilter(object):
    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        self.box_range = self.pcd_range[[0, 1, 3, 4]]

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]

        mask = ((gt_bboxes_3d[:, 0] > self.box_range[0])
                & (gt_bboxes_3d[:, 1] > self.box_range[1])
                & (gt_bboxes_3d[:, 0] < self.box_range[2])
                & (gt_bboxes_3d[:, 1] < self.box_range[3])
        )

        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][mask]
        if 'gt_attr_labels' in input_dict:
            input_dict['gt_attr_labels'] = input_dict['gt_attr_labels'][mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_dist_thred={self.class_dist_thred})"
        return repr_str

@PIPELINES.register_module()
class CircleObjectRangeFilter(object):
    def __init__(self, class_dist_thred=[52.5] * 5 + [31.5] + [42] * 3 + [31.5]):
        self.class_dist_thred = class_dist_thred

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        dist = np.sqrt(
            np.sum(gt_bboxes_3d[:, :2] ** 2, axis=-1)
        )
        mask = np.array([False] * len(dist))
        for label_idx, dist_thred in enumerate(self.class_dist_thred):
            mask = np.logical_or(
                mask,
                np.logical_and(gt_labels_3d == label_idx, dist <= dist_thred),
            )

        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]
        if "gt_agent_fut_trajs" in input_dict:
            input_dict["gt_agent_fut_trajs"] = input_dict["gt_agent_fut_trajs"][mask]
            input_dict["gt_agent_fut_masks"] = input_dict["gt_agent_fut_masks"][mask]
        if 'gt_attr_labels' in input_dict:
            input_dict['gt_attr_labels'] = input_dict['gt_attr_labels'][mask]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_dist_thred={self.class_dist_thred})"
        return repr_str

# 수정(2026-11-09): NAVSIM 1단계 발산 진단(docs/navsim_s1_divergence_20261109.md §6.1·§7, 검증 V4)의 GT 필터 2 개를 정식 transform 으로 옮김.
#   진단 구현 = results/diag_navsim_s1_div_20261109/tools/{diag_verify_fix.py, diag_fov_filter.py, fov_common.py}(CircleObjectRangeFilter
#   __call__ 덧씌우기). 판정·키 처리·적용 조건(not mask.all())을 그대로 옮겼다. config 에 넣은 경우에만 동작 → nuScenes·B2D 무변경.
_GT_INSTANCE_KEYS = ("gt_bboxes_3d", "gt_labels_3d", "instance_inds", "gt_agent_fut_trajs", "gt_agent_fut_masks", "gt_attr_labels")


def _apply_instance_mask(input_dict, mask):
    # 수정(2026-11-09): 진단 구현과 같은 키 목록·같은 조건(전부 통과면 배열을 건드리지 않음)
    if not mask.all():
        for k in _GT_INSTANCE_KEYS:
            if k in input_dict:
                input_dict[k] = input_dict[k][mask]
    return input_dict


@PIPELINES.register_module()
class ObjectHeightRangeFilter(object):
    """# 수정(2026-11-09): NAVSIM GT 높이 이상치 정리 — navtrain 에 z=-626 m 등 지면 아래 박스(55 m 안 z<-5 박스 268 개)가 있다.
    nuScenes 는 num_lidar_pts>0 관측 가능성 필터로 이런 박스가 빠지지만 NAVSIM 은 필터가 없다. z(box[:, 2]) 가
    [z_range[0], z_range[1]] 밖인 GT 를 뺀다(양 끝 포함, 표본은 그대로). 상한 10 m: navtrain z 최대 9.77 m — 정상 오르막 물체 보존
    (검토 1: 상한 3 m 는 정상 물체 7,685 개 제거). 진단 diag_verify_fix.py(Z0, Z1 = -5, 10)와 같은 판정."""

    def __init__(self, z_range=(-5.0, 10.0)):
        self.z_range = (float(z_range[0]), float(z_range[1]))

    def __call__(self, input_dict):
        b = input_dict["gt_bboxes_3d"]
        mask = (b[:, 2] >= self.z_range[0]) & (b[:, 2] <= self.z_range[1])
        return _apply_instance_mask(input_dict, mask)

    def __repr__(self):
        return f"{self.__class__.__name__}(z_range={self.z_range})"


@PIPELINES.register_module()
class CameraFOVObjectFilter(object):
    """# 수정(2026-11-09): 카메라 시야 관측 가능성 필터(NAVSIM 1단계 발산 검증 V4 에서 lr 4e-4 안정).
    NAVSIM GT 는 360° 라벨인데 입력은 전방 3 캠(L0/F0/R0)이라 GT 의 약 절반(자차 뒤)이 영상에 없다(프레임당 63.4 → 32.5 개).
    박스 8 모서리 중 하나라도 어느 카메라 영상 안(가로·세로 margin_frac 여유) + 깊이 > min_depth 이면 유지한다.
    투영 = input_dict["lidar2img"](증강 뒤 — ResizeCropFlipImage·BBoxRotation 다음에 둘 것), 영상 크기 = input_dict["img_shape"](H, W).
    사용 카메라 = input_dict 에 들어 있는 카메라 전부(NAVSIM = NAVSIM_CAMS L0/F0/R0). expect_cameras 를 주면 camera_names 와 대조(다르면 오류).
    진단 fov_common.visible_mask(MARGIN_FRAC 0.05, 깊이 > 0.1, crop_top 0)와 같은 수식·같은 연산 순서(float64)."""

    def __init__(self, margin_frac=0.05, min_depth=0.1, crop_top=0.0, expect_cameras=None):
        self.margin_frac = float(margin_frac)
        self.min_depth = float(min_depth)
        self.crop_top = float(crop_top)
        self.expect_cameras = None if expect_cameras is None else tuple(expect_cameras)

    @staticmethod
    def box_corners(b):
        """b: (N,>=7) [x,y,z,w,l,h,yaw] (HiP-AD box3d 순서). 중심 기준 8 모서리 (N,8,3). 진단 fov_common.box_corners 와 같다."""
        x, y, z, w, l, h, yaw = [b[:, i] for i in range(7)]
        sx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * 0.5
        sy = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * 0.5
        sz = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * 0.5
        dx = sx[None] * w[:, None]; dy = sy[None] * l[:, None]; dz = sz[None] * h[:, None]
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        return np.stack([x[:, None] + c * dx - s * dy, y[:, None] + s * dx + c * dy, z[:, None] + dz], -1)

    def visible_mask(self, boxes, lidar2img_list, hw_list):
        if len(boxes) == 0:
            return np.zeros(0, bool)
        cor = self.box_corners(np.asarray(boxes, np.float64))       # (N,8,3)
        P = np.concatenate([cor, np.ones_like(cor[..., :1])], -1)    # (N,8,4)
        vis = np.zeros(len(boxes), bool)
        for M, (H, W) in zip(lidar2img_list, hw_list):
            q = P @ np.asarray(M, np.float64).T                      # (N,8,4)
            d = q[..., 2]
            u = q[..., 0] / np.clip(d, 1e-5, None); v = q[..., 1] / np.clip(d, 1e-5, None)
            mu, mv = self.margin_frac * W, self.margin_frac * (H - self.crop_top)
            ok = (d > self.min_depth) & (u >= -mu) & (u < W + mu) & (v >= self.crop_top - mv) & (v < H + mv)
            vis |= ok.any(1)
        return vis

    def __call__(self, input_dict):
        l2i = input_dict["lidar2img"]
        hw = [tuple(s[:2]) for s in input_dict["img_shape"]]
        if len(l2i) != len(hw):
            raise ValueError(f"CameraFOVObjectFilter: lidar2img {len(l2i)} 개 != img_shape {len(hw)} 개")
        if self.expect_cameras is not None and tuple(input_dict.get("camera_names", ())) != self.expect_cameras:
            raise ValueError(f"CameraFOVObjectFilter: camera_names {input_dict.get('camera_names')} != expect_cameras {self.expect_cameras}")
        b = np.asarray(input_dict["gt_bboxes_3d"])
        mask = self.visible_mask(b[:, :7], l2i, hw)
        return _apply_instance_mask(input_dict, mask)

    def __repr__(self):
        return (f"{self.__class__.__name__}(margin_frac={self.margin_frac}, min_depth={self.min_depth}, "
                f"crop_top={self.crop_top}, expect_cameras={self.expect_cameras})")


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str
