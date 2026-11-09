# 수정(2026-10-15): B 융합 — LiDAR 다중 sweep / Radar 다중 sweep 점 읽기 transform(새 파일).
#   입력: NuScenes3DFusionDataset 가 넘기는 results['lidar_sweeps'], results['radar_sweeps']
#         (tools/data_converter/nuscenes_fusion_infos.py 산출: sensor2lidar R/t, timestamp(us)).
#   모델 좌표계 = LiDAR 키프레임 좌표계(GT 박스·projection_mat(lidar2img) 과 같음). 3D 증강 없음:
#   이 코드베이스의 nuScenes 증강은 영상 2D(ResizeCropFlipImage: lidar2img 만 바뀜)뿐이고 rot3d_range=[0,0],
#   BBoxRotation 미사용 → 점·GT 모두 회전하지 않으므로 일관된다. BEV 증강을 켜려면 점에도 같은 변환을 적용해야 한다
#   (이 파일의 transform 은 BEV 증강을 지원하지 않는다 — results['aug_config']['rotate_3d'] 가 0 이 아니면 즉시 실패).
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES

__all__ = ["LoadPointsFromMultiSweepsFusion", "LoadRadarPointsMultiSweeps", "PointsRangeFilterFusion",
           "FusionPointsToTensor"]


# _BLAS_NOTE: 이 환경의 numpy 1.23.5 번들 OpenBLAS 0.3.20 은 CPU(Xeon Gold 5416S)를 Cooperlake 로 잡고, 그 DGEMM 커널이
#   (N,4)x(4,4) float64 행렬곱을 틀리게 계산한다(2026-10-15 실측: einsum 대비 오차 ~10, OPENBLAS_CORETYPE=SkylakeX/Haswell 이면 0).
#   점 좌표 변환은 BLAS 를 거치지 않는 einsum 으로 계산해 커널 선택과 무관하게 한다.
# NFS(하드 마운트) 파일 하나 읽는 데 왕복 지연이 수십~수백 ms 라 sweep 파일(LiDAR 10개, Radar 30개)을 순차로 읽으면
#   지연이 그대로 더해진다(2026-10-15 실측: LiDAR 10 sweep 순차 2.6~3.8 s → 스레드 0.6~1.0 s, Radar 30 파일 2.5~9.0 s → 0.3~1.1 s).
#   데이터 워커 프로세스마다 I/O 스레드 풀을 하나 두고 파일 읽기만 병렬로 한다(계산·결과 순서는 순차와 같다).
_POOL = {}


def _io_map(fn, items, threads):
    if threads <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    pid = os.getpid()  # fork 된 워커마다 따로 만든다
    if _POOL.get("pid") != pid:
        _POOL.clear()
        _POOL.update(pid=pid, ex=ThreadPoolExecutor(threads))
    return list(_POOL["ex"].map(fn, items))


def _check_no_bev_aug(results):
    aug = results.get("aug_config") or {}
    if abs(float(aug.get("rotate_3d", 0.0))) > 0:
        raise ValueError("융합 점 transform 은 BEV 회전 증강을 지원하지 않는다(rot3d_range 를 [0,0] 으로 둘 것).")


def _remove_close(points, radius=1.0):
    """센서 좌표계에서 반경 radius 이내 점 제거(mmdet3d LoadPointsFromMultiSweeps / devkit remove_close 와 같음)."""
    near = (np.abs(points[:, 0]) < radius) & (np.abs(points[:, 1]) < radius)
    return points[~near]


@PIPELINES.register_module()
class LoadPointsFromMultiSweepsFusion(object):
    """키프레임 LiDAR(results[in_key], LoadPointsFromFile 로 이미 읽음, 5열 x,y,z,intensity,ring) +
    직전 sweep 최대 sweeps_num 개 → results[out_key] (N,5): x,y,z,intensity,time_lag(s, 키프레임=0).
    mmdet3d LoadPointsFromMultiSweeps 와 같은 규약: sweep 점은 센서 좌표계에서 반경 1m 이내를 지우고
    p_l = R p_s + t 로 키프레임 LiDAR 좌표계로 옮긴다. info 에 저장된 sweep 이 sweeps_num 이하이면 모두 쓴다
    (장면 첫 프레임은 sweep 이 적다 — 패딩하지 않는다)."""

    def __init__(self, sweeps_num=10, load_dim=5, in_key="points", out_key="lidar_points",
                 remove_close_radius=1.0, io_threads=16):
        self.io_threads = io_threads
        self.sweeps_num = sweeps_num
        self.load_dim = load_dim
        self.in_key = in_key
        self.out_key = out_key
        self.remove_close_radius = remove_close_radius

    def __call__(self, results):
        _check_no_bev_aug(results)
        key = np.asarray(results[self.in_key], dtype=np.float32)
        if key.shape[1] < 4:
            raise ValueError("키프레임 점에 intensity 열이 필요하다(LoadPointsFromFile load_dim=5, use_dim=5).")
        ts = results["timestamp"]  # s (dataset 이 info timestamp/1e6 으로 넣는다)
        out = [np.concatenate([key[:, :4], np.zeros((key.shape[0], 1), np.float32)], axis=1)]
        sweeps = results["lidar_sweeps"][: self.sweeps_num]
        raws = _io_map(lambda sw: np.fromfile(sw["data_path"], dtype=np.float32), sweeps, self.io_threads)
        for sw, raw in zip(sweeps, raws):
            pts = raw.reshape(-1, self.load_dim)
            pts = _remove_close(pts, self.remove_close_radius)
            R = np.asarray(sw["sensor2lidar_rotation"], np.float32)
            t = np.asarray(sw["sensor2lidar_translation"], np.float32)
            xyz = np.einsum("ij,kj->ik", pts[:, :3], R) + t  # = pts @ R.T (BLAS gemm 미사용: 위 _BLAS_NOTE)
            dt = np.full((pts.shape[0], 1), ts - sw["timestamp"] / 1e6, np.float32)
            out.append(np.concatenate([xyz, pts[:, 3:4], dt], axis=1))
        results[self.out_key] = np.concatenate(out, axis=0)
        return results


@PIPELINES.register_module()
class LoadRadarPointsMultiSweeps(object):
    """5 레이더 × 최대 sweeps_num sweep(키프레임 포함) → results[out_key] (N,7):
    x,y,z(키프레임 LiDAR 좌표), rcs, vx_comp, vy_comp(자차 운동 보상 속도를 LiDAR 좌표계로 회전), time_lag(s).
    - 점 읽기·필터: nuScenes devkit RadarPointCloud.from_file 의 기본 필터(invalid_states=[0], dynprop_states=0..6,
      ambig_states=[3])를 명시 인자로 넘긴다(클래스 전역 설정에 의존하지 않음). 이후 devkit remove_close(1.0) 와 같은 근접 제거.
    - 좌표: devkit from_file_multisweep 과 같은 사슬(sensor→ego(t_s)→global→ego(t_key)→lidar) 의 결과 R,t 를 적용.
    - 속도: vx_comp/vy_comp 는 센서 좌표계에서 표현된 지면 기준 속도이므로 위치 변환의 회전부 R 만 적용한다
      (v_l = R [vx, vy, 0]^T, global 은 관성계라 시각 차이에 의한 추가 항 없음). 점 위치의 물체 운동 보정은 하지 않는다
      (시간차 채널 time_lag 로 모델이 처리)."""

    CHANNELS = ("RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT")

    def __init__(self, sweeps_num=6, out_key="radar_points", remove_close_radius=1.0,
                 invalid_states=(0,), dynprop_states=tuple(range(7)), ambig_states=(3,), io_threads=16):
        self.io_threads = io_threads
        self.sweeps_num = sweeps_num
        self.out_key = out_key
        self.remove_close_radius = remove_close_radius
        self.invalid_states = list(invalid_states)
        self.dynprop_states = list(dynprop_states)
        self.ambig_states = list(ambig_states)

    def _load(self, path):
        from nuscenes.utils.data_classes import RadarPointCloud
        pc = RadarPointCloud.from_file(path, invalid_states=self.invalid_states,
                                       dynprop_states=self.dynprop_states, ambig_states=self.ambig_states)
        return pc.points.T.astype(np.float32)  # (n,18)

    def __call__(self, results):
        _check_no_bev_aug(results)
        ts = results["timestamp"]
        out = []
        sweeps = [sw for ch in self.CHANNELS for sw in results["radar_sweeps"][ch][: self.sweeps_num]]
        loaded = _io_map(lambda sw: self._load(sw["data_path"]), sweeps, self.io_threads)
        for sw, p in zip(sweeps, loaded):
            p = _remove_close(p, self.remove_close_radius)
            R = np.asarray(sw["sensor2lidar_rotation"], np.float32)
            t = np.asarray(sw["sensor2lidar_translation"], np.float32)
            xyz = np.einsum("ij,kj->ik", p[:, :3], R) + t
            v = np.einsum("ij,kj->ik", p[:, 8:10], R[:, :2])  # = [vx,vy,0] @ R.T
            dt = np.full((p.shape[0], 1), ts - sw["timestamp"] / 1e6, np.float32)
            out.append(np.concatenate([xyz, p[:, 5:6], v[:, :2], dt], axis=1))
        results[self.out_key] = (np.concatenate(out, axis=0) if out else np.zeros((0, 7), np.float32))
        return results


@PIPELINES.register_module()
class PointsRangeFilterFusion(object):
    """x,y,z 범위 필터(범위 밖 점은 BEV 격자에 들어갈 수 없다). point_cloud_range = [x0,y0,z0,x1,y1,z1], 반열린 구간."""

    def __init__(self, point_cloud_range, keys=("lidar_points",)):
        self.pcr = np.asarray(point_cloud_range, np.float32)
        self.keys = keys

    def __call__(self, results):
        for k in self.keys:
            p = results[k]
            m = ((p[:, 0] >= self.pcr[0]) & (p[:, 0] < self.pcr[3]) & (p[:, 1] >= self.pcr[1]) & (p[:, 1] < self.pcr[4])
                 & (p[:, 2] >= self.pcr[2]) & (p[:, 2] < self.pcr[5]))
            results[k] = np.ascontiguousarray(p[m])
        return results


@PIPELINES.register_module()
class FusionPointsToTensor(object):
    """가변 길이 점 배열 → DataContainer(stack=False): 배치 안에서 표본별 텐서 리스트로 모델에 전달."""

    def __init__(self, keys=("lidar_points",)):
        self.keys = keys

    def __call__(self, results):
        for k in self.keys:
            results[k] = DC(torch.from_numpy(np.ascontiguousarray(results[k], dtype=np.float32)),
                            stack=False, cpu_only=False)
        return results
