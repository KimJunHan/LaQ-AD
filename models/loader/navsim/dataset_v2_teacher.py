# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
"""NAVSIM LaQ v2(교사 캐시) 학습·평가용 데이터셋 — 새 파일(기존 NavsimDataset·패키지 __init__ 무수정).

등록: config 의 custom_imports = dict(imports=["models.loader.navsim.dataset_v2_teacher"]) 로만 import 된다
(models/loader/navsim/__init__.py 를 고치지 않으므로 nuScenes 학습의 import 경로는 전혀 바뀌지 않는다).

NavsimDataset(models/loader/navsim/dataset.py) 을 상속하고 바뀌는 것:
  1. 정보 파일: build_navsim_infos_hist.py 산출(kind="train_hist", 현재+과거 프레임만) / navtest(kind="test") 둘 다.
  2. 지도 GT: info 에 덧붙이는 navsim_devkit_steps.py map(제자리 pkl 수정) 대신, navsim_teacher_map.py 의 별도 지도 pkl
     (교사와 같은 지도: 0 ped_crossing / 1 divider / 2 boundary — nuScenes 와 같은 3 클래스 의미)을 읽어 라이다 프레임으로 돌린다.
     교사 프레임(+x 오른쪽, +y 전방) → 라이다=ego(+x 전방, +y 왼쪽): x_l = y_t, y_l = −x_t. ROI 교사 (30, 60) → 라이다 (60, 30).
  3. 라이다(보조 깊이 손실): .pcd → .bin 변환(수백 GB 복사) 대신 LoadPointsFromPcdNavsim 이 .pcd(binary) 를 직접 읽는다.
  4. 자차 앵커·크기: egometa(제자리 pkl 수정) 대신 nuPlan Pacifica 제원 상수 + 데이터 확인값으로 이 파일에서 계산하고 config 와 대조.
     중심 = rear-axle 전방 1.461 m, 높이 중심 z = 지면(−0.35 m: 20 m 안 차량 상자 바닥 z 중앙값, navtrain 20 로그 실측) + 1.777/2.
     lidar2ego 가 단위행렬인지 전 프레임 검사(아니면 즉시 실패).
나머지(카메라 3대·투영·ego_status 4·명령 입력·계획 8 포즈·evaluate)는 NavsimDataset 그대로.
"""
import os
import pickle

import numpy as np
from mmdet.datasets import DATASETS
from mmdet.datasets.builder import PIPELINES

from ..pipelines.loading import LoadPointsFromFile
from .dataset import NavsimDataset

TEACHER_MAP_CLASSES = ("ped_crossing", "divider", "boundary")
PACIFICA = dict(length=5.176, width=2.297, height=1.777, rear_axle_to_center=1.461)   # nuplan get_pacifica_parameters
GROUND_Z = -0.35          # ego(rear-axle) 프레임 지면 높이 추정 — 위 docstring 참조
NAVSIM_EGO_SIZE = [PACIFICA["length"], PACIFICA["width"], PACIFICA["height"]]
NAVSIM_EGO_ANCHOR = [PACIFICA["rear_axle_to_center"], 0.0, round(GROUND_Z + PACIFICA["height"] / 2, 4),
                     float(np.log(PACIFICA["length"])), float(np.log(PACIFICA["width"])), float(np.log(PACIFICA["height"])),
                     0.0, 1.0, 0.0, 0.0, 0.0]


def teacher_to_lidar(pl):
    pl = np.asarray(pl, np.float32)
    return np.stack([pl[:, 1], -pl[:, 0]], 1)


@PIPELINES.register_module()
class LoadPointsFromPcdNavsim(LoadPointsFromFile):
    """OpenScene MergedPointCloud .pcd(binary, FIELDS x y z intensity lidar_info ring, SIZE 4 4 4 1 1 1, TYPE F F F U U U)
    → (n, 5) float32 [x, y, z, intensity, ring] (navsim_devkit_steps.py lidar 의 .bin 과 같은 열). 헤더가 다르면 즉시 실패."""

    _EXPECT = dict(FIELDS="x y z intensity lidar_info ring", SIZE="4 4 4 1 1 1", TYPE="F F F U U U", COUNT="1 1 1 1 1 1")
    _DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "u1"), ("lidar_info", "u1"), ("ring", "u1")])

    def __init__(self, coord_type="LIDAR", load_dim=5, use_dim=5, **kw):
        super().__init__(coord_type=coord_type, load_dim=load_dim, use_dim=use_dim, **kw)

    @classmethod
    def parse(cls, buf):
        head, pos = {}, 0
        while True:
            end = buf.index(b"\n", pos)
            line = buf[pos:end].decode("ascii").strip()
            pos = end + 1
            if line.startswith("#") or not line:
                continue
            k, _, v = line.partition(" ")
            head[k] = v
            if k == "DATA":
                break
        for k, v in cls._EXPECT.items():
            if head.get(k) != v:
                raise ValueError(f"PCD 헤더 {k}={head.get(k)!r} != {v!r}")
        if head["DATA"] != "binary":
            raise ValueError(f"PCD DATA={head['DATA']} (binary 만 지원)")
        n = int(head["POINTS"])
        a = np.frombuffer(buf, dtype=cls._DT, count=n, offset=pos)
        if pos + n * cls._DT.itemsize != len(buf):
            raise ValueError(f"PCD 크기 불일치: 헤더 {pos} + {n}×{cls._DT.itemsize} != {len(buf)}")
        return np.stack([a["x"], a["y"], a["z"], a["intensity"].astype(np.float32), a["ring"].astype(np.float32)], 1)

    def _load_points(self, pts_filename):
        if not str(pts_filename).endswith(".pcd"):
            raise ValueError(f"{pts_filename}: .pcd 가 아니다")
        with open(pts_filename, "rb") as f:
            pts = self.parse(f.read())
        return pts.reshape(-1).astype(np.float32)   # 부모가 reshape(-1, load_dim) 한다


@DATASETS.register_module()
class NavsimTeacherDataset(NavsimDataset):
    def __init__(self, *args, teacher_map_files=None, ego_anchor=None, ego_size=None, map_roi=None, map_classes=None,
                 command_groups=4, ego_input_file=None, **kwargs):  # 수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건 (3) command_groups
        self.command_groups = int(command_groups)
        # 수정(2026-11-05): 사용자 결정 "E2 자차 상태 직접 입력" — ego_input_file(tools/navsim/build_navsim_ego_input.py 산출 pkl)이 있으면
        #   표본마다 input_dict["ego_input"] = [속도, 종방향 가속도, yaw rate, 유효] (교사 프롬프트 자차 상태 줄 값)을 넣는다.
        #   None(기본) = 키를 넣지 않음(종전과 동일). info 프레임이 표에 없으면 초기화 실패(대체값 없음).
        self.ego_input = None
        if ego_input_file is not None:
            _d = pickle.load(open(ego_input_file, "rb"))
            if _d["metadata"].get("fields") != ["speed_mps", "accel_long_mps2", "yaw_rate_rps", "available"]:
                raise ValueError(f"{ego_input_file}: fields {_d['metadata'].get('fields')}")
            self.ego_input = _d["ego_input"]
        kwargs.pop("require_map", None); kwargs.pop("require_lidar_bin", None)
        super().__init__(*args, ego_anchor=None, ego_size=None, map_roi=None, map_classes=map_classes,
                         require_map=False, require_lidar_bin=False, **kwargs)
        kind = self.metadata.get("kind")
        if kind not in ("train_hist", "test"):
            raise ValueError(f"{self.ann_file}: kind={kind} — build_navsim_infos_hist.py(navtrain) 또는 navtest(test) 산출물만")
        if not self.test_mode and kind != "train_hist":
            raise ValueError("학습은 현재+과거 프레임만 쓰는 train_hist info 로만 한다")
        l2e = np.stack([np.asarray(i["lidar2ego"], np.float64) for i in self.data_infos])
        if np.abs(l2e - np.eye(4)).max() > 1e-6:
            raise ValueError("lidar2ego 가 단위행렬이 아닌 프레임이 있다 — 자차 앵커·지도 회전 가정 위반")
        if ego_anchor is not None and not np.allclose(ego_anchor, NAVSIM_EGO_ANCHOR, atol=1e-4):
            raise ValueError(f"config ego_anchor {ego_anchor} != {NAVSIM_EGO_ANCHOR}")
        if ego_size is not None and not np.allclose(ego_size, NAVSIM_EGO_SIZE, atol=1e-4):
            raise ValueError(f"config ego_size {ego_size} != {NAVSIM_EGO_SIZE}")
        if tuple(map_classes or ()) != TEACHER_MAP_CLASSES:
            raise ValueError(f"map_classes {map_classes} != {TEACHER_MAP_CLASSES}")
        if map_roi is not None and tuple(float(x) for x in map_roi) != (60.0, 30.0):
            raise ValueError(f"map_roi {map_roi} != (60, 30) (라이다 x 전방 60 m · y 30 m = 교사 지도 ROI 회전)")
        if not teacher_map_files:
            raise ValueError("teacher_map_files 필요(navsim_teacher_map.py 산출)")
        self.map_frames = {}
        for p in teacher_map_files:
            d = pickle.load(open(p, "rb"))
            m = d["metadata"]
            if m.get("partial") or [m["classes"][k] for k in range(3)] != list(TEACHER_MAP_CLASSES) or list(m["roi"]) != [30.0, 60.0]:
                raise ValueError(f"{p}: 지도 pkl 메타 불일치/부분 산출물 {m}")
            for t, v in d["frames"].items():
                self.map_frames[t] = v["map_annos"]
        if self.ego_input is not None:  # 수정(2026-11-05): 사용자 결정 "E2 자차 상태 직접 입력"
            _m = [i["token"] for i in self.data_infos if i["token"] not in self.ego_input]
            if _m:
                raise RuntimeError(f"ego_input 표에 없는 프레임 {len(_m)} (예: {_m[:3]}) — build_navsim_ego_input.py 를 이 info 로 실행")
        miss = [i["token"] for i in self.data_infos if i["token"] not in self.map_frames]
        if miss:
            raise RuntimeError(f"지도 pkl 에 없는 프레임 {len(miss)} (예: {miss[:3]}) — navsim_teacher_map.py 를 이 info 로 실행")

    # 수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건 (3) 명령별 계획 앵커 3×16.
    #   NAVSIM driving_command one-hot [0 left, 1 straight, 2 right, 3 unknown] → 3 묶음 [left, straight, right] one-hot,
    #   unknown → straight(실측: navtrain 360 프레임·navtest 56 장면, 0.24%). 앵커 파일(navsim_kmeans_plan_cmd.py)과 같은 규칙.
    #   명령은 경로만 반영한 입력(docs/agents.md:79)이라 학습·추론 모두 같은 값을 쓴다.
    NAV2CMD3 = {0: 0, 1: 1, 2: 2, 3: 1}

    def get_ann_info(self, index):
        res = super().get_ann_info(index)
        if self.command_groups == 3:
            c = np.zeros(3, np.float32)
            c[self.NAV2CMD3[int(np.argmax(res["gt_ego_fut_cmd"]))]] = 1.0
            res["gt_ego_fut_cmd"] = c
        elif self.command_groups != 4:
            raise ValueError(f"command_groups {self.command_groups} (3 또는 4)")
        return res

    def get_data_info(self, index):
        d = super().get_data_info(index)
        info = self.data_infos[index]
        annos = {k: [teacher_to_lidar(pl) for pl in v] for k, v in self.map_frames[info["token"]].items()}
        d["map_infos"] = annos
        d["map_geoms"] = self.anno2geom(annos)
        d["map_location"] = info["map_location"]
        if self.ego_input is not None:  # 수정(2026-11-05): 사용자 결정 "E2 자차 상태 직접 입력"
            d["ego_input"] = np.array(self.ego_input[info["token"]], np.float32)
        return d
