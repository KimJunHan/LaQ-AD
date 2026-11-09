# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
"""NAVSIM LaQ v2 앵커 — navtrain(현재+과거 프레임 info, kind=train_hist) + 교사 지도 pkl 로 det/map/motion/plan 앵커.

navsim_kmeans.py(수정 안 함)와 같은 알고리즘·KMeans(n_init=10, random_state=0)이고 다른 점만:
  - 입력 kind 'train_hist' 허용(평가 분할 금지는 그대로), 지도 = navsim_teacher_map.py 의 3 클래스(교사 프레임) → 라이다 프레임 회전.
  - det: 프레임이 152,495 개·상자 ~1,000 만 개라 프레임 stride(--det-stride, 기본 8)로 부분추출(~125 만 상자, nuScenes 학습
    전체 상자 수와 비슷한 규모). 앵커 통계 추정용 표본 추출일 뿐 학습 표본을 버리는 것이 아니다.
  - plan: 자차 4 s(8 포즈) 누적 궤적(라이다=ego 프레임, +x 전방) — NAVSIM Trajectory 지평과 같다. 미래 8 포즈가 전부 유효한 프레임.
  - 라이다 전방 축 = +x(lidar2ego 단위행렬 — 데이터 확인) → 지도 직선 앵커 70% 를 x 축 방향으로.
출력(--out-dir): navsim_v2_det_900.npy (900,11) / navsim_v2_map_100.npy (100,20,2) / navsim_v2_motion_6.npy (7,6,12,2) /
                 navsim_v2_plan_temp_48.npy (48,8,2) + navsim_v2_kmeans_meta.json
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
from sklearn.cluster import KMeans

NAVSIM_DET_CLASSES = ("vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier", "czone_sign", "generic_object")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--teacher-map", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dist", type=float, default=55.0)
    ap.add_argument("--z-range", type=float, nargs=2, default=(-5.0, 3.0))
    ap.add_argument("--det-stride", type=int, default=8)
    ap.add_argument("--map-stride", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    d = pickle.load(open(a.infos, "rb"))
    infos, meta = d["infos"], d["metadata"]
    if meta.get("kind") not in ("train", "train_hist"):
        raise ValueError("navtrain info 만 쓴다(평가 분할로 앵커를 만들면 누설)")
    l2e = np.stack([np.asarray(i["lidar2ego"]) for i in infos[::97]])
    if np.abs(l2e - np.eye(4)).max() > 1e-6:
        raise ValueError("lidar2ego != I")
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    M = {}

    cen = []
    for i in infos[::a.det_stride]:
        b = np.asarray(i["gt_boxes"], np.float64).reshape(-1, 7)[:, :3]
        keep = (np.linalg.norm(b[:, :2], axis=1) < a.dist) & (b[:, 2] > a.z_range[0]) & (b[:, 2] < a.z_range[1])
        cen.append(b[keep])
    cen = np.concatenate(cen)
    det = KMeans(n_clusters=900, n_init=10, random_state=0).fit(cen).cluster_centers_
    det = np.concatenate([det, np.array([[1, 1, 1, 1, 0, 0, 0, 0]] * 900)], 1)
    np.save(os.path.join(a.out_dir, "navsim_v2_det_900.npy"), det)
    M["det"] = dict(boxes=len(cen), frame_stride=a.det_stride, seconds=round(time.time() - t0))
    print("det", det.shape, M["det"], flush=True)

    mp = pickle.load(open(a.teacher_map, "rb"))
    if mp["metadata"].get("partial"):
        raise ValueError("부분 지도 pkl")
    frames = mp["frames"]
    mc = []
    for i in infos[::a.map_stride]:
        for gs in frames[i["token"]]["map_annos"].values():
            for g in gs:
                g = np.asarray(g, np.float64)
                if len(g):
                    mc.append(np.array([g[:, 1].mean(), -g[:, 0].mean()]))   # 교사(x 오른쪽, y 전방) → 라이다(x 전방, y 왼쪽)
    mc = np.stack(mc)
    mk = KMeans(n_clusters=100, n_init=10, random_state=0).fit(mc).cluster_centers_
    fwd, lat = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    vecs = []
    for k in range(100):
        L = rng.uniform(8, 20)
        axis = fwd if rng.uniform() < 0.7 else lat
        vecs.append(mk[k][None] + np.linspace(-L / 2, L / 2, 20)[:, None] * axis[None])
    vecs = np.asarray(vecs)
    np.save(os.path.join(a.out_dir, "navsim_v2_map_100.npy"), vecs)
    M["map"] = dict(polylines=len(mc), frame_stride=a.map_stride)
    print("map", vecs.shape, M["map"], flush=True)

    T = int(meta["agent_fut_ts"])
    per = {c: [] for c in range(len(NAVSIM_DET_CLASSES))}
    for i in infos:
        b = np.asarray(i["gt_boxes"], np.float64).reshape(-1, 7)
        tr = np.asarray(i["gt_agent_fut_trajs"], np.float64)
        m = np.asarray(i["gt_agent_fut_masks"], np.float64)
        for j, n in enumerate(i["gt_names"]):
            if n not in NAVSIM_DET_CLASSES or m[j].sum() < T or np.linalg.norm(b[j, :2]) >= a.dist:
                continue
            c, s = np.cos(-b[j, 6]), np.sin(-b[j, 6])
            per[NAVSIM_DET_CLASSES.index(n)].append(np.cumsum(tr[j], 0) @ np.array([[c, s], [-s, c]]))
    motion = []
    for c in range(len(NAVSIM_DET_CLASSES)):
        if len(per[c]) < 6:
            raise RuntimeError(f"모션 앵커: 클래스 {NAVSIM_DET_CLASSES[c]} 표본 {len(per[c])} < 6")
        x = np.stack(per[c]).reshape(len(per[c]), -1)
        motion.append(KMeans(n_clusters=6, n_init=10, random_state=0).fit(x).cluster_centers_.reshape(6, T, 2))
    motion = np.stack(motion).astype(np.float32)
    np.save(os.path.join(a.out_dir, "navsim_v2_motion_6.npy"), motion)
    M["motion"] = {NAVSIM_DET_CLASSES[c]: len(per[c]) for c in per}
    print("motion", motion.shape, M["motion"], flush=True)

    E = int(meta["ego_fut_ts"])
    tr = [np.cumsum(np.asarray(i["gt_ego_fut_trajs"], np.float64), 0) for i in infos
          if np.asarray(i["gt_ego_fut_masks"]).min() >= 1]
    tr = np.stack(tr).reshape(len(tr), -1)
    plan = KMeans(n_clusters=48, n_init=10, random_state=0).fit(tr).cluster_centers_.reshape(48, E, 2).astype(np.float32)
    np.save(os.path.join(a.out_dir, "navsim_v2_plan_temp_48.npy"), plan)
    M["plan"] = dict(trajs=len(tr), horizon_poses=E, end_dist_m=[float(np.linalg.norm(plan[:, -1], axis=1).min()),
                                                                float(np.linalg.norm(plan[:, -1], axis=1).max())])
    print("plan", plan.shape, M["plan"], flush=True)
    M.update(infos=a.infos, teacher_map=a.teacher_map, seconds=round(time.time() - t0), z_range=list(a.z_range), dist=a.dist)
    json.dump(M, open(os.path.join(a.out_dir, "navsim_v2_kmeans_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
