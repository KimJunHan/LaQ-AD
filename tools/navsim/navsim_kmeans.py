"""수정(2026-10-09): LaQ-AD NAVSIM — navtrain info 로 검출/지도/모션/계획 앵커 생성.

nuScenes 와 같은 알고리즘(tools/kmeans/kmeans_nusc.py, kmeans_nusc_plan.py, kmeans_motion.py)을 NAVSIM 좌표계에 맞춰 옮겼다.
입력: build_navsim_infos.py --kind train 산출 + navsim_devkit_steps.py egometa/map 을 거친 pkl.
출력(data/kmeans/):
  navsim_det_900.npy   (900, 11)  [x,y,z] + [W,L,H,SIN,COS,VX,VY,VZ]=[1,1,1,1,0,0,0,0] (nuScenes 와 같은 초기화)
  navsim_map_100.npy   (100, 20, 2) 지도 폴리라인 중심 군집 + 길이 8~20 m 직선(70% 자차 전방 축 방향)
  navsim_motion_6.npy  (7, 6, 12, 2) 클래스별 에이전트 국소좌표 미래 궤적(_agent2lidar 규약: lidar = R(yaw) @ agent)
  navsim_plan_temp_48.npy (48, 8, 2) 자차 4 s 누적 궤적(라이다 프레임)
군집화 표본 선택(전 구간 유효 궤적만 등)은 앵커 통계 추정의 조건일 뿐 학습 표본을 버리는 것이 아니다(nuScenes 와 동일 규칙).
클래스에 표본이 K 개 미만이면 앵커 행이 빠져 클래스 인덱스가 어긋나므로 즉시 실패한다(b2d 스크립트는 조용히 건너뜀).
"""
import argparse
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans

NAVSIM_DET_CLASSES = ("vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier", "czone_sign", "generic_object")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True, help="navtrain info pkl")
    ap.add_argument("--out-dir", default="data/kmeans")
    ap.add_argument("--dist", type=float, default=55.0)
    ap.add_argument("--z-range", type=float, nargs=2, required=True,
                    help="검출 z 범위(라이다 프레임) — config point_cloud_range 의 z 와 같게")
    ap.add_argument("--map-stride", type=int, default=100, help="nuScenes 와 같이 지도 표본을 stride 로 부분추출")
    args = ap.parse_args()
    d = pickle.load(open(args.infos, "rb"))
    infos, meta = d["infos"], d["metadata"]
    if meta.get("kind") != "train":
        raise ValueError("navtrain(--kind train) info 만 쓴다 — 평가 분할로 앵커를 만들면 누설이다.")
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    # ---------------- det
    cen = []
    for i in infos:
        b = np.asarray(i["gt_boxes"], np.float64).reshape(-1, 7)[:, :3]
        keep = (np.linalg.norm(b[:, :2], axis=1) < args.dist) & (b[:, 2] > args.z_range[0]) & (b[:, 2] < args.z_range[1])
        cen.append(b[keep])
    cen = np.concatenate(cen)
    det = KMeans(n_clusters=900, n_init=10, random_state=0).fit(cen).cluster_centers_
    det = np.concatenate([det, np.array([[1, 1, 1, 1, 0, 0, 0, 0]] * 900)], 1)
    np.save(os.path.join(args.out_dir, "navsim_det_900.npy"), det)
    print("det", det.shape, "boxes", len(cen))

    # ---------------- map
    if "lidar_forward_axis" not in meta or "map_classes" not in meta:
        raise RuntimeError("egometa/map 단계가 안 된 info 다(navsim_devkit_steps.py).")
    fwd = np.asarray(meta["lidar_forward_axis"][:2], np.float64)
    fwd /= np.linalg.norm(fwd)
    lat = np.array([-fwd[1], fwd[0]])
    mc = [np.asarray(g, np.float64).mean(0) for i in infos[::args.map_stride]
          for gs in i["map_annos"].values() for g in gs if len(g)]
    mc = np.stack(mc)
    mk = KMeans(n_clusters=100, n_init=10, random_state=0).fit(mc).cluster_centers_
    vecs = []
    for k in range(100):
        L = rng.uniform(8, 20)
        axis = fwd if rng.uniform() < 0.7 else lat
        vecs.append(mk[k][None] + np.linspace(-L / 2, L / 2, 20)[:, None] * axis[None])
    vecs = np.asarray(vecs)
    np.save(os.path.join(args.out_dir, "navsim_map_100.npy"), vecs)
    print("map", vecs.shape, "polylines", len(mc))

    # ---------------- motion (agent frame)
    T = int(meta["agent_fut_ts"])
    per = {c: [] for c in range(len(NAVSIM_DET_CLASSES))}
    for i in infos:
        b = np.asarray(i["gt_boxes"], np.float64).reshape(-1, 7)
        tr = np.asarray(i["gt_agent_fut_trajs"], np.float64)
        m = np.asarray(i["gt_agent_fut_masks"], np.float64)
        for j, n in enumerate(i["gt_names"]):
            if n not in NAVSIM_DET_CLASSES or m[j].sum() < T or np.linalg.norm(b[j, :2]) >= args.dist:
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
    np.save(os.path.join(args.out_dir, "navsim_motion_6.npy"), motion)
    print("motion", motion.shape)

    # ---------------- plan (lidar frame cumulative)
    E = int(meta["ego_fut_ts"])
    tr = [np.cumsum(np.asarray(i["gt_ego_fut_trajs"], np.float64), 0) for i in infos
          if np.asarray(i["gt_ego_fut_masks"]).min() >= 1]
    tr = np.stack(tr).reshape(len(tr), -1)
    plan = KMeans(n_clusters=48, n_init=10, random_state=0).fit(tr).cluster_centers_.reshape(48, E, 2).astype(np.float32)
    np.save(os.path.join(args.out_dir, "navsim_plan_temp_48.npy"), plan)
    print("plan", plan.shape, "trajs", len(tr))


if __name__ == "__main__":
    main()
