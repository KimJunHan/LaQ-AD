# 수정(2026-10-15): 실험 C(LaQ 교정 레시피) — SparseDrive 원본 tools/kmeans/kmeans_map.py · kmeans_det.py 방식으로
#   nuScenes 지도·검출 앵커를 새 파일로 만든다(기존 nusc_map_100.npy / nusc_det_900.npy 는 건드리지 않는다).
#   SparseDrive 와 같은 점: train info 를 timestamp 로 정렬, 지도 = 전 표본의 모든 polyline 중심 k-means(K=100) 후
#   길이 8 m(±4 m) 세로(y) 직선 20점, 검출 = 거리 < 55 m 박스 중심 k-means(K=900) + [1,1,1,1,0,0,0,0] (z 필터 없음).
#   SparseDrive 와 다른 점(재현성): KMeans 에 random_state=0, n_init=10 을 명시했다(원본은 시드 미지정 → 매번 다른 앵커;
#   n_init=10 은 원본 당시 sklearn 기본값. 현재 sklearn 1.3 기본도 10 이지만 경고 억제·버전 고정 목적으로 명시).
#   종전 우리 방식(kmeans_nusc.py) 과의 차이: 지도 = 1/100 표본·길이 8~20 m 무작위·30% 가로 → SD 방식(전 표본·8 m·세로);
#   검출 = z ∈ (-5, 3) 필터 → SD 는 없음.
#   사용(저장소 루트, CPU): OMP_NUM_THREADS=8 PYTHONPATH=. python tools/kmeans/kmeans_nusc_sd.py [--det]
import argparse
import os

import mmcv
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

p = argparse.ArgumentParser()
p.add_argument("--det", action="store_true", help="검출 앵커(nusc_det_900_sd.npy)도 만든다")
p.add_argument("--info", default="data/infos/nuscenes_infos_train.pkl")
p.add_argument("--out_dir", default="data/kmeans")
args = p.parse_args()

map_out = os.path.join(args.out_dir, "nusc_map_100_sd.npy")
det_out = os.path.join(args.out_dir, "nusc_det_900_sd.npy")
for f in [map_out] + ([det_out] if args.det else []):
    if os.path.exists(f):
        raise FileExistsError(f"{f} 가 이미 있다 — 덮어쓰지 않는다(새 이름을 쓰거나 기존 파일을 확인할 것).")

data = mmcv.load(args.info)
infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))  # SparseDrive 와 같은 정렬
print(f"train 표본 {len(infos)}")

# ===== 지도 앵커 (SparseDrive kmeans_map.py) =====
K, num_sample = 100, 20
center = []
for info in tqdm(infos, desc="map polylines"):
    for cls, geoms in info["map_annos"].items():
        for geom in geoms:
            center.append(np.asarray(geom).mean(axis=0))
center = np.stack(center, axis=0)
print(f"지도 polyline 중심 {len(center)}개")
center = KMeans(n_clusters=K, n_init=10, random_state=0).fit(center).cluster_centers_
delta_y = np.linspace(-4, 4, num_sample)
delta_x = np.zeros([num_sample])
delta = np.stack([delta_x, delta_y], axis=-1)
vecs = center[:, np.newaxis] + delta[np.newaxis]
np.save(map_out, vecs)
print(f"저장 {map_out} shape={vecs.shape} x[{center[:,0].min():.1f},{center[:,0].max():.1f}] "
      f"y[{center[:,1].min():.1f},{center[:,1].max():.1f}]")

# ===== 검출 앵커 (SparseDrive kmeans_det.py) =====
if args.det:
    K_DET, DIS_THRESH = 900, 55
    c = []
    for info in tqdm(infos, desc="det boxes"):
        boxes = info["gt_boxes"][:, :3]
        if len(boxes) == 0:
            continue
        distance = np.linalg.norm(boxes[:, :2], axis=1)
        c.append(boxes[distance < DIS_THRESH])
    c = np.concatenate(c, axis=0)
    print(f"검출 박스 중심 {len(c)}개, K={K_DET}")
    cluster = KMeans(n_clusters=K_DET, n_init=10, random_state=0).fit(c).cluster_centers_
    others = np.array([1, 1, 1, 1, 0, 0, 0, 0])[np.newaxis].repeat(K_DET, axis=0)
    cluster = np.concatenate([cluster, others], axis=1)
    np.save(det_out, cluster)
    print(f"저장 {det_out} shape={cluster.shape} z[{cluster[:,2].min():.2f},{cluster[:,2].max():.2f}]")
