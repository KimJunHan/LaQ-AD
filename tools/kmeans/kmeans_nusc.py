# 수정: nuScenes 전용 det/map 앵커 생성 (기존 b2d 앵커가 CARLA용이라 nuScenes 검출 저하 → 근본수정).
#   로직은 kmeans_det.py / kmeans_map.py 와 동일하되 입력만 nuScenes train pkl.
import os
import mmcv
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

os.makedirs('data/kmeans', exist_ok=True)
infos = mmcv.load('data/infos/nuscenes_infos_train.pkl')['infos']
print(f'nuScenes train 샘플 {len(infos)}개')

# ===== det anchor: GT box (x,y,z) 위치 k-means (kmeans_det.py 동일) =====
K_DET = 900
DIS_THRESH = 55
Z_MIN, Z_MAX = -5.0, 3.0  # 수정: point_cloud_range z. z 이상치(예: 77m) 박스 제거
center = []
for info in tqdm(infos, desc='det boxes'):
    boxes = np.asarray(info['gt_boxes'], dtype=np.float64)[:, :3]
    if len(boxes) == 0:
        continue
    dist = np.linalg.norm(boxes[:, :2], axis=1)
    keep = (dist < DIS_THRESH) & (boxes[:, 2] > Z_MIN) & (boxes[:, 2] < Z_MAX)
    center.append(boxes[keep])
center = np.concatenate(center, axis=0)
print(f'det 클러스터링 대상 {len(center)}개 박스, K={K_DET} (수 분 소요)')
cluster = KMeans(n_clusters=K_DET, n_init=10, random_state=0).fit(center).cluster_centers_
# b2d 와 동일: [x,y,z] + [W,L,H,SIN_YAW,COS_YAW,VX,VY,VZ]=[1,1,1,1,0,0,0,0]
others = np.array([1, 1, 1, 1, 0, 0, 0, 0])[np.newaxis].repeat(K_DET, axis=0)
det_anchor = np.concatenate([cluster, others], axis=1)
np.save(f'data/kmeans/nusc_det_{K_DET}.npy', det_anchor)
print(f'  저장 nusc_det_{K_DET}.npy shape={det_anchor.shape}, xyz범위 '
      f'x[{cluster[:,0].min():.1f},{cluster[:,0].max():.1f}] '
      f'y[{cluster[:,1].min():.1f},{cluster[:,1].max():.1f}] '
      f'z[{cluster[:,2].min():.1f},{cluster[:,2].max():.1f}]')

# ===== map anchor: polyline 중심 k-means → 직선 vec (kmeans_map.py 동일) =====
K_MAP = 100
num_sample = 20
y_ratio = 0.7
rng = np.random.default_rng(0)
mcenter = []
for info in tqdm(infos[::100], desc='map polylines'):
    ma = info.get('map_annos', {})
    for label, geoms in ma.items():
        for geom in geoms:
            g = np.asarray(geom, dtype=np.float64)
            if len(g) > 0:
                mcenter.append(g.mean(axis=0))
mcenter = np.stack(mcenter, axis=0)
print(f'map 클러스터링 대상 {len(mcenter)}개 polyline 중심, '
      f'범위 x[{mcenter[:,0].min():.1f},{mcenter[:,0].max():.1f}] '
      f'y[{mcenter[:,1].min():.1f},{mcenter[:,1].max():.1f}]')
mcluster = KMeans(n_clusters=K_MAP, n_init=10, random_state=0).fit(mcenter).cluster_centers_
vecs = []
for k in range(K_MAP):
    length = rng.uniform(8, 20)
    if rng.uniform() < y_ratio:
        delta_y = np.linspace(-length / 2, length / 2, num_sample)
        delta_x = np.zeros([num_sample])
    else:
        delta_y = np.zeros([num_sample])
        delta_x = np.linspace(-length / 2, length / 2, num_sample)
    delta = np.stack([delta_x, delta_y], axis=-1)
    vecs.append(mcluster[k, np.newaxis] + delta)
vecs = np.array(vecs)
np.save(f'data/kmeans/nusc_map_{K_MAP}.npy', vecs)
print(f'  저장 nusc_map_{K_MAP}.npy shape={vecs.shape}')
