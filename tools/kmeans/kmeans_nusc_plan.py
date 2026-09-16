# 수정(2026-08-11): nuScenes 전용 계획(plan) 앵커 생성.
#
#   [배경] nuScenes config 가 CARLA 용 앵커(b2d_plan_spat_6x8_5m.npy)를 그대로 참조하고
#          있었다. foundation 이 HiP-AD 이므로 학습 커리큘럼도 HiP-AD 원본을 따른다 —
#          stage1 에 plan/ego 를 task_select 에 포함(손실 0, 순전파만), stage2 에서 켠다.
#          계획 쿼리가 공유 디코더의 쿼리 풀에 존재해야 인지 쿼리와 상호작용하므로,
#          손실이 0 인 stage1 에서도 앵커는 실주행 분포여야 한다.
#          (nuScenes 는 계획 '평가'를 하지 않는다 — 평가는 det/map/track/motion 만.)
#
#   [규약] models/plan/instance_bank.py 의 로딩 규칙:
#            3D 앵커 (M, T, 2) -> reshape(M, T*2)
#            4D 앵커 (C, M, T, 2) -> reshape(C*M, T*2)
#          config 는 ego_fut_cmd=1, ego_fut_mode=48 이므로 총 48 모드가 필요하다.
#          b2d 는 (6,8,6,2)=48 이었다. nuScenes 는 명령이 3종뿐이라 명령축을 따로 두지 않고
#          (48, 6, 2) 로 만든다 — 평탄화 결과가 (48,12) 로 동일하다.
#
#   [표현] b2d 앵커는 누적 궤적이다(mode0 y: 4.97, 9.86, ... 28.82).
#          nuScenes gt_ego_fut_trajs 는 스텝당 변위이므로 cumsum 후 군집화한다.
#
#   사용법: python tools/kmeans/kmeans_nusc_plan.py
import os
import mmcv
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

K_PLAN = 48          # config 의 ego_fut_mode
FUT_TS = 6           # nuScenes 자차 GT 지평: 2Hz x 6스텝 = 3초
OUT = f'data/kmeans/nusc_plan_temp_{K_PLAN}.npy'

os.makedirs('data/kmeans', exist_ok=True)
infos = mmcv.load('data/infos/nuscenes_infos_train.pkl')['infos']
print(f'nuScenes train 샘플 {len(infos)}개')

trajs = []
n_skip = 0
for info in tqdm(infos, desc='ego trajs'):
    t = np.asarray(info.get('gt_ego_fut_trajs', []), dtype=np.float64)
    m = np.asarray(info.get('gt_ego_fut_masks', []), dtype=np.float64)
    if t.shape[:1] != (FUT_TS,) or m.shape[:1] != (FUT_TS,):
        n_skip += 1
        continue
    # 전 구간이 유효한 표본만 사용한다. 부분 마스크를 0 으로 채우면
    # "정지" 궤적이 인위적으로 늘어 군집이 원점으로 쏠린다.
    if m.reshape(FUT_TS, -1).min() < 1:
        n_skip += 1
        continue
    trajs.append(np.cumsum(t, axis=0))   # 변위 -> 누적 (b2d 앵커와 동일 표현)

trajs = np.stack(trajs, axis=0)
print(f'군집화 대상 {len(trajs)}개 (제외 {n_skip}개), K={K_PLAN}')
print(f'  종점 거리 중앙값 {np.median(np.linalg.norm(trajs[:, -1], axis=-1)):.2f} m, '
      f'최대 {np.linalg.norm(trajs[:, -1], axis=-1).max():.2f} m')

flat = trajs.reshape(len(trajs), -1)     # (N, T*2)
centers = KMeans(n_clusters=K_PLAN, n_init=10, random_state=0).fit(flat).cluster_centers_
anchor = centers.reshape(K_PLAN, FUT_TS, 2).astype(np.float32)

np.save(OUT, anchor)
d = np.linalg.norm(anchor[:, -1], axis=-1)
print(f'저장 {OUT} shape={anchor.shape}')
print(f'  모드 종점 거리 {d.min():.2f} ~ {d.max():.2f} m (평균 {d.mean():.2f})')
print(f'  스텝별 |값| 평균: {np.round(np.abs(anchor).mean(axis=(0, 2)), 2)}')
