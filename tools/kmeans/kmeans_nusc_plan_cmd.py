# 수정(2026-10-09): nuScenes 명령별 계획 앵커 — "명령으로 계획 모드 선택"(사용자 지시) 용.
#   [근거] SparseDrive tools/kmeans/kmeans_plan.py 는 gt_ego_fut_cmd(argmax) 별로 따로 군집해 (3, K, T, 2) 를 만들고,
#          추론 시 decoder 가 GT 명령으로 모드 묶음을 고른다(SparseDrive decoder.py:170-187, 우리 plan/decoder.py:179-187 동일 기능).
#   [형식] models/decoder/temporal_bank/plan_bank.py 는 4D 앵커 (C, M, T, 2) 를 (C*M, T*2) 로 펼친다 → config ego_fut_cmd=3, ego_fut_mode=16.
#          총 48 모드로 기존(1×48)과 쿼리 수 동일. 누적 궤적 표현(cumsum)·전 구간 유효 표본만 사용은 kmeans_nusc_plan.py 와 같다.
#   사용법: python tools/kmeans/kmeans_nusc_plan_cmd.py
import os
import mmcv
import numpy as np
from sklearn.cluster import KMeans

K_PER_CMD, N_CMD, FUT_TS = 16, 3, 6
OUT = f'data/kmeans/nusc_plan_temp_{N_CMD}x{K_PER_CMD}.npy'

infos = mmcv.load('data/infos/nuscenes_infos_train.pkl')['infos']
by_cmd = [[] for _ in range(N_CMD)]
skip = 0
for info in infos:
    t = np.asarray(info.get('gt_ego_fut_trajs', []), np.float64)
    m = np.asarray(info.get('gt_ego_fut_masks', []), np.float64)
    if t.shape[:1] != (FUT_TS,) or m.reshape(-1).min() < 1:
        skip += 1; continue
    cmd = int(np.asarray(info['gt_ego_fut_cmd']).argmax())
    by_cmd[cmd].append(np.cumsum(t, 0))
anchor = np.zeros((N_CMD, K_PER_CMD, FUT_TS, 2), np.float32)
for c in range(N_CMD):
    X = np.stack(by_cmd[c]).reshape(len(by_cmd[c]), -1)
    assert len(X) >= K_PER_CMD, f'명령 {c} 표본 {len(X)} < K'
    anchor[c] = KMeans(n_clusters=K_PER_CMD, n_init=10, random_state=0).fit(X).cluster_centers_.reshape(K_PER_CMD, FUT_TS, 2)
    end = anchor[c, :, -1]
    print(f'cmd {c}: 표본 {len(X)}, 앵커 종점 x 평균 {end[:,0].mean():+.2f} m, y 평균 {end[:,1].mean():.2f} m')
print(f'제외 {skip}')
np.save(OUT, anchor); print('->', OUT, anchor.shape)
