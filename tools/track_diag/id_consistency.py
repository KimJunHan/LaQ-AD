# 수정(2026-10-08): 같은 ID 가 연속 프레임에서 같은 물체를 가리키는지(global 변위) 검사 — ID 정렬 버그 판별.
import pickle, sys, numpy as np
from pyquaternion import Quaternion
R = sys.argv[1]; THR = 0.2
res = pickle.load(open(R, 'rb'))
inf = pickle.load(open('data/infos/nuscenes_infos_val.pkl', 'rb')); inf = inf['infos'] if isinstance(inf, dict) else inf
def T(x): return x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x)
def to_global(xy3, info):
    p = Quaternion(info['lidar2ego_rotation']).rotation_matrix @ xy3.T + np.array(info['lidar2ego_translation'])[:, None]
    p = Quaternion(info['ego2global_rotation']).rotation_matrix @ p + np.array(info['ego2global_translation'])[:, None]
    return p.T
disp = []; vel_pred_err = []; prev = None; ps = None
for r, info in zip(res, inf):
    p = r['img_bbox']; b = T(p['boxes_3d']); s = T(p['scores_3d']); ids = T(p['instance_ids']).astype(int)
    g = to_global(b[:, :3], info); m = s >= THR
    cur = {t: (g[k], info['timestamp']) for k, t in enumerate(ids) if m[k]}
    if prev is not None and info['scene_token'] == ps:
        for t, (x, ts) in cur.items():
            if t in prev:
                disp.append(np.linalg.norm(x[:2] - prev[t][0][:2]))
    prev, ps = cur, info['scene_token']
d = np.array(disp)
print('연속 프레임 동일 ID 쌍 %d, global 변위 m: median %.2f p90 %.2f p99 %.2f  >2m %.2f%%  >5m %.2f%%' %
      (len(d), np.median(d), np.percentile(d, 90), np.percentile(d, 99), 100*(d > 2).mean(), 100*(d > 5).mean()))
