# 수정(2026-10-08): AMOTA 격차 원인 검증 — 저장된 results.pkl 로 ID 전환(IDS) 원인을 분류한다(재학습·GPU 불필요).
#   GT 트랙별로 연속 프레임의 매칭 예측 ID 가 바뀌면 전환으로 보고, 이전 ID 의 행방으로 분류한다:
#   alive_dup  — 이전 ID 가 이번 프레임에도 점수>=thr 로 살아있음(다른 박스 = 중복/선두교체)
#   below_thr  — 이전 ID 가 top-300 안에 있으나 점수<thr
#   gone       — 이전 ID 가 top-300 출력에서 사라짐(시간 뱅크 탈락)
import pickle, sys, numpy as np, collections
R = sys.argv[1]; THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
CLS = ['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']
TRK = {'car','truck','bus','trailer','motorcycle','bicycle','pedestrian'}
res = pickle.load(open(R, 'rb'))
inf = pickle.load(open('data/infos/nuscenes_infos_val.pkl', 'rb')); inf = inf['infos'] if isinstance(inf, dict) else inf
assert len(res) == len(inf)
def T(x): return x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x)
cnt = collections.Counter(); age_new = collections.Counter(); dup_dist = []; gone_prev_score = []
prev = {}; prev_scene = None; prev_ids_all = None; n_match = 0
for i, (r, info) in enumerate(zip(res, inf)):
    p = r['img_bbox']; b = T(p['boxes_3d']); s = T(p['scores_3d']); l = T(p['labels_3d']).astype(int); ids = T(p['instance_ids']).astype(int)
    if info['scene_token'] != prev_scene: prev = {}; prev_ids_all = None; first_seen = {}
    prev_scene = info['scene_token']
    for t in ids:
        first_seen.setdefault(t, i)
    keep = (s >= THR) & np.array([CLS[k] in TRK for k in l], dtype=bool)
    gb = info['gt_boxes']; gn = info['gt_names']; gi = info['instance_inds']
    gmask = np.array([n in TRK for n in gn], dtype=bool) & np.asarray(info['valid_flag'], dtype=bool)
    # 클래스별 탐욕 매칭(점수순, 2m) — 평가 임계와 동일
    used = set(); cur = {}
    for j in np.argsort(-s):
        if not keep[j]: continue
        best, bd = None, 2.0
        for g in np.where(gmask)[0]:
            if g in used or gn[g] != CLS[l[j]]: continue
            d = np.linalg.norm(gb[g, :2] - b[j, :2])
            if d < bd: best, bd = g, d
        if best is not None: used.add(best); cur[gi[best]] = (ids[j], j)
    n_match += len(cur)
    id2idx = {t: k for k, t in enumerate(ids)}
    for g, (tid, j) in cur.items():
        if g in prev and prev[g] != tid:
            pid = prev[g]
            if pid in id2idx:
                k = id2idx[pid]
                if keep[k]:
                    cnt['alive_dup'] += 1; dup_dist.append(np.linalg.norm(b[k, :2] - b[j, :2]))
                else: cnt['below_thr'] += 1
            else:
                cnt['gone'] += 1
            cnt['IDS'] += 1
            age_new['new_id_fresh' if first_seen[tid] == i else 'new_id_old'] += 1
    for g, (tid, j) in cur.items(): prev[g] = tid
    # 트랙 소실 후 재등장: prev 는 마지막 매칭을 유지(평가와 동일하게 공백을 건너 비교)
print(f'thr={THR} matched={n_match}', dict(cnt), dict(age_new))
if dup_dist:
    d = np.array(dup_dist); print('alive_dup: 이전ID 박스와 새 매칭 박스 거리 m: median %.2f  <1m %.1f%%  <2m %.1f%%' % (np.median(d), 100*(d<1).mean(), 100*(d<2).mean()))
