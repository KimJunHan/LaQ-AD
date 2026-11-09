# 수정(2026-10-08): ID 남발 가설의 재학습-없는 검증. 박스·점수는 그대로 두고 track ID 만 다시 잇는다.
#   mode=relink : 이번 프레임에 처음 나온(fresh) ID 의 박스가, 이번 프레임에 이어지지 못한(lost) 기존 트랙의
#                 예측 위치(마지막 위치 + 속도*dt) 근처면 그 트랙 ID 를 물려받는다(탐욕, 거리순).
#   mode=dedup  : relink + fresh ID 박스가 살아있는 기존 트랙 박스와 같은 클래스 1m 이내면 fresh 쪽 제거.
#   하이퍼파라미터는 val 로 튜닝하지 않고 CenterPoint(Yin et al., CVPR 2021) nuScenes 기본값으로 고정한다.
import json, sys, pickle, numpy as np
src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
GATE = dict(car=4, truck=4, bus=5.5, trailer=3, pedestrian=1, motorcycle=13, bicycle=3)  # CenterPoint 속도오차 게이트(m)
MAX_AGE = 3  # CenterPoint max_age(프레임)
DUP_R = 1.0
sub = json.load(open(src)); res = sub['results']
inf = pickle.load(open('data/infos/nuscenes_infos_val.pkl', 'rb')); inf = inf['infos'] if isinstance(inf, dict) else inf
inf = sorted(inf, key=lambda x: (x['scene_token'], x['timestamp']))
stat = dict(relinked=0, dedup=0, fresh=0)
prev_scene = None
for info in inf:
    tok = info['token']; boxes = res.get(tok, [])
    if info['scene_token'] != prev_scene:
        tracks = {}; alias = {}; seen = set(); prev_scene = info['scene_token']
    ts = info['timestamp'] * 1e-6
    for b in boxes: b['_raw'] = b['tracking_id']; b['tracking_id'] = alias.get(b['_raw'], b['_raw'])
    present = {b['tracking_id'] for b in boxes}
    fresh = [b for b in boxes if b['_raw'] not in seen]
    stat['fresh'] += len(fresh)
    # (1) lost 트랙으로 재연결
    lost = {t: v for t, v in tracks.items() if t not in present}
    cand = []
    for bi, b in enumerate(fresh):
        p = np.array(b['translation'][:2])
        for t, v in lost.items():
            if v['cls'] != b['tracking_name']: continue
            pred = v['pos'] + v['vel'] * (ts - v['ts'])
            d = np.linalg.norm(p - pred)
            if d < GATE[b['tracking_name']]: cand.append((d, bi, t))
    used_b, used_t = set(), set()
    for d, bi, t in sorted(cand):
        if bi in used_b or t in used_t: continue
        used_b.add(bi); used_t.add(t); b = fresh[bi]
        alias[b['_raw']] = t; b['tracking_id'] = t; stat['relinked'] += 1
    # (2) 중복 제거
    if mode == 'dedup':
        alive = [b for b in boxes if b['_raw'] in seen or b['_raw'] in alias]
        drop = set()
        for bi, b in enumerate(fresh):
            if bi in used_b: continue
            p = np.array(b['translation'][:2])
            for a in alive:
                if a['tracking_name'] == b['tracking_name'] and np.linalg.norm(p - np.array(a['translation'][:2])) < DUP_R:
                    drop.add(id(b)); break
        stat['dedup'] += len(drop)
        boxes = [b for b in boxes if id(b) not in drop]
    for b in boxes: seen.add(b['_raw'])
    # 트랙 상태 갱신
    for b in boxes:
        tracks[b['tracking_id']] = dict(pos=np.array(b['translation'][:2]), vel=np.array(b['velocity'][:2]),
                                        ts=ts, cls=b['tracking_name'], age=0)
    for t in list(tracks):
        if t not in {b['tracking_id'] for b in boxes}:
            tracks[t]['age'] += 1
            if tracks[t]['age'] > MAX_AGE: del tracks[t]
    for b in boxes: b.pop('_raw', None)
    res[tok] = boxes
print(mode, stat)
json.dump(sub, open(dst, 'w'))
