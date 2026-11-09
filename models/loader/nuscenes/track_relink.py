# 수정(2026-10-15): 경로 이동 mmdet3d_plugin/datasets/track_relink.py → models/loader/nuscenes/track_relink.py.
# 수정(2026-10-08): 추적 평가용 ID 재연결 후처리(사용자 결정: 평가 시 적용).
#   [왜] 시간 쿼리가 추적 중인 물체를 놓치고 신규 쿼리가 같은 물체를 새 ID 로 재검출하는 끊김이
#   Run9 IDS 5,090 의 73% 를 차지했다(tools/track_diag/diag_ids.py). 박스·점수는 그대로 두고
#   이번 프레임에 처음 나온(fresh) ID 를, 이번 프레임에 이어지지 못한(lost) 기존 트랙의 예측 위치
#   (마지막 global 위치 + 예측 속도 * dt) 근처면 그 트랙 ID 로 잇는다. 탐욕 매칭, 거리 오름차순.
#   [검증] Run9: AMOTA 0.3674 -> 0.3988, IDS 5,090 -> 588 (재연결 전 수치 재현 확인).
#   [하이퍼파라미터] val 로 튜닝하지 않았다. CenterPoint(Yin et al., CVPR 2021) nuScenes 트래커의
#   클래스별 속도오차 게이트와 max_age=3 을 그대로 쓴다.
#   [공시] HiP-AD/SparseDrive 는 이 후처리 없이 쿼리 전파 ID 만 쓴다 → 논문에 명시하고 재연결 전
#   수치(tracking_raw/*)를 함께 보고한다. 중복 제거 변형은 AMOTA 를 낮춰(0.3931) 채택하지 않았다.
import numpy as np

GATE = dict(car=4.0, truck=4.0, bus=5.5, trailer=3.0, pedestrian=1.0, motorcycle=13.0, bicycle=3.0)
MAX_AGE = 3


def relink_track_ids(nusc_results, data_infos, gate=GATE, max_age=MAX_AGE):
    """nuScenes tracking 제출 dict(token -> box list)의 tracking_id 를 장면 단위로 재연결한다(in-place).

    data_infos 는 scene_token / timestamp(us) / token 을 가진 val info 목록. 반환: 통계 dict.
    """
    order = sorted(range(len(data_infos)),
                   key=lambda i: (data_infos[i]["scene_token"], data_infos[i]["timestamp"]))
    stat = dict(fresh=0, relinked=0)
    prev_scene = None
    for i in order:
        info = data_infos[i]
        boxes = nusc_results.get(info["token"], [])
        if info["scene_token"] != prev_scene:
            # 장면 경계: 트랙 상태를 전부 비운다(장면 사이 ID 이월 없음).
            tracks, alias, seen = {}, {}, set()
            prev_scene = info["scene_token"]
        ts = info["timestamp"] * 1e-6
        raw = [b["tracking_id"] for b in boxes]
        for b, r in zip(boxes, raw):
            b["tracking_id"] = alias.get(r, r)
        present = {b["tracking_id"] for b in boxes}
        fresh = [k for k, r in enumerate(raw) if r not in seen]
        stat["fresh"] += len(fresh)

        lost = {t: v for t, v in tracks.items() if t not in present}
        cand = []
        for k in fresh:
            b = boxes[k]
            g = gate.get(b["tracking_name"])
            if g is None:
                continue
            p = np.asarray(b["translation"][:2])
            for t, v in lost.items():
                if v["cls"] != b["tracking_name"]:
                    continue
                d = float(np.linalg.norm(p - (v["pos"] + v["vel"] * (ts - v["ts"]))))
                if d < g:
                    cand.append((d, k, t))
        used_k, used_t = set(), set()
        for d, k, t in sorted(cand, key=lambda x: x[0]):
            if k in used_k or t in used_t:
                continue
            used_k.add(k)
            used_t.add(t)
            alias[raw[k]] = t
            boxes[k]["tracking_id"] = t
            stat["relinked"] += 1

        seen.update(raw)
        cur = set()
        for b in boxes:
            t = b["tracking_id"]
            cur.add(t)
            tracks[t] = dict(pos=np.asarray(b["translation"][:2], dtype=float),
                             vel=np.asarray(b["velocity"][:2], dtype=float),
                             ts=ts, cls=b["tracking_name"], age=0)
        for t in list(tracks):
            if t not in cur:
                tracks[t]["age"] += 1
                if tracks[t]["age"] > max_age:
                    del tracks[t]
    return stat
