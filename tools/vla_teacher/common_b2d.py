# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   B2D 교사 VLM 정답('v2-sensor-b2d', 고정 길이 547 토큰) 공용 모듈. nuScenes 판(common_v2.py)은 수정하지 않는다.
#
#   설계 원칙: nuScenes v2-sensor 정답과 "줄·필드·토큰 수"가 완전히 같다 → laq_cache_layout(7 폴더 고정 위치), answer_grammar
#   (위치별 문법 제약), extract_teacher_cache_v2(캐시 자르기)를 B2D 에서도 고치지 않고 쓸 수 있다
#   (이 모듈을 sys.modules["common_v2"] 로 끼워 넣는 B2D 실행 래퍼에서 — b2d_teacher_shim.py).
#   그래서 구조 토큰 문자열(<none> <na> <_> <static> <pc> <pc_on> <pc_none> <div> <bnd> <at0> <at10> <at20> <free>)은
#   nuScenes 와 같은 글자를 쓰고, B2D 에서의 뜻은 아래(GRAMMAR_B2D, 지시문)에 따로 정의한다.
#
#   nuScenes ↔ B2D 대응(같은 필드, 다른 정의 — 문서화):
#     클래스     : nuScenes 10 → B2D 9 (car van truck bicycle traffic_sign traffic_cone traffic_light pedestrian others)
#                  새 토큰 <van> <sign> <light> <others>; 겹치는 것은 nuScenes 와 같은 글자(<car> <truck> <bike> <cone> <ped>).
#     진행방위   : traffic_cone(경고 시설물 포함)·others(도로 위 잔해) = <na>(방향 정의 없음). traffic_sign·traffic_light 는 면이 향하는
#                  방향이 정의되어 있으므로 값 있음(어느 차로에 적용되는 표지/신호인지가 그 방향에 들어 있다).
#     속도       : B2D 박스 속도(gt_boxes[7:9], 라이다 좌표)는 시뮬레이터 값이라 NaN 이 없다(전수 확인은 생성기 stats).
#     <pc> 칸    : B2D 지도에는 횡단보도 클래스가 없다. 대신 "신호등·정지 표지 트리거 영역(map_infos trigger_volumes;
#                  자차가 멈춰야 하는 정지선 쪽 영역)" 중 20 m 안 최근접의 방위·거리. 영역 안이면 <pc_on>, 없으면 <pc_none>.
#     <div>/<bnd>: B2D 지도 클래스 = Broken / Solid / SolidSolid(차선 표시) + Center(차로 중심선). 도로 가장자리 클래스가 없다.
#                  <div> DD = 차선 표시 선 수(Broken+Solid+SolidSolid), <bnd> DD = 넘으면 안 되는 표시 선 수(Solid+SolidSolid).
#                  단면 <atY>: div_l/div_r = 좌/우 최근접 차선 표시(종류 무관 = 자기 차로 가장자리),
#                              bnd_l/bnd_r = 좌/우 최근접 넘으면 안 되는 표시(Solid/SolidSolid).
#     <free>     : 자차 폭 직선 회랑 0..30 m 에서 처음 만나는 "객체 박스"의 y. B2D 에는 물리적 도로 경계 클래스가 없어 지도 선은 쓰지
#                  않는다(차선 표시는 장애물이 아니다 — B2D 추월 시나리오는 실선을 합법적으로 넘는다).
#     차로변경   : (b) 미래 꺾은선이 차선 표시(0,1,2)를 가로지름 + (c) 출발 차로 중심선(Center) 기준 횡변위 |Δd| ≥ 1.6 m.
#                  두 신호 방향 일치 → lane change, 한 신호만(또는 방향 충돌) → uncertain(답 go straight, 가중 0.3). nuScenes 의
#                  (a) 지도 위상 신호(map-expansion lane 연결)는 B2D 가공 지도에 연결 정보가 없어 쓰지 않는다.
#     명령       : B2D 경로 계획기 RoadOption 6 종(LEFT, RIGHT, STRAIGHT, LANEFOLLOW, CHANGELANELEFT, CHANGELANERIGHT) 그대로 문장화.
#                  + 목표점(target point, HiP-AD B2D 학생 입력과 같은 command_far_xy, 라이다 좌표) 줄을 프롬프트에 넣는다(고정폭).
#     자차 상태  : 시뮬레이터 값(결측 없음). 고정폭 줄(B안) 그대로. 종가속·요레이트 정의는 ego_inputs_b2d 주석.
#     행동 토큰  : B2D 전용 코드북(build_b2d_action_codebook.py, train 키프레임 0.5 s 조각, K-disk ≈ 2048, B2D 자차 크기).
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common_v2 as V2  # noqa: E402  (nuScenes v2 — 수정 금지, 형식 함수·토큰 표 재사용만)
from common_v2 import (SEC_TOK, LON_TOK, LAT_TOK, HEAD_TOK, TTC_TOK, AT_TOK, LANE_Y, MISC_TOK, SENSOR_TOK,  # noqa: E402,F401
                       TOK_SEC, TOK_LON, TOK_LAT, TOK_HEAD, TOK_TTC, _r1, _r1c, _sn, _un, _fill, _rect, _eval_yaw,
                       ttc_bucket, merge_spans, loss_mask_spans, _SN_RE, _UN_RE, _I2_RE, _ACT_RE, _Cur, _cross_x,
                       MAX_OBJECTS, CORRIDOR_HALF, TTC_MAX, CLOSING_MIN, VRU_RANGE, STRAIGHT_MIN_LEN, TIER_NAMES,
                       SLOT_FIELDS_NUMERIC, SLOT_FIELDS_DIRONLY, MAP_LINE_FIELDS, FUT_STEPS, FREE_CAP, TTC_T,
                       EGO_MODES, SPEED_MODES, slot_tokens, _seg_dist)
from common import sector, bearing_xy, heading_series, STOP_V, DV_TH, TURN_TH_DEG, MIN_PATH, DT  # noqa: E402,F401

ANSWER_FORMAT = "v2-sensor-b2d"
DATASET = "b2d"

# ---------------------------------------------------------------- 클래스·토큰
DET_CLASSES = ["car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light", "pedestrian", "others"]  # configs/b2d/stage1/train.py
CLS_TOK = {"car": "<car>", "van": "<van>", "truck": "<truck>", "bicycle": "<bike>", "traffic_sign": "<sign>",
           "traffic_cone": "<cone>", "traffic_light": "<light>", "pedestrian": "<ped>", "others": "<others>"}
TOK_CLS = {v: k for k, v in CLS_TOK.items()}
NO_HEADING = ("traffic_cone", "others")            # 방향 정의 없음 → <na>
VRU = ("pedestrian", "bicycle")                    # B2D 에는 motorcycle 클래스가 없다
OBJ_RANGE = 55.0                                   # B2D 학습 파이프라인 CircleObjectRangeFilter 55 m
MOVING_TH = 0.5
MAP_CLASSES = ["Broken", "Solid", "SolidSolid", "Center"]   # 지도 라벨 0..3
MARK_IDS = (0, 1, 2)                               # 차선 표시(종류 무관)
NOCROSS_IDS = (1, 2)                               # 넘으면 안 되는 표시
CENTER_ID = 3
TC_RANGE = 20.0                                    # m, 신호/정지 트리거 영역(<pc> 칸) 탐색 반경 = nuScenes 횡단보도 반경과 같음

# B2D 명령(RoadOption, info['command_near'] 1..6; -1 → 4 = dataset.command2hot 과 같은 규칙)
#   문장은 토큰 수가 모두 같도록 고른다('\nCommand: <문장>.' = 7 토큰, Qwen2 토크나이저 실측): 캐시 문맥 길이 고정(B안).
#   ('follow lane'(6)·'change lane left'(7)처럼 섞으면 표본마다 문맥 길이가 달라진다 — check_teacher_gt_b2d.py 가 전수 검사.)
CMD_TEXT_B2D = {1: "turn left ahead", 2: "turn right ahead", 3: "go straight ahead", 4: "follow the lane",
                5: "change lane left", 6: "change lane right"}


def norm_command(c):
    c = int(c)
    return 4 if c < 0 else c


# 행동 코드북 크기는 B2D 코드북 파일에서 정한다(set_action_k). 기본값은 생성 전 자리 표시(문법 검사 때 반드시 덮어쓴다).
ACTION_K = int(os.environ.get("B2D_ACTION_K", "0")) or None
ACTION_TOK = []


def set_action_k(k):
    global ACTION_K, ACTION_TOK, SPECIAL_TOKENS
    ACTION_K = int(k)
    ACTION_TOK = [f"<action_{i}>" for i in range(ACTION_K)]
    SPECIAL_TOKENS = _special_tokens()
    return ACTION_K


def _special_tokens():
    toks = (list(CLS_TOK.values()) + list(SEC_TOK.values()) + MISC_TOK + list(LON_TOK.values()) + list(LAT_TOK.values()) +
            ACTION_TOK + SENSOR_TOK)
    assert len(toks) == len(set(toks))
    return toks


SPECIAL_TOKENS = _special_tokens()

# ---------------------------------------------------------------- 자차(B2D: Lincoln MKZ 2020, info['ego_size'] = [폭, 길이, 높이])
EGO_L, EGO_W = 4.89238167, 1.83671331   # 생성기가 info['ego_size'] 와 전 표본 일치를 검사한다
EGO_FWD = None                          # 라이다 원점 → 자차 박스 중심 전방 오프셋(m). 생성기가 lidar2ego 로 정해 set_ego_geom 으로 넣는다
FREE_HALF_W = EGO_W / 2
TTC_GAP = None                          # m — gen_teacher_gt_b2d.py --explore_ttc (train 전용) 결과로 고정(아래 set_ttc_gap)


def set_ego_geom(fwd):
    global EGO_FWD
    EGO_FWD = float(fwd)


def set_ttc_gap(g):
    global TTC_GAP
    TTC_GAP = float(g)


# ---------------------------------------------------------------- 표본 묶음(sample dict) 규약
# 생성기가 Bench2DriveDataset.get_data_info 결과에서 뽑아 만드는 표본 s (라이다 좌표: +x 오른쪽, +y 전방 — 생성기가 카메라 투영으로 검사):
#   s["boxes"]     (N, 9) remap 후 박스 [x, y, z, l, w, h, yaw, vx, vy]  (num_points>0 인 것만 = 학습 GT 와 같음)
#   s["names"]     (N,)   클래스 이름(NameMapping 후)
#   s["agent_fut"] (N, 6, 2) 0.5 s 증분, s["agent_mask"] (N, 6)
#   s["ego_fut"]   (6, 2) 0.5 s 증분, s["ego_mask"] (6,)
#   s["map"]       {라벨: [ (P,2) 폴리라인 ]} 라이다 좌표, ROI x±15 / y±30
#   s["tc"]        [(P,2) 트리거 영역 다각형(라이다 xy)] 신호등·정지 표지
def future_complete(s):
    return bool((np.asarray(s["ego_mask"]) == 1).all())


def fut_cum(s):
    return np.asarray(s["ego_fut"], np.float64).cumsum(0)[:6, :2]


def corridor_path(s, v0):
    if future_complete(s):
        return np.concatenate([np.zeros((1, 2)), fut_cum(s)], 0), "gt_future"
    L = max(3.0 * v0, STRAIGHT_MIN_LEN)
    return np.array([[0.0, 0.0], [0.0, L]]), "straight_cmd"


def _obj_ok(name, x, y):
    return name in DET_CLASSES and (x * x + y * y) ** 0.5 <= OBJ_RANGE


def select_objects(s, v0):
    """common_v2.select_objects 와 같은 중요도 규칙(경로 회랑 → 접근 TTC → VRU 20 m → 근접), 최대 8, 선택 후 거리순."""
    boxes = np.asarray(s["boxes"], np.float64)
    names = list(s["names"])
    path, path_src = corridor_path(s, v0)
    vego = np.array([0.0, v0])
    rows = []
    for j in range(len(boxes)):
        name = str(names[j])
        x, y = float(boxes[j, 0]), float(boxes[j, 1])
        if not _obj_ok(name, x, y):
            continue
        dist = (x * x + y * y) ** 0.5
        vx, vy = float(boxes[j, 7]), float(boxes[j, 8])
        fin = bool(np.isfinite(vx) and np.isfinite(vy))
        if not fin:
            vx = vy = 0.0
        spd = (vx * vx + vy * vy) ** 0.5
        moving = spd >= MOVING_TH
        p = np.array([x, y])
        cd = _seg_dist(p, path)
        rr = float((p @ (np.array([vx, vy]) - vego)) / max(dist, 1e-6))
        ttc = dist / -rr if (moving and rr <= -CLOSING_MIN) else None
        if cd <= CORRIDOR_HALF:
            tier, key = 1, dist
        elif ttc is not None and ttc <= TTC_MAX:
            tier, key = 2, ttc
        elif name in VRU and dist <= VRU_RANGE:
            tier, key = 3, dist
        else:
            tier, key = 4, dist
        rows.append(dict(name=name, sector=sector(bearing_xy(x, y)), lat=_r1(x), lon=_r1(y), dist=round(dist, 1),
                         vel_sector=sector(bearing_xy(vx, vy)) if moving else None, speed=_r1(spd) if moving else None,
                         reason=TIER_NAMES[tier], corridor_dist=round(cd, 2), ttc=None if ttc is None else round(ttc, 2),
                         x=round(x, 3), y=round(y, 3), box=[round(float(v), 4) for v in boxes[j, :7]],
                         vel_finite=fin, instance_ind=int(s["ids"][j]) if "ids" in s else -1, box_index=j,
                         _k=(tier, key, dist), _d=dist))
    n_in = len(rows)
    rows.sort(key=lambda r: r["_k"])
    sel = rows[:MAX_OBJECTS]
    sel.sort(key=lambda r: r["_d"])
    nearest8 = {id(r) for r in sorted(rows, key=lambda r: r["_d"])[:MAX_OBJECTS]}
    meta = dict(path_src=path_src, n_tier={TIER_NAMES[t]: sum(r["_k"][0] == t for r in rows) for t in TIER_NAMES},
                n_selected_not_in_nearest8=sum(id(r) not in nearest8 for r in sel))
    for r in sel:
        r.pop("_k"); r.pop("_d")
    return sel, n_in, meta


# ---------------------------------------------------------------- TTC(common_v2 와 같은 규칙, B2D 자차 크기·오프셋)
def ego_track(s):
    if not future_complete(s):
        return None
    if EGO_FWD is None:
        raise RuntimeError("EGO_FWD 미설정(set_ego_geom)")
    fut = fut_cum(s)
    yaw = _eval_yaw(fut)
    v = fut[5] - fut[4]
    pos = [np.zeros(2)] + list(fut) + [fut[5] + v * k for k in range(1, 7)]
    yws = [np.pi / 2] + list(yaw) + [yaw[-1]] * 6
    return [_rect(p[0] + EGO_FWD * np.cos(a), p[1] + EGO_FWD * np.sin(a), EGO_L, EGO_W, a) for p, a in zip(pos, yws)]


def agent_gap_series(s, j, ego):
    b = np.asarray(s["boxes"], np.float64)[j]
    f = np.asarray(s["agent_fut"], np.float64)[j]
    m = np.asarray(s["agent_mask"])[j]
    vel = b[7:9]
    p, a = b[:2].copy(), float(b[6])
    last = (vel if np.isfinite(vel).all() else np.zeros(2)) * DT
    gaps = [ego[0].distance(_rect(p[0], p[1], b[3], b[4], a))]
    for t in range(12):
        if t < 6 and (m[:t + 1] == 1).all():
            st = f[t]; last = st
        else:
            st = last
        p = p + st
        if np.linalg.norm(st) >= 0.25:
            a = float(np.arctan2(st[1], st[0]))
        gaps.append(ego[t + 1].distance(_rect(p[0], p[1], b[3], b[4], a)))
    return np.asarray(gaps)


def enrich_objects(s, objs, gap_th=None, prev_available=True):
    gap_th = TTC_GAP if gap_th is None else gap_th
    if gap_th is None:
        raise RuntimeError("TTC_GAP 미설정(--explore_ttc 로 정한 값을 set_ttc_gap 으로)")
    boxes = np.asarray(s["boxes"], np.float64)
    F = np.asarray(s["agent_fut"], np.float64)
    M = np.asarray(s["agent_mask"])
    ego = ego_track(s)
    for r in objs:
        j = r["box_index"]
        b = boxes[j]
        r["lat"] = _r1c(b[0], -99.9, 99.9, "lat"); r["lon"] = _r1c(b[1], -99.9, 99.9, "lon")
        r["length"] = _r1c(b[3], 0.0, 99.9, "length"); r["width"] = _r1c(b[4], 0.0, 99.9, "width")
        r["heading"] = None if r["name"] in NO_HEADING else sector(bearing_xy(np.cos(b[6]), np.sin(b[6])))
        if r["vel_finite"]:
            spd = float(np.hypot(b[7], b[8]))
            r["motion"] = "moving" if spd >= MOVING_TH else "static"
            r["speed"] = _r1c(spd, 0.0, 99.9, "speed")
        else:
            r["motion"], r["speed"], r["vel_sector"] = "na", None, None
        cum = F[j].cumsum(0)
        r["fut"] = [[_r1c(cum[h, 0], -99.9, 99.9, "fut_x"), _r1c(cum[h, 1], -99.9, 99.9, "fut_y")]
                    if (M[j, :h + 1] == 1).all() else None for h in FUT_STEPS]
        if ego is None:
            r["ttc"], r["ttc_t"], r["gap_min3"] = None, None, None
        else:
            g = agent_gap_series(s, j, ego)
            r["ttc"], r["ttc_t"] = ttc_bucket(g, gap_th)
            r["gap_min3"] = round(float(g[:7].min()), 3)
    return objs


# ---------------------------------------------------------------- 지도
def traffic_control(s):
    """<pc> 칸(B2D 뜻): 신호등·정지 표지 트리거 영역 중 20 m 안 최근접. 반환 None | dict(dist, sector|'here', kind)."""
    from shapely.geometry import Point, Polygon, LineString
    from shapely.ops import nearest_points
    o = Point(0.0, 0.0)
    best = None
    for poly, kind in zip(s.get("tc", []), s.get("tc_kind", [])):
        P = np.asarray(poly, np.float64)
        if len(P) < 2:
            continue
        g = Polygon(P) if len(P) >= 3 else LineString(P)
        if not g.is_valid:
            g = g.convex_hull
        d = g.distance(o)
        if best is None or d < best[0]:
            q = nearest_points(g, o)[0]
            best = (d, q.x, q.y, kind)
    if best is None or best[0] > TC_RANGE:
        return None
    d, qx, qy, kind = best
    return dict(dist=round(d, 1), sector="here" if d < 0.05 else sector(bearing_xy(qx, qy)), kind=kind)


def lane_offsets(ma):
    out = {}
    for y in LANE_Y:
        d = {}
        for key, ids in (("div", MARK_IDS), ("bnd", NOCROSS_IDS)):
            polys = [p for c in ids for p in ma.get(c, [])]
            xs = _cross_x(polys, float(y))
            L = [-x for x in xs if x < 0]; R = [x for x in xs if x >= 0]
            d[key + "_l"] = _r1c(min(L), 0.0, 99.9, "lane") if L else None
            d[key + "_r"] = _r1c(min(R), 0.0, 99.9, "lane") if R else None
        out[str(y)] = d
    return out


def free_distance(s):
    """자차 폭 직선 회랑(|x| ≤ 자차 반폭, 0 ≤ y ≤ 30 m)에서 처음 만나는 객체 박스(select_objects 와 같은 필터)의 최소 y. 없으면 30."""
    from shapely.geometry import box as sbox
    cor = sbox(-FREE_HALF_W, 0.0, FREE_HALF_W, FREE_CAP)
    best = FREE_CAP
    boxes = np.asarray(s["boxes"], np.float64)
    for j, name in enumerate(s["names"]):
        if not _obj_ok(str(name), boxes[j, 0], boxes[j, 1]):
            continue
        g = _rect(boxes[j, 0], boxes[j, 1], boxes[j, 3], boxes[j, 4], boxes[j, 6]).intersection(cor)
        if not g.is_empty:
            best = min(best, g.bounds[1])
    return _r1c(max(best, 0.0), 0.0, FREE_CAP, "free")


def map_context_sensor(s):
    ma = s.get("map")
    if ma is None:
        return None
    nd = sum(len(ma.get(c, [])) for c in MARK_IDS)
    nb = sum(len(ma.get(c, [])) for c in NOCROSS_IDS)
    if not (nd < 100 and nb < 100):
        raise ValueError(f"지도 개수 2 자리 초과: {nd}, {nb}")
    return dict(ped_crossing=traffic_control(s), n_divider=nd, n_boundary=nb, lanes=lane_offsets(ma),
                free_dist=free_distance(s), n_center=len(ma.get(CENTER_ID, [])))


# ---------------------------------------------------------------- 의도(라벨 — 미래 GT 사용 가능)
HALF_LANE = 1.6        # m (nuScenes lane_v2 와 같은 값)
ASSIGN_MAX = 2.0       # m, 출발 차로 중심선 배정 허용 횡거리
HEAD_TOL_DEG = 40.0    # 자차 진행방향과 중심선 방향 차 허용
UNCERTAIN_WEIGHT = 0.3


def divider_crossing(xy, polys):
    import common as C1
    return C1.divider_crossing(xy, polys)


def _signed_offset(poly, p, extend=True):
    """폴리라인 기준 점 p 의 부호 있는 횡거리(+ = 진행방향 오른쪽)와 투영 위치의 방향. 투영이 끝을 넘으면 끝 선분 연장."""
    P = np.asarray(poly, np.float64)
    best = None
    for a in range(len(P) - 1):
        q0, q1 = P[a], P[a + 1]
        u = q1 - q0; L2 = float(u @ u)
        if L2 < 1e-12:
            continue
        t = float((p - q0) @ u / L2)
        tc = t if (extend and ((a == 0 and t < 0) or (a == len(P) - 2 and t > 1))) else min(max(t, 0.0), 1.0)
        q = q0 + tc * u
        d = float(np.linalg.norm(p - q))
        if best is None or d < best[0]:
            cr = u[0] * (p - q0)[1] - u[1] * (p - q0)[0]   # u × (p-q0) > 0 → 왼쪽(+x 오른쪽 좌표계)
            best = (d, -np.sign(cr) * d, np.arctan2(u[1], u[0]))
    return None if best is None else (best[1], best[2])


def center_signal(xy, centers):
    """(c) 출발 차로 중심선 기준 횡변위. 반환 dict(dir='left'|'right'|None, d0, d6, ok)."""
    p0, p1, p6 = xy[0], xy[1], xy[-1]
    head0 = np.arctan2(*(p1 - p0)[::-1]) if np.linalg.norm(p1 - p0) > 0.25 else np.pi / 2
    cand = []
    for k, c in enumerate(centers):
        r = _signed_offset(c, p0, extend=False)
        if r is None:
            continue
        dh = abs((r[1] - head0 + np.pi) % (2 * np.pi) - np.pi)
        if abs(r[0]) <= ASSIGN_MAX and np.degrees(dh) <= HEAD_TOL_DEG:
            cand.append((abs(r[0]), k, r[0]))
    if not cand:
        return dict(dir=None, d0=None, d6=None, ok=False)
    _, k, d0 = min(cand)
    d6 = _signed_offset(centers[k], p6, extend=True)[0]
    dd = d6 - d0
    return dict(dir=None if abs(dd) < HALF_LANE else ("right" if dd > 0 else "left"), d0=round(d0, 3), d6=round(d6, 3), ok=True)


def intent_labels(s, v0):
    """종방향 = common.intent_labels 와 같은 규칙. 횡방향: 회전(±12.5°) → 차로변경(두 신호) → 직진."""
    import common as C1
    fut = fut_cum(s)
    ma = s["map"] or {}
    marks = [p for c in MARK_IDS for p in ma.get(c, [])]
    lab = C1.intent_labels(fut, v0, marks)       # lateral 은 아래에서 다시 정한다(종방향·회전·잔차는 그대로)
    lat_v1 = lab["lateral"]
    sig = None
    if lat_v1.startswith("turn") or lab["path_len"] < MIN_PATH:
        lat, unc = lat_v1, False
    else:
        xy = np.concatenate([np.zeros((1, 2)), fut], 0)
        b = lab["divider_crossing"]
        c = center_signal(xy, ma.get(CENTER_ID, []))
        sig = dict(b_divider=b, c_center=c["dir"], c_ok=c["ok"], d0=c["d0"], d6=c["d6"])
        if b is not None and c["dir"] == b:
            lat, unc = f"lane change {b}", False
        elif b is None and c["dir"] is None:
            lat, unc = "go straight", False
        else:
            lat, unc = "go straight", True
    return dict(lab, lateral=lat, lateral_v1=lat_v1, uncertain=unc, lc_signals=sig)


# ---------------------------------------------------------------- 프롬프트
SYSTEM_PROMPT = (
    "You are a driving teacher model for an autonomous vehicle in the CARLA simulator. You see the surround-view camera "
    "images of the current frame and of the previous frame (0.5 s earlier), and may be given the high-level navigation "
    "command, the next route target point and the current ego vehicle state. You must describe the driving-relevant "
    "objects and map context, decide the driving intent, and output the ego trajectory for the next 3 seconds as action tokens."
)


def build_instruction(ego_in_prompt="current", object_speed="numeric"):
    assert ego_in_prompt in EGO_MODES and object_speed in SPEED_MODES
    if ACTION_K is None:
        raise RuntimeError("ACTION_K 미설정(set_action_k — B2D 코드북 크기)")
    inputs = ("Inputs: 12 camera images: the 6 surround-view cameras of the previous frame (0.5 s earlier), then "
              "the 6 of the current frame, each in order front, front-left, front-right, back-left, back, back-right "
              "(if the previous frame is missing the current frame is repeated); the navigation command; the route target "
              "point in metres (+x = right, +y = ahead)")
    inputs += ("; the current ego state (speed, longitudinal acceleration, yaw rate). The command or the ego state "
               "may be unavailable.\n" if ego_in_prompt != "none" else ". The command may be unavailable.\n")
    motion = ("motion = <static> or the direction sector of motion, followed by the speed DD.D m/s; unknown = "
              "<na><_><_><_><_>" if object_speed == "numeric" else
              "motion = <static> or the direction sector of motion; unknown = <na>")
    n_slot = slot_tokens(object_speed)
    return (inputs +
            "Answer in exactly four lines with a fixed number of tokens, no spaces inside fields. Numbers are "
            "fixed-width with one decimal: signed SDD.D (e.g. +09.2, -05.6, zero = +00.0) or unsigned DD.D (e.g. 04.5). "
            "Fillers keep the length fixed: <none> = absent, <na> = unknown, each followed by <_> to the field's length.\n"
            "Objects: exactly 8 slots. Up to 8 objects most relevant to driving (in the ego path, approaching the ego, "
            "pedestrians and cyclists within 20m, then nearest), nearest first; then empty slots <none> followed by "
            f"{n_slot - 1} <_>. An object slot is <class><sector><x><y><length><width><heading><motion><future 1s>"
            "<future 2s><future 3s><risk>: x, y = signed centre position in metres of the current frame (+x = right, "
            "+y = ahead); length, width in metres; heading = direction the object faces relative to the ego "
            f"<hF><hFR><hR><hBR><hB><hBL><hL><hFL> (<na> for cones, warning props and debris); {motion}; future = signed x, y "
            "displacement in metres after 1, 2, 3 s (unknown = <na> and 9 <_>); risk = time until the object comes "
            f"within {TTC_GAP}m of the ego along the ego's future path: <ttc_lt2> (under 2 s), <ttc_2_4>, <ttc_4_6>, "
            "<ttc_safe> (none within 6 s), or <na>. Classes: <car><van><truck><bike><sign><cone><light><ped><others>. "
            "Sectors (clockwise from front): <F><FR><R><BR><B><BL><L><FL>.\n"
            "Map: <pc><sector><distance> (nearest traffic-light or stop-sign stop zone within 20m) or <pc_on> (inside "
            "one) or <pc_none> (each followed by 5 <_>), <div><number of lane markings DD><bnd><number of solid lane "
            "markings DD>, then for the cross-sections 0, 10, 20 m ahead <at0>, <at10>, <at20> each followed by the "
            "unsigned distance to the nearest left lane marking, right lane marking, left solid marking, right solid "
            "marking (<none><_><_><_> if none within 15m), then <free> and the free distance ahead in the ego's width "
            f"until the first object ({FREE_CAP:.1f} if none within {FREE_CAP:.0f}m); or <na> and {MAP_LINE_FIELDS - 1} "
            "<_> if the map is unknown.\n"
            "Intent: <longitudinal><lateral> with longitudinal one of <accelerate><keep_speed><decelerate><stop> and "
            "lateral one of <go_straight><turn_left><turn_right><lane_change_left><lane_change_right>; or <na><_>.\n"
            f"Action: six action tokens <action_k> (k in 0-{ACTION_K - 1}), one per 0.5 s for the next 3 s; or <na> and 5 <_>.")


CMD_DROPPED = "Command: unavailable."
# 자차 상태 줄 — common_v2 와 같은 문장 틀(B안 고정폭)이되 종가속을 부호+2 자리 정수('+06.2f')로 넓힌다:
#   B2D 전문가(시뮬레이터)는 0.5 s 안에 7 m/s → 0 으로 서는 급제동을 자주 한다(IMU 종가속 |a| val p99 17 m/s², 최대 24.3 m/s²,
#   conventions 실측). nuScenes 형식(|a| < 10)으로는 고정폭이 깨지고, 자르는 것은 금지라 형식을 넓혔다. 토큰 수 43(값 조합 전부 실측).
#   결측 문장도 43 토큰으로 맞춘다(캐시 문맥 길이 고정 — 실측 '\n' 포함 43). B2D 는 실제 결측이 없다(SFT 드롭아웃 전용).
EGO_DROPPED = ("Ego state: unavailable. The speed, longitudinal acceleration and yaw rate of the ego vehicle were not given for "
               "this frame, so infer its motion from the camera images (no vehicle state and no past poses).")
EGO_LINE_TOKENS = 43


def ego_line(speed, accel, yaw):
    if speed is None:
        return EGO_DROPPED
    if not (0.0 <= speed < 99.95 and abs(accel) < 99.995 and abs(yaw) < 9.9995):
        raise ValueError(f"자차 상태가 고정폭 범위 밖: speed {speed}, accel {accel}, yaw {yaw}")
    return (f"Ego state: speed {speed:04.1f} m/s, longitudinal acceleration {accel:+06.2f} m/s^2, "
            f"yaw rate {yaw:+.3f} rad/s (positive = turning left).")


def command_line(cmd):
    return f"Command: {CMD_TEXT_B2D[norm_command(cmd)]}."


def target_line(tp):
    """목표점(라이다 m) 고정폭 'SDDDD.D'(부호 + 4 자리 정수) — 범위 밖은 즉시 실패."""
    x, y = float(tp[0]), float(tp[1])
    if not (abs(x) < 9999.95 and abs(y) < 9999.95):
        raise ValueError(f"목표점 범위 밖: {tp}")
    return f"Target point: x {_r1(x):+07.1f} m, y {_r1(y):+07.1f} m."


def build_prompt_parts(cmd, target_xy, ego_vals, ego_in_prompt="current", object_speed="numeric"):
    e = ego_vals if ego_vals is not None else (None, None, None)
    return dict(instruction=build_instruction(ego_in_prompt, object_speed), command=command_line(cmd),
                target=target_line(target_xy),
                ego=None if ego_in_prompt == "none" else ego_line(*e),
                command_dropped=CMD_DROPPED, ego_dropped=EGO_DROPPED)


def join_prompt(parts, drop_command=False, drop_ego=False, ego_mode="blank"):
    lines = [parts["instruction"], parts["command_dropped"] if drop_command else parts["command"], parts["target"]]
    if parts["ego"] is not None and not (drop_ego and ego_mode == "omit"):
        lines.append(parts["ego_dropped"] if drop_ego else parts["ego"])
    return "\n".join(lines)


# ---------------------------------------------------------------- copy-baseline(common_v2 와 같은 규칙, B2D 명령 문장)
def copy_baseline(cmd, ego_vals, cb):
    v, w = (0.0, 0.0) if ego_vals is None else (float(ego_vals[0]), float(ego_vals[2]))
    traj = V2.ctrv_traj(v, w)
    toks, _ = cb.tokenize(traj)
    lat = {1: "turn left", 2: "turn right", 5: "lane change left", 6: "lane change right"}.get(norm_command(cmd), "go straight")
    return dict(intent=dict(longitudinal="stop" if v < STOP_V else "keep speed", lateral=lat),
                ego_used=None if ego_vals is None else [round(v, 4), round(w, 5)],
                traj=[[round(float(a), 4) for a in p] for p in traj], action_tokens=toks)


# ---------------------------------------------------------------- 정답 조립·엄격 파서(common_v2 와 같은 규칙, B2D 클래스 표)
def fmt_object(r, object_speed="numeric"):
    s = CLS_TOK[r["name"]] + SEC_TOK[r["sector"]] + _sn(r["lat"]) + _sn(r["lon"]) + _un(r["length"]) + _un(r["width"])
    s += "<na>" if r["heading"] is None else HEAD_TOK[r["heading"]]
    num = object_speed == "numeric"
    if r["motion"] == "na":
        s += _fill("<na>", 5) if num else "<na>"
    else:
        s += ("<static>" if r["motion"] == "static" else SEC_TOK[r["vel_sector"]]) + (_un(r["speed"]) if num else "")
    for p in r["fut"]:
        s += _fill("<na>", 10) if p is None else _sn(p[0]) + _sn(p[1])
    s += "<na>" if r["ttc"] is None else TTC_TOK[r["ttc"]]
    return s


fmt_map = V2.fmt_map   # 지도 줄 형식은 nuScenes 와 같다(필드 뜻만 B2D 정의)


def build_answer(objs, mp, intent_ans, toks, object_speed="numeric"):
    if len(objs) > MAX_OBJECTS:
        raise ValueError(f"객체 {len(objs)} > {MAX_OBJECTS}")
    if toks is not None and not all(0 <= k < ACTION_K for k in toks):
        raise ValueError("action 토큰 범위 밖")
    l1 = "Objects: " + "".join(fmt_object(r, object_speed) for r in objs) + \
         _fill("<none>", slot_tokens(object_speed)) * (MAX_OBJECTS - len(objs))
    l2 = fmt_map(mp)
    l3 = "Intent: " + ("<na><_>" if intent_ans is None else LON_TOK[intent_ans["longitudinal"]] + LAT_TOK[intent_ans["lateral"]])
    l4 = "Action: " + (_fill("<na>", 6) if toks is None else "".join(f"<action_{k}>" for k in toks))
    return "\n".join([l1, l2, l3, l4])


def _parse_object(c, object_speed, sp):
    num = object_speed == "numeric"
    o = dict(name=TOK_CLS[c.tok(CLS_TOK.values(), "클래스")], sector=TOK_SEC[c.tok(SEC_TOK.values(), "방위")])
    o["lat"] = float(c.rx(_SN_RE, "횡")[0]); o["lon"] = float(c.rx(_SN_RE, "종")[0])
    o["length"] = float(c.rx(_UN_RE, "길이")[0]); o["width"] = float(c.rx(_UN_RE, "폭")[0])
    if c.is_fill("<na>"):
        sp["na"].append(c.fill("<na>", 1, "진행방위")); o["heading"] = None
    else:
        o["heading"] = TOK_HEAD[c.tok(HEAD_TOK.values(), "진행방위")]
    if c.is_fill("<na>"):
        sp["na"].append(c.fill("<na>", 5 if num else 1, "운동")); o.update(motion="na", vel_sector=None, speed=None)
    else:
        t = c.tok(["<static>"] + list(SEC_TOK.values()), "운동")
        o["motion"] = "static" if t == "<static>" else "moving"
        o["vel_sector"] = None if t == "<static>" else TOK_SEC[t]
        if num:
            v, span = c.rx(_UN_RE, "속력"); o["speed"] = float(v); sp["speed"].append(span)
        else:
            o["speed"] = None
    o["fut"] = []
    for _ in FUT_STEPS:
        if c.is_fill("<na>"):
            sp["na"].append(c.fill("<na>", 10, "미래")); o["fut"].append(None)
        else:
            o["fut"].append([float(c.rx(_SN_RE, "미래 x")[0]), float(c.rx(_SN_RE, "미래 y")[0])])
    if c.is_fill("<na>"):
        sp["na"].append(c.fill("<na>", 1, "TTC")); o["ttc"] = None
    else:
        o["ttc"] = TOK_TTC[c.tok(TTC_TOK.values(), "TTC")]
    return o


def _parse_full(text, object_speed="numeric"):
    """common_v2._parse_full 과 같은 규칙. 객체 칸만 B2D 클래스 표로 읽고, 지도·의도·행동 줄은 같은 문법이다."""
    if object_speed not in SPEED_MODES:
        raise ValueError(object_speed)
    lines = text.split("\n")
    if len(lines) != 4:
        raise ValueError(f"줄 수 {len(lines)} != 4")
    pos, L = 0, {}
    for name, line in zip(["objects", "map", "intent", "action"], lines):
        L[name] = [pos, pos + len(line)]; pos += len(line) + 1
    sp = dict(speed=[], na=[])
    if not lines[0].startswith("Objects: "):
        raise ValueError("Objects 머리말")
    c = _Cur(lines[0][9:], base=9)
    objs, empty = [], 0
    for _ in range(MAX_OBJECTS):
        if c.is_fill("<none>"):
            c.fill("<none>", slot_tokens(object_speed), "빈 칸"); empty += 1
        else:
            if empty:
                raise ValueError("빈 칸 뒤에 채운 칸")
            objs.append(_parse_object(c, object_speed, sp))
    if not c.done():
        raise ValueError("Objects 줄 끝에 남은 문자")
    # 지도·의도 줄은 nuScenes 파서를 그대로 쓴다(객체 줄을 빈 칸으로 바꾼 문자열을 넘겨 구간 위치만 보정)
    empty_obj = "Objects: " + _fill("<none>", slot_tokens(object_speed)) * MAX_OBJECTS
    act_line = lines[3]
    if act_line != "Action: " + _fill("<na>", 6):
        body = act_line[8:] if act_line.startswith("Action: ") else None
        ms = list(_ACT_RE.finditer(body or ""))
        if body is None or len(ms) != 6 or "".join(m.group(0) for m in ms) != body:
            raise ValueError("Action 줄 문법 불일치")
        act = [int(m.group(1)) for m in ms]
        if ACTION_K is None or not all(0 <= k < ACTION_K for k in act):
            raise ValueError("action 토큰 범위 밖")
        act_line = "Action: " + "<action_0>" * 6          # nuScenes 파서의 범위(2041) 검사를 피하려 자리만 바꿔 넘긴다
    else:
        act = None
    p2, sp2 = V2._parse_full("\n".join([empty_obj, lines[1], lines[2], act_line]), object_speed)
    shift = len(lines[0]) - len(empty_obj)
    sp["na"] += [[a + shift, b + shift] for a, b in sp2["na"]]
    return dict(objects=objs, map=p2["map"], intent=p2["intent"], action=act), dict(lines=L, **sp)


def parse_answer(text, object_speed="numeric"):
    return _parse_full(text, object_speed)[0]


def format_parsed(p, object_speed="numeric"):
    mp = p["map"]
    if mp is not None and mp["ped_crossing"] is not None and mp["ped_crossing"]["sector"] == "here":
        mp = dict(mp, ped_crossing=dict(dist=0.0, sector="here"))
    return build_answer(p["objects"], mp, p["intent"], p["action"], object_speed)


def answer_spans(answer, object_speed="numeric"):
    sp = _parse_full(answer, object_speed)[1]
    return dict(lines=sp["lines"], object_speed=sp["speed"], speed=sp["speed"], na=sp["na"])


GRAMMAR_B2D = r"""# 교사 VLM 정답 v2-sensor-b2d 문법 (common_b2d.parse_answer 가 엄격히 강제; format_parsed 로 왕복 정확 일치)
# nuScenes v2-sensor(data/vla_teacher/v2/grammar.txt)와 줄·필드·토큰 수가 같다(547). 다른 것: 클래스 표, 필드의 B2D 정의(아래 *).
answer   := L1 "\n" L2 "\n" L3 "\n" L4
L1       := "Objects: " OBJ{k} EMPTY{8-k}                       (k = 0..8, 채운 칸은 거리 오름차순)
OBJ      := CLS SEC SN SN UN UN HEAD MOTION FUT FUT FUT RISK    (57 토큰) — x(+오른쪽), y(+전방), 길이, 폭(라이다 m)
HEAD     := <hF>|…|<hFL> | "<na>"     * traffic_cone·others(경고 시설물·잔해) = <na>; sign·light 는 면 방향
MOTION   := ("<static>" | SEC) UN | "<na>" "<_>"x4             (moving = 속력 ≥ 0.5 m/s, 시뮬레이터 속도)
FUT      := SN SN | "<na>" "<_>"x9                              (1/2/3 s 누적 변위, B2D 객체 미래 마스크)
RISK     := <ttc_lt2>|<ttc_2_4>|<ttc_4_6>|<ttc_safe> | "<na>"   (간격 ≤ TTC_GAP_B2D 첫 시각, B2D 자차 4.89×1.84 m)
EMPTY    := "<none>" "<_>"x56
L2       := "Map: " ( MAP | "<na>" "<_>"x67 )
MAP      := PC "<div>" I2 "<bnd>" I2 AT0 AT10 AT20 "<free>" UN
PC       := "<pc>" SEC UN | "<pc_on>" "<_>"x5 | "<pc_none>" "<_>"x5
           * B2D: 신호등·정지 표지 트리거 영역(정지 구역) 20 m 안 최근접 (B2D 지도에 횡단보도 없음; 토큰 글자는 nuScenes 와 공유)
"<div>" I2 * 차선 표시 선 수(Broken+Solid+SolidSolid),  "<bnd>" I2 * 넘으면 안 되는 표시 선 수(Solid+SolidSolid)
ATy      := "<aty>" LANE LANE LANE LANE   * 왼/오른 최근접 차선 표시(종류 무관), 왼/오른 최근접 실선(Solid/SolidSolid) |x|
LANE     := UN | "<none>" "<_>"x3                               (ROI |x| ≤ 15 m)
"<free>"  * 자차 폭 직선 회랑 0..30 m 의 첫 객체 박스 y(없으면 30.0) — 지도 선은 쓰지 않음(물리 경계 클래스 없음)
L3       := "Intent: " ( LON LAT | "<na><_>" )   * 차로변경 = 표시선 가로지름 + 출발 차로 중심선 기준 |Δd| ≥ 1.6 m
L4       := "Action: " ( ACT{6} | "<na>" "<_>"x5 )   ACT := <action_K>, K = 0..ACTION_K_B2D-1 (B2D 코드북)
CLS      := <car>|<van>|<truck>|<bike>|<sign>|<cone>|<light>|<ped>|<others>
SEC/LON/LAT/SN/UN/I2/손실 제외 규칙: nuScenes v2-sensor 와 같음.
"""
