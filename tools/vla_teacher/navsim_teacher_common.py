# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
"""NAVSIM 교사 GT — nuScenes v2-sensor 정답 형식의 NAVSIM 판 공용 모듈.

common.py / common_v2.py 는 수정하지 않는다(nuScenes 학습 진행 중). 형식 함수(fmt_object, fmt_map, _sn, _un, _fill,
_rect, _eval_yaw, agent_gap_series, ttc_bucket, lane_offsets, map_context, ActionCodebook, divider_crossing)는 import 해
그대로 쓰고, nuScenes 상수가 박혀 있는 함수(객체 선택·자차 박스·자유거리·의도 라벨·Action 줄)만 이 파일에 NAVSIM 값으로 둔다.

nuScenes → NAVSIM 대응(설계 문서: docs/navsim_teacher_gt_design.md)
  좌표   : 교사 프레임 = nuScenes lidar 규약(+x 오른쪽, +y 전방). NAVSIM ego(=lidar) 는 +x 전방 · +y 왼쪽 →
           x_t = −y_e, y_t = x_e, yaw_t = yaw_e + π/2 (속도·증분도 같은 회전). 원점 = rear-axle.
  클래스 : vehicle→<car>, pedestrian→<ped>, bicycle→<bike>, traffic_cone→<cone>, barrier→<barrier>,
           czone_sign→<czone>(신규), generic_object→<obj>(신규). 신규 2 개도 토큰 1 개라 칸 길이(57) 불변.
  객체 필터: nuScenes 는 num_lidar_pts>0 — OpenScene 주석에는 점 수가 없어 거리(55 m)·클래스만 쓴다.
  자차 박스: nuPlan Pacifica 5.176 × 2.297 m, 차체 중심 = rear-axle 전방 1.461 m (get_pacifica_parameters).
  지평  : 자차 정답 8 포즈 = 4 s (NAVSIM Trajectory). Action 8 토큰(0.5 s 간격) → 정답 549 토큰(nuScenes 547 + 2).
           TTC 자차 궤적 = 0–4 s GT + 4–6 s 마지막 증분 등속 외삽. 객체 미래 칸(1/2/3 s)은 nuScenes 와 같다.
  의도  : 같은 임계값을 4 s 끝 기준으로(종: 3.0–4.0 s 평균 속력 vs 현재 속력, 횡: 4 s 끝 진행각). 차로변경 = nuPlan 지도
           위상(같은 roadblock 의 왼/오른 차로로 이동, navsim_teacher_map.py) 우선, 위상 없이 구분선 교차만 있으면 'uncertain'
           (nuScenes v2 와 같이 go straight + 가중 0.3).
  명령  : NAVSIM driving_command one-hot [left, straight, right, unknown] (navsim_ref docs/agents.md:79) → 'turn left' /
           'go straight' / 'turn right' / unknown = 'Command: unavailable.'. (NAVSIM 의 left/right 는 회전·차로변경·급커브 포함.)
  자차 상태: speed = |(vx, vy)|, longitudinal accel = ax (ego_dynamic_state), yaw rate = 직전 프레임(t−0.5 s, 로그 포즈) 과의
           heading 차분 / Δt. 모두 현재·과거 정보(미래 없음).
"""
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common as C1  # noqa: E402
import common_v2 as C  # noqa: E402

# ---------------------------------------------------------------- 상수
NAV_CLS_MAP = {"vehicle": "car", "pedestrian": "pedestrian", "bicycle": "bicycle", "traffic_cone": "traffic_cone",
               "barrier": "barrier", "czone_sign": "czone_sign", "generic_object": "generic_object"}
NAV_TEACHER_CLASSES = tuple(NAV_CLS_MAP.values())
NEW_CLS_TOK = {"czone_sign": "<czone>", "generic_object": "<obj>"}
CLS_TOK = dict(C.CLS_TOK, **NEW_CLS_TOK)
TOK_CLS = {v: k for k, v in CLS_TOK.items()}
SPECIAL_TOKENS = list(C.SPECIAL_TOKENS) + list(NEW_CLS_TOK.values())   # nuScenes 순서 유지 + 뒤에 2 개
assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS))
# 수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건 — (4) NAVSIM 전용 행동 코드북·의도 임계값(build_navsim_codebook_intent.py 산출).
#   configure() 를 부르기 전에는 nuScenes 값(코드북 K 2041, 임계값 common.py)이다 — gen/check 스크립트는 반드시 configure 한다.
ACTION_K = C1.ACTION_K
THR = dict(STOP_V=C1.STOP_V, DV_TH=C1.DV_TH, TURN_TH_DEG=C1.TURN_TH_DEG, MIN_PATH=C1.MIN_PATH)
CONFIG = dict(codebook="nuscenes", thresholds="nuscenes")


class NavsimActionCodebook(C1.ActionCodebook):
    """common.ActionCodebook 과 같은 토큰화(폐루프 최근접), K 만 파일 값(nuScenes 의 K==2041 검사 없음)."""
    def __init__(self, path):
        z = np.load(path)
        self.book = np.asarray(z["codebook"], np.float64)
        self.K = len(self.book)


def configure(codebook_path, thresholds_path):
    global ACTION_K, SPECIAL_TOKENS
    import json as _json
    cb = NavsimActionCodebook(codebook_path)
    ACTION_K = cb.K
    base = [t for t in C.SPECIAL_TOKENS if not t.startswith("<action_")]
    i0 = next(k for k, t in enumerate(C.SPECIAL_TOKENS) if t.startswith("<action_"))
    SPECIAL_TOKENS = base[:i0] + [f"<action_{k}>" for k in range(ACTION_K)] + base[i0:] + list(NEW_CLS_TOK.values())
    assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS))
    th = _json.load(open(thresholds_path))
    for k in THR:
        THR[k] = float(th[k])
    CONFIG.update(codebook=os.path.abspath(codebook_path), thresholds=os.path.abspath(thresholds_path), action_k=ACTION_K, **THR)
    return cb
VRU = ("pedestrian", "bicycle")
EGO_L, EGO_W, EGO_FWD = 5.176, 2.297, 1.461       # nuplan get_pacifica_parameters (length, width, rear_axle_to_center)
FREE_HALF_W = EGO_W / 2
N_FUT = 8                                         # 자차 정답 포즈(4 s @ 2 Hz)
ACTION_N = 8
ANSWER_TOKENS = 547 + (ACTION_N - 6)              # 549
CAMS = ("CAM_L0", "CAM_F0", "CAM_R0")             # 전방 3캠(SparseDriveV2·QA 와 같은 순서: 왼앞, 앞, 오른앞)
CMD_TEXT_NAV = {0: "turn left", 1: "go straight", 2: "turn right"}   # 3 = unknown → CMD_DROPPED
TTC_GAP_NAV = None                                # navtrain 통계로 정한다(gen_teacher_gt_navsim.py --explore_ttc). None 이면 생성 거부


# ---------------------------------------------------------------- 좌표 변환(NAVSIM ego → 교사 프레임)
def _rot_xy(a):
    a = np.asarray(a, np.float64)
    out = np.empty_like(a)
    out[..., 0] = -a[..., 1]
    out[..., 1] = a[..., 0]
    return out


def teacher_view(info, map_annos):
    """NAVSIM info 프레임 → common_v2 함수가 읽는 키를 가진 교사 프레임 dict (원본 info 는 바꾸지 않는다)."""
    b = np.asarray(info["gt_boxes"], np.float64).reshape(-1, 7).copy()
    b[:, :2] = _rot_xy(b[:, :2])
    b[:, 6] = b[:, 6] + np.pi / 2
    names = np.asarray([NAV_CLS_MAP.get(str(n), "__other__") for n in info["gt_names"]])
    return dict(
        token=info["token"], gt_boxes=b, gt_names=names,
        gt_velocity=_rot_xy(np.asarray(info["gt_velocity"], np.float64).reshape(-1, 2)),
        num_lidar_pts=np.ones(len(b), np.int64), instance_inds=np.asarray(info["instance_inds"]),
        gt_agent_fut_trajs=_rot_xy(np.asarray(info["gt_agent_fut_trajs"], np.float64)),
        gt_agent_fut_masks=np.asarray(info["gt_agent_fut_masks"], np.float64),
        gt_ego_fut_trajs=_rot_xy(np.asarray(info["gt_ego_fut_trajs"], np.float64)),
        gt_ego_fut_masks=np.asarray(info["gt_ego_fut_masks"], np.float64),
        map_annos=map_annos,
    )


def future_complete(v):
    return bool((np.asarray(v["gt_ego_fut_masks"]) == 1).all())


def fut_cum(v):
    return np.asarray(v["gt_ego_fut_trajs"], np.float64).cumsum(0)[:N_FUT, :2]


# ---------------------------------------------------------------- 객체 선택(common_v2.select_objects 와 같은 규칙, 필터만 NAVSIM)
def select_objects(v, v0):
    boxes = np.asarray(v["gt_boxes"], np.float64); names = np.asarray(v["gt_names"])
    vel = np.asarray(v["gt_velocity"], np.float64); inst = list(v["instance_inds"])
    if future_complete(v):
        path, path_src = np.concatenate([np.zeros((1, 2)), fut_cum(v)], 0), "gt_future"
    else:
        path, path_src = np.array([[0.0, 0.0], [0.0, max(3.0 * v0, C.STRAIGHT_MIN_LEN)]]), "straight_cmd"
    vego = np.array([0.0, v0])
    rows = []
    for j in range(len(boxes)):
        name = str(names[j])
        if name not in NAV_TEACHER_CLASSES:
            continue
        x, y = float(boxes[j, 0]), float(boxes[j, 1])
        dist = (x * x + y * y) ** 0.5
        if dist > C1.OBJ_RANGE:
            continue
        vx, vy = float(vel[j, 0]), float(vel[j, 1])
        fin = bool(np.isfinite(vx) and np.isfinite(vy))
        if not fin:
            vx = vy = 0.0
        spd = (vx * vx + vy * vy) ** 0.5
        moving = spd >= C1.MOVING_TH
        p = np.array([x, y])
        cd = C._seg_dist(p, path)
        rr = float((p @ (np.array([vx, vy]) - vego)) / max(dist, 1e-6))
        ttc = dist / -rr if (moving and rr <= -C.CLOSING_MIN) else None
        if cd <= C.CORRIDOR_HALF:
            tier, key = 1, dist
        elif ttc is not None and ttc <= C.TTC_MAX:
            tier, key = 2, ttc
        elif name in VRU and dist <= C.VRU_RANGE:
            tier, key = 3, dist
        else:
            tier, key = 4, dist
        bear = C1.bearing_xy(x, y)
        rows.append(dict(
            name=name, sector=C1.sector(bear), lat=C._r1(x), lon=C._r1(y), dist=round(dist, 1),
            vel_sector=C1.sector(C1.bearing_xy(vx, vy)) if moving else None, speed=C._r1(spd) if moving else None,
            reason=C.TIER_NAMES[tier], corridor_dist=round(cd, 2), ttc=None if ttc is None else round(ttc, 2),
            x=round(x, 3), y=round(y, 3), box=[round(float(q), 4) for q in boxes[j, :7]],
            vel_finite=fin, instance_ind=int(inst[j]), box_index=j, _k=(tier, key, dist), _d=dist))
    n_in = len(rows)
    rows.sort(key=lambda r: r["_k"])
    sel = rows[:C.MAX_OBJECTS]
    sel.sort(key=lambda r: r["_d"])
    nearest8 = {id(r) for r in sorted(rows, key=lambda r: r["_d"])[:C.MAX_OBJECTS]}
    meta = dict(path_src=path_src, n_tier={C.TIER_NAMES[t]: sum(r["_k"][0] == t for r in rows) for t in C.TIER_NAMES},
                n_selected_not_in_nearest8=sum(id(r) not in nearest8 for r in sel))
    for r in sel:
        r.pop("_k"); r.pop("_d")
    return sel, n_in, meta


# ---------------------------------------------------------------- 자차 박스 궤적·TTC
def ego_track(v):
    """13 시각(0..6 s) 자차 박스. 0–4 s = GT 8 포즈, 4–6 s = 마지막 증분 등속 외삽."""
    if not future_complete(v):
        return None
    fut = fut_cum(v)
    yaw = C._eval_yaw(fut)
    st = fut[-1] - fut[-2]
    pos = [np.zeros(2)] + list(fut) + [fut[-1] + st * k for k in range(1, 13 - N_FUT)]
    yws = [np.pi / 2] + list(yaw) + [yaw[-1]] * (12 - N_FUT)
    assert len(pos) == 13 == len(yws)
    return [C._rect(p[0] + EGO_FWD * np.cos(a), p[1] + EGO_FWD * np.sin(a), EGO_L, EGO_W, a) for p, a in zip(pos, yws)]


def enrich_objects(v, objs, gap_th):
    boxes = np.asarray(v["gt_boxes"], np.float64)
    vel = np.asarray(v["gt_velocity"], np.float64)
    F = np.asarray(v["gt_agent_fut_trajs"], np.float64)
    M = np.asarray(v["gt_agent_fut_masks"])
    ego = ego_track(v)
    for r in objs:
        j = r["box_index"]
        b = boxes[j]
        r["lat"] = C._r1c(b[0], -99.9, 99.9, "lat"); r["lon"] = C._r1c(b[1], -99.9, 99.9, "lon")
        r["length"] = C._r1c(b[3], 0.0, 99.9, "length"); r["width"] = C._r1c(b[4], 0.0, 99.9, "width")
        r["heading"] = None if r["name"] == "traffic_cone" else C1.sector(C1.bearing_xy(np.cos(b[6]), np.sin(b[6])))
        if r["vel_finite"]:
            spd = float(np.hypot(vel[j, 0], vel[j, 1]))
            r["motion"] = "moving" if spd >= C1.MOVING_TH else "static"
            r["speed"] = C._r1c(spd, 0.0, 99.9, "speed")
        else:
            r["motion"], r["speed"], r["vel_sector"] = "na", None, None
        cum = F[j].cumsum(0)
        r["fut"] = [[C._r1c(cum[h, 0], -99.9, 99.9, "fut_x"), C._r1c(cum[h, 1], -99.9, 99.9, "fut_y")]
                    if (M[j, :h + 1] == 1).all() else None for h in C.FUT_STEPS]
        if ego is None:
            r["ttc"], r["ttc_t"], r["gap_min3"] = None, None, None
        else:
            g = C.agent_gap_series(v, j, ego)
            r["ttc"], r["ttc_t"] = C.ttc_bucket(g, gap_th)
            r["gap_min3"] = round(float(g[:7].min()), 3)
    return objs


def gap_series_for_explore(v, v0):
    if not future_complete(v):
        return []
    objs, _, _ = select_objects(v, v0)
    ego = ego_track(v)
    return [C.agent_gap_series(v, r["box_index"], ego).tolist() for r in objs]


# ---------------------------------------------------------------- 지도 줄
def free_distance(v, ma):
    from shapely.geometry import LineString, box as sbox
    cor = sbox(-FREE_HALF_W, 0.0, FREE_HALF_W, C.FREE_CAP)
    best = C.FREE_CAP
    for pl in ma.get(2, []):
        pl = np.asarray(pl, np.float64)
        if len(pl) < 2:
            continue
        g = LineString(pl).intersection(cor)
        if not g.is_empty:
            best = min(best, g.bounds[1])
    boxes = np.asarray(v["gt_boxes"], np.float64); names = np.asarray(v["gt_names"])
    for j in range(len(boxes)):
        if str(names[j]) not in NAV_TEACHER_CLASSES or np.hypot(boxes[j, 0], boxes[j, 1]) > C1.OBJ_RANGE:
            continue
        g = C._rect(boxes[j, 0], boxes[j, 1], boxes[j, 3], boxes[j, 4], boxes[j, 6]).intersection(cor)
        if not g.is_empty:
            best = min(best, g.bounds[1])
    return C._r1c(max(best, 0.0), 0.0, C.FREE_CAP, "free")


def map_context_sensor(v):
    mp = C1.map_context(v)
    if mp is None:
        return None
    ma = v["map_annos"]
    if not (mp["n_divider"] < 100 and mp["n_boundary"] < 100):
        raise ValueError(f"지도 개수 2 자리 초과: {mp['n_divider']}, {mp['n_boundary']}")
    mp["lanes"] = C.lane_offsets(ma)
    mp["free_dist"] = free_distance(v, ma)
    return mp


# ---------------------------------------------------------------- 의도(4 s)
def intent_labels(fut, v0, dividers):
    """common.intent_labels 와 같은 임계값, 끝 시각만 4 s(fut[7])."""
    xy = np.concatenate([np.zeros((1, 2)), fut], 0)
    th = float(C1.heading_series(xy)[-1])
    s = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    v_end = float(np.linalg.norm(fut[-1] - fut[-3]) / (2 * C1.DT))
    dv = v_end - v0
    lon = ("stop" if v_end < THR["STOP_V"] else "accelerate" if dv >= THR["DV_TH"] else
           "decelerate" if dv <= -THR["DV_TH"] else "keep speed")
    tdeg = float(np.degrees(th))
    cross = C1.divider_crossing(xy, dividers)
    if s < THR["MIN_PATH"]:
        lat = "go straight"
    elif tdeg >= THR["TURN_TH_DEG"]:
        lat = "turn left"
    elif tdeg <= -THR["TURN_TH_DEG"]:
        lat = "turn right"
    elif cross is not None:
        lat = f"lane change {cross}"
    else:
        lat = "go straight"
    return dict(longitudinal=lon, lateral=lat, v0=round(v0, 3), v_end=round(v_end, 3), dv=round(dv, 3),
                theta_end_deg=round(tdeg, 2), path_len=round(s, 3), divider_crossing=cross)


def lateral_decision(lab, topo):
    """회전은 각도 규칙 그대로. 그 밖: 위상 차로변경 → 'lane change X'(확실), 위상 없음 + 구분선 교차 → uncertain."""
    if lab["lateral"].startswith("turn"):
        return lab["lateral"], False, "turn"
    if topo is not None and topo.get("decision") in ("left", "right"):
        return f"lane change {topo['decision']}", False, "topology"
    if lab["divider_crossing"] is not None:
        return "go straight", True, "divider_only"
    return "go straight", False, "none"


# ---------------------------------------------------------------- 프롬프트
SYSTEM_PROMPT = (
    "You are a driving teacher model for an autonomous vehicle. You see the three front camera images "
    "(front-left, front, front-right) of the current frame and of the previous frame (0.5 s earlier), and may be "
    "given the high-level navigation command and the current ego vehicle state. You must describe the "
    "driving-relevant objects and map context, decide the driving intent, and output the ego trajectory for the "
    "next 4 seconds as action tokens."
)


def build_instruction(gap_th):
    base = C.build_instruction("current", "numeric")
    inputs_old = base.split("\n", 1)[0] + "\n"
    inputs = ("Inputs: 6 camera images: the 3 front cameras of the previous frame (0.5 s earlier), then the 3 of the "
              "current frame, each in order front-left, front, front-right (if the previous frame is missing the "
              "current frame is repeated); the navigation command; the current ego state (speed, longitudinal "
              "acceleration, yaw rate). The command or the ego state may be unavailable.\n")
    s = inputs + base[len(inputs_old):]
    reps = [
        ("Classes: <car><truck><bus><trailer><cv><ped><moto><bike><cone><barrier>.",
         "Classes: <car><ped><bike><cone><barrier><czone><obj> (<car> = any vehicle, <czone> = construction zone "
         "sign, <obj> = other object)."),
        (f"within {C.TTC_GAP}m of the ego", f"within {gap_th}m of the ego"),
        (f"Action: six action tokens <action_k> (k in 0-{C1.ACTION_K - 1}), one per 0.5 s for the next 3 s; or <na> and 5 <_>.",
         f"Action: eight action tokens <action_k> (k in 0-{ACTION_K - 1}), one per 0.5 s for the next 4 s; or <na> and 7 <_>."),
    ]
    for a, b in reps:
        if a not in s:
            raise RuntimeError(f"지시문 치환 실패(common_v2 문구 변경?): {a[:50]}")
        s = s.replace(a, b)
    return s


CMD_UNKNOWN = "Command: unknown direction."   # NAVSIM 4 번째 명령(경로 방향 미상). 다른 명령 줄과 같은 5 토큰 — 캐시 길이 고정
#   ('Command: unavailable.' 은 4 토큰이라 context 길이가 1 줄어든다(실측 757 vs 756). 그 문장은 SFT 드롭아웃 전용으로 남긴다.)


def command_line(cmd_id):
    return CMD_UNKNOWN if int(cmd_id) == 3 else f"Command: {CMD_TEXT_NAV[int(cmd_id)]}."


def build_prompt_parts(cmd_id, ego_vals, gap_th):
    e = ego_vals if ego_vals is not None else (None, None, None)
    return dict(instruction=build_instruction(gap_th), command=command_line(cmd_id), ego=C.ego_line(*e),
                command_dropped=C.CMD_DROPPED, ego_dropped=C.EGO_DROPPED)


# ---------------------------------------------------------------- 정답(1–3 줄 = common_v2 형식 그대로, 4 줄 = 8 행동)
_ACT_RE = re.compile(r"<action_(0|[1-9]\d*)>")
_NA_ACTION = C._fill("<na>", ACTION_N)


def fmt_object(r):
    """common_v2.fmt_object 와 같은 형식, 클래스 토큰만 NAVSIM 사전."""
    s = CLS_TOK[r["name"]]
    return s + C.fmt_object(dict(r, name="car"))[len(C.CLS_TOK["car"]):]


def build_answer(objs, mp, intent_ans, toks):
    if len(objs) > C.MAX_OBJECTS:
        raise ValueError(len(objs))
    l1 = "Objects: " + "".join(fmt_object(r) for r in objs) + C._fill("<none>", C.SLOT_FIELDS_NUMERIC) * (C.MAX_OBJECTS - len(objs))
    l2 = C.fmt_map(mp)
    l3 = "Intent: " + ("<na><_>" if intent_ans is None else C.LON_TOK[intent_ans["longitudinal"]] + C.LAT_TOK[intent_ans["lateral"]])
    if toks is not None and len(toks) != ACTION_N:
        raise ValueError(f"행동 토큰 {len(toks)} != {ACTION_N}")
    l4 = "Action: " + (_NA_ACTION if toks is None else "".join(f"<action_{k}>" for k in toks))
    return "\n".join([l1, l2, l3, l4])


def parse_answer(text):
    """엄격 파서. 1–3 줄은 common_v2._parse_full(클래스 토큰만 사전 교체), 4 줄은 8 행동. 실패 시 ValueError.
    반환 (parsed, spans) — spans 는 common_v2 와 같은 키(lines, speed, na; 문자 구간은 원문 기준)."""
    lines = text.split("\n")
    if len(lines) != 4:
        raise ValueError(f"줄 수 {len(lines)} != 4")
    # 클래스 토큰: '<czone>'(7 글자) ↔ '<truck>'(7 글자), '<obj>'(5) ↔ '<bus>'(5) 로 같은 길이 치환 → 구간 그대로 유효.
    #   원문에 이미 '<truck>'/'<bus>' 가 있으면 되돌릴 때 구별이 안 되므로(NAVSIM 정답에는 생기지 않는다) 즉시 실패.
    if "<truck>" in lines[0] or "<bus>" in lines[0]:
        raise ValueError("NAVSIM 정답에 nuScenes 전용 클래스 토큰")
    l0 = lines[0].replace("<czone>", "<truck>").replace("<obj>", "<bus>")
    fake = "\n".join([l0, lines[1], lines[2], "Action: " + C._fill("<na>", 6)])
    parsed, sp = C._parse_full(fake, "numeric")
    for o in parsed["objects"]:
        o["name"] = {"truck": "czone_sign", "bus": "generic_object"}.get(o["name"], o["name"])
    body = lines[3]
    if body == "Action: " + _NA_ACTION:
        act = None
    else:
        if not body.startswith("Action: "):
            raise ValueError("Action 머리말")
        ms = list(_ACT_RE.finditer(body[8:]))
        if len(ms) != ACTION_N or "".join(m.group(0) for m in ms) != body[8:]:
            raise ValueError("Action 줄 문법 불일치")
        act = [int(m.group(1)) for m in ms]
        if not all(0 <= k < ACTION_K for k in act):
            raise ValueError("action 토큰 범위 밖")
    parsed["action"] = act
    a = sp["lines"]["action"][0]
    sp["lines"]["action"] = [a, a + len(body)]
    if build_answer(parsed["objects"], _fmt_map_back(parsed["map"]), parsed["intent"], act) != text:
        raise ValueError("왕복 불일치")
    return parsed, sp


def _fmt_map_back(mp):
    if mp is not None and mp["ped_crossing"] is not None and mp["ped_crossing"]["sector"] == "here":
        mp = dict(mp, ped_crossing=dict(dist=0.0, sector="here"))
    return mp


def answer_spans(text):
    sp = parse_answer(text)[1]
    return dict(lines=sp["lines"], object_speed=sp["speed"], speed=sp["speed"], na=sp["na"])


def loss_mask_spans(spans, complete, prev_available):
    return C.loss_mask_spans(spans, complete, prev_available)
