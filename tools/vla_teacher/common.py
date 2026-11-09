# 수정(2026-10-15): 교사 VLM ① 정답 자동 생성 — 공용 모듈(문법·임계값·파서·토큰화).
#   교사 입력 = 6 카메라 + 주행 명령 + 자차 상태(CAN). 학생 예측·미래 정보(명령 제외) 입력 금지.
#   교사 출력 = 인지(3D 사실) → 판단(의도) → 행동(코드북 토큰) 순서의 결정적·파싱가능 텍스트.
#   기존 조각 재사용(수정 없이 import): tools/qwen3d_nusc/spatial_common.py (카메라 순서·로더 규약·구 파서·보상),
#   tools/qwen3d_nusc/nusc_action_codebook.py (heading_series — 코드북 생성과 같은 접선 방향 규약).
#   1행(Perception)은 구 공간 캡션(archive/tools/qwen3d/make_spatial_captions.py) 문법을 그대로 유지하고
#   각 객체 뒤에 "(<lat>m left|right, <lon>m ahead|behind)" 상대위치 단서만 덧붙인다 → 구 parse_caption /
#   smooth_reward 가 그대로 동작(검증: check_teacher_gt.py 의 왕복 검사).
import os, re, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "qwen3d_nusc"))
from spatial_common import CAMS as QWEN_CAMS, parse_caption, SECTORS  # noqa: E402  (F,FL,FR,BL,B,BR)
from nusc_action_codebook import heading_series  # noqa: E402

# ---------------------------------------------------------------- 상수(문서화된 규칙)
DET_CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
               "motorcycle", "bicycle", "pedestrian", "traffic_cone"]   # configs/nuscenes/stage1/train_v2.py det_class_names
OBJ_RANGE = 55.0          # v2 레시피 CircleObjectRangeFilter(전 클래스 55 m, 원형, lidar xy 거리)
MAX_OBJECTS = 12          # 구 캡션과 동일
MOVING_TH = 0.5           # m/s, 구 캡션과 동일(이상 → moving)
PED_CROSS_RANGE = 20.0    # m, 지도 문맥 줄
ACTION_K = 2041           # data/qwen3d/nusc_action_codebook_k2048.npz 의 실제 대표 수(K-disk 이분탐색 결과, 목표 2048 의 ±1%)
DT = 0.5                  # s, nuScenes 미래 궤적 간격(2 Hz × 6 = 3 s)

# 의도(meta-action) 임계값 — train split 통계로 고정(gen_teacher_gt.py --explore 로 재현, val 미사용).
#   종방향: v0 = 현재 자차 속도(CAN ego_status vel, CAN 결측 시 포즈 차분), v_end = 마지막 1 s 평균 속도 |p6-p4|/1.0.
#     stop        : v_end < STOP_V (0.5 m/s = 객체 moving 기준과 같은 값)
#     accelerate  : v_end - v0 >= +DV_TH
#     decelerate  : v_end - v0 <= -DV_TH
#     keep speed  : 그 외
#     DV_TH=1.0 m/s(Δt≈2.5 s → 평균 0.4 m/s²): train 에서 |CAN 속도 − 포즈 차분 속도|(측정 잡음) 의 p95=0.85 m/s 를
#     넘는 가장 작은 0.5 단위 값. (train Δv 분포 p20=-0.96, p80=+1.03 → 대략 20/60/20 분할)
#   횡방향: θ_end = 3 s 끝 진행 방향(heading_series, 반시계 + = 왼쪽), s = 경로 길이,
#     r = x_end − x_arc(θ_end, s) (등곡률 원호로 설명되지 않는 횡변위, +x = 오른쪽 — 참고값으로만 targets 에 기록)
#     s < MIN_PATH(2 m)                  → go straight (방향 불확정: 정지·서행)
#     θ_end >= +TURN_TH                  → turn left ;  θ_end <= −TURN_TH → turn right
#     그 외, 미래 경로(0→p6 꺾은선)가 차선 구분선(map_annos 라벨 1 = divider) 을 가로지르면
#                                        → lane change left/right (마지막 교차의 횡방향; 진행방향 기준 외적 부호)
#     그 외 go straight
#     TURN_TH=12.5°: train 에서 {5,10,12.5,15,17.5,20,25,30}° 중 nuScenes 표준 명령(x_end ±2 m 규칙)과의
#       일치율 최대(0.9434).
#     수정(2026-10-15, 근본수정): 처음엔 차로변경을 궤적 잔차 |r|>=1.0 m(비회전 train |r| p98=0.92 m)로 정했으나
#       contact sheet 점검에서 '회전 꼬리(현재 yaw rate −0.35 rad/s 로 우회전 마무리 중)' 가 lane change right 로
#       찍힌 것을 발견. train 점검: 잔차 규칙 차로변경 242건 중 177건은 현재 yaw rate 가 이동방향으로 >0.1 rad/s
#       (회전/곡선 탈출), 실제 구분선 교차 412건 중 잔차 규칙이 맞춘 것은 21건뿐. 궤적만으로는 '도로 대비 자차
#       방향'을 알 수 없는 것이 원인 → 지도 GT(구분선)로 차로변경을 정의한다(구분선 교차 train 표본의 현재
#       yaw rate 중앙값 ±0.03 rad/s — 차로변경과 부합). 한계: 3 s 안에 중심이 구분선을 넘는 경우만 잡힌다(추정).
#     (측정 잡음 p95 은 CAN 결측 장면 제외 train 기준. 수치는 --explore 출력으로 재현)
STOP_V = 0.5
DV_TH = 1.0
TURN_TH_DEG = 12.5
MIN_PATH = 2.0

CMD_TEXT = {0: "turn right", 1: "turn left", 2: "go straight"}   # nuscenes_converter: [1,0,0]=R, [0,1,0]=L, [0,0,1]=S
LONG_CLASSES = ["accelerate", "keep speed", "decelerate", "stop"]
LAT_CLASSES = ["go straight", "turn left", "turn right", "lane change left", "lane change right"]
MAP_NAMES = ["ped_crossing", "divider", "boundary"]   # map_annos 라벨 0/1/2 (converter MAP_CLASSES)


# ---------------------------------------------------------------- 기하
def sector(bearing_deg):  # 구 캡션과 동일
    return SECTORS[int(((bearing_deg + 22.5) % 360) // 45)]


def bearing_xy(x, y):     # lidar: +x 오른쪽, +y 전방 → 0°=front, 시계방향 +
    return float(np.degrees(np.arctan2(x, y))) % 360


def ego_speed_from_status(es):
    es = np.asarray(es, np.float64)
    return float(np.linalg.norm(es[6:8]))   # SparseDrive ego_status: [acc3, rot_rate3, vel3, steer]


def can_available(es):
    """converter get_ego_status 는 CAN 결측 장면에서 예외 → 0 벡터 10개. 이를 '결측'으로 명시한다."""
    return not bool(np.all(np.asarray(es) == 0))


# ---------------------------------------------------------------- 객체(인지)
def object_rows(info):
    boxes = np.asarray(info["gt_boxes"], np.float64)
    names = np.asarray(info["gt_names"])
    vel = np.asarray(info["gt_velocity"], np.float64)
    npts = np.asarray(info["num_lidar_pts"])
    inst = list(info.get("instance_inds", [-1] * len(boxes)))
    rows = []
    for j in range(len(boxes)):
        if not npts[j] > 0:                      # v2 레시피: num_lidar_pts>0
            continue
        name = str(names[j])
        if name not in DET_CLASSES:              # 10 검출 클래스 외 제외
            continue
        x, y = float(boxes[j, 0]), float(boxes[j, 1])
        dist = (x * x + y * y) ** 0.5
        if dist > OBJ_RANGE:                     # 원형 55 m
            continue
        vx, vy = float(vel[j, 0]), float(vel[j, 1])
        if not (np.isfinite(vx) and np.isfinite(vy)):
            vx = vy = 0.0                        # 구 캡션과 같은 처리(속도 미상 → 정지로 서술). 개수는 stats 에 기록
        spd = (vx * vx + vy * vy) ** 0.5
        bear = bearing_xy(x, y)
        rows.append(dict(
            name=name, dist=round(dist, 1), bearing=round(bear, 1), sector=sector(bear),
            speed=round(spd, 1), vel_sector=sector(bearing_xy(vx, vy)) if spd >= MOVING_TH else None,
            yaw_deg=round(float(np.degrees(boxes[j, 6])) % 360, 1),
            lat=round(abs(x), 1), lat_side="right" if x >= 0 else "left",
            lon=round(abs(y), 1), lon_side="ahead" if y >= 0 else "behind",
            x=round(x, 3), y=round(y, 3), box=[round(float(v), 4) for v in boxes[j, :7]],
            vel_finite=bool(np.isfinite(vel[j]).all()), num_lidar_pts=int(npts[j]),
            instance_ind=int(inst[j]), _d=dist))
    rows.sort(key=lambda r: r["_d"])               # 원값 거리로 정렬(가까운 순 = 깊이 순서)
    n_in_range = len(rows)
    rows = rows[:MAX_OBJECTS]
    for r in rows:
        r.pop("_d")
    return rows, n_in_range


def fmt_object(r):
    s = f"{r['sector']} {r['dist']:.1f}m {r['name']}"
    s += f" moving {r['vel_sector']} {r['speed']:.1f}m/s" if r["vel_sector"] else " stationary"
    s += f" ({r['lat']:.1f}m {r['lat_side']}, {r['lon']:.1f}m {r['lon_side']})"
    return s


# ---------------------------------------------------------------- 지도 문맥(선택)
def map_context(info):
    """map_annos(lidar 프레임, ROI x±15 m / y±30 m — converter roi_size=(30,60)) 에서 싸게 얻는 문맥.
    반환 dict 또는 None(map_annos 없음)."""
    from shapely.geometry import LineString, Polygon, Point
    from shapely.ops import nearest_points
    ma = info.get("map_annos")
    if ma is None:
        return None
    o = Point(0.0, 0.0)
    best = None
    for pl in ma.get(0, []):
        pl = np.asarray(pl, np.float64)
        if len(pl) < 2:
            continue
        g = Polygon(pl) if (len(pl) >= 4 and np.allclose(pl[0], pl[-1])) else LineString(pl)
        if not g.is_valid:
            g = LineString(pl)
        d = g.distance(o)
        if best is None or d < best[0]:
            q = nearest_points(g, o)[0]
            best = (d, q.x, q.y)
    out = dict(n_divider=len(ma.get(1, [])), n_boundary=len(ma.get(2, [])),
               ped_crossing=None)
    if best is not None and best[0] <= PED_CROSS_RANGE:
        d, qx, qy = best
        out["ped_crossing"] = dict(dist=round(d, 1),
                                   sector="here" if d < 0.05 else sector(bearing_xy(qx, qy)))
    return out


def fmt_map(m):
    if m is None:
        return "Map: unavailable."
    pc = m["ped_crossing"]
    if pc is None:
        s = f"no ped_crossing within {PED_CROSS_RANGE:.0f}m"
    elif pc["sector"] == "here":
        s = "on ped_crossing"
    else:
        s = f"ped_crossing {pc['dist']:.1f}m {pc['sector']}"
    return f"Map: {s}; dividers {m['n_divider']}; boundaries {m['n_boundary']}."


# ---------------------------------------------------------------- 의도(판단) — 라벨이므로 미래 사용 가능
def future_complete(info):
    return bool((np.asarray(info["gt_ego_fut_masks"]) == 1).all())


def fut_cum(info):
    return np.asarray(info["gt_ego_fut_trajs"], np.float64).cumsum(0)[:6, :2]


def divider_crossing(xy, dividers):
    """xy: (T+1,2) 경로(원점 포함, lidar: +x 오른쪽 +y 전방). dividers: map_annos[1] 폴리라인들.
    반환 'left'|'right'|None — 마지막 교차의 횡방향. 구분선 조각 방향 u 를 진행방향 v 와 같은 쪽으로 돌린 뒤
    u×v > 0 이면 진행방향 기준 왼쪽으로 넘어간 것(x 오른쪽·y 전방은 표준 우수 2D 좌표)."""
    last = None
    for a in range(len(xy) - 1):
        P0, v = xy[a], xy[a + 1] - xy[a]
        for d in dividers:
            d = np.asarray(d, np.float64)
            for b in range(len(d) - 1):
                Q0, u = d[b], d[b + 1] - d[b]
                den = v[0] * u[1] - v[1] * u[0]
                if abs(den) < 1e-9:
                    continue
                w = Q0 - P0
                t = (w[0] * u[1] - w[1] * u[0]) / den
                q = (w[0] * v[1] - w[1] * v[0]) / den
                if 0 <= t <= 1 and 0 <= q <= 1:
                    uu = u if np.dot(u, v) >= 0 else -u
                    last = "left" if (uu[0] * v[1] - uu[1] * v[0]) > 0 else "right"
    return last


def intent_labels(fut, v0, dividers):
    xy = np.concatenate([np.zeros((1, 2)), fut], 0)
    th = float(heading_series(xy)[-1])
    s = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    v_end = float(np.linalg.norm(fut[5] - fut[3]) / (2 * DT))
    dv = v_end - v0
    if v_end < STOP_V:
        lon = "stop"
    elif dv >= DV_TH:
        lon = "accelerate"
    elif dv <= -DV_TH:
        lon = "decelerate"
    else:
        lon = "keep speed"
    x_arc = -s * (1 - np.cos(th)) / th if abs(th) > 1e-6 else 0.0   # 왼쪽 회전(θ>0) → x 음수
    r = float(fut[5, 0] - x_arc)
    tdeg = float(np.degrees(th))
    cross = divider_crossing(xy, dividers)
    if s < MIN_PATH:
        lat = "go straight"
    elif tdeg >= TURN_TH_DEG:
        lat = "turn left"
    elif tdeg <= -TURN_TH_DEG:
        lat = "turn right"
    elif cross is not None:
        lat = f"lane change {cross}"
    else:
        lat = "go straight"
    return dict(longitudinal=lon, lateral=lat, v0=round(v0, 3), v_end=round(v_end, 3), dv=round(dv, 3),
                theta_end_deg=round(tdeg, 2), path_len=round(s, 3), lat_resid=round(r, 3),
                divider_crossing=cross)


# ---------------------------------------------------------------- 행동 토큰(코드북) — nusc_action_codebook.main().tokenize 와 동일 알고리즘
class ActionCodebook:
    def __init__(self, path):
        z = np.load(path)
        self.book = np.asarray(z["codebook"], np.float64)   # (K,3) Δx,Δy,Δθ (조각 시작 자세 기준)
        self.K = len(self.book)
        assert self.K == ACTION_K, f"코드북 크기 {self.K} != ACTION_K {ACTION_K} (문법의 토큰 범위와 불일치)"

    def tokenize(self, fut):
        book = self.book
        pos, th, toks, rec = np.zeros(2), 0.0, [], []
        for t in range(len(fut)):
            c, s = np.cos(-th), np.sin(-th)
            d = fut[t] - pos
            tgt = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
            j = int(np.argmin(np.linalg.norm(book[:, :2] - tgt, axis=1)))
            toks.append(j)
            c2, s2 = np.cos(th), np.sin(th)
            pos = pos + np.array([c2 * book[j, 0] - s2 * book[j, 1], s2 * book[j, 0] + c2 * book[j, 1]])
            th += book[j, 2]
            rec.append(pos.copy())
        return toks, np.array(rec)

    def decode(self, toks):
        pos, th, rec = np.zeros(2), 0.0, []
        for j in toks:
            c2, s2 = np.cos(th), np.sin(th)
            b = self.book[j]
            pos = pos + np.array([c2 * b[0] - s2 * b[1], s2 * b[0] + c2 * b[1]])
            th += b[2]
            rec.append(pos.copy())
        return np.array(rec)


# ---------------------------------------------------------------- 프롬프트(입력만: 명령 + 자차 상태)
SYSTEM_PROMPT = (
    "You are a driving teacher model for an autonomous vehicle. You see the current surround-view "
    "camera images and are given the high-level navigation command and the current ego vehicle state. "
    "You must describe what is around the vehicle, decide the driving intent, and output the ego "
    "trajectory for the next 3 seconds as action tokens."
)

INSTRUCTION = (
    "Inputs: 6 surround-view camera images in order: front, front-left, front-right, back-left, back, "
    "back-right; the navigation command (go straight, turn left or turn right); the current ego state "
    "from the vehicle bus (speed, longitudinal acceleration, yaw rate; it may be unavailable).\n"
    "Answer in exactly four lines, in this order:\n"
    "1) Perception: 'Ego speed <v> m/s', then up to 12 objects within 55m nearest first, separated by "
    "'; ' and ending with '.'. Each object is '<sector> <distance>m <class> moving <direction> "
    "<speed>m/s (<lateral>m left|right, <longitudinal>m ahead|behind)' or '<sector> <distance>m <class> "
    "stationary (<lateral>m left|right, <longitudinal>m ahead|behind)'. Sectors and directions: front, "
    "front-right, right, back-right, back, back-left, left, front-left. Classes: car, truck, "
    "construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone. "
    "Numbers have one decimal.\n"
    "2) Map: 'ped_crossing <distance>m <sector>' or 'on ped_crossing' or 'no ped_crossing within 20m', "
    "then '; dividers <n>; boundaries <n>.'\n"
    "3) Intent: '<longitudinal>, <lateral>.' with longitudinal one of accelerate, keep speed, decelerate, "
    "stop and lateral one of go straight, turn left, turn right, lane change left, lane change right.\n"
    f"4) Action: six tokens <action_k> (k in 0-{ACTION_K - 1}), one per 0.5 s for the next 3 s, separated by "
    "spaces and ending with '.'."
)


def ego_state_text(es):
    if not can_available(es):
        return "Ego state: unavailable."
    es = np.asarray(es, np.float64)
    return (f"Ego state: speed {ego_speed_from_status(es):.1f} m/s, longitudinal acceleration "
            f"{es[0]:+.2f} m/s^2, yaw rate {es[5]:+.3f} rad/s (positive = turning left).")


def build_user_prompt(cmd_id, es):
    """입력은 명령 id 와 ego_status(현재 CAN) 둘뿐 — 시그니처로 미래 정보 유입을 막는다."""
    return f"{INSTRUCTION}\nCommand: {CMD_TEXT[int(cmd_id)]}.\n{ego_state_text(es)}"


# 샘플별 가변부 정규식(누설 검사용): 이 두 줄 외 가변 텍스트가 없어야 한다.
PROMPT_VAR_RE = re.compile(
    r"\nCommand: (go straight|turn left|turn right)\.\n"
    r"Ego state: (?:unavailable\.|speed \d+\.\d m/s, longitudinal acceleration [+-]\d+\.\d\d m/s\^2, "
    r"yaw rate [+-]\d+\.\d\d\d rad/s \(positive = turning left\)\.)$")


# ---------------------------------------------------------------- 정답 조립 / 파서
def build_answer(ego_speed, objs, mp, intent, toks):
    parts = [f"Ego speed {ego_speed:.1f} m/s"] + [fmt_object(r) for r in objs]
    l1 = "Perception: " + "; ".join(parts) + "."
    l2 = fmt_map(mp)
    l3 = "Intent: unavailable." if intent is None else f"Intent: {intent['longitudinal']}, {intent['lateral']}."
    l4 = "Action: unavailable." if toks is None else "Action: " + " ".join(f"<action_{k}>" for k in toks) + "."
    return "\n".join([l1, l2, l3, l4])


_S = r"(front-right|front-left|back-right|back-left|front|back|right|left)"
_OBJ_FULL = re.compile(_S + r" (\d+\.\d)m ([a-z_]+)(?: moving " + _S + r" (\d+\.\d)m/s| stationary)"
                       r" \((\d+\.\d)m (left|right), (\d+\.\d)m (ahead|behind)\)$")
_L1 = re.compile(r"^Perception: Ego speed (\d+\.\d) m/s((?:; [^;]+)*)\.$")
_L2 = re.compile(r"^Map: (?:(no ped_crossing within 20m)|(on ped_crossing)|ped_crossing (\d+\.\d)m " + _S +
                 r"); dividers (\d+); boundaries (\d+)\.$|^Map: unavailable\.$")
_L3 = re.compile(r"^Intent: (?:unavailable|(accelerate|keep speed|decelerate|stop), "
                 r"(go straight|turn left|turn right|lane change left|lane change right))\.$")
_L4 = re.compile(r"^Action: (?:unavailable|((?:<action_\d+> ){5}<action_\d+>))\.$")


def parse_answer(text):
    """엄격 문법 파서. 실패 시 ValueError. 반환 dict(ego_speed, objects, map, intent, action)."""
    lines = text.split("\n")
    if len(lines) != 4:
        raise ValueError(f"줄 수 {len(lines)} != 4")
    m1 = _L1.match(lines[0])
    if not m1:
        raise ValueError("Perception 줄 문법 불일치")
    objs = []
    body = m1.group(2)
    for seg in ([s for s in body.split("; ")[1:]] if body else []):
        g = _OBJ_FULL.match(seg)
        if not g:
            raise ValueError(f"객체 문법 불일치: {seg}")
        objs.append(dict(sector=g.group(1), dist=float(g.group(2)), name=g.group(3), vel_sector=g.group(4),
                         speed=float(g.group(5)) if g.group(5) else None,
                         lat=float(g.group(6)), lat_side=g.group(7), lon=float(g.group(8)), lon_side=g.group(9)))
    m2 = _L2.match(lines[1])
    if not m2:
        raise ValueError("Map 줄 문법 불일치")
    if lines[1] == "Map: unavailable.":
        mp = None
    else:
        if m2.group(1):
            pc = None
        elif m2.group(2):
            pc = dict(dist=None, sector="here")
        else:
            pc = dict(dist=float(m2.group(3)), sector=m2.group(4))
        mp = dict(ped_crossing=pc, n_divider=int(m2.group(5)), n_boundary=int(m2.group(6)))
    m3 = _L3.match(lines[2])
    if not m3:
        raise ValueError("Intent 줄 문법 불일치")
    intent = None if m3.group(1) is None else dict(longitudinal=m3.group(1), lateral=m3.group(2))
    m4 = _L4.match(lines[3])
    if not m4:
        raise ValueError("Action 줄 문법 불일치")
    act = None if m4.group(1) is None else [int(k) for k in re.findall(r"<action_(\d+)>", m4.group(1))]
    return dict(ego_speed=float(m1.group(1)), objects=objs, map=mp, intent=intent, action=act)
