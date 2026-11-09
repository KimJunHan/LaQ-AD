# 수정(2026-10-15): 교사 VLM ① 정답 v2 (사용자 승인 4개 개선) — 공용 모듈(v2 문법·특수토큰·엄격 파서·
#   객체 중요도 선택·과거 포즈 자차 상태·copy-baseline·분리 가능한 프롬프트).
#   v1(common.py)은 수정하지 않는다 — 임계값·의도 규칙·코드북 토큰화·지도 문맥은 v1 함수를 그대로 import 해 재사용
#   (종방향·회전 판정과 행동 토큰은 v1 과 비트 단위 동일). 바뀌는 것:
#     ② 지름길 완화: 정답에서 자차 속도 제거(입력 중복), 프롬프트 = 분리 가능한 줄(prompt_parts) → SFT 입력 드롭아웃,
#        copy-baseline(명령만으로 의도, 등속·등yaw-rate 3 s 궤적 → 행동 토큰) 표본별 저장.
#     ③ CAN 결측: 과거 포즈(t−1, t−2 샘플)만으로 속도·종가속·yaw rate 추정(ego_status_src='pose_past'),
#        과거 없음(장면 첫 프레임) → 'unavailable'. ego_drop_ok 필드.
#     ④ 짧은 정답: 특수토큰 문법 + 중요도 순 객체 최대 8개(선택 후 거리순 정렬).
#     ① 차로변경은 lane_v2.py.
import os, re, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common as C1  # noqa: E402  (v1 — 수정 금지, 재사용만)
from common import (QWEN_CAMS, DET_CLASSES, OBJ_RANGE, MOVING_TH, ACTION_K, DT, CMD_TEXT,  # noqa: E402,F401
                    LONG_CLASSES, LAT_CLASSES, sector, bearing_xy, ego_speed_from_status, can_available,
                    map_context, future_complete, fut_cum, ActionCodebook, heading_series)

# ---------------------------------------------------------------- ④ 특수 토큰
CLS_TOK = {"car": "<car>", "truck": "<truck>", "bus": "<bus>", "trailer": "<trailer>",
           "construction_vehicle": "<cv>", "pedestrian": "<ped>", "motorcycle": "<moto>", "bicycle": "<bike>",
           "traffic_cone": "<cone>", "barrier": "<barrier>"}
SEC_TOK = {"front": "<F>", "front-left": "<FL>", "front-right": "<FR>", "left": "<L>", "right": "<R>",
           "back-left": "<BL>", "back": "<B>", "back-right": "<BR>"}
LON_TOK = {"accelerate": "<accelerate>", "keep speed": "<keep_speed>", "decelerate": "<decelerate>", "stop": "<stop>"}
LAT_TOK = {"go straight": "<go_straight>", "turn left": "<turn_left>", "turn right": "<turn_right>",
           "lane change left": "<lane_change_left>", "lane change right": "<lane_change_right>"}
MISC_TOK = ["<static>", "<none>", "<na>", "<pc>", "<pc_on>", "<pc_none>", "<div>", "<bnd>"]
ACTION_TOK = [f"<action_{k}>" for k in range(ACTION_K)]
# 수정(2026-10-15): 정답 형식 'v2-sensor'(고정 길이·센서 대체 정보) 신규 특수토큰 — 기존 토큰 순서는 그대로 두고 뒤에 붙인다.
#   <hX> = 객체 진행(박스 yaw) 방위(자차 전방 기준 8 섹터), <ttc_*> = GT 미래 기반 충돌 위험 구간,
#   <at0>/<at10>/<at20> = 자차 전방 0/10/20 m 단면의 차선·경계 횡거리 묶음 머리말, <free> = 전방 자유거리,
#   <_> = 고정 길이 채움(구조적 빈칸 — 감독함). <na>(값 미상 — 손실 제외)·<none>(없음 — 감독)은 v2 토큰 재사용.
HEAD_TOK = {k: "<h" + v[1:] for k, v in SEC_TOK.items()}          # front → <hF>, back-left → <hBL> ...
TTC_TOK = {"lt2": "<ttc_lt2>", "2_4": "<ttc_2_4>", "4_6": "<ttc_4_6>", "safe": "<ttc_safe>"}
LANE_Y = (0, 10, 20)                                                # m, 자차 전방(+y) 단면
AT_TOK = {y: f"<at{y}>" for y in LANE_Y}
SENSOR_TOK = list(HEAD_TOK.values()) + list(TTC_TOK.values()) + list(AT_TOK.values()) + ["<free>", "<_>"]
SPECIAL_TOKENS = (list(CLS_TOK.values()) + list(SEC_TOK.values()) + MISC_TOK + list(LON_TOK.values()) +
                  list(LAT_TOK.values()) + ACTION_TOK + SENSOR_TOK)
assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS))
_INV = lambda d: {v: k for k, v in d.items()}  # noqa: E731
TOK_CLS, TOK_SEC, TOK_LON, TOK_LAT = _INV(CLS_TOK), _INV(SEC_TOK), _INV(LON_TOK), _INV(LAT_TOK)
TOK_HEAD, TOK_TTC = _INV(HEAD_TOK), _INV(TTC_TOK)
ANSWER_FORMAT = "v2-sensor"   # 수정(2026-10-15): stats.json·레코드 modes 에 기록(이전 가변 길이 v2 정답과 구분)

# ---------------------------------------------------------------- ④ 객체 중요도 선택
MAX_OBJECTS = 8
CORRIDOR_HALF = 2.0      # m, 경로 회랑 반폭(객체 중심 ↔ 경로 꺾은선 거리)
TTC_MAX = 6.0            # s, 접근 객체(TTC 대리값) 상한
CLOSING_MIN = 0.5        # m/s, 접근 판정 최소 접근 속도(= moving 기준과 같은 값)
VRU = ("pedestrian", "bicycle", "motorcycle")
VRU_RANGE = 20.0         # m
STRAIGHT_MIN_LEN = 10.0  # m, 미래 불완전 시 '명령 직진 경로' 최소 길이(= max(3 s × v0, 10 m))
TIER_NAMES = {1: "path", 2: "approach", 3: "vru", 4: "near"}
# 문서화(라벨 규칙): 경로 회랑은 GT 미래 궤적(완전할 때)으로 정한다 — 정답(라벨) 쪽이므로 미래 사용 허용.
#   미래 불완전 표본은 현재 진행방향 직진 경로(길이 max(3 s·v0, 10 m))를 쓴다. 입력(prompt)에는 영향 없음.


def _r1(v):
    """1 소수 반올림 + −0.0 정규화(문자열 '+0.0' 와 왕복 일치)."""
    r = round(float(v), 1)
    return 0.0 if r == 0 else r


def _seg_dist(p, poly):
    best = np.inf
    for a in range(len(poly) - 1):
        q0, q1 = poly[a], poly[a + 1]
        u = q1 - q0; L2 = float(u @ u)
        t = 0.0 if L2 < 1e-12 else float(np.clip((p - q0) @ u / L2, 0, 1))
        best = min(best, float(np.linalg.norm(p - (q0 + t * u))))
    return best


def corridor_path(info, v0):
    if future_complete(info):
        return np.concatenate([np.zeros((1, 2)), fut_cum(info)], 0), "gt_future"
    L = max(3.0 * v0, STRAIGHT_MIN_LEN)
    return np.array([[0.0, 0.0], [0.0, L]]), "straight_cmd"


def select_objects(info, v0):
    """반환 (선택 rows(거리순), n_in_range, selection_meta). rows 는 v1 object_rows 와 같은 필터(num_lidar_pts>0,
    10 클래스, 55 m) 를 통과한 전부에서 고른다(v1 의 12개 절단 전 목록을 다시 만든다)."""
    boxes = np.asarray(info["gt_boxes"], np.float64)
    names = np.asarray(info["gt_names"]); vel = np.asarray(info["gt_velocity"], np.float64)
    npts = np.asarray(info["num_lidar_pts"]); inst = list(info.get("instance_inds", [-1] * len(boxes)))
    path, path_src = corridor_path(info, v0)
    vego = np.array([0.0, v0])                     # lidar 프레임 자차 속도(+y 전방)
    rows = []
    for j in range(len(boxes)):
        name = str(names[j])
        if not npts[j] > 0 or name not in DET_CLASSES:
            continue
        x, y = float(boxes[j, 0]), float(boxes[j, 1])
        dist = (x * x + y * y) ** 0.5
        if dist > OBJ_RANGE:
            continue
        vx, vy = float(vel[j, 0]), float(vel[j, 1])
        fin = bool(np.isfinite(vx) and np.isfinite(vy))
        if not fin:
            vx = vy = 0.0                            # v1 과 같은 처리(속도 미상 → 정지)
        spd = (vx * vx + vy * vy) ** 0.5
        moving = spd >= MOVING_TH
        p = np.array([x, y])
        cd = _seg_dist(p, path)
        rr = float((p @ (np.array([vx, vy]) - vego)) / max(dist, 1e-6))   # 거리 변화율(상대속도, − = 접근)
        ttc = dist / -rr if (moving and rr <= -CLOSING_MIN) else None
        if cd <= CORRIDOR_HALF:
            tier, key = 1, dist
        elif ttc is not None and ttc <= TTC_MAX:
            tier, key = 2, ttc
        elif name in VRU and dist <= VRU_RANGE:
            tier, key = 3, dist
        else:
            tier, key = 4, dist
        bear = bearing_xy(x, y)
        rows.append(dict(
            name=name, sector=sector(bear), lat=_r1(x), lon=_r1(y), dist=round(dist, 1),
            vel_sector=sector(bearing_xy(vx, vy)) if moving else None, speed=_r1(spd) if moving else None,
            reason=TIER_NAMES[tier], corridor_dist=round(cd, 2), ttc=None if ttc is None else round(ttc, 2),
            x=round(x, 3), y=round(y, 3), box=[round(float(v), 4) for v in boxes[j, :7]],
            vel_finite=fin, num_lidar_pts=int(npts[j]), instance_ind=int(inst[j]),
            box_index=j,   # 수정(2026-10-15): v2-sensor 필드(크기·진행방위·미래·TTC)를 같은 박스에서 계산하기 위한 infos 인덱스
            _k=(tier, key, dist), _d=dist))
    n_in = len(rows)
    rows.sort(key=lambda r: r["_k"])
    sel = rows[:MAX_OBJECTS]
    sel.sort(key=lambda r: r["_d"])                 # 선택 후 거리순(깊이 순서) 정렬
    nearest8 = {id(r) for r in sorted(rows, key=lambda r: r["_d"])[:MAX_OBJECTS]}
    meta = dict(path_src=path_src, n_tier={TIER_NAMES[t]: sum(r["_k"][0] == t for r in rows) for t in TIER_NAMES},
                n_selected_not_in_nearest8=sum(id(r) not in nearest8 for r in sel))
    for r in sel:
        r.pop("_k"); r.pop("_d")
    return sel, n_in, meta


# ---------------------------------------------------------------- ③ 과거 포즈 자차 상태
def _yaw(q):
    from pyquaternion import Quaternion
    return float(Quaternion(q).yaw_pitch_roll[0])


def ego_from_past_poses(cur, past):
    """cur: 현재 info, past: 같은 장면의 [t−1, t−2, ...] info(시간 역순). 미래 info 는 받지 않는다(시그니처로 차단).
    반환 dict(speed, accel_long, yaw_rate, n_past) 또는 None(과거 없음).
      speed     = |p0 − p−1| / Δt          (ego2global 위치 후진 차분, 0.5 s 평균 — 구간 중심 t−0.25 s)
      yaw_rate  = wrap(ψ0 − ψ−1) / Δt       (+ = 왼쪽, CAN rotation_rate z 와 같은 부호)
      accel_long= (v(t0) − v(t−1)) / Δt     (t−2 필요; 없으면 0.0 으로 두고 n_past=1 로 표시)"""
    if not past:
        return None

    def vel(a, b):
        dp = np.asarray(b["ego2global_translation"][:2], np.float64) - np.asarray(a["ego2global_translation"][:2], np.float64)
        return float(np.linalg.norm(dp) / ((b["timestamp"] - a["timestamp"]) * 1e-6))
    p1 = past[0]
    dt1 = (cur["timestamp"] - p1["timestamp"]) * 1e-6
    v0 = vel(p1, cur)
    dyaw = (_yaw(cur["ego2global_rotation"]) - _yaw(p1["ego2global_rotation"]) + np.pi) % (2 * np.pi) - np.pi
    w = dyaw / dt1
    if len(past) >= 2:
        acc = (v0 - vel(past[1], p1)) / dt1
        n = 2
    else:
        acc, n = 0.0, 1
    return dict(speed=v0, accel_long=acc, yaw_rate=w, n_past=n)


# ---------------------------------------------------------------- ② 프롬프트(분리 가능한 줄)
# 입력 모드(코디네이터 검토 → 사용자 결정 2026-10-15, gen_teacher_gt_v2 플래그로 선택·stats.json 기록):
#   ego_in_prompt: 'current'(기본: CAN, 결측이면 과거 포즈로 채움) | 'past_pose'(항상 과거 포즈 추정) | 'none'(자차 상태 줄 없음)
#     — 자차 상태 줄은 prompt_parts['ego'] 로 분리되어 있어 '캐시 프롬프트'(자차 상태 없는 변형; 학생 융합용 토큰 추출 시
#       HiP-AD Table 3 프로토콜 정합) 를 언제든 join_prompt(parts, drop_ego=True, ego_mode='omit') 로 만들 수 있다.
#   object_speed: 'numeric'(이동 방향 섹터 + 속력) | 'dir_only'(이동 방향 섹터만)
#   이전 키프레임(t−0.5 s) 6 카메라를 입력에 추가(cams_prev) — 단일 프레임으로는 관측 불가한 속력을 관측 가능하게.
EGO_MODES = ("current", "past_pose", "none")
SPEED_MODES = ("numeric", "dir_only")

SYSTEM_PROMPT = (
    "You are a driving teacher model for an autonomous vehicle. You see the surround-view camera images of the "
    "current frame and of the previous frame (0.5 s earlier), and may be given the high-level navigation command "
    "and the current ego vehicle state. You must describe the driving-relevant objects and map context, decide the "
    "driving intent, and output the ego trajectory for the next 3 seconds as action tokens."
)


def build_instruction(ego_in_prompt="current", object_speed="numeric"):
    assert ego_in_prompt in EGO_MODES and object_speed in SPEED_MODES
    inputs = ("Inputs: 12 camera images: the 6 surround-view cameras of the previous frame (0.5 s earlier), then "
              "the 6 of the current frame, each in order front, front-left, front-right, back-left, back, back-right "
              "(if the previous frame is missing the current frame is repeated); the navigation command")
    inputs += ("; the current ego state (speed, longitudinal acceleration, yaw rate). The command or the ego state "
               "may be unavailable.\n" if ego_in_prompt != "none" else ". The command may be unavailable.\n")
    motion = ("motion = <static> or the direction sector of motion, followed by the speed DD.D m/s; unknown = "
              "<na><_><_><_><_>" if object_speed == "numeric" else
              "motion = <static> or the direction sector of motion; unknown = <na>")
    n_slot = 57 if object_speed == "numeric" else 53   # = slot_tokens(object_speed)(아래 정의 — 모듈 끝 assert 로 일치 강제)
    # 수정(2026-10-15): 결정 1 — 교사 SFT 지시문이 v2-sensor(고정 길이) 정답 문법을 설명하도록 교체(입력 설명 문단은 그대로).
    #   학생 캐시용 지시문(tools/qwen_cache.py INSTRUCTION)과 명령·자차 상태 줄(B안 고정폭, 캐시 L 1,200)은 무관·무변경.
    return (inputs +
            "Answer in exactly four lines with a fixed number of tokens, no spaces inside fields. Numbers are "
            "fixed-width with one decimal: signed SDD.D (e.g. +09.2, -05.6, zero = +00.0) or unsigned DD.D (e.g. 04.5). "
            "Fillers keep the length fixed: <none> = absent, <na> = unknown, each followed by <_> to the field's length.\n"
            "Objects: exactly 8 slots. Up to 8 objects most relevant to driving (in the ego path, approaching the ego, "
            "vulnerable road users within 20m, then nearest), nearest first; then empty slots <none> followed by "
            f"{n_slot - 1} <_>. An object slot is <class><sector><x><y><length><width><heading><motion><future 1s>"
            "<future 2s><future 3s><risk>: x, y = signed centre position in metres of the current frame (+x = right, "
            "+y = ahead); length, width in metres; heading = direction the object faces relative to the ego "
            f"<hF><hFR><hR><hBR><hB><hBL><hL><hFL> (<na> for traffic cones); {motion}; future = signed x, y "
            "displacement in metres after 1, 2, 3 s (unknown = <na> and 9 <_>); risk = time until the object comes "
            "within 1.4m of the ego along the ego's future path: <ttc_lt2> (under 2 s), <ttc_2_4>, <ttc_4_6>, "
            "<ttc_safe> (none within 6 s), or <na>. Classes: <car><truck><bus><trailer><cv><ped><moto><bike><cone>"
            "<barrier>. Sectors (clockwise from front): <F><FR><R><BR><B><BL><L><FL>.\n"
            "Map: <pc><sector><distance> (nearest pedestrian crossing within 20m) or <pc_on> or <pc_none> (each "
            "followed by 5 <_>), <div><number of dividers DD><bnd><number of boundaries DD>, then for the cross-sections "
            "0, 10, 20 m ahead <at0>, <at10>, <at20> each followed by the unsigned distance to the nearest left divider, "
            "right divider, left boundary, right boundary (<none><_><_><_> if none within 15m), then <free> and the "
            "free distance ahead in the ego's width until the first boundary or object (30.0 if none within 30m); or "
            "<na> and 67 <_> if the map is unknown.\n"
            "Intent: <longitudinal><lateral> with longitudinal one of <accelerate><keep_speed><decelerate><stop> and "
            "lateral one of <go_straight><turn_left><turn_right><lane_change_left><lane_change_right>; or <na><_>.\n"
            f"Action: six action tokens <action_k> (k in 0-{ACTION_K - 1}), one per 0.5 s for the next 3 s; or <na> and 5 <_>.")


INSTRUCTION = build_instruction()   # 기본 모드(current, numeric)
CMD_DROPPED = "Command: unavailable."
# 수정(2026-10-15): 교사 토큰 자차 상태 포함(B안, 사용자 결정) — 캐시 프롬프트에 자차 상태 줄이 들어가므로 토큰 수를 고정한다.
#   Qwen2 토크나이저는 숫자를 한 자리씩 나눠 '4.2'(3) 와 '18.1'(4) 의 토큰 수가 달라 표본마다 캐시 길이 L 이 바뀌고,
#   배치 패딩 → flash varlen 역방향 경로(sm_120 에서 실패, qwen_token_attn.py 주석)를 탄다. 속도를 두 자리 고정폭(04.1f)으로 쓰고
#   (가속 +.2f · 요레이트 +.3f 는 |값|<10 이면 이미 고정폭), 결측 문장도 같은 토큰 수(앞 '\n' 포함 42)로 맞춘다.
#   실측(ckpts/qwen2-vl-2b 토크나이저): 값 조합 전부 42, 아래 결측 문장 42. 전 표본 검사는 check_teacher_gt_v2.py.
EGO_DROPPED = ("Ego state: unavailable. The speed, longitudinal acceleration and yaw rate of the ego vehicle were not "
               "measured for this frame, so infer its motion from the camera images (no CAN bus or past poses).")
EGO_LINE_TOKENS = 41   # 캐시 프롬프트(지시문 '…cues.' 뒤)에서 '\n' + 자차 상태 줄이 늘리는 토큰 수 — '.\n' 이 한 토큰으로 합쳐져
                       # 단독 측정(42)보다 1 적다. 실측 전 34,149 표본 41, 영상 포함 L = 1,159 → 1,200(값 있음·과거 포즈·결측 동일).


def command_line(cmd_id):
    return f"Command: {CMD_TEXT[int(cmd_id)]}."


def ego_line(speed, accel, yaw):
    """speed/accel/yaw 가 None 이면 unavailable. CAN 과 과거 포즈 추정은 같은 문장 형식(출처는 메타데이터에만)."""
    if speed is None:
        return EGO_DROPPED
    # 수정(2026-10-15): 고정폭(토큰 수 고정). 범위 밖이면 형식이 깨지므로 즉시 실패(실측 최대: 속도 18.1, |가속| 4.9, |요| 0.67).
    if not (0.0 <= speed < 99.95 and abs(accel) < 9.995 and abs(yaw) < 9.9995):
        raise ValueError(f"자차 상태가 고정폭 범위 밖: speed {speed}, accel {accel}, yaw {yaw}")
    return (f"Ego state: speed {speed:04.1f} m/s, longitudinal acceleration {accel:+.2f} m/s^2, "
            f"yaw rate {yaw:+.3f} rad/s (positive = turning left).")


def build_prompt_parts(cmd_id, ego_vals, ego_in_prompt="current", object_speed="numeric"):
    """입력 = 명령 id + 현재/과거 자차 상태 값(ego_vals = (speed, accel, yaw) 또는 None) 뿐.
    ego_in_prompt='none' 이면 parts['ego'] = None(줄 자체 없음)."""
    e = ego_vals if ego_vals is not None else (None, None, None)
    return dict(instruction=build_instruction(ego_in_prompt, object_speed), command=command_line(cmd_id),
                ego=None if ego_in_prompt == "none" else ego_line(*e),
                command_dropped=CMD_DROPPED, ego_dropped=EGO_DROPPED)


def join_prompt(parts, drop_command=False, drop_ego=False, ego_mode="blank"):
    """SFT 입력 드롭아웃용: 드롭 시 해당 줄을 '... unavailable.' 로 바꾼다(ego_mode='blank', 줄 위치 고정).
    ego_mode='omit' 이면 자차 상태 줄을 아예 뺀다(캐시 프롬프트 변형)."""
    lines = [parts["instruction"], parts["command_dropped"] if drop_command else parts["command"]]
    if parts["ego"] is not None and not (drop_ego and ego_mode == "omit"):
        lines.append(parts["ego_dropped"] if drop_ego else parts["ego"])
    return "\n".join(lines)


# ---------------------------------------------------------------- ② copy-baseline
def ctrv_traj(v, w, T=6, dt=DT):
    """등속·등yaw-rate 궤적(lidar: +x 오른쪽, +y 전방, w + = 왼쪽) → (T,2) 누적 위치."""
    t = dt * np.arange(1, T + 1)
    if abs(w) < 1e-6:
        return np.stack([np.zeros(T), v * t], 1)
    return np.stack([-(v / w) * (1 - np.cos(w * t)), (v / w) * np.sin(w * t)], 1)


def copy_baseline(cmd_id, ego_vals, cb):
    """명령만으로 의도(횡방향 = 명령, 종방향 = 등속 가정 → keep speed / v<0.5 면 stop), 자차 상태만으로 궤적."""
    v, w = (0.0, 0.0) if ego_vals is None else (float(ego_vals[0]), float(ego_vals[2]))
    traj = ctrv_traj(v, w)
    toks, rec = cb.tokenize(traj)
    return dict(intent=dict(longitudinal="stop" if v < C1.STOP_V else "keep speed", lateral=CMD_TEXT[int(cmd_id)]),
                ego_used=None if ego_vals is None else [round(v, 4), round(w, 5)],
                traj=[[round(float(a), 4) for a in p] for p in traj], action_tokens=toks)




# ================================================================ 수정(2026-10-15): 정답 형식 'v2-sensor'
#   (사용자 지시 'v2 와 통합': 별도 v3 대신 v2 정답을 확장. 프롬프트·B안 자차 상태 줄·캐시 L 1,200 은 그대로.)
#   목표: RGB 만으로 LiDAR/레이더가 주던 정보(위치·크기·진행방위·속도·미래 변위·충돌 위험·차선/경계 횡거리·자유거리)를
#   교사가 말하게 하고, 그 생성 토큰을 학생 쿼리에 캐시로 넣는다. 캐시 길이가 표본마다 같아야 하므로(가변 → 패딩 →
#   sm_120 에서 실패하는 flash varlen 역방향) 정답 토큰 수를 모든 표본에서 같게 만든다:
#     * 숫자는 고정폭(부호 있는 'SDD.D' = 5 토큰, 부호 없는 'DD.D' = 4 토큰, 정수 'DD' = 2 토큰 — Qwen2 는 숫자를 한 자리씩
#       자르고 '+', '-', '.' 은 숫자 사이에서 단독 토큰; 실측 검사는 check_teacher_gt_v2.py 가 전 표본에 대해 한다).
#     * 대체형(값 없음/미상/빈 칸)은 머리 토큰 1개 + '<_>' 채움으로 같은 토큰 수:
#         NONE 형 '<none><_>…' / '<pc_on><_>…' / '<pc_none><_>…' = 구조적으로 없음(결정적 → 감독)
#         NA  형 '<na><_>…'                                   = GT 값 미상(→ loss_mask_spans 로 손실 제외)
#     * 객체는 항상 8 칸(채운 칸 거리순 → 빈 칸). 범위를 넘는 값은 자르지 않고 즉시 실패(ValueError).
SLOT_FIELDS_NUMERIC = 57      # 칸당 토큰 수(object_speed='numeric'): 1+1+5+5+4+4+1+(1+4)+3·(5+5)+1
SLOT_FIELDS_DIRONLY = 53      # object_speed='dir_only' (속력 4 토큰 없음)
MAP_LINE_FIELDS = 68          # 'Map: ' 뒤 토큰 수: PC 6 + <div>DD<bnd>DD 6 + 3·(<atY> + 4·4) 51 + <free>DD.D 5
FUT_STEPS = (1, 3, 5)         # gt_agent_fut_trajs 인덱스(0.5 s 간격 증분) → 1 s, 2 s, 3 s 누적 변위
FREE_CAP = 30.0               # m, 자유거리 상한 = map_annos ROI 전방 한계(컨버터 roi_size=(30,60) → y ≤ 30 m).
                              #   사양 예시(50 m)보다 짧다: 30 m 너머 경계는 GT 에 없어서 50 m 까지 재면 '없음'과 '모름'이 섞인다.
EGO_L, EGO_W, EGO_FWD = 4.084, 1.85, 0.5   # tools/eval_nusc_planning.py(SparseDrive 평가기) 자차 박스·전방 오프셋
FREE_HALF_W = EGO_W / 2       # m, 자유거리 회랑 반폭(자차 폭)
# TTC(충돌 위험) 규칙 — 라벨(GT 미래) 쪽:
#   시각 t ∈ {0, 0.5, …, 6.0} s 에서 자차 박스(4.084×1.85, 0.5 m 전방 오프셋, 평가기 get_yaw 규약)와 객체 박스(l×w)의
#   최소 간격(shapely distance, 겹치면 0)이 TTC_GAP 이하가 되는 첫 t. 구간 t<2 → <ttc_lt2>, 2≤t<4 → <ttc_2_4>,
#   4≤t≤6 → <ttc_4_6>, 없음 → <ttc_safe>.
#   자차: 0–3 s = gt_ego_fut_trajs 누적, 3–6 s = 마지막 0.5 s 변위 등속 외삽(진행방향 고정). 자차 미래 불완전 → <na>(손실 제외).
#   객체: gt_agent_fut_trajs 누적(마스크 유효 구간), 무효가 시작된 뒤 = 마지막 유효 증분 등속 외삽, 유효 증분이 없으면
#         gt_velocity×0.5 s(NaN 이면 정지). 진행방향 = 증분 ≥0.25 m(0.5 m/s) 이면 증분 방향, 아니면 이전 값(t0 = 박스 yaw).
#   TTC_GAP: 순수 겹침(0 m)은 GT 미래끼리라 train 선택 객체의 99.6% 가 safe 로 퇴화(탐색 4,000 표본) → 간격 임계값을
#   train 전용 통계로 정한다: 미래 완전 train 표본의 선택 객체별 '0–3 s 최소 간격' 분포의 5 백분위수(0.1 m 반올림).
#   재현: gen_teacher_gt_v2.py --explore_ttc (train 전체, val 미사용). 값과 분포는 stats.json.thresholds 에 기록.
TTC_GAP = 1.4                 # m — --explore_ttc(train 23,854 미래완전 표본·177,777 객체): 0–3 s 최소 간격 p5 = 1.358 → 1.4.
                              #   이때 train 분포 lt2 4.5% · 2_4 1.1% · 4_6 0.85% · safe 93.6% (0 m 면 safe 99.6%).
TTC_T = [0.5 * k for k in range(13)]


def _r1c(v, lo, hi, what):
    """1 소수 반올림 + 범위 검사(자르지 않음 — 범위 밖이면 형식이 깨지므로 즉시 실패)."""
    r = _r1(v)
    if not (lo <= r <= hi):
        raise ValueError(f"{what}={v} 가 고정폭 범위 [{lo}, {hi}] 밖")
    return r


def _sn(v):   # 부호 있는 고정폭 'SDD.D'(5 토큰). −0.0 은 '+00.0'
    return f"{_r1(v):+05.1f}"


def _un(v):   # 부호 없는 고정폭 'DD.D'(4 토큰)
    return f"{_r1(v):04.1f}"


def _fill(head, n):
    return head + "<_>" * (n - 1)


def _rect(cx, cy, l, w, yaw):
    from shapely.geometry import Polygon
    c, s = np.cos(yaw), np.sin(yaw)
    return Polygon([(cx + c * a - s * b, cy + s * a + c * b) for a, b in
                    ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2))])


def _eval_yaw(traj):
    """tools/eval_nusc_planning.get_yaw 와 같은 규약(numpy 판)."""
    if np.linalg.norm(traj[-1] - traj[0]) < 0.5:
        return np.full(len(traj), np.pi / 2)
    cat = np.concatenate([np.zeros((1, 2)), traj], 0)
    y = np.zeros(len(traj) + 1)
    y[1:-1] = np.arctan2(cat[2:, 1] - cat[:-2, 1], cat[2:, 0] - cat[:-2, 0])
    y[-1] = np.arctan2(cat[-1, 1] - cat[-2, 1], cat[-1, 0] - cat[-2, 0])
    return y[1:]


def ego_track(info):
    """자차 13 시각(0..6 s) 박스 리스트 또는 None(자차 미래 불완전)."""
    if not future_complete(info):
        return None
    fut = fut_cum(info)
    yaw = _eval_yaw(fut)
    v = fut[5] - fut[4]
    pos = [np.zeros(2)] + list(fut) + [fut[5] + v * k for k in range(1, 7)]
    yws = [np.pi / 2] + list(yaw) + [yaw[-1]] * 6
    return [_rect(p[0] + EGO_FWD * np.cos(a), p[1] + EGO_FWD * np.sin(a), EGO_L, EGO_W, a) for p, a in zip(pos, yws)]


def agent_gap_series(info, j, ego):
    """객체 j 와 자차의 13 시각 최소 간격(m)."""
    b = np.asarray(info["gt_boxes"], np.float64)[j]
    f = np.asarray(info["gt_agent_fut_trajs"], np.float64)[j]
    m = np.asarray(info["gt_agent_fut_masks"])[j]
    vel = np.asarray(info["gt_velocity"], np.float64)[j]
    p, a = b[:2].copy(), float(b[6])
    last = (vel if np.isfinite(vel).all() else np.zeros(2)) * DT
    gaps = [ego[0].distance(_rect(p[0], p[1], b[3], b[4], a))]
    for t in range(12):
        if (m[:t + 1] == 1).all():
            st = f[t]; last = st
        else:
            st = last
        p = p + st
        if np.linalg.norm(st) >= 0.25:
            a = float(np.arctan2(st[1], st[0]))
        gaps.append(ego[t + 1].distance(_rect(p[0], p[1], b[3], b[4], a)))
    return np.asarray(gaps)


def ttc_bucket(gaps, gap_th):
    hit = np.nonzero(gaps <= gap_th)[0]
    if len(hit) == 0:
        return "safe", None
    t = TTC_T[int(hit[0])]
    return ("lt2" if t < 2 else "2_4" if t < 4 else "4_6"), t


def enrich_objects(info, objs, gap_th=None):
    """select_objects 의 rows 에 v2-sensor 필드 추가(제자리). 라벨이므로 GT 미래 사용 허용(입력 무관)."""
    gap_th = TTC_GAP if gap_th is None else gap_th
    boxes = np.asarray(info["gt_boxes"], np.float64)
    vel = np.asarray(info["gt_velocity"], np.float64)
    F = np.asarray(info["gt_agent_fut_trajs"], np.float64)
    M = np.asarray(info["gt_agent_fut_masks"])
    ego = ego_track(info)
    for r in objs:
        j = r["box_index"]
        b = boxes[j]
        r["lat"] = _r1c(b[0], -99.9, 99.9, "lat"); r["lon"] = _r1c(b[1], -99.9, 99.9, "lon")
        r["length"] = _r1c(b[3], 0.0, 99.9, "length"); r["width"] = _r1c(b[4], 0.0, 99.9, "width")
        # 진행방위: 박스 yaw(lidar, +x 기준 반시계) → 방향벡터 (cos, sin) → 자차 전방 기준 시계방향 방위 섹터.
        #   traffic_cone 은 방향이 정의되지 않음(nuScenes 평가도 cone 의 방향 오차 제외) → <na>(손실 제외)
        r["heading"] = None if r["name"] == "traffic_cone" else sector(bearing_xy(np.cos(b[6]), np.sin(b[6])))
        if r["vel_finite"]:
            spd = float(np.hypot(vel[j, 0], vel[j, 1]))
            r["motion"] = "moving" if spd >= MOVING_TH else "static"
            r["speed"] = _r1c(spd, 0.0, 99.9, "speed")
        else:
            r["motion"], r["speed"], r["vel_sector"] = "na", None, None   # GT 속도 NaN → 미상(손실 제외)
        cum = F[j].cumsum(0)
        r["fut"] = [[_r1c(cum[h, 0], -99.9, 99.9, "fut_x"), _r1c(cum[h, 1], -99.9, 99.9, "fut_y")]
                    if (M[j, :h + 1] == 1).all() else None for h in FUT_STEPS]
        if ego is None:
            r["ttc"], r["ttc_t"], r["gap_min3"] = None, None, None
        else:
            g = agent_gap_series(info, j, ego)
            r["ttc"], r["ttc_t"] = ttc_bucket(g, gap_th)
            r["gap_min3"] = round(float(g[:7].min()), 3)
    return objs


# ---------------------------------------------------------------- 지도 확장(차선·경계 횡거리, 자유거리)
def _cross_x(polys, y0):
    xs = []
    for pl in polys:
        pl = np.asarray(pl, np.float64)
        for a in range(len(pl) - 1):
            (x0, y0a), (x1, y1a) = pl[a], pl[a + 1]
            if y0a == y1a:
                if y0a == y0:
                    xs += [x0, x1]
                continue
            if (y0a - y0) * (y1a - y0) <= 0:
                xs.append(x0 + (y0 - y0a) * (x1 - x0) / (y1a - y0a))
    return xs


def lane_offsets(ma):
    """{y: dict(div_l, div_r, bnd_l, bnd_r)}: 자차 전방 y m 단면(직선 y=const, lidar 프레임)과 폴리라인 교점 중
    왼쪽(x<0) / 오른쪽(x≥0) 가장 가까운 것의 |x|(m). 없으면 None. ROI 가 |x|≤15 m 라 값 ≤ 15."""
    out = {}
    for y in LANE_Y:
        d = {}
        for key, cls in (("div", 1), ("bnd", 2)):
            xs = _cross_x(ma.get(cls, []), float(y))
            L = [-x for x in xs if x < 0]; R = [x for x in xs if x >= 0]
            d[key + "_l"] = _r1c(min(L), 0.0, 99.9, "lane") if L else None
            d[key + "_r"] = _r1c(min(R), 0.0, 99.9, "lane") if R else None
        out[str(y)] = d
    return out


def free_distance(info, ma):
    """자차 전방 자유거리(m): 회랑 |x| ≤ 자차 반폭(0.925 m), 0 ≤ y ≤ FREE_CAP 안에서 처음 만나는 도로 경계(map 라벨 2)
    또는 객체 박스(객체 필터 = select_objects 와 같음: num_lidar_pts>0·10 클래스·55 m)의 최소 y. 없으면 FREE_CAP.
    직선(현재 진행축) 회랑이다 — 차로 중심선을 따라가지 않는다(문서화된 단순화). 거리 기준 = lidar 원점."""
    from shapely.geometry import LineString, box as sbox
    cor = sbox(-FREE_HALF_W, 0.0, FREE_HALF_W, FREE_CAP)
    best = FREE_CAP
    for pl in ma.get(2, []):
        pl = np.asarray(pl, np.float64)
        if len(pl) < 2:
            continue
        g = LineString(pl).intersection(cor)
        if not g.is_empty:
            best = min(best, g.bounds[1])
    boxes = np.asarray(info["gt_boxes"], np.float64); names = np.asarray(info["gt_names"])
    npts = np.asarray(info["num_lidar_pts"])
    for j in range(len(boxes)):
        if not npts[j] > 0 or str(names[j]) not in DET_CLASSES or np.hypot(boxes[j, 0], boxes[j, 1]) > OBJ_RANGE:
            continue
        g = _rect(boxes[j, 0], boxes[j, 1], boxes[j, 3], boxes[j, 4], boxes[j, 6]).intersection(cor)
        if not g.is_empty:
            best = min(best, g.bounds[1])
    return _r1c(best, 0.0, FREE_CAP, "free")


def map_context_sensor(info):
    """v1 map_context(횡단보도·개수) + 단면 횡거리 + 자유거리. map_annos 없으면 None."""
    mp = map_context(info)
    if mp is None:
        return None
    ma = info["map_annos"]
    if not (mp["n_divider"] < 100 and mp["n_boundary"] < 100):
        raise ValueError(f"지도 개수 2 자리 초과: {mp['n_divider']}, {mp['n_boundary']}")
    mp["lanes"] = lane_offsets(ma)
    mp["free_dist"] = free_distance(info, ma)
    return mp


# ---------------------------------------------------------------- 정답 조립(고정 길이)
def slot_tokens(object_speed="numeric"):
    return SLOT_FIELDS_NUMERIC if object_speed == "numeric" else SLOT_FIELDS_DIRONLY


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


def _fmt_lane(v):
    return _fill("<none>", 4) if v is None else _un(v)


def fmt_map(m):
    if m is None:
        return "Map: " + _fill("<na>", MAP_LINE_FIELDS)
    pc = m["ped_crossing"]
    if pc is None:
        s = _fill("<pc_none>", 6)
    elif pc["sector"] == "here":
        s = _fill("<pc_on>", 6)
    else:
        s = "<pc>" + SEC_TOK[pc["sector"]] + _un(pc["dist"])
    s += f"<div>{m['n_divider']:02d}<bnd>{m['n_boundary']:02d}"
    for y in LANE_Y:
        d = m["lanes"][str(y)]
        s += AT_TOK[y] + "".join(_fmt_lane(d[k]) for k in ("div_l", "div_r", "bnd_l", "bnd_r"))
    return "Map: " + s + "<free>" + _un(m["free_dist"])


def build_answer(objs, mp, intent_ans, toks, object_speed="numeric"):
    """objs = enrich_objects 를 거친 rows(최대 8, 거리순). 빈 칸은 '<none><_>…'(칸 토큰 수와 같음).
    intent_ans = dict(longitudinal, lateral) 또는 None('<na><_>'), toks = 행동 6 토큰 또는 None('<na><_>×5')."""
    if len(objs) > MAX_OBJECTS:
        raise ValueError(f"객체 {len(objs)} > {MAX_OBJECTS}")
    l1 = "Objects: " + "".join(fmt_object(r, object_speed) for r in objs) + \
         _fill("<none>", slot_tokens(object_speed)) * (MAX_OBJECTS - len(objs))
    l2 = fmt_map(mp)
    l3 = "Intent: " + ("<na><_>" if intent_ans is None else LON_TOK[intent_ans["longitudinal"]] + LAT_TOK[intent_ans["lateral"]])
    l4 = "Action: " + (_fill("<na>", 6) if toks is None else "".join(f"<action_{k}>" for k in toks))
    return "\n".join([l1, l2, l3, l4])


# ---------------------------------------------------------------- 엄격 파서(커서 방식) + 구간
_SN_RE = re.compile(r"(?!-00\.0)[+-]\d\d\.\d")
_UN_RE = re.compile(r"\d\d\.\d")
_I2_RE = re.compile(r"\d\d")
_ACT_RE = re.compile(r"<action_(0|[1-9]\d*)>")


class _Cur:
    def __init__(self, s, base=0):
        self.s, self.p, self.base = s, 0, base

    def done(self):
        return self.p == len(self.s)

    def peek(self, opts):
        for o in sorted(opts, key=len, reverse=True):
            if self.s.startswith(o, self.p):
                return o
        return None

    def tok(self, opts, what):
        o = self.peek(opts)
        if o is None:
            raise ValueError(f"{what}: 위치 {self.p} 에서 {self.s[self.p:self.p + 20]!r}")
        self.p += len(o)
        return o

    def lit(self, t, what):
        return self.tok([t], what)

    def rx(self, r, what):
        m = r.match(self.s, self.p)
        if not m:
            raise ValueError(f"{what}: 위치 {self.p} 에서 {self.s[self.p:self.p + 20]!r}")
        self.p = m.end()
        return m.group(0), [self.base + m.start(), self.base + m.end()]

    def fill(self, head, n, what):
        a = self.p
        self.lit(head, what)
        for _ in range(n - 1):
            self.lit("<_>", what + " 채움")
        return [self.base + a, self.base + self.p]

    def is_fill(self, head):
        return self.s.startswith(head, self.p)


def _parse_object(c, object_speed, sp):
    """반환 객체 dict. sp = dict(speed=[], na=[]) 에 구간 추가."""
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
    if object_speed not in SPEED_MODES:
        raise ValueError(object_speed)
    lines = text.split("\n")
    if len(lines) != 4:
        raise ValueError(f"줄 수 {len(lines)} != 4")
    pos, L = 0, {}
    for name, line in zip(["objects", "map", "intent", "action"], lines):
        L[name] = [pos, pos + len(line)]; pos += len(line) + 1
    sp = dict(speed=[], na=[])
    # Objects: 8 칸(채운 칸 뒤에 빈 칸만)
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
    # Map
    if not lines[1].startswith("Map: "):
        raise ValueError("Map 머리말")
    b1 = L["map"][0] + 5
    c = _Cur(lines[1][5:], base=b1)
    if c.is_fill("<na>"):
        sp["na"].append(c.fill("<na>", MAP_LINE_FIELDS, "지도 미상")); mp = None
    else:
        if c.is_fill("<pc_none>"):
            c.fill("<pc_none>", 6, "횡단보도"); pc = None
        elif c.is_fill("<pc_on>"):
            c.fill("<pc_on>", 6, "횡단보도"); pc = dict(dist=None, sector="here")
        else:
            c.lit("<pc>", "횡단보도")
            sec = TOK_SEC[c.tok(SEC_TOK.values(), "횡단보도 방위")]
            pc = dict(dist=float(c.rx(_UN_RE, "횡단보도 거리")[0]), sector=sec)
        c.lit("<div>", "div"); nd = int(c.rx(_I2_RE, "div 수")[0])
        c.lit("<bnd>", "bnd"); nb = int(c.rx(_I2_RE, "bnd 수")[0])
        lanes = {}
        for y in LANE_Y:
            c.lit(AT_TOK[y], "단면")
            d = {}
            for k in ("div_l", "div_r", "bnd_l", "bnd_r"):
                if c.is_fill("<none>"):
                    c.fill("<none>", 4, "단면 값"); d[k] = None
                else:
                    d[k] = float(c.rx(_UN_RE, "단면 값")[0])
            lanes[str(y)] = d
        c.lit("<free>", "자유거리")
        fd = float(c.rx(_UN_RE, "자유거리")[0])
        mp = dict(ped_crossing=pc, n_divider=nd, n_boundary=nb, lanes=lanes, free_dist=fd)
    if not c.done():
        raise ValueError("Map 줄 끝에 남은 문자")
    # Intent
    if lines[2] == "Intent: <na><_>":
        intent = None
    else:
        c = _Cur(lines[2])
        c.lit("Intent: ", "Intent 머리말")
        intent = dict(longitudinal=TOK_LON[c.tok(LON_TOK.values(), "종방향")], lateral=TOK_LAT[c.tok(LAT_TOK.values(), "횡방향")])
        if not c.done():
            raise ValueError("Intent 줄 끝에 남은 문자")
    # Action
    if lines[3] == "Action: " + _fill("<na>", 6):
        act = None
    else:
        if not lines[3].startswith("Action: "):
            raise ValueError("Action 머리말")
        body = lines[3][8:]
        ms = list(_ACT_RE.finditer(body))
        if len(ms) != 6 or "".join(m.group(0) for m in ms) != body:
            raise ValueError("Action 줄 문법 불일치")
        act = [int(m.group(1)) for m in ms]
        if not all(0 <= k < ACTION_K for k in act):
            raise ValueError("action 토큰 범위 밖")
    return dict(objects=objs, map=mp, intent=intent, action=act), dict(lines=L, **sp)


def parse_answer(text, object_speed="numeric"):
    """엄격 파서(전체 일치·정규 고정폭 숫자만). 실패 시 ValueError. format_parsed(parse_answer(a)) == a 보장."""
    return _parse_full(text, object_speed)[0]


def format_parsed(p, object_speed="numeric"):
    """parse_answer 결과 → 정답 문자열(왕복 검사용 역함수)."""
    mp = p["map"]
    if mp is not None and mp["ped_crossing"] is not None and mp["ped_crossing"]["sector"] == "here":
        mp = dict(mp, ped_crossing=dict(dist=0.0, sector="here"))
    return build_answer(p["objects"], mp, p["intent"], p["action"], object_speed)


def answer_spans(answer, object_speed="numeric"):
    """손실 마스킹용 문자 구간 [start, end)(answer 기준). lines: 줄별 구간(머리말 포함, 줄바꿈 제외).
    speed = object_speed(동일, v2 키 유지): 객체 속력 숫자 구간. na: '<na>' 대체형 구간(Intent/Action 줄 제외 —
    그 둘은 future_complete=False 일 때 줄 전체로 가린다)."""
    sp = _parse_full(answer, object_speed)[1]
    return dict(lines=sp["lines"], object_speed=sp["speed"], speed=sp["speed"], na=sp["na"])


def merge_spans(spans):
    out = []
    for a, b in sorted(map(list, spans)):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def loss_mask_spans(spans, future_complete_, prev_available_):
    """규칙: 미래 불완전 → Intent·Action 줄 전체, 이전 프레임 없음 → 객체 속력 숫자, 항상 → '<na>' 대체형(GT 미상)."""
    lm = list(spans["na"])
    if not future_complete_:
        lm += [spans["lines"]["intent"], spans["lines"]["action"]]
    if not prev_available_:
        lm += spans["speed"]
    return merge_spans(lm)


# 수정(2026-10-15): build_instruction 의 빈 칸 채움 수(모듈 앞쪽 리터럴)가 실제 칸 토큰 수와 같은지 강제
assert f"followed by {SLOT_FIELDS_NUMERIC - 1} <_>" in build_instruction(object_speed="numeric")
assert f"followed by {SLOT_FIELDS_DIRONLY - 1} <_>" in build_instruction(object_speed="dir_only")
assert f"<na> and {MAP_LINE_FIELDS - 1} <_>" in INSTRUCTION
assert f"within {TTC_GAP}m of the ego" in INSTRUCTION and f"within {FREE_CAP:.0f}m" in INSTRUCTION
