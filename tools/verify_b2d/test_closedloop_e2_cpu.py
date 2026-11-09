# 수정(2026-11-06): 사용자 결정 "B2D 폐쇄루프 E2 + 나침반 NaN 틱 처리(권장안)" — 새 파일(원본 없음)
"""B2D 폐쇄루프 E2(자차 상태 4 값) 전달·나침반 NaN 틱 유지 규칙 CPU 단위 시험. CARLA·GPU 불필요.

  source /home1/irteam/data-vol1/hipad_env.sh
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tools/verify_b2d/test_closedloop_e2_cpu.py [--model]

(1) EgoStateHold: 정상 / 경로 시작 NaN(0 채움) / NaN 연속(직전 정상값 유지·나이) / IMU 만 비유한 / 속도 NaN → 예외 / 기록
(2) 학습 쪽과 같은 식: B2D 교사 GT jsonl(val) 키프레임의 원값을 폐쇄루프 센서 형식(converter 역변환)으로 넣으면
    프롬프트 자차 줄이 GT 의 prompt_parts.ego 와 문자 단위로 같고, 학생 값이 b2d_ego_input_val.pkl 과 비트 단위로 같다
(3) 에이전트(기반 SparseAgent 를 가짜로 바꿔 끼움): 교사 갱신 주기(10 틱)·사이 틱 재사용, 교사 프롬프트 숫자 = 학생 입력(비트),
    NaN 틱에서 교사 tick_data 도 유지값, (1,4) float32, E2 꺼진 모델 kwargs 불변, 계약 위반(ego_line 없음/다름) → 예외,
    destroy 기록 JSON
(4) --model: B2D 2단계 E2 config(test_e3_nuim) CPU 빌드 → 속성 경로·ego_input → metas["ego_input"] → ego_input_encoder 입력 확인
"""
import argparse
import glob
import json
import math
import os
import pickle
import sys
import tempfile
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEAM = os.path.join(REPO, "bench2drive", "leaderboard", "team_code")
sys.path.insert(0, TEAM)
import ego_state_hold as EH  # noqa: E402

GT_CANDIDATES = ["data/vla_teacher/b2d/gt_v2_sensor_b2d/teacher_gt_b2d_val.jsonl",
                 "/home1/irteam/data-vol1/dl/b2d_v2/teacher/gt_v2_sensor_b2d/teacher_gt_b2d_val.jsonl"]
CONV_CANDIDATES = ["/home1/irteam/data-vol1/dl/b2d_v2/teacher/gt_v2_sensor_b2d/b2d_conventions_train.json",
                   "results/b2d_prep/b2d_conventions_train.json"]
EGO_PKL = "data/infos/b2d_20261106/b2d_ego_input_val.pkl"  # 저장소 자체 산출물(build_b2d_ego_input.py) — 외부 입력 아님
FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not cond:
        FAILS.append(name)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc as e:
        return str(e)
    return None


NAN = float("nan")


# ---------------------------------------------------------------- (1)
def test_hold():
    print("(1) EgoStateHold 유지 규칙", flush=True)
    h = EH.EgoStateHold("r")
    a_ok = [1.5, 0.2, 9.8]; g_ok = [0.01, 0.02, -0.3]
    seq = []
    # 경로 시작 NaN 2 틱 → 0 채움
    for s in (0, 1):
        seq.append(h.update(s, 3.0, NAN, [0.0] * 3, [0.0] * 3))
    check("시작 NaN → zero_fill, 가속·yaw 0, 속도 그대로",
          all(r["state"] == "zero_fill" and r["vals"] == (3.0, 0.0, -0.0) and r["hold_age"] is None for r in seq),
          str([r["vals"] for r in seq]))
    r = h.update(2, -0.05, 1.0, a_ok, g_ok)
    check("정상 틱: speed=max(v,0), accel=a[0], yaw=-g[2]", r["state"] == "ok" and r["vals"] == (0.0, 1.5, 0.3), str(r["vals"]))
    held = [h.update(s, 4.0, NAN, [0.0] * 3, [0.0] * 3) for s in (3, 4, 5, 6)]
    check("NaN 연속 → 직전 정상 IMU 유지(가속 3·각속도 3)", all(x["state"] == "held" and x["accel3"] == a_ok and x["gyro3"] == g_ok
                                                     and x["vals"] == (4.0, 1.5, 0.3) for x in held))
    check("유지 나이(틱) 1..4", [x["hold_age"] for x in held] == [1, 2, 3, 4])
    r = h.update(7, 5.0, 0.5, [2.0, 0, 9.8], [0, 0, 0.1])
    check("정상 복귀 → 새 값", r["state"] == "ok" and r["vals"] == (5.0, 2.0, -0.1))
    r = h.update(8, 5.0, 0.5, [NAN, 0, 9.8], [0, 0, 0.1])
    check("compass 정상 + 가속 NaN → 유지(별도 계수)", r["state"] == "held" and r["vals"] == (5.0, 2.0, -0.1) and h.n_invalid_imu_only == 1)
    check("속도 NaN → 예외", raises(lambda: h.update(9, NAN, 0.5, a_ok, g_ok), ValueError) is not None)
    s = h.summary()
    check("기록: NaN 틱 7, 유지 5, 시작 0 채움 2, 최대 나이 4, 구간",
          (s["n_nan_ticks"], s["n_held"], s["n_zero_fill_start"], s["hold_age_ticks_max"], s["n_nan_compass"]) == (7, 5, 2, 4, 6)
          and s["nan_runs"] == [[0, 2, "zero_fill"], [3, 4, "held"], [8, 1, "held"]], json.dumps(s))
    # 범위 밖은 자르지 않고 실패(학습 형식 규칙)
    check("가속 |a| ≥ 99.995 → ego_line 예외(자르지 않음)", raises(lambda: EH.ego_line_and_input((1.0, 120.0, 0.0)), ValueError) is not None)
    line, x = EH.ego_line_and_input((3.04, -0.1332, 0.00083))
    check("줄 → 값 (1 / 2 / 3 자리)", line.startswith("Ego state: speed 03.0 m/s, longitudinal acceleration -00.13")
          and x.dtype == np.float32 and x.shape == (4,) and x.tolist() == np.asarray([3.0, -0.13, 0.001, 1.0], np.float32).tolist(), line)


# ---------------------------------------------------------------- (1b) 수정(2026-11-09): 형식 범위 밖 → 자르고 기록
def test_clip():  # 수정(2026-11-09): 사용자 결정 "범위 안 최댓값으로 자르고(부호 유지) 기록"
    print("(1b) 형식 범위 밖 값 자르기(가속 ±150, yaw ±20, 속도 120)", flush=True)  # 수정(2026-11-09)
    h = EH.EgoStateHold("c")  # 수정(2026-11-09)
    r0 = h.update(0, 5.0, 0.1, [1.0, 0, 9.8], [0, 0, -0.2])  # 수정(2026-11-09): 범위 안 → 값 그대로
    check("범위 안 값은 그대로(clipped 없음)", r0["clipped"] == [] and r0["vals"] == (5.0, 1.0, 0.2))  # 수정(2026-11-09)
    cases = [(1, 5.0, [150.0, 0, 9.8], [0, 0, 0.0], (5.0, 99.99, -0.0), [("accel", 150.0, 99.99)]),  # 수정(2026-11-09)
             (2, 5.0, [-150.0, 0, 9.8], [0, 0, 0.0], (5.0, -99.99, -0.0), [("accel", -150.0, -99.99)]),  # 수정(2026-11-09)
             (3, 5.0, [0.0, 0, 9.8], [0, 0, -20.0], (5.0, 0.0, 9.999), [("yaw", 20.0, 9.999)]),  # 수정(2026-11-09): yaw = -gyro
             (4, 5.0, [0.0, 0, 9.8], [0, 0, 20.0], (5.0, 0.0, -9.999), [("yaw", -20.0, -9.999)]),  # 수정(2026-11-09)
             (5, 120.0, [150.0, 0, 9.8], [0, 0, 20.0], (99.9, 99.99, -9.999),  # 수정(2026-11-09): 세 신호 동시
              [("speed", 120.0, 99.9), ("accel", 150.0, 99.99), ("yaw", -20.0, -9.999)])]  # 수정(2026-11-09)
    ok = True  # 수정(2026-11-09)
    lines = []  # 수정(2026-11-09)
    for st, sp, acc, gy, exp, ev in cases:  # 수정(2026-11-09)
        r = h.update(st, sp, 0.1, acc, gy)  # 수정(2026-11-09)
        line, x = EH.ego_line_and_input(r["vals"])  # 수정(2026-11-09): 예외 없이 줄 생성
        lines.append(line)  # 수정(2026-11-09)
        ok &= r["vals"] == exp and r["clipped"] == ev  # 수정(2026-11-09)
        ok &= x.tolist() == np.asarray(list(exp) + [1.0], np.float32).tolist()  # 수정(2026-11-09): 학생 값 = 자른 값
        ok &= EH.parse_ego_line(line).tobytes() == x.tobytes()  # 수정(2026-11-09): 줄 ↔ 학생 비트 동일
    check("가속 ±150 → ±99.99, yaw ±20 → ±9.999, 속도 120 → 99.9 (부호 유지, 줄·학생 같은 값)", ok, " | ".join(lines[-1:]))  # 수정(2026-11-09)
    check("자른 줄 형식: +99.99 / -99.99 / +9.999 / -9.999 / 99.9",  # 수정(2026-11-09)
          "acceleration +99.99" in lines[0] and "acceleration -99.99" in lines[1] and "yaw rate +9.999" in lines[2]
          and "yaw rate -9.999" in lines[3] and "speed 99.9 m/s" in lines[4])  # 수정(2026-11-09)
    rh = h.update(6, 5.0, NAN, [0, 0, 0], [0, 0, 0])  # 수정(2026-11-09): NaN 틱 → 직전(step 5) 범위 밖 IMU 유지 → 다시 자름
    check("NaN 유지값이 범위 밖이면 유지 후 자름", rh["state"] == "held" and rh["vals"] == (5.0, 99.99, -9.999)  # 수정(2026-11-09)
          and [e[0] for e in rh["clipped"]] == ["accel", "yaw"])  # 수정(2026-11-09)
    h.note_refresh(rh)  # 수정(2026-11-09): 갱신 틱 표시
    s_ = h.summary()  # 수정(2026-11-09)
    check("기록: 자른 틱 6, 신호별 {speed 1, accel 4, yaw 4}, 갱신 틱 자름 1, 사건 step·원값·refresh",  # 수정(2026-11-09)
          s_["n_clip_ticks"] == 6 and s_["n_clip_by_signal"] == {"speed": 1, "accel": 4, "yaw": 4} and s_["n_refresh_clipped"] == 1
          and s_["clip_events"][0] == dict(step=1, signal="accel", raw=150.0, clipped=99.99, refresh=False)
          and [e["refresh"] for e in s_["clip_events"] if e["step"] == 6] == [True, True], json.dumps(s_["clip_events"][:2]))  # 수정(2026-11-09)
    check("경계: 99.994 는 자르지 않음(+99.99 로 표기), 99.995 는 자름",  # 수정(2026-11-09)
          EH.clip_to_format((1.0, 99.994, 0.0))[1] == [] and EH.clip_to_format((1.0, 99.995, 0.0))[1] != []
          and EH.clip_to_format((99.94, 0.0, 9.9994))[1] == [])  # 수정(2026-11-09)
    check("속도 NaN 은 여전히 예외", raises(lambda: h.update(7, NAN, 0.1, [0, 0, 0], [0, 0, 0]), ValueError) is not None)  # 수정(2026-11-09)


# ---------------------------------------------------------------- (2)
def test_training_equivalence():
    print("(2) 학습 쪽 식·형식과 같은지(GT jsonl val 전 키프레임, 학생 pkl 비트 비교)", flush=True)
    conv = next((p for p in CONV_CANDIDATES if os.path.exists(p)), None)
    if conv is None:
        check("conventions 파일 존재", False, str(CONV_CANDIDATES)); return
    sgn = 1.0 if json.load(open(conv))["yaw_rate_regress_slope"] > 0 else -1.0
    check("YAW_SIGN = conventions 부호", sgn == EH.YAW_SIGN, f"{sgn} vs {EH.YAW_SIGN}")
    gt = next((p for p in GT_CANDIDATES if os.path.exists(p)), None)
    if gt is None or not os.path.exists(EGO_PKL):
        check("GT jsonl·학생 pkl 존재", False, f"{gt} {EGO_PKL}"); return
    tab = pickle.load(open(EGO_PKL, "rb"))["ego_input"]
    n = n_line = n_bit = 0
    bad = []
    for ln in open(gt):
        if not ln.strip():
            continue
        r = json.loads(ln)
        e = r["ego_status"]
        if not e["available"]:
            continue
        raw = e["raw"]
        # converter 역변환: ego_accel = [acc0, -acc1, acc2] → acc = [ego_accel0, -ego_accel1, ego_accel2];
        #   ego_rotation_rate = -angular_velocity → gyro = -ego_rot.  speed = ego_vel[0]
        acc = [raw["ego_accel"][0], -raw["ego_accel"][1], raw["ego_accel"][2]]
        gyro = [-v for v in raw["ego_rot"]]
        h = EH.EgoStateHold("x")
        res = h.update(0, raw["ego_vel"][0], 0.3, acc, gyro)
        line, x = EH.ego_line_and_input(res["vals"])
        n += 1
        if line == r["prompt_parts"]["ego"]:
            n_line += 1
        elif len(bad) < 3:
            bad.append((r["token"], line, r["prompt_parts"]["ego"]))
        y = np.asarray(tab[r["token"]], np.float32)
        if x.tobytes() == y.tobytes():
            n_bit += 1
        elif len(bad) < 3:
            bad.append((r["token"], x.tolist(), y.tolist()))
    check(f"프롬프트 줄 문자 단위 동일 {n_line}/{n}", n > 0 and n_line == n, str(bad))
    check(f"학생 값 = b2d_ego_input_val.pkl 비트 동일 {n_bit}/{n}", n > 0 and n_bit == n, str(bad))


# ---------------------------------------------------------------- (3) 에이전트
CAMERA = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


class FakeBaseAgent:
    """hipad_b2d_agent.SparseAgent 의 tick NaN 처리(compass NaN → 가속·각속도 0)만 같게 흉내."""

    def setup(self, path_to_conf_file):
        self.device = "cpu"
        self.frame_rate = 20
        self.step = -1
        self.model = _FAKE_MODEL[0]
        self.destroyed = False

    def tick(self, input_data):
        self.step += 1
        imgs = {c: np.zeros((4, 4, 3), np.uint8) for c in CAMERA}
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
        if math.isnan(compass):
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)
        return dict(imgs=imgs, speed=speed, compass=compass, acceleration=acceleration, angular_velocity=angular_velocity,
                    command=4, target_xy=(0, 0))

    def destroy(self):
        self.destroyed = True


class FakeDetector:
    def __init__(self, e2):
        dec = types.SimpleNamespace(with_close_loop=True, close_loop_hz=20, open_loop_hz=2,
                                    ego_input_embed_cfg=dict(in_dims=4, scale=[10.0, 5.0, 0.4, 1.0]) if e2 else None)
        self.head = types.SimpleNamespace(onedecoder_head=dec)
        self.qwen_cache_dir = "x"
        self.calls = []

    def __call__(self, *a, **kw):
        self.calls.append(kw)
        return "out"


_FAKE_MODEL = [None]


class FakeTeacher:
    """계약을 지키는 가짜 교사 빌더: 넘겨받은 ego 로 common_b2d.ego_line 을 직접 만들어 프롬프트에 쓰고 그 줄을 돌려준다."""

    def __init__(self, mode="ok"):
        self.mode = mode
        self.prompts = []

    def __call__(self, t, cams, cams_prev, prev_available, tick_data, ego, ego_line):
        line = EH._common_b2d().ego_line(*ego)
        self.prompts.append(dict(t=t, line=line, given=ego_line, acc=np.asarray(tick_data["acceleration"]).tolist(),
                                 gyro=np.asarray(tick_data["angular_velocity"]).tolist(), state=tick_data["ego_state"]))
        out = dict(qwen_hidden=torch.zeros(1, 3, 8), qwen_mask=torch.ones(1, 3, dtype=torch.uint8),
                   qwen_type=torch.zeros(1, 3, dtype=torch.long))
        if self.mode == "ok":
            out["ego_line"] = line
        elif self.mode == "wrong":
            out["ego_line"] = EH._common_b2d().ego_line(ego[0] + 1.0, ego[1], ego[2])
        return out


def load_agent_module():
    fake = types.ModuleType("hipad_b2d_agent")
    fake.SparseAgent = FakeBaseAgent
    fake.CAMERA = CAMERA
    sys.modules["hipad_b2d_agent"] = fake
    import importlib
    if "hipad_b2d_laq_teacher_agent" in sys.modules:
        return importlib.reload(sys.modules["hipad_b2d_laq_teacher_agent"])
    return importlib.import_module("hipad_b2d_laq_teacher_agent")


def imu_seq(n):
    """step → (speed, compass, acc3, gyro3). 시작 3 틱 NaN, 25..37 NaN 연속(교사 갱신 30 포함), 나머지 정상(값이 틱마다 바뀜)."""
    out = []
    for s in range(n):
        nan = s < 3 or 25 <= s <= 37
        acc = [0.37 * s - 4.0, 0.01 * s, 9.8]
        if s == 40:  # 수정(2026-11-09): 교사 갱신 틱(40)에 충돌 같은 범위 밖 가속도 → 자르고 로그
            acc = [-150.0, 0.0, 9.8]  # 수정(2026-11-09)
        gyro = [0.001, -0.002, 0.0123 * (s % 7) - 0.03]
        out.append((0.31 * s, NAN if nan else 0.1 * s, acc, gyro))
    return out


def run_agent(mod, e2, teacher, n=60, save_dir=None):
    _FAKE_MODEL[0] = FakeDetector(e2)
    os.environ["LAQ_TEACHER_BUILDER"] = "_fake_builder_mod:build"
    bm = types.ModuleType("_fake_builder_mod")
    bm.build = lambda device: teacher
    sys.modules["_fake_builder_mod"] = bm
    if save_dir:
        os.environ["SAVE_PATH"] = save_dir
        os.environ["ROUTES"] = os.path.join(save_dir, "dev10_route_x.xml")
    ag = mod.LaQTeacherSparseAgent()
    ag.setup("cfg")
    det = _FAKE_MODEL[0]
    recs = []
    for s, (sp, comp, acc, gyro) in enumerate(imu_seq(n)):
        inp = {"SPEED": (0, {"speed": sp}), "IMU": (0, np.asarray(acc + gyro + [comp], np.float64))}
        ag.tick(inp)
        ag.model(img="I", img_metas="M", timestamp=ag.step / 20, rescale=True, return_loss=False)
        recs.append(dict(step=ag.step, kw=det.calls[-1]))
    return ag, det, recs


def test_agent():
    print("(3) 에이전트 tick·모델 래퍼(가짜 기반 에이전트·가짜 교사)", flush=True)
    mod = load_agent_module()
    tmp = tempfile.mkdtemp(prefix="e2cl_")
    te = FakeTeacher("ok")
    ag, det, recs = run_agent(mod, True, te, save_dir=tmp)
    ref_steps = [s for s, _t, _a, r in ag.teacher_age_log if r]
    check("교사 갱신 틱 = 0,10,20,...", ref_steps == list(range(0, 60, 10)), str(ref_steps))
    check("교사가 받은 줄 = 교사가 프롬프트에 쓴 줄", all(p["given"] == p["line"] for p in te.prompts))
    ok_bit = ok_shape = ok_reuse = True
    for r in recs:
        x = r["kw"]["ego_input"]
        kf = r["step"] - r["step"] % 10
        line = te.prompts[kf // 10]["line"]
        exp = EH.parse_ego_line(line)
        ok_shape &= tuple(x.shape) == (1, 4) and x.dtype == torch.float32 and x.device.type == "cpu"
        ok_bit &= x.numpy().tobytes() == exp.reshape(1, 4).tobytes()
        if r["step"] % 10:
            ok_reuse &= torch.equal(x, recs[kf]["kw"]["ego_input"])
    check("학생 ego_input = 교사 프롬프트 줄 숫자(비트 동일, 매 틱)", ok_bit)
    check("사이 틱 = 직전 갱신 틱 값 재사용", ok_reuse)
    check("(1,4) float32 (에이전트 device)", ok_shape)
    seq = imu_seq(60)
    p0 = te.prompts[0]
    check("경로 시작 NaN 갱신 틱(0): 가속·yaw 0, 상태 zero_fill, available 1",
          p0["state"] == "zero_fill" and p0["acc"] == [0, 0, 0] and recs[0]["kw"]["ego_input"][0, 3].item() == 1.0
          and recs[0]["kw"]["ego_input"][0, 1].item() == 0.0 and recs[0]["kw"]["ego_input"][0, 2].item() == 0.0, p0["line"])
    p3 = te.prompts[3]   # step 30: NaN 구간(25..37), 직전 정상 = step 24
    exp_line = EH._common_b2d().ego_line(max(seq[30][0], 0.0), seq[24][2][0], -seq[24][3][2])
    check("NaN 구간 갱신 틱(30): 교사 tick_data·줄 = step 24 IMU 유지, 속도는 현재 속도계",
          p3["state"] == "held" and p3["acc"] == seq[24][2] and p3["gyro"] == seq[24][3] and p3["line"] == exp_line, p3["line"])
    check("NaN 갱신 틱 기록(30: 나이 6 틱)", (30, "held", 6) in ag.ego_input_log, str(ag.ego_input_log))
    s = ag.ego_hold.summary()
    check("기록: NaN 16 틱, 시작 0 채움 3, 유지 13, 최대 나이 13, 갱신 틱 상태",
          (s["n_nan_ticks"], s["n_zero_fill_start"], s["n_held"], s["hold_age_ticks_max"]) == (16, 3, 13, 13)
          and s["refresh_states"] == {"ok": 4, "held": 1, "zero_fill": 1} and s["nan_runs"] == [[0, 3, "zero_fill"], [25, 13, "held"]],
          json.dumps(s))
    ag.destroy()
    js = glob.glob(os.path.join(tmp, "ego_state_stats_dev10_route_x_*.json"))
    p4 = te.prompts[4]  # 수정(2026-11-09): step 40 갱신 틱 — 범위 밖 가속 −150 → −99.99, 교사 줄·학생 같은 값
    check("에이전트 갱신 틱 범위 밖: 교사 줄 -99.99, 학생 입력 = 같은 자른 값, 기록",  # 수정(2026-11-09)
          "acceleration -99.99" in p4["line"] and recs[40]["kw"]["ego_input"][0, 1].item() == np.float32(-99.99)
          and torch.equal(recs[45]["kw"]["ego_input"], recs[40]["kw"]["ego_input"])
          and s["n_refresh_clipped"] == 1 and s["clip_events"] == [dict(step=40, signal="accel", raw=-150.0, clipped=-99.99, refresh=True)],
          p4["line"])  # 수정(2026-11-09)
    check("destroy: 기록 JSON 1 개 + 기반 destroy 호출", len(js) == 1 and ag.destroyed
          and json.load(open(js[0]))["n_nan_ticks"] == 16, str(js))

    # E2 꺼진 모델: kwargs 종전과 동일(ego_input 없음), 교사가 ego_line 을 안 돌려줘도 동작
    te2 = FakeTeacher("none")
    ag2, det2, recs2 = run_agent(mod, False, te2, n=25, save_dir=tmp)
    keys = {tuple(sorted(r["kw"].keys())) for r in recs2}
    check("E2 꺼짐: kwargs = 기반 인자 + qwen 3 키(ego_input 없음)",
          keys == {tuple(sorted(["img", "img_metas", "timestamp", "rescale", "return_loss", "qwen_hidden", "qwen_mask", "qwen_type"]))},
          str(keys))
    # 계약 위반
    msg = raises(lambda: run_agent(mod, True, FakeTeacher("none"), n=3, save_dir=tmp), RuntimeError)
    check("E2 켜짐 + 교사 출력에 ego_line 없음 → 예외", msg is not None and "ego_line" in msg, str(msg))
    msg = raises(lambda: run_agent(mod, True, FakeTeacher("wrong"), n=3, save_dir=tmp), RuntimeError)
    check("교사 ego_line 이 넘긴 줄과 다름 → 예외", msg is not None and "!=" in msg, str(msg))
    msg = raises(lambda: run_agent(mod, False, FakeTeacher("wrong"), n=3, save_dir=tmp), RuntimeError)
    check("E2 꺼짐이어도 교사 줄 불일치 → 예외(교사 프롬프트 규칙 일관)", msg is not None, str(msg))


# ---------------------------------------------------------------- (4) 모델 CPU 빌드
def test_model():
    print("(4) B2D 2단계 E2 config CPU 빌드·ego_input 경로", flush=True)
    sys.path.insert(0, REPO)
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": kw.get("weights_only", False)})
    import projects.mmdet3d_plugin  # noqa: F401
    import tools.plan_diffusion_map_register  # noqa: F401
    import tools.b2d_teacher_register  # noqa: F401
    from mmcv import Config
    from mmdet.models import build_detector
    c = Config.fromfile("configs/b2d/v2_teacher/stage2/test_e3_nuim.py")
    m = build_detector(c.model); m.init_weights(); m.eval()
    dec = m.head.onedecoder_head
    check("head.onedecoder_head.ego_input_embed_cfg (in_dims 4, scale [10,5,0.4,1])",
          dec.ego_input_embed_cfg is not None and dec.ego_input_embed_cfg["in_dims"] == 4
          and torch.equal(dec.ego_input_scale, torch.tensor([10.0, 5.0, 0.4, 1.0], dtype=torch.float32)), str(dec.ego_input_embed_cfg))
    # simple_test → head(img, feature_maps, data) 의 data 가 decoder metas — 에이전트 kwargs 가 그대로 들어가는지
    seen = {}

    def fake_head(img, feature_maps, data):
        seen["data"] = data
        raise StopIteration
    m.extract_feat = lambda img, *a, **k: [torch.zeros(1)]
    m.head.forward = fake_head
    x = torch.from_numpy(np.asarray([3.0, -0.13, 0.001, 1.0], np.float32)).view(1, 4)
    try:
        m(img=torch.zeros(1, 6, 3, 8, 8), img_metas=[{}], timestamp=0.0, rescale=True, return_loss=False,
          qwen_hidden=torch.zeros(1, 3, 8), qwen_mask=torch.ones(1, 3, dtype=torch.uint8), qwen_type=torch.zeros(1, 3, dtype=torch.long),
          ego_input=x)
    except StopIteration:
        pass
    check("model(**kwargs) → head data(metas)['ego_input'] 같은 텐서", seen.get("data", {}).get("ego_input") is x)
    # 디코더가 쓰는 식: encoder(x / scale), 0 초기화 → 출력 0, 모양 (1, 256)
    e = dec.ego_input_encoder(x.float() / dec.ego_input_scale)
    check("ego_input_encoder 출력 (1, embed) · 0 초기화 → 0", e.shape == (1, dec.embed_dims) and float(e.abs().max()) == 0.0, str(tuple(e.shape)))
    # 디코더 forward 안의 모양 검사 문장이 (1,4) 를 받아들이는지(같은 조건식)
    check("디코더 모양 조건 dim 2, (B=1, 4)", x.dim() == 2 and x.shape[0] == 1 and x.shape[1] == dec.ego_input_scale.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true")
    a = ap.parse_args()
    os.chdir(REPO)
    test_hold()
    test_clip()  # 수정(2026-11-09): 형식 범위 밖 자르기 시험
    test_training_equivalence()
    test_agent()
    if a.model:
        test_model()
    print(f"\n결과: 실패 {len(FAILS)} {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
