# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   폐쇄루프 교사 2 Hz 갱신 스케줄러(bench2drive/leaderboard/team_code/teacher_refresh.py) CPU 단위 시험. CARLA·GPU 불필요.
#   CUDA_VISIBLE_DEVICES="" python tools/verify_b2d/test_teacher_refresh_cpu.py
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "bench2drive", "leaderboard", "team_code"))
from teacher_refresh import TeacherRefreshScheduler, keyframe_of, teacher_age_s  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


class FakeTeacher:
    def __init__(self):
        self.calls = []

    def __call__(self, t, **kw):
        self.calls.append(t)
        return dict(made_at=t, x=kw.get("x"))


def run(hz, n, scene="r0", sched=None, teacher=None):
    teacher = teacher or FakeTeacher()
    sched = sched or TeacherRefreshScheduler(teacher)
    outs = []
    for step in range(n):
        t = step / hz
        o, info = sched.step(t, scene_id=scene, x=step)
        outs.append((step, t, o, info))
    return teacher, sched, outs


# 1) 20 Hz(leaderboard): 10 틱마다 갱신, 사이는 같은 객체 재사용, 나이 0..0.45
te, sc, outs = run(20, 100)
ref_steps = [s for s, t, o, i in outs if i["refreshed"]]
check("20Hz 갱신 틱 = 0,10,20,...", ref_steps == list(range(0, 100, 10)), str(ref_steps[:6]))
check("20Hz 교사 호출 수 10", len(te.calls) == 10)
reuse_ok = all(o is outs[s - s % 10][2] for s, t, o, i in outs)
check("사이 틱은 직전 출력 객체 그대로(복사·재계산 없음)", reuse_ok)
ages = sorted({round(i["age_s"], 2) for s, t, o, i in outs})
check("20Hz 나이 0.00..0.45", ages == [round(0.05 * k, 2) for k in range(10)], str(ages))

# 2) 10 Hz(학습 데이터 주기): 5 프레임마다, 나이 0..0.4 = keyframe 규칙과 같음
te, sc, outs = run(10, 50)
check("10Hz 갱신 = keyframe(f%5==0)", [s for s, t, o, i in outs if i["refreshed"]] == list(range(0, 50, 5)))
check("10Hz 나이 = teacher_age_s(f)", all(abs(i["age_s"] - teacher_age_s(s)) < 1e-9 for s, t, o, i in outs))
check("10Hz 사용 출력의 시각 = keyframe_of(f)/10", all(abs(o["made_at"] - keyframe_of(s) / 10) < 1e-9 for s, t, o, i in outs))

# 2b) clip 첫 프레임 f0=1(수정 2026-10-19 규칙): 스케줄러 첫 틱 = f0 → keyframe 1, 6, 11 … 과 같은 출력
te = FakeTeacher(); sc = TeacherRefreshScheduler(te)
ok = True
for f in range(1, 40):
    o, i = sc.step((f - 1) / 10, scene_id="f0_1")
    ok &= abs(i["age_s"] - teacher_age_s(f, f0=1)) < 1e-9 and abs(o["made_at"] - (keyframe_of(f, f0=1) - 1) / 10) < 1e-9
check("f0=1 clip: 갱신 프레임 = f0+5k, 나이 = (f−kf)/10", ok and [keyframe_of(f, f0=1) for f in (1, 5, 6, 10, 11)] == [1, 1, 6, 6, 11])
try:
    keyframe_of(0, f0=1); check("f < f0 실패", False)
except ValueError:
    check("f < f0 실패", True)

# 3) 장면(route) 전환: 새 장면 첫 틱에 즉시 갱신, 이전 장면 출력 재사용 없음, 시각 0 부터 다시
te = FakeTeacher(); sc = TeacherRefreshScheduler(te)
run(20, 23, scene="rA", sched=sc, teacher=te)
o, info = sc.step(0.0, scene_id="rB", x=-1)
check("장면 전환 즉시 갱신", info["refreshed"] and o["made_at"] == 0.0 and o["x"] == -1 and info["age_s"] == 0.0)
o2, info2 = sc.step(0.05, scene_id="rB", x=-2)
check("장면 전환 뒤 재사용", (not info2["refreshed"]) and o2 is o)

# 4) 시각 역행(같은 장면) → 즉시 실패
try:
    sc.step(0.0, scene_id="rB")
    check("시각 역행 실패", False)
except RuntimeError:
    check("시각 역행 실패", True)


# 5) 교사 예외·None → 그대로 실패(직전 출력으로 대신하지 않음)
class Boom:
    n = 0

    def __call__(self, t, **kw):
        Boom.n += 1
        if t >= 0.5:
            raise ValueError("교사 실패")
        return dict(made_at=t)


sc = TeacherRefreshScheduler(Boom())
for s in range(10):
    sc.step(s / 20, scene_id="r")
try:
    sc.step(0.5, scene_id="r")
    check("교사 예외 전파", False)
except ValueError:
    check("교사 예외 전파", True)
sc = TeacherRefreshScheduler(lambda t, **k: None)
try:
    sc.step(0.0, scene_id="r")
    check("교사 None 실패", False)
except RuntimeError:
    check("교사 None 실패", True)

# 6) 지연 모드: 에이전트 기본값에서 거부, 오프라인(allow_delayed)에서는 한 주기 늦은 출력
try:
    TeacherRefreshScheduler(FakeTeacher(), mode="delayed")
    check("delayed 기본 거부", False)
except ValueError:
    check("delayed 기본 거부", True)
te = FakeTeacher()
sc = TeacherRefreshScheduler(te, mode="delayed", allow_delayed=True)
_, _, outs = run(10, 20, sched=sc, teacher=te)
made = [o["made_at"] for s, t, o, i in outs]
exp = [0.0] * 10 + [0.5] * 5 + [1.0] * 5    # 첫 주기는 현재 것(이전 없음), 이후 0.5 s 늦은 출력
check("delayed = 한 주기 늦은 출력", all(abs(a - b) < 1e-9 for a, b in zip(made, exp)), str(made[::5]))
check("delayed 나이 0.5..0.9(첫 주기 제외)", sorted({round(i['age_s'], 1) for s, t, o, i in outs[10:]}) == [0.5, 0.6, 0.7, 0.8, 0.9])

# 7) 부동소수: 0.05 누적 시각에서도 10 틱 간격 유지(1 시간 = 72,000 틱)
te = FakeTeacher(); sc = TeacherRefreshScheduler(te)
t = 0.0
for s in range(72000):
    sc.step(t, scene_id="long"); t += 0.05
check("누적 시각 0.05 합에서도 7,200 회 갱신", len(te.calls) == 7200, str(len(te.calls)))

print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
