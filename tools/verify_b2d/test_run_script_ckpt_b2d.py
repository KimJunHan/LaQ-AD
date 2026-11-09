"""수정(2026-11-09): 새 파일 — B2D 폐쇄루프 실행 스크립트의 재개·집계 재현 시험(가짜 checkpoint, CPU, CARLA 안 띄움).

  (2) 'Agent couldn't be set up' 경로가 재개 때 건너뛰어지는 문제: 실제 route_indexer.validate_and_resume 로
      고치기 전(건너뜀) / b2d_ckpt_tool.py prep 뒤(그 경로부터) 재개 index 를 확인. Crashed 직전 경로 덮어쓰기 기록(diff)도.
  (A) 잘린 json: prep 이 파싱 전 백업·rc≠0, run_split 이 평가기를 안 띄우고 중단.
  (B) 기동 전 포트 점검: pg 가 없어도 포트 해제 대기(놓으면 통과, 계속 쥐면 rc 1).
  (C) 같은 경로 연속 2 회 기반시설 실패 → 즉시 중단(다른 경로면 계속).
      (A)(C) 는 실행 스크립트의 함수 구간을 그대로 떼어 가짜 CARLA 서버·가짜 평가기로 run_split 을 돌린다.
  (1) 집계: 실행 스크립트를 DRYRUN=1 AGG_ONLY=1 로 돌려 (a) 정상 → DONE, (b) 집계 실패(route 중복) → DONE 없음·rc≠0,
      (c) 다른 NUM_SPLITS 폴더·같은 폴더의 엉뚱한 json 은 집계에 안 들어감.

실행: source /home1/irteam/data-vol1/hipad_env.sh; python tools/verify_b2d/test_run_script_ckpt_b2d.py [--skip_script]
"""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LB = os.path.join(R, "bench2drive", "leaderboard")
TOOL = os.path.join(LB, "scripts", "b2d_ckpt_tool.py")
SCRIPT = os.path.join(LB, "scripts", "run_eval_b2d_laq_teacher_e3_nuim.sh")
DEV10 = "/home1/irteam/data-vol1/src/Bench2Drive/leaderboard/data/drivetransformer_bench2drive_dev10.xml"
sys.path[:0] = ["/home1/irteam/data-vol1/carla/pylib_cp310", "/home1/irteam/data-vol1/carla/CARLA_0.9.15/PythonAPI/carla", LB, os.path.join(R, "bench2drive", "scenario_runner")]
os.chdir(R)
FAIL = []


def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {extra}" if extra else ""), flush=True)
    if not ok:
        FAIL.append(name)


def route_ids(xml):
    import xml.etree.ElementTree as ET
    return [f"RouteScenario_{r.get('id')}_rep0" for r in ET.parse(xml).getroot().iter("route")]


def rec(rid, status, score=50.0):
    inf = {k: [] for k in ("collisions_layout", "collisions_pedestrian", "collisions_vehicle", "red_light", "stop_infraction",
                           "outside_route_lanes", "min_speed_infractions", "yield_emergency_vehicle_infractions",
                           "scenario_timeouts", "route_dev", "vehicle_blocked", "route_timeout")}
    return dict(route_id=rid, status=status, index=-1, infractions=inf,
                scores=dict(score_route=100.0 if status in ("Perfect", "Completed") else score, score_penalty=1.0,
                            score_composed=score),
                meta=dict(route_length=100.0, duration_game=10.0, duration_system=20.0))


def ckpt(path, recs, progress, entry):
    d = dict(entry_status=entry, eligible=False, sensors=["carla_camera"], values=[], labels=[],
             _checkpoint=dict(global_record={}, progress=progress, records=recs))
    with open(path, "w") as f:
        json.dump(d, f, indent=4)


def tool(*a):
    r = subprocess.run([sys.executable, TOOL, *a], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def resume_index(xml, ck):
    from leaderboard.utils.route_indexer import RouteIndexer
    ri = RouteIndexer(xml, 1, "")
    ok = ri.validate_and_resume(ck)
    return ok, ri.index


def test_resume(tmp):
    print("(2) 'Agent couldn't be set up' 재개 — 실제 route_indexer.validate_and_resume", flush=True)
    ids = route_ids(DEV10)
    ck = os.path.join(tmp, "s.json")
    # 평가기 실제 동작: Agent_init 실패 → 기록 'Failed - Agent couldn't be set up', entry 'Started', crashed → progress [i+1, N]
    recs = [rec(ids[i], "Perfect", 100.0) for i in range(3)] + [rec(ids[3], "Failed - Agent couldn't be set up", 0.0)]
    ckpt(ck, recs, [4, 10], "Started")
    orig = open(ck).read()
    ok, idx = resume_index(DEV10, ck)
    check("고치기 전: 평가기 재개 index 4 → 경로 3 건너뜀(문제 재현)", ok and idx == 4, f"resume {ok}, index {idx}")
    rc, out = tool("state", ck)
    check("state = todo(기반시설 실패)", out.startswith("todo") and ids[3] in out, out.strip())
    bk = os.path.join(tmp, "s.before.json")
    rc, out = tool("prep", ck, bk)
    print("    " + out.strip().replace("\n", "\n    "))
    check("prep: 백업 = 원본", open(bk).read() == orig)
    d = json.load(open(ck))
    check("prep: progress [3,10]·기록 3·entry Started", d["_checkpoint"]["progress"] == [3, 10]
          and len(d["_checkpoint"]["records"]) == 3 and d["entry_status"] == "Started", str(d["_checkpoint"]["progress"]))
    check("prep: 바꾼 내용 로그('[ckpt][되돌림]' + 경로 id)", "[ckpt][되돌림]" in out and ids[3] in out)
    ok, idx = resume_index(DEV10, ck)
    check("고친 뒤: 평가기 재개 index 3 → 경로 3 부터 다시", ok and idx == 3, f"resume {ok}, index {idx}")
    rc, out = tool("prep", ck, bk + "2")
    check("두 번째 prep: 바꿀 것 없음(멱등)", "[ckpt][되돌림]" not in out and json.load(open(ck)) == d)

    # 뒤에 끝난 경로가 더 있는 경우(재시작 사이에 Agent_init 실패 뒤 다른 실패로 이어진 기록) + Crashed
    recs = ([rec(ids[i], "Perfect", 100.0) for i in range(2)] + [rec(ids[2], "Failed - Agent couldn't be set up", 0.0)]
            + [rec(ids[3], "Completed", 70.0), rec(ids[4], "Failed - Simulation crashed", 0.0)])
    ckpt(ck, recs, [5, 10], "Crashed")
    ok, idx = resume_index(DEV10, ck)
    check("고치기 전(Crashed): 재개 index 3 → 경로 2 건너뜀", ok and idx == 3, f"index {idx}")
    rc, out = tool("prep", ck, bk + "3")
    ok, idx = resume_index(DEV10, ck)
    d = json.load(open(ck))
    check("고친 뒤: 재개 index 2, 뒤 기록 잘라냄(백업 보존)", ok and idx == 2 and len(d["_checkpoint"]["records"]) == 2
          and len(json.load(open(bk + "3"))["_checkpoint"]["records"]) == 5, f"index {idx}")

    # 평가기 자체가 잡는 경우는 손대지 않음
    recs = [rec(ids[i], "Perfect", 100.0) for i in range(4)] + [rec(ids[4], "Failed - Simulation crashed", 0.0)]
    ckpt(ck, recs, [5, 10], "Crashed")
    before = open(ck).read()
    rc, out = tool("prep", ck, bk + "4")
    ok, idx = resume_index(DEV10, ck)
    check("Simulation crashed 는 그대로(평가기가 index 3 = 직전 경로부터 재실행)", open(ck).read() == before and idx == 3,
          f"index {idx}")
    # (7) 재실행으로 직전 경로(3) 기록이 덮어쓰임 → diff 로그
    d = json.load(open(ck))
    d["_checkpoint"]["records"][3] = rec(ids[3], "Completed", 61.5)
    d["_checkpoint"]["records"][4] = rec(ids[4], "Perfect", 100.0)
    d["entry_status"] = "Finished"
    json.dump(d, open(ck, "w"))
    rc, out = tool("diff", bk + "4", ck)
    print("    " + out.strip().replace("\n", "\n    "))
    check("(7) 덮어쓰인 경로 3 기록을 로그로(이전 DS 100 → 61.5, 백업 경로)", "[ckpt][덮어씀] index 3" in out and "100.0" in out
          and bk + "4" in out and "index 4" not in out)

    # state: 센서 무효 기록은 done 이 아님
    recs = [rec(ids[i], "Perfect", 100.0) for i in range(9)] + [rec(ids[9], "Failed - Agent's sensors were invalid", 0.0)]
    ckpt(ck, recs, [10, 10], "Finished")
    rc, out = tool("state", ck)
    check("state: 'sensors were invalid' 는 todo", out.startswith("todo"), out.strip())
    recs[9] = rec(ids[9], "Failed - Agent timed out", 10.0)
    ckpt(ck, recs, [10, 10], "Finished")
    rc, out = tool("state", ck)
    check("state: 주행 실패(Agent timed out)만 있으면 done", out.strip() == "done", out.strip())


# ------------------------------------------------------------------ (A)(B)(C) — 실행 스크립트의 함수 구간(# >>> funcs … # <<< funcs)을
# 그대로 떼어 가짜 CARLA 서버(setsid, 포트 listen)·가짜 평가기(checkpoint 를 평가기 형식으로 씀)로 run_split 을 돌린다.
FAKE_SERVER = r"""
import socket, sys, time
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen()
time.sleep(float(sys.argv[2]) if len(sys.argv) > 2 else 600)
"""
FAKE_EVAL = r"""
import json, os, random, sys
import xml.etree.ElementTree as ET
a = dict(x.split("=", 1) for x in sys.argv[1:] if x.startswith("--") and "=" in x)
ck, xml = a["--checkpoint"], a["--routes"]
cnt = os.environ["FAKE_COUNT"]
c = int(open(cnt).read()) if os.path.exists(cnt) else 0
open(cnt, "w").write(str(c + 1))
ids = [f"RouteScenario_{r.get('id')}_rep0" for r in ET.parse(xml).getroot().iter("route")]
plan = os.environ["FAKE_PLAN"].split(";")[min(c, len(os.environ["FAKE_PLAN"].split(";")) - 1)]  # 시도별 "실패할 index" (빈칸 = 없음)
bad = {int(x.lstrip("s")): ("Failed - Simulation crashed" if x.startswith("s") else "Failed - Agent crashed")
       for x in plan.split(",") if x}   # "0" = Agent crashed, "s0" = Simulation crashed
def rec(i):
    st = bad.get(i, "Perfect")
    return dict(route_id=ids[i], status=st, index=-1, infractions={}, scores=dict(score_route=100.0, score_penalty=1.0,
                score_composed=0.0 if i in bad else 100.0), meta=dict(route_length=1.0, duration_game=1.0,
                duration_system=random.random()))
d = dict(entry_status="Finished", eligible=True, sensors=[], values=[], labels=[],
         _checkpoint=dict(global_record={}, progress=[len(ids), len(ids)], records=[rec(i) for i in range(len(ids))]))
json.dump(d, open(ck, "w"))
print(f"fake eval call {c} bad {sorted(bad)}")
"""


def harness(tmp, name, body, env_extra=None, port=41000):
    import xml.etree.ElementTree as ET
    T = os.path.join(tmp, name)
    os.makedirs(os.path.join(T, "out", "logs"), exist_ok=True)
    os.makedirs(os.path.join(T, "out", "locks"), exist_ok=True)
    open(os.path.join(T, "fake_server.py"), "w").write(FAKE_SERVER)
    open(os.path.join(T, "fake_eval.py"), "w").write(FAKE_EVAL)
    open(os.path.join(T, "fake_start.sh"), "w").write(
        f'setsid nohup python3 {T}/fake_server.py $2 > $3 2>&1 < /dev/null &\necho $!\n')
    root = ET.parse(DEV10).getroot()
    new = ET.Element(root.tag, root.attrib)
    for r in list(root.iter("route"))[:3]:
        new.append(r)
    open(os.path.join(T, "routes.xml"), "w").write(ET.tostring(new, encoding="unicode"))
    funcs = subprocess.run(["sed", "-n", "/^# >>> funcs/,/^# <<< funcs/p", SCRIPT], capture_output=True, text=True).stdout
    assert "run_split()" in funcs and "stop_server()" in funcs
    sh = f"""set -u
log() {{ echo "[$(date +%T)] $*"; }}
PY={sys.executable}; CKPT_TOOL={TOOL}; OUT={T}/out; LDIR={T}/out/logs; NT=n1; SET_TAG=t; NUM_SPLITS=1; DRYRUN=0
LOCK_DIR={T}/plocks; BASE_PORT={port}; BASE_TM_PORT={port + 20000}; PORT_STRIDE=200; PARALLEL=1; MAX_RESTARTS=3; SERVER_WAIT_S=20; GPU=9
START_CARLA={T}/fake_start.sh; EVALUATOR={T}/fake_eval.py; TEAM_AGENT=x; TEAM_CONFIG=x; MAP_LOAD_WAIT_S=0
PORT_FREE_WAIT_S=${{PORT_FREE_WAIT_S:-120}}
split_xml() {{ echo {T}/routes.xml; }}; split_ck() {{ echo {T}/out/ck.json; }}
{funcs}
{body}
"""
    open(os.path.join(T, "h.sh"), "w").write(sh)
    env = dict(os.environ, FAKE_COUNT=os.path.join(T, "count"), **(env_extra or {}))
    r = subprocess.run(["bash", os.path.join(T, "h.sh")], capture_output=True, text=True, env=env, timeout=600)
    calls = int(open(os.path.join(T, "count")).read()) if os.path.exists(os.path.join(T, "count")) else 0
    return T, r.stdout + r.stderr, calls


def test_harness(tmp):
    print("(A) 잘린 checkpoint json", flush=True)
    ids = route_ids(DEV10)
    ck = os.path.join(tmp, "trunc.json")
    ckpt(ck, [rec(ids[i], "Perfect", 100.0) for i in range(3)], [3, 10], "Started")
    raw = open(ck, "rb").read()[:-40]
    open(ck, "wb").write(raw)
    rc, out = tool("prep", ck, ck + ".bk")
    check("prep: 파싱 전에 원본 바이트 백업", os.path.exists(ck + ".bk") and open(ck + ".bk", "rb").read() == raw)
    check("prep: rc≠0 + 오류 메시지", rc != 0 and "[ckpt][오류]" in out, f"rc {rc}")
    for m in ("state", "ok"):
        r = subprocess.run([sys.executable, TOOL, m, ck], capture_output=True, text=True)
        check(f"{m}: rc≠0·표준출력 비어 있음", r.returncode != 0 and r.stdout == "", f"rc {r.returncode} out '{r.stdout.strip()}'")
    T, log, calls = harness(tmp, "A", 'cp "$TRUNC" "$OUT/ck.json"; run_split 0 0; echo "RC=$?"', dict(TRUNC=ck))
    print("    " + "\n    ".join(l for l in log.splitlines() if "중단" in l or "RC=" in l))
    check("run_split: 평가기 안 띄우고 중단(RC=1, 평가기 호출 0)", "RC=1" in log and calls == 0 and "prep 실패" in log, f"calls {calls}")
    import glob
    bks = glob.glob(os.path.join(T, "out", "logs", "t_s0of1.ckpt_before_*_try0.json"))
    check("run_split: 원본 바이트가 logs 백업에 그대로", len(bks) == 1 and open(bks[0], "rb").read() == raw, str(bks))

    print("(B) 기동 전 포트 점검(pg 없음)", flush=True)
    body = """python3 $T_DIR/fake_server.py 41600 {sec} & sp=$!; sleep 1
t0=$(date +%s); stop_server "" 41600 61600; echo "SS_RC=$? SEC=$(( $(date +%s) - t0 ))"; kill $sp 2>/dev/null; wait 2>/dev/null"""
    os.makedirs(os.path.join(tmp, "B0"), exist_ok=True)
    T, log, _ = harness(tmp, "B1", body.format(sec=6).replace("$T_DIR", os.path.join(tmp, "B1")), dict(PORT_FREE_WAIT_S="30"))
    import re
    m = re.search(r"SS_RC=(\d+) SEC=(\d+)", log)
    check("다른 프로세스가 6 s 뒤 포트를 놓으면: 대기 후 rc 0(약 5–8 s)", m and m.group(1) == "0" and 4 <= int(m.group(2)) <= 10,
          m.group(0) if m else log[-200:])
    T, log, _ = harness(tmp, "B2", body.format(sec=60).replace("$T_DIR", os.path.join(tmp, "B2")), dict(PORT_FREE_WAIT_S="6"))
    m = re.search(r"SS_RC=(\d+) SEC=(\d+)", log)
    check("포트를 계속 쥐면: PORT_FREE_WAIT_S(6 s) 뒤 rc 1·로그", m and m.group(1) == "1" and "[중단] 포트 41600" in log,
          m.group(0) if m else log[-200:])
    r = subprocess.run(["bash", "-c", "python3 %s/B2/fake_server.py 41600 20 & sleep 1; "
                        "pg=; [ -n \"$pg\" ] || { echo OLD_RETURNED_IMMEDIATELY; }; ss -ltnH | grep -q ':41600 ' && echo BUSY; kill %%1" % tmp],
                       capture_output=True, text=True)
    check("참고: 옛 구문([ -n pg ] || return 0)은 포트가 쓰이는데도 바로 통과", "OLD_RETURNED_IMMEDIATELY" in r.stdout and "BUSY" in r.stdout)

    print("(C) 같은 경로 연속 기반시설 실패 → 즉시 중단", flush=True)
    T, log, calls = harness(tmp, "C1", 'run_split 0 0; echo "RC=$?"', dict(FAKE_PLAN="0;0;0;0;0"), port=41200)
    print("    " + "\n    ".join(l for l in log.splitlines() if "중단" in l or "기반시설" in l or "RC=" in l))
    check("경로 0 이 2 시도 연속 Agent crashed → 2 번째 시도 뒤 중단(RC=1, 평가기 2 회)",
          "RC=1" in log and calls == 2 and f"한도 도달: {ids[0]}(agent 2회)" in log, f"calls {calls}")
    check("중단 이유가 resume.log 에도", "연속 기반시설 실패로 중단" in open(os.path.join(T, "out", "logs", "t_s0of1.resume.log")).read())
    T, log, calls = harness(tmp, "C2", 'run_split 0 0; echo "RC=$?"', dict(FAKE_PLAN="0;1;"), port=41400)
    check("대조: 시도마다 다른 경로가 실패(0 → 1 → 없음)면 계속 → 완료(RC=0, 평가기 3 회)",
          "RC=0" in log and calls == 3 and "연속 2 회" not in log, f"calls {calls}")
    T, log, calls = harness(tmp, "C3", 'run_split 0 0; echo "RC=$?"', dict(FAKE_PLAN=""), port=41400)
    check("대조: 실패 없음 → 1 회로 완료(RC=0)", "RC=0" in log and calls == 1, f"calls {calls}")
    T, log, calls = harness(tmp, "C4", 'run_split 0 0; echo "RC=$?"', dict(FAKE_PLAN="s0;s0;s0;s0;s0"), port=41200)
    print("    " + "\n    ".join(l for l in log.splitlines() if "한도 도달" in l))
    check("Simulation crashed 는 한도 3: 경로 0 이 3 시도 연속 → 3 번째 뒤 중단(RC=1, 평가기 3 회)",
          "RC=1" in log and calls == 3 and f"{ids[0]}(sim 3회)" in log, f"calls {calls}")
    T, log, calls = harness(tmp, "C5", 'run_split 0 0; echo "RC=$?"', dict(FAKE_PLAN="s0;s0;"), port=41400)
    check("대조: Simulation crashed 2 회 연속 뒤 성공 → 중단 없이 완료(RC=0, 3 회)", "RC=0" in log and calls == 3
          and "한도 도달" not in log, f"calls {calls}")

    print("(D) 백업 이름에 실행 시각 — 다음 실행이 이전 백업을 덮어쓰지 않음", flush=True)
    seed = os.path.join(tmp, "seed.json")
    ckpt(seed, [rec(ids[0], "Completed", 61.5)], [1, 3], "Started")
    T, log, calls = harness(tmp, "D", 'cp "$SEED" "$OUT/ck.json"; run_split 0 0; echo "RC1=$?"; sleep 1.2; '
                            'cp "$SEED" "$OUT/ck.json"; run_split 0 0; echo "RC2=$?"', dict(SEED=seed, FAKE_PLAN=""), port=41400)
    bks = sorted(glob.glob(os.path.join(T, "out", "logs", "t_s0of1.ckpt_before_*_try0.json")))
    check("두 실행의 try0 백업이 서로 다른 이름으로 둘 다 남음", "RC1=0" in log and "RC2=0" in log and len(bks) == 2
          and all(open(b).read() == open(seed).read() for b in bks), str([os.path.basename(b) for b in bks]))
    ow = open(os.path.join(T, "out", "logs", "t_s0of1.overwritten.log")).read()
    refs = [b for b in bks if b in ow]
    check("overwritten.log 가 각 실행의 실제 백업 경로를 가리킴", len(refs) == 2 and ow.count("[ckpt][덮어씀]") == 2, ow[-300:])
    left = subprocess.run(["bash", "-c", "ss -ltnH | grep -E ':(41000|41200|41400|41600) ' || true"], capture_output=True, text=True).stdout
    check("시험 뒤 가짜 서버 포트가 남지 않음", left.strip() == "", left)


def run_agg(dry):
    env = dict(os.environ, DRYRUN="1", AGG_ONLY="1", DRY_DIR=dry, LAQ_TEACHER_FOLD="A", ROUTE_SET="dev10", NUM_SPLITS="2")
    r = subprocess.run(["bash", SCRIPT], capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


def test_agg(tmp):
    print("(1) 집계 — 실행 스크립트 DRYRUN=1 AGG_ONLY=1 dev10 NUM_SPLITS=2", flush=True)
    ids = route_ids(DEV10)
    dry = os.path.join(tmp, "dry")
    out = os.path.join(dry, "hipad_b2d_laq_v2_teacher_stage2_e3_nuim", "dev10")
    c2, c1 = os.path.join(out, "ckpt", "n2"), os.path.join(out, "ckpt", "n1")
    os.makedirs(c2); os.makedirs(c1)
    good = [rec(i, "Perfect", 100.0) for i in ids]
    ckpt(os.path.join(c2, "dev10_s0of2.json"), good[:5], [5, 5], "Finished")
    ckpt(os.path.join(c2, "dev10_s1of2.json"), good[5:], [5, 5], "Finished")
    # 섞이면 안 되는 파일: 다른 조각 수(n1) 결과, 같은 폴더의 엉뚱한 json(중복 route)
    ckpt(os.path.join(c1, "dev10_s0of1.json"), [rec(i, "Failed", 0.0) for i in ids], [10, 10], "Finished")
    ckpt(os.path.join(c2, "dev10_s0of2.json.bak.json"), good[:5], [5, 5], "Finished")
    rc, log = run_agg(dry)
    print("    " + "\n    ".join(l for l in log.splitlines() if "[FULL" in l or "[SUBSET" in l or "집계" in l or "DONE" in l or "중단" in l))
    check("(a) 정상: rc 0·DONE 생성·DS 100(n1·엉뚱한 json 제외)", rc == 0 and os.path.exists(os.path.join(out, "DONE"))
          and "Driving Score=100.00" in log, f"rc {rc}")
    agg = os.path.join(out, "score_n2")
    check("(a) 집계 폴더에는 조각 0..1 파일만", sorted(f for f in os.listdir(agg) if f.endswith(".json")) ==
          ["dev10_s0of2.json", "dev10_s1of2.json", "merged.json"], str(os.listdir(agg)))
    os.remove(os.path.join(out, "DONE"))
    # (b) 집계 실패: s1 에 s0 의 경로가 중복(각 조각 state 는 done) → 집계기 sys.exit → pipefail 로 rc 전달
    bad = good[5:9] + [rec(ids[0], "Perfect", 100.0)]
    ckpt(os.path.join(c2, "dev10_s1of2.json"), bad, [5, 5], "Finished")
    rc, log = run_agg(dry)
    print("    " + "\n    ".join(l for l in log.splitlines() if "실패" in l or "중복" in l))
    check("(b) 집계 실패: rc≠0·DONE 없음", rc != 0 and not os.path.exists(os.path.join(out, "DONE")), f"rc {rc}")
    # 고치기 전 구문의 동작(파이프 끝 tee 의 rc 만 봄) 재현
    r = subprocess.run(["bash", "-c", "if (exit 1) | tee /dev/null; then echo DONE; fi"], capture_output=True, text=True)
    r2 = subprocess.run(["bash", "-c", "set -o pipefail; if (exit 1) | tee /dev/null; then echo DONE; fi"], capture_output=True, text=True)
    check("참고: pipefail 없으면 실패해도 DONE(옛 구문), 있으면 안 찍힘", r.stdout.strip() == "DONE" and r2.stdout.strip() == "")
    # (c) 미완료 조각 → 집계 안 함·rc≠0
    ckpt(os.path.join(c2, "dev10_s1of2.json"), good[5:9], [4, 5], "Started")
    rc, log = run_agg(dry)
    check("(c) 미완료 조각: 집계 생략·rc≠0·DONE 없음", rc != 0 and "집계 생략" in log
          and not os.path.exists(os.path.join(out, "DONE")), f"rc {rc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_script", action="store_true", help="(1) 실행 스크립트 시험 생략(사전 점검 import 로 1–2 분)")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="b2d_ckpt_test_")
    test_resume(tmp)
    test_harness(tmp)
    if not a.skip_script:
        test_agg(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if not FAIL else f"FAIL {len(FAIL)}: {FAIL}")
    sys.exit(1 if FAIL else 0)
