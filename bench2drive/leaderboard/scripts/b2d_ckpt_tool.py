"""수정(2026-11-09): 새 파일 — B2D 폐쇄루프 조각 checkpoint(json) 도구. run_eval_b2d_laq_teacher_e3_nuim.sh 가 쓴다.

평가기(leaderboard_evaluator / route_indexer / statistics_manager) 소스는 건드리지 않고, 재시작 전에 checkpoint 를 고친다.

  state <ck>                 "done" | "todo <설명>"  (done = progress 끝, 기록 수 = 경로 수, entry Finished, 기반시설 실패·미완 기록 없음)
  ok <ck>                    최종 상태(주행 결과)인 기록 수 — 재시작이 진전했는지 판단용
  fails <prepped> <after>    이번 시도에서 새로 쓰인 기록 가운데 기반시설 실패인 "route_id 종류"(줄마다 하나; 종류 sim = Simulation crashed,
                             agent = Agent crashed·Agent couldn't be set up·sensors invalid — 스크립트가 연속 실패 한도를 종류별로 둔다).
                             <prepped> = 시도 직전(prep 뒤) 사본; 같은 index 기록이 그대로면 이번 시도에서 안 돈 것으로 본다.
  prep <ck> <backup>         재시작 직전: 원본 바이트를 <backup> 으로 먼저 복사(파싱 전 — 잘린 json 도 보존) 후,
                             route_indexer.validate_and_resume 가 재개 지점으로 보지 않는 기반시설 실패
                             ("Failed - Agent couldn't be set up", "Failed - Agent's sensors were invalid")가 있으면
                             progress[0] 을 그 경로 index 로 되돌리고 그 뒤 기록을 잘라낸다(건너뛰기 금지).
                             바꾼 내용은 표준출력으로(스크립트가 로그에 남김).
  diff <before> <after>      재실행으로 덮어쓰인(이미 최종 상태였던) 경로 기록을 출력 — 원본은 <before> 에 남아 있다.
  checkpoint json 을 읽지 못하면(잘림·깨짐) 모든 모드가 메시지를 내고 rc 2 로 끝난다(스크립트는 그 조각을 중단).

왜 필요한가(평가기 동작, 소스 확인):
  - Agent_init 실패 → 기록 'Failed - Agent couldn't be set up', entry 'Started', crashed=True → progress=[i+1, N] 로 멈춤.
  - validate_and_resume 은 상태가 ['Failed', 'Failed - Simulation crashed', 'Failed - Agent crashed'] 인 첫 기록만 재개 지점으로
    잡으므로 위 기록은 지나치고 progress[0]=i+1 부터 재개 → 경로 i 가 건너뛰어진다.
  - entry 'Crashed' 면 재개 index = 재개지점 − 1(직전 경로 재실행) → 이미 끝난 직전 경로 기록이 덮어쓰인다(→ prep 의 백업 + diff).
"""
import json
import os
import shutil
import sys

# 기반시설 실패(평가 환경 문제) 표식 — statistics_manager.FAILURE_MESSAGES 의 실제 문자열
INFRA = ("Simulation crashed", "Agent crashed", "Agent couldn't be set up", "Agent's sensors were invalid")
# route_indexer.validate_and_resume 가 재개 지점으로 잡는 상태(소스 83 행 그대로)
RESUMABLE = ("Failed", "Failed - Simulation crashed", "Failed - Agent crashed")


class Broken(Exception):
    pass


def load(p):
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    try:
        with open(p) as f:
            d = json.load(f)
        recs = d["_checkpoint"].get("records", [])
        [(r["route_id"], r["status"]) for r in recs]
        return d
    except Exception as e:  # 잘린 json(쓰는 도중 죽음)·형식 깨짐
        raise Broken(f"checkpoint 읽기 실패 {p}: {type(e).__name__}: {e}")


def is_infra(status):
    return status.startswith("Failed") and any(m in status for m in INFRA)


def is_final(status):
    return status not in ("Started", "") and not is_infra(status)


def state(p):
    d = load(p)
    if d is None:
        return "todo 없음"
    c = d.get("_checkpoint", {})
    pr, recs = c.get("progress") or [0, 1], c.get("records", [])
    infra = [r["route_id"] for r in recs if is_infra(r["status"])]
    started = [r["route_id"] for r in recs if r["status"] == "Started"]
    if pr[0] == pr[1] and len(recs) == pr[1] and d.get("entry_status") == "Finished" and not infra and not started:
        return "done"
    return f"todo progress {pr} records {len(recs)} entry {d.get('entry_status')} 기반시설실패 {infra[:3]} 미완 {started[:3]}"


def ok(p):
    d = load(p)
    if d is None:
        return 0
    return sum(is_final(r["status"]) for r in d.get("_checkpoint", {}).get("records", []))


def prep(p, backup):
    if os.path.exists(p):
        shutil.copy2(p, backup)   # 파싱 전에 원본 바이트 보존(잘린 json 이어도)
    d = load(p)
    if d is None:
        print(f"[ckpt] {p} 없음(또는 빈 파일) — 처음부터")
        return
    c = d["_checkpoint"]
    pr, recs = c.get("progress") or [0, 0], c.get("records", [])
    # 평가기가 잡을 재개 지점(validate_and_resume 과 같은 순서로 훑음)
    ev = next((i for i in range(min(pr[0], len(recs))) if recs[i]["status"] in RESUMABLE), None)
    bad = next((i for i, r in enumerate(recs) if is_infra(r["status"]) and r["status"] not in RESUMABLE), None)
    print(f"[ckpt] 백업 {backup} (progress {pr}, 기록 {len(recs)}, entry {d.get('entry_status')}, 평가기 재개지점 {ev})")
    if bad is not None and (ev is None or bad < ev):
        old = dict(progress=list(pr), n_records=len(recs), entry=d.get("entry_status"))
        dropped = [(r["route_id"], r["status"]) for r in recs[bad:]]
        c["progress"] = [bad, pr[1]]
        c["records"] = recs[:bad]
        d["entry_status"] = "Started"   # 'Crashed' 면 평가기가 index−1 로 한 칸 더 되돌리므로 정확히 bad 에서 시작하게
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=4)
        os.replace(tmp, p)
        print(f"[ckpt][되돌림] {recs[bad]['route_id']} 상태 '{recs[bad]['status']}' 는 평가기 재개 대상이 아니어서 건너뛰어짐 → "
              f"progress {old['progress']} → {c['progress']}, 기록 {old['n_records']} → {bad}, entry {old['entry']} → Started. "
              f"잘라낸 기록(백업에 보존) {dropped}")
    elif d.get("entry_status") == "Invalid":
        print("[ckpt][경고] entry 'Invalid' — 평가기는 resume 을 거부하고 기록을 비운 뒤 처음부터 돈다(이전 기록은 백업에 보존)")


def fails(prepped, after):
    a = load(after)
    if a is None:
        return []
    b = load(prepped) if os.path.exists(prepped) else None
    rb = b["_checkpoint"].get("records", []) if b else []
    out = []
    for i, r in enumerate(a["_checkpoint"].get("records", [])):
        if is_infra(r["status"]) and (i >= len(rb) or rb[i] != r):
            out.append((r["route_id"], "sim" if "Simulation crashed" in r["status"] else "agent"))
    return out


def diff(before, after):
    b, a = load(before), load(after)
    if b is None or a is None:
        return
    rb, ra = b["_checkpoint"].get("records", []), a["_checkpoint"].get("records", [])
    for i, r in enumerate(rb):
        if not is_final(r["status"]):
            continue
        if i >= len(ra):
            print(f"[ckpt][덮어씀] index {i} {r['route_id']} 기록이 사라짐 (이전 {r['status']}, 백업 {before})")
            continue
        n = ra[i]
        sc = lambda x: x.get("scores", {}).get("score_composed")
        if n["route_id"] != r["route_id"] or n["status"] != r["status"] or sc(n) != sc(r):
            print(f"[ckpt][덮어씀] index {i} {r['route_id']}: {r['status']} DS {sc(r)} → {n['route_id']} {n['status']} DS {sc(n)} "
                  f"(이전 기록은 {before})")


def main():
    m = sys.argv[1]
    if m == "fails":
        for rid, kind in fails(sys.argv[2], sys.argv[3]):
            print(rid, kind)
    elif m == "state":
        print(state(sys.argv[2]))
    elif m == "ok":
        print(ok(sys.argv[2]))
    elif m == "prep":
        prep(sys.argv[2], sys.argv[3])
    elif m == "diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        sys.exit(f"모드 {m}?")


if __name__ == "__main__":
    try:
        main()
    except Broken as e:
        print(f"[ckpt][오류] {e}", file=sys.stderr)
        sys.exit(2)
