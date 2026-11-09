"""수정(2026-10-09): LaQ-AD Bench2Drive 폐쇄루프 결과 집계(엄격판) — Driving Score / Success Rate.

왜 새 스크립트인가:
  - bench2drive/tools/statistic_route_json.py 는 'Completed/Perfect/TickRuntime/blocked' 이외 상태의 route 를
    **평균에서 빼고** 분모도 줄인다(실패 route 제외 = skip). 프로젝트 규칙(root-cause only, skip 금지)과 충돌하고
    문헌 수치와도 비교되지 않는다.
  - 공식 bench2drive/tools/merge_route_json.py 는 분모를 220 으로 고정하지만 220 이 아니면 경고만 하고 숫자를 낸다.
  이 스크립트는 공식 정의(DS = Σ score_composed / 220, SR = 성공 route 수 / 220, 성공 = Completed|Perfect 이고
  min_speed 외 위반 0)를 그대로 쓰되, 다음 경우 **숫자를 내지 않고 실패**한다:
    (1) route 수가 기대값(기본 220)과 다르거나 route_id 가 중복될 때,
    (2) 기반시설 실패(시뮬레이터/에이전트 예외: 'Failed - Simulation crashed', 'Failed - Agent crashed' 등)가 있을 때
        — 주행 실패가 아니라 평가 환경 문제이므로 원인을 고치고 그 route 를 다시 돌려야 한다.
  주행 중 실패(충돌·막힘·경로이탈·시간초과 등)는 정상적인 평가 결과이므로 그대로 집계한다.

사용:
  python tools/eval_b2d_laq_closedloop.py evaluation/hipad_b2d_laq_stage2            # 220 route 전체
  python tools/eval_b2d_laq_closedloop.py evaluation/hipad_b2d_laq_stage2 --expect 14 --subset  # 1개 split 점검용
  --subset 결과는 문헌 비교 금지(출력에 SUBSET 표기). 공식 merged.json 도 같은 폴더에 쓴다(--write-merged).
능력별(Multi-Ability) 지표는 공식 bench2drive/tools/ability_benchmark.py(CARLA 서버·carla 모듈 필요),
효율·승차감은 efficiency_smoothness_benchmark.py 를 merged.json 에 대해 실행한다.
"""
import argparse
import glob
import json
import os
import sys

# 기반시설 실패(평가 환경 문제) — leaderboard_evaluator.FAILURE_MESSAGES 와 statistics_manager 의 상태 문자열
INFRA_FAILURES = ("Simulation crashed", "Agent crashed", "Agent couldn't be set up", "Sensors", "Agent_init")


def is_success(rd):
    if rd["status"] not in ("Completed", "Perfect"):
        return False
    return all(len(v) == 0 for k, v in rd["infractions"].items() if k != "min_speed_infractions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("route_dir", help="run_evaluation_b2d_laq.sh 의 SAVE_PATH (route json 들이 있는 폴더)")
    ap.add_argument("--expect", type=int, default=220, help="기대 route 수(공식 220)")
    ap.add_argument("--subset", action="store_true", help="부분 평가 점검용 — 문헌 비교 금지 표기")
    ap.add_argument("--write-merged", action="store_true", help="공식 형식 merged.json 저장")
    a = ap.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(a.route_dir, "*.json")) if not p.endswith("merged.json"))
    if not paths:
        sys.exit(f"[실패] route json 없음: {a.route_dir}")
    records = []
    for p in paths:
        with open(p) as f:
            records.extend(json.load(f)["_checkpoint"]["records"])

    ids = [rd["route_id"] for rd in records]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    infra = [(rd["route_id"], rd["status"]) for rd in records
             if rd["status"].startswith("Failed") and any(m in rd["status"] for m in INFRA_FAILURES)]
    errs = []
    if dup:
        errs.append(f"route_id 중복 {len(dup)}개: {dup[:5]}")
    if len(set(ids)) != a.expect:
        errs.append(f"route 수 {len(set(ids))} != 기대 {a.expect}")
    if infra:
        errs.append(f"기반시설 실패 {len(infra)}건(원인 수정 후 해당 route 재실행 필요): {infra[:5]}")
    if errs:
        sys.exit("[실패] 점수를 내지 않는다 —\n  " + "\n  ".join(errs))

    n = a.expect
    ds = sum(rd["scores"]["score_composed"] for rd in records) / n
    succ = sum(is_success(rd) for rd in records)
    sr = succ / n * 100
    by_status = {}
    for rd in records:
        by_status[rd["status"]] = by_status.get(rd["status"], 0) + 1
    tag = "SUBSET(문헌 비교 금지)" if a.subset else "FULL"
    print(f"[{tag}] routes={n}  Driving Score={ds:.2f}  Success Rate={sr:.2f}% ({succ}/{n})")
    print("  상태별:", dict(sorted(by_status.items(), key=lambda x: -x[1])))
    if a.write_merged:
        merged = {"_checkpoint": {"records": sorted(records, key=lambda d: d["route_id"], reverse=True)},
                  "driving score": ds, "success rate": succ / n, "eval num": n}
        out = os.path.join(a.route_dir, "merged.json")
        with open(out, "w") as f:
            json.dump(merged, f, indent=4)
        print("  merged.json ->", out)


if __name__ == "__main__":
    main()
