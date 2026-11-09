# 수정(2026-11-05): navhard_two_stage 평가 준비(평가 전용) — 예측 변환·검사·CSV 판독. 새 파일, 기존 소스 무수정.
"""navsim_pred.pkl({token: (8,3) ego 국소 (x, y, heading)}, NavsimDataset.evaluate 산출) ↔ navhard 2단계 채점.

채점 경로(공식 러너 그대로, devkit 무수정):
  run_pdm_score.py(navhard_two_stage, synthetic_sensor_path/synthetic_scenes_path) + agent=navsim_laq_agent(pred_path=…)
  — 에이전트는 scene.scene_metadata.initial_token 으로 궤적을 꺼낸다. 원본 장면 initial_token = 현재 프레임 token,
  합성 장면 initial_token = pickle metadata.initial_token(= frames[3].token) 이고, 둘 다 우리 info 의 navsim_token 과 같다
  → navtest 와 같은 pred 형식이 그대로 쓰인다(별도 좌표 변환 없음). 래퍼: scripts/navsim/run_navhard_eval_cpu8.sh.
  리더보드 형식(run_pdm_score_from_submission.py; 단일 프로세스라 느림)은 `submission` 하위명령으로 만든다:
  first_stage_predictions = [1단계 450 token 궤적], second_stage_predictions = [2단계 5,462 token 궤적]
  (run_create_submission_pickle.py:111-123 형식, Trajectory(poses, TrajectorySampling(4 s, 0.5 s))).

하위명령(env navsim):
  check      --pred P --infos I              : token 집합 == info eval_tokens(1단계 450 + 2단계 5,462), (8,3) 유한값.
  cv         --infos I --out P               : 더미 등속(CV) 예측 — devkit ConstantVelocityAgent 와 같은 식(현재 프레임
                                               |(vx, vy)| float32, x 축 직진). 우리 변환·채점 경로 점검용(공식 CV 결과와 비교).
  submission --pred P --infos I --out S      : 리더보드 제출 pickle.
  csv        --csv C --infos I               : 채점 CSV 판독 — 5,912 token 전부 valid + 집계 행 3 개(pseudo closed-loop valid)
                                               이어야 점수 출력. 보고 값 = extended_pdm_score_combined 행(단순 평균 아님!).
"""
import argparse
import pickle
import sys

import numpy as np


def load_meta(p):
    m = pickle.load(open(p, "rb"))["metadata"]
    if "navhard" not in m:
        raise SystemExit(f"{p}: navhard info 가 아니다(build_navhard_infos.py 산출물만)")
    return m


def check_pred(pred, meta):
    exp = set(meta["eval_tokens"])
    got = set(pred)
    if got != exp:
        raise SystemExit(f"[중단] token 불일치: 누락 {len(exp - got)} (예 {sorted(exp - got)[:3]}), 초과 {len(got - exp)}")
    for t, p in pred.items():
        p = np.asarray(p)
        if p.shape != (8, 3) or not np.isfinite(p).all():
            raise SystemExit(f"[중단] {t}: 궤적 형식 {p.shape} / 유한값 아님")
    n1, n2 = len(meta["navhard"]["stage1_tokens"]), len(meta["navhard"]["stage2_tokens"])
    print(f"OK: pred {len(pred)} token = 1단계 {n1} + 2단계 {n2}, 모두 (8,3) 유한값")


def cmd_check(a):
    check_pred(pickle.load(open(a.pred, "rb")), load_meta(a.infos))


def cmd_cv(a):
    d = pickle.load(open(a.infos, "rb"))
    preds = {}
    for i in d["infos"]:
        if not i["is_eval_frame"]:
            continue
        v = np.asarray(i["ego_status"], np.float32)[:2]
        speed = (v ** 2).sum(-1) ** 0.5            # devkit constant_velocity_agent.py 와 같은 float32 식
        preds[i["navsim_token"]] = np.array([[(k + 1) * 0.5 * speed, 0.0, 0.0] for k in range(8)], dtype=np.float32)
    check_pred(preds, d["metadata"])
    with open(a.out, "wb") as f:
        pickle.dump(preds, f)
    print(f"CV 더미 예측 {len(preds)} -> {a.out}")


def cmd_submission(a):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    meta = load_meta(a.infos)
    pred = pickle.load(open(a.pred, "rb"))
    check_pred(pred, meta)
    ts = TrajectorySampling(time_horizon=4, interval_length=0.5)
    first = {t: Trajectory(np.asarray(pred[t], np.float32), ts) for t in meta["navhard"]["stage1_tokens"]}
    second = {t: Trajectory(np.asarray(pred[t], np.float32), ts) for t in meta["navhard"]["stage2_tokens"]}
    sub = {"team_name": a.team_name, "authors": [], "email": "", "institution": "", "country / region": "",
           "first_stage_predictions": [first], "second_stage_predictions": [second]}
    with open(a.out, "wb") as f:
        pickle.dump(sub, f)
    print(f"submission: 1단계 {len(first)}, 2단계 {len(second)} -> {a.out}")


def cmd_csv(a):
    import pandas as pd
    meta = load_meta(a.infos)
    exp = set(meta["eval_tokens"])
    df = pd.read_csv(a.csv, index_col=0)
    rows = df[df["token"].isin(exp)]
    bad = rows[~rows["valid"].astype(str).str.lower().eq("true")]
    missing = exp - set(rows["token"])
    agg = df.set_index("token").reindex(["extended_pdm_score_stage_one", "extended_pdm_score_stage_two",
                                         "extended_pdm_score_combined"])
    agg_ok = agg["valid"].astype(str).str.lower().eq("true").all() and agg["score"].notna().all()
    if len(bad) or missing or not agg_ok or len(rows) != len(exp):
        print(f"FAIL: invalid {len(bad)}, missing {len(missing)}, 행 {len(rows)}/{len(exp)}, 집계 행 valid {agg_ok} — 점수를 보고하지 않는다",
              file=sys.stderr)
        print("invalid 예:", bad["token"].head(10).to_list(), file=sys.stderr)
        sys.exit(1)
    s1 = set(meta["navhard"]["stage1_tokens"])
    print(f"OK: navhard {len(rows)} token(1단계 {len(s1)}, 2단계 {len(rows) - len(s1)}) 모두 valid, pseudo closed-loop 집계 valid")
    print(f"EPDMS navhard two-stage (combined) = {agg.loc['extended_pdm_score_combined', 'score']:.4f}  "
          f"(stage one {agg.loc['extended_pdm_score_stage_one', 'score']:.4f}, "
          f"stage two {agg.loc['extended_pdm_score_stage_two', 'score']:.4f})")
    cols = [c for c in agg.columns if c not in ("valid", "score")]
    print(agg[cols].T.to_string())


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("check"); p.add_argument("--pred", required=True); p.add_argument("--infos", required=True)
    p = sp.add_parser("cv"); p.add_argument("--infos", required=True); p.add_argument("--out", required=True)
    p = sp.add_parser("submission"); p.add_argument("--pred", required=True); p.add_argument("--infos", required=True)
    p.add_argument("--out", required=True); p.add_argument("--team-name", default="LaQ-AD")
    p = sp.add_parser("csv"); p.add_argument("--csv", required=True); p.add_argument("--infos", required=True)
    a = ap.parse_args()
    dict(check=cmd_check, cv=cmd_cv, submission=cmd_submission, csv=cmd_csv)[a.cmd](a)


if __name__ == "__main__":
    main()
