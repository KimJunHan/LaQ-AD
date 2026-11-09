"""수정(2026-10-09): LaQ-AD NAVSIM — HiP-AD 가 낸 계획 궤적(token -> (8,3)) 을 devkit 제출 pickle 로 감싼다.
(구 make_submission.py — 이름 규칙에 따라 navsim 태그. 용도: HuggingFace 리더보드 제출용.
 navtest EPDMS 자체 채점은 이 파일이 아니라 tools/navsim/run_navsim_eval.sh(1단계 러너 + navsim_laq_agent) 를 쓴다 —
 run_pdm_score_from_submission.py 는 2단계 분할의 reactive_all_mapping 으로 집계하기 때문이다(:327).)

실행 env: NAVSIM devkit 이 설치된 env(별도). 입력 pkl 은 hipad_bw env 에서 numpy 로만 저장한다(클래스 의존성 없음).
형식 근거(navsim_ref @0a380a9):
  - navsim/planning/script/run_create_submission_pickle.py:111-123 — {"team_name",..., "first_stage_predictions": [dict],
    "second_stage_predictions": [dict]}
  - navsim/planning/script/run_pdm_score_from_submission.py:302-310 — 각 리스트 길이 1(단일 시드)만 허용
  - navsim/common/dataclasses.py:Trajectory — poses (num_poses, 3)=(x, y, heading), 기본 TrajectorySampling(4 s, 0.5 s)
navtest 는 1단계 평가만 있으므로 second_stage 는 빈 dict.

no-skip: devkit 점수기는 제출에 없는 token 을 평가하지 않고, 실패 token 은 valid=False 로 남긴 채 진행한다.
그러면 일부 장면이 빠진 평균이 나온다. 따라서 여기서 navtest 전체 token 과 1:1 일치를 강제한다.
"""
import argparse
import pickle

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="{token: np.ndarray(8,3)} pkl (HiP-AD 추론 산출)")
    ap.add_argument("--expected-tokens", required=True, help="build_navsim_infos.py 산출 pkl(장면 token 목록 출처)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--team-name", default="LaQ-AD")
    args = ap.parse_args()

    pred = pickle.load(open(args.pred, "rb"))
    expected = set(pickle.load(open(args.expected_tokens, "rb"))["metadata"]["eval_tokens"] or [])  # 평가 장면 token(이력 프레임 제외)
    if set(pred) != expected:
        raise RuntimeError(f"token 불일치: 누락 {len(expected - set(pred))}, 초과 {len(set(pred) - expected)}")
    sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    first = {}
    for tok, p in pred.items():
        p = np.asarray(p, dtype=np.float32)
        if p.shape != (sampling.num_poses, 3) or not np.isfinite(p).all():
            raise ValueError(f"{tok}: 궤적 형식 오류 shape={p.shape}")
        first[tok] = Trajectory(p, sampling)
    sub = {"team_name": args.team_name, "authors": [], "email": "", "institution": "",
           "country / region": "", "first_stage_predictions": [first], "second_stage_predictions": [{}]}
    pickle.dump(sub, open(args.out, "wb"))
    print(f"wrote {len(first)} trajectories -> {args.out}")


if __name__ == "__main__":
    main()
