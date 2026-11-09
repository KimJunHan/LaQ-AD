"""수정(2026-10-09): LaQ-AD NAVSIM — EPDMS 점수 CSV 무결성 검사 (구 check_pdm_csv.py, 이름 규칙에 따라 navsim 태그).

근거(navsim_ref @0a380a9, run_pdm_score_one_stage.py):
  - 지표 캐시가 없는 장면은 경고만 하고 평가에서 뺀다(:256-262).
  - 에이전트가 실패한 장면은 valid=False 로 남기고(:116-121), 평균은 skipna 로 계산된다(:285-291).
즉 devkit 이 출력하는 평균은 "평가된 장면만의" 평균일 수 있다. 일부를 뺀 평균은 보고하지 않는다(no-skip):
navtest 의 모든 장면이 CSV 에 있고 valid 여야 점수를 출력한다.
"""
import argparse
import pickle
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--expected-tokens", required=True, help="build_navsim_infos.py --kind test 산출 pkl")
    args = ap.parse_args()
    meta = pickle.load(open(args.expected_tokens, "rb"))["metadata"]
    expected = set(meta["eval_tokens"] or [])
    if not expected:
        raise RuntimeError("기대 token 목록이 비었다 — --kind test info 를 줄 것")
    df = pd.read_csv(args.csv, index_col=0)
    rows = df[df["token"].isin(expected)]
    bad = rows[~rows["valid"].astype(bool)]
    missing = expected - set(rows["token"])
    if len(bad) or missing:
        print(f"FAIL: invalid {len(bad)}, missing {len(missing)} / {len(expected)} — 점수를 보고하지 않는다", file=sys.stderr)
        print("invalid 예:", bad["token"].head(10).to_list(), file=sys.stderr)
        print("missing 예:", sorted(missing)[:10], file=sys.stderr)
        sys.exit(1)
    score_cols = [c for c in rows.columns if c not in ("token", "valid")]
    mean = rows[score_cols].apply(pd.to_numeric, errors="coerce").mean()
    print(f"OK: navtest {len(rows)} scenes, all valid")
    print(f"EPDMS (score) = {mean['score']:.4f}")
    print(mean.to_string())
    extra = df[~df["token"].isin(expected)]
    if len(extra):
        print("\n[devkit 집계 행]")
        print(extra.to_string())


if __name__ == "__main__":
    main()
