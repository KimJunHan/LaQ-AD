# 수정(2026-11-04): 두 results.pkl(val 전체, 장면 순서)을 표본별로 비교한다 — 키별 최대 절대차(max/median/p99)와 완전 일치 표본 수.
#   eager 끼리(실행 간 atomicAdd 잡음 바닥)와 eager vs 래퍼를 같은 표로 내서 래퍼 차이가 바닥 수준인지 본다.
#   입력은 이 저장소의 tools/test.py·student_test.py 가 만든 자기 결과 파일만(pickle — 외부 파일에 쓰지 말 것).
#   사용: python tools/latency/compare_results_pkl.py A.pkl B.pkl [--out x.json]
import argparse
import json
import pickle

import numpy as np


def _arr(v):
    if hasattr(v, "detach"):
        return v.detach().cpu().double().numpy()
    if isinstance(v, list):
        return np.stack([np.asarray(x, dtype=np.float64) for x in v]) if len(v) else np.zeros(0)
    return np.asarray(v, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    A = pickle.load(open(a.a, "rb"))
    B = pickle.load(open(a.b, "rb"))
    assert len(A) == len(B), (len(A), len(B))
    per = {}
    for ra, rb in zip(A, B):
        da, db = ra["img_bbox"], rb["img_bbox"]
        assert da.keys() == db.keys()
        for k in da:
            x, y = _arr(da[k]), _arr(db[k])
            d = float(np.abs(x - y).max()) if x.shape == y.shape and x.size else (0.0 if x.shape == y.shape else float("inf"))
            per.setdefault(k, []).append(d)
    res = {"a": a.a, "b": a.b, "n": len(A), "keys": {}}
    for k, v in per.items():
        v = np.asarray(v)
        res["keys"][k] = {"max_abs": float(v.max()), "median": float(np.median(v)), "p99": float(np.percentile(v, 99)),
                          "samples_exact": int((v == 0).sum())}
        print(f"  {k:18s} max {v.max():.3e}  median {np.median(v):.3e}  p99 {np.percentile(v, 99):.3e}  exact {(v == 0).sum()}/{len(v)}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
