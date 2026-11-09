# 진단(2026-11-05): 두 2단계 학습 로그(.log.json)의 손실을 같은 iter 구간별 평균으로 비교. 읽기 전용.
# 사용: python tools/diag/compare_train_curves.py a=<log.json> b=<log.json> [--bins 3516]
import argparse, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs=2); ap.add_argument("--bins", type=int, default=3516)
a = ap.parse_args()
data = {}
for s in a.runs:
    n, p = s.split("=", 1); rows = []
    for line in open(p):
        d = json.loads(line)
        if d.get("mode") == "train": rows.append(d)
    data[n] = rows
keys = [k for k in data[list(data)[0]][0] if "loss" in k or k == "grad_norm"]
names = list(data)
for k in keys:
    print(f"== {k}")
    out = []
    for n in names:
        it = np.array([r["iter"] for r in data[n]]); v = np.array([r.get(k, np.nan) for r in data[n]], float)
        out.append([np.nanmean(v[(it > lo) & (it <= lo + a.bins)]) for lo in range(0, int(it.max()), a.bins)])
    for i, (x, y) in enumerate(zip(*out)):
        print(f"  ep{i+1:2d} {names[0]}={x:.4f} {names[1]}={y:.4f} diff={y-x:+.4f} ({(y-x)/x*100:+.1f}%)")
