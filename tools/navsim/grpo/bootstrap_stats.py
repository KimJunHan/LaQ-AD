# 수정(2026-11-09): C3 GRPO 설계 v4.1 §10 판단(사전 등록) — 신규 파일(기존 파일 무수정).
"""navtest 로그 블록 paired bootstrap(10k) → 1차 가설 IUT + 2차 비교 8 개 Holm. numpy 만 쓴다(어느 env 든).

입력: --runs 이름=CSV[,CSV2] … (공식 run_pdm_score_one_stage CSV: token, valid, …, score; 마지막 average 행은 뺀다).
  시드가 둘인 런(Full, B1)은 CSV 두 개를 쉼표로 준다 → **토큰별 시드 평균 EPDMS**(사전 고정)를 그 런 점수로 쓴다. 시드 간 차이도 보고.
  --infos navtest info pkl(navsim_token → log_name). 모든 런의 토큰 집합이 같아야 한다(아니면 중단).
  valid False 행: 공식 CSV 의 score 를 그대로 쓴다(공식 평균과 같은 처리). score 가 NaN 이면 중단.
재표집 단위: log_name 블록(같은 로그 토큰은 함께). 블록을 복원 추출(10k, 시드 고정 0)하고 Δ = 평균(런 A) − 평균(런 B) 를
  토큰 가중(블록 크기)으로 다시 계산한다.
판정
  1차 H(IUT): Full − B1, Full − C1 의 단측 95 % 하한(백분위 5 %) 이 **둘 다** > 0 이면 지지.
  2차 8 개(Holm, 족 α = 0.05, 양측): Full vs A−a, A−b, A−d, B2, R, P / B1 vs C1 / C1 vs C2.
    양측 p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)) (bootstrap 분포, +1 보정).
  G3 중단 런은 --g3-stopped 로 표시(기술 통계만, 검정 지지 근거로 쓰지 않음 — 표에 표시).
출력: --out json + 표(stdout). 하위 점수·실패 유형별 비율도 런별로 낸다(CSV 하위 점수 열 평균, < 1 비율).
"""
import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np

PRIMARY = [("Full", "B1"), ("Full", "C1")]
SECONDARY = [("Full", "A-a"), ("Full", "A-b"), ("Full", "A-d"), ("Full", "B2"), ("Full", "R"), ("Full", "P"), ("B1", "C1"),
             ("C1", "C2")]
SUBS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance",
        "ego_progress", "time_to_collision_within_bound", "lane_keeping", "history_comfort", "two_frame_extended_comfort"]


def read_csv(p):
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            t = r["token"]
            if t == "average_all_frames" or t.startswith("average"):
                continue
            s = float(r["score"])
            if not math.isfinite(s):
                raise SystemExit(f"{p}: {t} score 비유한")
            subs = {k: float(r[k]) for k in SUBS if k in r and r[k] != ""}
            out[t] = (s, subs, r.get("valid") == "True")
    return out


def holm(ps, alpha=0.05):
    order = np.argsort(ps)
    m = len(ps)
    rej = np.zeros(m, bool)
    for i, j in enumerate(order):
        if ps[j] <= alpha / (m - i):
            rej[j] = True
        else:
            break
    adj = np.zeros(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j]))
        adj[j] = run
    return rej, adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="이름=CSV[,CSV2]")
    ap.add_argument("--infos", required=True, help="navtest info pkl(log_name)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--g3-stopped", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pickle.load(open(a.infos, "rb"))
    log_of = {i["navsim_token"]: i["log_name"] for i in d["infos"] if i.get("is_eval_frame")}
    runs, seeds_diff, sub_stats = {}, {}, {}
    for spec in a.runs:
        name, paths = spec.split("=", 1)
        tabs = [read_csv(p) for p in paths.split(",")]
        keys = set(tabs[0])
        for t in tabs[1:]:
            if set(t) != keys:
                raise SystemExit(f"{name}: 시드 CSV 토큰 집합이 다르다")
        runs[name] = {k: float(np.mean([t[k][0] for t in tabs])) for k in keys}
        if len(tabs) == 2:
            dd = np.array([tabs[0][k][0] - tabs[1][k][0] for k in sorted(keys)])
            seeds_diff[name] = dict(mean_s0=float(np.mean([tabs[0][k][0] for k in keys])),
                                    mean_s1=float(np.mean([tabs[1][k][0] for k in keys])), mean_abs_token_diff=float(np.abs(dd).mean()))
        sub_stats[name] = {s: dict(mean=float(np.mean([np.mean([t[k][1][s] for t in tabs]) for k in keys])),
                                   fail_rate=float(np.mean([np.mean([t[k][1][s] < 1.0 for t in tabs]) for k in keys])))
                           for s in SUBS if all(s in t[next(iter(keys))][1] for t in tabs)}
    toks = sorted(next(iter(runs.values())))
    for n, r in runs.items():
        if set(r) != set(toks):
            raise SystemExit(f"{n}: 토큰 집합이 다른 런과 다르다")
    miss = [t for t in toks if t not in log_of]
    if miss:
        raise SystemExit(f"info 에 log_name 없는 토큰 {len(miss)}")
    logs = sorted({log_of[t] for t in toks})
    li = {l: i for i, l in enumerate(logs)}
    tl = np.array([li[log_of[t]] for t in toks])
    nb = len(logs)
    cnt = np.bincount(tl, minlength=nb).astype(np.float64)
    S = {n: np.bincount(tl, weights=np.array([r[t] for t in toks]), minlength=nb) for n, r in runs.items()}
    rng = np.random.default_rng(a.seed)
    W = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(a.n_boot)]).astype(np.float64)  # (n_boot, nb)
    den = W @ cnt

    def boot(x, y):
        delta = (S[x].sum() - S[y].sum()) / cnt.sum()
        bs = (W @ (S[x] - S[y])) / den
        return delta, bs

    res = dict(n_tokens=len(toks), n_logs=nb, n_boot=a.n_boot, means={n: float(np.mean(list(r.values()))) for n, r in runs.items()},
               seeds=seeds_diff, subscores=sub_stats, g3_stopped=a.g3_stopped, primary={}, secondary={})
    ok = True
    for x, y in PRIMARY:
        if x not in runs or y not in runs:
            raise SystemExit(f"1차 비교 {x}−{y} 런 없음")
        dl, bs = boot(x, y)
        lo = float(np.quantile(bs, 0.05))
        res["primary"][f"{x}-{y}"] = dict(delta=dl, lower95_one_sided=lo, ci90=[lo, float(np.quantile(bs, 0.95))])
        ok &= lo > 0
    res["primary"]["H_supported"] = bool(ok) and not any(r in a.g3_stopped for r in ("Full", "B1", "C1"))
    names, ps, rows = [], [], []
    for x, y in SECONDARY:
        if x not in runs or y not in runs:
            res["secondary"][f"{x}-{y}"] = dict(missing=True)
            continue
        dl, bs = boot(x, y)
        p = min(1.0, 2 * min((np.sum(bs <= 0) + 1) / (a.n_boot + 1), (np.sum(bs >= 0) + 1) / (a.n_boot + 1)))
        names.append(f"{x}-{y}"); ps.append(p)
        rows.append(dict(delta=dl, ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], p_two_sided=p,
                         g3_stopped=[r for r in (x, y) if r in a.g3_stopped]))
    if ps:
        rej, adj = holm(np.array(ps))
        for nm, r, rj, pa in zip(names, rows, rej, adj):
            r.update(holm_reject=bool(rj), p_holm=float(pa))
            res["secondary"][nm] = r
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"토큰 {len(toks)}, 로그 블록 {nb}, bootstrap {a.n_boot}")
    for n, m in res["means"].items():
        print(f"  {n:5s} EPDMS {m:.4f}" + ("  [G3 중단]" if n in a.g3_stopped else ""))
    for k, v in res["primary"].items():
        print(f"  1차 {k}: {v}")
    for k, v in res["secondary"].items():
        print(f"  2차 {k}: {v}")


if __name__ == "__main__":
    main()
