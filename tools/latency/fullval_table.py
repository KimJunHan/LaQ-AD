# 수정(2026-11-04): 전체 val 채점 결과(tools/test.py 로그 마지막 지표 dict + eval_nusc_planning.py json)를 한 표로 모은다.
#   사용: python tools/latency/fullval_table.py name=<eval.log>,<planning_nusc.json> ... [--out x.json]
import argparse
import ast
import json

KEYS = ["mAP", "NDS", "mATE", "mASE", "mAOE", "mAVE", "mAAE", "mAP_normal", "ped_crossing", "divider", "boundary",
        "tracking_raw/amota", "tracking/amota", "tracking/amotp", "tracking/ids",
        "car_min_ade_err", "car_min_fde_err", "car_miss_rate_err", "car_EPA", "pedestrian_min_ade_err", "pedestrian_EPA"]


def load(log, plan):
    last = None
    with open(log, errors="ignore") as f:
        for line in f.read().replace("\r", "\n").split("\n"):
            if line.startswith("{'mAP'"):
                last = line
    d = ast.literal_eval(last)
    r = {k: d.get(k) for k in KEYS}
    p = json.load(open(plan))
    for g in ("L2", "obj_col", "obj_box_col"):
        for t in ("1s", "2s", "3s", "avg"):
            r[f"plan_{g}_{t}"] = p[g][t]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = {}
    for s in a.runs:
        name, paths = s.split("=", 1)
        log, plan = paths.split(",")
        rows[name] = load(log, plan)
    names = list(rows)
    print(f"{'metric':26s}" + "".join(f"{n:>16s}" for n in names))
    for k in rows[names[0]]:
        print(f"{k:26s}" + "".join(f"{rows[n][k]:16.4f}" if isinstance(rows[n][k], float) else f"{str(rows[n][k]):>16s}" for n in names))
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
