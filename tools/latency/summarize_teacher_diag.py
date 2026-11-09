# 수정(2026-11-04): 교사 답 구간 진단 요약 — teacher_usage/*.json 과 teacher_subset/*/summary.json(·planning_nusc.json)을 표로.
#   사용: python tools/latency/summarize_teacher_diag.py > results/latency/teacher_diag_tables.md
import glob
import json
import os
import os.path as osp

R = osp.join(osp.dirname(__file__), "..", "..", "results", "latency")
FOLD = ("context", "detection", "tracking", "motion", "map", "plan", "action")


def usage():
    for p in sorted(glob.glob(osp.join(R, "teacher_usage", "*.json"))):
        d = json.load(open(p))
        s, info = d["stats"], d["info"]
        L = info["L"]
        print(f"\n## 사용도 {osp.basename(p)} — 프레임 {d['frames']}, 장면 {d['scenes']}, 쿼리 N={info['N']}, 토큰 L={L}\n")
        print("게이트 |g| 평균/최대 (계층/구간): " + ", ".join(
            f"{k} {v['mean_abs']:.3f}/{v['max_abs']:.3f}" for k, v in d["gates"].items() if isinstance(v, dict)))
        for q in ("det", "map", "plan", "ego"):
            if f"0|{q}|ALL|rel" not in s:
                continue
            print(f"\n### 쿼리 {q}: mass(질량) / 밀도비(mass÷(L_f/L)) / rel=||g⊙A_f||/||q|| (%)\n")
            print("| 계층 | 전체 rel% | " + " | ".join(FOLD) + " |")
            print("|---|---|" + "---|" * len(FOLD))
            for li in range(6):
                cells = []
                for f in FOLD:
                    m = s.get(f"{li}|{q}|{f}|mass"); r = s.get(f"{li}|{q}|{f}|rel")
                    if m is None:
                        cells.append("-"); continue
                    dens = m / (info[f"L_{f}"] / L)
                    cells.append(f"{m:.3f} / {dens:.2f} / {100 * r:.2f}")
                print(f"| {li} | {100 * s[f'{li}|{q}|ALL|rel']:.2f} | " + " | ".join(cells) + " |")
            # 계층 평균
            cells = []
            for f in FOLD:
                ms = [s[f"{li}|{q}|{f}|mass"] for li in range(6)]; rs = [s[f"{li}|{q}|{f}|rel"] for li in range(6)]
                cells.append(f"{sum(ms) / 6:.3f} / {sum(ms) / 6 / (info[f'L_{f}'] / L):.2f} / {100 * sum(rs) / 6:.2f}")
            tot = sum(s[f"{li}|{q}|ALL|rel"] for li in range(6)) / 6
            print(f"| 평균 | {100 * tot:.2f} | " + " | ".join(cells) + " |")


def ablation():
    rows = []
    for p in sorted(glob.glob(osp.join(R, "teacher_subset", "*", "summary.json"))):
        d = json.load(open(p))
        name = osp.basename(osp.dirname(p))
        m = d.get("metrics", {})
        pl = osp.join(osp.dirname(p), "planning_nusc.json")
        pl = json.load(open(pl)) if osp.exists(pl) else None
        rows.append((name, m, pl))
    print("\n## 전체 val 추론 전용 진단\n")
    print("| run | mAP | NDS | map mAP | AMOTA | AMOTA raw | IDS raw | L2 avg | Col avg % |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, m, pl in rows:
        g = lambda k: (f"{m[k]:.4f}" if k in m else "-")  # noqa: E731
        ids = f"{int(m['tracking_raw/ids'])}" if "tracking_raw/ids" in m else "-"
        l2 = f"{pl['L2']['avg']:.4f}" if pl else "-"
        col = f"{100 * pl['obj_box_col']['avg']:.3f}" if pl else "-"
        print(f"| {name} | {g('mAP')} | {g('NDS')} | {g('mAP_normal')} | {g('tracking/amota')} | {g('tracking_raw/amota')} | {ids} | {l2} | {col} |")


if __name__ == "__main__":
    usage()
    ablation()
