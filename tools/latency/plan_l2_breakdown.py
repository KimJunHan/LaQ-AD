# 수정(2026-11-05): 여러 results.pkl 의 계획 L2(tools/eval_nusc_planning.py 와 같은 식, 표본별 6 점 평균)를 표본별로 내고
#   실행 사이 차이가 어디서 나는지(장면 안 위치, 장면) 나눈다 — graph_post 와 eager 의 L2 평균 차가 잡음인지 체계적인지 확인용.
#   입력은 이 저장소에서 만든 자기 결과 파일만(pickle).
#   사용: python tools/latency/plan_l2_breakdown.py <config> name=<results.pkl> ... --out x.json
import argparse
import functools
import json
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import importlib
    importlib.import_module("mmdet3d_plugin")
    from mmcv import Config
    from mmdet.datasets import build_dataset
    ds = build_dataset(Config.fromfile(a.config).data.test)
    infos = ds.data_infos
    valid, gts, pos, scene = [], [], [], []
    last_scene, k = None, 0
    for info in infos:
        k = k + 1 if info["scene_token"] == last_scene else 0
        last_scene = info["scene_token"]
        pos.append(k)
        scene.append(info["scene_token"])
        m = np.asarray(info["gt_ego_fut_masks"], np.float32)
        valid.append(bool((m == 1).all()))
        gts.append(np.cumsum(np.asarray(info["gt_ego_fut_trajs"], np.float32), 0)[:6, :2])
    valid = np.asarray(valid)
    pos = np.asarray(pos)
    _tl = torch.load
    torch.load = functools.partial(_tl, map_location="cpu", weights_only=False)
    per = {}
    preds = {}
    try:
        for s in a.runs:
            name, path = s.split("=", 1)
            res = pickle.load(open(path, "rb"))
            P = np.stack([np.asarray(torch.as_tensor(r["img_bbox"]["plan_temp_2hz"], dtype=torch.float32))[:6, :2] for r in res])
            del res
            preds[name] = P
            # eval_nusc_planning 과 같음: x 부호 반전은 L2 에 영향 없음
            per[name] = np.sqrt(((P - np.stack(gts)) ** 2).sum(-1))  # (N,6)
    finally:
        torch.load = _tl
    names = list(per)
    out = {"n_valid": int(valid.sum()), "runs": {}}
    for n in names:
        e = per[n][valid]
        cum = [float(e[:, :j + 1].mean()) for j in range(6)]
        out["runs"][n] = {"L2_avg": (cum[1] + cum[3] + cum[5]) / 3}
    base = names[0]
    out["vs_" + base] = {}
    for n in names[1:]:
        d = per[n].mean(1) - per[base].mean(1)  # 표본별 6 점 평균 L2 차
        dv = d[valid]
        bins = {}
        for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 41)):
            m = valid & (pos >= lo) & (pos < hi)
            bins[f"pos{lo}-{hi - 1}"] = {"n": int(m.sum()), "sum_diff": float(d[m].sum()), "mean_diff": float(d[m].mean()) if m.any() else 0.0}
        top = np.argsort(-np.abs(np.where(valid, d, 0)))[:15]
        out["vs_" + base][n] = {"mean_diff": float(dv.mean()), "sum_diff": float(dv.sum()),
                                "frac_exact_pred": float((np.abs(preds[n] - preds[base]).max((1, 2)) == 0).mean()),
                                "by_pos_in_scene": bins,
                                "top_abs": [{"i": int(i), "pos": int(pos[i]), "diff": float(d[i])} for i in top]}
        print(n, "mean diff vs", base, f"{dv.mean():.5f}")
        for kk, v in bins.items():
            print(f"   {kk:9s} n={v['n']:5d} sum={v['sum_diff']:+.3f} mean={v['mean_diff']:+.5f}")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
