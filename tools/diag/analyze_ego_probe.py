# 진단(2026-11-05): ego_probe.py 덤프들(probe_<name>.npz)의 계획 L2(eval_nusc_planning 식)와 자차 상태 예측 오차를 표로.
#   GT 는 val info(ego_status 10 = accel3·rot3·vel3·steer, gt_ego_fut_trajs). 읽기 전용.
#   사용(저장소 루트): python tools/diag/analyze_ego_probe.py results/diag/probe_*.npz --out results/diag/ego_probe_summary.json
import argparse
import json
import os.path as osp
import pickle  # 자체 생성한 info pkl 만 읽는다

import numpy as np


def l2_123(e):
    cum = [e[:, :j + 1].mean() for j in range(6)]
    v = [cum[1], cum[3], cum[5]]
    return [round(float(x), 4) for x in v + [np.mean(v)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--ann", default="data/infos/nuscenes_infos_val.pkl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    infos = pickle.load(open(a.ann, "rb"))["infos"]
    # 결과 순서 = dataset.data_infos 순서 = pkl 의 infos 순서 그대로(NuScenes3DDataset 은 정렬하지 않는다).
    valid = np.array([bool((np.asarray(i["gt_ego_fut_masks"]) == 1).all()) for i in infos])
    G = np.stack([np.cumsum(np.asarray(i["gt_ego_fut_trajs"], np.float32), 0)[:6, :2] for i in infos])
    es = np.stack([np.asarray(i["ego_status"], np.float32) for i in infos])
    esv = np.isfinite(es).all(1) & (np.abs(es).sum(1) > 0)  # CAN 결측(전부 0) 제외
    speed = np.linalg.norm(es[:, 6:8], axis=1)
    gl = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(G), 1, 2)), G], 1), axis=1), axis=-1).sum(1)
    acc = gl / 3 - speed
    out = {}
    for p in a.npz:
        z = np.load(p)
        n = osp.basename(p)[6:-4]
        P, E = z["plan"], z["ego"]
        e = np.linalg.norm(P - G, axis=-1)
        pv = np.linalg.norm(E[:, 6:8], axis=1)
        m = esv
        r = {"L2": l2_123(e[valid]),
             "ego_abs_err": {"accel_xy": float(np.abs(E[m, 0:2] - es[m, 0:2]).mean()),
                             "yaw_rate": float(np.abs(E[m, 5] - es[m, 5]).mean()),
                             "vel_xy": float(np.abs(E[m, 6:8] - es[m, 6:8]).mean()),
                             "speed": float(np.abs(pv[m] - speed[m]).mean()),
                             "speed_bias": float((pv[m] - speed[m]).mean()),
                             "steer": float(np.abs(E[m, 9] - es[m, 9]).mean()),
                             "all10_L1": float(np.abs(E[m] - es[m]).mean())},
             "speed_err_by_speed": {}, "L2avg_by_acc": {}}
        for lo, hi in [(0, 0.5), (0.5, 3), (3, 6), (6, 9), (9, 99)]:
            mm = m & (speed >= lo) & (speed < hi)
            r["speed_err_by_speed"][f"{lo}-{hi}"] = round(float(np.abs(pv[mm] - speed[mm]).mean()), 3)
        for nm, mm in [("accel", acc > 1), ("const", np.abs(acc) <= 1), ("decel", acc < -1)]:
            mm = mm & valid
            r["L2avg_by_acc"][nm] = l2_123(e[mm])[3]
        # 자차 상태 오차와 계획 종방향 오차의 관계: 예측 3 s 이동거리 - GT
        plen = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(P), 1, 2)), P], 1), axis=1), axis=-1).sum(1)
        mm = m & valid
        r["corr_speedErr_lenErr"] = float(np.corrcoef(pv[mm] - speed[mm], plen[mm] - gl[mm])[0, 1])
        r["len_bias_3s"] = float((plen - gl)[valid].mean())
        out[n] = r
        print(n, json.dumps(r))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
