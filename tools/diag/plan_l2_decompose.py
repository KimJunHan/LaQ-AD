# 진단(2026-11-05): 두 results.pkl 의 계획 L2 를 분해한다(명령·속도 구간·모드 선택 vs 모드 품질·종/횡 오차·경로 길이 편향).
#   tools/eval_nusc_planning.py 와 같은 L2 식(시점별 L2 → 누적 평균, 1/2/3 s = 2/4/6 번째, avg = 셋 평균, 미래 GT 완전 표본만).
#   읽기 전용. 사용(저장소 루트): python tools/diag/plan_l2_decompose.py <test_config> a=<results.pkl> b=<results.pkl> --out x.json
import argparse
import functools
import json
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, ".")


def l2_123(e):  # e: (n,6) 시점별 L2 → (1s,2s,3s,avg)
    if len(e) == 0:
        return [float("nan")] * 4
    cum = [e[:, :j + 1].mean() for j in range(6)]
    v = [cum[1], cum[3], cum[5]]
    return [float(x) for x in v] + [float(np.mean(v))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("runs", nargs=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import importlib
    importlib.import_module("mmdet3d_plugin")
    from mmcv import Config
    from mmdet.datasets import build_dataset
    ds = build_dataset(Config.fromfile(a.config).data.test)
    infos = ds.data_infos
    valid = np.array([bool((np.asarray(i["gt_ego_fut_masks"]) == 1).all()) for i in infos])
    G = np.stack([np.cumsum(np.asarray(i["gt_ego_fut_trajs"], np.float32), 0)[:6, :2] for i in infos])
    cmd = np.array([int(np.argmax(i["gt_ego_fut_cmd"])) for i in infos])  # 0 우, 1 좌, 2 직진
    es = np.stack([np.asarray(i.get("ego_status", np.zeros(10)), np.float32) for i in infos])
    speed = np.linalg.norm(es[:, 6:8], axis=1)
    gt_len = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(G), 1, 2)), G], 1), axis=1), axis=-1).sum(1)

    _tl = torch.load
    torch.load = functools.partial(_tl, map_location="cpu", weights_only=False)
    R = {}
    try:
        for s in a.runs:
            name, path = s.split("=", 1)
            res = pickle.load(open(path, "rb"))
            P = np.stack([r["img_bbox"]["plan_temp_2hz"].float().numpy()[:6, :2] for r in res])
            M = np.stack([r["img_bbox"]["plan_modes"].float().numpy()[:, :6, :2] for r in res])
            Pr = np.stack([r["img_bbox"]["plan_mode_probs"].float().numpy() for r in res])
            del res
            R[name] = (P, M, Pr)
    finally:
        torch.load = _tl

    # 진행 방향(GT 접선) 기준 종/횡 분해: 시점 t 의 GT 접선 = G[t]-G[t-1] (t=0 은 원점→G[0]).
    Gp = np.concatenate([np.zeros((len(G), 1, 2)), G], 1)
    tang = Gp[:, 1:] - Gp[:, :-1]
    tn = np.linalg.norm(tang, axis=-1, keepdims=True)
    # 거의 정지한 시점은 전체 변위 방향, 그것도 없으면 차량 전방(+x 또는 +y 는 아래에서 자료로 판정)
    tot = G[:, -1:] - 0
    tot_n = np.linalg.norm(tot, axis=-1, keepdims=True)
    fwd_axis = int(np.argmax(np.abs(G[valid][:, -1]).mean(0)))
    default = np.zeros(2); default[fwd_axis] = 1.0
    u = np.where(tn > 0.2, tang / np.maximum(tn, 1e-6), np.where(tot_n > 0.5, tot / np.maximum(tot_n, 1e-6), default))
    nrm = np.stack([-u[..., 1], u[..., 0]], -1)

    out = {"n_valid": int(valid.sum()), "fwd_axis": fwd_axis, "runs": {}}
    sb = [(0, 0.5), (0.5, 3), (3, 6), (6, 9), (9, 12), (12, 99)]
    for n, (P, M, Pr) in R.items():
        e = np.linalg.norm(P - G, axis=-1)
        sel_idx = Pr.argmax(1)
        P_sel = M[np.arange(len(M)), sel_idx]
        consistent = float(np.abs(P_sel - P).max(-1).max(-1).mean() if True else 0)
        eM = np.linalg.norm(M - G[:, None], axis=-1)  # (N,16,6)
        best = eM.mean(-1).argmin(1)
        e_or = eM[np.arange(len(M)), best]
        d = P - G
        lon = (d * u).sum(-1)
        lat = (d * nrm).sum(-1)
        plen = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(P), 1, 2)), P], 1), axis=1), axis=-1).sum(1)
        v = valid
        r = {
            "L2": l2_123(e[v]),
            "oracle_best_of_16_L2": l2_123(e_or[v]),
            "sel_equals_argmax_maxdiff": consistent,
            "sel_is_oracle_frac": float((sel_idx == best)[v].mean()),
            "top_prob_mean": float(Pr.max(1)[v].mean()),
            "entropy_mean": float((-(Pr * np.log(Pr + 1e-12)).sum(1))[v].mean()),
            "abs_lon_err_3s": float(np.abs(lon[v, 5]).mean()),
            "abs_lat_err_3s": float(np.abs(lat[v, 5]).mean()),
            "signed_lon_err_3s": float(lon[v, 5].mean()),
            "rms_lon_err_3s": float(np.sqrt((lon[v, 5] ** 2).mean())),
            "rms_lat_err_3s": float(np.sqrt((lat[v, 5] ** 2).mean())),
            "lon_err_t": [float(np.abs(lon[v, t]).mean()) for t in range(6)],
            "lat_err_t": [float(np.abs(lat[v, t]).mean()) for t in range(6)],
            "path_len_bias_3s": float((plen - gt_len)[v].mean()),
            "path_len_abs_err_3s": float(np.abs(plen - gt_len)[v].mean()),
            "by_cmd": {}, "by_speed": {},
        }
        for c, cn in enumerate(["right", "left", "straight"]):
            m = v & (cmd == c)
            r["by_cmd"][cn] = {"n": int(m.sum()), "L2": l2_123(e[m]), "oracle": l2_123(e_or[m]),
                               "abs_lon3": float(np.abs(lon[m, 5]).mean()), "abs_lat3": float(np.abs(lat[m, 5]).mean())}
        for lo, hi in sb:
            m = v & (speed >= lo) & (speed < hi)
            r["by_speed"][f"{lo}-{hi}"] = {"n": int(m.sum()), "L2": l2_123(e[m]), "oracle": l2_123(e_or[m]),
                                           "abs_lon3": float(np.abs(lon[m, 5]).mean()), "abs_lat3": float(np.abs(lat[m, 5]).mean()),
                                           "len_bias3": float((plen - gt_len)[m].mean())}
        out["runs"][n] = r
        R[n] = R[n] + (e, e_or, lon, lat, plen)
    # 표본별 차(b - a)의 기여 분해
    na, nb = list(R)
    ea, eb = R[na][3], R[nb][3]
    d = (eb.mean(1) - ea.mean(1))
    tot = d[valid].sum()
    contrib = {}
    for c, cn in enumerate(["right", "left", "straight"]):
        m = valid & (cmd == c)
        contrib["cmd_" + cn] = float(d[m].sum() / valid.sum())
    for lo, hi in sb:
        m = valid & (speed >= lo) & (speed < hi)
        contrib[f"speed_{lo}-{hi}"] = float(d[m].sum() / valid.sum())
    out["per_sample_diff_mean"] = float(d[valid].mean())
    out["contrib_to_mean6_diff"] = contrib
    # 모드 품질 vs 선택: 오라클 차, 그리고 "b 의 선택을 a 의 순위로" 같은 교차는 모드 집합이 달라 불가 → 오라클·선택 격차로 대체.
    out["gap_sel_minus_oracle"] = {n: float((R[n][3].mean(1) - R[n][4].mean(1))[valid].mean()) for n in R}
    # 상위 차 표본 비율(꼬리 vs 전반): 차의 분위수
    dv = d[valid]
    out["diff_quantiles"] = {q: float(np.quantile(dv, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
    srt = np.sort(dv)[::-1]
    out["top_frac_share"] = {f"top{p}%": float(srt[: int(len(srt) * p / 100)].sum() / max(srt.sum(), 1e-9)) for p in (1, 5, 10)}
    # 속도와의 상관
    out["corr_diff_speed"] = float(np.corrcoef(dv, speed[valid])[0, 1])
    json.dump(out, open(a.out, "w"), indent=1)
    np.savez_compressed(a.out.replace(".json", "_persample.npz"), valid=valid, cmd=cmd, speed=speed, gt_len=gt_len,
                        **{f"{n}_{k}": R[n][i] for n in R for i, k in [(0, "P"), (3, "e"), (4, "e_or"), (5, "lon"), (6, "lat"), (7, "plen")]})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
