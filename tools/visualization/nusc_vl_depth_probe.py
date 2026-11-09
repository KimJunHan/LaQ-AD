"""수정(2026-10-15): 발표 그림 (d) 용 — VLM 영상 토큰(CAM_FRONT 10×18)에서 칸별 LiDAR 깊이를 복원하는 선형 탐침(사용자 지시:
   "(d) 색과 관련되게" → 색 = 토큰에서 복원한 깊이(m), (a) 와 같은 색 척도).
   학습 = train 분할 표본(등간격 N_TR), 평가 = val 분할 표본(등간격 N_VA, 학습에 쓰지 않음). 대상 = 칸 안 LiDAR 점(키프레임, ≥3점)의
   깊이 중앙값의 log. 예측 = 칸별 평균 log 깊이(위치 기준선, train) + ridge(토큰 − 칸별 평균 토큰). λ 는 train 안 분할(앞 80 % / 뒤 20 %)로 선택.
   R² 는 val 칸 전체 기준(전역 평균 대비)으로 '위치 기준선만' 과 '토큰 포함' 을 함께 보고 → 토큰이 위치 이상으로 깊이를 담는지 판정.
   캐시 경로를 바꾸면(--cache) 교사 캐시(laq_cache/context 등)에도 같은 탐침을 쓸 수 있다(교사 학습 전후 비교용).
   사용(CPU): CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=. python tools/visualization/nusc_vl_depth_probe.py
   → docs/ppt/figures/vl_depth_probe.npz (+ .json)
"""
import argparse, json, os, pickle
import numpy as np

GH, GW, H, W = 10, 18, 900, 1600


def load_infos(p):
    d = pickle.load(open(p, "rb"))
    return d["infos"] if isinstance(d, dict) else d


QCAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]  # Qwen 입력 순서
CAM_START = 15                              # 캠 k 영상 토큰 = 15 + 182k … +179 (lang_tokens 배치로 확인: 캠마다 시작·끝 표지 2개)


def cell_logdepth(info, cam_name="CAM_FRONT"):
    """키프레임 LiDAR → 해당 카메라 투영, 10×18 칸별 깊이 중앙값의 log (점 3개 미만 칸은 NaN)."""
    cam = info["cams"][cam_name]
    R, t, K = np.array(cam["sensor2lidar_rotation"]), np.array(cam["sensor2lidar_translation"]), np.array(cam["cam_intrinsic"])
    pts = np.fromfile(info["lidar_path"], dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
    pc = (pts - t) @ R                       # lidar → camera (p_l = R p_c + t)
    pc = pc[pc[:, 2] > 1.0]
    uv = pc @ K.T; uv = uv[:, :2] / uv[:, 2:3]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv, dep = uv[ok], pc[ok, 2]
    r = np.minimum((uv[:, 1] * GH / H).astype(int), GH - 1)
    c = np.minimum((uv[:, 0] * GW / W).astype(int), GW - 1)
    out = np.full(GH * GW, np.nan)
    k = r * GW + c
    for i in np.unique(k):
        m = k == i
        if m.sum() >= 3:
            out[i] = np.log(np.median(dep[m]))
    return out


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/nuscenes/qwen_l18")
    ap.add_argument("--tok_range", default="15,194", help="CAM_FRONT 영상 토큰 구간(포함)")
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--n_val", type=int, default=150)
    ap.add_argument("--fig_token", default="c62b6ac878934607be0c524a02f1692b")
    ap.add_argument("--out", default="docs/ppt/figures/vl_depth_probe")
    ap.add_argument("--all_cams", action="store_true", help="수정(2026-10-15 밤): 6캠 모두(캠·칸별 위치 평균), 26쪽 그림용")
    a = ap.parse_args()
    if a.all_cams:
        return main_all_cams(a)
    A, B = (int(x) for x in a.tok_range.split(","))
    tr_inf, va_inf = load_infos("data/infos/nuscenes_infos_train.pkl"), load_infos("data/infos/nuscenes_infos_val.pkl")
    tr = tr_inf[::max(len(tr_inf) // a.n_train, 1)][:a.n_train]
    va = [x for x in va_inf[::max(len(va_inf) // a.n_val, 1)][:a.n_val] if x["token"] != a.fig_token]
    fig = next(x for x in va_inf if x["token"] == a.fig_token)

    def feats_targets(infos):
        X = torch.stack([torch.load(os.path.join(a.cache, x["token"] + ".pt"), map_location="cpu")["hidden"][A:B + 1].float()
                         for x in infos]).double()
        Y = torch.from_numpy(np.stack([cell_logdepth(x) for x in infos]))
        return X, Y

    Xtr, Ytr = feats_targets(tr)
    Xva, Yva = feats_targets(va)
    Xfg, Yfg = feats_targets([fig])
    pos_x = Xtr.mean(0)                                         # 칸별 평균 토큰
    Mtr = ~torch.isnan(Ytr)
    pos_y = torch.nan_to_num(Ytr).sum(0) / Mtr.sum(0).clamp(min=1)   # 칸별 평균 log 깊이(위치 기준선)
    gy = float(Ytr[Mtr].mean())
    pos_y = torch.where(Mtr.sum(0) > 0, pos_y, torch.full_like(pos_y, gy))

    def design(X):
        Z = (X - pos_x).reshape(-1, X.shape[-1])
        return Z

    Ztr = design(Xtr); mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    norm = lambda Z: (Z - mu) / sd
    rtr = (Ytr - pos_y).reshape(-1); mtr = Mtr.reshape(-1)
    Zn = norm(Ztr)
    n = len(tr); cut = int(0.8 * n) * GH * GW
    best = None
    for lam in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
        Za, ya, ma = Zn[:cut], rtr[:cut], mtr[:cut]
        Zb, yb, mb = Zn[cut:], rtr[cut:], mtr[cut:]
        w = torch.linalg.solve(Za[ma].T @ Za[ma] + lam * torch.eye(Za.shape[1], dtype=Za.dtype), Za[ma].T @ ya[ma])
        e = float(((Zb[mb] @ w - yb[mb]) ** 2).mean())
        if best is None or e < best[0]:
            best = (e, lam)
    lam = best[1]
    w = torch.linalg.solve(Zn[mtr].T @ Zn[mtr] + lam * torch.eye(Zn.shape[1], dtype=Zn.dtype), Zn[mtr].T @ rtr[mtr])

    def predict(X):
        return (norm(design(X)) @ w).reshape(X.shape[0], -1) + pos_y

    def r2(pred, Y):
        m = ~torch.isnan(Y)
        y = Y[m]
        return float(1 - ((pred[m] - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    pva = predict(Xva)
    res = dict(cache=a.cache, n_train=len(tr), n_val=len(va), lam=lam,
               r2_val_token=r2(pva, Yva), r2_val_pos_only=r2(pos_y.expand_as(Yva), Yva),
               r2_fig_token=r2(predict(Xfg), Yfg), r2_fig_pos_only=r2(pos_y[None], Yfg),
               mae_m_val_token=float((pva.exp() - Yva.exp())[~torch.isnan(Yva)].abs().median()),
               mae_m_val_pos_only=float((pos_y.exp()[None] - Yva.exp())[~torch.isnan(Yva)].abs().median()))
    # 수정(2026-10-15 밤): 사용자 지시 — LiDAR 와 Qwen 깊이의 차이를 거리 구간별로(val 칸, LiDAR 깊이 기준 구간)
    _m = ~torch.isnan(Yva); _gt = Yva.exp()[_m]; _pt = pva.exp()[_m]; _pp = pos_y.exp()[None].expand_as(Yva)[_m]
    res["bins"] = []
    for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 50), (50, 1e9)):
        b = (_gt >= lo) & (_gt < hi)
        if b.sum() == 0:
            continue
        res["bins"].append(dict(range=[lo, hi if hi < 1e9 else None], n=int(b.sum()),
                                mae_token=float((_pt[b] - _gt[b]).abs().median()), mae_pos=float((_pp[b] - _gt[b]).abs().median()),
                                bias_token=float((_pt[b] - _gt[b]).median()), rel_token=float(((_pt[b] - _gt[b]).abs() / _gt[b]).median())))
    np.savez(a.out + ".npz", fig_pred_depth=predict(Xfg)[0].exp().numpy().reshape(GH, GW),
             fig_lidar_depth=Yfg[0].exp().numpy().reshape(GH, GW), pos_depth=pos_y.exp().numpy().reshape(GH, GW))
    json.dump(res, open(a.out + ".json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False))


def main_all_cams(a):
    """6캠: 표본 × 캠을 한 묶음으로 보고 같은 선형 탐침 1개를 학습. 위치 기준선·토큰 평균은 (캠, 칸)별."""
    import torch
    tr_inf, va_inf = load_infos("data/infos/nuscenes_infos_train.pkl"), load_infos("data/infos/nuscenes_infos_val.pkl")
    tr = tr_inf[::max(len(tr_inf) // a.n_train, 1)][:a.n_train]
    va = [x for x in va_inf[::max(len(va_inf) // a.n_val, 1)][:a.n_val] if x["token"] != a.fig_token]
    fig = next(x for x in va_inf if x["token"] == a.fig_token)

    def ft(infos):
        X = torch.stack([torch.cat([torch.load(os.path.join(a.cache, x["token"] + ".pt"), map_location="cpu")["hidden"]
                                    [CAM_START + 182 * k: CAM_START + 182 * k + 180] for k in range(6)]).float() for x in infos])
        Y = torch.from_numpy(np.stack([np.concatenate([cell_logdepth(x, c) for c in QCAMS]) for x in infos]))
        return X, Y                                            # (N, 1080, 1536) (N, 1080)

    Xtr, Ytr = ft(tr); Xva, Yva = ft(va); Xfg, Yfg = ft([fig])
    pos_x = Xtr.mean(0)
    Mtr = ~torch.isnan(Ytr)
    pos_y = torch.nan_to_num(Ytr).sum(0) / Mtr.sum(0).clamp(min=1)
    pos_y = torch.where(Mtr.sum(0) > 0, pos_y, torch.full_like(pos_y, float(Ytr[Mtr].mean())))
    Z = (Xtr - pos_x).reshape(-1, Xtr.shape[-1]); mu = Z.mean(0); sd = Z.std(0) + 1e-6
    Zn = ((Z - mu) / sd); r = (Ytr - pos_y).reshape(-1).double(); m = Mtr.reshape(-1)
    n = len(tr); cut = int(0.8 * n) * 1080
    def gram(Za, ya):                                         # float64 누적(메모리 절약을 위해 덩어리)
        G = torch.zeros(Za.shape[1], Za.shape[1], dtype=torch.float64); b = torch.zeros(Za.shape[1], dtype=torch.float64)
        for i in range(0, Za.shape[0], 50000):
            z = Za[i:i + 50000].double(); G += z.T @ z; b += z.T @ ya[i:i + 50000]
        return G, b
    ia, ib = torch.nonzero(m[:cut]).flatten(), cut + torch.nonzero(m[cut:]).flatten()
    Ga, ba = gram(Zn[ia], r[ia]); best = None
    for lam in (1e2, 1e3, 1e4, 1e5, 1e6):
        w = torch.linalg.solve(Ga + lam * torch.eye(Ga.shape[0], dtype=Ga.dtype), ba)
        e = float(((Zn[ib].double() @ w - r[ib]) ** 2).mean())
        if best is None or e < best[0]:
            best = (e, lam)
    im = torch.nonzero(m).flatten(); G, b = gram(Zn[im], r[im])
    w = torch.linalg.solve(G + best[1] * torch.eye(G.shape[0], dtype=G.dtype), b)
    pred = lambda X: ((((X - pos_x).reshape(-1, X.shape[-1]) - mu) / sd).double() @ w).reshape(X.shape[0], -1) + pos_y.double()
    def r2(P, Y):
        mm = ~torch.isnan(Y); y = Y[mm]
        return float(1 - ((P[mm] - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    Pva = pred(Xva)
    percam = {c: dict(token=r2(Pva[:, k * 180:(k + 1) * 180], Yva[:, k * 180:(k + 1) * 180]),
                      pos_only=r2(pos_y[k * 180:(k + 1) * 180].double().expand(len(va), -1), Yva[:, k * 180:(k + 1) * 180]))
              for k, c in enumerate(QCAMS)}
    res = dict(cache=a.cache, cams=QCAMS, n_train=len(tr), n_val=len(va), lam=best[1],
               r2_val_token=r2(Pva, Yva), r2_val_pos_only=r2(pos_y.double().expand(len(va), -1), Yva), per_cam=percam)
    np.savez(a.out + "_6cam.npz", fig_pred_depth=pred(Xfg)[0].exp().numpy().reshape(6, GH, GW),
             fig_lidar_depth=Yfg[0].exp().numpy().reshape(6, GH, GW))
    json.dump(res, open(a.out + "_6cam.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
