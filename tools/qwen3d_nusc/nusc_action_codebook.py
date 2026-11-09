"""수정(2026-10-09): AutoVLA 방식 물리 행동 코드북(nuScenes) — 사용자 지시("AutoVLA 기준으로 action 데이터 추가").

  AutoVLA(backup/reference/AutoVLA.pdf §Action Tokenization, 부록) 정의를 따른다:
    - 행동 토큰 하나 = 0.5 s 움직임 (Δx, Δy, Δθ). 코드북 = 0.5 s 조각들을 K-disk 군집해 고른 대표 K=2048 개.
    - 조각 간 거리 = 조각 끝 시점 자차 박스 윤곽(현재 프레임 기준)의 차이. 토큰 추가: <action_0> ... <action_{K-1}>.
    - AutoVLA 는 5 s = 10 토큰. nuScenes GT 는 3 s(2 Hz×6)이므로 6 토큰.
  좌표: nuScenes lidar 프레임, +y = 전방, +x = 우측(make_spatial_captions.py 실증 규약). GT 는 위치만 있으므로
    Δθ 는 궤적 접선 방향으로 구한다(SparseDrive 계획 평가기 get_yaw 와 같은 규약: 0.5 m 미만 이동은 방향 유지).
  산출: data/qwen3d/nusc_action_codebook_k2048.npz (codebook (K,3), delta, 재구성 오차 통계)
  재구성 검증: GT 궤적을 코드북으로 순차 토큰화(누적 자세 기준 최근접, 드리프트 없음) → ADE/FDE 보고(AutoVLA Table 5 와 같은 지표).
"""
import argparse, pickle
import numpy as np

EGO_L, EGO_W = 4.084, 1.85   # SparseDrive/UniAD 계획 평가기의 자차 크기


def corners(dx, dy, dth):
    """조각 끝 자세(조각 시작 프레임 기준)에서 자차 박스 4모서리 (N,4,2). 전방=+y."""
    hl, hw = EGO_L / 2, EGO_W / 2
    base = np.array([[-hw, hl], [hw, hl], [hw, -hl], [-hw, -hl]], np.float64)   # (x, y), 전방 +y
    c, s = np.cos(dth), np.sin(dth)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)          # (N,2,2) 반시계 +
    return np.einsum('nij,kj->nki', rot, base) + np.stack([dx, dy], -1)[:, None]


def heading_series(xy):
    """(T+1,2) 위치열(0번=현재 원점) → 각 시점 진행 방향(전방 +y 기준 반시계 각, rad)."""
    th = np.zeros(len(xy))
    for t in range(1, len(xy)):
        d = xy[t] - xy[t - 1]
        th[t] = th[t - 1] if np.linalg.norm(d) < 0.5 * 0.5 else np.arctan2(-d[0], d[1])
    return th


def segments_from_traj(fut):
    """fut: (6,2) 누적 위치(lidar). 반환 (6,3) 각 0.5 s 조각의 (Δx, Δy, Δθ) — 조각 시작 자세 기준."""
    xy = np.concatenate([np.zeros((1, 2)), fut], 0)
    th = heading_series(xy)
    segs = []
    for t in range(len(fut)):
        d = xy[t + 1] - xy[t]
        c, s = np.cos(-th[t]), np.sin(-th[t])
        local = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        segs.append([local[0], local[1], th[t + 1] - th[t]])
    return np.array(segs)


def kdisk(feat, k, iters=30, seed=0):
    """K-disk: 반경 δ 안에 이미 대표가 없으면 대표로 추가(그리디). δ 를 이분탐색해 대표 수 ≈ k."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(feat))
    f = feat[order]
    def run(delta):
        cen = [f[0]]
        C = np.empty((0, f.shape[1])); C = np.vstack([C, f[0]])
        for i in range(1, len(f)):
            if np.min(np.linalg.norm(C - f[i], axis=1)) > delta:
                C = np.vstack([C, f[i]])
                if len(C) > k * 1.5: break
        return C
    lo, hi = 1e-3, 5.0
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        C = run(mid)
        if len(C) > k: lo = mid
        else: hi = mid; best = (mid, C)
        if best and abs(len(best[1]) - k) <= k * 0.01: break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/infos/nuscenes_infos_train.pkl")
    ap.add_argument("--val", default="data/infos/nuscenes_infos_val.pkl")
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--out", default="data/qwen3d/nusc_action_codebook_k2048.npz")
    a = ap.parse_args()

    def load(p):
        d = pickle.load(open(p, "rb")); d = d["infos"] if isinstance(d, dict) else d
        S, T = [], []
        for info in d:
            m = np.asarray(info["gt_ego_fut_masks"])
            if not (m == 1).all(): continue
            fut = np.asarray(info["gt_ego_fut_trajs"], np.float64).cumsum(0)[:6, :2]
            S.append(segments_from_traj(fut)); T.append(fut)
        return np.concatenate(S), np.stack(T)
    seg_tr, traj_tr = load(a.train)
    seg_va, traj_va = load(a.val)
    feat = corners(seg_tr[:, 0], seg_tr[:, 1], seg_tr[:, 2]).reshape(len(seg_tr), -1)
    print(f"train 조각 {len(seg_tr)}개 (궤적 {len(traj_tr)}), val 궤적 {len(traj_va)}")
    delta, C = kdisk(feat, a.k)
    # 대표 = 대표 윤곽에 가장 가까운 실제 조각의 (Δx,Δy,Δθ)
    idx = [int(np.argmin(np.linalg.norm(feat - c, axis=1))) for c in C]
    book = seg_tr[idx]
    print(f"K-disk δ={delta:.4f} → 코드북 {len(book)}개")

    bc = corners(book[:, 0], book[:, 1], book[:, 2]).reshape(len(book), -1)
    def tokenize(fut):
        """현재 자세에서 시작해 매 0.5 s 마다 GT 다음 위치에 가장 가까운 코드를 고르고 자세를 누적(드리프트 없는 폐루프 토큰화)."""
        pos, th, toks, rec = np.zeros(2), 0.0, [], []
        for t in range(len(fut)):
            c, s = np.cos(-th), np.sin(-th)
            d = fut[t] - pos
            tgt = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
            e = np.linalg.norm(book[:, :2] - tgt, axis=1)
            j = int(np.argmin(e)); toks.append(j)
            c2, s2 = np.cos(th), np.sin(th)
            pos = pos + np.array([c2 * book[j, 0] - s2 * book[j, 1], s2 * book[j, 0] + c2 * book[j, 1]])
            th += book[j, 2]; rec.append(pos.copy())
        return toks, np.array(rec)
    errs = []
    for fut in traj_va:
        _, rec = tokenize(fut)
        errs.append(np.linalg.norm(rec - fut, axis=1))
    errs = np.array(errs)
    ade, fde = errs.mean(), errs[:, -1].mean()
    print(f"val 재구성(3 s): ADE {ade:.4f} m  FDE {fde:.4f} m  (AutoVLA Table 5, K=2048 nuPlan 5 s: ADE 0.0182 / FDE 0.0203)")
    np.savez(a.out, codebook=book, delta=delta, val_ade=ade, val_fde=fde, ego_size=np.array([EGO_L, EGO_W]))
    print("->", a.out)


if __name__ == "__main__":
    main()
