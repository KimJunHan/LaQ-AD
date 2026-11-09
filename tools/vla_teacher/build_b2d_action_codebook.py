# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   B2D 행동 코드북(AutoVLA 방식 0.5 s 조각 (Δx, Δy, Δθ), K-disk ≈ 2048). nuScenes 판 tools/qwen3d_nusc/nusc_action_codebook.py 의
#   알고리즘(조각 분해·자차 박스 윤곽 거리·K-disk 이분탐색·폐루프 토큰화)을 그대로 import 해 쓰고, 바꾸는 것은 데이터와 자차 크기뿐이다:
#     데이터  = B2D train 2 Hz 키프레임(gen_teacher_gt_b2d.py extract 산출, 미래 6 스텝 완전한 표본) — 정답이 토큰화하는 바로 그 표본.
#               val 은 재구성 오차 보고에만 쓴다(코드북 선택에 미사용).
#     자차 크기 = B2D 자차(Lincoln MKZ 2020) info['ego_size'] 길이 4.892 m, 폭 1.837 m (nuScenes 4.084 × 1.85 대신).
#   사용: CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=. python tools/vla_teacher/build_b2d_action_codebook.py
#   산출: results/b2d_prep/b2d_action_codebook.npz (codebook (K,3), delta, val_ade, val_fde, ego_size, n_train_segments)
import argparse
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "qwen3d_nusc"))
sys.path.insert(0, HERE)
import nusc_action_codebook as NAC  # noqa: E402
import common_b2d as B  # noqa: E402

RAW_DIR = "results/b2d_prep/teacher_gt_raw"
OUT = "results/b2d_prep/b2d_action_codebook.npz"


def load(split):
    d = pickle.load(open(f"{RAW_DIR}/{split}.pkl", "rb"))["samples"]
    S, T = [], []
    for s in d:
        if not (np.asarray(s["ego_mask"]) == 1).all():
            continue
        fut = np.asarray(s["ego_fut"], np.float64).cumsum(0)[:6, :2]
        S.append(NAC.segments_from_traj(fut)); T.append(fut)
    return np.concatenate(S), np.stack(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.chdir(REPO)
    if os.path.exists(a.out) and not a.force:
        raise SystemExit(f"[중단] {a.out} 있음(--force)")
    NAC.EGO_L, NAC.EGO_W = B.EGO_L, B.EGO_W        # corners() 가 모듈 전역을 읽는다 — B2D 자차 크기
    t0 = time.time()
    seg_tr, traj_tr = load("train")
    seg_va, traj_va = load("val")
    feat = NAC.corners(seg_tr[:, 0], seg_tr[:, 1], seg_tr[:, 2]).reshape(len(seg_tr), -1)
    print(f"train 조각 {len(seg_tr)} (궤적 {len(traj_tr)}), val 궤적 {len(traj_va)}", flush=True)
    delta, C = NAC.kdisk(feat, a.k)
    idx = [int(np.argmin(np.linalg.norm(feat - c, axis=1))) for c in C]
    book = seg_tr[idx]
    print(f"K-disk δ={delta:.4f} → 코드북 {len(book)} ({time.time() - t0:.0f}s)", flush=True)

    import common as C1

    class CB(C1.ActionCodebook):
        def __init__(self, b):
            self.book = np.asarray(b, np.float64); self.K = len(self.book)
    cb = CB(book)
    errs = np.array([np.linalg.norm(cb.tokenize(f)[1] - f, axis=1) for f in traj_va])
    ade, fde = float(errs.mean()), float(errs[:, -1].mean())
    print(f"val 재구성(3 s): ADE {ade:.4f} m  FDE {fde:.4f} m  (nuScenes 코드북 참고값은 data/qwen3d/nusc_action_codebook_k2048.npz)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, codebook=book, delta=delta, val_ade=ade, val_fde=fde, ego_size=np.array([B.EGO_L, B.EGO_W]),
             n_train_segments=len(seg_tr), n_train_traj=len(traj_tr), n_val_traj=len(traj_va))
    print("->", a.out)


if __name__ == "__main__":
    main()
