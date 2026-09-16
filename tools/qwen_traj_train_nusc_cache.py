#수정: VLA — nuScenes 용 traj_proj 를 "Qwen hidden 캐시" 로 학습 (Qwen live forward 없이, CPU 가능).
#  근거: traj_proj 는 last-token hidden(1536) → (T*2) MLP 뿐. 우리는 data/nuscenes/qwen/<token>.pt 에
#  hidden(L,1536)+mask 를 이미 캐시했고, GT ego 미래궤적은 info["gt_ego_fut_trajs"](offsets)에 있다.
#  → 캐시에서 last-valid-token hidden 추출 + GT(cumsum→위치) 로 작은 MLP 만 학습 → traj_proj.pth.
#  학습 GPU(GPU0)와 충돌 없이 CPU 로 돌릴 수 있다. 저장 포맷은 QwenTrajectoryGenerator.traj_proj 와 호환.
#
#  사용:
#    python tools/qwen_traj_train_nusc_cache.py \
#       --ann-file data/infos/nuscenes_infos_train.pkl \
#       --cache-dir data/nuscenes/qwen --out work_dirs/qwen_traj_nusc/traj_proj.pth
import argparse
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn


def _load_one(arg):
    """캐시 .pt 에서 last-valid-token hidden(1536) + GT 위치(12) + mask(12) 추출."""
    token, cache_dir, gt_traj, gt_mask, n_future = arg
    p = os.path.join(cache_dir, f"{token}.pt")
    if not os.path.exists(p):
        return None
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
        hidden = d["hidden"]            # (L, 1536) bf16
        mask = d["attn_mask"]           # (L,)
        valid_len = int(mask.sum().item()) - 1
        valid_len = max(0, min(valid_len, hidden.shape[0] - 1))
        h_last = hidden[valid_len].float().numpy()   # (1536,)
    except Exception:
        return None
    # GT offsets → cumsum 위치 (현재 ego 기준 lidar frame), (n_future, 2)
    gt = np.asarray(gt_traj, dtype=np.float32).reshape(-1, 2)[:n_future]
    pos = np.cumsum(gt, axis=0).reshape(-1)          # (n_future*2,)
    m = np.asarray(gt_mask, dtype=np.float32).reshape(-1)[:n_future]
    m2 = np.repeat(m, 2)                              # (n_future*2,)
    if pos.shape[0] < n_future * 2:                  # pad
        pad = n_future * 2 - pos.shape[0]
        pos = np.concatenate([pos, np.zeros(pad, np.float32)])
        m2 = np.concatenate([m2, np.zeros(pad, np.float32)])
    return h_last.astype(np.float32), pos, m2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-file", default="data/infos/nuscenes_infos_train.pkl")
    ap.add_argument("--cache-dir", default="data/nuscenes/qwen")
    ap.add_argument("--out", default="work_dirs/qwen_traj_nusc/traj_proj.pth")
    ap.add_argument("--n-future", type=int, default=6)
    ap.add_argument("--embed-dim", type=int, default=1536)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cpu")  #수정: 기본 CPU → 학습용 GPU0 과 충돌 없음
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.ann_file, "rb") as f:
        infos = pickle.load(f)
    infos = infos["infos"] if isinstance(infos, dict) else infos
    if args.max_samples:
        infos = infos[:args.max_samples]
    jobs = [(it["token"], args.cache_dir, it.get("gt_ego_fut_trajs"),
             it.get("gt_ego_fut_masks"), args.n_future)
            for it in infos if it.get("gt_ego_fut_trajs") is not None]
    print(f"samples with GT: {len(jobs)} / {len(infos)} | 캐시에서 last-token hidden 추출(병렬 {args.workers})...")

    t0 = time.time()
    with Pool(args.workers) as pool:
        res = [r for r in pool.map(_load_one, jobs) if r is not None]
    print(f"추출 완료: {len(res)} samples, {time.time()-t0:.0f}s")

    X = torch.tensor(np.stack([r[0] for r in res]))   # (N,1536)
    Y = torch.tensor(np.stack([r[1] for r in res]))   # (N,12)
    M = torch.tensor(np.stack([r[2] for r in res]))   # (N,12)
    dev = args.device
    X, Y, M = X.to(dev), Y.to(dev), M.to(dev)

    # traj_proj — QwenTrajectoryGenerator.traj_proj 와 동일 구조
    traj_proj = nn.Sequential(
        nn.Linear(args.embed_dim, args.embed_dim // 2),
        nn.GELU(),
        nn.Linear(args.embed_dim // 2, args.n_future * 2),
    ).to(dev)
    nn.init.zeros_(traj_proj[-1].weight)
    nn.init.zeros_(traj_proj[-1].bias)
    print(f"trainable traj_proj params: {sum(p.numel() for p in traj_proj.parameters())/1e6:.3f} M")

    optim = torch.optim.AdamW(traj_proj.parameters(), lr=args.lr, weight_decay=1e-3)
    huber = nn.SmoothL1Loss(reduction="none")
    N = X.shape[0]
    for ep in range(args.epochs):
        perm = torch.randperm(N, device=dev)
        tot = 0.0
        for i in range(0, N, args.batch_size):
            idx = perm[i:i + args.batch_size]
            pred = traj_proj(X[idx])
            loss = (huber(pred, Y[idx]) * M[idx]).sum() / M[idx].sum().clamp(min=1)
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(traj_proj.parameters(), 5.0)
            optim.step()
            tot += float(loss) * len(idx)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1}/{args.epochs}  loss {tot/N:.4f}")

    torch.save({"traj_proj": {k: v.cpu() for k, v in traj_proj.state_dict().items()},
                "n_future": args.n_future, "source": "nuscenes_cache"}, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
