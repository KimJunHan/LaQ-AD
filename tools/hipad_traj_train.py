"""수정: nuScenes-mini — HiP-AD backbone+neck feature 만으로 ego future trajectory regression.
SparseDetector 의 plan task 가 비활성된 상황에서 '모델이 예측한 주행경로' 를 viz 에 보여주기 위한 경량 head.

구조:
  img (B, 6, 3, H, W) → SparseDetector.extract_feat → feature_maps[List of (B, 6, C_i, H_i, W_i)]
  → 가장 작은 scale (level=-1) 만 사용 → spatial+cam avg pool → (B, 256)
  → traj_head (MLP 256 → 128 → T*2) → (B, T, 2)
  loss = smooth_L1(pred, gt_future_ego_lidar)
  학습 : backbone+neck freeze, traj_head 만.
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmdet3d_plugin  # noqa
from mmcv import Config
from mmcv.utils import build_from_cfg
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint

from mmdet3d_plugin.datasets.nuscenes_mini_dataset import (
    NuScenesMiniDataset, _quat_to_rot, _compose_rt,
)


def build_scene_index(nusc):
    token2scene = {s["token"]: s["scene_token"] for s in nusc.sample}
    scene2tokens = {}
    for scene in nusc.scene:
        toks, st = [], scene["first_sample_token"]
        while st:
            toks.append(st)
            st = nusc.get("sample", st)["next"]
        scene2tokens[scene["token"]] = toks
    return token2scene, scene2tokens


def compute_gt_future_lidar(info, future_infos, n_future=6):
    lidar2ego = _compose_rt(_quat_to_rot(info["lidar2ego_rotation"]),
                            np.asarray(info["lidar2ego_translation"]))
    ego2global = _compose_rt(_quat_to_rot(info["ego2global_rotation"]),
                             np.asarray(info["ego2global_translation"]))
    global2ego = np.linalg.inv(ego2global)
    ego2lidar = np.linalg.inv(lidar2ego)
    out = []
    for fi in future_infos[:n_future]:
        p = np.asarray(fi["ego2global_translation"])
        p_l = ego2lidar @ global2ego @ np.array([p[0], p[1], p[2], 1.0])
        out.append([float(p_l[0]), float(p_l[1])])
    while len(out) < n_future:
        out.append(out[-1] if out else [0.0, 0.0])
    return np.array(out, dtype=np.float32)


class TrajHead(nn.Module):
    def __init__(self, in_dim=256, hidden=128, n_future=6, traj_dim=2):
        super().__init__()
        self.n_future = n_future
        self.traj_dim = traj_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_future * traj_dim),
        )
        # zero init last layer → warm start at zero trajectory
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat):
        # feat: (B, in_dim) — pooled feature
        out = self.net(feat)
        return out.view(-1, self.n_future, self.traj_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/configs/hipad_nusc_mini_stage1.py")
    ap.add_argument("--ckpt", default="work_dirs/hipad_nusc_mini_stage1/iter_50000.pth")
    ap.add_argument("--out", default="work_dirs/qwen_traj/hipad_traj.pth")
    ap.add_argument("--n-future", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cfg = Config.fromfile(args.config)

    # 1) Detector 로드 + ckpt 적용 + freeze
    detector = build_detector(cfg.model)
    load_checkpoint(detector, args.ckpt, map_location="cpu", strict=False)
    detector = detector.to(args.device).eval()
    for p in detector.parameters():
        p.requires_grad_(False)
    print("detector loaded + frozen")

    # 2) NuScenes scene
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-mini", dataroot="data/nuscenes", verbose=False)
    token2scene, scene2tokens = build_scene_index(nusc)

    # 3) dataset (train) — 이미 자료를 ds.get_data_info + pipeline 으로 텐서화
    train_pipeline_cfg = cfg.train_pipeline
    train_data_cfg = dict(cfg.data["train"])
    train_data_cfg["test_mode"] = True  # aug 일관성 위해 test 모드
    train_data_cfg["pipeline"] = cfg.test_pipeline
    ds = build_from_cfg(train_data_cfg, DATASETS)
    print(f"dataset: {len(ds)} samples")

    token2info = {i["token"]: i for i in ds.data_infos}
    # val pkl 도 추가
    import pickle as _pkl
    val_ann = cfg.data["val"]["ann_file"]
    if os.path.exists(val_ann):
        with open(val_ann, "rb") as f:
            d = _pkl.load(f)
        for i in d.get("infos", []):
            token2info.setdefault(i["token"], i)

    # 4) head + optim
    head = TrajHead(in_dim=cfg.embed_dims, hidden=128,
                    n_future=args.n_future, traj_dim=2).to(args.device)
    n_train = sum(p.numel() for p in head.parameters())
    print(f"trainable traj_head params: {n_train/1e6:.4f} M")
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * (args.max_samples or len(ds)))

    # 5) 학습 루프
    losses = []
    step = 0
    t0 = time.time()
    n_max = args.max_samples or len(ds)
    for ep in range(args.epochs):
        order = np.random.permutation(min(len(ds), n_max))
        ep_loss = []
        for idx in order:
            info = ds.data_infos[idx]
            tok = info["token"]
            scene = token2scene.get(tok)
            if not scene:
                continue
            scene_toks = scene2tokens[scene]
            try:
                cur = scene_toks.index(tok)
            except ValueError:
                continue
            future_toks = scene_toks[cur + 1: cur + 1 + args.n_future]
            future_infos = [token2info[t] for t in future_toks if t in token2info]
            if len(future_infos) < 2:
                continue
            gt_traj = compute_gt_future_lidar(info, future_infos, n_future=args.n_future)
            gt_t = torch.from_numpy(gt_traj).to(args.device).unsqueeze(0)

            # dataset pipeline 으로 img 얻기
            data = ds[idx]  # dict with 'img' DataContainer 등
            # mmcv DataContainer 풀기 (test_pipeline 출력)
            img = data["img"]
            if hasattr(img, "data"):
                img = img.data
            if isinstance(img, list):
                img = img[0] if len(img) == 1 else torch.stack(img)
            img = img.to(args.device).unsqueeze(0) if img.dim() == 4 else img.to(args.device)

            with torch.no_grad():
                # backbone+neck 직접 호출 (use_deformable_func 우회)
                bs = img.shape[0]
                if img.dim() == 5:
                    num_cams = img.shape[1]
                    img_flat = img.flatten(end_dim=1)
                else:
                    num_cams = 1
                    img_flat = img
                if detector.use_grid_mask and not detector.training:
                    pass  # eval mode, grid_mask skipped
                fm = detector.img_backbone(img_flat)
                if detector.img_neck is not None:
                    fm = list(detector.img_neck(fm))
                # 가장 작은 scale 선택, (B*num_cams, C, H, W) → spatial+cam avg → (B, C)
                feat = fm[-1]
                feat = feat.mean(dim=(2, 3))                        # (B*num_cams, C)
                feat = feat.view(bs, num_cams, -1).mean(dim=1)      # (B, C)
                feat = feat.float()

            pred = head(feat)   # (1, T, 2)
            loss = F.smooth_l1_loss(pred, gt_t)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optim.step()
            scheduler.step()
            losses.append(loss.item())
            ep_loss.append(loss.item())
            step += 1
            if step % 20 == 0:
                avg = np.mean(losses[-50:])
                el = time.time() - t0
                lr = scheduler.get_last_lr()[0]
                print(f"  ep={ep+1}/{args.epochs} step={step} loss={loss.item():.4f} "
                      f"avg50={avg:.4f} lr={lr:.2e} elapsed={el:.0f}s")
        print(f"=== epoch {ep+1} done, mean={np.mean(ep_loss):.4f}, elapsed={time.time()-t0:.0f}s ===")
        torch.save({"head": head.state_dict(), "in_dim": cfg.embed_dims,
                    "n_future": args.n_future, "traj_dim": 2, "epoch": ep + 1},
                   args.out.replace(".pth", f"_ep{ep+1}.pth"))

    torch.save({"head": head.state_dict(), "in_dim": cfg.embed_dims,
                "n_future": args.n_future, "traj_dim": 2, "epoch": args.epochs},
               args.out)
    print(f"saved: {args.out}  steps={step}  time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
