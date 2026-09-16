"""수정: VLA-v0 / Bench2Drive — Qwen2-VL-2B 의 traj_proj 만 학습하는 경량 트레이너 (B2D판).
nuScenes 판(tools/qwen_traj_train.py)을 Bench2Drive 로 미러링:
  - NuScenes scene(next 포인터) 룩업 대신 B2D 의 folder + frame_idx 그룹핑으로 미래 frame 조회.
  - GT ego 미래 궤적: LIDAR_TOP world2lidar 로 미래 ego world 위치 → 현재 lidar 좌표계 변환.
  - 6-cam: info['sensors'][cam]['data_path'] (data_root 기준 상대경로).
구조/손실/저장은 nuScenes 판과 동일 (Qwen frozen, traj_proj 만 학습, smooth_L1, traj_proj state_dict 저장).
B2D 10Hz → stride=5(0.5s) × n_future=6 = 3s 시평 (INSTRUCTION 과 일치).
"""
import argparse
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmdet3d_plugin  # noqa
from mmdet3d_plugin.models.qwen_traj_generator import QwenTrajectoryGenerator
from transformers import AutoProcessor

CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
             "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
INSTRUCTION = (
    "These are 6 surround-view camera images of an ego vehicle (front, front-left, "
    "front-right, back-left, back, back-right). Predict the ego vehicle's future "
    "trajectory for the next 3 seconds as 6 waypoints (0.5s spacing) in lidar frame meters."
)


def compute_gt_future_lidar(cur, future_infos, n_future=6):
    """현재 frame lidar 좌표계 기준 미래 ego 위치 (n_future, 2)."""
    w2l_cur = np.asarray(cur["sensors"]["LIDAR_TOP"]["world2lidar"])
    out = []
    for fi in future_infos[:n_future]:
        l2w = np.linalg.inv(np.asarray(fi["sensors"]["LIDAR_TOP"]["world2lidar"]))
        p_world = l2w[:3, 3]
        p_l = w2l_cur @ np.array([p_world[0], p_world[1], p_world[2], 1.0])
        out.append([float(p_l[0]), float(p_l[1])])
    while len(out) < n_future:
        out.append(out[-1] if out else [0.0, 0.0])
    return np.array(out, dtype=np.float32)


def load_six_cams_pil(info, data_root):
    imgs = []
    for cam in CAM_ORDER:
        p = info["sensors"][cam]["data_path"]
        if not os.path.isabs(p):
            p = os.path.join(data_root, p)
        imgs.append(Image.open(p).convert("RGB") if os.path.exists(p)
                    else Image.new("RGB", (1600, 900), (0, 0, 0)))
    return imgs


def build_prompt(processor, imgs, instruction):
    msgs = [{"role": "user", "content":
             [{"type": "image", "image": im} for im in imgs] +
             [{"type": "text", "text": instruction}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=imgs, return_tensors="pt", padding=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-path", default="/workspace/src/HiP-AD/ckpts/qwen2-vl-2b")
    ap.add_argument("--data-root", default="data/bench2drive")
    ap.add_argument("--ann-file", default="data/infos/b2d_infos_train.pkl")
    ap.add_argument("--out", default="work_dirs/qwen_traj_b2d/traj_proj.pth")
    ap.add_argument("--n-future", type=int, default=6)
    ap.add_argument("--stride", type=int, default=5, help="B2D 10Hz → 5 = 0.5s 간격")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-pixels", type=int, default=200 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=200 * 28 * 28)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) infos 로드 + folder 그룹핑 (frame_idx 순)
    with open(args.ann_file, "rb") as f:
        d = pickle.load(f)
    infos = d["infos"] if isinstance(d, dict) else d
    by_folder = defaultdict(dict)
    for it in infos:
        by_folder[str(it["folder"])][int(it["frame_idx"])] = it
    print(f"infos: {len(infos)} | scenes: {len(by_folder)}")

    # 미래 frame 이 충분한 sample 만 학습 대상
    samples = []
    for folder, fmap in by_folder.items():
        for fidx, info in fmap.items():
            fut = [fmap[fidx + k * args.stride] for k in range(1, args.n_future + 1)
                   if (fidx + k * args.stride) in fmap]
            if len(fut) >= 2:
                samples.append((info, fut))
    print(f"trainable samples (충분한 미래): {len(samples)}")

    # 2) Qwen (frozen) + traj_proj
    print(f"Loading Qwen from {args.qwen_path} ...")
    qg = QwenTrajectoryGenerator(
        qwen_model_path=args.qwen_path, embed_dim=1536,
        traj_ts=args.n_future, traj_dim=2, freeze_qwen=True,
        torch_dtype="bf16", use_lora=False, gradient_checkpointing=False)
    assert not qg._stub, "Qwen 로드 실패"
    qg = qg.to(args.device); qg.qwen.eval()
    for p in qg.qwen.parameters():
        p.requires_grad_(False)
    for p in qg.traj_proj.parameters():
        p.requires_grad_(True)
    print(f"trainable traj_proj: {sum(p.numel() for p in qg.traj_proj.parameters())/1e6:.3f} M")

    processor = AutoProcessor.from_pretrained(
        args.qwen_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    optim = torch.optim.AdamW(qg.traj_proj.parameters(), lr=args.lr, weight_decay=1e-3)
    n_max = args.max_samples or len(samples)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs * n_max)

    # 3) 학습 루프
    losses, step, t0 = [], 0, time.time()
    for ep in range(args.epochs):
        order = np.random.permutation(min(len(samples), n_max))
        ep_loss = []
        for idx in order:
            info, fut = samples[idx]
            gt = compute_gt_future_lidar(info, fut, n_future=args.n_future)
            gt_t = torch.from_numpy(gt).to(args.device).unsqueeze(0)  # (1,T,2)
            imgs = load_six_cams_pil(info, args.data_root)
            inp = build_prompt(processor, imgs, INSTRUCTION)
            inp = {k_: v.to(args.device) for k_, v in inp.items()}
            pred = qg(qwen_pixel_values=inp["pixel_values"],
                      qwen_input_ids=inp["input_ids"],
                      qwen_attention_mask=inp["attention_mask"],
                      qwen_image_grid_thw=inp.get("image_grid_thw"))
            loss = F.smooth_l1_loss(pred, gt_t)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(qg.traj_proj.parameters(), 5.0)
            optim.step(); scheduler.step()
            losses.append(loss.item()); ep_loss.append(loss.item()); step += 1
            if step % 10 == 0:
                print(f"  ep={ep+1}/{args.epochs} step={step} loss={loss.item():.4f} "
                      f"avg50={np.mean(losses[-50:]):.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time()-t0:.0f}s")
        print(f"=== epoch {ep+1} done, mean loss={np.mean(ep_loss):.4f}, elapsed={time.time()-t0:.0f}s ===")
        torch.save({"traj_proj": qg.traj_proj.state_dict(), "n_future": args.n_future,
                    "traj_dim": 2, "epoch": ep + 1, "step": step},
                   args.out.replace(".pth", f"_ep{ep+1}.pth"))

    torch.save({"traj_proj": qg.traj_proj.state_dict(), "n_future": args.n_future,
                "traj_dim": 2, "epoch": args.epochs, "step": step}, args.out)
    print(f"saved final: {args.out} | steps {step} | time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
