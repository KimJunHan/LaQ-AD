"""수정: VLA-v0 / nuScenes-mini — Qwen2-VL-2B 의 traj_proj 만 학습하는 경량 트레이너.
구조:
  Qwen2VLForConditionalGeneration (frozen, bf16) — 6-cam + simple prompt → last hidden state (B, H=1536)
  traj_proj (학습 가능, MLP 1536 → 768 → 12) → (B, T=6, 2) waypoints

학습 :
  for sample in nuScenes mini train (323):
    pixel_values, input_ids = processor(6 cams + prompt)
    hidden = qwen(...).hidden_states[-1][:, -1]  # last token
    pred_traj = traj_proj(hidden)
    loss = smooth_L1(pred_traj, gt_future_ego_traj)
    optimizer.step()  # only traj_proj

GT ego trajectory: NuScenes API 의 scene 순서로 미래 6 frame ego_global → 현재 lidar 좌표계로 변환.
저장 : --out 에 traj_proj state_dict 만 (~1 MB).
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
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmdet3d_plugin  # noqa
from mmdet3d_plugin.models.qwen_traj_generator import QwenTrajectoryGenerator
from mmdet3d_plugin.datasets.nuscenes_mini_dataset import (
    NuScenesMiniDataset, _quat_to_rot, _compose_rt,
)
from transformers import AutoProcessor


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
    """현재 lidar 좌표계 기준 미래 ego 위치 (N, 2). info 다음부터 n_future 개."""
    lidar2ego = _compose_rt(_quat_to_rot(info["lidar2ego_rotation"]),
                            np.asarray(info["lidar2ego_translation"]))
    ego2global = _compose_rt(_quat_to_rot(info["ego2global_rotation"]),
                             np.asarray(info["ego2global_translation"]))
    global2ego = np.linalg.inv(ego2global)
    ego2lidar = np.linalg.inv(lidar2ego)
    out = []
    for fi in future_infos[:n_future]:
        p = np.asarray(fi["ego2global_translation"])
        p_homo = np.array([p[0], p[1], p[2], 1.0])
        p_l = ego2lidar @ global2ego @ p_homo
        out.append([float(p_l[0]), float(p_l[1])])
    # n_future 보다 짧으면 마지막 위치 반복
    while len(out) < n_future:
        out.append(out[-1] if out else [0.0, 0.0])
    return np.array(out, dtype=np.float32)   # (n_future, 2)


def load_six_cams_pil(info, data_root):
    imgs = []
    cam_order = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                 "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
    for cam in cam_order:
        c = info["cams"][cam]
        p = c["data_path"]
        if "samples/" in p:
            p = "samples/" + p.split("samples/", 1)[1]
        if not os.path.isabs(p):
            p = os.path.join(data_root, p)
        if os.path.exists(p):
            imgs.append(Image.open(p).convert("RGB"))
        else:
            imgs.append(Image.new("RGB", (1600, 900), (0, 0, 0)))
    return imgs


def build_prompt(processor, imgs, instruction):
    msgs = [
        {"role": "user", "content":
            [{"type": "image", "image": im} for im in imgs] +
            [{"type": "text", "text": instruction}],
        },
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-path", default="/workspace/src/HiP-AD/ckpts/qwen2-vl-2b")
    ap.add_argument("--data-root", default="data/nuscenes")
    ap.add_argument("--ann-file", default="data/nuscenes/nuscenes_infos_train.pkl")
    ap.add_argument("--out", default="work_dirs/qwen_traj/traj_proj.pth")
    ap.add_argument("--n-future", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-pixels", type=int, default=200 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=200 * 28 * 28)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) NuScenes scene index
    from nuscenes.nuscenes import NuScenes
    print("Loading NuScenes v1.0-mini ...")
    nusc = NuScenes(version="v1.0-mini", dataroot=args.data_root, verbose=False)
    token2scene, scene2tokens = build_scene_index(nusc)

    # 2) dataset (infos + 자동 path 정규화)
    ds = NuScenesMiniDataset(data_root=args.data_root, ann_file=args.ann_file,
                              pipeline=None, modality=dict(use_camera=True, use_lidar=False))
    print(f"dataset: {len(ds)} samples, scenes={len(scene2tokens)}")

    # 모든 token→info (val pkl 까지 합쳐 future lookup)
    token2info = {i["token"]: i for i in ds.data_infos}
    for vp in (args.ann_file.replace("train", "val"),):
        if os.path.exists(vp):
            import pickle
            with open(vp, "rb") as f:
                d = pickle.load(f)
            for i in d.get("infos", []):
                token2info.setdefault(i["token"], i)
    print(f"token2info: {len(token2info)}")

    # 3) Qwen generator (frozen) + processor
    print(f"Loading Qwen from {args.qwen_path} ...")
    qg = QwenTrajectoryGenerator(
        qwen_model_path=args.qwen_path, embed_dim=1536,
        traj_ts=args.n_future, traj_dim=2, freeze_qwen=True,
        torch_dtype="bf16", use_lora=False, gradient_checkpointing=False,
    )
    assert not qg._stub, "Qwen 로드 실패"
    qg = qg.to(args.device)
    qg.qwen.eval()
    # traj_proj 만 학습
    for p in qg.qwen.parameters():
        p.requires_grad_(False)
    for p in qg.traj_proj.parameters():
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in qg.traj_proj.parameters())
    print(f"trainable traj_proj params: {n_train/1e6:.3f} M")

    processor = AutoProcessor.from_pretrained(
        args.qwen_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels)

    optim = torch.optim.AdamW(qg.traj_proj.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs * (args.max_samples or len(ds)))

    INSTRUCTION = (
        "These are 6 surround-view camera images of an ego vehicle (front, front-left, "
        "front-right, back-left, back, back-right). Predict the ego vehicle's future "
        "trajectory for the next 3 seconds as 6 waypoints (0.5s spacing) in lidar frame meters."
    )

    # 4) 학습 루프
    losses = []
    step = 0
    t0 = time.time()
    n_max = args.max_samples or len(ds)
    for ep in range(args.epochs):
        order = np.random.permutation(min(len(ds), n_max))
        ep_loss = []
        for k, idx in enumerate(order):
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
                continue   # 미래 frame 부족 → skip
            gt_traj = compute_gt_future_lidar(info, future_infos, n_future=args.n_future)
            gt_traj_t = torch.from_numpy(gt_traj).to(args.device).unsqueeze(0)  # (1, T, 2)

            imgs = load_six_cams_pil(info, args.data_root)
            inputs = build_prompt(processor, imgs, INSTRUCTION)
            inputs = {k_: v.to(args.device) for k_, v in inputs.items()}

            pred_traj = qg(qwen_pixel_values=inputs["pixel_values"],
                           qwen_input_ids=inputs["input_ids"],
                           qwen_attention_mask=inputs["attention_mask"],
                           qwen_image_grid_thw=inputs.get("image_grid_thw"))
            loss = F.smooth_l1_loss(pred_traj, gt_traj_t)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qg.traj_proj.parameters(), 5.0)
            optim.step()
            scheduler.step()
            losses.append(loss.item())
            ep_loss.append(loss.item())
            step += 1
            if step % 10 == 0:
                avg = np.mean(losses[-50:])
                el = time.time() - t0
                lr = scheduler.get_last_lr()[0]
                print(f"  ep={ep+1}/{args.epochs} step={step} idx={idx} loss={loss.item():.4f} "
                      f"avg50={avg:.4f} lr={lr:.2e} elapsed={el:.0f}s")
        print(f"=== epoch {ep+1} done, mean loss={np.mean(ep_loss):.4f}, elapsed={time.time()-t0:.0f}s ===")
        # epoch-end ckpt
        torch.save({"traj_proj": qg.traj_proj.state_dict(),
                    "n_future": args.n_future, "traj_dim": 2,
                    "epoch": ep + 1, "step": step},
                   args.out.replace(".pth", f"_ep{ep+1}.pth"))

    # final
    torch.save({"traj_proj": qg.traj_proj.state_dict(),
                "n_future": args.n_future, "traj_dim": 2,
                "epoch": args.epochs, "step": step},
               args.out)
    print(f"saved final: {args.out}  | total steps {step}  | total time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
