"""수정: VLA-v0 — 더미 배치로 SparseDetector forward 1회 호출.
shape/wiring/cross-attn 분기 안전성 검증. 실제 데이터 / GPU 없어도 import + 빌드 검증은 됨.
"""
import argparse
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import mmdet3d_plugin  # registries
from mmcv import Config
from mmdet.models import build_detector


def make_dummy_batch(cfg, device="cpu"):
    bs = 1
    num_cams = cfg.num_cams
    H, W = cfg.input_shape[1], cfg.input_shape[0]  # config 의 (W,H)
    img = torch.randn(bs, num_cams, 3, H, W, device=device)

    # 기본 metas/data
    data = dict(
        img=img,
        timestamp=torch.tensor([0.0], device=device),
        projection_mat=torch.eye(4, device=device).reshape(1, 1, 4, 4).repeat(bs, num_cams, 1, 1),
        image_wh=torch.tensor([[W, H]] * num_cams, device=device).unsqueeze(0).float(),
        target_point=torch.zeros(bs, 2, device=device),
        gt_ego_fut_cmd=torch.zeros(bs, cfg.ego_fut_cmd, device=device),
        ego_status=torch.zeros(bs, 6, device=device),
        ego_status_mask=torch.ones(bs, 6, device=device),
        focal=torch.tensor([100.0] * num_cams, device=device).unsqueeze(0),
    )

    # GT 들 — 빈/제로 텐서로 채워 loss 가 의미는 없지만 forward 는 통과
    data["gt_bboxes_3d"] = [torch.zeros(0, 9, device=device)]
    data["gt_labels_3d"] = [torch.zeros(0, dtype=torch.long, device=device)]
    data["gt_map_labels"] = [torch.zeros(0, dtype=torch.long, device=device)]
    data["gt_map_pts"] = [torch.zeros(0, 20, 2, device=device)]
    data["gt_agent_fut_trajs"] = [torch.zeros(0, cfg.fut_ts, 2, device=device)]
    data["gt_agent_fut_masks"] = [torch.zeros(0, cfg.fut_ts, device=device)]
    for k in ("gt_ego_spat_trajs_2m", "gt_ego_spat_trajs_5m", "gt_ego_fut_trajs_2hz",
             "gt_ego_fut_trajs_5hz"):
        data[k] = torch.zeros(bs, cfg.ego_fut_ts, 2, device=device)
    for k in ("gt_ego_spat_masks_2m", "gt_ego_spat_masks_5m", "gt_ego_fut_masks_2hz",
             "gt_ego_fut_masks_5hz"):
        data[k] = torch.ones(bs, cfg.ego_fut_ts, device=device)

    # depth
    data["gt_depth"] = [torch.zeros(bs, num_cams, H // s, W // s, device=device) for s in cfg.strides[:cfg.num_depth_layers]]

    #수정: VLA-v0 — Qwen 입력 (stub 호환 shape)
    # 실제 processor 출력 형태로 보내려면 image_grid_thw + 패치 flatten 이 필요.
    # 여기선 stub 호환: traj_proj weight device/dtype 만 확인하므로 임의의 numpy 도 OK.
    data["qwen_pixel_values"] = torch.zeros(bs, num_cams, 3, 504, 896, device=device)
    data["qwen_input_ids"] = torch.zeros(bs, 64, dtype=torch.long, device=device)
    data["qwen_attention_mask"] = torch.ones(bs, 64, dtype=torch.long, device=device)
    data["qwen_image_grid_thw"] = torch.tensor([[[1, 36, 64]] * num_cams] * bs,
                                                dtype=torch.long, device=device)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="projects/configs/hipad_b2d_vla_v0.py")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--build-only", action="store_true", help="forward 없이 build 만")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    print(f"[OK] Config: {args.config}")

    detector = build_detector(cfg.model)
    print(f"[OK] build_detector → {type(detector).__name__}")
    print(f"     qwen_traj_generator: stub={detector.qwen_traj_generator._stub if detector.qwen_traj_generator else 'N/A'}")
    # 외부 trajectory 모듈 존재 확인
    head = detector.head.onedecoder_head
    print(f"     with_external_trajectory: {head.with_external_trajectory}")
    if head.with_external_trajectory:
        print(f"     external_trajectory_encoder: {type(head.external_trajectory_encoder).__name__}")
        print(f"     external_trajectory_cross_attn: {type(head.external_trajectory_cross_attn).__name__}")

    if args.build_only:
        return

    detector = detector.to(args.device)
    detector.train()

    data = make_dummy_batch(cfg, device=args.device)
    print(f"[OK] dummy batch on {args.device}; keys = {sorted(data.keys())[:10]} ...")

    # forward — Qwen 가 stub 이면 trajectory = 0, cross-attn 은 그대로 실행되어 wiring 검증됨
    try:
        with torch.no_grad():  # smoke 만, backward 없이
            output = detector.forward_train(data["img"], **{k: v for k, v in data.items() if k != "img"})
        print(f"[OK] forward_train returned: {type(output).__name__}, n_losses = {len(output)}")
        for k in sorted(output.keys())[:8]:
            v = output[k]
            if torch.is_tensor(v):
                print(f"     {k}: tensor(shape={tuple(v.shape)}, dtype={v.dtype}, mean={v.float().mean().item():.4f})")
            else:
                print(f"     {k}: {v}")
        if args.device == "cuda":
            print(f"[MEM] peak GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] forward_train: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
