"""수정(2026-10-12): 계획 시각화 그림(사용자 지시 "Planning 시각화 이미지를 이번 학습이 끝나면 추가하자").
   대표 모델(hipad_nusc_laq_stage2: 명령 조건 3x16 + 언어 조건 잘린 확산)의 계획 과정을 선행 모델(Run10)과 함께 보인다.

  한 행 = 운전 명령 하나(좌회전 / 직진 / 우회전):
    (a) 전방 카메라: GT 궤적(초록), 선행 모델 계획(청록), 대표 모델 계획(자홍) 투영
    (b) BEV 잘린 확산 입력: 그 명령의 앵커 16개(회색 선) → t=infer_t(설정값, 현재 8) 잡음 앵커(주황) — 평가 때 실제로 디코더에 들어간 값
        (학습된 앵커 = 체크포인트의 plan_instance_bank.anchor, 잡음 = 모델과 같은 고정 시드 CUDA 생성기)
    (c) BEV 디노이징 결과: 16모드(확률이 클수록 진하게) + 대표 모델 최종 계획(자홍) + 선행 모델(청록) + GT(초록) + 예측 지도·박스
  표본 선택(재현 가능한 규칙, 성공 사례 고르기 아님): 명령마다 후보(미래 GT 완비 · 3 s 이동 ≥ 10 m · GT 에이전트 ≥ 5 ·
    회전은 GT 횡변위 ≥ 3 m) 중 대표 모델 계획 L2 가 **중앙값**인 표본. 선행·대표 L2 를 캡션에 함께 적는다.
  사용(2단계 평가 후):
    python tools/visualization/nusc_fig_plan_diffusion.py select
    CUDA_VISIBLE_DEVICES=<gpu> python tools/visualization/nusc_fig_plan_diffusion.py render   # → docs/iv/figures/plan_diffusion.{pdf,jpg}
"""
import argparse, functools, os, pickle, sys
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "tools/visualization")
from nusc_fig_weather_qual import (INFO, lidar2img, densify, box_bev, MAP_COL, in_roi, plan_l2)  # noqa: E402

NEW = "evaluation/hipad_nusc_laq_stage2/results.pkl"
PRIOR = "evaluation/hipad_nusc_vla_s2r9/results.pkl"
CKPT = "results/hipad_nusc_laq_stage2/latest.pth"
CFG = "configs/nuscenes/stage2/test.py"  # 수정(2026-10-15): config 경로 이동
SEL = "evaluation/hipad_nusc_laq_stage2/paper_diag/plan_diffusion_sel.pkl"
OUT = "docs/iv/figures/plan_diffusion"
CMD_NAME = {0: "Turn Right", 1: "Turn Left", 2: "Go Straight"}   # kmeans_nusc_plan_cmd.py 로 실증한 순서
ROW_ORDER = [1, 2, 0]
CAMS = ["CAM_FRONT"]
C_GT, C_PRIOR, C_NEW, C_NOISE = "#2f855a", "#319795", "#d53f8c", "#dd6b20"  # 선행=청록(지도 경계 회색과 구분)


def load_res(path):
    import torch
    _tl = torch.load
    torch.load = functools.partial(_tl, map_location="cpu", weights_only=False)
    try:
        return pickle.load(open(path, "rb"))
    finally:
        torch.load = _tl


def np_(v):
    return v.numpy() if hasattr(v, "numpy") else v


def cmd_select(args):
    infos = pickle.load(open(INFO, "rb"))["infos"]
    new, prior = load_res(args.new), load_res(args.prior)
    assert len(new) == len(prior) == len(infos), "결과 표본 수가 검증 분할과 다르다"
    assert "plan_modes" in new[0]["img_bbox"], f"{args.new} 에 plan_modes 가 없다 — 대표 모델 평가 결과인지 확인"
    cand = {c: [] for c in CMD_NAME}
    for i, info in enumerate(infos):
        if not (np.asarray(info["gt_ego_fut_masks"]) == 1).all():
            continue
        gt = np.asarray(info["gt_ego_fut_trajs"], np.float64).cumsum(0)[:6, :2]
        if np.linalg.norm(gt[-1]) < 10 or int(np.asarray(info["valid_flag"]).sum()) < 5:
            continue
        cmd = int(np.asarray(info["gt_ego_fut_cmd"]).argmax())
        if cmd != 2 and abs(gt[-1, 0]) < 3:
            continue
        cand[cmd].append((plan_l2(np_(new[i]["img_bbox"]["plan_temp_2hz"])[:6, :2], gt), i))
    out = []
    for cmd in ROW_ORDER:
        c = sorted(cand[cmd])
        l2, i = c[len(c) // 2]                                     # 중앙값 표본
        gt = np.asarray(infos[i]["gt_ego_fut_trajs"], np.float64).cumsum(0)[:6, :2]
        l2p = plan_l2(np_(prior[i]["img_bbox"]["plan_temp_2hz"])[:6, :2], gt)
        out.append(dict(cmd=cmd, idx=i, l2_new=float(l2), l2_prior=float(l2p), n_cand=len(c), info=infos[i],
                        new={k: np_(v) for k, v in new[i]["img_bbox"].items()},
                        prior_plan=np_(prior[i]["img_bbox"]["plan_temp_2hz"])[:6, :2]))
        print(f"{CMD_NAME[cmd]:11s} idx {i:4d}  L2 대표 {l2:.3f} / 선행 {l2p:.3f} m  (후보 {len(c)}, 중앙값 표본)")
    os.makedirs(os.path.dirname(args.sel), exist_ok=True)
    pickle.dump(out, open(args.sel, "wb"))
    print("->", args.sel)


def noisy_anchors(args):
    """평가 때 디코더에 들어간 잡음 앵커를 그대로 재현: 체크포인트의 학습된 앵커 + PlanTruncatedDiffusion(eval)."""
    import torch
    from mmcv import Config
    import importlib
    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.models.decoder.noise_t.diffusion import PlanTruncatedDiffusion
    assert torch.cuda.is_available(), "모델 평가와 같은 CUDA 생성기로 잡음을 재현해야 한다 — CUDA_VISIBLE_DEVICES 지정"
    cfg = Config.fromfile(args.cfg).model.head.onedecoder_head.plan_diffusion
    kw = {k: v for k, v in cfg.items() if k != "type"}
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    key = [k for k in sd if k.endswith("plan_instance_bank.anchor")]
    assert len(key) == 1, key
    anchor = sd[key[0]].float().cuda()[None]                       # (1, 48, 12) 누적 궤적
    diff = PlanTruncatedDiffusion(**kw).cuda().eval()
    with torch.no_grad():
        x_t, _ = diff(anchor)
    T = kw["ego_fut_ts"]
    return anchor[0].cpu().numpy().reshape(3, -1, T, 2), x_t[0].cpu().numpy().reshape(3, -1, T, 2)


def front_strip(info, trajs, args):
    from PIL import Image
    d, (c0, c1) = args.down, args.crop
    tw, th = 1600 // d, int(900 * (c1 - c0)) // d
    gap = 4
    M = np.full((th, len(CAMS) * tw + (len(CAMS) - 1) * gap, 3), 255, np.uint8)
    segs = []
    for c, name in enumerate(CAMS):
        cam = info["cams"][name]
        img = Image.open("data/nuscenes/samples/" + cam["data_path"].split("samples/", 1)[1]).convert("RGB")
        img = img.crop((0, int(900 * c0), 1600, int(900 * c0) + int(900 * (c1 - c0)))).resize((tw, th))
        ox = c * (tw + gap)
        M[:, ox:ox + tw] = np.asarray(img)
        for color, traj in trajs:
            p = densify(traj)
            homo = np.concatenate([p, np.full((len(p), 1), -1.8), np.ones((len(p), 1))], 1) @ lidar2img(cam).T
            ok = homo[:, 2] > 1.0
            uv = homo[:, :2] / np.maximum(homo[:, 2:3], 1e-3)
            uv = np.stack([uv[:, 0] / d, (uv[:, 1] - 900 * c0) / d], 1)
            inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < tw) & (uv[:, 1] >= 0) & (uv[:, 1] < th)
            for k in range(len(p) - 1):
                if inside[k] and inside[k + 1]:
                    segs.append((color, uv[k:k + 2] + [ox, 0]))
    return M, segs


def bev_frame(ax, args, title):
    ego = box_bev(np.array([0, 0, 0, 4.084, 1.85, 1.56, np.pi / 2]))
    ax.fill(ego[:, 0], ego[:, 1], color="#2d3748", zorder=6)
    ax.set_xlim(-args.bev_x, args.bev_x); ax.set_ylim(args.bev_y0, args.bev_y1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("#999999")
    ax.set_title(title, fontsize=6.5, loc="left", pad=1.5)


def path(traj):
    return np.concatenate([np.zeros((1, 2)), traj], 0)


def cmd_render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    sel = pickle.load(open(args.sel, "rb"))
    anchors, noisy = noisy_anchors(args)
    n = len(sel)
    probe = front_strip(sel[0]["info"], [], args)[0].shape
    bw, bh = 2 * args.bev_x, args.bev_y1 - args.bev_y0
    fig = plt.figure(figsize=(7.16, args.row_h * n + 0.25))
    gs = GridSpec(n, 3, figure=fig, width_ratios=[probe[1] / probe[0], bw / bh, bw / bh],
                  wspace=0.03, hspace=0.14, left=0.03, right=0.998, top=1 - 0.25 / (args.row_h * n + 0.25), bottom=0.005)
    pcm = plt.get_cmap("Blues")
    for r, s in enumerate(sel):
        info, pred, cmd = s["info"], s["new"], s["cmd"]
        gt = np.asarray(info["gt_ego_fut_trajs"]).cumsum(0)[:6, :2]
        plan, prior = np.asarray(pred["plan_temp_2hz"])[:6, :2], s["prior_plan"]
        # (a) 전방 카메라
        M, segs = front_strip(info, [(C_GT, gt), (C_PRIOR, prior), (C_NEW, plan)], args)
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(M); ax.axis("off")
        for color, sg in segs:
            ax.plot(sg[:, 0], sg[:, 1], color=color, lw=1.6, solid_capstyle="round")
        ax.text(-0.015, 0.5, CMD_NAME[cmd], transform=ax.transAxes, rotation=90, fontsize=7, fontweight="bold",
                va="center", ha="right")
        ax.text(0.01, 0.03, f"L2  prior {s['l2_prior']:.2f} m  →  ours {s['l2_new']:.2f} m", transform=ax.transAxes,
                fontsize=5.5, color="white", va="bottom", bbox=dict(facecolor="black", alpha=0.5, pad=0.8, lw=0))
        # (b) 잘린 확산 입력
        ax = fig.add_subplot(gs[r, 1]); bev_frame(ax, args, f"anchors → noise t={__import__('mmcv').Config.fromfile(args.cfg).model.head.onedecoder_head.plan_diffusion.infer_t}" if r == 0 else "")
        for a_, x_ in zip(anchors[cmd], noisy[cmd]):
            ax.plot(*path(a_).T, color="#a0aec0", lw=0.7, zorder=2)
            ax.plot(*path(x_).T, color=C_NOISE, lw=0.6, alpha=0.75, zorder=3)
            ax.scatter(x_[:, 0], x_[:, 1], s=2.5, color=C_NOISE, zorder=4, lw=0)
        # (c) 디노이징 결과
        ax = fig.add_subplot(gs[r, 2]); bev_frame(ax, args, "denoised modes" if r == 0 else "")
        for v, sc, lab in zip(pred["vectors"], pred["scores"], pred["labels"]):
            if sc >= args.map_thr:
                v = np.asarray(v); ax.plot(v[:, 0], v[:, 1], color=MAP_COL[int(lab)], lw=0.7, alpha=0.6, zorder=1)
        bx, bs = np.asarray(pred["boxes_3d"]), np.asarray(pred["scores_3d"])
        for b in bx[bs >= args.box_thr]:
            if in_roi(b[None, :2], args):
                c = box_bev(b); ax.plot(c[:, 0], c[:, 1], color="#e53e3e", lw=0.5, zorder=2)
        modes, probs = np.asarray(pred["plan_modes"]), np.asarray(pred["plan_mode_probs"])
        for k in np.argsort(probs):
            q = probs[k] / probs.max()
            ax.plot(*path(modes[k]).T, color=pcm(0.25 + 0.75 * q), lw=0.5 + 1.2 * q, alpha=0.4 + 0.6 * q, zorder=3)
        ax.plot(*path(gt).T, color=C_GT, lw=1.6, ls="--", zorder=5)
        ax.plot(*path(prior).T, color=C_PRIOR, lw=1.4, zorder=5)
        ax.plot(*path(plan).T, color=C_NEW, lw=1.8, zorder=6)
        ax.text(0.02, 0.01, f"top mode p={probs.max():.2f}", transform=ax.transAxes, fontsize=5.5, va="bottom")
    handles = [Line2D([], [], color=C_GT, ls="--", lw=1.6, label="GT"),
               Line2D([], [], color=C_PRIOR, lw=1.4, label="prior (Run10)"),
               Line2D([], [], color=C_NEW, lw=1.8, label="ours (cmd-cond. + truncated diffusion)"),
               Line2D([], [], color=C_NOISE, lw=1.0, label="noised anchors"),
               Line2D([], [], color=pcm(0.8), lw=1.2, label="denoised modes")]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.0))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".pdf", dpi=args.dpi); fig.savefig(args.out + ".jpg", dpi=args.dpi, pil_kwargs={"quality": 92})
    print("->", args.out + ".pdf", f"{os.path.getsize(args.out + '.pdf') / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "render"])
    ap.add_argument("--new", default=NEW); ap.add_argument("--prior", default=PRIOR)
    ap.add_argument("--ckpt", default=CKPT); ap.add_argument("--cfg", default=CFG)
    ap.add_argument("--sel", default=SEL); ap.add_argument("--out", default=OUT)
    ap.add_argument("--box-thr", type=float, default=0.35)
    ap.add_argument("--map-thr", type=float, default=0.4)
    ap.add_argument("--bev-x", type=float, default=16.0)
    ap.add_argument("--bev-y0", type=float, default=-6.0)
    ap.add_argument("--bev-y1", type=float, default=36.0)
    ap.add_argument("--crop", type=float, nargs=2, default=[0.30, 0.85])
    ap.add_argument("--row-h", type=float, default=1.5)
    ap.add_argument("--down", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    cmd_select(a) if a.mode == "select" else cmd_render(a)
