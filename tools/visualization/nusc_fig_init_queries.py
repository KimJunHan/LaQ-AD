"""수정(2026-10-15): 발표 자료용 — 디코딩 전 "초기 쿼리(앵커)"가 무엇인지 시각화한다(사용자 지시). CPU 전용.

  쿼리 = (특징 F ∈ ℝ^256, 앵커 A). 앵커는 BEV 기하, 특징은 내용.
    검출 900: A = data/kmeans/nusc_det_900.npy (900×11 = [x,y,z, ln w, ln l, ln h, sin, cos, vx, vy, vz], core/box3d.py)
              k-평균은 학습 분할 GT 박스 중심(x,y,z)만 군집, 나머지는 상수 [1,1,1,1,0,0,0,0] (tools/kmeans/kmeans_nusc.py)
              → 학습 중 앵커도 파라미터(anchor_grad=True)라 움직인다: 1단계 최종 ckpt 의 det_instance_bank.anchor 와 비교.
    지도 100 : A = nusc_map_100.npy (100×20×2) — 학습 polyline 중심 k-평균 + 길이 U(8,20) m 직선(70% 전방 방향).
    계획 48  : A = nusc_plan_temp_3x16.npy (3 명령 × 16 모드 × 6 점 × 2) — 명령별 자차 3 s 궤적 k-평균.
    자차 1   : 고정 자차 상자(ego/instance_bank.py nus_ego_anchor).
  좌표계: nuScenes 라이다 좌표(x 오른쪽, y 전방). 그림은 전방이 위.
  numpy LAPACK(svd/eigh) 은 이 환경에서 깨져 있어 쓰지 않는다(필요 없음).
  사용: PYTHONPATH=. /home1/irteam/data-vol1/anaconda3/envs/hipad_bw/bin/python tools/visualization/nusc_fig_init_queries.py
  출력: docs/ppt/figures/init_queries.png, init_queries.json(그림·슬라이드 수치)
"""
import argparse, json, os
import numpy as np

DET = "data/kmeans/nusc_det_900.npy"
MAP = "data/kmeans/nusc_map_100.npy"
PLAN = "data/kmeans/nusc_plan_temp_3x16.npy"
CKPT = "results/hipad_nusc_laq_stage1/iter_175800.pth"
OUT = "docs/ppt/figures/init_queries"
P = "head.onedecoder_head."


def footprint(a):
    """SparseBox3DKeyPointsGenerator 와 같은 회전: 국소 (±w/2, ±l/2) → R(sin,cos) → +중심. a: (N,11)."""
    w, l = np.exp(a[:, 3]), np.exp(a[:, 4])
    s, c = a[:, 6], a[:, 7]
    lx = np.array([1, 1, -1, -1, 1]) * 0.5
    ly = np.array([1, -1, -1, 1, 1]) * 0.5
    px, py = lx[None] * w[:, None], ly[None] * l[:, None]
    return np.stack([c[:, None] * px - s[:, None] * py + a[:, :1], s[:, None] * px + c[:, None] * py + a[:, 1:2]], -1)


def main(args):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.collections import LineCollection
    fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fm.fontManager.addfont(fp)
    plt.rcParams.update({"font.family": fm.FontProperties(fname=fp).get_name(), "axes.unicode_minus": False, "font.size": 9})

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    d0 = np.load(DET); d1 = sd[P + "det_instance_bank.anchor"].double().numpy()
    m0 = np.load(MAP); m1 = sd[P + "map_instance_bank.anchor"].double().numpy().reshape(m0.shape)
    p0 = np.load(PLAN); p1 = sd[P + "plan_instance_bank.anchor"].double().numpy().reshape(p0.shape)
    ego = sd[P + "ego_instance_bank.anchor"].double().numpy()
    fdet = sd[P + "det_instance_bank.instance_feature"]; fmap = sd[P + "map_instance_bank.instance_feature"]
    assert d0.shape == (900, 11) and m0.shape == (100, 20, 2) and p0.shape == (3, 16, 6, 2)

    mv = np.linalg.norm(d1[:, :2] - d0[:, :2], axis=1)
    sz1 = np.exp(d1[:, 3:6])
    yaw1 = np.degrees(np.arctan2(d1[:, 6], d1[:, 7]))
    mmv = np.linalg.norm(m1 - m0, axis=-1)
    st = dict(
        ckpt=args.ckpt,
        det=dict(n=900, init_size=float(np.exp(d0[0, 3])), init_size_const=bool(np.allclose(d0[:, 3:6], d0[0, 3])),
                 init_yaw_deg=float(np.degrees(np.arctan2(d0[0, 6], d0[0, 7]))), init_vel0=bool(np.all(d0[:, 8:] == 0)),
                 r_max=float(np.linalg.norm(d0[:, :2], axis=1).max()), n_within30=int((np.linalg.norm(d0[:, :2], axis=1) < 30).sum()),
                 z_init_mean=float(d0[:, 2].mean()),
                 move_mean=float(mv.mean()), move_med=float(np.median(mv)), move_max=float(mv.max()),
                 size_min=sz1.min(0).tolist(), size_max=sz1.max(0).tolist(), size_mean=sz1.mean(0).tolist(),
                 yaw_mean=float(yaw1.mean()), yaw_std=float(yaw1.std()), vel_abs_max=float(np.abs(d1[:, 8:]).max()),
                 feat_abs_max=float(fdet.abs().max())),
        map=dict(n=100, pts=20, move_mean=float(mmv.mean()), move_max=float(mmv.max()),
                 feat_norm_mean=float(fmap.norm(dim=1).mean()), feat_abs_max=float(fmap.abs().max())),
        plan=dict(n=48, cmd=3, modes=16, ts=6, end_mean=p0[:, :, -1].mean(1).round(2).tolist(),
                  move_max=float(np.abs(p1 - p0).max())),
        ego=dict(anchor=ego[0].round(4).tolist(), size=np.exp(ego[0, 3:6]).round(2).tolist()),
    )

    fig, ax = plt.subplots(1, 3, figsize=(13.0, 5.0), gridspec_kw=dict(width_ratios=[1.0, 0.62, 0.62], wspace=0.18))
    # (a) 검출 앵커
    a = ax[0]
    fp1 = footprint(d1)
    a.add_collection(LineCollection(fp1, colors="#2F5597", linewidths=0.45))
    a.scatter(d0[:, 0], d0[:, 1], s=1.2, c="#C00000", lw=0, zorder=3)
    a.add_patch(plt.Rectangle((-0.865, -2.04 + 0.5), 1.73, 4.08, fc="#7F7F7F", ec="k", lw=0.5, zorder=4))
    a.set_xlim(-56, 56); a.set_ylim(-56, 56); a.set_aspect("equal")
    for r in (10, 30, 50):
        a.add_patch(plt.Circle((0, 0), r, fill=False, ls=":", lw=0.5, color="0.6"))
        a.text(r * 0.707 + 0.5, r * 0.707 + 0.5, f"{r} m", fontsize=7, color="0.45")
    a.set_title("(a) 검출 앵커 900개 — 빨강: k-평균 초기 중심, 파랑: 학습 후 앵커 상자(실제 크기)", fontsize=9)
    a.set_xlabel("x (m, 오른쪽)"); a.set_ylabel("y (m, 전방)")
    # 확대 삽입: 이동을 ×10 으로 과장한 화살표
    ins = a.inset_axes([0.64, 0.02, 0.34, 0.34])
    box = (-8, 8, 2, 18)
    k = (d0[:, 0] > box[0]) & (d0[:, 0] < box[1]) & (d0[:, 1] > box[2]) & (d0[:, 1] < box[3])
    ins.add_collection(LineCollection(fp1[k], colors="#2F5597", linewidths=0.6))
    ins.scatter(d0[k, 0], d0[k, 1], s=5, c="#C00000", lw=0, zorder=3)
    ins.quiver(d0[k, 0], d0[k, 1], (d1[k, 0] - d0[k, 0]) * 10, (d1[k, 1] - d0[k, 1]) * 10, angles="xy", scale_units="xy",
               scale=1, color="#C55A11", width=0.012, zorder=4)
    ins.set_xlim(box[0], box[1]); ins.set_ylim(box[2], box[3]); ins.set_aspect("equal")
    ins.set_xticks([]); ins.set_yticks([])
    ins.text(0.03, 0.97, "확대: 이동 ×10 (주황)", transform=ins.transAxes, fontsize=7, va="top",
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    a.indicate_inset_zoom(ins, edgecolor="0.4", lw=0.6)

    # (b) 지도 앵커
    a = ax[1]
    for k_ in range(100):
        a.plot(m0[k_, :, 0], m0[k_, :, 1], color="0.7", lw=1.6, solid_capstyle="round")
        a.plot(m1[k_, :, 0], m1[k_, :, 1], color="#548235", lw=0.8)
        a.plot(m1[k_, ::19, 0], m1[k_, ::19, 1], "o", color="#548235", ms=1.6)
    a.add_patch(plt.Rectangle((-15, -30), 30, 60, fill=False, ls="--", lw=0.8, ec="#C00000"))
    a.add_patch(plt.Rectangle((-0.865, -1.54), 1.73, 4.08, fc="#7F7F7F", ec="k", lw=0.5, zorder=4))
    a.set_xlim(-19, 19); a.set_ylim(-36, 36); a.set_aspect("equal")
    a.set_title("(b) 지도 앵커 100개 × 20점\n회색: 초기, 초록: 학습 후, 빨강 점선: ROI 30×60 m", fontsize=9)
    a.set_xlabel("x (m)")

    # (c) 계획 앵커
    a = ax[2]
    cols = ["#C00000", "#2F5597", "#548235"]
    #   이름은 종점 평균 x 로 정한다(라이다 x = 오른쪽): 0 → +4.6 m(우), 1 → −4.8 m(좌), 2 → +0.1 m(직진)
    ex = p0[:, :, -1, 0].mean(1)
    names = [f"명령 {c} ({'우' if ex[c] > 1 else '좌' if ex[c] < -1 else '직진'}, 종점 x {ex[c]:+.1f} m)" for c in range(3)]
    for c in range(3):
        for m in range(16):
            tr = np.concatenate([[[0, 0]], p1[c, m]], 0)
            a.plot(tr[:, 0], tr[:, 1], "-", color=cols[c], lw=0.9, alpha=0.85, label=names[c] if m == 0 else None)
            a.plot(tr[1:, 0], tr[1:, 1], ".", color=cols[c], ms=2.2)
    a.add_patch(plt.Rectangle((-0.865, -1.54), 1.73, 4.08, fc="#7F7F7F", ec="k", lw=0.5, zorder=4))
    a.set_xlim(-14, 14); a.set_ylim(-4, 52); a.set_aspect("equal")
    a.legend(loc="upper left", fontsize=7, framealpha=0.9, edgecolor="none", handlelength=1.2)
    a.set_title("(c) 계획 앵커 48개 = 3명령 × 16모드\n6점(0.5 s 간격, 3 s) · 회색 상자: 자차 쿼리 1개", fontsize=9)
    a.set_xlabel("x (m)")
    for a in ax:
        a.tick_params(labelsize=7)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
    json.dump(st, open(OUT + ".json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(st, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--dpi", type=int, default=200)
    main(ap.parse_args())
