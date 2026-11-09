"""수정(2026-10-10): 논문용 날씨·조명별 정성 결과 그림(nuScenes val) — 사용자 지시
   ("archive/cleanup_20261026/top/hipad_nusce_picture.png 를 참고해 우리 모델의 예측 경로를 낮·비·밤별로 정성 평가하는 페이지").

  형식(HiP-AD Fig.2 와 같음): 한 행 = [6캠 2×3 (예측 계획 투영)] | [예측 BEV] | [GT BEV], 행 아래에 운전 명령.
    예측 BEV: 검출 박스(점수≥--box-thr), 지도 벡터(점수≥--map-thr), 모션 최고 모드 궤적, 자차 계획.
    GT BEV  : GT 박스·GT 지도·GT 에이전트 미래 궤적·GT 자차 궤적.
  날씨 분할 규칙은 tools/eval_nusc_breakdown.py 와 같다(장면 설명 소문자 'night'→Night, 아니면 'rain'→Rain, 그 외 Day).
  표본 선택(재현 가능하게 규칙으로 고정): 날씨마다 회전 명령 1개 + 직진 1개.
    후보 = 미래 GT 완비 · 3 s 이동 ≥ 10 m · GT 에이전트 ≥ 5 · 회전 후보는 GT 횡변위 |x| ≥ 3 m.
    후보 중 계획 L2(1/2/3 s 누적 평균의 평균)가 가장 작은 표본. 같은 장면에서 두 개를 뽑지 않는다.
  좌표: nuScenes lidar 프레임(+x 우측, +y 전방). 박스 = (x,y,z,l,w,h,yaw), yaw 는 +x 기준.
  사용:
    python tools/visualization/nusc_fig_weather_qual.py select   # results.pkl(2 GB) → 선택 표본만 작은 pkl 로
    python tools/visualization/nusc_fig_weather_qual.py render   # → docs/iv/figures/qual_weather.{pdf,jpg}
"""
import argparse, json, os, pickle, sys
import numpy as np

sys.path.insert(0, ".")
RES = "evaluation/hipad_nusc_vla_s2r9/results.pkl"
INFO = "data/infos/nuscenes_infos_val.pkl"
SCENE_JSON = "data/nuscenes/v1.0-trainval/scene.json"
SEL = "evaluation/hipad_nusc_vla_s2r9/paper_diag/qual_weather_sel.pkl"
OUT = "docs/iv/figures/qual_weather"
CMD_NAME = {0: "Turn Right", 1: "Turn Left", 2: "Go Straight"}   # kmeans_nusc_plan_cmd.py 로 실증한 순서
CAMS_TOP = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
CAMS_BOT = ["CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]
WEATHERS = ["Day", "Rain", "Night"]


def weather_of(info, desc):
    d = desc[info["scene_token"]].lower()
    return "Night" if "night" in d else ("Rain" if "rain" in d else "Day")


def plan_l2(pred, gt):
    e = np.linalg.norm(pred - gt, axis=-1)
    cum = [e[:j + 1].mean() for j in range(6)]
    return (cum[1] + cum[3] + cum[5]) / 3


def cmd_select(args):
    import torch, functools
    infos = pickle.load(open(INFO, "rb"))["infos"]
    desc = {s["token"]: s["description"] for s in json.load(open(SCENE_JSON))}
    _tl = torch.load
    torch.load = functools.partial(_tl, map_location="cpu", weights_only=False)
    try:
        res = pickle.load(open(RES, "rb"))
    finally:
        torch.load = _tl
    assert len(res) == len(infos)
    cand = {w: {"turn": [], "straight": []} for w in WEATHERS}
    for i, info in enumerate(infos):
        if not (np.asarray(info["gt_ego_fut_masks"]) == 1).all():
            continue
        gt = np.asarray(info["gt_ego_fut_trajs"], np.float64).cumsum(0)[:6, :2]
        if np.linalg.norm(gt[-1]) < 10 or int(np.asarray(info["valid_flag"]).sum()) < 5:
            continue
        pred = res[i]["img_bbox"]["plan_temp_2hz"].numpy()[:6, :2]
        cmd = int(np.asarray(info["gt_ego_fut_cmd"]).argmax())
        kind = "straight" if cmd == 2 else "turn"
        if kind == "turn" and abs(gt[-1, 0]) < 3:
            continue
        cand[weather_of(info, desc)][kind].append((plan_l2(pred, gt), i))
    picks = []
    for w in WEATHERS:
        used = set()
        for kind in ("turn", "straight"):
            for l2, i in sorted(cand[w][kind]):
                if infos[i]["scene_token"] in used:
                    continue
                used.add(infos[i]["scene_token"]); picks.append((w, kind, i, l2)); break
    out = []
    for w, kind, i, l2 in picks:
        r = res[i]["img_bbox"]
        out.append(dict(weather=w, kind=kind, idx=i, l2=float(l2), n_cand=len(cand[w][kind]),
                        info=infos[i], pred={k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in r.items()}))
        print(f"{w:5s} {kind:8s} idx {i:4d}  L2 {l2:.3f} m  cmd {CMD_NAME[int(np.argmax(infos[i]['gt_ego_fut_cmd']))]}"
              f"  (후보 {len(cand[w][kind])})  '{desc[infos[i]['scene_token']]}'")
    pickle.dump(out, open(SEL, "wb"))
    print("->", SEL)


# ---------------------------------------------------------------- render
def _rt(q, t):
    from pyquaternion import Quaternion
    T = np.eye(4); T[:3, :3] = Quaternion(q).rotation_matrix; T[:3, 3] = t
    return T


def lidar2img(cam):
    c2l = np.eye(4); c2l[:3, :3] = np.asarray(cam["sensor2lidar_rotation"]); c2l[:3, 3] = cam["sensor2lidar_translation"]
    K = np.eye(4); K[:3, :3] = np.asarray(cam["cam_intrinsic"])
    return K @ np.linalg.inv(c2l)


def box_bev(b):
    x, y, l, w, yaw = b[0], b[1], b[3], b[4], b[6]
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2], [l / 2, w / 2]])
    return loc @ np.array([[c, s], [-s, c]]) + [x, y]


def densify(traj, n=8, origin=True):
    pts = np.concatenate([np.zeros((1, 2)), traj], 0) if origin else np.asarray(traj)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, n + 1)[1:]:
            out.append(a + (b - a) * t)
    return np.array(out)


def mosaic(info, plan, args):
    """2×3 카메라 모자이크(하늘·보닛 상하 크롭) + 앞줄 3캠에 예측 계획 투영 선분 목록(모자이크 좌표)."""
    from PIL import Image
    d, (c0, c1) = args.down, args.crop
    tw, th = 1600 // d, int(900 * (c1 - c0)) // d
    gap = 6
    M = np.full((2 * th + gap, 3 * tw + 2 * gap, 3), 255, np.uint8)
    segs = []
    p = densify(plan)
    for row, names in enumerate((CAMS_TOP, CAMS_BOT)):
        for c, name in enumerate(names):
            cam = info["cams"][name]
            path = "data/nuscenes/samples/" + cam["data_path"].split("samples/", 1)[1]
            img = Image.open(path).convert("RGB")
            img = img.crop((0, int(900 * c0), 1600, int(900 * c0) + int(900 * (c1 - c0)))).resize((tw, th))
            ox, oy = c * (tw + gap), row * (th + gap)
            M[oy:oy + th, ox:ox + tw] = np.asarray(img)
            if row:
                continue
            homo = np.concatenate([p, np.full((len(p), 1), -1.8), np.ones((len(p), 1))], 1) @ lidar2img(cam).T
            ok = homo[:, 2] > 1.0
            uv = homo[:, :2] / np.maximum(homo[:, 2:3], 1e-3)
            uv = np.stack([uv[:, 0] / d, (uv[:, 1] - 900 * c0) / d], 1)
            inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < tw) & (uv[:, 1] >= 0) & (uv[:, 1] < th)
            for k in range(len(p) - 1):
                if inside[k] and inside[k + 1]:
                    segs.append((k / (len(p) - 1), uv[k:k + 2] + [ox, oy]))
    return M, segs, (tw, th, gap)


def draw_grad(ax, traj, cmap, lw=2.0, z=3, origin=True):
    import matplotlib.pyplot as plt
    p = densify(traj, origin=origin)
    cols = plt.get_cmap(cmap)(np.linspace(0.15, 1, len(p)))
    for k in range(len(p) - 1):
        ax.plot(p[k:k + 2, 0], p[k:k + 2, 1], color=cols[k], lw=lw, zorder=z, solid_capstyle="round")


MAP_COL = {0: "#2b6cb0", 1: "#d69e2e", 2: "#4a5568"}   # ped_crossing, divider, boundary


def draw_bev(ax, boxes, maps, agent_trajs, ego_traj, title, args):
    for lab, pts in maps:
        ax.plot(pts[:, 0], pts[:, 1], color=MAP_COL[int(lab)], lw=0.9, zorder=1)
    for t in agent_trajs:
        draw_grad(ax, t, "autumn_r", lw=1.0, z=2, origin=False)   # 에이전트 궤적은 박스 중심에서 시작
    for b in boxes:
        c = box_bev(b); ax.plot(c[:, 0], c[:, 1], color="#e53e3e", lw=0.7, zorder=3)
    ego = box_bev(np.array([0, 0, 0, 4.084, 1.85, 1.56, np.pi / 2]))
    ax.fill(ego[:, 0], ego[:, 1], color="#38a169", zorder=4)
    draw_grad(ax, ego_traj, "cool", lw=2.4, z=5)
    ax.set_xlim(-args.bev_x, args.bev_x); ax.set_ylim(args.bev_y0, args.bev_y1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("#999999")
    ax.set_title(title, fontsize=7, loc="left", pad=1.5)


def in_roi(p, args):
    return bool(((np.abs(p[:, 0]) <= args.bev_x + 5) & (p[:, 1] >= args.bev_y0 - 5) & (p[:, 1] <= args.bev_y1 + 5)).any())


def cmd_render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from PIL import Image
    sel = pickle.load(open(SEL, "rb"))
    order = {w: k for k, w in enumerate(WEATHERS)}
    sel = sorted(sel, key=lambda s: (order[s["weather"]], s["kind"] != "turn"))
    n = len(sel)
    bw, bh = 2 * args.bev_x, args.bev_y1 - args.bev_y0
    fig = plt.figure(figsize=(7.16, args.row_h * n))
    probe = mosaic(sel[0]["info"], np.zeros((6, 2)), args)[0].shape
    gs = GridSpec(n, 3, figure=fig, width_ratios=[probe[1] / probe[0], bw / bh, bw / bh], wspace=0.025, hspace=0.12,
                  left=0.03, right=0.998, top=0.985, bottom=0.005)
    cm = plt.get_cmap("cool")
    for r, s in enumerate(sel):
        info, pred = s["info"], s["pred"]
        plan = np.asarray(pred["plan_temp_2hz"])[:6, :2]
        M, segs, (tw, th, gap) = mosaic(info, plan, args)
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(M); ax.axis("off")
        for t, sg in segs:
            ax.plot(sg[:, 0], sg[:, 1], color=cm(0.15 + 0.85 * t), lw=1.8, solid_capstyle="round")
        for row, names in enumerate((CAMS_TOP, CAMS_BOT)):
            for c, name in enumerate(names):
                ax.text(c * (tw + gap) + 4, row * (th + gap) + 4, name.replace("CAM_", ""), color="white", fontsize=3.8,
                        va="top", bbox=dict(facecolor="black", alpha=0.45, pad=0.5, lw=0))
        ax.text(-0.012, 0.5, s["weather"], transform=ax.transAxes, rotation=90, fontsize=7.5,
                fontweight="bold", va="center", ha="right")
        cmd = CMD_NAME[int(np.argmax(info["gt_ego_fut_cmd"]))]
        # ---- 예측 BEV
        sc = np.asarray(pred["scores_3d"]); bx = np.asarray(pred["boxes_3d"])
        keep = sc >= args.box_thr
        pboxes = [b for b in bx[keep] if in_roi(b[None, :2], args)]
        tr = np.asarray(pred["trajs_3d"])[keep]; ts = np.asarray(pred["trajs_score"])[keep]
        ptraj = [np.concatenate([b[None, :2], t[np.argmax(q)][:args.motion_steps]], 0)
                 for b, t, q in zip(bx[keep], tr, ts) if in_roi(b[None, :2], args)]
        pmaps = [(l, np.asarray(v)) for v, sc_, l in zip(pred["vectors"], pred["scores"], pred["labels"])
                 if sc_ >= args.map_thr]
        ax = fig.add_subplot(gs[r, 1])
        draw_bev(ax, pboxes, pmaps, ptraj, plan, "pred", args)
        ax.text(0.02, 0.01, cmd, transform=ax.transAxes, fontsize=6, va="bottom")
        # ---- GT BEV
        m = np.asarray(info["valid_flag"])
        gb = np.asarray(info["gt_boxes"])[m]
        gfut = np.asarray(info["gt_agent_fut_trajs"])[m][:, :args.motion_steps]
        gmask = np.asarray(info["gt_agent_fut_masks"])[m][:, :args.motion_steps]
        gtraj = []
        for b, f, mk in zip(gb, gfut, gmask):
            k = int(mk.cumprod().sum())
            if k >= 1 and in_roi(b[None, :2], args):
                gtraj.append(np.concatenate([b[None, :2], b[:2] + f[:k].cumsum(0)], 0))
        gmaps = [(l, np.asarray(v)) for l, vs in info["map_annos"].items() for v in vs]
        gego = np.asarray(info["gt_ego_fut_trajs"]).cumsum(0)[:6, :2]
        ax = fig.add_subplot(gs[r, 2])
        draw_bev(ax, [b for b in gb if in_roi(b[None, :2], args)], gmaps, gtraj, gego, "gt", args)
        ax.text(0.02, 0.01, cmd, transform=ax.transAxes, fontsize=6, va="bottom")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".pdf", dpi=args.dpi); fig.savefig(OUT + ".jpg", dpi=args.dpi, pil_kwargs={"quality": 92})
    print("->", OUT + ".pdf", f"{os.path.getsize(OUT + '.pdf') / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "render"])
    ap.add_argument("--box-thr", type=float, default=0.35)
    ap.add_argument("--map-thr", type=float, default=0.4)
    ap.add_argument("--motion-steps", type=int, default=6)   # 3 s (2 Hz)
    ap.add_argument("--bev-x", type=float, default=20.0)
    ap.add_argument("--bev-y0", type=float, default=-22.0)
    ap.add_argument("--bev-y1", type=float, default=30.0)   # 지도 GT ROI 가 ±30 m 이므로 전방 30 m 까지
    ap.add_argument("--crop", type=float, nargs=2, default=[0.20, 0.92])   # 카메라 세로 크롭(하늘·보닛)
    ap.add_argument("--row-h", type=float, default=1.38)
    ap.add_argument("--down", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=300)
    # 수정(2026-10-12): 결과 경로를 실험 이름으로 지정(기본=대표 모델 2단계). 종전 고정값은 선행 모델 Run10(hipad_nusc_vla_s2r9).
    ap.add_argument("--run", default="hipad_nusc_laq_stage2")
    a = ap.parse_args()
    RES = f"evaluation/{a.run}/results.pkl"
    SEL = f"evaluation/{a.run}/paper_diag/qual_weather_sel.pkl"
    os.makedirs(os.path.dirname(SEL), exist_ok=True)
    cmd_select(a) if a.mode == "select" else cmd_render(a)
