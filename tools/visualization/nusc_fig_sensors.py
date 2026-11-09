"""수정(2026-10-15): 발표 자료용 — "왜 언어 모델인가" 슬라이드의 센서 동기 그림(사용자 지시).
   nuScenes val 실측 표본 1개의 CAM_FRONT 위에
     (a) LIDAR_TOP 점 투영(색 = 깊이)        — LiDAR 가 주는 3D
     (b) RADAR_FRONT 점 투영 + 속도 화살표   — 레이더가 주는 3D(거리 · 시선 속도 성분)
     (c) 카메라만 + 정답 3D 상자(거리 표기)   — 카메라로 복원하려는 3D
     (d) 같은 영상의 VLM(Qwen2-VL-2B 18층) 영상 토큰 10×18 에서 선형 탐침으로 복원한 칸별 깊이(색 = m, (a) 와 같은 척도)
         수정(2026-10-15 밤): 사용자 지시 "(d) 색과 관련되게" — 주성분 RGB(고정 의미 없음)를 깊이 복원으로 교체.
         탐침은 tools/visualization/nusc_vl_depth_probe.py(train 600 학습, val 평가) 산출물 docs/ppt/figures/vl_depth_probe.{npz,json} 을 읽는다.
   좌표 변환은 nuscenes-devkit 의 map_pointcloud_to_image 와 같은 순서
   (센서 → 자차(센서 시각) → 전역 → 자차(카메라 시각) → 카메라 → 내부 행렬). 상자는 devkit Box(중심 = 기하 중심).
   주성분은 torch.linalg.svd 로 계산(이 환경 numpy LAPACK 결함). 위치 성분은 val 표본 40개의 같은 칸 평균을 빼서 제거.
   사용(CPU): CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tools/visualization/nusc_fig_sensors.py
   → docs/ppt/figures/sensor_motivation.png + sensor_motivation.json
"""
import json, os, pickle, sys
import numpy as np

sys.path.insert(0, ".")
DATAROOT = "data/nuscenes"
TOKEN = "c62b6ac878934607be0c524a02f1692b"   # lang_tokens · vl_cache 그림과 같은 표본(val, 'Intersection, peds, cars, scooter, parked bicycle')
CACHE = "data/nuscenes/qwen_l18"
CAM_RANGE = (15, 194)                        # vl_cache.json cam_ranges['CAM_FRONT'] (포함 구간)
GH, GW = 10, 18
OUT = "docs/ppt/figures/sensor_motivation"
N_POSMEAN = 40


def load_infos(p):
    d = pickle.load(open(p, "rb"))
    return d["infos"] if isinstance(d, dict) else d


def to_cam(nusc, pts_xyz, sd_pt, sd_cam):
    """(3, N) 센서 좌표 점 → 카메라 좌표. devkit map_pointcloud_to_image 와 같은 변환 순서."""
    from pyquaternion import Quaternion
    p = pts_xyz.copy()
    cs = nusc.get("calibrated_sensor", sd_pt["calibrated_sensor_token"])
    p = Quaternion(cs["rotation"]).rotation_matrix @ p + np.array(cs["translation"])[:, None]
    ep = nusc.get("ego_pose", sd_pt["ego_pose_token"])
    p = Quaternion(ep["rotation"]).rotation_matrix @ p + np.array(ep["translation"])[:, None]
    ep = nusc.get("ego_pose", sd_cam["ego_pose_token"])
    p = Quaternion(ep["rotation"]).rotation_matrix.T @ (p - np.array(ep["translation"])[:, None])
    cs = nusc.get("calibrated_sensor", sd_cam["calibrated_sensor_token"])
    p = Quaternion(cs["rotation"]).rotation_matrix.T @ (p - np.array(cs["translation"])[:, None])
    return p, np.array(cs["camera_intrinsic"])


def project(p_cam, K):
    uv = K @ p_cam
    return uv[:2] / uv[2:3]


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
    from nuscenes.utils.geometry_utils import BoxVisibility
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    fpath = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fm.fontManager.addfont(fpath)
    plt.rcParams.update({"font.family": fm.FontProperties(fname=fpath).get_name(), "axes.unicode_minus": False, "font.size": 9})

    nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
    sample = nusc.get("sample", TOKEN)
    scene = nusc.get("scene", sample["scene_token"])["description"]
    sd_cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    sd_lid = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    sd_rad = nusc.get("sample_data", sample["data"]["RADAR_FRONT"])
    img = np.asarray(Image.open(os.path.join(DATAROOT, sd_cam["filename"])).convert("RGB"))
    H, W = img.shape[:2]

    def in_img(uv, depth, min_d=1.0):
        return (depth > min_d) & (uv[0] > 1) & (uv[0] < W - 1) & (uv[1] > 1) & (uv[1] < H - 1)

    # (a) LiDAR
    lpc = LidarPointCloud.from_file(os.path.join(DATAROOT, sd_lid["filename"]))
    lc, K = to_cam(nusc, lpc.points[:3], sd_lid, sd_cam)
    luv = project(lc, K); lm = in_img(luv, lc[2])
    l_depth = lc[2][lm]
    # 대조: devkit 함수와 같은 결과인지 확인(점 수·깊이)
    dk_pts, dk_col, _ = nusc.explorer.map_pointcloud_to_image(sd_lid["token"], sd_cam["token"])
    assert dk_pts.shape[1] == int(lm.sum()), (dk_pts.shape, int(lm.sum()))
    assert np.allclose(np.sort(dk_col), np.sort(l_depth), atol=1e-4)

    # (b) 레이더 (devkit 기본 필터) — 보상 속도(vx_comp, vy_comp: 자차 운동 제거, 레이더 좌표)
    rpc = RadarPointCloud.from_file(os.path.join(DATAROOT, sd_rad["filename"]))
    rc, _ = to_cam(nusc, rpc.points[:3], sd_rad, sd_cam)
    vel = np.vstack([rpc.points[8], rpc.points[9], np.zeros(rpc.points.shape[1])])
    rc_end, _ = to_cam(nusc, rpc.points[:3] + vel * 2.0, sd_rad, sd_cam)   # 2 s 이동 끝점
    ruv = project(rc, K); rm = in_img(ruv, rc[2])
    ruv_end = project(rc_end, K)
    speed = np.linalg.norm(vel[:2], axis=0)

    # (c) 정답 상자 (카메라 좌표, 기하 중심)
    _, boxes, K2 = nusc.get_sample_data(sd_cam["token"], box_vis_level=BoxVisibility.ANY)
    boxes = [b for b in boxes if b.center[2] > 1.0]

    # (d) VLM 영상 토큰 → 칸별 깊이 복원(탐침 산출물). 수정(2026-10-15 밤): 종전 주성분 RGB 는 아래 계산만 유지(json 기록용)
    vinf = load_infos("data/infos/nuscenes_infos_val.pkl")
    toks = [v["token"] for v in vinf[::max(len(vinf) // N_POSMEAN, 1)]][:N_POSMEAN]
    a, b_ = CAM_RANGE
    ref = torch.stack([torch.load(os.path.join(CACHE, t + ".pt"), map_location="cpu")["hidden"][a:b_ + 1].float() for t in toks])
    pos_mean = ref.mean(0)                                   # (180, 1536) 칸별 평균 = 위치 성분
    x = torch.load(os.path.join(CACHE, TOKEN + ".pt"), map_location="cpu")["hidden"][a:b_ + 1].float() - pos_mean
    x = x - x.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    pcs = (x @ Vh[:3].T).numpy()                             # (180, 3)
    var3 = float((S[:3] ** 2).sum() / (S ** 2).sum())
    lo, hi = np.percentile(pcs, 2, axis=0), np.percentile(pcs, 98, axis=0)
    rgb = np.clip((pcs - lo) / (hi - lo + 1e-6), 0, 1).reshape(GH, GW, 3)

    # ---- 그림 ----
    fig, axs = plt.subplots(2, 2, figsize=(12.0, 7.25))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005, wspace=0.015, hspace=0.11)
    # 슬라이드에서 반폭(≈6 in)으로 줄어드므로 글자를 크게, 제목은 짧게(세부 설명은 슬라이드 본문)
    titles = [f"(a) LiDAR — 점 {int(lm.sum()):,}개, 색 = 깊이",
              f"(b) 레이더 — 점 {int(rm.sum())}개, 화살표 = 속도",
              f"(c) 카메라 + 정답 3D 상자 (거리 m)",
              f"(d) VLM 토큰 → 칸별 깊이 복원 (색 = 깊이)"]
    for ax, t in zip(axs.flat, titles):
        ax.imshow(img); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off"); ax.set_title(t, fontsize=17, loc="left")
    vmax = 50.0
    sc = axs[0, 0].scatter(luv[0][lm], luv[1][lm], c=np.minimum(l_depth, vmax), s=1.2, cmap="turbo", vmin=0, vmax=vmax, linewidths=0)
    axs[0, 1].scatter(ruv[0][rm], ruv[1][rm], c=np.minimum(rc[2][rm], vmax), s=55, cmap="turbo", vmin=0, vmax=vmax,
                      edgecolors="white", linewidths=0.8, zorder=3)
    n_arrow = 0
    for i in np.nonzero(rm & (speed > 0.5))[0]:
        if rc_end[2, i] > 1.0:
            axs[0, 1].annotate("", xy=(ruv_end[0, i], ruv_end[1, i]), xytext=(ruv[0, i], ruv[1, i]),
                               arrowprops=dict(arrowstyle="->", color="#ff2d2d", lw=1.8), zorder=4)
            n_arrow += 1
    placed = []
    for bx in sorted(boxes, key=lambda b: float(np.linalg.norm(b.center))):
        c = "#ff7f0e" if bx.name.startswith("human") else ("#1f77b4" if bx.name.startswith("vehicle") else "#2ca02c")
        bx.render(axs[1, 0], view=K2, normalize=True, colors=(c, c, c), linewidth=1.3)
        d = float(np.linalg.norm(bx.center))
        if d < 60:
            cuv = project(bx.center[:, None], K2)[:, 0]
            # 가까운 상자부터 표기, 이미 적은 숫자와 95 px 안이면 생략(겹침 방지 — 상자는 모두 그림)
            if 0 < cuv[0] < W and 0 < cuv[1] < H and all(np.hypot(cuv[0] - q[0], cuv[1] - q[1]) > 95 for q in placed):
                placed.append(cuv)
                axs[1, 0].text(cuv[0], cuv[1], f"{d:.0f}", fontsize=12, color="white", ha="center", va="center",
                               bbox=dict(boxstyle="round,pad=0.12", fc=c, ec="none", alpha=0.85))
    axs[1, 0].text(20, H - 25, "주황 사람 · 파랑 차량 · 초록 그 밖", fontsize=14, color="white", va="bottom",
                   bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55))
    axs[0, 1].text(20, H - 25, "높이 없음(평면 측정) → 수평 띠", fontsize=14, color="white", va="bottom",
                   bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55))
    PR = json.load(open("docs/ppt/figures/vl_depth_probe.json")); PZ = np.load("docs/ppt/figures/vl_depth_probe.npz")
    axs[1, 1].imshow(np.minimum(PZ["fig_pred_depth"], vmax), extent=(0, W, H, 0), interpolation="nearest", alpha=0.6,
                     cmap="turbo", vmin=0, vmax=vmax)
    axs[1, 1].text(20, H - 25, f"val R² {PR['r2_val_token']:.2f} (칸 위치만 {PR['r2_val_pos_only']:.2f})", fontsize=14, color="white",
                   va="bottom", bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55))
    for k in range(1, GW):
        axs[1, 1].axvline(k * W / GW, color="white", lw=0.3, alpha=0.5)
    for k in range(1, GH):
        axs[1, 1].axhline(k * H / GH, color="white", lw=0.3, alpha=0.5)
    cax = fig.add_axes([0.05, 0.545, 0.22, 0.016])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_ticks([0, 10, 20, 30, 40, 50])
    cb.ax.tick_params(labelsize=11, colors="white", length=2); cb.outline.set_edgecolor("white")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=160)

    st = dict(token=TOKEN, scene=scene, n_lidar_total=int(lpc.points.shape[1]), n_lidar_in_img=int(lm.sum()),
              lidar_depth_pct=[float(np.percentile(l_depth, q)) for q in (5, 50, 95)], lidar_depth_max=float(l_depth.max()),
              n_radar_total=int(rpc.points.shape[1]), n_radar_in_img=int(rm.sum()), n_radar_arrow=n_arrow,
              radar_depth_max=float(rc[2][rm].max()) if rm.any() else None,
              n_boxes=len(boxes), box_dist=[float(np.linalg.norm(b.center)) for b in boxes],
              pca_var3=var3, n_posmean=len(toks), depth_probe=PR, devkit_check="lidar projection == nusc.explorer.map_pointcloud_to_image")
    json.dump(st, open(OUT + ".json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in st.items() if k != "box_dist"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
