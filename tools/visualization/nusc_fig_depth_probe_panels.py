"""수정(2026-10-15 밤): 발표 '교사 VLM 의 거리 추론' 슬라이드용 — 그림 표본 CAM_FRONT 10×18 칸별 깊이 비교(사용자 지시로 개편).
   (1) LiDAR 로 잰 칸별 깊이(정답, 점 ≥3 칸만) (2) Qwen 토큰으로 예측한 칸별 깊이(선형 회귀, nusc_vl_depth_probe.py 산출)
   (3) 차이 = 예측 − LiDAR (m, 빨강 = 실제보다 멀게, 파랑 = 가깝게). 세 면 모두 실제 카메라 영상 위에 겹친다.
   사용(CPU): PYTHONPATH=. python tools/visualization/nusc_fig_depth_probe_panels.py → docs/ppt/figures/vl_depth_probe_panels.png"""
import json, pickle
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; fm.fontManager.addfont(fp)
plt.rcParams.update({"font.family": fm.FontProperties(fname=fp).get_name(), "axes.unicode_minus": False})
TOKEN = "c62b6ac878934607be0c524a02f1692b"
Z = np.load("docs/ppt/figures/vl_depth_probe.npz")
v = pickle.load(open("data/infos/nuscenes_infos_val.pkl", "rb")); v = v["infos"] if isinstance(v, dict) else v
info = next(x for x in v if x["token"] == TOKEN)
img = np.asarray(Image.open(info["cams"]["CAM_FRONT"]["data_path"]).convert("RGB"))
H, W = img.shape[:2]
gt, pr = Z["fig_lidar_depth"], Z["fig_pred_depth"]
diff = pr - gt                                            # NaN = LiDAR 점 부족 칸
m = ~np.isnan(gt)
ad = np.abs(diff[m])
fig, axs = plt.subplots(1, 3, figsize=(15, 3.9))
fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.27, wspace=0.03)
ext = (0, W, H, 0)
for ax in axs:
    ax.imshow(img, extent=ext); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
im1 = axs[0].imshow(np.ma.masked_invalid(np.minimum(gt, 50)), extent=ext, cmap="turbo", vmin=0, vmax=50, alpha=0.65, interpolation="nearest")
axs[1].imshow(np.minimum(pr, 50), extent=ext, cmap="turbo", vmin=0, vmax=50, alpha=0.65, interpolation="nearest")
im3 = axs[2].imshow(np.ma.masked_invalid(np.clip(diff, -20, 20)), extent=ext, cmap="RdBu_r", vmin=-20, vmax=20, alpha=0.8, interpolation="nearest")
axs[0].set_title("① LiDAR 로 잰 깊이 (정답)", fontsize=15)
axs[1].set_title("② Qwen 토큰으로 예측한 깊이", fontsize=15)
axs[2].set_title(f"③ 차이 = ② − ① (중앙 오차 {np.median(ad):.1f} m)", fontsize=15)
for ax in axs:
    for k in range(1, 18): ax.axvline(k * W / 18, color="white", lw=0.3, alpha=0.5)
    for k in range(1, 10): ax.axhline(k * H / 10, color="white", lw=0.3, alpha=0.5)
c1 = fig.add_axes([0.06, 0.17, 0.55, 0.03]); cb = fig.colorbar(im1, cax=c1, orientation="horizontal"); cb.set_label("깊이 (m) — ①·② 공통, 빈 칸 = LiDAR 점 부족", fontsize=11)
c3 = fig.add_axes([0.70, 0.17, 0.26, 0.03]); cb3 = fig.colorbar(im3, cax=c3, orientation="horizontal")
cb3.set_label("차이 (m): 빨강 = 멀게 예측 · 파랑 = 가깝게 예측", fontsize=11)
fig.savefig("docs/ppt/figures/vl_depth_probe_panels.png", dpi=150)
json.dump(dict(fig_median_abs_err=float(np.median(ad)), fig_n_cells=int(m.sum())), open("docs/ppt/figures/vl_depth_probe_panels.json", "w"))
print("saved", float(np.median(ad)), int(m.sum()))
