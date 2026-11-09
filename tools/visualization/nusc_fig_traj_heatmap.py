"""수정(2026-10-09): 논문 정성 그림 — nuScenes 자차 미래 궤적 분포 히트맵(사용자 예시 planexample.png (a) 형식).
  GT: train 분할의 3 s(2 Hz×6) 자차 미래 궤적 점을 BEV(전방 +y, 우측 +x) 격자에 누적, 로그 색. 명령(우·좌·직진)별 비율도 표기.
  출력: docs/iv/figures/traj_heatmap.pdf/png
"""
import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
d = pickle.load(open("data/infos/nuscenes_infos_train.pkl", "rb")); d = d["infos"] if isinstance(d, dict) else d
pts, cmds = [], []
for info in d:
    m = np.asarray(info["gt_ego_fut_masks"])
    if not (m == 1).all(): continue
    pts.append(np.asarray(info["gt_ego_fut_trajs"], np.float64).cumsum(0)[:6, :2]); cmds.append(int(np.argmax(info["gt_ego_fut_cmd"])))
P = np.concatenate(pts); cmds = np.array(cmds)
fig, ax = plt.subplots(figsize=(3.45, 2.9))
h = ax.hist2d(P[:, 0], P[:, 1], bins=[np.arange(-25, 25.5, 0.5), np.arange(-2, 45.5, 0.5)], norm=LogNorm(), cmap="Reds")
fig.colorbar(h[3], ax=ax, pad=0.02, label="count")
ax.set_xlabel("lateral x (m, right +)"); ax.set_ylabel("longitudinal y (m, forward +)")
r = [100 * np.mean(cmds == c) for c in range(3)]
ax.set_title(f"nuScenes train ego 3 s future ({len(pts):,} samples)\nright {r[0]:.1f}% · left {r[1]:.1f}% · straight {r[2]:.1f}%", fontsize=7.5)
ax.set_aspect("equal")
fig.tight_layout(pad=0.2)
fig.savefig("docs/iv/figures/traj_heatmap.pdf"); fig.savefig("docs/iv/figures/traj_heatmap.png", dpi=250)
print("저장", len(pts), [round(x, 1) for x in r])
