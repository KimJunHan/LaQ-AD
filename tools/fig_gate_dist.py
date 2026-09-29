"""수정(2026-09-16 v2): IV 논문 Fig.2 — 게이트 채널 분포, Fig.1과 같은 파스텔 스타일.

  [무엇을 보이는가] 0 초기화 채널별 게이트 g(256차원)의 학습 후 분포.
  (1) 전 채널이 0 에서 벗어남 (2) 부호가 양·음으로 갈림(채널 선택적 변조)
  (3) 소수 채널 집중 없음. 음수=파랑, 양수=주황으로 부호 분리를 색이 직접 보인다.
  [형식] IEEE 1단 폭 3.4in, DejaVu Sans(Fig.1과 통일), 출력은 docs/iv/figures/.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 수정(2026-09-28): 대표모델 s2hr(2단계 완주) 게이트로 교체 — 본문 표와 동일 체크포인트.
# 수정(2026-09-29): results/ 는 저장소에서 제외되므로 논문 폴더 사본으로 폴백(그림 단독 재현).
import os
_SRC = "results/gate_s2hr.json"
if not os.path.exists(_SRC):
    _SRC = "docs/iv/figures/gate_s2hr.json"
g = np.array(json.load(open(_SRC))["gate"])
BL_F, BL_E = "#DEEAF6", "#4A76A8"
OR_F, OR_E = "#FDEBD3", "#C9741F"
TXT = "#1C2733"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
})
fig, ax = plt.subplots(figsize=(3.4, 2.05))

bins = np.histogram_bin_edges(g, bins=32)
ax.hist(g[g < 0], bins=bins, color=BL_F, edgecolor=BL_E, linewidth=0.6)
n_p, _, _ = ax.hist(g[g > 0], bins=bins, color=OR_F, edgecolor=OR_E, linewidth=0.6)
nmax = max(np.histogram(g, bins=bins)[0].max(), 1)
ax.set_ylim(0, nmax * 1.42)
ax.axvline(0.0, color="#3B4653", linestyle=(0, (4, 2)), linewidth=0.9)

ax.annotate(r"initialization $g^{(0)}=\mathbf{0}$",
            xy=(0.0, nmax * 1.16), xytext=(0.010, nmax * 1.16),
            fontsize=7, color=TXT, va="center",
            arrowprops=dict(arrowstyle="-", color="#3B4653", linewidth=0.7,
                            shrinkA=0, shrinkB=2))

n_pos, n_neg, n_zero = int((g > 0).sum()), int((g < 0).sum()), int((g == 0).sum())
from matplotlib.patches import Patch
leg = ax.legend(handles=[
        Patch(fc=BL_F, ec=BL_E, lw=0.9, label=f"{n_neg} negative channels"),
        Patch(fc=OR_F, ec=OR_E, lw=1.1, label=f"{n_pos} positive channels"),
    ], loc="upper left", fontsize=6.2, frameon=True, fancybox=True,
    borderpad=0.6, labelspacing=0.45, handlelength=1.3, handleheight=0.9)
leg.get_frame().set_edgecolor("#B9C2CC")
leg.get_frame().set_linewidth(0.7)
ax.text(0.012, nmax * 0.94, f"{n_zero} channels remain at zero",
        fontsize=6.2, color="#7A8694", va="top", ha="left")

ax.set_xlabel(r"learned gate value $g_c$", fontsize=7.5, color=TXT)
ax.set_ylabel("channels", fontsize=7.5, color=TXT)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#8A94A0")
ax.tick_params(length=2.5, colors="#4B5866")
ax.grid(axis="y", color="#ECEFF2", linewidth=0.6)
ax.set_axisbelow(True)

fig.tight_layout(pad=0.3)
fig.savefig("docs/iv/figures/gate_dist.pdf")
fig.savefig("docs/iv/figures/gate_dist.png", dpi=220)
print(f"저장 완료 | 채널 {g.size}  음 {n_neg} 양 {n_pos} 정확히0 {n_zero}")
