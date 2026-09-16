"""수정(2026-09-02): ICRA 논문 Fig. 게이트 채널 분포.

  [무엇을 보이는가] 0 으로 초기화한 채널별 게이트 g(256차원)가 학습 후 어떤 값이 됐는지.
  주장은 세 가지다 — (1) 전 채널이 0 에서 벗어났다, (2) 부호가 양·음으로 갈렸다(단순 증폭이
  아니라 채널 선택적 변조), (3) 특정 소수 채널에 몰리지 않았다.
  히스토그램 하나로 셋 다 읽힌다.

  [형식] IEEE 1단 폭(3.4in), 흑백 인쇄 대응 — 단일 계열이므로 색으로 구분할 것이 없고
  회색 한 톤만 쓴다. 계열이 하나면 범례를 두지 않는다(제목이 계열을 지칭).
  GPU 미사용, CPU 전용.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

g = np.array(json.load(open("results/gate_stage1.json"))["gate"])

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
})
fig, ax = plt.subplots(figsize=(3.4, 2.05))

n, _, _ = ax.hist(g, bins=32, color="0.62", edgecolor="0.25", linewidth=0.5)
ax.set_ylim(0, n.max() * 1.42)          # 주석이 막대와 겹치지 않도록 위쪽 여백 확보
ax.axvline(0.0, color="0.15", linestyle=(0, (4, 2)), linewidth=0.9)

ax.annotate(r"initialization $g^{(0)}=\mathbf{0}$",
            xy=(0.0, n.max() * 1.18), xytext=(0.010, n.max() * 1.18),
            fontsize=7, color="0.15", va="center",
            arrowprops=dict(arrowstyle="-", color="0.15", linewidth=0.7, shrinkA=0, shrinkB=2))

n_pos, n_neg, n_zero = int((g > 0).sum()), int((g < 0).sum()), int((g == 0).sum())
ax.text(0.015, 0.985, f"{n_neg} negative, {n_pos} positive;\n"
                      f"{n_zero} channels remain at zero",
        transform=ax.transAxes, fontsize=7, color="0.25", va="top", linespacing=1.35)

ax.set_xlabel(r"learned gate value $g_c$")
ax.set_ylabel("channels")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(length=2.5)
ax.grid(axis="y", color="0.9", linewidth=0.5)
ax.set_axisbelow(True)

fig.tight_layout(pad=0.3)
fig.savefig("docs/icra/figures/gate_dist.pdf")
fig.savefig("docs/icra/figures/gate_dist.png", dpi=220)
print(f"저장 완료 | 채널 {g.size}  음 {n_neg} 양 {n_pos} 정확히0 {n_zero}  "
      f"|g| 평균 {np.abs(g).mean():.5f}  범위 [{g.min():.5f}, {g.max():.5f}]")
