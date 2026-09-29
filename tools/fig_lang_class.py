"""수정(2026-09-16): IV 논문 — 클래스별 언어 기여를 표(tab:lang_class) 대신
덤벨 차트로 가시화. 흑백 인쇄 대응(회색조, 열린/채운 마커), 1단 폭(3.45in).
데이터는 수정(2026-09-28) 대표모델 s2hr(896x448, 2단계 완주) 소거 실측으로 교체."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5})

rows = [  # (class, mismatched, correct, gain%)  수정(2026-09-28): s2hr 실측
    ("constr. veh.", 0.0465, 0.0637, "+37.2"),
    ("trailer",      0.0533, 0.0696, "+30.6"),
    ("traffic cone", 0.6259, 0.6740, "+7.7"),
    ("truck",        0.2954, 0.3175, "+7.5"),
    ("pedestrian",   0.4765, 0.5047, "+5.9"),
    ("motorcycle",   0.3860, 0.4071, "+5.5"),
    ("bicycle",      0.3951, 0.4007, "+1.4"),
    ("car",          0.6291, 0.6378, "+1.4"),
    ("barrier",      0.5809, 0.5863, "+0.9"),
    ("bus",          0.3123, 0.3052, "-2.3"),
]

fig, ax = plt.subplots(figsize=(3.45, 2.35))
ys = range(len(rows) - 1, -1, -1)          # 이득 큰 클래스가 위
for y, (name, a, b, g) in zip(ys, rows):
    ax.plot([a, b], [y, y], color="#B9C6D2", lw=1.6, zorder=1,
            solid_capstyle="round")
    ax.plot(a, y, "o", ms=4.2, mfc="white", mec="#4A76A8", mew=1.1, zorder=2)
    ax.plot(b, y, "o", ms=4.6, mfc="#C9741F", mec="#9A5512", mew=0.6, zorder=3)
    ax.text(0.755, y, g + "\\%" if False else g + "%", va="center", ha="left",
            fontsize=6.0, color="#1C2733")
ax.text(0.755, len(rows) - 0.25, "gain", fontsize=6.0, color="0.35",
        ha="left", va="center")

ax.set_yticks(list(ys))
ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
ax.set_xlim(0, 0.74); ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_xticks([0, 0.2, 0.4, 0.6])
ax.set_xlabel("per-class detection AP", fontsize=6.5, labelpad=2)
ax.tick_params(axis="both", length=2, pad=1.5, labelsize=6.0)
ax.grid(axis="x", color="#ECEFF2", lw=0.6, zorder=0)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color("#8A94A0")

handles = [
    Line2D([], [], marker="o", ls="", mfc="white", mec="#4A76A8", mew=1.1, ms=4.2,
           label="mismatched tokens"),
    Line2D([], [], marker="o", ls="", mfc="#C9741F", mec="#9A5512", mew=0.6, ms=4.6,
           label="correct tokens"),
]
ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.99),
          frameon=False, fontsize=6.0, handletextpad=0.3, borderaxespad=0.2,
          labelspacing=0.3)

fig.tight_layout(pad=0.3)
fig.savefig("docs/iv/figures/lang_class.pdf")
fig.savefig("docs/iv/figures/lang_class.png", dpi=220)
print("저장 완료")
