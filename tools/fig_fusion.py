"""수정(2026-10-09): IV 논문 Fig.2 — 영상 쿼리와 언어 토큰의 융합 상세(사용자 지시: Fig.1 은 개괄, 융합은 방법 절 별도 그림).

  표현 대상(대표모델 Run10, models/decoder/language_fusion/qwen_token_attn.py 와 일치):
    - 쿼리 특징 F (N×256, 검출·지도·계획/자차 구간)가 질의 Q
    - 6방향 영상 + 고정 지시문 → 동결 Qwen2-VL-2B 18층 은닉 H (1159×1536 = 시각 1080 + 문자 79), 오프라인 캐시
    - 사영기(2층 MLP 1536→512→256 + LayerNorm) → K, V
    - 다중 헤드 교차 어텐션(8 heads); 블록이 잔차를 포함하므로 출력 = F + A
    - 게이트 g ∈ R^{6×3×256}: 현재 계층 l·쿼리 i 의 과제 구간 s(i) 로 선택, 0 초기화
    - F'_i = F_i + g_{l,s(i)} ⊙ (F_i + A_i)   ( = 논문 식 (1+g)⊙F + g⊙A )
  규칙: 글자 ≥7pt, 직교 화살표, 교차 0 (Fig.1 v5 와 같은 색 규약).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig, ax = plt.subplots(figsize=(7.16, 2.45))
ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

BL_F, BL_E = "#DEEAF6", "#4A76A8"
OR_F, OR_E = "#FDEBD3", "#C9741F"
PU_F, PU_E = "#EAE4F4", "#7E6BB5"
GR_F, GR_E = "#F5F5F5", "#8A8A8A"
TXT, GEO, LANG = "#1C2733", "#3C5A78", "#C06818"
Q_DET, Q_MAP, Q_PLAN = "#4A76A8", "#5E9E68", "#C97A96"
FS, FS2 = 8.0, 7.0


def box(x, y, w, h, label, fc=BL_F, ec=BL_E, lw=1.0, fs=FS, ls="-", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25", fc=fc, ec=ec, lw=lw, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=TXT,
            linespacing=1.15, fontweight="bold" if bold else "normal")


def path(pts, color=GEO, lw=1.1):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=9, shrinkA=0, shrinkB=0))


def qbar(x, y, w, h):
    for sx, sw, sc in ((0, 0.6, Q_DET), (0.6, 0.22, Q_MAP), (0.82, 0.18, Q_PLAN)):
        ax.add_patch(Rectangle((x + sx * w, y), sw * w, h, fc=sc, ec="none"))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="#4B5866", lw=0.7))


# ── 질의: 쿼리 특징 F ────────────────────────────────────────────
ax.text(13.0, 20.2, "query features $F$  ($N{\\times}256$)", fontsize=FS2, color=TXT, va="center", ha="center")
qbar(2.0, 22.0, 22.0, 2.6)
for lab, xc in (("det", 2.0 + 0.3 * 22), ("map", 2.0 + 0.71 * 22), ("p/e", 2.0 + 0.91 * 22)):
    ax.text(xc, 23.3, lab, fontsize=FS2, color="white", ha="center", va="center")
path([(24.4, 23.3), (47.6, 23.3)])
ax.text(36.0, 24.4, "Q", fontsize=FS, color=GEO, ha="center", va="bottom")

# ── 키·값: 언어 토큰 ─────────────────────────────────────────────
for k in range(6):
    x0, y0 = 2.0 + (k % 3) * 3.6, 6.2 + (1 - k // 3) * 3.4
    ax.add_patch(Rectangle((x0, y0), 3.0, 2.6, fc="#E3E8EE", ec="#5B6B7A", lw=0.6))
ax.text(7.0, 4.6, "6 views", fontsize=FS2, color=TXT, ha="center", va="center")
box(16.2, 12.6, 11.6, 3.2, "fixed prompt", fc="white", ec="#8A96A3", fs=FS2)
box(16.2, 4.6, 11.6, 6.4, "Qwen2-VL-2B\n(frozen)\nlayer 18 / 28", fc=PU_F, ec=PU_E, lw=1.2, fs=FS2)
path([(13.2, 7.8), (15.8, 7.8)], color=LANG)
path([(22.0, 12.2), (22.0, 11.4)], color="#8A96A3")
# H 토큰 열: 시각 1080 + 문자 79
hx, hw = 31.0, 13.0
ax.add_patch(Rectangle((hx, 6.6), hw * 1080 / 1159, 2.6, fc=PU_E, ec="none", alpha=0.75))
ax.add_patch(Rectangle((hx + hw * 1080 / 1159, 6.6), hw * 79 / 1159, 2.6, fc=LANG, ec="none"))
ax.add_patch(Rectangle((hx, 6.6), hw, 2.6, fc="none", ec="#4B5866", lw=0.7, ls=(0, (3, 2))))
ax.text(hx + hw / 2, 11.0, "tokens $H$ (offline cache)", fontsize=FS2, color=TXT, ha="center", va="center")
ax.text(hx + hw / 2, 5.0, "$1159{\\times}1536$", fontsize=FS2, color=TXT, ha="center", va="center")
path([(28.2, 7.8), (30.6, 7.8)], color=LANG)
box(47.6, 4.0, 13.0, 7.6, "projector\n2-layer MLP\n+ LayerNorm", fc=OR_F, ec=OR_E, lw=1.3, fs=FS2)
path([(44.4, 7.8), (47.2, 7.8)], color=LANG)
path([(54.1, 12.0), (54.1, 16.6)], color=LANG)
ax.text(55.0, 14.3, "K, V  ($1159{\\times}256$)", fontsize=FS2, color=LANG, va="center")

# ── 교차 어텐션 ──────────────────────────────────────────────────
box(47.6, 17.0, 13.0, 8.4, "multi-head\ncross-attention\n(8 heads)", fc=OR_F, ec=OR_E, lw=1.5, bold=False)

# ── 게이트 선택·곱·잔차 ──────────────────────────────────────────
mx, my, r = 78.0, 21.2, 1.25
path([(61.0, my), (mx - r - 0.2, my)])
ax.text(69.0, 22.2, "$F + A$", fontsize=FS, color=TXT, ha="center", va="bottom")
ax.add_patch(Circle((mx, my), r, fc="white", ec=OR_E, lw=1.2))
ax.text(mx, my, "$\\odot$", fontsize=FS, ha="center", va="center", color=TXT)
# 게이트 격자 6계층 × 3과제 (현재 계층 l 강조)
gx, gy, cw, rh = 73.0, 4.6, 3.4, 1.35
for row in range(6):
    for col, c in enumerate((Q_DET, Q_MAP, Q_PLAN)):
        ax.add_patch(Rectangle((gx + col * cw, gy + (5 - row) * rh), cw, rh, fc=c,
                               alpha=0.9 if row == 2 else 0.25, ec="white", lw=0.4))
ax.add_patch(Rectangle((gx, gy + 3 * rh), 3 * cw, rh, fc="none", ec=OR_E, lw=1.6))
ax.add_patch(Rectangle((gx, gy), 3 * cw, 6 * rh, fc="none", ec="#4B5866", lw=0.7))
ax.text(gx - 0.6, gy + 3.5 * rh, "layer $l$", fontsize=FS2, color=OR_E, ha="right", va="center")
for col, lab in enumerate(("det", "map", "p/e")):
    ax.text(gx + (col + 0.5) * cw, gy - 0.9, lab, fontsize=FS2, color=TXT, ha="center", va="center")
ax.text(gx + 1.5 * cw, gy + 6 * rh + 1.0, "gate $g$ ($6{\\times}3{\\times}256$, init $0$)",
        fontsize=FS2, color=TXT, ha="center", va="bottom")
path([(mx, gy + 6 * rh + 3.0), (mx, my - r - 0.2)], color=OR_E)
ax.text(mx + 0.8, 16.8, "$g_{l,s(i)}$", fontsize=FS, color=OR_E, va="center")

px = 88.5
ax.add_patch(Circle((px, my), r, fc="white", ec="#3B4653", lw=1.1))
ax.text(px, my, "+", fontsize=FS + 1, ha="center", va="center", color=TXT)
path([(mx + r + 0.2, my), (px - r - 0.2, my)])
# 잔차 F (위로 우회)
ax.plot([13.0, 13.0, px], [24.8, 30.6, 30.6], color=GEO, lw=1.0)
path([(px, 30.6), (px, my + r + 0.2)])
ax.text(50.0, 31.4, "residual $F$", fontsize=FS2, color=GEO, ha="center", va="bottom")
qbar(91.2, my - 1.3, 8.0, 2.6)
path([(px + r + 0.2, my), (91.0, my)])
ax.text(95.2, my + 2.2, "$F'$", fontsize=FS, color=TXT, ha="center", va="bottom")

ax.text(99.5, 1.2, "$F'_i = F_i + g_{l,s(i)} \\odot (F_i + A_i)$,   $s(i)$: task of query $i$",
        fontsize=FS, color=TXT, ha="right", va="center")

fig.tight_layout(pad=0.15)
fig.savefig("docs/iv/figures/fusion.pdf")
fig.savefig("docs/iv/figures/fusion.png", dpi=250)
print("저장 완료")
