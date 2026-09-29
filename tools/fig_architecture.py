"""수정(2026-09-16 v4): IV 논문 Fig.1 — HiP-AD/SparseDrive 풍 컬러 디자인.

  사용자 요구: 회색조가 너무 공대 스타일 → 게재 논문들처럼 파스텔 색 구분 + 실제
  카메라 썸네일. 모듈 색 규약(범례와 일치):
    파랑 = HiP-AD 기반(학습됨) / 주황 = 추가분(ours, 학습됨 +0.53M) /
    보라+❄ = 동결 VLM(2B) / 회색 점선 = 오프라인 데이터·캐시.
  구성·좌표는 v3(겹침 제거·모듈 그룹·융합 상세 패널)을 유지. IEEE 2단 폭 7.1in.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
fig, ax = plt.subplots(figsize=(7.1, 3.37))
ax.set_xlim(0, 100); ax.set_ylim(0, 47.5); ax.axis("off")

# 팔레트
BL_F, BL_E = "#DEEAF6", "#4A76A8"     # 기반(학습)
OR_F, OR_E = "#FDEBD3", "#C9741F"     # ours(학습)
PU_F, PU_E = "#EAE4F4", "#7E6BB5"     # 동결 VLM
GR_F, GR_E = "#F5F5F5", "#8A8A8A"     # 데이터/캐시
TXT = "#1C2733"
A_GEO, A_LANG = "#3C5A78", "#C06818"  # 화살표
Q_DET, Q_MAP, Q_PLAN = "#4A76A8", "#5E9E68", "#C97A96"  # 쿼리 종류

def box(x, y, w, h, label, fc=BL_F, ec=BL_E, lw=1.0, fs=6.8,
        style="round,pad=0.10", tc=TXT, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec,
                                lw=lw, linestyle=ls))
    if label:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fs, color=tc, linespacing=1.3)

def arrow(x1, y1, x2, y2, lw=1.0, color=A_GEO, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=8, lw=lw, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0))

def seg_bar(x, y, w, h, lab, lx):
    for sx, sw, sc in ((0, 0.60, Q_DET), (0.60, 0.25, Q_MAP), (0.85, 0.15, Q_PLAN)):
        ax.add_patch(Rectangle((x + sx*w, y), sw*w, h, fc=sc, ec="none", alpha=0.85))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="#4B5866", lw=0.7))
    ax.text(lx, y - 0.85, lab, ha="center", va="center", fontsize=5.8, color=TXT)

def thumbs(x, y, w, h, imgs, dx=0.9, dy=0.75):
    for i, im in zip((2, 1, 0), imgs[::-1]):
        ex = (x + i*dx, x + i*dx + w, y + i*dy, y + i*dy + h)
        ax.imshow(im, extent=ex, zorder=3 - i, interpolation="bilinear")
        ax.add_patch(Rectangle((ex[0], ex[2]), w, h, fc="none", ec="white",
                               lw=1.2, zorder=3 - i + 0.1))
        ax.add_patch(Rectangle((ex[0], ex[2]), w, h, fc="none", ec="#5B6B7A",
                               lw=0.6, zorder=3 - i + 0.2))

IMGS = []
for cam, f in (("CAM_FRONT", "n015-2018-10-02-11-23-23+0800__CAM_FRONT__1538451148112460.jpg"),
               ("CAM_FRONT_LEFT", "n015-2018-11-14-19-52-02+0800__CAM_FRONT_LEFT__1542196583104844.jpg"),
               ("CAM_BACK", "n015-2018-08-03-12-54-49+0800__CAM_BACK__1533272191037525.jpg")):
    IMGS.append(plt.imread(f"data/nuscenes/samples/{cam}/{f}")[::6, ::6])

# ══ 범례 (우상단) ══════════════════════════════════════════════
ax.add_patch(FancyBboxPatch((74.5, 35.3), 24.7, 10.6, boxstyle="round,pad=0.10",
                            fc="white", ec="#B9C2CC", lw=0.7))
def leg(y, txt, fc, ec, lw, ls="-", double=False, snow=False):
    ax.add_patch(Rectangle((75.6, y), 2.4, 1.3, fc=fc, ec=ec, lw=lw, linestyle=ls))
    if double:
        ax.add_patch(Rectangle((75.9, y+0.22), 1.8, 0.86, fc="none", ec=ec, lw=0.5))
    if snow:
        ax.text(76.8, y+0.62, "❄", ha="center", va="center", fontsize=4.6, color=PU_E)
    ax.text(78.9, y+0.62, txt, ha="left", va="center", fontsize=6.4, color=TXT)
leg(43.7, "trainable — HiP-AD base",        BL_F, BL_E, 0.9)
leg(41.7, "trainable — ours ($+0.53$M)",    OR_F, OR_E, 1.3)
leg(39.7, "frozen — Qwen2-VL-2B (2B)",      PU_F, PU_E, 1.0, double=True, snow=True)
leg(37.7, "offline data / cache (no grad.)", GR_F, GR_E, 0.8, ls=(0, (3, 2)))
ax.text(75.6, 36.3, "gradients never reach the frozen VLM",
        fontsize=6.0, color="#7A8694", ha="left", va="center")

# ══ 상단: 영상 인코딩 (기하 경로) ══════════════════════════════
ax.text(1, 46.3, "geometry pathway (HiP-AD base)", fontsize=7.4,
        style="italic", color=TXT)
thumbs(2, 36.8, 8.6, 6.0, IMGS)
ax.text(6.3 + 0.9, 36.1, "6-view images", ha="center", va="center",
        fontsize=6.6, color="#4B5866")
ax.text(6.3 + 0.9, 35.1, "each $3{\\times}448{\\times}896$", ha="center", va="center",
        fontsize=5.6, color="#7A8694")
box(14.5, 36.8, 12, 7.2, "image encoder\nResNet-50 + FPN")
for i in (2, 1, 0):
    ax.add_patch(Rectangle((29.5 + i*1.1, 37.0 + i*0.8), 7.6, 5.6,
                           fc="#EAF1F9", ec=BL_E, lw=0.7))
ax.text(33.3, 39.8, "multi-scale\nfeatures", ha="center", va="center",
        fontsize=6.6, color=TXT)
ax.text(31.3, 35.4, "6 views · 4 scales\n$C{=}256$", ha="center", va="center",
        fontsize=5.6, color="#7A8694", linespacing=1.2)
arrow(11.9, 40.4, 14.2, 40.4)
arrow(26.7, 40.4, 29.2, 40.4)

# ══ 좌: 통합 입력 그룹 (영상 + 앵커/쿼리 + 인스턴스 뱅크) ══════
ax.add_patch(FancyBboxPatch((0.4, 11.6), 13.6, 33.9, boxstyle="round,pad=0.10",
                            fc="none", ec="#B9C2CC", lw=0.7))
ax.text(1.1, 44.95, "inputs", fontsize=6.0, color="#7A8694", ha="left")
ax.add_patch(FancyBboxPatch((0.9, 18.85), 12.9, 12.75, boxstyle="round,pad=0.10",
                            fc="#F3F7FC", ec=BL_E, lw=0.9))
ax.text(7.25, 30.7, "task anchors & queries", ha="center", va="center",
        fontsize=5.8, color=TXT)
ax.text(7.25, 29.75, "k-means anchors:", ha="center", va="center",
        fontsize=5.8, color="#7A8694")
# det: 3D 상자 글리프
ax.add_patch(Rectangle((2.5, 27.6), 1.5, 1.0, fc="none", ec=Q_DET, lw=0.9))
ax.add_patch(Rectangle((2.9, 27.95), 1.5, 1.0, fc="none", ec=Q_DET, lw=0.6, alpha=0.6))
for dx0, dy0 in ((0, 0), (1.5, 0), (0, 1.0), (1.5, 1.0)):
    ax.plot([2.5+dx0, 2.9+dx0], [27.6+dy0, 27.95+dy0], color=Q_DET, lw=0.5, alpha=0.6)
# map: 폴리라인 글리프
ax.plot([6.3, 6.9, 7.5, 8.1], [27.7, 28.8, 27.9, 28.8], color=Q_MAP, lw=1.2,
        marker="o", ms=1.6, mfc=Q_MAP)
# plan: waypoint 점열 글리프
wx=[10.0, 10.45, 10.95, 11.5, 12.0]; wy=[27.7, 28.05, 28.35, 28.55, 28.7]
ax.plot(wx, wy, color=Q_PLAN, lw=0.8, ls=(0, (2, 1.5)))
ax.plot(wx, wy, "o", ms=1.8, mfc=Q_PLAN, mec="none")
ax.text(3.1, 27.0, "det 900", ha="center", fontsize=4.6, color=TXT)
ax.text(7.25, 27.0, "map 100", ha="center", fontsize=4.6, color=TXT)
ax.text(11.4, 27.0, "plan/ego", ha="center", fontsize=4.6, color=TXT)
arrow(7.25, 26.55, 7.25, 26.3, lw=0.8)
box(1.65, 24.0, 11.2, 2.2, "anchor encoder (MLP)\n$\\to$ positional emb.",
    fc=BL_F, ec=BL_E, fs=5.5, lw=0.8)
arrow(10.35, 23.85, 10.35, 23.5, lw=0.8)
ax.add_patch(plt.Circle((10.35, 22.85), 0.6, fc="white", ec=BL_E, lw=0.9))
ax.text(10.35, 22.8, "+", ha="center", va="center", fontsize=6.5, color=TXT)
ax.text(9.7, 22.85, "learnable instance\nfeature ($256$-d)", ha="right",
        va="center", fontsize=4.6, color=TXT, linespacing=1.2)
arrow(10.35, 22.2, 10.35, 21.6, lw=0.8)
seg_bar(2.6, 20.3, 9.3, 1.2, "queries $F$: $N{\\times}256$", 7.25)
arrow(13.7, 20.85, 16.8, 20.85)
box(1, 12.2, 12, 5.7, "temporal\ninstance bank\n(600 det propagated)",
    fc=GR_F, ec=GR_E, fs=5.8, ls=(0, (4, 2)), lw=0.8)
arrow(18.1 - 11.1, 18.05, 7.0, 18.95)
arrow(17.5, 11.9, 13.3, 14.8, ls=(0, (3, 2)), lw=0.8, color="#8A8A8A")
ax.text(7, 11.05, "IDs $\\to$ next frame", fontsize=6.0, color="#7A8694",
        ha="center", va="center")

# ══ 중앙: unified decoder layer ×6 ═════════════════════════════
box(15.5, 12, 57.2, 20, "", fc="#F3F7FC", ec="#9DB6D0", lw=1.0)
ax.text(17, 29.6, "unified decoder layer", fontsize=7.6, color=TXT, ha="left")
ax.text(17.2, 26.35, "in: queries $F$ ($N{\\times}256$)",
        fontsize=6.2, color="#7A8694", ha="left")
ax.add_patch(FancyArrowPatch((69.5, 29.6), (67.1, 29.6), arrowstyle="-|>",
                             mutation_scale=7, lw=0.9, color=A_GEO,
                             connectionstyle="arc3,rad=-1.7"))
ax.text(70.6, 29.3, "$\\times 6$", fontsize=7.8, color=TXT)

blocks = [
    (17.0, 10.0, "self-attention\n(concat tasks)", BL_F, 1.0, BL_E),
    (29.5, 10.5, "task deformable\naggregation",   BL_F, 1.0, BL_E),
    (42.0,  5.0, "FFN",                            BL_F, 1.0, BL_E),
    (48.8, 14.6, "language-query fusion",          OR_F, 1.5, OR_E),
    (65.2,  6.5, "split &\nrefine",                BL_F, 1.0, BL_E),
]
for x, w, lab, fc, lw, ec in blocks:
    box(x, 18, w, 7, lab, fc=fc, lw=lw, ec=ec, fs=6.5)
for x1, x2 in ((27.0, 29.5), (40.0, 42.0), (47.0, 48.8), (63.4, 65.2)):
    arrow(x1, 21.5, x2, 21.5, lw=0.9)
ax.text(64.3, 22.15, "$F'$", fontsize=5.6, color="#4B5866", ha="center")
arrow(36.5, 36.7, 36.5, 25.4)
box(60.0, 24.2, 3.1, 1.7, "ours", fc=OR_E, ec=OR_E, fs=5.6, tc="white",
    style="round,pad=0.06")
ax.text(17.2, 14.4, "layers 2–6 also attend propagated temporal queries",
        fontsize=6.0, color="#7A8694", ha="left")

# ══ 우: task heads 그룹 ════════════════════════════════════════
ax.add_patch(FancyBboxPatch((78.2, 11.0), 21.0, 22.4, boxstyle="round,pad=0.10",
                            fc="#F6FAF6", ec="#A8C8A8", lw=0.8))
ax.text(88.7, 31.9, "task heads", fontsize=6.6, color="#4E7A57", ha="center")
box(79.5, 25.4, 18.5, 5.2, "detection head\n(track = instance-bank IDs)", fs=6.6)
box(79.5, 18.6, 18.5, 5.2, "map head\nvectorized elements", fs=6.6)
box(79.5, 11.8, 18.5, 5.2, "motion · planning head\n(stage 2)", fs=6.6)
ax.text(75.6, 30.0, "updated\nqueries\n$N{\\times}256$", fontsize=6.0, color="#7A8694", ha="center")
for yy in (28.0, 21.2, 14.4):
    arrow(72.8, 21.7, 79.3, yy, lw=0.9)

# ══ 하단: 언어 경로 (제안) ═════════════════════════════════════
ax.text(7, 5.6, "language pathway\n(ours)", fontsize=7.4, style="italic",
        color=OR_E, ha="center")
box(23.8, 1.0, 31.7, 9.6, "", fc="#FFFDF8", ec="#D9A45B", lw=0.9, ls=(0, (4, 2)))
ax.text(25.0, 9.55, "offline — once per sample, no VLM in training",
        fontsize=6.4, color="#9A6A28", ha="left")
# 좌측 입력의 같은 6-view 영상을 언어 경로로 우회 공급 (리사이즈만 다름)
ax.plot([1.9, 0.6], [40.0, 40.0], color=A_LANG, lw=0.9)
ax.plot([0.6, 0.6], [40.0, 3.2], color=A_LANG, lw=0.9)
ax.plot([0.6, 24.0], [3.2, 3.2], color=A_LANG, lw=0.9)
arrow(24.0, 3.2, 25.1, 3.55, color=A_LANG, lw=0.9)
ax.text(12.3, 3.9, "same 6 views, resized $200{\\times}28{\\times}28$/cam",
        fontsize=5.6, color=A_LANG, ha="center")
box(25.2, 2.8, 10.8, 5.2, "", fc=PU_F, ec=PU_E, lw=1.4)
box(25.7, 3.25, 9.8, 4.3, "❄ Qwen2-VL-2B\n(frozen VLM,\n28 layers)", fc=PU_F,
    ec=PU_E, lw=0.6, fs=6.0)
ax.text(39.6, 1.72, "fixed prompt: \u201cobjects, lanes, traffic state \u2026\u201d",
        ha="center", va="center", fontsize=5.6, color="#7A8694")
# 은닉 토큰열 시각화 (보라 = 시각 토큰, 주황 = 프롬프트 텍스트 토큰) + 캐시 점선 상자
box(37.3, 2.6, 14.6, 5.7, "", fc="white", ec=GR_E, lw=0.8, ls=(0, (3, 2)))
for i in range(34):
    bx = 38.0 + i * 0.39
    if i < 28:
        col, al = PU_E, 0.35 + ((i * 7) % 9) / 14.0
    else:
        col, al = "#C9741F", 0.45 + ((i * 5) % 7) / 12.0
    ax.add_patch(Rectangle((bx, 4.25), 0.27, 2.35, fc=col, ec="none", alpha=al))
ax.text(44.6, 7.35, "tokens $H$ (layer 18)",
        fontsize=5.8, color=TXT, ha="center")
ax.text(44.6, 3.45, "$1159{\\times}1536$, bf16 $\\to$ cache",
        fontsize=5.4, color="#7A8694", ha="center")
arrow(36.2, 5.4, 37.1, 5.4, color=A_LANG)
box(57.0, 2.2, 9.5, 5.8, "2-layer MLP\n$1536\\!\\to\\!512\\!\\to\\!256$",
    fc=OR_F, ec=OR_E, lw=1.3, fs=6.2)
arrow(52.1, 5.2, 56.8, 5.2, color=A_LANG)

# ── 융합 상세 패널: 융합의 실제 계산 단계 ──
ax.add_patch(FancyBboxPatch((68.5, 0.8), 31.0, 10.0, boxstyle="round,pad=0.10",
                            fc="#FFFDF8", ec="#D9A45B", lw=0.8))
ax.text(98.9, 10.35, "fusion detail", fontsize=6.0, color="#9A6A28", ha="right")
ax.text(98.9, 9.72, "(per decoder layer)", fontsize=5.4, color="#9A6A28", ha="right")

seg_bar(69.3, 7.6, 6.4, 1.3, "queries $F$: $N{\\times}256$", 73.0)
arrow(75.9, 8.25, 77.4, 8.25, color=A_GEO)
ax.text(76.6, 8.9, "Q", fontsize=6.0, color=TXT, ha="center")

box(77.6, 6.8, 8.6, 2.9, "multi-head\ncross-attn",
    fc=OR_F, ec=OR_E, lw=1.2, fs=6.2)
arrow(86.4, 8.25, 86.9, 8.25, color=A_LANG)
ax.add_patch(plt.Circle((87.5, 8.25), 0.58, fc="white", ec=OR_E, lw=1.0))
ax.text(87.5, 8.25, "$\\odot$", ha="center", va="center", fontsize=5.0, color=TXT)

# 게이트 g — 6계층 x 3과제 격자 (열 색 = 과제 색, 행 = 디코더 계층)
for r in range(6):
    for c, gc in enumerate((Q_DET, Q_MAP, Q_PLAN)):
        al = 0.22 + ((r * 3 + c) * 5 % 8) / 13.0
        ax.add_patch(Rectangle((88.4 + c*0.9, 6.95 + r*0.4333), 0.9, 0.4333,
                               fc=gc, ec="white", lw=0.3, alpha=al))
ax.add_patch(Rectangle((88.4, 6.95), 2.7, 2.6, fc="none", ec=OR_E, lw=1.2))
ax.text(89.75, 6.3, "$g$: $6{\\times}3{\\times}256$, init $\\mathbf{0}$",
        fontsize=5.6, color=TXT, ha="center")
arrow(91.3, 8.25, 92.5, 8.25, color=A_LANG)
ax.add_patch(plt.Circle((93.4, 8.25), 0.85, fc="white", ec="#3B4653", lw=1.0))
ax.text(93.4, 8.2, "+", ha="center", va="center", fontsize=7.5, color=TXT)

# 잔차 스킵 — 곡선으로 상단 우회
ax.add_patch(FancyArrowPatch((73.5, 8.95), (93.3, 9.15), arrowstyle="-|>",
                             mutation_scale=7, lw=0.8, color=A_GEO,
                             connectionstyle="arc3,rad=-0.22"))
ax.text(83.0, 10.55, "residual $F$", fontsize=5.4, color="#4B5866", ha="center")

arrow(94.3, 8.25, 95.3, 8.25, color="#3B4653")
seg_bar(95.4, 7.6, 3.7, 1.3, "$F'$", 97.25)

arrow(66.6, 4.4, 81.2, 6.6, lw=1.0, color=A_LANG)
ax.text(73.4, 4.05, "K, V: $1159{\\times}256$", fontsize=6.2, color=A_LANG, ha="center")
ax.text(84.0, 2.75,
        "$A{=}\\mathrm{softmax}(QK^{\\top}\\!/\\sqrt{d})\\,V$;   $F'{=}F+g\\odot A$,   $g\\in\\mathbb{R}^{6\\times3\\times256}$",
        fontsize=6.0, color=TXT, ha="center")
ax.text(84.0, 1.45,
        "per layer/task/channel gate;  $g\\!=\\!0$ $\\Rightarrow$ unfused forward",
        fontsize=5.8, color=TXT, ha="center")
ax.plot([49.3, 69.2], [17.9, 10.9], ls=(0, (3, 2)), lw=0.7, color="#C9A26B")
ax.plot([63.0, 75.5], [17.9, 10.9], ls=(0, (3, 2)), lw=0.7, color="#C9A26B")

fig.tight_layout(pad=0.2)
fig.savefig("docs/iv/figures/architecture.pdf")
fig.savefig("docs/iv/figures/architecture.png", dpi=220)
print("저장 완료")
