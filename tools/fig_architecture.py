"""수정(2026-10-14 v7): 첫 블록을 temporal & self-attn 으로, 시간 뱅크→어텐션 화살표 추가(코드 temp_gnn 반영).
수정(2026-10-12 v6): 대표 모델(hipad_nusc_laq) 반영 — 명령 조건 잘린 확산 계획 입력(주황, 쿼리 위)과 추가 파라미터 +1.32M(언어 융합 1.188M + 확산 시점 임베딩 0.132M, 실측).
수정(2026-10-09 v5): IV 논문 Fig.1 = 전체 구조 개괄(융합 세부는 Fig.2, tools/fig_fusion.py). 전면 재설계 — 사용자 지적(화살표 난잡·설명글 과다·글자 판독 불가) 반영.

  원칙: (1) 글자 최소 7pt(IEEE 2단 폭 7.16in 인쇄 기준), 모듈명만 남기고 설명문은 캡션으로,
        (2) 화살표는 수평·수직 직교 경로만, 교차 0, (3) 세 줄 구성 — 위: 영상 인코딩(기하),
        가운데: 통합 디코더 ×6, 아래: 언어 경로(오프라인). 색 규약: 파랑=HiP-AD 기반, 주황=ours,
        보라=동결 VLM, 회색 점선=오프라인 캐시. 대표모델 Run10 구조(18층 토큰·2층 MLP·계층×과제×채널 게이트).
  이전 v4 는 git 이력(tools/fig_architecture.py) 참조.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig, ax = plt.subplots(figsize=(7.16, 2.75))
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

BL_F, BL_E = "#DEEAF6", "#4A76A8"   # HiP-AD 기반(학습)
OR_F, OR_E = "#FDEBD3", "#C9741F"   # ours(학습)
PU_F, PU_E = "#EAE4F4", "#7E6BB5"   # 동결 VLM
GR_F, GR_E = "#F5F5F5", "#8A8A8A"   # 오프라인 캐시
TXT, GEO, LANG = "#1C2733", "#3C5A78", "#C06818"
Q_DET, Q_MAP, Q_PLAN = "#4A76A8", "#5E9E68", "#C97A96"
FS, FS2 = 8.0, 7.0                  # 모듈명 / 보조 글자


def box(x, y, w, h, label, fc=BL_F, ec=BL_E, lw=1.0, fs=FS, ls="-", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25", fc=fc, ec=ec,
                                lw=lw, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color=TXT, linespacing=1.15, fontweight="bold" if bold else "normal")


def path(pts, color=GEO, lw=1.1, ls="-"):
    """직교 꺾은선 + 마지막 구간 화살촉."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=lw, ls=ls,
            solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=9,
                                shrinkA=0, shrinkB=0))


# ── 입력 영상(nuScenes 6캠 중 3장 썸네일) ────────────────────────
IMGS = []
for cam, f in (("CAM_FRONT", "n015-2018-10-02-11-23-23+0800__CAM_FRONT__1538451148112460.jpg"),
               ("CAM_FRONT_LEFT", "n015-2018-11-14-19-52-02+0800__CAM_FRONT_LEFT__1542196583104844.jpg"),
               ("CAM_BACK", "n015-2018-08-03-12-54-49+0800__CAM_BACK__1533272191037525.jpg")):
    IMGS.append(plt.imread(f"data/nuscenes/samples/{cam}/{f}")[::6, ::6])
for i, im in zip((2, 1, 0), IMGS[::-1]):
    x0, y0, w, h = 2.2 + i * 0.9, 30.0 + i * 0.9, 8.4, 5.6
    ax.imshow(im, extent=(x0, x0 + w, y0, y0 + h), zorder=3 - i, interpolation="bilinear")
    ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec="#5B6B7A", lw=0.6, zorder=3.5 - i))
ax.text(7.3, 38.9, "6-view images", ha="center", va="center", fontsize=FS2, color=TXT)

# ── 위: 영상 인코딩 ─────────────────────────────────────────────
box(16.0, 30.5, 11.0, 6.0, "ResNet-50\n+ FPN")
box(31.0, 30.5, 11.0, 6.0, "multi-scale\nfeatures", fc="#EAF1F9")
path([(12.6, 33.5), (15.6, 33.5)])
path([(27.4, 33.5), (31.6, 33.5)])

# ── 가운데: 통합 디코더 ×6 ──────────────────────────────────────
ax.add_patch(FancyBboxPatch((17.0, 13.0), 61.5, 13.8, boxstyle="round,pad=0.3",
                            fc="#F3F7FB", ec="#9DB3C8", lw=0.9))
ax.text(18.0, 25.3, "unified decoder layer", fontsize=FS, color=TXT, va="center")
ax.text(77.6, 25.3, "×6", fontsize=FS + 1, color=TXT, va="center", ha="right", fontweight="bold")
Y, H = 15.0, 7.6
box(19.0, Y, 9.0, H, "temporal &\nself-attn", fs=FS2)  # 수정(2026-10-14): 코드 operation_order 의 temp_gnn+gnn(+inter_gnn) — 시간 뱅크 인스턴스를 키로 참조
box(31.0, Y, 11.0, H, "deformable\naggregation", fs=FS2)
box(45.0, Y, 6.0, H, "FFN")
box(54.0, Y, 11.5, H, "language\nfusion", fc=OR_F, ec=OR_E, lw=1.5, bold=True)
box(68.5, Y, 8.5, H, "split &\nrefine")
yc = Y + H / 2
for xa, xb in ((28.4, 30.6), (42.4, 44.6), (51.4, 53.6), (65.9, 68.1)):
    path([(xa, yc), (xb, yc)])
path([(36.5, 30.1), (36.5, Y + H + 0.4)])          # 특징 → 가변형 집약(수직)

# 쿼리 풀(좌) + 시간 뱅크
box(4.0, 16.0, 9.5, 6.4, "", fc="white", ec="#4B5866", lw=0.9)
ax.text(7.6, 20.6, "queries", ha="center", va="center", fontsize=FS, color=TXT)
for sx, sw, sc in ((0, 0.6, Q_DET), (0.6, 0.22, Q_MAP), (0.82, 0.18, Q_PLAN)):
    ax.add_patch(Rectangle((5.0 + sx * 7.5, 17.2), sw * 7.5, 1.5, fc=sc, ec="none"))
path([(13.9, yc), (18.6, yc)])
box(4.0, 8.6, 9.5, 4.6, "temporal\nbank", fc=GR_F, ec=GR_E, ls=(0, (3, 2)), fs=FS2)
path([(8.75, 13.6), (8.75, 15.6)], color="#6B7785")
path([(13.9, 10.9), (23.5, 10.9), (23.5, 14.6)], color="#6B7785")  # 수정(2026-10-14): 시간 뱅크 → temporal attention 키(2~6층)
# 수정(2026-10-12): 명령 조건 잘린 확산 — 3명령×16 k-평균 앵커를 t 로 오염시켜 계획 쿼리로(2단계)
box(3.4, 23.6, 10.7, 4.6, "cmd anchors\n3×16 + noise $t$", fc=OR_F, ec=OR_E, lw=1.3, fs=FS2)
path([(11.8, 23.2), (11.8, 18.9)], color=LANG)


# ── 오른쪽: 과제 헤드 ───────────────────────────────────────────
for yy, lab in ((27.0, "detection\n& tracking"), (19.0, "online map"), (11.0, "motion\n& planning")):  # 수정(2026-10-09): VLA 잔차 경로는 아래 점선
    box(84.0, yy - 2.6, 14.5, 5.2, lab)
path([(77.4, yc), (80.5, yc)])
ax.plot([80.5, 80.5], [11.0, 27.0], color=GEO, lw=1.1)
for yy in (27.0, 19.0, 11.0):
    path([(80.5, yy), (83.6, yy)])

# ── 아래: 언어 경로(오프라인, 학습 중 VLM 미실행) ────────────────
ax.plot([1.2, 1.2], [33.0, 4.2], color=LANG, lw=1.1)
ax.plot([1.2, 2.1], [33.0, 33.0], color=LANG, lw=1.1)
path([(1.2, 4.2), (15.6, 4.2)], color=LANG)
box(16.0, 1.4, 12.0, 5.6, "Qwen2-VL-2B\n(frozen)", fc=PU_F, ec=PU_E, lw=1.2)
box(31.5, 1.4, 12.5, 5.6, "VL tokens\n(offline cache)", fc=GR_F, ec=GR_E, ls=(0, (3, 2)), fs=FS2)
box(47.0, 1.4, 9.5, 5.6, "projector", fc=OR_F, ec=OR_E, lw=1.3)
path([(28.4, 4.2), (31.1, 4.2)], color=LANG)
path([(44.4, 4.2), (46.6, 4.2)], color=LANG)
path([(56.9, 4.2), (59.75, 4.2), (59.75, Y - 0.4)], color=LANG)
ax.text(60.6, 10.2, "K, V", fontsize=FS2, color=LANG, va="center", ha="left")

# 수정(2026-10-09): VLA 확장 — VLM 의 잔차 행동 토큰이 계획 헤드로(점선). 다른 화살표와 교차하지 않게 하단으로 우회.
ax.plot([22.0, 22.0], [1.0, 0.3], color=LANG, lw=1.0, ls=(0, (3, 2)))
ax.plot([22.0, 91.25], [0.3, 0.3], color=LANG, lw=1.0, ls=(0, (3, 2)))
path([(91.25, 0.3), (91.25, 8.0)], color=LANG)
ax.text(73.5, 1.15, "VLA: residual action tokens $\\Delta$", fontsize=FS2, color=LANG, ha="center", va="center")
# ── 범례(우상단) ────────────────────────────────────────────────
items = (("HiP-AD base", BL_F, BL_E, "-"), ("ours (+1.32M)", OR_F, OR_E, "-"),
         ("frozen VLM", PU_F, PU_E, "-"), ("offline cache", GR_F, GR_E, (0, (3, 2))))
for k, (lab, fc, ec, ls) in enumerate(items):
    xx, yy = 46.0 + (k % 2) * 19.0, 37.4 - (k // 2) * 3.4
    ax.add_patch(Rectangle((xx, yy - 0.9), 2.6, 1.8, fc=fc, ec=ec, lw=1.0, linestyle=ls))
    ax.text(xx + 3.4, yy, lab, fontsize=FS2, color=TXT, va="center")
for k, (lab, c) in enumerate((("det 900", Q_DET), ("map 100", Q_MAP), ("plan/ego", Q_PLAN))):
    xx = 46.0 + k * 12.7
    ax.add_patch(Rectangle((xx, 29.6), 2.6, 1.3, fc=c, ec="none"))
    ax.text(xx + 3.4, 30.25, lab, fontsize=FS2, color=TXT, va="center")

fig.tight_layout(pad=0.15)
fig.savefig("docs/iv/figures/architecture.pdf")
fig.savefig("docs/iv/figures/architecture.png", dpi=250)
print("저장 완료")
