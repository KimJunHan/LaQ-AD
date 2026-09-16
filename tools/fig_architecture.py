"""수정(2026-09-15): IV 논문 Fig.1 — LaQ-AD 아키텍처 개요.

  [구성] 좌: 6-cam → 백본/FPN → 공유 쿼리 풀(det 900/map 100/plan·ego)과 6계층 디코더.
        우: Qwen2-VL-2B (동결) → 18층 hidden 오프라인 캐시 → 2층 MLP 사영 → cross-attn
        → (6계층x3과제x256) 영초기화 게이트 → 잔차 합.
  [형식] IEEE 2단 폭(7.1in) 벡터 PDF. 흑백 인쇄 대응 — 채도 낮은 회색조 + 해칭 구분.
  CPU 전용.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.5})
fig, ax = plt.subplots(figsize=(7.1, 3.15)); ax.set_xlim(0,100); ax.set_ylim(0,44); ax.axis("off")

def box(x,y,w,h,label,fc="0.93",ec="0.25",lw=0.9,fs=7.5,style="round,pad=0.12",tc="0.1",hatch=None):
    b=FancyBboxPatch((x,y),w,h,boxstyle=style,fc=fc,ec=ec,lw=lw,hatch=hatch)
    ax.add_patch(b); ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=fs,color=tc,linespacing=1.35)
def arrow(x1,y1,x2,y2,lw=1.0,color="0.2",style="-|>",ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=9,lw=lw,color=color,linestyle=ls))

# ── 좌: 기하 경로 (기반 모델) ─────────────────────────────
box(1.5,33,12,8,"6-view\ncameras",fc="0.97")
box(17,33,13,8,"ResNet-50\n+ FPN",fc="0.90")
arrow(13.7,37,16.8,37)
box(1.5,14,28.5,13,"",fc="0.965",ec="0.35")   # 디코더 컨테이너
ax.text(15.7,25.2,"shared decoder  ×6 layers",ha="center",fontsize=7.5,color="0.1")
box(3,19.5,11,4,"self-attn\n(query pool)",fc="0.90",fs=6.6)
box(16,19.5,12.5,4,"deformable\naggregation",fc="0.90",fs=6.6)
box(3,15.2,11,3.4,"FFN",fc="0.90",fs=6.6)
box(16,15.2,12.5,3.4,"anchor refine",fc="0.90",fs=6.6)
arrow(23.5,32.8,23.5,27.4)                      # FPN → deformable
ax.text(24.2,29.8,"multi-view\nfeatures",fontsize=6.2,color="0.35",ha="left")
box(1.5,3,28.5,7.5,"query pool\n900 det   /   100 map   /   plan·ego",fc="0.885",fs=7.2)
arrow(15.7,10.7,15.7,13.8); arrow(9,13.8,9,10.9,style="-|>")

# ── 우: 언어 경로 (제안) ──────────────────────────────────
box(63,33,15,8,"",fc="0.985",ec="0.2",lw=1.6)      # 동결 = 이중 테두리
box(63.6,33.5,13.8,7,"Qwen2-VL-2B\n(frozen)",fc="0.985",ec="0.45",lw=0.7)
box(81.5,33,17,8,"offline cache\nlayer-18 hidden\n$1159\\times1536$, bf16",fc="0.97",fs=6.8)
arrow(78.2,37,81.2,37)
ax.text(70.5,30.2,"run once per sample, never during training",ha="center",fontsize=6.0,color="0.35")
box(81.5,21.5,17,6.5,"2-layer MLP proj\n$1536\\to512\\to256$",fc="0.90",fs=6.8)
arrow(90,32.8,90,28.2)
box(63,21.5,15,6.5,"cross-attn\nQ: queries\nK,V: tokens",fc="0.90",fs=6.6)
arrow(81.3,24.7,78.2,24.7)
box(63,10.5,35.5,7,"zero-init gate   $g\\in\\mathbb{R}^{6\\times3\\times256}$,  $g^{(0)}=\\mathbf{0}$\n(decoder layer × task × channel)",fc="1.0",ec="0.1",lw=1.3,fs=7.0)
arrow(70.5,21.3,70.5,17.7)
# 쿼리 → cross-attn (Q), 게이트 출력 → 잔차 합
arrow(30.2,21.5,62.8,24.0,lw=0.9,ls=(0,(4,2)))
ax.text(46,24.1,"queries (Q)",fontsize=6.4,color="0.35",ha="center")
ax.add_patch(plt.Circle((46,6.7),1.7,fc="1.0",ec="0.1",lw=1.1)); ax.text(46,6.62,"+",ha="center",va="center",fontsize=11)
arrow(62.8,13.0,47.8,7.4,lw=1.1)
ax.text(56,11.5,"$g\\odot$attn",fontsize=6.6,color="0.1",ha="center")
arrow(30.2,6.7,44.1,6.7,lw=1.1); ax.text(37,7.6,"$F$",fontsize=7.5,color="0.1",ha="center")
arrow(46,4.95,46,2.4,lw=1.1); ax.text(46,0.9,"$F' = F + g\\odot\\mathrm{Attn}(F,\\,H)$   →   det · map · plan heads",ha="center",fontsize=7.3,color="0.1")

ax.text(15.7,42.8,"geometry pathway (HiP-AD base)",ha="center",fontsize=8,color="0.1",style="italic")
ax.text(80.5,42.8,"language pathway (ours)",ha="center",fontsize=8,color="0.1",style="italic")
ax.add_patch(plt.Rectangle((33.5,0.2),0.02,43.4,fc="0.75"))

fig.tight_layout(pad=0.25)
fig.savefig("docs/iv/figures/architecture.pdf")
fig.savefig("docs/iv/figures/architecture.png",dpi=220)
print("저장: docs/iv/figures/architecture.{pdf,png}")
