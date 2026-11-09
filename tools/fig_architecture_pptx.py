"""수정(2026-09-17): Fig.1 을 개별 편집 가능한 PowerPoint 로 변환.

  tools/fig_architecture.py(v4, 큰글자판)의 좌표·색을 그대로 옮겨 모든 요소를
  PPT 네이티브 도형(사각형·화살표·텍스트·이미지)으로 재구성한다.
  슬라이드 13.33in x 6.33in (1 unit = 0.1333in), 글자 = matplotlib pt x1.878.
  출력: docs/iv/figures/architecture_editable.pptx
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
import copy

U = 13.33 / 100.0          # 1 좌표단위 = 0.1333 in
H = 47.5                    # 캔버스 높이(단위)
FS = 1.878                  # 글자 배율

def IN(v): return Inches(v * U)
def TY(y, h=0):             # matplotlib y(상향) → ppt y(하향)
    return Inches((H - y - h) * U)

def C(hex_or_none):
    if hex_or_none is None: return None
    return RGBColor.from_string(hex_or_none.lstrip("#"))

def blend(hex_c, alpha):    # 흰 배경과 알파 블렌딩
    r, g, b = (int(hex_c[i:i+2], 16) for i in (1, 3, 5))
    f = lambda v: int(round(v * alpha + 255 * (1 - alpha)))
    return "#%02X%02X%02X" % (f(r), f(g), f(b))

BL_F, BL_E = "#DEEAF6", "#4A76A8"
OR_F, OR_E = "#FDEBD3", "#C9741F"
PU_F, PU_E = "#EAE4F4", "#7E6BB5"
GR_F, GR_E = "#F5F5F5", "#8A8A8A"
TXT = "#1C2733"
A_GEO, A_LANG = "#3C5A78", "#C06818"
Q_DET, Q_MAP, Q_PLAN = "#4A76A8", "#5E9E68", "#C97A96"
GRAY = "#7A8694"

prs = Presentation()
prs.slide_width = Inches(13.33); prs.slide_height = Inches(H * U)
slide = prs.slides.add_slide(prs.slide_layouts[6])
SH = slide.shapes

def _style_line(shape, ec, lw, dash=None):
    if ec is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = C(ec); shape.line.width = Pt(lw * 0.75)
        if dash:
            ln = shape.line._get_or_add_ln()
            d = ln.makeelement(qn("a:prstDash"), {"val": dash})
            ln.append(d)

def box(x, y, w, h, text="", fc=BL_F, ec=BL_E, lw=1.0, fs=6.8, tc=TXT,
        dash=None, rounded=True, bold=False):
    shp = SH.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                       IN(x), TY(y, h), IN(w), IN(h))
    if rounded:
        try: shp.adjustments[0] = 0.10
        except Exception: pass
    if fc is None: shp.fill.background()
    else: shp.fill.solid(); shp.fill.fore_color.rgb = C(fc)
    _style_line(shp, ec, lw, dash)
    shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    lines = text.split("\n") if text else []
    for i, ln_txt in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln_txt; p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(round(fs * FS)); p.font.color.rgb = C(tc)
        p.font.bold = bold; p.font.name = "Arial"
    return shp

def text(x, y, s, fs, tc=TXT, ha="center", italic=False, bold=False, w=None):
    est_w = w if w else max(0.6, len(s) * fs * 0.115)
    if ha == "center": left = x - est_w / 2
    elif ha == "left": left = x
    else: left = x - est_w
    lines = s.split("\n")
    est_h = 0.95 * len(lines) * fs / 6.0 + 0.4
    tb = SH.add_textbox(IN(left), TY(y + est_h/2 - est_h/2, est_h/2)- Emu(0), IN(est_w), IN(est_h))
    tb.top = Inches((H - y - est_h/2) * U)
    tf = tb.text_frame; tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, ln_txt in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln_txt
        p.alignment = {"center": PP_ALIGN.CENTER, "left": PP_ALIGN.LEFT,
                       "right": PP_ALIGN.RIGHT}[ha]
        p.font.size = Pt(round(fs * FS)); p.font.color.rgb = C(tc)
        p.font.italic = italic; p.font.bold = bold; p.font.name = "Arial"
    return tb

def _arrowhead(conn):
    ln = conn.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)

def arrow(x1, y1, x2, y2, color=A_GEO, lw=1.0, dash=None, head=True, curve=False):
    kind = MSO_CONNECTOR.CURVE if curve else MSO_CONNECTOR.STRAIGHT
    conn = SH.add_connector(kind, IN(x1), Inches((H - y1) * U), IN(x2), Inches((H - y2) * U))
    conn.line.color.rgb = C(color); conn.line.width = Pt(lw * 0.9)
    if dash:
        ln = conn.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:prstDash"), {"val": dash}))
    if head: _arrowhead(conn)
    conn.shadow.inherit = False
    return conn

def circle(cx, cy, r, label, fc="#FFFFFF", ec=TXT, lw=1.0, fs=6.5):
    shp = SH.add_shape(MSO_SHAPE.OVAL, IN(cx - r), Inches((H - cy - r) * U), IN(2*r), IN(2*r))
    shp.fill.solid(); shp.fill.fore_color.rgb = C(fc); _style_line(shp, ec, lw)
    shp.shadow.inherit = False
    tf = shp.text_frame; tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = label; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(round(fs * FS)); p.font.color.rgb = C(TXT); p.font.name = "Arial"
    return shp

def seg_bar(x, y, w, h, lab, lx):
    for sx, sw, sc in ((0, 0.60, Q_DET), (0.60, 0.25, Q_MAP), (0.85, 0.15, Q_PLAN)):
        box(x + sx*w, y, sw*w, h, fc=blend(sc, 0.85), ec=None, rounded=False)
    box(x, y, w, h, fc=None, ec="#4B5866", lw=0.7, rounded=False)
    text(lx, y - 0.85, lab, 5.8)

# ══ 범례 ══
box(74.5, 35.3, 24.7, 10.6, fc="#FFFFFF", ec="#B9C2CC", lw=0.7)
def leg(y, txt, fc, ec, lw, dash=None, double=False, snow=False):
    box(75.6, y, 2.4, 1.3, fc=fc, ec=ec, lw=lw, dash=dash, rounded=False)
    if double: box(75.9, y+0.22, 1.8, 0.86, fc=None, ec=ec, lw=0.5, rounded=False)
    if snow: text(76.8, y+0.62, "❄", 4.6, PU_E)
    text(78.9, y+0.62, txt, 6.4, TXT, ha="left")
leg(43.7, "trainable — HiP-AD base", BL_F, BL_E, 0.9)
leg(41.7, "trainable — ours (+0.53M)", OR_F, OR_E, 1.3)
leg(39.7, "frozen — Qwen2-VL-2B (2B)", PU_F, PU_E, 1.0, double=True, snow=True)
leg(37.7, "offline data / cache (no grad.)", GR_F, GR_E, 0.8, dash="dash")
text(75.6, 36.3, "gradients never reach the frozen VLM", 6.0, GRAY, ha="left")

# ══ 상단 ══
text(1, 46.3, "geometry pathway (HiP-AD base)", 7.4, TXT, ha="left", italic=True)
import os as _os
_SCR = "/tmp/claude-0/-workspace-src/c5472e65-3761-46d4-be5b-41f40487e30c/scratchpad/thumbs"
IMGROOT = "data/nuscenes/samples"
IMGS = [f"{_SCR}/n015-2018-08-03-12-54-49+0800__CAM_BACK__1533272191037525.jpg",
        f"{_SCR}/n015-2018-11-14-19-52-02+0800__CAM_FRONT_LEFT__1542196583104844.jpg",
        f"{_SCR}/n015-2018-10-02-11-23-23+0800__CAM_FRONT__1538451148112460.jpg"]
for i, path in zip((2, 1, 0), IMGS):
    pic = SH.add_picture(path, IN(2 + i*0.9), Inches((H - (36.8 + i*0.75) - 6.0) * U),
                         IN(8.6), IN(6.0))
    pic.line.color.rgb = C("#5B6B7A"); pic.line.width = Pt(0.8)
text(7.2, 36.1, "6-view images", 6.6, "#4B5866")
text(7.2, 35.1, "each 3×448×896", 5.6, GRAY)
box(14.5, 36.8, 12, 7.2, "image encoder\nResNet-50 + FPN")
for i in (2, 1, 0):
    box(29.5 + i*1.1, 37.0 + i*0.8, 7.6, 5.6, fc="#EAF1F9", ec=BL_E, lw=0.7, rounded=False)
text(33.3, 39.8, "multi-scale\nfeatures", 6.6)
text(31.3, 35.4, "6 views · 4 scales\nC=256", 5.6, GRAY)
arrow(11.9, 40.4, 14.2, 40.4); arrow(26.7, 40.4, 29.2, 40.4)

# ══ 좌: 입력 그룹 ══
box(0.4, 11.6, 13.6, 33.9, fc=None, ec="#B9C2CC", lw=0.7)
text(1.1, 44.95, "inputs", 6.0, GRAY, ha="left")
box(0.9, 18.85, 12.9, 12.75, fc="#F3F7FC", ec=BL_E, lw=0.9)
text(7.25, 30.7, "task anchors & queries", 5.8)
text(7.25, 29.75, "k-means anchors:", 5.8, GRAY)
box(2.5, 27.6, 1.5, 1.0, fc=None, ec=Q_DET, lw=0.9, rounded=False)
box(2.9, 27.95, 1.5, 1.0, fc=None, ec=blend(Q_DET, 0.6), lw=0.6, rounded=False)
for dx0, dy0 in ((0, 0), (1.5, 0), (0, 1.0), (1.5, 1.0)):
    arrow(2.5+dx0, 27.6+dy0, 2.9+dx0, 27.95+dy0, color=blend(Q_DET, 0.6), lw=0.5, head=False)
pts = [(6.3, 27.7), (6.9, 28.8), (7.5, 27.9), (8.1, 28.8)]
for a, b in zip(pts, pts[1:]):
    arrow(a[0], a[1], b[0], b[1], color=Q_MAP, lw=1.2, head=False)
for px, py in pts:
    circle(px, py, 0.16, "", fc=Q_MAP, ec=Q_MAP, lw=0.3)
wpts = [(10.0, 27.7), (10.45, 28.05), (10.95, 28.35), (11.5, 28.55), (12.0, 28.7)]
for a, b in zip(wpts, wpts[1:]):
    arrow(a[0], a[1], b[0], b[1], color=Q_PLAN, lw=0.8, dash="dash", head=False)
for px, py in wpts:
    circle(px, py, 0.18, "", fc=Q_PLAN, ec=Q_PLAN, lw=0.3)
text(3.1, 27.0, "det 900", 4.6); text(7.25, 27.0, "map 100", 4.6); text(11.4, 27.0, "plan/ego", 4.6)
arrow(7.25, 26.55, 7.25, 26.3, lw=0.8)
box(1.65, 24.0, 11.2, 2.2, "anchor encoder (MLP)\n→ positional emb.", fs=5.5, lw=0.8)
arrow(10.35, 23.85, 10.35, 23.5, lw=0.8)
circle(10.35, 22.85, 0.6, "+", ec=BL_E, lw=0.9, fs=6.5)
text(9.7, 22.85, "learnable instance\nfeature (256-d)", 4.6, TXT, ha="right")
arrow(10.35, 22.2, 10.35, 21.6, lw=0.8)
seg_bar(2.6, 20.3, 9.3, 1.2, "queries F: N×256", 7.25)
arrow(13.7, 20.85, 16.8, 20.85)
box(1, 12.2, 12, 5.7, "temporal\ninstance bank\n(600 det propagated)",
    fc=GR_F, ec=GR_E, fs=5.8, dash="dash", lw=0.8)
arrow(7.0, 18.05, 7.0, 18.95)
arrow(17.5, 11.9, 13.3, 14.8, dash="dash", lw=0.8, color="#8A8A8A")
text(7, 11.05, "IDs → next frame", 6.0, GRAY)

# ══ 중앙: 디코더 ══
box(15.5, 12, 57.2, 20, fc="#F3F7FC", ec="#9DB6D0", lw=1.0)
text(17, 29.6, "unified decoder layer", 7.6, TXT, ha="left")
text(17.2, 26.35, "in: queries F (N×256)", 6.2, GRAY, ha="left")
arrow(69.5, 30.6, 67.1, 30.6, curve=True, lw=0.9)
text(70.6, 29.3, "×6", 7.8, TXT)
BLOCKS = [
    (17.0, 10.0, "self-attention\n(concat tasks)", BL_F, 1.0, BL_E),
    (29.5, 10.5, "task deformable\naggregation",   BL_F, 1.0, BL_E),
    (42.0,  5.0, "FFN",                            BL_F, 1.0, BL_E),
    (48.8, 14.6, "language-query fusion",          OR_F, 1.5, OR_E),
    (65.2,  6.5, "split &\nrefine",                BL_F, 1.0, BL_E),
]
for x, w, lab, fc, lw, ec in BLOCKS:
    box(x, 18, w, 7, lab, fc=fc, lw=lw, ec=ec, fs=6.5)
for x1, x2 in ((27.0, 29.5), (40.0, 42.0), (47.0, 48.8), (63.4, 65.2)):
    arrow(x1, 21.5, x2, 21.5, lw=0.9)
text(64.3, 22.15, "F′", 5.6, "#4B5866")
arrow(36.5, 36.7, 36.5, 25.4)
box(60.0, 24.2, 3.1, 1.7, "ours", fc=OR_E, ec=OR_E, fs=5.6, tc="#FFFFFF")
text(17.2, 14.4, "layers 2–6 also attend propagated temporal queries", 6.0, GRAY, ha="left")

# ══ 우: task heads ══
box(78.2, 11.0, 21.0, 22.4, fc="#F6FAF6", ec="#A8C8A8", lw=0.8)
text(88.7, 31.9, "task heads", 6.6, "#4E7A57")
box(79.5, 25.4, 18.5, 5.2, "detection head\n(track = instance-bank IDs)", fs=6.6)
box(79.5, 18.6, 18.5, 5.2, "map head\nvectorized elements", fs=6.6)
box(79.5, 11.8, 18.5, 5.2, "motion · planning head\n(stage 2)", fs=6.6)
text(75.6, 30.0, "updated\nqueries\nN×256", 6.0, GRAY)
for yy in (28.0, 21.2, 14.4):
    arrow(72.8, 21.7, 79.3, yy, lw=0.9)

# ══ 하단: 언어 경로 ══
text(7, 5.6, "language pathway\n(ours)", 7.4, OR_E, italic=True)
box(23.8, 1.0, 31.7, 9.6, fc="#FFFDF8", ec="#D9A45B", lw=0.9, dash="dash")
text(25.0, 9.55, "offline — once per sample, no VLM in training", 6.4, "#9A6A28", ha="left")
arrow(1.9, 40.0, 0.6, 40.0, color=A_LANG, lw=0.9, head=False)
arrow(0.6, 40.0, 0.6, 3.2, color=A_LANG, lw=0.9, head=False)
arrow(0.6, 3.2, 24.0, 3.2, color=A_LANG, lw=0.9, head=False)
arrow(24.0, 3.2, 25.1, 3.55, color=A_LANG, lw=0.9)
text(12.3, 3.9, "same 6 views, resized 200×28×28/cam", 5.6, A_LANG)
box(25.2, 2.8, 10.8, 5.2, fc=PU_F, ec=PU_E, lw=1.4)
box(25.7, 3.25, 9.8, 4.3, "❄ Qwen2-VL-2B\n(frozen VLM,\n28 layers)", fc=PU_F,
    ec=PU_E, lw=0.6, fs=6.0)
text(39.6, 1.72, "fixed prompt: “objects, lanes, traffic state …”", 5.6, GRAY)
box(37.3, 2.6, 14.6, 5.7, fc="#FFFFFF", ec=GR_E, lw=0.8, dash="dash")
for i in range(34):
    bx = 38.0 + i * 0.39
    col, al = (PU_E, 0.35 + ((i*7) % 9) / 14.0) if i < 28 else ("#C9741F", 0.45 + ((i*5) % 7) / 12.0)
    box(bx, 4.25, 0.27, 2.35, fc=blend(col, al), ec=None, rounded=False)
text(44.6, 7.35, "tokens H (layer 18)", 5.8)
text(44.6, 3.45, "1159×1536, bf16 → cache", 5.4, GRAY)
arrow(36.2, 5.4, 37.1, 5.4, color=A_LANG)
box(57.0, 2.2, 9.5, 5.8, "2-layer MLP\n1536→512→256", fc=OR_F, ec=OR_E, lw=1.3, fs=6.2)
arrow(52.1, 5.2, 56.8, 5.2, color=A_LANG)

# ══ 융합 상세 패널 ══
box(68.5, 0.8, 31.0, 10.0, fc="#FFFDF8", ec="#D9A45B", lw=0.8)
text(98.9, 10.35, "fusion detail", 6.0, "#9A6A28", ha="right")
text(98.9, 9.72, "(per decoder layer)", 5.4, "#9A6A28", ha="right")
seg_bar(69.3, 7.6, 6.4, 1.3, "queries F: N×256", 73.0)
arrow(75.9, 8.25, 77.4, 8.25); text(76.6, 8.9, "Q", 6.0)
box(77.6, 6.8, 8.6, 2.9, "multi-head\ncross-attn", fc=OR_F, ec=OR_E, lw=1.2, fs=6.2)
arrow(86.4, 8.25, 86.9, 8.25, color=A_LANG)
circle(87.5, 8.25, 0.58, "⊙", ec=OR_E, lw=1.0, fs=5.0)
for r in range(6):
    for c, gc in enumerate((Q_DET, Q_MAP, Q_PLAN)):
        al = 0.22 + ((r*3 + c) * 5 % 8) / 13.0
        box(88.4 + c*0.9, 6.95 + r*0.4333, 0.9, 0.4333, fc=blend(gc, al),
            ec="#FFFFFF", lw=0.3, rounded=False)
box(88.4, 6.95, 2.7, 2.6, fc=None, ec=OR_E, lw=1.2, rounded=False)
text(89.75, 6.3, "g: 6×3×256, init 0", 5.6)
arrow(91.3, 8.25, 92.5, 8.25, color=A_LANG)
circle(93.4, 8.25, 0.85, "+", ec="#3B4653", lw=1.0, fs=7.5)
arrow(73.5, 8.95, 93.3, 9.15, curve=True, lw=0.8)
text(83.0, 10.55, "residual F", 5.4, "#4B5866")
arrow(94.3, 8.25, 95.3, 8.25, color="#3B4653")
seg_bar(95.4, 7.6, 3.7, 1.3, "F′", 97.25)
arrow(66.6, 4.4, 80.8, 6.6, color=A_LANG, lw=1.0)
text(73.4, 4.05, "K, V: 1159×256", 6.2, A_LANG)
text(84.0, 2.75, "A = softmax(QKᵀ/√d)·V;   F′ = F + g⊙A,   g ∈ ℝ^(6×3×256)", 6.0)
text(84.0, 1.45, "per layer/task/channel gate;  g = 0 ⇒ unfused forward", 5.8)
arrow(49.3, 17.9, 69.2, 10.9, color="#C9A26B", lw=0.7, dash="dash", head=False)
arrow(63.0, 17.9, 75.5, 10.9, color="#C9A26B", lw=0.7, dash="dash", head=False)

prs.save("docs/iv/figures/architecture_editable.pptx")
print("저장: docs/iv/figures/architecture_editable.pptx")
