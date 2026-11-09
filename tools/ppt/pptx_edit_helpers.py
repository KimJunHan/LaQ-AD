# 수정(2026-11-09): 원래 생성기(make_laq_paper_pptx.py)가 클라우드 이관 때 옮겨지지 않아(tools/ppt 링크만 이관),
#   기존 발표 파일을 읽어 필요한 쪽만 바꾸는 편집 도우미를 새로 둔다. 서식(글꼴·크기·색)은 기존 런을 그대로 복제해 유지한다.
import copy

from pptx.util import Emu

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def _set_par_text(p_elm, text):
    """문단의 첫 런 서식을 유지한 채 텍스트만 교체(나머지 런·줄바꿈 제거)."""
    runs = p_elm.findall(A + "r")
    for extra in p_elm.findall(A + "br") + p_elm.findall(A + "fld"):
        p_elm.remove(extra)
    if not runs:
        r = p_elm.makeelement(A + "r", {})
        t = r.makeelement(A + "t", {})
        r.append(t)
        end = p_elm.find(A + "endParaRPr")
        if end is not None:
            rpr = copy.deepcopy(end)
            rpr.tag = A + "rPr"
            r.insert(0, rpr)
            end.addprevious(r)
        else:
            p_elm.append(r)
        runs = [r]
    for r in runs[1:]:
        p_elm.remove(r)
    runs[0].find(A + "t").text = text


def set_text(tf_owner, lines, bold=None):
    """shape(또는 cell) 의 text_frame 을 lines(문자열 또는 리스트)로 교체.
    문단 수가 늘면 마지막 문단을 복제, 줄면 뒤 문단 삭제. 문단별 서식 유지."""
    if isinstance(lines, str):
        lines = lines.split("\n")
    tf = tf_owner.text_frame
    body = tf._txBody
    ps = body.findall(A + "p")
    while len(ps) < len(lines):
        new = copy.deepcopy(ps[-1])
        ps[-1].addnext(new)
        ps = body.findall(A + "p")
    for extra in ps[len(lines):]:
        body.remove(extra)
    ps = body.findall(A + "p")
    for p_elm, line in zip(ps, lines):
        _set_par_text(p_elm, line)
        if bold is not None:
            for r in p_elm.findall(A + "r"):
                rpr = r.find(A + "rPr")
                if rpr is not None:
                    rpr.set("b", "1" if bold else "0")


def set_bold(cell_or_shape, bold=True):
    for r in cell_or_shape.text_frame._txBody.iter(A + "r"):
        rpr = r.find(A + "rPr")
        if rpr is None:
            continue
        rpr.set("b", "1" if bold else "0")
        for tag in ("latin", "ea"):
            f = rpr.find(A + tag)
            if f is not None and f.get("typeface", "").startswith("KoPubWorld"):
                f.set("typeface", "KoPubWorld돋움체 Bold" if bold else "KoPubWorld돋움체 Medium")


def set_fill(cell, rgb_hex):
    """표 칸 배경색(None = 채움 없음)."""
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("noFill", "solidFill"):
        for e in tcPr.findall(A + tag):
            tcPr.remove(e)
    if rgb_hex is None:
        e = tcPr.makeelement(A + "noFill", {})
    else:
        e = tcPr.makeelement(A + "solidFill", {})
        c = e.makeelement(A + "srgbClr", {"val": rgb_hex})
        e.append(c)
    # 선 요소(lnL..lnB) 뒤에 둔다
    tcPr.append(e)


def fill_table(shape, rows, template_row=None, row_h=None, bold_rows=(), fill_rows=None):
    """표를 rows(2차원 리스트, 머리 행 포함)로 다시 채움. 행 수가 다르면 template_row(기본 마지막 행)를 복제/삭제.
    열 수는 기존과 같아야 한다."""
    tbl = shape.table._tbl
    trs = tbl.tr_lst
    ncol = len(shape.table.columns)
    assert all(len(r) == ncol for r in rows), (ncol, [len(r) for r in rows])
    tmpl = trs[template_row if template_row is not None else len(trs) - 1]
    tmpl_copy = copy.deepcopy(tmpl)
    while len(tbl.tr_lst) < len(rows):
        tbl.tr_lst[-1].addnext(copy.deepcopy(tmpl_copy))
    while len(tbl.tr_lst) > len(rows):
        tbl.remove(tbl.tr_lst[-1])
    for i, (tr, vals) in enumerate(zip(tbl.tr_lst, rows)):
        if row_h is not None:
            tr.set("h", str(int(row_h)))
        for j, v in enumerate(vals):
            cell = shape.table.cell(i, j)
            set_text(cell, str(v))
            if i > 0:
                set_bold(cell, i in bold_rows)
                if fill_rows is not None:
                    set_fill(cell, fill_rows.get(i))
    shape.height = Emu(sum(int(tr.get("h")) for tr in tbl.tr_lst))


def set_col_widths(shape, widths_in):
    for col, w in zip(shape.table.columns, widths_in):
        col.width = Emu(int(w * 914400))
    shape.width = Emu(sum(int(w * 914400) for w in widths_in))


def shape_by_id(slide, sid):
    for sh in slide.shapes:
        if sh.shape_id == sid:
            return sh
    raise KeyError(sid)


def delete_shape(slide, sid):
    sh = shape_by_id(slide, sid)
    sh._element.getparent().remove(sh._element)


def delete_slide(prs, slide):
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst):
        if prs.part.related_part(sldId.rId) is slide.part:
            prs.part.drop_rel(sldId.rId)
            sldIdLst.remove(sldId)
            return
    raise KeyError("slide not found")


def move_slide(prs, slide, new_index):
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst):
        if prs.part.related_part(sldId.rId) is slide.part:
            sldIdLst.remove(sldId)
            sldIdLst.insert(new_index, sldId)
            return
    raise KeyError("slide not found")
