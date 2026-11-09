# 수정(2026-10-16): 교사 생성 답 캐시 7 폴더 — 폴더별 토큰 위치(고정) 정의·검사 공용 모듈.
"""laq_cache 7 폴더의 토큰 배치.

정답 문법(data/vla_teacher/v2/grammar.txt, v2-sensor, object_speed=numeric)은 고정 길이 547 토큰이다:
  'Objects: '(3) + 8 칸 × 57 + '\\nMap: '(4) + 지도 68 + '\\nIntent: '(4) + 의도 2 + '\\nAction: '(4) + 행동 6 = 547.
칸(57 토큰) 안의 필드 배치(문법 OBJ := CLS SEC SN SN UN UN HEAD MOTION FUT FUT FUT RISK):
  detection = CLS 1 + SEC 1 + x 5 + y 5 + 길이 4 + 폭 4 + HEAD 1 = 21   (칸 안 0..20)
  tracking  = MOTION 5 (<static>|SEC + 속력 DD.D, 또는 <na><_>×4)     (칸 안 21..25)
  motion    = FUT × 3 = 30 (1/2/3 s 누적 변위 dx,dy)                  (칸 안 26..55)
  plan      = RISK 1 (TTC 구간)                                       (칸 안 56)
빈 칸('<none>' + '<_>'×56)도 같은 위치로 자른다(위치 고정 — 학생 쿼리가 칸 번호로 읽는다).
map    = 'Map: ' 뒤 68 토큰 전부(<free> 5 토큰 포함), plan 은 8 칸 RISK 8 + <free>DD.D 5 = 13 (지도 줄과 공유).
action = 의도 2 + 행동 6 = 8. 머리말('Objects: ', '\\nMap: ', '\\nIntent: ', '\\nAction: ') 15 토큰은 어느 폴더에도 넣지 않는다.

정답 위치는 '정답 시작 기준 색인'(0..546). 템플릿은 문법 단위 수와 토크나이저로 계산하고(answer_template),
문자 구간 → 토큰 구간 독립 계산(folder_token_positions_from_text)과 같아야 한다(전 표본 검사: smoke_laq_cache_cpu.py).
context = 현재 프레임 6 캠 영상 토큰(<|image_pad|> 6×180 = 1,080, 시퀀스 순서) + user 텍스트 토큰
          (마지막 <|vision_end|> 바로 뒤 ~ user 턴의 <|im_end|> 직전: 지시문·명령·자차 상태 줄 = 752, 전 표본 실측) = 1,832.
          system 문장·채팅 틀 토큰(<|im_start|> 등)·<|vision_start/end|>·이전 프레임 영상 토큰은 넣지 않는다.
"""
import re

FOLDERS = ("context", "detection", "tracking", "motion", "map", "plan", "action")
ANSWER_FOLDERS = FOLDERS[1:]
N_SLOTS = 8
SLOT = 57                       # common_v2.SLOT_FIELDS_NUMERIC
SLOT_FIELDS = (("detection", 0, 21), ("tracking", 21, 26), ("motion", 26, 56), ("plan", 56, 57))
MAP_N = 68                      # common_v2.MAP_LINE_FIELDS
FREE_N = 5                      # '<free>' + DD.D
INTENT_N, ACTION_N = 2, 6
HEADERS = ("Objects: ", "\nMap: ", "\nIntent: ", "\nAction: ")
EXPECTED_COUNTS = dict(detection=168, tracking=40, motion=240, map=68, plan=13, action=8)
ANSWER_TOKENS = 547
TOKENS_PER_CAM = 180
NUM_CAMS = 6

_UNIT = re.compile(r"<[^<>]+>|.", re.S)


def _n(tok, s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


def answer_template(tok):
    """정답 시작 기준 폴더별 토큰 색인(list[int], 오름차순). 머리말 토큰 수는 토크나이저로 잰다."""
    h = [_n(tok, x) for x in HEADERS]
    pos = {k: [] for k in ANSWER_FOLDERS}
    p = h[0]
    for k in range(N_SLOTS):
        s = p + k * SLOT
        for name, a, b in SLOT_FIELDS:
            pos[name] += list(range(s + a, s + b))
    p += N_SLOTS * SLOT + h[1]
    pos["map"] = list(range(p, p + MAP_N))
    pos["plan"] = sorted(pos["plan"] + list(range(p + MAP_N - FREE_N, p + MAP_N)))
    p += MAP_N + h[2]
    pos["action"] = list(range(p, p + INTENT_N))
    p += INTENT_N + h[3]
    pos["action"] += list(range(p, p + ACTION_N))
    p += ACTION_N
    if p != ANSWER_TOKENS:
        raise RuntimeError(f"템플릿 총 토큰 {p} != {ANSWER_TOKENS} (머리말 {h}) — 토크나이저/문법 불일치")
    for k, n in EXPECTED_COUNTS.items():
        if len(pos[k]) != n:
            raise RuntimeError(f"{k} 토큰 {len(pos[k])} != {n}")
    return pos


def folder_char_spans(text):
    """파싱 가능한 정답 문자열 → 폴더별 문자 구간 목록(문법 단위: <...> 특수토큰 1 단위, 그 외 글자 1 단위).
    parse_answer 가 통과한 문자열에만 쓴다(문법 단위 수가 맞지 않으면 ValueError)."""
    lines = text.split("\n")
    if len(lines) != 4:
        raise ValueError("줄 수 != 4")
    starts, p = [], 0
    for ln in lines:
        starts.append(p); p += len(ln) + 1
    heads = ("Objects: ", "Map: ", "Intent: ", "Action: ")

    def units(i):
        if not lines[i].startswith(heads[i]):
            raise ValueError(f"머리말 {heads[i]!r} 없음")
        base = starts[i] + len(heads[i])
        return [(base + m.start(), base + m.end()) for m in _UNIT.finditer(lines[i][len(heads[i]):])]
    uo, um, ui, ua = units(0), units(1), units(2), units(3)
    if (len(uo), len(um), len(ui), len(ua)) != (N_SLOTS * SLOT, MAP_N, INTENT_N, ACTION_N):
        raise ValueError(f"문법 단위 수 {(len(uo), len(um), len(ui), len(ua))}")
    sp = {k: [] for k in ANSWER_FOLDERS}
    for k in range(N_SLOTS):
        for name, a, b in SLOT_FIELDS:
            sp[name] += uo[k * SLOT + a:k * SLOT + b]
    sp["map"] = list(um)
    sp["plan"] = sorted(sp["plan"] + um[MAP_N - FREE_N:])
    sp["action"] = ui + ua
    return sp


def token_char_offsets(tok, ids):
    """생성 토큰 id 열 → (전체 문자열, 토큰별 [start, end)). 토큰별 복호를 이어 붙인 것이 전체 복호와 같아야 한다
    (다르면 None — 다바이트 문자 등으로 토큰 경계가 문자 경계와 어긋남)."""
    pieces = [tok.decode([int(i)], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in ids]
    full = tok.decode([int(i) for i in ids], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if "".join(pieces) != full:
        return full, None
    offs, p = [], 0
    for s in pieces:
        offs.append((p, p + len(s))); p += len(s)
    return full, offs


def folder_token_positions_from_text(text, offs):
    """문자 구간 → 토큰 색인(문법 단위 하나가 토큰 하나와 정확히 겹쳐야 한다 — 아니면 ValueError)."""
    sp = folder_char_spans(text)
    start_to_tok = {a: i for i, (a, b) in enumerate(offs)}
    out = {}
    for k, spans in sp.items():
        idx = []
        for a, b in spans:
            i = start_to_tok.get(a)
            if i is None or offs[i][1] != b:
                raise ValueError(f"{k}: 문자 구간 {a, b} 이 토큰 하나와 일치하지 않는다")
            idx.append(i)
        out[k] = idx
    return out


def expected_folder_texts(parsed):
    """common_v2.parse_answer 결과(구조) → 폴더별 기대 문자열(문자 구간 방식과 독립인 형식 함수로 조립). 검사용."""
    import common_v2 as V
    f = V._fill
    t = {k: "" for k in ANSWER_FOLDERS}
    objs = parsed["objects"]
    for k in range(N_SLOTS):
        if k >= len(objs):
            t["detection"] += f("<none>", 21); t["tracking"] += "<_>" * 5; t["motion"] += "<_>" * 30; t["plan"] += "<_>"
            continue
        r = objs[k]
        t["detection"] += (V.CLS_TOK[r["name"]] + V.SEC_TOK[r["sector"]] + V._sn(r["lat"]) + V._sn(r["lon"]) +
                           V._un(r["length"]) + V._un(r["width"]) + ("<na>" if r["heading"] is None else V.HEAD_TOK[r["heading"]]))
        if r["motion"] == "na":
            t["tracking"] += f("<na>", 5)
        else:
            t["tracking"] += ("<static>" if r["motion"] == "static" else V.SEC_TOK[r["vel_sector"]]) + V._un(r["speed"])
        for p in r["fut"]:
            t["motion"] += f("<na>", 10) if p is None else V._sn(p[0]) + V._sn(p[1])
        t["plan"] += "<na>" if r["ttc"] is None else V.TTC_TOK[r["ttc"]]
    mp = parsed["map"]
    if mp is not None and mp["ped_crossing"] is not None and mp["ped_crossing"]["sector"] == "here":
        mp = dict(mp, ped_crossing=dict(dist=0.0, sector="here"))
    t["map"] = V.fmt_map(mp)[len("Map: "):]
    t["plan"] += "<_>" * 5 if mp is None else "<free>" + V._un(mp["free_dist"])
    it = parsed["intent"]
    t["action"] = ("<na><_>" if it is None else V.LON_TOK[it["longitudinal"]] + V.LAT_TOK[it["lateral"]])
    t["action"] += f("<na>", 6) if parsed["action"] is None else "".join(f"<action_{k}>" for k in parsed["action"])
    return t


def context_positions(ids, image_token_id, vision_end_id, im_end_id, cur_block, n_blocks=12):
    """한 행(패딩 포함 프롬프트 id 리스트) → context 위치 리스트(시퀀스 순서).
    cur_block = 현재 프레임 영상 블록 번호 범위(prev_first 면 6..11). 블록 = 연속된 <|image_pad|> 묶음."""
    ids = [int(x) for x in ids]
    blocks, i = [], 0
    while i < len(ids):
        if ids[i] == image_token_id:
            j = i
            while j < len(ids) and ids[j] == image_token_id:
                j += 1
            blocks.append((i, j)); i = j
        else:
            i += 1
    if len(blocks) != n_blocks:
        raise RuntimeError(f"영상 블록 {len(blocks)} != {n_blocks}")
    img = []
    for b in cur_block:
        a, e = blocks[b]
        if e - a != TOKENS_PER_CAM:
            raise RuntimeError(f"영상 블록 {b} 토큰 {e - a} != {TOKENS_PER_CAM}")
        img += list(range(a, e))
    ve = max(k for k, x in enumerate(ids) if x == vision_end_id)
    ie = next(k for k in range(ve + 1, len(ids)) if ids[k] == im_end_id)
    return img + list(range(ve + 1, ie)), (ve + 1, ie)
