# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
"""NAVSIM 교사 캐시 7 폴더 토큰 배치 — laq_cache_layout.py(nuScenes, 수정 안 함)의 NAVSIM 판.

다른 점은 두 가지뿐이다:
  - 정답: Action 8 토큰(4 s) → 정답 549 토큰, action 폴더 = 의도 2 + 행동 8 = 10. 나머지 폴더(detection 168 · tracking 40 ·
    motion 240 · map 68 · plan 13)와 칸 안 필드 위치는 nuScenes 와 같다.
  - context: 현재 프레임 전방 3 캠 영상 토큰(3 × 120 = 360, Qwen 128·28·28 → 224×420 → 캠당 120) + user 텍스트 토큰.
    영상 블록 = 직전 프레임 3 + 현재 3 = 6 블록, 현재 = 블록 3·4·5.
CLI: python -I tools/vla_teacher/laq_cache_layout_navsim.py --jsonl <teacher_gt_*.jsonl> [--n 2000]
  → 표본마다 템플릿 위치 == 문자 구간→토큰 위치(독립 계산) 인지 검사(smoke_laq_cache_cpu.py 의 NAVSIM 판).
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import laq_cache_layout as LL  # noqa: E402

FOLDERS = LL.FOLDERS
ANSWER_FOLDERS = LL.ANSWER_FOLDERS
N_SLOTS, SLOT, SLOT_FIELDS = LL.N_SLOTS, LL.SLOT, LL.SLOT_FIELDS
MAP_N, FREE_N, INTENT_N = LL.MAP_N, LL.FREE_N, LL.INTENT_N
ACTION_N = 8
HEADERS = LL.HEADERS
EXPECTED_COUNTS = dict(LL.EXPECTED_COUNTS, action=INTENT_N + ACTION_N)
ANSWER_TOKENS = LL.ANSWER_TOKENS + (ACTION_N - LL.ACTION_N)      # 549
TOKENS_PER_CAM = 120
NUM_CAMS = 3
N_IMAGE_BLOCKS = 2 * NUM_CAMS
CUR_BLOCKS = tuple(range(NUM_CAMS, 2 * NUM_CAMS))


def answer_template(tok):
    h = [LL._n(tok, x) for x in HEADERS]
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
        raise RuntimeError(f"템플릿 총 토큰 {p} != {ANSWER_TOKENS} (머리말 {h})")
    for k, n in EXPECTED_COUNTS.items():
        if len(pos[k]) != n:
            raise RuntimeError(f"{k} 토큰 {len(pos[k])} != {n}")
    return pos


def folder_char_spans(text):
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
        return [(base + m.start(), base + m.end()) for m in LL._UNIT.finditer(lines[i][len(heads[i]):])]
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


def folder_token_positions_from_text(text, offs):
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


def context_positions(ids, image_token_id, vision_end_id, im_end_id):
    return LL.context_positions(ids, image_token_id, vision_end_id, im_end_id, CUR_BLOCKS, n_blocks=N_IMAGE_BLOCKS) \
        if TOKENS_PER_CAM == LL.TOKENS_PER_CAM else _context_positions(ids, image_token_id, vision_end_id, im_end_id)


def _context_positions(ids, image_token_id, vision_end_id, im_end_id):
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
    if len(blocks) != N_IMAGE_BLOCKS:
        raise RuntimeError(f"영상 블록 {len(blocks)} != {N_IMAGE_BLOCKS}")
    img = []
    for b in CUR_BLOCKS:
        a, e = blocks[b]
        if e - a != TOKENS_PER_CAM:
            raise RuntimeError(f"영상 블록 {b} 토큰 {e - a} != {TOKENS_PER_CAM}")
        img += list(range(a, e))
    ve = max(k for k, x in enumerate(ids) if x == vision_end_id)
    ie = next(k for k in range(ve + 1, len(ids)) if ids[k] == im_end_id)
    return img + list(range(ve + 1, ie)), (ve + 1, ie)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--codebook", required=True)     # 수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건 (4)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    import navsim_teacher_common as N
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(_HERE, "..", "..", "ckpts/qwen2-vl-2b"))
    N.configure(a.codebook, a.thresholds)   # 수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건 (4)
    tok.add_tokens(N.SPECIAL_TOKENS, special_tokens=True)
    tpl = answer_template(tok)
    n = bad = 0
    with open(a.jsonl) as f:
        for ln in f:
            r = json.loads(ln)
            enc = tok(r["answer"], return_offsets_mapping=True, add_special_tokens=False)
            if len(enc["input_ids"]) != ANSWER_TOKENS:
                bad += 1
            else:
                got = folder_token_positions_from_text(r["answer"], enc["offset_mapping"])
                bad += any(got[k] != tpl[k] for k in ANSWER_FOLDERS)
            n += 1
            if n >= a.n:
                break
    print(json.dumps(dict(samples=n, mismatch=bad, answer_tokens=ANSWER_TOKENS,
                          counts={k: len(v) for k, v in tpl.items()}, RESULT="PASS" if bad == 0 else "FAIL")))


if __name__ == "__main__":
    main()
