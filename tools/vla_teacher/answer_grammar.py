# 수정(2026-10-16): 설계 검토 반영 — v2-sensor 정답의 문법 제약 생성(위치별 허용 토큰 + 문법 상태).
"""정답 문법 제약 생성(교사 캐시 추출·GRPO 표집·val 평가 공용).

왜: 교사 캐시(extract_teacher_cache_v2.py)는 생성 답을 '고정 위치'(laq_cache_layout.answer_template)로 자른다.
제약 없이 생성하면 토큰 하나만 어긋나도 그 뒤 위치가 모두 다른 필드를 가리킨다. 그래서 생성 단계에서 위치마다
허용 토큰을 강제해 '위치 = 필드'를 구조적으로 보장한다.

허용 집합(위치 t = 정답 시작 기준 0..546, 547 = <|im_end|>):
  기본 집합 base[t] = TRAIN split GT 정답 전체(28,130)의 위치 t 토큰 합집합. 숫자 자리(위치 t 에 숫자 토큰이 한 번이라도
    나온 자리)는 숫자 0..9 전체를 더한다. 행동 자리(<action_k>) 도 같은 '값 자리'라 코드북 번호 전체(0..2040)를 더한다
    — TRAIN 에서 그 자리에 실제로 나온 번호만 허용하면(자리별 최대 1,804 종) val GT 6,019 중 260 개(4.3 %, 전부 행동
    자리 541..546)를 거부했다(2026-10-16 실측): 아직 안 나온 궤적 번호를 막는 것은 문법이 아니라 데이터 우연이다.
    val·hold-out 정답은 이 집합을 만들 때 쓰지 않는다(검사에만 쓴다).
  문법 상태(common_v2._parse_full 과 같은 규칙)로 다시 좁힌다 — 위치별 집합만으로는 '<none> 뒤 숫자' 같은 조합이
    파서에서 실패하므로:
    * 채움 강제: 칸 머리 <none> → 다음 56 위치 <_>, 운동 <na> → 4, 미래 <na> → 9, 지도 <na> → 67, <pc_none>/<pc_on> → 5,
      단면 값 <none> → 3, 의도 <na> → 1, 행동 <na> → 5. 강제 구간 밖에서는 <_> 금지.
    * 빈 칸 뒤 빈 칸: 한 칸이 <none> 이면 그 뒤 칸 머리는 <none> 만.
    * 부호 숫자 '-00.0' 금지(파서 정규식 (?!-00\\.0)): '-','0','0','.' 다음 마지막 숫자에서 '0' 제외.
  최종 허용 = base[t] ∩ 문법 집합. 정적 검사(빌드 때): base[t] ⊆ 문법 집합 ∪ {<_>}, 강제 구간을 여는 토큰이 허용된 위치는
    그 구간 모든 위치의 base 에 <_> 가 있음, 앞 칸 <none> 허용이면 뒤 칸 머리 base 에도 <none> 있음 → 도달 가능한 어떤
    상태에서도 허용 집합이 비지 않는다(실행 중에도 비면 즉시 예외 — 건너뛰지 않음).
보장: 이 제약 아래 생성된 547 토큰은 (i) 위치마다 문법 단위 하나(머리말 위치는 고정 토큰 하나)라 고정 위치 자르기가 항상
  맞고, (ii) common_v2.parse_answer 를 항상 통과한다(무작위 보행·소형 무작위 모델로 검사: smoke_laq_cache_cpu.py).
파일: data/vla_teacher/v2/answer_position_mask.json (토큰 문자열로 저장 — 토크나이저 id 배정과 무관하게 다시 읽는다,
  train 파일 sha256·문법 표 sha256 대조). 만들기·검사:
  CUDA_VISIBLE_DEVICES="" python tools/vla_teacher/answer_grammar.py --build   # train 으로 만들고 val/hold-out 100 % 수용 검사
  CUDA_VISIBLE_DEVICES="" python tools/vla_teacher/answer_grammar.py --check   # 다시 읽어 검사만
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common_v2 as V  # noqa: E402
import laq_cache_layout as LY  # noqa: E402

TAG = "수정(2026-10-16): 설계 검토 반영"
MASK_FILE = "data/vla_teacher/v2/answer_position_mask.json"
FORMAT_VERSION = 1
CONSTRAINT_NAME = "answer_grammar_v1"   # 캐시 _meta.json·색인·config teacher_meta_expect 가 쓰는 이름
END_TOKEN = "<|im_end|>"
FILL = "<_>"
DIGITS = tuple("0123456789")
N_ANSWER = LY.ANSWER_TOKENS          # 547
N_POS = N_ANSWER + 1                 # + <|im_end|>
_UNIT = re.compile(r"<[^<>]+>|.", re.S)


def _one(tok, s):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise RuntimeError(f"{s!r} 가 토큰 1 개가 아니다: {ids} (특수토큰 등록·토크나이저 확인)")
    return int(ids[0])


# ---------------------------------------------------------------- 문법 표(토크나이저 → 위치별 문법 집합·강제 규칙)
def grammar_table(tok):
    """반환 dict(cls=[frozenset]*N_POS, heads=[{id: 강제 길이}]*N_POS, slot_heads=[위치], sn_last={위치: 부호 위치},
    header=[위치], ids=dict(이름→id)). 순서·길이는 common_v2.build_answer / laq_cache_layout.answer_template 와 같다."""
    one = lambda s: _one(tok, s)  # noqa: E731
    I = dict(fill=one(FILL), none=one("<none>"), na=one("<na>"), end=one(END_TOKEN), plus=one("+"), minus=one("-"),
             dot=one("."), zero=one("0"), static=one("<static>"), pc=one("<pc>"), pc_on=one("<pc_on>"),
             pc_none=one("<pc_none>"), div=one("<div>"), bnd=one("<bnd>"), free=one("<free>"))
    D = frozenset(one(d) for d in DIGITS)
    CLS = frozenset(one(t) for t in V.CLS_TOK.values())
    SEC = frozenset(one(t) for t in V.SEC_TOK.values())
    HEAD = frozenset(one(t) for t in V.HEAD_TOK.values())
    TTC = frozenset(one(t) for t in V.TTC_TOK.values())
    LON = frozenset(one(t) for t in V.LON_TOK.values())
    LAT = frozenset(one(t) for t in V.LAT_TOK.values())
    ACT = frozenset(one(f"<action_{k}>") for k in range(V.ACTION_K))
    SIGN = frozenset((I["plus"], I["minus"]))
    cls, heads, header, slot_heads, sn_last = [], [], [], [], {}

    def put(c, h=None):
        cls.append(frozenset(c)); heads.append(dict(h or {}))

    def lit(text):
        for i in tok(text, add_special_tokens=False)["input_ids"]:
            header.append(len(cls)); put({int(i)})

    def sn(h_extra=None):
        s0 = len(cls)
        put(SIGN | set(h_extra or {}), h_extra); put(D); put(D); put({I["dot"]}); put(D)
        sn_last[len(cls) - 1] = s0

    def un(h_extra=None):
        put(D | set(h_extra or {}), h_extra); put(D); put({I["dot"]}); put(D)

    slot = LY.SLOT
    lit(LY.HEADERS[0])
    for _ in range(LY.N_SLOTS):
        slot_heads.append(len(cls))
        put(CLS | {I["none"]}, {I["none"]: slot - 1})             # OBJ | EMPTY(<none> + 56 <_>)
        put(SEC); sn(); sn(); un(); un()                          # 방위·x·y·길이·폭
        put(HEAD | {I["na"]})                                     # 진행방위 | <na>(채움 없음)
        put(SEC | {I["static"], I["na"]}, {I["na"]: 4}); put(D); put(D); put({I["dot"]}); put(D)   # MOTION
        for _f in range(3):
            sn({I["na"]: 9}); sn()                                # FUT = SN SN | <na> + 9 <_>
        put(TTC | {I["na"]})                                      # RISK
    lit(LY.HEADERS[1])
    put({I["na"], I["pc"], I["pc_on"], I["pc_none"]}, {I["na"]: LY.MAP_N - 1, I["pc_on"]: 5, I["pc_none"]: 5})
    put(SEC); un()                                                # <pc> SEC UN
    put({I["div"]}); put(D); put(D); put({I["bnd"]}); put(D); put(D)
    for y in V.LANE_Y:
        put({one(V.AT_TOK[y])})
        for _k in range(4):
            un({I["none"]: 3})                                    # LANE = UN | <none> + 3 <_>
    put({I["free"]}); un()
    lit(LY.HEADERS[2])
    put(LON | {I["na"]}, {I["na"]: 1}); put(LAT)
    lit(LY.HEADERS[3])
    put(ACT | {I["na"]}, {I["na"]: LY.ACTION_N - 1})
    for _k in range(LY.ACTION_N - 1):
        put(ACT)
    if len(cls) != N_ANSWER:
        raise RuntimeError(f"문법 표 길이 {len(cls)} != {N_ANSWER} — common_v2 문법·laq_cache_layout 과 불일치")
    put({I["end"]})
    # 템플릿(폴더 위치)과 머리말 위치가 서로 보완인지(머리말 15 = 어느 폴더에도 없는 위치)
    tpl = LY.answer_template(tok)
    covered = set().union(*[set(v) for v in tpl.values()])
    if sorted(set(range(N_ANSWER)) - covered) != header:
        raise RuntimeError("문법 표의 머리말 위치가 laq_cache_layout 템플릿과 다르다")
    if [slot_heads[k] - slot_heads[0] for k in range(LY.N_SLOTS)] != [k * slot for k in range(LY.N_SLOTS)]:
        raise RuntimeError("칸 머리 위치 간격 != 57")
    return dict(cls=cls, heads=heads, header=header, slot_heads=slot_heads, sn_last=sn_last, ids=I, digits=D, act=ACT)


def table_sha(tok, T):
    """문법 표 식별 해시(토큰 문자열 기준 — id 배정과 무관)."""
    s = tok.convert_ids_to_tokens
    doc = [[sorted(s(list(c))), sorted((s(k), v) for k, v in h.items())] for c, h in zip(T["cls"], T["heads"])]
    doc.append([T["slot_heads"], sorted(T["sn_last"].items())])
    return hashlib.sha256(json.dumps(doc, ensure_ascii=False).encode()).hexdigest()


# ---------------------------------------------------------------- 기본 집합(TRAIN GT)
def answer_ids(tok, answers):
    """GT 정답 문자열들 → id 행렬(list[list[int]]), 전부 547 이어야 한다."""
    out = []
    for s in range(0, len(answers), 1024):
        enc = tok(answers[s:s + 1024], add_special_tokens=False)["input_ids"]
        for i, ids in enumerate(enc):
            if len(ids) != N_ANSWER:
                raise RuntimeError(f"GT 정답 #{s + i} 토큰 {len(ids)} != {N_ANSWER}")
        out += [list(map(int, x)) for x in enc]
    return out


def build_base(tok, train_answers, T):
    ids = answer_ids(tok, train_answers)
    base = [set() for _ in range(N_ANSWER)]
    for row in ids:
        for t, i in enumerate(row):
            base[t].add(i)
    n_digit_pos = n_act_pos = 0
    for t in range(N_ANSWER):
        if base[t] & T["digits"]:
            base[t] |= set(T["digits"]); n_digit_pos += 1
        if base[t] & T["act"]:            # 행동 자리 = 코드북 번호(숫자와 같은 값 자리) → <action_0..2040> 전체
            base[t] |= set(T["act"]); n_act_pos += 1
    base.append({T["ids"]["end"]})
    return base, n_digit_pos, n_act_pos


# ---------------------------------------------------------------- 제약기
class State:
    __slots__ = ("t", "force_until", "empty", "hist")

    def __init__(self):
        self.t, self.force_until, self.empty, self.hist = 0, 0, False, []


class AnswerGrammar:
    """base(위치별 TRAIN 집합) ∩ 문법 상태. 키(key) = 상태가 고르는 허용 목록 이름:
    ('F',) <_> 강제, ('N',) <none> 강제(빈 칸 뒤 칸 머리), ('Z', t) 위치 t 에서 '0' 제외, ('B', t) 위치 t 기본."""

    def __init__(self, tok, base, T, info=None):
        self.T, self.info = T, dict(info or {})
        self.n_pos = N_POS
        self.fill, self.none, self.end, self.zero = T["ids"]["fill"], T["ids"]["none"], T["ids"]["end"], T["ids"]["zero"]
        self.minus, self.dot = T["ids"]["minus"], T["ids"]["dot"]
        self.slot_heads = set(T["slot_heads"])
        self.sn_last = dict(T["sn_last"])
        self.heads = T["heads"]
        self.base = [frozenset(b) for b in base]
        self.allow = []
        for t in range(N_POS):
            extra = self.base[t] - T["cls"][t] - {self.fill}
            if extra:
                raise RuntimeError(f"위치 {t}: TRAIN 토큰 {tok.convert_ids_to_tokens(sorted(extra))[:5]} 가 문법 밖 — 문법 표 오류")
            a = self.base[t] & T["cls"][t]
            self.allow.append(sorted(a))
        self.allow_nz = {t: [i for i in self.allow[t] if i != self.zero] for t in self.sn_last}
        self._static_checks(tok)
        self._tcache = {}

    def _static_checks(self, tok):
        for t in range(N_POS):
            if not self.allow[t] and t not in self._only_forced_positions():
                raise RuntimeError(f"위치 {t}: 허용 집합이 비었다")
            for i, n in self.heads[t].items():
                if i in self.allow[t]:
                    bad = [u for u in range(t + 1, t + 1 + n) if self.fill not in self.base[u]]
                    if bad:
                        raise RuntimeError(f"위치 {t} 의 {tok.convert_ids_to_tokens(i)} 가 강제할 위치 {bad[:3]} 에 TRAIN <_> 없음")
        sh = sorted(self.slot_heads)
        for a, b in zip(sh, sh[1:]):
            if self.none in self.allow[a] and self.none not in self.allow[b]:
                raise RuntimeError(f"칸 머리 {a} 에 <none> 허용인데 뒤 칸 머리 {b} 에는 없다")
        for t, lst in self.allow_nz.items():
            if not lst:
                raise RuntimeError(f"위치 {t}: '0' 제외 뒤 허용 집합이 비었다")

    def _only_forced_positions(self):
        """모든 TRAIN 표본에서 강제 채움이었던 위치(비강제 허용 집합이 빈 것이 정상). base == {<_>}."""
        if not hasattr(self, "_of"):
            self._of = {t for t in range(N_POS) if self.base[t] == {self.fill}}
        return self._of

    # ---- 상태
    def start(self):
        return State()

    def key(self, st):
        t = st.t
        if t >= N_POS:
            raise IndexError(f"정답 위치 {t} ≥ {N_POS} (문법 고정 길이 초과)")
        if t < st.force_until:
            return ("F",)
        if st.empty and t in self.slot_heads:
            return ("N",)
        if t in self.sn_last and st.hist[-4:] == [self.minus, self.zero, self.zero, self.dot]:
            return ("Z", t)                       # '-00.' 뒤 마지막 숫자: '0' 금지(파서 (?!-00\.0))
        return ("B", t)

    def allowed_list(self, key):
        if key[0] == "F":
            return [self.fill]
        if key[0] == "N":
            return [self.none]
        lst = self.allow_nz[key[1]] if key[0] == "Z" else self.allow[key[1]]
        if not lst:
            raise RuntimeError(f"허용 집합이 비었다: {key} — 문법·기본 집합 점검(건너뛰지 않음)")
        return lst

    def step(self, st, i):
        """토큰 i 를 받아 상태 갱신. 허용 밖이면 ValueError."""
        i = int(i)
        k = self.key(st)
        if i not in self._set(k):
            raise ValueError(f"위치 {st.t}: 토큰 {i} 가 허용 집합({k}) 밖")
        t = st.t
        if k[0] != "F":                           # 'N'(빈 칸 뒤 <none>) 도 채움 56 을 연다
            n = self.heads[t].get(i)
            if n:
                st.force_until = t + 1 + n
            if t in self.slot_heads and i == self.none:
                st.empty = True
        st.hist.append(i)
        st.t += 1

    def _set(self, k):
        c = self._tcache.setdefault(("set",) + k, None)
        if c is None:
            c = frozenset(self.allowed_list(k))
            self._tcache[("set",) + k] = c
        return c

    def accepts(self, ids):
        """반환 (True, -1) 또는 (False, 처음 어긋난 위치)."""
        st = self.start()
        for t, i in enumerate(ids):
            try:
                self.step(st, i)
            except (ValueError, IndexError):
                return False, t
        return True, -1

    def replay_keys(self, ids):
        st, keys = self.start(), []
        for i in ids:
            keys.append(self.key(st)); self.step(st, i)
        return keys

    # ---- torch
    def key_tensor(self, key, device):
        import torch
        ck = ("t", key, str(device))
        x = self._tcache.get(ck)
        if x is None:
            x = torch.tensor(self.allowed_list(key), dtype=torch.long, device=device)
            self._tcache[ck] = x
        return x

    def gather_index(self, keys, device):
        """keys(list) → (idx (N,K) long, valid (N,K) bool). 빈 자리는 그 행 첫 허용 id 로 채우고 valid=False."""
        import torch
        ts = [self.key_tensor(k, device) for k in keys]
        K = max(int(x.numel()) for x in ts)
        idx = torch.empty((len(ts), K), dtype=torch.long, device=device)
        valid = torch.zeros((len(ts), K), dtype=torch.bool, device=device)
        for r, x in enumerate(ts):
            n = int(x.numel())
            idx[r, :n] = x; idx[r, n:] = x[0]; valid[r, :n] = True
        return idx, valid

    def mask_scores(self, scores, keys):
        """scores (B,V) → 허용 밖 −inf (HF LogitsProcessor 용). 채움 자리 id 는 그 행의 허용 id 라 값이 같아 무해."""
        import torch
        idx, _ = self.gather_index(keys, scores.device)
        out = torch.full_like(scores, float("-inf"))
        return out.scatter(1, idx, scores.gather(1, idx))


def make_logits_processor(grammar, prompt_len, batch):
    """HF generate 용 LogitsProcessor(왼쪽 패딩 배치: 모든 행의 프롬프트 길이 = prompt_len). 행마다 문법 상태를 들고
    매 step 직전 생성 토큰으로 갱신한 뒤 허용 밖을 −inf 로 만든다."""
    from transformers import LogitsProcessor

    class _P(LogitsProcessor):
        def __init__(self):
            self.states = [grammar.start() for _ in range(batch)]
            self.t = 0

        def __call__(self, input_ids, scores):
            t = int(input_ids.shape[1]) - prompt_len
            if t != self.t or input_ids.shape[0] != batch:
                raise RuntimeError(f"문법 제약기 위치 {t} != 기대 {self.t} 또는 배치 {input_ids.shape[0]} != {batch}")
            if t > 0:
                for b, st in enumerate(self.states):
                    grammar.step(st, input_ids[b, -1])
            self.t += 1
            return grammar.mask_scores(scores, [grammar.key(st) for st in self.states])

    return _P()


# ---------------------------------------------------------------- 저장·적재
def file_sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_grammar(tok, path=None, data_dir=None, check_train_sha=True):
    """저장 파일 → AnswerGrammar. 토큰 문자열 → id 왕복, 문법 표 sha256, (기본) 현재 TRAIN 파일 sha256 대조."""
    import sft_common as S
    p = Path(path or (S.REPO / MASK_FILE))
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음 — 먼저 python tools/vla_teacher/answer_grammar.py --build")
    d = json.load(open(p))
    if d.get("format_version") != FORMAT_VERSION:
        raise RuntimeError(f"{p}: format_version {d.get('format_version')} != {FORMAT_VERSION}")
    T = grammar_table(tok)
    if table_sha(tok, T) != d["grammar_table_sha256"]:
        raise RuntimeError(f"{p}: 문법 표 sha256 불일치(문법 코드 또는 토크나이저가 바뀌었다) — --build 다시")
    if check_train_sha:
        dd = Path(data_dir) if data_dir else S.resolve_data_dir(None)
        tf = Path(S.split_file(dd, "train"))
        if file_sha256(tf) != d["train_sha256"]:
            raise RuntimeError(f"{p}: TRAIN 파일 {tf} 의 sha256 이 만들 때와 다르다 — --build 다시")
    base = []
    for t, strs in enumerate(d["base_tokens"]):
        ids = tok.convert_tokens_to_ids(strs)
        if any(i is None or i == tok.unk_token_id for i in ids) or tok.convert_ids_to_tokens(ids) != strs:
            raise RuntimeError(f"{p}: 위치 {t} 토큰 문자열 → id 왕복 실패")
        base.append(set(map(int, ids)))
    if len(base) != N_POS:
        raise RuntimeError(f"{p}: 위치 수 {len(base)} != {N_POS}")
    g = AnswerGrammar(tok, base, T, info=dict(name=CONSTRAINT_NAME, path=str(p), sha256=file_sha256(p), train_sha256=d["train_sha256"],
                                               grammar_table_sha256=d["grammar_table_sha256"]))
    return g


def _tokenizer():
    import sft_common as S
    proc = S.make_processor(S.BASE_QWEN)
    tok = proc.tokenizer
    toks, _ = S.load_special_tokens(None, S.resolve_data_dir(None))
    S.add_special_tokens(tok, toks)
    return tok


def random_walk(g, n, seed=0):
    """허용 집합에서 균등 무작위로 n 개 정답 생성(모델 없이 문법 제약 자체 검사용)."""
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        st, ids = g.start(), []
        for _t in range(N_POS):
            i = rng.choice(g.allowed_list(g.key(st)))
            g.step(st, i); ids.append(i)
        out.append(ids)
    return out


def check(g, tok, recs_by_name, n_walk=2000):
    """반환 dict: 이름별 수용률, 무작위 보행 파싱률·위치 일치율."""
    res = {}
    for name, answers in recs_by_name.items():
        ids = answer_ids(tok, answers)
        bad = [(i, g.accepts(r + [g.end])[1]) for i, r in enumerate(ids) if not g.accepts(r + [g.end])[0]]
        res[name] = dict(n=len(ids), accepted=len(ids) - len(bad), first_bad=bad[:5])
    tpl = LY.answer_template(tok)
    walks = random_walk(g, n_walk)
    n_parse = n_pos = 0
    errs = []
    for ids in walks:
        a = ids[:N_ANSWER]
        text, offs = LY.token_char_offsets(tok, a)
        try:
            V.parse_answer(text); n_parse += 1
            pos = LY.folder_token_positions_from_text(text, offs) if offs is not None else None
            n_pos += int(pos is not None and all(pos[k] == tpl[k] for k in LY.ANSWER_FOLDERS))
        except ValueError as e:
            errs.append(str(e)[:120])
    res["random_walk"] = dict(n=n_walk, parse_ok=n_parse, positions_ok=n_pos, errors=errs[:3])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fold_file", default=None)
    ap.add_argument("--n_walk", type=int, default=2000)
    a = ap.parse_args()
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    import sft_common as S
    dd = S.resolve_data_dir(a.data_dir)
    out = Path(a.out) if a.out else S.REPO / MASK_FILE
    tok = _tokenizer()
    train = S.load_records(dd, "train")
    val = S.load_records(dd, "val")
    if a.build:
        T = grammar_table(tok)
        base, n_digit, n_act = build_base(tok, [r["answer"] for r in train], T)
        g = AnswerGrammar(tok, base, T)
        doc = dict(format_version=FORMAT_VERSION, created_by=TAG, created=datetime.datetime.now().isoformat(timespec="seconds"),
                   source="TRAIN split GT only (val·hold-out 미사용) + 숫자 자리에 0..9 전체 + 행동 자리에 <action_0..2040> 전체",
                   train_file=str(S.split_file(dd, "train")), train_sha256=file_sha256(S.split_file(dd, "train")),
                   n_train=len(train), n_digit_positions=n_digit, n_action_positions=n_act, n_positions=N_POS, end_token=END_TOKEN,
                   grammar_table_sha256=table_sha(tok, T),
                   allowed_sizes=[len(x) for x in g.allow],
                   base_tokens=[tok.convert_ids_to_tokens(sorted(b)) for b in base])
        tmp = out.with_name(out.name + ".tmp")
        json.dump(doc, open(tmp, "w"), ensure_ascii=False)
        os.replace(tmp, out)
        print(f"[build] {out} — TRAIN {len(train)} 정답, 숫자 자리 {n_digit}, 허용 크기 min {min(len(x) for x in g.allow)} "
              f"max {max(len(x) for x in g.allow)}", flush=True)
    g = load_grammar(tok, out, dd)
    d, _ = S.load_fold_split(a.fold_file or S.FOLD_FILE)
    hold = set(d["holdout"]["A"]["scenes"]) | set(d["holdout"]["B"]["scenes"])
    res = check(g, tok, {"train": [r["answer"] for r in train], "val": [r["answer"] for r in val],
                         "holdout_AB": [r["answer"] for r in train if r["scene_token"] in hold]}, a.n_walk)
    empty = sum(1 for t in range(N_POS) if not g.allow[t])
    res["positions_with_empty_nonforced_set"] = empty
    res["min_allowed_any_key"] = min(len(g.allowed_list(k)) for k in
                                     [("F",), ("N",)] + [("B", t) for t in range(N_POS) if g.allow[t]] + [("Z", t) for t in g.sn_last])
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    okv = all(res[k]["accepted"] == res[k]["n"] for k in ("train", "val", "holdout_AB"))
    okw = res["random_walk"]["parse_ok"] == res["random_walk"]["n"] == res["random_walk"]["positions_ok"]
    print("[check] " + ("OK" if okv and okw and res["min_allowed_any_key"] >= 1 else "FAIL"), flush=True)
    sys.exit(0 if okv and okw else 1)


if __name__ == "__main__":
    main()
