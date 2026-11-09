# 수정(2026-10-29): NAVSIM 교사 캐시 — 정답 문법 제약 생성의 NAVSIM 판(549 토큰). answer_grammar.py 는 수정하지 않는다.
"""NAVSIM 정답 문법 제약(교사 캐시 추출용).

answer_grammar.py(nuScenes, 547 토큰)의 규칙·코드를 그대로 쓰되, 그 파일이 모듈 수준에서 읽는 두 의존성만 NAVSIM 값으로 바꾼
'별도 모듈 사본'을 만든다(load_impl). nuScenes 의 sys.modules['answer_grammar'] 는 건드리지 않는다(사본 이름이 다르다).
  - V  (common_v2 자리): common_v2 의 모든 이름 + NAVSIM 덮어쓰기 3 개 — CLS_TOK(<czone>·<obj> 추가), ACTION_K(NAVSIM 코드북
        2,045; navsim_teacher_common.configure 뒤 값), parse_answer(navsim_teacher_common.parse_answer: 1–3 줄 = common_v2 문법,
        4 줄 = 행동 8). SEC/HEAD/TTC/LON/LAT/LANE_Y/AT_TOK 는 NAVSIM 생성기도 common_v2 것을 그대로 쓴다.
  - LY (laq_cache_layout 자리): laq_cache_layout 의 모든 이름 + laq_cache_layout_navsim 의 상수·함수(ANSWER_TOKENS 549,
        ACTION_N 8, answer_template, folder_token_positions_from_text, context_positions …). token_char_offsets 는 공용.
  결과: 위치 0..548 + <|im_end|> = 550 위치, 행동 줄 = <action_0..2044> 8 개 또는 <na> + <_> 7 개. 나머지 규칙(채움 강제, 빈 칸 뒤
  빈 칸, '-00.0' 금지, 숫자 자리 0..9 전체, 행동 자리 코드북 전체)은 nuScenes 와 같다.
기본 집합: NAVSIM TRAIN GT 전체(teacher_gt_navtrain.jsonl, navtrain 전 프레임 152,495 — nuScenes 가 train split 전체로 만든 것과 같은
  규칙). 검사(--build/--check): navtrain·navtest·hold-out(A∪B) GT 100 % 수용 + 무작위 보행 2,000 개 파싱·위치 100 %.
제약 이름은 학생 config(teacher_meta_expect)·로더(TEACHER_CONSTRAINT)가 요구하는 'answer_grammar_v1' 그대로(규칙이 같다) —
  NAVSIM 표 식별은 파일 sha256·grammar_table_sha256·variant='navsim_549' 로 한다.
사용(CPU):
  CUDA_VISIBLE_DEVICES="" python tools/vla_teacher/answer_grammar_navsim.py --build   # 만들고 검사
  CUDA_VISIBLE_DEVICES="" python tools/vla_teacher/answer_grammar_navsim.py --check   # 다시 읽어 검사만
"""
import argparse
import datetime
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common_v2 as C2  # noqa: E402
import laq_cache_layout as LL  # noqa: E402
import laq_cache_layout_navsim as LN  # noqa: E402
import navsim_sft_common as NS  # noqa: E402
import navsim_teacher_common as N  # noqa: E402

TAG = "수정(2026-10-29): NAVSIM 교사 캐시"
VARIANT = "navsim_549"
MASK_FILE = f"{NS.NAVSIM_ROOT}/navsim_grammar_20261029/answer_position_mask_navsim.json"
_LN_NAMES = ("FOLDERS", "ANSWER_FOLDERS", "N_SLOTS", "SLOT", "SLOT_FIELDS", "MAP_N", "FREE_N", "INTENT_N", "ACTION_N",
             "HEADERS", "EXPECTED_COUNTS", "ANSWER_TOKENS", "TOKENS_PER_CAM", "NUM_CAMS", "answer_template",
             "folder_char_spans", "folder_token_positions_from_text", "context_positions")
_IMPL = None


def configure(codebook=NS.NAVSIM_CODEBOOK, thresholds=NS.NAVSIM_THRESHOLDS):
    """navsim_teacher_common 을 NAVSIM 코드북·임계값으로 설정(정답 생성 때와 같은 값). 반환 ACTION_K."""
    N.configure(codebook, thresholds)
    return N.ACTION_K


def _views():
    V = types.ModuleType("common_v2__navsim_view")
    V.__dict__.update({k: v for k, v in vars(C2).items() if not k.startswith("__")})
    V.CLS_TOK, V.ACTION_K, V.parse_answer = dict(N.CLS_TOK), N.ACTION_K, N.parse_answer
    L = types.ModuleType("laq_cache_layout__navsim_view")
    L.__dict__.update({k: v for k, v in vars(LL).items() if not k.startswith("__")})
    for k in _LN_NAMES:
        setattr(L, k, getattr(LN, k))
    return V, L


def load_impl():
    """answer_grammar.py 를 NAVSIM 의존성(V, LY)으로 실행한 별도 모듈 사본. configure() 뒤에만(ACTION_K 가 NAVSIM 값)."""
    global _IMPL
    if _IMPL is not None:
        return _IMPL
    if N.CONFIG.get("codebook") == "nuscenes":
        raise RuntimeError("answer_grammar_navsim.load_impl: 먼저 configure() — 행동 코드북이 아직 nuScenes 값이다")
    V, L = _views()
    spec = importlib.util.spec_from_file_location("answer_grammar__navsim_impl", HERE / "answer_grammar.py")
    mod = importlib.util.module_from_spec(spec)
    saved = {k: sys.modules.get(k) for k in ("common_v2", "laq_cache_layout")}
    sys.modules["common_v2"], sys.modules["laq_cache_layout"] = V, L
    try:
        spec.loader.exec_module(mod)
    finally:                                   # 다른 모듈이 보는 sys.modules 는 즉시 원래대로
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    if mod.V is not V or mod.LY is not L or mod.N_ANSWER != LN.ANSWER_TOKENS or mod.N_POS != LN.ANSWER_TOKENS + 1:
        raise RuntimeError("answer_grammar NAVSIM 사본의 의존성 주입 실패")
    if sys.modules.get("common_v2") is V or sys.modules.get("laq_cache_layout") is L:
        raise RuntimeError("sys.modules 복원 실패")
    _IMPL = mod
    return mod


def navsim_tokenizer(base=None, special_tokens=None):
    """기본 Qwen 토크나이저 + NAVSIM 특수토큰(SFT 자료 special_tokens.txt — navsim_teacher_common.SPECIAL_TOKENS 와 같아야 함)."""
    import sft_common as S
    tok = S.make_processor(base or S.BASE_QWEN, NS.NAVSIM_PIXELS).tokenizer
    toks, _ = S.load_special_tokens(special_tokens or str(Path(NS.NAVSIM_SFT_DIR) / "special_tokens.txt"))
    if toks != list(N.SPECIAL_TOKENS):
        raise RuntimeError("SFT special_tokens.txt 가 navsim_teacher_common.SPECIAL_TOKENS(코드북 설정 뒤)와 다르다")
    S.add_special_tokens(tok, toks)
    return tok


def iter_gt(path, keys=("token", "log_name", "answer")):
    with open(path) as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                yield {k: r[k] for k in keys}


def build(tok, out=MASK_FILE, gt_dir=NS.NAVSIM_GT_DIR):
    M = load_impl()
    tf = Path(gt_dir) / NS.NAVSIM_SPLIT_FILES["train"]
    answers = [r["answer"] for r in iter_gt(tf)]
    T = M.grammar_table(tok)
    base, n_digit, n_act = M.build_base(tok, answers, T)
    g = M.AnswerGrammar(tok, base, T)
    doc = dict(format_version=M.FORMAT_VERSION, created_by=TAG, created=datetime.datetime.now().isoformat(timespec="seconds"),
               variant=VARIANT, constraint_name=M.CONSTRAINT_NAME,
               source=f"NAVSIM TRAIN GT 전체(navtrain 전 프레임, navtest·hold-out 미사용) + 숫자 자리 0..9 전체 + 행동 자리 "
                      f"<action_0..{N.ACTION_K - 1}> 전체",
               train_file=str(tf), train_sha256=M.file_sha256(tf), n_train=len(answers), n_digit_positions=n_digit,
               n_action_positions=n_act, n_positions=M.N_POS, end_token=M.END_TOKEN, action_k=N.ACTION_K,
               navsim_config=dict(N.CONFIG), grammar_table_sha256=M.table_sha(tok, T),
               allowed_sizes=[len(x) for x in g.allow], base_tokens=[tok.convert_ids_to_tokens(sorted(b)) for b in base])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    json.dump(doc, open(tmp, "w"), ensure_ascii=False)
    os.replace(tmp, out)
    print(f"[build] {out} — TRAIN {len(answers)} 정답, 숫자 자리 {n_digit}, 행동 자리 {n_act}, 허용 크기 "
          f"min {min(len(x) for x in g.allow)} max {max(len(x) for x in g.allow)}", flush=True)
    return out


def load(tok, path=MASK_FILE, gt_dir=NS.NAVSIM_GT_DIR, check_train_sha=True):
    """저장 파일 → AnswerGrammar(NAVSIM). answer_grammar.load_grammar 와 같은 검사 + variant·행동 K 대조 + NAVSIM TRAIN 파일 sha256."""
    M = load_impl()
    d = json.load(open(path))
    if d.get("variant") != VARIANT or d.get("action_k") != N.ACTION_K:
        raise RuntimeError(f"{path}: variant {d.get('variant')}·action_k {d.get('action_k')} != {VARIANT}·{N.ACTION_K}")
    if check_train_sha:
        tf = Path(gt_dir) / NS.NAVSIM_SPLIT_FILES["train"]
        if M.file_sha256(tf) != d["train_sha256"]:
            raise RuntimeError(f"{path}: TRAIN 파일 {tf} 의 sha256 이 만들 때와 다르다 — --build 다시")
    g = M.load_grammar(tok, path, check_train_sha=False)   # 형식·문법 표 sha256·토큰 왕복·위치 수 검사(TRAIN 해시는 위에서)
    g.info.update(variant=VARIANT, action_k=N.ACTION_K)
    return g


def check(tok, path=MASK_FILE, gt_dir=NS.NAVSIM_GT_DIR, fold_file=NS.NAVSIM_FOLD_FILE, n_walk=2000):
    import sft_common as S
    M = load_impl()
    g = load(tok, path, gt_dir)
    d, _ = S.load_fold_split(fold_file)
    hold = set(d["holdout"]["A"]["scenes"]) | set(d["holdout"]["B"]["scenes"])
    tr = list(iter_gt(Path(gt_dir) / NS.NAVSIM_SPLIT_FILES["train"]))
    va = list(iter_gt(Path(gt_dir) / NS.NAVSIM_SPLIT_FILES["val"]))
    res = M.check(g, tok, {"train": [r["answer"] for r in tr], "val": [r["answer"] for r in va],
                           "holdout_AB": [r["answer"] for r in tr if r["log_name"] in hold]}, n_walk)
    res["min_allowed_any_key"] = min(len(g.allowed_list(k)) for k in
                                     [("F",), ("N",)] + [("B", t) for t in range(M.N_POS) if g.allow[t]] + [("Z", t) for t in g.sn_last])
    okv = all(res[k]["accepted"] == res[k]["n"] for k in ("train", "val", "holdout_AB"))
    okw = res["random_walk"]["parse_ok"] == res["random_walk"]["n"] == res["random_walk"]["positions_ok"]
    res["RESULT"] = "PASS" if okv and okw and res["min_allowed_any_key"] >= 1 else "FAIL"
    return res, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=MASK_FILE)
    ap.add_argument("--gt_dir", default=NS.NAVSIM_GT_DIR)
    ap.add_argument("--n_walk", type=int, default=2000)
    a = ap.parse_args()
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    configure()
    tok = navsim_tokenizer()
    if a.build:
        build(tok, a.out, a.gt_dir)
    res, _ = check(tok, a.out, a.gt_dir, n_walk=a.n_walk)
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    print(f"[check] {res['RESULT']}", flush=True)
    sys.exit(0 if res["RESULT"] == "PASS" else 1)


if __name__ == "__main__":
    main()
