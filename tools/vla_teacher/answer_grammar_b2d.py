# 수정(2026-10-31): B2D 교사 체인 — 정답 문법 제약 생성의 B2D 판(547 토큰). answer_grammar.py 는 수정하지 않는다.
"""B2D 정답 문법 제약(교사 캐시 추출용). NAVSIM 의 answer_grammar_navsim.py 에 해당한다.

B2D 정답(v2-sensor-b2d)은 줄·필드·토큰 수가 nuScenes v2-sensor 와 같으므로(547) 위치 배치는 laq_cache_layout 그대로이고, 다른 것은
클래스 표(9 종)·행동 코드북 K(2040)·파서(common_b2d.parse_answer)뿐이다. 그래서 사본 모듈을 만들지 않고 b2d_teacher_shim 으로
sys.modules['common_v2'] = common_b2d 를 먼저 끼운 뒤 answer_grammar 를 import 한다(check_teacher_gt_b2d.py 와 같은 방식) —
이 프로세스 안의 answer_grammar·laq_cache_layout 은 B2D 표를 쓴다(nuScenes 와 한 프로세스에 섞지 않는다; shim 이 섞이면 실패시킨다).
기본 집합: B2D TRAIN GT 전체(teacher_gt_b2d_train.jsonl, 950 clip 키프레임 47,346 — nuScenes 가 train split 전체, NAVSIM 이 navtrain
  전체로 만든 것과 같은 규칙) + 숫자 자리 0..9 전체 + 행동 자리 <action_0..K-1> 전체. val 은 검사에만 쓴다.
검사(--build/--check): train·val·hold-out(A∪B) GT 100 % 수용 + 무작위 보행 2,000 개 파싱·위치 100 %.
제약 이름 = 'answer_grammar_v1'(학생 config teacher_meta_expect·로더 TEACHER_CONSTRAINT 가 요구, 규칙 동일). B2D 표 식별은 파일
  sha256·grammar_table_sha256·variant='b2d_547'·action_k 로 한다.
사용(CPU):
  CUDA_VISIBLE_DEVICES="" python tools/vla_teacher/answer_grammar_b2d.py --build   # 만들고 검사
  CUDA_VISIBLE_DEVICES="" python tools/vla_teacher/answer_grammar_b2d.py --check   # 다시 읽어 검사만
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import b2d_sft_common as BS  # noqa: E402

TAG = "수정(2026-10-31): B2D 교사 체인"
VARIANT = "b2d_547"
MASK_FILE = str(BS.B2D_GRAMMAR_FILE)


def impl(gt_dir=BS.B2D_GT_DIR):
    """shim 설치 뒤의 answer_grammar(B2D 표). 반환 (AG, LY, common_b2d)."""
    B = BS.install_shim(gt_dir)
    import answer_grammar as AG
    import laq_cache_layout as LY
    if AG.V is not B:
        raise RuntimeError("answer_grammar 가 common_b2d 가 아닌 common_v2 로 import 됐다(shim 설치 전에 import 됨)")
    return AG, LY, B


def b2d_tokenizer(base=None, gt_dir=BS.B2D_GT_DIR):
    """기본 Qwen 토크나이저 + B2D 특수토큰(SFT 와 같은 sft_common.add_special_tokens)."""
    import sft_common as S
    _, _, B = impl(gt_dir)
    tok = S.make_processor(base or S.BASE_QWEN, BS.B2D_PIXELS).tokenizer
    toks, _ = S.load_special_tokens(str(Path(gt_dir) / "special_tokens.txt"))
    if toks != list(B.SPECIAL_TOKENS):
        raise RuntimeError("special_tokens.txt != common_b2d.SPECIAL_TOKENS")
    S.add_special_tokens(tok, toks)
    return tok


def iter_gt(path, keys=("token", "scene_token", "answer")):
    with open(path) as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                yield {k: r[k] for k in keys}


def build(tok, out=MASK_FILE, gt_dir=BS.B2D_GT_DIR):
    AG, _, B = impl(gt_dir)
    tf = Path(gt_dir) / BS.B2D_SPLIT_FILES["train"]
    answers = [r["answer"] for r in iter_gt(tf)]
    T = AG.grammar_table(tok)
    base, n_digit, n_act = AG.build_base(tok, answers, T)
    g = AG.AnswerGrammar(tok, base, T)
    doc = dict(format_version=AG.FORMAT_VERSION, created_by=TAG, created=datetime.datetime.now().isoformat(timespec="seconds"),
               variant=VARIANT, constraint_name=AG.CONSTRAINT_NAME,
               source=f"B2D TRAIN GT 전체(950 clip 2 Hz 키프레임, val 미사용) + 숫자 자리 0..9 전체 + 행동 자리 <action_0..{B.ACTION_K - 1}> 전체",
               train_file=str(tf), train_sha256=AG.file_sha256(tf), n_train=len(answers), n_digit_positions=n_digit,
               n_action_positions=n_act, n_positions=AG.N_POS, end_token=AG.END_TOKEN, action_k=B.ACTION_K, ttc_gap_m=B.TTC_GAP,
               grammar_table_sha256=AG.table_sha(tok, T),
               allowed_sizes=[len(x) for x in g.allow], base_tokens=[tok.convert_ids_to_tokens(sorted(b)) for b in base])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    json.dump(doc, open(tmp, "w"), ensure_ascii=False)
    os.replace(tmp, out)
    print(f"[build] {out} — TRAIN {len(answers)} 정답, 숫자 자리 {n_digit}, 행동 자리 {n_act}, 허용 크기 "
          f"min {min(len(x) for x in g.allow)} max {max(len(x) for x in g.allow)}", flush=True)
    return out


def load(tok, path=MASK_FILE, gt_dir=BS.B2D_GT_DIR):
    """저장 파일 → AnswerGrammar(B2D). answer_grammar.load_grammar 검사(형식·문법 표 sha256·토큰 왕복·위치 수·TRAIN sha256)
    + variant·action_k 대조."""
    AG, _, B = impl(gt_dir)
    d = json.load(open(path))
    if d.get("variant") != VARIANT or d.get("action_k") != B.ACTION_K:
        raise RuntimeError(f"{path}: variant {d.get('variant')}·action_k {d.get('action_k')} != {VARIANT}·{B.ACTION_K}")
    g = AG.load_grammar(tok, path, data_dir=gt_dir, check_train_sha=True)   # data_dir 의 teacher_gt*_train.jsonl = B2D train
    g.info.update(variant=VARIANT, action_k=B.ACTION_K)
    return g


def check(tok, path=MASK_FILE, gt_dir=BS.B2D_GT_DIR, fold_file=BS.B2D_FOLD_FILE, n_walk=2000):
    import sft_common as S
    AG, _, _ = impl(gt_dir)
    g = load(tok, path, gt_dir)
    d, _ = S.load_fold_split(str(fold_file))
    hold = set(d["holdout"]["A"]["scenes"]) | set(d["holdout"]["B"]["scenes"])
    tr = list(iter_gt(Path(gt_dir) / BS.B2D_SPLIT_FILES["train"]))
    va = list(iter_gt(Path(gt_dir) / BS.B2D_SPLIT_FILES["val"]))
    res = AG.check(g, tok, {"train": [r["answer"] for r in tr], "val": [r["answer"] for r in va],
                            "holdout_AB": [r["answer"] for r in tr if r["scene_token"] in hold]}, n_walk)
    for k in ("train", "val", "holdout_AB"):
        res[k].pop("first_bad", None) if res[k]["accepted"] == res[k]["n"] else None
    res["min_allowed_any_key"] = min(len(g.allowed_list(k)) for k in
                                     [("F",), ("N",)] + [("B", t) for t in range(AG.N_POS) if g.allow[t]] + [("Z", t) for t in g.sn_last])
    okv = all(res[k]["accepted"] == res[k]["n"] for k in ("train", "val", "holdout_AB"))
    okw = res["random_walk"]["parse_ok"] == res["random_walk"]["n"] == res["random_walk"]["positions_ok"]
    res["RESULT"] = "PASS" if okv and okw and res["min_allowed_any_key"] >= 1 and res["holdout_AB"]["n"] > 0 else "FAIL"
    return res, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=MASK_FILE)
    ap.add_argument("--gt_dir", default=str(BS.B2D_GT_DIR))
    ap.add_argument("--n_walk", type=int, default=2000)
    a = ap.parse_args()
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    tok = b2d_tokenizer(gt_dir=a.gt_dir)
    if a.build:
        build(tok, a.out, a.gt_dir)
    res, _ = check(tok, a.out, a.gt_dir, n_walk=a.n_walk)
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    print(f"[check] {res['RESULT']}", flush=True)
    sys.exit(0 if res["RESULT"] == "PASS" else 1)


if __name__ == "__main__":
    main()
