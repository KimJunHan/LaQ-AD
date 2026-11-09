# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   B2D 교사 정답(teacher_gt_b2d_{train,val}.jsonl) 전수 검사. 하나라도 실패하면 종료 코드 1.
#     (1) 엄격 파서 통과 + format_parsed 왕복 문자열 일치 (common_b2d)
#     (2) 정답 토큰 수 = 547 고정(Qwen2-VL-2B 토크나이저 + special_tokens.txt) — 전 표본
#     (3) laq_cache_layout 템플릿(7 폴더 고정 위치) = 문자 구간 → 토큰 구간 독립 계산, 그리고 expected_folder_texts 와 일치(shim 경유)
#     (4) answer_grammar 문법 표(shim 경유)로 train 기본 집합을 만들고 val 정답 100 % 수용 + 무작위 보행 파싱·위치 일치
#     (5) 프롬프트: user 텍스트 토큰 수 고정(목표점·자차 상태 고정폭) + prompt_parts 조립 = prompt.user + 변수 줄 외 텍스트 동일(누설 없음)
#     (6) 손실 마스크 구간이 정답 범위 안·정렬, 미래 불완전 표본은 Intent/Action 줄이 가려짐
#     (7) 분할: train 파일 clip ⊆ 공식 train 950 clip, val 파일 clip = 공식 val 50 clip, 겹침 0. token 형식 '<clip>/<kf:05d>'
#   사용: CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=. python tools/vla_teacher/check_teacher_gt_b2d.py --data_dir <dir>
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import b2d_teacher_shim as SH  # noqa: E402

TOKENIZER = "/home1/irteam/data-vol1/src/HiP-AD/ckpts/qwen2-vl-2b"


def load(dd, split, max_n=None):
    out = []
    with open(os.path.join(dd, f"teacher_gt_b2d_{split}.jsonl")) as f:
        for ln in f:
            if ln.strip():
                out.append(json.loads(ln))
                if max_n and len(out) >= max_n:
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_walk", type=int, default=500)
    ap.add_argument("--max_n", type=int, default=0, help="0 = 전수")
    a = ap.parse_args()
    os.chdir(REPO)
    dd = a.data_dir
    st = json.load(open(os.path.join(dd, "stats.json")))
    K = st["action_K"]
    B = SH.install(K)
    B.set_ttc_gap(st["ttc_gap_m"])
    toks = [l.strip() for l in open(os.path.join(dd, "special_tokens.txt")) if l.strip()]
    if toks != B.SPECIAL_TOKENS:
        raise SystemExit("special_tokens.txt != common_b2d.SPECIAL_TOKENS")
    tok = SH.tokenizer_with_b2d_tokens(TOKENIZER, toks)
    import laq_cache_layout as LY
    import answer_grammar as AG
    res, fail = {}, []
    tr = load(dd, "train", a.max_n or None)
    va = load(dd, "val", a.max_n or None)
    tpl = LY.answer_template(tok)
    for name, recs in (("train", tr), ("val", va)):
        r = dict(n=len(recs), parse_roundtrip_bad=0, len_bad=0, layout_bad=0, folder_text_bad=0, mask_bad=0,
                 prompt_len=set(), parts_bad=0, token_bad=0)
        for rec in recs:
            ans = rec["answer"]
            try:
                p = B.parse_answer(ans)
                if B.format_parsed(p) != ans:
                    r["parse_roundtrip_bad"] += 1
            except ValueError:
                r["parse_roundtrip_bad"] += 1
                continue
            ids = tok(ans, add_special_tokens=False)["input_ids"]
            if len(ids) != LY.ANSWER_TOKENS:
                r["len_bad"] += 1
            text, offs = LY.token_char_offsets(tok, ids)
            try:
                pos = LY.folder_token_positions_from_text(text, offs)
                if any(pos[k] != tpl[k] for k in LY.ANSWER_FOLDERS):
                    r["layout_bad"] += 1
                exp = LY.expected_folder_texts(p)
                sp = LY.folder_char_spans(ans)
                if any("".join(ans[x:y] for x, y in sp[k]) != exp[k] for k in LY.ANSWER_FOLDERS):
                    r["folder_text_bad"] += 1
            except (ValueError, TypeError):
                r["layout_bad"] += 1
            lm = rec["loss_mask_spans"]
            ok = all(0 <= x < y <= len(ans) for x, y in lm) and lm == sorted(lm)
            if not rec["targets"]["future_complete"]:
                L = rec["answer_spans"]["lines"]
                ok &= any(x <= L["intent"][0] and L["action"][1] <= y for x, y in lm) or \
                      (any(x <= L["intent"][0] and L["intent"][1] <= y for x, y in lm) and any(x <= L["action"][0] and L["action"][1] <= y for x, y in lm))
            r["mask_bad"] += int(not ok)
            r["prompt_len"].add(len(tok(rec["prompt"]["user"], add_special_tokens=False)["input_ids"]))
            pp = rec["prompt_parts"]
            if "\n".join([pp["instruction"], pp["command"], pp["target"], pp["ego"]]) != rec["prompt"]["user"]:
                r["parts_bad"] += 1
            c, f = rec["token"].rsplit("/", 1)
            if c != rec["scene_token"] or len(f) != 5 or int(f) != rec["frame_idx"]:
                r["token_bad"] += 1
        r["prompt_len"] = sorted(r["prompt_len"])
        res[name] = r
        for k in ("parse_roundtrip_bad", "len_bad", "layout_bad", "folder_text_bad", "mask_bad", "parts_bad", "token_bad"):
            if r[k]:
                fail.append(f"{name}.{k}={r[k]}")
        if len(r["prompt_len"]) != 1:
            fail.append(f"{name}.prompt_len 가변 {r['prompt_len'][:5]}")
    instr = {r["prompt_parts"]["instruction"] for r in tr[:2000] + va[:2000]}
    if len(instr) != 1:
        fail.append(f"지시문 {len(instr)} 종(1 이어야 함)")
    # (4) 문법 제약
    T = AG.grammar_table(tok)
    base, n_digit, n_act = AG.build_base(tok, [r["answer"] for r in tr], T)
    g = AG.AnswerGrammar(tok, base, T)
    walk = AG.check(g, tok, {"train": [r["answer"] for r in random.Random(0).sample(tr, min(len(tr), 3000))],
                             "val": [r["answer"] for r in va]}, a.n_walk)
    res["grammar"] = dict(n_digit_pos=n_digit, n_action_pos=n_act, **{k: {kk: vv for kk, vv in v.items() if kk != "first_bad"}
                                                                       for k, v in walk.items()})
    for k in ("train", "val"):
        if walk[k]["accepted"] != walk[k]["n"]:
            fail.append(f"grammar.{k} 수용 {walk[k]['accepted']}/{walk[k]['n']}")
    if walk["random_walk"]["parse_ok"] != a.n_walk or walk["random_walk"]["positions_ok"] != a.n_walk:
        fail.append(f"grammar.random_walk {walk['random_walk']}")
    # (7) 분할
    sys.path.insert(0, HERE)
    import b2d_qa_common as QC
    off_val = QC.official_val_clips()
    trc, vac = {r["scene_token"] for r in tr}, {r["scene_token"] for r in va}
    res["split"] = dict(train_clips=len(trc), val_clips=len(vac), train_in_official_val=len(trc & off_val),
                        val_not_official=len(vac - off_val), overlap=len(trc & vac))
    if trc & off_val or (not a.max_n and vac != off_val) or trc & vac:
        fail.append(f"split {res['split']}")
    res["fail"] = fail
    print(json.dumps(res, indent=1, ensure_ascii=False, default=list))
    out = os.path.join(dd, "check_report.json")
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False, default=list)
    print("ALL PASS" if not fail else f"FAIL {fail}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
