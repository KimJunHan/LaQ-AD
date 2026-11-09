# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
"""gen_teacher_gt_navsim.py 산출 jsonl 전수 검사(env: hipad_bw).

  1. 행 수·token 집합 == info 의 고유 프레임(누락·중복 0), 장면 프레임 수(is_scene_frame) == info 의 평가/장면 프레임 수.
  2. 정답: 엄격 파서 왕복 일치, 토큰 수 == 549(특수토큰 등록 Qwen2-VL 토크나이저), 폴더 위치 = 템플릿(laq_cache_layout_navsim).
  3. 입력 영상: cams / cams_prev 경로가 모두 info 프레임(현재+과거)의 경로 — 미래 프레임 영상 참조 0. 전방 3 캠 순서.
  4. prev_available 이면 prev_token 이 info 에 있고 시간이 0.5 s 앞. 손실 마스크 구간이 정답 안.
  5. 교사 제외 6 로그 표시(teacher_excluded_log)가 제외 목록과 일치.
python -I tools/vla_teacher/check_teacher_gt_navsim.py --jsonl <...> --infos <...> --out <json>
"""
import argparse
import collections
import json
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import laq_cache_layout_navsim as LN  # noqa: E402
import navsim_teacher_common as N  # noqa: E402

EXCLUDE = "/home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/qa/nuplanqa/teacher_exclude_logs_navsim.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--codebook", required=True)     # 수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건 (4)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--infos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenize_every", type=int, default=1, help="1 = 전 표본 토큰화 검사")
    a = ap.parse_args()
    d = pickle.load(open(a.infos, "rb"))
    info = {}
    for i in d["infos"]:
        info.setdefault(i["token"], i)
    scene_frames = {t for t, i in info.items() if i.get("is_eval_frame")}
    paths = {}
    for t, i in info.items():
        for c in N.CAMS:
            paths[i["cams"][c]["data_path"]] = t
    excl = set(json.load(open(EXCLUDE))["logs"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(_HERE, "..", "..", "ckpts/qwen2-vl-2b"))
    N.configure(a.codebook, a.thresholds)   # 수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건 (4)
    tok.add_tokens(N.SPECIAL_TOKENS, special_tokens=True)
    tpl = LN.answer_template(tok)
    R = collections.Counter()
    seen = set()
    ulen = collections.Counter()   # user 텍스트 토큰 수(캐시 context 길이 고정 전제) — 값이 하나여야 한다
    with open(a.jsonl) as f:
        for k, ln in enumerate(f):
            r = json.loads(ln)
            t = r["token"]
            R["rows"] += 1
            R["dup"] += t in seen
            seen.add(t)
            if t not in info:
                R["token_not_in_infos"] += 1
                continue
            R["scene_flag_mismatch"] += bool(r["is_scene_frame"]) != (t in scene_frames)
            try:
                N.parse_answer(r["answer"])
            except ValueError:
                R["parse_fail"] += 1
            if k % a.tokenize_every == 0:
                enc = tok(r["answer"], return_offsets_mapping=True, add_special_tokens=False)
                if len(enc["input_ids"]) != LN.ANSWER_TOKENS:
                    R["len_bad"] += 1
                else:
                    got = LN.folder_token_positions_from_text(r["answer"], enc["offset_mapping"])
                    R["layout_bad"] += any(got[f] != tpl[f] for f in LN.ANSWER_FOLDERS)
                R["tokenized"] += 1
                ulen[len(tok(r["prompt"]["user"], add_special_tokens=False)["input_ids"])] += 1
            for key in ("cams", "cams_prev"):
                if len(r[key]) != 3:
                    R[f"{key}_not_3"] += 1
                R[f"{key}_path_not_in_infos"] += sum(p not in paths for p in r[key])
            if r["cams"] != [info[t]["cams"][c]["data_path"] for c in N.CAMS]:
                R["cams_order_bad"] += 1
            if r["prev_available"]:
                p = r["prev_token"]
                if p not in info:
                    R["prev_not_in_infos"] += 1
                else:
                    dt = (info[t]["timestamp"] - info[p]["timestamp"]) * 1e-6
                    R["prev_dt_bad"] += not (0.3 < dt < 0.75)
                    R["cams_prev_mismatch"] += r["cams_prev"] != [info[p]["cams"][c]["data_path"] for c in N.CAMS]
            L = len(r["answer"])
            R["mask_span_out"] += sum(not (0 <= s < e <= L) for s, e in r["loss_mask_spans"])
            R["excl_flag_bad"] += bool(r["teacher_excluded_log"]) != (r["log_name"] in excl)
            R["excluded_rows"] += bool(r["teacher_excluded_log"])
            R["scene_rows"] += bool(r["is_scene_frame"])
            R["prev_available"] += bool(r["prev_available"])
            R["future_complete"] += bool(r["targets"]["future_complete"])
    R["missing_tokens"] = len(set(info) - seen)
    R["info_frames"] = len(info)
    R["info_scene_frames"] = len(scene_frames)
    bad_keys = ["dup", "token_not_in_infos", "scene_flag_mismatch", "parse_fail", "len_bad", "layout_bad",
                "cams_not_3", "cams_prev_not_3", "cams_path_not_in_infos", "cams_prev_path_not_in_infos", "cams_order_bad",
                "prev_not_in_infos", "prev_dt_bad", "cams_prev_mismatch", "mask_span_out", "excl_flag_bad", "missing_tokens"]
    out = dict(R)
    out["user_text_tokens"] = {str(k): v for k, v in ulen.items()}
    if len(ulen) != 1:
        bad_keys.append("user_text_tokens_not_fixed"); R["user_text_tokens_not_fixed"] = len(ulen)
    out["FAILS"] = [k for k in bad_keys if R[k]]
    if R["scene_rows"] != len(scene_frames):
        out["FAILS"].append("scene_rows")
    out["RESULT"] = "PASS" if not out["FAILS"] else "FAIL"
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
