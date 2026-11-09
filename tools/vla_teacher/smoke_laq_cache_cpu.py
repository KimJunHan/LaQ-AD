# 수정(2026-10-16): 교사 생성 답 캐시 7 폴더 — CPU 전용 스모크 검사(실행: CUDA_VISIBLE_DEVICES="").
"""extract_teacher_cache_v2.py 검사.
  1) 배치 정의: train+val 전 표본(34,149) GT 정답으로 — 템플릿 위치 == 문자 구간→토큰 위치, 폴더 슬라이스 복호 == 구조에서 조립한
     기대 문자열, 폴더 토큰 수, user 텍스트 토큰 수 고정(context 길이). (GT 는 여기서 '토크나이저·문법 검사'에만 쓴다 — 캐시에는 안 씀)
  2) 소형 무작위 Qwen2-VL(실제 토크나이저 + 특수토큰, 실제 영상 12 장)로 파이프라인 실행: 7 폴더 파일 모양, 색인, meta.
  3) 저장값 == 단일 표본(패딩 없음) 순전파 hidden_states[layer] 의 해당 위치, 배치 생성 == 단일 생성, 영상별 시각 인코딩 == 묶음.
  4) 재개(완료 표본 건너뜀·파일 불변), 손상 파일 → 즉시 실패, --verify(부분 집합 OK / 전체 = 누락 보고 실패), qwen_l18 경로 거부.
  5) 수정(2026-10-16): 설계 검토 반영 — 문법 제약 생성: 소형 무작위 모델의 제약 출력(탐욕·무작위 표집)이 항상 파싱되고 고정 위치와
     일치, 색인 grammar_ok·layout_ok·constrained_decoding, _meta.json 의 prompt_mode="B"·teacher_chain_verified·교사별 chain,
     --verify 의 layout_bad 하드 게이트(색인 한 줄을 layout_ok=False 로 바꾸면 실패).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

import common_v2 as V  # noqa: E402
import laq_cache_layout as LY  # noqa: E402
import answer_grammar as AG  # noqa: E402  # 수정(2026-10-16): 설계 검토 반영
import sft_common as C  # noqa: E402

RES = {}
SCRIPT = HERE / "extract_teacher_cache_v2.py"
TEACHER_DIR = C.REPO / "results/vla_teacher/sft_foldA"


def ok(name, cond, info=""):
    RES[name] = dict(ok=bool(cond), info=str(info))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    return cond


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def layout_checks(tok, recs):
    tpl = LY.answer_template(tok)
    ok("template_counts", {k: len(v) for k, v in tpl.items()} == LY.EXPECTED_COUNTS,
       {k: len(v) for k, v in tpl.items()})
    bad_pos = bad_txt = bad_len = 0
    ex = None
    users = {}
    for r in recs:
        ids = tok(r["answer"], add_special_tokens=False)["input_ids"]
        if len(ids) != LY.ANSWER_TOKENS:
            bad_len += 1; continue
        text, offs = LY.token_char_offsets(tok, ids)
        assert text == r["answer"], "복호 != 정답(토크나이저 정리 규칙)"
        pos = LY.folder_token_positions_from_text(text, offs)
        if any(pos[k] != tpl[k] for k in LY.ANSWER_FOLDERS):
            bad_pos += 1
        want = LY.expected_folder_texts(V.parse_answer(r["answer"]))
        got = {k: tok.decode([ids[i] for i in tpl[k]]) for k in LY.ANSWER_FOLDERS}
        if got != want:
            bad_txt += 1
            ex = ex or (r["token"], [k for k in want if want[k] != got[k]])
        n = len(tok(C.build_user_text(r), add_special_tokens=False)["input_ids"])
        users[n] = users.get(n, 0) + 1
    ok("gt_answer_len_547_all", bad_len == 0, f"{len(recs)} 표본 중 길이 불일치 {bad_len}")
    ok("template_equals_charspan_positions_all", bad_pos == 0, f"불일치 {bad_pos}/{len(recs)}")
    ok("folder_slices_decode_to_expected_all", bad_txt == 0, f"불일치 {bad_txt} {ex}")
    ok("user_text_tokens_fixed", len(users) == 1, users)
    # 예시 한 표본: 폴더별 복호 문자열 앞부분
    r = next(rr for rr in recs if len(rr["targets"]["objects"]) >= 2)
    ids = tok(r["answer"], add_special_tokens=False)["input_ids"]
    for k in LY.ANSWER_FOLDERS:
        print(f"    {k:9s} {len(tpl[k]):3d}: {tok.decode([ids[i] for i in tpl[k]])[:110]}", flush=True)
    return tpl, list(users)[0]


def make_tiny_teacher(d, proc, tok):
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
    cfg = Qwen2VLConfig.from_pretrained(C.BASE_QWEN)
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 64, 128, 4
    cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
    cfg.rope_scaling = {"type": "mrope", "rope_type": "default", "mrope_section": [2, 3, 3]}
    vc = cfg.vision_config
    vc.depth, vc.embed_dim, vc.num_heads, vc.hidden_size, vc.mlp_ratio = 1, 32, 2, 64, 2
    cfg.tie_word_embeddings = True
    torch.manual_seed(0)
    m = Qwen2VLForConditionalGeneration(cfg).float()
    m.tie_weights()
    C.resize_for_tokens(m, len(tok))
    m.save_pretrained(d)
    proc.save_pretrained(d)
    meta = json.load(open(TEACHER_DIR / "teacher_meta.json"))
    meta["merged_dtype"] = "fp32"
    meta["smoke_note"] = "tiny random model for CPU smoke (smoke_laq_cache_cpu.py)"
    json.dump(meta, open(Path(d) / "teacher_meta.json", "w"), indent=1)
    return meta


def run(args, env, expect_ok=True):
    p = subprocess.run([sys.executable, str(SCRIPT)] + args, env=env, capture_output=True, text=True, cwd=str(C.REPO))
    tail = (p.stdout + p.stderr).strip().splitlines()[-6:]
    print("    $ extract " + " ".join(args) + f"  -> rc {p.returncode}\n      " + "\n      ".join(tail), flush=True)
    return p


def main():
    assert C.require_gpu_env(allow_cpu=True) == "cpu", "CUDA_VISIBLE_DEVICES='' 로 실행할 것"
    data_dir = C.REPO / "data/vla_teacher/v2"
    proc = C.make_processor(str(TEACHER_DIR / "processor"))
    tok = proc.tokenizer
    recs = C.load_records(data_dir, "train") + C.load_records(data_dir, "val")
    print(f"[smoke] GT 레코드 {len(recs)} (배치 검사 전용)", flush=True)
    tpl, n_user = layout_checks(tok, recs)
    L_ctx = 6 * 180 + n_user
    print(f"    context = 1080 + {n_user} = {L_ctx}", flush=True)
    del recs

    work = Path(os.environ.get("LAQ_SMOKE_DIR") or tempfile.mkdtemp(prefix="laq_smoke_"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    mdir, root = work / "tiny_teacher", work / "laq_cache"
    make_tiny_teacher(mdir, proc, tok)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(C.REPO), OPENBLAS_CORETYPE="SkylakeX")
    base = ["--teacher", "A", "--model", str(mdir), "--out_root", str(root), "--device", "cpu", "--dtype", "fp32",
            "--layer", "2", "--batch", "2", "--fwd_batch", "1"]

    p = run(base + ["--max_samples", "2"], env)
    ok("run2_rc0", p.returncode == 0)
    idx = [json.loads(x) for x in open(root / "_index/teacher_A.jsonl")]
    Ls = dict(context=L_ctx, **{k: len(tpl[k]) for k in LY.ANSWER_FOLDERS})
    shapes = {k: [tuple(torch.load(root / k / f"{e['token']}.pt", weights_only=True)["hidden"].shape) for e in idx]
              for k in LY.FOLDERS}
    ok("folder_shapes", all(s == [(Ls[k], 64)] * 2 for k, s in shapes.items()), {k: v[0] for k, v in shapes.items()})
    d0 = torch.load(root / "detection" / f"{idx[0]['token']}.pt", weights_only=True)
    ok("file_format", d0["hidden"].dtype == torch.bfloat16 and d0["attn_mask"].dtype == torch.uint8
       and bool((d0["attn_mask"] == 1).all()))
    metas = {k: json.load(open(root / k / "_meta.json")) for k in LY.FOLDERS}
    ok("meta_written", all(metas[k]["L"] == Ls[k] and metas[k]["teachers"]["A"]["n_done"] == 2 for k in LY.FOLDERS),
       {k: metas[k]["L"] for k in LY.FOLDERS})
    ok("meta_prompt_mode_chain_constraint",   # 수정(2026-10-16): 설계 검토 반영 — config teacher_meta_expect 가 요구하는 키
       all(metas[k]["prompt_mode"] == "B" and metas[k]["teacher_chain_verified"] is True and metas[k]["ego_in_prompt"] is True
           and metas[k]["constrained_decoding"] == AG.CONSTRAINT_NAME and metas[k]["teachers"]["A"]["chain"]["fold"] == "A"
           and metas[k]["teachers"]["A"]["chain"]["qa"]["kind"] == "original_qwen" for k in LY.FOLDERS),
       {k: metas["context"][k] for k in ("prompt_mode", "teacher_chain_verified", "constrained_decoding")})
    ok("index_entries", len(idx) == 2 and all(len(e["gen_ids"]) == 547 for e in idx),
       [(e["parse_ok"], e["layout_ok"], e["canonical"], e["fwd_gen_agree"], e["would_end_after"]) for e in idx])
    gt = {r["token"]: r["answer"] for r in C.load_records(data_dir, "train") + C.load_records(data_dir, "val")}
    ok("cache_is_generated_not_gt", all(e["gen_text"] != gt[e["token"]] for e in idx))
    ok("constrained_output_parses_positions_match",   # 수정(2026-10-16): 설계 검토 반영 — 제약 전: 무작위 모델 = 전부 형식 오류
       all(e["parse_ok"] and e["layout_ok"] and e["grammar_ok"] and not e["parse_failure"]
           and e["constrained_decoding"] == AG.CONSTRAINT_NAME for e in idx),
       [(e["parse_ok"], e["layout_ok"], e["grammar_ok"], e["unconstrained_agree"]) for e in idx])

    # 3) 저장값 == 단일 표본 순전파
    from transformers import Qwen2VLForConditionalGeneration
    sys.path.insert(0, str(HERE))
    import extract_teacher_cache_v2 as X
    model = Qwen2VLForConditionalGeneration.from_pretrained(str(mdir), torch_dtype=torch.float32, attn_implementation="sdpa").eval()
    meta = C.load_meta(str(mdir))
    items, _, _ = X.assigned_records(data_dir, C.FOLD_FILE, "A")
    r = X.strip_gt(items[0][1])
    ok("strip_gt", not any(k in r for k in X.GT_KEYS))
    assert r["token"] == idx[0]["token"]
    text = proc.apply_chat_template(C.build_messages(C.system_text(r, meta["system"]), C.build_user_text(r), 12),
                                    tokenize=False, add_generation_prompt=True)
    imgs = C.load_images(r, True, meta["frame_order"])
    enc = proc(text=[text], images=imgs, return_tensors="pt")
    with torch.no_grad():
        v_all = model.visual(enc["pixel_values"], grid_thw=enc["image_grid_thw"])
        X.patch_visual_per_image(model)
        v_one = model.visual(enc["pixel_values"], grid_thw=enc["image_grid_thw"])
    ok("visual_per_image_equals_batched", torch.allclose(v_all, v_one, atol=1e-5), f"max|Δ|={float((v_all - v_one).abs().max()):.1e}")
    P = enc["input_ids"].shape[1]
    from transformers import LogitsProcessorList
    grammar = AG.load_grammar(tok)   # 수정(2026-10-16): 설계 검토 반영 — 추출과 같은 제약
    with torch.no_grad():
        g1 = model.generate(**enc, do_sample=False, min_new_tokens=547, max_new_tokens=547,
                            eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")],
                            pad_token_id=tok.pad_token_id,
                            logits_processor=LogitsProcessorList([AG.make_logits_processor(grammar, P, 1)]))[0, P:].tolist()
    ok("single_generation_equals_batched", g1 == idx[0]["gen_ids"], f"일치 {sum(a == b for a, b in zip(g1, idx[0]['gen_ids']))}/547")
    full = torch.tensor([enc["input_ids"][0].tolist() + idx[0]["gen_ids"]])
    batch = dict(input_ids=full, attention_mask=torch.ones_like(full), pixel_values=enc["pixel_values"],
                 image_grid_thw=enc["image_grid_thw"])
    with torch.no_grad():
        hs = C.lm_forward(model, batch, "mrope", output_hidden_states=True).hidden_states[2][0]
    ids_of = tok.convert_tokens_to_ids
    ctx, (ua, ub) = LY.context_positions(full[0, :P].tolist(), ids_of("<|image_pad|>"), ids_of("<|vision_end|>"),
                                         ids_of("<|im_end|>"), range(6, 12))
    ok("context_text_span_is_user_text", tok.decode(full[0, ua:ub].tolist()) == C.build_user_text(r) and len(ctx) == L_ctx,
       f"user {ub - ua}, 영상 블록 6..11, 첫 위치 {ctx[0]}")
    errs = {}
    for k in LY.FOLDERS:
        pos = ctx if k == "context" else [P + q for q in tpl[k]]
        want = hs[pos].to(torch.bfloat16)
        got = torch.load(root / k / f"{r['token']}.pt", weights_only=True)["hidden"]
        errs[k] = float((got.float() - want.float()).abs().max())
    ok("stored_equals_single_forward_layer", max(errs.values()) <= 1e-2, {k: f"{v:.1e}" for k, v in errs.items()})

    # 수정(2026-10-16): 설계 검토 반영 — 5) 무작위 표집(온도 1)도 제약 아래에서는 항상 파싱·고정 위치 일치(3 행 × 씨드 2)
    sam_ok, sam_info = [], []
    for seed in (0, 1):
        torch.manual_seed(seed)
        enc3 = {k: (torch.cat([v] * 3, 0) if k in ("input_ids", "attention_mask") else
                    torch.cat([v] * 3, 0) if k in ("pixel_values", "image_grid_thw") else v) for k, v in enc.items()}
        with torch.no_grad():
            gs = model.generate(**enc3, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=547,
                                max_new_tokens=547, pad_token_id=tok.pad_token_id,
                                eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")],
                                logits_processor=LogitsProcessorList([AG.make_logits_processor(grammar, P, 3)]))[:, P:]
        for row in gs.tolist():
            e = X.analyse_answer(tok, row, tpl, grammar)
            sam_ok.append(e["parse_ok"] and e["layout_ok"] and e["grammar_ok"] and len(row) == 547)
            sam_info.append(e["parse_error"])
    ok("constrained_sampling_always_parses_positions_match", all(sam_ok) and len(sam_ok) == 6,
       f"{sum(sam_ok)}/{len(sam_ok)} {[x for x in sam_info if x][:2]}")
    text_u = None
    with torch.no_grad():   # 대조: 같은 모델·제약 없음 → 형식 오류(제약이 실제로 일을 한다는 확인)
        gu = model.generate(**enc, do_sample=False, min_new_tokens=547, max_new_tokens=547, pad_token_id=tok.pad_token_id,
                            eos_token_id=[tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")])[0, P:].tolist()
    eu = X.analyse_answer(tok, gu, tpl, grammar)
    ok("unconstrained_control_fails", not eu["grammar_ok"] and not eu["layout_ok"] and eu["parse_failure"],
       f"제약 없음: grammar_ok={eu['grammar_ok']} 첫 위반 위치 {eu['grammar_bad_pos']}")

    # 4) 재개
    before = {k: sha(root / k / f"{idx[0]['token']}.pt") for k in LY.FOLDERS}
    p = run(base + ["--max_samples", "3"], env)
    idx3 = [json.loads(x) for x in open(root / "_index/teacher_A.jsonl")]
    ok("resume_skips_done", p.returncode == 0 and "완료 건너뜀 2" in p.stdout and len(idx3) == 3
       and before == {k: sha(root / k / f"{idx[0]['token']}.pt") for k in LY.FOLDERS})
    # 손상 → 즉시 실패
    f = root / "detection" / f"{idx[0]['token']}.pt"
    keep = f.read_bytes()
    torch.save({"hidden": torch.zeros(5, 64, dtype=torch.bfloat16), "attn_mask": torch.ones(5, dtype=torch.uint8)}, f)
    p = run(base + ["--max_samples", "3"], env)
    ok("corrupt_file_fails_loudly", p.returncode != 0 and "파일 이상" in (p.stdout + p.stderr))
    f.write_bytes(keep)
    # 검증
    p = run(["--verify", "--teacher", "A", "--max_samples", "3", "--out_root", str(root), "--deep"], env)
    ok("verify_subset_ok", p.returncode == 0 and "OK" in p.stdout and "'layout_bad': 0" in p.stdout)
    # 수정(2026-10-16): 설계 검토 반영 — layout_bad 하드 게이트: 색인 한 줄을 layout_ok=False 로 바꾸면 --verify 실패(종료 코드 1)
    ip = root / "_index/teacher_A.jsonl"
    keep_idx = ip.read_text()
    lines = keep_idx.splitlines()
    e0 = json.loads(lines[1]); e0["layout_ok"] = False
    ip.write_text("\n".join([lines[0], json.dumps(e0, ensure_ascii=False)] + lines[2:]) + "\n")
    p = run(["--verify", "--teacher", "A", "--max_samples", "3", "--out_root", str(root)], env)
    ok("verify_layout_bad_hard_gate", p.returncode == 1 and "layout_bad: 1" in p.stdout and "'layout_bad': 1" in p.stdout)
    ip.write_text(keep_idx)
    p = run(["--verify", "--out_root", str(root)], env)
    ok("verify_full_reports_missing", p.returncode == 1 and "index_missing: 34146" in p.stdout)
    os.remove(root / "map" / f"{idx3[2]['token']}.pt")
    p = run(["--verify", "--teacher", "A", "--max_samples", "3", "--out_root", str(root)], env)
    ok("verify_detects_missing_file", p.returncode == 1 and "map_missing: 1" in p.stdout)
    # 보호 경로
    p = run(base[:4] + ["--out_root", str(work / "qwen_l18"), "--max_samples", "1", "--device", "cpu"], env)
    ok("refuse_qwen_l18_path", p.returncode != 0 and "거부" in (p.stdout + p.stderr))
    # 영상 없음 → 즉시 실패(대체 금지)
    rr = dict(r, cams=["data/nuscenes/samples/CAM_FRONT/__missing__.jpg"] + r["cams"][1:])
    try:
        C.load_images(rr, True); miss = False
    except FileNotFoundError:
        miss = True
    ok("missing_image_fails", miss)
    # 크기 추정(실제 D=1536, bf16)
    per = {k: Ls[k] * 1536 * 2 for k in LY.FOLDERS}
    tot = sum(per.values())
    print(f"[size] 표본당 {tot / 2**20:.2f} MiB {({k: round(v / 2**20, 3) for k, v in per.items()})} | 34,149 표본 "
          f"{tot * 34149 / 2**30:.1f} GiB", flush=True)
    n_fail = sum(not v["ok"] for v in RES.values())
    print(f"\n== {len(RES) - n_fail}/{len(RES)} 통과 == (작업 디렉터리 {work})", flush=True)
    out = C.REPO / "results/vla_teacher/smoke_laq_cache_cpu.json"
    json.dump(dict(results=RES, L=Ls, work=str(work)), open(out, "w"), indent=1, ensure_ascii=False)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
