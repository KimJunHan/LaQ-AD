# 수정(2026-10-16): 교사 QA 전체 학습 단계 — CPU 시험(GPU 불필요). 실행: CUDA_VISIBLE_DEVICES="" python tools/vla_teacher/smoke_qa_full_cpu.py
#   1) 대화 조립: 조각을 이어 붙인 문자열 == processor.apply_chat_template(전체 다중 턴, add_generation_prompt=False)
#   2) 조각별 토큰화(실제 영상 6 장) == 전체 문자열 한 번에 토큰화, 영상 토큰 6×180, 표본 길이 == jsonl n_tokens
#   3) 손실 마스크: 감독 토큰을 턴별로 디코드 == 각 답 + <|im_end|> (질문·영상·system·턴 머리말·줄바꿈 감독 없음)
#   4) 소형 무작위 Qwen2-VL(실제 config 축소, 묶인 임베딩 유지)로 train_qa_full.py 를 CPU 에서 실행:
#      시각 인코더 블록 불변 / merger·언어모델 변화 / 묶인 lm_head 유지 / final == latest / 재개 결과 == 끊김 없는 결과
#   5) 저장한 final 을 train_sft.py --base 로 적재해 1 step(LoRA SFT) 이 도는지
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import qa_common as Q  # noqa: E402
import sft_common as S  # noqa: E402

RES = {}


def ok(name, cond, info=""):
    RES[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)


def tiny_model_dir(d, proc):
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
    cfg = Qwen2VLConfig.from_pretrained(S.BASE_QWEN)
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 64, 128, 2
    cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
    cfg.rope_scaling = {"type": "mrope", "rope_type": "default", "mrope_section": [2, 3, 3]}
    vc = cfg.vision_config
    vc.depth, vc.embed_dim, vc.num_heads, vc.hidden_size, vc.mlp_ratio = 1, 32, 2, 64, 2
    cfg.tie_word_embeddings = True
    torch.manual_seed(0)
    m = Qwen2VLForConditionalGeneration(cfg).float()
    m.tie_weights()
    m.save_pretrained(d, safe_serialization=True)
    proc.save_pretrained(d)
    shutil.copy(Path(S.BASE_QWEN) / "generation_config.json", Path(d) / "generation_config.json")
    return d


def sd(path):
    from safetensors.torch import load_file
    out = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        out.update(load_file(str(f)))
    return out


def main():
    assert S.require_gpu_env(allow_cpu=True) == "cpu", "CUDA_VISIBLE_DEVICES='' 로 실행할 것"
    proc = S.make_processor(S.BASE_QWEN)
    tok = proc.tokenizer
    recs = {}
    for f in S.FOLDS:
        for p in ("train", "holdout"):
            recs[(f, p)] = Q.load_qa(f, p)[0]
    pick = []
    for ds in Q.DATASETS:
        rs = [r for r in recs[("A", "train")] if r["dataset"] == ds]
        pick += [rs[0], max(rs, key=lambda r: r["n_tokens"])]
    chunked = [r for r in recs[("A", "train")] if r["n_chunks"] > 1][:1]
    pick += chunked

    # 1) 조립 == chat template
    good = all("".join(t for t, _ in Q.conversation_pieces(r)) ==
               proc.apply_chat_template(Q.chat_messages(r), tokenize=False, add_generation_prompt=False) for r in pick)
    ok("pieces_equal_chat_template", good, f"{len(pick)} 표본(데이터셋별 첫·최장 + chunk 표본)")

    # 2)·3) 실제 영상 토큰화 · 마스크
    ds = Q.QAFrameDataset(pick, proc)
    g_whole, g_len, g_mask, g_img = True, True, True, True
    for i, r in enumerate(pick):
        it = ds[i]
        from PIL import Image
        full = "".join(t for t, _ in Q.conversation_pieces(r))
        enc = proc(text=[full], images=[Image.open(p).convert("RGB") for p in Q.cam_paths(r)], return_tensors="pt")
        g_whole &= enc["input_ids"][0].tolist() == it["input_ids"].tolist()
        g_img &= int((it["input_ids"] == Q.IMAGE_PAD_ID).sum()) == 6 * S.TOKENS_PER_CAM and \
            it["image_grid_thw"].tolist() == [[1, 20, 36]] * 6
        g_len &= len(it["input_ids"]) == r["n_tokens"]
        lab = it["labels"].tolist()
        spans, cur = [], []
        for x in lab:
            if x != -100:
                cur.append(x)
            elif cur:
                spans.append(cur); cur = []
        if cur:
            spans.append(cur)
        g_mask &= [tok.decode(s) for s in spans] == [t["a"] + "<|im_end|>" for t in r["turns"]]
    ok("tokenize_pieces_equal_whole", g_whole, "조각별 == 한 번에(영상 포함)")
    ok("image_tokens_6x180", g_img)
    ok("length_matches_jsonl", g_len, "QAFrameDataset 길이 == make_qa_data n_tokens(텍스트 기준 계산)")
    ok("loss_mask_answers_only", g_mask, "감독 구간 == 턴별 답+<|im_end|>, 그 밖 전부 -100")
    ok("chunk_sample_present", bool(chunked), chunked[0]["id"] if chunked else "chunk 표본 없음")

    # 4) 소형 모델 학습
    import train_qa_full as T
    tmp = Path(tempfile.mkdtemp(prefix="qa_full_smoke_"))
    try:
        base = tiny_model_dir(str(tmp / "tiny"), proc)
        common = ["--fold", "A", "--base", base, "--device", "cpu", "--test_subset", "4", "--batch", "2", "--accum", "1",
                  "--workers", "0", "--holdout_every", "2", "--save_every", "1", "--lr", "1e-3", "--lr_projector", "1e-3",
                  "--warmup", "0"]
        T.main(common + ["--max_steps", "2", "--out", str(tmp / "run")])
        w0, w1 = sd(base), sd(tmp / "run" / "final")
        vis_blocks = [k for k in w0 if k.startswith("visual.") and not k.startswith("visual.merger.")]
        ok("vision_blocks_unchanged", all(torch.equal(w0[k], w1[k]) for k in vis_blocks), f"{len(vis_blocks)} 텐서")
        ok("merger_changed", any(not torch.equal(w0[k], w1[k]) for k in w0 if k.startswith("visual.merger.")))
        ok("llm_changed", any(not torch.equal(w0[k], w1[k]) for k in w0 if k.startswith("model.layers.")) and
           not torch.equal(w0["model.embed_tokens.weight"], w1["model.embed_tokens.weight"]))
        ok("tied_lm_head_saved_once", "lm_head.weight" not in w1, "tie_word_embeddings → lm_head 는 embed 와 공유")
        ok("all_fp32", all(v.dtype == torch.float32 for v in w1.values()))
        wl = sd(tmp / "run" / "latest")
        ok("final_equals_latest", all(torch.equal(w1[k], wl[k]) for k in w1))
        ok("best_saved", (tmp / "run" / "best" / "qa_meta.json").exists() and (tmp / "run" / "best" / "preprocessor_config.json").exists())
        log = [json.loads(l) for l in open(tmp / "run" / "train_log.jsonl")]
        st = [x for x in log if "loss" in x]
        ok("log_fields", all(k in st[0] for k in ("loss", "gnorm", "lr", "s_step", "peak_gb", "tok_per_s")), list(st[0].keys()))
        ok("holdout_logged", any(x.get("kind") == "holdout_start" for x in log) and any(x.get("kind") == "holdout" for x in log))
        # 재개: 3 step 를 한 번에 vs 2 step 에서 멈춘 뒤 재개
        T.main(common + ["--max_steps", "3", "--out", str(tmp / "full3"), "--eval_at_start", "0"])
        T.main(common + ["--max_steps", "3", "--stop_at_step", "2", "--out", str(tmp / "res3"), "--eval_at_start", "0"])
        T.main(common + ["--max_steps", "3", "--resume", "--out", str(tmp / "res3"), "--eval_at_start", "0"])
        a3, b3 = sd(tmp / "full3" / "final"), sd(tmp / "res3" / "final")
        md = max(float((a3[k] - b3[k]).abs().max()) for k in a3)
        ok("resume_matches_uninterrupted", md < 1e-6, f"최대 차이 {md:.2e}")

        # 5) train_sft.py --base <final>
        import train_sft
        train_sft.main(["--base", str(tmp / "run" / "final"), "--fold", "A", "--device", "cpu", "--max_steps", "1",
                        "--max_train_samples", "2", "--max_holdout_samples", "1", "--batch", "1", "--accum", "1",
                        "--workers", "0", "--out", str(tmp / "sft"), "--holdout_every", "1", "--save_every", "1"])
        meta = json.load(open(tmp / "sft" / "final" / "teacher_meta.json"))
        ok("train_sft_loads_qa_ckpt", meta["base"] == str(tmp / "run" / "final") and (tmp / "sft" / "final" / "adapter_model.safetensors").exists(),
           f"teacher_meta.base = {meta['base']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bad = [k for k, v in RES.items() if not v]
    print("ALL PASS" if not bad else f"FAIL {bad}", flush=True)
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
