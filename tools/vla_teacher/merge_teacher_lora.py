# 수정(2026-10-15): 교사 VLM ② LoRA SFT — 어댑터(LoRA + 새 토큰 행)를 base 에 병합해 일반 Qwen2-VL 체크포인트로 저장.
#   학습과 똑같이 특수토큰 등록·임베딩 resize 를 재현한 뒤 PeftModel 을 적재해 merge_and_unload 한다
#   (새 토큰 행 값은 어댑터의 trainable_tokens 값으로 덮어써지므로 resize 초기값과 무관).
#   저장물: 모델(bf16 기본 — 캐시 파이프라인과 같은 dtype) + processor(토큰 추가된 토크나이저, min/max_pixels) + teacher_meta.json.
#   사용: python tools/vla_teacher/merge_teacher_lora.py --adapter results/vla_teacher/<run>/best --out results/vla_teacher/<run>/merged
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_common import add_special_tokens, load_meta, make_processor, require_gpu_env, resize_for_tokens, save_meta  # noqa: E402


def merge(adapter, out, dtype="bf16", device="cpu"):
    import torch
    from peft import PeftModel
    from transformers import Qwen2VLForConditionalGeneration
    meta = load_meta(adapter)
    dt = torch.bfloat16 if dtype == "bf16" else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(meta["base"], torch_dtype=dt)
    proc = make_processor(meta["base"], meta["pixels"])
    tok = proc.tokenizer
    new_ids, n0 = add_special_tokens(tok, meta["special_tokens"])
    assert n0 == meta["n_base_vocab"] and len(tok) == meta["vocab_after"], "학습 때와 토크나이저가 다르다"
    resize_for_tokens(model, len(tok))
    model.to(device)
    pm = PeftModel.from_pretrained(model, adapter)
    m = pm.merge_and_unload()
    if getattr(m.config, "tie_word_embeddings", False):
        assert m.get_output_embeddings().weight.data_ptr() == m.get_input_embeddings().weight.data_ptr()
    m.to(dt)
    os.makedirs(out, exist_ok=True)
    m.save_pretrained(out)
    proc.save_pretrained(out)
    save_meta(os.path.join(out, "teacher_meta.json"), **{k: v for k, v in meta.items() if k != "step"},
              merged_from=os.path.abspath(adapter), merged_dtype=dtype, adapter_step=meta.get("step"))
    print("merged ->", out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cpu", help="cpu(기본, GPU 불필요) | cuda")
    a = ap.parse_args()
    if a.device != "cpu":
        require_gpu_env()
    merge(a.adapter, a.out, a.dtype, a.device)


if __name__ == "__main__":
    main()
