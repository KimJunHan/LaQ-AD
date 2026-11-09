# 수정(2026-10-09): 기본값을 GRPO(sftref) 병합으로 변경. 원 설명(2026-09-21): B2 완료 후 LoRA 어댑터를 base 에 병합해 일반 모델로 저장 → qwen_cache.py 가 그대로 사용.
import argparse, torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
ap = argparse.ArgumentParser()
ap.add_argument("--qwen", default="ckpts/qwen2-vl-2b-3d")  # base = SFT 병합 모델
ap.add_argument("--adapter", default="results/qwen3d_nusc_grpo_sftref/final")
ap.add_argument("--out", default="ckpts/qwen2-vl-2b-3d-grpo")
a = ap.parse_args()
m = Qwen2VLForConditionalGeneration.from_pretrained(a.qwen, torch_dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, a.adapter)
m = m.merge_and_unload()
m.save_pretrained(a.out)
AutoProcessor.from_pretrained(a.qwen).save_pretrained(a.out)
print("merged ->", a.out)
