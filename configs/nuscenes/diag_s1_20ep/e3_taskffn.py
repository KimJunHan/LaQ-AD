# 수정(2026-10-27): 후보 3 — 통합 디코더의 공유 FFN·LayerNorm. E0 위에서 ffn·norm 을 쿼리 종류별로 따로 둔다
#   (models/decoder/unified_decoder.py task_specific_ffn_norm). 어텐션(gnn·inter_gnn·qwen_attn)과 나머지는 E0 과 같다.
_base_ = ["./e0_base.py"]
model = dict(head=dict(onedecoder_head=dict(task_specific_ffn_norm=True)))
run_name = "diag_s1_20ep_e3_taskffn"
