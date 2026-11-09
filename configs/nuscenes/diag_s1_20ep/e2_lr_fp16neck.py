# 수정(2026-10-27): 후보 2 — neck bf16. E1(lr 2e-4) 위에서 neck 을 HiP-AD 원본처럼 fp16(정적 배율 32)으로 계산한다.
#   E1 대비 바뀌는 것: neck_dtype "bf16" → None. 넘치면 fp16_strict_hook 이 멈춘다(건너뛰지 않음) — 그 자체가 결과.
_base_ = ["./e1_lr.py"]
model = dict(neck_dtype=None)
run_name = "diag_s1_20ep_e2_lr_fp16neck"
