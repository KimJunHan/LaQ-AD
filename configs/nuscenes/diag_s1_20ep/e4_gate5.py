# 수정(2026-10-27): 사용자 제안 "교사 비율을 5배로" — 평가 때 게이트를 키우는 것(×2 에서 mAP 0.4015→0.3624 하락)은
#   학습된 하류 계층과 크기가 안 맞아 역효과라, 학습 때 교사 게이트(gate·gate_sep)의 학습률을 5배로 해 게이트가 스스로 커질 수 있게 한다.
#   E0 대비 바뀌는 것: qwen_token_attn.gate 의 lr_mult 1→5 (그 밖의 교사 어텐션·사영 lr 은 그대로).
_base_ = ["./e0_base.py"]
optimizer = dict(paramwise_cfg=dict(custom_keys={"img_backbone": dict(lr_mult=0.5), "img_neck": dict(lr_mult=0.5),
                                                 "qwen_token_attn.gate": dict(lr_mult=5.0)}))
run_name = "diag_s1_20ep_e4_gate5"
