# 수정(2026-10-27): 후보 1 — lr 4e-4 + BN FPN 조합. HiP-AD 공개 코드 1단계 레시피(lr 2e-4, 백본 ×0.5, FPN 배율 없음)로 되돌린다.
#   E0 대비 바뀌는 것: lr 4e-4→2e-4, FPN lr_mult 0.5→1.0(실효 FPN lr 2e-4 로 E0 과 같음; 나머지 헤드·디코더 lr 이 절반). neck bf16 은 유지.
_base_ = ["./e0_base.py"]
optimizer = dict(lr=2e-4, paramwise_cfg=dict(custom_keys={"img_backbone": dict(lr_mult=0.5), "img_neck": dict(lr_mult=1.0)}))
run_name = "diag_s1_20ep_e1_lr"
