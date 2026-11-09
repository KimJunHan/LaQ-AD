# 수정(2026-11-02): 사용자 결정 "1번 후보 100epoch 재학습 진행하자" — 공유 FFN·norm(HiP-AD 원 설계, E0 와 같은 모델) + GPU 4장 DDP + EMA.
#   ./train_e3_ddp4.py 와 모든 값이 같고 task_specific_ffn_norm 만 False(통합 디코더 기본값 = ./train.py·E0).
#   근거: 100 에폭에서 E3(과제별 FFN) 원 가중치 이득 없음(E0 .4015 vs E3 .3988), 공유 FFN 이 HiP-AD 설계와 같다.
#   EMA decay 0.9998 / ramp 2000 = 규칙(창 5000·step/43900, ramp 2000·step/43900, step 43,900)과 같다.
_base_ = ["./train_e3_ddp4.py"]
model = dict(head=dict(onedecoder_head=dict(task_specific_ffn_norm=False)))
run_name = "hipad_nusc_laq_v2_teacher_stage1_e0_ddp4"
