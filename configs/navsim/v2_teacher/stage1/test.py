# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
#   NAVSIM LaQ v2 교사 캐시 1단계 평가 config(navtest 추론 → navsim_pred.pkl → EPDMS). ./train.py 상속.
#   batch 1 강제(장면 경계 시간 뱅크 초기화, SparseDetector._maybe_reset_banks). 1단계는 계획 손실 0 → EPDMS 참고용.
_base_ = ["./train.py"]
data = dict(samples_per_gpu=1, test=dict(samples_per_gpu=1))
evaluation = dict(jsonfile_prefix="evaluation/hipad_navsim_laq_v2_teacher_stage1")
