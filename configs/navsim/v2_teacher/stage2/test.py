# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
#   NAVSIM LaQ v2 교사 캐시 2단계 평가 config(navtest 추론 → evaluation/.../navsim_pred.pkl → EPDMS 채점). ./train.py 상속.
#   batch 1 강제(장면 경계 시간 뱅크 초기화). 확산 추론은 고정 시드(infer_seed 0)·infer_t 8.
_base_ = ["./train.py"]
data = dict(samples_per_gpu=1, test=dict(samples_per_gpu=1))
evaluation = dict(jsonfile_prefix="evaluation/hipad_navsim_laq_v2_teacher_stage2")
