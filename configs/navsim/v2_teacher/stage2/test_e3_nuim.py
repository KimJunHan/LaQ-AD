# 수정(2026-11-05): 사용자 결정 "NAVSIM 1단계 = nuImages + E3" — ./train_e3_nuim.py 의 navtest 평가 config(./test.py 와 같은 평가 설정, batch 1).
_base_ = ["./train_e3_nuim.py"]
data = dict(samples_per_gpu=1, test=dict(samples_per_gpu=1))
evaluation = dict(jsonfile_prefix="evaluation/hipad_navsim_laq_v2_teacher_stage2_e3_nuim")
