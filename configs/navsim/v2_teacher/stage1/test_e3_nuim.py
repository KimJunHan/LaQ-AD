# 수정(2026-11-05): 사용자 결정 "NAVSIM 1단계 = nuImages + E3" — ./train_e3_nuim.py 의 navtest 평가 config(./test.py 와 같은 평가 설정).
#   batch 1 강제(장면 경계 시간 뱅크 초기화). 모델 구조(E3 과제별 FFN·norm, FPN bf16)는 학습 config 를 따른다. 백본 초기값 설정은
#   체크포인트 가중치가 덮어써 결과에 영향 없음. 스트림 그룹은 학습 전용(평가에서는 쓰지 않음).
_base_ = ["./train_e3_nuim.py"]
data = dict(samples_per_gpu=1, test=dict(samples_per_gpu=1))
evaluation = dict(jsonfile_prefix="evaluation/hipad_navsim_laq_v2_teacher_stage1_e3_nuim")
