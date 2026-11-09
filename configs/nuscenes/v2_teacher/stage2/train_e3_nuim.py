# 수정(2026-11-04): 사용자 지시 "GPU 1 에서 nuImages 2단계 E3를 시작하자" — ./train_e3.py 와 모든 값이 같고(지도 정렬 선택적 잡음,
#   과제별 FFN·norm, 계획·자차 lr ×10, EMA 규칙) 시작 가중치만 nuImages R50 1단계 E3 EMA 로 바꾼다.
#   이미지 백본 초기값(./train.py 의 ImageNet 설정)은 load_from 이 전부 덮어쓴다.
_base_ = ["./train_e3.py"]

load_from = "./results/hipad_nusc_laq_v2_teacher_stage1_e3_nuim_ddp4/iter_43900_ema.pth"

run_name = "hipad_nusc_laq_v2_teacher_stage2_e3_nuim"
