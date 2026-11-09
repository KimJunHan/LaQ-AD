# 수정(2026-11-02): 공유 FFN 4-GPU 1단계(./train_e0_ddp4.py, ./train_e0_nuim_ddp4.py) 평가 config — ./test.py(E0)와 같은 모델·평가 설정. run_name 만 다르다.
#   (평가는 체크포인트 가중치를 쓰므로 백본 초기값 설정은 결과에 영향 없음 — 두 학습 모두 이 config 로 평가)
_base_ = ["./test.py"]
run_name = "hipad_nusc_laq_v2_teacher_stage1_e0_ddp4"
