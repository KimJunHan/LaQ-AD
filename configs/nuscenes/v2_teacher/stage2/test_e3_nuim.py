# 수정(2026-11-04): ./train_e3_nuim.py 평가 config — ./test_e3.py 와 같다(구조 동일, 실행 이름만 다름).
_base_ = ["./test_e3.py"]
run_name = "hipad_nusc_laq_v2_teacher_stage2_e3_nuim"
