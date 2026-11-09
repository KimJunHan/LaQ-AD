# 수정(2026-11-01): nuImages 백본 E3 4-GPU 1단계 평가 config — ./test_e3_ddp4.py 와 같다(평가는 체크포인트 가중치를 쓰므로
#   백본 초기값 설정은 결과에 영향 없음). run_name 만 다르다.
_base_ = ["./test_e3_ddp4.py"]
run_name = "hipad_nusc_laq_v2_teacher_stage1_e3_nuim_ddp4"
