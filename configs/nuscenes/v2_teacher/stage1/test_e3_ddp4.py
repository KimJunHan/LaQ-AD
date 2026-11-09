# 수정(2026-10-30): E3 4-GPU 1단계 평가 config — ./test.py 와 같은 평가 설정, 모델만 E3(task_specific_ffn_norm). 평가는 1 GPU.
_base_ = ["./test.py"]
model = dict(head=dict(onedecoder_head=dict(task_specific_ffn_norm=True)))
run_name = "hipad_nusc_laq_v2_teacher_stage1_e3_ddp4"
