# 수정(2026-10-31): 새 본 2단계(train_e3.py) 평가 config — ./test.py 와 같고 과제별 FFN·norm 만 켠다(체크포인트 구조 일치).
_base_ = ["./test.py"]
model = dict(head=dict(onedecoder_head=dict(task_specific_ffn_norm=True)))
run_name = "hipad_nusc_laq_v2_teacher_stage2_e3"
