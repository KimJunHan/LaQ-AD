# 수정(2026-11-06): 사용자 결정 "B2D 학생 = 최종 레시피 + E2" — ./train_e3_nuim.py(E3, FPN bf16, E2 자차 상태 입력) 의 개방루프 평가 config.
#   평가 설정은 ./test.py 와 같다(계획만, HiP-AD 원본과 같음). 모델 구조(E3·FPN bf16·ego_input_encoder)와 val 파이프라인
#   (Collect "ego_input")·data.test.ego_input_file(b2d_ego_input_val.pkl)은 학습 config 를 따른다. 스트림 그룹은 학습 전용 → 1.
_base_ = ["./train_e3_nuim.py"]
evaluation = dict(
    eval_mode=dict(with_det=False, with_tracking=False, with_map=False, with_motion=False,
                   with_planning=True, tracking_threshold=0.2, motion_threshhold=0.2),
)
model = dict(head=dict(onedecoder_head=dict(num_stream_groups=1)))
