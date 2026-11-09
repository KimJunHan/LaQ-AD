# 수정(2026-10-15): 경로 이동 configs/b2d/hipad_b2d_laq_stage1_test.py → configs/b2d/stage1/test.py
# 수정(2026-10-09): 테스트용 config. 학습 config 를 그대로 상속하고 평가 설정만 명시한다.
#   사용: bash tools/dist_test.sh configs/b2d/hipad_b2d_laq_stage1_test.py <ckpt> 1 --eval bbox  (단일 GPU·launcher 없음 — 추적 ID 는 추론 순서 의존)
_base_ = ["./train.py"]

# B2D 1단계 개방루프 평가 = det / map (1단계는 계획·자차 손실 0 이라 계획 평가는 무의미).
#   GitHub 원본은 1·2단계 모두 with_planning 만 켠다 — 1단계 인지 점검을 위해 det/map 으로 바꾼다.
evaluation = dict(
    eval_mode=dict(with_det=True, with_tracking=False, with_map=True, with_motion=False,
                   with_planning=False, tracking_threshold=0.2, motion_threshhold=0.2),
)

# 수정(2026-10-16): v2 학습 config 반영 — 스트림 그룹(num_stream_groups)은 학습 전용. 폐쇄루프 에이전트(hipad_b2d_agent.py:274)가
#   with_close_loop=True 로 빌드하는데 디코더는 둘을 함께 허용하지 않는다(models/decoder/unified_decoder.py:342).
#   평가에서는 1 로 둔다. 개방루프 추론(model.eval())은 원래 그룹 분기를 타지 않고, state_dict 도 같다(그룹 상태는 파이썬 속성).
model = dict(head=dict(onedecoder_head=dict(num_stream_groups=1)))
