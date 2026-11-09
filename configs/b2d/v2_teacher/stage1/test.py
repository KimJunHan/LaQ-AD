# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   B2D 교사 캐시 1단계 평가 config. ./train.py 상속, 평가 설정은 ../../stage1/test.py 와 같다(det/map; 1단계 계획 손실 0).
#   교사 로더(LoadTeacherCacheB2D, 2 Hz 키프레임)는 ./train.py 의 test_pipeline 을 그대로 물려받는다.
_base_ = ["./train.py"]
evaluation = dict(
    eval_mode=dict(with_det=True, with_tracking=False, with_map=True, with_motion=False,
                   with_planning=False, tracking_threshold=0.2, motion_threshhold=0.2),
)
# 스트림 그룹은 학습 전용 — 평가·폐쇄루프(with_close_loop=True)는 1(../../stage1/test.py 와 같은 이유).
model = dict(head=dict(onedecoder_head=dict(num_stream_groups=1)))
