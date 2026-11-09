# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   B2D 교사 캐시 2단계 평가 config. ./train.py 상속, 평가 설정은 ../../stage2/test.py 와 같다(개방루프 = 계획만, HiP-AD 원본과 같음).
#   폐쇄루프에서는 에이전트가 2 Hz 로 교사를 다시 돌리고 사이 프레임은 직전 출력을 재사용한다(캐시 로더 대신).
_base_ = ["./train.py"]
evaluation = dict(
    eval_mode=dict(with_det=False, with_tracking=False, with_map=False, with_motion=False,
                   with_planning=True, tracking_threshold=0.2, motion_threshhold=0.2),
)
model = dict(head=dict(onedecoder_head=dict(num_stream_groups=1)))
