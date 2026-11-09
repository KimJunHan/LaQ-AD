# 수정(2026-10-19): 사용자 결정 — 언어 토큰 자리에 LiDAR/레이더 토큰만 바꿔 넣는 비교(인코더 함께 학습)
#   2단계 평가 config. ./train.py 상속, 평가 설정은 ../../v2_teacher/stage2/test.py 와 같다(평가 프로토콜 불변).
_base_ = ["./train.py"]
evaluation = dict(
    eval_mode=dict(with_det=True, with_tracking=True, with_map=True, with_motion=True,
                   with_planning=False, tracking_threshold=0.2, tracking_relink=True,
                   motion_threshhold=0.2),
)
