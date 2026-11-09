# 수정(2026-10-27): e4_gate5 평가 config — 모델·파이프라인은 학습 config 그대로, 검출·지도만(추적 끔; 비교 지표 = mAP·NDS·지도 mAP·범주별 AP).
_base_ = ["./e4_gate5.py"]
evaluation = dict(
    eval_mode=dict(with_det=True, with_tracking=False, with_map=True, with_motion=False,
                   with_planning=False, tracking_threshold=0.2, tracking_relink=True, motion_threshhold=0.2),
)
