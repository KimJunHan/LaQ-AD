# 수정(2026-10-15): 경로 이동 configs/nuscenes/hipad_nusc_laq_stage2_nodiff_test.py → configs/nuscenes/stage2/nodiff_test.py
# 수정(2026-10-12): 확산만 끈 2단계의 평가 config. 대표 모델 2단계 평가(hipad_nusc_laq_stage2_test.py)와 같은 평가 설정.
_base_ = ["./nodiff_train.py"]

evaluation = dict(
    eval_mode=dict(with_det=True, with_tracking=True, with_map=True, with_motion=True,
                   with_planning=False, tracking_threshold=0.2, tracking_relink=True,
                   motion_threshhold=0.2),
)
