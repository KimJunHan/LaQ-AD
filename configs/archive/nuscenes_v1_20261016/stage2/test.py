# 수정(2026-10-15): 경로 이동 configs/nuscenes/hipad_nusc_laq_stage2_test.py → configs/nuscenes/stage2/test.py
# 수정(2026-10-09): 테스트용 config. 학습 config 를 그대로 상속하고 평가 설정만 명시한다.
#   사용: bash tools/dist_test.sh configs/nuscenes/hipad_nusc_laq_stage2_test.py <ckpt> 1 --eval bbox  (단일 GPU·launcher 없음 — 추적 ID 는 추론 순서 의존)
_base_ = ["../../../nuscenes/stage2/_base_common.py"]  # 수정(2026-10-16): v1 실험 config 제거 — 기록 보관용으로 이동, 종전 ./train.py(= 현 nuscenes/stage2/_base_common.py, 내용 동일)를 가리킴. 실행용 아님

# 2단계 평가 = det / track / map / motion (계획 L2 는 ego-status 문제로 보고하지 않음 — 논문 한계 절).
evaluation = dict(
    eval_mode=dict(with_det=True, with_tracking=True, with_map=True, with_motion=True,
                   with_planning=False, tracking_threshold=0.2, tracking_relink=True,
                   motion_threshhold=0.2),
)
