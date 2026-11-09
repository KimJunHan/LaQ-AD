# 수정(2026-10-16): 교사 캐시 7 폴더 · 토큰 유형 임베딩 — 2단계 평가 config. ./train.py 상속, 평가 설정은 ../../stage2/test_v2.py 와 같다.
_base_ = ["./train.py"]
# 수정(2026-10-16): 설계 검토 반영 — teacher_meta_expect(prompt_mode="B"·teacher_chain_verified·constrained_decoding)와
#   색인 layout_ok 검사(require_index)는 ./train.py 의 teacher_loader 를 그대로 물려받아 val/test 파이프라인에도 적용된다
#   (여기서 변수를 다시 정의해도 이미 만들어진 파이프라인에는 반영되지 않으므로 다시 적지 않는다; test_teacher_cache_cpu.py (2) 가 4 개 config 모두 검사).
evaluation = dict(
    eval_mode=dict(with_det=True, with_tracking=True, with_map=True, with_motion=True,
                   with_planning=False, tracking_threshold=0.2, tracking_relink=True,
                   motion_threshhold=0.2),
)
