# 수정(2026-11-06): 사용자 결정 "B2D 학생 = 최종 레시피" — ./train_e3_nuim.py(nuImages + E3, FPN bf16) 의 개방루프 평가 config.
#   평가 설정은 ./test.py 와 같다(det·map; 1단계 계획 손실 0). 모델 구조(E3 과제별 FFN·norm, FPN bf16)는 학습 config 를 따른다.
#   백본 초기값 설정(init_cfg)은 체크포인트 가중치가 덮어써 결과에 영향 없음. 스트림 그룹은 학습 전용 → 1(./test.py 와 같은 이유).
#   DDP2 로 학습한 경우(./train_e3_nuim_ddp2.py)도 모델 구조가 같아 이 config 로 평가한다.
_base_ = ["./train_e3_nuim.py"]
evaluation = dict(
    eval_mode=dict(with_det=True, with_tracking=False, with_map=True, with_motion=False,
                   with_planning=False, tracking_threshold=0.2, motion_threshhold=0.2),
)
model = dict(head=dict(onedecoder_head=dict(num_stream_groups=1)))
