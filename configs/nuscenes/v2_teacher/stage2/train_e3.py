# 수정(2026-10-31): 사용자 지시 "전체 진행" — 새 1단계(E3 과제별 FFN·norm, 4-GPU 100 에폭, EMA)에서 시작하는 본 2단계.
#   ./train.py(본 2단계: 지도 정렬 선택적 잡음, 1 GPU 8×6, G=6, 10 에폭)와 레시피·데이터·평가가 같고 아래만 다르다.
#   (1) 과제별 FFN·norm(task_specific_ffn_norm=True) — 1단계 E3 와 같은 구조(체크포인트 키를 그대로 읽기 위함).
#   (2) load_from = 1단계 EMA 가중치(iter_43900_ema.pth, tools/ema_to_ckpt.py 로 만든 것; 1단계 val 에서 원 가중치보다 전 지표 높음).
#   (3) EMA 가중치 함께 저장(EMASidecarHook, 학습에는 영향 없음; optimizer step = 6 micro-iter 마다 갱신).
#   (4) 계획·자차 쿼리 전용 FFN·norm(layers.N.plan / layers.N.ego)에 lr ×10 — 1단계에는 계획·자차 손실이 없어 이 모듈들은
#       기울기 0(실측: 1단계 체크포인트에서 norm weight 0.9913·bias 0 = 초기값에 weight decay 만 적용된 값)인 사실상 신규 모듈이다.
#       2단계 신규 모듈은 모두 ×10 이라는 기존 규칙(../../stage2/train_v2.py 의 ego_* 수정 (8)과 같은 이유)을 그대로 적용.
#       공유 FFN(./train.py)에서는 계획 쿼리가 1단계에서 학습된 FFN 을 썼으므로 이 차이는 E3 구조에서만 생긴다.
#       키는 '.plan.'·'.ego.' 부분 문자열 — 1단계 체크포인트에서 이 둘에 걸리는 키는 정확히 layers.*.plan/ego 144 개뿐임을 확인.
_base_ = ["./train.py"]

cumulative_iters = 6  # ../../stage2/train_v2.py 와 같은 값(재기재)

model = dict(head=dict(onedecoder_head=dict(task_specific_ffn_norm=True)))

optimizer = dict(paramwise_cfg=dict(custom_keys={".plan.": dict(lr_mult=10.0), ".ego.": dict(lr_mult=10.0)}))

load_from = "./results/hipad_nusc_laq_v2_teacher_stage1_e3_ddp4/iter_43900_ema.pth"

# EMA 설정은 1단계와 "학습 길이 대비 같은 비율"로 맞춘다(결과를 보기 전 결정). 1단계: optimizer step 43,900, decay 0.9998(평균 창 ≈ 5,000 step
#   = 11.4 %), ramp 2,000(4.6 %). 2단계: optimizer step 5,860(35,160 / 6) → 같은 비율이면 창 ≈ 667 step(decay = 1 − 1/667 ≈ 0.9985),
#   ramp ≈ 267(= 2,000 × 5,860 / 43,900). 1단계 값을 그대로 쓰면 ramp 때문에 마지막 decay 가 0.946(창 ≈ 19 step)으로
#   사실상 원 가중치와 같아진다(독립 검토 지적).
custom_hooks = [dict(type="EMASidecarHook", decay=0.9985, ramp_steps=267, update_every=cumulative_iters,
                     max_keep=3, priority="ABOVE_NORMAL")]

run_name = "hipad_nusc_laq_v2_teacher_stage2_e3"
