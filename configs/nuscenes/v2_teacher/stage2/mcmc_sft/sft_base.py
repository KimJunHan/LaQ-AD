# 수정(2026-11-05): 사용자 지시 "MCMC SFT 확인한대로 진행" — 3단계 계획 SFT 공통 config(설계 docs/mcmc_sft_plan_design.md v2).
#   = 군 (b) GT-SFT 그대로(정답 = GT, 소스 옵션 모두 끔 → 학습 경로는 ../train_e3_nuim.py 와 비트 동일).
#   ../train_e3_nuim.py(nuImages 2단계 E3) 와 모델·데이터·손실·plan_target("output")·fp16 설정이 같고 아래만 다르다.
#   (1) load_from = 기반 체크포인트 nuImages 2단계 E3 EMA(iter_35160_ema.pth). optimizer 상태는 새로 시작(load_from = 가중치만).
#   (2) 2 에폭 = 2 × 3,516 micro-iter = 7,032 (optimizer step 1,172).
#   (3) lr 고정(Fixed, warmup 없음) = 2단계 정점 lr 의 1/10 = 2e-6.
# 수정(2026-11-05): 사용자 결정 "lr 1/10, 부드러운 잡음" — v2 의 "일정 마지막 값"(2.000004e-08)은 사실상 학습이 없어 바꿈.
#       정점 출처: configs/nuscenes/stage2/_base_common.py:711 `lr=2e-5`(AdamW 기본 lr; warmup 1/3→1 뒤 CosineAnnealing 시작값 = 정점).
#       train_v2.py → v2_teacher/stage2/train.py → train_e3.py → train_e3_nuim.py 어디에서도 lr 을 덮어쓰지 않는다(배수만 추가).
#       파라미터 그룹 배수(img_backbone ×0.5, 계획·자차·모션 등 ×10, '.plan.'/'.ego.' ×10)는 상속 그대로(유효 ×10 그룹 = 2e-5).
#   (4) EMA 같은 규칙(../train_e3.py 주석: 1단계와 "학습 길이 대비 같은 비율" — 창 = 5,000/43,900 = 11.39 %, ramp = 2,000/43,900):
#       optimizer step 1,172 → 창 ≈ 133.5 step(decay = 1 − 1/133.5 = 0.99251), ramp ≈ 53. update_every = 6(같음).
#       EMA 초기값 = 시작 가중치(기반 EMA 체크포인트).
#   주의: 생성 덤프는 test 파이프라인·장면 순서 연속 추론, 학습은 sequences_split_num=2 로 장면을 반으로 나눠 시간 뱅크를 리셋 — 설계 문서 명시 사항.
_base_ = ["../train_e3_nuim.py"]

cumulative_iters = 6
num_iters_per_epoch = 28130 // 8  # 3516 (base 와 같은 값)
num_epochs = 2

load_from = "./results/hipad_nusc_laq_v2_teacher_stage2_e3_nuim/iter_35160_ema.pth"

sft_lr = 2e-6  # 수정(2026-11-05): 사용자 결정 "lr 1/10, 부드러운 잡음" — 정점 2e-5(_base_common.py:711) × 1/10 (위 (3))
optimizer = dict(lr=sft_lr)
lr_config = dict(_delete_=True, policy="Fixed")
runner = dict(type="IterBasedRunner", max_iters=num_iters_per_epoch * num_epochs)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)

custom_hooks = [dict(type="EMASidecarHook", decay=0.99251, ramp_steps=53, update_every=cumulative_iters,
                     max_keep=3, priority="ABOVE_NORMAL")]

run_name = "hipad_nusc_laq_v2_teacher_stage3_sft_gt"
