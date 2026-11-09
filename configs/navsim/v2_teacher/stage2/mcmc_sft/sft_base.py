# 수정(2026-11-06): 사용자 결정 "NAVSIM MCMC SFT" — 3단계 계획 SFT 공통 config = 군 "기반 이어 학습(base-continue)":
#   정답 변경 없음(GT, 데이터셋 기본 키 gt_ego_fut_trajs_2hz; SFT 옵션 모두 끔 → 학습 경로는 ../train_e3_nuim.py 와 같고 일정만 다르다).
#   nuScenes configs/nuscenes/v2_teacher/stage2/mcmc_sft/sft_base.py(v2.3)와 같은 규칙. 설계 docs/mcmc_sft_navsim_plan.md §5.
#   ../train_e3_nuim.py(NAVSIM 2단계, 1 GPU 8 × 누적 6) 와 모델·데이터·손실·plan_target·fp16(정적 32)·lr 배수가 같고 아래만 다르다.
#   (1) load_from = NAVSIM 2단계 EMA(results/hipad_navsim_laq_v2_teacher_stage2_e3_nuim/iter_129060_ema.pth —
#       scripts/navsim/run_v2_teacher_e3_nuim_20261105.sh 의 ema_convert 산출 이름 = iter_<M2>_ema.pth, M2 = 129,060). optimizer 상태는 새로(가중치만).
#   (2) 2 에폭 = 2 × 12,906 micro-iter = 25,812 (optimizer step 4,302). 근거: nuScenes 는 2단계 10 에폭(step 5,860) 뒤 2 에폭(step 1,172)
#       = 2단계 길이의 20 %. NAVSIM 2단계도 10 에폭(step 21,510) → 같은 20 % = 2 에폭(에폭 = navtrain 장면 103,288 기준, ../train_e3_nuim.py).
#   (3) lr 고정(Fixed, warmup 없음) = 2단계 정점 lr 의 1/10 = 2e-6. 정점 출처 configs/navsim/stage2/train.py:731 `lr=2e-5`
#       (v2_teacher/stage2/train.py → train_e3_nuim.py 어디에서도 lr 을 덮어쓰지 않고 lr_mult 만 더한다). 배수(img_backbone ×0.5,
#       계획·자차·모션·plan_diffusion·'.plan.'·'.ego.'·ego_input_encoder ×10)는 상속 그대로.
#   (4) EMA 같은 규칙(학습 길이 대비 같은 비율 — ../train.py 주석: 창 11.39 %, ramp 4.556 % = 2,000/43,900):
#       step 4,302 → 창 0.11390 × 4,302 ≈ 490(decay = 1 − 1/490 = 0.997959), ramp 0.04556 × 4,302 ≈ 196, update_every 6(같음).
#   NAVSIM 학습은 sequences_split_num=1(장면을 나누지 않음) → 덤프(장면 순서 연속 추론)와 시간 뱅크 리셋 차이가 nuScenes 보다 작다.
# 수정(2026-11-09): GT 필터 2단계로 교체(docs/navsim_gt_filter_20261109.md) — 상속 = ../train_e3_nuim_gtfilt.py(train 파이프라인에 z·시야 GT 필터,
#   평가 파이프라인 불변), load_from = 그 2단계 EMA. 위 (1)·머리 주석의 ../train_e3_nuim.py·..._stage2_e3_nuim 은 ..._gtfilt 로 읽을 것.
_base_ = ["../train_e3_nuim_gtfilt.py"]   # 수정(2026-11-09): #이전 ["../train_e3_nuim.py"]

cumulative_iters = 6
num_iters_per_epoch = 12906   # = ../train_e3_nuim.py 값(103,288 // 48 × 6)
num_epochs = 2

load_from = "./results/hipad_navsim_laq_v2_teacher_stage2_e3_nuim_gtfilt/iter_129060_ema.pth"   # 수정(2026-11-09): #이전 ..._stage2_e3_nuim/iter_129060_ema.pth

sft_lr = 2e-6
optimizer = dict(lr=sft_lr)
lr_config = dict(_delete_=True, policy="Fixed")
runner = dict(type="IterBasedRunner", max_iters=num_iters_per_epoch * num_epochs)   # 25,812
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
custom_hooks = [dict(type="EMASidecarHook", decay=0.997959, ramp_steps=196, update_every=cumulative_iters,
                     max_keep=3, priority="ABOVE_NORMAL")]
evaluation = dict(interval=num_iters_per_epoch, jsonfile_prefix="evaluation/hipad_navsim_laq_v2_teacher_stage3_sft_base")
run_name = "hipad_navsim_laq_v2_teacher_stage3_sft_base"
