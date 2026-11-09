# 수정(2026-11-09): C3 GRPO 설계 v4.1 §9.7·§10 — 신규 파일. GRPO 학습 루프(tools/navsim/grpo/train_loop.py) 전용 설정(grpo dict 만).
# 모델 config·시작 체크포인트·관문 산출물(κ, σ_ref, G0 커널, B0 기준, μ_ref)은 여기 두지 않는다 — 체인(scripts/navsim/run_grpo_navsim_chain.sh)이
#   MODEL_CFG·GRPO_INIT_CKPT·GATES_DIR 로 넘긴다(1·2단계 이름이 ..._ddp2_gtfilt 계열로 바뀌어도 이 파일은 그대로).
# 값은 설계 문서 사전 고정값(val 로 고르지 않음). 런마다 variant·seed 만 다르다(§10 공통 조건).
grpo = dict(
    variant="Full", seed=0,
    steps=3000, M=4, G=8, micro_batch=8, accum=6, workers=12,       # §9.2: 8 × 6 = 48 프레임/step, 스트림 그룹 6
    lr=1e-5, wd=1e-4, warmup=100, warmup_ratio=0.1, clip=1.0,       # §9.7: AdamW, 학습 대상 모두 lr 1e-5(lr_mult 1), cosine
    ema_decay=0.99708, ema_ramp=137,                                # §9.7: 1 − 1/342, ramp 137 step
    lambda_bc=0.1, lambda_cls=0.5, rho=0.8,                         # §8, §2.1
    ckpt_interval=250, log_interval=10, monitor_interval=50,
    g3_start=250, g3_window=500, g3_stop_pp=0.05, g3_ep_drop=0.05,  # §9.8
    bc_target="gt",                                                 # §8 L_BC = 2단계 계획 손실(GT). "sft" 는 MCMC 정답(사용자 확인 대상)
    score_workers=16,
)
