# 수정(2026-11-06): 사용자 결정 "NAVSIM MCMC SFT" — 군 MCMC-SFT: 정답 = MH 체인 최종 상태 x*(N=10, ε=0.5 m, C = GT 상대 7 하위 점수·후진 금지; 설계 §1–§2).
_base_ = ["./sft_target_base.py"]
data = dict(train=dict(plan_sft_target_file="results/mcmc_sft_navsim/s2e3nuim_gtfilt_ema_v1/targets_mcmc.npz"))   # 수정(2026-11-09): GT 필터 2단계 덤프 #이전 s2e3nuim_ema_v1
evaluation = dict(jsonfile_prefix="evaluation/hipad_navsim_laq_v2_teacher_stage3_sft_mcmc")
run_name = "hipad_navsim_laq_v2_teacher_stage3_sft_mcmc"
