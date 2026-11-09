# 수정(2026-11-05): 사용자 지시 "MCMC SFT 확인한대로 진행" — 군 (c) MCMC-SFT: 정답 = MH 체인 최종 상태 x*(N=10, ε=0.5 m).
_base_ = ["./sft_target_base.py"]
# 수정(2026-11-05): 사용자 결정 "lr 1/10, 부드러운 잡음" — v2.1 부드러운 제안(축별 제곱지수 핵 ℓ=1 s)으로 다시 만든 정답(v2 독립 잡음 정답 nuim_e3_ema/ 는 기록으로 둠)
# 수정(2026-11-05): 사용자 결정 "속도 구간별 Σ" — v2.2 정답(현재 속도 구간별 Σ, 경계 0.5/3/6/9 m/s; v21 은 기록으로 둠)
# 수정(2026-11-05): 사용자 결정 "C 에 후진 금지" — v2.3 정답(v2.2 + C 에 GT 에 없는 5 cm 넘는 후진 금지; v22·v21 은 기록으로 둠)
data = dict(train=dict(plan_sft_target_file="results/mcmc_sft/nuim_e3_ema_v23/targets_mcmc.npz"))
run_name = "hipad_nusc_laq_v2_teacher_stage3_sft_mcmc"
