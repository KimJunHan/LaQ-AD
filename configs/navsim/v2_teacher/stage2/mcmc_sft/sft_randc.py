# 수정(2026-11-06): 사용자 결정 "NAVSIM MCMC SFT" — 군 C 안 무작위(randc): 같은 제안·10 단계, 수용 = 1[x' ∈ C](밀도 없음) — 목표 흐리기 효과 분리용.
# 수정(2026-11-06): 사용자 결정 "GT-SFT 군 제거 → randc 로 교체" — 기본 체인 군으로 승격. #이전 "선택 군, 기본 체인에는 없음(RUNS 로 추가)"
_base_ = ["./sft_target_base.py"]
data = dict(train=dict(plan_sft_target_file="results/mcmc_sft_navsim/s2e3nuim_gtfilt_ema_v1/targets_randc.npz"))   # 수정(2026-11-09): GT 필터 2단계 덤프 #이전 s2e3nuim_ema_v1
evaluation = dict(jsonfile_prefix="evaluation/hipad_navsim_laq_v2_teacher_stage3_sft_randc")
run_name = "hipad_navsim_laq_v2_teacher_stage3_sft_randc"
