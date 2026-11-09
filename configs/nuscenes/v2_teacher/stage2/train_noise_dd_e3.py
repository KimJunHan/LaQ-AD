# 수정(2026-10-31): 기여 3 비교(DiffusionDrive 원래 잡음) 2단계를 새 1단계에서 다시 — ./train_noise_dd.py 와 같은 잡음 설정,
#   그 밖(과제별 FFN·norm, 1단계 EMA 시작, EMA 저장)은 ./train_e3.py 와 같다. 본 2단계와 비교 2단계의 출발점이 같도록.
_base_ = ["./train_e3.py"]

embed_dims = 256
model = dict(
    head=dict(
        onedecoder_head=dict(
            plan_diffusion=dict(_delete_=True, type="PlanTruncatedDiffusion", embed_dims=embed_dims, ego_fut_ts=6,
                                num_train_timesteps=1000, trunc_steps=50, infer_t=8, beta_start=1e-4, beta_end=0.02,
                                norm_scale=(13.0, 29.0), norm_offset=(-1.0, 27.0), infer_seed=0),
        ),
    ),
)

run_name = "hipad_nusc_laq_v2_teacher_stage2_noise_dd_e3"
