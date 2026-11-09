# 수정(2026-10-23): 사용자 결정 "비교 1회 추가" — 기여 3(지도 정렬·선택적 가우시안 잡음)의 효과를 보이기 위한 비교 2단계.
#   ./train.py 와 **계획 확산 잡음만** 다르다: DiffusionDrive(Liao et al., CVPR 2025) 원래 방식
#   (PlanTruncatedDiffusion: 학습 t ~ U{0..49}, 정규화 공간 등방 잡음, 지도 정렬·경계 잘라내기·선택 없음).
#   같은 1단계 체크포인트(load_from)·같은 레시피·같은 지도 기억 뱅크(PlanMapMemoryInstanceBank, 잡음에 쓰이지 않음)·같은 평가.
#   일정·정규화·추론 t(8)·시드·시점 임베딩 파라미터는 같다(state_dict 동일).
#   run_name hipad_nusc_laq_v2_teacher_stage2_noise_dd
_base_ = ["./train.py"]

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

# 수정(2026-10-23): 무결점 검사 1차 지적 D1 — run_name 을 물려받으면 본 2단계(hipad_nusc_laq_v2_teacher_stage2) 이름이 되어
#   --work-dir 없이 실행할 때 본 런 폴더를 덮어쓸 수 있다 → 비교 런 이름을 명시한다.
run_name = "hipad_nusc_laq_v2_teacher_stage2_noise_dd"
