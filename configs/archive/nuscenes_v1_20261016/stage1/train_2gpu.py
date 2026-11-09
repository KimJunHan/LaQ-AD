# 수정(2026-10-15): 경로 이동 configs/nuscenes/hipad_nusc_laq_stage1_train_2gpu.py → configs/nuscenes/stage1/train_2gpu.py
# 수정(2026-10-10): 1단계 2-GPU(DDP) 실행용 — 사용자가 GPU1 을 임시 허가("다른 연구자가 사용할 때까지만").
#   학습 레시피는 hipad_nusc_laq_stage1_train.py(1-GPU, 16×4)와 같다: GPU당 16 × 2 GPU × 누적 2 = 유효 64,
#   optimizer step 총 43,950 회·warmup 500 step·코사인 일정이 동일하도록 micro-iter 단위 값만 절반으로 환산.
#   GPU1 을 비워야 하면 tools/nusc_ckpt_2gpu_to_1gpu.py 로 latest.pth 의 iter 를 ×2 환산해
#   1-GPU config 로 이어서 학습한다(같은 optimizer step 위치에서 같은 lr).
_base_ = ["../../../nuscenes/stage1/_base_common.py"]  # 수정(2026-10-16): v1 실험 config 제거 — 기록 보관용으로 이동, 종전 ./train.py(= 현 nuscenes/stage1/_base_common.py, 내용 동일)를 가리킴. 실행용 아님
num_gpus = 2
num_iters_per_epoch = 28130 // (2 * 16)          # 879 (GPU 2장 × 16 표본/iter)
optimizer_config = dict(cumulative_iters=2)      # 16 × 2 GPU × 2 = 64
lr_config = dict(warmup_iters=500 * 2)           # optimizer step 500 회
runner = dict(max_iters=num_iters_per_epoch * 100)   # 87,900 micro-iter = 43,950 optimizer step
# 수정(2026-10-12): 저장 간격 2 epoch(1,758 iter, 짝수) — 저장 iter 가 항상 누적 경계(2)이고 ×2 환산 시 1-GPU 누적 경계(4)도 되도록.
#   (879 는 홀수라 홀수 배 저장점은 1-GPU 로 되돌릴 수 없다.)
checkpoint_config = dict(interval=num_iters_per_epoch * 2)
evaluation = dict(interval=num_iters_per_epoch)
