# 수정(2026-11-06): 사용자 결정 "B2D 1단계: GPU1 이 비어 있으면 GPU1+GPU2 DDP2" — ./train_e3_nuim.py(1 GPU × 16 × 누적 4)의 분산 등가 config.
#   본뜬 것: configs/navsim/v2_teacher/stage1/train_e3_nuim_ddp2.py(NAVSIM 1단계 DDP2), nuScenes ../../../nuscenes/v2_teacher/stage1/train_e3_ddp4.py.
#   레시피·데이터·모델·lr 2e-4·EMA decay/ramp 는 ./train_e3_nuim.py 와 같고 분산 등가만 바꾼다.
#   (1) GPU 2장 × batch 16 × 누적 2(GPU 당 스트림 그룹 2): optimizer step 당 독립 시퀀스 64·표본 64·lr 그대로.
#       샘플러 슬롯 = 전역 (rank × 2 + k) × 16 + j → 1 GPU 그룹 4개(k × 16 + j)와 같은 전역 슬롯 0..63(group_in_batch_sampler.py).
#   (2) iter 기준 값 = optimizer step × 2(1 GPU 는 × 4): 에폭당 micro-iter 14,672 → 7,336(step 3,668 불변), 총 176,064 → 88,032
#       (step 44,016 불변), warmup 2,000 → 1,000(= step 500), 체크포인트 매 에폭, EMA update_every 4 → 2
#       (decay 0.999801·ramp 2,005 는 step 기준이라 불변: 창 5,000 × 44,016 / 43,900, ramp 2,000 × 44,016 / 43,900).
#   (3) DDP 규칙 = NAVSIM·nuScenes DDP config 와 같다: backbone_cp_non_reentrant(DDP + with_cp 충돌 해결, 수식 동일),
#       seed_rank_offset(rank 별 주 프로세스 난수; 샘플러 장면 순열은 rank 공통 seed). 1 GPU 대비 차이: BN 이동 통계 rank0,
#       rank 별 증강 난수(B2D 는 크기·자르기·좌우 뒤집기·광도 왜곡 — 표본별 증강 값이 1 GPU 런과 다름, 분포는 같음), reduce_mean(num_pos) 2 rank 평균.
#   결과 폴더 results/hipad_b2d_laq_v2_teacher_stage1_e3_nuim_ddp2 (1 GPU 런과 분리). 2단계는 그 폴더의 iter_88032_ema.pth 에서 시작
#   (scripts/b2d/run_v2_teacher_e3_nuim_20261106.sh 가 --cfg-options load_from 으로 넘김; 2단계 config 무수정).
#   평가 config = ./test_e3_nuim.py(모델 구조 같음).
_base_ = ["./train_e3_nuim.py"]

num_gpus, batch_size, cumulative_iters, num_epochs = 2, 16, 2, 12
num_stream_groups = cumulative_iters
b2d_train_frames = 234769
num_iters_per_epoch = int(b2d_train_frames // (num_gpus * batch_size * cumulative_iters)) * cumulative_iters   # 7,336

data = dict(samples_per_gpu=batch_size, num_stream_groups=num_stream_groups)
optimizer_config = dict(cumulative_iters=cumulative_iters)
lr_config = dict(warmup_iters=500 * cumulative_iters)                    # 1,000 micro-iter = step 500
runner = dict(max_iters=num_iters_per_epoch * num_epochs)                # 88,032 = step 44,016
checkpoint_config = dict(interval=num_iters_per_epoch)
evaluation = dict(interval=num_iters_per_epoch)
custom_hooks = [dict(type="EMASidecarHook", decay=0.999801, ramp_steps=2005, update_every=cumulative_iters,
                     max_keep=3, priority="ABOVE_NORMAL")]
model = dict(head=dict(onedecoder_head=dict(num_stream_groups=num_stream_groups)))
backbone_cp_non_reentrant = True
seed_rank_offset = True

run_name = "hipad_b2d_laq_v2_teacher_stage1_e3_nuim_ddp2"
