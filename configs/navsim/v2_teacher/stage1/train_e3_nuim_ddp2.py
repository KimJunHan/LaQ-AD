# 수정(2026-11-06): 사용자 결정 — 1단계를 "처음부터" 이 config(GPU1+GPU3 DDP2, torchrun 2)로 학습한다(1 GPU → DDP 전환 폐기).
#   scripts/navsim/run_v2_teacher_e3_nuim_20261105.sh 가 직접 torchrun 으로 띄운다(초기화 = nuImages R50 init_cfg, ./train_e3_nuim.py 와 같음).
#   아래 (4) 의 전환 도구·전환 스크립트(run_stage1_e3_nuim_ddp2_switch_20261105.sh)는 쓰지 않음(deprecated). 레시피·반복 수 불변.
# 수정(2026-11-05): 사용자 결정 "nuScenes MCMC 체인(GPU1)이 끝나면 NAVSIM 1단계를 GPU1+GPU3 DDP 로 이어 간다" — 전환(재개) 전용 config.
#   ./train_e3_nuim.py(1 GPU × 16 × 누적 4, 스트림 그룹 4)와 레시피·데이터·모델·lr·EMA decay/ramp 가 같고 분산 등가만 바꾼다.
#   (1) GPU 2장 × batch 16 × 누적 2(GPU 당 스트림 그룹 2): optimizer step 당 독립 시퀀스 64·표본 64·lr 4e-4 그대로.
#       샘플러 슬롯 = 전역 (rank × 2 + k) × 16 + j → 가상 GPU 4개 배정이 1 GPU 그룹 4개(k × 16 + j)와 같은 전역 슬롯 0..63
#       (models/loader/samplers/group_in_batch_sampler.py) → 같은 seed 면 슬롯별 장면 순서가 1 GPU 런과 같다.
#   (2) iter 기준 값 = optimizer step × 2 (1 GPU 는 × 4): 에폭당 6,452 → 3,226, 총 174,204 → 87,102(step 43,551 불변),
#       warmup 2,000 → 1,000(= step 500), 체크포인트 매 에폭, EMA update_every 4 → 2(decay 0.999798·ramp 1,984 는 step 기준 불변).
#       LR(cosine·linear warmup)은 runner.iter / max_iters 로 계산 → step 기준으로 같은 곡선(누적 묶음 안 위치 차이 ≤ 0.25 step, 상대 lr 차 < 1e-5).
#   (3) DDP 규칙 = nuScenes ../../../nuscenes/v2_teacher/stage1/train_e3_ddp4.py 와 같다: backbone_cp_non_reentrant(DDP+with_cp 충돌 해결,
#       수식 동일), seed_rank_offset(rank 별 주 프로세스 난수; 샘플러 장면 순열은 rank 공통 seed). 1 GPU 대비 차이: BN 이동 통계 rank0,
#       rank 별 증강 난수(NAVSIM 은 크기 고정·뒤집기/회전 없음 → 광도 왜곡 난수만), reduce_mean(num_pos) 2 rank 평균.
#   (4) 1 GPU 체크포인트에서의 전환 = tools/navsim/convert_1gpu_to_ddp_ckpt.py(iter·EMA 파일 번호 변환) → scripts/navsim/
#       run_stage1_e3_nuim_ddp2_switch_20261105.sh. 결과 폴더는 1 GPU 런과 분리(results/hipad_navsim_laq_v2_teacher_stage1_e3_nuim_ddp2).
#       2단계는 이 폴더의 iter_87102_ema.pth 에서 시작(전환 스크립트가 --cfg-options load_from 으로 넘김; 2단계 config 무수정).
_base_ = ["./train_e3_nuim.py"]

num_gpus, batch_size, cumulative_iters, num_epochs = 2, 16, 2, 27
num_stream_groups = cumulative_iters
navsim_train_scenes = 103288
num_iters_per_epoch = int(navsim_train_scenes // (num_gpus * batch_size * cumulative_iters)) * cumulative_iters   # 3,226

data = dict(samples_per_gpu=batch_size, num_stream_groups=num_stream_groups)
optimizer_config = dict(cumulative_iters=cumulative_iters)
lr_config = dict(warmup_iters=500 * cumulative_iters)                    # 1,000 micro-iter = step 500
runner = dict(max_iters=num_iters_per_epoch * num_epochs)                # 87,102 = step 43,551
checkpoint_config = dict(interval=num_iters_per_epoch)
evaluation = dict(interval=num_iters_per_epoch,
                  jsonfile_prefix="evaluation/hipad_navsim_laq_v2_teacher_stage1_e3_nuim_ddp2")   # 수정(2026-11-06): 표시용(--no-validate) #이전 1 GPU 런 이름 상속
custom_hooks = [dict(type="EMASidecarHook", decay=0.999798, ramp_steps=1984, update_every=cumulative_iters,
                     max_keep=3, priority="ABOVE_NORMAL")]
model = dict(head=dict(onedecoder_head=dict(num_stream_groups=num_stream_groups)))
backbone_cp_non_reentrant = True
seed_rank_offset = True

run_name = "hipad_navsim_laq_v2_teacher_stage1_e3_nuim_ddp2"
