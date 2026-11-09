# 수정(2026-11-05): 사용자 결정 "NAVSIM 1단계 = nuImages + E3" — nuScenes 1단계 최종(configs/nuscenes/v2_teacher/stage1/train_e3_nuim_ddp4.py,
#   E3 + nuImages R50, results/hipad_nusc_laq_v2_teacher_stage1_e3_nuim_ddp4/iter_43900_ema.pth)과 같은 레시피로 NAVSIM 1단계를 맞춘다.
#   ./train.py(NAVSIM 1단계, 무수정)를 상속하고 아래 (1)~(4)만 바꾼다. 데이터·지도·앵커·교사 캐시·증강·lr 4e-4·wd·cosine·clip·fp16(정적 32)·
#   EMA 규칙·에폭 정의(장면 103,288 / 에폭)는 ./train.py 와 같다.
#
#   (1) E3: onedecoder_head.task_specific_ffn_norm=True — 통합 디코더의 ffn·norm 자리를 쿼리 종류(det/map/ego/plan)별 모듈로 분리
#       (models/decoder/unified_decoder.py; 데이터셋과 무관한 구조 변경). nuScenes train_e3_ddp4.py (1) 과 같다.
#   (2) nuImages R50 백본 초기값: nuScenes train_e3_nuim_ddp4.py 와 같은 가중치·같은 방식(pretrained=None + init_cfg Pretrained,
#       prefix "backbone." → 검출기 안의 backbone.* 만 읽음, FPN 은 구조가 달라 읽지 않음). ImageNet(ckpts/resnet50-19c8e357.pth) 대신.
#       [비교 조건] HiP-AD·SparseDrive 원본은 ImageNet 초기값 → "nuImages 사전학습 백본" 으로 따로 표기.
#   (3) FPN(img_neck) bfloat16 + FPN lr ×0.5: nuScenes 1단계 기본 레시피(configs/nuscenes/stage1/train_v2.py, 2026-10-20 근본 원인 수정 —
#       BN 있는 FPN + lr 4e-4 에서 FPN 출력이 fp16 최대를 넘어 Inf)에 있는데 NAVSIM ../../stage1/train.py(10-16 동기화)에는 빠져 있다.
#       E3 고유 항목은 아니지만 nuScenes 최종(E0·E3 모두)이 이 값으로 학습됐고, NAVSIM 도 같은 fp16·BN FPN·lr 4e-4 라 같은 넘침 위험이 있다.
#       (mmcv custom_keys 는 dict 병합이라 백본 ×0.5 를 다시 적는다.)
#   (4) micro-batch 2 × 누적 32(스트림 그룹 32) → 16 × 누적 4(스트림 그룹 4): optimizer step 당 독립 시퀀스 64·표본 64·lr 4e-4 그대로
#       (nuScenes 1 GPU 레시피 16 × 4 = 4-GPU DDP 16 × 1 과 같은 등가 규칙; 샘플러 슬롯 = batch × 그룹). 한 forward 의 BN 배치가
#       nuScenes 최종(GPU 당 16 표본)과 같아진다(종전 2 표본). iter 기준 값은 누적 수에 맞춘다: 에폭당 micro-iter 51,616 → 6,452
#       (step 1,613 불변), 총 5,161,600 → 645,200(step 161,300 불변), warmup 16,000 → 2,000 micro-iter(= step 500 불변),
#       EMA update_every 32 → 4(decay·ramp 는 step 기준이라 배치 변경과 무관). (총 길이는 아래 (5)에서 27 에폭으로 줄였다.)
#   (5) 수정(2026-11-05): 사용자 결정(권고안 (a) 승인) — 학습 길이 100 → 27 에폭. NAVSIM 에폭(장면 103,288 = step 1,613)이 nuScenes(28,130 표본
#       = step 439)의 3.7 배라, 27 에폭 = step 43,551 ≈ nuScenes 최종 1단계 43,900 step(표본 약 2.79M vs 2.81M)으로 optimizer step 수를 맞춘다.
#       navtest 를 보기 전에 정한 일정(속도 측정 결과 100 에폭 = 9~16.5 일로 11-30 마감 불가). 총 micro-iter 645,200 → 174,204.
#       EMA 는 같은 규칙(창 = 5,000 × step / 43,900, ramp = 2,000 × step / 43,900)으로 다시 계산: 창 4,960.2 → decay 1 − 1/4,960.2 = 0.999798,
#       ramp 1,984.1 → 1,984. (100 에폭이었다면 0.999946 / 7,349.) 2단계 load_from 도 iter_174204_ema.pth 로 바꿨다.
#   nuScenes 최종 중 옮기지 않는 것(근거): GPU 4장 DDP(NAVSIM 은 GPU3 1장 — 위 (4)가 등가), backbone_cp_non_reentrant·seed_rank_offset
#   (DDP 전용), Collect meta_keys 에서 stream_group 제거(그룹 1 일 때만 — 여기는 그룹 4), log interval 25(표시만), TF32 끔(기본값 그대로).
#   데이터셋 고유 차이(유지): 증강(NAVSIM 은 고정 크기·뒤집기/회전 없음), 지도 앵커 생성 방식, 클래스·카메라·해상도·계획 8 포즈.
_base_ = ["./train.py"]

# ---- (4) 배치·누적·반복 수 ----
num_gpus, batch_size, cumulative_iters, num_epochs = 1, 16, 4, 27   # (5) #이전 num_epochs 100
num_stream_groups = cumulative_iters
navsim_train_scenes = 103288   # ./train.py 와 같음(에폭 = 장면 수 기준)
num_iters_per_epoch = int(navsim_train_scenes // (num_gpus * batch_size * cumulative_iters)) * cumulative_iters   # 6,452

data = dict(samples_per_gpu=batch_size, num_stream_groups=num_stream_groups)
optimizer_config = dict(cumulative_iters=cumulative_iters)
lr_config = dict(warmup_iters=500 * cumulative_iters)                    # 2,000 micro-iter = step 500
runner = dict(max_iters=num_iters_per_epoch * num_epochs)                # 174,204 = step 43,551  #이전 645,200(100 에폭)
checkpoint_config = dict(interval=num_iters_per_epoch)
evaluation = dict(interval=num_iters_per_epoch,
                  jsonfile_prefix="evaluation/hipad_navsim_laq_v2_teacher_stage1_e3_nuim")
custom_hooks = [dict(type="EMASidecarHook", decay=0.999798, ramp_steps=1984, update_every=cumulative_iters,  # (5) #이전 0.999946 / 7349
                     max_keep=3, priority="ABOVE_NORMAL")]

# ---- (1) E3 · (2) nuImages R50 · (3) FPN bf16 ----
model = dict(
    neck_dtype="bf16",
    img_backbone=dict(
        pretrained=None,
        init_cfg=dict(type="Pretrained", prefix="backbone.",
                      checkpoint="ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth")),
    head=dict(onedecoder_head=dict(task_specific_ffn_norm=True, num_stream_groups=num_stream_groups)),
)
# ---- (3) FPN lr ×0.5 ----
optimizer = dict(paramwise_cfg=dict(custom_keys={"img_backbone": dict(lr_mult=0.5), "img_neck": dict(lr_mult=0.5)}))

run_name = "hipad_navsim_laq_v2_teacher_stage1_e3_nuim"
