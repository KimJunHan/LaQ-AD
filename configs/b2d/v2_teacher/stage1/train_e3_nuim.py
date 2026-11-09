# 수정(2026-11-06): B2D 1단계 후보 — nuScenes·NAVSIM 최종 1단계(nuImages + E3, FPN bf16·FPN lr ×0.5)에 맞춘 B2D 판. 사용자 승인 전 기동 금지.
#   ./train.py(B2D v2 교사 1단계, 무수정)를 상속하고 아래 (1)~(4)만 바꾼다. lr 2e-4(HiP-AD 원본 B2D)·wd 0.001·cosine·clip 25·fp16(정적 32)·
#   12 에폭·해상도 640×352·증강·앵커·클래스·교사 캐시 7 폴더·EMA 규칙·에폭 정의(train 234,769 프레임 / 에폭)는 ./train.py 와 같다.
#   본뜬 것: configs/navsim/v2_teacher/stage1/train_e3_nuim.py, configs/nuscenes/v2_teacher/stage1/train_e3_nuim_ddp4.py.
#   (1) E3: onedecoder_head.task_specific_ffn_norm=True (쿼리 종류별 FFN·norm, 데이터셋과 무관한 구조 변경).
#   (2) nuImages R50 백본 초기값(pretrained=None + init_cfg Pretrained prefix "backbone."). ImageNet(ckpts/resnet50-19c8e357.pth) 대신.
#       [비교 조건] HiP-AD 원본은 ImageNet → "nuImages 사전학습 백본" 으로 따로 표기.
#   (3) FPN(img_neck) bfloat16 + FPN lr ×0.5 (nuScenes 1단계 기본 레시피·NAVSIM 최종과 같음; custom_keys 는 dict 병합이라 백본 ×0.5 도 다시 적음).
#   (4) micro-batch 2 × 누적 32(스트림 그룹 32) → 16 × 누적 4(스트림 그룹 4). optimizer step 당 표본 64·독립 시퀀스 64·lr 그대로
#       (NAVSIM 최종 1단계와 같은 등가 규칙). BN 배치 = 16(nuScenes 최종 GPU 당 16 과 같음, 종전 2).
#       iter 기준 값만 누적 수에 맞춘다: 에폭당 micro-iter 117,376 → 14,672(step 3,668 불변), 총 1,408,512 → 176,064(step 44,016 불변),
#       warmup 16,000 → 2,000 micro-iter(= step 500 불변), EMA update_every 32 → 4(decay 0.999801·ramp 2,005 는 step 기준이라 불변).
#       (참고: base 주석의 "batch16 → NaN"(2026-07-06) 은 FlashAttention @auto_fp16 강제 fp16 이 원인으로 같은 날 근본 수정됨.
#        16 × 4 의 수치 안정성은 GPU0 50 iter 스모크만 확인 — docs/b2d_prep_20261106.md.)
#   workers_per_gpu 는 ./train.py 의 6(2026-07-15 사용자 RAM 지시) 그대로 — 스모크 data_time 참고.
#   2단계 짝: ../stage2/train_e3_nuim.py (load_from = 이 run 의 iter_176064_ema.pth).
_base_ = ["./train.py"]

# ---- (4) 배치·누적·반복 수 ----
num_gpus, batch_size, cumulative_iters, num_epochs = 1, 16, 4, 12
num_stream_groups = cumulative_iters
b2d_train_frames = 234769   # ./train.py 와 같음(B2D 공식 train 분할)
num_iters_per_epoch = int(b2d_train_frames // (num_gpus * batch_size * cumulative_iters)) * cumulative_iters   # 14,672

data = dict(samples_per_gpu=batch_size, num_stream_groups=num_stream_groups)
optimizer_config = dict(cumulative_iters=cumulative_iters)
lr_config = dict(warmup_iters=500 * cumulative_iters)                    # 2,000 micro-iter = step 500
runner = dict(max_iters=num_iters_per_epoch * num_epochs)                # 176,064 = step 44,016
checkpoint_config = dict(interval=num_iters_per_epoch)
evaluation = dict(interval=num_iters_per_epoch)
custom_hooks = [dict(type="EMASidecarHook", decay=0.999801, ramp_steps=2005, update_every=cumulative_iters,
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

run_name = "hipad_b2d_laq_v2_teacher_stage1_e3_nuim"
