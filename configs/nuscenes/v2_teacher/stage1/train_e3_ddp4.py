# 수정(2026-10-30): 사용자 결정 — 마지막 nuScenes 1단계 = E3(통합 디코더 FFN·LayerNorm 쿼리 종류별 분리) 100 에폭 처음부터,
#   GPU 4장 분산(DDP)으로 학습 시간 단축 + EMA 가중치 함께 저장. TF32 끔(사용자 결정, 기존과 같은 fp32 행렬곱).
#   ./train.py(본 1단계 100 에폭)를 상속하고 아래만 바꾼다.
#   (1) E3: onedecoder_head.task_specific_ffn_norm=True (diag_s1_20ep/e3_taskffn.py 와 같은 변경).
#   (2) 분산 등가: 종전 1 GPU × batch 16 × 누적 4(스트림 그룹 4) = GPU 4장 × batch 16 × 누적 1(스트림 그룹 1).
#       optimizer step 당 독립 시퀀스 64·표본 64·lr 4e-4 그대로. 샘플러 슬롯 배정도 같다(group_in_batch_sampler.py 주석:
#       그룹 k 슬롯 = 가상 GPU rank k). iter 수 = optimizer step 수가 되므로 iter 기준 값은 누적 수(4)로 나눈다:
#       에폭당 1,756 → 439, 총 175,600 → 43,900, warmup 2,000 → 500(= optimizer step 500, 종전과 같음), 체크포인트 매 에폭.
#   (3) EMA: mmdet3d_plugin/apis/ema_sidecar_hook.py — 매 step 갱신(decay 0.9998, 워밍업 2,000 step), 체크포인트마다 ema_iter_<N>.pth.
_base_ = ["./train.py"]
batch_size = 16
cumulative_iters = 1
num_gpus = 4
num_epochs = 100
num_iters_per_epoch = int(28130 // (batch_size * num_gpus))   # 439 (= 종전 1,756 / 4)
runner = dict(type="IterBasedRunner", max_iters=num_iters_per_epoch * num_epochs)   # 43,900
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
evaluation = dict(interval=num_iters_per_epoch)
optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=cumulative_iters,
                        grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(policy="CosineAnnealing", warmup="linear", warmup_iters=500, warmup_ratio=1.0 / 3, min_lr_ratio=1e-3)
# ---- ./train.py 와 같은 값·학습 파이프라인(상속 config 는 base 변수를 참조할 수 없어 그대로 다시 적는다) ----
#   유일한 차이: Collect meta_keys 에서 "stream_group" 을 뺀다(스트림 그룹 1개면 샘플러가 그 키를 붙이지 않는다 — 종전 G=1 config 와 같음).
det_class_names = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
map_roi_size = (30, 60)
map_num_pts = 20
strides = [4, 8, 16, 32]
num_depth_layers = 3
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# ---- 교사 캐시 ----
teacher_cache_root = "data/nuscenes/laq_cache"
teacher_folders = ["context", "detection", "tracking", "motion", "map", "plan", "action"]  # 고정 순서(K/V 배치 = 유형 번호 0..6)
teacher_expect_lens = None   # None = 폴더 _meta.json → 첫 파일 순으로 길이를 정하고 전 표본 동일 검사
teacher_meta_expect = {f: dict(prompt_mode="B", teacher_chain_verified=True, constrained_decoding="answer_grammar_v1")
                       for f in teacher_folders}   # 수정(2026-10-16): 설계 검토 반영 — B안·교사 연결·제약 생성 강제(키 = 추출기 기록 키)
qwen_cache_dir = teacher_cache_root  # SparseDetector 의 "언어 융합 ON" 표시. 파이프라인이 qwen_hidden 을 넣으므로 모델은 디스크를 읽지 않는다

teacher_loader = dict(type="LoadTeacherCacheNusc", cache_root=teacher_cache_root, folders=teacher_folders,
                      hidden_dim=1536, expect_lens=teacher_expect_lens, meta_expect=teacher_meta_expect)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(type="ResizeCropFlipImage"),
    dict(type="MultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=map_roi_size, simplify=False, normalize=False,
         sample_num=map_num_pts, permute=True),
    teacher_loader,  # 수정(2026-10-16): LoadQwenCacheNusc(qwen_l18) → 교사 캐시 7 폴더
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "qwen_type", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "ego_status", "ego_status_mask", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token"],
    ),
]

data = dict(samples_per_gpu=batch_size, num_stream_groups=1, train=dict(pipeline=train_pipeline))
model = dict(head=dict(onedecoder_head=dict(task_specific_ffn_norm=True, num_stream_groups=1)))
log_config = dict(interval=25)
custom_hooks = [dict(type="EMASidecarHook", decay=0.9998, ramp_steps=2000, update_every=cumulative_iters, max_keep=3,
                     priority="ABOVE_NORMAL")]
# 수정(2026-10-30): DDP 에서 백본 with_cp(재진입 체크포인트) 충돌 해결 — mmdet3d_plugin/apis/mmdet_train.py 참고(수식 동일)
backbone_cp_non_reentrant = True
# 수정(2026-10-30): DDP 검토 반영 — rank 별 주 프로세스 난수(tools/train.py). 샘플러 장면 순열은 rank 공통 seed 그대로
seed_rank_offset = True
run_name = "hipad_nusc_laq_v2_teacher_stage1_e3_ddp4"
