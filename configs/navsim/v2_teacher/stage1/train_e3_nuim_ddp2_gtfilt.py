# 수정(2026-11-09): NAVSIM 1단계 발산(원 런 iter 5,725 정지) 대응 — GT 필터 2 개를 넣은 새 1단계 config(처음부터 학습).
#   근거: docs/navsim_s1_divergence_20261109.md §7(검증 V4: z(-5,10) + 카메라 시야 필터, lr 4e-4 그대로 안정), 정식화 docs/navsim_gt_filter_20261109.md.
#   ./train_e3_nuim_ddp2.py(GPU1+3 DDP2, 2 × 16 × 누적 2, 87,102 micro-iter, lr 4e-4, EMA 0.999798/1,984)를 상속하고 바꾸는 것은 셋뿐:
#   (1) train_pipeline: CircleObjectRangeFilter 바로 뒤(InstanceNameFilter 앞)에 ObjectHeightRangeFilter(z ∈ [-5, 10] m)
#       → CameraFOVObjectFilter(박스 8 모서리 중 하나라도 L0/F0/R0 중 한 영상 안(가로·세로 5 % 여유) + 깊이 > 0.1 m) 순서로 넣는다.
#       위치·순서는 V4(진단 모듈이 CircleObjectRangeFilter.__call__ 뒤에 z → 시야 순서로 덧씌움)와 같다. 투영은 증강 뒤 lidar2img·img_shape.
#       나머지 단계는 ../../stage1/train.py → ./train.py 의 train_pipeline 을 그대로 다시 적은 것(리스트는 상속 병합이 안 된다).
#   (2) run_name·evaluation.jsonfile_prefix 를 새 이름(..._ddp2_gtfilt)으로. 결과 폴더 = results/<run_name>(체인이 --work-dir 로 넘김) —
#       기존 발산 런 폴더(results/hipad_navsim_laq_v2_teacher_stage1_e3_nuim_ddp2, latest.pth = iter_3226)와 섞이지 않는다.
#   (3) 평가(test/val) 파이프라인은 그대로 — 평가 GT 정의 불변.
#   기준 config(./train.py, ./train_e3_nuim.py, ./train_e3_nuim_ddp2.py)는 고치지 않는다.
_base_ = ["./train_e3_nuim_ddp2.py"]

# ---- ./train.py 와 같은 값(상속 config 에서 base 변수를 직접 참조할 수 없어 다시 적는다; 값 불변) ----
det_class_names = ["vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier", "czone_sign", "generic_object"]
map_num_pts = 20
strides = [4, 8, 16, 32]
num_depth_layers = 3
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
map_roi_size = (60, 30)
teacher_folders = ["context", "detection", "tracking", "motion", "map", "plan", "action"]
teacher_loader = dict(type="LoadTeacherCacheNusc", cache_root="data/navsim/laq_cache", folders=teacher_folders,
                      hidden_dim=1536, expect_lens=None,
                      meta_expect={f: dict(prompt_mode="B", teacher_chain_verified=True, constrained_decoding="answer_grammar_v1")
                                   for f in teacher_folders})
# (1) GT 필터 정의 = docs/navsim_gt_filter_20261109.md. expect_cameras = models/loader/navsim/dataset.py NAVSIM_CAMS(대조만; 투영은 lidar2img 전부).
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromPcdNavsim", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(type="ResizeCropFlipImage"),
    dict(type="MultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
    dict(type="ObjectHeightRangeFilter", z_range=(-5.0, 10.0)),   # 수정(2026-11-09): (1) GT 높이 이상치 정리
    dict(type="CameraFOVObjectFilter", margin_frac=0.05, min_depth=0.1, crop_top=0.0,
         expect_cameras=("CAM_L0", "CAM_F0", "CAM_R0")),                            # 수정(2026-11-09): (1) 카메라 시야 필터
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=map_roi_size, simplify=False, normalize=False, sample_num=map_num_pts, permute=True),
    teacher_loader,
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "qwen_type", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "ego_status", "ego_status_mask", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token", "stream_group"]),
]
data = dict(train=dict(pipeline=train_pipeline))

# ---- (2) 이름 ----
evaluation = dict(jsonfile_prefix="evaluation/hipad_navsim_laq_v2_teacher_stage1_e3_nuim_ddp2_gtfilt")
run_name = "hipad_navsim_laq_v2_teacher_stage1_e3_nuim_ddp2_gtfilt"

# 수정(2026-11-09): 학습 속도 수정(docs/train_speedfix_20261109.md, 검토 2회 통과) — 같은 (lr, wd) 파라미터 그룹 병합(1,588→2) + fused AdamW.
#   lr·wd·custom_keys(img_backbone·img_neck ×0.5) 는 base 와 dict 병합으로 그대로. 런 도중 바꾸지 말 것(그룹 수 바뀌면 optimizer 상태 재개 불가).
optimizer = dict(constructor="GroupedOptimizerConstructor", fused=True)
