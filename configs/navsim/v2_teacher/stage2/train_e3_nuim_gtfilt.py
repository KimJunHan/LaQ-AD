# 수정(2026-11-09): GT 필터 1단계(../stage1/train_e3_nuim_ddp2_gtfilt.py)에서 시작하는 2단계. 독립 검토 2회 전 기동 금지.
#   근거·정의: docs/navsim_gt_filter_20261109.md. ./train_e3_nuim.py(GPU3 1장 8 × 누적 6, 129,060 micro-iter, lr 2e-5, E2 자차 입력)를
#   상속하고 바꾸는 것은 셋뿐:
#   (1) train_pipeline: 1단계와 같은 GT 필터(det GT 정의 일관) — CircleObjectRangeFilter 바로 뒤에 ObjectHeightRangeFilter(z ∈ [-5, 10] m)
#       → CameraFOVObjectFilter(8 모서리·L0/F0/R0·5 % 여유·깊이 > 0.1 m). 모션 GT(gt_agent_fut_trajs/masks)도 같은 마스크로 함께 빠진다.
#       나머지 단계는 ./train_e3_nuim.py 의 train_pipeline 을 그대로 다시 적은 것. 평가(val/test) 파이프라인은 그대로 — 평가 GT 불변.
#   (2) load_from = GT 필터 1단계 DDP2 의 EMA 변환본(iter_87102_ema.pth; tools/ema_to_ckpt.py --config ../stage1/train_e3_nuim_ddp2_gtfilt.py).
#   (3) run_name·evaluation.jsonfile_prefix = ..._stage2_e3_nuim_gtfilt.
#   기준 config(./train.py, ./train_e3_nuim.py)는 고치지 않는다. 추론은 ./test_e3_nuim.py 를 그대로 쓴다(모델·test 파이프라인 같음).
_base_ = ["./train_e3_nuim.py"]

det_class_names = ["vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier", "czone_sign", "generic_object"]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
teacher_folders = ["context", "detection", "tracking", "motion", "map", "plan", "action"]
teacher_loader = dict(type="LoadTeacherCacheNusc", cache_root="data/navsim/laq_cache", folders=teacher_folders, hidden_dim=1536,
                      expect_lens=None,
                      meta_expect={f: dict(prompt_mode="B", teacher_chain_verified=True, constrained_decoding="answer_grammar_v1")
                                   for f in teacher_folders})
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromPcdNavsim", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(type="ResizeCropFlipImage"),
    dict(type="MultiScaleDepthMapGenerator", downsample=[4, 8, 16]),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
    dict(type="ObjectHeightRangeFilter", z_range=(-5.0, 10.0)),   # 수정(2026-11-09): (1) GT 높이 이상치 정리
    dict(type="CameraFOVObjectFilter", margin_frac=0.05, min_depth=0.1, crop_top=0.0,
         expect_cameras=("CAM_L0", "CAM_F0", "CAM_R0")),          # 수정(2026-11-09): (1) 카메라 시야 필터
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=(60, 30), simplify=False, normalize=False, sample_num=20, permute=True),
    teacher_loader,
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "qwen_type", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "ego_status", "ego_status_mask", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz",
               "gt_agent_fut_trajs", "gt_agent_fut_masks", "ego_input"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token", "stream_group"]),
]
data = dict(train=dict(pipeline=train_pipeline))

load_from = "./results/hipad_navsim_laq_v2_teacher_stage1_e3_nuim_ddp2_gtfilt/iter_87102_ema.pth"   # 수정(2026-11-09): (2) #이전 ..._stage1_e3_nuim_ddp2/iter_87102_ema.pth
evaluation = dict(jsonfile_prefix="evaluation/hipad_navsim_laq_v2_teacher_stage2_e3_nuim_gtfilt")
run_name = "hipad_navsim_laq_v2_teacher_stage2_e3_nuim_gtfilt"
