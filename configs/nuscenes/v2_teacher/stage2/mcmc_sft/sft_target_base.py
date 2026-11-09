# 수정(2026-11-05): 사용자 지시 "MCMC SFT 확인한대로 진행" — 군 (c)(d) 공통: 계획 정답을 생성 파일의 SFT 키로 바꾼다.
#   ./sft_base.py 와 다른 점은 셋뿐이다.
#   (1) data.train.plan_sft_target_file = 정답 파일(하위 config 에서 지정; 여기서는 None 이면 데이터셋이 키를 만들지 않아
#       디코더가 KeyError 로 멈춘다 — 대체 없음).
#   (2) train Collect keys 에 "gt_ego_plan_sft_2hz" 추가(파이프라인의 나머지는 ../train.py 와 같은 목록을 다시 적은 것;
#       같음은 tools/mcmc_sft/check_sft_configs.py 로 검사).
#   (3) 디코더 plan_sft_key="gt_ego_plan_sft_2hz" → get_gt_trajs 가 ("temp","2hz") 정답 궤적만 이 키로 읽는다(마스크·평가 키 불변).
_base_ = ["./sft_base.py"]

det_class_names = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
map_roi_size = (30, 60)
map_num_pts = 20
strides = [4, 8, 16, 32]
num_depth_layers = 3
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
teacher_cache_root = "data/nuscenes/laq_cache"
teacher_folders = ["context", "detection", "tracking", "motion", "map", "plan", "action"]
teacher_meta_expect = {f: dict(prompt_mode="B", teacher_chain_verified=True, constrained_decoding="answer_grammar_v1")
                       for f in teacher_folders}
teacher_loader = dict(type="LoadTeacherCacheNusc", cache_root=teacher_cache_root, folders=teacher_folders,
                      hidden_dim=1536, expect_lens=None, meta_expect=teacher_meta_expect)

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
    teacher_loader,
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "qwen_type", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "ego_status", "ego_status_mask", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz",
               "gt_agent_fut_trajs", "gt_agent_fut_masks",
               "gt_ego_plan_sft_2hz"],  # 수정(2026-11-05): 사용자 지시 "MCMC SFT 확인한대로 진행" — 학습 전용 SFT 정답
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token", "stream_group"],
    ),
]

data = dict(train=dict(pipeline=train_pipeline, plan_sft_target_file=None))
model = dict(head=dict(onedecoder_head=dict(plan_sft_key="gt_ego_plan_sft_2hz")))
