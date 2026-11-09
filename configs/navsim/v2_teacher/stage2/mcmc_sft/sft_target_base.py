# 수정(2026-11-06): 사용자 결정 "NAVSIM MCMC SFT" — 정답 파일 군(MCMC·randc) 공통(# 수정(2026-11-06): GT-SFT 군 제거 #이전 GT-SFT·MCMC·randc): 계획 정답을 생성 파일의 SFT 키로 바꾼다.
#   ./sft_base.py 와 다른 점은 셋뿐(tools/mcmc_sft/check_sft_configs_navsim.py 가 단언):
#   (1) data.train.plan_sft_target_file = 정답 파일(하위 config 에서 지정; None 이면 데이터셋이 키를 만들지 않아 디코더가 KeyError 로 멈춤 — 대체 없음).
#   (2) train Collect keys 끝에 "gt_ego_plan_sft_2hz"(나머지 목록은 ../train_e3_nuim.py 를 다시 적은 것 — 리스트는 상속 병합이 안 된다).
#       수정(2026-11-09): 상속 기준이 ../train_e3_nuim_gtfilt.py 로 바뀌어 GT 필터 2 단계(z·시야)도 같은 위치에 다시 적는다.
#   (3) 디코더 plan_sft_key="gt_ego_plan_sft_2hz" → get_gt_trajs 가 ("temp","2hz") 정답 궤적만 이 키로 읽는다(마스크·평가 키 불변).
_base_ = ["./sft_base.py"]

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
    dict(type="ObjectHeightRangeFilter", z_range=(-5.0, 10.0)),   # 수정(2026-11-09): GT 필터 2단계(../train_e3_nuim_gtfilt.py)와 같게
    dict(type="CameraFOVObjectFilter", margin_frac=0.05, min_depth=0.1, crop_top=0.0,
         expect_cameras=("CAM_L0", "CAM_F0", "CAM_R0")),          # 수정(2026-11-09): 같음
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=(60, 30), simplify=False, normalize=False, sample_num=20, permute=True),
    teacher_loader,
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "qwen_type", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "ego_status", "ego_status_mask", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz",
               "gt_agent_fut_trajs", "gt_agent_fut_masks", "ego_input",
               "gt_ego_plan_sft_2hz"],  # 수정(2026-11-06): 사용자 결정 "NAVSIM MCMC SFT" — 학습 전용 SFT 정답
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token", "stream_group"]),
]
data = dict(train=dict(pipeline=train_pipeline, plan_sft_target_file=None))
model = dict(head=dict(onedecoder_head=dict(plan_sft_key="gt_ego_plan_sft_2hz")))
