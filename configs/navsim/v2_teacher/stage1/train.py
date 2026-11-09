# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
#   NAVSIM LaQ v2 + 교사 캐시(7 폴더) 1단계 학습 config. nuScenes configs/nuscenes/v2_teacher/stage1/train.py 와 같은 구조를
#   NAVSIM 1단계(../../stage1/train.py: 전방 3캠 512×256, 7 클래스, 명령 입력 4, 계획 8 포즈, lr 4e-4, eff 64, 100 ep)에 얹는다.
#   ../../stage1/train.py 는 수정하지 않고 상속해서 바뀌는 것만 덮어쓴다:
#     1) 데이터: NavsimTeacherDataset(새 파일, custom_imports 로만 등록) — navtrain = 현재+과거 프레임 info
#        (data/infos/navsim_20261019/navsim_navtrain.pkl, 152,495 프레임 / 장면 103,288 / 시퀀스 14,147),
#        navtest = data/infos/navsim_20261019/navsim_navtest.pkl(→ navsim_20261016, 장면별 이력 4 프레임).
#     2) 지도 GT: 교사와 같은 nuPlan 지도 3 클래스(ped_crossing / divider / boundary, navsim_teacher_map.py 별도 pkl)를 라이다
#        프레임(+x 전방)으로 회전 → map_roi_size (60, 30). 종전 결정 대기 클래스(lane_centerline/road_boundary/walkway) 대신
#        nuScenes 와 같은 의미의 3 클래스 — 2단계 지도 정렬 잡음(divider=1, boundary=2)과 바로 맞는다. [사용자 승인 2026-10-19 — NAVSIM 권장안 8건 (2)]
#     3) 라이다(깊이 보조 손실): .pcd 직접 읽기(LoadPointsFromPcdNavsim; nuplan 판독기와 값 일치 확인) — .bin 변환 불필요.
#     4) 자차 앵커·크기: Pacifica 제원 + 지면 높이 실측(dataset_v2_teacher.NAVSIM_EGO_ANCHOR; 데이터셋이 config 값과 대조).
#     5) 앵커: navsim_v2_*.npy (tools/navsim/navsim_kmeans_v2.py, navtrain 만). 계획 앵커 48 × 8 포즈(4 s).
#     6) 언어 토큰: data/navsim/laq_cache/{context,…,action}/<token>.pt (교사 캐시, GPU 추출 대기) + 유형 임베딩 7.
#   학습 표본 = info 의 모든 프레임(장면 현재 프레임 + 과거 프레임; 모두 GT 있음, 미래 영상 없음).
custom_imports = dict(imports=["tools.navsim_v2_teacher_register"], allow_failed_imports=False)
_base_ = ["../../stage1/train.py"]

# ---- base 와 같은 값(상속 config 에서는 base 변수를 직접 참조할 수 없어 다시 적는다; 값 불변) ----
det_class_names = ["vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier", "czone_sign", "generic_object"]
map_num_pts = 20
strides = [4, 8, 16, 32]
num_depth_layers = 3
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
num_gpus, batch_size, cumulative_iters, num_epochs = 1, 2, 32, 100
project_dir = "/home1/irteam/data-vol1/src/HiP-AD"

# ---- 바뀌는 값 ----
navsim_train_frames = 152495   # build_navsim_infos_hist.py 출력(check_navsim_infos_hist.py PASS). 데이터셋이 전수 대조
navsim_ego_anchor = [1.461, 0.0, 0.5385, 1.644033, 0.831604, 0.574927, 0.0, 1.0, 0.0, 0.0, 0.0]   # = dataset_v2_teacher.NAVSIM_EGO_ANCHOR
navsim_ego_size = [5.176, 2.297, 1.777]
map_class_names = ["ped_crossing", "divider", "boundary"]
map_roi_size = (60, 30)        # 라이다 x(전방) 60 m · y 30 m
anno_root = "data/infos/navsim_20261019/"
teacher_map_root = "/home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/teacher/navsim_gt_v2sensor_20261019"
teacher_map_train = [f"{teacher_map_root}/teacher_map_navtrain.pkl"]
teacher_map_test = [f"{teacher_map_root}/teacher_map_navtest.pkl"]
# 수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건 — (7) 에폭 = 장면 수 기준(navtrain 장면 103,288). 샘플러(GroupInBatchSampler, 스트림 그룹)는 시퀀스의 모든
#   프레임(장면 프레임 + 과거 프레임)을 차례로 소비하므로 1 에폭 = 103,288 프레임 소비 ≈ 152,495 프레임의 0.68 바퀴.
#   과거 프레임에도 손실이 걸린다(모델에 표본별 손실 끄기 기능이 없음 — 공용 코드 변경 필요, 보고 사항).
navsim_train_scenes = 103288
num_iters_per_epoch = int(navsim_train_scenes // (num_gpus * batch_size * cumulative_iters)) * cumulative_iters
checkpoint_epoch_interval = 1

anchor_paths = {
    "det": f"{project_dir}/data/kmeans/navsim_v2_det_900.npy",
    "map": f"{project_dir}/data/kmeans/navsim_v2_map_100.npy",
    "motion": f"{project_dir}/data/kmeans/navsim_v2_motion_6.npy",
}
# 수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건 — (3) 명령별 계획 앵커 3×16(nuScenes v2 와 같음). 묶음 순서 = NAVSIM [left, straight, right],
#   unknown → straight(NavsimTeacherDataset command_groups=3). 명령 임베딩 OFF(앵커 묶음 선택으로 명령 사용 — nuScenes v2 와 같음).
plan_anchor_paths = f"{project_dir}/data/kmeans/navsim_v2_plan_temp_3x16.npy"  # tools/navsim/navsim_kmeans_plan_cmd.py
ego_fut_cmd, ego_fut_mode = 3, 16

# ---- 교사 캐시 ----
teacher_cache_root = "data/navsim/laq_cache"
teacher_folders = ["context", "detection", "tracking", "motion", "map", "plan", "action"]
teacher_expect_lens = None
teacher_meta_expect = {f: dict(prompt_mode="B", teacher_chain_verified=True, constrained_decoding="answer_grammar_v1")
                       for f in teacher_folders}
qwen_cache_dir = teacher_cache_root
teacher_loader = dict(type="LoadTeacherCacheNusc", cache_root=teacher_cache_root, folders=teacher_folders,
                      hidden_dim=1536, expect_lens=teacher_expect_lens, meta_expect=teacher_meta_expect)

model = dict(
    qwen_cache_dir=qwen_cache_dir,
    head=dict(
        onedecoder_head=dict(
            motion_anchor=anchor_paths["motion"],
            det_instance_bank=dict(anchor=anchor_paths["det"]),
            map_instance_bank=dict(anchor=anchor_paths["map"]),
            ego_instance_bank=dict(custom_anchor=navsim_ego_anchor),
            with_command_embed=False,  # 수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건 (3)
            plan_instance_bank=dict(anchor_paths=plan_anchor_paths, ego_fut_cmd=ego_fut_cmd, ego_fut_mode=ego_fut_mode,
                                    num_temp_mode=ego_fut_mode),
            plan_refine_layer=dict(ego_fut_cmd=ego_fut_cmd, ego_fut_mode=ego_fut_mode),
            plan_sampler=dict(ego_fut_cmd=ego_fut_cmd, ego_fut_mode=ego_fut_mode),
            align_sampler=dict(ego_fut_cmd=ego_fut_cmd, ego_fut_mode=ego_fut_mode),
            map_sampler=dict(roi_size=map_roi_size),
            loss_map_reg=dict(roi_size=map_roi_size),
            plan_decoder=dict(ego_size=navsim_ego_size, ego_fut_cmd=ego_fut_cmd, ego_fut_mode=ego_fut_mode),
            qwen_token_attn=dict(num_token_types=7, token_type_names=teacher_folders, route_segs=None),
        ),
    ),
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromPcdNavsim", coord_type="LIDAR", load_dim=5, use_dim=5),   # .pcd 직접(→ (n,5))
    dict(type="ResizeCropFlipImage"),
    dict(type="MultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
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
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    teacher_loader,
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "qwen_type", "gt_bboxes_3d", "gt_labels_3d", "timestamp", "projection_mat",
               "image_wh", "ego_status", "gt_ego_fut_cmd"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token", "token"]),
]
eval_pipeline = [
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=map_roi_size, simplify=True, normalize=False),
    dict(type="Collect", keys=["vectors", "gt_bboxes_3d", "gt_labels_3d"], meta_keys=["token", "timestamp"]),
]

navsim_ds_common = dict(type="NavsimTeacherDataset", map_classes=map_class_names, map_roi=map_roi_size, command_groups=3,
           ego_anchor=navsim_ego_anchor, ego_size=navsim_ego_size)
data = dict(
    train=dict(**navsim_ds_common, ann_file=anno_root + "navsim_navtrain.pkl", num_frames=navsim_train_frames,
               teacher_map_files=teacher_map_train, pipeline=train_pipeline,
               sequences_split_num=1),  # 수정(2026-10-19): 시퀀스(로그 내 연속 이력 프레임, 길이 1~65·중앙 9)는 이미 짧다. 2 로 나누면 길이 1 시퀀스에서 샘플러 검사 실패(실측) → 나누지 않음
    val=dict(**navsim_ds_common, ann_file=anno_root + "navsim_navtest.pkl", teacher_map_files=teacher_map_test, pipeline=test_pipeline),
    test=dict(**navsim_ds_common, ann_file=anno_root + "navsim_navtest.pkl", teacher_map_files=teacher_map_test, pipeline=test_pipeline),
)

checkpoint_config = dict(interval=num_iters_per_epoch)
lr_config = dict(warmup_iters=500 * cumulative_iters)
runner = dict(max_iters=num_iters_per_epoch * num_epochs)
evaluation = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval,
    jsonfile_prefix="evaluation/hipad_navsim_laq_v2_teacher_stage1",
    eval_config=dict(**navsim_ds_common, ann_file=anno_root + "navsim_navtest.pkl", teacher_map_files=teacher_map_test,
                     pipeline=eval_pipeline, test_mode=True),
)
# 수정(2026-11-01): EMA 가중치 함께 저장(사용자 규칙, nuScenes 최종 1단계 train_e3_ddp4.py 와 같은 훅). 학습에는 영향 없음.
#   optimizer step(누적 32 micro-iter)마다 갱신. nuScenes 와 같은 규칙(창 = 전체 step 의 11.4 %, ramp 4.6 %; 결과 보기 전 결정):
#   전체 step 161,300(= 1,613 × 100) → 창 ≈ 18,371(decay = 1 − 1/18,371 ≈ 0.999946), ramp ≈ 7,349(= 2,000 × 161,300 / 43,900).
custom_hooks = [dict(type="EMASidecarHook", decay=0.999946, ramp_steps=7349, update_every=cumulative_iters,
                     max_keep=3, priority="ABOVE_NORMAL")]

run_name = "hipad_navsim_laq_v2_teacher_stage1"
