# 수정(2026-10-19): 사용자 결정 — 언어 토큰 자리에 LiDAR/레이더 토큰만 바꿔 넣는 비교(인코더 함께 학습)
#   카메라 + LiDAR 토큰 1단계 학습 config. run_name hipad_nusc_token_lidar_stage1
#   → results/hipad_nusc_token_lidar_stage1, evaluation/hipad_nusc_token_lidar_stage1.
#   ../../v2_teacher/stage1/train.py(LaQ v2 교사 캐시 1단계)를 그대로 상속하고 **언어 교차 어텐션에 들어가는 토큰열만** 바꾼다:
#     종전  교사 캐시 7 폴더 이어 붙인 2,369 토큰 × 1536 (파이프라인 LoadTeacherCacheNusc)
#     이번  LiDAR 키프레임 + 직전 10 sweep(시간차 채널) → 기둥 BEV 인코더(PillarBEVEncoder, 기둥 102.4/384 = 0.267 m,
#           384² → stride 2·2·2 → 48²×256) → BEV 칸마다 토큰 1 개 = 48×48 = 2,304 토큰 × 256 + 칸 위치 임베딩(사인 → MLP).
#           모델 안에서 계산(SensorTokenSparseDetector) → 인코더는 처음부터(무작위 초기화) 디코더 손실로 함께 학습된다.
#   바뀌는 것(전부):
#     (a) model.type SparseDetector → SensorTokenSparseDetector(+ sensor_token_encoder), qwen_cache_dir None
#     (b) qwen_token_attn.qwen_hidden_dim 1536 → 256(사영 첫 층 입력 차원만), 유형 임베딩 7 종 → 1 종(0 초기화, 이름 "lidar")
#     (c) 파이프라인: 교사 캐시 읽기 → LiDAR sweep 읽기·범위 필터·텐서화(같은 자리), Collect 의 qwen_* → lidar_points,
#         평가 파이프라인에 키프레임 LiDAR 읽기(LoadPointsFromFile) 추가(sweep 좌표 기준·키프레임 점 자체도 토큰 입력)
#     (d) 데이터셋 NuScenes3DFusionDataset + data/infos/nuscenes_fusion_infos_{train,val}.pkl
#         (기존 info 와 표본 수·순서·GT 동일 — sweeps 를 다시 만들고 radars 를 더한 것뿐. 평가 GT 불변)
#     (e) custom_imports(새 모듈 등록; 기존 __init__.py 를 고치지 않으려고)
#   그대로인 것: operation_order(qwen_attn 포함)·QwenTokenAttn 의 사영 MLP 구조(→512→256→LayerNorm)·게이트(계층×과제×채널, 0 초기화)·
#     손실·lr 4e-4·배치 16×4 누적·스트림 4 그룹·100 epoch·fp16 정적 배율 32·평가 설정.
_base_ = ["../../v2_teacher/stage1/train.py"]

# ---- base 와 같은 값(상속 config 에서는 base 변수를 직접 참조할 수 없어 다시 적는다; 값 불변) ----
det_class_names = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
map_roi_size = (30, 60)
map_num_pts = 20
strides = [4, 8, 16, 32]
num_depth_layers = 3
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

custom_imports = dict(imports=["tools.sensor_tokens_register"], allow_failed_imports=False)

# ---- 교사 캐시 변수 무효화(파이프라인·모델 어디에서도 쓰지 않음을 config 에 드러낸다) ----
teacher_cache_root = None
teacher_folders = None
teacher_expect_lens = None
teacher_meta_expect = None
teacher_loader = None
qwen_cache_dir = None

# ---- LiDAR 토큰 ----
token_pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]   # x,y ±51.2 m(B 융합 비교와 같은 범위), z 는 nuScenes 관례
token_grid = (48, 48)                                       # 2,304 토큰 ≈ 교사 2,369 (−2.7 %), 칸 2.133 m
lidar_sweeps = 10
sensor_points_key = "lidar_points"
sensor_token_encoder = dict(
    type="SensorBEVTokenizer", token_grid=token_grid, pos_dim=256,
    bev_encoder=dict(
        type="PillarBEVEncoder", in_channels=5, points_key=sensor_points_key,   # x,y,z,intensity,time_lag
        voxel_size=[102.4 / 384, 102.4 / 384], pc_range=token_pc_range, pfn_channels=64,
        stage_channels=[64, 128, 256], stage_layers=[3, 5, 5], stage_strides=[2, 2, 2],   # B 융합 LiDAR 와 같은 백본
        out_indices=[2], out_channels=256),                                                # 48² 한 수준만
)
points_pipeline = [
    dict(type="LoadPointsFromMultiSweepsFusion", sweeps_num=lidar_sweeps, in_key="points", out_key=sensor_points_key),
    dict(type="PointsRangeFilterFusion", point_cloud_range=token_pc_range, keys=[sensor_points_key]),
    dict(type="FusionPointsToTensor", keys=[sensor_points_key]),
]

model = dict(
    type="SensorTokenSparseDetector",
    qwen_cache_dir=None,
    sensor_token_encoder=sensor_token_encoder,
    head=dict(
        onedecoder_head=dict(
            qwen_token_attn=dict(
                qwen_hidden_dim=256,           # 사영 첫 층 입력 = BEV 채널(교사 1536)
                num_token_types=1,             # 센서 토큰 1 종, 0 초기화 (1, 256)
                token_type_names=["lidar"],
                route_segs=None,
            ),
        ),
    ),
)

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
    *points_pipeline,  # 수정(2026-10-19): 교사 캐시 읽기 자리 → LiDAR 점
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", sensor_points_key, "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "ego_status", "ego_status_mask", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token", "stream_group"],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),  # 수정(2026-10-19): 키프레임 LiDAR(토큰 입력)
    dict(type="ResizeCropFlipImage"),
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    *points_pipeline,  # 수정(2026-10-19): 교사 캐시 읽기 자리 → LiDAR 점
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", sensor_points_key, "gt_bboxes_3d", "gt_labels_3d", "timestamp", "projection_mat",
               "image_wh", "ego_status", "gt_ego_fut_cmd"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token", "token"],
    ),
]

anno_root = "data/infos/"
data = dict(
    train=dict(type="NuScenes3DFusionDataset", ann_file=anno_root + "nuscenes_fusion_infos_train.pkl", pipeline=train_pipeline),
    val=dict(type="NuScenes3DFusionDataset", ann_file=anno_root + "nuscenes_fusion_infos_val.pkl", pipeline=test_pipeline),
    test=dict(type="NuScenes3DFusionDataset", ann_file=anno_root + "nuscenes_fusion_infos_val.pkl", pipeline=test_pipeline),
)

run_name = "hipad_nusc_token_lidar_stage1"
