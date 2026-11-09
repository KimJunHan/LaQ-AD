# 수정(2026-10-19): 사용자 결정 — 언어 토큰 자리에 LiDAR/레이더 토큰만 바꿔 넣는 비교(인코더 함께 학습)
#   카메라 + 레이더 토큰 1단계 학습 config. run_name hipad_nusc_token_radar_stage1
#   → results/hipad_nusc_token_radar_stage1, evaluation/hipad_nusc_token_radar_stage1.
#   ../../token_lidar/stage1/train.py(= LaQ v2 교사 1단계에서 토큰열만 LiDAR 로 바꾼 것)와 코드·레시피가 같고 센서만 다르다:
#     레이더 5 대 × 6 sweep(키프레임 포함, 13 Hz → 약 0.4 s) 누적, 점 7 열 x,y,z,rcs,vx_comp,vy_comp(LiDAR 좌표로 회전),time_lag,
#     devkit 기본 필터 → 기둥 BEV(RadarPillarBEVEncoder, 기둥 102.4/96 = 1.067 m, 96² → stride 1·2 → 48²×256, B 융합 레이더와 같은 얕은 백본)
#     → BEV 칸마다 토큰 1 개 = 2,304 토큰 × 256 + 칸 위치 임베딩. 유형 임베딩 1 종(이름 "radar").
#   LiDAR sweep 은 읽지 않는다. 키프레임 LiDAR 1 장은 교사 런과 같은 깊이 보조 감독(DenseDepthNet) GT 용으로만 학습 때 읽는다.
_base_ = ["../../token_lidar/stage1/train.py"]

# ---- base 와 같은 값(재기재; 값 불변) ----
det_class_names = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
map_roi_size = (30, 60)
map_num_pts = 20
strides = [4, 8, 16, 32]
num_depth_layers = 3
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
token_pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
token_grid = (48, 48)

# ---- 레이더 토큰 ----
lidar_sweeps = None
radar_sweeps = 6
sensor_points_key = "radar_points"
sensor_token_encoder = dict(
    _delete_=True,
    type="SensorBEVTokenizer", token_grid=token_grid, pos_dim=256,
    bev_encoder=dict(
        type="RadarPillarBEVEncoder", in_channels=7, points_key=sensor_points_key,
        voxel_size=[102.4 / 96, 102.4 / 96], pc_range=token_pc_range, pfn_channels=64,
        stage_channels=[64, 128], stage_layers=[2, 2], stage_strides=[1, 2],   # B 융합 레이더와 같은 백본
        out_indices=[1], out_channels=256),                                     # 48² 한 수준만
)
points_pipeline = [
    dict(type="LoadRadarPointsMultiSweeps", sweeps_num=radar_sweeps, out_key=sensor_points_key),
    dict(type="PointsRangeFilterFusion", point_cloud_range=token_pc_range, keys=[sensor_points_key]),
    dict(type="FusionPointsToTensor", keys=[sensor_points_key]),
]

model = dict(
    sensor_token_encoder=sensor_token_encoder,
    head=dict(onedecoder_head=dict(qwen_token_attn=dict(token_type_names=["radar"]))),
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),  # 깊이 보조 GT 용(교사 런과 같음)
    dict(type="ResizeCropFlipImage"),
    dict(type="MultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=map_roi_size, simplify=False, normalize=False,
         sample_num=map_num_pts, permute=True),
    *points_pipeline,  # 수정(2026-10-19): 교사 캐시 읽기 자리 → 레이더 점
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
    dict(type="ResizeCropFlipImage"),
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    *points_pipeline,  # 수정(2026-10-19): 교사 캐시 읽기 자리 → 레이더 점
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", sensor_points_key, "gt_bboxes_3d", "gt_labels_3d", "timestamp", "projection_mat",
               "image_wh", "ego_status", "gt_ego_fut_cmd"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token", "token"],
    ),
]

data = dict(train=dict(pipeline=train_pipeline), val=dict(pipeline=test_pipeline), test=dict(pipeline=test_pipeline))

run_name = "hipad_nusc_token_radar_stage1"
