log_level = "INFO"
dist_params = dict(backend="nccl")

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

num_gpus = 1  #수정: nuScenes full 멀티태스크, GPU0 단독
batch_size = 4  #수정: per-GPU micro-batch (motion 추가로 stage1보다 무거움)
#수정(2026-08-03): effective batch 48 유지 — 상류 대조 결과 stage2 는 48 이 맞다.
#   HiP-AD 공개 stage2(hipad_b2d_stage2.py): 8gpu × batch 6 = total 48
#   SparseDrive nuScenes stage2: total_batch_size = 48  → 두 레퍼런스 일치.
#   (stage1 만 64 로 정정했다. stage1/stage2 배치가 다른 것이 원본 설계다.)
cumulative_iters = 12  # effective batch = 1*4*12 = 48
# num_iters_per_epoch = int(234769 // (num_gpus * batch_size))
num_iters_per_epoch = int(28130 // (num_gpus * batch_size))  #수정: full nuScenes (micro-iter 기준)
num_epochs = 10  #수정: HiP-AD/SparseDrive nuScenes stage2 표준(=10ep, load_from stage1), 비교 타당성 위해 일치
#checkpoint_epoch_interval = 20
checkpoint_epoch_interval = 1

checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)  #수정: ckpt 보존 3
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)
load_from = "./work_dirs/hipad_nusc_stage1/latest.pth"  #수정: stage1 가중치 로드
resume_from = None
workflow = [("train", 1)]
find_unused_parameters = True  #수정: 미사용 plan/ego 모듈(빌드만)용
fp16 = None  #수정: fp32 고정(NaN 원천차단, 프로젝트 규칙)
#수정(2026-08-03): (640,352) → (704,256). stage1 과 동일 근거
#   (HiP-AD 논문 Appendix B "resolution is changed to 704 × 256" + SparseDrive nuScenes config).
#   stage1/stage2 해상도가 달라지면 load_from 가중치가 어긋나므로 반드시 함께 바꾼다.
input_shape = (704, 256)
num_cams = 6

# det & map
det_class_names = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]  #수정: nuScenes 10 det class
map_class_names = ["ped_crossing", "divider", "boundary"]  #수정: nuScenes 3 map class

num_det_classes = len(det_class_names)
num_map_classes = len(map_class_names)

map_roi_size = (30, 60)
map_num_pts = 20

# traj
fut_ts = 12  #수정: nuScenes motion GT 와 일치(12 steps)
fut_mode = 6
ego_fut_ts = 12  #수정: nuScenes motion GT 와 일치(12 steps)
ego_fut_cmd = 1
ego_fut_mode = 48

# model
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = True
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
decouple_attn = True
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]  #수정: nuScenes surround range

# temporal
temporal = True
temporal_det = True
temporal_map = True
temporal_ego = True
temporal_plan = True

# tasks
task_config = dict(with_onedecoder=True)

task_select = ["det", "map", "motion"]  #수정: nuScenes stage2 = stage1 + motion
query_select = ["det", "map"]

single_frame_layer = ["concat", "gnn", "inter_gnn", "norm", "split", "deformable", "concat", "ffn", "norm", "split", "refine"]
temporal_frame_layer = ["concat", "temp_gnn", "gnn", "inter_gnn", "norm", "split", "deformable", "concat", "ffn", "norm", "split", "refine"]

operation_order = single_frame_layer * num_single_frame_decoder + \
                  temporal_frame_layer * (num_decoder - num_single_frame_decoder)

# anchors
project_dir = "/workspace/src/HiP-AD"

anchor_paths = {
    "det" : f"{project_dir}/data/kmeans/nusc_det_900.npy",
    "map" : f"{project_dir}/data/kmeans/nusc_map_100.npy",
    "motion": f"{project_dir}/data/kmeans/nusc_motion_6.npy",  #수정: nuScenes 10클래스 12ts anchor
}

plan_anchor_paths = f"{project_dir}/data/kmeans/b2d_plan_spat_6x8_5m.npy"
plan_anchor_refer = ("temp", "2hz")
plan_anchor_types = [("temp", "2hz")]


model = dict(
    type="SparseDetector",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=False,  #수정: DDP+find_unused 충돌 회피(메모리 여유)
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained="ckpts/resnet50-19c8e357.pth",
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        norm_cfg=dict(type="BN", requires_grad=True),
        no_norm_on_lateral=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    depth_branch=dict(  # for auxiliary supervision only
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),
    head=dict(
        type="SparseHead",
        task_config=task_config,
        evaluate_bench2dive=False,  #수정: nuScenes det eval — B2D planning metric(plan_temp_2hz) 비활성. DetectionEval(mAP/NDS) 사용
        onedecoder_head=dict(
            type="SparseOneDecoder",
            task_select=task_select,
            query_select=query_select,
            operation_order=operation_order,
            num_single_frame_decoder=num_single_frame_decoder,
            plan_anchor_refer=plan_anchor_refer,
            with_command_embed=False,  #수정
            with_target_point_embed=False,  #수정
            with_supervise_ego_status=False,  #수정: ego status loss(weight0) shape충돌 회피
            with_ego_instance_feature=False,  #수정
            with_incremental_plan_refine=False,  #수정
            motion_anchor=anchor_paths["motion"],
            cls_threshold_to_reg=0.05,
            # instance_bank
            det_instance_bank=dict(
                type="InstanceBank",
                num_anchor=900,
                embed_dims=embed_dims,
                anchor=anchor_paths["det"],
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=600 if temporal_det else -1,
                confidence_decay=0.6,
                feat_grad=False,
                class_names=det_class_names,
                zero_velocity_classes=["barrier", "traffic_cone"],  #수정: nuScenes 정적 클래스
            ),
            map_instance_bank=dict(
                type="InstanceBank",
                num_anchor=100,
                embed_dims=embed_dims,
                anchor=anchor_paths["map"],
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
                num_temp_instances=0 if temporal_map else -1,
                confidence_decay=0.6,
                feat_grad=True,
            ),
            ego_instance_bank=dict(
                type="EgoInstanceBank",
                anchor_type="b2d",
                embed_dims=embed_dims,
                num_temp_instances=1 if temporal_ego else -1,
                feature_map_scale=(input_shape[1] / strides[-1], input_shape[0] / strides[-1]),
            ),
            plan_instance_bank=dict(
                type="PlanningInstanceBank",
                embed_dims=embed_dims,
                ego_fut_ts=ego_fut_ts,
                ego_fut_cmd=ego_fut_cmd,
                ego_fut_mode=ego_fut_mode,
                num_temp_mode=ego_fut_mode if temporal_plan else -1,
                feature_map_scale=(input_shape[1] / strides[-1], input_shape[0] / strides[-1]),
                anchor_paths=plan_anchor_paths,
                anchor_types=plan_anchor_types,
            ),
            # anchor encoder
            det_anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=1,
                out_loops=4 if decouple_attn else 2,
            ),
            map_anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=map_num_pts,
                return_points_embed=True,
            ),
            plan_anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=ego_fut_ts,
                return_points_embed=True,
            ),
            # operation
            custom_op=dict(type="CustomOperation"),
            temp_graph_model=dict(
                type="TemporalSeparateAttention",
                query_select=query_select,
                query_list=[["det"], ["map"]],  #수정: det+map
                key_list=[["det"], ["map"]],  #수정
                decouple_list=[True, False],  #수정
                attn=[
                    dict(
                        type="MultiheadFlashAttention",
                        embed_dims=embed_dims * 2,
                        num_heads=num_groups,
                        batch_first=True,
                        dropout=drop_out,
                    ),
                    dict(
                        type="MultiheadFlashAttention",
                        embed_dims=embed_dims,
                        num_heads=num_groups,
                        batch_first=True,
                        dropout=drop_out,
                    ),
                    dict(
                        type="MultiheadFlashAttention",
                        embed_dims=embed_dims,
                        num_heads=num_groups,
                        batch_first=True,
                        dropout=drop_out,
                    ),
                ],
            ) if temporal else None,
            graph_model=dict(
                type="SeparateAttention",
                query_select=query_select,
                separate_list=[["det"], ["map"]],
                decouple_list=[True, False],
                attn=[
                    dict(
                        type="MultiheadFlashAttention",
                        embed_dims=embed_dims * 2,
                        num_heads=num_groups,
                        batch_first=True,
                        dropout=drop_out,
                    ),
                    dict(
                        type="MultiheadFlashAttention",
                        embed_dims=embed_dims,
                        num_heads=num_groups,
                        batch_first=True,
                        dropout=drop_out,
                    ),
                ],
            ),
            inter_graph_model=None,  #수정: plan/ego 없으면 불필요
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            # deformable
            det_deformable=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[
                        [0, 0, 0],
                        [0.45, 0, 0],
                        [-0.45, 0, 0],
                        [0, 0.45, 0],
                        [0, -0.45, 0],
                        [0, 0, 0.45],
                        [0, 0, -0.45],
                    ],
                ),
            ),
            map_deformable=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims,
                    num_sample=map_num_pts,
                    num_learnable_pts=3,
                    fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023,  # ground height in lidar frame
                ),
            ),
            ego_deformable=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=12,
                    fix_scale=[
                        [0.45, 0, 0],
                    ],
                ),
            ),
            plan_deformable=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims,
                    num_sample=ego_fut_ts,
                    num_learnable_pts=3,
                    fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023,  # ground height in lidar frame
                ),
            ),
            # refine
            det_refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims,
                num_cls=num_det_classes,
                refine_yaw=True,
                with_quality_estimation=True,
            ),
            map_refine_layer=dict(
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims,
                num_sample=map_num_pts,
                num_cls=num_map_classes,
            ),
            ego_refine_layer=dict(
                type="EgoStatusRefinementModule",
                embed_dims=embed_dims,
            ),
            plan_refine_layer=dict(
                type="SparsePlanAlignRefinementModule",
                embed_dims=embed_dims,
                ego_fut_ts=ego_fut_ts,
                ego_fut_cmd=ego_fut_cmd,
                ego_fut_mode=ego_fut_mode,
                anchor_types=plan_anchor_types,
            ),
            motion_refine_layer=dict(
                type="SparseMotionRefinementModule",
                embed_dims=embed_dims,
                fut_ts=fut_ts,
                fut_mode=fut_mode,
            ),
            # sampler
            det_sampler=dict(
                type="SparseBox3DTarget",
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,
                max_dn_gt=32,
                add_neg_dn=True,
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
                cls_wise_reg_weights={
                    det_class_names.index("traffic_cone"): [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                },
            ),
            map_sampler=dict(
                type="SparsePoint3DTarget",
                assigner=dict(
                    type="HungarianLinesAssigner",
                    cost=dict(
                        type="MapQueriesCost",
                        cls_cost=dict(type="FocalLossCost", weight=1.0),
                        reg_cost=dict(type="LinesL1Cost", weight=10.0, beta=0.01, permute=True),
                    ),
                ),
                num_cls=num_map_classes,
                num_sample=map_num_pts,
                roi_size=map_roi_size,
            ),
            plan_sampler=dict(
                type="SparsePlanTarget",
                ego_fut_ts=ego_fut_ts,
                ego_fut_cmd=ego_fut_cmd,
                ego_fut_mode=ego_fut_mode
            ),
            align_sampler=dict(
                type="AlignPlanTarget",
                ego_fut_ts=ego_fut_ts,
                ego_fut_cmd=ego_fut_cmd,
                ego_fut_mode=ego_fut_mode
            ),
            motion_sampler=dict(
                type="SparseMotionTarget"
            ),
            # loss
            loss_det_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_det_reg=dict(type="SparseBox3DLoss",
                              loss_box=dict(type="L1Loss", loss_weight=0.25),
                              loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                              loss_yawness=dict(type="GaussianFocalLoss")),
            loss_map_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_map_reg=dict(type="SparseLineLoss",
                              loss_line=dict(type="LinesL1Loss", loss_weight=10.0, beta=0.01),
                              num_sample=map_num_pts,
                              roi_size=map_roi_size),
            loss_ego_status=dict(type="L1Loss", loss_weight=0.0),
            loss_plan_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.0),
            loss_plan_reg=dict(type="L1Loss", loss_weight=0.0),
            loss_motion_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.2),
            loss_motion_reg=dict(type="L1Loss", loss_weight=0.2),
            # weights
            det_reg_weights=[2.0] * 3 + [1.0] * 7,
            map_reg_weights=[1.0] * 40,
            # decoder
            det_decoder=dict(type="SparseBox3DDecoder"),
            map_decoder=dict(type="SparsePoint3DDecoder"),
            plan_decoder=dict(type="SparsePlanDecoder", ego_fut_ts=ego_fut_ts, ego_fut_cmd=ego_fut_cmd,
                              ego_fut_mode=ego_fut_mode, ego_vehicle="b2d", anchor_types=plan_anchor_types,
                              anchor_refer=plan_anchor_refer, with_rescore=True),
            motion_decoder=dict(type="SparseMotionDecoder"),
        ),
    ),
)

# ================== data ========================
dataset_type = "NuScenesMiniDataset"
data_root = "data/nuscenes"
anno_root = "data/infos/"
file_client_args = dict(backend="disk")

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),  #수정: depth aux 용 lidar
    dict(type="ResizeCropFlipImage"),
    dict(type="MultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),  #수정: depth GT 생성
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=map_roi_size, simplify=False, normalize=False,
         sample_num=map_num_pts, permute=True),  #수정: permute=True → gt_map_labels/gt_map_pts
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "ego_status", "ego_status_mask", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts", "gt_agent_fut_trajs", "gt_agent_fut_masks"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token"],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "gt_bboxes_3d", "gt_labels_3d", "timestamp", "projection_mat", "image_wh", "ego_status"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token", "token"],
    ),
]

eval_pipeline = [
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=map_roi_size, simplify=True, normalize=False),
    dict(type="Collect",
         keys=["vectors", "gt_bboxes_3d", "gt_labels_3d"],
         meta_keys=["token", "timestamp"]),
]

input_modality = dict(use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=False)

data_basic_config = dict(
    type=dataset_type, data_root=data_root,
    det_classes=det_class_names, map_classes=map_class_names,
    plan_anchor_types=plan_anchor_types, modality=input_modality,
)

data_aug_conf = {
    "resize_lim": (0.40, 0.47), "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0), "rot_lim": (-5.4, 5.4),
    "H": 900, "W": 1600, "rand_flip": True, "rot3d_range": [0, 0],
}

data = dict(
    samples_per_gpu=batch_size, workers_per_gpu=batch_size,
    train=dict(**data_basic_config, ann_file=anno_root + "nuscenes_infos_train.pkl",
               pipeline=train_pipeline, test_mode=False, data_aug_conf=data_aug_conf,
               with_seq_flag=True, sequences_split_num=2, keep_consistent_seq_aug=True),
    val=dict(**data_basic_config, ann_file=anno_root + "nuscenes_infos_val.pkl",
             pipeline=test_pipeline, data_aug_conf=data_aug_conf, test_mode=True),
    test=dict(**data_basic_config, ann_file=anno_root + "nuscenes_infos_val.pkl",
              pipeline=test_pipeline, data_aug_conf=data_aug_conf, test_mode=True),
)

# ================== training ========================
optimizer = dict(
    type="AdamW",
    lr=2e-4, #수정: HiP-AD upstream 값으로 복원(2.5e-5 는 fp16 NaN 회피 핵). effective batch 48 에 맞는 lr
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)
optimizer_config = dict(  #수정: grad accumulation 으로 effective batch 48 재현
    type="GradientCumulativeOptimizerHook",
    cumulative_iters=cumulative_iters,
    grad_clip=dict(max_norm=25, norm_type=2),
)
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500 * cumulative_iters,  #수정: warmup 을 optim-step 기준(=500)으로 유지
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)

# ================== eval ========================
#수정(2026-08-03): HiP-AD Table 4 전 항목 산출. stage2 는 task_select 에 motion 이 있으므로
#  with_motion=True — Table 4 의 4개 열(det/map/track/motion)이 모두 여기서 나온다.
#  planning 은 이 config 의 task_select 에 없으므로 False.
eval_mode = dict(
    with_det=True,
    with_tracking=True,
    with_map=True,
    with_motion=True,
    with_planning=False,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)
#수정(2026-08-03): map 평가기용 GT dataset cfg (stage1 과 동일).
eval_config = dict(
    **data_basic_config,
    ann_file=anno_root + "nuscenes_infos_val.pkl",
    pipeline=eval_pipeline,
    test_mode=True,
)
evaluation = dict(
    interval=num_iters_per_epoch*checkpoint_epoch_interval,
    jsonfile_prefix="val/",
    eval_mode=eval_mode,
    eval_config=eval_config,
)