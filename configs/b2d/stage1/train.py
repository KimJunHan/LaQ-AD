# 수정(2026-10-16): v2 학습 config 반영 — nuScenes v2(configs/nuscenes/stage{1,2}/train_v2.py)의 모델·학습 방식 항목과 버그 수정을 옮김.
#   데이터셋 고유 항목(클래스·카메라·해상도·앵커·캐시·명령/계획 형식·epoch·batch·lr 의 HiP-AD 원본 B2D 기준값)은 그대로.
#   기록: backup/configs_pre_v2_sync_20261016/REPORT.md (원본 사본 같은 폴더)
# 수정(2026-10-15): 경로 이동 configs/b2d/hipad_b2d_laq_stage1_train.py → configs/b2d/stage1/train.py
#   run_name: 결과 디렉터리 이름(results/<run_name>, evaluation/<run_name>). tools/train.py·test.py 가 --work-dir 미지정 시 쓴다.
# 수정(2026-10-09): 벤치마크·stage·학습/테스트 단위 config 정리(사용자 지시).
#   원본: configs/hipad_b2d_vla_stage1.py(현재 archive/configs/) 를 그대로 복사한 학습용 config. B2D 1단계(GitHub hipad_b2d_stage1 + Run10 융합).
#   테스트용 짝 config 는 같은 폴더의 *_test.py (_base_ 로 이 파일을 상속, 평가 설정만 덮어씀).
# =====================================================================
# 수정(2026-10-09): LaQ-AD Bench2Drive stage1 (det+map, plan/ego 손실 0) — 폐쇄루프 DS 평가용 계보.
#   출처: archive/configs/hipad_b2d_full_stage1.py(B2D full, fp32) + Run10(hipad_nusc_vla_s2r9) 융합 모듈.
#   하이퍼파라미터는 업스트림 hipad_base_b2d_stage1.py 주석값(8 GPU x 8 = eff 64, 12 ep, lr 2e-4)에 맞춘다.
#   VRAM 상한 24GB(사용자 지시) → micro-batch 2 x 누적 32 = eff 64. 실측 전 보수값이므로 스모크로 확정할 것.
#   상세: docs/benchmark_prep.md
# =====================================================================
log_level = "INFO"
dist_params = dict(backend="nccl")

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

num_gpus = 1  #수정: GPU0 단독
batch_size = 2  #수정(2026-10-09): VRAM 24GB 상한. #이전(b2d_full) batch_size = 8  #수정: batch16 은 디코더 어텐션 activation 폭발→NaN 발산(2026-07-06 진단 확정). batch8 로 환원(검증됨: fresh batch8 은 iter 61k 무-NaN). eff batch 48
# num_iters_per_epoch = int(234769 // (num_gpus * batch_size))
cumulative_iters = 32  #수정(2026-10-09): eff 64(업스트림 8x8). #이전 cumulative_iters = 6  #수정: eff batch 48 유지(8*6=48). batch8 환원에 맞춰 3→6
num_iters_per_epoch = int(234769 // (num_gpus * batch_size * cumulative_iters)) * cumulative_iters  # 수정(2026-10-16): v2 학습 config 반영 — B2D 공식 train 분할 234,769 프레임(사용자 확인; 종전 242,403 은 다른 info 기준). epoch 당 micro-iter 를 누적 수의 배수로(체크포인트·재개 지점 = optimizer step 경계, 스트림 그룹 순환 보존). #이전 int(242403 // (num_gpus * batch_size))
num_stream_groups = cumulative_iters  # 수정(2026-10-16): v2 학습 config 반영 — (2) 독립 스트림 2×32=64 (= HiP-AD 원본 GPU 8 × 8 의 step 당 독립 시퀀스 64). 종전: 같은 2 시퀀스의 연속 32 프레임
num_epochs = 12  #수정: b2d upstream
#checkpoint_epoch_interval = 20
checkpoint_epoch_interval = 1

checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=1)  #수정(2026-10-09): 정책 기본값(1벌)
#이전: checkpoint_config = dict(interval=5000, max_keep_ckpts=3)  #수정: 공유 GPU OOM crash 보험(5000 iter≈6h마다). 주기는 하이퍼파라미터 무관, 비교 영향 없음
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)
load_from = None
resume_from = None
workflow = [("train", 1)]
fp16 = dict(loss_scale=32.0)  # 수정(2026-10-18): 사용자 결정 — HiP-AD 원본과 같은 fp16(정적 배율 32, 넘친 step 은 apis/fp16_strict_hook.py 가 멈춤). #이전 fp16 = None
find_unused_parameters = True  # 수정(2026-10-16): v2 학습 config 반영 — 누씬과 같은 모양(단일 프로세스 학습에서는 영향 없음)
cudnn_benchmark = True  #수정: 입력 고정(352×640)→conv autotune 가속. compute-bound이므로 유일한 안전 가속(하이퍼파라미터 무관)
input_shape = (640, 352)
num_cams = 6

# det & map
det_class_names = ["car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light", "pedestrian", "others"]
map_class_names = ["Broken", "Solid", "SolidSolid", "Center"]

num_det_classes = len(det_class_names)
num_map_classes = len(map_class_names)

map_roi_size = (30, 60)
map_num_pts = 20

# traj
fut_ts = 6
fut_mode = 6
ego_fut_ts = 6
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
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

# temporal
temporal = True
temporal_det = True
temporal_map = True
temporal_ego = True
temporal_plan = True

# tasks
task_config = dict(with_onedecoder=True)

task_select = ["det", "map", "plan", "ego"]
query_select = ["det", "map", "plan", "ego"]  # with query initial order

single_frame_layer = ["concat", "gnn", "inter_gnn", "norm", "split", "deformable", "concat", "ffn", "norm", "qwen_attn", "split", "refine"]  #수정(2026-10-09): Run10 과 동일 위치(ffn+norm 뒤, split 앞)
temporal_frame_layer = ["concat", "temp_gnn", "gnn", "inter_gnn", "norm", "split", "deformable", "concat", "ffn", "norm", "qwen_attn", "split", "refine"]  #수정(2026-10-09)

operation_order = single_frame_layer * num_single_frame_decoder + \
                  temporal_frame_layer * (num_decoder - num_single_frame_decoder)

# anchors
project_dir = "/home1/irteam/data-vol1/src/HiP-AD"

anchor_paths = {
    "det" : f"{project_dir}/data/kmeans/b2d_det_900.npy",
    "map" : f"{project_dir}/data/kmeans/b2d_map_100.npy",
    "motion": f"{project_dir}/data/kmeans/b2d_motion_{fut_mode}.npy",
}

plan_anchor_paths = f"{project_dir}/data/kmeans/b2d_plan_spat_6x8_5m.npy"
plan_anchor_refer = ("temp", "2hz")
plan_anchor_types = [("temp", "2hz")]


#수정(2026-10-09): B2D Qwen2-VL-2B 18층 캐시(tools/qwen_cache_b2d.py 로 생성). 지시문·화소예산(200x28x28/cam)·
#  계층은 nuScenes Run10 캐시(qwen_l18)와 동일하게 맞춘다. 경로는 생성 위치 확정 후 갱신(NAS 권장, ~0.9TB 추정).
qwen_cache_dir = "data/bench2drive_qwen_l18"

model = dict(
    type="SparseDetector",
    qwen_cache_dir=qwen_cache_dir,  #수정(2026-10-09): B2D VLA 캐시 배선(누락 표본은 즉시 실패)
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,  # 수정(2026-10-16): v2 학습 config 반영 — 검토 결과 유지. HiP-AD 원본 B2D(reference/hipad_base_b2d_stage1.py:104)가 True, v2 도 True
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
        evaluate_bench2dive=True,
        onedecoder_head=dict(
            type="SparseOneDecoder",
            task_select=task_select,
            query_select=query_select,
            operation_order=operation_order,
            num_stream_groups=num_stream_groups,  # 수정(2026-10-16): v2 학습 config 반영 — (2) 스트림 그룹별 시간 뱅크 상태
            #수정(2026-10-09): B2D VLA — Run10(hipad_nusc_vla_s2r9) 과 동일한 언어-쿼리 융합 모듈.
            #  사영·어텐션 1벌을 6계층이 공유, 게이트만 계층(6)×과제구간(3)×채널(256), 0 초기화.
            qwen_token_attn=dict(
                type="QwenTokenAttn",
                embed_dims=embed_dims,
                num_heads=8,
                qwen_hidden_dim=1536,
                gate_init=0.0,
                num_layers=6,
                seg_bounds=[900, 1000],  # [0,900)=det, [900,1000)=map, 나머지=plan/ego (query_select 순서와 일치해야 함)
                proj_hidden=512,
            ),
            num_single_frame_decoder=num_single_frame_decoder,
            plan_anchor_refer=plan_anchor_refer,
            with_command_embed=True,
            with_target_point_embed=True,
            with_supervise_ego_status=True,
            with_ego_instance_feature=True,
            with_incremental_plan_refine=True,
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
                zero_velocity_classes=["traffic_sign", "traffic_cone", "traffic_light"],
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
                query_list=[["det"], ["map"], ["plan", "ego"]],
                key_list=[["det"], ["map"], ["det", "map"]],
                decouple_list=[True, False, False],
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
            inter_graph_model=dict(
                type="InteractiveAttention",
                query_select=query_select,
                query_list=[["plan", "ego"]],
                key_list=[["det", "map"]],
                decouple_list=[False],
                attn=[
                    dict(
                        type="MultiheadFlashAttention",
                        embed_dims=embed_dims,
                        num_heads=num_groups,
                        batch_first=True,
                        dropout=drop_out,
                    ),
                ],
            ),
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
                              loss_yawness=dict(type="GaussianFocalLoss"),
                              cns_detach=False),  # 수정(2026-10-16): v2 학습 config 반영 — (3) centerness 타깃 detach 안 함(SparseDrive·HiP-AD 원본 코드와 같음)
            loss_map_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_map_reg=dict(type="SparseLineLoss",
                              loss_line=dict(type="LinesL1Loss", loss_weight=10.0, beta=0.01),
                              num_sample=map_num_pts,
                              roi_size=map_roi_size),
            loss_ego_status=dict(type="L1Loss", loss_weight=0.0),  # 수정복원: HiP-AD 커리큘럼 stage1=det+map+motion (plan/ego는 stage2)
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
                              anchor_refer=plan_anchor_refer, with_rescore=True,  # 수정(2026-11-06): 다음 줄에 rescore 인자를 더하려고 닫는 괄호를 옮김. #이전 with_rescore=True)
                              rescore_ego_offset=0.384),  # 수정(2026-11-06): B2D 궤적 원점=라이다(액터 x −0.39 m), 박스 중심 bbx_loc x −0.006 m → 라이다→중심 전방 0.384 m. yaw 는 y 전진이라 기본 π/2. #이전 (없음 = 0.5)
            motion_decoder=dict(type="SparseMotionDecoder"),
        ),
    ),
)

# ================== data ========================
dataset_type = "Bench2DriveDataset"
data_root = "data/bench2drive"
anno_root = "data/infos/"
file_client_args = dict(backend="disk")

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="B2DLoadPointsFromFile"),
    dict(type="ResizeCropFlipImage"),
    dict(type="B2DMultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizePloyLine",
         roi_size=map_roi_size,
         simplify=False,
         normalize=False,
         sample_num=map_num_pts,
         permute=True,
    ),
    dict(type="LoadQwenCacheNusc", cache_dir=qwen_cache_dir),  # 수정(2026-10-16): v2 학습 config 반영 — Qwen 캐시를 데이터 워커에서 읽음(누씬 10-10 속도 수정, 값 동일·누락 시 즉시 실패)
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "gt_ego_fut_cmd", "target_point", "ego_status", "ego_status_mask",
               "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               "gt_agent_fut_trajs", "gt_agent_fut_masks", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz"
               ],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token", "stream_group"],  #수정(2026-10-09): Qwen 캐시 조회 키. # 수정(2026-10-16): v2 학습 config 반영 — (2) stream_group
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    # dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="LoadQwenCacheNusc", cache_dir=qwen_cache_dir),  # 수정(2026-10-16): v2 학습 config 반영 — Qwen 캐시를 데이터 워커에서 읽음(누씬 10-10 속도 수정, 값 동일·누락 시 즉시 실패)
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "gt_bboxes_3d", "gt_labels_3d", "timestamp", "projection_mat", "image_wh", "ego_status",
               "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz", "gt_attr_labels",  "target_point"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token", "token"],  #수정(2026-10-09): Qwen 캐시 조회 키
    ),
]

eval_pipeline = [
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap",
         roi_size=map_roi_size,
         simplify=True,
         normalize=False,
    ),
    dict(type="Collect",
         keys=["vectors", "gt_bboxes_3d", "gt_labels_3d", "gt_agent_fut_trajs", "gt_agent_fut_masks",
               "gt_ego_fut_trajs", "gt_ego_fut_masks", "gt_ego_fut_cmd", "fut_boxes", "gt_attr_labels", "target_point"],
         meta_keys=["token", "timestamp"]
    ),
]

inference_only_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "gt_ego_fut_cmd", "projection_mat", "timestamp", "image_wh", "target_point"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token"],  #수정(2026-10-09): 폐쇄루프는 캐시 대신 에이전트가 qwen_hidden 을 온라인 주입(token 없음)
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    det_classes=det_class_names,
    map_classes=map_class_names,
    plan_anchor_types=plan_anchor_types,
    modality=input_modality,
)


data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [0, 0],
}

data = dict(
    num_stream_groups=num_stream_groups,  # 수정(2026-10-16): v2 학습 config 반영 — (2) 샘플러 그룹 수(mmdet_train 이 누적 수·디코더 값과 같은지 검사)
    samples_per_gpu=batch_size,
    workers_per_gpu=6,  #수정: RAM/swap OOM 방지 — data_infos 직렬화(COW-safe)와 함께 worker 6 으로 하향(사용자 지시 2026-07-15). map_infos COW 복제가 worker 수에 비례하므로 상한.
    persistent_workers=True,  #수정: epoch마다 worker 재spawn 방지 → map COW 재복사·data_time 스파이크 방지
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "b2d_infos_train.pkl",
        map_file=anno_root + "b2d_map_infos.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "b2d_infos_val.pkl",
        map_file=anno_root + "b2d_map_infos.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "b2d_infos_val.pkl",
        map_file=anno_root + "b2d_map_infos.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
    ),
)

# ================== training ========================
optimizer = dict(
    type="AdamW",
    lr=2e-4,  #수정: HiP-AD upstream lr 복원. # 수정(2026-10-16): v2 학습 config 반영 — B2D 는 HiP-AD 원본 2e-4 유지(사용자 지시: lr 은 벤치마크별 HiP-AD 원본; 누씬 v2 의 4e-4 는 SparseDrive 누씬 값)
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)
optimizer_config = dict(  #수정: grad accum(eff batch) + GradientCumulative
    type="GradientCumulativeOptimizerHook",
    cumulative_iters=cumulative_iters,
    grad_clip=dict(max_norm=25, norm_type=2),
)
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500 * cumulative_iters,  #수정(2026-10-09): warmup 을 optim-step 기준 500 으로(업스트림은 누적 없음). micro-iter 로 500 이면 ~16 step 만에 끝남
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)

# ================== eval ========================
eval_mode = dict(
    with_det=False,
    with_tracking=False,
    with_map=False,
    with_motion=False,
    with_planning=True,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)
evaluation = dict(
    interval=num_iters_per_epoch*checkpoint_epoch_interval,
    jsonfile_prefix="val/",
    eval_mode=eval_mode,
)

# 수정(2026-10-15): 결과 디렉터리 이름(파일명이 train.py 로 바뀌어 파일명 기반 기본값을 쓸 수 없다).
run_name = "hipad_b2d_laq_stage1"
