# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
#   NAVSIM LaQ v2 + 교사 캐시(7 폴더) 2단계 학습 config. ../../stage2/train.py(NAVSIM 2단계: +motion·plan, lr 2e-5 + 신규 모듈 ×10,
#   eff 48, 10 ep)를 상속하고 ../../v2_teacher/stage1/train.py 와 같은 데이터·지도·앵커·교사 캐시 치환을 한다. 추가로 nuScenes
#   v2_teacher 2단계(configs/nuscenes/v2_teacher/stage2/train.py)와 같은 계획 구조를 맞춘다:
#     - 잘린 확산 계획 + 지도 정렬 선택적 잡음(MapAlignedPlanTruncatedDiffusion, 지도 = 직전 프레임 모델 예측, PlanMapMemoryInstanceBank)
#       파라미터(score_thr 0.4, divider·boundary = 클래스 1·2, search_radius 3.5, margin 0.1)는 nuScenes 와 같다. 수정(2026-10-19): anisotropy_ratio 인자 삭제(모듈 API 변경 — 정규화 잡음 타원 크기 유지 + 차선 접선으로 회전). [주의] NAVSIM 계획 프레임은 x 가 전진(norm_scale[0]=34), 모듈은 norm_scale[1] 을 전진축으로 가정 — 전진축 옵션이 중앙에서 추가되기 전에는 2단계 학습 금지.
#         → 수정(2026-11-01): 사용자 지시 — 모듈에 forward_axis 옵션 추가(plan_diffusion_map.py (4), 기본 1 = 종전·nuScenes/B2D 비트 동일),
#           여기서 forward_axis=0(+x 전진). 확인: 라이다 = ego(lidar2ego 단위행렬), CAM_F0 광축 = 라이다 +x, navtest 4 s 끝 변위 x 평균 19.5 m·|y| 2.6 m,
#           navtest 300 프레임 경유점 2,091 개에서 잡음 긴 축과 차선 접선 각도 중앙값: forward_axis=1 89.1°(직각 — 종전 문제) → 0 에서 0.9°. 금지 해제.
#       ego_fut_ts 8(4 s). 정규화 = nuScenes 와 같은 규칙(학습 분할 GT 누적 궤적 min/max → (x−offset)/scale ∈ [−1, 1]):
#         navtrain(152,495 프레임, 유효 1,215,075 점): 전진 x [−1.99, 65.26], 횡 y(왼쪽+) [−23.73, 22.62]
#         → x: offset 32, scale 34 / y: offset −1, scale 24  (참고: DiffusionDrive NAVSIM norm_odo 는 x offset 27.25·scale 28.45,
#           y offset 3·scale 23 — 꼬리를 자른 값. 우리는 nuScenes 와 같은 규칙을 따른다.)
#       추론 infer_t 8·학습 t∈[0,50)·β 선형은 nuScenes 와 같다(DiffusionDrive 값).
#       수정(2026-11-02): 학습 t 는 이제 gauss(infer_t 8 중심, σ 8) 추출 + selective — 아래 plan_diffusion 주석 참고(위 "t∈[0,50)" 는 이전 서술).
#     - plan_diffusion lr ×10(신규 시점 임베딩) — nuScenes 와 같다.
#   수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건 (3) 계획 앵커 = 명령별 3×16(nuScenes v2 와 같음, 명령 임베딩 OFF).
#     (종전: 1×48 + 명령 임베딩 ON(num_command 4)). 지도 3 클래스(2)도 승인됨.
custom_imports = dict(imports=["tools.navsim_v2_teacher_register", "tools.plan_diffusion_map_register"], allow_failed_imports=False)
_base_ = ["../../stage2/train.py"]

det_class_names = ["vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier", "czone_sign", "generic_object"]
map_num_pts = 20
strides = [4, 8, 16, 32]
num_depth_layers = 3
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
num_gpus, batch_size, cumulative_iters, num_epochs = 1, 2, 24, 10
project_dir = "/home1/irteam/data-vol1/src/HiP-AD"
embed_dims = 256
ego_fut_ts = 8

navsim_train_frames = 152495
navsim_ego_anchor = [1.461, 0.0, 0.5385, 1.644033, 0.831604, 0.574927, 0.0, 1.0, 0.0, 0.0, 0.0]
navsim_ego_size = [5.176, 2.297, 1.777]
map_class_names = ["ped_crossing", "divider", "boundary"]
map_roi_size = (60, 30)
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
            map_instance_bank=dict(type="PlanMapMemoryInstanceBank", anchor=anchor_paths["map"]),
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
            plan_diffusion=dict(type="MapAlignedPlanTruncatedDiffusion", embed_dims=embed_dims, ego_fut_ts=ego_fut_ts,
                                num_train_timesteps=1000, trunc_steps=50, infer_t=8, beta_start=1e-4, beta_end=0.02,
                                norm_scale=(34.0, 24.0), norm_offset=(32.0, -1.0), infer_seed=0,
                                score_thr=0.4, lane_class_ids=(1, 2), boundary_class_ids=(2,), search_radius=3.5,
                                boundary_margin=0.1, clip_to_boundary=True,
                                # 수정(2026-11-02): 사용자 결정 "누씬하고 같게 맞추자" — nuScenes 최종 2단계(configs/nuscenes/v2_teacher/stage2/train.py)와 같은
                                #   학습 t 추출(infer_t 8 중심 이산 가우시안, σ 8)과 지도 확신도 선택(selective). #이전 uniform·selective False(모듈 기본)
                                t_sampling="gauss", t_std=8.0, selective=True,
                                forward_axis=0),  # 수정(2026-11-01): 사용자 지시 — NAVSIM 계획 좌표 +x 전진(위 [주의] 해결). #이전 (없음 = 1, +y 전진 가정)
            qwen_token_attn=dict(num_token_types=7, token_type_names=teacher_folders, route_segs=None),
        ),
    ),
)
optimizer = dict(paramwise_cfg=dict(custom_keys={"plan_diffusion": dict(lr_mult=10.0)}))

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromPcdNavsim", coord_type="LIDAR", load_dim=5, use_dim=5),
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
               "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz",
               "gt_agent_fut_trajs", "gt_agent_fut_masks"],
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
    val=dict(**navsim_ds_common, ann_file=anno_root + "navsim_navtest.pkl", teacher_map_files=teacher_map_test,
             pipeline=test_pipeline),
    test=dict(**navsim_ds_common, ann_file=anno_root + "navsim_navtest.pkl", teacher_map_files=teacher_map_test,
              pipeline=test_pipeline),
)

# 수정(2026-11-01): 사용자 결정 — 2단계 시작 = 1단계 EMA 가중치(B2D D안·nuScenes train_e3.py 와 같은 방식). 1단계 max_iters 5,161,600
#   (= 51,616 × 100, ../../v2_teacher/stage1/train.py) 의 ema_iter_5161600.pth 를 tools/ema_to_ckpt.py --config 로 바꾼 파일.
#   1단계 길이를 바꾸면 이 숫자도 같이 바꿀 것(실행 스크립트가 1단계 config 의 max_iters 와 대조해 다르면 멈춘다).
load_from = "./results/hipad_navsim_laq_v2_teacher_stage1/iter_5161600_ema.pth"  #이전 ./results/hipad_navsim_laq_v2_teacher_stage1/latest.pth(1단계 원 가중치)
checkpoint_config = dict(interval=num_iters_per_epoch)
lr_config = dict(warmup_iters=500 * cumulative_iters)
runner = dict(max_iters=num_iters_per_epoch * num_epochs)
evaluation = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval,
    jsonfile_prefix="evaluation/hipad_navsim_laq_v2_teacher_stage2",
    eval_config=dict(**navsim_ds_common, ann_file=anno_root + "navsim_navtest.pkl", teacher_map_files=teacher_map_test,
                     pipeline=eval_pipeline, test_mode=True),
)
# 수정(2026-11-01): EMA 가중치 함께 저장(사용자 규칙, nuScenes 최종 2단계 train_e3.py 와 같은 훅). 학습에는 영향 없음.
#   optimizer step(누적 24 micro-iter)마다 갱신. nuScenes 와 같은 규칙(창 11.4 %, ramp 4.6 %; 결과 보기 전 결정):
#   전체 step 21,510(= 2,151 × 10) → 창 ≈ 2,450(decay = 1 − 1/2,450 ≈ 0.999592), ramp ≈ 980(= 2,000 × 21,510 / 43,900).
custom_hooks = [dict(type="EMASidecarHook", decay=0.999592, ramp_steps=980, update_every=cumulative_iters,
                     max_keep=3, priority="ABOVE_NORMAL")]

run_name = "hipad_navsim_laq_v2_teacher_stage2"
