# 수정(2026-11-06): B2D 2단계 후보 — ../stage1/train_e3_nuim.py(nuImages + E3) 에서 시작하는 2단계. 사용자 승인 전 기동 금지.
#   ./train.py(B2D v2 교사 2단계 = HiP-AD 논문 4.2 절 값 lr 2e-4·wd 0.01·eff 32, D안 6 에폭, 잡음 = nuScenes 최종, 무수정)를 상속하고
#   아래 (1)~(4)만 바꾼다.
#   (1) task_specific_ffn_norm=True — 1단계 E3 체크포인트의 layers.N.{det,map,ego,plan} 키를 읽기 위해 필수(빠지면 load_from 이
#       엄격하지 않아 과제별 키가 조용히 버려진다 — NAVSIM train_e3_nuim.py (1) 과 같은 이유).
#   (2) FPN bf16(neck_dtype) — 1단계와 같은 수치 경로(nuScenes·NAVSIM 2단계와 같음). FPN lr ×0.5 는 nuScenes·NAVSIM 2단계에도 없어 넣지 않는다.
#   (3) 계획·자차 '.plan.'·'.ego.' lr ×10 은 넣지 않는다: nuScenes·NAVSIM 은 기본 lr 2e-5 라 ×10 으로 실효 2e-4 를 주던 것이고,
#       B2D 2단계는 논문 값으로 모든 모듈이 이미 2e-4 다(×10 이면 2e-3).
#   (4) micro-batch 2 × 누적 16(스트림 그룹 16) → 8 × 누적 4(스트림 그룹 4): step 당 표본 32·독립 시퀀스 32·lr 그대로
#       (nuScenes 최종 2단계 8 × 6 과 같은 형태). 에폭당 micro-iter 117,376 → 29,344(step 7,336 불변), 총 704,256 → 176,064(step 44,016 불변),
#       warmup 8,000 → 2,000(= step 500), EMA update_every 16 → 4(decay 0.999801·ramp 2,005 불변). 저장 = 매 에폭, 학습 중 평가 없음.
#   E2(자차 상태 직접 입력)는 넣지 않았다: B2D 데이터셋에 ego_input_file 경로·B2D 교사 프롬프트 자차 줄 추출 도구가 없어 소스 수정이 필요
#     (NAVSIM 은 models/loader/navsim/dataset_v2_teacher.py + tools/navsim/build_navsim_ego_input.py). 별도 승인 항목.  #이전 서술 — 아래 (5)로 대체
#   (5) 수정(2026-11-06): 사용자 결정 "B2D E2" — NAVSIM 2단계 (5)와 같은 E2: 교사 프롬프트 자차 상태 줄의 숫자(속도·종방향 가속도·yaw rate)를
#       ego 쿼리에 직접 더한다. 값 = data/infos/b2d_20261106/b2d_ego_input_{train,val}.pkl (tools/b2d/build_b2d_ego_input.py: 교사 GT jsonl 의
#       prompt.user 자차 줄을 그대로 읽음 = 교사 캐시 프롬프트와 같은 숫자; 10 Hz 프레임은 LoadTeacherCacheB2D 와 같은 키프레임(나이 ≤ 0.4 s)의 값;
#       현재 프레임 시뮬레이터 값만 — 도구가 infos 현재 프레임으로 다시 계산해 확인). 결측 = 0,0,0,0(B2D 실제 결측 0).
#       모듈 = SparseOneDecoder.ego_input_encoder(Linear 4→256, ReLU, Linear 256→256, 마지막 층 0 초기화 → 시작 출력 = E2 없는 모델).
#       입력 scale [10 m/s, 5 m/s², 0.4 rad/s, 1]: 정규화 후 표준편차를 NAVSIM(0.34 / 0.73 / 0.58)과 맞춘 값
#       (B2D train 234,769 프레임 표준편차 3.81 / 3.58 / 0.222 → 0.38 / 0.72 / 0.55; NAVSIM 은 [10, 1, 0.2, 1]). B2D IMU 가속도는 꼬리가 길다(|a| p99 15.6, 최대 42.5).
#       lr: 배수 없음 — NAVSIM 의 ×10 은 기본 2e-5 를 실효 2e-4 로 맞추던 것이고 B2D 2단계 기본 lr 이 이미 2e-4(위 (3)과 같은 이유).
#       Collect(학습·평가)에 "ego_input" 추가. 1단계에는 넣지 않는다(NAVSIM 과 같음; 1단계는 계획 손실 0).
#       폐쇄루프 에이전트는 교사 갱신 시점(2 Hz)의 같은 숫자를 metas["ego_input"] 로 넘겨야 한다(미구현 — 별도 작업).
#   load_from = 1단계 후보 EMA(max_iters 176,064 의 ema_iter_176064.pth 를 tools/ema_to_ckpt.py --config ../stage1/train_e3_nuim.py 로 바꾼 파일).
#   평가용 test config(neck_dtype·task_specific_ffn_norm 포함)는 아직 없다 — 승인 시 함께 만든다.
_base_ = ["./train.py"]

# ---- (4) 배치·누적·반복 수 ----
num_gpus, batch_size, cumulative_iters, num_epochs = 1, 8, 4, 6
num_stream_groups = cumulative_iters
b2d_train_frames = 234769
num_iters_per_epoch = int(b2d_train_frames // (num_gpus * batch_size * cumulative_iters)) * cumulative_iters   # 29,344
checkpoint_epoch_interval = 20   # ./train.py 와 같음(> 6 에폭 → 학습 중 평가 없음)

# data 는 아래 (5) 블록에 한 번에 적는다(같은 파일 안 재대입은 병합되지 않음).
optimizer_config = dict(cumulative_iters=cumulative_iters)
lr_config = dict(warmup_iters=500 * cumulative_iters)                    # 2,000 micro-iter = step 500
runner = dict(max_iters=num_iters_per_epoch * num_epochs)                # 176,064 = step 44,016
checkpoint_config = dict(interval=num_iters_per_epoch)
evaluation = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval)
custom_hooks = [dict(type="EMASidecarHook", decay=0.999801, ramp_steps=2005, update_every=cumulative_iters,
                     max_keep=3, priority="ABOVE_NORMAL")]

# ---- (1) E3 · (2) FPN bf16 · (5) E2 ----
model = dict(neck_dtype="bf16",
             head=dict(onedecoder_head=dict(task_specific_ffn_norm=True, num_stream_groups=num_stream_groups,
                                            ego_input_embed=dict(in_dims=4, scale=[10.0, 5.0, 0.4, 1.0]))))  # 수정(2026-11-06): 사용자 결정 "B2D E2"

load_from = "./results/hipad_b2d_laq_v2_teacher_stage1_e3_nuim/iter_176064_ema.pth"
run_name = "hipad_b2d_laq_v2_teacher_stage2_e3_nuim"

# ---- (5) E2: 파이프라인(./train.py 와 같고 Collect keys 끝에 "ego_input" 만 추가; 목록은 상속 병합이 안 돼 다시 적는다) ----
#   수정(2026-11-06): 사용자 결정 "B2D E2"
det_class_names = ["car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light", "pedestrian", "others"]
map_roi_size = (30, 60)
map_num_pts = 20
strides = [4, 8, 16, 32]
num_depth_layers = 3
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
teacher_folders = ["context", "detection", "tracking", "motion", "map", "plan", "action"]
teacher_loader = dict(type="LoadTeacherCacheB2D", cache_root="data/bench2drive/laq_cache", folders=teacher_folders,
                      hidden_dim=1536, expect_lens=None,
                      meta_expect={f: dict(prompt_mode="B", teacher_chain_verified=True, constrained_decoding="answer_grammar_v1")
                                   for f in teacher_folders},
                      keyframe_stride=5)
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
    dict(type="VectorizePloyLine", roi_size=map_roi_size, simplify=False, normalize=False,
         sample_num=map_num_pts, permute=True),
    teacher_loader,
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "qwen_type", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "gt_ego_fut_cmd", "target_point", "ego_status", "ego_status_mask",
               "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               "gt_agent_fut_trajs", "gt_agent_fut_masks",
               "gt_ego_spat_trajs_2m", "gt_ego_spat_masks_2m", "gt_ego_spat_trajs_5m", "gt_ego_spat_masks_5m",
               "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz", "gt_ego_fut_trajs_5hz", "gt_ego_fut_masks_5hz", "ego_input"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token", "stream_group"],
    ),
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
               "image_wh", "ego_status", "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz", "gt_attr_labels",
               "target_point", "ego_input"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token", "token"],
    ),
]
ego_input_root = "data/infos/b2d_20261106/"
data = dict(
    samples_per_gpu=batch_size, num_stream_groups=num_stream_groups,
    train=dict(pipeline=train_pipeline, ego_input_file=ego_input_root + "b2d_ego_input_train.pkl"),
    val=dict(pipeline=test_pipeline, ego_input_file=ego_input_root + "b2d_ego_input_val.pkl"),
    test=dict(pipeline=test_pipeline, ego_input_file=ego_input_root + "b2d_ego_input_val.pkl"),
)
