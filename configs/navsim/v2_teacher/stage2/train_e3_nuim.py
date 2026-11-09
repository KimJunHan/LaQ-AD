# 수정(2026-11-05): 사용자 결정 "NAVSIM 1단계 = nuImages + E3" — 그 1단계(../stage1/train_e3_nuim.py)에서 시작하는 2단계. 독립 검토 2회 전 기동 금지.
#   ./train.py(NAVSIM 2단계, 무수정)를 상속하고 nuScenes 최종 2단계(configs/nuscenes/v2_teacher/stage2/train_e3_nuim.py → train_e3.py)와
#   같은 항목 + E2 만 바꾼다. 배치(2 × 누적 24)·lr 2e-5·10 에폭·지도 정렬 잡음(gauss·selective·forward_axis=0)·EMA 0.999592/980 은 ./train.py 그대로.
#   (1) task_specific_ffn_norm=True — 1단계 E3 체크포인트의 layers.N.{det,map,ego,plan} 키를 그대로 읽기 위해 필수
#       (빠지면 load_from 이 엄격하지 않아 과제별 FFN·norm 키 288 개가 조용히 버려지고 공유 FFN 이 새로 초기화된다 — CPU 확인).
#   (2) 계획·자차 쿼리 전용 FFN·norm('.plan.'·'.ego.') lr ×10 — nuScenes train_e3.py (4)와 같은 이유(1단계에 계획·자차 손실이 없어
#       이 모듈들은 1단계에서 기울기 0 = 사실상 신규 모듈). 사용자 승인(2026-11-05). 걸리는 파라미터 = layers.*.plan/ego 144 개뿐(CPU 확인).
#   (3) FPN bf16(neck_dtype) — 1단계와 같은 수치 경로(nuScenes 2단계도 bf16).
#   (4) load_from = 새 1단계 EMA(max_iters 174,204 = 6,452 × 27 의 ema_iter_174204.pth 를 tools/ema_to_ckpt.py --config 로 바꾼 파일).
#       1단계 길이를 바꾸면 이 숫자도 같이 바꿀 것(실행 스크립트가 1단계 config max_iters 와 대조해 다르면 멈춘다).
#   (5) 수정(2026-11-05): 사용자 결정 "E2 자차 상태 직접 입력" — 교사 프롬프트에 이미 주는 자차 상태(속도·종방향 가속도·yaw rate)를
#       ego 쿼리에 직접 더한다(근거: docs/diag/stage2_nuim_l2_regression.md — nuImages nuScenes 모델이 교사 텍스트의 가속도를 못 읽어
#       a_x 상관 .956 → .717, 종방향 L2 악화). 학생은 지금까지 ego_status 를 손실 정답으로만 쓰고 입력으로는 쓰지 않았다(코드 확인).
#       값 = data/infos/navsim_20261019/navsim_ego_input_{navtrain,navtest}.pkl (tools/navsim/build_navsim_ego_input.py: 교사 프롬프트
#       자차 상태 줄을 그대로 읽은 [speed, accel_long, yaw_rate, available]; 결측 = 0,0,0,0 — 교사도 "unavailable").
#       모듈 = SparseOneDecoder.ego_input_encoder(Linear 4→256, ReLU, Linear 256→256, 마지막 층 0 초기화 → 시작 출력 = 종전 모델),
#       입력 정규화 scale [10 m/s, 1 m/s², 0.2 rad/s, 1](navtrain 표준편차 3.4 / 0.73 / 0.12 수준). 신규 모듈이라 lr ×10.
#       Collect 에 "ego_input" 추가(학습·평가). 옵션 기본값(None)이면 모든 기존 config 동작 동일.
#   (6) 수정(2026-11-06): 사용자 결정 — micro-batch 2 × 누적 24(스트림 그룹 24) → 8 × 누적 6(스트림 그룹 6), GPU3 1장.
#       nuScenes 최종 2단계(configs/nuscenes/stage2/train_v2.py: 1 GPU 8 × 6, G=6, workers 12)와 같은 형태. step 당 독립 시퀀스 48·표본 48·
#       lr 2e-5 그대로. iter 기준 값 = optimizer step × 6(종전 × 24): 에폭당 micro-iter 51,624 → 12,906(step 2,151 불변),
#       총 516,240 → 129,060(step 21,510 불변), warmup 12,000 → 3,000(= step 500 불변), 체크포인트 매 에폭(12,906),
#       EMA update_every 24 → 6(decay 0.999592·ramp 980 은 step 기준이라 불변 — 1단계 train_e3_nuim.py (4) 와 같은 규칙).
#       workers 8 → 12(nuScenes 8 × 6 과 같음; batch 4 배라 로더 부하 증가, RAM 2 TB).
#       DDP2(GPU1+3, 8 × 누적 3)는 쓰지 않는다: 2단계 DDP 는 어느 데이터셋에서도 돌려 본 적이 없고(nuScenes 2단계 = 1 GPU 8 × 6),
#       GPU1 은 nuScenes MCMC SFT 체인과 공유 — 1단계(DDP2)가 끝나면 GPU1 을 돌려준다. #이전 2 × 24(./train.py 값), workers 8
#   (7) 수정(2026-11-06): 사용자 결정 — 1단계를 처음부터 GPU1+3 DDP2(../stage1/train_e3_nuim_ddp2.py, max_iters 87,102 = step 43,551)로
#       학습 → load_from = 그 폴더의 iter_87102_ema.pth(tools/ema_to_ckpt.py --config ../stage1/train_e3_nuim_ddp2.py 변환본).
#       #이전 ./results/hipad_navsim_laq_v2_teacher_stage1_e3_nuim/iter_174204_ema.pth(1 GPU 1단계)
_base_ = ["./train.py"]

# ---- (6) 배치·누적·반복 수 ----
num_gpus, batch_size, cumulative_iters, num_epochs = 1, 8, 6, 10   # #이전 1, 2, 24, 10(./train.py)
num_stream_groups = cumulative_iters
navsim_train_scenes = 103288   # ./train.py 와 같음(에폭 = 장면 수 기준)
num_iters_per_epoch = int(navsim_train_scenes // (num_gpus * batch_size * cumulative_iters)) * cumulative_iters   # 12,906
# data 의 samples_per_gpu·workers_per_gpu·num_stream_groups 는 아래 data = dict(...) 안에 적는다(같은 파일 안 재대입은 병합되지 않음).
optimizer_config = dict(cumulative_iters=cumulative_iters)
lr_config = dict(warmup_iters=500 * cumulative_iters)                    # 3,000 micro-iter = step 500
runner = dict(max_iters=num_iters_per_epoch * num_epochs)                # 129,060 = step 21,510
checkpoint_config = dict(interval=num_iters_per_epoch)
custom_hooks = [dict(type="EMASidecarHook", decay=0.999592, ramp_steps=980, update_every=cumulative_iters,
                     max_keep=3, priority="ABOVE_NORMAL")]

model = dict(neck_dtype="bf16",
             head=dict(onedecoder_head=dict(task_specific_ffn_norm=True,
                                            num_stream_groups=num_stream_groups,   # 수정(2026-11-06): (6) #이전 24(./train.py)
                                            ego_input_embed=dict(in_dims=4, scale=[10.0, 1.0, 0.2, 1.0]))))
optimizer = dict(paramwise_cfg=dict(custom_keys={".plan.": dict(lr_mult=10.0), ".ego.": dict(lr_mult=10.0),
                                                 "ego_input_encoder": dict(lr_mult=10.0)}))
load_from = "./results/hipad_navsim_laq_v2_teacher_stage1_e3_nuim_ddp2/iter_87102_ema.pth"   # 수정(2026-11-06): (7)

# ---- (5) E2: 파이프라인(./train.py 와 같고 Collect keys 에 "ego_input" 만 추가; 목록은 상속 병합이 안 돼 다시 적는다) ----
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
test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="BEVObjectRangeFilter", point_cloud_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    teacher_loader,
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "qwen_type", "gt_bboxes_3d", "gt_labels_3d", "timestamp", "projection_mat",
               "image_wh", "ego_status", "gt_ego_fut_cmd", "ego_input"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token", "token"]),
]
ego_input_root = "data/infos/navsim_20261019/"
data = dict(
    samples_per_gpu=batch_size, workers_per_gpu=12, num_stream_groups=num_stream_groups,   # 수정(2026-11-06): (6) #이전 2, 8, 24(./train.py)
    train=dict(pipeline=train_pipeline, ego_input_file=ego_input_root + "navsim_ego_input_navtrain.pkl"),
    val=dict(pipeline=test_pipeline, ego_input_file=ego_input_root + "navsim_ego_input_navtest.pkl"),
    test=dict(pipeline=test_pipeline, ego_input_file=ego_input_root + "navsim_ego_input_navtest.pkl"),
)

evaluation = dict(interval=num_iters_per_epoch,   # 수정(2026-11-06): (6) #이전 51,624(./train.py, 2 × 24)
                  jsonfile_prefix="evaluation/hipad_navsim_laq_v2_teacher_stage2_e3_nuim")
run_name = "hipad_navsim_laq_v2_teacher_stage2_e3_nuim"
