# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   LaQ v2 + 교사 캐시 7 폴더 · 토큰 유형 임베딩 — B2D 2단계 학습 config (configs/nuscenes/v2_teacher/stage2/train.py 의 B2D 짝).
#   ../../stage2/train.py(B2D LaQ 2단계, HiP-AD 원본 B2D 값: 18 epoch, lr 2e-4, eff batch 48, +motion·plan/ego 손실)를 상속하고
#     (a) 언어 토큰 출처: data/bench2drive_qwen_l18 → data/bench2drive/laq_cache 7 폴더(2 Hz 키프레임, LoadTeacherCacheB2D)
#     (b) 0 초기화 토큰 유형 임베딩(7×256) ON, route_segs OFF
#     (c) nuScenes v2 와 같은 구조: 잘린 확산 계획 + 지도 정렬 선택적 잡음(MapAlignedPlanTruncatedDiffusion + PlanMapMemoryInstanceBank).
#         B2D base 2단계에는 확산이 없으므로 여기서 새로 넣는다. 값은 nuScenes 에서 베끼지 않고 같은 규칙으로 B2D 데이터에서 다시 구했다
#         (tools/b2d_plan_diffusion_stats.py → results/b2d_prep/b2d_plan_diffusion_stats.json, 아래 주석).
#   그 밖(백본·앵커·클래스·640×352·fp16·스트림 그룹·평가·lr 정책)은 base 와 같다.
#   lr: base 와 같은 2e-4(HiP-AD 원본 B2D, 모듈별 ×10 없음 — 사용자 지시 "lr 은 벤치마크별 HiP-AD 원본"). base 의 paramwise 규칙은
#     img_backbone 0.5 하나뿐이라 plan_diffusion(t_mlp)도 기본 2e-4 를 받는다 → paramwise 키를 따로 넣지 않는다
#     (nuScenes 2단계의 plan_diffusion ×10 은 그쪽 기본 lr 이 2e-5 라 실효 2e-4 로 맞추던 것; B2D 는 이미 2e-4).
#   배치: base 의 micro-batch 2 × 누적 24(= eff 48, HiP-AD 원본 8 GPU × 6). micro-batch 는 VRAM 실측 뒤 확정 —
#     바꿀 때 eff 48 이 유지되도록 cumulative_iters·num_stream_groups 를 함께 바꿀 것(base 의 num_iters_per_epoch 식도 같이).
#   수정(2026-11-01): 사용자 결정 — B2D 2단계 하이퍼파라미터는 GitHub config 대신 HiP-AD 논문 4.2 절 값을 따른다
#     ("8 GPU, total batch size 32, AdamW + Cosine, lr 2e-4, weight decay 0.01"). "논문 값 덮어쓰기" 블록(변수 = 교사 캐시 앞, optimizer·lr·runner = 파일 끝):
#     eff batch 48 → 32 (micro-batch 2 × 누적 24 → 2 × 16, 스트림 그룹 24 → 16), weight_decay 0.001 → 0.01. lr·cosine·18 에폭 불변.
#     (18 에폭은 같은 날 아래 D안으로 6 에폭이 됐다.)
#     base(../../stage2/train.py)는 LaQ v1 B2D 2단계(../../stage2/test.py)도 쓰므로 고치지 않고 여기서만 덮어쓴다.
#     (위 "eff 48" 서술은 이전 값 — 기록으로 둔다.)
#   수정(2026-11-01): 사용자 결정(D안) — 1단계 12 에폭 + 2단계 6 에폭, 2단계 시작 = 1단계 EMA 가중치.
#     논문 "주행 스타일 헤드 없이 12 에폭 → 켜고 6 에폭" 중 우리 1단계에는 주행 스타일 헤드가 없으므로 1단계(12 에폭)는 그대로 두고
#     2단계를 6 에폭으로 한다(종전 18). 에폭당 micro-iter(117,376)·warmup(step 500)·lr 2e-4·wd 0.01·eff 32 는 그대로,
#     전체 길이에 묶인 값만 다시 구했다: max_iters 704,256(step 44,016), cosine 길이 = runner.max_iters(mmcv 가 자동으로 씀),
#     EMA decay 0.999801·ramp 2,005(파일 끝, 같은 규칙). 저장 간격은 에폭 단위(117,376)라 그대로, 학습 중 평가는 여전히 없음(20 에폭 > 6).
#     초기 가중치: 아래 load_from 을 1단계 EMA 변환본(iter_<1단계 max_iters>_ema.pth, tools/ema_to_ckpt.py --config)으로 바꿨다
#     (nuScenes train_e3.py 와 같은 방식). 실행 스크립트 scripts/b2d/run_v2_teacher_20261101.sh 의 기본 S2_INIT=ema 도 같은 파일을 쓴다.
#     E3 의 task_specific_ffn_norm·계획/자차 lr ×10 은 B2D 에 넣지 않는다(미승인).
#   B안: 계획 결과는 "w/ ego status" 범주.
#   run_name hipad_b2d_laq_v2_teacher_stage2, load_from = results/hipad_b2d_laq_v2_teacher_stage1/latest.pth(이전 값 — 2026-11-01 D안으로 1단계 EMA 로 바뀜, 파일 끝)
#   (확산 t_mlp 는 1단계에 없는 새 파라미터 → load_from 에서 빠진 키로 0 초기화 마지막 층 그대로 시작; 지도 뱅크는 파라미터 추가 없음).
_base_ = ["../../stage2/train.py"]

custom_imports = dict(imports=["tools.b2d_teacher_register", "tools.plan_diffusion_map_register"], allow_failed_imports=False)

# ---- base 와 같은 값(재기재, 값 불변) ----
det_class_names = ["car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light", "pedestrian", "others"]
map_roi_size = (30, 60)
map_num_pts = 20
strides = [4, 8, 16, 32]
num_depth_layers = 3
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
embed_dims = 256
ego_fut_ts = 6

# ---- 수정(2026-11-01): 논문 값 덮어쓰기(HiP-AD 논문 4.2 절, 사용자 결정) ----
#   상속 config 는 base 의 파생값(num_iters_per_epoch·warmup_iters·max_iters·interval)을 다시 계산하지 않으므로 같은 식으로 모두 다시 적는다.
#   1 GPU × micro-batch 2 × 누적 16 = optimizer step 당 표본 32·독립 스트림 32(= 논문 8 GPU × 4). 재개·저장 지점이 step 경계가 되도록
#   에폭당 micro-iter 를 누적 수의 배수로 내림(base 와 같은 식): 234,769 // 32 = 7,336 step × 16 = 117,376 micro-iter/에폭.
#   warmup 은 base 와 같은 "optimizer step 500" → 500 × 16 = 8,000 micro-iter.
num_gpus = 1
batch_size = 2
cumulative_iters = 16  #이전(base) 24 → eff 48
num_epochs = 6         # 수정(2026-11-01): 사용자 결정(D안) — 1단계 12 + 2단계 6 에폭(논문 12 + 6 의 6 에폭 부분). #이전 18(base 와 같음)
num_iters_per_epoch = int(234769 // (num_gpus * batch_size * cumulative_iters)) * cumulative_iters  # 117,376  #이전 117,384
num_stream_groups = cumulative_iters  # 16 (아래 model 의 onedecoder_head.num_stream_groups 와 data.num_stream_groups 에 쓰임)  #이전 24
checkpoint_epoch_interval = 20  # base 와 같음(> 18 에폭 → 학습 중 평가 없음; run_train 은 --no-validate)
#   optimizer·lr·runner·저장 간격 덮어쓰기는 파일 끝(같은 날짜 주석).

# ---- 교사 캐시 ----
teacher_cache_root = "data/bench2drive/laq_cache"
teacher_folders = ["context", "detection", "tracking", "motion", "map", "plan", "action"]
teacher_expect_lens = None
teacher_meta_expect = {f: dict(prompt_mode="B", teacher_chain_verified=True, constrained_decoding="answer_grammar_v1")
                       for f in teacher_folders}
qwen_cache_dir = teacher_cache_root

teacher_loader = dict(type="LoadTeacherCacheB2D", cache_root=teacher_cache_root, folders=teacher_folders,
                      hidden_dim=1536, expect_lens=teacher_expect_lens, meta_expect=teacher_meta_expect, keyframe_stride=5)

# ---- 잘린 확산 + 지도 정렬 선택적 잡음 (B2D 값) ----
#   근거: tools/b2d_plan_diffusion_stats.py → results/b2d_prep/b2d_plan_diffusion_stats.json (train 234,769 프레임 전부, CPU, 2026-10-19).
#   일정·시점(β 선형 1e-4→0.02/1000, 학습 t ∈ [0,50), 추론 t=8, 시드 0)은 nuScenes 와 같다(DiffusionDrive 값 — 데이터 무관).
#   norm (DiffusionDrive norm_odo 규칙 = nuScenes 와 같은 규칙: train GT 누적 궤적 [min,max] → [-1,1], 정수 반올림):
#     B2D 2단계가 쓰는 계획 목표 4 종(temp 2hz·5hz, spat 2m·5m) 유효점 5,128,589 개의 합집합 범위
#     x [-30.48, 30.14], y [-14.72, 47.12] → offset (0, 16), scale (31, 32).
#     (x 폭은 spat 5m 의 회전 구간 — 30 m 경로 끝이 옆으로 감; y 최대는 temp 2hz 3 s 고속, y 최소는 spat 5m 의 후진·되돌아옴 451 점.
#      nuScenes 처럼 이상치 제거 없이 원 [min,max] 를 쓴다 — 규칙 동일성 우선. 0.1/99.9 백분위는 x ±26, y [-0.07, 30.4].)
#   lane_class_ids (0,1,2,3) = Broken·Solid·SolidSolid·Center 전부: 넷 다 차로와 평행한 선이라 "차로 방향" 근거가 된다
#     (Center = 차로 중심선이 가장 직접적; nuScenes 의 divider·boundary 에 해당하는 표시선도 같은 방향).
#   clip_to_boundary=False, boundary_class_ids=(): B2D 지도에는 도로 가장자리(주행 불가 경계) 클래스가 없다(4 종 모두 차로 표시선/중심선).
#     train GT 2hz 경로(6 칸 유효 196,769 프레임)가 ROI 안에서 가로지르는 비율: Broken 7.08 %, Solid 3.23 %, SolidSolid 0.033 %,
#     Center 43.5 % → 전문가가 Solid 를 Broken 의 절반 꼴로 실제로 넘는다(교차로·…TwoWays 추월 등). Solid 에서 잡음을 자르면 GT 가
#     있는 쪽을 막게 된다. SolidSolid 는 거의 안 넘지만(65 프레임) "지도 밖(도로 밖)" 이 아니라 "반대 차로" 라 nuScenes 의 경계 자르기와
#     뜻이 다르다 → 자르기 끔(잡음은 지도 정렬 이방성만). 모듈은 clip_to_boundary=False 면 _clip_boundary 를 부르지 않아 빈 목록이 안전하다.
#   search_radius 3.5 m = 차로 폭 1 개: Center 선 중 같은 도로·같은 진행 방향·이웃 lane_id 쌍 간격의 중앙값 3.50 m
#     (2,581 쌍; Town12/13/15 는 3.25, 나머지 3.50).
#   수정(2026-10-19): 사용자 결정(API 변경) — anisotropy_ratio 인자 제거(모듈이 더 받지 않음). 새 규칙: 정규화 공간 잡음 타원의 크기
#     (축별 표준편차 ∝ norm_scale)는 그대로 두고 긴 축(norm_scale[1] = 전방 y 축)만 지역 차선 접선 방향으로 회전한다.
#     B2D 계획 좌표(라이다 = 교사 GT 좌표)의 전방 = +y: 카메라 투영(라이다 (0,15) → CAM_FRONT, (+8,8) → CAM_FRONT_RIGHT) 과
#     train GT 미래 y 범위([-14.7, 47.1], x 는 ±30)로 확인 → 모듈의 가정(norm_scale[1] = 전방·긴 축) 성립.
#     단 B2D norm_scale = (31, 32) 로 두 축이 거의 같아(비 1.03) 회전 효과는 사실상 없다(타원이 거의 원) — 확인 필요 항목으로 보고.
#   score_thr 0.4·boundary_margin 0.1 m 은 nuScenes 와 같다(margin 은 자르기 끔이라 쓰이지 않음).
plan_diffusion = dict(
    type="MapAlignedPlanTruncatedDiffusion", embed_dims=embed_dims, ego_fut_ts=ego_fut_ts,
    num_train_timesteps=1000, trunc_steps=50, infer_t=8, beta_start=1e-4, beta_end=0.02,
    norm_scale=(31.0, 32.0), norm_offset=(0.0, 16.0), infer_seed=0,
    score_thr=0.4, lane_class_ids=(0, 1, 2, 3), boundary_class_ids=(), search_radius=3.5,
    boundary_margin=0.1, clip_to_boundary=False,
    # 수정(2026-11-02): 사용자 결정 "누씬하고 같게 맞추자" — nuScenes 최종 2단계(configs/nuscenes/v2_teacher/stage2/train.py)와 같은
    #   학습 t 추출(infer_t 8 중심 이산 가우시안, σ 8)과 지도 확신도 선택(selective). #이전 uniform·selective False(모듈 기본)
    t_sampling="gauss", t_std=8.0, selective=True,
)

model = dict(
    qwen_cache_dir=qwen_cache_dir,
    head=dict(
        onedecoder_head=dict(
            num_stream_groups=num_stream_groups,  # 수정(2026-11-01): 논문 eff batch 32 — 16 (아래 덮어쓰기 블록). #이전 base 값 24
            plan_diffusion=plan_diffusion,
            map_instance_bank=dict(type="PlanMapMemoryInstanceBank"),
            qwen_token_attn=dict(
                num_token_types=7,
                token_type_names=teacher_folders,
                route_segs=None,
            ),
        ),
    ),
)

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
    teacher_loader,  # LoadQwenCacheNusc(bench2drive_qwen_l18) → 교사 캐시 7 폴더(2 Hz 키프레임)
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "qwen_type", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "gt_ego_fut_cmd", "target_point", "ego_status", "ego_status_mask",
               "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               "gt_agent_fut_trajs", "gt_agent_fut_masks",
               "gt_ego_spat_trajs_2m", "gt_ego_spat_masks_2m", "gt_ego_spat_trajs_5m", "gt_ego_spat_masks_5m",
               "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz", "gt_ego_fut_trajs_5hz", "gt_ego_fut_masks_5hz"],
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
               "target_point"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token", "token"],
    ),
]

data = dict(
    num_stream_groups=num_stream_groups,  # 수정(2026-11-01): 논문 eff batch 32 — 16. #이전 base 값 24
    samples_per_gpu=batch_size,           # base 와 같은 2(재기재)
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

# 수정(2026-11-01): 사용자 결정(D안) — 2단계 시작 = 1단계 EMA 가중치(nuScenes train_e3.py 와 같은 방식). 1단계 max_iters 1,408,512
#   (= 117,376 × 12, configs/b2d/v2_teacher/stage1/train.py) 의 ema_iter_1408512.pth 를 tools/ema_to_ckpt.py --config 로 바꾼 파일.
#   1단계 길이를 바꾸면 이 숫자도 같이 바꿀 것(실행 스크립트가 1단계 config 의 max_iters 와 대조해 다르면 멈춘다).
load_from = "./results/hipad_b2d_laq_v2_teacher_stage1/iter_1408512_ema.pth"  #이전 ./results/hipad_b2d_laq_v2_teacher_stage1/latest.pth(1단계 원 가중치)

# ---- 수정(2026-11-01): 논문 값 덮어쓰기 — optimizer·lr·runner·저장 간격(변수는 위 "논문 값 덮어쓰기" 블록) ----
optimizer = dict(weight_decay=0.01)  #이전(base) 0.001 — lr 2e-4·paramwise(img_backbone ×0.5)는 base 그대로
optimizer_config = dict(cumulative_iters=cumulative_iters)  #이전 24
lr_config = dict(warmup_iters=500 * cumulative_iters)        # 8,000  #이전 12,000 (= 둘 다 optimizer step 500)
runner = dict(max_iters=num_iters_per_epoch * num_epochs)     # 수정(2026-11-01): D안 6 에폭 — 704,256 (= step 44,016). cosine 길이도 이 값(mmcv IterBasedRunner → by_epoch=False → runner.max_iters)  #이전 18 에폭 2,112,768 (step 132,048), 그 전 2,112,912 (step 88,038)
checkpoint_config = dict(interval=num_iters_per_epoch)        # 매 에폭  #이전 117,384
evaluation = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval)  # 20 에폭 > 6 에폭 → 학습 중 평가 없음  #이전 2,347,680

run_name = "hipad_b2d_laq_v2_teacher_stage2"

# 수정(2026-11-01): EMA 가중치 함께 저장(사용자 규칙 "EMA 가중치는 체크포인트와 함께 저장", nuScenes 최종 레시피 train_e3.py 와 같은 훅).
#   학습에는 영향 없음(mmdet3d_plugin/apis/ema_sidecar_hook.py — 읽기만). optimizer step(누적 16 micro-iter)마다 갱신.
#   값은 nuScenes 와 같은 규칙(학습 길이 대비 같은 비율: 평균 창 = 전체 step 의 11.4 %, ramp = 4.6 %)으로 정했다(결과 보기 전 결정):
#   (18 에폭 때) 전체 step 132,048 → 창 ≈ 15,040 step(decay = 1 − 1/15,040 ≈ 0.999934), ramp ≈ 6,016(= 2,000 × 132,048 / 43,900).
#   수정(2026-11-01): 사용자 결정(D안) 6 에폭 — 같은 규칙(창 = 5,000 × 전체 step / 43,900, ramp = 2,000 × 전체 step / 43,900, decay 소수 6 자리)으로 다시:
#     전체 step 44,016(= 7,336 × 6) → 창 5,013.2 step → decay = 1 − 1/5,013.2 = 0.9998005 → 0.999801, ramp 2,005.3 → 2,005.
#     (B2D 1단계도 전체 step 44,016 인데 0.9998/2,000 을 쓴다 — 1단계는 nuScenes 값을 그대로 둔 반올림. 차이: 창 5,000 vs 5,013, 0.3 %.)
#     수정(2026-11-02): 위 괄호는 이전 상태 — 사용자 결정으로 B2D 1단계도 같은 규칙 값 0.999801/2,005 가 됐다.
#   평가용 변환: tools/ema_to_ckpt.py <work_dir> <N> --config <이 파일>.
custom_hooks = [dict(type="EMASidecarHook", decay=0.999801, ramp_steps=2005, update_every=cumulative_iters,  # 수정(2026-11-01): D안 6 에폭 #이전 0.999934 / 6016(18 에폭)
                     max_keep=3, priority="ABOVE_NORMAL")]
