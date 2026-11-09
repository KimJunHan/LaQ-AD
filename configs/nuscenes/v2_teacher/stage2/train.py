# 수정(2026-10-16): 교사 캐시 7 폴더 · 토큰 유형 임베딩 — LaQ v2 + 교사(QA→SFT; GRPO 는 시도 후 제외 — 수정 2026-10-19) 캐시 2단계 학습 config.
#   ../../stage2/train_v2.py(실험 C "LaQ 교정 레시피" 2단계)를 상속하고 **언어 토큰 출처만** 바꾼다:
#     종전  data/nuscenes/qwen_l18/<token>.pt (1,159 토큰, 원본 Qwen2-VL-2B 18층)
#     이번  data/nuscenes/laq_cache/{context,detection,tracking,motion,map,plan,action}/<token>.pt 를 이 순서로 이어 붙인 K/V
#           (계획 길이 ≈ 1,860+168+40+240+68+13+8 = 2,397; 실제 길이는 각 폴더 _meta.json 또는 첫 파일에서 읽고 전 표본 동일 검사)
#   + 0 초기화 토큰 유형 임베딩(7×256, 사영 뒤 K/V 에 더함) ON, 쿼리 구간별 폴더 경로 지정(route_segs) OFF(모든 쿼리가 모든 토큰).
#   그 밖(학습 레시피·디코더·손실·해상도·fp32·lr·스트림 그룹·평가)은 train_v2.py 와 완전히 같다.
#   B안(사용자 결정 2026-10-15): 교사 문맥(context 폴더)에 지시문 뒤 자차 상태 줄이 들어 있다(언어 경유 자차 상태).
#     → 계획 결과는 "w/ ego status" 범주로 보고. 학생의 ego_status 입력·Collect 는 v2 와 같다(변경 없음).
#     수정(2026-10-16): 설계 검토 반영 — 교사 캐시 생성기(extract_teacher_cache_v2.py)가 7 폴더 _meta.json 모두에
#     prompt_mode="B"(자차 상태 줄 포함)·teacher_chain_verified=True(QA→SFT 단계 fold·분할 해시 검사 통과; 병합 교사 = SFT 가중치)·
#     constrained_decoding="answer_grammar_v1"(문법 제약 생성) 을 기록한다 → 아래 teacher_meta_expect 로 셋 다 강제 검사.
#   run_name hipad_nusc_laq_v2_teacher_stage2, load_from = results/hipad_nusc_laq_v2_teacher_stage1/latest.pth
#   (유형 임베딩은 1단계에서 det/map 손실로 학습된 값을 이어받는다 — 기본 lr, 별도 lr 배수 없음. 2단계 optimizer 는 v2 와 같다).
_base_ = ["../../stage2/train_v2.py"]

# 수정(2026-10-19): 사용자 결정 — 지도 정렬 선택적 Gaussian 잡음(지도 밖 잡음 제거)
#   계획 확산의 잡음만 바꾼다(PlanTruncatedDiffusion → MapAlignedPlanTruncatedDiffusion, 지도 뱅크 → PlanMapMemoryInstanceBank).
#   지도 = 직전 프레임 모델 예측(detach)을 자차 이동으로 현재 좌표에 옮긴 것(정답 지도 아님, 학습·평가 같은 규칙).
#   t·일정·정규화·시드·시점 임베딩·출력·파라미터는 종전과 같다(state_dict 동일 → 1단계 체크포인트 그대로 읽힘).
#   이 config 를 상속하는 token_lidar/token_radar 2단계도 같이 바뀐다(token_lidar/stage2/train.py 의 custom_imports 에 등록 모듈 추가).
#   1단계 config(확산 없음)는 바꾸지 않는다.
custom_imports = dict(imports=["tools.plan_diffusion_map_register"], allow_failed_imports=False)

# ---- base 와 같은 값(상속 config 에서는 base 변수를 직접 참조할 수 없어 다시 적는다; 값 불변) ----
det_class_names = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
map_roi_size = (30, 60)
map_num_pts = 20
strides = [4, 8, 16, 32]
num_depth_layers = 3
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# ---- 교사 캐시 ----
teacher_cache_root = "data/nuscenes/laq_cache"
teacher_folders = ["context", "detection", "tracking", "motion", "map", "plan", "action"]  # 고정 순서(K/V 배치 = 유형 번호 0..6)
teacher_expect_lens = None   # None = 폴더 _meta.json → 첫 파일 순으로 길이를 정하고 전 표본 동일 검사
teacher_meta_expect = {f: dict(prompt_mode="B", teacher_chain_verified=True, constrained_decoding="answer_grammar_v1")
                       for f in teacher_folders}   # 수정(2026-10-16): 설계 검토 반영 — B안·교사 연결·제약 생성 강제(키 = 추출기 기록 키)
qwen_cache_dir = teacher_cache_root  # SparseDetector 의 "언어 융합 ON" 표시. 파이프라인이 qwen_hidden 을 넣으므로 모델은 디스크를 읽지 않는다

teacher_loader = dict(type="LoadTeacherCacheNusc", cache_root=teacher_cache_root, folders=teacher_folders,
                      hidden_dim=1536, expect_lens=teacher_expect_lens, meta_expect=teacher_meta_expect)

model = dict(
    qwen_cache_dir=qwen_cache_dir,
    head=dict(
        onedecoder_head=dict(
            # 수정(2026-10-19): 사용자 결정 — 지도 정렬 선택적 Gaussian 잡음(지도 밖 잡음 제거)
            #   score_thr 0.4: 저장소 지도 시각화 기본값(--map-thr 0.4)과 같은 "믿을 만한 지도 요소" 기준.
            #   search_radius 3.5 m: 차선 폭 1 개 — 차로 안 경유점이면 자기 차로의 구분선/경계가 이 안에 든다.
            # 수정(2026-10-19): 사용자 결정 — 기존 잡음 타원을 차선 방향으로 회전(새 비율 값 없음)
            #   잡음 크기는 기존 DiffusionDrive 식 정규화 잡음과 같다(m 좌표 표준편차 = norm_scale (13, 29) × sqrt(1−ᾱ_t)).
            #   가까운 차선이 있으면 그 타원을 회전만 해 긴 축(29 m)을 차선 방향, 짧은 축(13 m)을 차선 가로 방향에 맞춘다.
            #   새 비율 값이나 조정값은 없다(종전 anisotropy_ratio 1.5 제거). 전방축 차선이면 기존 등방 잡음과 비트 동일.
            #   boundary_margin 0.1 m: 잘린 점이 경계선 위에 정확히 놓이지 않도록 하는 수치 여유(차로 폭보다 훨씬 작음).
            plan_diffusion=dict(type="MapAlignedPlanTruncatedDiffusion", score_thr=0.4, lane_class_ids=(1, 2),
                                boundary_class_ids=(2,), search_radius=3.5,  # 수정(2026-10-19): 사용자 결정 — 기존 잡음 타원을 차선 방향으로 회전(새 비율 값 없음) — anisotropy_ratio 제거
                                boundary_margin=0.1, clip_to_boundary=True,
                                # 수정(2026-10-23): 사용자 결정 — ① 학습 잡음 세기 t 를 추론 t(8) 중심 이산 가우시안(σ = 8)에서 뽑기,
                                #   ② 지도 확신도(가까운 차선 점수) 확률로만 차선 회전 타원(나머지는 종전 전방축 타원)
                                t_sampling="gauss", t_std=8.0, selective=True),
            map_instance_bank=dict(type="PlanMapMemoryInstanceBank"),
            qwen_token_attn=dict(
                num_token_types=7,                 # 0 초기화 유형 임베딩 (7, 256) ON
                token_type_names=teacher_folders,  # 유형 번호 → 폴더 이름
                route_segs=None,                   # 경로 지정 OFF. 켜는 예(별도 config 로):
                #   [["context", "detection", "tracking", "motion"],            # det 쿼리 [0, 900)
                #    ["context", "map"],                                          # map 쿼리 [900, 1000)
                #    ["context", "plan", "action", "motion", "map"]]              # plan/ego 쿼리 [1000, N)
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
    teacher_loader,  # 수정(2026-10-16): LoadQwenCacheNusc(qwen_l18) → 교사 캐시 7 폴더
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "qwen_type", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "ego_status", "ego_status_mask", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz",
               "gt_agent_fut_trajs", "gt_agent_fut_masks"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token", "stream_group"],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    teacher_loader,  # 수정(2026-10-16): LoadQwenCacheNusc(qwen_l18) → 교사 캐시 7 폴더
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "qwen_type", "gt_bboxes_3d", "gt_labels_3d", "timestamp", "projection_mat",
               "image_wh", "ego_status", "gt_ego_fut_cmd"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token", "token"],
    ),
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

load_from = "./results/hipad_nusc_laq_v2_teacher_stage1/latest.pth"  # 교사 캐시 1단계에서 시작

run_name = "hipad_nusc_laq_v2_teacher_stage2"
