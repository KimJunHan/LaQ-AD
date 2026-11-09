# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   LaQ v2 + 교사 캐시 7 폴더 · 토큰 유형 임베딩 — B2D 1단계 학습 config (configs/nuscenes/v2_teacher/stage1/train.py 의 B2D 짝).
#   ../../stage1/train.py(B2D LaQ 1단계, HiP-AD 원본 B2D 값: 12 epoch, lr 2e-4, eff batch 64)를 상속하고 **언어 토큰 출처만** 바꾼다:
#     종전  data/bench2drive_qwen_l18/<token>.pt (원본 Qwen2-VL-2B 18층, LoadQwenCacheNusc)
#     이번  data/bench2drive/laq_cache/{context,detection,tracking,motion,map,plan,action}/<clip>/<kf:05d>.pt 를 이 순서로 이어 붙인 K/V
#           (정답 문법은 nuScenes 와 같은 547 토큰·7 폴더 배치 → 폴더 길이는 _meta.json 또는 첫 파일에서 읽고 전 표본 동일 검사)
#   2 Hz 키프레임 규칙(사용자 결정): 캐시는 frame_idx % 5 == 0 키프레임만 있다. 10 Hz 학습 프레임은 자기 시각 이하 가장 최근
#     키프레임 캐시를 쓴다(교사 나이 0~0.4 s; 폐쇄루프 2 Hz 갱신·직전 출력 재사용과 같은 규칙) — LoadTeacherCacheB2D.
#   + 0 초기화 토큰 유형 임베딩(7×256) ON, route_segs OFF. 그 밖(백본·앵커·클래스·해상도 640×352·fp16·lr·스트림 그룹·평가)은 base 와 같다.
#   배치: base 의 micro-batch 2 × 누적 32(= eff 64, HiP-AD 원본 8 GPU × 8)를 그대로 둔다. micro-batch 는 VRAM 실측 뒤 확정
#     (교사 K/V 가 nuScenes 와 같은 길이로 늘어 메모리가 base 와 다르다) — 바꿀 때 eff batch 64 가 유지되도록 cumulative_iters 도 함께.
#   B안: context 폴더(지시문 뒤 자차 상태 줄 포함) → 계획 결과는 "w/ ego status" 범주.
#   run_name hipad_b2d_laq_v2_teacher_stage1 → results/hipad_b2d_laq_v2_teacher_stage1.
_base_ = ["../../stage1/train.py"]

custom_imports = dict(imports=["tools.b2d_teacher_register"], allow_failed_imports=False)

# ---- base 와 같은 값(상속 config 에서는 base 변수를 직접 참조할 수 없어 다시 적는다; 값 불변) ----
det_class_names = ["car", "van", "truck", "bicycle", "traffic_sign", "traffic_cone", "traffic_light", "pedestrian", "others"]
map_roi_size = (30, 60)
map_num_pts = 20
strides = [4, 8, 16, 32]
num_depth_layers = 3
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# ---- 교사 캐시 ----
teacher_cache_root = "data/bench2drive/laq_cache"   # 아직 없음(교사 SFT·캐시 추출 뒤 생성 — GPU 작업)
teacher_folders = ["context", "detection", "tracking", "motion", "map", "plan", "action"]  # 고정 순서(유형 번호 0..6)
teacher_expect_lens = None
teacher_meta_expect = {f: dict(prompt_mode="B", teacher_chain_verified=True, constrained_decoding="answer_grammar_v1")
                       for f in teacher_folders}
qwen_cache_dir = teacher_cache_root  # SparseDetector 의 "언어 융합 ON" 표시(파이프라인이 qwen_hidden 을 넣으므로 모델은 디스크를 읽지 않음)

teacher_loader = dict(type="LoadTeacherCacheB2D", cache_root=teacher_cache_root, folders=teacher_folders,
                      hidden_dim=1536, expect_lens=teacher_expect_lens, meta_expect=teacher_meta_expect, keyframe_stride=5)

model = dict(
    qwen_cache_dir=qwen_cache_dir,
    head=dict(
        onedecoder_head=dict(
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
               "gt_agent_fut_trajs", "gt_agent_fut_masks", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz"],
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
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

# 수정(2026-11-01): EMA 가중치 함께 저장(사용자 규칙, nuScenes 최종 1단계 train_e3_ddp4.py 와 같은 훅). 학습에는 영향 없음.
#   optimizer step(누적 32 micro-iter)마다 갱신. 전체 step 44,016(= 3,668 × 12) 이 nuScenes 1단계(43,900)와 거의 같아
#   같은 규칙(창 11.4 %, ramp 4.6 %)의 값이 nuScenes 값 그대로다: decay 0.9998(창 5,000), ramp 2,000.
#   ../../stage1/train.py 의 cumulative_iters(32)를 상속 config 에서 참조할 수 없어 숫자로 적는다 — base 를 바꾸면 같이 바꿀 것.
# 수정(2026-11-02): 사용자 결정 "B2D 1단계 EMA 값도 맞추자" — 반올림 대신 같은 규칙으로 계산한 값(2단계와 같은 식):
#   창 = 5,000 × 44,016 / 43,900 = 5,013.2 step → decay = 1 − 1/5,013.2 = 0.9998005 → 0.999801, ramp = 2,000 × 44,016 / 43,900 = 2,005.3 → 2,005.
#   위 "nuScenes 값 그대로" 서술은 이전 값(0.9998 / 2000).
custom_hooks = [dict(type="EMASidecarHook", decay=0.999801, ramp_steps=2005, update_every=32,  #이전 0.9998 / 2000
                     max_keep=3, priority="ABOVE_NORMAL")]

run_name = "hipad_b2d_laq_v2_teacher_stage1"
