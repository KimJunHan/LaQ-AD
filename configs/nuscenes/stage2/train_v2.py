# 수정(2026-10-15): 실험 C "LaQ 교정 레시피" 2단계 학습 config (새 실험 = 새 config; ./train.py 는 그대로 둔다).
#   run_name hipad_nusc_laq_v2_stage2, load_from = results/hipad_nusc_laq_v2_stage1/latest.pth.
#   ./train.py(2단계: +motion·plan, 확산 계획, lr 2e-5 + 신규 모듈 ×10) 를 상속하고 1단계 v2 와 같은 데이터·손실 교정을 적용한다.
#   inter_graph_model(InteractiveAttention [plan,ego]←[det,map]) 과 temp_graph query_list [["det"],["map"],["plan","ego"]] 은
#   base 가 이미 HiP-AD 원본(9610ed8 projects/configs/hipad_b2d_stage2.py:228-230, 277-282)과 같아 그대로 둔다.
_base_ = ["./_base_common.py"]  # 수정(2026-10-16): v1 실험 config 제거 — 종전 ./train.py 를 _base_common.py 로 이름 변경(내용 동일)

# ---- base 와 같은 값(재기재, 값 불변) ----
batch_size = 8
cumulative_iters = 6
det_class_names = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
map_roi_size = (30, 60)
map_num_pts = 20
strides = [4, 8, 16, 32]
num_depth_layers = 3
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
qwen_cache_dir = "data/nuscenes/qwen_l18"
project_dir = "/home1/irteam/data-vol1/src/HiP-AD"

# 수정(2026-10-15): (2) 독립 스트림 — 1단계 v2 와 같은 방식. 2단계는 8×6 = eff 48(SD stage2 GPU 8 × 6 과 같은 step 당 독립 시퀀스 48).
#   종전: 같은 8 시퀀스의 연속 6 프레임(독립 8). epoch 당 micro-iter 3516 = 28130//48 × 6 (base 값과 같다).
num_stream_groups = cumulative_iters

load_from = "./results/hipad_nusc_laq_v2_stage1/latest.pth"  # 수정(2026-10-15): v2 1단계에서 시작

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            # 수정(2026-10-15): (8) 버그 A(results/review_s2/REPORT.md) — 자차 모듈은 1단계에서 기울기 0(Adam exp_avg_sq==0)인
            #   사실상 신규 모듈인데 ×10 이 빠져 lr 2e-5 로 학습됐다. plan_* 와 같이 ×10(유효 2e-4).
            "ego_refine": dict(lr_mult=10.0),
            "ego_deformable": dict(lr_mult=10.0),
            "ego_instance_bank": dict(lr_mult=10.0),
        }
    ),
)

model = dict(
    neck_dtype="bf16",  # 수정(2026-10-20): fp16 넘침 원인 수정 — FPN 출력이 fp16 최대(65,504)를 넘어 Inf(1단계 iter ~7,500). neck 만 bfloat16(사용자 결정), 처음부터 재학습
    img_backbone=dict(with_cp=True),  # 수정(2026-10-15): (7) 1단계 v2 와 같은 결정 = SD stage2(:99). 수치 동일, BN running stats 갱신 2회/iter 만 다름
    head=dict(
        onedecoder_head=dict(
            num_stream_groups=num_stream_groups,  # 수정(2026-10-15): (2)
            map_instance_bank=dict(
                # 수정(2026-10-15): (6) 1단계 v2 와 같은 SD 방식 지도 앵커(load_from 이 학습된 앵커 파라미터로 덮어쓴다 — 초기값 일관성용)
                anchor=f"{project_dir}/data/kmeans/nusc_map_100_sd.npy",
            ),
            loss_det_reg=dict(
                cns_detach=False,  # 수정(2026-10-15): (3) SD 원본
                cls_allow_reverse=[det_class_names.index("barrier")],  # base 와 같음(재기재)
            ),
        ),
    ),
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(type="ResizeCropFlipImage"),
    dict(type="MultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),
    dict(type="BBoxRotation"),  # 수정(2026-10-15): (7) 항등(rot3d_range=[0,0])
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),  # 수정(2026-10-15): (4) SD 원형 55 m
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=map_roi_size, simplify=False, normalize=False,
         sample_num=map_num_pts, permute=True),
    dict(type="LoadQwenCacheNusc", cache_dir=qwen_cache_dir),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "ego_status", "ego_status_mask", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz",
               "gt_agent_fut_trajs", "gt_agent_fut_masks"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token", "stream_group"],  # 수정(2026-10-15): (2)
    ),
]

data = dict(
    num_stream_groups=num_stream_groups,  # 수정(2026-10-15): (2)
    train=dict(
        pipeline=train_pipeline,
        use_valid_flag=False,  # 수정(2026-10-15): (4) num_lidar_pts>0 (SD). 평가 쪽 불변.
    ),
)

run_name = "hipad_nusc_laq_v2_stage2"

# 수정(2026-10-18): 사용자 결정 — HiP-AD 원본과 같은 fp16 혼합 정밀도(백본·FPN 만 fp16, 디코더·손실·언어 모듈은 fp32, 정적 배율 32).
#   근거: results/diag_fp16_20261016/REPORT.md(fp16 구간 최대 2,268 < 65,504, 32배 기울기 최대 98, 시험 560 iter 넘침 0).
#   넘친 step 은 건너뛰지 않고 멈춤(apis/fp16_strict_hook.py, mmdet_train.py fp16 분기). attention 핵심은 bf16 유지(B2D 넘침 실측).
#   base(_base_common.py)가 fp16 = None 이라 _delete_ 필요.
fp16 = dict(_delete_=True, loss_scale=32.0)
