# 수정(2026-10-15): 실험 C "LaQ 교정 레시피" 1단계 학습 config (새 실험 = 새 config; ./train.py 는 그대로 둔다).
#   run_name hipad_nusc_laq_v2_stage1 → results/hipad_nusc_laq_v2_stage1, 평가 evaluation/hipad_nusc_laq_v2_stage1.
#   근거: results/diag_source/REPORT.md(정렬 목록 1~8), diag_recipe/REPORT.md(레시피 대조). 대조 원본은
#   /home1/irteam/data-vol1/src/SparseDrive_ref/projects/configs/sparsedrive_small_stage1.py (아래 "SD:" 는 그 파일의 줄).
#   ./train.py 를 _base_ 로 상속하고 아래 항목만 덮어쓴다. 유지: 언어 모듈(QwenTokenAttn), fp32(fp16=None), 704×256,
#   HiP-AD 통합 디코더 구조, wd 1e-3, backbone lr_mult 0.5, warmup 500 step, cosine, clip 25, 100 epoch, eff batch 64.
#   state_dict 키는 ./train.py 와 같다(새 파라미터·버퍼 없음 — 스트림 그룹 상태는 파이썬 속성).
_base_ = ["./_base_common.py"]  # 수정(2026-10-16): v1 실험 config 제거 — 종전 ./train.py 를 _base_common.py 로 이름 변경(내용 동일)

# ---- base 와 같은 값(상속 config 에서는 base 변수를 직접 참조할 수 없어 다시 적는다; 값 불변) ----
batch_size = 16
cumulative_iters = 4
num_epochs = 100
det_class_names = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
map_roi_size = (30, 60)
map_num_pts = 20
strides = [4, 8, 16, 32]
num_depth_layers = 3
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
qwen_cache_dir = "data/nuscenes/qwen_l18"
project_dir = "/home1/irteam/data-vol1/src/HiP-AD"

# 수정(2026-10-15): (2) 독립 스트림 64 개 — 1 GPU 에서 "GPU 4 개 × 16"(= SD GPU 8 × 8 과 같은 step 당 독립 시퀀스 64) 등가.
#   샘플러가 64 개 스트림을 4 그룹(각 16)으로 나눠 micro-step k=0..3 마다 그룹 k 를 내보내고, 디코더가 그룹별 뱅크 상태를
#   저장·복원한다(models/loader/samplers/group_in_batch_sampler.py, models/decoder/temporal_bank/stream_state.py).
#   종전(./train.py): 같은 16 시퀀스의 연속 4 프레임 = step 당 독립 시퀀스 16. eff batch 64(16×4) 는 그대로.
num_stream_groups = cumulative_iters
# 수정(2026-10-15): epoch 당 micro-iter 를 누적 수의 배수로(= SD 의 28130 // 64 = 439 step × 4). 체크포인트·재개 지점이
#   optimizer step 경계와 맞아 재개 시 그룹 순환이 어긋나지 않는다. 종전 1758 → 1756 (총 175,600 = SD 43,900 step × 4).
num_iters_per_epoch = int(28130 // (batch_size * cumulative_iters)) * cumulative_iters
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
runner = dict(type="IterBasedRunner", max_iters=num_iters_per_epoch * num_epochs)
evaluation = dict(interval=num_iters_per_epoch)

# 수정(2026-10-15): (1) lr 4e-4 = SD stage1(SD:685, eff batch 64). 종전 2e-4. wd·lr_mult·warmup·cosine·clip 은 base 그대로.
optimizer = dict(
    lr=4e-4,
    # 수정(2026-10-20): 근본 원인 수정(사용자 결정 "FPN 학습률 절반 + bf16") — BN 있는 FPN(HiP-AD) 에 SD 의 lr 4e-4 를 쓴 조합은 원본 어디에도 없고,
    #   BN 앞 합성곱 가중치가 손실에 영향 없이 커져(쏠림 비중 0.23→0.76, FPN 출력 표준편차 18→1,494) fp16 넘침을 낳았다.
    #   FPN(img_neck) 만 HiP-AD 원본 짝인 2e-4(×0.5)로, 백본 ×0.5 는 그대로. mmcv custom_keys 는 dict 병합이라 백본 키를 다시 적는다.
    paramwise_cfg=dict(custom_keys={"img_backbone": dict(lr_mult=0.5), "img_neck": dict(lr_mult=0.5)}),
)

model = dict(
    neck_dtype="bf16",  # 수정(2026-10-20): fp16 넘침 원인 수정 — FPN 출력이 fp16 최대(65,504)를 넘어 Inf(1단계 iter ~7,500). neck 만 bfloat16(사용자 결정), 처음부터 재학습
    img_backbone=dict(
        # 수정(2026-10-15): (7) with_cp=True = SD:99. 체크포인트 재계산이라 forward·기울기 수식은 같고(첫 iter 손실 det_cls
        #   13.6884 vs 13.6883 — deformable 커널 atomicAdd 잡음 수준), 차이는 재계산 forward 에서 BN running stats 가 iter 당
        #   2회 갱신되는 것뿐(SD 원본과 같은 조건). 스모크(GPU1, batch 4): 메모리 24.9→22.4 GB, s/iter +2%(공유 GPU 잡음 범위).
        with_cp=True,
    ),
    head=dict(
        onedecoder_head=dict(
            num_stream_groups=num_stream_groups,  # 수정(2026-10-15): (2) 그룹별 뱅크 상태
            map_instance_bank=dict(
                # 수정(2026-10-15): (6) SD kmeans_map 방식(전 표본, 8 m 세로선) 지도 앵커 — tools/kmeans/kmeans_nusc_sd.py 로 생성.
                #   종전 nusc_map_100.npy(1/100 표본, 8~20 m, 30% 가로)는 그대로 둔다. 검출 앵커는 종전 nusc_det_900 유지(보고서 참조).
                anchor=f"{project_dir}/data/kmeans/nusc_map_100_sd.npy",
            ),
            loss_det_reg=dict(
                # 수정(2026-10-15): (3) centerness 타깃 detach 원복(SD 원본: 예측 중심을 detach 하지 않는다).
                cns_detach=False,
                # 수정(2026-10-15): (5) barrier 앞뒤 구분 없음(SD:262). 종전 1단계에는 없었다(2단계에만).
                cls_allow_reverse=[det_class_names.index("barrier")],
            ),
        ),
    ),
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(type="ResizeCropFlipImage"),
    dict(type="MultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),
    dict(type="BBoxRotation"),  # 수정(2026-10-15): (7) SD:531 과 같은 위치. rot3d_range=[0,0] 이라 항등(행렬 ×I, yaw +0 — 값 불변)
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    # 수정(2026-10-15): (4) 학습 GT 거리 필터 — 정사각 ±50 m(BEVObjectRangeFilter) → SD 원형 55 m(SD:535, 전 클래스 55).
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=map_roi_size, simplify=False, normalize=False,
         sample_num=map_num_pts, permute=True),
    dict(type="LoadQwenCacheNusc", cache_dir=qwen_cache_dir),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "ego_status", "ego_status_mask", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz"],
         # 수정(2026-10-15): (2) stream_group — 샘플러가 붙인 그룹 번호를 디코더로 전달
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token", "stream_group"],
    ),
]

data = dict(
    num_stream_groups=num_stream_groups,  # 수정(2026-10-15): (2) 샘플러 그룹 수(mmdet_train 이 cumulative_iters·디코더 값과 같은지 검사)
    train=dict(
        pipeline=train_pipeline,
        # 수정(2026-10-15): (4) GT 유효 마스크 = num_lidar_pts>0 (SD use_valid_flag=False). 종전 valid_flag(lidar+radar>0).
        #   평가(val/test) 쪽은 바꾸지 않는다(평가 프로토콜 불변).
        use_valid_flag=False,
    ),
)

run_name = "hipad_nusc_laq_v2_stage1"

# 수정(2026-10-18): 사용자 결정 — HiP-AD 원본과 같은 fp16 혼합 정밀도(백본·FPN 만 fp16, 디코더·손실·언어 모듈은 fp32, 정적 배율 32).
#   근거: results/diag_fp16_20261016/REPORT.md(fp16 구간 최대 2,268 < 65,504, 32배 기울기 최대 98, 시험 560 iter 넘침 0).
#   넘친 step 은 건너뛰지 않고 멈춤(apis/fp16_strict_hook.py, mmdet_train.py fp16 분기). attention 핵심은 bf16 유지(B2D 넘침 실측).
#   base(_base_common.py)가 fp16 = None 이라 _delete_ 필요.
fp16 = dict(_delete_=True, loss_scale=32.0)
