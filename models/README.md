# models/ — LaQ-AD(HiP-AD 기반) 모델·데이터 소스

수정(2026-10-15): 소스 구조 정리. 미사용 코드를 걷어내고(`archive/refactor_20261015/`), 모델 코드를 **데이터 로더 → 인코더 → 넥 →
디코더 → 헤드** 로 나눈 뒤, 각 하위 폴더를 발표 자료 "전체 구조" 그림의 상자와 1:1 로 맞췄다(2차 재배치).
레지스트리 `type` 이름과 모듈 속성 이름(=체크포인트 state_dict 키)은 하나도 바꾸지 않았다 — 기존 config·체크포인트가 그대로 동작한다.

## 아키텍처 그림 상자 → 폴더 → 주요 클래스

| 그림 상자 | 폴더 | 주요 클래스 / 함수 (레지스트리 type) |
|---|---|---|
| 6캠 영상 입력·데이터 | `loader/{nuscenes,bench2drive,navsim,pipelines,samplers,evaluation}/` | `NuScenes3DDataset`(=`NuScenesMiniDataset`), `Bench2DriveDataset`, `NavsimDataset`, 파이프라인 transforms, `GroupInBatchSampler`, 평가기 |
| ResNet-50 + FPN / multi-scale features | `encoder/image_resnet50_fpn/` (+ config 의 mmdet `ResNet`·`FPN`) | `GridMask`; 특징 추출은 `SparseDetector.extract_feat` |
| Qwen2-VL-2B | `encoder/vlm_qwen2vl/` | `QwenOnlineEncoder`(qwen_online.py), `QwenTrajectoryGenerator`(VLA-v0, 미등록) |
| VL tokens (offline cache) | `encoder/vl_tokens_cache/` | `LoadQwenCacheNusc`(파이프라인; `loader/pipelines` 가 다시 내보냄), 대체 경로 `SparseDetector._maybe_load_qwen_cache` |
| projector | `encoder/projector/` | `build_qwen_projector` (1536→512→GELU→256→LN) |
| multi-scale features 보조 깊이 | `neck/multi_scale_depth/` | `DenseDepthNet` |
| 통합 디코더(전체 배선) | `decoder/unified_decoder.py` | `SparseOneDecoder` (`operation_order` 디스패치) |
| query anchors / cmd anchors 3×16 | `decoder/query_anchors/` | `SparseBox3DEncoder`(검출·자차), `SparsePoint3DEncoder`(지도·계획), `load_plan_anchors`, `load_motion_anchor`·`build_motion_anchor_encoder`, `ego_anchor` |
| noise t | `decoder/noise_t/` | `PlanTruncatedDiffusion` (잘린 확산 오염 + 시점 임베딩) |
| temporal bank | `decoder/temporal_bank/` | `InstanceBank`(검출·지도, 추적 id), `PlanningInstanceBank`, `EgoInstanceBank`, `topk` |
| temporal / self attention | `decoder/temporal_self_attn/` | `TemporalSeparateAttention`(temp_gnn), `SeparateAttention`(gnn), `InteractiveAttention`(inter_gnn), `MultiheadFlashAttention` |
| deformable aggregation | `decoder/deformable_aggregation/` | `DeformableFeatureAggregation`, `SparseBox3DKeyPointsGenerator`, `SparsePoint3DKeyPointsGenerator`, `AsymmetricFFN` |
| language fusion (gated cross-attn) | `decoder/language_fusion/` | `QwenTokenAttn` |
| split / refine | `decoder/split_refine/` | `SparseBox3DRefinementModule`, `SparsePoint3DRefinementModule`, `SparseMotionRefinementModule`, `SparsePlanAlignRefinementModule`, `EgoStatusRefinementModule`, `CustomOperation` |
| (출력 디스패치) | `head/sparse_head.py` | `SparseHead` |
| detection / tracking | `head/detection_tracking/` | `SparseBox3DDecoder`(instance_ids 출력), `SparseBox3DTarget`, `SparseBox3DLoss` |
| online map | `head/online_map/` | `SparsePoint3DDecoder`, `SparsePoint3DTarget`, `HungarianLinesAssigner`, `LinesL1Cost`, `MapQueriesCost`, `LinesL1Loss`, `SparseLineLoss` |
| motion / planning | `head/motion_planning/` | `SparseMotionDecoder`, `SparseMotionTarget`, `SparsePlanDecoder`, `SparsePlanTarget`, `AlignPlanTarget` |
| (전체 연결) | `detector.py` | `SparseDetector` |
| (그림 밖 공용) | `common/` | 박스 인덱스 상수, `box3d_to_corners*`, `linear_relu_ln`, `gen_sineembed_for_position`, 동기화 제거 도우미, `BaseTargetWithDenoising`(검출·지도 target 공용 베이스) |
| (별도 연구 트랙) | `residual_rl/` | Residual RL 안전제어 (`tools/residual_rl/` 가 경로로 적재) |

### 그림과 1:1 이 아닌 부분(코드 제약)

- **ResNet-50 / FPN** 은 mmdet 모듈이라 이 저장소에 코드가 없다(config `img_backbone`/`img_neck`). 폴더에는 GridMask 만 있다.
- **projector** 는 `build_qwen_projector()` 함수만 여기에 있고, 모듈 객체는 `QwenTokenAttn.qwen_token_proj` 로 language_fusion 이
  소유한다. 별도 nn.Module 로 감싸면 state_dict 키(`...qwen_token_attn.qwen_token_proj.0.weight` 등)가 바뀌기 때문이다.
- **VL tokens cache** 의 `LoadQwenCacheNusc` 는 실체가 `encoder/vl_tokens_cache/` 에 있고, `loader/pipelines/__init__.py` 가 다시 내보낸다
  (레지스트리 등록은 한 번 — 모듈이 한 번만 로드된다).
- **query anchors**: 앵커 값 적재 함수는 여기 있지만, 앵커 파라미터 자체는 종전과 같은 속성(`plan_instance_bank.anchor`,
  `ego_instance_bank.anchor`, `motion_anchor`, `motion_anchor_encoder`)으로 각 뱅크·디코더가 소유한다(키 불변).
  자차 앵커 임베딩은 검출 앵커 인코더를 공유한다(`ego_anchor_encoder is det_anchor_encoder`).
- **detection / tracking**: 추적 ID 는 `decoder/temporal_bank/instance_bank.py` 의 `get_instance_id` 가 만들고 검출 decoder 가 내보낸다.

## import 규칙 (중요)

```
HiP-AD/
├── models/                     ← 실체(여기를 편집한다)
├── mmdet3d_plugin/
│   ├── models -> ../models     ← 심볼릭 링크. 패키지 경로는 projects.mmdet3d_plugin.models.*
│   ├── datasets/__init__.py    ← models.loader 를 다시 내보내는 얇은 층(하위 호환)
│   ├── ops/                    ← deformable_aggregation CUDA op(.so)
│   └── apis/  core/            ← 학습 루프·평가 훅
└── projects/mmdet3d_plugin -> ../mmdet3d_plugin   (config 의 plugin_dir)
```

- `models/` 안에서는 **상대 import 만** 쓴다(절대형은 `projects.mmdet3d_plugin.*`/`mmdet3d_plugin.*` 두 이름으로 이중 로드 → 레지스트리 이중 등록).
- `models/` 는 단독 패키지로 import 할 수 없다(`from ....ops import` 로 상위 `mmdet3d_plugin.ops` 참조).

## 데이터 흐름(한 프레임)

```
loader(영상·GT, encoder/vl_tokens_cache: VL 토큰) → SparseDetector.extract_feat(ResNet-50→FPN [→ neck: DenseDepthNet 보조 손실])
 → SparseOneDecoder: temporal_bank.get → query_anchors 인코딩(+ noise_t) → concat → temporal_self_attn(temp_gnn/gnn/inter_gnn)
   → split → deformable_aggregation → concat → ffn → norm → language_fusion(QwenTokenAttn, projector) → split → split_refine × 6층
 → SparseHead.loss(head/*/target·loss) 또는 post_process(head/*/decoder)
```

## 이동 표 (정리 전 → 정리 후)

| 정리 전 | 정리 후 |
|---|---|
| `models/sparse_detector.py` | `models/detector.py` |
| `models/sparse_onedecoder.py` | `models/decoder/unified_decoder.py` |
| `models/sparse_head.py` | `models/head/sparse_head.py` |
| `models/grid_mask.py` | `models/encoder/image_resnet50_fpn/grid_mask.py` |
| `models/qwen_online.py`, `models/qwen_traj_generator.py` | `models/encoder/vlm_qwen2vl/` |
| `datasets/pipelines/transform.py` 의 `LoadQwenCacheNusc` | `models/encoder/vl_tokens_cache/loading.py` |
| `models/qwen_token_attn.py` (사영 MLP 구성) | `models/encoder/projector/qwen_projector.py` (함수) |
| `models/qwen_token_attn.py` | `models/decoder/language_fusion/qwen_token_attn.py` |
| `models/blocks.py` `DenseDepthNet` | `models/neck/multi_scale_depth/depth.py` |
| `models/blocks.py` `DeformableFeatureAggregation`·`AsymmetricFFN` | `models/decoder/deformable_aggregation/aggregation.py` |
| `models/blocks.py` `CustomOperation` | `models/decoder/split_refine/ops.py` |
| `models/blocks.py` `linear_relu_ln`, `models/attention.py` `gen_sineembed_for_position` | `models/common/layers.py` |
| `models/attention.py` | `models/decoder/temporal_self_attn/flash_attention.py` |
| `models/separate_attn.py` | `models/decoder/temporal_self_attn/separate_attn.py` |
| `models/instance_bank.py` | `models/decoder/temporal_bank/instance_bank.py` |
| `models/plan/instance_bank.py` | `models/decoder/temporal_bank/plan_bank.py` (앵커 적재 → `decoder/query_anchors/plan_anchor.py`) |
| `models/ego/instance_bank.py` | `models/decoder/temporal_bank/ego_bank.py` (앵커 값 → `decoder/query_anchors/ego_anchor.py`) |
| 3중 정의 `topk` | `models/decoder/temporal_bank/topk.py` |
| `models/det/blocks.py` `SparseBox3DEncoder` | `models/decoder/query_anchors/box3d_encoder.py` |
| `models/map/blocks.py` `SparsePoint3DEncoder` | `models/decoder/query_anchors/point3d_encoder.py` |
| `models/det/blocks.py`·`models/map/blocks.py` 키포인트 생성기 | `models/decoder/deformable_aggregation/aggregation.py` |
| `models/sparse_onedecoder.py` 모션 앵커·인코더 구성 | `models/decoder/query_anchors/motion_anchor.py` (함수) |
| `models/plan/diffusion.py` | `models/decoder/noise_t/diffusion.py` |
| `models/{det,map,motion,plan,ego}/blocks.py` refine 모듈 | `models/decoder/split_refine/{det,map,motion,plan,ego}_refine.py` |
| `models/det/{decoder,target,losses}.py` | `models/head/detection_tracking/` |
| `models/map/{decoder,target,loss,match_cost}.py` | `models/head/online_map/` |
| `models/motion/{decoder,target}.py` | `models/head/motion_planning/motion_{decoder,target}.py` |
| `models/plan/{decoder,target}.py` | `models/head/motion_planning/plan_{decoder,target}.py` |
| `models/base_target.py` | `models/common/base_target.py` |
| `models/sync_free.py`, `core/box3d.py`, `datasets/utils.py`(꼭짓점) | `models/common/{sync_free,box3d,box_ops}.py` (`core/box3d.py` 는 재노출) |
| `mmdet3d_plugin/datasets/{builder,pipelines,samplers,evaluation}` | `models/loader/{builder.py,pipelines,samplers,evaluation}` |
| `datasets/nuscenes_mini_dataset.py`, `track_relink.py`, `map_utils/` | `models/loader/nuscenes/{dataset.py,track_relink.py,map_utils/}` |
| `datasets/bench2drive_dataset.py`, `bench2drive_eval_utils.py` | `models/loader/bench2drive/{dataset.py,eval_utils.py}` |
| `datasets/navsim_dataset.py` | `models/loader/navsim/dataset.py` |

## 2026-10-15 정리에서 제거한 것(요약)

미사용 파일(`optimizer_dedup.py`, `det/det_head.py`(Sparse4DHead), `plan/loss.py`, `utils.py`, `core/hook/nan_diag_hook.py`,
`pipelines/qa_loading.py`, A800 전용 op), 미사용 클래스(`SparseEgoRefinementModule`, `KeyPoint3DEncoder`, `MotionTarget`,
`SparseBox3DMotionDecoder`, `HierarchicalPlanningDecoder`, `SparsePlanRefinementModule`, `PlanningTarget`, `Grid`),
어떤 config 도 켜지 않던 분기(scenes 토큰, attention mask, 점 단위 지도·계획 쿼리, 외부 궤적·계획 전용 Qwen 어텐션, top-k 계획 모드,
DN 노이즈 쿼리, 보조 Qwen 캐시 혼합, 단층 사영·단일 구간 게이트 등). 제거된 기능을 config 로 켜면 **조용히 무시하지 않고 즉시 실패**한다.
원본은 `archive/refactor_20261015/`(삭제 파일 + 정리 직전 전체 스냅샷)에 있다.

## 2026-10-15 실험 B 융합(카메라+LiDAR / 카메라+Radar, 언어 없음) 추가분

수정(2026-10-15): B 융합. 새 파일 위주, 공유 파일은 config 로 켤 때만 동작(기본값이면 종전과 state_dict·순전파 동일 — 기존 18개 config 시그니처·초기값 해시 불변 확인).

| 상자 | 위치 | 내용 |
|---|---|---|
| LiDAR 인코더 | `encoder/lidar_pointpillars/pillar_bev.py` | `PillarBEVEncoder`: 동적 기둥(절단 없음) + PFN + 2D 백본(k2s2 다운샘플) + FPN 식 출력 |
| Radar 인코더 | `encoder/radar_pillars/` | `RadarPillarBEVEncoder`(같은 구조, 7열 입력·0.8 m 기둥) |
| 점 읽기 | `loader/pipelines/loading_points.py` | `LoadPointsFromMultiSweepsFusion`, `LoadRadarPointsMultiSweeps`, `PointsRangeFilterFusion`, `FusionPointsToTensor` |
| 데이터셋 | `loader/nuscenes/fusion_dataset.py` | `NuScenes3DFusionDataset` (info 의 sweeps/radars 전달) |
| BEV 표본추출 | `decoder/deformable_aggregation/aggregation.py` | `bev_num_levels>0` 일 때만: 키포인트 (x,y) → BEV bilinear, 카메라와 공동 softmax |
| 배선 | `detector.py` | `bev_encoders` cfg 가 있을 때만 `data['fusion_bev_feats']` 생성 |
| info 생성 | `tools/data_converter/nuscenes_fusion_infos.py` | `data/infos/nuscenes_fusion_infos_{train,val}.pkl` |
| config/스크립트 | `configs/nuscenes/fusion_{lidar,radar}/stage{1,2}/`, `scripts/nuscenes/run_cam_{lidar,radar}.sh` | |
