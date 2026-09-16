# 수정(2026-09-07): Run7 — 검출 앵커에 크기 사전정보 부여. Run1(896x448) 위에 얹는다.
#
#   [진단] mAP 격차의 대부분이 trailer(AP 0.066)·construction_vehicle(0.065)에서 나오고,
#   실체는 오분류가 아니라 **순수 미검출**이다 — trailer GT 의 92.6%가 2m 내 예측조차 없고
#   truck 으로 오인한 것은 0.4%뿐이다. 거리별로 보면 trailer 는 0-10m 에서 92.3% 잡다가
#   10-20m 에서 23.7%로 급락한다(car 98.8->93.1, bus 98.1->82.4 와 대조되는 절벽).
#   기각한 가설: GT 결손 / 단순 희소성 / 손실 불균형(가중치 실험에서 오히려 악화) /
#   valid_flag 학습 필터링(trailer 3.6% 제외 vs car 11.3% 제외인데 car 는 정상).
#
#   [구조적 원인] 앵커 900개의 크기 열이 전부 1.0(log) = **exp(1)=2.72m 정육면체**다.
#   models/det/blocks.py:192 에서 deformable 키포인트가 `anchor[[W,L,H]].exp()` 로
#   스케일되므로, 초기 샘플링 발자국이 대상 크기와 무관하게 2.72m 로 고정된다.
#   13.24m 짜리 trailer 를 2.72m 발자국으로 찾는 셈이다. 가까이서는 물체가 화면을 채워
#   걸리지만 10m 를 넘으면 판별 부위를 놓친다 — 관측된 절벽과 일치한다.
#   실제로 100+12 epoch 학습 후에도 앵커 크기는 1.77~3.71m 대역을 벗어나지 못했다.
#
#   [수정] full train GT(710,471 박스, 50m 이내, valid_flag 통과)에서 클래스별로 앵커를
#   배분해(빈도의 제곱근 비례, 최소 20개) 각 클러스터의 중앙 크기를 부여했다.
#   결과: 길이 최대 16.38m, >12m 앵커 67개, >4m 앵커 432개. 기존은 >4m 가 0개였다.
#   -> data/kmeans/nusc_det_900_cls.npy
#
#   [주의] 앵커는 nn.Parameter 라 config 의 파일 경로만 바꾸면 체크포인트의 학습된 옛 값이
#   덮어쓴다. 그래서 체크포인트 안의 anchor 텐서를 직접 교체한 사본에서 출발한다
#   (latest_newanchor.pth, 앵커 외 텐서 변경 0 검증 완료). 옵티마이저 상태는 제거했다 —
#   앵커가 바뀌면 그에 대한 모멘텀은 무의미하다.
#
#   [바꾸는 것은 앵커 하나뿐] 해상도·배치·lr·스케줄·증강 전부 Run1 과 동일.
# 수정(2026-08-26): 고해상도 fine-tune — HiP-AD Table 4(det mAP 0.424) 초과가 목표.
#
#   [왜 해상도인가] 모델 구조가 HiP-AD 원본과 완전 일치함을 확인했다(det 앵커 900 /
#   temporal 전파 600 / confidence_decay 0.6 / cls_threshold_to_reg 0.05 /
#   single_frame_decoder 1 / map 앵커 100). 구조적 어긋남이 없으므로 남은 지렛대는 입력이다.
#   우리 자체 실측도 같은 방향이다 — 640x352(225k px) 0.3996 vs 704x256(180k px) 0.3867.
#
#   [설정] 704x256(180k) -> 896x448(401k, 2.23배). 세로가 256->448 로 늘어 원본 종횡비
#   (1600x900=1.78:1)에 가까워진다(2.75:1 -> 2:1). 세로 크롭이 줄어 근거리·대형 객체에 유리하다.
#   처음부터 재학습(10일)하지 않고 100epoch 체크포인트에서 이어 학습한다 —
#   해상도 전환 fine-tune 은 표준 기법이고 이득의 대부분을 며칠에 얻는다.
#
#   896x448 은 임의값이 아니다 — BEVCar 등 최근 nuScenes 카메라 모델이 쓰는 해상도와 같아
#   비교·재현 시 참조점이 있다.
#
#   [주의] HiP-AD 의 0.424 는 704x256 값이다. 논문에는 두 해상도 행을 모두 싣고
#   설정 차이를 명시할 것. 숨기면 안 된다.
#
#   [배치] 픽셀 2.23배 -> per-GPU batch 8 -> 4, 누적 8 -> 16 으로 유효 배치 64 유지.
#          VRAM 추정 약 31GB (현재 28GB@batch8@180k px 기준). 상한 70GB 이내.
#   [lr] 2e-4 -> 2e-5. 100epoch 코사인으로 수렴한 지점을 파괴하지 않기 위함
#        (2026-08-11 stage2 인지 하락의 확정 원인이 lr 2000배 재시작이었다).
#
#   [이번 run 에서 바꾸는 것은 해상도 하나뿐이다] temporal map 확장 등 다른 후보는
#   이 run 의 결과를 본 뒤 별도 config 로 시험한다 — 두 변경을 섞으면 귀속이 불가능해진다.
# 수정(2026-08-11): ReLaQ-AD 언어-쿼리 융합(VLA) 적용 stage1.
#   hipad_nusc_full_stage1.py (융합 OFF 기반) + Qwen2-VL-2B 토큰 cross-attn.
#
#   [무엇을 하는가] Qwen2-VL-2B 가 6방향 카메라 영상을 읽고 낸 hidden token 열
#   (L=1159, 1536dim, bf16)을 K/V 로, 통합 디코더의 과제 쿼리를 Q 로 두어 cross-attn 하고
#   0 초기화 게이트를 통해 잔차로 더한다. Qwen 본체는 forward 하지 않는다 — 전 표본의
#   hidden 을 data/nuscenes/qwen 에 미리 캐시해 두었다(train 28130 + val 6019 = 34149, 누락 0).
#
#   [삽입 위치] deformable(영상 특징 집약) 직후, ffn 앞.
#   시각적으로 접지된 특징을 언어가 변조하는 순서이며, DETR 계열의
#   self-attn -> image cross-attn -> text cross-attn -> FFN 관례와 일치한다.
#   instance_feature 가 통합 쿼리인 구간(concat~split)이어야 모든 과제 쿼리가 함께 융합된다.
#
#   [0 초기화 게이트] models/qwen_token_attn.py 의 gate 를 0 으로 시작한다. step 0 의 출력이
#   융합 OFF 모델과 정확히 동일하므로 (a) 사전학습 표현을 초기에 교란하지 않고
#   (b) ON/OFF 차이를 융합 효과로 귀속할 수 있다. stage2 에서 모션을 전강도로 주입했다가
#   추적 AMOTA 가 23.7% 떨어진 전례가 이 설계의 근거다.
#
#   [데이터] full only — train 28,130 / val 6,019. 축소본 사용 금지(사용자 지시).
# 수정(2026-08-11): nuScenes stage1 을 HiP-AD 원본 커리큘럼으로 정렬.
#
#   [문제] 기존 hipad_trainval_nusc_stage1.py 는 task_select=["det","map"] 이라
#          계획·자차 쿼리가 공유 디코더의 쿼리 풀에서 아예 빠져 있었다. 손실이 0 이어도
#          쿼리가 존재하면 GNN·어텐션 단계에서 인지 쿼리와 상호작용하므로 구조가 다르다.
#          HiP-AD 의 핵심 기여가 "모든 과제 쿼리가 하나의 디코더를 공유"하는 것인데
#          그 부분이 비활성이었고, 실제 성능도 SparseDrive-S 근방(-2~4%)에 머물렀다.
#          (실측 stage1: mAP 0.3996 / NDS 0.5046 / map 0.5399 / AMOTA 0.3521,
#           HiP-AD Table 4 대비 -5.8% / -5.7% / -5.4% / -13.3%)
#
#   [수정] 상위 configs/hipad_base_b2d_stage1.py 와 동일하게
#            task_select = ["det","map","plan","ego"], 계획·자차 손실 가중치 0.0
#          로 두어 순전파만 시킨다. 손실은 stage2 에서 켠다(ego 1.0/plan_cls 0.5/plan_reg 1.0).
#
#   [nuScenes 전용화] B2D(CARLA) 잔재를 제거한다:
#            - plan_anchor_paths : b2d_plan_spat_6x8_5m.npy -> nusc_plan_temp_48.npy
#              (tools/kmeans/kmeans_nusc_plan.py 로 nuScenes 자차 궤적에서 생성.
#               48모드, 종점 0.06~51.8m, b2d 는 CARLA 분포라 최대 30m 로 달랐다)
#            - ego_instance_bank.anchor_type : "b2d" -> "nus"
#              (models/ego/instance_bank.py 에 nus 분기 존재: 자차 4.08x1.73x1.56 vs
#               CARLA 4.89x1.84x1.49)
#            - plan_instance_bank.ego_vehicle : "b2d" -> "nus"
#
#   [평가] nuScenes 평가는 det/map/track/motion 만 한다(사용자 방침). 계획은
#          Bench2Drive / nuPlan / NAVSIM 에서 평가한다. 여기서 계획을 켜는 것은
#          쿼리 풀 구성을 HiP-AD 와 일치시키기 위한 것이지 계획 지표 산출이 목적이 아니다.
#
#   나머지(해상도 704x256, lr 2e-4, 100ep, 유효배치 64, fp32)는 원본과 동일.
log_level = "INFO"
dist_params = dict(backend="nccl")

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

num_gpus = 1  #수정: nuScenes full 멀티태스크, GPU0 단독
batch_size = 4  #수정(hires): 원래 8. 픽셀 2.23배라 절반. #원: per-GPU micro-batch (deformable 커널 한계 내 최대, mem 31GB)
#수정(2026-08-03): effective batch 48 → 64. 근거 = 상류 config 실측 대조.
#   HiP-AD 공개 stage1(projects/configs/hipad_b2d_stage1.py): 8gpu × batch 8 = total 64
#   SparseDrive nuScenes stage1(sparsedrive_small_stage1.py): total_batch_size = 64
#   두 레퍼런스가 모두 64 로 일치 → 48(이전 가정 "8gpu*6")은 근거 없는 값이었다.
#   1 GPU 이므로 micro-batch 8 × grad accumulation 8 = 64 로 재현한다.
cumulative_iters = 16  # effective batch = 1*4*16 = 64 (유지)
# num_iters_per_epoch = int(234769 // (num_gpus * batch_size))
num_iters_per_epoch = int(28130 // (num_gpus * batch_size))  #수정: full nuScenes (micro-iter 기준)
num_epochs = 12  #수정(hires): fine-tune. 원래 100: HiP-AD/SparseDrive nuScenes stage1 표준(=100ep), 비교 타당성 위해 일치
#checkpoint_epoch_interval = 20
checkpoint_epoch_interval = 1

checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)
load_from = "./results/hipad_nusc_vla_hires/latest_newanchor.pth"  #수정(Run7): Run1 + 새 앵커
resume_from = None
workflow = [("train", 1)]
find_unused_parameters = True  #수정: 미사용 plan/ego 모듈(빌드만)용
fp16 = None  #수정: fp32 고정(NaN 원천차단, 프로젝트 규칙)
#수정(2026-08-03): (640,352) → (704,256). HiP-AD 논문 Appendix B 가 nuScenes 는
#   "the resolution is changed to 704 × 256" 이라고 명시하고, SparseDrive nuScenes config 도 동일.
#   (640,352)는 Bench2Drive 해상도를 그대로 들고 온 값이라 Table 4 비교가 성립하지 않았다.
#   data_aug_conf 의 나머지 항목은 이미 SparseDrive 와 동일 → 이 한 줄로 정렬 완료.
input_shape = (896, 448)  #수정(hires): 180k -> 401k px
num_cams = 6

# det & map
det_class_names = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]  #수정: nuScenes 10 det class
map_class_names = ["ped_crossing", "divider", "boundary"]  #수정: nuScenes 3 map class

num_det_classes = len(det_class_names)
num_map_classes = len(map_class_names)

map_roi_size = (30, 60)
map_num_pts = 20

# traj
fut_ts = 6
fut_mode = 6
#수정(2026-08-11): 자차 상태 차원. B2D 는 6(속도,가속2,각속2,조향), nuScenes 는 10
#   (가속3, 각속3, 속도3, 조향1). EgoStatusRefinementModule 기본값이 6 이라 그대로 두면
#   loss_ego 에서 pred(6) vs target(10) 로 assert 실패한다.
ego_status_dims = 10
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
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]  #수정: nuScenes surround range

# temporal
temporal = True
temporal_det = True
temporal_map = True
temporal_ego = True
temporal_plan = True

# tasks
task_config = dict(with_onedecoder=True)

task_select = ["det", "map", "plan", "ego"]  #수정: HiP-AD 원본 커리큘럼과 일치(손실은 0.0, 순전파만)
query_select = ["det", "map", "plan", "ego"]  #수정: 쿼리 풀도 HiP-AD 와 일치 — 계획·자차 쿼리가 공유 디코더에 존재해야 인지 쿼리와 상호작용한다

#수정: qwen_attn 은 ffn+norm 뒤, split 앞에 넣는다.
#   concat 직후 구간은 Sparse4D 의 fc_before 로 특징이 2*embed_dims(512)로 확장된 상태라
#   256 기준으로 만든 cross-attn 을 붙일 수 없다(실측: mat 8392x512 vs 256x256 불일치).
#   ffn+norm 을 거치면 256 으로 복원되고, split 전이므로 여전히 통합 쿼리다 —
#   영상 특징 집약(deformable) 이후이자 과제별 분기(split) 이전이라
#   "시각 접지 -> 언어 변조 -> 과제별 정제" 순서가 유지된다.
single_frame_layer = ["concat", "gnn", "inter_gnn", "norm", "split", "deformable", "concat", "ffn", "norm", "qwen_attn", "split", "refine"]
temporal_frame_layer = ["concat", "temp_gnn", "gnn", "inter_gnn", "norm", "split", "deformable", "concat", "ffn", "norm", "qwen_attn", "split", "refine"]

operation_order = single_frame_layer * num_single_frame_decoder + \
                  temporal_frame_layer * (num_decoder - num_single_frame_decoder)

# anchors
project_dir = "/workspace/src/HiP-AD"

anchor_paths = {
    #수정(Run7): 크기 사전정보를 담은 앵커(길이 최대 16.4m, >4m 432개). 기존은 전부 2.72m.
    "det" : f"{project_dir}/data/kmeans/nusc_det_900_cls.npy",
    "map" : f"{project_dir}/data/kmeans/nusc_map_100.npy",
    "motion": f"{project_dir}/data/kmeans/nusc_motion_{fut_mode}.npy",  #수정: CARLA 앵커 -> nuScenes 앵커(stage1 에선 미사용이나 stage2 와 일관)
}

plan_anchor_paths = f"{project_dir}/data/kmeans/nusc_plan_temp_48.npy"  #수정: CARLA 앵커 -> nuScenes 앵커
plan_anchor_refer = ("temp", "2hz")
plan_anchor_types = [("temp", "2hz")]


#수정: Qwen hidden 캐시 경로. 지정 시 sparse_detector 가 배치 토큰으로 캐시를 읽어
#   metas['qwen_hidden'] / metas['qwen_mask'] 로 주입한다. 누락 시 조용히 넘기지 않고 실패한다.
qwen_cache_dir = "data/nuscenes/qwen"

model = dict(
    type="SparseDetector",
    qwen_cache_dir=qwen_cache_dir,  #수정: VLA 캐시 배선
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=False,  #수정: DDP+find_unused 충돌 회피(메모리 여유)
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
        evaluate_bench2dive=False,  #수정: nuScenes det eval
        onedecoder_head=dict(
            type="SparseOneDecoder",
            task_select=task_select,
            query_select=query_select,
            operation_order=operation_order,
            #수정: 언어-쿼리 융합 모듈. 전 디코더 계층이 이 단일 모듈을 공유한다(파라미터 공유).
            qwen_token_attn=dict(
                type="QwenTokenAttn",
                embed_dims=embed_dims,
                num_heads=8,
                qwen_hidden_dim=1536,   # Qwen2-VL-2B hidden dim
                gate_init=0.0,          # 0 초기화 -> step 0 출력이 융합 OFF 와 동일
            ),
            num_single_frame_decoder=num_single_frame_decoder,
            plan_anchor_refer=plan_anchor_refer,
            with_command_embed=False,  #수정
            with_target_point_embed=False,  #수정
            with_supervise_ego_status=True,  #수정: 상위 HiP-AD 와 일치. 이전 False 는 shape 충돌 회피용 우회였다
            with_ego_instance_feature=True,  #수정: 상위 HiP-AD 와 일치
            with_incremental_plan_refine=False,  #수정
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
                zero_velocity_classes=["barrier", "traffic_cone"],  #수정: nuScenes 정적 클래스
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
                anchor_type="nus",  #수정: nuScenes 자차 제원
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
                query_list=[["det"], ["map"]],  #수정: det+map
                key_list=[["det"], ["map"]],  #수정
                decouple_list=[True, False],  #수정
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
            inter_graph_model=None,  #수정: plan/ego 없으면 불필요
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
                status_dims=ego_status_dims,  #수정: 기본 6(B2D) -> nuScenes 10. 이전에는
                #   차원 불일치를 with_supervise_ego_status=False 로 우회하고 있었다.
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
                              loss_yawness=dict(type="GaussianFocalLoss")),
            loss_map_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_map_reg=dict(type="SparseLineLoss",
                              loss_line=dict(type="LinesL1Loss", loss_weight=10.0, beta=0.01),
                              num_sample=map_num_pts,
                              roi_size=map_roi_size),
            loss_ego_status=dict(type="L1Loss", loss_weight=0.0),
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
                              ego_fut_mode=ego_fut_mode, ego_vehicle="nus", anchor_types=plan_anchor_types,
                              anchor_refer=plan_anchor_refer, with_rescore=True),
            motion_decoder=dict(type="SparseMotionDecoder"),
        ),
    ),
)

# ================== data ========================
dataset_type = "NuScenesMiniDataset"
data_root = "data/nuscenes"
anno_root = "data/infos/"
file_client_args = dict(backend="disk")

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),  #수정: depth aux 용 lidar
    dict(type="ResizeCropFlipImage"),
    dict(type="MultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),  #수정: depth GT 생성
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=map_roi_size, simplify=False, normalize=False,
         sample_num=map_num_pts, permute=True),  #수정: permute=True → gt_map_labels/gt_map_pts
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "ego_status", "ego_status_mask", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               #수정: 계획 쿼리 학습용 자차 궤적 GT(2Hz 접미사 키)
               "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token"],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "gt_bboxes_3d", "gt_labels_3d", "timestamp", "projection_mat", "image_wh", "ego_status"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "scene_token", "token"],
    ),
]

eval_pipeline = [
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=map_roi_size, simplify=True, normalize=False),
    dict(type="Collect",
         keys=["vectors", "gt_bboxes_3d", "gt_labels_3d"],
         meta_keys=["token", "timestamp"]),
]

input_modality = dict(use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=False)

data_basic_config = dict(
    type=dataset_type, data_root=data_root,
    det_classes=det_class_names, map_classes=map_class_names,
    plan_anchor_types=plan_anchor_types, modality=input_modality,
)

data_aug_conf = {
    #수정(hires): final_dim 폭 704->896(x1.273)에 비례해 리사이즈 비율도 상향
    "resize_lim": (0.51, 0.60), "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0), "rot_lim": (-5.4, 5.4),
    "H": 900, "W": 1600, "rand_flip": True, "rot3d_range": [0, 0],
}

data = dict(
    samples_per_gpu=batch_size, workers_per_gpu=batch_size,
    train=dict(**data_basic_config, ann_file=anno_root + "nuscenes_infos_train.pkl",
               pipeline=train_pipeline, test_mode=False, data_aug_conf=data_aug_conf,
               with_seq_flag=True, sequences_split_num=2, keep_consistent_seq_aug=True),
    val=dict(**data_basic_config, ann_file=anno_root + "nuscenes_infos_val.pkl",
             pipeline=test_pipeline, data_aug_conf=data_aug_conf, test_mode=True),
    test=dict(**data_basic_config, ann_file=anno_root + "nuscenes_infos_val.pkl",
              pipeline=test_pipeline, data_aug_conf=data_aug_conf, test_mode=True),
)

# ================== training ========================
optimizer = dict(
    type="AdamW",
    lr=2e-5, #수정(hires): 수렴점 보존. 원래 2e-4: HiP-AD upstream 값으로 복원(2.5e-5 는 fp16 NaN 회피 핵). effective batch 48 에 맞는 lr
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)
optimizer_config = dict(  #수정: grad accumulation 으로 effective batch 48 재현
    type="GradientCumulativeOptimizerHook",
    cumulative_iters=cumulative_iters,
    grad_clip=dict(max_norm=25, norm_type=2),
)
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500 * cumulative_iters,  #수정: warmup 을 optim-step 기준(=500)으로 유지(micro-iter=3000)
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)

# ================== eval ========================
#수정(2026-08-03): HiP-AD Table 4(검출·매핑·추적·모션) 산출용으로 정정.
#  기존 값은 Bench2Drive config 를 그대로 복사한 것이라 det/map 이 전부 False 이고
#  이 config 에 존재하지도 않는 planning 만 True 였다 → stage1 을 평가해도 아무 지표도 안 나온다.
#  stage1 task_select = ["det","map"] 이므로 motion 은 False(stage2 에서 True).
#  tracking 은 학습 태스크가 아니라 temporal instance bank 의 ID 를 그대로 track ID 로 쓰는
#  후처리 평가이므로 stage1 에서도 산출 가능하다.
eval_mode = dict(
    with_det=True,
    with_tracking=True,
    with_map=True,
    with_motion=False,
    with_planning=False,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)
#수정(2026-08-03): map 평가기(VectorEvaluate)는 GT map vector 를 예측 결과가 아니라
#  별도 dataset(eval_pipeline)에서 다시 로드한다 → 그 dataset cfg 를 여기서 준다.
#  SparseDrive sparsedrive_small_stage1.py 의 eval_config 와 동일 구성.
eval_config = dict(
    **data_basic_config,
    ann_file=anno_root + "nuscenes_infos_val.pkl",
    pipeline=eval_pipeline,
    test_mode=True,
)
evaluation = dict(
    interval=num_iters_per_epoch*checkpoint_epoch_interval,
    jsonfile_prefix="val/",
    eval_mode=eval_mode,
    eval_config=eval_config,
)