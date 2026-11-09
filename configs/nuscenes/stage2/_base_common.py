# 수정(2026-10-16): v1 실험 config 제거(사용자 지시) — 이 파일은 v2 학습 config 의 공통 바탕일 뿐 단독 실행 금지
#   (종전 이름 nuscenes/stage2/train.py = v1 학습 config. v1 결과 기록용 짝 config 는 configs/archive/nuscenes_v1_20261016/stage2/)
# 수정(2026-10-15): 경로 이동 configs/nuscenes/hipad_nusc_laq_stage2_train.py → configs/nuscenes/stage2/train.py
#   run_name: 결과 디렉터리 이름(results/<run_name>, evaluation/<run_name>). tools/train.py·test.py 가 --work-dir 미지정 시 쓴다.
# 수정(2026-10-10): 통합 실험(사용자 결정) — 704 1단계 100ep 위의 2단계에 planfix(명령별 모드·계획 lr×10, 이미 반영)와
#   plandiff(언어 조건 잘린 확산 계획, models/plan/diffusion.py)를 함께 넣는다. 설정값은 plandiff config 와 같다.
# 수정(2026-10-09): 벤치마크·stage·학습/테스트 단위 config 정리(사용자 지시).
#   원본: configs/hipad_nusc_vla_s2r9.py 를 복사해 만든 학습용 config. 수정(2026-10-09): 입력 해상도를 704x256 으로 복원했으므로 Run9/Run10(896x448) 과는 다른 새 실험이다. = Run10(2단계, 논문 대표모델). load_from 은 Run9 체크포인트 경로 그대로.
#   테스트용 짝 config 는 같은 폴더의 *_test.py (_base_ 로 이 파일을 상속, 평가 설정만 덮어씀).
# 수정(2026-10-08): Run10 — Run9(s1hr, 896x448 100ep stage1) 위에 stage2(+motion, plan/ego 손실)를 얹는다.
#   기반: hipad_nusc_vla_s1hr.py 를 복사. stage2 레시피(lr 보호·모션 가중치)는 hipad_nusc_vla_s2hr.py 와 동일.
#   [s2hr 대비 추가로 바로잡은 것 — 6차 재검증(2026-10-05) 결함]
#   (1) inter_graph_model 복원: 상위 hipad_base_b2d_stage1.py 의 InteractiveAttention
#       (query=plan/ego, key=det/map). 기존 None 은 "plan/ego 없으면 불필요" 라는 stale 판단이었다.
#   (2) temp_graph_model 의 3번째 그룹(plan/ego -> 시간 det/map) 복원 — 상위와 동일.
#       attn 3번째 블록은 Run9 에도 빌드돼 있었으나 query_list 가 2그룹이라 한 번도 쓰이지 않았다(초기값 그대로).
#       (1)(2) 가 없으면 계획·자차 쿼리가 검출·지도를 전혀 보지 못한 채 학습된다.
#   (3) barrier 를 cls_allow_reverse 로 지정 — 평가가 barrier yaw 를 mod pi 로 본다.
#   코드 수정(모든 config 에 적용): motion 손실의 계층별 det 매칭, cns 타깃 detach,
#   map 뱅크 dt 가 det dt 를 덮어쓰던 문제, ego_status_mask 의 정차 제외.
#   (1)(2) 의 attention 은 사실상 미학습이므로 신규 모듈과 같이 lr_mult 10 을 준다.
# 수정(2026-09-15): Run9 — 896x448 에서 stage1 을 **처음부터 100 epoch** 학습.
#
#   [왜] 기존 896x448 모델(Run1)은 704x256 으로 100 epoch 학습한 가중치를 12 epoch
#   미세조정한 것이다. 즉 고해상도의 이득을 12 epoch 어치만 취한 상태다. 기준 문헌들은
#   자기 해상도에서 100 epoch 을 온전히 학습한다. 동일 예산으로 처음부터 학습하면
#   미세조정으로는 닿지 못하는 지점에 갈 수 있다.
#
#   [배치] 4 -> 8, 누적 16 -> 8 로 유효 배치 64 를 유지한다. VRAM 추정 48GB(상한 70GB 이내,
#   스모크로 실측 확인할 것). 총 iter 는 703,200 -> 351,600 으로 줄지만 표본 수는 같으므로
#   벽시계는 약 13.6일로 비슷하다. 배치를 키우는 이득은 속도가 아니라 누적 스텝 감소에 따른
#   옵티마이저 오버헤드 감소와 기울기 잡음 감소다.
#
#   [논문 기재 의무] 이 모델을 논문에 쓰려면 실험 설정 절에 입력 크기를 **반드시 명시**해야
#   한다. 기여를 해상도로 주장하지 않는 것과, 설정을 숨기는 것은 다르다. 후자는 심사에서
#   반드시 지적된다. 소거 실험(핵심 기여)은 파이프라인 내부 비교라 해상도와 무관하게 유효하다.
#
#   [후속] 이 모델이 대표 모델이 되면 소거 실험 3종(게이트=0 / 토큰 셔플)을 이 체크포인트에서
#   다시 산출해야 표가 일관된다(평가 3회 x 약 2시간).
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
batch_size = 8  #수정(2026-10-10): 속도 — GPU0 단독·VRAM 상한 90GB(사용자), 실측 표본당 시간 최소. eff 48 유지. #이전 #수정(Run10): stage2 는 motion 추가로 무거움(s2hr 와 동일). #이전(Run9)=8: 896x448 에서도 8 유지(VRAM 추정 48GB). #원: per-GPU micro-batch (deformable 커널 한계 내 최대, mem 31GB)
#수정(2026-08-03): effective batch 48 → 64. 근거 = 상류 config 실측 대조.
#   HiP-AD 공개 stage1(projects/configs/hipad_b2d_stage1.py): 8gpu × batch 8 = total 64
#   SparseDrive nuScenes stage1(sparsedrive_small_stage1.py): total_batch_size = 64
#   두 레퍼런스가 모두 64 로 일치 → 48(이전 가정 "8gpu*6")은 근거 없는 값이었다.
#   1 GPU 이므로 micro-batch 8 × grad accumulation 8 = 64 로 재현한다.
cumulative_iters = 6  #수정(2026-10-10): eff 48 = 8×6. #이전 #수정(Run10): effective batch = 1*4*12 = 48 (상위 stage2 표준, s2hr 와 동일)
# num_iters_per_epoch = int(234769 // (num_gpus * batch_size))
num_iters_per_epoch = int(28130 // (num_gpus * batch_size))  #수정: full nuScenes (micro-iter 기준)
cudnn_benchmark = True  # 수정(2026-10-10): 입력 크기 고정 → 합성곱 알고리즘 자동선택(수치 레시피 무관)
num_epochs = 10  #수정(Run10): HiP-AD/SparseDrive nuScenes stage2 표준(=10ep, load_from stage1)
#checkpoint_epoch_interval = 20
checkpoint_epoch_interval = 1

checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)
load_from = "./results/hipad_nusc_laq_stage1/latest.pth"  #수정(2026-10-09): 704x256 1단계(새 규약 경로). #이전(Run10) Run9(896x448) iter_351600
resume_from = None
workflow = [("train", 1)]
find_unused_parameters = True  #수정: 미사용 plan/ego 모듈(빌드만)용
fp16 = None  #수정: fp32 고정(NaN 원천차단, 프로젝트 규칙)
#수정(2026-08-03): (640,352) → (704,256). HiP-AD 논문 Appendix B 가 nuScenes 는
#   "the resolution is changed to 704 × 256" 이라고 명시하고, SparseDrive nuScenes config 도 동일.
#   (640,352)는 Bench2Drive 해상도를 그대로 들고 온 값이라 Table 4 비교가 성립하지 않았다.
#   data_aug_conf 의 나머지 항목은 이미 SparseDrive 와 동일 → 이 한 줄로 정렬 완료.
input_shape = (704, 256)  #수정(2026-10-09): 해상도 HiP-AD nuScenes 원래 값으로 복원(사용자 지시: 데이터셋별 원래 해상도). #이전(Run9/10) (896, 448)
num_cams = 6

# det & map
det_class_names = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]  #수정: nuScenes 10 det class
map_class_names = ["ped_crossing", "divider", "boundary"]  #수정: nuScenes 3 map class

num_det_classes = len(det_class_names)
num_map_classes = len(map_class_names)

map_roi_size = (30, 60)
map_num_pts = 20

# traj
fut_ts = 12  #수정(Run10): nuScenes agent 모션 GT 지평(2Hz x 12스텝 = 6초)
fut_mode = 6
#수정(2026-08-11): 자차 상태 차원. B2D 는 6(속도,가속2,각속2,조향), nuScenes 는 10
#   (가속3, 각속3, 속도3, 조향1). EgoStatusRefinementModule 기본값이 6 이라 그대로 두면
#   loss_ego 에서 pred(6) vs target(10) 로 assert 실패한다.
ego_status_dims = 10
ego_fut_ts = 6
ego_fut_cmd = 3  #수정(2026-10-09, 2번안): 명령별 계획 모드(우·좌·직진), 추론 시 GT 명령 선택 — SparseDrive 동일
ego_fut_mode = 16  #수정(2026-10-09, 2번안): 명령당 16 × 3 = 48

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

task_select = ["det", "map", "plan", "ego", "motion"]  #수정(Run10): stage2 = stage1 + motion
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
project_dir = "/home1/irteam/data-vol1/src/HiP-AD"

anchor_paths = {
    "det" : f"{project_dir}/data/kmeans/nusc_det_900.npy",
    "map" : f"{project_dir}/data/kmeans/nusc_map_100.npy",
    "motion": f"{project_dir}/data/kmeans/nusc_motion_{fut_mode}.npy",  #수정: CARLA 앵커 -> nuScenes 앵커(stage1 에선 미사용이나 stage2 와 일관)
}

plan_anchor_paths = f"{project_dir}/data/kmeans/nusc_plan_temp_3x16.npy"  #수정(2026-10-09, 2번안): 명령별 앵커  #수정: CARLA 앵커 -> nuScenes 앵커
plan_anchor_refer = ("temp", "2hz")
plan_anchor_types = [("temp", "2hz")]


#수정: Qwen hidden 캐시 경로. 지정 시 sparse_detector 가 배치 토큰으로 캐시를 읽어
#   metas['qwen_hidden'] / metas['qwen_mask'] 로 주입한다. 누락 시 조용히 넘기지 않고 실패한다.
#수정(2026-09-15, Run9): 마지막 계층 -> 18층 캐시로 전환.
#  근거 — VLM 내부 분석(arXiv 2606.06890): 시각적으로 접지된 신호는 중간 계층에서 가장
#  강하고 마지막 계층에서는 언어 prior 에 눌린다(28계층 모델 기준 14~19층 부근).
#  Qwen2-VL-2B 도 28계층이므로 18층(64% 깊이)을 주력으로 쓴다. 14/22층도 같은 순전파에서
#  함께 저장해 두어(비용 0) 추후 계층 ablation 이 가능하다.
qwen_cache_dir = "data/nuscenes/qwen_l18"

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
        with_cp=False,  #수정(Run10): batch 4 라 VRAM 여유(s2hr 와 동일). with_cp=True 는 BN running stats 를 iter 당 2회 갱신한다.
        #이전(Run9) with_cp=True:  #수정(2026-10-02): GPU1 50GB 상한 — 활성화 체크포인팅으로 VRAM 절감.
                        # 단일 프로세스 학습(distributed: False)이라 과거의 DDP+find_unused 충돌은 해당 없음.
                        # 순전파·역전파 수학은 동일하므로 iter 158220 에서의 재개 유효성 불변.
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
                #수정(2026-09-15, Run9): 언어 융합 확장 3종. 전부 0 초기화 게이트 뒤에
                #  있으므로 step 0 출력은 여전히 기반 모델과 동일하다.
                num_layers=6,           # (A) 계층별 게이트 — 디코더 6계층 각각 독립
                seg_bounds=[900, 1000], # (B) 과제별 게이트 — [0,900)=det, [900,1000)=map, 나머지=plan/ego
                proj_hidden=512,        # (C) K/V 사영 1536->512->256 2층 MLP(GELU)
                #수정(2026-10-12): plan/ego 구간(seg 2) 게이트 분리 — 1단계 손실 0 이라 정확히 0 으로 들어온다
                #  (iter 75,594 체크포인트 실측). 아래 optimizer 에서 다른 계획 모듈과 같은 lr x10.
                sep_gate_segs=[2],
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
            # 수정(2026-10-10): 잘린 확산 계획(A안). β 선형 1e-4→0.02/1000, 학습 t∈[0,50), 추론 t=49 1스텝, 정규화 (10,50) m
            plan_diffusion=dict(type="PlanTruncatedDiffusion", embed_dims=embed_dims, ego_fut_ts=ego_fut_ts,
                                num_train_timesteps=1000, trunc_steps=50, infer_t=8, beta_start=1e-4, beta_end=0.02,
                                norm_scale=(13.0, 29.0), norm_offset=(-1.0, 27.0), infer_seed=0),
            #수정(2026-10-12): DiffusionDrive(hustvl/DiffusionDrive transfuser_model_v2.py) 와 일치시킴.
            #  - infer_t 49 -> 8: DiffusionDrive 추론은 trunc_timesteps=8 에서 시작한다(학습 t~U{0..49} 는 동일).
            #    t=49 는 추론 전진 잡음 σ 8.5 m 로 앵커 최근접 간격(중앙값 1.6 m)·0.5s 전진(3.4 m)을 덮었다.
            #    검증셋 튜닝값이 아니라 인용 방법의 값을 사전에 고정한 것이다.
            #  - 정규화: DiffusionDrive norm_odo 와 같은 규칙((x-offset)/scale 로 데이터 [min,max] -> [-1,1]).
            #    nuScenes 학습 분할 GT 누적 궤적(마스크 유효 154,080점): 횡 x [-13.97, 11.77], 전진 y [-2.02, 55.79]
            #    -> x: offset -1, scale 13 / y: offset 27, scale 29 (DiffusionDrive 전진 offset 27.25, scale 28.45).
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
                #수정(Run10): 상위와 동일하게 plan/ego -> det/map 시간 attention 그룹 복원
                query_list=[["det"], ["map"], ["plan", "ego"]],
                key_list=[["det"], ["map"], ["det", "map"]],
                decouple_list=[True, False, False],
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
            #수정(Run10): 상위 hipad_base_b2d_stage1.py 와 동일하게 복원. plan/ego 쿼리가 det/map 을 본다.
            inter_graph_model=dict(
                type="InteractiveAttention",
                query_select=query_select,
                query_list=[["plan", "ego"]],
                key_list=[["det", "map"]],
                decouple_list=[False],
                attn=[
                    dict(
                        type="MultiheadFlashAttention",
                        embed_dims=embed_dims,
                        num_heads=num_groups,
                        batch_first=True,
                        dropout=drop_out,
                    ),
                ],
            ),
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
                              loss_yawness=dict(type="GaussianFocalLoss"),
                              cls_allow_reverse=[det_class_names.index("barrier")]),  #수정(Run10): barrier 는 앞뒤 구분 없음(평가 yaw mod pi)
            loss_map_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_map_reg=dict(type="SparseLineLoss",
                              loss_line=dict(type="LinesL1Loss", loss_weight=10.0, beta=0.01),
                              num_sample=map_num_pts,
                              roi_size=map_roi_size),
            loss_ego_status=dict(type="L1Loss", loss_weight=1.0),  #수정(Run10): stage2 에서 켬
            loss_plan_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.5),  #수정(Run10): stage2 에서 켬
            loss_plan_reg=dict(type="L1Loss", loss_weight=1.0),  #수정(Run10): stage2 에서 켬
            loss_motion_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.2),
            loss_motion_reg=dict(type="L1Loss", loss_weight=0.1),  #수정(Run10): 0.2 -> 0.1, fut_ts 6->12 지평 정규화(s2hr 와 동일)
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
dataset_type = "NuScenes3DDataset"  # 수정(2026-10-15): 클래스 이름 변경(옛 이름 NuScenesMiniDataset 도 같은 클래스로 등록되어 있음)
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
    dict(type="LoadQwenCacheNusc", cache_dir=qwen_cache_dir),  # 수정(2026-10-10): Qwen 캐시를 데이터 워커에서 선읽기(속도, 값 동일)
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "ego_status", "ego_status_mask", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               #수정: 계획 쿼리 학습용 자차 궤적 GT(2Hz 접미사 키)
               "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz",
               #수정(Run10): stage2 모션 과제용 agent 미래 궤적 GT
               "gt_agent_fut_trajs", "gt_agent_fut_masks"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token"],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="LoadQwenCacheNusc", cache_dir=qwen_cache_dir),  # 수정(2026-10-10): Qwen 캐시를 데이터 워커에서 선읽기(속도, 값 동일)
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "gt_bboxes_3d", "gt_labels_3d", "timestamp", "projection_mat", "image_wh", "ego_status", "gt_ego_fut_cmd"],  # 수정(2026-10-10): 추론 시 명령으로 계획 모드 묶음 선택(decoder.select) — 누락 시 KeyError
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
    #수정(Run9): final_dim 폭 704->896(x1.273)에 비례해 리사이즈 비율 상향
    "resize_lim": (0.40, 0.47), "final_dim": input_shape[::-1],  #수정(2026-10-09): 해상도 HiP-AD nuScenes 원래 값으로 복원(사용자 지시: 데이터셋별 원래 해상도). #이전 (0.51, 0.60)
    "bot_pct_lim": (0.0, 0.0), "rot_lim": (-5.4, 5.4),
    "H": 900, "W": 1600, "rand_flip": True, "rot3d_range": [0, 0],
}

data = dict(
    samples_per_gpu=batch_size, workers_per_gpu=12,  #수정(2026-10-10): GPU0 단독 → 12. #이전 8(Run10, RAM 안전상한)
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
    lr=2e-5,  #수정(Run10): 2e-4 -> 2e-5. 기존 파라미터를 stage1 수렴점 근방에 유지(2026-08-11 하락 진단)
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),           # 유효 1e-5
            "motion_anchor_encoder": dict(lr_mult=10.0), # 유효 2e-4 (신규 모듈)
            "motion_refine": dict(lr_mult=10.0),         # 유효 2e-4 (신규 모듈)
            #수정(2026-10-09, 2번안): 계획 모듈은 stage1 손실 0 → 사실상 신규, 모션과 같은 실효 2e-4
            "plan_instance_bank": dict(lr_mult=10.0),
            "plan_anchor_encoder": dict(lr_mult=10.0),
            "plan_deformable": dict(lr_mult=10.0),
            "plan_refine": dict(lr_mult=10.0),
            "plan_diffusion": dict(lr_mult=10.0),  # 수정(2026-10-10): 신규 확산 시점 임베딩
            "qwen_token_attn.gate_sep": dict(lr_mult=10.0),  # 수정(2026-10-12): plan/ego 언어 게이트(1단계에서 0)
            #수정(Run10): Run9 에서 미학습인 attention — inter_gnn 전체, temp_gnn 의 3번째 그룹(attns.2)
            **{f"onedecoder_head.layers.{i}.": dict(lr_mult=10.0)
               for i, op in enumerate(operation_order) if op == "inter_gnn"},
            **{f"onedecoder_head.layers.{i}.attns.2.": dict(lr_mult=10.0)
               for i, op in enumerate(operation_order) if op == "temp_gnn"},
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
    with_motion=True,  #수정(Run10)
    with_planning=False,
    tracking_threshold=0.2,
    tracking_relink=True,  #수정(2026-10-08): 평가 시 ID 재연결 후처리(사용자 결정). 원본은 tracking_raw/* 로 병기
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

# 수정(2026-10-15): 결과 디렉터리 이름(파일명이 train.py 로 바뀌어 파일명 기반 기본값을 쓸 수 없다).
run_name = "hipad_nusc_laq_stage2"
