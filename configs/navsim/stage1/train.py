# 수정(2026-10-16): v2 학습 config 반영 — nuScenes v2(configs/nuscenes/stage{1,2}/train_v2.py)의 모델·학습 방식 항목과 버그 수정을 옮김.
#   데이터셋 고유 항목(클래스·카메라·해상도·앵커·캐시·명령/계획 형식·epoch·batch·lr 의 누씬 v2(HiP-AD 에 NAVSIM 설정 없음) 기준값)은 그대로.
#   기록: backup/configs_pre_v2_sync_20261016/REPORT.md (원본 사본 같은 폴더)
# 수정(2026-10-15): 경로 이동 configs/navsim/hipad_navsim_laq_stage1_train.py → configs/navsim/stage1/train.py
#   run_name: 결과 디렉터리 이름(results/<run_name>, evaluation/<run_name>). tools/train.py·test.py 가 --work-dir 미지정 시 쓴다.
# 수정(2026-10-09): LaQ-AD NAVSIM stage1 학습 config (navtrain). 사용자 확정: 주 지표 EPDMS(navtest), 학습 navtrain.
#   출처: configs/nuscenes/hipad_nusc_laq_stage1_train.py 전문 복사 + 아래 치환(각 치환은 생성 스크립트가 1회 일치를 assert).
#   구조·학습 일정(에폭·유효배치·lr·fp32·Qwen 18층 융합)은 nuScenes 와 동일. 바뀐 것:
#     카메라 8대 / 물체 7클래스(navsim_scenario_utils.py:22-31) / 지도 3클래스(navsim_devkit_steps.py) / 앵커 navsim_*.npy /
#     자차 계획 8포즈(4 s @ 2 Hz) / ego_status 4차원 / 경로 명령 임베딩 ON(num_command=4) / 자차 앵커·크기 custom /
#     micro-batch 2 + 누적(VRAM 24GB 상한) / 추론 시 장면 경계 뱅크 초기화(reset_bank_on_scene_change).
#   [데이터로 확정 — 값이 틀리면 NavsimDataset 이 즉시 실패] navsim_ego_anchor / navsim_ego_size (egometa 출력),
#     map_roi_size(egometa 의 라이다 전방 축), navsim_train_frames(navtrain info 프레임 수), image_hw(영상 크기),
#     point_cloud_range z(navsim_kmeans --z-range 와 같게).
# 수정(2026-10-09): 벤치마크·stage·학습/테스트 단위 config 정리(사용자 지시).
#   원본: configs/hipad_nusc_vla_s1hr.py 를 그대로 복사한 학습용 config. = Run9(1단계, 896x448, 100ep scratch) 의 config.
#   주의: config 는 같아도 코드는 Run9 이후 바뀌었다(10-08: cns 목표 detach, map dt 분리 등) → 지금 다시 학습하면
#   Run9 와 동일 실험이 아니다(Run9 결과의 재현이 아니라 결함 수정판). 배리어 cls_allow_reverse 는 이 config 에 없음(2단계에만).
#   테스트용 짝 config 는 같은 폴더의 *_test.py (_base_ 로 이 파일을 상속, 평가 설정만 덮어씀).
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

# ================== NAVSIM 전용 상수 (수정 2026-10-09) ==================
#   PROVISIONAL: 아래 3개는 데이터·devkit 없이 정할 수 없어 임시값이다. 변환기 출력으로 바꾸기 전에는
#   NavsimDataset 의 일치 검사가 학습 시작 시점에 실패한다(조용히 틀린 값으로 학습하지 않음).
navsim_ego_anchor = [0.0, 0.0, 0.0, 1.6441, 0.8316, 0.5750, 0.0, 1.0, 0.0, 0.0, 0.0]  # PROVISIONAL: egometa 출력으로 교체
navsim_ego_size = [5.176, 2.297, 1.777]   # 수정(2026-11-06): 확인됨 — navsim env nuplan get_pacifica_parameters(): length 5.176·width 2.297·height 1.777. rescore_ego_offset 1.461 = half_length 2.588 − rear_length 1.127(= rear_axle_to_center, 후륜축→박스 중심). #이전 PROVISIONAL: nuPlan Pacifica 통상값 — egometa(get_pacifica_parameters) 출력으로 교체
navsim_train_frames = 0                    # PROVISIONAL: build_navsim_infos --kind train 의 frames 수로 교체(0 이면 학습 불가)
navsim_image_hw = (1080, 1920)             # 확인 필요: 데이터 영상 크기(NavsimDataset 이 첫 표본에서 검사)

num_gpus = 1  #수정: nuScenes full 멀티태스크, GPU0 단독
batch_size = 2  #수정(2026-10-09): NAVSIM — VRAM 24GB 상한(사용자). (이전 8캠 기준 주석) 현재 3캠×512x256 — VRAM 재측정 필요. #nuScenes Run9 = 8
#수정(2026-08-03): effective batch 48 → 64. 근거 = 상류 config 실측 대조.
#   HiP-AD 공개 stage1(projects/configs/hipad_b2d_stage1.py): 8gpu × batch 8 = total 64
#   SparseDrive nuScenes stage1(sparsedrive_small_stage1.py): total_batch_size = 64
#   두 레퍼런스가 모두 64 로 일치 → 48(이전 가정 "8gpu*6")은 근거 없는 값이었다.
#   1 GPU 이므로 micro-batch 8 × grad accumulation 8 = 64 로 재현한다.
cumulative_iters = 32  #수정(2026-10-09): eff 64 = 1*2*32 (nuScenes 1단계와 동일)
# num_iters_per_epoch = int(234769 // (num_gpus * batch_size))
num_iters_per_epoch = int(navsim_train_frames // (num_gpus * batch_size * cumulative_iters)) * cumulative_iters  # 수정(2026-10-16): v2 학습 config 반영 — epoch 당 micro-iter 를 누적 수의 배수로(체크포인트·재개 지점 = optimizer step 경계, 스트림 그룹 순환 보존). #이전 int(navsim_train_frames // (num_gpus * batch_size))
num_stream_groups = cumulative_iters  # 수정(2026-10-16): v2 학습 config 반영 — (2) 독립 스트림 2×32=64 (= 누씬 v2·SparseDrive 의 step 당 독립 시퀀스 64). 종전: 같은 2 시퀀스의 연속 프레임
num_epochs = 100  #수정: HiP-AD/SparseDrive nuScenes stage1 표준(=100ep), 비교 타당성 위해 일치
#checkpoint_epoch_interval = 20
checkpoint_epoch_interval = 1

checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)
load_from = None  # NAVSIM 1단계는 처음부터(nuScenes Run9 와 같은 일정)
resume_from = None
workflow = [("train", 1)]
find_unused_parameters = True  #수정: 미사용 plan/ego 모듈(빌드만)용
fp16 = dict(loss_scale=32.0)  # 수정(2026-10-18): 사용자 결정 — HiP-AD 원본과 같은 fp16(정적 배율 32, 넘친 step 은 apis/fp16_strict_hook.py 가 멈춤). #이전 fp16 = None
cudnn_benchmark = True  # 수정(2026-10-16): v2 학습 config 반영 — 입력 크기 고정 → 합성곱 알고리즘 자동선택(수치 설정 무관)
#수정(2026-08-03): (640,352) → (704,256). HiP-AD 논문 Appendix B 가 nuScenes 는
#   "the resolution is changed to 704 × 256" 이라고 명시하고, SparseDrive nuScenes config 도 동일.
#   (640,352)는 Bench2Drive 해상도를 그대로 들고 온 값이라 Table 4 비교가 성립하지 않았다.
#   data_aug_conf 의 나머지 항목은 이미 SparseDrive 와 동일 → 이 한 줄로 정렬 완료.
input_shape = (512, 256)  # 수정(2026-10-09): SparseDriveV2 NAVSIM 설정(사용자 결정) — 전방 3캠 l0·f0·r0, 1프레임 (swc-17/SparseDriveV2 sparsedrive_config.py:63): 카메라당 512x256 — 명시 지정
# 수정(2026-11-06): NAVSIM 전방 카메라 색인 — 학생 카메라 순서 NAVSIM_CAMS = (CAM_L0, CAM_F0, CAM_R0)(models/loader/navsim/dataset.py) 에서 정면 CAM_F0 = 1.
#   계획·자차 뱅크가 카메라 0(= CAM_L0, 왼앞)을 정면으로 가정하던 것을 바로잡는다. 1단계는 계획·자차 손실 0 이고 det/map 쿼리가
#   plan/ego 쿼리를 읽지 않아 인식 결과는 같다(영향 = 계획·자차 인코더 BN 이동 통계 뿐). 1단계 시작 전 적용(재개 도중 바꾸지 않음).
#   docs/navsim_front_view_fix_20261106.md
navsim_front_view_idx = 1  # 수정(2026-11-06): NAVSIM 전방 카메라 색인
num_cams = 3  # 수정(2026-10-09): SparseDriveV2 NAVSIM 설정(사용자 결정) — 전방 3캠 l0·f0·r0, 1프레임 (swc-17/SparseDriveV2 sparsedrive_config.py:63)

# det & map
det_class_names = ["vehicle", "pedestrian", "bicycle", "traffic_cone", "barrier", "czone_sign", "generic_object"]  #수정(2026-10-09): NAVSIM 7 클래스(navsim_scenario_utils.py:22-31, ego 제외)
map_class_names = ["lane_centerline", "road_boundary", "walkway"]  #수정(2026-10-09): NAVSIM 지도 클래스(navsim_devkit_steps.py MAP_CLASSES) — 사용자 확인 필요

num_det_classes = len(det_class_names)
num_map_classes = len(map_class_names)

map_roi_size = (30, 60)  #수정(2026-10-09): 확인 필요 — (라이다 x 길이, y 길이). 라이다 전방 축이 x 면 (60, 30). navsim_devkit_steps.py map --roi 와 같게
map_num_pts = 20

# traj
fut_ts = 6
fut_mode = 6
#수정(2026-08-11): 자차 상태 차원. B2D 는 6(속도,가속2,각속2,조향), nuScenes 는 10
#   (가속3, 각속3, 속도3, 조향1). EgoStatusRefinementModule 기본값이 6 이라 그대로 두면
#   loss_ego 에서 pred(6) vs target(10) 로 assert 실패한다.
ego_status_dims = 4  #수정(2026-10-09): NAVSIM ego_dynamic_state (vx, vy, ax, ay)
ego_fut_ts = 8  #수정(2026-10-09): NAVSIM Trajectory 4 s @ 0.5 s = 8 포즈(dataclasses.py:284)
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
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]  #수정(2026-10-09): 확인 필요 — NAVSIM 라이다 높이에 맞춘 z 범위(navsim_kmeans --z-range 와 같게)

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
project_dir = "/home1/irteam/data-vol1/src/HiP-AD"

anchor_paths = {
    "det" : f"{project_dir}/data/kmeans/navsim_det_900.npy",
    "map" : f"{project_dir}/data/kmeans/navsim_map_100.npy",
    "motion": f"{project_dir}/data/kmeans/navsim_motion_{fut_mode}.npy",  #수정: CARLA 앵커 -> nuScenes 앵커(stage1 에선 미사용이나 stage2 와 일관)
}

plan_anchor_paths = f"{project_dir}/data/kmeans/navsim_plan_temp_48.npy"  #수정: CARLA 앵커 -> nuScenes 앵커
plan_anchor_refer = ("temp", "2hz")
plan_anchor_types = [("temp", "2hz")]


#수정: Qwen hidden 캐시 경로. 지정 시 sparse_detector 가 배치 토큰으로 캐시를 읽어
#   metas['qwen_hidden'] / metas['qwen_mask'] 로 주입한다. 누락 시 조용히 넘기지 않고 실패한다.
#수정(2026-09-15, Run9): 마지막 계층 -> 18층 캐시로 전환.
#  근거 — VLM 내부 분석(arXiv 2606.06890): 시각적으로 접지된 신호는 중간 계층에서 가장
#  강하고 마지막 계층에서는 언어 prior 에 눌린다(28계층 모델 기준 14~19층 부근).
#  Qwen2-VL-2B 도 28계층이므로 18층(64% 깊이)을 주력으로 쓴다. 14/22층도 같은 순전파에서
#  함께 저장해 두어(비용 0) 추후 계층 ablation 이 가능하다.
qwen_cache_dir = "data/navsim_qwen_l18"  #수정(2026-10-09): tools/qwen_cache_navsim.py 산출(NAS 경로 심링크). 8캠 → L 은 _meta.txt 참조

model = dict(
    type="SparseDetector",
    reset_bank_on_scene_change=True,  #수정(2026-10-09): navtest 장면(겹친 창) 경계에서 시간 뱅크 초기화(추론 전용 동작)
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
        with_cp=True,  # 수정(2026-10-16): v2 학습 config 반영 — (7) v2 와 같이 True(SparseDrive 원본). 수식은 같고 재계산 forward 에서 BN running stats 가 iter 당 2회 갱신되는 것만 다름. #이전 False
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
            num_stream_groups=num_stream_groups,  # 수정(2026-10-16): v2 학습 config 반영 — (2) 스트림 그룹별 시간 뱅크 상태
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
            ),
            num_single_frame_decoder=num_single_frame_decoder,
            plan_anchor_refer=plan_anchor_refer,
            with_command_embed=True,   #수정(2026-10-09): NAVSIM 경로 명령(좌/직/우/미상) — 정답 누설 아님(docs/agents.md:79)
            num_command=4,             #수정(2026-10-09): NAVSIM driving_command 4차원
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
                anchor_type="custom",  #수정(2026-10-09): NAVSIM 자차(nuPlan Pacifica)
                custom_anchor=navsim_ego_anchor,
                embed_dims=embed_dims,
                num_temp_instances=1 if temporal_ego else -1,
                front_view_idx=navsim_front_view_idx,  # 수정(2026-11-06): NAVSIM 전방 카메라 색인
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
                front_view_idx=navsim_front_view_idx,  # 수정(2026-11-06): NAVSIM 전방 카메라 색인
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
                num_cams=num_cams,  #수정(2026-10-09): 하드코딩 6 → 변수(NAVSIM 3캠). 6 이면 카메라 축이 어긋남
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
                num_cams=num_cams,  #수정(2026-10-09): 하드코딩 6 → 변수(NAVSIM 3캠). 6 이면 카메라 축이 어긋남
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
                num_cams=num_cams,  #수정(2026-10-09): 하드코딩 6 → 변수(NAVSIM 3캠). 6 이면 카메라 축이 어긋남
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
                num_cams=num_cams,  #수정(2026-10-09): 하드코딩 6 → 변수(NAVSIM 3캠). 6 이면 카메라 축이 어긋남
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
                              cns_detach=False,  # 수정(2026-10-16): v2 학습 config 반영 — (3) centerness 타깃 detach 안 함(SparseDrive·HiP-AD 원본 코드와 같음)
                              cls_allow_reverse=[det_class_names.index("barrier")]),  # 수정(2026-10-16): v2 학습 config 반영 — (5) barrier 앞뒤 구분 없음(SparseDrive; 종전엔 2단계에만)
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
                              ego_fut_mode=ego_fut_mode, ego_vehicle="custom", ego_size=navsim_ego_size, anchor_types=plan_anchor_types,
                              anchor_refer=plan_anchor_refer, with_rescore=True,  # 수정(2026-11-06): 다음 줄에 rescore 인자를 더하려고 닫는 괄호를 옮김. #이전 with_rescore=True)
                              ego_start_yaw=0.0,  # 수정(2026-11-06): NAVSIM 계획 좌표 +x 전진(lidar2ego=I, GT 끝 변위 x 우세) → 정지 모드 yaw 0. #이전 (없음 = π/2)
                              rescore_ego_offset=1.461),  # 수정(2026-11-06): 후륜축 원점 → 박스 중심 = Pacifica half_length 2.588 − rear_length 1.127. #이전 (없음 = 0.5)
            motion_decoder=dict(type="SparseMotionDecoder"),
        ),
    ),
)

# ================== data ========================
dataset_type = "NavsimDataset"  #수정(2026-10-09)
data_root = "data/navsim"  #수정(2026-10-09): 영상 경로는 info 의 절대경로를 쓴다(build_navsim_infos --sensor-root)
anno_root = "data/infos/"
file_client_args = dict(backend="disk")

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),  #수정: depth aux 용 lidar
    dict(type="ResizeCropFlipImage"),
    dict(type="MultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),  #수정: depth GT 생성
    dict(type="BBoxRotation"),  # 수정(2026-10-16): v2 학습 config 반영 — (7) SparseDrive 와 같은 위치. rot3d_range=[0,0] 이라 항등(값 불변)
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),  # 수정(2026-10-16): v2 학습 config 반영 — (4) 학습 GT 거리 필터 정사각 ±50 m → 원형 55 m(SparseDrive·HiP-AD B2D 원본과 같음). 평가 쪽은 그대로
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=map_roi_size, simplify=False, normalize=False,
         sample_num=map_num_pts, permute=True),  #수정: permute=True → gt_map_labels/gt_map_pts
    dict(type="LoadQwenCacheNusc", cache_dir=qwen_cache_dir),  # 수정(2026-10-16): v2 학습 config 반영 — Qwen 캐시를 데이터 워커에서 읽음(누씬 10-10 속도 수정, 값 동일·누락 시 즉시 실패)
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "ego_status", "ego_status_mask", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               #수정: 계획 쿼리 학습용 자차 궤적 GT(2Hz 접미사 키)
               "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token", "token", "stream_group"],  # 수정(2026-10-16): v2 학습 config 반영 — (2) stream_group
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="BEVObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="LoadQwenCacheNusc", cache_dir=qwen_cache_dir),  # 수정(2026-10-16): v2 학습 config 반영 — Qwen 캐시를 데이터 워커에서 읽음(누씬 10-10 속도 수정, 값 동일·누락 시 즉시 실패)
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "qwen_hidden", "qwen_mask", "gt_bboxes_3d", "gt_labels_3d", "timestamp", "projection_mat", "image_wh", "ego_status",
               "gt_ego_fut_cmd"],  #수정(2026-10-09): 명령 임베딩 입력은 추론에도 필요
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
    ego_anchor=navsim_ego_anchor, ego_size=navsim_ego_size, image_hw=navsim_image_hw,  #수정(2026-10-09): 변환기 산출물과 일치 검사
    map_roi=map_roi_size,
)

data_aug_conf = {
    #수정(Run9): final_dim 폭 704->896(x1.273)에 비례해 리사이즈 비율 상향
    "resize_lim": (512/1920, 512/1920), "final_dim": input_shape[::-1],  #수정(2026-10-09): SparseDriveV2 고정 배율(크기 증강 없음). 1920x1080→512x288→위 32행 crop→512x256
    #   수정(2026-10-09): HiP-AD nuScenes 704x256 은 1600x900 에 resize_lim (0.40, 0.47) (중앙 0.435 ≈ 704/1600=0.44).
    #   NAVSIM 1920x1080 은 같은 16:9 이므로 같은 상대 배율을 쓰도록 폭 비로 환산: (0.40, 0.47) x 1600/1920 = (0.333, 0.392)
    #   (중앙 0.3625 ≈ 704/1920=0.367). 640x352 로 바꾸면 HiP-AD B2D 규칙을 같은 방식으로 다시 환산할 것.
    "bot_pct_lim": (0.0, 0.0), "rot_lim": (0.0, 0.0),  #수정(2026-10-09): SparseDriveV2 rot_lim 0. #이전 (-5.4, 5.4)
    "H": navsim_image_hw[0], "W": navsim_image_hw[1],  #수정(2026-10-09)
    "rand_flip": False, "rot3d_range": [0, 0],  #수정(2026-10-09): SparseDriveV2 rand_flip False. #이전 True
}

data = dict(
    num_stream_groups=num_stream_groups,  # 수정(2026-10-16): v2 학습 config 반영 — (2) 샘플러 그룹 수
    samples_per_gpu=batch_size, workers_per_gpu=8,  #수정(2026-10-09): worker 8 고정(RAM 안전상한)
    train=dict(**data_basic_config, ann_file=anno_root + "navsim_navtrain.pkl", num_frames=navsim_train_frames,
               pipeline=train_pipeline, test_mode=False, data_aug_conf=data_aug_conf,
               with_seq_flag=True, sequences_split_num=2, keep_consistent_seq_aug=True),
    val=dict(**data_basic_config, ann_file=anno_root + "navsim_navtest.pkl",
             pipeline=test_pipeline, data_aug_conf=data_aug_conf, test_mode=True),
    test=dict(**data_basic_config, ann_file=anno_root + "navsim_navtest.pkl",
              pipeline=test_pipeline, data_aug_conf=data_aug_conf, test_mode=True),
)

# ================== training ========================
optimizer = dict(
    type="AdamW",
    lr=4e-4,  # 수정(2026-10-16): v2 학습 config 반영 — (1) 누씬 v2 1단계와 같은 4e-4(SparseDrive 1단계, eff 64). HiP-AD 에 NAVSIM 설정이 없어 v2 를 따름 → 판단 필요. #이전 2e-4
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
    tracking_relink=True,  #수정(2026-10-08): 평가 시 ID 재연결 후처리(사용자 결정). 원본은 tracking_raw/* 로 병기
    motion_threshhold=0.2,
)
#수정(2026-08-03): map 평가기(VectorEvaluate)는 GT map vector 를 예측 결과가 아니라
#  별도 dataset(eval_pipeline)에서 다시 로드한다 → 그 dataset cfg 를 여기서 준다.
#  SparseDrive sparsedrive_small_stage1.py 의 eval_config 와 동일 구성.
eval_config = dict(
    **data_basic_config,
    ann_file=anno_root + "navsim_navtest.pkl",
    pipeline=eval_pipeline,
    test_mode=True,
)
evaluation = dict(
    interval=num_iters_per_epoch*checkpoint_epoch_interval,
    jsonfile_prefix="evaluation/navsim",  #수정(2026-10-09): NavsimDataset.evaluate 가 navsim_pred.pkl 저장
    eval_mode=eval_mode,
    eval_config=eval_config,
)

# 수정(2026-10-15): 결과 디렉터리 이름(파일명이 train.py 로 바뀌어 파일명 기반 기본값을 쓸 수 없다).
run_name = "hipad_navsim_laq_stage1"
