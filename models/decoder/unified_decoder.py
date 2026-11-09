# 수정(2026-10-15): 경로 이동 models/sparse_onedecoder.py → models/decoder/unified_decoder.py 및 정리.
#   통합 디코더(SparseOneDecoder): 하나의 쿼리 풀(det/map/plan/ego)을 operation_order 순서의 연산
#   (concat/gnn/temp_gnn/inter_gnn/norm/split/deformable/ffn/qwen_attn/refine)으로 갱신하고 과제별 손실·후처리를 낸다.
#   제거(어떤 live config 도 켜지 않음 — state_dict·출력 동일, 2026-10-15 검증):
#     - scenes(장면 토큰) 쿼리·손실 경로 → forward 반환은 5-튜플(det, map, ego, plan, motion)
#     - attention mask 일체(with_attn_mask / attn_mask_dict / with_distance_attn_mask / with_velocity_attn_mask /
#       with_plan_group_attn_mask, distance_tau·velocity_tau)
#     - 점 단위 지도·계획 쿼리(with_concat_{map,plan}_points / with_deform_{map,plan}_points, suqueeze_* MLP)
#     - VLA-v0 외부 궤적 교차 어텐션(with_external_trajectory), 계획 전용 Qwen 교차 어텐션(plan_qwen_token_attn)
#     - 계획 top-k 모드(with_topk_mode / topk_mode_list / keep_topk_relative_pos)
#     - with_target_point_next_embed / with_custom_status_embed, 쓰지 않던 계획 보조 손실(loss_plan_col/dir/bound/status)
#     - independent_{gnn,temp_gnn,inter_gnn}=False 경로(그래프 어텐션은 항상 독립 모듈 호출)
#     - with_supervise_ego_status=False 경로(자차 궤적 회귀: ego_decoder / ego_sampler / loss_ego_cls / loss_ego_reg)
#   with_incremental_plan_refine 은 원래부터 어디서도 읽지 않는 키다(**kwargs 로 흡수, B2D config 에 True 로 남아 있음).
import copy
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)

from mmcv.utils import build_from_cfg
from mmcv.runner import BaseModule, force_fp32

from mmdet.core import reduce_mean
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.models import HEADS, LOSSES

from typing import List, Optional

from ..common.box3d import *
from ..common.layers import linear_relu_ln, gen_sineembed_for_position
from ..common.sync_free import const_like, gather_to_host  # 수정(2026-10-10): 동기화 제거 # 수정(2026-11-09): gather_to_host(정합 비용 일괄 전송)
from .query_anchors.motion_anchor import load_motion_anchor, build_motion_anchor_encoder
from .temporal_bank.stream_state import bank_state_keys, save_bank_state, load_bank_state, empty_bank_state  # 수정(2026-10-15): 실험 C
import re  # 수정(2026-11-09): C3 GRPO §9.3 격리 전제 검사

# 수정(2026-11-09): C3 GRPO §9.3 — 격리 켬일 때 학습 대상으로 허용하는 디코더 매개변수(onedecoder_head 기준 이름).
#   tools/navsim/grpo/rollout.TRAINABLE 과 같은 목록(접두 head.onedecoder_head. 제외). 바꾸면 둘 다 바꾼다.
_GRPO_PLAN_PARAM = re.compile(r"^(plan_refine\.|plan_anchor_encoder\.|plan_deformable\.|plan_diffusion\.t_mlp\.|layers\.\d+\.plan\.|"
                              r"qwen_token_attn\.gate_sep$)")

__all__ = ["SparseOneDecoder"]


@HEADS.register_module()
class SparseOneDecoder(BaseModule):
    # 2026-10-15 소스 정리에서 제거한 인자. 값이 참(켜짐)으로 들어오면 __init__ 에서 실패한다.
    _REMOVED_KEYS = (
        "scenes_instance_bank", "scenes_attention", "scenes_refine_layer", "loss_scenes_reg",
        "loss_plan_col", "loss_plan_dir", "loss_plan_bound", "loss_plan_status",
        "loss_ego_cls", "loss_ego_reg", "ego_decoder", "ego_sampler", "ego_anchor_encoder",
        "attn_mask_dict", "with_attn_mask", "with_distance_attn_mask", "with_velocity_attn_mask",
        "with_plan_group_attn_mask", "with_target_point_next_embed", "with_custom_status_embed",
        "with_external_trajectory", "external_trajectory_encoder", "external_trajectory_cross_attn",
        "plan_qwen_token_attn", "with_concat_map_points", "with_deform_map_points",
        "with_concat_plan_points", "with_deform_plan_points", "with_topk_mode", "topk_mode_list",
        "keep_topk_relative_pos",
    )

    def __init__(
            self,
            ffn: dict = None,
            init_cfg: dict = None,
            custom_op: dict = None,
            embed_dims: int = 256,
            num_decoder: int = 6,
            num_single_frame_decoder: int = -1,

            decouple_attn: bool = True,
            with_instance_id: bool = True,
            cls_threshold_to_reg: float = -1,

            norm_layer: dict = None,
            graph_model: dict = None,
            temp_graph_model: dict = None,
            inter_graph_model: dict = None,
            operation_order: Optional[List[str]] = None,

            det_instance_bank: dict = None,
            map_instance_bank: dict = None,
            ego_instance_bank: dict = None,
            plan_instance_bank: dict = None,

            det_anchor_encoder: dict = None,
            map_anchor_encoder: dict = None,
            plan_anchor_encoder: dict = None,

            det_deformable: dict = None,
            map_deformable: dict = None,
            ego_deformable: dict = None,
            plan_deformable: dict = None,

            det_refine_layer: dict = None,
            map_refine_layer: dict = None,
            ego_refine_layer: dict = None,
            plan_refine_layer: dict = None,
            motion_refine_layer: dict = None,

            loss_det_cls: dict = None,
            #수정(2026-09-04): 검출 분류 손실의 클래스별 가중치.
            #  [문제] 격차가 희소 클래스 두 개에 집중돼 있다 — trailer AP 0.066 /
            #  construction_vehicle 0.065 인 반면 나머지 8개는 0.330~0.674 다. 이 둘만
            #  0.20 이면 mAP 0.4034 -> 0.4303 으로 기준선을 넘는다.
            #  희소성 자체는 원인이 아니다: bicycle(9,478개) AP 0.376 인데
            #  trailer(20,701개)가 0.066 이다. GT 개수·크기·앵커는 모두 정상임을 확인했다.
            #  [원인] FocalLoss 를 전체 양성 수(avg_factor=num_pos)로만 나누므로, GT 의
            #  42%를 차지하는 car 와 19%인 pedestrian 의 기울기에 2.1%인 trailer 가 묻힌다.
            #  [왜 CBGS 가 아닌가] 이 모델은 GroupInBatchSampler 로 시퀀스 단위 배칭을 하는
            #  시간적 모델이라, CBGS 재샘플링은 프레임 연속성을 깨 인스턴스 뱅크 전파를
            #  무너뜨린다(데이터셋에 get_cat_ids 도 없다). 샘플러를 건드리지 않고 손실에서
            #  직접 균형을 잡는다.
            det_cls_class_weights=None,
            loss_det_reg: dict = None,
            loss_map_cls: dict = None,
            loss_map_reg: dict = None,
            loss_ego_status: dict = None,
            loss_plan_cls: dict = None,
            loss_plan_reg: dict = None,
            loss_motion_cls: dict = None,
            loss_motion_reg: dict = None,

            det_reg_weights: List = None,
            map_reg_weights: List = None,

            det_decoder: dict = None,
            map_decoder: dict = None,
            plan_decoder: dict = None,
            motion_decoder: dict = None,

            det_sampler: dict = None,
            map_sampler: dict = None,
            plan_sampler: dict = None,
            align_sampler: dict = None,
            motion_sampler: dict = None,

            motion_anchor=None,
            plan_speed_refer=None,
            plan_anchor_refer=None,
            combine_layer_loss=True,

            task_select=["det", "map", "motion", "plan"],
            query_select=["det", "map", "plan"],

            num_command=6,

            with_close_loop=False,  # default False, True means use closed-loop memory bank
            open_loop_hz=2,
            close_loop_hz=20,
            open_loop_bank_length=None,
            close_loop_bank_length=None,

            with_command_embed=False,           # embeds command to planning query
            with_target_point_embed=False,      # embeds target points to planning query
            with_supervise_ego_status=True,     # ego query predicts (supervised) ego status
            with_ego_instance_feature=False,    # enable plan_instance_feature+= ego_instance_feature

            #수정: VLA-v1 — Qwen hidden state sequence K/V ↔ unified query Q cross-attn.
            qwen_token_attn: dict = None,
            plan_diffusion: dict = None,  # 수정(2026-10-10): 잘린 확산 계획(A안). None 이면 기존과 동일
            # 수정(2026-10-16): F1 계획 정답 모드 = GT 최근접 깨끗한 앵커(DiffusionDrive 방식)
            #   'output'(기본, 종전과 비트 동일): 층마다 정제된 출력 중 GT 최근접 모드를 정답으로 쓴다.
            #   'clean_anchor': 잡음 넣기 전 깨끗한 앵커(plan_instance_bank.anchor) 중 GT 명령 그룹에서 GT 최근접 모드를
            #   모든 층의 공통 정답으로 쓴다(hustvl/DiffusionDrive multimodal_loss.py LossComputer.forward 와 같은 규칙).
            #   검증: HiP-AD 트리 nuScenes 계획부 미세조정 시험(configs/nuscenes/hipad_nusc_laq_stage2_f1_planft_*.py, TODO.md 2026-10-16).
            plan_target: str = "output",
            num_stream_groups: int = 1,  # 수정(2026-10-15): 실험 C — 학습 시 스트림 그룹별 뱅크 상태(1 = 종전 동작, 분기 없음)
            # 수정(2026-10-27): 1단계 미달 후보 3 진단 — True 면 ffn·norm 을 쿼리 종류(query_select)별로 따로 둔다
            #   (SparseDrive 처럼 검출·지도가 FFN/LayerNorm 을 공유하지 않음). False(기본) = HiP-AD 통합 디코더 그대로.
            task_specific_ffn_norm: bool = False,
            # 수정(2026-11-05): 사용자 지시 "MCMC SFT 확인한대로 진행" — 계획 SFT 정답 키(학습 전용). None(기본) = 종전과 비트 동일.
            #   문자열(예 "gt_ego_plan_sft_2hz")이면 get_gt_trajs 가 ("temp","2hz") 계획 정답 궤적만 data[그 키] 에서 읽는다
            #   (마스크는 gt_ego_fut_masks_2hz 그대로; 평가가 읽는 gt_ego_fut_trajs_2hz 는 건드리지 않음). 키가 없으면 KeyError(대체 없음).
            plan_sft_key: str = None,
            # 수정(2026-11-05): 사용자 결정 "E2 자차 상태 직접 입력" — 교사 프롬프트에 이미 주는 CAN 자차 상태(속도·종방향 가속도·yaw rate·유효)를
            #   텍스트 해독에 의존하지 않고 ego 쿼리에 직접 더한다. None(기본) = 모듈·파라미터·forward 모두 종전과 동일.
            #   dict(in_dims=4, scale=[10.0, 1.0, 0.2, 1.0]): metas["ego_input"] (B, in_dims) / scale → MLP(마지막 Linear 0 초기화)
            #   → ego_instance_feature 에 잔차로 더함. 0 초기화라 시작 모델 출력 = 종전 모델(+0). 키가 없으면 KeyError(대체 없음).
            ego_input_embed: dict = None,
            # 수정(2026-11-09): C3 GRPO 설계 §9.1·§9.3 — RL 롤아웃 전용 플래그 2 개. 둘 다 False(기본) = 종전 학습·평가 동작·값·그래프 그대로.
            #   force_stream_groups: eval() 에서도 스트림 그룹 뱅크 상태 저장·복원(_stream_group_enter)을 켠다(GRPO 롤아웃은 eval 모드 + 스트림 그룹 6).
            #   plan_grad_isolation: 기울기가 켜진 forward 에서 det·map·ego 구간의 계산 그래프를 만들지 않는다(값은 그대로, forward 의 [격리] 주석).
            #   파이썬 속성이라 state_dict 불변. GRPO 루프(tools/navsim/grpo/rollout.py)가 실행 중 켠다.
            force_stream_groups: bool = False,
            plan_grad_isolation: bool = False,

            **kwargs,
    ):
        super(SparseOneDecoder, self).__init__(init_cfg)
        # 수정(2026-10-15): 제거된 기능을 켜는 설정을 **kwargs 가 조용히 삼키지 않도록 즉시 실패시킨다(no-skip 규칙).
        _enabled_removed = sorted(k for k in self._REMOVED_KEYS if kwargs.get(k))
        if _enabled_removed:
            raise ValueError(f"SparseOneDecoder: 2026-10-15 소스 정리에서 제거된 기능이 켜져 있다: {_enabled_removed}")
        if not with_supervise_ego_status:
            raise ValueError("with_supervise_ego_status=False(자차 궤적 회귀) 경로는 2026-10-15 소스 정리에서 제거됐다.")
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder

        self.with_instance_id = with_instance_id
        self.with_command_embed = with_command_embed
        self.with_target_point_embed = with_target_point_embed

        self.with_ego_instance_feature = with_ego_instance_feature
        self.with_supervise_ego_status = with_supervise_ego_status

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order
        self.cls_threshold_to_reg = cls_threshold_to_reg

        self.embed_dims = embed_dims
        self.task_select = task_select
        self.query_select = query_select

        self.with_close_loop = with_close_loop
        self.open_loop_hz = open_loop_hz
        self.close_loop_hz = close_loop_hz
        self.open_loop_bank_length = open_loop_bank_length
        self.close_loop_bank_length = close_loop_bank_length

        # 수정(2026-10-16): F1 계획 정답 모드 = GT 최근접 깨끗한 앵커(DiffusionDrive 방식)
        assert plan_target in ("output", "clean_anchor"), plan_target
        self.plan_target = plan_target

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        assert len(self.task_select)
        assert len(self.query_select)

        num_refine = operation_order.count('refine')
        num_deform = operation_order.count('deformable')

        if 'det' in self.query_select:
            self.det_instance_bank = build(det_instance_bank, PLUGIN_LAYERS)
            self.det_anchor_encoder = build(det_anchor_encoder, POSITIONAL_ENCODING)
            self.det_deformable = nn.ModuleList([build(det_deformable, ATTENTION) for _ in range(num_deform)])
            self.det_refine = nn.ModuleList([build(det_refine_layer, PLUGIN_LAYERS) for _ in range(num_refine)])
            self.det_sampler = build(det_sampler, BBOX_SAMPLERS)
            self.det_decoder = build(det_decoder, BBOX_CODERS)
            self.loss_det_cls = build(loss_det_cls, LOSSES)
            self.loss_det_reg = build(loss_det_reg, LOSSES)
            self.det_reg_weights = det_reg_weights
            self.num_det_anchor = self.det_instance_bank.num_anchor
            self.num_temp_det_anchor = self.det_instance_bank.num_temp_instances
            #수정(2026-09-04): 클래스별 가중치 버퍼. 길이 = num_classes+1, 마지막은 배경(=1.0).
            self.det_cls_class_weights = None
            if det_cls_class_weights is not None:
                w = torch.tensor(list(det_cls_class_weights) + [1.0], dtype=torch.float32)
                self.register_buffer("_det_cls_w", w, persistent=False)
                self.det_cls_class_weights = True

        if 'map' in self.query_select:
            self.map_instance_bank = build(map_instance_bank, PLUGIN_LAYERS)
            self.map_anchor_encoder = build(map_anchor_encoder, POSITIONAL_ENCODING)
            self.map_deformable = nn.ModuleList([build(map_deformable, ATTENTION) for _ in range(num_deform)])
            self.map_refine = nn.ModuleList([build(map_refine_layer, PLUGIN_LAYERS) for _ in range(num_refine)])
            self.map_sampler = build(map_sampler, BBOX_SAMPLERS)
            self.map_decoder = build(map_decoder, BBOX_CODERS)
            self.map_reg_weights = map_reg_weights
            self.loss_map_cls = build(loss_map_cls, LOSSES)
            self.loss_map_reg = build(loss_map_reg, LOSSES)
            self.num_map_anchor = self.map_instance_bank.num_anchor
            self.num_temp_map_anchor = self.map_instance_bank.num_temp_instances

        if 'ego' in self.query_select:
            self.ego_instance_bank = build(ego_instance_bank, PLUGIN_LAYERS)
            # 자차 앵커는 검출 앵커 인코더를 공유한다(같은 모듈 객체 — state_dict 에 두 이름으로 나타난다).
            if hasattr(self, 'det_anchor_encoder'):
                self.ego_anchor_encoder = self.det_anchor_encoder
            else:
                self.ego_anchor_encoder = build(det_anchor_encoder, POSITIONAL_ENCODING)
            self.ego_deformable = nn.ModuleList([build(ego_deformable, ATTENTION) for _ in range(num_deform)])
            self.ego_refine = nn.ModuleList([build(ego_refine_layer, PLUGIN_LAYERS) for _ in range(num_refine)])
            self.loss_ego_status = build(loss_ego_status, LOSSES)

        if 'plan' in self.query_select:
            self.plan_instance_bank = build(plan_instance_bank, PLUGIN_LAYERS)
            self.plan_anchor_encoder = build(plan_anchor_encoder, POSITIONAL_ENCODING)
            self.plan_deformable = nn.ModuleList([build(plan_deformable, ATTENTION) for _ in range(num_deform)])
            self.plan_refine = nn.ModuleList([build(plan_refine_layer, PLUGIN_LAYERS) for _ in range(num_refine)])
            self.plan_decoder = build(plan_decoder, BBOX_CODERS)
            self.plan_sampler = build(plan_sampler, BBOX_SAMPLERS)
            self.align_sampler = build(align_sampler, BBOX_SAMPLERS)
            self.loss_plan_cls = build(loss_plan_cls, LOSSES)
            self.loss_plan_reg = build(loss_plan_reg, LOSSES)

            self.ego_fut_ts = self.plan_refine[0].ego_fut_ts
            self.ego_fut_cmd = self.plan_refine[0].ego_fut_cmd
            self.ego_fut_mode = self.plan_refine[0].ego_fut_mode
            self.plan_anchor_group = self.plan_instance_bank.anchor_group
            self.plan_anchor_types = self.plan_instance_bank.anchor_types
            self.plan_speed_refer = plan_speed_refer
            self.plan_anchor_refer = plan_anchor_refer
            # 수정(2026-11-05): 사용자 지시 "MCMC SFT 확인한대로 진행" — SFT 정답은 ("temp","2hz") 한 묶음 설정에서만 정의된다(다른 묶음은 GT 를 읽어 정답이 섞이므로 금지)
            self.plan_sft_key = plan_sft_key
            if plan_sft_key is not None:
                _types = [tuple(t) for t in self.plan_anchor_types]
                if _types != [("temp", "2hz")] or tuple(self.plan_anchor_refer) != ("temp", "2hz"):
                    raise ValueError(f"plan_sft_key 는 plan_anchor_types=[('temp','2hz')] 전용: {_types}, refer {self.plan_anchor_refer}")

            if self.with_command_embed:
                self.command_embed_encoder = nn.Sequential(
                    *linear_relu_ln(self.embed_dims, 2, 1, input_dims=num_command),
                    Linear(self.embed_dims, self.embed_dims)
                )

            if self.with_target_point_embed:
                self.target_point_encoder = nn.Sequential(
                    *linear_relu_ln(self.embed_dims, 2, 1),
                    Linear(self.embed_dims, self.embed_dims)
                )

        # 수정(2026-11-05): 사용자 결정 "E2 자차 상태 직접 입력"
        #   새 모듈 초기화가 전역 난수를 소비하지 않게 fork_rng 안에서 만든다(켜도 그 뒤 모듈들의 초기값이 꺼졌을 때와 같다).
        self.ego_input_embed_cfg = ego_input_embed
        if ego_input_embed is not None:
            if 'ego' not in self.query_select:
                raise ValueError("ego_input_embed 는 ego 쿼리가 있을 때만(query_select 에 'ego')")
            _in = int(ego_input_embed["in_dims"])
            _sc = torch.as_tensor(ego_input_embed["scale"], dtype=torch.float32)
            if _sc.shape != (_in,) or not bool((_sc > 0).all()):
                raise ValueError(f"ego_input_embed.scale {ego_input_embed['scale']} 은 in_dims={_in} 길이의 양수여야 한다")
            self.register_buffer("ego_input_scale", _sc, persistent=False)
            with torch.random.fork_rng(devices=[]):
                self.ego_input_encoder = nn.Sequential(
                    nn.Linear(_in, self.embed_dims), nn.ReLU(inplace=True), nn.Linear(self.embed_dims, self.embed_dims))
            nn.init.zeros_(self.ego_input_encoder[2].weight)
            nn.init.zeros_(self.ego_input_encoder[2].bias)

        #수정: VLA-v1 — Qwen token attn 모듈 (task_select 조건과 무관하게 빌드)
        if qwen_token_attn is not None:
            self.qwen_token_attn = build(qwen_token_attn, ATTENTION)
        else:
            self.qwen_token_attn = None

        # 수정(2026-10-10): 잘린 확산 계획(A안) — 계획 앵커 오염 + 확산 시점 임베딩. models/decoder/noise_t/diffusion.py 참조.
        self.plan_diffusion = build(plan_diffusion, PLUGIN_LAYERS) if plan_diffusion is not None else None
        # 수정(2026-10-19): 사용자 결정 — 지도 정렬 선택적 Gaussian 잡음(지도 밖 잡음 제거)
        #   잡음 모듈이 지도 기억을 요구하면(MapAlignedPlanTruncatedDiffusion) 지도 뱅크가 그 기억을 가져야 한다. 조용히 등방으로
        #   되돌리지 않고 즉시 실패시킨다. 종전 모듈(PlanTruncatedDiffusion)은 이 분기와 무관(동작 비트 동일).
        self.plan_diffusion_uses_map = bool(getattr(self.plan_diffusion, "uses_map_memory", False))
        if self.plan_diffusion_uses_map and not ("map" in self.query_select and "map" in self.task_select
                                                 and hasattr(self.map_instance_bank, "plan_map_memory")):
            raise ValueError("plan_diffusion 이 지도 기억을 쓰려면 map 쿼리·과제가 켜져 있고 map_instance_bank 가 "
                             "PlanMapMemoryInstanceBank 여야 한다.")

        if 'motion' in self.task_select:
            # 수정(2026-10-15, 2차): 모션 앵커 적재·모드 쿼리 인코더 구성은 decoder/query_anchors/motion_anchor.py (키 동일)
            self.motion_anchor = load_motion_anchor(motion_anchor)
            self.motion_anchor_encoder = build_motion_anchor_encoder(self.embed_dims)
            self.motion_refine = nn.ModuleList([build(motion_refine_layer, PLUGIN_LAYERS) for _ in range(num_refine)])
            self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
            self.motion_decoder = build(motion_decoder, BBOX_CODERS)
            self.loss_motion_cls = build(loss_motion_cls, LOSSES)
            self.loss_motion_reg = build(loss_motion_reg, LOSSES)

            self.fut_ts = self.motion_refine[0].fut_ts
            self.fut_mode = self.motion_refine[0].fut_mode

        # operation
        self.op_config_map = {
            "concat": [custom_op, PLUGIN_LAYERS],
            "split": [custom_op, PLUGIN_LAYERS],
            "gnn": [graph_model, ATTENTION],
            "temp_gnn": [temp_graph_model, ATTENTION],
            "inter_gnn": [inter_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [custom_op, PLUGIN_LAYERS],
            "refine": [custom_op, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList([build(*self.op_config_map.get(op, [None, None])) for op in self.operation_order])
        # 수정(2026-10-27): 후보 3 진단 — ffn·norm 자리를 쿼리 종류별 모듈 묶음으로 바꾼다(같은 설정으로 종류마다 하나씩 빌드).
        self.task_specific_ffn_norm = bool(task_specific_ffn_norm)
        if self.task_specific_ffn_norm:
            for i, op in enumerate(self.operation_order):
                if op in ("ffn", "norm"):
                    self.layers[i] = nn.ModuleDict(
                        {m: build(*self.op_config_map[op]) for m in self.query_select})

        self.fc_before = nn.Identity()
        self.fc_after = nn.Identity()

        if self.decouple_attn:
            self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
            self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)

        self.run_step = 0
        self.is_init_bank_list = False
        self.combine_layer_loss = combine_layer_loss
        # 수정(2026-10-15): 실험 C — 스트림 그룹 G 개의 뱅크 상태(파이썬 속성: 파라미터·버퍼 아님 → state_dict 불변).
        self.num_stream_groups = int(num_stream_groups)
        assert self.num_stream_groups >= 1
        assert not (self.num_stream_groups > 1 and with_close_loop), "스트림 그룹은 개방루프 학습 전용"
        self._stream_states = None
        self.force_stream_groups = bool(force_stream_groups)  # 수정(2026-11-09): C3 GRPO §9.1(위 인자 설명)
        self.plan_grad_isolation = bool(plan_grad_isolation)  # 수정(2026-11-09): C3 GRPO §9.3(위 인자 설명)

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif 'refine' not in op:
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

        self.init_instance_bank_list()

    def init_instance_bank_list(self):
        self.is_init_bank_list = True

        # open-loop training and evaluation
        if not self.with_close_loop:
            self.bank_length = self.open_loop_bank_length if self.open_loop_bank_length is not None else 1
            if 'det' in self.query_select:
                self.det_instance_bank_list = [self.det_instance_bank]
            if 'map' in self.query_select:
                self.map_instance_bank_list = [self.map_instance_bank]
            if 'ego' in self.query_select:
                self.ego_instance_bank_list = [self.ego_instance_bank]
            if 'plan' in self.query_select:
                self.plan_instance_bank_list = [self.plan_instance_bank]

        # close-loop evaluation
        else:
            self.bank_length = self.close_loop_bank_length if self.close_loop_bank_length is not None \
                else self.close_loop_hz // self.open_loop_hz
            if 'det' in self.task_select:
                self.det_instance_bank_list = [copy.deepcopy(self.det_instance_bank) for _ in range(self.bank_length)]
            if 'map' in self.task_select:
                self.map_instance_bank_list = [copy.deepcopy(self.map_instance_bank) for _ in range(self.bank_length)]
            if 'ego' in self.task_select:
                self.ego_instance_bank_list = [copy.deepcopy(self.ego_instance_bank) for _ in range(self.bank_length)]
            if 'plan' in self.task_select:
                self.plan_instance_bank_list = [copy.deepcopy(self.plan_instance_bank) for _ in range(self.bank_length)]

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar

    def get_motion_anchor(self, classification, prediction):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def graph_model(self, index, query, key=None, value=None, query_pos=None, key_pos=None, **kwargs):
        # 그래프 어텐션 래퍼(SeparateAttention 등)는 과제 구간을 직접 다루므로 항상 독립 호출한다.
        #   (decouple 시 fc_before / fc_after 는 kwargs 로 넘겨 래퍼 안에서 적용된다.)
        return self.layers[index](query, key, value, query_pos=query_pos, key_pos=key_pos, **kwargs)

    def forward(self, img, feature_maps, metas):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        if not self.is_init_bank_list:
            self.init_instance_bank_list()

        deform_i = 0
        refine_i = 0

        batch_size = feature_maps[0].shape[0]
        bank_idx = self.run_step % self.bank_length
        # 수정(2026-11-09): C3 GRPO 설계 §9.3 계획 기울기 격리 [격리] — 플래그가 켜지고 기울기가 켜진 forward 에서만.
        #   ① split 직후 det·map·ego 구간(인스턴스 특징·앵커 임베딩, 시간 구간 포함)을 detach → 그 뒤 deformable·refine·motion 은 그래프 없음.
        #   ② 과제별(ModuleDict) norm·ffn 은 비계획 구간을 no_grad 로 따로 계산(종전도 구간별 호출이라 값 동일).
        #   ③ gnn·inter_gnn·temp_gnn 래퍼는 grad_modalities=("plan",) 로 쿼리에 plan 이 없는 그룹을 no_grad 로 계산(separate_attn._group_ctx).
        #   ④ qwen_attn 은 전체 쿼리를 no_grad 로 계산하고, plan 행만 기울기를 켠 채 따로 계산해 바꿔 끼운다(row_offset; 교차 어텐션은 행 독립).
        #      ego 행은 no_grad 값(갱신된 gate_sep 사용)을 쓴다 — 설계 §9.3. plan 행 값은 따로 부른 계산이라 GEMM 모양 차이만큼(ulp) 다를 수 있다(G2-3).
        #   det·map·motion(rescore 입력) 값은 바뀌지 않는다. E3(task_specific_ffn_norm=True) 전용 — 아니면 오류.
        iso = self.plan_grad_isolation and torch.is_grad_enabled()
        if iso and not self.task_specific_ffn_norm:
            raise RuntimeError("plan_grad_isolation 은 task_specific_ffn_norm=True(E3) 전용 — 공유 norm·ffn 을 구간별로 나누면 값이 달라질 수 있다")
        if iso and "plan" not in self.query_select:
            raise RuntimeError("plan_grad_isolation: query_select 에 plan 이 없다")
        iso_kw = {"grad_modalities": ("plan",)} if iso else {}
        if iso:
            # 수정(2026-11-09): C3 GRPO §9.3 [격리] 전제 검사(검토 A 권장) — 격리는 비계획 구간의 **그래프**만 끊는다. 비계획 경로의
            #   매개변수가 학습 대상이면 그 경로 출력(det·map·motion = rescore 입력)이 RL 갱신으로 바뀌어 "RL 동안 rescore 입력 불변"
            #   (설계 §9.3·§9.7)이 깨진다. 그래서 학습 대상(requires_grad)은 계획 매개변수(tools/navsim/grpo/rollout.TRAINABLE 과 같은 목록)만
            #   허용한다. gate_sep(plan·ego 행 공유 게이트)은 설계 결정으로 포함 — ego 행은 no_grad 값(갱신된 gate_sep 사용, §9.3).
            _bad = [n for n, p in self.named_parameters() if p.requires_grad and not _GRPO_PLAN_PARAM.search(n)]
            if _bad:
                raise RuntimeError(f"plan_grad_isolation: 비계획 매개변수 {len(_bad)} 개가 requires_grad=True {_bad[:5]} — 격리 전제 위반")
        stream_k = self._stream_group_enter(metas, bank_idx)  # 수정(2026-10-15): 실험 C — 그룹 k 뱅크 상태 복원(G=1/추론이면 None, 아무것도 안 함)

        # ========= initial query ============
        with_temp = False

        if "ego" in self.query_select:
            ego_instance_feature, ego_anchor, temp_ego_instance_feature, temp_ego_anchor = \
                self.ego_instance_bank_list[bank_idx].get(batch_size, metas, feature_maps)
            ego_anchor_embed = self.ego_anchor_encoder(ego_anchor)
            temp_ego_anchor_embed = self.ego_anchor_encoder(temp_ego_anchor) if temp_ego_anchor is not None else None
            self.num_ego_anchor = ego_anchor.size(1)
            self.num_temp_ego_anchor = temp_ego_anchor.size(1) if temp_ego_anchor is not None else 0
            if self.ego_input_embed_cfg is not None:  # 수정(2026-11-05): 사용자 결정 "E2 자차 상태 직접 입력"
                _x = metas["ego_input"]
                if _x.dim() != 2 or _x.shape[0] != batch_size or _x.shape[1] != self.ego_input_scale.numel():
                    raise ValueError(f"metas['ego_input'] 모양 {tuple(_x.shape)} != ({batch_size}, {self.ego_input_scale.numel()})")
                _e = self.ego_input_encoder(_x.float() / self.ego_input_scale)
                ego_instance_feature = ego_instance_feature + _e.to(ego_instance_feature.dtype).unsqueeze(1)
        else:
            ego_instance_feature, ego_anchor, ego_anchor_embed = None, None, None
            temp_ego_instance_feature, temp_ego_anchor, temp_ego_anchor_embed = None, None, None
            self.num_ego_anchor, self.num_temp_ego_anchor = 0, 0

        if "det" in self.query_select:
            det_instance_feature, det_anchor, temp_det_instance_feature, temp_det_anchor, time_interval = \
                self.det_instance_bank_list[bank_idx].get(batch_size, metas)
            det_anchor_embed = self.det_anchor_encoder(det_anchor)
            temp_det_anchor_embed = self.det_anchor_encoder(temp_det_anchor) if temp_det_anchor is not None else None
            self.num_det_anchor = det_anchor.size(1)
            self.num_temp_det_anchor = temp_det_anchor.size(1) if temp_det_anchor is not None else 0
        else:
            det_instance_feature, det_anchor, det_anchor_embed = None, None, None
            temp_det_instance_feature, temp_det_anchor, temp_det_anchor_embed = None, None, None
            self.num_det_anchor, self.num_temp_det_anchor = 0, 0

        if "map" in self.query_select:
            #수정(2026-10-08): map 뱅크의 dt 를 별도 변수로 받는다. 같은 이름 time_interval 에 재할당하면
            #  det 뱅크의 실제 dt 를 덮어써(map 은 num_temp_instances=0 이라 상수 0.5) det 속도 디코딩이 어긋난다.
            map_instance_feature, map_anchor, temp_map_instance_feature, temp_map_anchor, map_time_interval = \
                self.map_instance_bank_list[bank_idx].get(batch_size, metas)
            map_anchor_embed, _ = self.map_anchor_encoder(map_anchor)
            temp_map_anchor_embed = self.map_anchor_encoder(temp_map_anchor)[0] if temp_map_anchor is not None else None
            self.num_map_anchor = map_anchor.size(1)
            self.num_temp_map_anchor = temp_map_anchor.size(1) if temp_map_anchor is not None else 0
        else:
            map_instance_feature, map_anchor, map_anchor_embed = None, None, None
            temp_map_instance_feature, temp_map_anchor, temp_map_anchor_embed = None, None, None
            self.num_map_anchor, self.num_temp_map_anchor = 0, 0

        if "plan" in self.query_select:
            plan_instance_feature, plan_anchor, temp_plan_instance_feature, temp_plan_anchor = \
                self.plan_instance_bank_list[bank_idx].get(batch_size, metas, feature_maps)
            # 수정(2026-10-10): 잘린 확산(A안) — 고정 k-means 앵커를 x_t 로 오염시키고 시점 임베딩을 계획 쿼리에 더한다.
            #   시간 계획 메모리(temp_plan_*)는 이전 프레임 예측(x0)이므로 오염시키지 않는다.
            if self.plan_diffusion is not None:
                if self.plan_diffusion_uses_map:  # 수정(2026-10-19): 지도 정렬 선택적 잡음 — 직전 프레임 지도 예측(현재 좌표로 옮김, detach)
                    plan_anchor, plan_t_embed = self.plan_diffusion(
                        plan_anchor, map_memory=self.map_instance_bank_list[bank_idx].plan_map_memory)
                else:
                    plan_anchor, plan_t_embed = self.plan_diffusion(plan_anchor)
                plan_instance_feature = plan_instance_feature + plan_t_embed
            plan_anchor_embed, _ = self.plan_anchor_encoder(plan_anchor)
            temp_plan_anchor_embed = self.plan_anchor_encoder(temp_plan_anchor)[0] if temp_plan_anchor is not None else None
            self.num_plan_anchor = plan_anchor.size(1)
            self.num_temp_plan_anchor = temp_plan_anchor.size(1) if temp_plan_anchor is not None else 0
        else:
            plan_instance_feature, plan_anchor, plan_anchor_embed = None, None, None
            temp_plan_instance_feature, temp_plan_anchor, temp_plan_anchor_embed = None, None, None
            self.num_plan_anchor, self.num_temp_plan_anchor = 0, 0

        if (temp_ego_anchor is not None or
            temp_det_anchor is not None or
            temp_map_anchor is not None or
            temp_plan_anchor is not None):
            with_temp = True

        self.num_anchor_list = []
        for q in self.query_select:
            self.num_anchor_list.append(getattr(self, "num_{}_anchor".format(q)))

        self.num_temp_anchor_list = []
        for q in self.query_select:
            self.num_temp_anchor_list.append(getattr(self, "num_temp_{}_anchor".format(q)))

        self.num_anchor_cumsum = np.cumsum([0] + self.num_anchor_list)
        self.num_temp_anchor_cumsum = np.cumsum([0] + self.num_temp_anchor_list)

        self.total_num_anchor = self.num_anchor_cumsum[-1]
        self.total_num_temp_anchor = self.num_temp_anchor_cumsum[-1]

        self.num_anchor_section = {q: [self.num_anchor_cumsum[i], self.num_anchor_cumsum[i + 1]]
                                   for i, q in enumerate(self.query_select)}
        self.num_temp_anchor_section = {q: [self.num_temp_anchor_cumsum[i], self.num_temp_anchor_cumsum[i + 1]]
                                        for i, q in enumerate(self.query_select)}

        # =================== forward the layers ====================
        det_prediction = []
        det_classification = []
        det_quality = []

        map_prediction = []
        map_classification = []
        map_quality = []

        ego_prediction = []
        ego_classification = []
        ego_status = []

        plan_prediction = []
        plan_classification = []
        plan_status = []

        motion_prediction = []
        motion_classification = []

        #수정(2026-09-15, Run9): 계층별 게이트용 카운터. "refine" 이 각 디코더 계층의 마지막
        #  연산이므로 그 횟수가 곧 현재 계층 번호다. qwen_attn 이 refine 앞에 오므로
        #  qwen_attn 호출 시점의 값이 정확히 해당 계층 인덱스(0..5)가 된다.
        _qwen_layer_idx = 0
        # 수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 2) — 교사 토큰 사영(K/V)을 이 forward 안에서 한 번만 계산해 6 계층이 재사용.
        #   지역 변수라 forward 가 끝나면 사라진다(다음 forward·다른 배치/스트림 그룹과 섞이지 않음). 계산은 첫 qwen_attn 에서.
        _qwen_kv = None
        _qwen_mask_all_valid = None  # 수정(2026-11-09): qwen_mask 패딩 판정도 _qwen_kv 와 함께 forward 당 1 회

        for i, op in enumerate(self.operation_order):
            #수정: VLA-v1 — qwen_attn 은 op_config_map 에 없어 layers[i]=None 이 된다.
            #  None-skip 가드에서 제외해야 forward 가 elif op=="qwen_attn" 핸들러에 도달.
            if self.layers[i] is None and op != "qwen_attn":
                continue

            elif op == "concat":
                anchor_embed_list = []
                instance_feature_list = []
                temp_anchor_embed_list = []
                temp_instance_feature_list = []

                instance_num_list = []
                temp_instance_num_list = []

                for modality in self.query_select:
                    if modality == "ego":
                        cur = (ego_anchor_embed, ego_instance_feature, temp_ego_anchor_embed, temp_ego_instance_feature)
                    elif modality == "det":
                        cur = (det_anchor_embed, det_instance_feature, temp_det_anchor_embed, temp_det_instance_feature)
                    elif modality == "map":
                        cur = (map_anchor_embed, map_instance_feature, temp_map_anchor_embed, temp_map_instance_feature)
                    elif modality == "plan":
                        cur = (plan_anchor_embed, plan_instance_feature, temp_plan_anchor_embed, temp_plan_instance_feature)
                    else:
                        raise NotImplementedError(f"query {modality} is not supported.")
                    m_anchor_embed, m_instance_feature, m_temp_anchor_embed, m_temp_instance_feature = cur
                    anchor_embed_list.append(m_anchor_embed)
                    instance_feature_list.append(m_instance_feature)
                    instance_num_list.append(m_instance_feature.size(1))
                    if with_temp and m_temp_instance_feature is not None:
                        temp_anchor_embed_list.append(m_temp_anchor_embed)
                        temp_instance_feature_list.append(m_temp_instance_feature)
                        temp_instance_num_list.append(m_temp_instance_feature.size(1))
                    else:
                        temp_instance_num_list.append(0)

                anchor_embed = torch.cat(anchor_embed_list, dim=1)
                instance_feature = torch.cat(instance_feature_list, dim=1)

                if with_temp:
                    temp_anchor_embed = torch.cat(temp_anchor_embed_list, dim=1)
                    temp_instance_feature = torch.cat(temp_instance_feature_list, dim=1)
                else:
                    temp_anchor_embed = None
                    temp_instance_feature = None

            elif op == "split":
                instance_num_cumsum = np.cumsum([0] + instance_num_list)
                if iso:  # 수정(2026-11-09): C3 GRPO §9.3 [격리] ① — 비계획 구간 detach(값 동일, 그래프만 끊음)
                    _d = lambda m, x: x if (m == "plan" or x is None) else x.detach()
                else:
                    _d = lambda m, x: x

                for modality, query_start, query_end in zip(
                        self.query_select, instance_num_cumsum[:-1], instance_num_cumsum[1:]):
                    if modality == "ego":
                        ego_anchor_embed = _d(modality, anchor_embed[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①
                        ego_instance_feature = _d(modality, instance_feature[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①
                    elif modality == "det":
                        det_anchor_embed = _d(modality, anchor_embed[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①
                        det_instance_feature = _d(modality, instance_feature[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①
                    elif modality == "map":
                        map_anchor_embed = _d(modality, anchor_embed[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①
                        map_instance_feature = _d(modality, instance_feature[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①
                    elif modality == "plan":
                        plan_anchor_embed = _d(modality, anchor_embed[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①
                        plan_instance_feature = _d(modality, instance_feature[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①

                if with_temp:
                    temp_instance_num_cumsum = np.cumsum([0] + temp_instance_num_list)
                    for modality, query_start, query_end in zip(
                            self.query_select, temp_instance_num_cumsum[:-1], temp_instance_num_cumsum[1:]):
                        if modality == "ego":
                            temp_ego_anchor_embed = _d(modality, temp_anchor_embed[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①
                            temp_ego_instance_feature = _d(modality, temp_instance_feature[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①
                        elif modality == "det":
                            temp_det_anchor_embed = _d(modality, temp_anchor_embed[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①
                            temp_det_instance_feature = _d(modality, temp_instance_feature[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①
                        elif modality == "map":
                            temp_map_anchor_embed = _d(modality, temp_anchor_embed[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①
                            temp_map_instance_feature = _d(modality, temp_instance_feature[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①
                        elif modality == "plan":
                            temp_plan_anchor_embed = _d(modality, temp_anchor_embed[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①
                            temp_plan_instance_feature = _d(modality, temp_instance_feature[:, query_start: query_end])  # 수정(2026-11-09): C3 GRPO §9.3 ①

            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    num_anchor_cumsum=self.num_anchor_cumsum,
                    num_temp_anchor_cumsum=self.num_temp_anchor_cumsum,
                    fc_before=self.fc_before,
                    fc_after=self.fc_after,
                    **iso_kw,  # 수정(2026-11-09): C3 GRPO §9.3 [격리] ③(격리 꺼짐이면 빈 dict = 종전 호출)
                )

            elif op == "gnn" or op == "inter_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    num_anchor_cumsum=self.num_anchor_cumsum,
                    fc_before=self.fc_before,
                    fc_after=self.fc_after,
                    **iso_kw,  # 수정(2026-11-09): C3 GRPO §9.3 [격리] ③(격리 꺼짐이면 빈 dict = 종전 호출)
                )

            elif op == "norm" or op == "ffn":
                if isinstance(self.layers[i], nn.ModuleDict):
                    # 수정(2026-10-27): 후보 3 진단 — 이어 붙인 쿼리를 같은 계층 concat 의 종류별 개수로 나눠 각자의 모듈에 통과
                    assert sum(instance_num_list) == instance_feature.size(1), (instance_num_list, instance_feature.shape)
                    if iso:  # 수정(2026-11-09): C3 GRPO §9.3 [격리] ② — 비계획 구간은 detach 입력·no_grad(같은 모듈·같은 입력 → 값 동일)
                        _parts = []
                        for m, f in zip(self.query_select, torch.split(instance_feature, instance_num_list, dim=1)):
                            if m == "plan":
                                _parts.append(self.layers[i][m](f))
                            else:
                                with torch.no_grad():
                                    _parts.append(self.layers[i][m](f.detach()))
                        instance_feature = torch.cat(_parts, dim=1)
                    else:
                        instance_feature = torch.cat([
                            self.layers[i][m](f) for m, f in zip(
                                self.query_select, torch.split(instance_feature, instance_num_list, dim=1))], dim=1)
                else:
                    instance_feature = self.layers[i](instance_feature)

            #수정: VLA-v1 — Qwen token K/V ↔ unified query Q cross-attn.
            #  metas['qwen_hidden'] (B, L, 1536) bf16, metas['qwen_mask'] (B, L) uint8.
            elif op == "qwen_attn":
                qh = metas.get("qwen_hidden") if isinstance(metas, dict) else None
                #수정(2026-10-02): 조용한 skip 제거(프로젝트 절대 규칙 — Skip 금지).
                #   종전엔 qwen_hidden 이 없으면 아무 말 없이 융합을 건너뛰었다. 그러면
                #   qwen_cache_dir 가 누락·오버라이드된 평가가 **에러 없이 비융합 모델로
                #   측정되고 그 수치가 "LaQ-AD" 로 기록된다**. operation_order 에
                #   qwen_attn 이 있고 모듈이 빌드된 상태라면 토큰 부재는 설정 오류이므로
                #   즉시 드러나야 한다.
                if self.qwen_token_attn is not None and qh is None:
                    raise RuntimeError(
                        "operation_order 에 'qwen_attn' 이 있고 qwen_token_attn 이 "
                        "빌드되었으나 img_metas 에 qwen_hidden 이 없다. "
                        "model.qwen_cache_dir 설정을 확인하라(조용히 건너뛰지 않는다)."
                    )
                if qh is not None and self.qwen_token_attn is not None:
                    qm = metas.get("qwen_mask")
                    # 수정(2026-10-16): 교사 캐시 7 폴더 · 토큰 유형 임베딩 — 토큰별 폴더 번호(있을 때만) 전달
                    qt = metas.get("qwen_type")
                    # 수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 2) — 사영은 첫 계층에서 한 번(project_kv), 이후 계층은 재사용.
                    #   공유 모듈 하나의 같은 가중치·같은 입력이라 계층마다 계산하던 값과 같다(qwen_token_attn.py project_kv 주석).
                    if _qwen_kv is None:
                        _qwen_kv = self.qwen_token_attn.project_kv(qh, qwen_type=qt, dtype=instance_feature.dtype)
                        # 수정(2026-11-09): 학습 속도 — 패딩 유무 판정(bool(mask.all()), 호스트 동기화)을 계층마다(6 회) 하지 않고
                        #   forward 당 한 번만 한다. 6 계층이 같은 qm 텐서를 받으므로 판정값이 같다(kv 와 같은 재사용 규칙).
                        _qwen_mask_all_valid = qm is None or bool(qm.bool().all())
                    _kv, _qt = _qwen_kv
                    if iso:  # 수정(2026-11-09): C3 GRPO §9.3 [격리] ④ — 전체는 no_grad, plan 행만 기울기 켜고 따로(row_offset) 계산해 바꿔 끼움
                        _ps, _pe = (int(v) for v in self.num_anchor_section["plan"])
                        if sum(instance_num_list) != instance_feature.size(1) or \
                                list(np.cumsum([0] + instance_num_list)[[self.query_select.index("plan"),
                                                                         self.query_select.index("plan") + 1]]) != [_ps, _pe]:
                            raise RuntimeError("plan_grad_isolation: qwen_attn 의 plan 구간이 concat 구간과 다르다")
                        with torch.no_grad():
                            _full = self.qwen_token_attn(
                                instance_feature.detach(), qh, qm, layer_idx=_qwen_layer_idx, kv=_kv,
                                mask_all_valid=_qwen_mask_all_valid,  # 수정(2026-11-09): 학습 속도 수정(mask_all_valid)과 손으로 합침
                                **({} if qt is None else {"qwen_type": _qt}))
                        _plan = self.qwen_token_attn(
                            instance_feature[:, _ps:_pe], qh, qm, layer_idx=_qwen_layer_idx, kv=_kv, row_offset=_ps,
                            mask_all_valid=_qwen_mask_all_valid,  # 수정(2026-11-09): 학습 속도 수정(mask_all_valid)과 손으로 합침
                            **({} if qt is None else {"qwen_type": _qt}))
                        instance_feature = torch.cat([_full[:, :_ps], _plan, _full[:, _pe:]], dim=1)
                    else:
                        instance_feature = self.qwen_token_attn(
                            instance_feature, qh, qm, layer_idx=_qwen_layer_idx, kv=_kv,
                            mask_all_valid=_qwen_mask_all_valid,
                            **({} if qt is None else {"qwen_type": _qt}))

            elif op == "deformable":
                for modality in self.query_select:
                    if modality == "ego":
                        ego_instance_feature = self.ego_deformable[deform_i](
                            ego_instance_feature, ego_anchor, ego_anchor_embed, feature_maps, metas)
                    elif modality == "det":
                        det_instance_feature = self.det_deformable[deform_i](
                            det_instance_feature, det_anchor, det_anchor_embed, feature_maps, metas)
                    elif modality == "map":
                        map_instance_feature = self.map_deformable[deform_i](
                            map_instance_feature, map_anchor, map_anchor_embed, feature_maps, metas)
                    elif modality == "plan":
                        plan_instance_feature = self.plan_deformable[deform_i](
                            plan_instance_feature, plan_anchor, plan_anchor_embed, feature_maps, metas)

                deform_i += 1

            elif op == "refine":
                _qwen_layer_idx += 1
                if "det" in self.task_select:
                    det_anchor, det_cls, det_qt = self.det_refine[refine_i](
                        det_instance_feature, det_anchor, det_anchor_embed, time_interval=time_interval,
                        return_cls=True)
                    det_prediction.append(det_anchor)
                    det_classification.append(det_cls)
                    det_quality.append(det_qt)

                    if len(det_prediction) == self.num_single_frame_decoder:
                        det_instance_feature, det_anchor = self.det_instance_bank_list[bank_idx].update(
                            det_instance_feature, det_anchor, det_cls)

                    det_anchor_embed = self.det_anchor_encoder(det_anchor)

                    if len(det_prediction) > self.num_single_frame_decoder and temp_det_anchor_embed is not None:
                        temp_det_anchor_embed = det_anchor_embed[:,
                                                : self.det_instance_bank_list[bank_idx].num_temp_instances]

                if "map" in self.task_select:
                    map_anchor, map_cls, map_qt = self.map_refine[refine_i](
                        map_instance_feature, map_anchor, map_anchor_embed, time_interval=map_time_interval,  #수정(2026-10-08): map 자체 dt
                        return_cls=True)
                    map_prediction.append(map_anchor)
                    map_classification.append(map_cls)
                    map_quality.append(map_qt)

                    if len(map_prediction) == self.num_single_frame_decoder:
                        map_instance_feature, map_anchor = self.map_instance_bank_list[bank_idx].update(
                            map_instance_feature, map_anchor, map_cls)

                    map_anchor_embed, _ = self.map_anchor_encoder(map_anchor)

                    if len(map_prediction) > self.num_single_frame_decoder and temp_map_anchor_embed is not None:
                        temp_map_anchor_embed = map_anchor_embed[:, : self.map_instance_bank_list[bank_idx].num_temp_instances]

                if "motion" in self.task_select:
                    motion_anchor = self.get_motion_anchor(det_cls, det_anchor)
                    motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :]))
                    motion_query = motion_mode_query + (det_instance_feature + det_anchor_embed).unsqueeze(2)
                    motion_cls, motion_reg = self.motion_refine[refine_i](motion_query)

                    motion_classification.append(motion_cls)
                    motion_prediction.append(motion_reg)

                if "ego" in self.task_select:
                    # only predict (supervised) ego status
                    ego_status_ = self.ego_refine[refine_i](ego_instance_feature, ego_anchor_embed)
                    ego_classification.append(None)
                    ego_prediction.append(None)
                    ego_status.append(ego_status_)

                if "plan" in self.task_select:
                    use_plan_anchor_embed = True
                    if self.with_target_point_embed:
                        target_point = metas['target_point'].unsqueeze(1).unsqueeze(1)
                        target_point_embed = self.target_point_encoder(gen_sineembed_for_position(target_point))
                        plan_anchor_embed += target_point_embed.squeeze(1)

                    if self.with_command_embed:
                        ego_command = metas['gt_ego_fut_cmd'].unsqueeze(1).unsqueeze(1)
                        command_embed = self.command_embed_encoder(ego_command)
                        plan_anchor_embed += command_embed.squeeze(1)

                    if self.with_ego_instance_feature:
                        plan_instance_feature = plan_instance_feature + ego_instance_feature
                        plan_anchor_embed = plan_anchor_embed + ego_anchor_embed

                    plan_reg, plan_cls = self.plan_refine[refine_i](
                        plan_instance_feature, plan_anchor, plan_anchor_embed, use_plan_anchor_embed)

                    plan_anchor = plan_reg.clone()
                    bs, nj, _ = plan_reg.shape

                    plan_reg = plan_reg.clone().reshape(bs, 1, nj, self.ego_fut_ts, 2)
                    plan_reg[..., 1:, :] = plan_reg[..., 1:, :] - plan_reg[..., :-1, :]

                    plan_prediction.append(plan_reg)
                    plan_classification.append(plan_cls.reshape(bs, 1, -1))
                    plan_status.append(None)

                    plan_anchor_embed, _ = self.plan_anchor_encoder(plan_anchor)

                refine_i += 1

            else:
                raise NotImplementedError(f"{op} is not supported.")

        det_output = {
            "classification": det_classification,
            "prediction": det_prediction,
            "quality": det_quality,
            "instance_feature": det_instance_feature,
            "anchor_embed": det_anchor_embed,
        }
        map_output = {
            "classification": map_classification,
            "prediction": map_prediction,
            "quality": map_quality,
            "instance_feature": map_instance_feature,
            "anchor_embed": map_anchor_embed,
        }
        ego_output = {
            "classification": ego_classification,
            "prediction": ego_prediction,
            "status": ego_status
        }
        plan_output = {
            "classification": plan_classification,
            "prediction": plan_prediction,
            "status": plan_status,
        }
        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
        }

        # cache current instances for temporal modeling
        for modality in self.query_select:
            if modality == "ego":
                self.ego_instance_bank_list[bank_idx].cache(
                    ego_instance_feature, ego_anchor, metas, feature_maps)
            if modality == "det":
                self.det_instance_bank_list[bank_idx].cache(
                    det_instance_feature, det_anchor, det_cls, metas, feature_maps)
            if modality == "map":
                self.map_instance_bank_list[bank_idx].cache(
                    map_instance_feature, map_anchor, map_cls, metas, feature_maps)
            if modality == "plan":
                self.plan_instance_bank_list[bank_idx].cache(
                    plan_instance_feature, plan_anchor, plan_cls, metas, feature_maps)

        if self.with_instance_id and "det" in self.task_select:
            det_instance_id = self.det_instance_bank_list[bank_idx].get_instance_id(
                det_cls, det_anchor, self.det_decoder.score_threshold)
            det_output["instance_id"] = det_instance_id

        self._stream_group_exit(stream_k, bank_idx)  # 수정(2026-10-15): 실험 C — 그룹 k 뱅크 상태 저장
        self.run_step += 1

        return det_output, map_output, ego_output, plan_output, motion_output

    # 수정(2026-10-15): 실험 C — 스트림 그룹별 시간 뱅크 상태 저장·복원(설명: temporal_bank/stream_state.py).
    def _stream_banks(self, bank_idx):
        banks = {}
        for name in ("ego", "det", "map", "plan"):
            if name in self.query_select:
                banks[name] = getattr(self, f"{name}_instance_bank_list")[bank_idx]
        return banks

    def _stream_group_enter(self, metas, bank_idx):
        if self.num_stream_groups <= 1 or not (self.training or self.force_stream_groups):  # 수정(2026-11-09): C3 GRPO §9.1 force_stream_groups. #이전 or not self.training
            return None
        img_metas = metas.get("img_metas") if isinstance(metas, dict) else None
        if not isinstance(img_metas, (list, tuple)) or not all(isinstance(m, dict) for m in img_metas):
            raise RuntimeError("num_stream_groups>1 인데 metas['img_metas'] 가 dict 목록이 아니다.")
        ks = {m.get("stream_group") for m in img_metas}
        if None in ks or len(ks) != 1:
            # 조용히 넘기지 않는다(no-skip): 그룹 번호 없이 뱅크를 섞으면 시간 상태가 다른 시퀀스로 샌다.
            raise RuntimeError(f"num_stream_groups={self.num_stream_groups} 인데 배치의 stream_group={ks}. "
                               "샘플러(data.num_stream_groups)와 Collect meta_keys 의 'stream_group' 을 확인할 것.")
        k = int(ks.pop())
        if not 0 <= k < self.num_stream_groups:
            raise RuntimeError(f"stream_group={k} 가 범위 밖(G={self.num_stream_groups}).")
        if self._stream_states is None:
            self._stream_states = [None] * self.num_stream_groups
        banks = self._stream_banks(bank_idx)
        if self._stream_states[k] is None:
            self._stream_states[k] = {n: empty_bank_state(b, bank_state_keys(b)) for n, b in banks.items()}
        for n, b in banks.items():
            load_bank_state(b, self._stream_states[k][n])
        return k

    def _stream_group_exit(self, k, bank_idx):
        if k is None:
            return
        self._stream_states[k] = {n: save_bank_state(b, bank_state_keys(b))
                                  for n, b in self._stream_banks(bank_idx).items()}

    @force_fp32(apply_to=("model_outs"))
    def loss(self, det_output, map_output, ego_output, plan_output, motion_output, data):
        losses = dict()

        # 수정(2026-11-09): 학습 속도 — det·map 의 모든 디코더 층 헝가리안 비용을 먼저 계산해 GPU→CPU 를 한 번에 옮긴다
        #   (종전: 층마다 sample() 안에서 .cpu() 1 회 → det 6 + map 6 = 12 회 대기, 대기열이 빈 뒤 CPU 정합·재기동 동안 GPU 정지).
        #   비용·정합·목표 계산 코드는 sampler.prepare/finish 로 나눈 종전 sample() 그대로라 결과(정합 인덱스·목표·가중치)가 같다.
        assign = self._batched_assign(det_output if "det" in self.task_select else None,
                                      map_output if "map" in self.task_select else None, data)
        if "det" in self.task_select:
            loss_det = self.loss_det(det_output, data, assign=assign["det"])
            losses.update(loss_det)
        if "map" in self.task_select:
            loss_map = self.loss_map(map_output, data, assign=assign["map"])
            losses.update(loss_map)
        if "ego" in self.task_select:
            loss_ego = self.loss_ego(ego_output, data)
            losses.update(loss_ego)
        if "motion" in self.task_select:
            loss_motion = self.loss_motion(motion_output, data)
            losses.update(loss_motion)
        if "plan" in self.task_select:
            loss_plan = self.loss_plan(det_output, map_output, motion_output, plan_output, data)
            losses.update(loss_plan)

        return losses

    @force_fp32(apply_to=("det_outs", "map_outs"))
    def _batched_assign(self, det_outs, map_outs, data):
        """# 수정(2026-11-09): det·map 전 층의 헝가리안 정합을 한 번의 GPU→CPU 로 처리한다.
        loss_det/loss_map 과 같은 force_fp32 변환(fp16→fp32)·같은 reg 자르기·같은 sampler 비용 코드를 쓴다 → 층별 sample() 결과와 같다.
        반환 {"det": [(cls_target, reg_target, reg_weights, indices)] * 층, "map": [(cls_target, reg_target, reg_weights)] * 층}
        (해당 과제가 없으면 None)."""
        jobs = []
        if det_outs is not None:
            for cls, reg in zip(det_outs["classification"], det_outs["prediction"]):
                reg = reg[..., : len(self.det_reg_weights)]
                jobs.append(("det", self.det_sampler, self.det_sampler.prepare(
                    cls, reg, data["gt_labels_3d"], data["gt_bboxes_3d"])))
        if map_outs is not None:
            for cls, reg in zip(map_outs["classification"], map_outs["prediction"]):
                reg = reg[..., : len(self.map_reg_weights)]
                jobs.append(("map", self.map_sampler, self.map_sampler.prepare(
                    cls, reg, data["gt_map_labels"], data["gt_map_pts"])))
        flat = gather_to_host([c for _, _, st in jobs for c in st["costs"]])  # 동기화 1 회(비용이 하나도 없으면 0 회)
        out = {"det": [] if det_outs is not None else None, "map": [] if map_outs is not None else None}
        off = 0
        for task, sampler, st in jobs:
            n = len(st["costs"])
            res = sampler.finish(st, flat[off:off + n]); off += n
            out[task].append(res + ((sampler.indices,) if task == "det" else ()))
        return out

    @force_fp32(apply_to=("model_outs"))
    def loss_det(self, model_outs, data, assign=None):
        quality = model_outs["quality"]
        reg_preds = model_outs["prediction"]
        cls_scores = model_outs["classification"]

        output = {}
        #수정(2026-10-08): 계층별 헝가리안 매칭 결과를 보관한다. det_sampler.indices 는 sample()
        #  호출마다 덮어써져 마지막 계층 것만 남으므로, loss_motion 이 이를 그대로 쓰면 계층 k 의
        #  모션 쿼리가 다른 GT 의 미래 궤적으로 지도된다. det/motion 은 같은 refine 단계에서
        #  1:1 로 append 되므로 계층 번호로 짝지을 수 있다.
        self._det_layer_indices = []
        for decoder_idx, (cls, reg, qt) in enumerate(zip(cls_scores, reg_preds, quality)):
            reg = reg[..., : len(self.det_reg_weights)]
            if assign is not None:  # 수정(2026-11-09): _batched_assign 이 미리 계산한 이 층의 정합(sample() 과 같은 값)
                cls_target, reg_target, reg_weights, self.det_sampler.indices = assign[decoder_idx]
            else:
                cls_target, reg_target, reg_weights = self.det_sampler.sample(
                    cls, reg, data["gt_labels_3d"], data["gt_bboxes_3d"])
            self._det_layer_indices.append(self.det_sampler.indices)
            reg_target = reg_target[..., : len(self.det_reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = reduce_mean(torch.sum(mask).to(dtype=reg.dtype)).clamp(min=1.0)  # 수정(2026-10-10): max(텐서,1.0) 의 파이썬 비교(동기화) 제거, 값 동일

            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(mask, cls.max(dim=-1).values.sigmoid() > threshold)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            #수정(2026-09-04): 클래스별 가중치를 표본 단위 weight 로 전달한다.
            #  FocalLoss.forward(pred, target, weight, avg_factor) 의 weight 는 (N,) 로
            #  브로드캐스트되므로 손실의 형태·정규화는 그대로 두고 기여도만 재조정한다.
            if getattr(self, 'det_cls_class_weights', None) is not None:
                idx = cls_target.clamp(min=0, max=self._det_cls_w.numel() - 1)
                cls_w = self._det_cls_w.to(cls.dtype)[idx]
                cls_loss = self.loss_det_cls(cls, cls_target, weight=cls_w, avg_factor=num_pos)
            else:
                cls_loss = self.loss_det_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            # 수정(2026-10-10): 동기화 제거 — 같은 불리언 마스크 인덱싱(매번 nonzero 동기화)을 인덱스 1회 계산으로, 상수는 캐시(값 동일)
            pos_idx = mask.nonzero(as_tuple=True)[0]
            reg_weights = reg_weights * const_like(reg, self.det_reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[pos_idx]
            reg = reg.flatten(end_dim=1)[pos_idx]
            reg_weights = reg_weights.flatten(end_dim=1)[pos_idx]
            reg_target = torch.where(reg_target.isnan(), const_like(reg, 0.0), reg_target)
            cls_target = cls_target[pos_idx]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[pos_idx]

            reg_loss = self.loss_det_reg(reg, reg_target, weight=reg_weights, avg_factor=num_pos,
                                         prefix="det_", suffix=f"_{decoder_idx}", quality=qt, cls_target=cls_target)

            if self.combine_layer_loss:
                if "det_loss_cls" not in output:
                    output["det_loss_cls"] = 0.0
                if "det_loss_box" not in output:
                    output["det_loss_box"] = 0.0
                    output["det_loss_cns"] = 0.0
                    output["det_loss_yns"] = 0.0
                output["det_loss_cls"] += cls_loss
                output["det_loss_box"] += reg_loss[f"det_loss_box_{decoder_idx}"]
                output["det_loss_cns"] += reg_loss[f"det_loss_cns_{decoder_idx}"]
                output["det_loss_yns"] += reg_loss[f"det_loss_yns_{decoder_idx}"]
            else:
                output[f"det_loss_cls_{decoder_idx}"] = cls_loss
                output.update(reg_loss)

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_map(self, model_outs, data, assign=None):
        quality = model_outs["quality"]
        reg_preds = model_outs["prediction"]
        cls_scores = model_outs["classification"]

        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(zip(cls_scores, reg_preds, quality)):
            reg = reg[..., : len(self.map_reg_weights)]
            if assign is not None:  # 수정(2026-11-09): _batched_assign 이 미리 계산한 이 층의 정합(sample() 과 같은 값)
                cls_target, reg_target, reg_weights = assign[decoder_idx]
            else:
                cls_target, reg_target, reg_weights = self.map_sampler.sample(
                    cls, reg, data["gt_map_labels"], data["gt_map_pts"])

            reg_target = reg_target[..., : len(self.map_reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = reduce_mean(torch.sum(mask).to(dtype=reg.dtype)).clamp(min=1.0)  # 수정(2026-10-10): max(텐서,1.0) 의 파이썬 비교(동기화) 제거, 값 동일

            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(mask, cls.max(dim=-1).values.sigmoid() > threshold)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_map_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            # 수정(2026-10-10): 동기화 제거 — 같은 불리언 마스크 인덱싱(매번 nonzero 동기화)을 인덱스 1회 계산으로, 상수는 캐시(값 동일)
            pos_idx = mask.nonzero(as_tuple=True)[0]
            reg_weights = reg_weights * const_like(reg, self.map_reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[pos_idx]
            reg = reg.flatten(end_dim=1)[pos_idx]
            reg_weights = reg_weights.flatten(end_dim=1)[pos_idx]
            reg_target = torch.where(reg_target.isnan(), const_like(reg, 0.0), reg_target)
            cls_target = cls_target[pos_idx]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[pos_idx]

            reg_loss = self.loss_map_reg(reg, reg_target, weight=reg_weights, avg_factor=num_pos,
                                         prefix="map_", suffix=f"_{decoder_idx}", quality=qt, cls_target=cls_target)

            if self.combine_layer_loss:
                if "map_loss_cls" not in output:
                    output["map_loss_cls"] = 0.0
                if "map_loss_line" not in output:
                    output["map_loss_line"] = 0.0
                output["map_loss_cls"] += cls_loss
                output["map_loss_line"] += reg_loss[f"map_loss_line_{decoder_idx}"]
            else:
                output[f"map_loss_cls_{decoder_idx}"] = cls_loss
                output.update(reg_loss)

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_ego(self, model_outs, data):
        status_preds = model_outs["status"]
        output = {}
        for decoder_idx, status in enumerate(status_preds):
            # only predict (supervised) ego status
            gt_ego_status = data['ego_status']
            ego_status_weight = data['ego_status_mask']
            status_loss = self.loss_ego_status(status.squeeze(1), gt_ego_status, weight=ego_status_weight)
            status_loss = torch.nan_to_num(status_loss)

            if self.combine_layer_loss:
                if "ego_loss_status" not in output:
                    output["ego_loss_status"] = 0.0
                output["ego_loss_status"] += status_loss
            else:
                output[f"ego_loss_status_{decoder_idx}"] = status_loss

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data):
        reg_preds = model_outs["prediction"]
        cls_scores = model_outs["classification"]

        output = {}
        #수정(2026-10-08): 계층마다 그 계층의 det 매칭을 쓴다(loss_det 의 _det_layer_indices 참조).
        layer_indices = getattr(self, "_det_layer_indices", None)
        assert layer_indices is not None and len(layer_indices) == len(reg_preds), \
            "loss_motion 은 loss_det 이후, det 과 같은 계층 수로 호출되어야 한다"
        for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):
            motion_loss_cache = dict(indices=layer_indices[decoder_idx])
            cls_target, cls_weight, reg_pred, reg_target, reg_weight, num_pos = self.motion_sampler.sample(
                reg, data["gt_agent_fut_trajs"], data["gt_agent_fut_masks"], motion_loss_cache)

            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.loss_motion_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.loss_motion_reg(reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos)

            if self.combine_layer_loss:
                if "motion_loss_cls" not in output:
                    output["motion_loss_cls"] = 0.0
                if "motion_loss_reg" not in output:
                    output["motion_loss_reg"] = 0.0
                output["motion_loss_cls"] += cls_loss
                output["motion_loss_reg"] += reg_loss
            else:
                output[f"motion_loss_cls_{decoder_idx}"] = cls_loss
                output[f"motion_loss_reg_{decoder_idx}"] = reg_loss

        return output

    def align_plan_traj_loss(self,
                             ref_pred_cls, ref_pred_reg,
                             ref_gt_trajs, ref_gt_masks,
                             pred_cls, pred_reg,
                             gt_trajs, gt_masks, data, fixed_target=None):
        # 수정(2026-10-16): F1 계획 정답 모드 = GT 최근접 깨끗한 앵커(DiffusionDrive 방식)
        #   fixed_target=(ref_target, ref_cls_weight) 가 주어지면 층별 출력 기반 정답 대신 그것을 쓴다.
        if fixed_target is None:
            _, ref_target, ref_cls_weight, _, _, _ = self.plan_sampler.sample(
                ref_pred_cls, ref_pred_reg, ref_gt_trajs, ref_gt_masks, data)
        else:
            ref_target, ref_cls_weight = fixed_target

        cls, _, _, reg_pred, reg_target, reg_weight = self.align_sampler.sample(
            pred_cls, pred_reg, gt_trajs, gt_masks, data, ref_target)

        cls = cls.flatten(end_dim=1)
        ref_target = ref_target.flatten(end_dim=1)
        ref_cls_weight = ref_cls_weight.flatten(end_dim=1)
        cls_loss = self.loss_plan_cls(cls, ref_target, weight=ref_cls_weight)

        reg_weight = reg_weight.flatten(end_dim=1)
        reg_pred = reg_pred.cumsum(dim=-2)
        reg_target = reg_target.cumsum(dim=-2)
        reg_pred = reg_pred.flatten(end_dim=1)
        reg_target = reg_target.flatten(end_dim=1)
        reg_weight = reg_weight.unsqueeze(-1)
        reg_loss = self.loss_plan_reg(reg_pred, reg_target, weight=reg_weight)

        return cls_loss, reg_loss

    def align_speed_traj_loss(self,
                              ref_pred_cls, ref_pred_reg,
                              ref_gt_trajs, ref_gt_masks,
                              ref_speed_trajs, ref_speed_masks,
                              pred_cls_list, pred_reg_list,
                              gt_trajs, gt_masks, speed_areas, data, fixed_target=None):
        if fixed_target is None:
            _, ref_target, _, _, _, _ = self.plan_sampler.sample(
                ref_pred_cls, ref_pred_reg, ref_gt_trajs, ref_gt_masks, data)
        else:  # 수정(2026-10-16): F1 계획 정답 모드 = GT 최근접 깨끗한 앵커(DiffusionDrive 방식)
            ref_target = fixed_target[0]

        aligned_cls_list = []
        aligned_reg_list = []
        bs_indices = torch.arange(ref_target.shape[0], device=ref_target.device)
        for pred_cls, pred_reg in zip(pred_cls_list, pred_reg_list):
            aligned_cls, _, _, aligned_reg, _, _ = self.align_sampler.sample(
                pred_cls, pred_reg, gt_trajs, gt_masks, data, ref_target)
            aligned_cls = aligned_cls.squeeze(1)[bs_indices, ref_target.squeeze(-1)][:, None, None]
            aligned_reg = aligned_reg[:, :, None]
            aligned_cls_list.append(aligned_cls)
            aligned_reg_list.append(aligned_reg)

        aligned_cls = torch.cat(aligned_cls_list, dim=-1)
        aligned_reg = torch.cat(aligned_reg_list, dim=-3)

        cls, cls_target, cls_weight, reg_pred, reg_target, reg_weight = self.speed_sample(
            ref_speed_trajs, ref_speed_masks, aligned_cls, aligned_reg, gt_trajs, gt_masks, speed_areas, data)

        cls = cls.flatten(end_dim=1)
        cls_target = cls_target.flatten(end_dim=1)
        cls_weight = cls_weight.flatten(end_dim=1)
        cls_loss = self.loss_plan_cls(cls, cls_target, weight=cls_weight)

        reg_weight = reg_weight.flatten(end_dim=1)
        reg_pred = reg_pred.cumsum(dim=-2)
        reg_target = reg_target.cumsum(dim=-2)
        reg_pred = reg_pred.flatten(end_dim=1)
        reg_target = reg_target.flatten(end_dim=1)
        reg_weight = reg_weight.unsqueeze(-1)
        reg_loss = self.loss_plan_reg(reg_pred, reg_target, weight=reg_weight)

        return cls_loss, reg_loss

    def speed_sample(self, ref_speed_trajs, ref_speed_masks, cls_pred, reg_pred, gt_trajs, gt_masks, speed_areas, data):
        ego_fut_mode = cls_pred.shape[2]
        gt_trajs = gt_trajs.unsqueeze(1)
        gt_masks = gt_masks.unsqueeze(1)

        ref_speed_trajs = ref_speed_trajs.unsqueeze(1)
        ref_speed_masks = ref_speed_masks.unsqueeze(1)

        bs, _, num_traj = cls_pred.shape
        bs_indices = torch.arange(bs, device=reg_pred.device)

        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1) if self.ego_fut_cmd > 1 else 0

        cls_pred = cls_pred.reshape(bs, self.ego_fut_cmd, 1, ego_fut_mode)
        reg_pred = reg_pred.reshape(bs, self.ego_fut_cmd, 1, ego_fut_mode, self.ego_fut_ts, 2)

        cls_pred = cls_pred[bs_indices, cmd]
        reg_pred = reg_pred[bs_indices, cmd]

        target_dist = torch.linalg.norm(ref_speed_trajs, dim=-1)
        target_dist_sum = target_dist.sum(-1)
        gt_masks_sum = ref_speed_masks.sum(-1)
        refer_interval = 1 / float(self.plan_speed_refer[1].split('hz')[0])
        gt_speed = target_dist_sum / (gt_masks_sum * refer_interval + 1e-4)

        mode_idx = torch.ones_like(gt_speed, dtype=torch.long)
        for speed_idx, (start, end) in enumerate(speed_areas):
            speed_mask = torch.logical_and(gt_speed >= start, gt_speed < end)
            mode_idx[speed_mask] = speed_idx

        cls_target = mode_idx
        cls_weight = ref_speed_masks.any(dim=-1)
        tmp_mode_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, self.ego_fut_ts, 2)
        best_reg = torch.gather(reg_pred, 2, tmp_mode_idx).squeeze(2)

        return cls_pred, cls_target, cls_weight, best_reg, gt_trajs, gt_masks

    def get_pred_trajs(self, cls, reg, type):
        num_group = self.plan_anchor_group
        num_mode = reg.size(2) // num_group

        pred_index = self.plan_anchor_types.index(type)
        pred_start = self.ego_fut_cmd * num_mode * pred_index
        pred_end = self.ego_fut_cmd * num_mode * (pred_index + 1)
        return cls[:, :, pred_start:pred_end], reg[:, :, pred_start:pred_end]

    def get_gt_trajs(self, data, type):
        # temporal and speed waypoints
        if type[0] == "temp" or type[0] == "speed":
            gt_trajs = data['gt_ego_fut_trajs_{}'.format(type[1])]
            gt_masks = data['gt_ego_fut_masks_{}'.format(type[1])]
            # 수정(2026-11-05): 사용자 지시 "MCMC SFT 확인한대로 진행" — 옵션일 때만 계획 정답 궤적을 SFT 키로(마스크 불변)
            if getattr(self, "plan_sft_key", None) is not None and type[0] == "temp" and type[1] == "2hz":
                gt_trajs = data[self.plan_sft_key]
        # spatial waypoints
        elif type[0] == "spat":
            gt_trajs = data['gt_ego_spat_trajs_{}'.format(type[1])]
            gt_masks = data['gt_ego_spat_masks_{}'.format(type[1])]
        else:
            raise NotImplementedError

        return gt_trajs, gt_masks

    # 수정(2026-10-16): F1 계획 정답 모드 = GT 최근접 깨끗한 앵커(DiffusionDrive 방식)
    def get_clean_anchor_plan_target(self, gt_trajs, gt_masks, data):
        """깨끗한(잡음 전) 계획 앵커 중 GT 명령 그룹에서 GT 와 가장 가까운 모드를 정답으로 고른다.
        gt_trajs (B,T,2) 시점별 변위, gt_masks (B,T). 앵커는 누적 좌표 (N_all, T*2), 순서 = [앵커종류][명령][모드].
        거리 = 유효 시점(마스크)만의 누적 위치 L2 평균. 반환 (ref_target (B,1) long, cls_weight (B,1) bool) —
        SparsePlanTarget.sample 이 내는 ref_target / cls_weight 와 같은 모양이다."""
        bs = gt_trajs.shape[0]
        anchor = self.plan_instance_bank.anchor.detach().float()
        num_mode_all = anchor.shape[0] // self.plan_anchor_group
        idx = self.plan_anchor_types.index(self.plan_anchor_refer)
        anchor = anchor[num_mode_all * idx: num_mode_all * (idx + 1)]
        anchor = anchor.reshape(self.ego_fut_cmd, -1, self.ego_fut_ts, 2)  # (cmd, mode, T, 2) 누적
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1) if self.ego_fut_cmd > 1 else torch.zeros(
            bs, dtype=torch.long, device=gt_trajs.device)
        anchor = anchor[cmd]  # (B, mode, T, 2)
        gt_cum = gt_trajs.float().cumsum(dim=-2)  # (B, T, 2)
        mask = gt_masks.float()
        dist = torch.linalg.norm(anchor - gt_cum.unsqueeze(1), dim=-1)  # (B, mode, T)
        dist = (dist * mask.unsqueeze(1)).sum(-1) / mask.sum(-1, keepdim=True).clamp(min=1.0)
        ref_target = dist.argmin(dim=-1).unsqueeze(1)  # (B, 1)
        cls_weight = (gt_masks > 0).any(dim=-1).unsqueeze(1)  # (B, 1) — 미래 GT 가 전혀 없으면 손실 0
        return ref_target, cls_weight

    @force_fp32(apply_to=("plan_output"))
    def loss_plan(self,
                  det_output,
                  map_output,
                  motion_output,
                  plan_output,
                  data):
        reg_preds = plan_output["prediction"]
        cls_scores = plan_output["classification"]

        output = {}
        # 수정(2026-10-16): F1 계획 정답 모드 = GT 최근접 깨끗한 앵커(DiffusionDrive 방식)
        #   정답 모드는 층과 무관하게 한 번만 정한다(DiffusionDrive 와 같음).
        fixed_target = None
        if self.plan_target == "clean_anchor":
            ref_gt_trajs0, ref_gt_masks0 = self.get_gt_trajs(data, self.plan_anchor_refer)
            fixed_target = self.get_clean_anchor_plan_target(ref_gt_trajs0, ref_gt_masks0, data)
        for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):

            temp_cls_loss = 0.0
            temp_reg_loss = 0.0
            spat_cls_loss = 0.0
            spat_reg_loss = 0.0
            speed_cls_loss = 0.0
            speed_reg_loss = 0.0

            # select reference anchor
            ref_pred_cls, ref_pred_reg = self.get_pred_trajs(cls, reg, self.plan_anchor_refer)
            ref_gt_trajs, ref_gt_masks = self.get_gt_trajs(data, self.plan_anchor_refer)

            speed_dict = dict()
            for anchor_type in self.plan_anchor_types:
                pred_cls, pred_reg = self.get_pred_trajs(cls, reg, anchor_type)
                gt_trajs, gt_masks = self.get_gt_trajs(data, anchor_type)

                if anchor_type[0] in ["temp", "spat"]:
                    cls_loss, reg_loss = self.align_plan_traj_loss(
                        ref_pred_cls, ref_pred_reg, ref_gt_trajs, ref_gt_masks, pred_cls, pred_reg, gt_trajs, gt_masks, data,
                        fixed_target=fixed_target)  # 수정(2026-10-16): F1

                    if anchor_type[0] == "temp":
                        temp_cls_loss += cls_loss
                        temp_reg_loss += reg_loss
                    elif anchor_type[0] == "spat":
                        spat_cls_loss += cls_loss
                        spat_reg_loss += reg_loss

                elif anchor_type[0] == "speed":
                    if anchor_type[1] not in speed_dict:
                        speed_dict[anchor_type[1]] = {
                            "pred_cls": [pred_cls],
                            "pred_reg": [pred_reg],
                            "gt_trajs": gt_trajs,
                            "gt_masks": gt_masks,
                            "speed_areas": [anchor_type[2]]
                        }
                    else:
                        assert (speed_dict[anchor_type[1]]["gt_trajs"] == gt_trajs).all()
                        assert (speed_dict[anchor_type[1]]["gt_masks"] == gt_masks).all()
                        speed_dict[anchor_type[1]]["pred_cls"].append(pred_cls)
                        speed_dict[anchor_type[1]]["pred_reg"].append(pred_reg)
                        speed_dict[anchor_type[1]]["speed_areas"].append(anchor_type[2])

            # speed loss
            if len(speed_dict):
                for k, v in speed_dict.items():
                    ref_speed_trajs, ref_speed_masks = self.get_gt_trajs(data, self.plan_speed_refer)
                    cls_loss, reg_loss = self.align_speed_traj_loss(
                        ref_pred_cls, ref_pred_reg, ref_gt_trajs, ref_gt_masks, ref_speed_trajs, ref_speed_masks,
                        v["pred_cls"], v["pred_reg"], v["gt_trajs"], v["gt_masks"], v["speed_areas"], data,
                        fixed_target=fixed_target,  # 수정(2026-10-16): F1
                    )
                    speed_cls_loss += cls_loss
                    speed_reg_loss += reg_loss

            loss_types = []
            for anchor_type in self.plan_anchor_types:
                if anchor_type[0] not in loss_types:
                    loss_types.append(anchor_type[0])

            if self.combine_layer_loss:
                if "temp" in loss_types:
                    if "plan_loss_temp_cls" not in output:
                        output["plan_loss_temp_cls"] = 0.0
                    if "plan_loss_temp_reg" not in output:
                        output["plan_loss_temp_reg"] = 0.0
                    output["plan_loss_temp_cls"] += temp_cls_loss
                    output["plan_loss_temp_reg"] += temp_reg_loss
                if "spat" in loss_types:
                    if "plan_loss_spat_cls" not in output:
                        output["plan_loss_spat_cls"] = 0.0
                    if "plan_loss_spat_reg" not in output:
                        output["plan_loss_spat_reg"] = 0.0
                    output["plan_loss_spat_cls"] += spat_cls_loss
                    output["plan_loss_spat_reg"] += spat_reg_loss
                if "speed" in loss_types:
                    if "plan_loss_speed_cls" not in output:
                        output["plan_loss_speed_cls"] = 0.0
                    if "plan_loss_speed_reg" not in output:
                        output["plan_loss_speed_reg"] = 0.0
                    output["plan_loss_speed_cls"] += speed_cls_loss
                    output["plan_loss_speed_reg"] += speed_reg_loss

            else:
                if "temp" in loss_types:
                    output[f"plan_loss_temp_cls_{decoder_idx}"] = temp_cls_loss
                    output[f"plan_loss_temp_reg_{decoder_idx}"] = temp_reg_loss
                if "spat" in loss_types:
                    output[f"plan_loss_spat_cls_{decoder_idx}"] = spat_cls_loss
                    output[f"plan_loss_spat_reg_{decoder_idx}"] = spat_reg_loss
                if "speed" in loss_types:
                    output[f"plan_loss_speed_cls_{decoder_idx}"] = speed_cls_loss
                    output[f"plan_loss_speed_reg_{decoder_idx}"] = speed_reg_loss

        return output

    @force_fp32(apply_to=("det_output", "map_output", "ego_output", "plan_output", "motion_output"))
    def post_process(self, det_output, map_output, ego_output, plan_output, motion_output, data, output_idx=-1):
        det_result, map_result, ego_result, plan_result, motion_result = None, None, None, None, None

        if "det" in self.task_select:
            det_result = self.det_decoder.decode(
                det_output["classification"], det_output["prediction"],
                det_output.get("instance_id"), det_output.get("quality"), output_idx=output_idx)
        if "map" in self.task_select:
            map_result = self.map_decoder.decode(
                map_output["classification"], map_output["prediction"],
                map_output.get("instance_id"), map_output.get("quality"), output_idx=output_idx)
        if "motion" in self.task_select:
            motion_result = self.motion_decoder.decode(
                det_output["classification"], det_output["prediction"],
                det_output.get("instance_id"), det_output.get("quality"), motion_output)
        # 자차 출력은 상태 감독용이라 후처리 결과가 없다(ego_result=None).
        if "plan" in self.task_select:
            plan_result = self.plan_decoder.decode(
                ego_output, det_output, motion_output, plan_output, data)

        return det_result, map_result, ego_result, plan_result, motion_result
