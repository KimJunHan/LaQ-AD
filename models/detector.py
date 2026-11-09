# 수정(2026-10-15): 경로 이동 models/sparse_detector.py → models/detector.py 및 정리(아래 __init__ 주석 참조).
#   구성: 영상 백본(mmdet ResNet) → 넥(mmdet FPN) [+ neck/DenseDepthNet 보조 깊이] → head/SparseHead(통합 디코더).
from inspect import signature

import torch

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .encoder.image_resnet50_fpn.grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["SparseDetector"]


@DETECTORS.register_module()
class SparseDetector(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
        qwen_cache_dir=None,        #수정: VLA-v1 — pre-computed Qwen hidden state cache dir
        reset_bank_on_scene_change=False,  #수정(2026-10-09): NAVSIM — 추론 시 scene_token 이 바뀌면 시간 뱅크 초기화(기본 False = 기존 동작)
        bev_encoders=None,  # 수정(2026-10-15): B 융합 — LiDAR/Radar 기둥 BEV 인코더 cfg 리스트. None(기본) = 종전과 같음(모듈 미생성)
        neck_dtype=None,  # 수정(2026-10-20): fp16 넘침 원인 수정 — "bf16" 이면 FPN(neck)을 bfloat16 autocast 로 계산(None = 종전과 같음)
        **kwargs,
    ):
        super(SparseDetector, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        # 수정(2026-10-20): LaQ-AD 1단계 iter ~7,500 에서 FPN 출력 합성곱(img_neck.fpn_convs.1)의 출력이 fp16 최대 65,504 를 넘어 Inf
        #   (직전 최대 4.98e4→6.25e4, 입력은 최대 515 로 정상) → 디코더 분류 손실 NaN. 백본은 fp16 그대로 두고 neck 만 지수 범위가
        #   fp32 와 같은 bfloat16 으로 계산해 넘침 자체를 없앤다(사용자 결정: FPN bf16, 처음부터 재학습). 진단: results/diag_nan_v2t_s1_20261020.
        assert neck_dtype in (None, "bf16"), f"neck_dtype 은 None 또는 'bf16' (받은 값 {neck_dtype!r})"
        self.neck_dtype = neck_dtype
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        # 수정(2026-10-15): 소스 정리 — 어떤 config 도 쓰지 않던 scenes_tokenizer, VLA-v0 qwen_traj_generator /
        #   qwen_inject_hidden(라이브 Qwen 궤적·은닉 주입), 보조 캐시 혼합(qwen_cache_dir_aux / qwen_aux_anneal_iters)
        #   을 제거했다. 켜는 설정이 들어오면 조용히 무시하지 않고 즉시 실패한다.
        _removed_keys = ("scenes_tokenizer", "qwen_traj_generator", "qwen_inject_hidden",
                         "qwen_cache_dir_aux", "qwen_aux_anneal_iters")
        _unknown = sorted(set(kwargs) - set(_removed_keys))
        if _unknown:
            raise TypeError(f"SparseDetector: 알 수 없는 인자 {_unknown}")
        _removed = sorted(k for k in _removed_keys if kwargs.get(k))
        if _removed:
            raise ValueError(f"SparseDetector: 2026-10-15 소스 정리에서 제거된 기능이 켜져 있다: {_removed}")
        #수정: VLA-v1 — Qwen cache 디렉토리 저장 (per-sample .pt 로드용)
        self.qwen_cache_dir = qwen_cache_dir
        #수정(2026-10-09): NAVSIM navtest 는 장면(이력 4프레임)들이 같은 로그에서 겹친 창으로 나온다.
        #   시간 뱅크의 리셋은 |dt|<=2s 게이트뿐이라(models/decoder/temporal_bank/instance_bank.py get()), 다음 장면 첫 프레임이
        #   직전 장면(시간상 미래일 수도 있는) 인스턴스를 물려받는다. scene_token 으로 명시 초기화한다.
        #   기본 False — nuScenes/B2D 동작과 기존 체크포인트 평가는 바뀌지 않는다.
        self.reset_bank_on_scene_change = bool(reset_bank_on_scene_change)
        self._last_scene_token = None
        # 수정(2026-10-15): B 융합 — 점 → BEV 특징. 인코더별 출력 수준을 순서대로 이어 data['fusion_bev_feats'] 로 넘기고,
        #   DeformableFeatureAggregation(bev_num_levels>0) 이 metas 에서 읽어 키포인트를 BEV 에 표본추출한다.
        #   None 이면 속성을 만들지 않는다(state_dict·난수열 불변).
        if bev_encoders:
            self.bev_encoders = torch.nn.ModuleList([build_backbone(c) for c in bev_encoders])
        else:
            self.bev_encoders = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        if self.img_neck is not None:
            if self.neck_dtype == "bf16" and img.is_cuda and getattr(self, "fp16_enabled", False):
                # 수정(2026-10-20): fp16 넘침 원인 수정 — neck 은 bfloat16(지수 범위 = fp32).
                # 수정(2026-10-20): 무결점 검사 1 지적(치명) — mmdet FPN.forward 의 @auto_fp16 이 wrap_fp16_model 뒤 fp16_enabled=True 가 되어
                #   안쪽에서 autocast(fp16)를 다시 열어 bf16 을 덮어썼다. 이 블록 안에서만 neck 의 fp16_enabled 를 끄고(장식이 그대로 통과)
                #   끝나면 원래 값으로 되돌린다. fp16 학습일 때만 적용(fp32 학습에서 정밀도가 오히려 떨어지는 것 방지, 지적 2).
                _prev = getattr(self.img_neck, "fp16_enabled", False)
                self.img_neck.fp16_enabled = False
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        feature_maps = list(self.img_neck(feature_maps))
                finally:
                    self.img_neck.fp16_enabled = _prev
                feature_maps = [f.float() for f in feature_maps]   # 이후(깊이 가지·디코더)는 종전처럼 fp32 로 받는다
            else:
                feature_maps = list(self.img_neck(feature_maps))
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
        if return_depth:
            return feature_maps, depths
        return feature_maps


    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def _maybe_load_qwen_cache(self, data):
        """수정: VLA-v1 — per-sample Qwen hidden state cache 디스크 로드 → data['qwen_kv'].
        token 은 metas 또는 img_metas 에서 추출. 모든 sample 의 cache 있어야 batch.
        """
        if not self.qwen_cache_dir:
            return data
        # 수정(2026-10-10): 데이터 워커가 이미 읽어 온 경우(파이프라인 LoadQwenCacheNusc) 그대로 쓴다 — 값·dtype 은 아래 경로와 동일
        #   (bf16 hidden, uint8 mask, 배치 내 L 축 0 패딩). 메인 프로세스의 NAS 순차 읽기·표본별 H2D 동기화가 사라진다.
        if "qwen_hidden" in data:
            data["qwen_hidden"] = data["qwen_hidden"].squeeze(1).to(dtype=torch.bfloat16)  # (B,1,L,H)→(B,L,H)
            data["qwen_mask"] = data["qwen_mask"].squeeze(1).to(dtype=torch.uint8)
            # 수정(2026-10-16): 교사 캐시 7 폴더 · 토큰 유형 임베딩 — LoadTeacherCacheNusc 의 토큰별 폴더 번호 (B,1,L)→(B,L)
            if "qwen_type" in data:
                data["qwen_type"] = data["qwen_type"].squeeze(1).long()
            return data
        if "qwen_type" in data:
            raise RuntimeError("qwen_type 이 있는데 qwen_hidden 이 없다 — 파이프라인 Collect keys 확인(교사 캐시 경로).")
        import os as _os
        import torch as _torch
        # token 추출 — batch 내 "모든" sample.
        #수정: VLA-FULL — 기존엔 metas[0] 1개만 뽑아 qwen_hidden batch=1 → batch>1 학습 시
        #  query(batch=B) 와 flash-attn batch 차원 불일치(assert). (batch=1 스모크는 1==1 로 우연 통과.)
        #  → metas 전체에서 sample 순서대로 token 수집.
        tokens = data.get("token") or data.get("frame_token")
        if tokens is None:
            metas = data.get("img_metas") or data.get("metas")
            if isinstance(metas, list) and len(metas) > 0:
                if all(isinstance(m, dict) for m in metas):
                    tokens = [m.get("token") for m in metas]
                elif isinstance(metas[0], (list, tuple)):  # [[m0, m1, ...]] 중첩 형태
                    tokens = [m.get("token") for m in metas[0] if isinstance(m, dict)]
        #수정(2026-08-11): 조용한 skip 제거. 일부 배치만 언어 없이 학습되면 결과가 재현되지
        #   않고 ON/OFF ablation 도 성립하지 않는다(프로젝트 규칙: root-cause only, skip 금지).
        #   qwen_cache_dir 를 지정했다는 것은 언어 융합을 켜겠다는 뜻이므로, 조건이 갖춰지지
        #   않으면 조용히 넘기지 말고 즉시 실패시킨다.
        if tokens is None:
            raise RuntimeError(
                "qwen_cache_dir 가 설정됐는데 metas 에서 sample token 을 찾지 못했다. "
                "Collect 의 meta_keys 에 'token' 이 포함되어 있는지 확인할 것.")
        if not isinstance(tokens, (list, tuple)):
            tokens = [tokens]
        if any(t is None for t in tokens):
            raise RuntimeError(
                f"배치 내 token 이 None 인 표본이 있다(batch={len(tokens)}). "
                "일부만 언어 없이 학습되는 상황을 허용하지 않는다.")
        hidden_list, mask_list = [], []
        max_L = 0
        for tok in tokens:
            _dir = self.qwen_cache_dir
            p = _os.path.join(_dir, f"{tok}.pt")
            if not _os.path.exists(p):
                #수정(2026-08-11): cache miss 조용한 skip 제거(위와 같은 이유).
                raise FileNotFoundError(
                    f"Qwen 캐시 누락: {p}. tools/qwen_cache.py 로 해당 분할 전체를 먼저 생성할 것. "
                    f"(2026-08-11 확인: data/nuscenes/qwen 에 train 28130 + val 6019 = 34149 완비)")
            d = _torch.load(p, map_location="cpu")
            hidden_list.append(d["hidden"])
            mask_list.append(d["attn_mask"])
            max_L = max(max_L, d["hidden"].shape[0])
        # pad to max_L
        H = hidden_list[0].shape[1]
        device = next(self.parameters()).device
        qwen_hidden = _torch.zeros(len(hidden_list), max_L, H, dtype=_torch.bfloat16, device=device)
        qwen_mask = _torch.zeros(len(hidden_list), max_L, dtype=_torch.uint8, device=device)
        for i, (h, m) in enumerate(zip(hidden_list, mask_list)):
            L = h.shape[0]
            qwen_hidden[i, :L] = h.to(device)
            qwen_mask[i, :L] = m.to(device)
        data["qwen_hidden"] = qwen_hidden
        data["qwen_mask"] = qwen_mask
        return data

    def _maybe_extract_bev(self, data):
        """수정(2026-10-15): B 융합 — 인코더마다 data[points_key](표본별 점 텐서 리스트) → BEV 맵들을 이어 붙인다."""
        if self.bev_encoders is None:
            return data
        feats = []
        for enc in self.bev_encoders:
            pts = data[enc.points_key]
            if isinstance(pts, torch.Tensor):  # 배치 1 이 텐서로 풀린 경우
                pts = [pts]
            feats.extend(enc([p.float() for p in pts]))
        data["fusion_bev_feats"] = feats
        return data

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        data = self._maybe_extract_bev(data)  # 수정(2026-10-15): B 융합(bev_encoders 없으면 그대로)
        #수정: VLA-v1 — Qwen cache K/V 디스크 로드 → metas
        data = self._maybe_load_qwen_cache(data)
        model_outs = self.head(img, feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def _maybe_reset_banks(self, data):
        """수정(2026-10-09): scene_token 이 바뀌면 모든 시간 인스턴스 뱅크를 비운다(opt-in)."""
        if not self.reset_bank_on_scene_change:
            return
        metas = data.get("img_metas")
        if isinstance(metas, list) and len(metas) > 0 and isinstance(metas[0], (list, tuple)):
            metas = metas[0]
        if not isinstance(metas, list) or not all(isinstance(m, dict) and "scene_token" in m for m in metas):
            raise RuntimeError("reset_bank_on_scene_change=True 인데 img_metas 에 scene_token 이 없다 "
                               "(test pipeline Collect meta_keys 에 'scene_token' 필요).")
        toks = {m["scene_token"] for m in metas}
        if len(toks) != 1:
            # 배치 안에 장면이 섞이면 표본별 리셋이 불가하다 — 순차 추론(batch 1)만 허용.
            raise RuntimeError(f"한 배치에 장면 {len(toks)}개가 섞였다. 추론은 samples_per_gpu=1 로 돌릴 것.")
        tok = toks.pop()
        if tok != self._last_scene_token:
            dec = getattr(self.head, "onedecoder_head", None)
            if dec is None:
                raise RuntimeError("onedecoder_head 가 없는 head 에서는 뱅크 초기화를 지원하지 않는다.")
            # 개방루프는 *_instance_bank_list 가 같은 객체 1개, 폐쇄루프는 복사본 여러 개(decoder/unified_decoder.py init_instance_bank_list)
            #   → 목록과 원본을 모두 비운다.
            for name in ("det_instance_bank", "map_instance_bank", "ego_instance_bank", "plan_instance_bank"):
                banks = list(getattr(dec, name + "_list", []) or []) + [getattr(dec, name, None)]
                for bank in banks:
                    if bank is not None:
                        bank.reset()
            self._last_scene_token = tok

    def simple_test(self, img, **data):
        self._maybe_reset_banks(data)  #수정(2026-10-09): NAVSIM 장면 경계 초기화(opt-in)
        feature_maps = self.extract_feat(img)
        data = self._maybe_extract_bev(data)  # 수정(2026-10-15): B 융합(bev_encoders 없으면 그대로)
        data = self._maybe_load_qwen_cache(data) #수정: VLA-v1
        model_outs = self.head(img, feature_maps, data)
        results = self.head.post_process(model_outs, data)

        output = []
        for result in results:
            out_dict = {}
            if 'metric_results' in result:
                metric_results = result.pop('metric_results')
                out_dict['metric_results'] = metric_results
            out_dict['img_bbox'] = result
            output.append(out_dict)

        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
