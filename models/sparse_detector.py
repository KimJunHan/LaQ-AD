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
from .grid_mask import GridMask

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
        scenes_tokenizer=None,
        qwen_traj_generator=None,  #수정: VLA-v0 — Qwen2-VL-2B trajectory generator
        qwen_cache_dir=None,        #수정: VLA-v1 — pre-computed Qwen hidden state cache dir
        qwen_inject_hidden=False,   #수정: VLA-v1(B2D base) — live Qwen hidden 을 metas['qwen_hidden'] 주입
    ):
        super(SparseDetector, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
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
        if scenes_tokenizer is not None:
            self.scenes_tokenizer = build_from_cfg(scenes_tokenizer, PLUGIN_LAYERS)
        else:
            self.scenes_tokenizer = None
        #수정: VLA-v0 — Qwen trajectory generator. config 미제공 시 None (cross-attn 분기 skip).
        if qwen_traj_generator is not None:
            self.qwen_traj_generator = build_head(qwen_traj_generator)
        else:
            self.qwen_traj_generator = None
        #수정: VLA-v1 — Qwen cache 디렉토리 저장 (per-sample .pt 로드용)
        self.qwen_cache_dir = qwen_cache_dir
        #수정: VLA-v1(B2D base) — 캐시 대신 live Qwen forward 로 hidden 주입할지
        self.qwen_inject_hidden = qwen_inject_hidden
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


    def extract_scenes(self, img, feature_maps, data):
        scenes_tokens, scenes_embeds = self.scenes_tokenizer(img, feature_maps)
        data['scenes_tokens'] = scenes_tokens
        data['scenes_embeds'] = scenes_embeds
        data['temp_scenes_tokens'] = self.scenes_tokenizer.cached_scenes_tokens
        data['temp_scenes_embeds'] = self.scenes_tokenizer.cached_scenes_embeds

        # extract future feats
        if "fut_img" in data and self.training:
            fut_img = data["fut_img"]
            fut_mask = data["fut_mask"]
            fut_data = {key.split("fut_")[-1]: value for key, value in data.items() if key.startswith("fut_")}
            with torch.no_grad():
                fut_feature_maps, fut_depths = self.extract_feat(fut_img, True, fut_data)
                fut_data = self.extract_scenes(fut_img, fut_feature_maps, fut_data)
            data["fut_mask"] = fut_mask
            data["fut_scenes_tokens"] = fut_data['scenes_tokens']
            data["fut_scenes_embeds"] = fut_data['scenes_embeds']

        return data

    def extract_fut_feat(self, img, feature_maps, data):
        fut_img = data["fut_img"]
        fut_mask = data["fut_mask"]
        fut_data = {key.split("fut_")[-1]: value for key, value in data.items() if key.startswith("fut_")}
        with torch.no_grad():
            fut_feature_maps = self.extract_feat(fut_img, False, fut_data)
        data["fut_mask"] = fut_mask
        data["fut_feature_maps"] = fut_feature_maps
        return data

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def _maybe_run_qwen(self, data):
        """수정: VLA-v0 — Qwen2-VL-2B 가 (B,T,2) trajectory 생성 → data['external_trajectory'] 주입.
        Qwen 모듈 없거나 입력 키 미존재 시 graceful skip (cross-attn 분기는 None 체크로 자동 스킵)."""
        if self.qwen_traj_generator is None:
            return data
        required = ("qwen_pixel_values", "qwen_input_ids", "qwen_attention_mask")
        if any(k not in data for k in required):
            return data
        trajectory = self.qwen_traj_generator(
            qwen_pixel_values=data["qwen_pixel_values"],
            qwen_input_ids=data["qwen_input_ids"],
            qwen_attention_mask=data["qwen_attention_mask"],
            qwen_image_grid_thw=data.get("qwen_image_grid_thw"),
        )
        data["external_trajectory"] = trajectory   # (B, T, 2)
        #수정: VLA-v1(B2D base) — 캐시 없이 live Qwen LM hidden 을 token-fusion K/V 로 주입.
        if self.qwen_inject_hidden and "qwen_hidden" not in data:
            qh, qm = self.qwen_traj_generator.run_hidden(
                qwen_pixel_values=data["qwen_pixel_values"],
                qwen_input_ids=data["qwen_input_ids"],
                qwen_attention_mask=data["qwen_attention_mask"],
                qwen_image_grid_thw=data.get("qwen_image_grid_thw"),
            )
            if qh is not None:
                data["qwen_hidden"] = qh
                data["qwen_mask"] = qm
        return data

    def _maybe_load_qwen_cache(self, data):
        """수정: VLA-v1 — per-sample Qwen hidden state cache 디스크 로드 → data['qwen_kv'].
        token 은 metas 또는 img_metas 에서 추출. 모든 sample 의 cache 있어야 batch.
        """
        if not self.qwen_cache_dir:
            return data
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
            p = _os.path.join(self.qwen_cache_dir, f"{tok}.pt")
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

    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        if "fut_img" in data and self.training:
            data = self.extract_fut_feat(img, feature_maps, data)
        #수정: VLA-v0 — Qwen trajectory 주입
        data = self._maybe_run_qwen(data)
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

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)
        data = self._maybe_run_qwen(data)        #수정: VLA-v0
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
