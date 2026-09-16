#수정: VLA-v0 — Qwen2-VL-2B 기반 trajectory generator.
#  데이터 파이프라인에서 받은 (qwen_pixel_values, qwen_input_ids, qwen_attention_mask, qwen_image_grid_thw) 를
#  Qwen2-VL forward → 마지막 hidden state → 작은 MLP projection → (B, T, 2) waypoint.
#  이렇게 하면 end-to-end gradient 가 cross-attn → trajectory_proj → Qwen LLM 까지 흐름.
#  텍스트 생성 / parsing 없음 → 학습 안정성, 결정성, 속도 모두 ↑.
#  학습 시 추가로 LM next-token CE loss (gt_qwen_traj_text) 를 보조 supervision 으로 줄 수 있음 (선택).
#  transformers >= 4.45 + Qwen2-VL-2B-Instruct 가중치 필요. 없으면 stub 모드 (zero trajectory 출력) 로 graceful.
import os
import warnings

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models import HEADS


@HEADS.register_module()
class QwenTrajectoryGenerator(BaseModule):
    """Qwen2-VL-2B-Instruct 를 trajectory generator 로 wrap.

    Args:
        qwen_model_path (str): HF id 또는 로컬 경로. None 이면 stub 모드.
        embed_dim (int): Qwen hidden size (Qwen2-VL-2B = 1536).
        traj_ts (int): 출력 trajectory step 수 (HiP-AD ego_fut_ts = 6).
        traj_dim (int): waypoint 좌표 차원 (2 = (x, y) ego frame).
        freeze_qwen (bool): Qwen 본체 freeze. True 면 projection 만 학습 — v0 default.
        torch_dtype (str): 'fp16' 또는 'bf16'. Qwen2-VL 은 보통 bf16.
        use_lora (bool): LoRA wrap. v0 는 False (projection 만).
        lora_r (int): LoRA rank.
        lora_alpha (int): LoRA alpha.
        gradient_checkpointing (bool): 메모리 절약.
        init_cfg: mmcv init config.
    """

    def __init__(self,
                 qwen_model_path=None,
                 embed_dim=1536,
                 traj_ts=6,
                 traj_dim=2,
                 freeze_qwen=True,
                 torch_dtype="bf16",
                 use_lora=False,
                 lora_r=16,
                 lora_alpha=32,
                 gradient_checkpointing=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.qwen_model_path = qwen_model_path
        self.embed_dim = embed_dim
        self.traj_ts = traj_ts
        self.traj_dim = traj_dim
        self.freeze_qwen = freeze_qwen
        self.torch_dtype_str = torch_dtype
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.gradient_checkpointing = gradient_checkpointing

        self._stub = False
        self.qwen = None
        self._try_load_qwen()

        # Trajectory projection head: hidden_state → (T * 2)
        # 학습 안정성을 위해 traj_dim 별로 출력 후 reshape.
        self.traj_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, traj_ts * traj_dim),
        )
        # 좋은 초기화: 마지막 linear 의 weight/bias 를 0 근처로 → 초기 trajectory ≈ 0 (stable warm start).
        nn.init.zeros_(self.traj_proj[-1].weight)
        nn.init.zeros_(self.traj_proj[-1].bias)

    @property
    def torch_dtype(self):
        return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[self.torch_dtype_str]

    def _try_load_qwen(self):
        if self.qwen_model_path is None:
            warnings.warn("QwenTrajectoryGenerator: qwen_model_path=None → stub 모드.")
            self._stub = True
            return
        # 로컬 경로 우선 (다운로드된 가중치)
        path = self.qwen_model_path
        if not os.path.isdir(path) and not path.startswith("Qwen/"):
            warnings.warn(f"QwenTrajectoryGenerator: path '{path}' 가 디렉토리 아님. HF id 로 시도.")
        try:
            from transformers import Qwen2VLForConditionalGeneration
            # torch 1.13 (HiP-AD env) 은 SDPA 미지원 → eager attention. torch >= 2.1.1 면 sdpa/flash_attention_2 가능.
            import torch as _torch
            attn_impl = "sdpa" if _torch.__version__ >= "2.1.1" else "eager"
            self.qwen = Qwen2VLForConditionalGeneration.from_pretrained(
                path, torch_dtype=self.torch_dtype, attn_implementation=attn_impl
            )
            if self.gradient_checkpointing:
                self.qwen.gradient_checkpointing_enable()
            if self.freeze_qwen:
                for p in self.qwen.parameters():
                    p.requires_grad_(False)
                self.qwen.eval()
            if self.use_lora:
                self._apply_lora()
        except Exception as e:
            warnings.warn(f"QwenTrajectoryGenerator: Qwen 로드 실패 ({type(e).__name__}: {e}). stub 모드.")
            self.qwen = None
            self._stub = True

    def _apply_lora(self):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            warnings.warn("peft 미설치 — LoRA 비활성, 기본 freeze 만.")
            return
        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.qwen = get_peft_model(self.qwen, lora_cfg)

    def _zero_trajectory(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.traj_ts, self.traj_dim, device=device, dtype=dtype)

    def forward(self, qwen_pixel_values, qwen_input_ids, qwen_attention_mask,
                qwen_image_grid_thw=None, **kwargs):
        """
        Args:
            qwen_pixel_values: (B, num_imgs=6, 3, H, W) 또는 Qwen processor 가 만든 flat 형태.
            qwen_input_ids:    (B, L)
            qwen_attention_mask: (B, L)
            qwen_image_grid_thw: (B, 6, 3) 또는 (sum_imgs, 3)

        Returns:
            trajectory: (B, traj_ts, traj_dim) — metas['external_trajectory'] 로 주입됨.
        """
        # tensor 정규화 (numpy 면 변환)
        if not torch.is_tensor(qwen_input_ids):
            return self._zero_trajectory(
                batch_size=qwen_input_ids.shape[0] if hasattr(qwen_input_ids, "shape") else 1,
                device=self.traj_proj[0].weight.device,
                dtype=self.traj_proj[0].weight.dtype,
            )

        batch_size = qwen_input_ids.shape[0]
        device = qwen_input_ids.device

        if self._stub or self.qwen is None:
            # stub: zero trajectory — 학습/평가는 가능, cross-attn 분기는 의미있는 값 못 받지만 shape 일관성 유지
            return self._zero_trajectory(batch_size, device, self.traj_proj[0].weight.dtype)

        # Qwen forward — Qwen2VLForConditionalGeneration.forward 가 pixel_values + image_grid_thw 받음.
        # output_hidden_states=True 로 hidden_states tuple 받고 last layer 사용.
        outputs = self.qwen(
            input_ids=qwen_input_ids,
            attention_mask=qwen_attention_mask,
            pixel_values=qwen_pixel_values.to(self.torch_dtype) if qwen_pixel_values is not None else None,
            image_grid_thw=qwen_image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )
        # outputs.hidden_states: tuple of (B, L, H) per layer. last = -1 = LM output 직전.
        hidden = outputs.hidden_states[-1]  # (B, L, H)
        valid_len = qwen_attention_mask.sum(dim=1) - 1   # (B,)
        b_idx = torch.arange(batch_size, device=device)
        last_token_hidden = hidden[b_idx, valid_len]      # (B, H)

        # projection 은 fp32 로 (안정성). last_token_hidden 은 bf16/fp16 일 수 있음.
        proj_in = last_token_hidden.to(self.traj_proj[0].weight.dtype)
        traj_flat = self.traj_proj(proj_in)               # (B, T*2)
        trajectory = traj_flat.reshape(batch_size, self.traj_ts, self.traj_dim)
        return trajectory

    @torch.no_grad()
    def run_hidden(self, qwen_pixel_values, qwen_input_ids, qwen_attention_mask,
                   qwen_image_grid_thw=None, **kwargs):
        """수정: VLA-v1 (B2D base, 캐시 비현실) — Qwen LM 마지막 layer hidden 전체를 반환.
        token-fusion(qwen_attn / plan_qwen_token_attn)의 K/V 로 metas['qwen_hidden'] 에 주입용.
        Returns: (hidden (B, L, H) bf16, mask (B, L) uint8)  — stub 이면 (None, None)."""
        if self._stub or self.qwen is None or not torch.is_tensor(qwen_input_ids):
            return None, None
        outputs = self.qwen(
            input_ids=qwen_input_ids,
            attention_mask=qwen_attention_mask,
            pixel_values=qwen_pixel_values.to(self.torch_dtype) if qwen_pixel_values is not None else None,
            image_grid_thw=qwen_image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]                       # (B, L, H)
        mask = qwen_attention_mask.to(torch.uint8)               # (B, L)
        return hidden, mask

    def __repr__(self):
        return (f"{self.__class__.__name__}(stub={self._stub}, "
                f"qwen_path={self.qwen_model_path}, freeze={self.freeze_qwen}, "
                f"traj_ts={self.traj_ts}, traj_dim={self.traj_dim})")
