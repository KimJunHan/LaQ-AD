# 수정(2026-10-15): 경로 이동 models/plan/diffusion.py → models/decoder/noise_t/diffusion.py (소스 정리, 동작 동일).
# 수정(2026-10-10): 언어 조건 잘린 확산 계획(사용자 결정 A안) — DiffusionDrive(Liao et al., CVPR 2025)의
#   "앵커 + 잘린 확산 일정" 을 HiP-AD 단일 디코더의 계획 쿼리에 이식한다.
#   - 학습: k-means 계획 앵커(누적 궤적)를 정규화 공간에서 t ~ U{0..trunc_steps-1} 의 DDPM 전방 과정으로 오염시킨다.
#           x_t = sqrt(ab_t)·a + sqrt(1-ab_t)·eps. 디코더의 계단식 refine 층이 x_t 에서 깨끗한 궤적(x0)을 예측한다
#           (기존 계획 손실 = 최근접 모드 회귀 + 분류 를 그대로 쓴다 → x0 예측 목적과 같다).
#   - 조건: 매 디코더 층의 qwen_attn(게이트 교차 어텐션)이 계획 쿼리에 언어 토큰을 넣으므로 디노이징이 언어 조건을 받는다.
#           확산 시점 t 는 사인 임베딩 → MLP(마지막 층 0 초기화) 로 계획 쿼리 특징에 더한다(시작 시 시점 임베딩만 0 — 앵커는 첫 스텝부터 오염되므로 출력 동일은 아님, 수정 2026-10-14).
#   - 추론: infer_t 시점으로 오염시킨 앵커에서 x0 를 한 번에 예측(DDIM 1 스텝). 잡음은 고정 시드 생성기로 만들어
#           평가를 재현 가능하게 한다. 여러 스텝 DDIM 은 디코더 전체 재실행이 필요하고 시간 인스턴스 뱅크 상태
#           (cache/update)가 한 프레임에 두 번 갱신되므로 이번 구현에 넣지 않았다.
import math
import torch
from torch import nn
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

__all__ = ["PlanTruncatedDiffusion"]


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


@PLUGIN_LAYERS.register_module()
class PlanTruncatedDiffusion(nn.Module):
    def __init__(self, embed_dims=256, ego_fut_ts=6, num_train_timesteps=1000, trunc_steps=50, infer_t=None,
                 beta_start=1e-4, beta_end=0.02, norm_scale=(15.0, 35.0), infer_seed=0, norm_offset=(0.0, 0.0)):
        # 수정(2026-10-12): norm_offset — DiffusionDrive 의 norm_odo 처럼 (x - offset)/scale 로 데이터 범위
        #   [min,max] 를 [-1,1] 에 사상한다. 전진축은 항상 양수라 대칭 정규화(offset 0)는 범위 절반을 낭비하고
        #   같은 t 에서 잡음(m)을 키운다. 기본값 (0,0) 이면 종전과 동일.
        super().__init__()
        self.embed_dims, self.ego_fut_ts = embed_dims, ego_fut_ts
        self.trunc_steps = int(trunc_steps)
        self.infer_t = self.trunc_steps - 1 if infer_t is None else int(infer_t)
        assert 0 <= self.infer_t < num_train_timesteps and 0 < self.trunc_steps <= num_train_timesteps
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float64)
        ab = torch.cumprod(1.0 - betas, 0)
        self.register_buffer("sqrt_ab", ab.sqrt().float(), persistent=False)
        self.register_buffer("sqrt_1mab", (1.0 - ab).sqrt().float(), persistent=False)
        self.register_buffer("norm_scale", torch.tensor(norm_scale, dtype=torch.float32), persistent=False)
        self.register_buffer("norm_offset", torch.tensor(norm_offset, dtype=torch.float32), persistent=False)
        self.infer_seed = infer_seed
        self.t_mlp = nn.Sequential(nn.Linear(embed_dims, embed_dims), nn.SiLU(), nn.Linear(embed_dims, embed_dims))
        nn.init.zeros_(self.t_mlp[-1].weight); nn.init.zeros_(self.t_mlp[-1].bias)

    def forward(self, anchor):
        """anchor: (B, N, T*2) 누적 궤적(lidar, m). 반환: 오염된 앵커 (B, N, T*2), 시점 임베딩 (B, 1, C)."""
        B, N, D = anchor.shape
        a = (anchor.reshape(B, N, self.ego_fut_ts, 2) - self.norm_offset) / self.norm_scale
        if self.training:
            t = torch.randint(0, self.trunc_steps, (B,), device=anchor.device)
            eps = torch.randn_like(a)
        else:
            t = torch.full((B,), self.infer_t, device=anchor.device, dtype=torch.long)
            g = torch.Generator(device=anchor.device); g.manual_seed(self.infer_seed)
            eps = torch.randn(a.shape, generator=g, device=anchor.device, dtype=a.dtype)
        x_t = self.sqrt_ab[t].view(B, 1, 1, 1) * a + self.sqrt_1mab[t].view(B, 1, 1, 1) * eps
        x_t = (x_t * self.norm_scale + self.norm_offset).reshape(B, N, D)
        t_emb = self.t_mlp(timestep_embedding(t, self.embed_dims)).unsqueeze(1)
        return x_t, t_emb
