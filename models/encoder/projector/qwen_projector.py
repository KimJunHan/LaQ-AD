# 수정(2026-10-15, 2차): 아키텍처 그림의 "projector" 상자. QwenTokenAttn 안에 있던 VL 토큰 사영 MLP 구성을 함수로 분리했다.
#   1536(Qwen2-VL-2B 은닉) → proj_hidden(512) → GELU → embed_dims(256) → LayerNorm.
#   nn.Sequential 을 그대로 돌려주므로 QwenTokenAttn.qwen_token_proj 의 state_dict 키(qwen_token_proj.0/.2/.3.*)와
#   파라미터 생성 순서(난수 소비)는 종전과 같다. 모듈 객체는 QwenTokenAttn(decoder/language_fusion) 이 소유한다.
import torch.nn as nn

__all__ = ["build_qwen_projector"]


def build_qwen_projector(qwen_hidden_dim=1536, proj_hidden=512, embed_dims=256):
    return nn.Sequential(
        nn.Linear(qwen_hidden_dim, proj_hidden),
        nn.GELU(),
        nn.Linear(proj_hidden, embed_dims),
        nn.LayerNorm(embed_dims),
    )
