# 수정(2026-10-15, 2차): projector — VL 토큰 1536→512→256 사영 MLP 구성 함수. 모듈 객체는 QwenTokenAttn.qwen_token_proj 로 소유된다
#   (state_dict 키 head.onedecoder_head.qwen_token_attn.qwen_token_proj.* 를 바꾸지 않기 위해).
from .qwen_projector import build_qwen_projector

__all__ = ["build_qwen_projector"]
