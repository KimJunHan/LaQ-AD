#수정: VLA-v1 — Qwen2-VL hidden state sequence (L, 1536) 를 HiP-AD query 와 cross-attention 으로 fuse.
#  Qwen 본체는 forward 안 함 — pre-computed cache (data/qwen_cache/<token>.pt) 에서 로드.
#  학습 가능 모듈:
#    - qwen_token_proj : Linear(1536 → embed_dims=256) + LayerNorm
#    - cross_attn      : MultiheadFlashAttention (모든 query type 공유)
#    - gate            : 채널별 스칼라, 0 초기화 (아래 [1] 참조)
#  task_select 의 모든 query 가 같은 cross-attn 통과 → enriched feature.
#
#수정(2026-08-11): 세 가지 결함을 근본 수정. 셋 다 "새 신호를 공유 표현에 그대로 주입하면
#  이미 학습된 능력이 잠식된다"는 같은 실패 양상과 관련이 있다. 실제로 stage2 에서 모션
#  과제를 그렇게 주입했다가 추적 AMOTA 가 23.7% 떨어진 전례가 있다.
#
#  [1] 0 초기화 게이트 부재.
#      이전 구현은 `return query + out` 으로 잔차를 처음부터 전강도로 더했다. 학습 초기의
#      cross-attn 출력은 무작위이므로, 사전학습된 인스턴스 특징을 즉시 교란한다.
#      채널별 게이트를 0 으로 초기화하면 step 0 의 출력이 기반 모델과 **정확히 동일**하고,
#      언어 정보는 손실을 낮추는 만큼만 점진적으로 반영된다. ablation 의 ON/OFF 비교도
#      이때 비로소 성립한다 — OFF 가 ON 의 초기 상태와 같아야 차이를 융합 효과로 귀속할 수 있다.
#
#  [2] qwen_mask 미사용.
#      인자로 받기만 하고 어텐션에 넘기지 않아 패딩 토큰까지 참조했다. 배치 내 문장 길이가
#      다르면 짧은 표본일수록 의미 없는 패딩에 어텐션이 분산된다. key_padding_mask 로 넘긴다.
#
#  [3] 캐시 누락 시 조용한 skip.
#      호출부가 cache miss 를 graceful skip 으로 처리하면, 일부 표본만 언어 없이 학습되어
#      결과가 재현되지 않는다. 프로젝트 규칙(root-cause only, skip 금지)에 어긋난다.
#      캐시 로더가 누락을 명시적으로 알리고, 호출부가 실패하도록 한다.

import os
import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.utils import build_from_cfg
from mmcv.runner import BaseModule


@ATTENTION.register_module()
class QwenTokenAttn(BaseModule):
    """Qwen token K/V ↔ HiP-AD query Q cross-attention (0 초기화 게이트 잔차).

    Forward: query (B, N, embed_dims), qwen_hidden (B, L, qwen_hidden_dim),
             qwen_mask (B, L) — 1=유효 토큰, 0=패딩
    Returns: query + gate * attn(query, kv, kv)
    """

    def __init__(self, embed_dims=256, num_heads=8, attn_drop=0.1, proj_drop=0.1,
                 qwen_hidden_dim=1536, gate_init=0.0, attn_cfg=None, init_cfg=None,
                 #수정(2026-09-15, Run9): 게이트 분해와 사영 강화 — 셋 다 선택적이며
                 #  기본값이면 종전과 완전히 동일하게 동작한다(하위 호환).
                 #  num_layers>0  : 계층별 게이트(A). 언어가 유용한 깊이는 계층마다 다르다.
                 #  seg_bounds    : 과제별 게이트(B). 실측 이득이 지도 +30.8% vs 검출 +9.7% 로
                 #                  크게 달라, 과제마다 주입 강도를 따로 배우게 한다.
                 #                  [det_end, map_end] 형식(그 뒤는 plan/ego 구간).
                 #  proj_hidden>0 : K/V 사영을 2층 MLP 로(C). 1536->256 6배 압축을 단층에
                 #                  맡기던 병목을 완화한다.
                 num_layers=0, seg_bounds=None, proj_hidden=0):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        if proj_hidden and proj_hidden > 0:
            self.qwen_token_proj = nn.Sequential(
                nn.Linear(qwen_hidden_dim, proj_hidden),
                nn.GELU(),
                nn.Linear(proj_hidden, embed_dims),
                nn.LayerNorm(embed_dims),
            )
        else:
            self.qwen_token_proj = nn.Sequential(
                nn.Linear(qwen_hidden_dim, embed_dims),
                nn.LayerNorm(embed_dims),
            )
        self.num_layers = int(num_layers or 0)
        self.seg_bounds = list(seg_bounds) if seg_bounds else None
        self.num_segs = (len(self.seg_bounds) + 1) if self.seg_bounds else 1
        if attn_cfg is None:
            attn_cfg = dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                batch_first=True,
            )
        self.attn = build_from_cfg(attn_cfg, ATTENTION)
        # [1] 채널별 게이트. gate_init=0 이면 초기 출력이 기반 모델과 동일하다.
        #수정(Run9): (계층 L x 과제 S x 채널 d) 로 분해. L=S=1 이면 종전과 동일 형상 의미.
        #  전 원소 0 초기화이므로 분해 여부와 무관하게 step 0 출력은 기반 모델과 같다.
        L = max(self.num_layers, 1); S = self.num_segs
        self.gate = nn.Parameter(torch.full((L, S, embed_dims), float(gate_init)))

    def forward(self, query, qwen_hidden, qwen_mask=None, layer_idx=0):
        kv = self.qwen_token_proj(qwen_hidden.to(self.qwen_token_proj[0].weight.dtype))
        kv = kv.to(query.dtype)

        # [2] 패딩 토큰 차단. MultiheadFlashAttention 은 key_padding_mask 를 받는다(True=무시).
        #
        #수정(2026-08-11): 마스크가 실제로 가릴 토큰이 있을 때에만 넘긴다.
        #   본 캐시는 이미지 해상도와 지시문이 고정이라 전 표본이 (1159, 1536) 로 길이가
        #   같고 attn_mask 가 전부 1 이다(34,149개 파일 크기 1종으로 확인). 패딩이 없는데
        #   마스크를 넘기면 flash-attn 의 varlen 경로를 타는데, 이 경로의 역방향 커널이
        #   Blackwell(sm_120)에서 "invalid configuration argument" 로 실패한다.
        #   가릴 것이 없는 마스크를 넘기지 않는 것은 계산 결과를 바꾸지 않는다 — 회피가
        #   아니라 항등 연산을 제거하는 것이다. 실제 패딩이 생기면 마스크를 그대로 넘기고,
        #   커널이 실패하면 조용히 넘기지 않고 그대로 드러나게 둔다.
        kwargs = {}
        if qwen_mask is not None and not bool(qwen_mask.bool().all()):
            kwargs["key_padding_mask"] = ~(qwen_mask.bool())

        out = self.attn(query=query, key=kv, value=kv, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        #수정(Run9): 계층·과제별 게이트 적용.
        #  li: 호출한 디코더 계층(범위 밖이면 마지막 게이트 재사용 — 계층 수 불일치 안전장치).
        #  구간: [0,det_end)=det, [det_end,map_end)=map, [map_end,N)=plan/ego.
        li = min(int(layer_idx), self.gate.shape[0] - 1)
        g_layer = self.gate[li]                                   # (S, d)
        if self.num_segs == 1:
            g = g_layer[0]
        else:
            N = query.shape[1]
            g = query.new_zeros(N, self.embed_dims)
            prev = 0
            for si, end in enumerate(list(self.seg_bounds) + [N]):
                end = min(int(end), N)
                if end > prev:
                    g[prev:end] = g_layer[si]
                prev = end
        return query + g.to(out.dtype) * out


def load_qwen_cache(token, cache_dir, device=None, required=True):
    """Per-sample Qwen K/V 디스크 로드.

    Returns: (hidden (L, 1536), attn_mask (L,))
    [3] required=True 면 캐시 누락 시 FileNotFoundError 를 던진다. 일부 표본만 언어 없이
        학습되는 상황을 조용히 허용하지 않기 위한 것이다.
    """
    p = os.path.join(cache_dir, f"{token}.pt")
    if not os.path.exists(p):
        if required:
            raise FileNotFoundError(
                f"Qwen 캐시 누락: {p}. 일부 표본만 언어 없이 학습되면 결과가 재현되지 않는다. "
                f"tools/build_qwen_cache.py 로 전체 분할의 캐시를 먼저 생성할 것."
            )
        return None, None
    d = torch.load(p, map_location=device or "cpu")
    return d["hidden"], d["attn_mask"]
