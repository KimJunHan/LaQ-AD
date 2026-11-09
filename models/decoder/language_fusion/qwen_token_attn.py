# 수정(2026-10-15): 경로 이동 models/qwen_token_attn.py → models/decoder/language_fusion/qwen_token_attn.py (아키텍처 그림의 language fusion, gated cross-attn). 미사용 load_qwen_cache 제거(캐시는 파이프라인 LoadQwenCacheNusc 또는 SparseDetector 가 읽는다).
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
#      언어 정보는 손실을 낮추는 만큼만 점진적으로 반영된     다. ablation 의 ON/OFF 비교도
#      이때 비로소 성립한다 — OFF 가 ON 의 초기 상태와 같아야 차이를 융합 효과로 귀속할 수 있다.
#
#  [2] qwen_mask 미사용.
#      인자로 받기만 하고 어텐션에 넘기지 않아 패딩 토큰까지 참조했다. 배치 내 문장 길이가
#      다르면 짧은 표본일수록 의미 없는 패딩에 어텐션이 분산된다. key_padding_mask 로 넘긴다.

#  [3] 캐시 누락 시 조용한 skip.
#      호출부가 cache miss 를 graceful skip 으로 처리하면, 일부 표본만 언어 없이 학습되어
#      결과가 재현되지 않는다. 프로젝트 규칙(root-cause only, skip 금지)에 어긋난다.
#      캐시 로더가 누락을 명시적으로 알리고, 호출부가 실패하도록 한다.

import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.utils import build_from_cfg
from mmcv.runner import BaseModule

from ...encoder.projector.qwen_projector import build_qwen_projector


@ATTENTION.register_module()
class QwenTokenAttn(BaseModule):
    """Qwen token K/V ↔ HiP-AD query Q cross-attention (0 초기화 게이트 잔차).

    Forward: query (B, N, embed_dims), qwen_hidden (B, L, qwen_hidden_dim),
             qwen_mask (B, L) — 1=유효 토큰, 0=패딩
    Returns: query + gate * attn(query, kv, kv)
      수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 1) — attn(...) 은 잔차를 뺀 교차 어텐션 출력 A 다. 종전 구현은
      MultiheadFlashAttention 의 identity 잔차가 섞여 실제로는 (1+g)⊙query + g⊙A 였다. 이제 정확히 query + g⊙A
      (g=0 → query 와 비트 동일, g≠0 → 출력 − query = g⊙A).
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
                 num_layers=0, seg_bounds=None, proj_hidden=0,
                 #수정(2026-10-12): sep_gate_segs — 지정한 과제 구간의 게이트를 별도 파라미터(gate_sep)로
                 #  분리해 optimizer 에서 따로 lr 배수를 줄 수 있게 한다. 1단계는 계획·자차 손실이 0 이라
                 #  plan/ego 구간(seg 2) 게이트가 75k iter 후에도 정확히 0 이다(체크포인트 실측). 2단계에서
                 #  다른 계획 모듈과 같은 lr x10 을 주되, 이미 수렴한 det/map 게이트는 기본 lr 로 둔다.
                 #  None 이면 종전과 완전히 동일.
                 sep_gate_segs=None,
                 #수정(2026-10-16): 교사 캐시 7 폴더 · 토큰 유형 임베딩
                 #  num_token_types>0 : 사영 뒤(256 차원) K/V 에 더하는 0 초기화 유형 임베딩 (T, embed_dims).
                 #                      0(기본)이면 파라미터를 만들지 않는다 — 종전 config 의 state_dict·난수열 불변.
                 #  token_type_names  : 유형 번호 → 폴더 이름(경로 지정 route_segs 에 필요). 기본 = 교사 캐시 7 폴더 순서.
                 #  route_segs        : 쿼리 구간별 읽을 폴더 목록(구간 수 = len(seg_bounds)+1). None(기본) = 모든 쿼리가
                 #                      모든 토큰을 본다. 마스크 대신 구간별로 K/V 를 잘라 따로 어텐션한다(아래 forward 참조).
                 num_token_types=0, token_type_names=None, route_segs=None,
                 # 수정(2026-10-26): 사용자 요청 "교사 토큰 반영 비율을 높여 평가" — 진단 전용. 평가(eval 모드)에서만 g 에 곱하는 배율.
                 #   1.0(기본) = 종전과 비트 단위 동일. 학습 중에는 쓰지 않는다. val 로 값을 고르면 val 튜닝이므로 판정 수치로 쓰지 않는다.
                 eval_gate_scale=1.0):
        super().__init__(init_cfg)
        self.eval_gate_scale = float(eval_gate_scale)
        self.embed_dims = embed_dims
        #수정(2026-10-15): 소스 정리 — 단층 선형 사영(proj_hidden=0)과 단일 구간 게이트(seg_bounds 미지정) 분기를
        #  제거했다. 모든 config 가 proj_hidden=512 / seg_bounds=[900, 1000] 을 쓴다. 지정하지 않으면 조용히
        #  다른 구조로 빌드하지 않고 즉시 실패한다. legacy (d,) 게이트 체크포인트 호환(_load_from_state_dict)은 유지.
        if not (proj_hidden and proj_hidden > 0):
            raise ValueError("QwenTokenAttn: proj_hidden>0 이 필요하다(단층 선형 사영 분기는 2026-10-15 제거).")
        if not seg_bounds:
            raise ValueError("QwenTokenAttn: seg_bounds 가 필요하다(단일 구간 게이트 분기는 2026-10-15 제거).")
        # 수정(2026-10-15, 2차): 사영 MLP 구성은 encoder/projector/qwen_projector.py(아키텍처 그림의 projector). 키·순서 동일.
        self.qwen_token_proj = build_qwen_projector(qwen_hidden_dim, proj_hidden, embed_dims)
        self.num_layers = int(num_layers or 0)
        self.seg_bounds = list(seg_bounds)
        self.num_segs = len(self.seg_bounds) + 1
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
        self.sep_gate_segs = list(sep_gate_segs) if sep_gate_segs else []
        assert all(0 <= si < S for si in self.sep_gate_segs), (self.sep_gate_segs, S)
        if self.sep_gate_segs:
            # self.gate 의 해당 구간 원소는 forward 에서 쓰지 않는다(기울기 0 → 0 유지).
            self.gate_sep = nn.Parameter(torch.full((L, len(self.sep_gate_segs), embed_dims), float(gate_init)))
        # 수정(2026-10-16): 교사 캐시 7 폴더 · 토큰 유형 임베딩
        #   위치 = 사영(1536→512→256→LayerNorm) **뒤**, 256 차원. 이유:
        #   (1) LayerNorm 뒤에 더하므로 정규화가 유형 신호를 다시 눌러 없애지 않는다(BERT segment embedding 과 같은 자리).
        #   (2) 0 초기화면 kv 가 비트 단위로 종전과 같다 — 사영 앞(1536)에 더하면 0 이라도 같은 값이지만, 학습 뒤
        #       GELU·LayerNorm 을 거쳐 토큰 내용과 비선형으로 섞여 "폴더 구분" 효과만 따로 해석하기 어렵다.
        #   (3) 파라미터 7×256=1,792 (1536 이면 10,752) — 작고, 어텐션 in_proj 가 K·V 각각에 선형으로 옮긴다.
        #   K 와 V 둘 다에 더한다: K 쪽은 "어느 폴더를 볼지"를, V 쪽은 "읽은 정보가 어느 폴더에서 왔는지"를 쿼리에 전달한다.
        self.num_token_types = int(num_token_types or 0)
        if self.num_token_types > 0:
            self.token_type_embed = nn.Parameter(torch.zeros(self.num_token_types, embed_dims))
        if token_type_names is None:
            from ...encoder.vl_tokens_cache.loading import TEACHER_CACHE_FOLDERS
            token_type_names = TEACHER_CACHE_FOLDERS
        self.token_type_names = list(token_type_names)
        self.route_type_ids = None
        if route_segs is not None:
            if len(route_segs) != self.num_segs:
                raise ValueError(f"QwenTokenAttn: route_segs 구간 수 {len(route_segs)} != {self.num_segs}")
            bad = [n for seg in route_segs for n in seg if n not in self.token_type_names]
            if bad or any(len(seg) == 0 for seg in route_segs):
                raise ValueError(f"QwenTokenAttn: route_segs 오류(알 수 없는 폴더 {bad} 또는 빈 구간): {route_segs}")
            self.route_type_ids = [sorted(self.token_type_names.index(n) for n in seg) for seg in route_segs]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """수정(2026-10-02): 게이트 shape 하위호환.

        Run9 에서 게이트를 (d,) -> (L,S,d) 로 바꿨는데 "기본값이면 종전과 동일"이라는
        가정이 틀렸다. shape 가 달라져 mmcv 의 strict=False 적재가 이 키를 **조용히
        건너뛰었고**, stage2 가 stage1 이 배운 융합을 버리고 0 에서 다시 시작했다
        (results/hipad_nusc_vla_s2hr/20260918_155442.log:932 의 size mismatch).
        legacy 체크포인트의 (d,) 게이트를 현재 shape 로 브로드캐스트해 복원한다.
        """
        k = prefix + "gate"
        if k in state_dict:
            src = state_dict[k]
            tgt = self.gate
            if src.shape != tgt.shape and src.numel() == tgt.shape[-1]:
                state_dict[k] = src.reshape(1, 1, -1).expand_as(tgt).contiguous()
                print(f"[qwen_token_attn] legacy 게이트 {tuple(src.shape)} -> "
                      f"{tuple(tgt.shape)} 브로드캐스트 복원", flush=True)
        #수정(2026-10-12): gate_sep 이 없는 체크포인트(1단계)에서는 self.gate 의 해당 구간 값으로 채운다.
        ks = prefix + "gate_sep"
        if self.sep_gate_segs and ks not in state_dict and k in state_dict \
                and state_dict[k].shape == self.gate.shape:
            state_dict[ks] = state_dict[k][:, self.sep_gate_segs].clone()
            print(f"[qwen_token_attn] gate_sep 을 gate[:, {self.sep_gate_segs}] 에서 초기화 "
                  f"(|g| max {float(state_dict[ks].abs().max()):.4g})", flush=True)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    # 수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 2) — 교사 토큰 사영을 디코더 forward 당 1 회만 계산.
    #   사영(qwen_token_proj)·유형 임베딩(token_type_embed)은 이 모듈 하나가 소유하고 디코더 6 계층이 같은 모듈을 공유한다
    #   (unified_decoder.py 의 self.qwen_token_attn 1 개). 사영 입력(qwen_hidden·qwen_type)도 6 계층 모두 같고, 사영 MLP 에는
    #   dropout 이 없어(Linear→GELU→Linear→LayerNorm) 결과가 계층과 무관하게 같다 → 계층마다 다시 계산하던 것은 같은 값을
    #   6 번 만들고 그 중간 활성(B×L×1536 fp32 사본 포함)을 역전파용으로 6 벌 저장했다. 아래 project_kv 를 디코더가 forward 시작 뒤
    #   첫 qwen_attn 에서 한 번 부르고 결과를 지역 변수로 들고 다니며 forward(kv=...) 로 넘긴다(전역·텐서 신원 캐시 없음 →
    #   forward 가 끝나면 자동 소멸, 배치·스트림 그룹이 바뀌어도 섞일 수 없다). 역전파는 autograd 가 6 계층의 기울기를 kv 에서
    #   합친 뒤 사영을 한 번 지난다(수식 동일, 부동소수 합산 순서만 다름). 디코더에는 활성 체크포인팅이 없다(with_cp 는 영상 백본만).
    def project_kv(self, qwen_hidden, qwen_type=None, dtype=None):
        """교사(VL) 토큰 → K/V 공간(B, L, embed_dims). 반환 (kv, qwen_type(long 또는 None)).

        dtype: kv 를 바꿀 dtype(= 쿼리 dtype). 종전 forward 와 같은 순서(사영 → dtype 변환 → 유형 임베딩 더하기)로 계산한다.
        """
        kv = self.qwen_token_proj(qwen_hidden.to(self.qwen_token_proj[0].weight.dtype))
        if dtype is not None:
            kv = kv.to(dtype)
        # 수정(2026-10-16): 교사 캐시 7 폴더 · 토큰 유형 임베딩 — 유형 번호가 필요한 설정인데 없으면 즉시 실패.
        if (self.num_token_types > 0 or self.route_type_ids is not None):
            if qwen_type is None:
                raise RuntimeError("QwenTokenAttn: num_token_types/route_segs 가 켜졌는데 qwen_type 이 없다 "
                                   "(파이프라인 LoadTeacherCacheNusc 와 Collect keys 의 'qwen_type' 확인).")
            if qwen_type.shape != kv.shape[:2]:
                raise ValueError(f"QwenTokenAttn: qwen_type {tuple(qwen_type.shape)} != 토큰 {tuple(kv.shape[:2])}")
            qwen_type = qwen_type.long()
        if self.num_token_types > 0:
            if bool(((qwen_type < 0) | (qwen_type >= self.num_token_types)).any()):  # 호스트 동기화 1 회(종전 mask 검사와 같은 수준)
                raise ValueError(f"QwenTokenAttn: qwen_type 범위 [{int(qwen_type.min())}, {int(qwen_type.max())}] "
                                 f"이 [0, {self.num_token_types}) 밖이다")
            kv = kv + self.token_type_embed.to(kv.dtype)[qwen_type]
        return kv, qwen_type

    def _attn_only(self, q, kv, **kw):
        # 수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 1) — 게이트는 언어 항에만 건다.
        #   MultiheadFlashAttention.forward 는 identity(기본 = query) + attn 을 돌려준다. 종전에는 그 값을 그대로 게이트에 곱해
        #   query + g·(query + A) = (1+g)⊙query + g⊙A 가 되어 게이트가 쿼리 자체의 채널별 재척도까지 맡았다(1단계 |g|≤0.113 → 최대 11%).
        #   identity 를 0 으로 넘겨 교차 어텐션 출력 A 만 받는다(0 + A = A, 부동소수 정확). MultiheadFlashAttention 자체는 바꾸지 않으므로
        #   다른 호출부(gnn·temp_gnn·inter_gnn 등)의 잔차 동작은 그대로다.
        o = self.attn(query=q, key=kv, value=kv, identity=torch.zeros_like(q), **kw)
        return o[0] if isinstance(o, tuple) else o

    def forward(self, query, qwen_hidden, qwen_mask=None, layer_idx=0, qwen_type=None, kv=None, mask_all_valid=None, row_offset=0):  # 수정(2026-11-09): C3 GRPO §9.3 row_offset(학습 속도 수정의 mask_all_valid 뒤에 손으로 합침)
        """반환 query + g ⊙ A, A = 교차 어텐션 출력(잔차 미포함). 수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 1).

        kv: 수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 2) — project_kv(qwen_hidden, qwen_type, query.dtype) 결과의 kv.
            주면 사영을 다시 계산하지 않는다(이때 qwen_type 은 project_kv 가 돌려준 값을 넘긴다). None 이면 종전처럼 여기서 계산.
        mask_all_valid: 수정(2026-11-09): 학습 속도 — 호출자가 같은 qwen_mask 로 미리 구한 `qwen_mask is None or bool(qwen_mask.bool().all())`.
            주면 계층마다 하던 판정(호스트 동기화)을 다시 하지 않는다. None 이면 종전처럼 여기서 판정.
        row_offset: 수정(2026-11-09): C3 GRPO 설계 §9.3 계획 기울기 격리 — query 가 통합 쿼리 풀의 행 [row_offset, row_offset + N) 구간만
            담고 있을 때 그 시작 행 번호. 구간 게이트(seg_bounds)·경로 지정(route_segs)을 풀 전체의 행 번호 기준으로 고른다.
            교차 어텐션은 쿼리 행끼리 상호작용이 없어(각 행 softmax 독립) 일부 행만 넣어도 그 행의 값은 같다(G2-3 에서 수치 확인). 0(기본) = 종전.
        """
        row_offset = int(row_offset)  # 수정(2026-11-09): C3 GRPO §9.3
        if row_offset < 0:
            raise ValueError(f"QwenTokenAttn: row_offset {row_offset} < 0")
        if kv is None:
            kv, qwen_type = self.project_kv(qwen_hidden, qwen_type, query.dtype)
        else:
            # 미리 계산한 kv 가 이번 입력과 어긋나면 조용히 쓰지 않고 즉시 실패한다.
            if kv.dtype != query.dtype or kv.shape[:2] != qwen_hidden.shape[:2] or kv.shape[-1] != self.embed_dims:
                raise ValueError(f"QwenTokenAttn: 미리 계산한 kv {tuple(kv.shape)} {kv.dtype} 가 query {query.dtype}/"
                                 f"qwen_hidden {tuple(qwen_hidden.shape)} 와 맞지 않는다")
            if self.route_type_ids is not None and qwen_type is None:
                raise RuntimeError("QwenTokenAttn: route_segs 가 켜졌는데 qwen_type 이 없다")

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
        if mask_all_valid is None:  # 수정(2026-11-09): 미리 판정한 값이 없을 때만 여기서 판정(동기화 1 회) — 값은 같다
            mask_all_valid = qwen_mask is None or bool(qwen_mask.bool().all())
        if qwen_mask is not None and not mask_all_valid:
            #수정(2026-10-02): 극성 반전 버그. flash-attn 2.8.3 unpad_input 의 규약은
            #   "attention_mask: 1 means valid and 0 means not valid" 이므로 유효=True 를
            #   넘겨야 한다. 종전엔 ~mask(True=패딩)를 넘겨 **실제 토큰을 전부 버리고 패딩만
            #   참조**했다. 지금까지 모든 캐시가 동일 길이(L=1159, mask 전부 1)여서 이 분기를
            #   타지 않아 드러나지 않았으나, 길이 다른 캐시를 섞으면(주 1159 / qwen3d 1207)
            #   즉시 발동한다.
            kwargs["key_padding_mask"] = qwen_mask.bool()

        if self.route_type_ids is None:
            out = self._attn_only(query, kv, **kwargs)  # 수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 1) — 잔차 없는 A
        else:
            out = self._routed_attn(query, kv, qwen_mask, qwen_type, row_offset=row_offset)  # 수정(2026-11-09): C3 GRPO §9.3 row_offset. #이전 (query, kv, qwen_mask, qwen_type)
        #수정(Run9): 계층·과제별 게이트 적용.
        #  li: 호출한 디코더 계층(범위 밖이면 마지막 게이트 재사용 — 계층 수 불일치 안전장치).
        #  구간: [0,det_end)=det, [det_end,map_end)=map, [map_end,N)=plan/ego.
        li = min(int(layer_idx), self.gate.shape[0] - 1)
        g_layer = self.gate[li]                                   # (S, d)
        N = query.shape[1]
        g = query.new_zeros(N, self.embed_dims)
        prev = 0
        # 수정(2026-11-09): C3 GRPO §9.3 — 구간 경계를 이 query 의 지역 행 번호로 옮긴다(row_offset 0 이면 종전과 같은 경계·같은 대입).
        for si, end in enumerate([b - row_offset for b in self.seg_bounds] + [N]):  # #이전 enumerate(list(self.seg_bounds) + [N])
            end = max(min(int(end), N), 0)  # 수정(2026-11-09): 경계가 이 구간 앞이면 0(그 구간 행 없음). #이전 end = min(int(end), N)
            if end > prev:
                #수정(2026-10-12): 분리된 구간은 gate_sep 사용
                g[prev:end] = (self.gate_sep[li][self.sep_gate_segs.index(si)]
                               if si in self.sep_gate_segs else g_layer[si])
            prev = max(prev, end)  # 수정(2026-11-09): row_offset 0 이면 경계가 단조라 종전 prev = end 와 같다. #이전 prev = end
        if not self.training and self.eval_gate_scale != 1.0:
            g = g * self.eval_gate_scale  # 수정(2026-10-26): 진단 전용 평가 배율(위 __init__ 설명)
        return query + g.to(out.dtype) * out

    def _routed_attn(self, query, kv, qwen_mask, qwen_type, row_offset=0):  # 수정(2026-11-09): C3 GRPO §9.3 row_offset(forward 설명)
        """수정(2026-10-16): 교사 캐시 7 폴더 · 토큰 유형 임베딩 — 쿼리 구간별 폴더 경로 지정.

        MultiheadFlashAttention 은 attn_mask(쿼리×키 마스크/편향)를 지원하지 않고(assert attn_mask is None),
        flash-attn varlen 커널도 임의 편향을 받지 않는다. key_padding_mask 는 표본 단위(모든 쿼리 공통)라 구간별
        경로 지정을 표현할 수 없고, 쓰면 Blackwell 에서 역방향이 실패하는 unpad 경로를 탄다.
        교차 어텐션은 쿼리끼리 상호작용이 없으므로(각 쿼리 행의 softmax 가 독립), "구간 s 의 쿼리는 허용 폴더만 본다"는
        블록 마스크 어텐션은 "구간 s 쿼리 × 허용 폴더 토큰만 모은 K/V" 의 밀집 어텐션과 수학적으로 같다.
        그래서 구간마다 K/V 를 잘라(index_select — 폴더 길이가 고정이라 배치 전체에 같은 색인, 패딩 없음) 같은 attn
        모듈(같은 가중치)을 따로 부른다. 마스크가 없으므로 종전과 같은 비패딩 flash 경로다.
        수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 1) — 구간별 호출도 _attn_only(identity=0)로 잔차 없는 A 만 받아 이어 붙인다
        (종전 문장 "attn 은 identity(=쿼리) 잔차를 포함해 반환" 은 더 이상 맞지 않는다). 게이트·잔차는 forward 가 query + g⊙A 로 한 번 더한다.
        """
        if not torch.equal(qwen_type, qwen_type[:1].expand_as(qwen_type)):
            raise ValueError("QwenTokenAttn: route_segs 는 배치 내 모든 표본의 토큰 유형 배치가 같아야 한다(폴더별 고정 길이).")
        t0 = qwen_type[0]
        N = query.shape[1]
        outs, prev = [], 0
        for si, end in enumerate([b - int(row_offset) for b in self.seg_bounds] + [N]):  # 수정(2026-11-09): C3 GRPO §9.3. #이전 list(self.seg_bounds) + [N]
            end = max(min(int(end), N), 0)  # 수정(2026-11-09): C3 GRPO §9.3 — 앞 구간 경계 → 0. #이전 min(int(end), N)
            if end <= prev:
                prev = max(prev, end)
                continue
            allowed = torch.tensor(self.route_type_ids[si], device=t0.device, dtype=t0.dtype)
            idx = torch.nonzero(torch.isin(t0, allowed), as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                raise ValueError(f"QwenTokenAttn: 구간 {si} 의 허용 폴더 "
                                 f"{[self.token_type_names[i] for i in self.route_type_ids[si]]} 토큰이 입력에 없다")
            kv_s = kv.index_select(1, idx)
            kw = {}
            if qwen_mask is not None:
                m_s = qwen_mask.index_select(1, idx)
                if not bool(m_s.bool().all()):
                    kw["key_padding_mask"] = m_s.bool()  # 실제 패딩이 있을 때만(종전 규칙과 같음)
            outs.append(self._attn_only(query[:, prev:end], kv_s, **kw))  # 수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 1) — 잔차 없는 A
            prev = end
        return torch.cat(outs, dim=1)
