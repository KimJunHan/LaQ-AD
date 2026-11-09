# 수정(2026-10-19): 사용자 결정 — 언어 토큰 자리에 LiDAR/레이더 토큰만 바꿔 넣는 비교(인코더 함께 학습)
#   점 → 기존 기둥 BEV 인코더(PillarBEVEncoder / RadarPillarBEVEncoder, encoder/lidar_pointpillars) → BEV 특징 맵 1 수준
#   (B, C, Hg, Wg) → BEV 칸마다 토큰 1 개 (B, Hg*Wg, C) + 칸 위치 임베딩.
#   이 토큰열이 LaQ v2 교사 캐시 자리(qwen_hidden)에 들어가 같은 QwenTokenAttn(사영 MLP → 0 초기화 유형 임베딩 →
#   flash 교차 어텐션 → 계층·과제·채널별 0 초기화 게이트)을 지난다. 바뀌는 것은 사영 첫 층의 입력 차원(1536 → C)뿐이다.
#
#   격자: 토큰 수를 교사 토큰 수(7 폴더 합 1,832+168+40+240+68+13+8 = 2,369)와 비슷하게 48×48 = 2,304 (−2.7 %)로 둔다.
#     범위는 융합 비교와 같은 x,y ±51.2 m → 칸 크기 102.4/48 = 2.133 m. 인코더의 기둥 크기를 이 격자에 정확히 맞춰
#     (LiDAR 102.4/384 m 기둥 → stride 2·2·2 → 48², Radar 102.4/96 m 기둥 → stride 1·2 → 48²) 인코더 마지막 출력 칸 하나가
#     곧 토큰 하나가 되게 한다 — 다시 표본추출(보간·풀링)하지 않으므로 칸 중심 좌표가 정확하다.
#     출력 해상도가 token_grid 와 다르면 즉시 실패한다(조용히 보간하지 않는다).
#   위치 임베딩: 칸 중심 (x, y) 를 범위로 [0,1] 정규화 → 사인·코사인(models/common/layers.gen_sineembed_for_position,
#     DAB-DETR 방식, 256 차원) → 2층 MLP → C 차원, BEV 특징에 더한다. 학습 표(2,304×C)가 아니라 사인 → MLP 로 둔 이유:
#     (1) 어텐션은 토큰 순서를 모르므로 위치를 명시해야 하는데, 사인 함수는 이웃 칸끼리 값이 연속이라 점이 드문 먼 칸
#         (특히 레이더)도 가까운 칸에서 배운 위치 표현을 함께 쓴다. 학습 표는 칸마다 따로 배워야 한다.
#     (2) 파라미터가 작다(256·C + C·C ≈ 13 만 vs 표 59 만).
#     위치 임베딩은 0 초기화하지 않는다 — 0 초기화 게이트가 이미 step 0 출력을 카메라 단독 모델과 같게 만든다.
#   토큰 순서: 행 우선(iy * Wg + ix), iy 는 y 축, ix 는 x 축(PillarBEVEncoder 격자 규약과 같음).
#   마스크·유형: 교사와 같이 고정 길이·전부 유효. 유형은 센서 토큰 1 종(번호 0) — 0 초기화 유형 임베딩 (1, 256) 하나.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models import BACKBONES, build_backbone

from ...common.layers import gen_sineembed_for_position

__all__ = ["SensorBEVTokenizer"]


@BACKBONES.register_module()
class SensorBEVTokenizer(BaseModule):
    """점 리스트 → BEV 칸 토큰 (B, L=Hg*Wg, C).

    Args:
        bev_encoder: 기둥 BEV 인코더 cfg(PillarBEVEncoder / RadarPillarBEVEncoder). out_indices 는 1 개여야 한다.
        token_grid: (Hg, Wg) — 인코더 출력 해상도와 정확히 같아야 한다.
        pos_dim: 사인 임베딩 차원(짝수, 4 의 배수).
    """

    def __init__(self, bev_encoder, token_grid=(48, 48), pos_dim=256, init_cfg=None):
        super().__init__(init_cfg)
        self.bev_encoder = build_backbone(bev_encoder)
        enc = self.bev_encoder
        if len(enc.out_indices) != 1:
            raise ValueError(f"SensorBEVTokenizer: 인코더 출력 수준은 1 개여야 한다(out_indices={enc.out_indices}).")
        stride = enc.out_strides[0]
        out_hw = (enc.H // stride, enc.W // stride)
        if enc.H % stride or enc.W % stride or tuple(out_hw) != tuple(token_grid):
            raise ValueError(f"SensorBEVTokenizer: 인코더 출력 {out_hw}(캔버스 {enc.H}×{enc.W} / stride {stride}) "
                             f"!= token_grid {tuple(token_grid)} — 기둥 크기·stride 를 격자에 맞출 것(보간하지 않는다).")
        self.points_key = enc.points_key
        self.Hg, self.Wg = int(token_grid[0]), int(token_grid[1])
        self.num_tokens = self.Hg * self.Wg
        self.out_channels = enc.lateral[0].out_channels
        self.pos_dim = int(pos_dim)
        C = self.out_channels
        self.pos_mlp = nn.Sequential(nn.Linear(self.pos_dim, C), nn.ReLU(inplace=True), nn.Linear(C, C))
        # 칸 중심 정규화 좌표 (L, 2) = ((ix+0.5)/Wg, (iy+0.5)/Hg), 행 우선. 상수라 state_dict 에 넣지 않는다.
        iy, ix = torch.meshgrid(torch.arange(self.Hg, dtype=torch.float32),
                                torch.arange(self.Wg, dtype=torch.float32), indexing="ij")
        centers = torch.stack([(ix + 0.5) / self.Wg, (iy + 0.5) / self.Hg], dim=-1).reshape(-1, 2)
        self.register_buffer("cell_centers", centers, persistent=False)
        x0, y0, _, x1, y1, _ = enc.pc_range
        self.cell_size = ((x1 - x0) / self.Wg, (y1 - y0) / self.Hg)

    def cell_centers_metric(self):
        """칸 중심의 실제 좌표 (L, 2) [m] — 시험·시각화용."""
        x0, y0 = self.bev_encoder.pc_range[0], self.bev_encoder.pc_range[1]
        sx, sy = self.cell_size
        c = self.cell_centers
        return torch.stack([x0 + c[:, 0] * self.Wg * sx, y0 + c[:, 1] * self.Hg * sy], dim=-1)

    def pos_embed(self):
        return self.pos_mlp(gen_sineembed_for_position(self.cell_centers, self.pos_dim))  # (L, C)

    def forward(self, points_list):
        feats = self.bev_encoder(points_list)
        f = feats[0]                                                   # (B, C, Hg, Wg)
        if f.shape[-2:] != (self.Hg, self.Wg):
            raise RuntimeError(f"SensorBEVTokenizer: BEV 출력 {tuple(f.shape[-2:])} != {(self.Hg, self.Wg)}")
        tok = f.flatten(2).transpose(1, 2)                             # (B, L, C), 행 우선
        return tok + self.pos_embed().to(tok.dtype)[None]
