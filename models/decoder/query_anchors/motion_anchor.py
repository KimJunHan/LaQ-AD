# 수정(2026-10-15, 2차): 아키텍처 그림의 query anchors 상자 — 모션 앵커(클래스별 k-means 궤적) 적재와 모드 쿼리 인코더.
#   SparseOneDecoder.__init__ 에 있던 코드를 함수로 옮겼다. 반환 객체를 디코더가 같은 속성 이름(motion_anchor,
#   motion_anchor_encoder)으로 소유하므로 state_dict 키·생성 순서는 같다.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import Linear

from ...common.layers import linear_relu_ln

__all__ = ["load_motion_anchor", "build_motion_anchor_encoder"]


def load_motion_anchor(path):
    return nn.Parameter(torch.tensor(np.load(path), dtype=torch.float32), requires_grad=False)


def build_motion_anchor_encoder(embed_dims):
    return nn.Sequential(*linear_relu_ln(embed_dims, 1, 1), Linear(embed_dims, embed_dims))
