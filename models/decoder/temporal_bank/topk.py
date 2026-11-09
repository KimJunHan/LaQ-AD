# 수정(2026-10-15): instance_bank.py / plan/instance_bank.py / ego/instance_bank.py 에 3중 정의되어 있던 같은 topk 를 하나로 합쳤다.
import torch

__all__ = ["topk"]


def topk(confidence, k, *inputs):
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (indices + torch.arange(bs, device=indices.device)[:, None] * N).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs
