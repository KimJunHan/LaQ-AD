# 수정(2026-10-15): 경로 이동 models/sync_free.py → models/common/sync_free.py (소스 정리).
# 수정(2026-10-10): 학습 속도 근본 수정 — 호스트↔GPU 동기화 제거용 도우미.
#   [원인] 1단계 프로파일에서 step 당 cudaStreamSynchronize 790 회(GPU 연산 0.52 s 대비 동기화 대기 0.92 s).
#   파이썬 상수/리스트 인덱스/numpy 정합 결과를 매 호출 GPU 로 복사(pageable H2D = 동기화)하는 코드가 대부분이었다.
#   [원칙] 값은 바꾸지 않는다. 상수는 기존과 같은 연산(CPU float32)으로 한 번 만들어 장치별로 캐시하고,
#   매 step 바뀌는 numpy 값은 고정 메모리(pinned)에서 비동기 복사한다(스트림 순서가 보장되므로 결과 동일).
import numpy as np
import torch

_CACHE = {}


def dev_const(ref, key, builder):
    """builder() 가 만드는 CPU 텐서를 ref 의 장치로 한 번만 옮겨 캐시한다(key 는 값을 결정하는 해시 가능 객체)."""
    k = (key, ref.device, ref.dtype)
    t = _CACHE.get(k)
    if t is None:
        t = builder().to(device=ref.device, dtype=ref.dtype)
        _CACHE[k] = t
    return t


def const_like(ref, values):
    """ref.new_tensor(values) 와 같은 값(같은 dtype 변환)을 장치별 캐시로 돌려준다."""
    key = tuple(np.asarray(values, dtype=np.float64).ravel().tolist()) + (np.asarray(values).shape,)
    return dev_const(ref, ("const",) + key, lambda: torch.tensor(values, dtype=ref.dtype))


def to_device_async(x, device, dtype=None):
    """numpy/리스트/CPU 텐서 → device. 고정 메모리 + non_blocking 이라 호스트가 GPU 를 기다리지 않는다."""
    t = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x, dtype=dtype)
    if t.device.type != "cpu" or torch.device(device).type != "cuda":
        # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — 목적지가 CUDA 가 아니면(CPU 시험 등) 고정 메모리를 쓰지 않는다.
        #   pin_memory() 는 CUDA 장치가 없으면 "No CUDA GPUs are available" 로 실패한다(검토 재현). 종전 호출부는 모두 CUDA 목적지라 동작 동일.
        return t.to(device=device)
    return t.pin_memory().to(device=device, non_blocking=True)


def gather_to_host(tensors):
    """# 수정(2026-11-09): 학습 속도 — 헝가리안 정합 비용 행렬을 det·map 디코더 전 층에서 모아 GPU→CPU 를 한 번으로(종전 층마다 1 회).
    tensors: 텐서 또는 None 의 리스트. 같은 dtype 끼리 1 차원으로 이어 붙여 dtype 당 한 번 .cpu() 하고, 원래 모양의 numpy 뷰
    (None 은 None)를 같은 순서로 돌려준다. 이어 붙이기·복사는 값을 바꾸지 않는다(종전 target.sample 의 표본 묶음과 같은 방식).
    dtype 이 섞이면 승격하지 않고 dtype 별로 따로 옮긴다(값 그대로 보존)."""
    out = [None] * len(tensors)
    by_dtype = {}
    for i, t in enumerate(tensors):
        if t is not None:
            by_dtype.setdefault(t.dtype, []).append(i)
    for idx in by_dtype.values():
        flat = torch.cat([tensors[i].detach().reshape(-1) for i in idx]).cpu().numpy()
        off = 0
        for i in idx:
            n = tensors[i].numel()
            out[i] = flat[off:off + n].reshape(tuple(tensors[i].shape))
            off += n
    return out
