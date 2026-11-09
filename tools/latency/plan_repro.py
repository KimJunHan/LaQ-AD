# 수정(2026-11-05): 프로세스가 다르면 계획 출력이 결정적 DAF 에서도 비트 단위로 달라지는 원인을 찾는다.
#   첫 --frames 프레임 동안 head 아래 모든 모듈의 출력 텐서 지문(바이트 해시)을 호출 순서대로 기록 → 두 실행을 비교해
#   처음 어긋나는 모듈 호출을 찾는다(같은 프로세스 안의 원본·래퍼 비교에서는 나타나지 않는 차이).
#   사용: python tools/latency/plan_repro.py --out a.json [--deterministic-daf]; python tools/latency/plan_repro.py --diff a.json b.json
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _fp(x):
    import torch
    if torch.is_tensor(x):
        t = x.detach()
        return hashlib.sha1((t.contiguous().view(torch.int16) if t.dtype == torch.bfloat16 else t.contiguous()).cpu().numpy().tobytes()).hexdigest()[:16] + f"|{tuple(t.shape)}|{t.dtype}"
    if isinstance(x, (list, tuple)):
        return [_fp(v) for v in x]
    if isinstance(x, dict):
        return {k: _fp(v) for k, v in x.items()}
    return None


def record(a):
    import student_common as common
    import torch
    cfg, dataset, loader, model, fp16 = common.build()
    if a.deterministic_daf:
        import student_fast
        student_fast._DetDAF.on = True
        student_fast.install_global_patches(model.module)
    det = model.module
    log = []

    def hook(name):
        def f(m, i, o):
            log.append([name, _fp(o), _fp([v for v in i if torch.is_tensor(v)])])
        return f
    for name, mod in det.head.named_modules():
        if name:
            mod.register_forward_hook(hook(name))
    it = iter(loader)
    frames = []
    with torch.no_grad():
        for _ in range(a.frames):
            log.clear()
            model(return_loss=False, rescale=True, **next(it))
            frames.append(list(log))
    json.dump({"frames": frames}, open(a.out, "w"))
    print("calls per frame", [len(f) for f in frames])


def diff(pa, pb):
    A, B = json.load(open(pa))["frames"], json.load(open(pb))["frames"]
    for fi, (fa, fb) in enumerate(zip(A, B)):
        for ci, (ca, cb) in enumerate(zip(fa, fb)):
            if ca[0] != cb[0]:
                print(f"frame {fi} call {ci}: 호출 순서 다름 {ca[0]} vs {cb[0]}")
                return
            if ca[2] != cb[2]:
                print(f"frame {fi} call {ci}: 입력이 처음 다름 — {ca[0]}")
                return
            if ca[1] != cb[1]:
                print(f"frame {fi} call {ci}: 같은 입력에 출력이 다름 — {ca[0]}")
                return
        print(f"frame {fi}: 모든 호출 동일 ({len(fa)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--deterministic-daf", action="store_true")
    ap.add_argument("--diff", nargs=2)
    a = ap.parse_args()
    if a.diff:
        diff(*a.diff)
    else:
        record(a)
