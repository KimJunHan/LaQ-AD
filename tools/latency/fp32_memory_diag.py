# 수정(2026-11-04): fp32 추론 최대 메모리(33.7 GB, fp16 wrap 은 1.1 GB)의 출처를 찾는다.
#   예열 뒤 한 프레임을 torch.cuda.memory._record_memory_history 로 기록해 가장 큰 할당(크기·파이썬 스택)을 나열하고,
#   모듈별(깊이 ≤ 3) 구간 최대 메모리 증가를 forward 훅으로 잰다.
#   사용: python tools/latency/fp32_memory_diag.py [--fp32] --out results/latency/fp32_memory_diag.json
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import student_common as common  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    cfg, dataset, loader, model, fp16 = common.build(fp32=a.fp32)
    it = iter(loader)
    det = model.module
    with torch.no_grad():
        for _ in range(a.warmup):
            model(return_loss=False, rescale=True, **next(it))
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        # (1) 모듈별 구간 최대(자식 포함): pre 에서 현재 할당·누적 최대를 스택에 넣고 reset, post 에서 구간 최대 = max(현재 최대, 자식 최대)
        stack, rows = [], {}

        def pre(name):
            def f(m, i):
                stack.append([name, torch.cuda.memory_allocated(), 0])
                torch.cuda.reset_peak_memory_stats()
            return f

        def post(name):
            def f(m, i, o):
                nm, a0, child_peak = stack.pop()
                peak = max(torch.cuda.max_memory_allocated(), child_peak)
                r = rows.setdefault(nm, {"calls": 0, "max_peak_over_entry_gb": 0.0})
                r["calls"] += 1
                r["max_peak_over_entry_gb"] = max(r["max_peak_over_entry_gb"], (peak - a0) / 2 ** 30)
                if stack:
                    stack[-1][2] = max(stack[-1][2], peak)
                torch.cuda.reset_peak_memory_stats()
            return f
        hs = []
        for name, mod in det.named_modules():
            if not name or name.count(".") > 3 or isinstance(mod, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                continue
            hs += [mod.register_forward_pre_hook(pre(name)), mod.register_forward_hook(post(name))]
        torch.cuda.reset_peak_memory_stats()
        model(return_loss=False, rescale=True, **next(it))
        torch.cuda.synchronize()
        for h in hs:
            h.remove()
        # (2) 할당 기록
        torch.cuda.memory._record_memory_history(max_entries=200000, stacks="python")
        torch.cuda.reset_peak_memory_stats()
        model(return_loss=False, rescale=True, **next(it))
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        try:
            snap = torch.cuda.memory._snapshot()
        except RuntimeError as e:  # 기록 실패는 내용으로 남긴다(모듈 표는 그대로 유효)
            print("snapshot 실패:", e)
            snap = {}
        torch.cuda.memory._record_memory_history(enabled=None)
    allocs = []
    for tr in snap.get("device_traces", []):
        for e in tr:
            if e.get("action") == "alloc":
                fr = [f"{os.path.relpath(x['filename'], common._REPO) if x['filename'].startswith(common._REPO) else x['filename']}:{x['line']} {x['name']}"
                      for x in e.get("frames", []) if x["filename"].endswith(".py")]
                allocs.append({"size_gb": e["size"] / 2 ** 30, "frames": fr[:8]})
    allocs.sort(key=lambda x: -x["size_gb"])
    top_mod = sorted(rows.items(), key=lambda kv: -kv[1]["max_peak_over_entry_gb"])[:25]
    out = {"fp16_wrapped": fp16, "allocated_before_frame_gb": base / 2 ** 30, "frame_peak_gb": peak / 2 ** 30,
           "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
           "top_allocs": allocs[:15], "top_modules": top_mod}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"fp16={fp16} frame peak {peak / 2 ** 30:.2f} GB (before {base / 2 ** 30:.2f} GB)")
    for x in allocs[:8]:
        print(f"  {x['size_gb']:.3f} GB  " + " <- ".join(x["frames"][:4]))
    for k, v in top_mod[:12]:
        print(f"  module {k:60s} peak+{v['max_peak_over_entry_gb']:.3f} GB")


if __name__ == "__main__":
    main()
