# 수정(2026-11-04): 학생 추론 지연(batch 1) — 원본 eager 와 FastStudent 변형을 같은 조건으로 잰다.
#   조건: tools/benchmark_latency.py·profile_gpu.py 와 같음(val 앞에서부터 장면 순서, 교사 토큰 캐시, workers 0,
#         평가와 같은 fp16 wrap / --fp32). 한 프레임 = DataLoader 표본(CPU) → 결과 dict(원본과 같은 형식) 까지, 앞뒤 synchronize.
#         (데이터 로딩·전처리 시간은 제외 — benchmark_latency.py 의 "model" 구간과 같다.)
#   예열 --warmup 프레임(첫 프레임 eager + 캡처/컴파일 포함) 뒤 --n 프레임의 median/p90. 빌드(컴파일·캡처) 시간은 따로 기록.
#   GPU0 공용 — 측정은 flock /tmp/claude-500/gpu0_timing.lock 로 감싸서 돌린다. 측정 전후 GPU0 의 다른 프로세스를 기록한다.
#   사용: flock /tmp/claude-500/gpu0_timing.lock python tools/latency/student_bench.py --variant graph_post [--fp32] --out x.json
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import student_common as common  # noqa: E402
import torch  # noqa: E402


def main():
    from student_equiv import VARIANTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["eager"] + sorted(k for k in VARIANTS if k != "eager2"))
    ap.add_argument("--config", default=common.DEFAULT_CFG)
    ap.add_argument("--checkpoint", default=common.DEFAULT_CKPT)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    load_before = common.gpu_other_procs()
    cfg, dataset, loader, model, fp16 = common.build(a.config, a.checkpoint, fp32=a.fp32)
    fs = None
    if a.variant == "eager":
        def run(data):
            return model(return_loss=False, rescale=True, **data)
    else:
        from student_fast import FastStudent
        fs = FastStudent(model, **VARIANTS[a.variant])
        run = fs
    it = iter(loader)
    times, first = [], []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(a.warmup + a.n):
            data = next(it)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = run(data)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            if i >= a.warmup:
                times.append(dt)
            elif i < 3:
                first.append(dt)
            if (i + 1) % 100 == 0:
                print(f"[{i + 1}/{a.warmup + a.n}] median {np.median(times) if times else float('nan'):.2f} ms", flush=True)
    load_after = common.gpu_other_procs()
    res = {"variant": a.variant, "kwargs": None if a.variant == "eager" else
           {k: (list(v) if isinstance(v, tuple) else v) for k, v in VARIANTS[a.variant].items()},
           "fp16_wrapped": fp16, "gpu": torch.cuda.get_device_name(0), "warmup": a.warmup, "n": a.n,
           "frame_ms": common.summarize(times), "first_frames_ms": first,
           "build_timings_s": fs.timings if fs is not None else None,
           "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
           "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2 ** 30,
           "gpu0_load_before": load_before, "gpu0_load_after": load_after,
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    s = res["frame_ms"]
    print(f"{a.variant} fp16={fp16}: median {s['median_ms']:.2f} p90 {s['p90_ms']:.2f} ms, build {res['build_timings_s']}, "
          f"peak {res['peak_allocated_gb']:.2f} GB; other procs before={len(load_before.get('other_procs', []))} "
          f"after={len(load_after.get('other_procs', []))}")


if __name__ == "__main__":
    main()
