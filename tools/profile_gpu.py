# 수정(2026-11-03): 사용자 지시 "파이썬 그대로 프로파일링" — nuScenes 추론(배치 1) GPU 프로파일링. 모델 소스는 건드리지 않는다.
#   조건은 tools/benchmark_latency.py 와 같다(batch 1, 704x256, val 앞쪽 표본 시간 순서, 교사 토큰 캐시, 평가와 같은 fp16 wrap / --fp32).
#   세 가지를 한 번에 낸다(<out>/):
#     (1) modules.json·modules.md: 모듈별 GPU 시간(CUDA event, 포함 시간·프레임당 합) — forward 훅으로 잰다.
#         같은 모듈이 한 프레임에 여러 번 불리면(층별 det_deformable[i] 등) 프레임당 합. 부모 시간은 자식을 포함한다.
#         훅이 forward 를 거치지 않는 부분(instance bank 갱신·후처리 등)은 "head 미귀속" 으로 남는다.
#     (2) torch_profiler/: torch.profiler 추적(chrome trace, 커널별 표 key_averages, 메모리) — --prof-steps 프레임.
#     (3) NVTX 범위: 같은 훅이 모듈 이름으로 NVTX push/pop. nsys 로 돌리면(--nsys) 측정 구간만 cudaProfilerStart/Stop.
#   훅·event 기록 자체가 프레임당 수백 µs 를 더할 수 있다 → 지연 수치는 benchmark_latency.py 결과를 쓰고, 여기서는 비율을 본다
#   (훅 없이 잰 model 시간도 함께 기록해 차이를 남긴다).
#   사용: python tools/profile_gpu.py <config> <ckpt> --out results/profile/x [--fp32] [--nsys]
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_latency import import_plugin, summarize  # noqa: E402  (torch.load weights_only 호환 패치도 함께 적용됨)

from mmcv import Config, DictAction  # noqa: E402
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmcv.runner import load_checkpoint, wrap_fp16_model  # noqa: E402
from mmdet.datasets import build_dataset, replace_ImageToTensor  # noqa: E402
from mmdet.datasets import build_dataloader as build_dataloader_origin  # noqa: E402
from mmdet.models import build_detector  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True, help="결과 폴더")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--n", type=int, default=200, help="모듈 시간 측정 프레임 수")
    ap.add_argument("--prof-steps", type=int, default=5, help="torch.profiler 기록 프레임 수(0 = 건너뜀)")
    ap.add_argument("--depth", type=int, default=3, help="훅을 거는 모듈 깊이(0 = 최상위 자식)")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--nsys", action="store_true", help="nsys --capture-range=cudaProfilerApi 와 함께: 모듈 측정 구간만 기록")
    ap.add_argument("--cfg-options", nargs="+", action=DictAction)
    return ap.parse_args()


class ModuleTimer:
    """forward pre/post 훅으로 모듈별 CUDA event 를 기록하고 NVTX 범위를 연다."""

    def __init__(self, model, depth):
        self.on = False
        self.pending = []          # (이름, 시작 event, 끝 event) — 프레임 끝에 동기화 후 계산
        self.stack = defaultdict(list)
        self.frames = []           # 프레임별 {이름: ms 합}
        self.handles = []
        for name, mod in model.named_modules():
            if not name or name.count(".") > depth:
                continue
            if isinstance(mod, (torch.nn.ModuleList, torch.nn.ModuleDict)):
                continue           # 직접 불리지 않는다(원소에 훅이 걸린다)
            self.handles.append(mod.register_forward_pre_hook(self._pre(name)))
            self.handles.append(mod.register_forward_hook(self._post(name)))

    def _pre(self, name):
        def f(mod, inp):
            if not self.on:
                return
            torch.cuda.nvtx.range_push(name)
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.stack[name].append(ev)
        return f

    def _post(self, name):
        def f(mod, inp, out):
            if not self.on or not self.stack[name]:
                return
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self.pending.append((name, self.stack[name].pop(), ev))
            torch.cuda.nvtx.range_pop()
        return f

    def end_frame(self):
        torch.cuda.synchronize()
        acc = defaultdict(float)
        for name, s, e in self.pending:
            acc[name] += s.elapsed_time(e)
        self.pending.clear()
        self.frames.append(dict(acc))

    def remove(self):
        for h in self.handles:
            h.remove()


def build(a):
    cfg = Config.fromfile(a.config)
    if a.cfg_options:
        cfg.merge_from_dict(a.cfg_options)
    import_plugin(cfg, a.config)
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.data.test.pop("samples_per_gpu", 1)
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader_origin(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16 = cfg.get("fp16", None) is not None and not a.fp32
    if fp16:
        wrap_fp16_model(model)
    load_checkpoint(model, a.checkpoint, map_location="cpu")
    return cfg, loader, MMDataParallel(model, device_ids=[0]).eval(), fp16


def run_frame(model, data):
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    torch.cuda.nvtx.range_push("frame")
    model(return_loss=False, rescale=True, **data)
    torch.cuda.nvtx.range_pop()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg, loader, model, fp16 = build(a)
    inner = model.module
    timer = ModuleTimer(inner, a.depth)
    it = iter(loader)
    t_plain, t_hooked = [], []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        # 예열 + 훅 없는 기준 시간(시간 순서 표본이라 시간 축 메모리가 정상 상태로 들어간다)
        for i in range(a.warmup + a.n // 2):
            t = run_frame(model, next(it))
            if i >= a.warmup:
                t_plain.append(t)
        print(f"훅 없음 model median {np.median(t_plain):.2f} ms", flush=True)

        # (1)·(3) 모듈 시간 + NVTX
        if a.nsys:
            torch.cuda.cudart().cudaProfilerStart()
        timer.on = True
        for i in range(a.n):
            t_hooked.append(run_frame(model, next(it)))
            timer.end_frame()
            if (i + 1) % 50 == 0:
                print(f"[{i + 1}/{a.n}] 훅 있음 model median {np.median(t_hooked):.2f} ms", flush=True)
        timer.on = False
        if a.nsys:
            torch.cuda.cudart().cudaProfilerStop()
        peak_gb = torch.cuda.max_memory_allocated() / 2**30

        # (2) torch.profiler
        if a.prof_steps > 0 and not a.nsys:
            from torch.profiler import ProfilerActivity, profile
            pdir = os.path.join(a.out, "torch_profiler")
            os.makedirs(pdir, exist_ok=True)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                         profile_memory=True, with_stack=False) as prof:
                for _ in range(a.prof_steps):
                    run_frame(model, next(it))
            prof.export_chrome_trace(os.path.join(pdir, "trace.json"))
            ka = prof.key_averages()
            with open(os.path.join(pdir, "kernels_by_cuda_time.txt"), "w") as f:
                f.write(ka.table(sort_by="self_cuda_time_total", row_limit=80))
            with open(os.path.join(pdir, "ops_by_cuda_time.txt"), "w") as f:
                f.write(ka.table(sort_by="cuda_time_total", row_limit=80))
            with open(os.path.join(pdir, "ops_by_input_shape.txt"), "w") as f:
                f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=60))
            with open(os.path.join(pdir, "memory.txt"), "w") as f:
                f.write(ka.table(sort_by="self_cuda_memory_usage", row_limit=40))
    timer.remove()

    # 모듈 표: 프레임별 합 → 중앙값. 프레임에서 한 번도 안 불린 모듈은 0 으로 넣지 않는다(조건부 경로 구분)
    names = sorted({k for fr in timer.frames for k in fr})
    frame_med = float(np.median(t_hooked))
    rows = []
    for k in names:
        xs = [fr[k] for fr in timer.frames if k in fr]
        rows.append({"module": k, "depth": k.count("."), "frames_called": len(xs),
                     "median_ms": float(np.median(xs)), "mean_ms": float(np.mean(xs)),
                     "pct_of_frame": 100.0 * float(np.median(xs)) / frame_med})
    rows.sort(key=lambda r: -r["median_ms"])
    top = {r["module"]: r["median_ms"] for r in rows if r["depth"] == 0}
    head_children = sum(r["median_ms"] for r in rows if r["module"].count(".") == 1 and r["module"].startswith("head."))
    # head 는 SparseHead → onedecoder_head 하나라 깊이 2(onedecoder_head 의 자식) 합으로 미귀속을 잰다
    od = [r for r in rows if r["module"].startswith("head.onedecoder_head.") and r["depth"] == 2]
    od_sum = sum(r["median_ms"] for r in od)
    out = {
        "config": a.config, "checkpoint": a.checkpoint, "fp16_wrapped": fp16, "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "warmup": a.warmup, "n": a.n, "depth": a.depth,
        "model_no_hooks": summarize(t_plain), "model_with_hooks": summarize(t_hooked),
        "hook_overhead_ms_median": float(np.median(t_hooked) - np.median(t_plain)),
        "peak_allocated_gb": peak_gb, "top_level_ms": top,
        "onedecoder_children_sum_ms": od_sum, "onedecoder_ms": next((r["median_ms"] for r in rows if r["module"] == "head.onedecoder_head"), None),
        "frame_minus_top_level_ms": frame_med - sum(top.values()),
        "head_children_sum_ms": head_children, "modules": rows,
        "note": "중앙값의 합은 합의 중앙값과 다를 수 있다. 부모 시간은 자식 포함. 미귀속 = 부모 - 자식 합(훅 밖 연산·CPU 대기).",
    }
    with open(os.path.join(a.out, "modules.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    with open(os.path.join(a.out, "modules.md"), "w") as f:
        f.write(f"# GPU 프로파일 ({out['gpu']}, fp16 wrap={fp16})\n\n")
        f.write(f"- 훅 없음 model median {np.median(t_plain):.2f} ms / 훅 있음 {frame_med:.2f} ms, 최대 할당 {peak_gb:.2f} GB\n")
        f.write(f"- 최상위 합 밖(데이터 scatter·후처리 등) {out['frame_minus_top_level_ms']:.2f} ms; "
                f"onedecoder {out['onedecoder_ms'] or 0:.2f} ms 중 자식 합 {od_sum:.2f} ms\n\n")
        f.write("| 모듈 | 깊이 | 호출 프레임 | median ms | 프레임 대비 % |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['module']} | {r['depth']} | {r['frames_called']} | {r['median_ms']:.3f} | {r['pct_of_frame']:.1f} |\n")
    print(json.dumps({k: out[k] for k in ("model_no_hooks", "model_with_hooks", "hook_overhead_ms_median",
                                          "peak_allocated_gb", "top_level_ms", "frame_minus_top_level_ms")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
