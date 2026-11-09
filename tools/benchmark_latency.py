# 수정(2026-10-29): 사용자 지시 "측정 스크립트 새로 만들어서 내일 새벽에 재자" — nuScenes 추론 시간(배치 1) 측정.
#   이전 측정(TODO.md 10-15·10-16, 사내 서버 results/benchmark/)과 같은 조건: batch 1, 704x256, val 앞쪽 표본 시간 순서,
#   예열 --warmup 회 뒤 --n 회 측정. 교사 토큰은 캐시에서 읽는다(실시간 Qwen 은 측정하지 않음).
#   구간: data = 데이터로더가 다음 표본을 내줄 때까지(CPU 전처리, workers 0 이면 직렬 전처리 시간 그대로),
#         model = scatter(GPU 복사) + forward, 앞뒤 torch.cuda.synchronize.
#   정밀도: 기본 = 평가(tools/test.py)와 같게 config 의 fp16 이 있으면 wrap_fp16_model. --fp32 = fp16 설정을 지워 전부 fp32.
#   사용: python tools/benchmark_latency.py <config> <ckpt> --out results/benchmark/x.json [--fp32] [--workers 0]
import argparse
import importlib
import json
import os
import statistics
import time

import numpy as np
import torch

_orig_torch_load = torch.load
def _torch_load_compat(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _torch_load_compat

from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import build_dataset, replace_ImageToTensor
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True, help="결과 JSON 경로")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--fp32", action="store_true", help="config 의 fp16 설정을 무시하고 전부 fp32 로")
    ap.add_argument("--workers", type=int, default=0, help="0 = 메인 프로세스에서 직렬 전처리(전처리 시간 측정용)")
    ap.add_argument("--cfg-options", nargs="+", action=DictAction)
    return ap.parse_args()


def import_plugin(cfg, config_path):
    if not cfg.get("plugin", False):
        return
    d = os.path.dirname(cfg.plugin_dir) if hasattr(cfg, "plugin_dir") else os.path.dirname(config_path)
    importlib.import_module(".".join(d.split("/")))


def summarize(xs):
    a = np.asarray(xs)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)), "p90_ms": float(np.percentile(a, 90)),
            "min_ms": float(a.min()), "max_ms": float(a.max()), "std_ms": float(a.std()),
            "fps_median": float(1000.0 / np.median(a))}


def main():
    a = parse_args()
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
    loader = build_dataloader_origin(dataset, samples_per_gpu=1, workers_per_gpu=a.workers, dist=False, shuffle=False)

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16 = cfg.get("fp16", None) is not None and not a.fp32
    if fp16:
        wrap_fp16_model(model)
    load_checkpoint(model, a.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0]).eval()

    total = a.warmup + a.n
    t_data, t_model, n_nonfinite = [], [], 0
    it = iter(loader)
    with torch.no_grad():
        for i in range(total):
            t0 = time.perf_counter()
            data = next(it)
            t1 = time.perf_counter()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            result = model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            if i >= a.warmup:
                t_data.append((t1 - t0) * 1000)
                t_model.append((t3 - t2) * 1000)
                # 계획 출력이 유한한지(fp16 넘침 확인용)
                r = result[0] if isinstance(result, list) else result
                r = r.get("img_bbox", r) if isinstance(r, dict) else r
                for k, v in (r.items() if isinstance(r, dict) else []):
                    if "plan" in k and torch.is_tensor(v) and not torch.isfinite(v.float()).all():
                        n_nonfinite += 1
                        break
            if (i + 1) % 50 == 0:
                print(f"[{i + 1}/{total}] model median {statistics.median(t_model) if t_model else float('nan'):.1f} ms", flush=True)

    e2e = [d + m for d, m in zip(t_data, t_model)]
    out = {
        "config": a.config, "checkpoint": a.checkpoint, "fp16_wrapped": fp16, "config_fp16": cfg.get("fp16", None), "workers": a.workers,
        "warmup": a.warmup, "n": a.n, "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "teacher_tokens": "cache", "nonfinite_plan_samples": n_nonfinite,
        "model": summarize(t_model), "data": summarize(t_data), "end_to_end": summarize(e2e),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("fp16_wrapped", "model", "data", "end_to_end", "nonfinite_plan_samples")}, indent=1))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
