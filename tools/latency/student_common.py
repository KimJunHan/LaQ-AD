# 수정(2026-11-04): 지연 단축(CUDA Graph / torch.compile) 작업용 공용 도우미. 기존 파일은 건드리지 않는다.
#   build(): tools/profile_gpu.py·benchmark_latency.py 와 같은 조건(batch 1, val 시간 순서, 평가와 같은 fp16 wrap / --fp32)으로
#   데이터셋·로더·모델을 만든다.
import importlib
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_TOOLS = os.path.join(_REPO, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

_orig_torch_load = torch.load


def _torch_load_compat(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)


torch.load = _torch_load_compat

from mmcv import Config  # noqa: E402
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmcv.runner import load_checkpoint, wrap_fp16_model  # noqa: E402
from mmdet.datasets import build_dataset, replace_ImageToTensor  # noqa: E402
from mmdet.datasets import build_dataloader as build_dataloader_origin  # noqa: E402
from mmdet.models import build_detector  # noqa: E402

DEFAULT_CFG = "configs/nuscenes/v2_teacher/stage2/test_e3.py"
DEFAULT_CKPT = "results/hipad_nusc_laq_v2_teacher_stage2_e3/iter_35160_ema.pth"


def import_plugin(cfg, config_path):
    if not cfg.get("plugin", False):
        return
    d = os.path.dirname(cfg.plugin_dir) if hasattr(cfg, "plugin_dir") else os.path.dirname(config_path)
    importlib.import_module(".".join(d.split("/")))


def build(config=DEFAULT_CFG, checkpoint=DEFAULT_CKPT, fp32=False, workers=0, cfg_options=None):
    cfg = Config.fromfile(config)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    import_plugin(cfg, config)
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.data.test.pop("samples_per_gpu", 1)
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader_origin(dataset, samples_per_gpu=1, workers_per_gpu=workers, dist=False, shuffle=False)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16 = cfg.get("fp16", None) is not None and not fp32
    if fp16:
        wrap_fp16_model(model)
    load_checkpoint(model, checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0]).eval()
    return cfg, dataset, loader, model, fp16


def summarize(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {"n": int(a.size), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p90_ms": float(np.percentile(a, 90)), "p99_ms": float(np.percentile(a, 99)),
            "min_ms": float(a.min()), "max_ms": float(a.max()), "std_ms": float(a.std())}


def gpu_other_procs():
    """GPU0 에 이 프로세스 말고 다른 연산 프로세스가 있는지(측정 당시 부하 기록용)."""
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader",
                              "-i", os.environ.get("HIPAD_PHYS_GPU", "0")], capture_output=True, text=True).stdout
        me = os.getpid()
        rows = [r.strip() for r in out.strip().splitlines() if r.strip()]
        others = [r for r in rows if int(r.split(",")[0]) != me]
        util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                               "-i", os.environ.get("HIPAD_PHYS_GPU", "0")], capture_output=True, text=True).stdout.strip()
        return {"other_procs": others, "util_pct_snapshot": util}
    except Exception as e:  # 기록용일 뿐이다 — 실패를 감추지 않고 내용으로 남긴다
        return {"error": repr(e)}
