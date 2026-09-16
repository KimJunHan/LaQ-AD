# 수정: Blackwell 환경 검증용 forward 스모크. 데이터(NAS) 미마운트 상황에서
# 이미지 로딩만 합성 픽셀로 패치하고, pkl 의 실제 메타/캘리브로 모델 forward 가 GPU 에서 도는지 확인.
import argparse, os, warnings
import numpy as np
import torch
import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

warnings.filterwarnings("ignore")

# --- 이미지 로딩 패치: 파일이 없으면 합성 uint8 이미지 반환 ---
_orig_imread = mmcv.imread
def _patched_imread(name, flag="color", *a, **k):
    if isinstance(name, str) and not os.path.exists(name):
        return np.random.randint(0, 256, (900, 1600, 3), dtype=np.uint8)
    return _orig_imread(name, flag, *a, **k)
mmcv.imread = _patched_imread
# loading.py 가 `from ... import mmcv` 가 아니라 모듈로 호출하므로 모듈 속성 교체로 충분

# --- torch 2.x ↔ mmcv 1.7.1 비호환 패치: MMDataParallel scatter 의 _get_stream 이
#     int device-id 를 받는데 torch 2.7 은 torch.device 를 기대함 → 변환 래핑 ---
import mmcv.parallel._functions as _mpf
_orig_get_stream = _mpf._get_stream
def _safe_get_stream(device):
    if isinstance(device, int):
        device = torch.device("cuda", device)
    return _orig_get_stream(device)
_mpf._get_stream = _safe_get_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    import importlib
    importlib.import_module(cfg.plugin_dir.replace("/", ".").rstrip("."))

    from mmdet.datasets import build_dataset
    from mmdet.datasets import build_dataloader
    from mmdet.models import build_detector

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    cfg.data.test.work_dir = cfg.work_dir if cfg.get("work_dir") else "./work_dirs/smoke"
    mmcv.mkdir_or_exist(os.path.abspath(cfg.data.test.work_dir))

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False)

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()
    print(f"[smoke] device cap = {torch.cuda.get_device_capability(0)} | "
          f"name = {torch.cuda.get_device_name(0)}")

    n_ok = 0
    for i, data in enumerate(loader):
        if i >= args.n:
            break
        with torch.no_grad():
            out = model(return_loss=False, rescale=True, **data)
        # 출력에서 텐서를 모아 finite 검사
        tensors = []
        def _collect(o):
            if torch.is_tensor(o):
                tensors.append(o)
            elif isinstance(o, dict):
                [ _collect(v) for v in o.values() ]
            elif isinstance(o, (list, tuple)):
                [ _collect(v) for v in o ]
        _collect(out)
        finite = all(torch.isfinite(t.float()).all().item() for t in tensors if t.numel())
        nkeys = list(out[0].keys()) if isinstance(out, list) and out and isinstance(out[0], dict) else type(out)
        print(f"[smoke] sample {i}: forward OK | out_tensors={len(tensors)} | "
              f"all_finite={finite} | top_keys={nkeys}")
        n_ok += 1
        torch.cuda.synchronize()
    print(f"[smoke] DONE: {n_ok} forward passes succeeded on GPU, peak_mem="
          f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB")


if __name__ == "__main__":
    main()
