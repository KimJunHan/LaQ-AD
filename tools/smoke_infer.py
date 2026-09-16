# 수정: VLA-v0 — B2D stage2 추론 스모크. 모델 로드 + 2 sample forward 후 출력 키/shape 확인.
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataset
from mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet.models import build_detector
import mmdet3d_plugin

CFG = "projects/configs/hipad_b2d_stage2.py"
CKPT = "work_dirs/hipad_b2d_stage2/latest.pth"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2

cfg = Config.fromfile(CFG)
ds = build_dataset(cfg.data["val"])
dl = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, CKPT, map_location="cpu")
model = MMDataParallel(model.cuda().eval(), device_ids=[0])

for i, data in enumerate(dl):
    if i >= N:
        break
    with torch.no_grad():
        out = model(return_loss=False, rescale=True, **data)
    r = out[0]["img_bbox"]
    print(f"\n=== sample {i} keys ===")
    for k, v in r.items():
        if torch.is_tensor(v):
            print(f"  {k}: tensor {tuple(v.shape)} {v.dtype}")
        elif isinstance(v, np.ndarray):
            print(f"  {k}: ndarray {v.shape}")
        else:
            print(f"  {k}: {type(v).__name__}")
print("\nSMOKE OK")
