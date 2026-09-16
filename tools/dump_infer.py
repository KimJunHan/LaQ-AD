# 수정: VLA-v0 — B2D stage2 open-loop 추론 결과를 results.pkl 로 덤프 (시각화 입력용).
#   dist_test 대신 단일 GPU 로 전체 val 을 돌려 outputs(list of {'img_bbox':{...}}) 를 저장.
import warnings; warnings.filterwarnings("ignore")
import sys, mmcv, torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataset
from mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet.models import build_detector
import mmdet3d_plugin

CFG = "projects/configs/hipad_b2d_stage2.py"
CKPT = "work_dirs/hipad_b2d_stage2/latest.pth"
OUT = sys.argv[1] if len(sys.argv) > 1 else "work_dirs/hipad_b2d_stage2/results_val.pkl"

cfg = Config.fromfile(CFG)
ds = build_dataset(cfg.data["val"])
dl = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, CKPT, map_location="cpu")
model = MMDataParallel(model.cuda().eval(), device_ids=[0])

# tensor -> cpu 로 옮겨 pkl 크기/이식성 확보
def to_cpu(r):
    out = {}
    for k, v in r.items():
        out[k] = v.cpu() if torch.is_tensor(v) else v
    return out

results = []
prog = mmcv.ProgressBar(len(ds))
for data in dl:
    with torch.no_grad():
        out = model(return_loss=False, rescale=True, **data)
    results.append({"img_bbox": to_cpu(out[0]["img_bbox"])})
    prog.update()

mmcv.dump(results, OUT)
print(f"\nDUMP DONE: {len(results)} samples -> {OUT}")
