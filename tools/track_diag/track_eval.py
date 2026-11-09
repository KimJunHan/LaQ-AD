# 수정(2026-10-08): tracking json 단독 평가 — models/loader/nuscenes/dataset.py NuScenes3DDataset._evaluate_single_tracking 과 동일 설정.
import sys, json, collections, collections.abc
for n in ("Iterable", "Mapping", "Sequence", "Callable"):
    if not hasattr(collections, n): setattr(collections, n, getattr(collections.abc, n))
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.common.config import config_factory
TrackingEval(config=config_factory("tracking_nips_2019"), result_path=sys.argv[1], eval_set="val",
             output_dir=sys.argv[2], verbose=False, nusc_version="v1.0-trainval", nusc_dataroot="data/nuscenes").main()
m = json.load(open(sys.argv[2] + "/metrics_summary.json"))
print(sys.argv[2], {k: round(m[k], 4) for k in ["amota", "amotp", "recall", "motar", "mota", "ids", "frag", "faf", "fp", "fn"]})
