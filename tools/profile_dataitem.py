#수정: data_time 병목 근본원인 측정용 프로파일러(일회성 진단, 2026-07-16).
#      __getitem__ 을 구성단계로 쪼개 각 transform 소요를 측정 → 진짜 병목 단계를 특정.
import os, sys, time, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # GPU 불필요
sys.path.insert(0, ".")
import numpy as np
from mmcv import Config
from mmcv.utils import build_from_cfg
from mmdet.datasets import DATASETS

import importlib
importlib.import_module("projects.mmdet3d_plugin")  # 커스텀 모듈 등록


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()

    cfg = Config.fromfile(args.cfg)
    train_cfg = cfg.data.train
    train_cfg.pop("type", None)
    t0 = time.time()
    ds = DATASETS.get("Bench2DriveDataset")(**train_cfg)
    print(f"[build] dataset init {time.time()-t0:.1f}s, len={len(ds)}")

    pipeline = ds.pipeline
    transforms = pipeline.transforms
    names = [type(t).__name__ for t in transforms]
    print("pipeline:", names)

    # 워커 워밍업(첫 접근의 lazy 비용 제외)
    rng = np.random.default_rng(0)
    idxs = rng.integers(0, len(ds), size=args.n + 3).tolist()
    for j in idxs[:3]:
        _ = ds[j]

    agg_getinfo = 0.0
    agg_map = 0.0
    agg_tf = {nm: 0.0 for nm in names}
    n = 0
    for j in idxs[3:]:
        aug = ds.get_augmentation()
        t = time.time(); data = ds.get_data_info(j); agg_getinfo += time.time() - t
        # get_map_info 단독 (get_data_info 안에서도 호출되지만 순수비용 측정)
        t = time.time(); _ = ds.get_map_info(j); agg_map += time.time() - t
        data["aug_config"] = aug
        for tf, nm in zip(transforms, names):
            t = time.time()
            data = tf(data)
            agg_tf[nm] += time.time() - t
            if data is None:
                break
        n += 1

    print(f"\n=== 샘플 {n}개 평균 (ms) ===")
    print(f"  get_data_info(전체, get_ann/get_map 포함): {agg_getinfo/n*1000:8.1f}")
    print(f"  └ get_map_info(단독):                     {agg_map/n*1000:8.1f}")
    print("  --- pipeline transforms ---")
    for nm in names:
        print(f"    {nm:38s}: {agg_tf[nm]/n*1000:8.1f}")
    total = (agg_getinfo + sum(agg_tf.values())) / n * 1000
    print(f"  {'≈ 샘플당 총합':38s}: {total:8.1f}  (배치8/worker6 → data_time≈총합*8/6/1000 s 근사)")


if __name__ == "__main__":
    main()
