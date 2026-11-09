# 진단(2026-11-05): 2단계 모델의 val 자차 상태 예측(ego_refine 마지막 계층 출력 10 차원)과 계획 출력을 fp32 로 덤프.
#   tools/latency/eval_teacher_subset.py 의 파이프라인 변환(LatencyTeacherFolderFilter·TokenSwap)을 import 해 재사용(그 파일은 수정 안 함).
#   차이: wrap_fp16_model 을 하지 않는다(fp32), dataset.evaluate 대신 계획·자차 상태만 저장한다.
#   진단 전용 — val 로 무엇도 고르지 않는다.
#   사용(저장소 루트): python tools/diag/ego_probe.py <config> <ckpt> --out results/diag/x.npz [--drop ctx_txt] [--lag 1] [--limit N]
import argparse
import json
import os.path as osp
import pickle
import sys
import time

import numpy as np
import torch

_REPO = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, osp.join(_REPO, "tools", "latency"))
import eval_teacher_subset as ets  # noqa: E402  (pipeline 등록 + chdir(_REPO) + torch.load shim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop", default="")
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    drop = [f for f in a.drop.split(",") if f]
    cfg = ets.Config.fromfile(a.config)
    import importlib
    importlib.import_module("projects.mmdet3d_plugin")
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    ets.set_random_seed(0, deterministic=False)
    pipe = list(cfg.data.test.pipeline)
    li = [i for i, p in enumerate(pipe) if p["type"] == "LoadTeacherCacheNusc"][0]
    infos = pickle.load(open(cfg.data.test.ann_file, "rb"))["infos"]
    if a.lag:
        tmap, _ = ets.build_token_map(infos, a.lag)
        mf = a.out.replace(".npz", "_tokmap.json")
        json.dump(tmap, open(mf, "w"))
        pipe = pipe[:li] + [dict(type="LatencyTeacherTokenSwap", map_file=mf)] + pipe[li:]
        li += 1
    pipe = pipe[:li + 1] + [dict(type="LatencyTeacherFolderFilter", drop=drop, mask=[])] + pipe[li + 1:]
    cfg.data.test.pipeline = pipe
    dataset = ets.build_dataset(cfg.data.test)
    n = len(dataset) if not a.limit else a.limit
    if a.limit:
        dataset = torch.utils.data.Subset(dataset, list(range(a.limit)))
    loader = ets.build_dataloader_origin(dataset, samples_per_gpu=1, workers_per_gpu=a.workers, dist=False, shuffle=False)
    cfg.model.train_cfg = None
    model = ets.build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    # fp32: wrap_fp16_model 생략
    ets.load_checkpoint(model, a.checkpoint, map_location="cpu")
    model = ets.MMDataParallel(model, device_ids=[0]).eval()
    cap = {}
    ego_mods = [(nm, m) for nm, m in model.named_modules() if nm.endswith("ego_refine.5")]
    assert len(ego_mods) == 1, [x[0] for x in ego_mods]
    ego_mods[0][1].register_forward_hook(lambda m, i, o: cap.__setitem__("ego", o.detach().float().cpu()))
    E, P, M, Pr = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for k, data in enumerate(loader):
            out = model(return_loss=False, rescale=True, **data)
            r = out[0]["img_bbox"]
            E.append(cap["ego"].reshape(-1)[:10].numpy())
            P.append(r["plan_temp_2hz"].float().numpy()[:6, :2])
            M.append(r["plan_modes"].float().numpy()[:, :6, :2])
            Pr.append(r["plan_mode_probs"].float().numpy())
            if k % 500 == 0:
                print(k, n, f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(a.out, ego=np.stack(E), plan=np.stack(P), modes=np.stack(M), probs=np.stack(Pr),
                        drop=",".join(drop), lag=a.lag, ckpt=a.checkpoint, sec=time.time() - t0,
                        peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    print("done", time.time() - t0, torch.cuda.max_memory_allocated() / 1e9)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
