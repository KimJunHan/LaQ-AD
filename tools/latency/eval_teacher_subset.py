# 수정(2026-11-04): 교사 답 구간 사용도·교사 주기 진단 — 추론 전용 평가 래퍼. 기존 파일(모델·config·파이프라인·evaluation)은 건드리지 않는다.
#   tools/test.py 의 단일 GPU 평가 경로(같은 seed 0·fp16 wrap·MMDataParallel·single_gpu_test·dataset.evaluate)를 그대로 따르고,
#   테스트 파이프라인의 LoadTeacherCacheNusc 앞뒤에만 변환을 끼운다:
#     --drop f1,f2   : 그 폴더 토큰을 qwen_hidden/qwen_mask/qwen_type 에서 잘라 낸다(모델이 보기 전에).
#                      교차 어텐션은 쿼리 행마다 독립 softmax 이므로 "잘라 냄" = "key_padding_mask 로 가림" 과 수학적으로 같다.
#                      잘라 내면 마스크가 전부 1 이라 평가와 같은 비패딩 flash 경로를 탄다(unpad 경로 수치 차이 없음).
#     --mask f1,..   : (대안) 잘라 내지 않고 qwen_mask 를 0 으로 — QwenTokenAttn 이 key_padding_mask 로 넘긴다.
#     --lag K        : 프레임 t 가 같은 장면의 t-K 프레임 교사 캐시를 쓴다(장면 앞 K 프레임은 max(t-K, 장면 첫 프레임)).
#     --period P     : 교사를 장면 안 P 프레임마다 한 번(색인 0, P, 2P, ...) 돌리고 그 사이는 마지막 교사 출력을 쓴다(나이 0..P-1).
#   진단 전용(val 로 무엇도 고르지 않는다 — 결과는 판정·튜닝 수치가 아니다).
#   사용(저장소 루트에서):
#     source /home1/irteam/data-vol1/hipad_env.sh; export CUDA_VISIBLE_DEVICES=0 HIPAD_ALLOW_GPU0=1
#     python tools/latency/eval_teacher_subset.py <config> <ckpt> --out-dir results/latency/teacher_subset/<name> [--drop map] [--lag 1]
import argparse
import json
import os
import os.path as osp
import sys
import time

import torch

_REPO = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)

_orig_torch_load = torch.load


def _torch_load_compat(*a, **kw):  # tools/test.py 와 같은 shim
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)


torch.load = _torch_load_compat

import mmcv  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmcv.parallel import DataContainer as DC  # noqa: E402
from mmcv.runner import load_checkpoint, wrap_fp16_model  # noqa: E402
from mmdet.apis import single_gpu_test, set_random_seed  # noqa: E402
from mmdet.datasets import build_dataset  # noqa: E402
from mmdet.datasets import build_dataloader as build_dataloader_origin  # noqa: E402
from mmdet.datasets.builder import PIPELINES  # noqa: E402
from mmdet.models import build_detector  # noqa: E402

FOLDERS = ("context", "detection", "tracking", "motion", "map", "plan", "action")
# context 폴더 안 구간(laq_cache_layout.context_positions 순서): 현재 프레임 6 캠 영상 토큰 1,080 다음 user 텍스트 752.
CTX_PARTS = {"ctx_img": (0, 1080), "ctx_txt": (1080, 1832)}


@PIPELINES.register_module()
class LatencyTeacherTokenSwap(object):
    """LoadTeacherCacheNusc 앞: token 을 교사 캐시를 읽을 프레임의 token 으로 바꾼다(원래 값은 _orig_token 에 보관)."""

    def __init__(self, map_file):
        with open(map_file) as f:
            self.m = json.load(f)

    def __call__(self, d):
        d["_orig_token"] = d["token"]
        d["token"] = self.m[d["token"]]
        return d


@PIPELINES.register_module()
class LatencyTeacherFolderFilter(object):
    """LoadTeacherCacheNusc 뒤: token 복원 + 폴더 제거(drop) 또는 가림(mask)."""

    def __init__(self, drop=(), mask=()):
        self.drop_parts = [CTX_PARTS[f] for f in drop if f in CTX_PARTS]
        self.drop = sorted(FOLDERS.index(f) for f in drop if f not in CTX_PARTS)
        self.mask = sorted(FOLDERS.index(f) for f in mask)

    def __call__(self, d):
        if "_orig_token" in d:
            d["token"] = d.pop("_orig_token")
        if not self.drop and not self.mask and not self.drop_parts:
            return d
        h, m, t = d["qwen_hidden"].data, d["qwen_mask"].data, d["qwen_type"].data  # (1,L,H) (1,L) (1,L)
        if self.drop or self.drop_parts:
            keep = ~torch.isin(t[0], torch.tensor(self.drop, dtype=t.dtype))
            ctx = torch.nonzero(t[0] == 0).squeeze(1)
            if self.drop_parts and ctx.numel() != 1832:
                raise ValueError(f"context 길이 {ctx.numel()} != 1832")
            for a0, a1 in self.drop_parts:
                keep[ctx[a0:a1]] = False
            if not bool(keep.any()):
                raise ValueError("모든 토큰을 제거할 수 없다")
            h, m, t = h[:, keep], m[:, keep], t[:, keep]
        if self.mask:
            m = m.clone()
            m[0, torch.isin(t[0], torch.tensor(self.mask, dtype=t.dtype))] = 0
        d["qwen_hidden"] = DC(h.contiguous(), stack=True, pad_dims=None)
        d["qwen_mask"] = DC(m.contiguous(), stack=True, pad_dims=None)
        d["qwen_type"] = DC(t.contiguous(), stack=True, pad_dims=None)
        return d


def build_token_map(infos, lag=0, period=1, cross_scene=0, const=None):
    """프레임 token → 교사 캐시를 읽을 token. 장면 안 시간 순서 기준. 반환 (map, 나이 통계).
    cross_scene=S>0: 장면 순서에서 S 칸 뒤 장면(순환)의 같은 상대 위치(길이 넘으면 마지막) 프레임 — 다른 장면 교사 토큰(대조 진단).
    const=token: 모든 프레임이 그 한 프레임의 교사 토큰(대조 진단)."""
    by_scene = {}
    for i, info in enumerate(infos):
        by_scene.setdefault(info["scene_token"], []).append(i)
    m, ages = {}, []
    order = [sorted(v, key=lambda i: infos[i]["timestamp"]) for v in by_scene.values()]
    if const is not None or cross_scene:
        for si, idxs in enumerate(order):
            other = order[(si + cross_scene) % len(order)]
            for p, i in enumerate(idxs):
                m[infos[i]["token"]] = const if const is not None else infos[other[min(p, len(other) - 1)]]["token"]
                ages.append(-1)
        return m, ages
    for idxs in order:
        for p, i in enumerate(idxs):
            q = max(p - lag, 0)
            if period > 1:
                q = (q // period) * period
            m[infos[i]["token"]] = infos[idxs[q]]["token"]
            ages.append(p - q)
    return m, ages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--drop", default="")
    ap.add_argument("--mask", default="")
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--period", type=int, default=1)
    ap.add_argument("--cross-scene", type=int, default=0)
    ap.add_argument("--const-token", default=None)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--result-file", default=None, help="이미 만든 results.pkl 로 채점만")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cudnn-benchmark", type=int, default=0,
                    help="1 = tools/test.py 처럼 cfg.cudnn_benchmark(True)를 켠다. 기본 0(결정적 기본 알고리즘 — 진단 런 사이 비교용)")
    ap.add_argument("--limit", type=int, default=0, help="앞 N 프레임만 추론(결정성 검사용, 채점 안 함)")
    a = ap.parse_args()
    drop = [f for f in a.drop.split(",") if f]
    mask = [f for f in a.mask.split(",") if f]
    for f in drop + mask:
        if f not in FOLDERS and not (f in CTX_PARTS and f in drop):
            raise SystemExit(f"알 수 없는 폴더 {f}")
    os.makedirs(a.out_dir, exist_ok=True)

    cfg = Config.fromfile(a.config)
    import importlib
    importlib.import_module("projects.mmdet3d_plugin")  # tools/test.py 의 plugin import 와 같음(plugin_dir 기준)
    if a.cudnn_benchmark and cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    set_random_seed(0, deterministic=False)  # tools/test.py 기본(--seed 0)
    cfg.work_dir = osp.join("./work_dirs", "latency_teacher_subset")
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.data.test.work_dir = cfg.work_dir

    pipe = list(cfg.data.test.pipeline)
    li = [i for i, p in enumerate(pipe) if p["type"] == "LoadTeacherCacheNusc"]
    assert len(li) == 1, li
    li = li[0]
    ages = None
    if a.lag or a.period > 1 or a.cross_scene or a.const_token:
        import pickle
        infos = pickle.load(open(cfg.data.test.ann_file, "rb"))["infos"]
        tmap, ages = build_token_map(infos, a.lag, a.period, a.cross_scene, a.const_token)
        mf = osp.join(a.out_dir, "teacher_token_map.json")
        with open(mf, "w") as f:
            json.dump(tmap, f)
        pipe = pipe[:li] + [dict(type="LatencyTeacherTokenSwap", map_file=mf)] + pipe[li:]
        li += 1
    pipe = pipe[:li + 1] + [dict(type="LatencyTeacherFolderFilter", drop=drop, mask=mask)] + pipe[li + 1:]
    cfg.data.test.pipeline = pipe

    meta = dict(config=a.config, checkpoint=a.checkpoint, drop=drop, mask=mask, lag=a.lag, period=a.period,
                cross_scene=a.cross_scene, const_token=a.const_token,
                pipeline=[p["type"] for p in pipe], started=time.strftime("%F %T"))
    if ages is not None:
        import collections
        meta["teacher_age_hist"] = dict(sorted(collections.Counter(ages).items()))
    print(json.dumps(meta, ensure_ascii=False), flush=True)

    dataset = build_dataset(cfg.data.test)
    meta["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    if a.limit:
        dataset = torch.utils.data.Subset(dataset, list(range(a.limit)))
        a.no_eval = True
    if a.result_file:
        outputs = mmcv.load(a.result_file)
    else:
        loader = build_dataloader_origin(dataset, samples_per_gpu=1,
                                         workers_per_gpu=a.workers if a.workers is not None else cfg.data.workers_per_gpu,
                                         dist=False, shuffle=False)
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
        if cfg.get("fp16", None) is not None:
            wrap_fp16_model(model)
        load_checkpoint(model, a.checkpoint, map_location="cpu")
        model = MMDataParallel(model, device_ids=[0])
        t0 = time.time()
        outputs = single_gpu_test(model, loader, False, None)
        meta["infer_sec"] = time.time() - t0
        meta["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        mmcv.dump(outputs, osp.join(a.out_dir, "results.pkl"))
        del model
        torch.cuda.empty_cache()
    if not a.no_eval:
        ek = cfg.get("evaluation", {}).copy()
        for k in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
            ek.pop(k, None)
        ek.update(dict(metric="bbox", jsonfile_prefix=a.out_dir))
        res = dataset.evaluate(outputs, **ek)
        print(res)
        meta["metrics"] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in res.items()}
        # 대용량 제출 파일은 지운다(지표·results.pkl 은 남김).
        for fn in ("results_nusc.json", "results_nusc_tracking.json", "results_nusc_tracking_relink.json",
                   "submission_vector.json"):
            p = osp.join(a.out_dir, fn)
            if osp.exists(p):
                os.remove(p)
    meta["finished"] = time.strftime("%F %T")
    with open(osp.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
