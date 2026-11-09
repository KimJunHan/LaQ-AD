# 수정(2026-11-05): 사용자 지시 "MCMC SFT 확인한대로 진행" — 새 파일. SFT config·데이터 경로 검증(CPU).
"""두 가지 모드.
  configs : 기반 2단계 config(train_e3_nuim) 대비 SFT config 4 개의 병합 결과 차이를 평탄화해 출력하고, (c)(d)의 train 파이프라인이
            (b) 와 Collect keys 에 'gt_ego_plan_sft_2hz' 하나만 더한 것인지 단언한다.
  samples : config 의 data.train 으로 데이터셋을 만들고 지정 색인의 표본을 고정 시드로 뽑아 pickle 로 저장(비교는 compare 모드).
  compare : 두 samples 파일을 키·바이트 단위로 비교. --allow_extra 키는 한쪽에만 있어도 된다(그 값은 --targets 파일 값과 비트 비교).
사용(저장소 루트, PYTHONPATH=.):
  python tools/mcmc_sft/check_sft_configs.py configs
  python tools/mcmc_sft/check_sft_configs.py samples <config> <out.pkl> --idx 0 1 500 20000
  python tools/mcmc_sft/check_sft_configs.py compare a.pkl b.pkl [--allow_extra gt_ego_plan_sft_2hz --targets x.npz]
"""
import argparse
import importlib
import pickle
import random
import sys

import numpy as np

C = "configs/nuscenes/v2_teacher/stage2/"
BASE = C + "train_e3_nuim.py"
SFT = {k: C + f"mcmc_sft/sft_{k}.py" for k in ("gt", "mcmc", "randc")}
SFT["target_base"] = C + "mcmc_sft/sft_target_base.py"


def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{p}.{k}" if p else str(k)))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            out.update(flat(v, f"{p}[{i}]"))
        if not d:
            out[p] = repr(d)
    else:
        out[p] = repr(d)
    return out


def cmd_configs():
    from mmcv import Config
    base = flat(Config.fromfile(BASE)._cfg_dict.to_dict())
    cfgs = {k: Config.fromfile(v) for k, v in SFT.items()}
    for k, c in cfgs.items():
        f = flat(c._cfg_dict.to_dict())
        diff = sorted(x for x in set(base) | set(f) if base.get(x) != f.get(x))
        print(f"== {k} vs train_e3_nuim: {len(diff)} 항목")
        for x in diff:
            print(f"   {x}: {base.get(x)} -> {f.get(x)}")
    pg = cfgs["gt"].data.train.pipeline
    for k in ("mcmc", "randc"):
        pk = cfgs[k].data.train.pipeline
        assert len(pk) == len(pg)
        for a, b in zip(pg, pk):
            if a["type"] == "Collect":
                assert list(b["keys"]) == list(a["keys"]) + ["gt_ego_plan_sft_2hz"], b["keys"]
                assert b["meta_keys"] == a["meta_keys"]
            else:
                assert a == b, (a, b)
        assert cfgs[k].data.train.plan_sft_target_file.endswith(f"targets_{k}.npz")
        assert cfgs[k].model.head.onedecoder_head.plan_sft_key == "gt_ego_plan_sft_2hz"
    assert "plan_sft_target_file" not in cfgs["gt"].data.train and "plan_sft_key" not in cfgs["gt"].model.head.onedecoder_head
    for k in ("mcmc", "randc"):
        for split in ("val", "test"):
            assert "plan_sft_target_file" not in cfgs[k].data[split]
    print("[OK] (c)(d) 파이프라인 = (b) + Collect 키 1 개, (b) 는 SFT 옵션 없음, val/test 에 SFT 파일 없음")


def to_np(x):
    from mmcv.parallel import DataContainer
    import torch
    if isinstance(x, DataContainer):
        return to_np(x.data)
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:  # 교사 캐시(bf16): 비트 그대로 int16 으로 본다(바이트 비교용)
            return x.view(torch.int16).numpy()
        return x.numpy()
    if isinstance(x, dict):
        return {k: to_np(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_np(v) for v in x]
    return x


def cmd_samples(cfg_path, out, idx):
    import torch
    from mmcv import Config
    cfg = Config.fromfile(cfg_path)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    importlib.import_module("projects.mmdet3d_plugin")
    from mmdet.datasets import build_dataset
    ds = build_dataset(cfg.data.train)
    res = {}
    for i in idx:
        np.random.seed(1000 + i); random.seed(1000 + i); torch.manual_seed(1000 + i)
        # GroupInBatchSampler 와 같은 형식(dict: idx·aug_config·stream_group) — 학습 경로 그대로
        res[i] = to_np(ds[dict(idx=i, aug_config=ds.get_augmentation(), stream_group=0)])
        res[i]["_token"] = ds.data_infos[i]["token"]
    pickle.dump(res, open(out, "wb"))
    print(f"[저장] {out}: {len(res)} 표본, 키 {sorted(res[idx[0]].keys())}")


def eq(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a = np.asarray(a); b = np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    return type(a) == type(b) and a == b


def cmd_compare(pa, pb, allow_extra, targets):
    # 이 스크립트가 samples 모드로 직접 만든 로컬 파일만 읽는다(외부 입력 아님)
    A = pickle.load(open(pa, "rb")); B = pickle.load(open(pb, "rb"))
    assert A.keys() == B.keys()
    T = None
    if targets:
        z = np.load(targets)
        T = {str(t): z["gt_ego_plan_sft_2hz"][k] for k, t in enumerate(z["token"])}
    bad = 0
    for i in A:
        ka, kb = set(A[i]), set(B[i])
        extra = (ka ^ kb)
        if not extra <= set(allow_extra):
            print(f"색인 {i}: 허용되지 않은 키 차이 {extra}"); bad += 1
        for k in ka & kb:
            if not eq(A[i][k], B[i][k]):
                print(f"색인 {i}: 키 {k} 값이 다르다"); bad += 1
        for k in extra:
            v = (A[i] if k in A[i] else B[i])[k]
            tgt = T[A[i]["_token"]]
            same = np.asarray(v).tobytes() == tgt.tobytes() and np.asarray(v).dtype == tgt.dtype
            gt = A[i].get("gt_ego_fut_trajs_2hz", B[i].get("gt_ego_fut_trajs_2hz"))
            print(f"색인 {i}: {k} == 정답 파일 값 {same}; GT 와 최대차 {np.abs(np.asarray(v) - np.asarray(gt)).max():.4f} m")
            bad += int(not same)
    print("[OK] 동일" if bad == 0 else f"[실패] {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["configs", "samples", "compare"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--idx", type=int, nargs="+", default=[0, 1, 500, 20000])
    ap.add_argument("--allow_extra", nargs="*", default=[])
    ap.add_argument("--targets", default=None)
    a = ap.parse_args()
    if a.mode == "configs":
        cmd_configs()
    elif a.mode == "samples":
        cmd_samples(a.args[0], a.args[1], a.idx)
    else:
        cmd_compare(a.args[0], a.args[1], a.allow_extra, a.targets)
