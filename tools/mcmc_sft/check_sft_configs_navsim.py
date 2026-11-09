# 수정(2026-11-06): 사용자 결정 "NAVSIM MCMC SFT" — 새 파일. tools/mcmc_sft/check_sft_configs.py(nuScenes, 무수정)의 NAVSIM 판.
"""모드
  configs : NAVSIM 2단계 train_e3_nuim 대비 SFT config(base/mcmc/randc/target_base; 수정(2026-11-06) GT-SFT 군 제거) 병합 차이를 출력하고 단언한다 —
            base 는 SFT 옵션 없음, 정답 파일 군의 train 파이프라인 = base + Collect 키 'gt_ego_plan_sft_2hz' 하나, val/test 에 SFT 파일 없음,
            max_iters 25,812 / lr 2e-6 Fixed / EMA 0.997959·196·6 / load_from = 2단계 EMA.
  samples, compare : check_sft_configs.py 와 같다(데이터셋 표본 저장·바이트 비교). samples 는 --target_file 로 정답 파일을 바꿀 수 있다.
사용(저장소 루트, hipad env, PYTHONPATH=.):
  python tools/mcmc_sft/check_sft_configs_navsim.py configs
  python tools/mcmc_sft/check_sft_configs_navsim.py samples <config> <out.pkl> --idx 0 1 500 [--target_file x.npz]
  python tools/mcmc_sft/check_sft_configs_navsim.py compare a.pkl b.pkl [--allow_extra gt_ego_plan_sft_2hz --targets x.npz]
"""
import argparse
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_sft_configs as NU  # noqa: E402  (flat, to_np, eq 재사용)

C = "configs/navsim/v2_teacher/stage2/"
BASE = C + "train_e3_nuim_gtfilt.py"   # 수정(2026-11-09): GT 필터 2단계 #이전 "train_e3_nuim.py"
SFT = {k: C + f"mcmc_sft/sft_{k}.py" for k in ("base", "mcmc", "randc", "target_base")}  # 수정(2026-11-06): 사용자 결정 "GT-SFT 군 제거" #이전 ("base", "gt", "mcmc", "randc", "target_base")
TARGET_DIR = "results/mcmc_sft_navsim/s2e3nuim_gtfilt_ema_v1"   # 수정(2026-11-09): #이전 s2e3nuim_ema_v1


def cmd_configs():
    from mmcv import Config
    base = NU.flat(Config.fromfile(BASE)._cfg_dict.to_dict())
    cfgs = {k: Config.fromfile(v) for k, v in SFT.items()}
    for k, c in cfgs.items():
        f = NU.flat(c._cfg_dict.to_dict())
        diff = sorted(x for x in set(base) | set(f) if base.get(x) != f.get(x))
        print(f"== {k} vs train_e3_nuim: {len(diff)} 항목")
        for x in diff:
            print(f"   {x}: {base.get(x)} -> {f.get(x)}")
    b = cfgs["base"]
    pb = Config.fromfile(BASE).data.train.pipeline
    assert b.data.train.pipeline == pb, "base 파이프라인이 2단계와 다르다"
    assert "plan_sft_target_file" not in b.data.train and "plan_sft_key" not in b.model.head.onedecoder_head
    for k in ("mcmc", "randc", "target_base"):  # 수정(2026-11-06): 사용자 결정 "GT-SFT 군 제거" #이전 ("gt", "mcmc", "randc", "target_base")
        c = cfgs[k]
        pk = c.data.train.pipeline
        assert len(pk) == len(pb)
        for x, y in zip(pb, pk):
            if x["type"] == "Collect":
                assert list(y["keys"]) == list(x["keys"]) + ["gt_ego_plan_sft_2hz"], y["keys"]
                assert y["meta_keys"] == x["meta_keys"]
            else:
                assert x == y, (x, y)
        assert c.model.head.onedecoder_head.plan_sft_key == "gt_ego_plan_sft_2hz"
        want = None if k == "target_base" else f"{TARGET_DIR}/targets_{k}.npz"
        assert c.data.train.plan_sft_target_file == want, (k, c.data.train.plan_sft_target_file)
        for split in ("val", "test"):
            assert "plan_sft_target_file" not in c.data[split]
    for k, c in cfgs.items():
        assert c.runner.max_iters == 25812 and c.optimizer.lr == 2e-6 and c.lr_config == dict(policy="Fixed"), k
        h = c.custom_hooks[0]
        assert (h["type"], h["decay"], h["ramp_steps"], h["update_every"]) == ("EMASidecarHook", 0.997959, 196, 6), k
        assert c.load_from == "./results/hipad_navsim_laq_v2_teacher_stage2_e3_nuim_gtfilt/iter_129060_ema.pth", k   # 수정(2026-11-09)
        assert c.optimizer_config.cumulative_iters == 6 and c.data.samples_per_gpu == 8, k
        assert c.data.train.sequences_split_num == 1, k
    print("[OK] base = SFT 옵션 없음·2단계 파이프라인, 정답 파일 군 = base + Collect 키 1 개, 일정·EMA·load_from 단언 통과")


def cmd_samples(cfg_path, out, idx, target_file):
    import importlib
    import numpy as np
    import torch
    from mmcv import Config
    cfg = Config.fromfile(cfg_path)
    if target_file is not None:
        cfg.data.train.plan_sft_target_file = target_file
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    importlib.import_module("projects.mmdet3d_plugin")
    from mmdet.datasets import build_dataset
    ds = build_dataset(cfg.data.train)
    res = {}
    for i in idx:
        np.random.seed(1000 + i); random.seed(1000 + i); torch.manual_seed(1000 + i)
        res[i] = NU.to_np(ds[dict(idx=i, aug_config=ds.get_augmentation(), stream_group=0)])
        res[i]["_token"] = ds.data_infos[i]["token"]
    pickle.dump(res, open(out, "wb"))
    print(f"[저장] {out}: {len(res)} 표본, 키 {sorted(res[idx[0]].keys())}")


def cmd_compare(pa, pb, allow_extra, targets):
    """check_sft_configs.cmd_compare 와 같은 판정. 정답 파일 배열을 한 번만 읽는다(종전 함수는 token 마다 npz 를 다시 풀어 15 만 token 에서 매우 느림)."""
    import numpy as np
    A = pickle.load(open(pa, "rb")); B = pickle.load(open(pb, "rb"))   # 이 스크립트 samples 모드가 만든 로컬 파일
    assert A.keys() == B.keys()
    T = None
    if targets:
        z = np.load(targets)
        arr = z["gt_ego_plan_sft_2hz"]
        T = {str(t): k for k, t in enumerate(z["token"])}
    bad = 0
    for i in A:
        ka, kb = set(A[i]), set(B[i])
        extra = ka ^ kb
        if not extra <= set(allow_extra):
            print(f"색인 {i}: 허용되지 않은 키 차이 {extra}"); bad += 1
        for k in ka & kb:
            if not NU.eq(A[i][k], B[i][k]):
                print(f"색인 {i}: 키 {k} 값이 다르다"); bad += 1
        for k in extra:
            v = np.asarray((A[i] if k in A[i] else B[i])[k])
            tgt = arr[T[A[i]["_token"]]]
            same = v.tobytes() == tgt.tobytes() and v.dtype == tgt.dtype
            gt = A[i].get("gt_ego_fut_trajs_2hz", B[i].get("gt_ego_fut_trajs_2hz"))
            print(f"색인 {i}: {k} == 정답 파일 값 {same}; GT 와 최대차 {np.abs(v - np.asarray(gt)).max():.4f} m")
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
    ap.add_argument("--target_file", default=None, help="samples: data.train.plan_sft_target_file 덮어쓰기(시험용)")
    a = ap.parse_args()
    if a.mode == "configs":
        cmd_configs()
    elif a.mode == "samples":
        cmd_samples(a.args[0], a.args[1], a.idx, a.target_file)
    else:
        cmd_compare(a.args[0], a.args[1], a.allow_extra, a.targets)
