# 수정(2026-11-09): N-A13(docs/pipeline_gap_navsim_b2d_20261109.md §1.6) — NAVSIM PTQ(학습 후 양자화) 정확도 경로. 새 파일.
#   nuScenes 도구 tools/quant/ptq_fakequant.py(가짜 양자화 함수·hook·BN 접기·ego DAF 탐침)를 그대로 import 해 쓰고,
#   데이터 쪽(보정 = navtrain, 평가 = navtest → navsim_pred.pkl)만 NAVSIM 에 맞게 바꾼다. 모델 소스·config·기존 도구는 고치지 않는다.
#   가짜 양자화 = 정확도 시뮬레이션(연산은 원 정밀도 그대로) — 지연 이득을 뜻하지 않는다.
#
#   흐름(추론 전용):
#     1) --quant 가 none 이 아니면 모델 A(보정용)에 BN 접기 + 관측 hook. 보정 표본 = **navtrain 전용** 장면 창
#        (navtest 와 같은 모양: 연속 4 프레임 = 과거 3 + 현재(is_eval_frame), 0.5 s 간격, 같은 로그·연속 frame_index) N 개(기본 64 × 4 = 256, seed 고정).
#        창마다 scene_token 을 "ptq_calib#<번호>" 로 바꾼 보정 info pkl 을 출력 폴더에 쓰고(기록), 모델의 reset_bank_on_scene_change 가
#        navtest 와 똑같이 창 경계에서 시간 뱅크를 비우게 한다. 보정 프레임 token 이 navtest·navhard info 의 프레임 token 과 하나라도 겹치면 멈춘다.
#     2) 모델 B 를 새로 만들어 같은 접기·가중치 양자화·활성 양자화 hook(보정 amax)을 단다(보정 상태 누출 차단).
#     3) navtest(또는 --limit-scenes N: 앞 N 장면 = 4N 프레임의 부분 info pkl 을 출력 폴더에 만들어) 순차 추론(batch 1) →
#        dataset.evaluate(= tools/test.py --eval bbox 와 같은 NavsimDataset.evaluate) → <out>/navsim_pred.pkl.
#        EPDMS 채점은 이 도구에서 하지 않는다(scripts/navsim/run_ptq_navsim_20261109.sh 가 CPU 에서 따로).
#   ego DAF 출력 사영(ego_deformable.*.output_proj) 규칙(docs/ptq_nuscenes_20261109.md §2 를 NAVSIM 에 옮김, bb_dec 범위에서만 의미):
#     --probe-ego-windows N: val/test 대신 navtrain 창 N 개(--probe-seed, 보정과 다른 seed)로 (a) 원 모델 / (b) --quant 모델에서
#       층별 비0 프레임·amax 를 기록(json). 그 뒤 --ego-amax-json 으로 amax = max(보정 amax, 탐침 비0 amax), 비0 사례가 전혀 없으면 --exclude-ego-out.
#   정밀도(--precision): mixed = config 그대로(학습·평가 정밀도: 백본 fp16 auto_fp16, FPN bf16, 디코더 fp32), fp32 = fp16 끔.
#   사용:
#     python tools/quant/ptq_navsim.py <config> <ckpt> --out-dir <dir> --precision mixed --quant int8 --scope bb [--limit-scenes 50]
import argparse
import copy
import json
import os
import os.path as osp
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import ptq_fakequant as F  # noqa: E402  (저장소 경로 등록·torch.load shim 포함)

import mmcv  # noqa: E402
from mmcv import Config, DictAction  # noqa: E402
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmdet.apis import single_gpu_test, set_random_seed  # noqa: E402
from mmdet.datasets import build_dataset  # noqa: E402
from mmdet.datasets import build_dataloader as build_dataloader_origin  # noqa: E402

NAVHARD_INFOS = ["data/infos/navsim_navhard_20261105/navsim_navhard_two_stage.pkl"]   # 보정 누출 검사용(평가 분할)
WIN_LEN = 4   # navtest 장면 창 = 과거 3 + 현재 (navsim_navtest.pkl: 장면마다 4 프레임)


def load_pkl(p):
    d = pickle.load(open(p, "rb"))
    return d["infos"], d.get("metadata", {})


def eval_frame_tokens(cfg):
    """평가 분할(navtest = cfg.data.test 원 ann_file, navhard) 프레임 token 전체 — 보정·탐침 누출 검사용."""
    toks = set()
    srcs = [cfg.data.test.ann_file] + [p for p in NAVHARD_INFOS if osp.exists(p)]
    for p in srcs:
        infos, _ = load_pkl(p)
        toks |= {x["token"] for x in infos}
    return toks, srcs


def navtrain_windows(infos, n, seed, length=WIN_LEN):
    """navtrain 의 현재 프레임(is_eval_frame) 중 앞 length-1 프레임이 같은 로그 장면·연속 frame_index·0.5 s 간격인 것만 후보 →
    seed 고정 무작위 n 개(중복 없음). 반환 = [[i0, i1, i2, i3], ...] (시간 순)."""
    cand = []
    for i, x in enumerate(infos):
        if not x.get("is_eval_frame") or i < length - 1:
            continue
        w = list(range(i - length + 1, i + 1))
        if any(infos[j]["scene_token"] != x["scene_token"] for j in w):
            continue
        fi = [infos[j]["frame_index_in_log"] for j in w]
        ts = np.diff([infos[j]["timestamp"] for j in w]) / 1e6
        if any(b - a != 1 for a, b in zip(fi, fi[1:])) or not np.all((ts > 0.4) & (ts < 0.6)):
            continue
        cand.append(w)
    if len(cand) < n:
        raise RuntimeError(f"navtrain 창 후보 {len(cand)} < 요청 {n}")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cand), size=n, replace=False)
    return [cand[int(p)] for p in pick], len(cand)


def write_window_ann(cfg, n, seed, out_path, tag):
    """navtrain 창 n 개의 info pkl(창마다 scene_token = f"{tag}#번호") — 보정/탐침 표본 기록이자 데이터셋 입력."""
    src = cfg.data.train.ann_file
    infos, meta = load_pkl(src)
    if meta.get("kind") != "train_hist" or "navtrain" not in osp.basename(src):
        raise ValueError(f"보정은 navtrain(train_hist) 만: {src} kind={meta.get('kind')}")
    wins, n_cand = navtrain_windows(infos, n, seed)
    ev, srcs = eval_frame_tokens(cfg)
    flat = [i for w in wins for i in w]
    leak = sorted({infos[i]["token"] for i in flat} & ev)
    if leak:
        raise RuntimeError(f"보정 표본에 평가 분할(navtest/navhard) 프레임 token {len(leak)} 개 — 평가셋 사용 금지 ({srcs})")
    out = []
    for k, w in enumerate(wins):
        for i in w:
            x = dict(infos[i])
            x["scene_token"] = f"{tag}#{k:05d}"
            x["src_scene_token"] = infos[i]["scene_token"]
            out.append(x)
    m = dict(meta)
    m.update(eval_tokens=[infos[w[-1]]["token"] for w in wins], num_scenes=len(wins), partial=True,
             ptq_windows=dict(src=src, n=n, seed=seed, length=WIN_LEN, n_candidates=n_cand, leak_checked_against=srcs))
    with open(out_path, "wb") as f:
        pickle.dump(dict(infos=out, metadata=m), f)
    return dict(ann_file=out_path, src=src, n_windows=n, n_frames=len(out), seed=seed, n_candidates=n_cand,
                first_window_current_tokens=[infos[w[-1]]["token"] for w in wins[:8]], leak_checked_against=srcs)


def navtrain_dataset_cfg(cfg, ann_file):
    """평가 파이프라인(데이터 증강 없음·교사 캐시 동일)에 navtrain 쪽 지도·ego_input 표를 붙인 test_mode 데이터셋 설정."""
    t = copy.deepcopy(cfg.data.test)
    t.ann_file = ann_file
    t.test_mode = True
    for k in ("teacher_map_files", "ego_input_file"):
        if k in cfg.data.train:
            t[k] = cfg.data.train[k]
        elif k in t:
            raise KeyError(f"data.test 에는 {k} 가 있는데 data.train 에는 없다 — navtrain 쪽 값을 정할 수 없다")
    t.pop("samples_per_gpu", None)
    return t


def write_test_subset(ann, n_scenes, out_path):
    """navtest 앞 n_scenes 장면(장면 = 연속 4 프레임, 마지막이 평가 프레임) 부분 info pkl. eval_tokens 도 그 장면만."""
    infos, meta = load_pkl(ann)
    order, seen = [], set()
    for x in infos:
        if x["scene_token"] not in seen:
            seen.add(x["scene_token"]); order.append(x["scene_token"])
    keep = set(order[:n_scenes])
    sub = [x for x in infos if x["scene_token"] in keep]
    ev = [x["navsim_token"] for x in sub if x.get("is_eval_frame")]
    if len(ev) != n_scenes or len(sub) != WIN_LEN * n_scenes:
        raise RuntimeError(f"부분 info 모양 이상: 평가 프레임 {len(ev)}, 프레임 {len(sub)} (장면 {n_scenes})")
    m = dict(meta); m.update(eval_tokens=ev, num_scenes=n_scenes, partial=True, subset_of=ann)
    with open(out_path, "wb") as f:
        pickle.dump(dict(infos=sub, metadata=m), f)
    return out_path


def check_reset_on_scene(model):
    if not getattr(model, "reset_bank_on_scene_change", False):
        raise RuntimeError("모델 reset_bank_on_scene_change=False — NAVSIM 창 경계 뱅크 초기화가 navtest 와 달라진다")


def run_windows(model, dcfg, workers):
    """창 info 데이터셋을 순차(batch 1)로 돌린다. 뱅크 초기화는 모델의 scene_token 경계 초기화(navtest 와 같은 경로)."""
    ds = build_dataset(dcfg)
    dl = build_dataloader_origin(ds, samples_per_gpu=1, workers_per_gpu=workers, dist=False, shuffle=False)
    dp = MMDataParallel(model, device_ids=[0])
    with torch.no_grad():
        for data in dl:
            dp(return_loss=False, rescale=True, **data)
    return len(ds)


def build_quant_model(cfg, args, stats, calib_ann):
    """보정(모델 A) → 새 모델 B 에 가중치·활성 가짜 양자화. 반환 (model B, hooks)."""
    mA = F.build_model(cfg, args.checkpoint).cuda()
    check_reset_on_scene(mA)
    hA, excluded = F.attach(mA, args.scope, args.quant)
    t0 = time.time()
    n = run_windows(mA, navtrain_dataset_cfg(cfg, calib_ann), cfg.data.workers_per_gpu)
    stats["calib_sec"] = time.time() - t0
    stats["calib_frames"] = n
    print(f"[ptq-navsim] 보정 {n} 프레임 {stats['calib_sec']:.0f}s", flush=True)
    amax = {k: h.amax for k, (m, h) in hA.items()}
    calls = {k: h.calls for k, (m, h) in hA.items()}
    del mA, hA
    torch.cuda.empty_cache()
    model = F.build_model(cfg, args.checkpoint)
    hooks, excluded_b = F.attach(model, args.scope, args.quant)
    assert set(hooks) == set(amax) and excluded_b == excluded, "보정·평가 모델의 대상 모듈이 다르다"
    ego_names = [k for k in hooks if k.startswith("head.onedecoder_head.ego_deformable.") and k.endswith(".output_proj")]
    if args.exclude_ego_out:
        for k in ego_names:
            m, h = hooks.pop(k); m._forward_pre_hooks.clear(); excluded.append(k + "(ego DAF 출력, 규칙상 제외)")
    elif args.ego_amax_json:
        ov = {}
        for pj in args.ego_amax_json:
            pr = json.load(open(pj))
            if pr.get("dataset") != "navsim" or pr.get("split") != "navtrain":
                raise ValueError(f"{pj}: NAVSIM navtrain 탐침 결과가 아니다")
            for k, v in pr["summary"].items():
                ov[k] = max(ov.get(k, 0.0), v["amax"])
            for k, v in pr.get("probe_quant_nonzero_pass", {}).items():
                ov[k] = max(ov.get(k, 0.0), v["amax"])
        stats["ego_amax_override"] = {k: dict(calib=amax[k], probe=ov.get(k, 0.0), used=max(amax[k], ov.get(k, 0.0))) for k in ego_names}
        for k in ego_names:
            amax[k] = max(amax[k], ov.get(k, 0.0))
    never = sorted(k for k in hooks if calls[k] == 0)
    wrel = {}
    for k, (m, h) in list(hooks.items()):
        if calls[k] == 0:
            continue
        h.amax = amax[k]; h.mode = "quant"
        wrel[k] = F.quantize_weight_(m, args.quant)
    for k in never:   # 보정에서 한 번도 호출되지 않은 모듈은 양자화하지 않음(목록 보고 — 조용히 넘기지 않음)
        m, h = hooks.pop(k); m._forward_pre_hooks.clear()
    stats.update(n_quantized=len(hooks), never_called=never, excluded=excluded, act_amax=amax, calib_calls=calls,
                 weight_rel_err=wrel, amax_zero_modules=sorted(k for k, (m, h) in hooks.items() if h.amax == 0),
                 n_by_part={p: sum(1 for k in hooks if k.startswith(p)) for p in ("img_backbone", "img_neck", "head")})
    print(f"[ptq-navsim] 양자화 모듈 {len(hooks)} {stats['n_by_part']} / 미호출 {len(never)} / 제외 {len(excluded)}", flush=True)
    return model, hooks


def probe(cfg, args, out_json):
    """ego DAF 탐침(navtrain 창 전용). nuScenes probe_ego 와 같은 json 구조 + dataset/split 표시."""
    st = dict(mode="probe_ego", dataset="navsim", split="navtrain", precision=args.precision, quant=args.quant, scope=args.scope,
              started=time.strftime("%Y-%m-%d %H:%M:%S"))
    od = osp.dirname(osp.abspath(out_json))
    win = write_window_ann(cfg, args.probe_ego_windows, args.probe_seed, osp.join(od, f"probe_windows_s{args.probe_seed}.pkl"), "ptq_probe")
    st["windows"] = {k: v for k, v in win.items() if k != "first_window_current_tokens"}
    hooks = {}
    if args.quant != "none":
        cal = write_window_ann(cfg, args.calib_windows, args.calib_seed, osp.join(od, "calib_navtrain_windows.pkl"), "ptq_calib")
        st["calib"] = cal
        model, hooks = build_quant_model(cfg, args, st, cal["ann_file"])
        for k, (m, h) in hooks.items():
            h.probe = True
    else:
        model = F.build_model(cfg, args.checkpoint)
    check_reset_on_scene(model)
    pr = F.EgoProbe(model)
    model = model.cuda()
    ds = build_dataset(navtrain_dataset_cfg(cfg, win["ann_file"]))
    dl = build_dataloader_origin(ds, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)
    dp = MMDataParallel(model, device_ids=[0])
    t0 = time.time()
    with torch.no_grad():
        for k, data in enumerate(dl):
            dp(return_loss=False, rescale=True, **data)
            pr.end_frame(ds.data_infos[k]["token"], k)
    summ = {}
    for i in range(len(pr.layers)):
        rs = [r["layers"][i] for r in pr.rows]
        nz = [r for r in rs if r["nonzero"] > 0]
        summ[F.EGO_OUT.format(i)] = dict(frames=len(rs), nonzero_frames=len(nz), amax=max([r["amax"] for r in rs] + [0.0]),
                                         frames_kps_inside=sum(1 for r in rs if r["n_inside"] > 0),
                                         nonzero_and_inside=sum(1 for r in nz if r["n_inside"] > 0),
                                         nonzero_not_inside=sum(1 for r in nz if r["n_inside"] == 0),
                                         inside_but_zero=sum(1 for r in rs if r["n_inside"] > 0 and r["nonzero"] == 0))
    st.update(probe_frames=len(pr.rows), probe_sec=time.time() - t0, summary=summ,
              nonzero_rows=[r for r in pr.rows if any(l["nonzero"] > 0 for l in r["layers"])][:200],
              example_rows=pr.rows[:4], finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    if hooks:
        st["probe_quant_nonzero_pass"] = {k: dict(n=h.probe_nonzero, amax=h.probe_amax) for k, (m, h) in hooks.items() if h.probe_nonzero}
    json.dump(st, open(out_json, "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps(summ, indent=1), f"\n[probe] → {out_json}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("checkpoint")
    ap.add_argument("--out-dir", required=True, help="새 폴더(있으면 멈춤). navsim_pred.pkl·ptq_stats.json·보정 info 가 생긴다")
    ap.add_argument("--precision", choices=["fp32", "mixed"], required=True)
    ap.add_argument("--quant", choices=["none", "int8", "fp8"], default="none")
    ap.add_argument("--scope", choices=["bb", "bb_dec"], default="bb")
    ap.add_argument("--limit-scenes", type=int, default=0, help=">0 이면 navtest 앞 N 장면(4N 프레임)만 — 경로 검증용")
    ap.add_argument("--calib-windows", type=int, default=64, help="navtrain 창 수(창 = 4 프레임; 기본 64 × 4 = 256 프레임)")
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--probe-ego-windows", type=int, default=0, help=">0: 평가 대신 navtrain 창 N 개 ego DAF 탐침만(--out-dir/probe_<quant>.json)")
    ap.add_argument("--probe-seed", type=int, default=1)
    ap.add_argument("--ego-amax-json", nargs="*", default=None)
    ap.add_argument("--exclude-ego-out", action="store_true")
    ap.add_argument("--save-results", action="store_true", help="모델 원 출력 results.pkl 도 저장(행 간 출력 비교용, 크다)")
    ap.add_argument("--max-gpu-gb", type=float, default=20.0, help="이 프로세스 GPU 메모리 상한(공유 GPU 보호; 넘으면 OOM 으로 멈춤)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    args = ap.parse_args()

    if osp.exists(args.out_dir) and os.listdir(args.out_dir):
        raise FileExistsError(f"{args.out_dir} 가 비어 있지 않다(덮어쓰지 않음)")
    mmcv.mkdir_or_exist(args.out_dir)
    if args.max_gpu_gb > 0:
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gb * 2 ** 30 / tot), 0)

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    F.import_plugin(cfg)
    if cfg.data.test.type not in ("NavsimTeacherDataset", "NavsimDataset"):
        raise ValueError(f"NAVSIM config 가 아니다(data.test.type={cfg.data.test.type})")
    if args.precision == "fp32":
        cfg.fp16 = None
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.data.test.pop("samples_per_gpu", None)
    set_random_seed(args.seed, deterministic=False)   # tools/test.py 와 같음
    if cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs", cfg.get("run_name") or "ptq_navsim")
    cfg.data.test.work_dir = cfg.work_dir

    if args.probe_ego_windows > 0:
        probe(cfg, args, osp.join(args.out_dir, f"probe_{args.quant}_{args.scope}.json"))
        return

    stats = dict(config=args.config, checkpoint=args.checkpoint, precision=args.precision, quant=args.quant, scope=args.scope,
                 limit_scenes=args.limit_scenes, fp16_cfg=str(cfg.get("fp16", None)), neck_dtype=cfg.model.get("neck_dtype"),
                 torch=torch.__version__, gpu=torch.cuda.get_device_name(0), started=time.strftime("%Y-%m-%d %H:%M:%S"))
    hooks = {}
    if args.quant != "none":
        cal = write_window_ann(cfg, args.calib_windows, args.calib_seed, osp.join(args.out_dir, "calib_navtrain_windows.pkl"), "ptq_calib")
        stats["calib"] = cal
        model, hooks = build_quant_model(cfg, args, stats, cal["ann_file"])
    else:
        model = F.build_model(cfg, args.checkpoint)
    check_reset_on_scene(model)

    if args.limit_scenes > 0:
        cfg.data.test.ann_file = write_test_subset(cfg.data.test.ann_file, args.limit_scenes,
                                                   osp.join(args.out_dir, f"navtest_first{args.limit_scenes}.pkl"))
    stats["test_ann_file"] = cfg.data.test.ann_file
    ds = build_dataset(cfg.data.test)
    dl = build_dataloader_origin(ds, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)
    for k, (m, h) in hooks.items():
        h.calls = 0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    outputs = single_gpu_test(MMDataParallel(model, device_ids=[0]), dl, False, None)
    stats.update(infer_sec=time.time() - t0, n_frames=len(outputs), peak_allocated_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
    if hooks:
        stats["test_uncalled_quantized"] = [k for k, (m, h) in hooks.items() if h.calls == 0]
        stats["test_zero_pass"] = {k: h.zero_pass for k, (m, h) in hooks.items() if h.zero_pass}
    print(f"\n[ptq-navsim] 추론 {len(outputs)} 프레임 {stats['infer_sec']:.0f}s, 최대 할당 {stats['peak_allocated_gb']:.1f} GB", flush=True)
    if args.save_results:
        mmcv.dump(outputs, osp.join(args.out_dir, "results.pkl"))
    stats["evaluate"] = ds.evaluate(outputs, jsonfile_prefix=args.out_dir)   # → navsim_pred.pkl (NavsimDataset.evaluate)
    stats["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(stats, open(osp.join(args.out_dir, "ptq_stats.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print(f"[ptq-navsim] → {args.out_dir}/navsim_pred.pkl, ptq_stats.json", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
