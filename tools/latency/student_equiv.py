# 수정(2026-11-04): 빠른 경로(FastStudent) 출력이 원본 eager 와 같은지 — val 앞에서부터 연속 N 프레임(장면 순서, 시간 상태 포함).
#   같은 프로세스에 원본 모델과 빠른 경로 모델을 따로 만들어(각자 시간 뱅크) 같은 표본을 번갈아 넣고 출력 키마다 최대 절대 차를 잰다.
#   사용: python tools/latency/student_equiv.py --variant graph_post [--fp32] [--n 240] --out results/latency/equiv_x.json
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import student_common as common  # noqa: E402
import torch  # noqa: E402

VARIANTS = {
    # 이름: FastStudent 인자. "eager2" = 원본 모델 하나 더(같은 코드) — atomicAdd 합산 순서로 생기는 eager 자체의 실행 간 잡음 바닥.
    "eager2": None,
    "eager_patched": dict(use_graph=False, post_graph=False),
    "eager_patched_post": dict(use_graph=False, post_graph=True),
    "graph": dict(use_graph=True, post_graph=False),
    "graph_post": dict(use_graph=True, post_graph=True),
    # 진단용(계획 L2 차이 원인 가르기): 예열 스트림만 바꾼 변형
    "graph_post_cw": dict(use_graph=True, post_graph=True, warmup_stream="current"),
    "eager_patched_post_sw": dict(use_graph=False, post_graph=True, warmup_stream="side"),
    "graph_post_compile_bb": dict(use_graph=True, post_graph=True, compile_mode="max-autotune-no-cudagraphs",
                                  compile_targets=("backbone", "neck")),
    "graph_post_compile_default_bb": dict(use_graph=True, post_graph=True, compile_mode="default",
                                          compile_targets=("backbone", "neck")),
    "eager_patched_post_compile_ro_bb": dict(use_graph=False, post_graph=True, compile_mode="reduce-overhead",
                                             compile_targets=("backbone", "neck")),
    "graph_post_compile_head": dict(use_graph=True, post_graph=True, compile_mode="max-autotune-no-cudagraphs",
                                    compile_targets=("head",)),
    "graph_post_compile_default_head": dict(use_graph=True, post_graph=True, compile_mode="default",
                                            compile_targets=("head",)),
    "graph_post_compile_dec": dict(use_graph=True, post_graph=True, compile_mode="max-autotune-no-cudagraphs",
                                   compile_targets=("backbone", "neck", "decoder_layers")),
    "graph_post_compile_default_dec": dict(use_graph=True, post_graph=True, compile_mode="default",
                                           compile_targets=("backbone", "neck", "decoder_layers")),
}


def as_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _unused_compare(ref, out, acc):
    r, o = ref[0]["img_bbox"], out[0]["img_bbox"]
    if set(r) != set(o):
        raise AssertionError(f"키 불일치: {sorted(set(r) ^ set(o))}")
    for k in r:
        a, b = r[k], o[k]
        if k == "vectors":
            a, b = np.stack(a), np.stack(b)
        a, b = as_np(a), as_np(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f"{k}: 모양/dtype 불일치 {a.shape}{a.dtype} vs {b.shape}{b.dtype}")
        e = acc.setdefault(k, {"max_abs": 0.0, "frames_exact": 0, "frames": 0, "nonfinite_mismatch": 0, "max_rel": 0.0})
        e["frames"] += 1
        if np.issubdtype(a.dtype, np.floating):
            fa, fb = np.isfinite(a), np.isfinite(b)
            if not np.array_equal(fa, fb):
                e["nonfinite_mismatch"] += 1
            m = fa & fb
            d = np.abs(a[m].astype(np.float64) - b[m].astype(np.float64))
            md = float(d.max()) if d.size else 0.0
            e["max_abs"] = max(e["max_abs"], md)
            den = np.abs(a[m].astype(np.float64)).max() if d.size else 0.0
            if den > 0:
                e["max_rel"] = max(e["max_rel"], md / den)
            e["frames_exact"] += int(np.array_equal(a, b, equal_nan=True))
        else:
            e["max_abs"] = max(e["max_abs"], float(np.abs(a.astype(np.int64) - b.astype(np.int64)).max()) if a.size else 0.0)
            e["frames_exact"] += int(np.array_equal(a, b))


RAW_KEYS = [("det", "classification"), ("det", "prediction"), ("det", "quality"), ("det", "instance_id"),
            ("map", "classification"), ("map", "prediction"), ("motion", "classification"), ("motion", "prediction"),
            ("plan", "classification"), ("plan", "prediction"), ("ego", "status")]
_IDX = {"det": 0, "map": 1, "ego": 2, "plan": 3, "motion": 4}


def raw_view(model_outs):
    """디코더 마지막 층 출력(앵커 순서 — 정렬 전) 사본."""
    out = {}
    for t, k in RAW_KEYS:
        v = model_outs[_IDX[t]][k]
        v = v[-1] if isinstance(v, list) else v
        out[f"raw_{t}_{k}"] = v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu()
    return out


def bank_view(det):
    od = det.head.onedecoder_head
    out = {}
    for n, attrs in (("det_instance_bank", ("cached_feature", "cached_anchor", "confidence", "instance_id", "prev_id")),
                     ("map_instance_bank", ("plan_map_pts", "plan_map_scores")),
                     ("ego_instance_bank", ("cached_feature",)), ("plan_instance_bank", ("cached_feature", "cached_anchor", "confidence"))):
        for a in attrs:
            v = getattr(getattr(od, n), a)
            v = v if torch.is_tensor(v) else torch.tensor(v)
            out[f"state_{n.split('_')[0]}_{a}"] = v.detach().cpu()
    return out


def compare_dict(r, o, acc):
    for k in r:
        a, b = as_np(r[k]), as_np(o[k])
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(f"{k}: 모양/dtype 불일치 {a.shape}{a.dtype} vs {b.shape}{b.dtype}")
        e = acc.setdefault(k, {"max_abs": 0.0, "frames_exact": 0, "frames": 0, "nonfinite_mismatch": 0, "per_frame_max": []})
        e["frames"] += 1
        if np.issubdtype(a.dtype, np.floating):
            fa, fb = np.isfinite(a), np.isfinite(b)
            if not np.array_equal(fa, fb):
                e["nonfinite_mismatch"] += 1
            m = fa & fb
            d = np.abs(a[m].astype(np.float64) - b[m].astype(np.float64))
            md = float(d.max()) if d.size else 0.0
            e["frames_exact"] += int(np.array_equal(a, b, equal_nan=True))
        else:
            md = float(np.abs(a.astype(np.int64) - b.astype(np.int64)).max()) if a.size else 0.0
            e["frames_exact"] += int(np.array_equal(a, b))
        e["max_abs"] = max(e["max_abs"], md)
        e["per_frame_max"].append(md)


def post_view(res):
    r = dict(res[0]["img_bbox"])
    r["vectors"] = np.stack(r["vectors"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--config", default=common.DEFAULT_CFG)
    ap.add_argument("--checkpoint", default=common.DEFAULT_CKPT)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--free-run", action="store_true",
                    help="상태 동기화 없이 각자 시간 뱅크로 계속 돌린다(누적 차). 기본은 매 프레임 원본 상태를 후보에 복사(프레임별 차)")
    ap.add_argument("--deterministic-daf", action="store_true",
                    help="원본·후보 모두 결정적 DAF(student_fast._DetDAF)로 — 래퍼 자체의 비트 동치 확인용")
    ap.add_argument("--det-daf-after-build", action="store_true",
                    help="진단: 0 프레임·빌드(예열·캡처)는 실제 DAF, 빌드 직후부터 원본·후보 모두 결정적 DAF — 예열이 남기는 지속 효과 확인")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import student_fast  # noqa: F401  (custom op 등록)
    from student_fast import FastStudent, _DetDAF, install_global_patches
    _DetDAF.on = bool(a.deterministic_daf)
    cfg, dataset, loader, ref, fp16 = common.build(a.config, a.checkpoint, fp32=a.fp32)
    _, _, _, cand, _ = common.build(a.config, a.checkpoint, fp32=a.fp32)
    rdet, cdet = ref.module, cand.module
    install_global_patches(rdet)  # 모듈 수준 래퍼(플래그 꺼짐 = 원본 그대로; --deterministic-daf 면 결정적 DAF)
    stash = {}

    def hook(name):
        def f(mod, inp, out):
            stash[name] = out
        return f
    rdet.head.register_forward_hook(hook("ref"))
    if a.variant == "eager2":
        cdet.head.register_forward_hook(hook("cand"))
        fs = None

        def run_cand(data):
            return cand(return_loss=False, rescale=True, **data)

        def load_state(src):
            sod, cod = src.head.onedecoder_head, cdet.head.onedecoder_head
            for n in ("det_instance_bank", "map_instance_bank", "ego_instance_bank", "plan_instance_bank"):
                sb, cb = getattr(sod, n), getattr(cod, n)
                for k, v in vars(sb).items():
                    if torch.is_tensor(v):
                        setattr(cb, k, v.detach().clone())
                if isinstance(getattr(sb, "metas", None), dict) and "timestamp" in sb.metas:
                    cb.metas = dict(cb.metas, timestamp=sb.metas["timestamp"].detach().clone())
    else:
        fs = FastStudent(cand, **VARIANTS[a.variant])
        if a.det_daf_after_build:
            _orig_build = fs._build

            def _build_then_det(data):
                _orig_build(data)
                _DetDAF.on = True
            fs._build = _build_then_det
        run_cand = fs
        load_state = fs.load_state_from
    acc_post, acc_raw, acc_state = {}, {}, {}
    mismatch = []
    it = iter(loader)
    scenes = set()
    t0 = time.time()
    with torch.no_grad():
        for i in range(a.n):
            data = next(it)
            scenes.add(data["img_metas"].data[0][0]["scene_token"])
            if i > 0 and not a.free_run:
                load_state(rdet)
            r = ref(return_loss=False, rescale=True, **data)
            r_raw = raw_view(stash["ref"])
            o = run_cand(data)
            c_outs = stash.get("cand") if fs is None else (fs.last_model_outs or stash.get("ref_dummy"))
            if fs is not None and fs.last_model_outs is None:  # 첫 프레임(eager) — 후보 head 출력은 따로 잡지 않았다
                c_outs = None
            before = {id(acc): {k: v["frames_exact"] for k, v in acc.items()} for acc in (acc_raw, acc_post, acc_state)}
            if c_outs is not None:
                compare_dict(r_raw, raw_view(c_outs), acc_raw)
            compare_dict(post_view(r), post_view(o), acc_post)
            compare_dict(bank_view(rdet), bank_view(cdet), acc_state)
            # 수정: 어긋난 프레임 기록(전체 val 비트 동치 확인 때 위치를 찾으려고) — 이 프레임에서 exact 가 늘지 않은 키
            bad = [k for acc in (acc_raw, acc_post, acc_state) for k, v in acc.items()
                   if v["frames_exact"] == before[id(acc)].get(k, 0)]
            if bad:
                meta = data["img_metas"].data[0][0]
                mismatch.append({"frame": i, "scene": meta.get("scene_token"), "sample": meta.get("token"), "keys": bad})
                if len(mismatch) <= 5:
                    print(f"[불일치] frame {i} {bad[:6]}", flush=True)
            if (i + 1) % 40 == 0:
                worst = max(((k, v["max_abs"]) for k, v in acc_raw.items() if "instance_id" not in k), key=lambda kv: kv[1],
                            default=("-", 0))
                print(f"[{i + 1}/{a.n}] raw 최대 차 {worst[0]} {worst[1]:.3g} ({time.time() - t0:.0f}s)", flush=True)

    def summ(acc):
        out = {}
        for k, v in acc.items():
            pf = np.asarray(v["per_frame_max"])
            out[k] = {"max_abs": v["max_abs"], "median_frame_max": float(np.median(pf)),
                      "p99_frame_max": float(np.percentile(pf, 99)), "frames_exact": v["frames_exact"], "frames": v["frames"],
                      "nonfinite_mismatch": v["nonfinite_mismatch"]}
        return out
    res = {"variant": a.variant, "kwargs": None if VARIANTS[a.variant] is None else
           {k: (list(v) if isinstance(v, tuple) else v) for k, v in VARIANTS[a.variant].items()},
           "fp16_wrapped": fp16, "n_frames": a.n, "n_scenes": len(scenes), "state_sync": not a.free_run, "deterministic_daf": bool(a.deterministic_daf),
           "build_timings": fs.timings if fs is not None else None,
           "mismatch_frames": len(mismatch), "mismatch_first": mismatch[:200],
           "raw_last_layer": summ(acc_raw), "post": summ(acc_post), "bank_state_after_frame": summ(acc_state)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ("variant", "fp16_wrapped", "n_frames", "n_scenes", "state_sync")}, ensure_ascii=False))
    for name in ("raw_last_layer", "post", "bank_state_after_frame"):
        print(f"== {name}")
        for k, v in res[name].items():
            print(f"  {k:34s} max {v['max_abs']:.3e}  median {v['median_frame_max']:.3e}  p99 {v['p99_frame_max']:.3e}  "
                  f"exact {v['frames_exact']}/{v['frames']}")


if __name__ == "__main__":
    main()
