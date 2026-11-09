# 수정(2026-11-04): val 전체 추론을 원본 eager 또는 FastStudent 변형으로 돌려 results.pkl 을 만든다(채점은 기존 경로 그대로).
#   tools/test.py 의 단일 GPU 경로(single_gpu_test: 표본마다 model(return_loss=False, rescale=True, **data) 를 이어 붙임)와 같은
#   데이터로더(workers = cfg.data.workers_per_gpu)·fp16 wrap 으로 추론만 바꾼다. 채점:
#     python tools/test.py <cfg> <ckpt> --result_file <out>/results.pkl --eval bbox --cfg-options evaluation.jsonfile_prefix=<out>
#     python tools/eval_nusc_planning.py <cfg> <out>/results.pkl --out <out>/planning_nusc.json
#   출력은 results/latency/ 아래에만 쓴다(evaluation/ 금지).
#   사용: python tools/latency/student_test.py --variant graph_post --out results/latency/fullval/graph_post
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import student_common as common  # noqa: E402
import mmcv  # noqa: E402
import torch  # noqa: E402


def main():
    from student_equiv import VARIANTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["eager"] + sorted(k for k in VARIANTS if k != "eager2"))
    ap.add_argument("--config", default=common.DEFAULT_CFG)
    ap.add_argument("--checkpoint", default=common.DEFAULT_CKPT)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--deterministic-daf", action="store_true",
                    help="원본·래퍼 모두 결정적 DAF(student_fast._DetDAF) — 합산 순서 잡음 없이 지표를 비교하는 진단용(지연 측정용 아님)")
    ap.add_argument("--daf-impl", choices=["default", "orig", "mine"], default="default",
                    help="진단용: DAF 확장 강제 선택(orig = 원본 확장, mine = tools/latency/daf_stream 확장)")
    a = ap.parse_args()
    assert os.path.abspath(a.out).startswith(os.path.abspath("results/latency")), "출력은 results/latency/ 아래만"
    os.makedirs(a.out, exist_ok=True)
    from mmcv import Config
    cfg = Config.fromfile(a.config)
    workers = cfg.data.workers_per_gpu
    cfg, dataset, loader, model, fp16 = common.build(a.config, a.checkpoint, fp32=a.fp32, workers=workers)
    if a.daf_impl != "default":
        import student_fast
        student_fast._DAFDispatch.force = a.daf_impl
        student_fast.install_global_patches(model.module)
    if a.deterministic_daf:
        import student_fast
        student_fast._DetDAF.on = True
        student_fast.install_global_patches(model.module)
    if a.variant == "eager":
        def run(data):
            return model(return_loss=False, rescale=True, **data)
        fs = None
    else:
        from student_fast import FastStudent
        fs = FastStudent(model, **VARIANTS[a.variant])
        run = fs
    results = []
    t0 = time.time()
    prog = mmcv.ProgressBar(len(dataset))
    with torch.no_grad():
        for data in loader:
            r = run(data)
            results.extend(r)
            prog.update()
    t_inf = time.time() - t0
    print(f"\n추론 {len(results)} 표본 {t_inf:.0f}s", flush=True)
    mmcv.dump(results, os.path.join(a.out, "results.pkl"))
    with open(os.path.join(a.out, "infer_meta.json"), "w") as f:
        json.dump({"variant": a.variant, "fp16_wrapped": fp16, "config": a.config, "checkpoint": a.checkpoint,
                   "n": len(results), "infer_s": t_inf, "deterministic_daf": bool(a.deterministic_daf), "daf_impl": a.daf_impl, "workers": workers,
                   "build_timings_s": fs.timings if fs is not None else None,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
