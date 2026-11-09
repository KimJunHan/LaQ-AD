# 수정(2026-11-05): 원본 deformable_aggregation_ext 와 tools/latency/daf_stream 확장(같은 .cu, 현재 스트림 실행)이 같은 입력에서
#   같은 분포의 값을 내는지 본다. 실제 val 프레임의 DAF 입력을 잡아, 각 확장을 K 번 돌려 (a) 실행 간 흔들림(atomicAdd 합산 순서),
#   (b) 두 확장 평균의 차(체계적 차이 = 코드 생성 차이 의심)를 비교한다.
#   사용: python tools/latency/daf_ext_check.py --frames 3 --k 20 --out results/latency/daf_ext_check.json
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import student_common as common  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import student_fast
    cfg, dataset, loader, model, fp16 = common.build()
    det = model.module
    agg = student_fast._module_of(det.head.onedecoder_head.det_deformable[0])
    orig = agg.DAF
    calls = []

    def grab(*args):
        calls.append([x.detach().clone() for x in args])
        return orig(*args)
    agg.DAF = grab
    it = iter(loader)
    with torch.no_grad():
        for _ in range(a.frames):
            model(return_loss=False, rescale=True, **next(it))
    agg.DAF = orig
    E0 = sys.modules[orig.__module__ + ".deformable_aggregation"].deformable_aggregation_ext  # 모델이 실제로 쓰는 원본 확장
    E1 = student_fast._daf_ext()
    rows = []
    with torch.no_grad():
        for ci, args in enumerate(calls):
            f, ss, ssi, loc, w = args
            ins = (f.contiguous().float(), ss.contiguous().int(), ssi.contiguous().int(), loc.contiguous().float(), w.contiguous().float())
            o0 = torch.stack([E0.deformable_aggregation_forward(*ins) for _ in range(a.k)]).double()
            o1 = torch.stack([E1.forward(*ins) for _ in range(a.k)]).double()
            m0, m1 = o0.mean(0), o1.mean(0)
            r = {"call": ci, "shape": list(o0.shape[1:]),
                 "orig_run_spread_max": float((o0 - m0).abs().max()), "mine_run_spread_max": float((o1 - m1).abs().max()),
                 "orig_vs_orig_first_two_max": float((o0[0] - o0[1]).abs().max()),
                 "mean_diff_max": float((m0 - m1).abs().max()), "mean_diff_signed_mean": float((m1 - m0).mean()),
                 "first_run_exact": bool(torch.equal(o0[0], o1[0])),
                 "abs_mean": float(m0.abs().mean())}
            rows.append(r)
    out = {"n_calls": len(calls), "k": a.k, "rows": rows}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for r in rows[:12]:
        print(r)


if __name__ == "__main__":
    main()
