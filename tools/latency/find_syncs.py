# 수정(2026-11-04): 학생 추론(배치 1) 한 프레임 안의 호스트↔GPU 동기화 지점을 file:line 으로 모은다.
#   torch.cuda.set_sync_debug_mode("warn") 이 동기화 연산(.item()/nonzero/bool 마스크 색인/pageable H2D·D2H 복사 등)마다
#   경고를 내면, 그때의 파이썬 호출 스택에서 저장소 안의 가장 안쪽 프레임(과 바로 위 호출자)을 기록한다.
#   사용: python tools/latency/find_syncs.py [--fp32] [--frames 5] --out results/latency/syncs_fp16.json
import argparse
import collections
import json
import os
import sys
import traceback
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import student_common as common  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=common.DEFAULT_CFG)
    ap.add_argument("--checkpoint", default=common.DEFAULT_CKPT)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg, dataset, loader, model, fp16 = common.build(a.config, a.checkpoint, fp32=a.fp32)
    repo = common._REPO
    counts = collections.Counter()
    first_stack = {}
    inner_only = collections.Counter()

    def show(message, category, filename, lineno, file=None, line=None):
        msg = str(message)
        if "synchroniz" not in msg:
            return
        st = traceback.extract_stack()[:-1]
        repo_frames = [f for f in st if f.filename.startswith(repo) and "/tools/latency/" not in f.filename]
        if repo_frames:
            inner = repo_frames[-1]
            caller = repo_frames[-2] if len(repo_frames) > 1 else None
            key = f"{os.path.relpath(inner.filename, repo)}:{inner.lineno}"
            ckey = f"{os.path.relpath(caller.filename, repo)}:{caller.lineno}" if caller else ""
        else:
            # 저장소 밖(mmcv scatter 등)
            outer = [f for f in st if "/tools/latency/" not in f.filename and "warnings" not in f.filename]
            inner = outer[-1]
            key = f"{inner.filename}:{inner.lineno}"
            ckey = ""
        k = (key, ckey, inner.line)
        counts[k] += 1
        inner_only[key] += 1
        if k not in first_stack:
            first_stack[k] = [f"{f.filename}:{f.lineno} {f.name}: {f.line}" for f in st[-12:]]

    it = iter(loader)
    with torch.no_grad():
        for _ in range(a.warmup):
            model(return_loss=False, rescale=True, **next(it))
        torch.cuda.synchronize()
        warnings.showwarning = show
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        for _ in range(a.frames):
            model(return_loss=False, rescale=True, **next(it))
        torch.cuda.set_sync_debug_mode("default")
    rows = [{"site": k[0], "caller": k[1], "code": k[2], "per_frame": v / a.frames, "stack": first_stack[k]}
            for k, v in counts.most_common()]
    out = {"fp16_wrapped": fp16, "frames": a.frames, "total_per_frame": sum(counts.values()) / a.frames,
           "by_site": {k: v / a.frames for k, v in inner_only.most_common()}, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"동기화 {out['total_per_frame']:.1f} 회/프레임")
    for r in rows:
        print(f"{r['per_frame']:6.1f}  {r['site']:70s} <- {r['caller']:55s} | {r['code']}")


if __name__ == "__main__":
    main()
