# 수정(2026-10-16): 2-fold 장면 분할 — 교사 SFT 교차 적합용 장면 분할 파일 생성(CPU, 수 초).
#   TRAIN 장면 → A/B(표본 수 균형), VAL 장면 → A/B, fold 별 체크포인트 선택 hold-out(그 fold 안 25 장면),
#   교사별 정답 생성 배정(teacher_A → train.B·val.B, teacher_B → train.A·val.A)을 JSON 하나에 기록한다.
#   기존 파일이 있으면 --force 없이는 덮어쓰지 않는다(이미 학습한 교사와 분할이 어긋나는 것을 막음).
import argparse
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_common as C  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/vla_teacher/v2")
    ap.add_argument("--out", default=C.FOLD_FILE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_holdout", type=int, default=C.FOLD_HOLDOUT_SCENES, help="fold 별 체크포인트 선택 장면 수")
    ap.add_argument("--force", action="store_true", help="기존 분할 파일 덮어쓰기(이미 학습한 교사가 있으면 쓰지 말 것)")
    a = ap.parse_args(argv)
    out = Path(a.out) if os.path.isabs(a.out) else C.REPO / a.out
    if out.exists() and not a.force:
        raise SystemExit(f"[거부] {out} 이미 있음 — 덮어쓰려면 --force")
    data_dir = C.resolve_data_dir(a.data_dir)
    tr, va = C.load_records(data_dir, "train"), C.load_records(data_dir, "val")
    sources = {}
    for split, recs in (("train", tr), ("val", va)):
        f = C.split_file(data_dir, split)
        sources[split] = dict(path=os.path.relpath(f, C.REPO), sha256=C.file_sha256(f), n_samples=len(recs),
                              n_scenes=len({r["scene_token"] for r in recs}))
    d = C.make_fold_split(tr, va, seed=a.seed, n_holdout=a.n_holdout, sources=sources)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(out, "w"), indent=1, ensure_ascii=False)
    for part in ("train", "val", "holdout"):
        for f in C.FOLDS:
            e = d[part][f]
            print(f"[fold] {part}.{f}: {e['n_scenes']} 장면 / {e['n_samples']} 표본 (sha256 {e['sha256'][:12]})")
    print(f"[fold] 저장 {out} (sha256 {d['sha256'][:12]}), 균형 {d['balance']}")


if __name__ == "__main__":
    main()
