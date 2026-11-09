# 수정(2026-10-16): B2D 교사 QA 준비 — Bench2Drive 2-fold clip 분할 파일 생성(CPU). make_fold_split.py 와 같은 규칙.
"""data/vla_teacher/b2d/fold_split_b2d.json 생성.

  OPENBLAS_CORETYPE=SkylakeX CUDA_VISIBLE_DEVICES="" python tools/vla_teacher/make_fold_split_b2d.py

규칙(make_fold_split.py / sft_common.make_fold_split 과 같은 함수를 그대로 사용):
  * 장면 단위 = B2D clip(= route. Base-1000 은 clip 마다 Route 번호가 다르고 겹치는 route 없음).
  * TRAIN = 공식 Bench2DriveZoo split 의 val 50 clip 을 뺀 950 clip, VAL = 그 50 clip.
  * 표본 수(균형 기준) = clip 의 rgb_front jpg 프레임 수(QA 원본과 무관한 데이터 자체의 크기 — QA 출처가 늘어도 분할이 바뀌지 않음).
  * TRAIN → A/B, VAL → A/B (표본 수 균형, seed 0, 결정적), fold 별 hold-out = 그 반쪽 안의 clip.
  * hold-out clip 수 = nuScenes 와 같은 비율(25/350 장면) → round(25/350 × 475) = 34 clip.
기존 파일이 있으면 --force 없이는 덮어쓰지 않는다.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2d_qa_common as B  # noqa: E402
import sft_common as S  # noqa: E402

N_HOLDOUT_B2D = round(S.FOLD_HOLDOUT_SCENES / 350 * 475)   # = 34 (nuScenes 25/350 비율 유지)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=B.FOLD_FILE_B2D)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_holdout", type=int, default=N_HOLDOUT_B2D)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out) if os.path.isabs(a.out) else S.REPO / a.out
    if out.exists() and not a.force:
        raise SystemExit(f"[거부] {out} 이미 있음 — 덮어쓰려면 --force")
    val = B.official_val_clips()
    clips = B.all_clips()
    if not val <= set(clips):
        raise RuntimeError(f"공식 val clip 중 B2D 폴더에 없는 것 {sorted(val - set(clips))}")
    listing = B.image_listing(S.REPO)
    counts = B.frames_per_clip(listing)
    missing = [c for c in clips if c not in counts or counts[c] == 0]
    if missing:
        raise RuntimeError(f"rgb_front 프레임이 없는 clip {missing}")
    # sft_common.make_fold_split 은 표본 레코드(token, scene_token)를 받는다 — 프레임 하나 = 표본 하나
    def recs(cs):
        return [dict(token=f"{c}/{i}", scene_token=c) for c in cs for i in range(counts[c])]
    tr_clips = [c for c in clips if c not in val]
    va_clips = [c for c in clips if c in val]
    sources = dict(
        official_split=dict(path=B.OFFICIAL_SPLIT, sha256=S.file_sha256(B.OFFICIAL_SPLIT), n_val_clips=len(val)),
        train=dict(n_clips=len(tr_clips), n_samples=sum(counts[c] for c in tr_clips), unit="rgb_front jpg frames"),
        val=dict(n_clips=len(va_clips), n_samples=sum(counts[c] for c in va_clips), unit="rgb_front jpg frames"),
        b2d_root=B.B2D_ROOT)
    d = S.make_fold_split(recs(tr_clips), recs(va_clips), seed=a.seed, n_holdout=a.n_holdout, sources=sources)
    # B2D 용 표시(내용이 바뀌므로 해시 재계산 — load_fold_split 이 검증)
    d["benchmark"] = "b2d"
    d["scene_unit"] = "Bench2Drive Base clip (= route; 1 route per clip)"
    d["n_holdout_rule"] = "round(25/350 × 475) — nuScenes fold hold-out 비율과 같음"
    d["assignment"]["teacher_A"]["out_default"] = "results/vla_teacher/b2d/qa_full_foldA"
    d["assignment"]["teacher_B"]["out_default"] = "results/vla_teacher/b2d/qa_full_foldB"
    d["sha256"] = S.fold_doc_hash(d)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(out, "w"), indent=1, ensure_ascii=False)
    S.load_fold_split(str(out))          # 해시 검증 왕복
    for part in ("train", "val", "holdout"):
        for f in S.FOLDS:
            e = d[part][f]
            print(f"[fold-b2d] {part}.{f}: {e['n_scenes']} clip / {e['n_samples']} 프레임 (sha256 {e['sha256'][:12]})")
    print(f"[fold-b2d] 저장 {out} (sha256 {d['sha256'][:12]}), 균형 {d['balance']}")


if __name__ == "__main__":
    main()
