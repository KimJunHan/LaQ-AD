# 수정(2026-10-16): 교사 QA — NAVSIM 트랙. make_fold_split.py 와 같은 2-fold 분할을 NAVSIM 로그 단위로 만든다(CPU).
"""NAVSIM 2-fold 로그 분할 → data/vla_teacher/navsim/fold_split_navsim.json

nuScenes(make_fold_split.py)와 같은 함수(sft_common.make_fold_split)를 그대로 쓴다. 대응 관계:
  nuScenes 장면(scene_token)  ↔ NAVSIM 로그(log_name)        — 같은 로그의 프레임은 한쪽 fold 에만 들어간다.
  nuScenes 표본(키프레임)      ↔ NAVSIM 장면 token(학생이 쓰는 navtrain/navtest 장면, devkit filter_scenes 재구성)
  train(700 장면)             ↔ navtrain 로그(navtrain.yaml log_names)
  val(150 장면)               ↔ navtest 로그(navtest.yaml log_names) — 교사 학습 금지, 캐시만 반반 생성
  hold-out 25/350 장면        ↔ fold 안 로그 수 × 25/350 (반올림) — 같은 비율
추가 기록: teacher_exclude_logs(NuPlanQA-Eval 프레임이 있는 navtrain 로그 6개) — 분할에는 남기되(학생 캐시 생성 대상),
  교사 학습 표본에서는 make_qa_data_navsim.py 가 전부 뺀다.
실행(내려받은 데이터 → python -I):
  CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX python -I tools/vla_teacher/make_fold_split_navsim.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import navsim_qa_common as N  # noqa: E402
import sft_common as C  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=N.FOLD_FILE_NAVSIM)
    ap.add_argument("--index_cache", required=True, help="NAVSIM 프레임 색인 캐시(pkl, 없으면 생성)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_holdout", type=int, default=0, help="fold 별 hold-out 로그 수(0 = nuScenes 비율 25/350 로 계산)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out) if os.path.isabs(a.out) else C.REPO / a.out
    if out.exists() and not a.force:
        raise SystemExit(f"[거부] {out} 이미 있음 — 덮어쓰려면 --force")
    idx = N.build_frame_index(a.index_cache, a.workers)
    ftr, fte = N.load_filter(N.NAVTRAIN_YAML), N.load_filter(N.NAVTEST_YAML)
    if set(ftr["log_names"]) & set(fte["log_names"]):
        raise RuntimeError("navtrain·navtest 로그 겹침")
    tr_by_log = N.scene_tokens_by_log(idx, ftr, "trainval")
    te_by_log = N.scene_tokens_by_log(idx, fte, "test")
    # 장면 token 이 없는 로그(필터 log_names 에는 있으나 장면 0)는 표본 0 → sft_common 의 표본 기반 분할에서 빠진다. 세어서 기록.
    empty_tr = sorted(l for l, v in tr_by_log.items() if not v)
    empty_te = sorted(l for l, v in te_by_log.items() if not v)
    tr = [dict(token=t, scene_token=l) for l, v in tr_by_log.items() for t in v]
    te = [dict(token=t, scene_token=l) for l, v in te_by_log.items() for t in v]
    n_logs_tr = len({r["scene_token"] for r in tr})
    n_hold = a.n_holdout or int(round(n_logs_tr / 2 * C.FOLD_HOLDOUT_SCENES / 350))
    sources = dict(
        navtrain=dict(path=N.NAVTRAIN_YAML, sha256=C.file_sha256(N.NAVTRAIN_YAML), n_samples=len(tr), n_scenes=n_logs_tr,
                      unit="sample = navtrain scene token, scene = log"),
        navtest=dict(path=N.NAVTEST_YAML, sha256=C.file_sha256(N.NAVTEST_YAML), n_samples=len(te),
                     n_scenes=len({r["scene_token"] for r in te}), unit="sample = navtest scene token, scene = log"),
        logs=dict(trainval=N.TRAIN_LOGS, test=N.TEST_LOGS),
        filter_logs_without_scenes=dict(navtrain=empty_tr, navtest=empty_te))
    d = C.make_fold_split(tr, te, seed=a.seed, n_holdout=n_hold, sources=sources)
    excl = json.load(open(N.EXCLUDE_LOGS))
    d.pop("sha256")
    d["benchmark"] = "navsim"
    d["unit"] = "scene = NAVSIM log_name (OpenScene log), sample = NAVSIM scene token"
    d["n_holdout_rule"] = f"round(fold 로그 수 {n_logs_tr / 2:.1f} × 25/350) = {n_hold} (nuScenes 와 같은 비율)"
    d["teacher_exclude_logs"] = dict(path=N.EXCLUDE_LOGS, sha256=C.file_sha256(N.EXCLUDE_LOGS), logs=sorted(excl["logs"]),
                                     rule=excl["rule"],
                                     in_train=dict({f: sorted(set(excl["logs"]) & set(d["train"][f]["scenes"])) for f in C.FOLDS}),
                                     in_holdout=dict({f: sorted(set(excl["logs"]) & set(d["holdout"][f]["scenes"])) for f in C.FOLDS}))
    d["sha256"] = C.fold_doc_hash(d)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x" if not a.force else "w") as fp:
        json.dump(d, fp, indent=1, ensure_ascii=False)
    C.load_fold_split(str(out))   # 해시 자기검사
    for part in ("train", "val", "holdout"):
        for f in C.FOLDS:
            e = d[part][f]
            print(f"[fold] {part}.{f}: {e['n_scenes']} 로그 / {e['n_samples']} 장면 token (sha256 {e['sha256'][:12]})")
    print(f"[fold] 장면 없는 필터 로그: navtrain {len(empty_tr)}, navtest {len(empty_te)}")
    print(f"[fold] 제외 로그 위치: {d['teacher_exclude_logs']['in_train']}, hold-out 안: {d['teacher_exclude_logs']['in_holdout']}")
    print(f"[fold] 저장 {out} (sha256 {d['sha256'][:12]}), 균형 {d['balance']}")


if __name__ == "__main__":
    main()
