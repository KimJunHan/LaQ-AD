# 수정(2026-10-16): 교사 QA 전체 학습 단계 — QA 표본 장면 유출 검사(CPU, 수 초).
"""data/vla_teacher/qa/fold{A,B}_{train,holdout}.jsonl 이
  (1) nuScenes 공식 val 150 장면(nuscenes-devkit splits.val → v1.0-trainval scene.json)과 겹침 0,
  (2) 다른 fold 의 train 장면(holdout 포함)과 겹침 0,
  (3) teacher_<F> 의 학습 = train.<F> − holdout.<F>, hold-out = holdout.<F> 와 정확히 일치(train_sft.py --fold 와 같은 집합),
  (4) 표본 토큰의 장면이 공식 sample.json 과 일치, 같은 QA(원본 위치 src)가 두 파일에 중복되지 않음
임을 세어서 보인다. 하나라도 어긋나면 종료 코드 1.
  python -I tools/vla_teacher/check_qa_split.py
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_common as Q  # noqa: E402
import sft_common as S  # noqa: E402
from make_qa_data import NUSC_META, official_scene_maps  # noqa: E402


def main():
    s2scene, off_train, off_val = official_scene_maps()
    fd, _ = S.load_fold_split(S.FOLD_FILE)
    sets = {f: dict(train=set(fd["train"][f]["scenes"]) - set(fd["holdout"][f]["scenes"]),
                    holdout=set(fd["holdout"][f]["scenes"]), all=set(fd["train"][f]["scenes"])) for f in S.FOLDS}
    fold_val = set(fd["val"]["A"]["scenes"]) | set(fd["val"]["B"]["scenes"])
    ok = True

    def check(name, cond, info):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'} {name}: {info}", flush=True)

    check("official_val_eq_fold_val", fold_val == off_val, f"공식 val {len(off_val)} 장면, fold_split val {len(fold_val)} 장면")
    srcs = Counter()
    for f in S.FOLDS:
        other = "B" if f == "A" else "A"
        for part in ("train", "holdout"):
            recs, p = Q.load_qa(f, part)
            scenes = {r["scene_token"] for r in recs}
            n_val = sum(r["scene_token"] in off_val for r in recs)
            n_other = sum(r["scene_token"] in sets[other]["all"] for r in recs)
            n_out = sum(r["scene_token"] not in sets[f][part] for r in recs)
            n_meta = sum(s2scene.get(r["token"]) != r["scene_token"] for r in recs)
            n_qa = sum(len(r["turns"]) for r in recs)
            by = Counter(r["dataset"] for r in recs)
            qa_by = Counter()
            for r in recs:
                qa_by[r["dataset"]] += len(r["turns"])
                for t in r["turns"]:
                    srcs[(r["dataset"], t["src"])] += 1
            tag = f"fold{f}_{part}"
            print(f"[{tag}] 표본 {len(recs)} {dict(by)} / QA {n_qa} {dict(qa_by)} / 장면 {len(scenes)} / 프레임 "
                  f"{len({(r['dataset'], r['token']) for r in recs})}", flush=True)
            check(f"{tag}_val_overlap", n_val == 0 and not (scenes & off_val), f"val 장면 표본 {n_val}, 장면 {len(scenes & off_val)}")
            check(f"{tag}_other_fold_overlap", n_other == 0 and not (scenes & sets[other]["all"]),
                  f"fold {other} 장면 표본 {n_other}, 장면 {len(scenes & sets[other]['all'])}")
            check(f"{tag}_in_assigned_scenes", n_out == 0, f"배정 밖 표본 {n_out} (배정 = "
                  f"{'train.' + f + ' − holdout.' + f if part == 'train' else 'holdout.' + f}, {len(sets[f][part])} 장면 중 {len(scenes)} 사용)")
            check(f"{tag}_scene_matches_official", n_meta == 0, f"공식 sample.json 과 다른 장면 {n_meta}")
    dup = sum(1 for v in srcs.values() if v > 1)
    check("no_duplicate_qa", dup == 0, f"두 번 이상 들어간 QA {dup} / 전체 {len(srcs)}")
    trA, hoA = {r["scene_token"] for r in Q.load_qa("A", "train")[0]}, {r["scene_token"] for r in Q.load_qa("A", "holdout")[0]}
    check("foldA_train_holdout_disjoint", not (trA & hoA), f"겹침 {len(trA & hoA)}")
    trB, hoB = {r["scene_token"] for r in Q.load_qa("B", "train")[0]}, {r["scene_token"] for r in Q.load_qa("B", "holdout")[0]}
    check("foldB_train_holdout_disjoint", not (trB & hoB), f"겹침 {len(trB & hoB)}")
    print("ALL PASS" if ok else "SOME FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
