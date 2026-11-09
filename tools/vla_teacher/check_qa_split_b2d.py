# 수정(2026-10-16): B2D 교사 QA 준비 — B2D QA 표본의 clip 유출·영상·답·길이 검사(CPU). check_qa_split.py 의 B2D 판.
"""data/vla_teacher/b2d/qa/{foldA,foldB}_{train,holdout}.jsonl + val.jsonl 이
  (1) 공식 Bench2DriveZoo val 50 clip 과 학습 파일 겹침 0, val.jsonl 은 val clip 만,
  (2) 다른 fold 의 train clip(holdout 포함)과 겹침 0, (3) 배정 = train.<F> − holdout.<F> / holdout.<F> 와 정확히 일치,
  (4) 같은 QA(원본 위치 src)·같은 id 가 두 번 나오지 않음, (5) 모든 답·질문이 비어 있지 않음, n_tokens ≤ max_seq_tokens,
  (6) 참조한 영상 전부 존재(clip 별 6 카메라 폴더 listdir 결과와 대조 — 새로 listdir 함, 캐시 아님) + 무작위 200 장은 실제로 열어 크기 확인,
  (7) 파일 sha256 == qa_manifest.json
임을 세어서 보인다. 하나라도 어긋나면 종료 코드 1. 결과는 data/vla_teacher/b2d/qa/check_report.json 에도 저장.
  OPENBLAS_CORETYPE=SkylakeX CUDA_VISIBLE_DEVICES="" python -I tools/vla_teacher/check_qa_split_b2d.py
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2d_qa_common as B  # noqa: E402
import sft_common as S  # noqa: E402


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_dir", default=B.QA_DIR_B2D)
    ap.add_argument("--fold_file", default=B.FOLD_FILE_B2D)
    ap.add_argument("--threads", type=int, default=8, help="NFS listdir 대기용 스레드")
    ap.add_argument("--n_open", type=int, default=200)
    a = ap.parse_args(argv)
    qa_dir = Path(a.qa_dir) if os.path.isabs(a.qa_dir) else S.REPO / a.qa_dir
    man = json.load(open(qa_dir / "qa_manifest.json"))
    fd, _ = S.load_fold_split(a.fold_file)
    off_val = B.official_val_clips()
    sets = {f: dict(train=set(fd["train"][f]["scenes"]) - set(fd["holdout"][f]["scenes"]),
                    holdout=set(fd["holdout"][f]["scenes"]), all=set(fd["train"][f]["scenes"]), val=set(fd["val"][f]["scenes"]))
            for f in S.FOLDS}
    ok, rep = True, {}

    def check(name, cond, info):
        nonlocal ok
        ok &= bool(cond)
        rep[name] = dict(ok=bool(cond), info=info)
        print(f"{'PASS' if cond else 'FAIL'} {name}: {info}", flush=True)

    check("fold_hash_matches_manifest", man["fold_sha256"] == fd["sha256"], fd["sha256"][:12])
    check("fold_val_eq_official_val", (sets["A"]["val"] | sets["B"]["val"]) == off_val, f"{len(off_val)} clip")
    check("fold_train_val_disjoint", not ((sets["A"]["all"] | sets["B"]["all"]) & off_val) and not (sets["A"]["all"] & sets["B"]["all"]),
          "train.A∩train.B = ∅, train∩val = ∅")
    srcs, ids = Counter(), Counter()
    all_recs = []
    files = [(f, p) for f in S.FOLDS for p in ("train", "holdout")] + [("val", None)]
    for f, part in files:
        name = "val" if f == "val" else f"fold{f}_{part}"
        p = qa_dir / f"{name}.jsonl"
        check(f"{name}_sha256", S.file_sha256(p) == man["files"][name]["sha256"], str(p))
        recs = load_jsonl(p)
        all_recs += recs
        clips = {r["scene_token"] for r in recs}
        by = Counter(r["dataset"] for r in recs)
        qa_by = Counter()
        for r in recs:
            qa_by[r["dataset"]] += len(r["turns"])
            ids[r["id"]] += 1
            for t in r["turns"]:
                srcs[(r["dataset"], t["src"])] += 1
        print(f"[{name}] 표본 {len(recs)} {dict(by)} / QA {sum(qa_by.values())} {dict(qa_by)} / clip {len(clips)} / 프레임 "
              f"{len({(r['dataset'], r['token']) for r in recs})}", flush=True)
        rep[f"{name}_counts"] = dict(samples=len(recs), by_dataset=dict(by), qa=dict(qa_by), clips=len(clips))
        if f == "val":
            check("val_only_official_val", clips <= off_val, f"val 밖 clip {len(clips - off_val)}, 사용 {len(clips)}/50")
            check("val_fold_field", all(r["scene_token"] in sets[r["fold"]]["val"] and r["part"] == "val" for r in recs), "fold = val 반쪽")
            continue
        other = "B" if f == "A" else "A"
        check(f"{name}_val_overlap", not (clips & off_val), f"val clip {len(clips & off_val)}")
        check(f"{name}_other_fold_overlap", not (clips & sets[other]["all"]), f"fold {other} clip {len(clips & sets[other]['all'])}")
        check(f"{name}_in_assigned", clips <= sets[f][part] and all(r["fold"] == f and r["part"] == part for r in recs),
              f"배정 밖 clip {len(clips - sets[f][part])} (배정 {len(sets[f][part])} clip 중 {len(clips)} 사용)")
    check("no_duplicate_qa", all(v == 1 for v in srcs.values()), f"중복 QA {sum(v > 1 for v in srcs.values())} / 전체 {len(srcs)}")
    check("no_duplicate_id", all(v == 1 for v in ids.values()), f"중복 id {sum(v > 1 for v in ids.values())}")
    empty = sum(1 for r in all_recs for t in r["turns"] if not t["a"].strip() or not t["q"].strip())
    check("answers_nonempty", empty == 0 and all(r["turns"] for r in all_recs), f"빈 Q/A {empty}")
    over = sum(r["n_tokens"] > man["max_seq_tokens"] for r in all_recs)
    check("length_le_max", over == 0, f"초과 {over}, 최대 {max(r['n_tokens'] for r in all_recs)} / {man['max_seq_tokens']}")
    fmt = sum(r["cams"] != [B.cam_path(r["scene_token"], cd, r["token"].split("/")[1]) for cd in B.CAM_DIRS] for r in all_recs)
    check("cam_paths_format", fmt == 0, f"경로 형식/순서 다른 표본 {fmt}")

    # 영상 존재(새 listdir — 참조된 clip × 6 카메라 폴더)
    need = defaultdict(set)
    for r in all_recs:
        for cd, pth in zip(B.CAM_DIRS, r["cams"]):
            need[(r["scene_token"], cd)].add(os.path.basename(pth))

    def ls(k):
        d = f"{B.B2D_ROOT}/{k[0]}/camera/{k[1]}"
        return k, set(os.listdir(d)) if os.path.isdir(d) else set()
    with ThreadPoolExecutor(a.threads) as ex:
        have = dict(ex.map(ls, sorted(need)))
    n_ref = sum(len(v) for v in need.values())
    miss = sorted((k[0], k[1], f) for k, v in need.items() for f in v - have[k])
    check("images_exist", not miss, f"참조 영상 {n_ref} 장(고유), {len(need)} 폴더 listdir, 없음 {len(miss)} {miss[:3]}")
    rep["images"] = dict(n_unique_images=n_ref, n_dirs=len(need), missing=miss[:100], n_missing=len(miss))
    from PIL import Image
    rng = random.Random(0)
    sample = rng.sample([p for r in all_recs for p in r["cams"]], min(a.n_open, n_ref))
    sizes = Counter()
    for p in sample:
        with Image.open(p) as im:
            im.load()
            sizes[im.size] += 1
    check("images_open", sum(sizes.values()) == len(sample), f"{len(sample)} 장 열기, 크기 {dict((str(k), v) for k, v in sizes.items())}")
    print("ALL PASS" if ok else "SOME FAIL", flush=True)
    rep["all_pass"] = ok
    json.dump(rep, open(qa_dir / "check_report.json", "w"), indent=1, ensure_ascii=False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
