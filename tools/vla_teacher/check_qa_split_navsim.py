# 수정(2026-10-16): 교사 QA — NAVSIM 트랙. QA 표본 유출·형식 검사(CPU). check_qa_split.py 의 NAVSIM 판.
"""data/vla_teacher/navsim/fold{A,B}_{train,holdout}.jsonl 이
  (1) navtest 로그·navtest 장면 token·test 로그 영상과 겹침 0,
  (2) teacher_exclude_logs(NuPlanQA-Eval 6 로그) 표본 0, NuPlanQA-Eval sample_token/프레임과 겹침 0,
  (3) 다른 fold 로그와 겹침 0, teacher_<F> 학습 = train.<F> − holdout.<F>, hold-out = holdout.<F> 와 정확히 일치,
  (4) 같은 QA(src)가 두 번 들어가지 않음, 빈 답 0, 카메라 3장·경로 = NAVSIM trainval sensor_blobs, 영상 전부 실재(로그별 listdir),
  (5) manifest 해시·n_cams·pixels 일치
임을 세어서 보인다. 하나라도 어긋나면 종료 코드 1.
  CUDA_VISIBLE_DEVICES="" python -I tools/vla_teacher/check_qa_split_navsim.py
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import navsim_qa_common as N  # noqa: E402
import qa_common as Q  # noqa: E402
import sft_common as S  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_dir", default=N.QA_DIR_NAVSIM)
    ap.add_argument("--fold_file", default=N.FOLD_FILE_NAVSIM)
    ap.add_argument("--out", default=None, help="결과 JSON(선택)")
    a = ap.parse_args(argv)
    qa_dir = Path(a.qa_dir) if os.path.isabs(a.qa_dir) else S.REPO / a.qa_dir
    fd, _ = S.load_fold_split(a.fold_file)
    man = json.load(open(qa_dir / "qa_manifest.json"))
    fte = N.load_filter(N.NAVTEST_YAML)
    navtest_logs, navtest_tokens = set(fte["log_names"]), set(fte["tokens"])
    excl = set(fd["teacher_exclude_logs"]["logs"])
    val = set(fd["val"]["A"]["scenes"]) | set(fd["val"]["B"]["scenes"])
    sets = {f: dict(train=set(fd["train"][f]["scenes"]) - set(fd["holdout"][f]["scenes"]),
                    holdout=set(fd["holdout"][f]["scenes"]), all=set(fd["train"][f]["scenes"])) for f in S.FOLDS}
    ev = [json.loads(l) for l in open(qa_dir / "nuplanqa_eval_navsim.jsonl")]
    ev_sample = {r["sample_token"] for r in ev}
    ev_frames = {r["navsim"]["frame_token"] for r in ev if r["navsim"]}
    ok, report = True, {}

    def check(name, cond, info):
        nonlocal ok
        ok &= bool(cond)
        report[name] = dict(pass_=bool(cond), info=info)
        print(f"{'PASS' if cond else 'FAIL'} {name}: {info}", flush=True)

    check("fold_hash_matches_manifest", man["fold_sha256"] == fd["sha256"], fd["sha256"][:12])
    check("manifest_cams_pixels", man["n_cams"] == 3 and man["pixels"] == N.NAVSIM_PIXELS and man["cams"] == list(N.QA_CAMS),
          f"n_cams {man['n_cams']} pixels {man['pixels']} cams {man['cams']} 캠당 토큰 {man['tokens_per_cam']}")
    check("val_eq_navtest_logs", val == {l for l in navtest_logs if l in val} and val <= navtest_logs,
          f"fold val 로그 {len(val)}, navtest 로그 {len(navtest_logs)}")
    srcs = Counter()
    by_log_files = defaultdict(set)
    all_tokens = Counter()
    for f in S.FOLDS:
        other = "B" if f == "A" else "A"
        for part in ("train", "holdout"):
            recs, p = Q.load_qa(f, part, str(qa_dir))
            check(f"fold{f}_{part}_file_hash", S.file_sha256(p) == man["files"][f"fold{f}_{part}"]["sha256"], os.path.basename(p))
            logs = {r["scene_token"] for r in recs}
            tag = f"fold{f}_{part}"
            n_navtest = sum(r["scene_token"] in navtest_logs or r["token"] in navtest_tokens for r in recs)
            n_excl = sum(r["scene_token"] in excl for r in recs)
            n_other = sum(r["scene_token"] in sets[other]["all"] for r in recs)
            n_out = sum(r["scene_token"] not in sets[f][part] for r in recs)
            n_evtok = sum(bool(set(r.get("nuplanqa_sample_tokens", [])) & ev_sample) for r in recs)
            n_evfr = sum(r["token"] in ev_frames for r in recs)
            n_empty = sum(1 for r in recs for t in r["turns"] if not t["a"].strip() or not t["q"].strip())
            n_badcam = 0
            for r in recs:
                if len(r["cams"]) != 3:
                    n_badcam += 1; continue
                for c, pth in zip(N.QA_CAMS, r["cams"]):
                    pp = pth.split("/")
                    if not pth.startswith(N.TRAIN_BLOBS + "/") or pp[-3] != r["scene_token"] or pp[-2] != c:
                        n_badcam += 1; break
                    by_log_files[(pp[-3], c)].add(pp[-1])
                all_tokens[(r["dataset"], r["token"], r["chunk"])] += 1
                for t in r["turns"]:
                    srcs[(r["dataset"], t["src"])] += 1
            qa_by = Counter()
            for r in recs:
                qa_by[r["dataset"]] += len(r["turns"])
            print(f"[{tag}] 표본 {len(recs)} {dict(Counter(r['dataset'] for r in recs))} / QA {dict(qa_by)} / 로그 {len(logs)} / "
                  f"프레임 {len({r['token'] for r in recs})}", flush=True)
            check(f"{tag}_navtest_overlap", n_navtest == 0 and not (logs & navtest_logs), f"navtest 로그/token 표본 {n_navtest}")
            check(f"{tag}_teacher_exclude_logs", n_excl == 0, f"제외 6 로그 표본 {n_excl}")
            check(f"{tag}_nuplanqa_eval_overlap", n_evtok == 0 and n_evfr == 0, f"Eval sample_token 겹침 {n_evtok}, Eval 프레임 겹침 {n_evfr}")
            check(f"{tag}_other_fold_overlap", n_other == 0, f"fold {other} 로그 표본 {n_other}")
            check(f"{tag}_in_assigned_logs", n_out == 0, f"배정 밖 표본 {n_out}")
            check(f"{tag}_nonempty_qa", n_empty == 0, f"빈 Q/A {n_empty}")
            check(f"{tag}_cams_format", n_badcam == 0, f"카메라 형식 위반 {n_badcam}")
    dup = sum(1 for v in srcs.values() if v > 1)
    check("no_duplicate_qa", dup == 0, f"두 번 이상 들어간 QA {dup} / 전체 {len(srcs)}")
    check("no_duplicate_sample_id", all(v == 1 for v in all_tokens.values()), f"표본 {len(all_tokens)}")
    # 영상 실재: 로그×카메라 listdir
    miss, tot = 0, 0
    miss_ex = []
    for (lg, c), names in sorted(by_log_files.items()):
        try:
            have = set(os.listdir(os.path.join(N.TRAIN_BLOBS, lg, c)))
        except FileNotFoundError:
            have = set()
        m = names - have
        miss += len(m); tot += len(names)
        miss_ex += [f"{lg}/{c}/{x}" for x in sorted(m)[:2]]
    check("all_images_exist", miss == 0, f"영상 {tot} 개 중 없음 {miss} {miss_ex[:3]}")
    print("ALL PASS" if ok else "SOME FAIL", flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(all_pass=ok, checks=report), fh, indent=1, ensure_ascii=False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
