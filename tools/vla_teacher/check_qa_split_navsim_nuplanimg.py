# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
"""make_qa_data_navsim_nuplanimg.py 산출물 검사(check_qa_split_navsim.py 의 원본 영상 판 — 기존 파일은 바꾸지 않음).

  (1) navtest 로그·navtest 장면 token 과 겹침 0 (표본 token = NuPlanQA sample_token 또는 NAVSIM 프레임 token, 둘 다 nuPlan 프레임 token)
  (2) teacher_exclude_logs(NuPlanQA-Eval 6 로그) 표본 0, NuPlanQA-Eval sample_token·NAVSIM 프레임·영상 파일과 겹침 0
  (3) 다른 fold 로그와 겹침 0, 배정(train.F − holdout.F / holdout.F) 밖 0
  (4) 같은 QA(src) 두 번 0, 표본 id 중복 0, 빈 Q/A 0,
      카메라 3장 형식: NuPlanQA = nuPlan 원본 sensor_blobs/<log>/<cam>/, ReCogDrive = NAVSIM trainval sensor_blobs/<log>/<cam>/
  (5) 영상 전부 실재(로그×카메라 listdir, 스레드 8), NuPlanQA-Eval 1,801 전부 원본 영상 실재
  (6) manifest 의 분할 해시·파일 해시·n_cams·pixels 일치
하나라도 어긋나면 종료 코드 1.
  CUDA_VISIBLE_DEVICES="" python -I tools/vla_teacher/check_qa_split_navsim_nuplanimg.py --qa_dir <dir> --out <json>
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import navsim_qa_common as N  # noqa: E402
import qa_common as Q  # noqa: E402
import sft_common as S  # noqa: E402

ROOTS = {"nuplanqa": N.NUPLAN_RAW_BLOBS, "recogdrive_navsim": N.TRAIN_BLOBS}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_dir", required=True)
    ap.add_argument("--fold_file", default=N.FOLD_FILE_NAVSIM)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    qa_dir = Path(a.qa_dir)
    fd, _ = S.load_fold_split(a.fold_file)
    man = json.load(open(qa_dir / "qa_manifest.json"))
    fte = N.load_filter(N.NAVTEST_YAML)
    navtest_logs, navtest_tokens = set(fte["log_names"]), set(fte["tokens"])
    excl = set(fd["teacher_exclude_logs"]["logs"])
    val = set(fd["val"]["A"]["scenes"]) | set(fd["val"]["B"]["scenes"])
    sets = {f: dict(train=set(fd["train"][f]["scenes"]) - set(fd["holdout"][f]["scenes"]),
                    holdout=set(fd["holdout"][f]["scenes"]), all=set(fd["train"][f]["scenes"])) for f in S.FOLDS}
    evp = qa_dir / "nuplanqa_eval_navsim.jsonl"
    ev = [json.loads(l) for l in open(evp)]
    ev_sample = {r["sample_token"] for r in ev}
    ev_frames = {r["navsim"]["frame_token"] for r in ev if r["navsim"]}
    ev_imgs = {os.path.basename(p)[:-4] for r in ev if r["cams"] for p in r["cams"]}
    ok, report = True, {}

    def check(name, cond, info):
        nonlocal ok
        ok &= bool(cond)
        report[name] = dict(pass_=bool(cond), info=info)
        print(f"{'PASS' if cond else 'FAIL'} {name}: {info}", flush=True)

    check("fold_hash_matches_manifest", man["fold_sha256"] == fd["sha256"], fd["sha256"][:12])
    check("manifest_cams_pixels", man["n_cams"] == 3 and man["pixels"] == N.NAVSIM_PIXELS and man["cams"] == list(N.QA_CAMS),
          f"n_cams {man['n_cams']} pixels {man['pixels']} cams {man['cams']} 캠당 토큰 {man['tokens_per_cam']}")
    check("val_eq_navtest_logs", val <= navtest_logs, f"fold val 로그 {len(val)}, navtest 로그 {len(navtest_logs)}")
    check("eval_file_hash", S.file_sha256(evp) == man["eval_file"]["sha256"], evp.name)
    srcs, all_ids = Counter(), Counter()
    by_dir = defaultdict(set)
    summary = {}
    for f in S.FOLDS:
        other = "B" if f == "A" else "A"
        for part in ("train", "holdout"):
            recs, p = Q.load_qa(f, part, str(qa_dir))
            check(f"fold{f}_{part}_file_hash", S.file_sha256(p) == man["files"][f"fold{f}_{part}"]["sha256"], os.path.basename(p))
            tag = f"fold{f}_{part}"
            logs = {r["scene_token"] for r in recs}
            n_navtest = sum(r["scene_token"] in navtest_logs or r["token"] in navtest_tokens
                            or r.get("navsim_frame_token") in navtest_tokens for r in recs)
            n_excl = sum(r["scene_token"] in excl for r in recs)
            n_other = sum(r["scene_token"] in sets[other]["all"] for r in recs)
            n_out = sum(r["scene_token"] not in sets[f][part] for r in recs)
            n_evtok = sum(bool(set(r.get("nuplanqa_sample_tokens", [])) & ev_sample) or r["token"] in ev_sample for r in recs)
            n_evfr = sum(r["token"] in ev_frames or (r.get("navsim_frame_token") in ev_frames) for r in recs)
            n_evimg = sum(any(os.path.basename(pth)[:-4] in ev_imgs for pth in r["cams"]) for r in recs)
            n_empty = sum(1 for r in recs for t in r["turns"] if not t["a"].strip() or not t["q"].strip())
            n_badcam = 0
            for r in recs:
                root = ROOTS.get(r["dataset"])
                if root is None or len(r["cams"]) != 3:
                    n_badcam += 1; continue
                for c, pth in zip(N.QA_CAMS, r["cams"]):
                    pp = pth.split("/")
                    if not pth.startswith(root + "/") or pp[-3] != r["scene_token"] or pp[-2] != c:
                        n_badcam += 1; break
                    by_dir[(root, pp[-3], c)].add(pp[-1])
                all_ids[r["id"]] += 1
                for t in r["turns"]:
                    srcs[(r["dataset"], t["src"])] += 1
            qa_by, n_by = Counter(), Counter()
            for r in recs:
                qa_by[r["dataset"]] += len(r["turns"]); n_by[r["dataset"]] += 1
            summary[tag] = dict(samples=len(recs), by_dataset=dict(n_by), qa=dict(qa_by), logs=len(logs),
                                frames=len({r["token"] for r in recs}))
            print(f"[{tag}] {summary[tag]}", flush=True)
            check(f"{tag}_navtest_overlap", n_navtest == 0 and not (logs & navtest_logs), f"navtest 로그/token 표본 {n_navtest}")
            check(f"{tag}_teacher_exclude_logs", n_excl == 0, f"제외 6 로그 표본 {n_excl}")
            check(f"{tag}_nuplanqa_eval_overlap", n_evtok == 0 and n_evfr == 0 and n_evimg == 0,
                  f"Eval sample_token 겹침 {n_evtok}, Eval NAVSIM 프레임 겹침 {n_evfr}, Eval 영상 파일 겹침 {n_evimg}")
            check(f"{tag}_other_fold_overlap", n_other == 0, f"fold {other} 로그 표본 {n_other}")
            check(f"{tag}_in_assigned_logs", n_out == 0, f"배정 밖 표본 {n_out}")
            check(f"{tag}_nonempty_qa", n_empty == 0, f"빈 Q/A {n_empty}")
            check(f"{tag}_cams_format", n_badcam == 0, f"카메라 형식 위반 {n_badcam}")
    dup = sum(1 for v in srcs.values() if v > 1)
    check("no_duplicate_qa", dup == 0, f"두 번 이상 들어간 QA {dup} / 전체 {len(srcs)}")
    check("no_duplicate_sample_id", all(v == 1 for v in all_ids.values()), f"표본 {len(all_ids)}")
    # Eval: 1,801 전부, 원본 영상, 제외 6 로그 안
    n_ev_bad = sum(1 for r in ev if not r["cams"] or r["image_source"] != "nuplan_raw_sensor_blobs"
                   or any(not p.startswith(N.NUPLAN_RAW_BLOBS + "/") for p in r["cams"]))
    n_ev_out = sum(1 for r in ev if r.get("nuplan_log") not in excl)
    for r in ev:
        if r["cams"]:
            for c, pth in zip(N.QA_CAMS, r["cams"]):
                pp = pth.split("/")
                by_dir[(N.NUPLAN_RAW_BLOBS, pp[-3], pp[-2])].add(pp[-1])
    check("eval_all_items", len(ev) == 1801 and len({r["id"] for r in ev}) == 1801, f"Eval 문항 {len(ev)}")
    check("eval_original_images", n_ev_bad == 0, f"원본 영상 아닌/없는 Eval 문항 {n_ev_bad}")
    check("eval_in_teacher_exclude_logs", n_ev_out == 0, f"제외 6 로그 밖 Eval 문항 {n_ev_out}")

    def miss_of(key):
        root, lg, c = key
        try:
            have = set(os.listdir(os.path.join(root, lg, c)))
        except FileNotFoundError:
            have = set()
        m = by_dir[key] - have
        return len(by_dir[key]), len(m), [f"{lg}/{c}/{x}" for x in sorted(m)[:2]]

    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(miss_of, sorted(by_dir)))
    tot, miss = sum(r[0] for r in res), sum(r[1] for r in res)
    check("all_images_exist", miss == 0, f"영상 {tot} 개(폴더 {len(by_dir)}) 중 없음 {miss} {[x for r in res for x in r[2]][:3]}")
    print("ALL PASS" if ok else "SOME FAIL", flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(all_pass=ok, summary=summary, checks=report), fh, indent=1, ensure_ascii=False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
