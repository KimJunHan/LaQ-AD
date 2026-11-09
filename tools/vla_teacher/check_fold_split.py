# 수정(2026-10-16): 2-fold 장면 분할 — 분할 파일 검사(CPU 전용, 모델·영상 미사용).
#   1) 파일 해시(전체·장면 목록별)와 원본 jsonl sha256 일치, 같은 seed 로 재생성 시 완전 동일(결정성)
#   2) train: A∩B 장면 0, A∪B = 전체 train 장면, 모든 train 표본이 정확히 한 번 배정, 표본 수·균형
#   3) hold-out ⊂ 자기 fold, 다른 fold·val 과 겹침 0, 크기
#   4) val: A∩B 0, A∪B = 전체 val 장면, 표본 정확히 한 번, train 과 장면 겹침 0
#   5) 정답 생성 배정: 각 교사가 생성하는 train 장면 = 자기가 학습하지 않은 반쪽, train 표본 전체를 정확히 한 번 덮음
#   6) train_sft.select_train_holdout: --fold none 이 이전 코드(전체 − 고정 50 장면)와 표본 목록·순서까지 동일,
#      --fold A/B 의 학습+hold-out = 그 반쪽 표본 전체, 두 fold 합 = train 전체
import json
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_common as C  # noqa: E402
import train_sft as T  # noqa: E402

RESULTS = {}


def ok(name, cond, info=""):
    RESULTS[name] = dict(ok=bool(cond), info=str(info))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    return cond


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/vla_teacher/v2")
    ap.add_argument("--fold_file", default=C.FOLD_FILE)
    ap.add_argument("--out", default="results/vla_teacher/check_fold_split.json")
    a = ap.parse_args()
    data_dir = C.resolve_data_dir(a.data_dir)
    tr, va = C.load_records(data_dir, "train"), C.load_records(data_dir, "val")
    d, path = C.load_fold_split(a.fold_file)            # 해시 불일치면 여기서 실패
    ok("fold_file_hash", True, f"{path} sha256 {d['sha256'][:12]}")
    for split in ("train", "val"):
        f = C.split_file(data_dir, split)
        ok(f"source_sha256_{split}", C.file_sha256(f) == d["sources"][split]["sha256"], os.path.relpath(f, C.REPO))

    ctr, cva = C.scene_sample_counts(tr), C.scene_sample_counts(va)
    re = C.make_fold_split(tr, va, seed=d["seed"], n_holdout=d["holdout"]["A"]["n_scenes"], sources=d["sources"])
    ok("deterministic_rebuild", re == d, "같은 seed 로 다시 만들면 파일과 완전히 같음")
    stats = {}
    for split, recs, cnt in (("train", tr, ctr), ("val", va, cva)):
        A, B = set(d[split]["A"]["scenes"]), set(d[split]["B"]["scenes"])
        ok(f"{split}_disjoint", not (A & B), f"A∩B = {len(A & B)}")
        ok(f"{split}_complete", (A | B) == set(cnt), f"A∪B {len(A | B)} / 전체 {len(cnt)} 장면")
        side = {}
        for r in recs:
            side.setdefault(r["token"], []).append("A" if r["scene_token"] in A else ("B" if r["scene_token"] in B else "-"))
        once = all(len(v) == 1 and v[0] in "AB" for v in side.values()) and len(side) == len(recs)
        ok(f"{split}_each_sample_once", once, f"{len(recs)} 표본")
        nA, nB = sum(cnt[s] for s in A), sum(cnt[s] for s in B)
        ok(f"{split}_counts_match_file", (nA, nB) == (d[split]["A"]["n_samples"], d[split]["B"]["n_samples"])
           and nA + nB == len(recs), f"A {len(A)} 장면/{nA} 표본, B {len(B)} 장면/{nB} 표본")
        stats[split] = dict(A_scenes=len(A), A_samples=nA, B_scenes=len(B), B_samples=nB,
                            diff_samples=abs(nA - nB), diff_pct=round(100 * abs(nA - nB) / len(recs), 3))
        ok(f"{split}_balance", abs(nA - nB) <= max(cnt.values()), f"차이 {abs(nA - nB)} 표본({stats[split]['diff_pct']}%) "
           f"≤ 최대 장면 크기 {max(cnt.values())}")
    ok("train_val_scene_disjoint", not (set(ctr) & set(cva)), f"{len(set(ctr) & set(cva))}")

    for f in C.FOLDS:
        H = set(d["holdout"][f]["scenes"])
        mine, other = set(d["train"][f]["scenes"]), set(d["train"]["B" if f == "A" else "A"]["scenes"])
        ok(f"holdout_{f}_inside_fold", H <= mine and not (H & other) and not (H & set(cva)),
           f"{len(H)} 장면 / {sum(ctr[s] for s in H)} 표본, 다른 fold 겹침 {len(H & other)}, val 겹침 {len(H & set(cva))}")
        stats[f"holdout_{f}"] = dict(scenes=len(H), samples=sum(ctr[s] for s in H))

    # 정답 생성 배정
    gen = {f: d["assignment"][f"teacher_{f}"]["generate"] for f in C.FOLDS}
    ok("assignment", gen == {"A": ["train.B", "val.B"], "B": ["train.A", "val.A"]}, json.dumps(gen))
    cover = {}
    for f in C.FOLDS:
        for part in gen[f]:
            sp, h = part.split(".")
            for s in d[sp][h]["scenes"]:
                cover.setdefault((sp, s), []).append(f)
                ok_trained = s not in set(d["train"][f]["scenes"])
                if not ok_trained:
                    ok(f"gen_unseen_{f}_{s}", False, "교사가 자기가 학습한 장면을 생성")
    all_sc = [("train", s) for s in ctr] + [("val", s) for s in cva]
    ok("generation_covers_all_once", all(len(cover.get(k, [])) == 1 for k in all_sc) and len(cover) == len(all_sc),
       f"train {len(ctr)} + val {len(cva)} 장면, 각 1 교사")

    # train_sft 의 데이터 선택(--fold none 은 이전 코드와 동일해야 함)
    a0 = T.parse_args([])
    tr0, ho0, hold0, h0, p0, fi0 = T.select_train_holdout(a0, tr)
    old_hold, old_h, _ = C.holdout_scenes(tr, 50, 0, "results/vla_teacher/holdout_train_scenes.json")   # 이전 main 의 식 그대로
    old_tr = [r for r in tr if r["scene_token"] not in old_hold]
    old_ho = [r for r in tr if r["scene_token"] in old_hold]
    ok("fold_none_identical", [r["token"] for r in tr0] == [r["token"] for r in old_tr]
       and [r["token"] for r in ho0] == [r["token"] for r in old_ho] and h0 == old_h and fi0 is None
       and a0.out == "results/vla_teacher/sft_default" and a0.holdout_file == "results/vla_teacher/holdout_train_scenes.json",
       f"train {len(tr0)} / hold-out {len(ho0)} 표본, 해시 {h0[:12]}, out {a0.out}")
    stats["fold_none"] = dict(train=len(tr0), holdout=len(ho0), holdout_scenes=len(hold0))
    seen = []
    for f in C.FOLDS:
        af = T.parse_args(["--fold", f])
        trf, hof, holdf, hf, pf, fi = T.select_train_holdout(af, tr)
        mine = set(d["train"][f]["scenes"])
        ok(f"fold_{f}_select", {r["scene_token"] for r in trf} == mine - holdf and {r["scene_token"] for r in hof} == holdf
           and len(trf) + len(hof) == d["train"][f]["n_samples"] and af.out == f"results/vla_teacher/sft_fold{f}",
           f"학습 {len(trf)} 표본/{len(mine - holdf)} 장면, hold-out {len(hof)} 표본/{len(holdf)} 장면, out {af.out}")
        stats[f"fold_{f}"] = dict(train=len(trf), train_scenes=len(mine - holdf), holdout=len(hof), holdout_scenes=len(holdf))
        seen += [r["token"] for r in trf + hof]
    ok("folds_union_all_train", sorted(seen) == sorted(r["token"] for r in tr), f"{len(seen)} = {len(tr)}")
    try:
        T.parse_args(["--fold", "A", "--holdout_file", "x.json"]); rej = False
    except SystemExit:
        rej = True
    ok("fold_rejects_holdout_file", rej)

    n_fail = sum(not v["ok"] for v in RESULTS.values())
    out = C.REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(fold_file=path, fold_sha256=d["sha256"], stats=stats, checks=RESULTS, n_fail=n_fail),
              open(out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"[check] {len(RESULTS)} 검사, 실패 {n_fail} → {out}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
