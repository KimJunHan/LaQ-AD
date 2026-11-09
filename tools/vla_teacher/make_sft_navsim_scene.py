# 수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건
"""NAVSIM 교사 SFT 표본 = navtrain 장면 프레임만(103,288 중 교사 제외 6 로그 뺌) — 권장안 6. GPU 실행 없음(데이터만 준비).

입력: gen_teacher_gt_navsim.py 의 teacher_gt_navtrain.jsonl(전 프레임 152,495), fold_split_navsim.json(로그 단위 2-fold, sha256 고정).
출력(--out_dir): fold{A,B}_{train,holdout}.jsonl, special_tokens.txt·instruction.txt(생성기 산출 복사), manifest.json(수·해시·제외 이유).
  - 행 = 장면 프레임(is_scene_frame) 만. 과거 프레임은 SFT 입력의 이전 프레임 영상(cams_prev)으로만 쓰인다(레코드에 이미 있음).
  - fold 단위 = 로그: 레코드의 scene_token(시퀀스 id)은 seq_token 으로 옮기고 scene_token = log_name 으로 둔다
    (sft_common.fold_select 와 같은 "장면 = 분할 단위" 규칙. NAVSIM QA jsonl 과 같다).
  - 교사 제외 6 로그(NuPlanQA-Eval)는 모든 파일에서 뺀다. 그래서 sft_common.fold_select 의 fold 별 표본 수 검사
    (분할 파일 n_samples 와 같아야 함)는 이 파일들에 쓰지 않는다 — 대신 이 스크립트가 fold 파일과 대조한 수를 manifest 에 남긴다.
    train_sft.py 로 학습하려면 (a) 전방 3 캠·화소 128·28·28 인자, (b) fold 파일을 직접 읽는 입력이 필요하다(공용 파일 — 별도 승인).
"""
import argparse
import hashlib
import json
import os
import shutil

FOLD = "/home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/teacher/navsim/fold_split_navsim.json"
EXCLUDE = "/home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/qa/nuplanqa/teacher_exclude_logs_navsim.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    src = os.path.join(a.gt_dir, "teacher_gt_navtrain.jsonl")
    fold = json.load(open(FOLD))
    excl = set(json.load(open(EXCLUDE))["logs"])
    sets = {(s, f): set(fold[s][f]["scenes"]) for s in ("train", "holdout") for f in ("A", "B")}
    os.makedirs(a.out_dir, exist_ok=True)
    outs = {(s, f): open(os.path.join(a.out_dir, f"fold{f}_{s}.jsonl"), "w") for s in ("train", "holdout") for f in ("A", "B")}
    cnt = {f"fold{f}_{s}": 0 for s in ("train", "holdout") for f in ("A", "B")}
    reasons = dict(history_frame=0, excluded_log=0, scene_rows=0, not_in_fold_train=0)
    per_fold_scene_all = {"A": 0, "B": 0}
    with open(src) as fi:
        for ln in fi:
            r = json.loads(ln)
            if not r["is_scene_frame"]:
                reasons["history_frame"] += 1
                continue
            reasons["scene_rows"] += 1
            log = r["log_name"]
            fa = "A" if log in sets[("train", "A")] else "B" if log in sets[("train", "B")] else None
            if fa is None:
                reasons["not_in_fold_train"] += 1
                raise SystemExit(f"{log}: 분할 파일 train 에 없는 navtrain 로그")
            per_fold_scene_all[fa] += 1
            if log in excl:
                reasons["excluded_log"] += 1
                continue
            r["seq_token"], r["scene_token"] = r["scene_token"], log
            s = "holdout" if log in sets[("holdout", fa)] else "train"
            outs[(s, fa)].write(json.dumps(r, ensure_ascii=False) + "\n")
            cnt[f"fold{fa}_{s}"] += 1
    for f in outs.values():
        f.close()
    for n in ("special_tokens.txt", "instruction.txt"):
        shutil.copy(os.path.join(a.gt_dir, n), os.path.join(a.out_dir, n))
    man = dict(created="2026-10-19", rule="navtrain 장면 프레임만, 교사 제외 6 로그 제외, 분할 단위 = 로그(scene_token=log_name)",
               source=src, source_sha256=sha(src), fold_file=FOLD, fold_sha256=fold["sha256"], counts=cnt, reasons=reasons,
               fold_train_n_samples_file={f: fold["train"][f]["n_samples"] for f in ("A", "B")},
               fold_train_scene_rows_before_exclusion=per_fold_scene_all,
               match_fold_file={f: per_fold_scene_all[f] == fold["train"][f]["n_samples"] for f in ("A", "B")},
               files={k: sha(os.path.join(a.out_dir, k + ".jsonl")) for k in cnt})
    json.dump(man, open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: man[k] for k in ("counts", "reasons", "fold_train_n_samples_file", "fold_train_scene_rows_before_exclusion",
                                          "match_fold_file")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
