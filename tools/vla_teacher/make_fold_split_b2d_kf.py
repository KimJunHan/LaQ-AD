# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   B2D 2-fold 분할의 "2 Hz 키프레임 판". 기존 fold_split_b2d.json(make_fold_split_b2d.py, 표본 단위 = rgb_front 10 Hz 프레임)은
#   그대로 두고, clip 배정(train.A/B · val.A/B · holdout.A/B 의 clip 목록)을 한 글자도 바꾸지 않은 채 표본 수만 교사 정답
#   (teacher_gt_b2d_{train,val}.jsonl = 2 Hz 키프레임) 기준으로 다시 센 분할 파일을 만든다.
#   왜: sft_common.fold_select 는 "분할 파일의 clip 별 표본 수 = 지금 정답 데이터의 표본 수" 를 요구한다(PREP_REPORT 해결 안 된 것 1).
#       clip 배정을 그대로 두므로 QA 단계(기존 QA 데이터)와 SFT 단계의 교차 적합(teacher A 는 train.B 를 보지 않음)이 그대로 유지된다.
#   QA 단계와 SFT 단계가 같은 분할 해시를 요구하므로(sft_common.base_chain_info), QA 데이터 "보기 폴더"(qa_kf2hz/)를 같이 만든다:
#     jsonl 은 기존 qa/ 파일로 가는 심볼릭 링크(내용·sha256 그대로), qa_manifest.json 은 기존 것을 복사하고 fold_file·fold_sha256 만
#     새 분할로 바꾼 뒤 derived_from 에 원본 manifest 해시와 "clip 배정 동일" 검사 결과를 적는다.
#   사용: CUDA_VISIBLE_DEVICES="" python tools/vla_teacher/make_fold_split_b2d_kf.py --gt_dir <teacher_gt_b2d 폴더> [--force]
import argparse
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sft_common as S  # noqa: E402
import b2d_qa_common as QC  # noqa: E402

SRC_FOLD = QC.FOLD_FILE_B2D                                   # data/vla_teacher/b2d/fold_split_b2d.json
OUT_FOLD = "data/vla_teacher/b2d/fold_split_b2d_kf2hz.json"
SRC_QA = QC.QA_DIR_B2D                                        # data/vla_teacher/b2d/qa
OUT_QA = "data/vla_teacher/b2d/qa_kf2hz"


def counts(path):
    c = {}
    toks = set()
    with open(path) as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r["token"] in toks:
                raise RuntimeError(f"token 중복 {r['token']}")
            toks.add(r["token"])
            c[r["scene_token"]] = c.get(r["scene_token"], 0) + 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.chdir(S.REPO)
    out = S.REPO / OUT_FOLD
    qa_out = S.REPO / OUT_QA
    if (out.exists() or qa_out.exists()) and not a.force:
        raise SystemExit(f"[거부] {out} 또는 {qa_out} 있음(--force)")
    src, src_path = S.load_fold_split(SRC_FOLD)
    ctr = counts(Path(a.gt_dir) / "teacher_gt_b2d_train.jsonl")
    cva = counts(Path(a.gt_dir) / "teacher_gt_b2d_val.jsonl")
    tr_all = set(src["train"]["A"]["scenes"]) | set(src["train"]["B"]["scenes"])
    va_all = set(src["val"]["A"]["scenes"]) | set(src["val"]["B"]["scenes"])
    if set(ctr) != tr_all or set(cva) != va_all:
        raise SystemExit(f"정답 clip 집합 ≠ 분할 clip 집합: train 누락 {len(tr_all - set(ctr))} 초과 {len(set(ctr) - tr_all)}, "
                         f"val 누락 {len(va_all - set(cva))} 초과 {len(set(cva) - va_all)} — 조용히 맞추지 않음")
    d = json.loads(json.dumps(src))
    for part, cnt in (("train", ctr), ("val", cva), ("holdout", ctr)):
        for f in S.FOLDS:
            sc = src[part][f]["scenes"]
            e = S._half_entry(sc, cnt)
            if e["sha256"] != src[part][f]["sha256"]:
                raise RuntimeError("clip 목록 해시가 바뀌었다(있을 수 없음)")
            d[part][f].update(n_samples=e["n_samples"])
    d["balance"] = dict(train_sample_diff=abs(d["train"]["A"]["n_samples"] - d["train"]["B"]["n_samples"]),
                        val_sample_diff=abs(d["val"]["A"]["n_samples"] - d["val"]["B"]["n_samples"]))
    d["created"] = "2026-10-19"
    d["sample_unit"] = "2 Hz keyframe (frame_idx % 5 == 0) = teacher_gt_b2d record"
    d["derived_from"] = dict(path=str(src_path), sha256=src["sha256"], file_sha256=S.file_sha256(src_path),
                             rule="clip 배정(train/val/holdout × A/B) 동일, n_samples 만 키프레임 수로 재계산",
                             gt_train_sha256=S.file_sha256(Path(a.gt_dir) / "teacher_gt_b2d_train.jsonl"),
                             gt_val_sha256=S.file_sha256(Path(a.gt_dir) / "teacher_gt_b2d_val.jsonl"))
    d["sha256"] = S.fold_doc_hash(d)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    json.dump(d, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, out)
    S.load_fold_split(str(out))
    # QA 보기 폴더
    man_src = S.REPO / SRC_QA / "qa_manifest.json"
    man = json.load(open(man_src))
    if man["fold_sha256"] != src["sha256"]:
        raise SystemExit(f"기존 QA manifest 분할 해시 {man['fold_sha256'][:12]} != 원 분할 {src['sha256'][:12]}")
    qa_out.mkdir(parents=True, exist_ok=True)
    for name, e in man["files"].items():
        p_src = (S.REPO / e["path"]).resolve()
        if S.file_sha256(p_src) != e["sha256"]:
            raise SystemExit(f"{p_src}: manifest 해시 불일치")
        link = qa_out / Path(e["path"]).name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(p_src, link)
        e["path"] = f"{OUT_QA}/{link.name}"
    man["derived_from"] = dict(manifest=str(man_src.resolve()), manifest_sha256=S.file_sha256(man_src),
                               fold_file=man["fold_file"], fold_sha256=man["fold_sha256"],
                               note="jsonl 은 원본 링크(내용·해시 동일). clip 배정 동일 분할(키프레임 표본 수)로 해시만 교체")
    man["fold_file"] = str(out)
    man["fold_sha256"] = d["sha256"]
    man["tag"] = "# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)"
    json.dump(man, open(qa_out / "qa_manifest.json", "w"), indent=1, ensure_ascii=False)
    for part in ("train", "val", "holdout"):
        for f in S.FOLDS:
            e = d[part][f]
            print(f"[fold-b2d-kf] {part}.{f}: {e['n_scenes']} clip / {e['n_samples']} 키프레임")
    print(f"[fold-b2d-kf] {out} (sha256 {d['sha256'][:12]}), QA 보기 {qa_out}, 균형 {d['balance']}")


if __name__ == "__main__":
    main()
