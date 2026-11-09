# 수정(2026-11-05): navhard_two_stage 평가 준비 — 교사 캐시 추출 래퍼(새 파일; extract_teacher_cache_navsim.py 무수정 import).
"""navhard 프레임(1단계 원본 1,125 + 2단계 합성 13,655 = 고유 14,780) 교사 캐시. GPU 필요 — 실행은 사용자 승인 후(GPU3 = NAVSIM 전용).

extract_teacher_cache_navsim.py 의 run()(모델·프롬프트·문법 제약 생성·은닉 상태 저장·색인·_meta)을 그대로 쓰고, 표본 목록만 바꾼다:
  - 표본 = make_navhard_teacher_inputs.py 의 teacher_inputs_navhard.jsonl(입력 필드만; 정답 없음).
  - 배정 = 분할 파일의 navtest 규칙(로그 ∈ val.A → teacher_B, ∈ val.B → teacher_A). 원 scan_assignment 는 navtrain/navtest GT 파일의
    part 별 장면 수를 분할 파일과 대조하므로 navhard 에는 쓸 수 없어 같은 반환 형식의 navhard 판으로 바꾼다.
  - 누설 검사(check_unseen: 배정 로그 ∩ train.<교사> = ∅, QA·SFT 사용 장면 ⊆ train.<교사>, 배정과 겹침 0)·교사 체인 검사
    (verify_teacher_chain)·출처 검사(check_sources: navtest GT 폴더의 special_tokens/instruction == 교사 SFT 자료)는 원 함수 그대로.
  - 출력 = 별도 루트(기본 data/navsim/laq_cache_navhard). 학생 학습·navtest 용 data/navsim/laq_cache 와 섞지 않는다
    (그 캐시의 --verify --infos 는 navtrain+navtest token 집합 일치를 요구하므로 navhard token 이 섞이면 깨진다).
--dry: 모델을 올리지 않고(CPU) 체인·출처·배정·누설 검사 + 입력 영상 존재 확인 + 표본 수·예상 시간만 출력.

예(교사별 2 샤드 = 4 프로세스, navtest 캐시와 같은 방식):
  CUDA_VISIBLE_DEVICES=3 python tools/navsim/extract_teacher_cache_navhard.py --teacher A \
      --model results/vla_teacher/navsim_merged_foldA_qa --batch 16 --shard 0 --num_shards 2
  CUDA_VISIBLE_DEVICES="" python tools/navsim/extract_teacher_cache_navhard.py --teacher A --model … --dry
  CUDA_VISIBLE_DEVICES="" python tools/navsim/extract_teacher_cache_navhard.py --verify     # 색인·파일 = 입력 token 전부
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vla_teacher"))
import extract_teacher_cache_navsim as E  # noqa: E402
import sft_common as SC  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
INPUTS = os.path.join(REPO, "data/infos/navsim_navhard_20261105/teacher_inputs_navhard.jsonl")
OUT_ROOT = "/home1/irteam/data-vol1/src/HiP-AD/data/navsim/laq_cache_navhard"
SEC_PER_SAMPLE = 3.0      # navtest 캐시 실측(배치 16, 프로세스당)


COPY_FIELDS = ("L", "hidden_dim", "dtype", "layer", "answer_tokens", "positions_sha256", "prompt_mode", "ego_in_prompt",
               "constrained_decoding", "answer_grammar_sha256", "dataset", "format")
MAIN_ROOT = "/home1/irteam/data-vol1/src/HiP-AD/data/navsim/laq_cache"


def copy_index_path(root, T):
    return Path(root) / "_index" / f"teacher_{T}.copy.jsonl"


def copied_tokens(root):
    out = {}
    for T in ("A", "B"):
        out.update(E.X.read_index(copy_index_path(root, T)))
    return out


def make_scan(inputs_path, root=None, require_copy=False):
    """root 를 주면 복사 색인(_index/teacher_<T>.copy.jsonl)에 있는 token 은 생성 목록에서 뺀다.
    require_copy=True(실제 생성)면 navtest 입력(input_source=navtest_gt) token 이 모두 복사돼 있어야 한다(순서 강제: 복사 → 생성)."""
    def scan_assignment(gt_dir, fold_file, need_records_for=None):
        done = copied_tokens(root) if root is not None else {}
        d, fp = E.load_fold_split(fold_file)
        owner = {}
        for T in ("A", "B"):
            for s in d["assignment"][f"teacher_{T}"]["generate"]:
                split, f = s.split(".")
                if split == "val":
                    for log in d["val"][f]["scenes"]:
                        owner[log] = (T, s)
        assign, items, stats = {}, [], {}
        with open(inputs_path) as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                if r["log_name"] not in owner:
                    raise RuntimeError(f"{r['token']}: 로그 {r['log_name']} 가 val.A/val.B 에 없다(navhard 는 navtest 로그만)")
                T, part = owner[r["log_name"]]
                if r.get("teacher") != T or r.get("part") != part:
                    raise RuntimeError(f"{r['token']}: 입력 파일 배정 {r.get('teacher')}/{r.get('part')} != 분할 파일 {T}/{part}")
                if r["token"] in assign:
                    raise RuntimeError(f"token 중복 {r['token']}")
                assign[r["token"]] = (T, part, r["log_name"])
                st = stats.setdefault(part, dict(rows=0, synthetic=0, logs=set()))
                st["rows"] += 1; st["synthetic"] += int(bool(r.get("synthetic"))); st["logs"].add(r["log_name"])
                if r.get("input_source") == "navtest_gt" and require_copy and r["token"] not in done:
                    raise RuntimeError(f"{r['token']}: navtest 캐시 복사가 안 됐다 — --copy_from 을 먼저 실행")
                st["copied"] = st.get("copied", 0) + int(r["token"] in done)
                if need_records_for == T and r["token"] not in done:
                    items.append((part, E.strip_to_input(r)))
        for st in stats.values():
            st["logs"] = len(st["logs"])
        return assign, items, d, fp, stats
    return scan_assignment


def dry(a):
    meta = E.load_meta(a.model)
    chain = E.verify_teacher_chain(meta, a.teacher, a.fold_file)
    src = E.check_sources(meta, a.gt_dir)
    assign, items, fdoc, _, stats = E.scan_assignment(a.gt_dir, a.fold_file, need_records_for=a.teacher)
    unseen = E.check_unseen(meta, chain, fdoc, a.teacher, items)
    miss = 0
    for _, r in items:
        miss += len(SC.cam_paths(r, True, "prev_first", E.LN.NUM_CAMS)) != 2 * E.LN.NUM_CAMS   # 파일 없으면 FileNotFoundError
    n = len(items)
    print(json.dumps(dict(teacher=a.teacher, samples=n, all_samples=len(assign), stats=stats, unseen_check=unseen, sources=src,
                          image_count_bad=miss,
                          est_hours_1proc=round(n * SEC_PER_SAMPLE / 3600, 2),
                          est_hours_per_shard=round(n * SEC_PER_SAMPLE / 3600 / a.num_shards, 2)), indent=1, ensure_ascii=False))
    print("DRY OK — 모델 미적재, 저장 없음")


def sha_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main_meta(main_root):
    return {k: json.load(open(Path(main_root) / k / "_meta.json")) for k in E.LN.FOLDERS}


def copy_from(root, main_root, inputs_path):
    """navtest 캐시(main_root)에 이미 있는 1단계 원본 프레임(input_source=navtest_gt)을 그대로 복사한다.
    조건: main 색인 항목의 교사·part == navhard 배정(같은 분할 파일 규칙), 7 폴더 파일 존재, 복사본 sha256 == 원본.
    복사 색인 = _index/teacher_<T>.copy.jsonl(main 항목 + copied_from·sha256). 이미 복사된 token 은 sha 재확인 후 건너뜀."""
    import shutil
    root, main_root = Path(root), Path(main_root)
    midx = E.read_all_index(main_root)
    rows = [json.loads(ln) for ln in open(inputs_path) if ln.strip()]
    want = [r for r in rows if r.get("input_source") == "navtest_gt"]
    (root / "_index").mkdir(parents=True, exist_ok=True)
    for k in E.LN.FOLDERS:
        (root / k).mkdir(parents=True, exist_ok=True)
    done = copied_tokens(root)
    fh = {T: open(copy_index_path(root, T), "a") for T in ("A", "B")}
    n_new = n_old = 0
    for r in want:
        t, T = r["token"], r["teacher"]
        e = midx.get(t)
        if e is None:
            raise SystemExit(f"[중단] {t}: navtest 캐시 색인에 없다({main_root})")
        if e["teacher"] != T or e.get("part") != r["part"]:
            raise SystemExit(f"[중단] {t}: navtest 캐시 교사/part {e['teacher']}/{e.get('part')} != navhard 배정 {T}/{r['part']}")
        shas = {}
        for k in E.LN.FOLDERS:
            src, dst = main_root / k / f"{t}.pt", root / k / f"{t}.pt"
            hs = sha_file(src)
            if t in done:
                if sha_file(dst) != hs or done[t]["copy_sha256"][k] != hs:
                    raise SystemExit(f"[중단] {t} {k}: 기존 복사본이 원본과 다르다")
            else:
                tmp = dst.with_name(dst.name + f".tmp{os.getpid()}")
                shutil.copyfile(src, tmp)
                if sha_file(tmp) != hs:
                    raise SystemExit(f"[중단] {t} {k}: 복사 sha256 불일치")
                os.replace(tmp, dst)
            shas[k] = hs
        if t in done:
            n_old += 1
            continue
        fh[T].write(json.dumps(dict(e, copied_from=str(main_root), copy_sha256=shas), ensure_ascii=False) + "\n")
        fh[T].flush(); os.fsync(fh[T].fileno())
        n_new += 1
    for f in fh.values():
        f.close()
    mm = main_meta(main_root)
    man = dict(main_root=str(main_root), n_copied=len(want), teachers={T: dict(checkpoint=mm["context"]["teachers"][T]["checkpoint"],
               checkpoint_sha=mm["context"]["teachers"][T]["checkpoint_sha"]) for T in ("A", "B")},
               fixed={k: {f: mm[k].get(f) for f in COPY_FIELDS} for k in E.LN.FOLDERS})
    json.dump(man, open(root / "_index" / "copy_manifest.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(dict(copy_target=len(want), copied_now=n_new, already=n_old), indent=1))


def verify(root, inputs_path):
    """색인·파일 = 입력 token 전부 + 복사본 sha256 재확인 + _meta 의 고정 설정·교사 체크포인트 sha == navtest 캐시(복사 원본)."""
    rc = 0
    cp = copied_tokens(root)
    manp = Path(root) / "_index" / "copy_manifest.json"
    if cp:
        man = json.load(open(manp))
        bad_sha = sum(1 for t, e in cp.items() for k in E.LN.FOLDERS if sha_file(Path(root) / k / f"{t}.pt") != e["copy_sha256"][k])
        bad_meta = []
        for k in E.LN.FOLDERS:
            m = json.load(open(Path(root) / k / "_meta.json"))
            for f in COPY_FIELDS:
                if m.get(f) != man["fixed"][k].get(f):
                    bad_meta.append((k, f))
            for T in ("A", "B"):
                if m.get("teachers", {}).get(T, {}).get("checkpoint_sha") != man["teachers"][T]["checkpoint_sha"]:
                    bad_meta.append((k, f"teacher_{T}.checkpoint_sha"))
        print(json.dumps(dict(copied=len(cp), copied_sha_bad=bad_sha, meta_mismatch=bad_meta[:10]), indent=1))
        rc |= int(bool(bad_sha or bad_meta))
    want = set()
    with open(inputs_path) as fh:
        for ln in fh:
            if ln.strip():
                want.add(json.loads(ln)["token"])
    idx = E.read_all_index(root)
    miss_idx = want - set(idx)
    extra = set(idx) - want
    miss_f = [t for t in want for k in E.LN.FOLDERS if not (root / k / f"{t}.pt").exists()]
    bad = sum(1 for t in want if t in idx and idx[t].get("parse_failure"))
    print(json.dumps(dict(want=len(want), indexed=len(idx), missing_index=len(miss_idx), extra_index=len(extra),
                          missing_files=len(miss_f), parse_failure=bad), indent=1))
    return rc | (1 if (miss_idx or extra or miss_f) else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=INPUTS)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--copy_from", help="navtest 캐시 루트 — 1단계 원본 프레임(navtest 입력) 복사만 하고 끝낸다")
    a, rest = ap.parse_known_args()
    out_root = next((rest[i + 1] for i, x in enumerate(rest) if x == "--out_root"), OUT_ROOT)
    root = E.out_root_checked(out_root)
    E.OUT_ROOT = OUT_ROOT
    if a.copy_from:
        return copy_from(root, a.copy_from, a.inputs)
    E.scan_assignment = make_scan(a.inputs, root, require_copy=not a.dry and "--verify" not in rest)
    if "--verify" in rest:
        sys.exit(verify(root, a.inputs))
    if a.dry:
        p = argparse.ArgumentParser()
        p.add_argument("--teacher", choices=["A", "B"], required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--gt_dir", default=E.NS.NAVSIM_GT_DIR)
        p.add_argument("--fold_file", default=E.NS.NAVSIM_FOLD_FILE)
        p.add_argument("--num_shards", type=int, default=2)
        p.add_argument("--out_root", default=OUT_ROOT)   # 복사 색인 확인용(위 make_scan 이 이미 읽음)
        return dry(p.parse_args(rest))
    sys.argv = [sys.argv[0]] + rest
    E.main()     # 원 main: --teacher --model [--shard --num_shards --batch …] → run(a, root)


if __name__ == "__main__":
    main()
