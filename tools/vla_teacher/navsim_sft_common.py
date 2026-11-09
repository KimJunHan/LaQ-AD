# 수정(2026-10-29): NAVSIM 교사 SFT·캐시 — NAVSIM 전용 상수·자료 선택(새 파일, nuScenes/B2D 경로 무변경).
"""NAVSIM 교사 SFT(train_sft.py --navsim_sft_dir)와 캐시 추출(extract_teacher_cache_navsim.py)이 함께 쓰는 NAVSIM 전용 부분.

왜 따로 두나: sft_common.fold_select 는 (i) 자료를 data_dir/teacher_gt*_train.jsonl 한 파일에서 읽고 (ii) fold 별 표본 수가
분할 파일 n_samples 와 같아야 통과한다. NAVSIM SFT 자료(make_sft_navsim_scene.py 산출)는 fold{A,B}_{train,holdout}.jsonl 로
이미 나뉘어 있고, 교사 제외 6 로그(NuPlanQA-Eval)를 뺐기 때문에 (ii) 가 성립하지 않는다(make_sft_navsim_scene.py 설명).
그래서 (ii) 를 '건너뛰는' 대신 그보다 강한 검사로 대체한다(navsim_fold_select):
  - 분할 파일 = sft_common.load_fold_split(전체·부분 해시 검증), manifest 의 fold_sha256 와 같아야 함.
  - 4 파일 sha256 = manifest.files, 줄 수 = manifest.counts.
  - 제외 로그 목록 = 분할 파일 teacher_exclude_logs(sha256 기록), manifest.reasons.excluded_log 와 함께 검산:
    fold F 학습 로그 집합 == train.F − holdout.F − 제외 로그,  hold-out 로그 집합 == holdout.F − 제외 로그 (정확히 같음).
  - fold F 장면 프레임 수(제외 전) == 분할 파일 train.F.n_samples (manifest.fold_train_scene_rows_before_exclusion).
  - 모든 행: is_scene_frame=True, teacher_excluded_log=False, scene_token(=로그) 이 그 파일의 집합 안, token 중복 0.
  - 수정(2026-10-29): train.A ∩ train.B = ∅, train.A ∪ train.B = GT navtrain 전 로그, manifest source_sha256 = GT navtrain 해시.
값(nuScenes 와 다른 것만): 전방 3 캠(CAM_L0·F0·R0), 화소 128·28·28 → 224×420 → 캠당 120 토큰(QA 단계 run_qa_navsim.sh 와 같은 값).
"""
import hashlib
import json
import os
from pathlib import Path

NAVSIM_ROOT = "/home1/irteam/data-vol1/dl/navsim_v2/teacher"
NAVSIM_SFT_DIR = f"{NAVSIM_ROOT}/navsim_sft_scene_20261019"
NAVSIM_FOLD_FILE = f"{NAVSIM_ROOT}/navsim/fold_split_navsim.json"
NAVSIM_GT_DIR = f"{NAVSIM_ROOT}/navsim_gt_v2sensor_cb_20261019"          # SFT 자료의 원본(manifest.source 와 sha256 대조)
NAVSIM_CODEBOOK = f"{NAVSIM_ROOT}/navsim_codebook_20261019/navsim_action_codebook.npz"
NAVSIM_THRESHOLDS = f"{NAVSIM_ROOT}/navsim_codebook_20261019/navsim_intent_thresholds.json"
NAVSIM_NUM_CAMS = 3
NAVSIM_PIXELS = 128 * 28 * 28          # = 100352, run_qa_navsim.sh --pixels 와 같다
NAVSIM_TOKENS_PER_CAM = 120
NAVSIM_SPLIT_FILES = {"train": "teacher_gt_navtrain.jsonl", "val": "teacher_gt_navtest.jsonl"}   # 분할 파일의 train/val ↔ GT 파일


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path):
    out = []
    with open(path) as f:
        for ln in f:
            if ln.strip():
                out.append(json.loads(ln))
    return out


def load_manifest(sft_dir):
    p = Path(sft_dir) / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음 — make_sft_navsim_scene.py 산출 폴더가 아니다")
    return json.load(open(p)), str(p)


def exclude_logs(d):
    """분할 파일에 기록된 교사 제외 로그(NuPlanQA-Eval 6 로그) 집합."""
    ex = d.get("teacher_exclude_logs")
    if not isinstance(ex, dict) or "logs" not in ex:
        raise RuntimeError("분할 파일에 teacher_exclude_logs.logs 가 없다 — NAVSIM 분할 파일이 아니다")
    return set(ex["logs"])


def gt_navtrain_logs(gt_dir=NAVSIM_GT_DIR):  # 수정(2026-10-29): 검토 반영 — GT navtrain 의 sha256 과 로그 집합(한 번 읽기)
    """→ (sha256, 전 행 로그 set, 장면 프레임 로그 set, 경로). 2.2 GB 를 한 번 읽어 해시와 log_name 을 함께 얻는다(약 15 초)."""
    p = Path(gt_dir) / NAVSIM_SPLIT_FILES["train"]
    h, all_logs, scene_logs = hashlib.sha256(), set(), set()
    with open(p, "rb") as f:
        for ln in f:
            h.update(ln)
            if ln.strip():
                r = json.loads(ln)
                all_logs.add(r["log_name"])
                if r.get("is_scene_frame"):
                    scene_logs.add(r["log_name"])
    return h.hexdigest(), all_logs, scene_logs, str(p)


def navsim_fold_select(sft_dir, fold, fold_file, gt_dir=NAVSIM_GT_DIR):
    """NAVSIM SFT 자료 → (학습 표본, hold-out 표본, hold-out 로그 set, hold-out sha256, 분할 파일 경로, 분할 dict, 출처 dict).
    sft_common.fold_select 와 같은 반환 순서(+출처 dict). 어긋나면 RuntimeError(조용히 맞추지 않음)."""
    from sft_common import FOLDS, load_fold_split
    if fold not in FOLDS:
        raise ValueError(f"NAVSIM SFT 는 --fold A|B 만(받은 값 {fold})")
    d, fp = load_fold_split(fold_file)
    man, man_p = load_manifest(sft_dir)
    if man.get("fold_sha256") != d["sha256"]:
        raise RuntimeError(f"{man_p}: fold_sha256 {str(man.get('fold_sha256'))[:12]} != 분할 파일 {d['sha256'][:12]} ({fp})")
    ex = exclude_logs(d)
    # 수정(2026-10-29): 검토 반영 — (i) train.A ∩ train.B = ∅, train.A ∪ train.B = GT navtrain 의 모든 로그(전 행·장면 프레임 둘 다),
    #   (ii) SFT manifest 의 source(파일 이름)·source_sha256 = 지금 GT navtrain 파일 해시(SFT 자료가 이 GT 에서 만들어졌는지).
    fa, fb = set(d["train"]["A"]["scenes"]), set(d["train"]["B"]["scenes"])
    if fa & fb:
        raise RuntimeError(f"{fp}: train.A ∩ train.B 로그 {len(fa & fb)} 개(교차 적합 분할이 아니다)")
    gt_sha, gt_all, gt_scene, gt_p = gt_navtrain_logs(gt_dir)
    if (fa | fb) != gt_all or (fa | fb) != gt_scene:
        raise RuntimeError(f"{fp}: train.A ∪ train.B({len(fa | fb)}) ≠ GT navtrain 로그(전 행 {len(gt_all)}, 장면 프레임 {len(gt_scene)}) "
                           f"— 누락 {len(gt_all - (fa | fb))}, 초과 {len((fa | fb) - gt_all)} ({gt_p})")
    if not ex <= gt_all:
        raise RuntimeError(f"{fp}: 교사 제외 로그 {len(ex - gt_all)} 개가 GT navtrain 에 없다")
    if Path(str(man.get("source"))).name != NAVSIM_SPLIT_FILES["train"] or man.get("source_sha256") != gt_sha:
        raise RuntimeError(f"{man_p}: source {man.get('source')} / source_sha256 {str(man.get('source_sha256'))[:12]} ≠ GT {gt_p} {gt_sha[:12]} "
                           f"(SFT 자료가 지금 GT navtrain 에서 만들어진 것이 아니다)")
    mine, hold_all = set(d["train"][fold]["scenes"]), set(d["holdout"][fold]["scenes"])
    if not hold_all <= mine:
        raise RuntimeError(f"{fp}: holdout.{fold} 가 train.{fold} 밖의 로그를 포함")
    want = {"train": mine - hold_all - ex, "holdout": hold_all - ex}
    recs, files = {}, {}
    for part in ("train", "holdout"):
        name = f"fold{fold}_{part}"
        p = Path(sft_dir) / f"{name}.jsonl"
        sha = sha256_file(p)
        if sha != man["files"][name]:
            raise RuntimeError(f"{p}: sha256 {sha[:12]} != manifest {man['files'][name][:12]} (자료가 바뀌었다)")
        rs = read_jsonl(p)
        if len(rs) != man["counts"][name]:
            raise RuntimeError(f"{p}: 줄 {len(rs)} != manifest {man['counts'][name]}")
        logs = {r["scene_token"] for r in rs}
        if logs != want[part]:
            raise RuntimeError(f"{p}: 로그 집합이 분할 파일과 다르다(기대 {len(want[part])}, 실제 {len(logs)}, "
                               f"초과 {len(logs - want[part])}, 누락 {len(want[part] - logs)})")
        bad = [r["token"] for r in rs if not r.get("is_scene_frame") or r.get("teacher_excluded_log")
               or r.get("log_name") != r["scene_token"]]
        if bad:
            raise RuntimeError(f"{p}: 장면 프레임 아님·제외 로그·scene_token≠log_name 행 {len(bad)} (예 {bad[:3]})")
        recs[part], files[part] = rs, dict(path=str(p), sha256=sha, n=len(rs), n_logs=len(logs))
    toks = [r["token"] for r in recs["train"] + recs["holdout"]]
    if len(set(toks)) != len(toks):
        raise RuntimeError(f"fold {fold} 학습·hold-out 표본 token 중복 {len(toks) - len(set(toks))}")
    n_before = man.get("fold_train_scene_rows_before_exclusion", {}).get(fold)
    if n_before != d["train"][fold]["n_samples"]:
        raise RuntimeError(f"{man_p}: fold {fold} 제외 전 장면 프레임 {n_before} != 분할 파일 n_samples {d['train'][fold]['n_samples']}")
    info = dict(kind="navsim_sft_scene", sft_dir=str(Path(sft_dir).resolve()), manifest_sha256=sha256_file(man_p),
                source=man.get("source"), source_sha256=man.get("source_sha256"), files=files,
                gt_navtrain=dict(path=gt_p, sha256=gt_sha, n_logs=len(gt_all), fold_logs_disjoint_cover=True),  # 수정(2026-10-29): 검토 반영
                excluded_logs=sorted(ex & mine), n_excluded_rows_fold=n_before - len(recs["train"]) - len(recs["holdout"]),
                rule="train = train.F − holdout.F − teacher_exclude_logs, holdout = holdout.F − teacher_exclude_logs (장면 프레임만)")
    return recs["train"], recs["holdout"], hold_all, d["holdout"][fold]["sha256"], fp, d, info
