# 수정(2026-10-31): B2D 교사 체인 — Bench2Drive 교사 SFT·문법·캐시가 함께 쓰는 B2D 전용 상수·출처 검사(새 파일, nuScenes/NAVSIM 경로 무변경).
"""B2D 교사 체인(QA → SFT → 병합 → 캐시) 공용 부분. NAVSIM 의 navsim_sft_common.py 에 해당한다.

nuScenes 와 같은 것(그래서 nuScenes 코드를 그대로 쓴다):
  - 정답 = v2-sensor-b2d, 줄·필드·토큰 수 547 이 nuScenes v2-sensor 와 같다(common_b2d.py 설계) → laq_cache_layout·answer_grammar·
    extract_teacher_cache_v2 의 도우미를 b2d_teacher_shim(common_v2 자리에 common_b2d)으로 그대로 쓴다.
  - 영상 = 직전 키프레임(0.5 s) 6 캠 + 현재 6 캠 = 12 장, 화소 200·28·28(= sft_common.PIXELS, 캠당 180 토큰) — B2D QA 단계
    (qa_meta pixels 156800·cams 'current 6')와 같다.
  - SFT 자료 형식 = train_sft.py 의 nuScenes 경로(--data_dir 의 teacher_gt*_train.jsonl + sft_common.fold_select): 분할 파일
    fold_split_b2d_kf2hz.json 이 표본 수를 이 정답(2 Hz 키프레임) 기준으로 센 판이라 fold_select 의 '장면별 표본 수 일치' 검사가
    그대로 성립한다(NAVSIM 처럼 검사를 대체할 필요 없음 — CPU dry run 으로 확인).
B2D 에서 다른 것(여기서 정함):
  - 경로: 정답 data/vla_teacher/b2d/gt_v2_sensor_b2d, 분할 fold_split_b2d_kf2hz.json, 학생 캐시 data/bench2drive/laq_cache
    (configs/b2d/v2_teacher/stage1/train.py teacher_cache_root), 파일 = <folder>/<clip>/<kf:05d>.pt (token 에 '/' 포함).
  - 행동 코드북 K = stats.json action_K(2040), TTC 간격 = stats.json ttc_gap_m — shim 설치 때 common_b2d 에 넣는다.
  - 출처 검사(check_b2d_sources): GT 두 파일 sha256 = 분할 파일 derived_from.gt_{train,val}_sha256(분할이 이 정답으로 세어졌는지),
    special_tokens.txt = common_b2d.SPECIAL_TOKENS(K 설정 뒤), GT check_report.json 의 fail = [].
"""
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
B2D_TEACHER_ROOT = REPO / "data/vla_teacher/b2d"
B2D_GT_DIR = B2D_TEACHER_ROOT / "gt_v2_sensor_b2d"
B2D_FOLD_FILE = B2D_TEACHER_ROOT / "fold_split_b2d_kf2hz.json"
B2D_QA_DIR = B2D_TEACHER_ROOT / "qa_kf2hz"
B2D_GRAMMAR_FILE = B2D_TEACHER_ROOT / "grammar_b2d_20261031/answer_position_mask_b2d.json"
B2D_CACHE_ROOT = "/home1/irteam/data-vol1/src/HiP-AD/data/bench2drive/laq_cache"   # = HiP-AD_refactor/data/bench2drive/laq_cache(링크)
B2D_INFOS = {"train": REPO / "data/infos/b2d_infos_train.pkl", "val": REPO / "data/infos/b2d_infos_val.pkl"}
B2D_NUM_CAMS = 6
B2D_PIXELS = 200 * 28 * 28          # = sft_common.PIXELS = B2D QA qa_meta.pixels(156800)
B2D_QA_CAMS = "current 6"
B2D_SPLIT_FILES = {"train": "teacher_gt_b2d_train.jsonl", "val": "teacher_gt_b2d_val.jsonl"}
KEYFRAME_STRIDE = 5                 # 10 Hz → 2 Hz (학생 로더 LoadTeacherCacheB2D 와 같은 값)


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def same_path(a, b):
    return os.path.realpath(str(a)) == os.path.realpath(str(b))


def gt_stats(gt_dir=B2D_GT_DIR):
    st = json.load(open(Path(gt_dir) / "stats.json"))
    if st.get("answer_format") != "v2-sensor-b2d" or not isinstance(st.get("action_K"), int) or st.get("ttc_gap_m") is None:
        raise RuntimeError(f"{gt_dir}/stats.json: answer_format {st.get('answer_format')} / action_K {st.get('action_K')} / "
                           f"ttc_gap_m {st.get('ttc_gap_m')} — B2D v2-sensor-b2d 정답 폴더가 아니다")
    return st


_SHIM = None


def install_shim(gt_dir=B2D_GT_DIR):
    """b2d_teacher_shim.install(K) + TTC 간격. answer_grammar·laq_cache_layout·extract_teacher_cache_v2 를 import 하기 '전에' 부른다
    (이미 nuScenes common_v2 로 import 됐으면 shim 이 실패시킨다 — 조용한 혼용 금지). 반환 common_b2d 모듈."""
    global _SHIM
    if _SHIM is not None:
        return _SHIM
    import b2d_teacher_shim as SH
    st = gt_stats(gt_dir)
    B = SH.install(st["action_K"])
    B.set_ttc_gap(st["ttc_gap_m"])
    _SHIM = B
    return B


def check_b2d_sources(gt_dir=B2D_GT_DIR, fold_file=B2D_FOLD_FILE, special_tokens=None):
    """B2D 정답·분할·특수토큰 출처 검사. 어긋나면 RuntimeError. 반환 기록용 dict.
    special_tokens(목록)를 주면 그것도 special_tokens.txt 와 같아야 한다(교사 meta 대조용)."""
    import sft_common as S
    gt_dir = Path(gt_dir)
    d, fp = S.load_fold_split(str(fold_file))
    df = d.get("derived_from") or {}
    sha = {k: sha256_file(gt_dir / f) for k, f in B2D_SPLIT_FILES.items()}
    if df.get("gt_train_sha256") != sha["train"] or df.get("gt_val_sha256") != sha["val"]:
        raise RuntimeError(f"{fp}: derived_from GT sha256 (train {str(df.get('gt_train_sha256'))[:12]}, val {str(df.get('gt_val_sha256'))[:12]}) "
                           f"!= 지금 GT (train {sha['train'][:12]}, val {sha['val'][:12]}) — 분할이 이 정답으로 세어진 것이 아니다")
    if "2 Hz keyframe" not in str(d.get("sample_unit")):
        raise RuntimeError(f"{fp}: sample_unit {d.get('sample_unit')!r} — 2 Hz 키프레임 분할(fold_split_b2d_kf2hz)이 아니다")
    rep = json.load(open(gt_dir / "check_report.json"))
    if rep.get("fail") != [] or rep.get("train", {}).get("n") is None:
        raise RuntimeError(f"{gt_dir}/check_report.json: fail={rep.get('fail')} — check_teacher_gt_b2d.py 가 통과한 정답이 아니다")
    B = install_shim(gt_dir)
    toks, _ = S.load_special_tokens(str(gt_dir / "special_tokens.txt"))
    if toks != list(B.SPECIAL_TOKENS):
        raise RuntimeError(f"{gt_dir}/special_tokens.txt 가 common_b2d.SPECIAL_TOKENS(K={B.ACTION_K}) 와 다르다")
    if special_tokens is not None and list(special_tokens) != toks:
        raise RuntimeError("교사 meta 특수토큰 목록 ≠ B2D special_tokens.txt")
    return dict(kind="b2d_teacher_gt", gt_dir=str(gt_dir.resolve()), gt_sha256=sha, fold_file=fp, fold_sha256=d["sha256"],
                special_tokens_sha256=sha256_file(gt_dir / "special_tokens.txt"), action_k=B.ACTION_K,
                ttc_gap_m=B.TTC_GAP, check_report_sha256=sha256_file(gt_dir / "check_report.json"))


def check_b2d_qa_base(base, fold, fold_file=B2D_FOLD_FILE):
    """--base 가 B2D QA 전체 학습 체크포인트(같은 fold·분할·영상 설정)인지. 어긋나면 RuntimeError. 반환 qa_meta."""
    import sft_common as S
    qm_p = Path(base) / "qa_meta.json"
    if not qm_p.exists():
        raise RuntimeError(f"{qm_p} 없음 — B2D SFT 의 --base 는 B2D QA 전체 학습 체크포인트여야 한다")
    qm = json.load(open(qm_p))
    d, _ = S.load_fold_split(str(fold_file))
    bad = []
    if qm.get("stage") != "teacher_qa_full" or qm.get("fold") != fold:
        bad.append(f"stage {qm.get('stage')} / fold {qm.get('fold')} (기대 teacher_qa_full / {fold})")
    if qm.get("fold_sha256") != d["sha256"]:
        bad.append(f"fold_sha256 {str(qm.get('fold_sha256'))[:12]} != 분할 {d['sha256'][:12]}")
    if qm.get("pixels") != B2D_PIXELS or qm.get("cams") != B2D_QA_CAMS:
        bad.append(f"pixels {qm.get('pixels')} / cams {qm.get('cams')!r} (기대 {B2D_PIXELS} / {B2D_QA_CAMS!r})")
    qd = (qm.get("args") or {}).get("qa_dir")
    if qd is None or not same_path(qd if os.path.isabs(qd) else REPO / qd, B2D_QA_DIR):
        bad.append(f"QA 자료 {qd} != {B2D_QA_DIR}")
    ss = qm.get("scene_sets") or {}
    mine = set(d["train"][fold]["scenes"])
    used = set(ss.get("train_scenes_used", [])) | set(ss.get("holdout_scenes_used", []))
    if "train_scenes_used" not in ss or not used <= mine:
        bad.append(f"QA 실제 사용 장면이 train.{fold} 밖 {len(used - mine)} 개(또는 기록 없음)")
    if bad:
        raise RuntimeError(f"{qm_p}: B2D QA base 검사 실패 — " + "; ".join(bad))
    return qm
