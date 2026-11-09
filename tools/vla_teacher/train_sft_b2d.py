# 수정(2026-10-31): B2D 교사 체인 — Bench2Drive 교사 LoRA SFT 실행기(새 파일). train_sft.py·sft_common.py 는 고치지 않는다.
"""B2D 교사 SFT = train_sft.py 의 nuScenes 경로(--data_dir + --fold A|B + sft_common.fold_select) 그대로 + B2D 출처 검사·저장 안전.

왜 train_sft.py 를 고치지 않나: 지금 NAVSIM SFT(GPU3)가 train_sft.py 를 쓰고 있고, NAVSIM 체인의 출처 검사(sft_prov)가 끝난 뒤
  train_sft.parse_args·resume_arg_diff 를 다시 부른다. 인자를 하나라도 더하면 NAVSIM 메타 args 와 키가 달라져 그 검사가 멈춘다.
  B2D 는 정답 형식(547)·영상(6 캠×2·200·28·28)이 nuScenes 와 같아 train_sft.py 의 nuScenes 경로를 학습 수식 그대로 쓸 수 있다
  (CPU dry run: fold A 22,028 / hold-out 1,648, fold B 22,253 / 1,417, 정답 548 토큰 고정).
이 실행기가 더하는 것(NAVSIM 경로가 train_sft.py 안에서 하던 것과 같은 규칙):
  1) 인자 고정: --fold A|B, --data_dir = B2D 정답 폴더, --fold_file = fold_split_b2d_kf2hz.json, --navsim_sft_dir·--num_cams·
     --pixels 금지(기본 6·200·28·28). 다른 값이면 거부.
  2) 출처: b2d_sft_common.check_b2d_sources(GT sha256 = 분할 파일 derived_from, special_tokens = common_b2d, check_report PASS),
     check_b2d_qa_base(QA best = 같은 fold·분할 해시·pixels 156800·cams 'current 6'·QA 자료 qa_kf2hz·실제 사용 장면 ⊆ train.F).
  3) 저장 안전: --resume 이 아니고 출력 폴더에 이전 산출물이 있으면 거부(덮어써 기록이 섞이지 않게). --resume 이면 latest 메타에
     run_id·b2d 기록이 있어야 한다(이 실행기로 시작한 B2D SFT 의 latest).
  4) 메타 기록: 모든 teacher_meta.json(출력 루트·latest·best·final)에 run_id(한 실행 식별자 — 재개해도 같음)와 b2d(출처 dict)를
     더한다(sft_common.save_meta 를 이 프로세스 안에서만 감싼다). 체인이 final·best 가 한 실행인지 확인하고, 병합(merge_teacher_lora)
     ·캐시 추출(extract_teacher_cache_b2d)이 그대로 물려받는다. trainer_state.pt 의 run_id 는 train_sft 내부 값(별개, 재개 일관성용).
학습 하이퍼파라미터·손실·저장 순서·재개 규칙은 train_sft.py 그대로(체인 SFT_HP 에서 nuScenes 교사와 같은 값을 명시).
사용(GPU2 = B2D 전용):
  CUDA_VISIBLE_DEVICES=2 python tools/vla_teacher/train_sft_b2d.py --fold A --base results/vla_teacher/b2d/qa_full_foldA/best \
      --out results/vla_teacher/b2d/b2d_sft_foldA_qa <SFT_HP> [--resume]
  CUDA_VISIBLE_DEVICES="" python tools/vla_teacher/train_sft_b2d.py --dry_run --fold A --base ... --out ...
"""
import json
import os
import sys
import uuid
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b2d_sft_common as BS  # noqa: E402
import sft_common as S  # noqa: E402
import train_sft as T  # noqa: E402

TAG = "수정(2026-10-31): B2D 교사 체인"
_PREV_OUTPUTS = ("latest", "latest.tmp", "latest.old", "best", "best.tmp", "best.old", "final", "final.tmp")


def b2d_argv(argv):
    """B2D 기본 경로(--data_dir·--fold_file)를 채운 인자 목록. 사용자가 준 값은 그대로 두고 parse 뒤에 검사한다."""
    argv = list(argv)
    if "--data_dir" not in argv:
        argv += ["--data_dir", str(BS.B2D_GT_DIR)]
    if "--fold_file" not in argv:
        argv += ["--fold_file", str(BS.B2D_FOLD_FILE)]
    return argv


def parse_args(argv=None):
    """train_sft.parse_args(B2D 기본 채움) + B2D 고정값 검사. 체인의 출처 검사도 이 함수로 현재 인자를 만든다."""
    argv = b2d_argv(sys.argv[1:] if argv is None else argv)
    a = T.parse_args(argv)
    bad = []
    if a.fold not in ("A", "B"):
        bad.append(f"--fold {a.fold} (B2D 교사는 A|B 교차 적합만)")
    if a.navsim_sft_dir is not None:
        bad.append("--navsim_sft_dir 금지")
    if a.num_cams != BS.B2D_NUM_CAMS or a.pixels != BS.B2D_PIXELS:
        bad.append(f"num_cams {a.num_cams}·pixels {a.pixels} != {BS.B2D_NUM_CAMS}·{BS.B2D_PIXELS}")
    dd = a.data_dir if os.path.isabs(a.data_dir) else S.REPO / a.data_dir
    ff = a.fold_file if os.path.isabs(a.fold_file) else S.REPO / a.fold_file
    if not BS.same_path(dd, BS.B2D_GT_DIR):
        bad.append(f"--data_dir {a.data_dir} != {BS.B2D_GT_DIR}")
    if not BS.same_path(ff, BS.B2D_FOLD_FILE):
        bad.append(f"--fold_file {a.fold_file} != {BS.B2D_FOLD_FILE}")
    if a.special_tokens is not None:
        bad.append("--special_tokens 금지(정답 폴더의 special_tokens.txt 만)")
    if a.prev == "off":
        bad.append("--prev off 금지(설계 = 직전·현재 12 장)")
    if bad:
        raise SystemExit("[거부] B2D SFT 인자: " + "; ".join(bad))
    return a, argv


def check_out_state(a, out):
    """train_sft.check_resume_state 의 NAVSIM 전용 규칙을 B2D 에도(그 함수는 nuScenes 경로에서 이 둘을 하지 않는다)."""
    if a.resume:
        mp = out / "latest" / "teacher_meta.json"
        if mp.exists():
            lm = json.load(open(mp))
            if not lm.get("run_id") or (lm.get("b2d") or {}).get("kind") != "b2d_teacher_gt":
                raise SystemExit(f"[거부] {mp} 에 run_id·b2d 기록 없음 — 이 실행기로 시작한 B2D SFT 의 latest 가 아니다")
            return lm["run_id"]
        return None   # 나머지(latest 없음·교체 중 중단 등)는 train_sft.check_resume_state 가 거부·안내한다
    prev = [n for n in _PREV_OUTPUTS if (out / n).exists()]
    if prev:
        raise SystemExit(f"[거부] {out} 에 이전 학습 산출물 {prev} — latest 가 없고 latest.tmp/latest.old 만 있으면 그것을 latest 로 "
                         f"옮긴 뒤 --resume, 아니면 --resume 으로 이어 가거나 폴더를 옮긴 뒤 새로 시작")
    return None


def main(argv=None):
    a, full_argv = parse_args(argv)
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    try:
        src = BS.check_b2d_sources(a.data_dir if os.path.isabs(a.data_dir) else S.REPO / a.data_dir, a.fold_file)
        qm = BS.check_b2d_qa_base(a.base, a.fold, a.fold_file)
    except RuntimeError as e:
        raise SystemExit(f"[거부] B2D 출처 검사: {e}")
    b2d = dict(src, tag=TAG, qa_base_meta_step=qm.get("step"), qa_base_final=bool(qm.get("final")))
    print(f"[b2d] 출처 OK: GT train {src['gt_sha256']['train'][:12]} val {src['gt_sha256']['val'][:12]}, 분할 {src['fold_sha256'][:12]}, "
          f"K={src['action_k']}, QA base step {qm.get('step')}", flush=True)
    if a.dry_run:
        return T.main(full_argv)
    out = Path(a.out) if os.path.isabs(a.out) else S.REPO / a.out
    rid = check_out_state(a, out) or uuid.uuid4().hex
    orig = S.save_meta

    def save_meta_b2d(path, **kw):   # 이 프로세스의 train_sft 메타 저장에만 run_id·b2d 를 더한다
        if kw.get("run_id") not in (None, rid):
            raise RuntimeError(f"메타 run_id {kw.get('run_id')} != B2D 실행 {rid}")
        kw = dict(kw, run_id=rid, b2d=b2d)
        return orig(path, **kw)

    S.save_meta = save_meta_b2d
    try:
        print(f"[b2d] run_id {rid} ({'재개' if a.resume else '새 실행'})", flush=True)
        return T.main(full_argv)
    finally:
        S.save_meta = orig


if __name__ == "__main__":
    main()
