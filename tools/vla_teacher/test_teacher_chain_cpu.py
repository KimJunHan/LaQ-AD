# 수정(2026-10-16): 설계 검토 반영 — 교사 단계 연결(QA 전체 학습 → SFT → GRPO → 병합 → 캐시 추출) fold·분할 해시 검사 CPU 시험.
#   실행(저장소 루트): CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX python tools/vla_teacher/test_teacher_chain_cpu.py
#   (1) sft_common.base_chain_info: 원본 Qwen 허용, qa_meta 없는 다른 경로 거부, QA 모델 fold·해시·stage·QA base 검사
#   (2) sft_common.verify_teacher_chain: SFT·GRPO(→ 병합) meta 의 전 단계 검사, base 교체·SFT 체크포인트 덮어쓰기 감지
#   (3) 진입점: train_sft --dry_run / train_grpo --dry_run / extract_teacher_cache_v2.run 이 어긋난 연결에서 시작 전에 멈춘다
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sft_common as S  # noqa: E402

RES = {}


def ok(name, cond, info=""):
    RES[name] = dict(ok=bool(cond), info=str(info))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)


def expect_exit(name, fn, must=None):
    try:
        fn()
    except SystemExit as e:
        msg = str(e)
        ok(name, must is None or must in msg, msg[:110])
        return
    except Exception as e:  # 다른 예외 = 시험 실패(종료 대신 다른 경로로 깨짐)
        ok(name, False, f"SystemExit 기대, 실제 {type(e).__name__}: {e}")
        return
    ok(name, False, "멈추지 않았다")


def write_json(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(p, "w"), indent=1)


def qa_dir(root, name, **kw):
    d = Path(root) / name
    meta = dict(stage="teacher_qa_full", fold="A", base=S.BASE_QWEN)
    meta.update(kw)
    write_json(d / S.QA_META, meta)
    return str(d)


def main():
    assert S.require_gpu_env(allow_cpu=True) == "cpu", "CUDA_VISIBLE_DEVICES='' 로 실행할 것"
    fd, _ = S.load_fold_split()
    sha = fd["sha256"]
    tmp = tempfile.mkdtemp(prefix="teacher_chain_")

    print("(1) base_chain_info", flush=True)
    ok("original_qwen_ok", S.base_chain_info(S.BASE_QWEN, "A")["kind"] == "original_qwen")
    ok("original_qwen_fold_none_ok", S.base_chain_info(S.BASE_QWEN, "none")["kind"] == "original_qwen")
    os.makedirs(Path(tmp) / "random_model")
    expect_exit("unknown_base_without_meta_refused", lambda: S.base_chain_info(str(Path(tmp) / "random_model"), "A"), "출처 미상")
    qa_a = qa_dir(tmp, "qa_A", fold_sha256=sha)
    bc = S.base_chain_info(qa_a, "A")
    ok("qa_base_matching_ok", bc["kind"] == "qa_full" and bc["fold"] == "A" and bc["fold_sha256"] == sha and len(bc["qa_meta_sha256"]) == 64, bc)
    qa_alias = qa_dir(tmp, "qa_alias", fold_split_sha256=sha)
    ok("qa_meta_key_fold_split_sha256_accepted", S.base_chain_info(qa_alias, "A")["kind"] == "qa_full")
    qa_b = qa_dir(tmp, "qa_B", fold="B", fold_sha256=sha)
    expect_exit("qa_other_fold_refused", lambda: S.base_chain_info(qa_b, "A"), "fold=B")
    qa_sha = qa_dir(tmp, "qa_badsha", fold_sha256="0" * 64)
    expect_exit("qa_wrong_split_sha_refused", lambda: S.base_chain_info(qa_sha, "A"), "분할 해시")
    expect_exit("qa_with_fold_none_refused", lambda: S.base_chain_info(qa_a, "none"), "--fold none")
    qa_stage = qa_dir(tmp, "qa_stage", stage="something_else", fold_sha256=sha)
    expect_exit("qa_wrong_stage_refused", lambda: S.base_chain_info(qa_stage, "A"), "stage")
    qa_base = qa_dir(tmp, "qa_base", fold_sha256=sha, base=str(Path(tmp) / "random_model"))
    expect_exit("qa_non_original_base_refused", lambda: S.base_chain_info(qa_base, "A"), "원본 Qwen")

    print("(2) verify_teacher_chain", flush=True)
    sft = dict(base=qa_a, fold=dict(fold="A", fold_sha256=sha), base_chain=S.base_chain_info(qa_a, "A"))
    ch = S.verify_teacher_chain(sft, "A")
    ok("sft_chain_ok", ch["qa"]["kind"] == "qa_full" and ch["grpo"] is None and ch["fold_sha256"] == sha)
    expect_exit("sft_chain_other_fold_refused", lambda: S.verify_teacher_chain(sft, "B"), "fold=A")
    expect_exit("sft_chain_wrong_sha_refused",
                lambda: S.verify_teacher_chain(dict(sft, fold=dict(fold="A", fold_sha256="1" * 64)), "A"), "분할 해시")
    expect_exit("sft_meta_without_base_chain_but_qa_base_refused",
                lambda: S.verify_teacher_chain({k: v for k, v in sft.items() if k != "base_chain"}, "A"), "base_chain")
    legacy = json.load(open(S.REPO / "results/vla_teacher/sft_foldA/teacher_meta.json"))
    ok("legacy_sft_meta_original_qwen_ok", "base_chain" not in legacy and
       S.verify_teacher_chain(legacy, "A")["qa"]["kind"] == "original_qwen", "기존 results/vla_teacher/sft_foldA (base = 원본 Qwen)")
    # GRPO(→ 병합) meta: SFT meta 사본 + grpo.sft_ckpt
    sft_ck = Path(tmp) / "sft_A" / "best"
    write_json(sft_ck / "teacher_meta.json", sft)
    grpo = dict(sft, grpo=dict(sft_ckpt=str(sft_ck), sft_meta_sha256=S.file_sha256(sft_ck / "teacher_meta.json")))
    ch = S.verify_teacher_chain(grpo, "A")
    ok("grpo_chain_ok", ch["grpo"] is not None and ch["sft"]["teacher_meta_sha256"] == grpo["grpo"]["sft_meta_sha256"], ch["sft"])
    merged = dict(grpo, merged_from="x", merged_dtype="bf16")       # merge_teacher_lora.py 가 meta 를 그대로 복사
    ok("merged_grpo_chain_ok", S.verify_teacher_chain(merged, "A")["grpo"] is not None)
    # QA 모델 meta 가 SFT 뒤에 바뀜(다른 QA 모델로 교체) → base_chain 해시 불일치
    qm = Path(qa_a) / S.QA_META
    keep = qm.read_text()
    write_json(qm, dict(json.loads(keep), note="replaced"))
    expect_exit("qa_model_replaced_after_sft_refused", lambda: S.verify_teacher_chain(sft, "A"), "base_chain")
    qm.write_text(keep)
    # SFT 체크포인트 meta 가 GRPO 뒤에 바뀜(best 덮어쓰기)
    write_json(sft_ck / "teacher_meta.json", dict(sft, step=999))
    expect_exit("sft_ckpt_overwritten_after_grpo_refused", lambda: S.verify_teacher_chain(grpo, "A"), "바뀌었다")
    write_json(sft_ck / "teacher_meta.json", dict(sft, fold=dict(fold="B", fold_sha256=sha)))
    expect_exit("grpo_sft_ckpt_other_fold_refused",
                lambda: S.verify_teacher_chain(dict(grpo, grpo=dict(sft_ckpt=str(sft_ck))), "A"), "fold=B")
    expect_exit("grpo_sft_ckpt_missing_refused",
                lambda: S.verify_teacher_chain(dict(grpo, grpo=dict(sft_ckpt=str(Path(tmp) / "nope"))), "A"), "teacher_meta.json")
    write_json(sft_ck / "teacher_meta.json", sft)

    print("(3) 진입점", flush=True)
    import train_sft as TS
    expect_exit("train_sft_dry_run_qa_other_fold_refused",
                lambda: TS.main(["--fold", "A", "--base", qa_b, "--dry_run"]), "fold=B")
    expect_exit("train_sft_dry_run_qa_with_fold_none_refused",
                lambda: TS.main(["--fold", "none", "--base", qa_a, "--dry_run"]), "--fold none")
    expect_exit("train_sft_dry_run_unknown_base_refused",
                lambda: TS.main(["--fold", "A", "--base", str(Path(tmp) / "random_model"), "--dry_run"]), "출처 미상")
    import train_grpo as TG
    bad_ck = Path(tmp) / "sft_B_meta" / "best"
    write_json(bad_ck / "teacher_meta.json", dict(sft, fold=dict(fold="B", fold_sha256=sha)))
    (bad_ck / "adapter_model.safetensors").write_bytes(b"")
    expect_exit("train_grpo_dry_run_other_fold_sft_refused",
                lambda: TG.main(["--fold", "A", "--sft_ckpt", str(bad_ck), "--dry_run", "--out", str(Path(tmp) / "g")]), "fold=B")
    nb_ck = Path(tmp) / "sft_nobase" / "best"
    write_json(nb_ck / "teacher_meta.json", {k: v for k, v in sft.items() if k != "base_chain"})
    (nb_ck / "adapter_model.safetensors").write_bytes(b"")
    expect_exit("train_grpo_dry_run_unrecorded_qa_base_refused",
                lambda: TG.main(["--fold", "A", "--sft_ckpt", str(nb_ck), "--dry_run", "--out", str(Path(tmp) / "g")]), "base_chain")
    import extract_teacher_cache_v2 as X
    mdl = Path(tmp) / "merged_B"
    write_json(mdl / "teacher_meta.json", dict(sft, fold=dict(fold="B", fold_sha256=sha)))
    a = argparse.Namespace(model=str(mdl), teacher="A", device="cpu", fold_file=S.FOLD_FILE, data_dir=None)
    expect_exit("extract_other_fold_teacher_refused", lambda: X.run(a, Path(tmp) / "cache"), "fold=B")
    write_json(mdl / "teacher_meta.json", grpo)
    (sft_ck / "teacher_meta.json").write_text(json.dumps(dict(sft, step=5)))
    a = argparse.Namespace(model=str(mdl), teacher="A", device="cpu", fold_file=S.FOLD_FILE, data_dir=None)
    expect_exit("extract_grpo_with_overwritten_sft_refused", lambda: X.run(a, Path(tmp) / "cache"), "바뀌었다")
    ok("extract_wrote_nothing", not (Path(tmp) / "cache").exists(), "검사 실패 시 출력 폴더를 만들지 않음")

    n_fail = sum(not v["ok"] for v in RES.values())
    print(f"\n== {len(RES) - n_fail}/{len(RES)} 통과 ==", flush=True)
    json.dump(RES, open(S.REPO / "results/vla_teacher/test_teacher_chain_cpu.json", "w"), indent=1, ensure_ascii=False)
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
