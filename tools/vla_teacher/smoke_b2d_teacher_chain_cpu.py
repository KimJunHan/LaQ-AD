# 수정(2026-10-31): B2D 교사 체인 — CPU 전용 스모크(실행: CUDA_VISIBLE_DEVICES=""). GPU·실제 교사 체크포인트를 쓰지 않는다.
"""train_sft_b2d.py → merge_teacher_lora.py → extract_teacher_cache_b2d.py → LoadTeacherCacheB2D 를 소형 무작위 Qwen2-VL 로 잇는다.
  1) 실행기 인자 거부(--fold none, 다른 --data_dir, --num_cams).
  2) 소형 QA base(실제 B2D QA fold A best 의 qa_meta 복사 + 무작위 소형 가중치) → SFT 2 step(CPU, 실제 B2D 정답·영상 12 장):
     latest/best/final 메타 run_id 같음·b2d 출처 기록, 같은 폴더 새 실행 거부, --resume(이미 total) = final 만 다시·run_id 유지.
  3) 병합(fp32, CPU) → 메타 run_id·b2d 유지.
  4) 캐시 추출 교사 A 샤드 0/2·1/2 각 1 표본(문법 제약 생성 547, layout_ok) → <folder>/<clip>/<kf>.pt, 재실행 = 건너뜀,
     --verify(부분 집합) OK, 학생 로더(LoadTeacherCacheB2D, config 의 meta_expect)로 키프레임+2 프레임 적재(나이 0.2 s).
  5) 학생 info 10 Hz 전 프레임 → GT 키프레임(전 배정 49,929 를 색인으로 둔 가짜 캐시 루트, config 로더 규칙) 사상 실패 0.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2d_sft_common as BS  # noqa: E402

RES = {}
PY = sys.executable


def ok(name, cond, info=""):
    RES[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    return cond


def run(args, expect_rc=0, tail=6):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(BS.REPO), OPENBLAS_CORETYPE="SkylakeX", TOKENIZERS_PARALLELISM="false")
    p = subprocess.run([PY] + [str(x) for x in args], cwd=str(BS.REPO), env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    if (p.returncode == 0) != (expect_rc == 0):
        print(out[-4000:], flush=True)
    return p.returncode, out


def make_tiny_qa_base(d):
    import torch
    import sft_common as S
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
    cfg = Qwen2VLConfig.from_pretrained(S.BASE_QWEN)
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 64, 128, 4
    cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
    cfg.rope_scaling = {"type": "mrope", "rope_type": "default", "mrope_section": [2, 3, 3]}
    vc = cfg.vision_config
    vc.depth, vc.embed_dim, vc.num_heads, vc.hidden_size, vc.mlp_ratio = 1, 32, 2, 64, 2
    cfg.tie_word_embeddings = True
    torch.manual_seed(0)
    m = Qwen2VLForConditionalGeneration(cfg).float()
    m.tie_weights()
    d.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(d)
    S.make_processor(S.BASE_QWEN, BS.B2D_PIXELS).save_pretrained(d)
    qm = json.load(open(BS.REPO / "results/vla_teacher/b2d/qa_full_foldA/best/qa_meta.json"))   # 실제 B2D QA fold A 출처 기록(읽기만)
    qm["smoke_note"] = "tiny random weights (smoke_b2d_teacher_chain_cpu.py) — qa_meta 는 실제 B2D QA fold A best 복사"
    json.dump(qm, open(d / "qa_meta.json", "w"), indent=1, ensure_ascii=False)


def main():
    work = Path(os.environ.get("SMOKE_DIR", "/tmp/claude-500/-home1-irteam/c9160d51-3404-4a92-b520-1bbc181ad11c/scratchpad/b2d_smoke"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    S_ = HERE / "train_sft_b2d.py"
    # 1) 인자 거부
    for bad in (["--fold", "none"], ["--fold", "A", "--data_dir", "data/vla_teacher/v2"], ["--fold", "A", "--num_cams", "3"]):
        rc, out = run([S_, "--dry_run", "--base", "x"] + bad, expect_rc=1)
        ok(f"wrapper_rejects {' '.join(bad)}", rc != 0 and ("[거부]" in out or "--navsim_sft_dir 와 함께" in out), out.strip().splitlines()[-1][:160])
    # 2) SFT
    base, sft = work / "tiny_qa_foldA", work / "b2d_sft_foldA_qa"
    make_tiny_qa_base(base)
    hp = ["--fold", "A", "--base", base, "--out", sft, "--device", "cpu", "--max_steps", "2", "--max_train_samples", "4",
          "--max_holdout_samples", "2", "--batch", "2", "--accum", "1", "--holdout_every", "1", "--save_every", "1",
          "--workers", "0", "--lora_r", "4", "--lora_alpha", "8"]
    rc, out = run([S_] + hp)
    ok("sft_cpu_2steps", rc == 0 and (sft / "final/teacher_meta.json").exists(), [l for l in out.splitlines() if l.startswith("[done]")][-1:])
    ms = {t: json.load(open(sft / t / "teacher_meta.json")) for t in ("latest", "best", "final")}
    rids = {t: m.get("run_id") for t, m in ms.items()}
    ok("sft_meta_run_id_same", len(set(rids.values())) == 1 and None not in rids.values(), rids)
    ok("sft_meta_b2d_record", all((m.get("b2d") or {}).get("kind") == "b2d_teacher_gt" and m["b2d"]["action_k"] == 2040 for m in ms.values()))
    ok("sft_meta_nuscenes_path_args", all("navsim_sft_dir" not in m["args"] and m["pixels"] == BS.B2D_PIXELS and m["use_prev"] for m in ms.values()))
    ok("sft_meta_fold", ms["final"]["fold"]["fold"] == "A" and ms["final"]["fold"]["generate"] == ["train.B", "val.B"])
    losses = [json.loads(l) for l in open(sft / "train_log.jsonl") if '"loss"' in l]
    import math
    ok("sft_loss_finite", losses and all(math.isfinite(r["loss"]) for r in losses), [round(r["loss"], 3) for r in losses])
    rc, out = run([S_] + hp, expect_rc=1)
    ok("sft_rejects_overwrite", rc != 0 and "[거부]" in out)
    rc, out = run([S_] + hp + ["--resume"])
    m2 = json.load(open(sft / "final/teacher_meta.json"))
    ok("sft_resume_done_keeps_run_id", rc == 0 and m2["run_id"] == rids["final"] and "학습 없이 final" in out)
    # 3) 병합
    merged = work / "b2d_merged_foldA_qa"
    rc, out = run([HERE / "merge_teacher_lora.py", "--adapter", sft / "best", "--out", merged, "--dtype", "fp32", "--device", "cpu"])
    mm = json.load(open(merged / "teacher_meta.json")) if rc == 0 else {}
    ok("merge_cpu_fp32", rc == 0 and mm.get("run_id") == rids["best"] and mm.get("merged_dtype") == "fp32" and "b2d" in mm)
    # 4) 캐시
    root = work / "bench2drive" / "laq_cache"
    X_ = HERE / "extract_teacher_cache_b2d.py"
    cb = ["--teacher", "A", "--model", merged, "--out_root", root, "--device", "cpu", "--dtype", "fp32", "--layer", "2",
          "--batch", "1", "--fwd_batch", "1", "--max_samples", "1", "--num_shards", "2"]
    for k in (0, 1):
        rc, out = run([X_] + cb + ["--shard", k])
        ok(f"cache_shard{k}of2", rc == 0 and f"DONE teacher_A shard {k}/2 이번 저장 1" in out,
           [l for l in out.splitlines() if l.startswith("DONE")][-1:][0][:300] if rc == 0 else out[-300:])
    rc, out = run([X_] + cb + ["--shard", "0"])
    ok("cache_resume_skips", rc == 0 and "완료 건너뜀 1 | 할 일 0" in out)
    idx = [json.loads(l) for p in sorted((root / "_index").glob("teacher_A.s*of2.jsonl")) for l in open(p)]
    ok("cache_index_flags", len(idx) == 2 and all(e["layout_ok"] and e["grammar_ok"] and e["parse_ok"] and e["length_ok"]
                                                  and e["constrained_decoding"] == "answer_grammar_v1" for e in idx),
       [(e["token"], e["part"], e["layout_ok"]) for e in idx])
    files = sorted(str(p.relative_to(root)) for p in root.rglob("*.pt"))
    ok("cache_file_layout", len(files) == 14 and all(len(Path(f).parts) == 3 for f in files), files[:2])
    meta_c = json.load(open(root / "context/_meta.json"))
    ok("cache_meta_keys", meta_c["prompt_mode"] == "B" and meta_c["teacher_chain_verified"] is True and meta_c["dataset"] == "b2d"
       and meta_c["constrained_decoding"] == "answer_grammar_v1" and meta_c["L"] == 1080 + 809, {k: meta_c[k] for k in ("L", "dataset")})
    for k in (0, 1):
        rc, out = run([X_, "--verify", "--teacher", "A", "--out_root", root, "--max_samples", "1", "--num_shards", "2", "--shard", k])
        ok(f"cache_verify_subset_s{k}", rc == 0 and "[verify] OK" in out, [l for l in out.splitlines() if "[verify]" in l][-1:])
    rc, out = run([X_, "--verify", "--out_root", root], expect_rc=1)
    ok("cache_verify_full_reports_missing", rc != 0 and "index_missing" in out)
    # 학생 로더(config 값, hidden_dim 만 소형 모델 64)
    sys.path.insert(0, str(BS.REPO))
    import extract_teacher_cache_b2d as E
    ld0, clip_of, kf_for, tl = E.student_loader(root)
    tl = dict(tl, hidden_dim=64)
    from projects.mmdet3d_plugin.models.encoder.vl_tokens_cache.loading_b2d import LoadTeacherCacheB2D
    ld = LoadTeacherCacheB2D(**tl)
    tok0 = idx[0]["token"]
    clip, kf = tok0.rsplit("/", 1)
    o = ld(dict(folder=f"v1/{clip}", frame_idx=int(kf) + 2, token="orig"))
    h = o["qwen_hidden"].data
    ok("student_loader_reads_cache", o["teacher_key"] == tok0 and abs(o["teacher_age"] - 0.2) < 1e-9 and o["token"] == "orig"
       and tuple(h.shape) == (1, sum(ld.expect_lens.values()), 64), (tuple(h.shape), o["teacher_age"]))
    # 5) 학생 info 프레임 → GT 키프레임 사상(가짜 루트: 전 배정 색인 + _meta)
    fake = work / "bench2drive" / "laq_cache_indexonly"
    exp = {}
    for T in ("A", "B"):
        its, _, _ = E.assigned(T, str(BS.B2D_FOLD_FILE))
        exp.update({r["token"]: T for _, r in its})
    (fake / "_index").mkdir(parents=True)
    with open(fake / "_index/teacher_AB.jsonl", "w") as f:
        for t, T in exp.items():
            f.write(json.dumps(dict(token=t, teacher=T, layout_ok=True, constrained_decoding="answer_grammar_v1")) + "\n")
    for k in tl["folders"]:
        (fake / k).mkdir()
        json.dump(dict(tl["meta_expect"][k], L=1), open(fake / k / "_meta.json", "w"))
    problems = {}
    n = E.verify_infos(fake, exp, problems, load_samples=False)
    ok("infos_all_frames_map_to_gt_keyframes", not problems and len(exp) == 49929, f"info 프레임 {n}, 배정 {len(exp)}, 문제 {list(problems)}")
    print("SMOKE", "ALL PASS" if all(RES.values()) else f"FAIL {[k for k, v in RES.items() if not v]}", flush=True)
    sys.exit(0 if all(RES.values()) else 1)


if __name__ == "__main__":
    main()
