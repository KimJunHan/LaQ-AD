# 수정(2026-10-16): B2D 교사 QA 준비 — B2D QA 파일을 train_qa_full.py 데이터 경로가 그대로 읽는지 CPU 시험(GPU 불필요).
#   실행: OPENBLAS_CORETYPE=SkylakeX CUDA_VISIBLE_DEVICES="" python tools/vla_teacher/smoke_qa_b2d_cpu.py
#   1) 대화 조립: 조각 이어 붙임 == processor.apply_chat_template(다중 턴)   (두 데이터셋 첫·최장 + chunk 표본 + val 표본)
#   2) QAFrameDataset(실제 B2D 6 캠 영상) == 전체 문자열 한 번에 토큰화, 영상 토큰 6×180·grid [1,20,36], 길이 == jsonl n_tokens
#   3) 손실 마스크 == 턴별 답 + <|im_end|>
#   4) train_qa_full.load_fold_data 가 --qa_dir/--fold_file(B2D)로 fold A·B 를 읽고 장면 재검사를 통과
#   5) 소형 무작위 Qwen2-VL 로 train_qa_full.main 을 B2D 파일로 CPU 2 step(hold-out 평가 포함) — 손실 유한
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import b2d_qa_common as B  # noqa: E402
import qa_common as Q  # noqa: E402
import sft_common as S  # noqa: E402
from smoke_qa_full_cpu import tiny_model_dir  # noqa: E402

RES = {}


def ok(name, cond, info=""):
    RES[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)


def first_and_longest(path, k=1):
    """jsonl 을 한 줄씩 읽어 데이터셋별 첫 표본·최장 표본 + chunk 표본 1 개(메모리 절약)."""
    first, longest, chunk = {}, {}, None
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ds = r["dataset"]
            first.setdefault(ds, r)
            if ds not in longest or r["n_tokens"] > longest[ds]["n_tokens"]:
                longest[ds] = r
            if chunk is None and r["n_chunks"] > 1:
                chunk = r
    return list(first.values()) + list(longest.values()) + ([chunk] if chunk else [])


def main():
    assert S.require_gpu_env(allow_cpu=True) == "cpu", "CUDA_VISIBLE_DEVICES='' 로 실행할 것"
    qa_dir = S.REPO / B.QA_DIR_B2D
    proc = S.make_processor(S.BASE_QWEN)
    tok = proc.tokenizer
    pick = first_and_longest(qa_dir / "foldA_train.jsonl") + first_and_longest(qa_dir / "val.jsonl")[:1]
    ok("picked_both_datasets", {r["dataset"] for r in pick} == set(B.DATASETS_B2D), [r["id"] for r in pick])
    ok("chunk_sample_present", any(r["n_chunks"] > 1 for r in pick))

    good = all("".join(t for t, _ in Q.conversation_pieces(r)) ==
               proc.apply_chat_template(Q.chat_messages(r), tokenize=False, add_generation_prompt=False) for r in pick)
    ok("pieces_equal_chat_template", good, f"{len(pick)} 표본")

    from PIL import Image
    ds = Q.QAFrameDataset(pick, proc)
    g_whole, g_len, g_mask, g_img = True, True, True, True
    for i, r in enumerate(pick):
        it = ds[i]
        full = "".join(t for t, _ in Q.conversation_pieces(r))
        enc = proc(text=[full], images=[Image.open(p).convert("RGB") for p in Q.cam_paths(r)], return_tensors="pt")
        g_whole &= enc["input_ids"][0].tolist() == it["input_ids"].tolist()
        g_img &= int((it["input_ids"] == Q.IMAGE_PAD_ID).sum()) == 6 * S.TOKENS_PER_CAM and \
            it["image_grid_thw"].tolist() == [[1, 20, 36]] * 6
        g_len &= len(it["input_ids"]) == r["n_tokens"]
        lab = it["labels"].tolist()
        spans, cur = [], []
        for x in lab:
            if x != -100:
                cur.append(x)
            elif cur:
                spans.append(cur); cur = []
        if cur:
            spans.append(cur)
        g_mask &= [tok.decode(s) for s in spans] == [t["a"] + "<|im_end|>" for t in r["turns"]]
    ok("tokenize_pieces_equal_whole", g_whole, "조각별 == 한 번에(B2D 영상 6 장 포함)")
    ok("image_tokens_6x180_grid_1x20x36", g_img, "B2D 1600×900 → nuScenes 와 같은 280×504 / 180 토큰")
    ok("length_matches_jsonl", g_len)
    ok("loss_mask_answers_only", g_mask)

    import train_qa_full as T
    for f in S.FOLDS:
        a = T.parse_args(["--fold", f, "--qa_dir", B.QA_DIR_B2D, "--fold_file", B.FOLD_FILE_B2D])
        tr, ho, fd, fpath, man = T.load_fold_data(a)
        ok(f"load_fold_data_{f}", len(tr) > 0 and len(ho) > 0 and fpath.endswith("fold_split_b2d.json"),
           f"학습 {len(tr)} / hold-out {len(ho)} 표본, 분할 {fd['sha256'][:12]}")
        del tr, ho

    tmp = Path(tempfile.mkdtemp(prefix="qa_b2d_smoke_"))
    try:
        base = tiny_model_dir(str(tmp / "tiny"), proc)
        T.main(["--fold", "A", "--base", base, "--device", "cpu", "--test_subset", "4", "--batch", "2", "--accum", "1",
                "--workers", "0", "--holdout_every", "2", "--save_every", "2", "--lr", "1e-3", "--lr_projector", "1e-3",
                "--warmup", "0", "--max_steps", "2", "--qa_dir", B.QA_DIR_B2D, "--fold_file", B.FOLD_FILE_B2D,
                "--out", str(tmp / "run")])
        log = [json.loads(l) for l in open(tmp / "run" / "train_log.jsonl")]
        st = [x for x in log if "loss" in x]
        ok("train_steps_finite", len(st) == 2 and all(torch.isfinite(torch.tensor(x["loss"])) for x in st),
           [round(x["loss"], 3) for x in st])
        ok("holdout_eval_ran", any(x.get("kind") == "holdout" for x in log))
        meta = json.load(open(tmp / "run" / "final" / "qa_meta.json"))
        ok("meta_points_to_b2d_fold", meta["fold_file"].endswith("fold_split_b2d.json"), meta["fold_file"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bad = [k for k, v in RES.items() if not v]
    print("ALL PASS" if not bad else f"FAIL {bad}", flush=True)
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
