# 수정(2026-10-16): 교사 QA — NAVSIM 트랙 CPU 시험(GPU 불필요). smoke_qa_full_cpu.py 의 NAVSIM 판(전방 3캠·128*28*28).
#   실행: CUDA_VISIBLE_DEVICES="" python tools/vla_teacher/smoke_qa_full_navsim_cpu.py
#   1) 조각 조립 == processor.apply_chat_template(3 영상, 다중 턴)
#   2) QAFrameDataset(n_cams=3, NAVSIM 해상도)가 실제 NAVSIM 영상을 읽고: 조각별 == 한 번에 토큰화, 영상 토큰 3×캠당 토큰,
#      표본 길이 == jsonl n_tokens, 손실 마스크 == 턴별 답 + <|im_end|>
#   3) train_qa_full.py 를 NAVSIM 파일·분할(--qa_dir/--fold_file/--num_cams 3/--pixels 100352)로 소형 무작위 모델 CPU 2 step
#   4) 인자·manifest 불일치(--num_cams 6)면 거부되는지
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import navsim_qa_common as N  # noqa: E402
import qa_common as Q  # noqa: E402
import sft_common as S  # noqa: E402
from smoke_qa_full_cpu import sd, tiny_model_dir  # noqa: E402

RES = {}


def ok(name, cond, info=""):
    RES[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)


def main():
    assert S.require_gpu_env(allow_cpu=True) == "cpu", "CUDA_VISIBLE_DEVICES='' 로 실행할 것"
    qa_dir = str(S.REPO / N.QA_DIR_NAVSIM)
    man = json.load(open(Path(qa_dir) / "qa_manifest.json"))
    tpc, ncam = man["tokens_per_cam"], man["n_cams"]
    proc = S.make_processor(S.BASE_QWEN, man["pixels"])
    tok = proc.tokenizer
    tr = Q.load_qa("A", "train", qa_dir)[0]
    pick = []
    for ds in ("nuplanqa", "recogdrive_navsim"):
        rs = [r for r in tr if r["dataset"] == ds]
        if rs:
            pick += [rs[0], max(rs, key=lambda r: r["n_tokens"])]
    chunked = [r for r in tr if r["n_chunks"] > 1][:1]
    pick += chunked
    good = all("".join(t for t, _ in Q.conversation_pieces(r, n_images=ncam)) ==
               proc.apply_chat_template(Q.chat_messages(r, n_images=ncam), tokenize=False, add_generation_prompt=False) for r in pick)
    ok("pieces_equal_chat_template", good, f"{len(pick)} 표본")
    ds = Q.QAFrameDataset(pick, proc, n_cams=ncam)
    g_whole, g_len, g_mask, g_img = True, True, True, True
    from PIL import Image
    grids = set()
    for i, r in enumerate(pick):
        it = ds[i]
        full = "".join(t for t, _ in Q.conversation_pieces(r, n_images=ncam))
        enc = proc(text=[full], images=[Image.open(p).convert("RGB") for p in Q.cam_paths(r, ncam)], return_tensors="pt")
        g_whole &= enc["input_ids"][0].tolist() == it["input_ids"].tolist()
        g_img &= int((it["input_ids"] == Q.IMAGE_PAD_ID).sum()) == ncam * tpc
        grids.add(str(it["image_grid_thw"].tolist()))
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
    ok("tokenize_pieces_equal_whole", g_whole)
    ok(f"image_tokens_{ncam}x{tpc}", g_img, f"grid {grids}")
    ok("length_matches_jsonl", g_len)
    ok("loss_mask_answers_only", g_mask)
    ok("chunk_sample_present", bool(chunked), chunked[0]["id"] if chunked else "chunk 표본 없음(4096 안에 다 들어감)")

    import train_qa_full as T
    tmp = Path(tempfile.mkdtemp(prefix="qa_full_navsim_smoke_"))
    try:
        base = tiny_model_dir(str(tmp / "tiny"), proc)
        common = ["--fold", "A", "--base", base, "--device", "cpu", "--test_subset", "4", "--batch", "2", "--accum", "1",
                  "--workers", "0", "--holdout_every", "2", "--save_every", "1", "--lr", "1e-3", "--lr_projector", "1e-3",
                  "--warmup", "0", "--qa_dir", qa_dir, "--fold_file", str(S.REPO / N.FOLD_FILE_NAVSIM)]
        T.main(common + ["--num_cams", "3", "--pixels", str(man["pixels"]), "--max_steps", "2", "--out", str(tmp / "run")])
        w0, w1 = sd(base), sd(tmp / "run" / "final")
        vis_blocks = [k for k in w0 if k.startswith("visual.") and not k.startswith("visual.merger.")]
        ok("vision_blocks_unchanged", all(torch.equal(w0[k], w1[k]) for k in vis_blocks))
        ok("merger_changed", any(not torch.equal(w0[k], w1[k]) for k in w0 if k.startswith("visual.merger.")))
        ok("llm_changed", any(not torch.equal(w0[k], w1[k]) for k in w0 if k.startswith("model.layers.")))
        meta = json.load(open(tmp / "run" / "final" / "qa_meta.json"))
        ok("meta_cams_pixels", meta["cams"] == "current 3" and meta["pixels"] == man["pixels"], f"{meta['cams']} / {meta['pixels']}")
        log = [json.loads(l) for l in open(tmp / "run" / "train_log.jsonl")]
        st = [x for x in log if "loss" in x]
        ok("finite_loss", all(x["loss"] == x["loss"] and abs(x["loss"]) < 1e9 for x in st), [round(x["loss"], 3) for x in st])
        ok("holdout_by_dataset", any(x.get("kind") == "holdout" for x in log),
           [k for k in next(x for x in log if x.get("kind") == "holdout") if k not in ("kind", "step", "epoch", "eval_s")])
        rejected = False
        try:
            T.main(common + ["--max_steps", "1", "--out", str(tmp / "bad")])   # 기본 6캠 → manifest(3캠)와 불일치
        except SystemExit as e:
            rejected = "manifest n_cams/pixels" in str(e)
        ok("rejects_cam_mismatch", rejected, "--num_cams 기본 6 으로 NAVSIM 데이터 → 거부")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bad = [k for k, v in RES.items() if not v]
    print("ALL PASS" if not bad else f"FAIL {bad}", flush=True)
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
