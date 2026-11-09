# 수정(2026-10-16): 교사 QA 전체 학습 단계 — 교사 SFT 전에 공개 nuScenes QA 로 Qwen2-VL-2B-Instruct 를 전체 파라미터 미세조정.
"""교사 QA 전체 학습(fold 별).

  CUDA_VISIBLE_DEVICES=1 python tools/vla_teacher/train_qa_full.py --fold A      # → results/vla_teacher/qa_full_foldA/{best,final}
  CUDA_VISIBLE_DEVICES=1 python tools/vla_teacher/train_qa_full.py --fold B
  이후 교사 SFT: train_sft.py --base results/vla_teacher/qa_full_foldA/best --fold A --out results/vla_teacher/sft_foldA_qa

데이터: make_qa_data.py 산출 data/vla_teacher/qa/fold<F>_{train,holdout}.jsonl — 장면 집합이 train_sft.py --fold <F> 와 같다
  (학습 = train.<F> − holdout.<F>, hold-out = holdout.<F>, nuScenes val 150 장면 전부 제외). 시작 때 fold_split.json 해시와
  manifest 의 해시를 대조하고, 모든 표본 장면이 허용 집합 안에 있는지 다시 검사한다(어긋나면 실패).
학습 대상(기본): 언어모델 전체(28 층 + 입력 임베딩 = 묶인 lm_head) + projector(visual.merger). 시각 인코더(patch_embed + 32
  블록)는 동결 — 공식 Qwen 미세조정 스크립트 기본(tune_mm_vision False, tune_mm_mlp True, tune_mm_llm True) 과 같다.
  --train_vision 이면 시각 인코더도 학습(lr --lr_vision, 공식 vision_tower_lr 1e-6).
  수정(2026-10-29): 검토 결과 qa_common.vision_features 가 patch_embed·블록을 항상 no_grad 로 계산하므로 --train_vision 을 줘도
  시각 인코더는 기울기가 없어 갱신되지 않는다(실제 = 동결, 체크포인트 대조 차이 0). 진행 중 런의 재개 호환(optimizer 그룹)을 위해
  파라미터 그룹은 그대로 두고 기록(qa_meta.vision_frozen)만 사실대로 남긴다.
학습률(기본): 언어모델 1e-6, projector 1e-5. 근거: QwenLM/Qwen2.5-VL qwen-vl-finetune(README '2025.04.08 … fine-tuning
  Qwen2-VL and Qwen2.5-VL') 의 README 예시 learning_rate 2e-7 · mm_projector_lr 1e-5 · vision_tower_lr 1e-6, 권장
  범위 'from 1e-6 to 2e-7', warmup_ratio 0.03, cosine, weight_decay 0, max_grad_norm 1. 여기서는 권장 범위의 상단 1e-6
  (유효 배치가 공식 예시(4×4×GPU 수)보다 작지 않게 16, 1 에폭). LLaMA-Factory examples/train_full/qwen2vl_full_sft.yaml 은
  1e-5(시각 인코더·projector 동결) — --lr 로 선택 가능.
정밀도: fp32 전용(bf16/fp16 학습 없음). SDPA 는 sft_common.exact_fp32_attention()(math 커널 고정 + 자기검사) 후
  check_sdpa_fp32() 를 모델 적재 뒤 한 번 더 실행 — 이 GPU 의 fp32 memory-efficient 커널은 부정확하다.
손실: 각 assistant 답 토큰 + <|im_end|> 만. 한 optimizer step(누적 마이크로배치 전체)의 감독 토큰 수로 나눈 토큰 평균 CE.
비유한 손실·기울기 노름: 진단 출력 후 즉시 중단(건너뛰지 않음).
저장: best(hold-out 손실 최소)·final = 완전한 HF 모델 폴더(fp32 safetensors + processor + qa_meta.json) → train_sft.py --base 로 적재.
  latest = 재개용(모델 + trainer_state.pt(AdamW 상태 포함)).
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qa_common as Q  # noqa: E402
import sft_common as S  # noqa: E402
from sft_common import BASE_QWEN, PIXELS, REPO, LengthGroupedBatchSampler, collate, make_processor, require_gpu_env  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=["A", "B"], required=True)
    ap.add_argument("--base", default=BASE_QWEN)
    ap.add_argument("--qa_dir", default=Q.QA_DIR)
    ap.add_argument("--fold_file", default=S.FOLD_FILE)
    ap.add_argument("--out", default=None, help="기본 results/vla_teacher/qa_full_fold<F>")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=2, help="마이크로배치(표본 수)")
    ap.add_argument("--accum", type=int, default=8, help="유효 배치 = batch × accum (기본 16, 교사 SFT 와 같음)")
    ap.add_argument("--lr", type=float, default=1e-6, help="언어모델(공식 Qwen 권장 1e-6~2e-7)")
    ap.add_argument("--lr_projector", type=float, default=1e-5, help="visual.merger(공식 mm_projector_lr)")
    ap.add_argument("--train_vision", action="store_true", help="시각 인코더 파라미터를 optimizer 에 넣는다 — 단 vision_features 가 no_grad 라 실제로는 갱신되지 않음(2026-10-29 검토)")
    ap.add_argument("--lr_vision", type=float, default=1e-6, help="--train_vision 일 때(공식 vision_tower_lr)")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--rope", choices=["mrope"], default="mrope")
    ap.add_argument("--holdout_every", type=int, default=250)
    ap.add_argument("--eval_at_start", type=int, default=1, help="1 = 학습 전 hold-out 손실(base 기준값) 측정")
    ap.add_argument("--save_every", type=int, default=100, help="latest(재개용) 저장 간격(optimizer step)")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_vram_gb", type=float, default=45.0,
                    help="이 프로세스 VRAM 상한(GPU1 합계 ≤ 90 GB 규칙 — 같이 도는 작업 사용량을 빼고 정할 것)")
    ap.add_argument("--tf32", action="store_true", help="TF32 matmul 허용(기본 끔 = 엄격 fp32)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--drop_trainer_state_on_finish", action="store_true",
                    help="정상 종료 후 latest/trainer_state.pt(AdamW 상태, 약 12.7 GB) 삭제 — 디스크 절약. 재개 불필요할 때만")
    ap.add_argument("--max_steps", type=int, default=0, help="시험용(0 = 에폭 기준). 스케줄 길이도 이 값")
    ap.add_argument("--stop_at_step", type=int, default=0, help="시험용: 스케줄은 전체 길이 그대로 두고 이 step 에서 멈춤")
    ap.add_argument("--test_subset", type=int, default=0,
                    help="시험 전용(0 = 전체): 학습·hold-out 앞 N 표본만. 실제 학습에 쓰지 말 것(전체 데이터 규칙)")
    ap.add_argument("--test_longest_first", action="store_true",
                    help="시험 전용: 가장 긴 표본부터 배치(최대 메모리 측정). --test_subset 과 함께만")
    ap.add_argument("--device", default=None, help="'cpu' = CPU 시험(CUDA_VISIBLE_DEVICES='' 일 때만)")
    # 수정(2026-10-16): NAVSIM 트랙 — 카메라 수·해상도(기본값 = nuScenes 6캠·200*28*28 그대로). NAVSIM: --num_cams 3 --pixels 100352
    ap.add_argument("--num_cams", type=int, default=S.NUM_CAMS, help="표본 영상 수(manifest 의 n_cams 와 대조)")
    ap.add_argument("--pixels", type=int, default=PIXELS, help="Qwen min=max_pixels(manifest 의 pixels 와 대조)")
    a = ap.parse_args(argv)
    if a.test_longest_first and not a.test_subset:
        raise SystemExit("--test_longest_first 는 --test_subset 과 함께만")
    a.out = a.out or f"results/vla_teacher/qa_full_fold{a.fold}"
    return a


# ---------------------------------------------------------------- 데이터(장면 집합 재검사)
def load_fold_data(a):
    fd, fpath = S.load_fold_split(a.fold_file)
    qa_dir = Path(a.qa_dir) if os.path.isabs(a.qa_dir) else REPO / a.qa_dir
    man = json.load(open(qa_dir / "qa_manifest.json"))
    if man["fold_sha256"] != fd["sha256"]:
        raise SystemExit(f"QA 데이터의 분할 해시 {man['fold_sha256'][:12]} != 현재 분할 {fd['sha256'][:12]} — make_qa_data.py 재실행")
    # 수정(2026-10-16): NAVSIM 트랙 — 데이터가 만든 카메라 수·해상도와 학습 인자가 같아야 한다(nuScenes manifest 는 n_cams 키 없음 → 6)
    if man.get("n_cams", S.NUM_CAMS) != a.num_cams or man.get("pixels", PIXELS) != a.pixels:
        raise SystemExit(f"manifest n_cams/pixels {man.get('n_cams', S.NUM_CAMS)}/{man.get('pixels', PIXELS)} != 인자 {a.num_cams}/{a.pixels}")
    tr, tr_p = Q.load_qa(a.fold, "train", qa_dir)
    ho, ho_p = Q.load_qa(a.fold, "holdout", qa_dir)
    for name, p in ((f"fold{a.fold}_train", tr_p), (f"fold{a.fold}_holdout", ho_p)):
        if S.file_sha256(p) != man["files"][name]["sha256"]:
            raise SystemExit(f"{p}: manifest 해시 불일치(파일이 바뀌었다)")
    mine = set(fd["train"][a.fold]["scenes"])
    hold = set(fd["holdout"][a.fold]["scenes"])
    forbidden = set(fd["val"]["A"]["scenes"]) | set(fd["val"]["B"]["scenes"]) | set(fd["train"]["B" if a.fold == "A" else "A"]["scenes"])
    bad_tr = [r["id"] for r in tr if r["scene_token"] not in mine - hold or r["scene_token"] in forbidden]
    bad_ho = [r["id"] for r in ho if r["scene_token"] not in hold or r["scene_token"] in forbidden]
    if bad_tr or bad_ho:
        raise SystemExit(f"허용 장면 밖 표본: 학습 {len(bad_tr)} {bad_tr[:3]}, hold-out {len(bad_ho)} {bad_ho[:3]}")
    if a.test_subset:
        if a.test_longest_first:
            tr = sorted(tr, key=lambda r: -r["n_tokens"])
            ho = sorted(ho, key=lambda r: -r["n_tokens"])
        tr, ho = tr[:a.test_subset], ho[:a.test_subset]
    return tr, ho, fd, fpath, man


# ---------------------------------------------------------------- 모델
def set_trainable(model, train_vision):
    """반환 {그룹: [(이름, 파라미터)]}. 묶인 lm_head 는 embed_tokens 와 같은 파라미터라 named_parameters 에 한 번만 나온다."""
    groups = {"llm": [], "projector": [], "vision": []}
    for n, p in model.named_parameters():
        if n.startswith("model.") or n.startswith("lm_head."):
            g = "llm"
        elif n.startswith("visual.merger."):
            g = "projector"
        elif n.startswith("visual."):
            g = "vision"
        else:
            raise RuntimeError(f"분류 안 된 파라미터 {n}")
        p.requires_grad_(g != "vision" or train_vision)
        groups[g].append((n, p))
    if model.config.tie_word_embeddings and model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise RuntimeError("tie_word_embeddings=True 인데 lm_head 와 embed_tokens 가 공유되지 않는다")
    return groups


def build_model(path, device, train_vision):
    import torch
    from transformers import Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="sdpa")
    groups = set_trainable(model, train_vision)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.to(device)
    for n, p in model.named_parameters():
        if p.dtype != torch.float32:
            raise RuntimeError(f"fp32 가 아닌 파라미터 {n} {p.dtype}")
    return model, groups


def save_full(model, proc, d, meta, trainer_state=None):
    """train_sft.py --base 가 그대로 읽는 HF 폴더(가중치 fp32 safetensors + config + processor/tokenizer/chat_template).
    수정(2026-10-29): trainer_state 를 주면 tmp 안에 먼저 쓰고 교체 — 교체 뒤에 쓰던 종전 방식은 그 사이 중단 시
    상태 파일 없는 latest 가 남아 재실행이 처음부터 돌았다."""
    d = Path(d)
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.config.use_cache = True        # 추론 기본값으로 저장(학습 중엔 False)
    model.save_pretrained(tmp, safe_serialization=True, max_shard_size="5GB")
    model.config.use_cache = False
    proc.save_pretrained(tmp)
    if (Path(meta["base"]) / "generation_config.json").exists() and not (tmp / "generation_config.json").exists():
        shutil.copy(Path(meta["base"]) / "generation_config.json", tmp / "generation_config.json")
    S.save_meta(tmp / "qa_meta.json", **meta)
    if trainer_state is not None:
        import torch
        torch.save(trainer_state, tmp / "trainer_state.pt")
    # 수정(2026-10-29): rmtree(d) → rename 이던 교체를 rename 두 번으로 — rmtree 도중 중단되면 일부만 지워진 d 가 남아
    #   부수 파일 없이 조용히 적재될 수 있었다. 이제 어느 순간 중단돼도 d 는 완전하거나 없고(없으면 .old/.tmp 가 남아 시작 검사가 거부).
    old = d.with_name(d.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if d.exists():
        d.rename(old)
    tmp.rename(d)                        # 저장 도중 중단돼도 이전 체크포인트가 깨지지 않게
    if old.exists():
        shutil.rmtree(old)
    return d


def link_copy(src, dst):
    """final = 마지막 step 의 latest 와 같은 가중치 — 하드링크로 디스크를 더 쓰지 않는다(trainer_state 제외)."""
    src, dst = Path(src), Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for f in src.iterdir():
        if f.name == "trainer_state.pt":
            continue
        os.link(f, tmp / f.name)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)


def evaluate(model, dl, device, rope):
    """hold-out 토큰 평균 CE·토큰 정확도(전체 + 데이터셋별). dl 은 batch_size=1(패딩 없음, 표본별 데이터셋 집계)."""
    import torch
    model.eval()
    agg = {}
    with torch.no_grad():
        for batch in dl:
            b = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            ce, n, c = Q.qa_forward_loss(model, b, rope, train_merger=False)
            if not torch.isfinite(ce):
                raise FloatingPointError(f"hold-out 비유한 손실 {batch['tokens']} — 원인 진단 필요(건너뛰지 않음)")
            for key in (batch["tokens"][0].split(":")[0], "all"):
                s = agg.setdefault(key, [0.0, 0, 0])
                s[0] += float(ce); s[1] += n; s[2] += c
    model.train()
    model.visual.eval()
    return {k: dict(loss=v[0] / max(v[1], 1), tok_acc=v[2] / max(v[1], 1), n_tok=v[1]) for k, v in agg.items()}


def grad_norms_by_group(groups):
    import torch
    out = {}
    with torch.no_grad():
        for g, lst in groups.items():
            s, bad = 0.0, []
            for n, p in lst:
                if p.grad is None:
                    continue
                v = float(p.grad.float().pow(2).sum())
                if not math.isfinite(v):
                    bad.append(n)
                s += v if math.isfinite(v) else 0.0
            out[g] = dict(norm=math.sqrt(s), nonfinite=bad[:10], n_nonfinite=len(bad))
    return out


def main(argv=None):
    a = parse_args(argv)
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    dev_kind = require_gpu_env(allow_cpu=(a.device == "cpu"))
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup
    device = torch.device("cpu" if dev_kind == "cpu" else "cuda")
    torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
    torch.backends.cudnn.allow_tf32 = bool(a.tf32)
    attn_info = "cpu(math)"
    if device.type == "cuda":
        attn_info = S.exact_fp32_attention(torch.float32)          # math 커널 고정 + 자기검사(실패 시 예외)
        if not attn_info.startswith("sdpa=math"):
            raise RuntimeError(f"정확한 fp32 어텐션 설정 실패: {attn_info}")
        print(f"[attn] {attn_info}", flush=True)
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_vram_gb / tot), 0)
    random.seed(a.seed); torch.manual_seed(a.seed)

    out = Path(a.out) if os.path.isabs(a.out) else REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    tr, ho, fd, fpath, man = load_fold_data(a)
    by_ds = {}
    for r in tr:
        by_ds[r["dataset"]] = by_ds.get(r["dataset"], 0) + 1
    print(f"[data] fold {a.fold}: 학습 {len(tr)} 표본 {by_ds}, hold-out {len(ho)} 표본, 분할 sha256 {fd['sha256'][:12]}", flush=True)

    resume_dir = out / "latest"
    resuming = a.resume and (resume_dir / "trainer_state.pt").exists()
    # 수정(2026-10-29): 조용히 처음부터 다시 돌면 약 하루를 재학습하고 best 를 이른 체크포인트로 덮어쓴다 → 멈춘다.
    if a.resume and not resuming:
        raise SystemExit(f"[거부] --resume 인데 {resume_dir}/trainer_state.pt 없음(학습 종료 후 삭제됐거나 저장 중 중단) — "
                         f"처음부터 돌리지 않는다. latest 교체 중 중단이면 완전한 latest.tmp(trainer_state.pt 있음, 더 최신) 또는 latest.old 를 "
                         f"latest 로 옮긴 뒤 재개. 학습 완료(final/ 있음)면 다시 돌릴 필요 없음")
    if a.resume:
        # 수정(2026-10-29): best 교체 중 중단(best 없음 + best.old 남음)이면 재개 뒤 best 가 다시 안 만들어질 수 있다 → 멈춘다.
        if (out / "best.old").exists() and not (out / "best").exists():
            raise SystemExit(f"[거부] {out}/best 없이 best.old 만 있다(best 교체 중 중단) — best.old 를 best 로 옮긴 뒤 재개(best.tmp 는 미완성일 수 있어 쓰지 않음 — 그 hold-out 개선분은 재개 뒤 다시 평가됨)")
    if not a.resume:
        prev = [n for n in ("latest", "latest.tmp", "latest.old", "best", "best.old", "final") if (out / n).exists()]
        if prev:
            raise SystemExit(f"[거부] {out} 에 이전 학습 산출물 {prev} 이 있다 — latest 가 없고 latest.tmp/latest.old 만 있으면 그것을 latest 로 옮긴 뒤 "
                             f"--resume, 아니면 --resume 으로 이어 가거나 폴더를 옮긴 뒤 새로 시작")
    load_path = str(resume_dir) if resuming else a.base
    proc = make_processor(a.base if not resuming else str(resume_dir), a.pixels)  # 수정(2026-10-16): NAVSIM 트랙 — 기본 PIXELS
    tok = proc.tokenizer
    model, groups = build_model(load_path, device, a.train_vision)
    if device.type == "cuda":  # 설계 검토 요구: 모델 적재 직후 다시 고정 + 자기검사(전역 설정이 모델 적재로 바뀌지 않았음을 확인)
        attn_info = S.exact_fp32_attention(torch.float32)
        if not attn_info.startswith("sdpa=math"):
            raise RuntimeError(f"정확한 fp32 어텐션 설정 실패: {attn_info}")
        print(f"[attn] {attn_info} (모델 적재 후)", flush=True)
        print(f"[attn] check_sdpa_fp32: {S.check_sdpa_fp32()}", flush=True)
    n_tr = {g: sum(p.numel() for _, p in lst if p.requires_grad) for g, lst in groups.items()}
    print(f"[model] 학습 파라미터 {n_tr} (합 {sum(n_tr.values()):,}), 동결 visual(patch_embed+blocks) — vision_features 가 no_grad{' (--train_vision: optimizer 에만 포함, 갱신 없음)' if a.train_vision else ''}", flush=True)

    ds_tr = Q.QAFrameDataset(tr, proc, n_cams=a.num_cams)  # 수정(2026-10-16): NAVSIM 트랙 — 기본 6
    ds_ho = Q.QAFrameDataset(sorted(ho, key=lambda r: r["n_tokens"]), proc, n_cams=a.num_cams)
    lens = [r["n_tokens"] for r in tr]
    sampler = LengthGroupedBatchSampler(lens, a.batch, seed=a.seed)
    if a.test_longest_first:
        sampler.batches = lambda: [list(range(i, min(i + a.batch, len(tr)))) for i in range(0, len(tr), a.batch)]
    col = lambda b: collate(b, tok.pad_token_id)  # noqa: E731
    dl_ho = DataLoader(ds_ho, batch_size=1, shuffle=False, num_workers=a.workers, collate_fn=col)
    n_b = len(sampler)
    steps_per_epoch = math.ceil(n_b / a.accum)
    total = a.max_steps or int(math.ceil(steps_per_epoch * a.epochs))

    pg = [dict(params=[p for _, p in groups["llm"]], lr=a.lr, weight_decay=a.weight_decay, name="llm"),
          dict(params=[p for _, p in groups["projector"]], lr=a.lr_projector, weight_decay=a.weight_decay, name="projector")]
    if a.train_vision:
        pg.append(dict(params=[p for _, p in groups["vision"]], lr=a.lr_vision, weight_decay=a.weight_decay, name="vision"))
    opt = torch.optim.AdamW(pg, betas=(0.9, 0.999), eps=1e-8, fused=(device.type == "cuda"))
    sch = get_cosine_schedule_with_warmup(opt, int(a.warmup * total), total)
    train_params = [p for g in pg for p in g["params"]]

    train_scenes = sorted(set(fd["train"][a.fold]["scenes"]) - set(fd["holdout"][a.fold]["scenes"]))
    holdout_scenes = sorted(fd["holdout"][a.fold]["scenes"])
    # 다음 SFT(train_sft.py --base)가 fold 일치를 검사할 수 있게: fold / fold_sha256 / scene_sets / base (키 이름 고정)
    meta = dict(stage="teacher_qa_full", tag=Q.TAG, base=a.base, fold=a.fold, fold_file=fpath, fold_sha256=fd["sha256"],
                scene_sets=dict(train=f"train.{a.fold} − holdout.{a.fold}", holdout=f"holdout.{a.fold}",
                                excluded=["val.A", "val.B", f"train.{'B' if a.fold == 'A' else 'A'}"],
                                train_scenes=train_scenes, holdout_scenes=holdout_scenes,
                                train_scenes_sha256=S.scenes_hash(train_scenes), holdout_scenes_sha256=fd["holdout"][a.fold]["sha256"],
                                train_scenes_used=sorted({r["scene_token"] for r in tr}),
                                holdout_scenes_used=sorted({r["scene_token"] for r in ho})),
                qa_manifest_sha256=S.file_sha256(Path(a.qa_dir if os.path.isabs(a.qa_dir) else REPO / a.qa_dir) / "qa_manifest.json"),
                qa_files={k: v for k, v in man["files"].items() if k.startswith(f"fold{a.fold}_")},
                n_train=len(tr), n_holdout=len(ho), pixels=a.pixels, cams=f"current {a.num_cams}",  # 수정(2026-10-16): NAVSIM 트랙
                rope=a.rope, dtype="fp32",
                attention=attn_info, trainable=n_tr, vision_frozen=True,  # 수정(2026-10-29): vision_features 가 항상 no_grad → --train_vision 과 무관하게 실제 동결
                vision_in_optimizer=bool(a.train_vision), args=vars(a))
    step, epoch, best, start_batch = 0, 0, float("inf"), 0
    if resuming:
        st = torch.load(resume_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"])
        step, epoch, best, start_batch = st["step"], st["epoch"], st["best"], st["batch_in_epoch"]
        if st.get("total") != total:
            raise SystemExit(f"재개: 전체 step {st.get('total')} != 현재 설정 {total} — 같은 인자로 재개할 것")
        random.setstate(st["py_rng"]); torch.set_rng_state(st["torch_rng"])
        print(f"[resume] step {step} epoch {epoch} batch {start_batch} best {best:.4f}", flush=True)
    S.save_meta(out / "qa_meta.json", **meta)
    log_f = open(out / "train_log.jsonl", "a")

    def log(rec):
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); log_f.flush()

    def run_eval(kind):
        nonlocal best
        t = time.time()
        ev = evaluate(model, dl_ho, device, a.rope)
        log(dict(kind=kind, step=step, epoch=epoch, eval_s=round(time.time() - t, 1), **ev))
        if kind != "holdout_start" and ev["all"]["loss"] < best:
            best = ev["all"]["loss"]
            save_full(model, proc, out / "best", dict(meta, step=step, holdout=ev))
        return ev

    model.train(); model.visual.eval()
    if a.eval_at_start and step == 0:
        run_eval("holdout_start")
    t0, step0, tok_seen = time.time(), step, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    done = step >= total
    while not done:
        sampler.set_epoch(epoch)
        sampler.skip = start_batch
        dl = DataLoader(ds_tr, batch_sampler=sampler, num_workers=a.workers, collate_fn=col,
                        pin_memory=(device.type == "cuda"))
        it = iter(dl)
        bi = start_batch
        while bi < n_b:
            ts = time.time()
            micro = []
            while len(micro) < a.accum and bi < n_b:   # 에폭 끝 잔여 마이크로배치도 버리지 않고 한 step(토큰 정규화라 보정 불필요)
                micro.append(next(it)); bi += 1
            n_step = sum(int((m["labels"][:, 1:] != -100).sum()) for m in micro)
            ce_tot, corr_tot = 0.0, 0
            for m in micro:
                b = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v) for k, v in m.items()}
                ce, n, c = Q.qa_forward_loss(model, b, a.rope, train_merger=True)
                if not torch.isfinite(ce):
                    raise FloatingPointError(f"비유한 손실 step {step + 1} 표본 {m['tokens']} 길이 {list(m['attention_mask'].sum(1).tolist())} "
                                             f"— 원인 진단 필요(건너뛰지 않음)")
                (ce / n_step).backward()
                ce_tot += float(ce); corr_tot += c
                tok_seen += int(m["attention_mask"].sum())
            gn = torch.nn.utils.clip_grad_norm_(train_params, a.clip)
            if not torch.isfinite(gn):
                diag = grad_norms_by_group({k: v for k, v in groups.items() if k != "vision" or a.train_vision})
                log(dict(kind="nonfinite_grad", step=step + 1, gnorm=float(gn), groups=diag, samples=[m["tokens"] for m in micro]))
                raise FloatingPointError(f"비유한 기울기 노름 step {step + 1} — 원인 진단 필요(건너뛰지 않음)")
            lrs = {g["name"]: g["lr"] for g in opt.param_groups}
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            if step % a.log_every == 0 or step == 1:
                el = time.time() - t0
                rate = el / max(step - step0, 1)
                log(dict(step=step, epoch=epoch, loss=ce_tot / max(n_step, 1), tok_acc=corr_tot / max(n_step, 1),
                         n_answer_tok=n_step, gnorm=float(gn), lr=lrs, s_step=round(time.time() - ts, 2),
                         s_per_step_avg=round(rate, 2), tok_per_s=round(tok_seen / el, 1),
                         peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None,
                         eta_h=round((total - step) * rate / 3600, 2)))
            if step % a.holdout_every == 0 or step == total:
                run_eval("holdout")
            if step % a.save_every == 0 or step == total:
                save_full(model, proc, out / "latest", dict(meta, step=step),   # 수정(2026-10-29): 상태 파일도 교체 전에 기록
                          trainer_state=dict(opt=opt.state_dict(), sch=sch.state_dict(), step=step, epoch=epoch, best=best,
                                             total=total, batch_in_epoch=bi, py_rng=random.getstate(),
                                             torch_rng=torch.get_rng_state()))
            if step >= total or (a.stop_at_step and step >= a.stop_at_step):
                done = True
                break
        start_batch = 0
        sampler.skip = 0
        if not done:
            epoch += 1
    if step >= total:
        link_copy(out / "latest", out / "final")
        S.save_meta(out / "final" / "qa_meta.json.tmp", **dict(meta, step=step, final=True))
        os.replace(out / "final" / "qa_meta.json.tmp", out / "final" / "qa_meta.json")  # 하드링크된 meta 를 덮지 않고 교체
        if a.drop_trainer_state_on_finish:
            os.remove(out / "latest" / "trainer_state.pt")
    el = time.time() - t0
    summary = dict(steps=step, total=total, wall_s=el, s_per_step=el / max(step - step0, 1), tok_per_s=tok_seen / max(el, 1e-9),
                   best_holdout_loss=best,
                   peak_gb=(torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None))
    json.dump(summary, open(out / "train_summary.json", "w"), indent=1)
    print("[done]", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
