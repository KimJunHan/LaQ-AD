# 수정(2026-10-16): 교사 GRPO — SFT LoRA 교사(fold A|B)를 GT 보상 기반 GRPO 로 강화학습(LoRA·새 토큰 행만, fp32).
"""교사 VLM GRPO (group relative policy optimization).

참고 알고리즘: DeepSeekMath GRPO(arXiv 2402.03300 §4.1), AutoVLA RFT(정책 갱신 = step 당 1 회 → 중요도 비 1, clip·old policy
불필요; SFT 참조 KL 을 k3 추정자로; lr 3e-5; LoRA 만). 수식·보상 정의는 grpo_common.py 머리 주석·RewardConfig.

step 하나:
  1) fold 학습 표본(장면 단위 fold 분할의 train.<fold> − holdout.<fold>, 전체 — 부분집합 아님)을 epoch 씨드 순열로 돌며
     프롬프트 P 개(--prompts_per_step, 기본 2) 를 꺼낸다. 프롬프트·영상은 SFT 와 같은 조립(meta 의 system·이전 프레임·순서).
  2) 영상 특징을 프롬프트당 한 번(이미지 단위 인코딩, 비전 동결) 계산 → 표집·참조·정책 순전파에서 재사용.
  3) 프롬프트마다 G 개(--group, 기본 8) 표집: temperature 1.0, top_p 1.0(전체 분포 — on-policy 로그확률과 일치),
     최대 548 토큰(정답 547 + <|im_end|>). 접두 1 회 + KV 캐시 G 복제.
     수정(2026-10-16): 설계 검토 반영 — 문법 제약 표집(answer_grammar.py, 캐시 추출과 같은 제약). 표집·참조·정책 로그확률은
     모두 허용 집합으로 다시 정규화한 log q(근거: grpo_common.py 머리 주석). 시작 때 fp32 SDPA math 커널 고정·자기검사,
     SFT 체크포인트는 QA → SFT 연결(fold·분할 해시·base 출처)을 sft_common.verify_teacher_chain 으로 검사.
  4) 보상(GT, ∈[0,1]) → 그룹 이점. 5) π_ref 로그확률(no_grad, SFT 값으로 잠깐 교체). 6) 정책 순전파(gradient checkpointing)
     + 손실 backward(마이크로배치 누적, 정확히 step 평균). 7) 기울기 노름(그룹별)·clip 1.0·AdamW.
  비유한 값(보상·로그확률·손실·기울기)은 즉시 진단 정보와 함께 예외(건너뛰지 않음 — 프로젝트 규칙).
체크포인트 선택: fold 의 holdout 장면 전 표본에 탐욕 생성 → 평균 보상(--holdout_every step 마다·마지막). 최고 → <out>/best.
저장: <out>/{latest,best,final}/ = adapter_model.safetensors(+토크나이저, teacher_meta.json — merge_teacher_lora.py 그대로 사용 가능),
      latest/trainer_state.pt(옵티마이저·스케줄·step·순열 커서·RNG) → --resume.
GPU0 금지(sft_common.require_gpu_env). 실행 예: CUDA_VISIBLE_DEVICES=1 python tools/vla_teacher/train_grpo.py --fold A
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_common as S  # noqa: E402  (읽기 전용)
import grpo_common as GC  # noqa: E402
import answer_grammar as AG  # noqa: E402  # 수정(2026-10-16): 설계 검토 반영 — 문법 제약 표집


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=["A", "B"], required=True)
    ap.add_argument("--sft_dir", default=None, help="기본 results/vla_teacher/sft_fold<F>")
    ap.add_argument("--sft_ckpt", default="best", help="sft_dir 아래 체크포인트(best|final|latest) 또는 절대 경로")
    ap.add_argument("--out", default=None, help="기본 results/vla_teacher/grpo_fold<F>")
    ap.add_argument("--data_dir", default=None, help="기본 SFT meta 의 data_dir")
    ap.add_argument("--fold_file", default=S.FOLD_FILE)
    ap.add_argument("--infos", default="data/infos/nuscenes_infos_train.pkl", help="미래 박스(충돌 보상) — fold·holdout 은 train split")
    ap.add_argument("--group", type=int, default=8, help="G: 프롬프트당 표집 수")
    ap.add_argument("--prompts_per_step", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=GC.ANSWER_TOKENS + 1)
    ap.add_argument("--beta", type=float, default=0.04, help="KL(π_θ‖π_ref) 계수(k3)")
    ap.add_argument("--lr", type=float, default=3e-5, help="LoRA lr (AutoVLA RFT)")
    ap.add_argument("--lr_tokens", type=float, default=None, help="새 토큰 행 lr(기본 = --lr)")
    ap.add_argument("--warmup_steps", type=int, default=10, help="선형 warmup 후 상수 lr")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--adv_eps", type=float, default=1e-4)
    ap.add_argument("--loss_norm", choices=["seq", "token"], default="seq",
                    help="seq = 행별 토큰 평균 후 행 평균(DeepSeekMath), token = 전체 감독 토큰 평균")
    ap.add_argument("--pg_exclude_intent", action="store_true",
                    help="Intent 줄 토큰을 정책 기울기·KL 에서 제외(보상에 의도 항이 없으므로). 기본: 포함(표준 GRPO)")
    ap.add_argument("--reward_cfg", default="", help="RewardConfig 덮어쓰기(JSON 문자열 또는 파일)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--micro", type=int, default=8, help="정책·참조 순전파 마이크로배치(행 수)")
    ap.add_argument("--holdout_every", type=int, default=100)
    ap.add_argument("--early_stop_patience", type=int, default=0,  # 수정(2026-10-18): 사용자 승인 — holdout 보상 정체 시 종료
                    help=">0 이면 holdout 평균 보상이 이 step 수 동안 최고값을 넘지 못하면 종료(best = 그때까지 최고). 0 = 끔")
    ap.add_argument("--start_gate", action="store_true",  # 수정(2026-10-18): 사용자 승인 — GRPO 시작 조건
                    help="학습 전 SFT 상태로 holdout 평가: 형식 유효율 1.0 이 아니면 중단, 맞으면 그 보상을 best 기준선으로 저장")
    ap.add_argument("--eval_batch", type=int, default=16, help="holdout 탐욕 생성 배치(프롬프트 수)")
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_vram_gb", type=float, default=80.0)
    ap.add_argument("--tf32", action="store_true", help="TF32 matmul 허용(기본 끔 = 엄격 fp32)")
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa",  # 수정(2026-10-16): 설계 검토 반영
                    help="fp32 정확 경로만: sdpa(시작 때 math 커널 고정·자기검사) | eager")
    ap.add_argument("--device", default=None, help="스모크용 'cpu' (CUDA_VISIBLE_DEVICES='' 일 때만)")
    ap.add_argument("--dry_run", action="store_true", help="데이터 구성·비용 추정만 출력(모델 미적재)")
    # 스모크 전용(실제 학습에서 쓰지 말 것 — --smoke 없이는 거부)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_holdout_samples", type=int, default=0)
    a = ap.parse_args(argv)
    if not a.smoke and (a.max_train_samples or a.max_holdout_samples or a.max_new_tokens < GC.ANSWER_TOKENS + 1):
        raise SystemExit("--max_train_samples / --max_holdout_samples / 짧은 --max_new_tokens 는 --smoke 전용(전체 데이터 규칙)")
    if a.group < 2:
        raise SystemExit("--group ≥ 2 (그룹 상대 이점)")
    if a.top_p < 1.0:
        print("[주의] top_p<1: 표집 분포가 π_θ 와 달라져 on-policy 가정이 깨진다(학습 log π 는 전체 분포)", flush=True)
    a.sft_dir = a.sft_dir or f"results/vla_teacher/sft_fold{a.fold}"
    a.out = a.out or f"results/vla_teacher/grpo_fold{a.fold}"
    return a


def _abs(p):
    return Path(p) if os.path.isabs(p) else S.REPO / p


def resolve_sft_ckpt(a):
    p = Path(a.sft_ckpt) if os.path.isabs(a.sft_ckpt) else _abs(a.sft_dir) / a.sft_ckpt
    if not (p / "adapter_model.safetensors").exists() or not (p / "teacher_meta.json").exists():
        raise SystemExit(f"SFT 체크포인트 없음: {p} (adapter_model.safetensors + teacher_meta.json 필요 — train_sft.py 의 "
                         f"<out>/best|final|latest)")
    return p


def check_fold_meta(meta, a):
    """SFT 가 같은 fold·같은 분할 파일로 학습되었는지 확인(다른 fold 장면으로 RL 하면 교차 적합이 깨진다).
    # 수정(2026-10-16): 설계 검토 반영 — QA(base) → SFT 전 단계를 sft_common.verify_teacher_chain 으로 검사(fold·분할 해시·base 출처)."""
    return S.verify_teacher_chain(meta, a.fold, a.fold_file)


def lr_lambda(warm):
    return lambda s: min(1.0, (s + 1) / max(warm, 1))


def run_holdout(pm, proc, tok, meta, recs, ctxs, codebook, cfg, a, device, grammar=None):
    """holdout 전 표본 탐욕 생성 → 평균 보상·성분. (체크포인트 선택 전용; val 미사용)"""
    import torch
    eos, pad = tok.convert_tokens_to_ids(GC.END_TOKEN), tok.pad_token_id
    tot, comp, n_fmt = [], {"perc": [], "map": [], "act": []}, 0
    t0 = time.time()
    for s in range(0, len(recs), a.eval_batch):
        rs = recs[s:s + a.eval_batch]
        prompts = [GC.encode_prompt(proc, r, meta) for r in rs]
        feats = [GC.image_features(pm, x["pixel_values"], x["grid"]) for x in prompts]
        out = GC.sample_answers(pm, prompts, feats, 1, a.max_new_tokens, eos, pad, greedy=True, grammar=grammar)  # 수정(2026-10-16): 설계 검토 반영
        for i, r in enumerate(rs):
            n = int(out["n_gen"][i]); ended = bool(out["ended"][i])
            text = tok.decode(out["ids"][i, :n - 1 if ended else n].tolist(), skip_special_tokens=False)
            R, d = GC.score_answer(text, ended, r, codebook, ctxs.get(r["token"]), cfg)
            tot.append(R); n_fmt += int(d["fmt"] > 0)
            for k in comp:
                if d[k] is not None:
                    comp[k].append(d[k])
        del prompts, feats
    mean = lambda x: float(sum(x) / len(x)) if x else None  # noqa: E731
    return dict(kind="holdout", n=len(tot), reward=mean(tot), format_valid=n_fmt / max(len(tot), 1),
                **{f"r_{k}": mean(v) for k, v in comp.items()}, wall_s=time.time() - t0)


def main(argv=None):
    a = parse_args(argv)
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    dev_kind = S.require_gpu_env(allow_cpu=(a.device == "cpu" or a.dry_run))
    import numpy as np
    import torch
    device = torch.device("cpu" if dev_kind == "cpu" else "cuda")
    torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
    torch.backends.cudnn.allow_tf32 = bool(a.tf32)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if device.type == "cuda":
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_vram_gb / tot), 0)
    if not a.dry_run:   # 수정(2026-10-16): 설계 검토 반영 — fp32 SDPA math 커널 고정 + float64 기준 자기검사(CUDA 일 때; CPU 는 'unchanged')
        print(f"[attn] 시작: {S.exact_fp32_attention(torch.float32)}", flush=True)

    ck_dir = Path(a.sft_ckpt) if os.path.isabs(a.sft_ckpt) else _abs(a.sft_dir) / a.sft_ckpt
    if a.dry_run and not (ck_dir / "teacher_meta.json").exists():   # SFT 완료 전에도 데이터 구성은 볼 수 있게
        ck, meta = ck_dir, None
        print(f"[dry_run] SFT 체크포인트 아직 없음({ck_dir}) — fold 검사 생략", flush=True)
    else:
        ck = resolve_sft_ckpt(a)
        meta = GC.load_sft_meta(ck)
        chain = check_fold_meta(meta, a)   # 수정(2026-10-16): 설계 검토 반영
        print(f"[chain] {json.dumps(chain, ensure_ascii=False)}", flush=True)
    data_dir = S.resolve_data_dir(a.data_dir or (meta or {}).get("data_dir"))
    recs = S.load_records(data_dir, "train")
    tr, ho, hold, hold_hash, fpath, fd = S.fold_select(recs, a.fold, a.fold_file)
    if meta is not None and hold_hash != meta.get("holdout_sha256"):
        raise SystemExit("holdout 해시가 SFT meta 와 다르다(다른 분할로 학습된 SFT)")
    if a.max_train_samples:
        tr = tr[:a.max_train_samples]
    if a.max_holdout_samples:
        ho = ho[:a.max_holdout_samples]
    P, G = a.prompts_per_step, a.group
    out = _abs(a.out)
    info = dict(fold=a.fold, sft_ckpt=str(ck), n_train=len(tr), n_train_scenes=len({r["scene_token"] for r in tr}),
                n_holdout=len(ho), n_holdout_scenes=len(hold), P=P, G=G, rows_per_step=P * G, steps=a.steps,
                prompts_seen=P * a.steps, fold_epochs=P * a.steps / max(len(tr), 1), out=str(out))
    print("[data]", json.dumps(info, ensure_ascii=False), flush=True)
    if a.dry_run:
        return info

    cfg = GC.RewardConfig.from_json(a.reward_cfg)
    from common import ActionCodebook
    codebook = ActionCodebook(str(S.REPO / S.CODEBOOK))
    t0 = time.time()
    ctxs = GC.build_action_contexts(tr + ho, str(_abs(a.infos)), codebook, cfg)
    print(f"[reward] 행동 문맥 {len(ctxs)} (future_complete) {time.time() - t0:.0f}s, cfg {json.dumps(cfg.to_dict())}", flush=True)

    pm, proc, tok, _, new_ids, n_base = GC.load_policy(ck, device, a.attn)
    grammar = AG.load_grammar(tok, data_dir=data_dir)   # 수정(2026-10-16): 설계 검토 반영 — 표집·참조·정책 로그확률 모두 같은 제약 분포
    if a.max_new_tokens > grammar.n_pos:
        raise SystemExit(f"--max_new_tokens {a.max_new_tokens} > 문법 고정 길이 {grammar.n_pos}")
    print(f"[grammar] {grammar.info}", flush=True)
    summ = S.trainable_summary(pm)
    print(f"[model] {ck} 학습 파라미터 {summ}", flush=True)
    snap = GC.RefSnapshot(pm)                        # π_ref = SFT (재개 전에 찍는다)
    eos, pad = tok.convert_tokens_to_ids(GC.END_TOKEN), tok.pad_token_id
    named = GC.trainable_params(pm)
    lora_p = [p for n, p in named if "lora_" in n]
    tok_p = [p for n, p in named if "trainable_tokens" in n]
    if len(lora_p) + len(tok_p) != len(named):
        raise RuntimeError("예상 외 학습 파라미터")
    opt = torch.optim.AdamW([dict(params=lora_p, lr=a.lr, weight_decay=0.0),
                             dict(params=tok_p, lr=a.lr_tokens or a.lr, weight_decay=0.0)], betas=(0.9, 0.999), eps=1e-8)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.warmup_steps))

    out.mkdir(parents=True, exist_ok=True)
    gmeta = {k: v for k, v in meta.items() if k not in ("step", "holdout", "args", "trainable")}
    gmeta.update(sft_args=meta.get("args"), grpo=dict(args=vars(a), reward_cfg=cfg.to_dict(), sft_ckpt=str(ck), data=info,
                                                     trainable=summ, tag=GC.TAG,
                                                     sft_meta_sha256=S.file_sha256(ck / "teacher_meta.json"),  # 수정(2026-10-16): 설계 검토 반영
                                                     chain=chain, answer_grammar=grammar.info))
    S.save_meta(out / "teacher_meta.json", **gmeta)
    if not (out / "processor").exists():
        proc.save_pretrained(out / "processor")

    step, epoch, cursor, best = 0, 0, 0, -1.0
    if a.resume and (out / "latest" / "trainer_state.pt").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        st = torch.load(out / "latest" / "trainer_state.pt", map_location="cpu", weights_only=False)
        if st["sft_ckpt"] != str(ck):
            raise SystemExit(f"재개 상태의 SFT 체크포인트 {st['sft_ckpt']} ≠ {ck}")
        set_peft_model_state_dict(pm, load_file(str(out / "latest" / "adapter_model.safetensors")))
        for p in pm.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"])
        step, epoch, cursor, best = st["step"], st["epoch"], st["cursor"], st["best"]
        random.setstate(st["py_rng"]); np.random.set_state(st["np_rng"]); torch.set_rng_state(st["torch_rng"])
        print(f"[resume] step {step} epoch {epoch} cursor {cursor} best {best:.4f}", flush=True)

    def save(tag, extra=None):
        d = out / tag
        pm.save_pretrained(d)
        tok.save_pretrained(d)
        S.save_meta(d / "teacher_meta.json", **gmeta, step=step, **(extra or {}))
        return d

    def order(ep):
        return np.random.RandomState(a.seed + 1000 * ep).permutation(len(tr)).tolist()

    log_f = open(out / "train_log.jsonl", "a")

    def log(rec):
        print(json.dumps(rec, ensure_ascii=False), flush=True); log_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); log_f.flush()

    last_improve = step  # 수정(2026-10-18): 조기 종료 기준 step
    if a.start_gate and step == 0:  # 수정(2026-10-18): 시작 조건 — SFT 상태 holdout(형식 유효율 1.0 필요), 기준선 = best
        pm.eval()
        ev0 = run_holdout(pm, proc, tok, meta, ho, ctxs, codebook, cfg, a, device, grammar)
        ev0.update(step=0, kind="holdout_start")
        print(json.dumps(ev0, ensure_ascii=False), flush=True)
        with open(out / "train_log.jsonl", "a") as f0:
            f0.write(json.dumps(ev0, ensure_ascii=False) + "\n")
        if ev0["format_valid"] < 1.0:
            raise SystemExit(f"[시작 조건 불충족] SFT holdout 형식 유효율 {ev0['format_valid']:.4f} < 1.0 — GRPO 시작 안 함")
        best = ev0["reward"]
        save("best", dict(holdout=ev0))
    perm = order(epoch)
    pm.train(); S.base_of(pm).visual.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start, step0 = time.time(), step
    while step < a.steps:
        ts = time.time()
        # ---- 1) 프롬프트
        batch = []
        while len(batch) < P:
            if cursor >= len(perm):
                epoch += 1; cursor = 0; perm = order(epoch)
            batch.append(tr[perm[cursor]]); cursor += 1
        prompts = [GC.encode_prompt(proc, r, meta) for r in batch]
        feats = [GC.image_features(pm, x["pixel_values"], x["grid"]) for x in prompts]
        t_enc = time.time() - ts
        # ---- 2) 표집
        gen_rng = torch.Generator(device=device).manual_seed(a.seed * 1_000_003 + step)
        tg = time.time()
        smp = GC.sample_answers(pm, prompts, feats, G, a.max_new_tokens, eos, pad, a.temperature, a.top_p, generator=gen_rng,
                                grammar=grammar)   # 수정(2026-10-16): 설계 검토 반영
        t_gen = time.time() - tg
        ids, n_gen, ended = smp["ids"], smp["n_gen"], smp["ended"]
        R_, T = ids.shape
        # ---- 3) 보상·이점
        rew = np.zeros((P, G)); comps = {k: [] for k in ("perc", "map", "act")}; n_fmt = 0; errs = {}
        for i in range(R_):
            r = batch[i // G]
            n = int(n_gen[i]); e = bool(ended[i])
            text = tok.decode(ids[i, :n - 1 if e else n].tolist(), skip_special_tokens=False)
            Rv, d = GC.score_answer(text, e, r, codebook, ctxs.get(r["token"]), cfg)
            rew[i // G, i % G] = Rv; n_fmt += int(d["fmt"] > 0)
            for k in comps:
                if d[k] is not None:
                    comps[k].append(d[k])
            if "err" in d:
                k = d["err"].split(":")[0]; errs[k] = errs.get(k, 0) + 1
        adv = GC.group_advantages(rew, a.adv_eps).reshape(-1)
        zero_std = float((np.std(rew, axis=1) == 0).mean())
        # ---- 4) 감독 마스크(GT 위치 규칙 ∧ 생성 길이 안)
        mask = torch.zeros((R_, T), dtype=torch.bool)
        for i in range(R_):
            gm = GC.gt_position_mask(tok, batch[i // G], T, a.pg_exclude_intent)
            mask[i] = torch.as_tensor(gm) & (torch.arange(T) < int(n_gen[i]))
        rows_p = [prompts[i // G] for i in range(R_)]
        rows_f = [feats[i // G] for i in range(R_)]
        denom = dict(kind="seq", value=float(R_)) if a.loss_norm == "seq" else dict(kind="token", value=float(mask.sum().clamp(min=1)))
        # ---- 5) 참조 로그확률(SFT 값, no_grad)
        tr_ = time.time()
        lp_ref = torch.zeros((R_, T))
        pm.eval()
        with snap.active(pm), torch.no_grad():
            for s in range(0, R_, a.micro):
                sl = slice(s, s + a.micro)
                lp_ref[sl] = GC.answer_logprobs(pm, rows_p[sl], rows_f[sl], ids[sl], n_gen[sl], a.temperature,
                                                grammar=grammar).cpu()   # 수정(2026-10-16): 설계 검토 반영
        pm.train(); S.base_of(pm).visual.eval()
        t_ref = time.time() - tr_
        if not torch.isfinite(lp_ref[mask]).all():
            raise FloatingPointError(f"step {step}: 참조 로그확률 비유한 {int((~torch.isfinite(lp_ref[mask])).sum())}개 "
                                     f"(토큰 {[r['token'] for r in batch]}) — 원인 진단 필요(건너뛰지 않음)")
        # ---- 6) 정책 순전파·역전파
        tp = time.time()
        opt.zero_grad(set_to_none=True)
        agg = dict(kl_sum=0.0, n_tok=0.0, kl_max=0.0, pg_sum=0.0)
        loss_tot, cons = 0.0, []
        for s in range(0, R_, a.micro):
            sl = slice(s, s + a.micro)
            lp = GC.answer_logprobs(pm, rows_p[sl], rows_f[sl], ids[sl], n_gen[sl], a.temperature, grammar=grammar)  # 수정(2026-10-16): 설계 검토 반영
            m = mask[sl].to(lp.device)
            if not torch.isfinite(lp[m]).all():
                raise FloatingPointError(f"step {step}: 정책 로그확률 비유한 (행 {s}..) — 원인 진단 필요(건너뛰지 않음)")
            loss, st = GC.grpo_loss(lp, lp_ref[sl].to(lp.device), adv[sl].to(lp.device), m, a.beta, denom)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"step {step}: 비유한 손실 {loss.item()} (보상 {rew.tolist()}, 이점 {adv.tolist()}) — 원인 진단 필요")
            loss.backward()
            loss_tot += float(loss)
            for k in ("kl_sum", "n_tok", "pg_sum"):
                agg[k] += st[k]
            agg["kl_max"] = max(agg["kl_max"], st["kl_max"])
            with torch.no_grad():   # on-policy 일치 진단: 표집 시점 log π vs 학습 순전파 log π (같은 파라미터)
                v = (lp.detach().cpu() - smp["logp"][sl])[mask[sl]]
                if v.numel():
                    cons.append(float(v.abs().max()))
            del lp, loss
        t_pol = time.time() - tp
        gn_l, gn_t = GC.grad_norm(lora_p), GC.grad_norm(tok_p)
        gn = torch.nn.utils.clip_grad_norm_(lora_p + tok_p, a.clip)
        if not torch.isfinite(gn):
            raise FloatingPointError(f"step {step}: 비유한 기울기 노름 (lora {gn_l}, tokens {gn_t}) — 원인 진단 필요(건너뛰지 않음)")
        opt.step(); sch.step()
        step += 1
        el = time.time() - ts
        mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        mstd = lambda x: (float(np.mean(x)), float(np.std(x))) if x else (None, None)  # noqa: E731
        rec = dict(step=step, epoch=epoch, loss=loss_tot, reward_mean=float(rew.mean()), reward_std=float(rew.std()),
                   reward_group_std_mean=float(rew.std(axis=1, ddof=1).mean()), zero_std_groups=zero_std,
                   **{f"r_{k}_mean": mstd(v)[0] for k, v in comps.items()}, **{f"r_{k}_std": mstd(v)[1] for k, v in comps.items()},
                   format_valid=n_fmt / R_, ended=float(ended.float().mean()), gen_len_mean=float(n_gen.float().mean()),
                   errors=errs, kl=agg["kl_sum"] / max(agg["n_tok"], 1), kl_max=agg["kl_max"], sup_tokens=agg["n_tok"],
                   adv_absmax=float(adv.abs().max()), gnorm_lora=gn_l, gnorm_tokens=gn_t, gnorm_total=float(gn),
                   logp_consistency_maxabs=max(cons) if cons else None,
                   lr=sch.get_last_lr()[0], lr_tokens=sch.get_last_lr()[1], s_step=el, s_encode=t_enc, s_gen=t_gen,
                   s_ref=t_ref, s_policy=t_pol, peak_gb=round(mem, 2),
                   eta_h=(a.steps - step) * (time.time() - t_start) / max(step - step0, 1) / 3600)
        log(rec)
        del prompts, feats, smp, rows_p, rows_f
        if step % a.holdout_every == 0 or step == a.steps:
            pm.eval()
            ev = run_holdout(pm, proc, tok, meta, ho, ctxs, codebook, cfg, a, device, grammar)  # 수정(2026-10-16): 설계 검토 반영
            pm.train(); S.base_of(pm).visual.eval()
            ev.update(step=step)
            log(ev)
            if ev["reward"] > best:
                best = ev["reward"]
                save("best", dict(holdout=ev))
                last_improve = step
            elif a.early_stop_patience and step - last_improve >= a.early_stop_patience:  # 수정(2026-10-18): 정체 → 종료
                log(dict(kind="early_stop", step=step, best=best, last_improve=last_improve))
                break
        if step % a.save_every == 0 or step == a.steps:
            d = save("latest")
            torch.save(dict(opt=opt.state_dict(), sch=sch.state_dict(), step=step, epoch=epoch, cursor=cursor, best=best,
                            sft_ckpt=str(ck), py_rng=random.getstate(), np_rng=np.random.get_state(),
                            torch_rng=torch.get_rng_state()), d / "trainer_state.pt")
    save("final")
    summary = dict(steps=step, wall_s=time.time() - t_start, best_holdout_reward=best,
                   ref_param_maxabs_diff=snap.max_abs_diff(pm),
                   peak_gb=(torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None))
    json.dump(summary, open(out / "train_summary.json", "w"), indent=1)
    print("[done]", json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    main()
