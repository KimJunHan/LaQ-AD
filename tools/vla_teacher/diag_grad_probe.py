# 수정(2026-10-16): SFT 발산 원인 진단 — 저장된 학습 상태(latest)에서 특정 optimizer step 의 배치를 표본 단위로 다시
#   통과시켜, 기울기가 어느 층·어느 위치·어느 토큰에서 커지는지와 SDPA 커널(기본 vs math)·유한차분으로 그 기울기가
#   실제 함수의 기울기인지(수치 오류가 아닌지) 확인한다. 학습 코드와 같은 함수(build_model, TeacherSFTDataset, sft_loss)를 쓴다.
"""사용 예:
  CUDA_VISIBLE_DEVICES=1 python tools/vla_teacher/diag_grad_probe.py --fold A --ckpt results/vla_teacher/diag_foldA_ckpt/latest \
      --steps 34 --sdpa default math
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_sft as T  # noqa: E402
from sft_common import (PIXELS, TeacherSFTDataset, LengthGroupedBatchSampler, answer_lengths, collate,  # noqa: E402
                        load_records, load_special_tokens, make_processor, require_gpu_env, resolve_data_dir, sft_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="train_sft 의 <out>/latest (adapter + trainer_state.pt)")
    ap.add_argument("--fold", default="A")
    ap.add_argument("--steps", type=int, nargs="+", required=True, help="조사할 optimizer step(1부터). step s = 마이크로배치 2(s-1), 2(s-1)+1")
    ap.add_argument("--sdpa", nargs="+", default=["default"], choices=["default", "math"])
    ap.add_argument("--samples", type=int, nargs="*", default=None, help="배치 안 표본 색인(기본 전부)")
    ap.add_argument("--fd", action="store_true", help="임베딩 기울기 방향 유한차분 검사")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    require_gpu_env()
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    args = T.parse_args(["--fold", a.fold])
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_dir = resolve_data_dir(args.data_dir)
    toks, _ = load_special_tokens(args.special_tokens, data_dir)
    recs = load_records(data_dir, "train")
    tr, *_ = T.select_train_holdout(args, recs)
    proc = make_processor(args.base, PIXELS)
    tok = proc.tokenizer
    pm, new_ids, n_base = T.build_model(args, tok, toks, device)
    sd = load_file(os.path.join(a.ckpt, "adapter_model.safetensors"))
    res = set_peft_model_state_dict(pm, sd)
    print("[load] unexpected", len(res.unexpected_keys), flush=True)
    for p in pm.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    base = pm.get_base_model()
    base.gradient_checkpointing_disable()
    pm.eval()                                   # LoRA dropout 끔(결정적), 기울기는 계산
    action_ids = [i for t, i in zip(toks, new_ids) if t.startswith("<action_")]
    ds = TeacherSFTDataset(tr, proc, args.system, 0.0, 0.0, args.seed, max_answer_tokens=args.max_answer_tokens,
                           action_ids=action_ids, use_prev=True, frame_order=args.frame_order, with_segments=True)
    sampler = LengthGroupedBatchSampler(answer_lengths(tok, tr), args.batch, seed=args.seed)
    sampler.set_epoch(0)
    batches = sampler.batches()
    layers = base.model.layers
    tok_p = [p for n, p in pm.named_parameters() if p.requires_grad and "trainable_tokens" in n][0]
    named = dict(pm.named_parameters())
    q0 = [p for n, p in named.items() if "layers.0.self_attn.q_proj.lora_B" in n][0]
    results = []
    for s in a.steps:
        for mb in (2 * (s - 1), 2 * (s - 1) + 1):
            idxs = batches[mb]
            for j, i in enumerate(idxs):
                if a.samples is not None and j not in a.samples:
                    continue
                item = ds[i]
                b = collate([item], tok.pad_token_id)
                b = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in b.items()}
                ids = b["input_ids"][0].tolist()
                for mode in a.sdpa:
                    pm.zero_grad(set_to_none=True)
                    store = {}
                    hooks = []
                    for li, L in enumerate(layers):
                        def fh(mod, inp, out, li=li):
                            h = out[0]; h.retain_grad(); store[li] = h
                        hooks.append(L.register_forward_hook(fh))
                    for li in (0, 1, 13, 27):                 # q/k 헤드별 노름(RoPE 는 노름 보존) → |로짓| 상한 = |q||k|/√d
                        at = layers[li].self_attn
                        def qh(mod, inp, out, li=li):
                            store[f"q{li}"] = out.detach().view(out.shape[0], out.shape[1], -1, 128).norm(dim=-1).max().item()
                        def kh(mod, inp, out, li=li):
                            store[f"k{li}"] = out.detach().view(out.shape[0], out.shape[1], -1, 128).norm(dim=-1).max().item()
                        hooks.append(at.q_proj.register_forward_hook(qh)); hooks.append(at.k_proj.register_forward_hook(kh))
                    def eh(mod, inp, out):
                        out.retain_grad(); store["emb"] = out
                    hooks.append(base.model.embed_tokens.register_forward_hook(eh))
                    mk_ctx = lambda: (sdpa_kernel([SDPBackend.MATH]) if mode == "math" else sdpa_kernel(  # noqa: E731
                        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]))
                    with mk_ctx():
                        loss, n, st = sft_loss(pm, b, args.rope, diag_n_base=n_base)
                        loss.backward()
                    for h in hooks:
                        h.remove()
                    ge = store["emb"].grad[0].norm(dim=-1)
                    top = ge.topk(5)
                    per_layer = []
                    for li in range(len(layers)):
                        g = store[li].grad[0].norm(dim=-1)
                        k = int(g.argmax())
                        per_layer.append((li, float(g.max()), k, tok.convert_ids_to_tokens(ids[k]), float(g.median())))
                    rows = tok_p.grad.norm(dim=-1)
                    tr_ = rows.topk(3)
                    r = dict(step=s, micro=mb, j=j, token=item["token"], sdpa=mode, loss=loss.item(), n=n,
                             g_tok=float(rows.norm()), g_q0B=float(q0.grad.norm()),
                             tok_rows_top=[(tok.convert_ids_to_tokens(n_base + int(t)), float(v)) for v, t in zip(tr_.values, tr_.indices)],
                             emb_top=[(int(p), tok.convert_ids_to_tokens(ids[int(p)]), float(v)) for v, p in zip(top.values, top.indices)],
                             emb_med=float(ge.median()), layers=per_layer, n_prompt=item["n_prompt"],
                             qk_logit_bound={li: store[f"q{li}"] * store[f"k{li}"] / math.sqrt(128) for li in (0, 1, 13, 27)})
                    if a.fd:
                        # 손실을 임베딩 출력의 함수로 보고 기울기 방향 유한차분: (L(e+εd)−L(e−εd))/2ε vs ⟨g,d⟩
                        g = store["emb"].grad.detach()
                        d = g / g.norm()
                        import sft_common as C
                        fds = []
                        with torch.no_grad(), mk_ctx():
                            # 영상 특징을 끼워 넣은 실제 입력 임베딩(영상 위치의 기울기는 0 이라 d 도 0)
                            e0 = C.embed_inputs(pm, b["input_ids"], b.get("pixel_values"), b.get("image_grid_thw"))
                            for eps in (1e-1, 3e-2, 1e-2):
                                vals = []
                                for sg in (1, -1):
                                    emb = e0 + sg * eps * d
                                    m_img = b["input_ids"] == base.config.image_token_id
                                    pos = C.position_ids_for(pm, b["input_ids"], b.get("image_grid_thw"), b["attention_mask"], args.rope)
                                    out = base.model(input_ids=None, inputs_embeds=emb, attention_mask=b["attention_mask"],
                                                     position_ids=pos, use_cache=False, return_dict=True)
                                    hh = out.last_hidden_state
                                    lab = b["labels"][:, 1:]; w = b["weights"][:, 1:]; mm = lab != -100
                                    lg = base.lm_head(hh[:, :-1][mm]).float()
                                    ce = torch.nn.functional.cross_entropy(lg, lab[mm], reduction="none")
                                    vals.append(float((ce * w[mm]).sum() / mm.sum()))
                                fds.append((eps, (vals[0] - vals[1]) / (2 * eps)))
                        r["fd"] = dict(grad_norm=float(g.norm()), fd=fds)
                    print(json.dumps(dict((k, v) for k, v in r.items() if k != "layers")), flush=True)
                    print("   layers(max,pos,tok,median):", " ".join(f"{li}:{m:.3g}@{p}{t}/{md:.2g}" for li, m, p, t, md in per_layer), flush=True)
                    results.append(r)
                    del store
                    torch.cuda.empty_cache()
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
