# 수정(2026-10-16): 교사 GRPO — CPU 전용 스모크 검사(실행: CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=.).
#   A) 보상: GT 정답 = 최대(1.0), 손상 정답 = 사이 값(0<R<1), 형식 오류 = 최소(0), 충돌·실현가능성 항, 범위 [0,1]
#   B) 그룹 이점(평균 0·표준편차 1·전부 같으면 0), k3 KL(≥0, π=π_ref 면 0, 몬테카를로 불편성), GRPO 손실 기울기(해석식 대조)
#   C) 소형 무작위 Qwen2-VL(smoke_sft_cpu.py 와 같은 축소 config)에 가짜 SFT 어댑터를 저장 → grpo_common.load_policy 로 적재:
#      어댑터 값 일치, 표집(KV 캐시·M-RoPE·왼쪽 패딩) log π == 학습 순전파(오른쪽 패딩) log π, sft_common.lm_forward 와 동일,
#      참조 교체: 초기 KL 정확히 0·교체 후 정책 값 비트 복원, 1 회 갱신: LoRA·새 토큰 행만 변화(기존 어휘 행·비전·base 불변), NaN 없음
#   D) train_grpo.main 종단 실행(소형 모델, --smoke) 2 step + 재개 1 step
# 수정(2026-10-16): 설계 검토 반영:
#   E) fp32 정확 어텐션: grpo_common.load_policy 가 정책 적재 직후 sft_common.exact_fp32_attention(torch.float32) 를 부르고,
#      train_grpo.main 은 시작 때 한 번 더(순서: 시작 → 정책 구성 → 적재 후) 부른다(가로채기 함수로 호출·인자·순서 확인).
#   F) 문법 제약 표집: 생성 토큰 전부 문법 수용, 표집 log q == 학습 순전파 log q(정규화), log q ≥ log π, 허용 1 개 위치 log q = 0,
#      548 토큰 전체 생성 → 파싱·고정 위치 일치, 제약 손실 기울기 유한. QA 단계 meta(qa_meta.json)를 둔 base 로 연결 검사.
import copy
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import sft_common as S  # noqa: E402
import grpo_common as GC  # noqa: E402
import answer_grammar as AG  # noqa: E402  # 수정(2026-10-16): 설계 검토 반영
import laq_cache_layout as LY  # noqa: E402  # 수정(2026-10-16): 설계 검토 반영

ATTN_EVENTS = []   # 수정(2026-10-16): 설계 검토 반영 — exact_fp32_attention 호출 기록(가로채기)
_orig_exact = S.exact_fp32_attention
_orig_build = GC.build_policy_from_base


def _spy_exact(dtype):
    ATTN_EVENTS.append(("attn", dtype))
    return _orig_exact(dtype)


def _spy_build(*a, **kw):
    ATTN_EVENTS.append(("policy_built", None))
    return _orig_build(*a, **kw)


S.exact_fp32_attention = _spy_exact
GC.build_policy_from_base = _spy_build
import common_v2 as V2  # noqa: E402

RESULTS = {}


def ok(name, cond, info=""):
    RESULTS[name] = dict(ok=bool(cond), info=str(info))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    return cond


# ---------------------------------------------------------------- A) 보상
def reward_tests(cb, cfg):
    val = S.load_records(S.resolve_data_dir(None), "val", max_n=600)
    t0 = time.time()
    ctx = GC.build_action_contexts(val, str(S.REPO / "data/infos/nuscenes_infos_val.pkl"), cb, cfg)
    ok("ctx_built", len(ctx) == sum(S.future_complete(r) for r in val), f"{len(ctx)} future_complete, {time.time() - t0:.1f}s")
    R = [GC.score_answer(r["answer"], True, r, cb, ctx.get(r["token"]), cfg)[0] for r in val]
    kinds = dict(fc=sum(S.future_complete(r) for r in val), map_none=sum(r["targets"]["map"] is None for r in val),
                 no_prev=sum(not S.prev_available(r) for r in val))
    ok("gt_answer_max_reward", min(R) == 1.0 and max(R) == 1.0, f"{len(R)} 표본 전부 1.0 ({kinds})")

    r = next(x for x in val if S.future_complete(x) and x["targets"]["map"] is not None and len(x["targets"]["objects"]) >= 4
             and all(o.get("motion") != "na" for o in x["targets"]["objects"]))
    c = ctx[r["token"]]
    base = V2.parse_answer(r["answer"])

    def sc(p):
        txt = V2.format_parsed(p)
        V2.parse_answer(txt)                       # 손상본도 문법상 유효해야 '사이 값' 검사가 의미 있다
        return GC.score_answer(txt, True, r, cb, c, cfg)

    def mod(f):
        p = copy.deepcopy(base); f(p); return sc(p)
    cases = {
        "obj_shift_3m": lambda p: p["objects"][0].update(lat=round(p["objects"][0]["lat"] + 3.0, 1)),
        "obj_class": lambda p: p["objects"][1].update(name="bus" if p["objects"][1]["name"] != "bus" else "car"),
        "obj_dropped": lambda p: p["objects"].pop(),
        "obj_future": lambda p: p["objects"][0].update(fut=[[f[0] + 2.0, f[1]] if f else None for f in p["objects"][0]["fut"]]),
        "obj_ttc": lambda p: p["objects"][0].update(ttc="lt2" if p["objects"][0]["ttc"] != "lt2" else "safe"),
        "map_counts": lambda p: p["map"].update(n_divider=p["map"]["n_divider"] + 2),
        "map_lane": lambda p: p["map"]["lanes"]["10"].update(bnd_l=None if p["map"]["lanes"]["10"]["bnd_l"] is not None else 3.0),
        "action_random": lambda p: p.update(action=[int(k) for k in np.random.RandomState(0).randint(0, 2041, 6)]),
        "action_na": lambda p: p.update(action=None),
        "map_na": lambda p: p.update(map=None),
    }
    vals = {}
    for k, f in cases.items():
        v, d = mod(f)
        vals[k] = round(v, 4)
    ok("corrupted_lower", all(0.0 < v < 1.0 for v in vals.values()), vals)
    ok("format_bonus_floor", all(v >= cfg.w_fmt for v in vals.values()), f"유효 정답 ≥ w_fmt={cfg.w_fmt}")
    mal = {"garbage": GC.score_answer("garbage", True, r, cb, c, cfg)[0],
           "truncated": GC.score_answer(r["answer"][:-9], True, r, cb, c, cfg)[0],
           "no_end_token": GC.score_answer(r["answer"], False, r, cb, c, cfg)[0],
           "bad_header": GC.score_answer(r["answer"].replace("Map:", "Mapp:"), True, r, cb, c, cfg)[0],
           "extra_line": GC.score_answer(r["answer"] + "\nMore.", True, r, cb, c, cfg)[0]}
    ok("malformed_min_reward", all(v == 0.0 for v in mal.values()), mal)
    # 인지: 예측 0 개(GT 있음) → 0, 완전 → 1
    ok("perception_empty_pred_zero", GC.perception_reward([], r["targets"]["objects"], cfg)[0] == 0.0)
    # 행동: GT 궤적 그대로 → 1 / 미래 박스 중심으로 돌진 → 충돌 0 / 순간이동 → 실현가능 < 1
    ra, da = c.reward(c.gt, cfg)
    ok("action_gt_traj_max", abs(ra - 1.0) < 1e-12 and da["col"] == 1.0 and da["feas"] == 1.0, da)
    hit = None
    for rr in val:
        cc = ctx.get(rr["token"])
        if cc is None or any(len(b) == 0 for b in cc.fut):
            continue
        j = int(np.argmin([np.hypot(*b[:2]) for b in cc.fut[0]]))
        tr = np.array([cc.fut[t][min(j, len(cc.fut[t]) - 1)][:2] for t in range(6)], np.float64)
        rv, dv = cc.reward(tr, cfg)
        if dv["collide"]:
            hit = (rr["token"], rv, dv); break
    ok("action_collision_penalised", hit is not None and hit[2]["col"] == 0.0 and hit[1] < 1.0, hit and dict(r=round(hit[1], 3), **hit[2]))
    tele = c.gt.copy(); tele[2:] += np.array([0.0, 40.0])
    rt, dt_ = c.reward(tele, cfg)
    ok("action_infeasible_penalised", dt_["feas"] < 1.0 and rt < 1.0, dt_)
    # 실현가능 한계가 GT(코드북 복호)에서 거의 항상 통과하는지(train 아님 — val 600 표본 진단만; 한계 근거는 train 통계)
    fe = np.mean([cc.or_feas.all() for cc in ctx.values()])
    ok("feasibility_oracle_pass_rate", fe > 0.98, f"oracle 전 검사 통과 비율 {fe:.4f}")
    # 무작위 정답 손상 100 개 범위 검사
    rng = np.random.RandomState(1)
    rs = []
    for rr in val[:100]:
        p = V2.parse_answer(rr["answer"])
        for o in p["objects"]:
            o["lat"] = round(float(np.clip(o["lat"] + rng.randn() * 2, -99.9, 99.9)), 1)
        if p["action"] is not None:
            p["action"] = [int(k) for k in rng.randint(0, 2041, 6)]
        rs.append(GC.score_answer(V2.format_parsed(p), True, rr, cb, ctx.get(rr["token"]), cfg)[0])
    ok("reward_range_random_corruptions", all(0.0 <= x <= 1.0 for x in rs) and np.mean(rs) < 1.0,
       f"min {min(rs):.3f} mean {np.mean(rs):.3f} max {max(rs):.3f}")
    return val


# ---------------------------------------------------------------- B) 이점·KL·손실
def math_tests():
    A = GC.group_advantages([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5], [0.2, 0.9, 0.4, 0.7]])
    ok("adv_group_mean0_std1", torch.allclose(A[[0, 2]].mean(1), torch.zeros(2), atol=1e-6)
       and torch.allclose(A[[0, 2]].std(1, unbiased=True), torch.ones(2), atol=1e-3), A.tolist())
    ok("adv_equal_group_zero", bool((A[1] == 0).all()))
    ok("adv_hand_value", abs(float(A[0, 0]) - 0.75 / (0.5 + 1e-4)) < 1e-4, f"{float(A[0, 0]):.5f} (=0.75/0.5)")
    try:
        GC.group_advantages([[float("nan"), 1.0]]); nan_raise = False
    except FloatingPointError:
        nan_raise = True
    ok("adv_nonfinite_raises", nan_raise)
    g = torch.Generator().manual_seed(0)
    lp, lr = -torch.rand(1000, generator=g) * 5, -torch.rand(1000, generator=g) * 5
    k = GC.kl_k3(lp, lr)
    ok("kl_k3_nonneg", bool((k >= 0).all()), f"min {float(k.min()):.2e}")
    ok("kl_k3_zero_when_equal", bool((GC.kl_k3(lp, lp) == 0).all()))
    # 불편성: x ~ π, E[k3] = KL(π‖ref)
    lg_p, lg_r = torch.randn(50, generator=g), torch.randn(50, generator=g)
    p, q = torch.softmax(lg_p, 0), torch.softmax(lg_r, 0)
    x = torch.multinomial(p, 400000, replacement=True, generator=g)
    est = float(GC.kl_k3(torch.log(p[x]), torch.log(q[x])).mean())
    true = float((p * (p.log() - q.log())).sum())
    ok("kl_k3_unbiased_mc", abs(est - true) / true < 0.02, f"MC {est:.4f} vs KL {true:.4f}")
    # 손실 기울기 해석식: β=0 → ∂L/∂logp_t = −A_i/(N·|o_i|) (seq 정규화)
    logp = (-torch.rand(3, 5, generator=g)).requires_grad_(True)
    adv = torch.tensor([1.0, -0.5, 0.0])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], dtype=torch.bool)
    loss, _ = GC.grpo_loss(logp, logp.detach(), adv, mask, 0.0, dict(kind="seq", value=3.0))
    loss.backward()
    want = torch.zeros(3, 5)
    for i in range(3):
        want[i, mask[i]] = -adv[i] / (3.0 * mask[i].sum())
    ok("loss_grad_analytic_pg", torch.allclose(logp.grad, want, atol=1e-7), logp.grad[0].tolist())
    # KL 항 기울기: ∂k3/∂logp = 1 − exp(logp_ref − logp)
    logp2 = (-torch.rand(2, 4, generator=g)).requires_grad_(True)
    ref2 = -torch.rand(2, 4, generator=g)
    m2 = torch.ones(2, 4, dtype=torch.bool)
    loss2, st2 = GC.grpo_loss(logp2, ref2, torch.zeros(2), m2, 0.5, dict(kind="token", value=8.0))
    loss2.backward()
    ok("loss_grad_analytic_kl", torch.allclose(logp2.grad, 0.5 * (1 - torch.exp(ref2 - logp2.detach())) / 8.0, atol=1e-7),
       f"kl_mean {st2['kl_mean']:.4f}")


# ---------------------------------------------------------------- C) 소형 모델
def make_tiny_sft(tmp, toks, recs_A, fold_doc):
    """소형 base(실제 config 축소) 저장 + 가짜 'SFT' 어댑터(LoRA B·새 토큰 행에 소잡음 → 비자명) 저장. meta 는 train_sft 형식."""
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
    cfg = Qwen2VLConfig.from_pretrained(S.BASE_QWEN)
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 64, 128, 2
    cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
    cfg.rope_scaling = {"type": "mrope", "rope_type": "default", "mrope_section": [2, 3, 3]}
    vc = cfg.vision_config
    vc.depth, vc.embed_dim, vc.num_heads, vc.hidden_size, vc.mlp_ratio = 1, 32, 2, 64, 2
    cfg.tie_word_embeddings = True
    torch.manual_seed(0)
    m = Qwen2VLForConditionalGeneration(cfg).float()
    m.tie_weights()
    bdir = os.path.join(tmp, "base")
    m.save_pretrained(bdir)
    proc0 = S.make_processor(S.BASE_QWEN)
    proc0.save_pretrained(bdir)
    # 수정(2026-10-16): 설계 검토 반영 — 소형 base 를 'QA 단계 산출'로 표시(연결 검사: 원본 Qwen 이 아닌 base 는 qa_meta 필수)
    S.save_meta(os.path.join(bdir, S.QA_META), stage="teacher_qa_full", fold="A", fold_sha256=fold_doc["sha256"],
                base=S.BASE_QWEN, smoke_note="tiny random base for smoke_grpo_cpu.py")
    # SFT 와 같은 구성으로 어댑터 생성
    m = Qwen2VLForConditionalGeneration.from_pretrained(bdir, torch_dtype=torch.float32, attn_implementation="sdpa")
    proc = S.make_processor(bdir)
    tok = proc.tokenizer
    new_ids, n0 = S.add_special_tokens(tok, toks)
    S.resize_for_tokens(m, len(tok))
    S.init_new_rows(m, tok, toks, new_ids, n0, "desc")
    for p in m.parameters():
        p.requires_grad_(False)
    pm = S.build_peft(m, new_ids, 4, 8, 0.05, "attn_mlp")
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=g) * (0.05 if "lora_B" in n else 0.01))
    sdir = os.path.join(tmp, "sft", "best")
    pm.save_pretrained(sdir)
    saved = {n: p.detach().clone() for n, p in pm.named_parameters() if p.requires_grad}
    hold = fold_doc["holdout"]["A"]
    meta = dict(base=bdir, data_dir=str(S.resolve_data_dir(None)), special_tokens=toks, special_tokens_src="smoke",
                n_base_vocab=n0, new_ids=[new_ids[0], new_ids[-1]], vocab_after=len(tok), pixels=S.PIXELS, rope="mrope",
                system="qwen_default", dtype="fp32", use_prev=True, frame_order="prev_first", holdout_sha256=hold["sha256"],
                fold=dict(fold="A", fold_sha256=fold_doc["sha256"]), lora=dict(r=4, alpha=8, dropout=0.05, targets="attn_mlp"),
                new_token_init="desc", step=0,
                base_chain=S.base_chain_info(bdir, "A"))   # 수정(2026-10-16): 설계 검토 반영
    S.save_meta(os.path.join(sdir, "teacher_meta.json"), **meta)
    return sdir, saved


def tiny_tests(tmp, toks, recs_A, fold_doc, cb, cfg):
    sdir, saved = make_tiny_sft(tmp, toks, recs_A, fold_doc)
    ATTN_EVENTS.clear()
    pm, proc, tok, meta, new_ids, n0 = GC.load_policy(sdir, torch.device("cpu"))
    ok("attn_exact_called_after_policy_load", ATTN_EVENTS == [("policy_built", None), ("attn", torch.float32)],   # 수정(2026-10-16): 설계 검토 반영
       f"{ATTN_EVENTS} (CPU 에서는 'unchanged' 반환 — CUDA 에서 math 커널 고정·자기검사)")
    loaded = {n: p for n, p in pm.named_parameters() if p.requires_grad}
    ok("tiny_load_adapter_exact", set(loaded) == set(saved) and all(torch.equal(loaded[n], saved[n]) for n in saved),
       f"{len(saved)} 학습 텐서 비트 일치")
    ok("tiny_dropout_zero", all(m.p == 0.0 for m in pm.modules() if isinstance(m, torch.nn.Dropout)))
    ok("tiny_trainable_only_lora_tokens", all(("lora_" in n or "trainable_tokens" in n) and ".visual." not in n for n in loaded),
       S.trainable_summary(pm))
    eos, pad = tok.convert_tokens_to_ids(GC.END_TOKEN), tok.pad_token_id
    r0, r1 = recs_A[0], recs_A[1]
    prompts = [GC.encode_prompt(proc, r0, meta), GC.encode_prompt(proc, r1, meta, drop_cmd=True, drop_ego=True)]
    ok("tiny_prompt_12_images", int((prompts[0]["ids"] == 151655).sum()) == 12 * S.TOKENS_PER_CAM,
       f"길이 {[len(x['ids']) for x in prompts]} (서로 달라 왼쪽 패딩 경로 검사)")
    feats = [GC.image_features(pm, x["pixel_values"], x["grid"]) for x in prompts]
    G, TNEW = 3, 24
    gen = torch.Generator().manual_seed(0)
    t0 = time.time()
    smp = GC.sample_answers(pm, prompts, feats, G, TNEW, eos, pad, 1.0, 1.0, generator=gen)
    ok("tiny_sample_shapes", tuple(smp["ids"].shape) == (2 * G, TNEW) and bool(torch.isfinite(smp["logp"]).all()),
       f"{tuple(smp['ids'].shape)} {time.time() - t0:.1f}s, 새 토큰 비율 {float((smp['ids'] >= n0).float().mean()):.3f}")
    rows_p = [prompts[i // G] for i in range(2 * G)]
    rows_f = [feats[i // G] for i in range(2 * G)]
    lp = GC.answer_logprobs(pm, rows_p, rows_f, smp["ids"], smp["n_gen"])
    d = float((lp.detach() - smp["logp"]).abs().max())
    ok("tiny_sample_vs_train_logprob", d < 1e-4, f"max|Δlog π| = {d:.2e} (KV 캐시·M-RoPE 델타·왼/오른 패딩 일치)")
    ok("tiny_group_rows_differ", len({tuple(x) for x in smp["ids"][:G].tolist()}) > 1, "같은 프롬프트의 G 표집이 서로 다름")
    # sft_common.lm_forward(SFT 학습 경로)와 같은 로그확률
    x = rows_p[0]
    full = torch.cat([x["ids"], smp["ids"][0]])[None]
    b = dict(input_ids=full, attention_mask=torch.ones_like(full), pixel_values=x["pixel_values"], image_grid_thw=x["grid"])
    with torch.no_grad():
        h = S.lm_forward(pm, b, "mrope").last_hidden_state[0, len(x["ids"]) - 1:-1]
        lg = S.base_of(pm).lm_head(h).float()
        lp_sft = torch.log_softmax(lg, -1).gather(-1, smp["ids"][0][:, None])[:, 0]
    d2 = float((lp_sft - lp[0].detach()).abs().max())
    ok("tiny_logprob_equals_sft_forward", d2 < 1e-4, f"max|Δ| = {d2:.2e}")
    # 탐욕 표집도 같은 규칙
    gr = GC.sample_answers(pm, prompts[:1], feats[:1], 1, 8, eos, pad, greedy=True)
    lpg = GC.answer_logprobs(pm, prompts[:1], feats[:1], gr["ids"], gr["n_gen"])
    ok("tiny_greedy_consistent", float((lpg.detach() - gr["logp"]).abs().max()) < 1e-4)
    grammar_tests(pm, tok, prompts, feats, eos, pad)   # 수정(2026-10-16): 설계 검토 반영

    # 참조 교체·KL
    snap = GC.RefSnapshot(pm)
    with snap.active(pm), torch.no_grad():
        lp_ref = GC.answer_logprobs(pm, rows_p, rows_f, smp["ids"], smp["n_gen"])
    ok("tiny_initial_kl_exact_zero", torch.equal(lp_ref, lp.detach()) and float(GC.kl_k3(lp.detach(), lp_ref).abs().max()) == 0.0)

    # 1 회 갱신(합성 보상 — 소형 무작위 모델 출력은 전부 형식 오류(보상 0) 라 실제 보상으로는 이점이 0 이 된다)
    real = [GC.score_answer(tok.decode(smp["ids"][i, :int(smp["n_gen"][i])].tolist()), bool(smp["ended"][i]),
                            [r0, r1][i // G], cb, None, cfg)[0] for i in range(2 * G)]
    ok("tiny_real_reward_malformed_zero", all(v == 0.0 for v in real) and
       bool((GC.group_advantages(np.array(real).reshape(2, G)) == 0).all()), "무작위 모델 → 보상 0 → 이점 0(신호 없음)")
    rew = np.array([[0.9, 0.1, 0.5], [0.2, 0.6, 0.3]])
    adv = GC.group_advantages(rew).reshape(-1)
    mask = torch.zeros_like(smp["ids"], dtype=torch.bool)
    for i in range(2 * G):
        mask[i] = torch.as_tensor(GC.gt_position_mask(tok, [r0, r1][i // G], smp["ids"].shape[1])) & \
            (torch.arange(smp["ids"].shape[1]) < int(smp["n_gen"][i]))
    before = {n: p.detach().clone() for n, p in pm.named_parameters()}
    W_base0 = S.base_of(pm).model.embed_tokens.token_adapter.base_layer.weight.detach().clone()
    named = GC.trainable_params(pm)
    lora_p = [p for n, p in named if "lora_" in n]; tok_p = [p for n, p in named if "trainable_tokens" in n]
    opt = torch.optim.AdamW([dict(params=lora_p, lr=1e-3, weight_decay=0.0), dict(params=tok_p, lr=1e-3, weight_decay=0.0)])
    pm.train(); S.base_of(pm).visual.eval()
    opt.zero_grad()
    tot = 0.0
    for s in range(0, 2 * G, 2):
        sl = slice(s, s + 2)
        lpp = GC.answer_logprobs(pm, rows_p[sl], rows_f[sl], smp["ids"][sl], smp["n_gen"][sl])
        loss, st = GC.grpo_loss(lpp, lp_ref[sl], adv[sl], mask[sl], 0.04, dict(kind="seq", value=float(2 * G)))
        loss.backward(); tot += float(loss)
    gl, gt = GC.grad_norm(lora_p), GC.grad_norm(tok_p)
    ok("tiny_loss_grad_finite", np.isfinite(tot) and np.isfinite(gl) and np.isfinite(gt) and gl > 0 and gt > 0,
       f"loss {tot:.4e} |g_lora| {gl:.3e} |g_tok| {gt:.3e}")
    vis_grad = any(p.grad is not None for p in S.base_of(pm).visual.parameters())
    ok("tiny_vision_no_grad", not vis_grad)
    torch.nn.utils.clip_grad_norm_(lora_p + tok_p, 1.0)
    opt.step()
    after = dict(pm.named_parameters())
    changed = [n for n in before if not torch.equal(before[n], after[n].detach())]
    ok("tiny_update_only_trainable", len(changed) > 0 and all(after[n].requires_grad for n in changed)
       and any("lora_" in n for n in changed) and any("trainable_tokens" in n for n in changed),
       f"변화 {len(changed)}개 텐서, 전부 LoRA/토큰 행")
    W_base1 = S.base_of(pm).model.embed_tokens.token_adapter.base_layer.weight.detach()
    ok("tiny_base_vocab_rows_unchanged", torch.equal(W_base0, W_base1), "기존 어휘·base 임베딩 비트 불변")
    ok("tiny_no_nan_params", all(torch.isfinite(p).all() for p in pm.parameters()))
    post = {n: p.detach().clone() for n, p in GC.trainable_params(pm)}
    with snap.active(pm), torch.no_grad():
        lp_ref2 = GC.answer_logprobs(pm, rows_p, rows_f, smp["ids"], smp["n_gen"])
    ok("tiny_ref_unchanged_after_update", torch.equal(lp_ref2, lp_ref), "참조(SFT) 로그확률 불변")
    ok("tiny_policy_restored_after_ref", all(torch.equal(post[n], p.detach()) for n, p in GC.trainable_params(pm)))
    with torch.no_grad():
        lp_new = GC.answer_logprobs(pm, rows_p, rows_f, smp["ids"], smp["n_gen"])
    kl = GC.kl_k3(lp_new, lp_ref)[mask]
    ok("tiny_kl_positive_after_update", float(kl.mean()) > 0 and bool((kl >= 0).all()), f"KL {float(kl.mean()):.3e}")
    # 이점 방향: 높은 보상 표본의 log π 가 오르고 낮은 표본은 내려간다(평균)
    dl = ((lp_new - lp.detach()) * mask).sum(1) / mask.sum(1).clamp(min=1)
    corr = float(np.corrcoef(dl.numpy(), adv.numpy())[0, 1])
    ok("tiny_update_follows_advantage", corr > 0.5, f"corr(Δlog π, A) = {corr:.3f}")
    return sdir


def grammar_tests(pm, tok, prompts, feats, eos, pad):
    """수정(2026-10-16): 설계 검토 반영 — 문법 제약 표집과 정규화 로그확률."""
    g = AG.load_grammar(tok)
    was_training = pm.training
    G, TNEW = 3, 40
    smp = GC.sample_answers(pm, prompts, feats, G, TNEW, eos, pad, 1.0, 1.0, generator=torch.Generator().manual_seed(3), grammar=g)
    acc = [g.accepts(r.tolist())[0] for r in smp["ids"]]
    ok("grammar_sampled_tokens_all_allowed", all(acc), f"{sum(acc)}/{len(acc)} 행(앞 {TNEW} 토큰)")
    rows_p = [prompts[i // G] for i in range(2 * G)]
    rows_f = [feats[i // G] for i in range(2 * G)]
    lq = GC.answer_logprobs(pm, rows_p, rows_f, smp["ids"], smp["n_gen"], grammar=g)
    d = float((lq.detach() - smp["logp"]).abs().max())
    ok("grammar_sample_vs_train_logq", d < 1e-4, f"max|Δlog q| = {d:.2e} (표집·학습 모두 허용 집합 정규화)")
    lu = GC.answer_logprobs(pm, rows_p, rows_f, smp["ids"], smp["n_gen"]).detach()
    ge = bool((lq.detach() >= lu - 1e-5).all())
    gt = int((lq.detach() > lu + 1e-3).sum())
    ok("grammar_logq_ge_logpi", ge and gt > 0, f"log q ≥ log π 전 위치, 차이 >1e-3 위치 {gt}/{lu.numel()} (log Z ≤ 0)")
    single = [t for t in range(TNEW) if len(g.allowed_list(g.key(_state_at(g, smp["ids"][0], t)))) == 1]
    ok("grammar_single_choice_logq_zero", len(single) >= 3 and all(float(lq[0, t]) == 0.0 for t in single),
       f"허용 1 개 위치 {single[:6]}… log q = 0 (머리말·강제 채움)")
    # 제약 손실 기울기(유한, 0 아님)
    pm.train(); S.base_of(pm).visual.eval()
    pm.zero_grad(set_to_none=True)
    lqg = GC.answer_logprobs(pm, rows_p[:2], rows_f[:2], smp["ids"][:2], smp["n_gen"][:2], grammar=g)
    m = torch.ones_like(lqg, dtype=torch.bool)
    loss, _ = GC.grpo_loss(lqg, lq[:2].detach(), torch.tensor([1.0, -1.0]), m, 0.04, dict(kind="seq", value=2.0))
    loss.backward()
    gn = GC.grad_norm([p for _, p in GC.trainable_params(pm)])
    ok("grammar_loss_grad_finite", np.isfinite(float(loss)) and np.isfinite(gn) and gn > 0, f"|g| {gn:.3e}")
    pm.zero_grad(set_to_none=True); pm.eval()
    # 548 토큰 전체(정답 547 + <|im_end|>): 무작위 소형 모델도 항상 파싱·고정 위치 일치
    full = GC.sample_answers(pm, prompts[:1], feats[:1], 2, g.n_pos, eos, pad, 1.0, 1.0,
                             generator=torch.Generator().manual_seed(4), grammar=g)
    tpl = LY.answer_template(tok)
    res = []
    for i in range(2):
        ids = full["ids"][i, :int(full["n_gen"][i])].tolist()
        text, offs = LY.token_char_offsets(tok, ids[:-1])
        try:
            V2.parse_answer(text)
            pos = LY.folder_token_positions_from_text(text, offs)
            res.append(bool(full["ended"][i]) and len(ids) == g.n_pos and all(pos[k] == tpl[k] for k in LY.ANSWER_FOLDERS))
        except ValueError as e:
            res.append(False); print("   ", e)
    ok("grammar_full_answer_parses_positions_match", all(res), f"{res}, 끝 토큰 위치 548")
    pm.train(was_training)
    if was_training:
        S.base_of(pm).visual.eval()


def _state_at(g, ids, t):
    st = g.start()
    for i in ids[:t].tolist():
        g.step(st, i)
    return st


def mask_tests(tok, recs):
    fc = next(r for r in recs if S.future_complete(r) and S.prev_available(r))
    inc = next(r for r in recs if not S.future_complete(r))
    m_fc = GC.gt_position_mask(tok, fc, GC.ANSWER_TOKENS + 1)
    _, lab, _ = S.tokenize_answer(tok, fc)
    ok("mask_matches_sft_labels", int((~m_fc).sum()) == sum(l == -100 for l in lab), f"제외 {int((~m_fc).sum())} 위치(<na> 등)")
    m_inc = GC.gt_position_mask(tok, inc, GC.ANSWER_TOKENS + 1)
    seg = S.answer_segments(tok, inc)
    ia = [S.SEG_NAMES.index(k) for k in ("intent", "action", "end")]
    ok("mask_incomplete_future_excludes_intent_action", all(not m_inc[i] for i, s in enumerate(seg) if s in ia))
    m_noint = GC.gt_position_mask(tok, fc, GC.ANSWER_TOKENS + 1, exclude_intent=True)
    seg_fc = S.answer_segments(tok, fc)
    n_int = sum(1 for i, s in enumerate(seg_fc) if s == S.SEG_NAMES.index("intent") and m_fc[i])
    ok("mask_exclude_intent_flag", n_int > 0 and int(m_fc.sum() - m_noint.sum()) == n_int, f"Intent 제외 {n_int} 위치")


def e2e_tests(tmp):
    import train_grpo as TG
    out = os.path.join(tmp, "grpo")
    ATTN_EVENTS.clear()   # 수정(2026-10-16): 설계 검토 반영
    common = ["--fold", "A", "--sft_dir", os.path.join(tmp, "sft"), "--sft_ckpt", "best", "--out", out, "--device", "cpu",
              "--smoke", "--max_train_samples", "4", "--max_holdout_samples", "2", "--max_new_tokens", "12", "--group", "2",
              "--prompts_per_step", "2", "--holdout_every", "2", "--save_every", "1", "--micro", "2", "--eval_batch", "2"]
    t0 = time.time()
    s1 = TG.main(common + ["--steps", "2"])
    ok("e2e_attn_exact_startup_and_after_load",   # 수정(2026-10-16): 설계 검토 반영
       [e[0] for e in ATTN_EVENTS] == ["attn", "policy_built", "attn"] and all(e[1] == torch.float32 for e in ATTN_EVENTS if e[0] == "attn"),
       [e[0] for e in ATTN_EVENTS])
    logs = [json.loads(l) for l in open(os.path.join(out, "train_log.jsonl"))]
    st = [l for l in logs if "loss" in l]
    ho = [l for l in logs if l.get("kind") == "holdout"]
    keys = ["reward_mean", "reward_std", "r_perc_mean", "kl", "gnorm_lora", "gnorm_tokens", "s_step", "peak_gb",
            "logp_consistency_maxabs", "zero_std_groups"]
    ok("e2e_two_steps", len(st) == 2 and len(ho) == 1 and all(k in st[0] for k in keys)
       and all(np.isfinite(l["loss"]) for l in st), f"{time.time() - t0:.0f}s, step1 {json.dumps({k: st[0][k] for k in keys[:6]})}")
    ok("e2e_consistency_logged", all(l["logp_consistency_maxabs"] is not None and l["logp_consistency_maxabs"] < 1e-4 for l in st),
       [l["logp_consistency_maxabs"] for l in st])
    ok("e2e_saved", all(os.path.exists(os.path.join(out, d, "adapter_model.safetensors")) for d in ("latest", "best", "final"))
       and os.path.exists(os.path.join(out, "latest", "trainer_state.pt")) and s1["ref_param_maxabs_diff"] >= 0)
    m = json.load(open(os.path.join(out, "final", "teacher_meta.json")))
    ok("e2e_meta_mergeable", all(k in m for k in ("base", "pixels", "special_tokens", "n_base_vocab", "vocab_after", "grpo")))
    ok("e2e_meta_records_grammar_and_chain",   # 수정(2026-10-16): 설계 검토 반영
       m["grpo"].get("answer_grammar", {}).get("name") == AG.CONSTRAINT_NAME and m["grpo"]["chain"]["qa"]["kind"] == "qa_full"
       and len(m["grpo"].get("sft_meta_sha256") or "") == 64 and S.verify_teacher_chain(m, "A")["grpo"] is not None,
       "GRPO 산출 meta → verify_teacher_chain(QA → SFT → GRPO) 통과")
    TG.main(common + ["--steps", "3", "--resume"])
    logs = [json.loads(l) for l in open(os.path.join(out, "train_log.jsonl"))]
    st = [l["step"] for l in logs if "loss" in l]
    ok("e2e_resume", st == [1, 2, 3], st)
    try:
        TG.main(common + ["--steps", "1", "--fold", "B"])     # argparse: 마지막 --fold 가 적용 → SFT meta(fold A) 와 불일치
        wrong_fold = False
    except SystemExit:
        wrong_fold = True
    ok("e2e_rejects_wrong_fold", wrong_fold)
    try:
        TG.parse_args(["--fold", "A", "--max_train_samples", "4"]); guard = False
    except SystemExit:
        guard = True
    ok("e2e_subset_requires_smoke_flag", guard)


def main():
    assert S.require_gpu_env(allow_cpu=True) == "cpu", "CUDA_VISIBLE_DEVICES='' 로 실행할 것"
    torch.manual_seed(0)
    from common import ActionCodebook
    cb = ActionCodebook(str(S.REPO / S.CODEBOOK))
    cfg = GC.RewardConfig()
    reward_tests(cb, cfg)
    math_tests()
    data_dir = S.resolve_data_dir(None)
    toks, _ = S.load_special_tokens(None, data_dir)
    recs = S.load_records(data_dir, "train")
    trA, hoA, *_ , fold_doc = S.fold_select(recs, "A")
    proc = S.make_processor(S.BASE_QWEN)
    tok = proc.tokenizer
    S.add_special_tokens(tok, toks)
    mask_tests(tok, trA[:3000])
    with tempfile.TemporaryDirectory() as tmp:
        tiny_tests(tmp, toks, trA, fold_doc, cb, cfg)
        e2e_tests(tmp)
    n_fail = sum(1 for v in RESULTS.values() if not v["ok"])
    print(f"\n== {len(RESULTS) - n_fail}/{len(RESULTS)} 통과 ==", flush=True)
    out = S.REPO / "results/vla_teacher/smoke_grpo_cpu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(RESULTS, open(out, "w"), indent=1, ensure_ascii=False)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
