# 수정(2026-10-15): 교사 VLM ② LoRA SFT — CPU 전용 스모크 검사(GPU 불필요, 실행: CUDA_VISIBLE_DEVICES="").
#   1) 데이터셋 → 토큰화 배치 모양(영상 토큰 6×180, pixel_values, grid)
#   2) 라벨 마스킹(프롬프트 -100, 정답+<|im_end|> 만 감독, 미래 불완전 표본은 Intent/Action/종료 제외, intent_weight 적용)
#   3) 특수토큰 왕복(2,041 개 모두 단일 토큰, 디코드 일치)
#   4) 파서·채점: GT 를 그대로 넣으면 만점, 깨진 텍스트는 형식 오류, copy 기준선
#   5) 프롬프트 dropout(v1 줄 제거, v2 형 prompt_parts 모의 레코드)
#   6) 소형 무작위 Qwen2-VL(실제 config 를 축소, tie_word_embeddings 유지)로 LoRA+새 토큰 행 학습 메커니즘:
#      기존 어휘 행 불변 / 새 행 변화 / 비전 동결 / LoRA 는 언어모델에만 / 병합 전후 출력 일치 / 1d·mrope 위치 경로
#   7) 기존 캐시 프롬프트 길이 = 1,159 (processor 만)
import copy
import json
import os
import sys
import tempfile

os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import sft_common as C  # noqa: E402
import teacher_metrics as M  # noqa: E402

RESULTS = {}


def ok(name, cond, info=""):
    RESULTS[name] = dict(ok=bool(cond), info=str(info))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    return cond


def main():
    assert C.require_gpu_env(allow_cpu=True) == "cpu", "CUDA_VISIBLE_DEVICES='' 로 실행할 것"
    torch.manual_seed(0)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="기본 자동(v2 → v1). v1 검사: --data_dir data/vla_teacher")
    a = ap.parse_args()
    data_dir = C.resolve_data_dir(a.data_dir)
    print(f"[smoke] data_dir = {data_dir}", flush=True)
    recs = C.load_records(data_dir, "train", max_n=2000)
    toks, src = C.load_special_tokens(None, data_dir)
    proc = C.make_processor(C.BASE_QWEN)
    tok = proc.tokenizer
    new_ids, n0 = C.add_special_tokens(tok, toks)
    C.check_answers_use_known_tokens(recs, toks)
    ok("special_tokens_registered", len(new_ids) == len(toks) and new_ids[0] == n0, f"{src} n={len(toks)} ids {new_ids[0]}..{new_ids[-1]}")

    # 3) 특수토큰 왕복
    s = " ".join(toks)
    ids = tok(s, add_special_tokens=False)["input_ids"]
    atomic = [i for i in ids if i >= n0]
    ok("special_tokens_atomic", atomic == new_ids, f"{len(atomic)}/{len(toks)} 단일 토큰")
    ok("special_tokens_decode", tok.decode(ids) == s)
    ex = "Action: <action_0> <action_2040> <action_13>."
    ok("special_tokens_roundtrip_line", tok.decode(tok(ex, add_special_tokens=False)["input_ids"]) == ex)

    # 1) 데이터셋 배치 모양
    action_ids = [i for t, i in zip(toks, new_ids) if t.startswith("<action_")]
    # 수정(2026-10-15): v2 는 이전 프레임이 없으면 객체 속력 숫자를 손실에서 가린다(loss_mask_spans). '정답 전체 감독' 검사는
    #   가림이 없는 완전 표본으로 하고, 가림 표본은 아래 label_mask_speed_spans 로 따로 검사한다(검사 가정 오류 수정, 학습 코드 무변경).
    comp = [r for r in recs if C.future_complete(r) and not r.get("loss_mask_spans")][:2]
    masked = [r for r in recs if C.future_complete(r) and r.get("loss_mask_spans")][:1]
    inc = [r for r in recs if not C.future_complete(r)][:1]
    ds = C.TeacherSFTDataset(comp + inc, proc, "qwen_default", action_ids=action_ids)
    items = [ds[i] for i in range(len(ds))]
    b = C.collate(items, tok.pad_token_id)
    n_img = int((b["input_ids"] == 151655).sum())
    ok("batch_image_tokens", n_img == 3 * 6 * C.TOKENS_PER_CAM, f"{n_img} (=3×6×180)")
    ok("batch_pixel_values", tuple(b["pixel_values"].shape) == (3 * 6 * 720, 1176), tuple(b["pixel_values"].shape))
    ok("batch_grid_thw", b["image_grid_thw"].tolist() == [[1, 20, 36]] * 18, b["image_grid_thw"][0].tolist())
    lens = [len(x["input_ids"]) for x in items]
    ok("batch_shapes", b["input_ids"].shape[1] == max(lens) and b["attention_mask"].sum(1).tolist() == lens,
       f"lens {lens}, prompt {[x['n_prompt'] for x in items]}")

    # 2) 라벨 마스킹
    good = True
    for x, r in zip(items, comp):
        lab = x["labels"]
        sup = lab[lab != -100].tolist()
        good &= bool((lab[:x["n_prompt"]] == -100).all())
        good &= tok.decode(sup) == r["answer"] + "<|im_end|>"
    ok("label_mask_complete", good, "프롬프트 전부 -100, 감독 토큰 디코드 == 정답+<|im_end|>")
    if masked:  # 수정(2026-10-15): 가림 표본 — 가린 문자 구간의 토큰만 빠지고 나머지는 감독
        mx = C.TeacherSFTDataset(masked, proc, "qwen_default", action_ids=action_ids)[0]
        r0 = masked[0]
        sup = tok.decode(mx["labels"][mx["labels"] != -100].tolist())
        keep = "".join(ch for i, ch in enumerate(r0["answer"])
                       if not any(a0 <= i < b0 for a0, b0 in r0["loss_mask_spans"])) + "<|im_end|>"
        ok("label_mask_speed_spans", bool((mx["labels"][:mx["n_prompt"]] == -100).all()) and sup == keep,
           f"가림 {r0['loss_mask_spans']}")
    x, r = items[2], inc[0]
    sup = tok.decode(x["labels"][x["labels"] != -100].tolist())
    # 수정(2026-10-15): v2-sensor 는 Objects/Map 줄 안에도 '<na>' 손실 제외 구간이 있다 → 그 문자를 뺀 앞 두 줄 + '\n'
    lm_inc = r.get("loss_mask_spans") or []
    want = "".join(ch for i, ch in enumerate("\n".join(r["answer"].split("\n")[:2]) + "\n")
                   if not any(a0 <= i < b0 for a0, b0 in lm_inc))
    ok("label_mask_incomplete", sup == want, repr(sup[-40:]))
    r2 = copy.deepcopy(comp[0]); r2["intent_weight"] = 0.3
    a_ids, a_lab, a_w = C.tokenize_answer(tok, r2)
    iw_txt = tok.decode([t for t, w in zip(a_ids, a_w) if abs(w - 0.3) < 1e-9])
    ok("intent_weight_span", iw_txt.startswith("Intent:") and "Action" not in iw_txt, repr(iw_txt))
    act_lab = x_act = items[0]["is_action_label"].sum().item()
    ok("action_label_count", act_lab == 6, f"{x_act} action 라벨")

    # 2b) 이전 키프레임 12 장 + prev_available=False 속력 마스크
    rp = copy.deepcopy(comp[0]); rp["cams_prev"] = comp[1]["cams"]; rp["prev_available"] = True
    paths = C.cam_paths(rp, True, "prev_first")
    ok("prev_order_prev_first", len(paths) == 12 and paths[0].endswith(comp[1]["cams"][0]) and paths[6].endswith(rp["cams"][0]))
    ok("prev_order_cur_first", C.cam_paths(rp, True, "cur_first")[0].endswith(rp["cams"][0]))
    rn = copy.deepcopy(comp[0]); rn.pop("cams_prev", None); rn["prev_available"] = False
    ok("prev_missing_repeats_current", C.cam_paths(rn, True)[:6] == C.cam_paths(rn, True)[6:])
    dsp = C.TeacherSFTDataset([rp, rn], proc, "qwen_default", action_ids=action_ids, use_prev=True)
    xp = dsp[0]
    n_img = int((xp["input_ids"] == 151655).sum())
    ok("prev_12_images_tokens", n_img == 12 * C.TOKENS_PER_CAM and tuple(xp["pixel_values"].shape) == (12 * 720, 1176),
       f"영상 토큰 {n_img}, 시퀀스 {len(xp['input_ids'])} (프롬프트 {xp['n_prompt']})")
    sp = C.speed_spans(rn)
    want_sp = [rn["answer"][a:b] for a, b in sp]
    if rn.get("modes", {}).get("answer_format") == "v2-sensor":   # 수정(2026-10-15): 고정폭 속력, static 도 속력 표기(NaN 만 없음)
        gt_sp = [f"{o['speed']:04.1f}" for o in rn["targets"]["objects"] if o.get("motion") != "na"]
    else:
        gt_sp = [f"{o['speed']:.1f}" for o in rn["targets"]["objects"] if o.get("vel_sector")]
    ok("speed_spans_match_targets", want_sp == gt_sp, f"{want_sp}")
    a_ids, a_lab, _ = C.tokenize_answer(tok, rn)
    masked = "".join(tok.decode([t]) for t, l in zip(a_ids, a_lab) if l == -100)
    ok("speed_mask_without_prev", masked == "".join(want_sp) and len(want_sp) > 0, repr(masked))
    a_ids2, a_lab2, _ = C.tokenize_answer(tok, rp)
    ok("no_speed_mask_with_prev", all(l != -100 for l in a_lab2))
    import tempfile as _tf
    with _tf.TemporaryDirectory() as d:
        hp = os.path.join(d, "h.json")
        h1, hh1, _ = C.holdout_scenes(recs, 5, 0, hp)
        h2, hh2, _ = C.holdout_scenes(recs, 5, 0, hp)
        ok("holdout_fixed_file_hash", h1 == h2 and hh1 == hh2 and len(h1) == 5)
        dd = json.load(open(hp)); dd["scenes"][0] = "x" * 32; json.dump(dd, open(hp, "w"))
        try:
            C.holdout_scenes(recs, 5, 0, hp); tamper = False
        except RuntimeError:
            tamper = True
        ok("holdout_tamper_detected", tamper)

    # 5) 프롬프트 dropout(드롭 줄은 '... unavailable.' 로 교체, 줄 수 고정)
    r0 = comp[0]
    lines = r0["prompt"]["user"].split("\n")
    u = C.build_user_text(r0, drop_cmd=True)
    ok("drop_cmd", "Command: unavailable." in u and lines[-2] not in u and lines[-1] in u
       and len(u.split("\n")) == len(lines), u.split("\n")[-2:])
    u = C.build_user_text(r0, drop_ego=True)
    ok("drop_ego", "Ego state: unavailable." in u and lines[-1] not in u and lines[-2] in u, u.split("\n")[-2:])
    ok("prompt_roundtrip", C.check_prompt_parts_roundtrip(recs, n=2000) == 0, "드롭 없는 조립 == prompt.user (2000 표본)")
    if r0.get("prompt_parts") is not None:
        import common_v2
        same = all(C.build_user_text(r, dc, de) == common_v2.join_prompt(r["prompt_parts"], dc, de)
                   for r in recs[:300] for dc in (False, True) for de in (False, True)
                   if r.get("ego_drop_ok", True) and r.get("command_drop_ok", True))
        ok("prompt_drop_equals_common_v2_join", same)
    r3 = copy.deepcopy(r0); r3["ego_drop_ok"] = False
    u = C.build_user_text(r3, drop_cmd=True, drop_ego=True)
    ok("drop_respects_ego_drop_ok", lines[-1] in u and "Command: unavailable." in u)
    ds_d = C.TeacherSFTDataset(recs[:200], proc, drop_cmd_p=0.5, drop_ego_p=0.0, seed=0)
    fr = np.mean([ds_d.drops(i)[0] for i in range(200)])
    ds_d.set_epoch(1)
    fr2 = [ds_d.drops(i)[0] for i in range(200)]
    ds_d.set_epoch(0)
    ok("dropout_rate_and_epoch_variation", 0.35 < fr < 0.65 and fr2 != [ds_d.drops(i)[0] for i in range(200)], f"p̂={fr:.2f}")

    # 4) 파서·채점
    val = C.load_records(data_dir, "val", max_n=300)
    pname, parser, rates = M.select_parser(None, val)
    ok("parser_select", pname.endswith("parse_answer"), f"{pname} {rates}")
    from common import ActionCodebook
    cb = ActionCodebook(str(C.REPO / C.CODEBOOK))
    perc, mp, it = M.PerceptionAcc(), M.MapAcc(), M.IntentAcc()
    for r in val:
        p = parser(r["answer"])
        perc.add(p["objects"], r["targets"]["objects"], p.get("ego_speed"), r["targets"].get("ego_speed"))
        mp.add(p["map"], r["targets"]["map"])
        if C.future_complete(r):
            it.add(p["intent"], r["targets"]["intent"])
    pr, mr, ir = perc.result(), mp.result(), it.result()
    ok("score_gt_perfect_perception", pr["recall"] == 1 and pr["precision"] == 1 and pr["dist_mae"] == 0
       and pr["depth_order_acc"] == 1.0, json.dumps({k: pr[k] for k in ("recall", "precision", "dist_mae", "speed_mae", "depth_order_acc")}))
    ok("score_gt_perfect_map", mr["line_exact"] == 1.0, mr["line_exact"])
    ok("score_gt_perfect_intent", ir["joint_acc"] == 1.0, ir["n"])
    # 깨진 텍스트
    bad = val[0]["answer"].replace("Map:", "Mapp:")
    try:
        parser(bad); broken = False
    except Exception:
        broken = True
    ok("parser_rejects_malformed", broken)
    # 객체 일부 누락·거리 오차
    r = next(rr for rr in val if len(rr["targets"]["objects"]) >= 4)
    P = copy.deepcopy(parser(r["answer"])["objects"])[:3]
    if "dist" in P[0]:
        P[0]["dist"] += 1.0
    else:                                   # v2: 거리 없음 → 위치를 지름 방향으로 1 m 밀기
        x, y = P[0]["lat"], P[0]["lon"]; d = float(np.hypot(x, y))
        P[0]["lat"], P[0]["lon"] = x * (d + 1) / d, y * (d + 1) / d
    pa = M.PerceptionAcc(); pa.add(P, r["targets"]["objects"])
    rr_ = pa.result()
    ok("score_partial", abs(rr_["recall"] - 3 / len(r["targets"]["objects"])) < 1e-9 and rr_["precision"] == 1.0
       and abs(rr_["dist_mae"] - 1 / 3) < 0.05, json.dumps({k: rr_[k] for k in ("recall", "precision", "dist_mae", "pos_err_mean")}))
    # 행동 복호 = action_recon_traj
    rc = next(rr for rr in val if C.future_complete(rr))
    dec = cb.decode(parser(rc["answer"])["action"])
    ok("action_decode_matches_recon", np.allclose(dec, rc["targets"]["action_recon_traj"], atol=1e-3))
    ci, ct, srcb = M.copy_baseline(rc)
    ok("copy_baseline", ct.shape == (6, 2) and ci["lateral"] == rc["command"]["text"], f"{srcb} v={rc['ego_status']['speed']}")
    # 형식 오류 생성 시 채점 경로
    gens = {rr["token"]: dict(token=rr["token"], text=("garbage" if k % 2 else rr["answer"]), ended=True)
            for k, rr in enumerate(val[:20])}
    n_ok = sum(1 for g in gens.values() if g["text"] != "garbage")
    pa = M.PerceptionAcc()
    for rr in val[:20]:
        try:
            p = parser(gens[rr["token"]]["text"])
        except Exception:
            p = None
        pa.add(p["objects"] if p else [], rr["targets"]["objects"])
    ok("score_with_invalid_outputs", 0 < pa.result()["recall"] < 1, f"{n_ok}/20 유효")

    # 7) 기존 캐시 프롬프트 길이
    instr = C.original_cache_instruction()
    from PIL import Image
    imgs = C.load_images(comp[0])
    msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": instr}]}]
    proc0 = C.make_processor(C.BASE_QWEN)
    t = proc0.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    L = proc0(text=[t], images=imgs, return_tensors="pt")["input_ids"].shape[1]
    ok("cache_prompt_len_1159", L == 1159, L)
    del Image

    # 6) 소형 무작위 모델 메커니즘
    tiny_mechanics(proc, tok, toks, new_ids, n0, b)

    n_fail = sum(1 for v in RESULTS.values() if not v["ok"])
    print(f"\n== {len(RESULTS) - n_fail}/{len(RESULTS)} 통과 ==", flush=True)
    out = C.REPO / "results/vla_teacher/smoke_cpu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(RESULTS, open(out, "w"), indent=1, ensure_ascii=False)
    sys.exit(1 if n_fail else 0)


def tiny_mechanics(proc, tok, toks, new_ids, n0, batch):
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
    cfg = Qwen2VLConfig.from_pretrained(C.BASE_QWEN)
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 64, 128, 2
    cfg.num_attention_heads, cfg.num_key_value_heads = 4, 2
    cfg.rope_scaling = {"type": "mrope", "rope_type": "default", "mrope_section": [2, 3, 3]}
    vc = cfg.vision_config
    vc.depth, vc.embed_dim, vc.num_heads, vc.hidden_size, vc.mlp_ratio = 1, 32, 2, 64, 2
    cfg.tie_word_embeddings = True
    torch.manual_seed(0)
    model = Qwen2VLForConditionalGeneration(cfg).float()
    model.config._attn_implementation = "sdpa"
    model.tie_weights()
    C.resize_for_tokens(model, len(tok))
    C.init_new_rows(model, tok, toks, new_ids, n0, "desc")
    W0 = model.get_input_embeddings().weight.detach().clone()
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm = C.build_peft(model, new_ids, 4, 8, 0.0, "attn_mlp")
    summ = C.trainable_summary(pm)
    names = [n for n, p in pm.named_parameters() if p.requires_grad]
    ok("tiny_lora_only_lm", all(".visual." not in n for n in names) and any("mlp.gate_proj.lora_A" in n for n in names)
       and any("self_attn.q_proj.lora_A" in n for n in names), summ)
    ok("tiny_trainable_fp32", all(p.dtype == torch.float32 for p in pm.parameters() if p.requires_grad))
    pm.train()
    lora_p = [p for n, p in pm.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_p, lr=1e-2, weight_decay=0.0)
    bb = {k: v for k, v in batch.items() if k != "tokens"}
    losses = []
    for _ in range(3):
        loss, n, st = C.sft_loss(pm, bb, "mrope")
        loss.backward(); opt.step(); opt.zero_grad()
        losses.append(loss.item())
    ok("tiny_loss_finite_decreasing", all(np.isfinite(losses)) and losses[-1] < losses[0], [round(x, 3) for x in losses])
    base = pm.get_base_model()
    # 순전파에서 쓰이는 실제 가중치 = base 가중치 + 새 행 덮어쓰기
    tt = base.model.embed_tokens.token_adapter          # peft TrainableTokensWrapper → TrainableTokensLayer
    W_base = tt.base_layer.weight.detach()
    ok("tiny_base_rows_unchanged", torch.equal(W_base[:n0], W0[:n0]), "기존 151,657 행 비트 불변")
    vals = tt.trainable_tokens_delta["default"].detach()
    ok("tiny_new_rows_changed", not torch.allclose(vals, W0[new_ids]), f"Δ={float((vals - W0[new_ids]).abs().mean()):.2e}")
    ok("tiny_vision_frozen", all(not p.requires_grad for p in base.visual.parameters()))
    # 병합 전후 출력 일치 + tied lm_head
    pm.eval()
    with torch.no_grad():
        h1 = C.lm_forward(pm, bb, "mrope").last_hidden_state
        lg1 = base.lm_head(h1[:, -5:])
    with tempfile.TemporaryDirectory() as d:
        pm.save_pretrained(d)
        sd_keys = list(__import__("safetensors.torch", fromlist=["load_file"]).load_file(os.path.join(d, "adapter_model.safetensors")).keys())
        ok("tiny_adapter_saves_token_rows", any("trainable_tokens" in k for k in sd_keys), [k for k in sd_keys if "trainable_tokens" in k][:2])
    merged = pm.merge_and_unload()
    with torch.no_grad():
        h2 = C.lm_forward(merged, bb, "mrope").last_hidden_state
        lg2 = merged.lm_head(h2[:, -5:])
    ok("tiny_merge_equivalent", torch.allclose(h1, h2, atol=1e-4) and torch.allclose(lg1, lg2, atol=1e-3),
       f"max|Δh|={float((h1 - h2).abs().max()):.2e}")
    ok("tiny_merged_tied", merged.get_output_embeddings().weight.data_ptr() == merged.get_input_embeddings().weight.data_ptr())
    # 1d vs mrope 위치 경로 — 1d 는 HF forward(position_ids 없음)와 같아야 한다
    with torch.no_grad():
        h1d = C.lm_forward(merged, bb, "1d", output_hidden_states=True).hidden_states[1]
        hf = merged(input_ids=bb["input_ids"], attention_mask=bb["attention_mask"], pixel_values=bb["pixel_values"],
                    image_grid_thw=bb["image_grid_thw"], output_hidden_states=True).hidden_states[1]
    ok("tiny_1d_equals_hf_forward", torch.allclose(h1d, hf, atol=1e-5), "기존 캐시 경로(HF forward)는 1d 위치")
    with torch.no_grad():
        hm = C.lm_forward(merged, bb, "mrope", output_hidden_states=True).hidden_states[1]
    # 수정(2026-10-15): '다르다' 판정 기준을 위 '같다' 기준(atol 1e-5)의 10배(1e-4)로. 이전 1e-3 은 소형 무작위 모델의 1층 은닉
    #   크기에 비해 임의로 컸다 — 지시문이 길어진 v2-sensor 배치에서 실제 차이 9.7e-4(같음 허용치의 ~100배)를 '같음'으로 판정했다.
    ok("tiny_mrope_differs_from_1d", not torch.allclose(hm, h1d, atol=1e-4), f"max|Δ|={float((hm - h1d).abs().max()):.2e}")


if __name__ == "__main__":
    main()
