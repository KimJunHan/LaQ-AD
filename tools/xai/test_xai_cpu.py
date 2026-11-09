# 수정(2026-10-20): 사용자 승인 — 설명가능성 4가지(학습 변경 없음)
"""설명가능성 도구 CPU 시험(합성 + 실제 1 표본). GPU 를 쓰지 않는다(CUDA_VISIBLE_DEVICES="" 강제).

  (1) explicit_cross_attn == 독립 참조(torch F.multi_head_attention_forward, 같은 in/out 사영 가중치) — 출력·가중치
  (2) 어텐션 가중치: 쿼리·머리마다 합 1, 가린 토큰 가중치 정확히 0(가린 뒤에도 합 1)
  (3) 실제 QwenTokenAttn(2단계 config 의 설정 그대로 CPU 빌드, 체크포인트 가중치) + 실제 교사 캐시 1 표본:
      a. 훅(명시적 경로) 출력 == query + g⊙A_ref (A_ref = 참조 구현) — 훅이 모듈 수식을 바꾸지 않음
      b. 폴더 7 개 가중치 합 = 1 (계획·ego 쿼리 행), TokenLayout 폴더 길이 1832/168/40/240/68/13/8, 칸 57 토큰
      c. g = 0 이면 어느 폴더를 가려도 출력 == query (비트 동일)
      d. g ≠ 0 이면 폴더를 가리면 출력이 바뀐다, zero_lang 이면 출력 == query
      e. uninstall 뒤 인스턴스 덮어쓰기가 남지 않는다
  (4) 의도 라벨: GT 미래 궤적 → 같은 함수 → GT 의도 라벨과 일치율(구성상 100 % 기대) — val 미래 완비 표본 전체
  (5) 일치도 집계 함수(rates) 합성 검사
사용: PYTHONPATH=. OPENBLAS_CORETYPE=SkylakeX python tools/xai/test_xai_cpu.py [--ckpt <pth>] [--quick]
"""
import argparse
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import xai_common as X  # noqa: E402

STAGE2_CFG = os.path.join(X.REPO, "configs/nuscenes/v2_teacher/stage2/test.py")
DEFAULT_CKPT = os.path.join(X.REPO, "results/hipad_nusc_laq_v2_teacher_stage1/latest.pth")
PASS, FAIL = [], []


def check(name, ok, info=""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'통과' if ok else '실패'}] {name} {info}", flush=True)


def ref_attn(mfa, q, kv, keep=None):
    """독립 참조: torch.nn.functional.multi_head_attention_forward(같은 가중치). 반환 (A, 머리 평균 가중치)."""
    import torch.nn.functional as F
    fm = mfa.attn
    qT, kT = q.float().transpose(0, 1), kv.float().transpose(0, 1)
    out, w = F.multi_head_attention_forward(
        qT, kT, kT, fm.embed_dim, fm.num_heads, fm.in_proj_weight.float(), fm.in_proj_bias.float(), None, None, False, 0.0,
        fm.out_proj.weight.float(), fm.out_proj.bias.float(), training=False,
        key_padding_mask=None if keep is None else ~keep.bool(), need_weights=True, average_attn_weights=True)
    return out.transpose(0, 1), w


def build_qta():
    from mmcv import Config
    from mmcv.utils import build_from_cfg
    from mmcv.cnn.bricks.registry import ATTENTION
    cfg = X.load_cfg(STAGE2_CFG)
    c = dict(cfg.model.head.onedecoder_head.qwen_token_attn)
    qta = build_from_cfg(c, ATTENTION).eval()
    return cfg, qta


def t_synthetic():
    torch.manual_seed(0)
    _, qta = build_qta()
    mfa = qta.attn
    q, kv = torch.randn(1, 37, 256), torch.randn(1, 53, 256)
    A, P = X.explicit_cross_attn(mfa, q, kv)
    Ar, wr = ref_attn(mfa, q, kv)
    check("(1) 명시적 출력 == F.multi_head_attention_forward", torch.allclose(A, Ar, atol=1e-5, rtol=1e-4), f"max {float((A - Ar).abs().max()):.2e}")
    check("(1) 명시적 머리평균 가중치 == 참조", torch.allclose(P.mean(1), wr, atol=1e-6), f"max {float((P.mean(1) - wr).abs().max()):.2e}")
    check("(2) 가중치 합 = 1 (쿼리·머리마다)", torch.allclose(P.sum(-1), torch.ones_like(P.sum(-1)), atol=1e-5))
    keep = torch.ones(1, 53, dtype=torch.bool); keep[0, 10:30] = False
    A2, P2 = X.explicit_cross_attn(mfa, q, kv, keep)
    Ar2, wr2 = ref_attn(mfa, q, kv, keep)
    check("(2) 가린 토큰 가중치 = 0", float(P2[..., 10:30].abs().max()) == 0.0)
    check("(2) 가린 뒤 합 = 1", torch.allclose(P2.sum(-1), torch.ones_like(P2.sum(-1)), atol=1e-5))
    check("(1) 가린 경우도 참조와 같음", torch.allclose(A2, Ar2, atol=1e-5, rtol=1e-4), f"max {float((A2 - Ar2).abs().max()):.2e}")


def load_real_sample(cfg, token=None):
    from mmcv.utils import build_from_cfg
    from mmdet.datasets.builder import PIPELINES
    tl = cfg.teacher_loader
    loader = build_from_cfg(dict(tl), PIPELINES)
    if token is None:
        token = X.load_infos()[0]["token"]
    d = loader({"token": token})
    return token, d["qwen_hidden"].data.float(), d["qwen_mask"].data, d["qwen_type"].data


def t_real(ckpt, quick):
    cfg, qta = build_qta()
    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        pre = "head.onedecoder_head.qwen_token_attn."
        sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        missing, unexpected = qta.load_state_dict(sub, strict=False)
        check("(3) 체크포인트 qwen_token_attn 적재", len(sub) > 0 and not unexpected and all(k == "gate_sep" for k in missing),
              f"{os.path.basename(os.path.realpath(ckpt))}: 키 {len(sub)}, missing {missing}")
        del sd
    qta.eval()
    tok, qh, qm, qt = load_real_sample(cfg)
    lay = X.TokenLayout(qt[0].numpy())
    lens = [len(lay.folder_idx[f]) for f in X.FOLDERS]
    check("(3b) 폴더 길이 1832/168/40/240/68/13/8", lens == [1832, 168, 40, 240, 68, 13, 8], str(lens))
    part = np.zeros(lay.L, int)
    for f in X.FOLDERS:
        part[lay.folder_idx[f]] += 1
    check("(3b) 폴더가 토큰을 정확히 한 번씩 덮음", bool((part == 1).all()))
    slot_union = np.sort(np.concatenate([lay.seg[f"slot{k}"] for k in range(8)] + [lay.seg["plan_free"]]))
    obj_union = np.sort(np.concatenate([lay.folder_idx[f] for f in ("detection", "tracking", "motion", "plan")]))
    check("(3b) 칸 8×57 + plan_free 5 = det∪trk∪mot∪plan", np.array_equal(slot_union, obj_union) and
          all(len(lay.seg[f"slot{k}"]) == 57 for k in range(8)))
    torch.manual_seed(1)
    N = 1049   # det 900 + map 100 + plan 48 + ego 1 (2단계 config)
    query = torch.randn(1, N, 256)
    with torch.no_grad():
        kv, qtl = qta.project_kv(qh, qt, torch.float32)
    hook = X.XaiHook(qta, list(range(1000, 1048)), [1048], (0, 900), (900, 1000))
    hook.impl, hook.record = "explicit", True
    hook.seg_mat = lay.matrix(torch)
    hook.install()
    try:
        # (3a) 훅 출력 == query + g⊙A_ref (레이어마다)
        if not any(float(p.abs().max()) > 0 for p in (qta.gate,) + ((qta.gate_sep,) if qta.sep_gate_segs else ())):
            with torch.no_grad():
                qta.gate.normal_(0, 0.05)
        worst = 0.0
        layers = range(6) if not quick else (0, 5)
        for l in layers:
            hook.clear()
            with torch.no_grad():
                out = qta(query, qh, qm, layer_idx=l, kv=kv, qwen_type=qtl)
                Ar, _ = ref_attn(qta.attn, query, kv)
                exp = query + X.gate_rows(qta, l, N)[None] * Ar
            worst = max(worst, float((out - exp).abs().max()))
            agg = hook.rec["agg"][l]
            fs = agg[:, [lay.names.index(f) for f in X.FOLDERS]].sum(-1)
            if l == layers[0]:
                check("(3b) 계획·ego 행 폴더 가중치 합 = 1", torch.allclose(fs, torch.ones_like(fs), atol=1e-4), f"[{float(fs.min()):.6f}, {float(fs.max()):.6f}]")
                ctx_parts = agg[:, [lay.names.index(p) for p in lay.names if p.startswith("ctx_")]].sum(-1)
                check("(3b) context 부분 7 합 = context", torch.allclose(ctx_parts, agg[:, lay.names.index("context")], atol=1e-5))
        check("(3a) 훅 출력 == query + g⊙A_ref (실제 가중치·실제 캐시)", worst < 1e-4, f"max {worst:.2e}")
        # (3c) g = 0 → 어느 폴더를 가려도 출력 == query
        saved = (qta.gate.detach().clone(), qta.gate_sep.detach().clone() if qta.sep_gate_segs else None)
        with torch.no_grad():
            qta.gate.zero_()
            if qta.sep_gate_segs:
                qta.gate_sep.zero_()
        ok0 = True
        for f in X.FOLDERS:
            hook.keep = lay.keep_mask([f])
            with torch.no_grad():
                out = qta(query, qh, qm, layer_idx=3, kv=kv, qwen_type=qtl)
            ok0 &= torch.equal(out, query)
        check("(3c) g=0: 폴더 7 개 각각 가려도 출력 == query (비트 동일)", ok0)
        with torch.no_grad():
            qta.gate.copy_(saved[0])
            if saved[1] is not None:
                qta.gate_sep.copy_(saved[1])
            if float(qta.gate.abs().max()) == 0:
                qta.gate.normal_(0, 0.05)
        # (3d) g ≠ 0 → 가리면 바뀐다(구간별), zero_lang → query
        hook.keep = None
        with torch.no_grad():
            base = qta(query, qh, qm, layer_idx=3, kv=kv, qwen_type=qtl)
        g3 = X.gate_rows(qta, 3, N)
        seg_nz = {s: bool(g3[a:b].abs().max() > 0) for s, (a, b) in (("det", (0, 900)), ("map", (900, 1000)), ("plan_ego", (1000, N)))}
        changed = {}
        for f in X.FOLDERS:
            hook.keep = lay.keep_mask([f])
            with torch.no_grad():
                out = qta(query, qh, qm, layer_idx=3, kv=kv, qwen_type=qtl)
            changed[f] = {s: float((out - base)[0, a:b].abs().max()) for s, (a, b) in
                          (("det", (0, 900)), ("map", (900, 1000)), ("plan_ego", (1000, N)))}
        ok = all((changed[f][s] > 0) == seg_nz[s] for f in X.FOLDERS for s in seg_nz)
        check("(3d) g≠0 구간은 폴더를 가리면 출력이 바뀌고, g=0 구간은 그대로", ok,
              f"게이트≠0 구간 {seg_nz}; 예: context 가림 변화 {changed['context']}")
        hook.keep = None; hook.zero_lang = True
        with torch.no_grad():
            out = qta(query, qh, qm, layer_idx=3, kv=kv, qwen_type=qtl)
        check("(3d) zero_lang(언어 항 제거): 출력 == query", torch.equal(out, query))
        hook.zero_lang = False
    finally:
        hook.uninstall()
    check("(3e) uninstall 뒤 인스턴스 덮어쓰기 없음", "forward" not in qta.__dict__ and "_attn_only" not in qta.__dict__)
    return tok


def t_intent(quick):
    infos = X.load_infos()
    gts = X.load_gt_val()
    lab = X.IntentLabeler()
    n = ok_lon = ok_lat = 0
    bad = []
    t0 = time.time()
    for i, info in enumerate(infos):
        if quick and n >= 400:
            break
        rec = gts[info["token"]]
        if not rec["targets"].get("future_complete"):
            continue
        g = rec["targets"]["intent"]
        s = lab.label(np.asarray(rec["targets"]["fut_traj"]), info, rec)
        n += 1
        ok_lon += s["longitudinal"] == g["longitudinal"]; ok_lat += s["lateral"] == g["lateral"]
        if s["longitudinal"] != g["longitudinal"] or s["lateral"] != g["lateral"]:
            bad.append(info["token"])
    check("(4) GT 궤적 의도 == GT 의도 라벨(구성상 100 %)", ok_lon == n and ok_lat == n,
          f"n={n} 종 {ok_lon / n:.4f} 횡 {ok_lat / n:.4f} 불일치 {bad[:3]} ({time.time() - t0:.0f}s)")
    # 합성: 직진 등속(v0 = 5 m/s) → keep speed / go straight
    traj = np.stack([np.zeros(6), 5.0 * 0.5 * np.arange(1, 7)], 1)
    import common as C1
    lab0 = C1.intent_labels(traj, 5.0, [])
    check("(4) 합성 직진 등속 → keep speed / go straight", lab0["longitudinal"] == "keep speed" and lab0["lateral"] == "go straight")
    lab1 = C1.intent_labels(np.stack([np.zeros(6), 0.1 * np.arange(1, 7)], 1), 5.0, [])
    check("(4) 합성 급감속 → stop", lab1["longitudinal"] == "stop")
    th = np.deg2rad(np.linspace(5, 30, 6)); r = 10.0
    left = np.stack([-(r - r * np.cos(th)), r * np.sin(th)], 1)
    lab2 = C1.intent_labels(left, 4.0, [])
    check("(4) 합성 좌회전 호(θ_end 30°) → turn left", lab2["lateral"] == "turn left", str(lab2["theta_end_deg"]))


def t_rates():
    import xai_consistency as XC
    rows = [dict(teacher=dict(longitudinal="stop", lateral="go straight"), student=dict(longitudinal="stop", lateral="go straight")),
            dict(teacher=dict(longitudinal="accelerate", lateral="turn left"), student=dict(longitudinal="keep speed", lateral="turn right")),
            dict(teacher=None, student=dict(longitudinal="stop", lateral="go straight"))]
    r = XC.rates(rows, "teacher", "student")
    check("(5) 일치도 집계(합성)", r["n"] == 2 and r["lon_agree"] == 0.5 and r["lat_agree"] == 0.5 and r["joint_agree"] == 0.5
          and r["lat_opposite_direction"] == 0.5 and sum(map(sum, r["lat_confusion"]["M"])) == 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--quick", action="store_true", help="의도 검사 400 표본·계층 2 개만")
    a = ap.parse_args()
    t0 = time.time()
    t_synthetic()
    tok = t_real(a.ckpt, a.quick)
    t_intent(a.quick)
    t_rates()
    print(f"\n통과 {len(PASS)} / 실패 {len(FAIL)} (실제 표본 {tok}, {time.time() - t0:.0f}s)")
    if FAIL:
        print("실패:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
