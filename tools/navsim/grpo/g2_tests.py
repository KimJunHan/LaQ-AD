# 수정(2026-11-09): C3 GRPO 설계 v4.1 §12 G2 시험 1·2·3 (GPU) — 신규 파일(기존 파일 무수정).
"""G2 시험(GPU). G2-4 는 tests/test_train_loop_cpu.py(CPU), G2-5 는 train_loop.py --measure-sigma-ref,
G2-6 은 gate_score.py g2_6, G2-7 은 train_loop.py --hold-all-graphs(격리 켬/끔) 로 한다(체인 MODE=gates 가 모두 부른다).

1 batch 1 : 한 장면의 연속 K 프레임을 (A) 배포 경로 simple_test(장면 경계 뱅크 초기화, 플래그 끔) 와 (B) 롤아웃 경로
            (force_stream_groups, 스트림 그룹 0, 빈 상태에서 시작, z·r broadcast)로 돌려 ξ=0 명령 묶음 16 모드 μ 차 ≤ 1e-4 m,
            rescore 후 argmax·상위 M 색인 일치. (비트 동일 아님: DAF atomicAdd·cudnn_benchmark — §9.1)
2 batch 8 : 장면 8 개를 (B) 한 배치(그룹 0)로, (A) 장면마다 batch 1 simple_test 로 돌려 같은 기준.
3 격리    : (a) 격리 켬 + grad 켬 순전파에서 det·map·motion·ego 출력 requires_grad False, 계획 출력 True.
            (b) 학습 대상 매개변수 섭동(상대 1e-2) 전후 det·map·motion 출력 최대 차(허용 1e-5, DAF 비결정성 기록).
            (c) 격리 켬/끔 순전파 μ16·raw16 최대 차, 같은 손실의 학습 대상 기울기 최대 상대 차(qwen 구간 분리 호출 ulp 차 포함 — 기록).
            (d) 저장된 스트림 뱅크 상태 텐서에 grad_fn·requires_grad 없음.
실행: CUDA_VISIBLE_DEVICES=<3 또는 0(HIPAD_ALLOW_GPU0=1)> PYTHONPATH=. python tools/navsim/grpo/g2_tests.py \
        --model-config $MODEL_CFG --ckpt <ckpt> --out <dir>/g2_123.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.abspath(__file__)).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import rollout as RO  # noqa: E402

TOL_M = 1e-4


def _scenes(ds, n_scenes, K):
    """연속 K 프레임 이상인 장면 앞에서부터 n 개 → [[색인 K 개]]."""
    infos = ds.data_infos
    by = {}
    for i, inf in enumerate(infos):
        by.setdefault(inf["scene_token"], []).append(i)
    out = []
    for s, idx in by.items():
        idx = sorted(idx, key=lambda i: infos[i]["timestamp"])
        if len(idx) >= K:
            out.append(idx[:K])
        if len(out) == n_scenes:
            break
    return out


def _item(ds, i, group=None):
    it = dict(idx=i, aug_config=ds.get_augmentation())
    if group is not None:
        it["stream_group"] = group
    return ds[it]


def _batch(items):
    from mmcv.parallel import collate
    return collate(items, samples_per_gpu=len(items))


def _reset(det):
    dec = det.head.onedecoder_head
    for name in ("det_instance_bank", "map_instance_bank", "ego_instance_bank", "plan_instance_bank"):
        for bank in list(getattr(dec, name + "_list", []) or []) + [getattr(dec, name, None)]:
            if bank is not None:
                bank.reset()
    dec._stream_states = None
    det._last_scene_token = None


def path_A(det, ds, idxs):
    """배포 경로: simple_test(batch 1), 장면 경계 초기화. → [(mu16, raw16, resc16)]"""
    import torch
    dec = det.head.onedecoder_head
    pdec = dec.plan_decoder
    dec.force_stream_groups = False
    cap = []
    orig = pdec.rescore

    def spy(plan_cls, *a, **k):
        cap.append(plan_cls.detach().float().clone())
        return orig(plan_cls, *a, **k)
    pdec.rescore = spy
    _reset(det)
    res = []
    try:
        with torch.no_grad():
            for i in idxs:
                data = RO.to_device(_batch([_item(ds, i, 0)]), "cuda:0")   # RL 파이프라인 Collect 가 stream_group 을 요구 — 플래그 끔이면 디코더가 쓰지 않는다
                cap.clear()
                det.simple_test(data.pop("img"), **data)
                reg, cls = pdec._vis_modes
                res.append((reg[0].double().cpu().numpy(), cap[0][0].double().cpu().numpy(), cls[0].double().cpu().numpy()))
    finally:
        pdec.rescore = orig
        dec.force_stream_groups = True
    return res


def path_B(det, ds, idx_lists):
    """롤아웃 경로: 스트림 그룹 0, batch = len(idx_lists), j 번째 프레임끼리 한 배치. → [j][b] = (mu16, raw16, resc16)"""
    import torch
    _reset(det)
    out = []
    with torch.no_grad():
        for j in range(len(idx_lists[0])):
            data = RO.to_device(_batch([_item(ds, l[j], 0) for l in idx_lists]), "cuda:0")
            outs, data = RO.forward_heads(det, data, grad=False)
            raw, mu, _ = RO.plan_bundle(det, outs, data)
            resc = RO.rescore16(det, outs, raw, mu)
            out.append([(mu[b].double().cpu().numpy(), raw[b].double().cpu().numpy(), resc[b].double().cpu().numpy())
                        for b in range(len(idx_lists))])
    return out


def compare(a, b, M=4):
    d = float(np.abs(a[0] - b[0]).max())
    am = int(np.argmax(a[2])) == int(np.argmax(b[2]))
    tm = list(RO.top_modes(a[2], a[1], M)) == list(RO.top_modes(b[2], b[1], M))
    return d, am, tm


def test12(det, ds, K, nb):
    sc = _scenes(ds, nb, K)
    A = [path_A(det, ds, s) for s in sc]
    r = {}
    B1 = path_B(det, ds, [sc[0]])
    c1 = [compare(A[0][j], B1[j][0]) for j in range(K)]
    r["g2_1"] = dict(max_dmu_m=max(c[0] for c in c1), argmax_match=all(c[1] for c in c1), top_match=all(c[2] for c in c1),
                     frames=K, pass_=bool(max(c[0] for c in c1) <= TOL_M and all(c[1] and c[2] for c in c1)))
    B8 = path_B(det, ds, sc)
    c8 = [compare(A[b][j], B8[j][b]) for j in range(K) for b in range(len(sc))]
    r["g2_2"] = dict(max_dmu_m=max(c[0] for c in c8), argmax_match_frac=float(np.mean([c[1] for c in c8])),
                     top_match_frac=float(np.mean([c[2] for c in c8])), frames=len(c8), batch=len(sc),
                     max_dmu_m_by_frame=[max(compare(A[b][j], B8[j][b])[0] for b in range(len(sc))) for j in range(K)],
                     pass_=bool(max(c[0] for c in c8) <= TOL_M and all(c[1] and c[2] for c in c8)))
    # G2-2' 배치 짝 독립성: 장면 0 을 다른 짝 장면들과 묶어 같은 배치 크기로 다시 돌려 장면 0 출력이 짝에 의존하는지(누설 검사).
    #   결정적 DAF 에서 0 이어야 한다. 배치 1 대비 차(위 g2_2)는 배치 모양에 따른 fp16 커널 수치 차(같은 표본 8 복사도 위치별로 다름 —
    #   진단 2026-11-09)이고 누설이 아님을 여기서 구분한다.
    alt = _scenes(ds, 2 * nb - 1, K)[nb:]
    if len(alt) == nb - 1:
        B8b = path_B(det, ds, [sc[0]] + alt)
        r["g2_2_partner_independence"] = dict(
            max_dmu_m=max(float(np.abs(B8[j][0][0] - B8b[j][0][0]).max()) for j in range(K)),
            max_draw=max(float(np.abs(B8[j][0][1] - B8b[j][0][1]).max()) for j in range(K)),
            note="0 이면 배치 짝 누설 없음(결정적 DAF 에서만 엄밀)")
    return r


FORCE_DET_GRAD = [False]


def _grad_cmp(names, a, b):
    rel = []
    for n, x, y in zip(names, a["g"], b["g"]):
        if x is None or y is None:
            continue
        rel.append((float((x - y).abs().max() / max(float(y.abs().max()), 1e-12)), n))
    v = [t[0] for t in rel]
    return dict(max_dmu_m=float((a["mu"] - b["mu"]).abs().max()), max_draw=float((a["raw"] - b["raw"]).abs().max()),
                grad_rel_median=float(np.median(v)), grad_rel_max=max(v), n_params=len(v),
                params_rel_gt_1e2=sorted([(round(x, 4), n) for x, n in rel if x > 1e-2], reverse=True))


def test3c_det(det, ds):
    """G2-3c(결정적 DAF, 기울기 포함, 1 장면): 같은 모드 반복(켬·켬, 끔·끔)과 켬·끔 비교. 판정은 chain_guard.py(기준 승인 여부)."""
    import torch
    dec = det.head.onedecoder_head
    names = RO.select_trainable(det)
    pm = dict(det.named_parameters())
    params = [pm[n] for n in names]
    sc = _scenes(ds, 1, 1)
    data_cpu = _batch([_item(ds, s[0], 0) for s in sc])
    _, raw0, mu0 = _grad_forward(det, data_cpu)
    gen = torch.Generator(device="cuda").manual_seed(0)
    wmu = torch.randn(mu0.shape, generator=gen, device=mu0.device)
    wraw = torch.randn(raw0.shape, generator=gen, device=raw0.device)
    del raw0, mu0, _
    torch.cuda.empty_cache()

    def run(iso):
        dec.plan_grad_isolation = iso
        torch.cuda.reset_peak_memory_stats()
        o, raw, mu = _grad_forward(det, data_cpu)
        gs = torch.autograd.grad((mu * wmu).sum() + (raw * wraw).sum(), params, allow_unused=True)
        r = dict(mu=mu.detach().double().cpu(), raw=raw.detach().double().cpu(),
                 g=[None if x is None else x.detach().double().cpu() for x in gs], mem=torch.cuda.max_memory_allocated() / 2 ** 20)
        del o, raw, mu, gs
        torch.cuda.empty_cache()
        return r
    try:
        on1, on2, off1, off2 = run(True), run(True), run(False), run(False)
    finally:
        dec.plan_grad_isolation = True
    rep_on, rep_off, onoff = _grad_cmp(names, on1, on2), _grad_cmp(names, off1, off2), _grad_cmp(names, on1, off1)
    return dict(scenes=1, rep_on=rep_on, rep_off=rep_off, on_vs_off=onoff, peak_mem_mib_on=on1["mem"], peak_mem_mib_off=off1["mem"],
                note="큰 켬/끔 차는 격리가 ego 행 경로를 끊는 정의 차(gate_sep·중간 층 plan) — impl_log §8.1")


def _grad_forward(det, data_cpu, seed=0, grad=True):
    import torch
    _reset(det)
    data = RO.to_device(data_cpu, "cuda:0")
    outs, data = RO.forward_heads(det, data, grad=grad)
    raw, mu, _ = RO.plan_bundle(det, outs, data)
    return outs, raw, mu


def test3(det, ds, nb):
    import torch
    dec = det.head.onedecoder_head
    names = RO.select_trainable(det)
    pm = dict(det.named_parameters())
    params = [pm[n] for n in names]
    sc = _scenes(ds, nb, 1)
    data_cpu = _batch([_item(ds, s[0], 0) for s in sc])
    r = {}
    # (a)
    outs, raw, mu = _grad_forward(det, data_cpu)
    rg = {}
    for nm, o in zip(("det", "map", "ego", "plan", "motion"), outs):
        flags = [t.requires_grad for v in o.values() if isinstance(v, list) for t in v if isinstance(t, torch.Tensor)]
        flags += [v.requires_grad for v in o.values() if isinstance(v, torch.Tensor)]
        rg[nm] = dict(any_requires_grad=bool(any(flags)), n=len(flags))
    # (d)
    bad = 0
    for st in dec._stream_states or []:
        if st is None:
            continue
        for bank in st.values():
            for v in bank.values():
                if isinstance(v, torch.Tensor) and (v.requires_grad or v.grad_fn is not None):
                    bad += 1
    r["g2_3a_requires_grad"] = rg
    r["g2_3a_pass"] = (not any(rg[k]["any_requires_grad"] for k in ("det", "map", "ego", "motion"))) and rg["plan"]["any_requires_grad"]
    r["g2_3d_bank_tensors_with_grad"] = bad
    # (c) 격리 켬/끔
    rng = torch.Generator(device="cuda").manual_seed(0)
    wmu = torch.randn(mu.shape, generator=rng, device=mu.device)
    wraw = torch.randn(raw.shape, generator=rng, device=raw.device)
    res = {}
    for iso in (True, False):
        dec.plan_grad_isolation = iso
        torch.cuda.reset_peak_memory_stats()
        outs, raw, mu = _grad_forward(det, data_cpu)
        loss = (mu * wmu).sum() + (raw * wraw).sum()
        gs = torch.autograd.grad(loss, params, allow_unused=True)
        res[iso] = dict(mu=mu.detach().double().cpu(), raw=raw.detach().double().cpu(),
                        g=[None if g is None else g.detach().double().cpu() for g in gs],
                        mem=torch.cuda.max_memory_allocated() / 2 ** 20)
        del outs, raw, mu, loss, gs
    dec.plan_grad_isolation = True
    dmu = float((res[True]["mu"] - res[False]["mu"]).abs().max())
    draw = float((res[True]["raw"] - res[False]["raw"]).abs().max())
    rel = []
    nonedif = 0
    for a, b in zip(res[True]["g"], res[False]["g"]):
        if (a is None) != (b is None):
            nonedif += 1
            continue
        if a is None:
            continue
        rel.append(float((a - b).abs().max() / max(float(b.abs().max()), 1e-12)))
    r["g2_3c"] = dict(max_dmu_m=dmu, max_draw=draw, grad_max_rel_diff=max(rel) if rel else None,
                      grad_median_rel_diff=float(np.median(rel)) if rel else None, grad_none_mismatch=nonedif,
                      peak_mem_mib_iso_on=res[True]["mem"], peak_mem_mib_iso_off=res[False]["mem"],
                      note="값·기울기 차는 qwen 구간 분리 호출(GEMM 모양 차)·DAF atomicAdd 의 부동소수 차 — 허용 오차 판정 아님, 기록")
    # (b) 섭동
    with torch.no_grad():
        outs0 = _grad_forward(det, data_cpu, grad=False)[0]
        ref = {nm: [t.detach().clone() for t in o["prediction"]] + [t.detach().clone() for t in o["classification"]]
               for nm, o in zip(("det", "map", "motion"), (outs0[0], outs0[1], outs0[4]))}
        plan0 = outs0[3]["prediction"][-1].detach().clone()
        saved = [p.detach().clone() for p in params]
        for p in params:
            p.add_(torch.randn_like(p) * 1e-2 * (p.abs().mean() + 1e-6))
        outs1 = _grad_forward(det, data_cpu, grad=False)[0]
        dd = {}
        for nm, o in zip(("det", "map", "motion"), (outs1[0], outs1[1], outs1[4])):
            cur = [t for t in o["prediction"]] + [t for t in o["classification"]]
            dd[nm] = max(float((x - y).abs().max()) for x, y in zip(ref[nm], cur))
        dplan = float((outs1[3]["prediction"][-1] - plan0).abs().max())
        for p, s in zip(params, saved):
            p.copy_(s)
    r["g2_3b"] = dict(max_diff=dd, plan_diff_sanity=dplan, pass_=bool(max(dd.values()) <= 1e-5 and dplan > 0))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--scenes", type=int, default=8)
    ap.add_argument("--scenes-test3", type=int, default=0,
                    help="G2-3 배치 장면 수(0=--scenes). 결정적 DAF 는 역전파용 중간값이 커서 20 GB 안에 맞추려면 줄인다")
    ap.add_argument("--max-gpu-mib", type=int, default=0)
    ap.add_argument("--deterministic-daf", action="store_true",
                    help="DAF(deformable aggregation) 를 atomicAdd 없는 결정적 구현(tools/latency/student_fast._DetDAF)으로 바꿔 "
                         "실행 간 잡음 없이 G2-1·2·3 을 판정(설계 §9.6: 정확 검사 전용, 학습·평가에는 쓰지 않음)")
    a = ap.parse_args()
    dev = os.environ.get("CUDA_VISIBLE_DEVICES")
    if dev in (None, "", "2") or "," in dev or (dev == "0" and os.environ.get("HIPAD_ALLOW_GPU0") != "1"):
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={dev} — 한 장, GPU2 금지, GPU0 은 HIPAD_ALLOW_GPU0=1")
    if dev not in ("0", "3") and os.environ.get("GRPO_ALLOW_GPU") != dev:
        raise SystemExit(f"GPU{dev} — NAVSIM 은 GPU3 전용(병렬은 GRPO_ALLOW_GPU={dev}, 사용자 승인 사항)")
    os.chdir(ROOT)
    import torch
    if a.max_gpu_mib > 0:
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gpu_mib * 2 ** 20 / tot), 0)
    cfg = RO.load_model_cfg(a.model_config)
    from mmdet.apis import set_random_seed
    set_random_seed(0, deterministic=False)
    det = RO.build_rl_model(cfg, a.ckpt, "cuda:0")
    if a.deterministic_daf:
        sys.path.insert(0, str(ROOT / "tools" / "latency"))
        import torch
        from student_fast import _DetDAF, _module_of, install_global_patches
        _DetDAF.on = True
        install_global_patches(det)
        # 결정적 DAF 는 역전파용 중간값(카메라·축척별 grid_sample)이 커서 기울기 전파 시 20 GB 를 넘는다 →
        #   기울기 없는 순전파(G2-1·2 전부, G2-3b 섭동)에만 쓰고, 기울기 순전파(G2-3a·c)는 원본 DAF(잡음 바닥 수준 차 포함)
        agg = _module_of(det.head.onedecoder_head.det_deformable[0])
        disp = agg.DAF

        def _daf_grad_aware(*args):
            return disp.orig(*args) if (torch.is_grad_enabled() and not FORCE_DET_GRAD[0]) else disp(*args)
        agg.DAF = _daf_grad_aware
    ds = RO.build_rl_dataset(cfg)
    if not getattr(det, "reset_bank_on_scene_change", False):
        raise SystemExit("reset_bank_on_scene_change=False — 배포 경로(§9.1) 가정과 다르다")
    r = dict(ckpt=os.path.abspath(a.ckpt), model_config=a.model_config, deterministic_daf=bool(a.deterministic_daf))
    r.update(test12(det, ds, a.frames, a.scenes))
    r.update(test3(det, ds, a.scenes_test3 or a.scenes))
    r["scenes_test3"] = a.scenes_test3 or a.scenes
    if a.deterministic_daf:
        torch.cuda.empty_cache()
        FORCE_DET_GRAD[0] = True            # 1 장면이면 결정적 DAF 로도 기울기 순전파가 20 GB 안(켬 2.4 GB, 끔 8.9 GB — 11-09 측정)
        try:
            r["g2_3c_det"] = test3c_det(det, ds)
        finally:
            FORCE_DET_GRAD[0] = False
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(r, ensure_ascii=False, indent=1))
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
