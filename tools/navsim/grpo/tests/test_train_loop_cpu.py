# 수정(2026-11-09): C3 GRPO 학습 루프·덤프·통계 도구 CPU 단위 시험 + G2-4(출력 공간 기울기 정확도) — 신규 파일.
"""hipad_bw env(CPU): PYTHONPATH=. python tools/navsim/grpo/tests/test_train_loop_cpu.py
  1 lr 일정(warmup 선형 → cosine), 2 상위 M 정렬 규칙(rescore 후 → 원시 → 색인), 3 장면 선택 규칙,
  4 G2-4: 구현 손실 ∂L/∂μ vs −(σ_ref²/N) Σ A log N(η(μ)) autograd(기저·σ detach, 고곡률 ≥ 0.3 rad/경유점, 축마다 다른 σ),
    B1(균등 A)·Full(가중 w·u) 둘 다, 요소별 상대 오차 ≤ 1e-6(분모 max(|ref|, 1e-8‖ref‖∞)) — fp64 로 계산하고 fp32 도 기록,
  5 EMA 식(EMASidecarHook 과 같음), 6 Holm, 7 로그 블록 bootstrap(합성: 차이 0 이면 하한 < 0, 큰 차이면 > 0),
  8 L_cls 목표·−999 제외, 9 후보 구성(K = 16 + M·G, η 재현), 10 grpo config 12 개 형식.
  검토 B(11-09) 운영 수정: 11 work_dir 잠금, 12 이어하기 기록 잘라내기·백업, 13 서버 기동 확인(이전 로그 무시·p.poll),
  14 CONT_REF_PROGRESS 소스 검사, 15 관문 강제(엄격/승인 기준·G2-7), 16 RUNS 분배 claim(겹침 die·죽은 pid 무시),
  17 덤프 DONE 맨 마지막·RNG 복원, 18 평가 전 엄격 적재 검사, 19 GPU 규칙(GPU1 은 GRPO_ALLOW_GPU), 20 체인 문법·가드 존재,
  21 모델 diff 의 격리 전제 정규식(_GRPO_PLAN_PARAM) = rollout.TRAINABLE(검토 A) + diff 가 현재 소스에 적용 가능(또는 이미 적용),
  22 체인 실제 실행(가짜 트리): DRYRUN 이 태그 폴더를 안 만들고, 이어진 MODE=gates 가 claim_dir 를 지나 덤프 단계까지 감.
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE.parent))
import advantage as AD  # noqa: E402
import ar1  # noqa: E402
import bootstrap_stats as BS  # noqa: E402
import chain_guard as CG  # noqa: E402
import dump_rollout as DR  # noqa: E402
import rollout as RO  # noqa: E402
import train_loop as TL  # noqa: E402


def t1_lr():
    g = dict(TL.GRPO_DEFAULT)
    assert abs(TL.lr_at(0, g) - 1e-6) < 1e-15
    assert abs(TL.lr_at(100, g) - 1e-5) < 1e-15
    assert TL.lr_at(50, g) < TL.lr_at(99, g) < 1e-5
    assert abs(TL.lr_at(1550, g) - 0.5e-5) < 1e-12
    assert TL.lr_at(2999, g) < 1e-8


def t2_top():
    resc = np.array([1, 3, -996, 3, 2, 2, 0, -999, 5, 0, 0, 0, 0, 0, 0, 0], float)
    raw = np.array([1, 3, 3, 4, 2, 7, 0, 1, 5, 0, 0, 0, 0, 0, 0, 0], float)
    top = RO.top_modes(resc, raw, 4)
    assert list(top) == [8, 3, 1, 5], top          # 5 > 3=3(원시 4 > 3) > 2,2(원시 7 > 2)
    resc2 = np.zeros(16); raw2 = np.zeros(16)
    assert list(RO.top_modes(resc2, raw2, 4)) == [0, 1, 2, 3]


def t3_select():
    infos = [dict(scene_token=f"s{i // 3}", token=f"f{i}") for i in range(30)]
    assert RO.scene_select(infos, "all") is None
    idx = RO.scene_select(infos, "hash_first:4")
    assert len(idx) == 12 and len({infos[i]["scene_token"] for i in idx}) == 4
    a = RO.scene_select(infos, "hash_mod:2:0"); b = RO.scene_select(infos, "hash_mod:2:1")
    assert sorted(a + b) == list(range(30)) and not set(a) & set(b)


def _basis(rng, n):
    th0 = rng.uniform(-np.pi, np.pi, (n, 1))
    th = th0 + np.cumsum(rng.choice([-1, 1], (n, 8)) * rng.uniform(0.3, 0.6, (n, 8)), 1)   # ≥ 0.3 rad/경유점
    e_lon = np.stack([np.cos(th), np.sin(th)], -1)
    e_lat = np.stack([-np.sin(th), np.cos(th)], -1)
    return np.stack([e_lon, e_lat], -2)


def t4_g2_4():
    import torch
    rng = np.random.default_rng(0)
    n, sref, Nn = 32, 0.15, 64
    E = _basis(rng, n)
    sig = np.stack([0.15 * np.sqrt(np.arange(8) + 1), 0.05 * np.sqrt(np.arange(8) + 1) * np.linspace(1, 2.5, 8)], -1)  # 축마다 다른 σ
    mu0 = rng.normal(0, 3, (n, 8, 2))
    eta = rng.standard_normal((n, 8, 2))
    tau = ar1.sample_tau(mu0, E, sig, eta)
    w = rng.uniform(0.2, 2, (n, 8)); w *= 8 / w.sum(-1, keepdims=True)
    u = rng.uniform(0.5, 1.5, (n, 1)); u = np.concatenate([u, 2 - u], -1)
    Ahat = rng.normal(0, 1, n)
    worst = {}
    for name, A in (("B1", np.repeat(np.repeat(Ahat[:, None, None], 8, 1), 2, 2)),
                    ("Full", Ahat[:, None, None] * w[..., None] * u[:, None, :])):
        for dt in (torch.float64, torch.float32):
            mu = torch.tensor(mu0, dtype=dt, requires_grad=True)
            L = ar1.loss_rl_torch(mu, E, sig, eta, A, sigma_ref=sref, N=Nn)
            g_impl, = torch.autograd.grad(L, mu)
            mu2 = torch.tensor(mu0, dtype=dt, requires_grad=True)
            et = ar1.innovations_torch(torch.tensor(tau, dtype=dt), mu2, E, sig)
            logp = -0.5 * et ** 2
            Lref = -(sref ** 2 / Nn) * (torch.tensor(A, dtype=dt) * logp).sum()
            g_ref, = torch.autograd.grad(Lref, mu2)
            r = g_ref.double().numpy(); i = g_impl.double().numpy()
            den = np.maximum(np.abs(r), 1e-8 * np.abs(r).max())
            worst[(name, str(dt))] = float((np.abs(i - r) / den).max())
            # 같은 τ 에서 η(μ0) = η 확인(표본 재구성)
            assert np.abs(et.detach().double().numpy() - eta).max() < (1e-9 if dt == torch.float64 else 1e-4)
    for k, v in worst.items():
        print(f"    G2-4 {k[0]:4s} {k[1]:14s} 요소별 최대 상대 오차 {v:.2e}")
        if "float64" in k[1]:
            assert v <= 1e-6, (k, v)
    # 둘째 항에 e_{a,k} 를 쓰는 오류를 잡는지: 일부러 틀린 손실과 비교
    mu = torch.tensor(mu0, requires_grad=True)
    Ew = torch.tensor(E)
    p_wrong = (Ew / torch.tensor(sig)[..., None] * mu[..., None, :]).sum(-1)
    pk1 = (Ew[:, 1:] / torch.tensor(sig)[1:, :, None] * mu[:, :-1, None, :]).sum(-1)      # 틀림: e_{a,k} 로 μ_{k−1} 사영
    br = torch.cat([p_wrong[:, :1], p_wrong[:, 1:] - 0.8 * pk1], 1)
    c = torch.tensor(ar1.c_diag())
    A = Ahat[:, None, None] * w[..., None] * u[:, None, :]
    Lw = -(sref ** 2 / Nn) * (torch.tensor(A) * torch.tensor(eta) / c[:, None] * br).sum()
    gw, = torch.autograd.grad(Lw, mu)
    mu2 = torch.tensor(mu0, requires_grad=True)
    et = ar1.innovations_torch(torch.tensor(tau), mu2, E, sig)
    gr, = torch.autograd.grad(-(sref ** 2 / Nn) * (torch.tensor(A) * -0.5 * et ** 2).sum(), mu2)
    rel = (np.abs(gw.numpy() - gr.numpy()) / np.maximum(np.abs(gr.numpy()), 1e-8 * np.abs(gr.numpy()).max())).max()
    assert rel > 1e-3, rel
    print(f"    G2-4 틀린 교차항 검출: 상대 오차 {rel:.2e} > 1e-3")


def t5_ema():
    d0, ramp = 0.99708, 137
    for n in (1, 10, 137, 1000):
        d = d0 * (1 - math.exp(-n / ramp))
        assert 0 <= d < d0
    assert d0 * (1 - math.exp(-1 / ramp)) < 0.01


def t6_holm():
    rej, adj = BS.holm(np.array([0.01, 0.04, 0.03, 0.2]))
    assert list(rej) == [True, False, False, False], rej
    assert abs(adj[0] - 0.04) < 1e-12 and abs(adj[2] - 0.09) < 1e-12


def t7_bootstrap(tmp):
    import csv
    import pickle
    rng = np.random.default_rng(1)
    toks = [f"t{i}" for i in range(600)]
    logs = [f"log{i // 10}" for i in range(600)]
    pickle.dump(dict(infos=[dict(navsim_token=t, log_name=l, is_eval_frame=True) for t, l in zip(toks, logs)]),
                open(tmp / "inf.pkl", "wb"))
    base = rng.uniform(0.5, 0.9, 600) + np.repeat(rng.normal(0, 0.1, 60), 10)

    def wr(name, sc):
        with open(tmp / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["", "token", "valid", "score"])
            for i, (t, s) in enumerate(zip(toks, sc)):
                w.writerow([i, t, True, s])
            w.writerow([600, "average_all_frames", True, float(np.mean(sc))])
    wr("full0", base + 0.05); wr("full1", base + 0.05 + rng.normal(0, 0.01, 600)); wr("b1a", base); wr("b1b", base)
    wr("c1", base + 0.049)
    for v in ("A-a", "A-b", "A-d", "B2", "R", "P", "C2"):
        wr(v, base + rng.normal(0, 0.01, 600))
    args = ["--runs", f"Full={tmp}/full0.csv,{tmp}/full1.csv", f"B1={tmp}/b1a.csv,{tmp}/b1b.csv", f"C1={tmp}/c1.csv"]
    args += [f"{v}={tmp}/{v}.csv" for v in ("A-a", "A-b", "A-d", "B2", "R", "P", "C2")]
    sys.argv = ["bs"] + args + ["--infos", str(tmp / "inf.pkl"), "--out", str(tmp / "bs.json"), "--n-boot", "2000"]
    BS.main()
    import json
    r = json.loads((tmp / "bs.json").read_text())
    assert r["primary"]["Full-B1"]["lower95_one_sided"] > 0
    assert r["primary"]["Full-C1"]["lower95_one_sided"] < 0.01
    assert r["n_logs"] == 60


def t8_cls():
    e = np.linspace(0, 1, 16)
    t, v, why = AD.cls_target(e, rescored_logits=np.zeros(16))
    assert v and abs(t.sum() - 1) < 1e-12 and t.argmax() == 15
    lg = np.zeros(16); lg[3] = -999
    t, v, why = AD.cls_target(e, rescored_logits=lg)
    assert not v and why == "rescore_sentinel"


def t9_cands():
    rng = np.random.default_rng(2)
    mu16 = np.cumsum(rng.uniform(0.5, 1.5, (16, 8, 2)) * [1, 0.1], 1)
    xs = np.linspace(-10, 120, 261)
    cl = dict(xyh=np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], -1), valid=np.ones(261, bool))
    top = np.array([3, 1, 7, 0])
    sig = ar1.sigma_table(1.0)
    c1, E, eta, info = RO.policy_candidates(mu16, top, "tokA", cl, xs, sig, 0, 5)
    c2, *_ = RO.policy_candidates(mu16, top, "tokA", cl, xs, sig, 0, 5)
    c3, *_ = RO.policy_candidates(mu16, top, "tokA", cl, xs, sig, 0, 6)
    assert c1.shape == (48, 8, 2) and np.array_equal(c1, c2) and not np.array_equal(c1, c3)
    assert np.array_equal(c1[:16], mu16)
    assert np.allclose(E[:, :, 0], [1, 0]) and np.allclose(E[:, :, 1], [0, 1])          # 직선 +x 중심선
    eta_rec = ar1.innovations(c1[16:24], mu16[3], E[3], sig)
    assert np.abs(eta_rec - eta[0]).max() < 1e-9
    ce, Ee, _, _ = RO.policy_candidates(mu16, top, "tokA", None, None, sig, 0, 5, axis="ego")
    assert ce.shape == (48, 8, 2)


def t10_configs():
    from mmcv import Config
    root = HERE.parents[3]
    fs = sorted((root / "configs/navsim/v2_teacher/grpo").glob("*_s[01].py"))
    seen = set()
    for f in fs:
        c = Config.fromfile(str(f))
        g = TL.load_grpo(str(f), [], False)
        assert g["variant"] in TL.ALL_VARIANTS and g["steps"] == 3000 and g["accum"] == 6 and g["micro_batch"] == 8
        seen.add((g["variant"], g["seed"]))
    assert len(fs) == 12 and len(seen) == 12, (len(fs), seen)
    assert {v for v, s in seen if s == 1} == {"Full", "B1"}


def _raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit as e:
        return str(e)
    raise AssertionError(f"{fn.__name__} 가 SystemExit 를 내지 않았다")


def t11_lock(tmp):
    p = tmp / ".train.lock"
    f1 = TL.acquire_lock(p)
    assert "잠겨" in _raises(TL.acquire_lock, p)
    rc = subprocess.run([sys.executable, "-c", "import fcntl,sys;f=open(sys.argv[1],'a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)", str(p)],
                        capture_output=True).returncode
    assert rc != 0                                   # 다른 프로세스도 못 잡는다
    f1.close()
    TL.acquire_lock(p).close()                       # 풀린 뒤 다시 잡힌다


def t12_truncate(tmp):
    lg, mo = tmp / "log.jsonl", tmp / "monitor.jsonl"
    lg.write_text("".join(json.dumps(dict(step=i)) + "\n" for i in range(10)) + '{"step": 10, "lo')   # 끊긴 마지막 줄
    mo.write_text("".join(json.dumps(dict(step=s)) + "\n" for s in (50, 100, 150)))
    assert TL.truncate_jsonl(lg, lambda r: r["step"] < 5, "t") == (5, 6)
    assert [json.loads(x)["step"] for x in lg.read_text().splitlines()] == [0, 1, 2, 3, 4]
    bk = (tmp / "log.jsonl.t.dropped").read_text().splitlines()
    assert len(bk) == 6 and bk[-1].startswith('{"step": 10')
    assert TL.truncate_jsonl(mo, lambda r: r["step"] <= 100, "t") == (2, 1)
    assert TL.truncate_jsonl(tmp / "none.jsonl", lambda r: True, "t") == (0, 0)
    assert TL.truncate_jsonl(mo, lambda r: r["step"] <= 100, "t2") == (2, 0) and not (tmp / "monitor.jsonl.t2.dropped").exists()


def t13_server_ready(tmp):
    lp = tmp / "srv.log"
    lp.write_text("old\n[ready] socket=a/score.sock workers=8\n")      # 이전 기동의 준비 줄
    off = lp.stat().st_size
    p = subprocess.Popen(["sleep", "30"])
    try:
        assert "시간 초과" in _raises(TL.wait_ready, p, lp, off, "[ready] socket=a/score.sock", 1.0, 0.1)
        with open(lp, "a") as f:
            f.write("[ready] socket=a/score.sock workers=8\n")
        TL.wait_ready(p, lp, off, "[ready] socket=a/score.sock", 2.0, 0.1)
    finally:
        p.kill(); p.wait()
    q = subprocess.Popen(["true"]); q.wait()
    lp2 = tmp / "srv2.log"; lp2.write_text("")
    assert "종료" in _raises(TL.wait_ready, q, lp2, 0, "[ready]", 5.0, 0.1)


def t14_cont(tmp):
    cont = TL.check_cont_index()
    assert cont[TL.CONT_REF_PROGRESS] == "ref_progress_m"
    fake = tmp / "fake_scorer.py"
    fake.write_text('X = 1\nCONT = ["ref_progress_m", "progress_m"]\n')
    assert "색인" in _raises(TL.check_cont_index, fake)
    fake.write_text("CONT2 = []\n")
    assert "찾지" in _raises(TL.check_cont_index, fake)


def _fake_gates(d, frac=0.3, partner=0.0, det_c=True):
    d.mkdir(parents=True, exist_ok=True)
    (d / "g1.json").write_text(json.dumps(dict(decision="go", kappa=1.0)))
    g2 = dict(deterministic_daf=True, g2_1=dict(pass_=True), g2_2=dict(pass_=False, max_dmu_m=9.97e-3, argmax_match_frac=0.958),
              g2_2_partner_independence=dict(max_dmu_m=partner, max_draw=0.0), g2_3a_pass=True, g2_3b=dict(pass_=True),
              g2_3d_bank_tensors_with_grad=0)
    if det_c:
        rep = dict(max_dmu_m=0.0, grad_rel_median=0.0, grad_rel_max=4e-7)
        g2["g2_3c_det"] = dict(rep_on=rep, rep_off=rep, on_vs_off=dict(max_dmu_m=7.1e-4, grad_rel_median=3.8e-4, grad_rel_max=0.62,
                                                                           params_rel_gt_1e2=[[0.62, "gate_sep"]]))
    (d / "g2_123.json").write_text(json.dumps(g2))
    (d / "g2_7_on").mkdir(exist_ok=True)
    (d / "g2_7_on" / "mem_summary.json").write_text(json.dumps(dict(frac=frac, isolation=True, hold_all_graphs=True)))
    (d / "sigma_ref.json").write_text(json.dumps(dict(sigma_ref=0.3)))
    (d / "g2_6.json").write_text("{}")


def t15_gates(tmp):
    d = tmp / "gates"
    _fake_gates(d)
    r = CG.judge_gates(d, approved=False)
    assert not r["passed"] and set(r["fails"]) == {"G2-2", "G2-3c"}, r["fails"]
    r = CG.judge_gates(d, approved=True)
    assert r["passed"], r["fails"]
    _fake_gates(d, frac=0.85)
    assert CG.judge_gates(d, approved=True)["fails"] == ["G2-7"]
    _fake_gates(d, partner=1e-6)
    assert CG.judge_gates(d, approved=True)["fails"] == ["G2-2"]
    _fake_gates(d, det_c=False)
    assert CG.judge_gates(d, approved=True)["fails"] == ["G2-3c"]
    _fake_gates(d)
    (d / "sigma_ref.json").unlink()
    assert CG.judge_gates(d, approved=True)["fails"] == ["G2-5"]
    _fake_gates(d)                                         # CLI: 승인 파일이 없거나 무효면 엄격 기준 → 종료 코드 1
    init = "results/x/mcmc_s0_ema.pth"
    ap = tmp / "approved"
    cli = [sys.executable, str(HERE.parent / "chain_guard.py"), "gates", "--gates-dir", str(d), "--approved", str(ap), "--init", init]

    def run_cli():
        rc = subprocess.run(cli, capture_output=True).returncode
        return rc, json.loads((d / "gates_check.json").read_text())["approval_status"]
    assert run_cli() == (1, "승인 파일 없음")
    for body, why in (("", "빈 승인 파일"), ("\n  \n", "빈 승인 파일"), (f"G2-2·G2-3c 기준 변경 승인 {init}\n", "결정 일자(YYYY-MM-DD) 없음"),
                      ("2026-11-09 G2-2·G2-3c 기준 변경 승인 results/other.pth\n", f"대상 INIT({init}) 기재 없음")):
        ap.write_text(body)
        assert run_cli() == (1, why), (body, run_cli())
    ap.write_text(f"2026-11-09 사용자 결정: G2-2·G2-3c 기준 변경(impl_log §8·§8.1) 승인. 대상 INIT {init}\n")
    assert run_cli() == (0, "유효")
    assert json.loads((d / "gates_check.json").read_text())["approved_criteria"] is True
    ap.write_text(f"2026-11-09 승인 {os.path.abspath(init)}\n")              # 절대 경로 기재도 유효
    assert run_cli() == (0, "유효")


def t16_claim(tmp):
    d = tmp / "claims_root"
    other = subprocess.Popen(["sleep", "30"])
    try:
        ok, _ = CG.claim(d, "runs", "3", ["full_s0", "b1_s0"], other.pid)
        assert ok
        ok, conf = CG.claim(d, "runs", "1", ["b1_s0", "c1_s0"], os.getpid())
        assert not ok and conf[0]["overlap"] == ["b1_s0"]
        ok, conf = CG.claim(d, "eval", "1", ["full_s0"], os.getpid())       # 다른 mode 도 겹침 검사
        assert not ok
        ok, _ = CG.claim(d, "runs", "1", ["c1_s0", "full_s1"], os.getpid())
        assert ok
    finally:
        other.kill(); other.wait()
    ok, _ = CG.claim(d, "eval", "1", ["full_s0", "b1_s0"], os.getpid())     # 죽은 pid 의 claim 은 무시
    assert ok
    assert subprocess.run([sys.executable, str(HERE.parent / "chain_guard.py"), "claim", "--dir", str(d), "--mode", "runs", "--gpu", "1",
                           "--runs", "nope_s0", "--pid", str(os.getpid())], capture_output=True).returncode == 1


def t17_dump_finalize(tmp):
    import torch
    out = tmp / "dump"; out.mkdir()
    rec = dict(tokens=["a", "b"], frame_tokens=["fa", "fb"], cmd=[0, 1], mu16=[np.zeros((16, 8, 2), np.float32)] * 2,
               raw16=[np.zeros(16, np.float32)] * 2, resc16=[np.zeros(16, np.float32)] * 2, top=[np.arange(4)] * 2, sel=[0, 0])
    (out / "state.pt").write_text("x")
    orig = os.replace

    def bad_replace(a, b):
        if str(b).endswith("manifest.json"):
            raise OSError("모의 중단")
        return orig(a, b)
    os.replace = bad_replace
    try:
        DR.finalize(out, rec, dict(goal=2), 4, 3, time.time())
        raise AssertionError("중단이 전파되지 않았다")
    except OSError:
        pass
    finally:
        os.replace = orig
    assert not (out / "DONE").exists() and (out / "state.pt").exists() and (out / "dump.npz").exists()
    DR.finalize(out, rec, dict(goal=2), 4, 3, time.time())
    assert (out / "DONE").exists() and not (out / "state.pt").exists()
    last = max((out / f).stat().st_mtime_ns for f in ("dump.npz", "sens.npz", "mu_ref.npz", "manifest.json"))
    assert (out / "DONE").stat().st_mtime_ns >= last
    assert json.loads((out / "manifest.json").read_text())["n_recorded"] == 2
    torch.manual_seed(5); np.random.seed(5)
    st = DR.rng_state()
    a1, b1 = torch.rand(3), np.random.rand(3)
    torch.manual_seed(99); np.random.seed(99)
    DR.set_rng_state(st)
    assert torch.equal(torch.rand(3), a1) and np.array_equal(np.random.rand(3), b1)


def t18_check_load(tmp):
    import torch
    import check_load as CL
    root = HERE.parents[3]
    # 체크포인트·config 는 환경변수로(체인은 GRPO_TEST_CKPT=$INIT, GRPO_TEST_CFG=$MODEL_CFG). 기본값은 11-06 스모크 체크포인트.
    cfg = os.environ.get("GRPO_TEST_CFG") or str(root / "configs/navsim/v2_teacher/stage2/test_e3_nuim.py")
    ck = Path(os.environ.get("GRPO_TEST_CKPT") or root / "results/smoke_navsim_prep_20261106/s2/iter_72.pth")
    if not ck.is_absolute():
        ck = root / ck
    if not os.path.isabs(cfg):
        cfg = str(root / cfg)
    assert ck.exists(), f"시험 18 체크포인트 없음: {ck} — GRPO_TEST_CKPT 로 있는 체크포인트를 지정할 것(건너뛰지 않음)"
    assert Path(cfg).exists(), f"시험 18 config 없음: {cfg} — GRPO_TEST_CFG 로 지정"
    cwd = os.getcwd(); os.chdir(root)
    try:
        assert CL.check(cfg, str(ck))["ok"]
        sd = RO._plain_state_dict(torch.load(ck, map_location="cpu", weights_only=False))
        drop = next(k for k in sd if "plan_refine" in k)
        sd.pop(drop)
        bad = tmp / "bad.pth"
        torch.save(dict(state_dict=sd), bad)
        try:
            CL.check(cfg, str(bad))
            raise AssertionError("빠진 키를 놓쳤다")
        except RuntimeError as e:
            assert "빠진 키" in str(e)
        bad.unlink()
    finally:
        os.chdir(cwd)


def t19_gpu_rules():
    class A:
        smoke = False; measure_sigma_ref = 0; hold_all_graphs = False
    env = dict(os.environ)
    try:
        for dev, extra, ok in (("3", {}, True), ("1", {}, False), ("1", {"GRPO_ALLOW_GPU": "1"}, True), ("1", {"GRPO_ALLOW_GPU": "3"}, False),
                               ("2", {"GRPO_ALLOW_GPU": "2"}, False), ("0", {"HIPAD_ALLOW_GPU0": "1"}, False), ("1,3", {}, False)):
            os.environ.pop("GRPO_ALLOW_GPU", None); os.environ.pop("HIPAD_ALLOW_GPU0", None)
            os.environ["CUDA_VISIBLE_DEVICES"] = dev
            os.environ.update(extra)
            if ok:
                assert TL.check_gpu(A()) == dev
            else:
                _raises(TL.check_gpu, A())
        A.smoke = True
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"; os.environ["HIPAD_ALLOW_GPU0"] = "1"
        assert TL.check_gpu(A()) == "0"
    finally:
        os.environ.clear(); os.environ.update(env)


def t20_chain():
    sh = HERE.parents[3] / "scripts/navsim/run_grpo_navsim_chain.sh"
    assert subprocess.run(["bash", "-n", str(sh)]).returncode == 0
    txt = sh.read_text()
    for k in ("chain_guard.py gates", "chain_guard.py claim", ".eval.lock", "check_load.py", '"$1/DONE"', "for r in $ALL_RUNS",
              ".grpo_gate_criteria_approved", "train_gpu$GPU", "g2_7_off/rc", "GRPO_TEST_CKPT=\"$INIT\""):
        assert k in txt, k
    assert "run_mcmc_sft_" not in txt and ".mcmc_sft_chain.lock" not in txt


def t21_diff_regex():
    import re
    root = HERE.parents[3]
    dp = root / "docs/grpo_model_flag_20261109.diff"
    txt = dp.read_text()
    m = re.search(r'^\+_GRPO_PLAN_PARAM = re\.compile\(r"(.*?)"\n\+\s*r"(.*?)"\)', txt, re.M)
    assert m, "diff 에 _GRPO_PLAN_PARAM 없음"
    dec = re.compile(m.group(1) + m.group(2))
    tr = [re.compile(p) for p in RO.TRAINABLE]
    names = ["plan_refine.0.w", "plan_anchor_encoder.1.b", "plan_deformable.2.kps_generator.learnable_fc.bias", "plan_diffusion.t_mlp.0.weight",
             "plan_diffusion.other.weight", "layers.46.plan.layers.0.0.weight", "layers.46.det.layers.0.0.weight", "qwen_token_attn.gate_sep",
             "qwen_token_attn.gate", "qwen_token_attn.attn.in_proj_weight", "det_refine.0.w", "map_deformable.0.w", "ego_refine.0.w",
             "motion_plan_decoder.x", "plan_decoder.rescore.w", "layers.3.plan_x.w"]
    for n in names:
        a = bool(dec.search(n)); b = any(p.search("head.onedecoder_head." + n) for p in tr)
        assert a == b, (n, a, b)
    fwd = subprocess.run(["git", "apply", "--check", str(dp)], cwd=str(root), capture_output=True)
    rev = subprocess.run(["git", "apply", "--reverse", "--check", str(dp)], cwd=str(root), capture_output=True)
    assert fwd.returncode == 0 or rev.returncode == 0, ("diff 가 현재 소스에 적용되지도, 이미 적용된 상태도 아니다", fwd.stderr.decode()[:300])


def t22_chain_run(tmp):
    """체인을 실제로 실행(가짜 트리): DRYRUN 은 태그 폴더를 만들지 않고, 이어진 MODE=gates 본 실행이 claim_dir 를 통과해
    첫 GPU 단계(덤프)까지 간다(검토 B 재검토 결함: 잠금용 mkdir 이 표식 없는 폴더를 만들어 claim_dir 가 die)."""
    import shutil
    import torch
    root = HERE.parents[3]
    T = tmp / "chainroot"
    (T / "scripts/navsim").mkdir(parents=True)
    shutil.copy(root / "scripts/navsim/run_grpo_navsim_chain.sh", T / "scripts/navsim/")
    (T / "scripts/common.sh").write_text('export PYTHONPATH=.\nPY=${PY:-%s}\nlog() { echo "[log] $*"; }\ndie() { log "[중단] $*"; exit 1; }\n'
                                         % sys.executable)
    (T / "scripts/navsim/navsim_env.sh").write_text(f'NAVSIM_PY=/bin/true\nNAVSIM_EXP_ROOT={T}/exp\n')
    (T / "exp/metric_cache_navtrain/metadata").mkdir(parents=True)
    (T / "exp/metric_cache_navtrain/metadata/m.csv").write_text("x\n")
    (T / "data/navsim/grpo/pdm_centerlines_navtrain").mkdir(parents=True)
    (T / "data/navsim/grpo/pdm_centerlines_navtrain/index.json").write_text("{}")
    for f, txt in (("models/decoder/noise_t/plan_diffusion_map.py", "infer_noise_broadcast"), ("models/decoder/unified_decoder.py", "plan_grad_isolation")):
        (T / f).parent.mkdir(parents=True, exist_ok=True); (T / f).write_text(txt)
    gc = T / "configs/navsim/v2_teacher/grpo"; gc.mkdir(parents=True)
    for r in CG.ALL_RUNS:
        (gc / f"{r}.py").write_text("grpo = dict()\n")
    (T / "mcfg.py").write_text("x = 1\n")
    g = T / "tools/navsim/grpo"; (g / "tests").mkdir(parents=True)
    (g / "tests/test_train_loop_cpu.py").write_text("import os,sys\nsys.exit(0 if os.environ.get('GRPO_TEST_CKPT') else 7)\n")
    (g / "check_load.py").write_text("import sys\nsys.exit(0)\n")
    (g / "dump_rollout.py").write_text("import sys\nprint('FAKE_DUMP');sys.exit(1)\n")
    (T / "results").mkdir()
    env = dict(os.environ, GRPO_INIT_CKPT=str(T / "init.pth"), MODEL_CFG="mcfg.py", GRPO_TAG="t22", GRPO_GPU="0", HIPAD_ALLOW_GPU0="1",
               MODE="gates", CHAIN_DRYRUN="1", PY=sys.executable)
    sh = ["bash", str(T / "scripts/navsim/run_grpo_navsim_chain.sh")]
    r1 = subprocess.run(sh, env=env, capture_output=True, text=True, timeout=300)
    assert r1.returncode == 0 and "DRYRUN 끝" in r1.stdout and "아직 없음" in r1.stdout, r1.stdout[-800:] + r1.stderr[-800:]
    assert not (T / "results/grpo_navsim/t22").exists(), "DRYRUN 이 태그 폴더를 만들었다"
    torch.save({"state_dict": {f"k{i}": torch.zeros(1) for i in range(101)}}, T / "init.pth")
    env["CHAIN_DRYRUN"] = "0"
    r2 = subprocess.run(sh, env=env, capture_output=True, text=True, timeout=300)
    out = r2.stdout + r2.stderr
    assert r2.returncode != 0 and "표식" not in out and "덤프 hash_first:1000" in out and "결정적 실패" in out, out[-1500:]
    assert (T / "results/grpo_navsim/t22/.chain_grpo_navsim").exists() and (T / "results/grpo_navsim/t22/gates/.chain_grpo_navsim").exists()
    assert (T / "results/.grpo_navsim_t22_gates.lock").exists()
    r3 = subprocess.run(sh, env=env, capture_output=True, text=True, timeout=300)   # 다시 실행해도 같은 지점(표식 있는 폴더 재사용)
    assert "표식" not in r3.stdout + r3.stderr and "덤프 hash_first:1000" in r3.stdout


def main():
    import tempfile
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
    tests = [t1_lr, t2_top, t3_select, t4_g2_4, t5_ema, t6_holm, lambda: t7_bootstrap(tmp), t8_cls, t9_cands, t10_configs,
             lambda: t11_lock(tmp), lambda: t12_truncate(tmp), lambda: t13_server_ready(tmp), lambda: t14_cont(tmp),
             lambda: t15_gates(tmp), lambda: t16_claim(tmp), lambda: t17_dump_finalize(tmp), lambda: t18_check_load(tmp),
             t19_gpu_rules, t20_chain, t21_diff_regex, lambda: t22_chain_run(tmp)]
    for i, t in enumerate(tests, 1):
        t()
        print(f"[통과] {i}", flush=True)
    print(f"[전부 통과] {len(tests)}")


if __name__ == "__main__":
    main()
