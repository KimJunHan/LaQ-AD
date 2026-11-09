# 수정(2026-11-09): C3 GRPO 설계 v4.1 §9.6(G1)·§9.8/§12-⑤(B0 argmax 채점)·§12 G2-6 — 신규 파일(기존 파일 무수정).
"""덤프(dump_rollout.py 의 dump.npz) → navsim env CPU 채점 관문. 워커 ≤ 16.

모드
  g1  : 장면마다 상위 M 모드 μ 주위 κ ∈ {0.5, 1, 2, 4} 의 AR(1) 차선축 표본 G = 8(η = sample_eta(seed 0, step 0, token, mode) —
        κ 사이 공통 난수)을 [16 ξ=0, M×G] 로 채점(guide 켬, 학습과 같은 K = 48).
        규칙(§9.6 사전 고정): 후보 = {κ : 앵커 그룹 중 EPDMS 분산 > 0(표준편차 > 1e-6) 비율 ≥ 25 %, 그리고 comfort 실패 증가 ≤ 5 %p}.
        comfort 실패 = hc 하위 점수 < 1(채점기 subscores[:,7]). 증가 = 정책 표본의 실패율 − 그 앵커 ξ=0(μ) 실패율(같은 장면·앵커).
        후보 중 가장 작은 κ, 비면 no-go. 같은 덤프로 단계 ①–④ 비율(advantage.scene_advantage Full, 커널 없음 = 사전)도 보고.
        → g1.json {decision, kappa, curves{κ: var_frac, comfort_inc, stage_frac}}
  b0  : 장면마다 ξ=0 최종 궤적(rescore 후 argmax) 하나 채점(진행도 정규화는 후보별 독립이라 K = 1 로 충분).
        → b0_stats.json {n, epdms_mean, ep_mean, stop_rate, speed_mean, subscore_means} + b0_tokens.npz(토큰별 epdms·ep·stop)
        정지 = 끝 변위 < 2 m ∧ ref_progress_m > 5 m(§9.8 과 같은 정의). --token-subset 으로 내부 보류 집합만 셀 수 있다.
  g2_6: 정지 모드(끝 변위 < 0.5 m, rescore 의 static_dis_thresh)의 rescore −999 비율과 그중 공식 nc = 1 비율.
실행: source scripts/navsim/navsim_env.sh && $NAVSIM_PY tools/navsim/grpo/gate_score.py g1 --dump <dir>/dump.npz --out <dir> --workers 16
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import advantage as AD  # noqa: E402
import ar1  # noqa: E402

KAPPAS = (0.5, 1.0, 2.0, 4.0)
VAR_FRAC_MIN, COMFORT_PP_MAX = 0.25, 0.05
HC, EP, NC = 7, 4, 0
_S = {}


def _init(cl_dir, metric_cache):
    import grpo_scorer as GS
    from extract_pdm_centerlines import CenterlineStore
    GS._init_worker(True)
    _S.update(GS=GS, store=CenterlineStore(cl_dir) if cl_dir else None, paths=GS.cache_paths(metric_cache))


def _g1(job):
    tok, mu16, top, resc, G = job
    GS, store = _S["GS"], _S["store"]
    mc = GS.load_cache(_S["paths"][tok])
    cl = store.get(tok)
    E16, _ = ar1.axis_basis(mu16, cl["xyh"][:, :2].astype(np.float64), s_grid=store.s_grid, valid=cl["valid"])
    eta = np.stack([ar1.sample_eta(0, 0, tok, int(q), G) for q in top])
    res = {}
    for kp in KAPPAS:
        sig = ar1.sigma_table(kp)
        tau = np.stack([ar1.sample_tau(mu16[q], E16[q], sig, eta[m]) for m, q in enumerate(top)])
        cands = np.concatenate([mu16, tau.reshape(-1, 8, 2)])
        out = GS.score_token(mc, GS._as_poses(cands), True, guide=True)
        M = len(top)
        e = np.asarray(out["epdms"])[16:].reshape(M, G)
        hc_pol = (np.asarray(out["subscores"])[16:, HC] < 1.0).reshape(M, G)
        hc_mu = np.asarray(out["subscores"])[top, HC] < 1.0
        sc = AD.build_scene(tok, out, M=M, G=G, mu16=mu16, top_modes=top)
        st = AD.scene_advantage(sc, "Full", None, rescored_logits=resc)["stage"]
        res[kp] = dict(var=(e.std(-1) > AD.EQ_TOL).tolist(), hc_pol=float(hc_pol.mean()), hc_mu=float(np.repeat(hc_mu, G).mean()),
                       stage=st.tolist())
    return tok, res


def _b0(job):
    tok, traj = job
    GS = _S["GS"]
    mc = GS.load_cache(_S["paths"][tok])
    out = GS.score_token(mc, GS._as_poses(traj[None]), True, guide=False)
    return tok, float(out["epdms"][0]), np.asarray(out["subscores"][0], np.float64), float(out["cont"][0, 1])


def _g26(job):
    tok, modes = job
    GS = _S["GS"]
    mc = GS.load_cache(_S["paths"][tok])
    out = GS.score_token(mc, GS._as_poses(modes), True, guide=False)
    return tok, np.asarray(out["subscores"])[:, NC]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["g1", "b0", "g2_6"])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric-cache", default=os.path.join(os.environ.get("NAVSIM_EXP_ROOT", ""), "metric_cache_navtrain"))
    ap.add_argument("--centerlines", default="data/navsim/grpo/pdm_centerlines_navtrain")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--token-subset", default=None, help="b0: 이 npz/txt 의 토큰만(내부 보류 집합)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.workers > 16:
        raise SystemExit("워커 상한 16")
    import grpo_scorer as GS
    CONT = list(GS.CONT)
    if CONT[1] != "ref_progress_m":
        raise SystemExit(f"CONT[1] = {CONT[1]} — ref_progress_m 가정 위반")
    z = np.load(a.dump, allow_pickle=False)
    toks = [str(t) for t in z["tokens"]]
    n = len(toks) if not a.max_tokens else min(a.max_tokens, len(toks))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ctx = get_context("fork")
    if a.mode == "g1":
        jobs = [(toks[i], z["mu16"][i].astype(np.float64), z["top"][i], z["resc16"][i].astype(np.float64), a.G) for i in range(n)]
        acc = {kp: dict(var=[], hc_pol=[], hc_mu=[], stage=[]) for kp in KAPPAS}
        with ctx.Pool(a.workers, initializer=_init, initargs=(a.centerlines, a.metric_cache)) as pool:
            for j, (tok, r) in enumerate(pool.imap(_g1, jobs, chunksize=4)):
                for kp in KAPPAS:
                    for key in acc[kp]:
                        (acc[kp][key].extend if isinstance(r[kp][key], list) else acc[kp][key].append)(r[kp][key])
                if (j + 1) % 100 == 0:
                    print(f"[G1] {j + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
        curves, cand = {}, []
        for kp in KAPPAS:
            var = float(np.mean(acc[kp]["var"]))
            inc = float(np.mean(acc[kp]["hc_pol"]) - np.mean(acc[kp]["hc_mu"]))
            st = np.bincount(np.asarray(acc[kp]["stage"]), minlength=5)[1:]
            curves[str(kp)] = dict(var_frac=var, comfort_fail_pol=float(np.mean(acc[kp]["hc_pol"])),
                                   comfort_fail_mu=float(np.mean(acc[kp]["hc_mu"])), comfort_inc=inc,
                                   stage_frac=(st / max(1, st.sum())).tolist(), zero_var_frac=1 - var)
            if var >= VAR_FRAC_MIN and inc <= COMFORT_PP_MAX:
                cand.append(kp)
        res = dict(decision="go" if cand else "no-go", kappa=min(cand) if cand else None, candidates=cand, curves=curves,
                   rule=f"var_frac ≥ {VAR_FRAC_MIN} ∧ comfort_inc ≤ {COMFORT_PP_MAX}, 가장 작은 κ(§9.6)", n_tokens=n,
                   dump=os.path.abspath(a.dump), seconds=time.time() - t0)
        (out / "g1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(json.dumps(res, ensure_ascii=False, indent=1))
    elif a.mode == "b0":
        keep = None
        if a.token_subset:
            p = Path(a.token_subset)
            keep = set(np.load(p)["tokens"].astype(str)) if p.suffix == ".npz" else set(p.read_text().split())
        idx = [i for i in range(n) if keep is None or toks[i] in keep]
        jobs = [(toks[i], z["mu16"][i][int(z["sel"][i])].astype(np.float64)) for i in idx]
        rows = []
        with ctx.Pool(a.workers, initializer=_init, initargs=(None, a.metric_cache)) as pool:
            for j, r in enumerate(pool.imap(_b0, jobs, chunksize=16)):
                rows.append(r)
                if (j + 1) % 2000 == 0:
                    print(f"[B0] {j + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
        ep = np.array([r[2][EP] for r in rows])
        epdms = np.array([r[1] for r in rows])
        endd = np.array([np.linalg.norm(z["mu16"][i][int(z["sel"][i])][-1]) for i in idx])
        refp = np.array([r[3] for r in rows])
        stop = (endd < 2.0) & (refp > 5.0)
        sub = np.stack([r[2] for r in rows])
        res = dict(n=len(rows), epdms_mean=float(epdms.mean()), ep_mean=float(ep.mean()), stop_rate=float(stop.mean()),
                   speed_mean=float((endd / 4.0).mean()), subscore_means=dict(zip(list(GS.SUBSCORES), sub.mean(0).tolist())),
                   dump=os.path.abspath(a.dump), token_subset=a.token_subset, seconds=time.time() - t0,
                   stop_def="끝 변위 < 2 m ∧ ref_progress_m > 5 m (§9.8)")
        name = f"b0_stats{a.tag}.json"
        (out / name).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        np.savez(out / f"b0_tokens{a.tag}.npz", tokens=np.array([r[0] for r in rows]), epdms=epdms, ep=ep, stop=stop,
                 subscores=sub)
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        jobs, meta = [], []
        for i in range(n):
            mu = z["mu16"][i].astype(np.float64)
            st = np.linalg.norm(mu[:, -1] - mu[:, 0], axis=-1) < 0.5
            if st.any():
                jobs.append((toks[i], mu[st]))
                meta.append(z["resc16"][i][st] < z["raw16"][i][st] - 1.0)
        nc = []
        with ctx.Pool(a.workers, initializer=_init, initargs=(None, a.metric_cache)) as pool:
            for r in pool.imap(_g26, jobs, chunksize=8):
                nc.append(r[1])
        m = np.concatenate(meta) if meta else np.zeros(0, bool)
        ncv = np.concatenate(nc) if nc else np.zeros(0)
        res = dict(n_static_modes=int(m.size), n_tokens_with_static=len(jobs), sentinel_rate=float(m.mean()) if m.size else None,
                   official_nc1_among_sentinel=float((ncv[m] == 1.0).mean()) if m.any() else None,
                   official_nc1_among_static=float((ncv == 1.0).mean()) if m.size else None, seconds=time.time() - t0)
        (out / "g2_6.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
