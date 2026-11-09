# 수정(2026-11-06): C3 GRPO 설계 v4 §4.2 G0(시뮬레이터 민감도 → 커널 w_f·축 책임 u_a) 구현 M5 — 신규 파일.
"""G0 민감도 측정 → advantage.py 가 읽는 커널 json.

절차
  1) 선별: 장면마다 기준 궤적 τ0 (학생 G0: 덤프 μ; 예비 G0: 사람 GT·PDM-Closed 0.5 s × 8) 주위에 AR(1) 차선축 잡음
     (ar1, κ, ρ 0.8, RNG = default_rng([seed, token_hash, base_id])) 표본 G 개를 뽑아 채점하고, 주 실패 f*(advantage.primary_failure,
     사람 필터 반영)가 있는 표본을 유형별로 모은다(유형당 상한 --per-type, token 정렬 순서로 결정적).
  2) 측정: 고른 표본마다 혁신 하나 η_{k,a} 를 ±1 바꾼 32 후보(k 0..7 × a 2 × 부호 2)를 다시 채점한다.
     · 연속 여유 m_f: collision → collision_time_s(첫 at-fault 충돌 시각, 없으면 4.0), ttc → ttc_time_s, offroad → drivable_margin_m,
       direction → oncoming_max_m, red_light → 첫 적신호 교차 시각(없으면 4.0).
     · 거리 기반 대안(충돌·ttc, 검토 1-3, 11-06): 첫 충돌 시각만 쓰면 충돌 뒤 혁신의 영향이 0 으로 잘린다. 그래서
       min_obj_dist_m(모든 객체 최소 거리)·min_obj_dist_fs_m(전방·측방 객체만, §6 c 와 같은 정의) 감도로 만든 커널도 같이 낸다.
       json 의 types_by_margin[<여유 키>][<유형>] 에 넣고, advantage 가 읽는 types 는 --coll-margin(기본 time = 종전) 으로 고른다.
       기본값 변경은 사용자 결정 전까지 하지 않는다.
     · 감도 = |m(+1) − m(−1)| / 2, Δ = k − j_f ∈ [−7, 7].
     · 이진: 그 유형의 실패 상태(fail_wp; 없음 = −1)가 기준 표본과 달라진 섭동 비율.
  3) 커널(§4.2 확정 규칙, 선택지 (i)):
     S_{a,f}[Δ] = 감도 평균, S_f = Σ_a S_{a,f}, shape_f[Δ] = S_f / max_Δ S_f, u_a = 2 Σ_Δ S_{a,f} / Σ_{a',Δ} S_{a',f}.
     표본 ≥ 50 이면 status = measured, 아니면 prior(advantage 가 사전 커널·기본 u 를 쓴다).
     표본이 없는 Δ 는 가장 가까운 측정 Δ 값으로 채우고 filled_delta 에 기록한다(advantage 는 NaN 을 거부한다).
     진단: 축별 정규화 모양, 두 축 최대 위치 차(≥ 2 경유점이면 axis_peak_mismatch), max/min 비(원 S_f, w0 바닥 적용 w 둘 다),
     이진 변화율과 연속 감도의 순위 상관(Spearman).

실행(navsim env, CPU, 워커 ≤ 16):
  source scripts/navsim/navsim_env.sh
  $NAVSIM_PY tools/navsim/grpo/sensitivity.py --metric-cache $NAVSIM_EXP_ROOT/metric_cache_navtrain --bases human,pdm \
      --per-type 50 --max-tokens 4000 --workers 16 --label 예비 --out <dir>
  학생 G0(G1 덤프)는 --dump <npz: tokens (n,), mu (n, Q, 8, 2) ego 프레임> 로 기준 궤적을 준다.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import advantage as AD  # noqa: E402
import ar1  # noqa: E402
from extract_pdm_centerlines import S_GRID, extract_one  # noqa: E402  (읽기만)

MAX_WORKERS = 16
KT = AD.KTYPES
MARGIN = {"collision_front": "collision_time_s", "collision_side": "collision_time_s", "ttc": "ttc_time_s",
          "offroad": "drivable_margin_m", "direction": "oncoming_max_m", "red_light": "red_light_time_s"}
DIST_KEYS = ("min_obj_dist_m", "min_obj_dist_fs_m")
DIST_TYPES = ("collision_front", "collision_side", "ttc")
FCOL = {"collision_front": 0, "collision_side": 0, "ttc": 1, "offroad": 2, "direction": 3, "red_light": 4}
MIN_N = 50
_S = {}


def _init(margins=True):
    import grpo_scorer as GS
    GS._init_worker(margins)
    _S["GS"] = GS


def _bases_for(mc, which, dump_mu=None):
    GS = _S["GS"]
    out = []
    if "human" in which and mc.human_trajectory is not None:
        out.append(("human", np.asarray(mc.human_trajectory.poses, np.float64)[:, :2]))
    if "pdm" in which:
        st = GS.get_trajectory_as_array(mc.trajectory, _S["GS"]._W["sampling"], mc.ego_state.time_point)
        out.append(("pdm", GS._ego_frame_poses(st, mc.ego_state)[:, :2]))
    if dump_mu is not None:
        for q, m in enumerate(dump_mu):
            out.append((f"mode{q}", np.asarray(m, np.float64)))
    return out


def _margins(o, k):
    c = o["cont"][k]
    GS = _S["GS"]
    m = {n: float(c[GS.CONT.index(n)]) for n in ("collision_time_s", "ttc_time_s", "drivable_margin_m", "oncoming_max_m",
                                                   "min_obj_dist_m")}
    m["min_obj_dist_fs_m"] = float(o["min_obj_dist_fs_m"][k])
    t = o["fail_idx"][k, 4]
    m["red_light_time_s"] = 4.0 if t < 0 else 0.1 * float(t)
    return m


def _screen(args):
    tok, path, which, kappa, G, seed, dump_mu = args
    GS = _S["GS"]
    mc = GS.load_cache(path)
    cl = extract_one(mc)
    sig = ar1.sigma_table(kappa)
    cands, meta = [], []
    for bi, (bname, base) in enumerate(_bases_for(mc, which, dump_mu)):
        E, info = ar1.axis_basis(base, cl["xyh"][:, :2].astype(np.float64), s_grid=S_GRID, valid=cl["valid"])
        rng = np.random.default_rng([int(seed), ar1.token_hash(tok), bi])
        eta = rng.standard_normal((G, 8, 2))
        for g in range(G):
            cands.append(ar1.sample_tau(base, E, sig, eta[g]))
            meta.append(dict(base=bname, bi=bi, eta=eta[g], tau0=base, E=E, oor=int(info["out_of_range"].sum())))
    if not cands:
        return tok, []
    o = GS.score_token(mc, GS._as_poses(np.stack(cands)), True)
    found = []
    for k, mt in enumerate(meta):
        f, jf = AD.primary_failure(o["fail_idx"][k], o["fail_wp"][k], bool(o["coll_front"][k]))
        if f is None:
            continue
        found.append(dict(tok=tok, path=path, ftype=f, jf=jf, epdms=float(o["epdms"][k]), ctype=o["coll_ctype"][k],
                          fail_wp=o["fail_wp"][k].copy(), m0=_margins(o, k), **mt))
    return tok, found


def _measure(args):
    s, kappa = args
    GS = _S["GS"]
    mc = GS.load_cache(s["path"])
    sig = ar1.sigma_table(kappa)
    cands = [ar1.sample_tau(s["tau0"], s["E"], sig, s["eta"])]
    idx = []
    for k in range(8):
        for a in range(2):
            for sg in (1.0, -1.0):
                e = s["eta"].copy()
                e[k, a] += sg
                cands.append(ar1.sample_tau(s["tau0"], s["E"], sig, e))
                idx.append((k, a, sg))
    o = GS.score_token(mc, GS._as_poses(np.stack(cands)), True)
    f = s["ftype"]
    col = FCOL[f]
    base_state = int(o["fail_wp"][0, col])
    base_f, _ = AD.primary_failure(o["fail_idx"][0], o["fail_wp"][0], bool(o["coll_front"][0]))
    sens = np.zeros((8, 2))
    sens_obj = np.zeros((8, 2))
    sens_fs = np.zeros((8, 2))
    flip = np.zeros((8, 2))
    mvals = {}
    for j, (k, a, sg) in enumerate(idx):
        mvals[(k, a, sg)] = (_margins(o, j + 1), int(o["fail_wp"][j + 1, col]))
    for k in range(8):
        for a in range(2):
            (mp, sp), (mm, sm) = mvals[(k, a, 1.0)], mvals[(k, a, -1.0)]
            sens[k, a] = abs(mp[MARGIN[f]] - mm[MARGIN[f]]) / 2
            sens_obj[k, a] = abs(mp["min_obj_dist_m"] - mm["min_obj_dist_m"]) / 2          # NaN(대상 객체 없음)은 build_kernel 이 뺀다
            sens_fs[k, a] = abs(mp["min_obj_dist_fs_m"] - mm["min_obj_dist_fs_m"]) / 2
            flip[k, a] = ((sp != base_state) + (sm != base_state)) / 2
    return dict(tok=s["tok"], base=s["base"], ftype=f, jf=s["jf"], sens=sens, sens_obj=sens_obj, sens_fs=sens_fs, flip=flip,
                repro=(base_f == f), ctype=s["ctype"])


def _spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


SENS_FIELD = {"time": "sens", "min_obj_dist_m": "sens_obj", "min_obj_dist_fs_m": "sens_fs"}


def _type_entry(rs, field, w0, min_n):
    """한 유형의 커널 항목. field = 결과 dict 의 감도 배열 키(sens | sens_obj | sens_fs). NaN 감도(거리 정의 안 됨)는 평균에서 뺀다."""
    n = len(rs)
    e = dict(n=n, status="measured" if n >= min_n else "prior")
    if n == 0:
        return e
    S_a = np.full((15, 2), np.nan)
    F = np.full(15, np.nan)
    cnt = np.zeros(15, int)
    n_nan = 0
    for d in range(-7, 8):
        vals, fl = [], []
        for r in rs:
            k = r["jf"] + d
            if 0 <= k < 8:
                v = r[field][k]
                if not np.isfinite(v).all():
                    n_nan += 1
                    continue
                vals.append(v)
                fl.append(r["flip"][k].mean())
        cnt[d + 7] = len(vals)
        if vals:
            S_a[d + 7] = np.mean(vals, 0)
            F[d + 7] = np.mean(fl)
    S = S_a.sum(-1)
    meas = np.isfinite(S)
    if not meas.any():
        e.update(status="prior", note="측정 Δ 없음(감도 전부 NaN)")
        return e
    filled = []
    Sf = S.copy()
    Saf = S_a.copy()
    mi = np.nonzero(meas)[0]
    for i in np.nonzero(~meas)[0]:
        j = mi[np.argmin(np.abs(mi - i))]
        Sf[i], Saf[i] = S[j], S_a[j]
        filled.append(int(i - 7))
    mx = Sf.max()
    if mx <= 0:
        e.update(status="prior", note="모든 Δ 감도 0 — 커널 정의 불가(사전 커널 사용)")
        return e
    shape = Sf / mx
    tot = Saf[meas].sum(0)
    u = 2 * tot / tot.sum()
    wfl = np.maximum(w0, shape)
    ax_shape = {a: (Saf[:, i] / Saf[:, i].max()).tolist() if Saf[:, i].max() > 0 else [0.0] * 15
                for i, a in enumerate(("lon", "lat"))}
    pk = [int(mi[np.argmax(S_a[mi, i])]) - 7 for i in range(2)]      # 최대 위치는 측정 Δ 에서만(채운 Δ 제외)
    smin = S[meas].min()
    e.update(shape=shape.tolist(), u=dict(lon=float(u[0]), lat=float(u[1])), S_axis=Saf.tolist(), count_per_delta=cnt.tolist(),
             filled_delta=filled, shape_axis=ax_shape, axis_peak_delta=dict(lon=pk[0], lat=pk[1]),
             axis_peak_mismatch=bool(abs(pk[0] - pk[1]) >= 2),
             max_min_ratio_S=float(S[meas].max() / smin) if smin > 0 else float("inf"),
             max_min_ratio_w=float(wfl.max() / wfl.min()),
             flat=bool((wfl.max() / wfl.min()) < 1.5),
             flip_rate=np.where(np.isfinite(F), F, -1).tolist(), spearman_S_vs_flip=_spearman(S, F),
             repro_frac=float(np.mean([r["repro"] for r in rs])),
             by_base={b: int(sum(r["base"] == b for r in rs)) for b in sorted({r["base"] for r in rs})},
             n_nan_excluded=int(n_nan),
             # 진단: 표본 < 10 인 Δ(평균이 소수 표본에 좌우됨)와, 그런 Δ 를 뺀 max/min·최대 위치(커널 규칙은 바꾸지 않음)
             low_count_delta=[int(i - 7) for i in range(15) if 0 < cnt[i] < 10],
             max_min_ratio_S_n10=(float(S[cnt >= 10].max() / S[cnt >= 10].min()) if (cnt >= 10).any() and S[cnt >= 10].min() > 0
                                  else float("inf")),
             peak_delta_n10=(int(np.nonzero(cnt >= 10)[0][np.argmax(S[cnt >= 10])]) - 7) if (cnt >= 10).any() else None)
    if field == "sens":
        e["S_obj_mean"] = float(np.nanmean([r["sens_obj"].sum(-1).mean() for r in rs]))
    return e


def build_kernel(results, label, w0=AD.W0, min_n=MIN_N, meta=None, coll_margin="time"):
    """types = advantage 가 읽는 커널(충돌·ttc 는 coll_margin 의 여유, 나머지는 MARGIN 표).
    types_by_margin = {"time": 시각 여유(종전), "min_obj_dist_m": ..., "min_obj_dist_fs_m": ...} — 거리 키는 충돌·ttc 만."""
    if coll_margin not in SENS_FIELD:
        raise ValueError(f"coll_margin {coll_margin} ∉ {list(SENS_FIELD)}")
    by = {}
    for key, field in SENS_FIELD.items():
        fl = KT if key == "time" else DIST_TYPES
        by[key] = {}
        for f in fl:
            rs = [r for r in results if r["ftype"] == f]
            if key != "time" and rs and field not in rs[0]:
                continue                                        # 옛 결과(sens_fs 없음)
            e = _type_entry(rs, field, w0, min_n)
            e["margin"] = (MARGIN[f] if key == "time" else key)
            by[key][f] = e
    types = {f: (by[coll_margin][f] if (f in DIST_TYPES and f in by[coll_margin]) else by["time"][f]) for f in KT}
    return dict(label=label, w0=w0, rule="docs/c3_grpo_design.md §4.2 선택지 (i)", meta=meta or {}, coll_margin=coll_margin,
                types=types, types_by_margin=by)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bases", default="human,pdm")
    ap.add_argument("--dump", default=None, help="학생 G0: npz(tokens, mu (n,Q,8,2) ego 프레임)")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--per-type", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="예비")
    ap.add_argument("--coll-margin", default="time", choices=list(SENS_FIELD),
                    help="types(advantage 가 읽음)의 충돌·ttc 여유. 기본 time = 종전(첫 충돌·TTC 시각). 거리 커널은 항상 types_by_margin 에 같이 낸다")
    a = ap.parse_args()
    if a.workers > MAX_WORKERS:
        raise ValueError(f"워커 상한 {MAX_WORKERS}")
    sys.path.insert(0, str(HERE.parent))
    import grpo_scorer as GS
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = GS.cache_paths(a.metric_cache)
    dump = None
    if a.dump:
        z = np.load(a.dump)
        dump = {str(t): m for t, m in zip(z["tokens"], z["mu"])}
        toks = sorted(dump)
    else:
        toks = sorted(paths)
    rng = np.random.default_rng(a.seed)
    order = [toks[i] for i in rng.permutation(len(toks))[:a.max_tokens]]
    which = a.bases.split(",") if not a.dump else []
    picked = {f: [] for f in KT}
    t0 = time.time()
    n_tok = 0
    n_samples = 0
    with get_context("fork").Pool(a.workers, initializer=_init) as pool:
        for c0 in range(0, len(order), a.chunk):
            chunk = order[c0:c0 + a.chunk]
            jobs = [(t, paths[t], which, a.kappa, a.G, a.seed, None if dump is None else dump[t]) for t in chunk]
            for tok, found in pool.imap(_screen, jobs):          # 순서 보존 → 결정적 선택
                n_tok += 1
                n_samples += a.G * (len(which) if dump is None else len(dump[tok]))
                for s in found:
                    if len(picked[s["ftype"]]) < a.per_type:
                        picked[s["ftype"]].append(s)
            print(f"[선별] token {n_tok} ({time.time() - t0:.0f}s): " + ", ".join(f"{f} {len(picked[f])}" for f in KT), flush=True)
            if all(len(picked[f]) >= a.per_type for f in KT):
                break
        sel = [s for f in KT for s in picked[f]]
        t1 = time.time()
        res = list(pool.imap(_measure, [(s, a.kappa) for s in sel]))
        print(f"[측정] 표본 {len(res)} × 32 섭동 ({time.time() - t1:.0f}s)", flush=True)
    meta = dict(bases=a.bases if not a.dump else "student_dump", kappa=a.kappa, G=a.G, seed=a.seed, tokens_screened=n_tok,
                samples_screened=n_samples, per_type_cap=a.per_type, metric_cache=a.metric_cache,
                date="2026-11-06", note="예비(논문 예비 결과 전용) — 커널 확정은 학생 G1 덤프 G0 로만(§4.2)" if a.label == "예비" else "")
    k = build_kernel(res, a.label, meta=meta, coll_margin=a.coll_margin)
    (out / "g0_kernel.json").write_text(json.dumps(k, ensure_ascii=False, indent=1))
    np.savez(out / "g0_raw.npz", tokens=np.array([r["tok"] for r in res]), base=np.array([r["base"] for r in res]),
             ftype=np.array([r["ftype"] for r in res]), jf=np.array([r["jf"] for r in res]),
             sens=np.stack([r["sens"] for r in res]) if res else np.zeros((0, 8, 2)),
             sens_obj=np.stack([r["sens_obj"] for r in res]) if res else np.zeros((0, 8, 2)),
             sens_fs=np.stack([r["sens_fs"] for r in res]) if res else np.zeros((0, 8, 2)),
             flip=np.stack([r["flip"] for r in res]) if res else np.zeros((0, 8, 2)),
             ctype=np.array([r["ctype"] for r in res]), repro=np.array([r["repro"] for r in res]))
    rows = [("time", f, k["types_by_margin"]["time"][f]) for f in KT]
    rows += [(m, f, k["types_by_margin"][m][f]) for m in DIST_KEYS for f in DIST_TYPES if f in k["types_by_margin"][m]]
    for m, f, e in rows:
        if "shape" in e:
            print(f"[{m:17s}] {f:16s} n {e['n']:3d} {e['status']:8s} u lon/lat {e['u']['lon']:.2f}/{e['u']['lat']:.2f} "
                  f"max/min S {e['max_min_ratio_S']:.2f} w {e['max_min_ratio_w']:.2f} peak lon/lat {e['axis_peak_delta']} "
                  f"filled {e['filled_delta']} ρ(S,flip) {e['spearman_S_vs_flip']:.2f}")
        else:
            print(f"[{m:17s}] {f:16s} n {e['n']:3d} {e['status']}")
    print(f"→ {out / 'g0_kernel.json'} (총 {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
