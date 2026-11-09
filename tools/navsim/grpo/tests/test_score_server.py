# 수정(2026-11-06): §12-③ 채점 서버 동등성·처리량 시험(신규 파일).
"""score_server 시험 — hipad_bw env 에서 실행(클라이언트 쪽). navsim env 하위 프로세스로 기준값과 서버를 띄운다.
  source /home1/irteam/data-vol1/hipad_env.sh
  python tools/navsim/grpo/tests/test_score_server.py --n 200 --K 48 --mb 2 --workers 8 16 --scratch <dir>
절차
  1) [navsim env] --mode ref: 무작위 navtrain n token, 후보 K 개 = 사람 GT 미래(MetricCache.human_trajectory) + 경유점별로 커지는
     섭동(짝수 token (K,8,2), 홀수 token (K,8,3) heading 섭동 포함; 후보 0 = GT 그대로) → GRPOScorer.score(blocking, 워커 16)
     결과·시간을 npz 로 저장.
  2) [hipad_bw] 워커 수마다 서버를 띄우고 ScoreClient 로 mb token 씩 micro-batch 를 연달아 submit → poll 로 모두 받음.
     모든 출력 키가 blocking 결과와 비트 동일(NaN 위치 포함)인지 확인, 처리량(장면/s, 후보/s)·micro-batch 지연 출력.
  3) 클라이언트 프로세스에 nuplan·navsim 이 import 되지 않았는지, 잘못된 요청(없는 token·형식·중복 id)이 예외인지 확인.
실패는 AssertionError/RuntimeError(비영 종료).
"""
import os

# BLAS 스레드 1 고정은 numpy import 전에 해야 한다(grpo_scorer 의 setdefault 는 numpy 가 먼저 import 되면 효과 없음).
# 실측(11-06): numpy 를 먼저 import 해 OpenBLAS 가 다중 스레드면 같은 입력의 frenet·lat_dev_wp_m 이 호출마다 1 ulp 씩 달라졌다
# (7 token × 4 반복 중 1 회); 1 스레드로 고정하면 0 회.
# 수정(2026-11-06, 검토 2): 이 1 ulp 의 원인은 BLAS 스레드가 아니라 numpy cos/sin 경로 선택(아래 ULP_KEYS 주석). 고정은 유지.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE.parents[1]))
NAVSIM_SH = REPO / "scripts/navsim/navsim_env.sh"


def _navsim_cmd(args):
    """깨끗한 환경에서 navsim env 를 source 하고 실행(hipad_bw 변수 섞임 방지)."""
    env = {k: os.environ[k] for k in ("HOME", "USER", "LOGNAME", "TMPDIR") if k in os.environ}
    env["PATH"] = "/usr/bin:/bin"
    # 수정(2026-11-06): 검토 2 — --npy-disable 이면 numpy SIMD 기능을 끈 채로(SVML 경로 제거) 기준값·서버를 띄운다.
    #   실측: 기본(AVX512_SKX 켬)은 같은 입력도 워커·할당 이력에 따라 frenet·lat_dev 가 ≤ 1e-15, cont(comfort_ratio_max, np.round(8)
    #   경계 증폭)가 ~5e-9 달라질 수 있고, NPY_DISABLE_CPU_FEATURES=AVX512_SKX 면 워커 16/8/5 결과가 모든 키 비트 동일.
    if os.environ.get("_TSS_NPY_DISABLE"):
        env["NPY_DISABLE_CPU_FEATURES"] = os.environ["_TSS_NPY_DISABLE"]
    cmd = f"source {NAVSIM_SH} && cd {REPO} && exec $NAVSIM_PY " + " ".join(map(str, args))
    return ["bash", "-c", cmd], env


# ---------------------------------------------------------------- navsim env: 기준값
def _ref(a):
    sys.path.insert(0, str(REPO / "tools/navsim"))
    from multiprocessing import get_context
    import grpo_scorer as G
    paths = G.cache_paths(a.metric_cache)
    toks = sorted(paths)
    rng = np.random.default_rng(a.seed)
    toks = [toks[i] for i in rng.choice(len(toks), a.n, replace=False)]
    with get_context("fork").Pool(16) as p:
        gts = p.map(_gt_pose, [paths[t] for t in toks])
    sc = np.linspace(0.15, 1.0, 8)[None, :, None]
    batch = {}
    for i, (t, gt) in enumerate(zip(toks, gts)):
        c = np.repeat(gt[None], a.K, 0)
        c[1:, :, :2] += rng.normal(0, a.noise, (a.K - 1, 8, 2)) * sc
        if i % 2 == 0:
            c = c[:, :, :2]
        else:
            c[1:, :, 2] += rng.normal(0, 0.05, (a.K - 1, 8)) * sc[..., 0]
        batch[t] = c
    with G.GRPOScorer(a.metric_cache, workers=16, margins=True, guide=True) as s:   # 수정(2026-11-06): 서버 기본 guide=True 와 같은 키
        t0 = time.time()
        out = s.score(batch)
        wall = time.time() - t0
    save = {"tokens": np.array(toks), "wall": np.array(wall)}
    for t in toks:
        save[f"in/{t}"] = batch[t]
        for k, v in out[t].items():
            save[f"out/{t}/{k}"] = np.asarray(v)
    np.savez(a.out, **save)
    print(f"[ref] blocking GRPOScorer.score 워커 16: {a.n} token × K={a.K} {wall:.1f}s → {a.n / wall:.2f} 장면/s "
          f"({a.n * a.K / wall:.0f} 후보/s)", flush=True)


def _gt_pose(path):
    sys.path.insert(0, str(REPO / "tools/navsim/grpo"))
    import extract_pdm_centerlines as E
    mc = E.load_cache(path)
    return np.asarray(mc.human_trajectory.poses, np.float64)


# ---------------------------------------------------------------- hipad_bw: 서버 시험
# 수정(2026-11-06, 검토 2): 원인 확인 — numpy 1.23 np.cos/np.sin 이 출력 버퍼와 입력의 인접(겹침) 여부로 SVML·libm 경로를 고르고
#   두 경로가 마지막 ulp 에서 다르다(OpenBLAS 스레드 아님, _frenet 에 BLAS 없음). ≤ 1e-15 m 라 허용. #이전 "미해결(…확인 필요)"
# frenet·lat_dev_wp_m 은 서버 없이 직접 score_token 을 불러도 프로세스·환경에 따라
# 마지막 몇 ulp(≤ 1e-15 m) 가 달라진다(실측: 같은 입력, 셸 env 프로세스 vs 깨끗한 env 프로세스, 깨끗한 env 에서는 앞서 채점한 token 에도
# 의존). 전송은 무손실(_dec(_enc(x)) 입력의 채점 = 원 입력 채점, 비트 동일). 그래서 이 두 키만 |차| ≤ ULP_TOL 로 비교하고 개수를 보고한다.
ULP_KEYS = ("frenet", "lat_dev_wp_m")
ULP_TOL = 1e-12
ULP_STATS = {"elems": 0, "max": 0.0}
# 수정(2026-11-06): 검토 3 — 사용자 결정으로 NPY_DISABLE_CPU_FEATURES 를 적용하지 않으므로, 같은 SVML/libm ulp 차이가 시뮬레이션
#   상태를 거쳐 comfort 신호의 np.round(…, 8) 경계에서 커진 cont[:, comfort_ratio_max] 차이(실측 4.8e-9, 같은 token·원소에서 재현)를
#   이 한 열에만 ≤ 2e-8(검토 4, 종전 1e-7) 허용하고 개수를 보고한다. 하위 점수·EPDMS·fail_idx 는 여전히 비트 동일을 요구한다.
#   --npy-disable AVX512_SKX 로 돌리면 이 허용 없이 전 키 비트 동일(검토 2 실측).
CONT_ROUND_COL = 8          # grpo_scorer.CONT.index("comfort_ratio_max")
CONT_ROUND_TOL = 2e-8      # 수정(2026-11-06): 검토 4 — 1e-7 → 2e-8(실측 최대 4.8e-9)
CONT_STATS = {"elems": 0, "max": 0.0}


def _cmp(ref, got, tok):
    keys = {k.split("/", 2)[2] for k in ref.files if k.startswith(f"out/{tok}/")}
    gk = set(got) - {"_timing"}
    assert keys == gk, (tok, keys ^ gk)
    for k in keys:
        a, b = ref[f"out/{tok}/{k}"], np.asarray(got[k])
        assert a.shape == b.shape and a.dtype == b.dtype, (tok, k, a.shape, b.shape, a.dtype, b.dtype)
        if k in ULP_KEYS and not np.array_equal(a, b, equal_nan=True):
            diff = np.abs(a - b)
            if not (np.isnan(a) == np.isnan(b)).all() or np.nanmax(diff) > ULP_TOL:
                raise AssertionError(f"{tok} {k}: |차| {np.nanmax(diff):.3e} > {ULP_TOL}")
            ULP_STATS["elems"] += int((diff > 0).sum())
            ULP_STATS["max"] = max(ULP_STATS["max"], float(np.nanmax(diff)))
            continue
        if k == "cont" and not np.array_equal(a, b, equal_nan=True):          # 수정(2026-11-06): 검토 3 — 위 CONT_ROUND 주석
            other = np.ones(a.shape[-1], bool)
            other[CONT_ROUND_COL] = False
            ca, cb = a[:, CONT_ROUND_COL], b[:, CONT_ROUND_COL]
            if np.array_equal(a[:, other], b[:, other], equal_nan=True) and np.array_equal(np.isnan(ca), np.isnan(cb)):
                d = np.abs(np.where(np.isnan(ca), 0.0, ca - cb))
                if d.max() <= CONT_ROUND_TOL:
                    CONT_STATS["elems"] += int((d > 0).sum())
                    CONT_STATS["max"] = max(CONT_STATS["max"], float(d.max()))
                    continue
        if not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
            msg = f"{tok} {k}: blocking 과 서버 결과 다름"
            if a.dtype.kind == "f":
                bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
                idx = np.argwhere(bad)
                msg += (f" — 다른 원소 {int(bad.sum())}/{a.size}, 최대 |차| {np.nanmax(np.abs(a - b)):.3e}, 처음 위치 "
                        f"{idx[:4].tolist()}, ref {a[tuple(idx[0])]!r} 서버 {b[tuple(idx[0])]!r}")
            raise AssertionError(msg)


def run_server_case(a, ref, workers):
    import score_server as S
    sock = Path(a.scratch) / f"s{workers}.sock"
    for f in (sock, Path(str(sock) + ".key")):
        if f.exists():
            raise RuntimeError(f"{f} 이 남아 있다 — 이전 서버 확인")
    log = open(Path(a.scratch) / f"server_w{workers}.log", "w")
    cmd, env = _navsim_cmd([REPO / "tools/navsim/grpo/score_server.py", "--socket", sock, "--workers", workers,
                            "--metric-cache", a.metric_cache])
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        cl = S.ScoreClient(str(sock), connect_timeout=180)
        toks = list(ref["tokens"])
        # 오류 경로: 건너뛰지 않고 예외
        for bad, what in (({"ffffffffffffffff": np.zeros((2, 8, 2))}, "없는 token"),
                          ({toks[0]: np.zeros((2, 7, 2))}, "형식")):
            try:
                cl.submit(f"bad-{what}", bad)
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError(f"{what} 요청이 거부되지 않았다")
        mbs = [toks[i:i + a.mb] for i in range(0, len(toks), a.mb)]
        t_sub, t_done, res = {}, {}, {}
        t0 = time.time()
        for i, m in enumerate(mbs):
            t_sub[i] = time.time()
            cl.submit(i, {t: ref[f"in/{t}"] for t in m})
            if i == 0:
                try:
                    cl.submit(0, {m[0]: ref[f"in/{m[0]}"]})
                except ValueError:
                    pass
                else:
                    raise AssertionError("중복 mb_id 가 거부되지 않았다")
            for mb_id, r in cl.poll(block=False):           # 학습 루프처럼 제출 사이사이 수거
                t_done[mb_id] = time.time()
                res[mb_id] = r
        while len(res) < len(mbs):
            for mb_id, r in cl.poll(block=True, timeout=60):
                t_done[mb_id] = time.time()
                res[mb_id] = r
        wall = time.time() - t0
        n = 0
        for i, m in enumerate(mbs):
            assert set(res[i]) == set(m), (i, set(res[i]) ^ set(m))
            for t in m:
                _cmp(ref, res[i][t], t)
                n += 1
        # wait() 경로: 2 개 제출 후 하나만 기다리고 나머지는 poll 로
        cl.submit("w0", {toks[0]: ref[f"in/{toks[0]}"]})
        cl.submit("w1", {toks[1]: ref[f"in/{toks[1]}"]})
        g = cl.wait(["w1"])
        _cmp(ref, g["w1"][toks[1]], toks[1])
        rest = []
        while not rest:
            rest = cl.poll(block=True, timeout=60)
        assert [m for m, _ in rest] == ["w0"], rest
        _cmp(ref, rest[0][1][toks[0]], toks[0])
        st = cl.status()
        lat = np.array([t_done[i] - t_sub[i] for i in range(len(mbs))])
        K = ref[f"in/{toks[0]}"].shape[0]
        print(f"[서버 워커 {workers}] {n} token: {ULP_KEYS} 외 모든 출력 키 비트 동일(blocking 대비); {ULP_KEYS} 비트 불일치 원소 "
              f"{ULP_STATS['elems']} 개, 최대 |차| {ULP_STATS['max']:.2e}; cont comfort_ratio_max 불일치 {CONT_STATS['elems']} 개 "
              f"최대 {CONT_STATS['max']:.2e}. {len(mbs)} micro-batch × {a.mb} token, "
              f"K={K}: {wall:.1f}s → {n / wall:.2f} 장면/s ({n * K / wall:.0f} 후보/s); micro-batch 지연 중앙 {np.median(lat):.2f}s "
              f"p90 {np.percentile(lat, 90):.2f}s; 서버 status workers={st['workers']}", flush=True)
        cl.shutdown_server()
        proc.wait(timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"서버 종료 코드 {proc.returncode} (로그 {log.name})")
        assert not sock.exists(), "소켓 파일이 남았다"
        return n / wall
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=60)
        log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="client", choices=["client", "ref"])
    ap.add_argument("--metric-cache", default="/home1/irteam/data-vol1/datasets/navsim/exp/metric_cache_navtrain")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--workers", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reuse-ref", action="store_true")
    ap.add_argument("--npy-disable", default=None, help="수정(2026-11-06): navsim 하위 프로세스에 NPY_DISABLE_CPU_FEATURES 로 전달(예 AVX512_SKX)")
    a = ap.parse_args()
    if a.npy_disable:
        os.environ["_TSS_NPY_DISABLE"] = a.npy_disable
    if a.mode == "ref":
        return _ref(a)
    Path(a.scratch).mkdir(parents=True, exist_ok=True)
    ref_path = Path(a.scratch) / f"ref_n{a.n}_K{a.K}_s{a.seed}{'_' + a.npy_disable if a.npy_disable else ''}.npz"
    if not (a.reuse_ref and ref_path.exists()):
        cmd, env = _navsim_cmd([HERE, "--mode", "ref", "--n", a.n, "--K", a.K, "--noise", a.noise, "--seed", a.seed,
                                "--metric-cache", a.metric_cache, "--scratch", a.scratch, "--out", ref_path])
        subprocess.run(cmd, env=env, check=True)
    ref = np.load(ref_path, allow_pickle=True)   # 이 시험이 방금 만든 scratch 파일(충돌 token 등 object 배열 포함)
    for w in a.workers:
        ULP_STATS.update(elems=0, max=0.0)
        CONT_STATS.update(elems=0, max=0.0)   # 수정(2026-11-06): 검토 3
        run_server_case(a, ref, w)
    bad = [m for m in sys.modules if m.split(".")[0] in ("nuplan", "navsim", "grpo_scorer", "shapely")]
    assert not bad, f"클라이언트 프로세스에 navsim 계열 모듈이 import 됨: {bad[:5]}"
    print("[통과] 클라이언트는 nuplan/navsim/shapely 를 import 하지 않음")


if __name__ == "__main__":
    main()
