# 수정(2026-11-06): 검토 2 수정 시험(신규 파일) — 서버 guide → advantage 단계 ③, 워커 죽음, comfort 귀속 창.
"""navsim env 에서 실행(채점기·서버 모두 navsim env 하위 프로세스):
  source scripts/navsim/navsim_env.sh
  $NAVSIM_PY tools/navsim/grpo/tests/test_scorer_robust.py --scratch /tmp/<짧은 경로> [--workers 4]
시험
  1) comfort(수정(2026-11-06) 검토 3): n_hist 13–16 에서 저크·종방향 저크는 savgol interp 앞 가장자리 때문에 이력 첫 시점(0)부터
     후보 미래에 의존함을 확인(시점 창 규칙이 틀린 근거). 값 비교 규칙 자체는 test_scorer_ext 가 실 장면에서 재계산·단언.
  7) comfort 열이 advantage 에 안 쓰임(수정(2026-11-06) 검토 4).  8) 같은 경로 동시 기동(수정(2026-11-06) 검토 4).
  6) 고아 future 회수(수정(2026-11-06) 검토 3, 가짜 executor 로 terminate_broken 경쟁 재현).
  5) 서버 부모 SIGKILL(수정(2026-11-06) 검토 3): 워커가 0 개 남고(PDEATHSIG), 남은 소켓·키 파일은 다음 기동이 정리하고 뜬다.
  2) 서버(guide 기본 켬) → advantage(읽기 전용 호출) 단계 ③: 모든 정책 후보가 크게 옆으로 벗어나 EPDMS 0 인 token 을 찾아
     build_scene → scene_advantage(Full, P) 가 단계 ③ 이고 guide_target(Lg) 이 원천 pdm/human 으로 나오는지,
     서버 결과의 ref_traj/ref_epdms 가 blocking GRPOScorer(guide=True) 와 비트 동일한지 확인. guide 키를 빼면(종전 서버)
     guide_target 이 ValueError 인지도 확인(회귀 재현).
  3) 워커 죽음(GRPOScorer.score): 채점 도중 워커 하나를 SIGKILL → BrokenProcessPool 이 시간 안에 올라와야 한다(멈추면 실패).
  4) 워커 죽음(서버): 진행 중 micro-batch 의 poll 이 RuntimeError(BrokenProcessPool) 를 내고, 이후 submit 은 거부, status.broken 기록.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "tools/navsim"))
sys.path.insert(0, str(REPO / "tools/navsim/grpo"))
import advantage as AD  # noqa: E402  (읽기 전용 호출)
import grpo_scorer as G  # noqa: E402
import score_server as S  # noqa: E402

GUIDE_KEYS = ("ref_traj", "ref_sub", "ref_epdms", "human_traj", "human_sub", "human_epdms")


def t_comfort_edge():
    # 수정(2026-11-06): 검토 3 — #이전 t_comfort_reach(COMFORT_REACH 11 단언, 규칙 폐기)
    rows = {}
    for nh in (13, 14, 15, 16):
        m = G._comfort_edge_dependence(nh)
        assert m["jerk"] == 0 and m["lon_jerk"] == 0, (nh, m)          # 이력 전 구간 의존
        assert all(v <= nh for v in m.values()), (nh, m)               # 모든 신호가 이력 일부에 의존(후보 미래 → 이력 값)
        rows[nh] = m
    print(f"[1 통과] 후보 미래가 바꾸는 가장 이른 시점(신호별): {rows} — 저크는 n_hist 13–16 모두 0(이력 전 구간)")


def _allfail_cands(rng, K=48):
    """옆으로 25–45 m 벗어나는 후보(주행 가능 영역 밖으로 나가 EPDMS 0 이 되게). ego 프레임 (K,8,2)."""
    t = np.arange(1, 9)[None, :] / 8.0
    side = rng.choice([-1.0, 1.0], K)[:, None]
    lat = side * rng.uniform(25, 45, (K, 1)) * np.minimum(1.0, 3 * t)
    lon = rng.uniform(2, 20, (K, 1)) * t
    return np.stack([lon, lat], -1)


def _start_server(sock, a, extra=()):
    log = open(str(sock) + ".log", "w")
    proc = subprocess.Popen([sys.executable, str(REPO / "tools/navsim/grpo/score_server.py"), "--socket", str(sock),
                             "--workers", str(a.workers), "--metric-cache", a.metric_cache, *extra],
                            stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
    return proc, log


def t_server_guide_stage3(a):
    sock = Path(a.scratch) / "g.sock"
    proc, log = _start_server(sock, a)
    try:
        cl = S.ScoreClient(str(sock), connect_timeout=180)
        st = cl.status()
        assert st["guide"] is True and st["broken"] is None and len(st["worker_pids"]) == a.workers, st
        paths = G.cache_paths(a.metric_cache)
        toks = sorted(paths)[::613]
        rng = np.random.default_rng(3)
        found = None
        for i, tok in enumerate(toks[:a.max_try]):
            c = _allfail_cands(rng)
            cl.submit(i, {tok: c})
            out = cl.wait([i], timeout=300)[i][tok]
            for k in GUIDE_KEYS:
                assert k in out, f"서버 결과에 {k} 없음(guide 전달 안 됨)"
            if np.all(out["epdms"][16:] == 0.0):
                found = (tok, c, out)
                break
        assert found is not None, f"{a.max_try} token 안에서 정책 후보 전부 EPDMS 0 인 장면 못 찾음"
        tok, c, out = found
        sc = AD.build_scene(tok, out, mu16=c[:16])
        for v in ("Full", "P"):
            r = AD.scene_advantage(sc, v)
            assert (r["stage"] == 3).all(), r["stage"]
            assert r["Lg"] is not None and r["Lg"]["source"] in ("pdm", "human", "none"), r["Lg"]
            assert np.all(r["A"] == 0.0)
        lg = AD.scene_advantage(sc, "Full")["Lg"]
        # 종전 서버(guide 키 없음) 재현: guide_target 이 ValueError
        old = {k: v for k, v in out.items() if k not in GUIDE_KEYS}
        try:
            AD.scene_advantage(AD.build_scene(tok, old, mu16=c[:16]), "Full")
        except ValueError:
            pass
        else:
            raise AssertionError("guide 키 없는 결과인데 단계 ③ 이 ValueError 를 내지 않았다")
        cl.shutdown_server()
        proc.wait(timeout=120)
        assert proc.returncode == 0, proc.returncode
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=60)
        log.close()
    # blocking 경로와 안내 키 비트 동일
    with G.GRPOScorer(a.metric_cache, workers=1, margins=True, guide=True) as gs:
        ref = gs.score({tok: c})[tok]
    for k in GUIDE_KEYS + ("epdms", "subscores", "fail_idx"):
        x, y = np.asarray(ref[k]), np.asarray(out[k])
        assert x.shape == y.shape and np.array_equal(x, y, equal_nan=True), k
    print(f"[2 통과] 서버(guide 기본) → advantage 단계 ③: token {tok}, Lg 원천 {lg['source']} r* {lg['r_star']:.3f} "
          f"q* {lg['q_star']} 가중 {lg['weight']:.3f}; 안내 키 blocking 과 비트 동일; guide 키 없으면 ValueError(회귀 재현)")


def _big_batch(a, n, K=48, seed=0):
    paths = G.cache_paths(a.metric_cache)
    toks = sorted(paths)[1::397][:n]
    rng = np.random.default_rng(seed)
    anchors = np.load(REPO / "data/kmeans/navsim_v2_plan_temp_48.npy").astype(np.float64)
    return {t: G._bench_cands(rng, anchors, K, 1.5) for t in toks}


def t_scorer_worker_kill(a):
    batch = _big_batch(a, 6 * a.workers)
    gs = G.GRPOScorer(a.metric_cache, workers=a.workers, margins=True, guide=True)
    pids = gs.worker_pids()
    res = {}

    def run():
        try:
            gs.score(batch)
            res["err"] = None
        except BaseException as e:   # 시험: 어떤 예외인지 기록
            res["err"] = e

    th = threading.Thread(target=run, daemon=True)
    th.start()
    time.sleep(3.0)
    os.kill(pids[0], signal.SIGKILL)
    th.join(timeout=180)
    assert not th.is_alive(), "워커 죽음 뒤 score() 가 180 s 안에 끝나지 않았다(멈춤)"
    assert isinstance(res.get("err"), BrokenProcessPool), f"BrokenProcessPool 이 아님: {res.get('err')!r}"
    try:
        gs.score(dict(list(batch.items())[:1]))
    except BrokenProcessPool:
        pass
    else:
        raise AssertionError("깨진 풀로 채점이 됐다")
    gs.close()
    print(f"[3 통과] GRPOScorer.score 도중 워커 SIGKILL → BrokenProcessPool ({type(res['err']).__name__}), 이후 호출도 거부")


def t_server_worker_kill(a):
    sock = Path(a.scratch) / "k.sock"
    proc, log = _start_server(sock, a)
    try:
        cl = S.ScoreClient(str(sock), connect_timeout=180)
        pids = cl.status()["worker_pids"]
        batch = _big_batch(a, 6 * a.workers, seed=1)
        items = list(batch.items())
        h = len(items) // 2
        cl.submit("a", dict(items[:h]))
        cl.submit("b", dict(items[h:]))
        time.sleep(3.0)
        os.kill(pids[-1], signal.SIGKILL)
        errs, got, t0 = [], set(), time.time()
        while cl.pending or cl._ready:
            if time.time() - t0 > 180:
                raise AssertionError(f"워커 죽음 뒤 180 s 안에 micro-batch 가 끝나지 않았다(멈춤): 오류 {len(errs)} 정상 {got}")
            try:
                for m, _ in cl.poll(block=True, timeout=10):
                    got.add(m)
            except RuntimeError as e:
                assert "BrokenProcessPool" in str(e), str(e)[:400]
                errs.append(str(e))
        assert errs, "어떤 micro-batch 도 오류로 끝나지 않았다(토큰이 조용히 사라짐?)"
        failed = {m for m in ("a", "b") if f"[mb {m}]" in "".join(errs)}
        assert failed | got == {"a", "b"} and not (failed & got), (failed, got)
        st = cl.status()
        assert st["broken"] and st["inflight"] == 0, st
        try:
            cl.submit("c", dict(items[:1]))
        except RuntimeError as e:
            assert "깨졌다" in str(e), str(e)
        else:
            raise AssertionError("깨진 서버가 submit 을 받았다")
        cl.shutdown_server()
        proc.wait(timeout=120)
        assert proc.returncode == 0, proc.returncode
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=60)
        log.close()
    print(f"[4 통과] 서버 워커 SIGKILL → micro-batch 실패 {sorted(failed)}(BrokenProcessPool), 정상 완료 {sorted(got)}, "
          f"이후 submit 거부, status.broken 기록, 정상 종료")


def _alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


def t_server_parent_kill(a):
    """수정(2026-11-06): 검토 3 — 서버 kill -9 → 워커 0 개 생존, 다음 기동이 잔재 소켓·키를 정리."""
    sock = Path(a.scratch) / "o.sock"
    proc, log = _start_server(sock, a)
    try:
        cl = S.ScoreClient(str(sock), connect_timeout=180)
        pids = cl.status()["worker_pids"]
        assert len(pids) == a.workers, pids
        cl.submit("x", _big_batch(a, 2 * a.workers, seed=2))          # 작업 중에 죽인다
        time.sleep(2.0)
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
        cl.close()
        t0 = time.time()
        while any(_alive(p) for p in pids) and time.time() - t0 < 30:
            time.sleep(0.2)
        alive = [p for p in pids if _alive(p)]
        for p in alive:                                                 # 시험 정리(실패 보고는 아래 단언)
            os.kill(p, signal.SIGKILL)
        assert not alive, f"서버 kill -9 뒤 워커 {len(alive)} 개가 30 s 넘게 살아 있다(고아): {alive}"
        assert sock.exists() and Path(str(sock) + ".key").exists(), "kill -9 인데 소켓·키 파일이 없다(시험 전제 깨짐)"
    finally:
        if proc.poll() is None:
            proc.kill()
        log.close()
    # 같은 경로로 다시 기동 → 잔재 정리 후 정상 동작
    proc2, log2 = _start_server(sock, a)
    try:
        cl = S.ScoreClient(str(sock), connect_timeout=180)
        st = cl.status()
        assert len(st["worker_pids"]) == a.workers and st["broken"] is None, st
        # 살아 있는 서버가 있으면 같은 경로 기동은 거부
        r = subprocess.run([sys.executable, str(REPO / "tools/navsim/grpo/score_server.py"), "--socket", str(sock), "--workers", "1",
                            "--metric-cache", a.metric_cache], capture_output=True, text=True, timeout=300, env=dict(os.environ))
        # 수정(2026-11-06): 검토 4 — 이제 잠금(<소켓>.lock)에서 먼저 거부된다 #이전 "이미 받고 있다"(접속 확인 거부)만 허용
        assert r.returncode != 0 and ("잡고 있다" in (r.stdout + r.stderr) or "이미 받고 있다" in (r.stdout + r.stderr)), \
            (r.returncode, (r.stdout + r.stderr)[-400:])
        cl.shutdown_server()
        proc2.wait(timeout=120)
        assert proc2.returncode == 0, proc2.returncode
    finally:
        if proc2.poll() is None:
            proc2.terminate()
            proc2.wait(timeout=60)
        log2.close()
    cleaned = "[정리]" in open(str(sock) + ".log").read()
    assert cleaned, "재기동 로그에 잔재 정리 기록이 없다"
    print(f"[5 통과] 서버 kill -9 → 워커 {len(pids)} 개 모두 종료(고아 0), 재기동이 잔재 소켓·키 정리 후 정상, 살아 있는 서버 경로 재사용은 거부")


def t_orphan_future_reap():
    """수정(2026-11-06): 검토 3 — py3.9 terminate_broken 경쟁 재현(가짜 executor): 풀이 깨졌고 관리 스레드가 끝났는데 완료되지 않은
    future 가 남으면, poll(block=True, timeout=None) 이 무한 대기하지 않고 그 token 을 명시 오류로 실패 처리하는지."""
    from concurrent.futures import Future
    from types import SimpleNamespace
    srv = object.__new__(S.ScoreServer)
    srv.ex = SimpleNamespace(_broken="시험용 깨짐", _executor_manager_thread=None)
    srv.broken, srv.stop, srv.n_tok = None, threading.Event(), 0
    srv._Broken = BrokenProcessPool
    c = S._Conn()
    f_done, f_orphan = Future(), Future()
    c.inflight["m"] = dict(pending={"t1", "t2"}, out={}, err=None, futs={"t1": f_done, "t2": f_orphan})
    f_done.set_result(None)
    srv._on_done(c, "m", ("t1", {}, 0.0, 0.0))
    t0 = time.time()
    out = srv._poll(c, True, None)
    assert len(out) == 1 and out[0][0] == "m" and out[0][1] is None and "고아 future" in out[0][2], out
    assert srv.broken and not c.inflight and time.time() - t0 < 10
    srv._on_err(c, "m", "t2", RuntimeError("늦은 통지"))                 # 끝난 micro-batch 에 늦은 통지 → 무시(예외 없음)
    print("[6 통과] 깨진 풀의 고아 future → poll(block, timeout=None) 이 즉시 그 token 을 오류로 끝냄, 늦은 통지는 무시")


def t_comfort_not_in_advantage():
    """수정(2026-11-06): 검토 4 — comfort 열(fail_idx/fail_wp[:, 6])은 기록·진단용: 값을 바꿔도 advantage 결과가 비트 동일한지(읽기 전용 호출)."""
    rng = np.random.default_rng(11)
    for trial in range(50):
        K = 16 + 32
        fi = rng.integers(-1, 41, (K, 7))
        fi[rng.random((K, 7)) < 0.6] = -1
        fw = np.where(fi < 0, -1, np.clip(np.ceil(fi / 5.0).astype(int) - 1, 0, 7))
        out = dict(epdms=np.where(rng.random(K) < 0.5, 0.0, rng.random(K)), fail_idx=fi, fail_wp=fw,
                   coll_front=rng.random(K) < 0.5, min_obj_dist_fs_m=np.where(rng.random(K) < 0.2, np.nan, rng.random(K) * 5),
                   lat_dev_wp_m=rng.random(K), ref_traj=rng.normal(size=(8, 3)), ref_epdms=float(rng.random()),
                   human_traj=rng.normal(size=(8, 3)), human_epdms=float(rng.random()))
        if trial % 5 == 0:
            out["epdms"][16:] = 0.0                                     # 단계 ③ 도 포함
        mu16 = rng.normal(size=(16, 8, 2))
        for v in AD.VARIANTS:
            base = AD.scene_advantage(AD.build_scene("t", out, mu16=mu16), v)
            o2 = dict(out, fail_idx=fi.copy(), fail_wp=fw.copy())
            o2["fail_idx"][:, 6] = rng.choice([-2, -1, 0, 3, 17, 40], K)
            o2["fail_wp"][:, 6] = np.where(o2["fail_idx"][:, 6] < 0, o2["fail_idx"][:, 6], np.clip(np.ceil(o2["fail_idx"][:, 6] / 5) - 1, 0, 7))
            alt = AD.scene_advantage(AD.build_scene("t", o2, mu16=mu16), v)
            for k in ("stage", "A", "Ahat", "w", "u", "jf", "src", "ftype", "cls_target"):
                assert np.array_equal(np.asarray(base[k]), np.asarray(alt[k])), (trial, v, k)
            assert (base["Lg"] is None) == (alt["Lg"] is None) and (base["Lg"] is None or base["Lg"]["q_star"] == alt["Lg"]["q_star"])
    print(f"[7 통과] comfort 열(-2/-1/0/…)을 바꿔도 advantage 출력(단계·A·w·u·주 실패·L_g·L_cls) 비트 동일 — 50 장면 × 변형 {len(AD.VARIANTS)}")


def t_concurrent_start(a):
    """수정(2026-11-06): 검토 4 — 빈 경로에 서버 둘을 동시에 띄우면 하나만 뜨고(다른 하나는 잠금으로 즉시 거부) 산 서버는 인증·채점 정상."""
    sock = Path(a.scratch) / "c.sock"
    procs = []
    for i in range(2):
        log = open(f"{sock}.{i}.log", "w")
        procs.append((subprocess.Popen([sys.executable, str(REPO / "tools/navsim/grpo/score_server.py"), "--socket", str(sock),
                                        "--workers", str(a.workers), "--metric-cache", a.metric_cache],
                                       stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ)), log))
    try:
        t0 = time.time()
        while sum(p.poll() is not None for p, _ in procs) < 1 and time.time() - t0 < 120:
            time.sleep(0.2)
        dead = [i for i, (p, _) in enumerate(procs) if p.poll() is not None]
        assert len(dead) == 1, f"정확히 하나가 거부돼야 한다: 끝난 프로세스 {dead}"
        loser = dead[0]
        assert procs[loser][0].returncode != 0
        procs[loser][1].flush()
        msg = open(f"{sock}.{loser}.log").read()
        assert "잡고 있다" in msg, msg[-400:]
        cl = S.ScoreClient(str(sock), connect_timeout=180)
        st = cl.status()
        assert len(st["worker_pids"]) == a.workers, st
        b = _big_batch(a, 2, K=8, seed=3)
        cl.submit("m", b)
        r = cl.wait(["m"], timeout=300)["m"]
        assert set(r) == set(b)
        cl.shutdown_server()
        winner = 1 - loser
        procs[winner][0].wait(timeout=120)
        assert procs[winner][0].returncode == 0
        assert not sock.exists() and not Path(str(sock) + ".key").exists()
    finally:
        for p, log in procs:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=60)
            log.close()
    print(f"[8 통과] 동시 기동 2 개 → 서버 {1 - loser} 만 뜨고 서버 {loser} 는 잠금으로 즉시 거부, 산 서버 인증·채점·종료 정상")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric-cache", default=os.path.join(os.environ.get("NAVSIM_EXP_ROOT", ""), "metric_cache_navtrain"))
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-try", type=int, default=30)
    ap.add_argument("--only", nargs="*", default=["1", "2", "3", "4", "5", "6", "7", "8"])
    a = ap.parse_args()
    if a.workers > G.MAX_WORKERS:
        raise ValueError("워커 상한 16")
    Path(a.scratch).mkdir(parents=True, exist_ok=True)
    if "1" in a.only:
        t_comfort_edge()
    if "2" in a.only:
        t_server_guide_stage3(a)
    if "3" in a.only:
        t_scorer_worker_kill(a)
    if "4" in a.only:
        t_server_worker_kill(a)
    if "5" in a.only:
        t_server_parent_kill(a)
    if "6" in a.only:
        t_orphan_future_reap()
    if "7" in a.only:
        t_comfort_not_in_advantage()
    if "8" in a.only:
        t_concurrent_start(a)
    print("[통과] test_scorer_robust")


if __name__ == "__main__":
    main()
