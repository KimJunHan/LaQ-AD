# 수정(2026-11-06): C3 GRPO 설계 §9.3–9.4 / §12-③ — navsim env 비동기 채점 서버 + nuplan 없는 클라이언트(신규 파일).
"""GRPO 비동기 채점 서버 (docs/c3_grpo_design.md §9.3 1-pass 비동기 채점, §9.4 채점 서버).

왜: 학습 env(hipad_bw)에는 nuplan 이 없고, GRPOScorer.score 는 끝까지 막는 호출이다. 1-pass 학습은
micro-batch 마다 표본을 보내고 점수가 도착한 micro-batch 부터 역전파하므로, 별도 navsim env 서버가 micro-batch 단위로 결과를 돌려준다.

구성
  서버(navsim env): tools/navsim/grpo_scorer.GRPOScorer 의 워커 풀(초기화·채점 함수 그대로, workers 명시 인자, 기본 16)에
    token 마다 ProcessPoolExecutor.submit 으로 grpo_scorer._job 을 넣고, 완료 콜백이 micro-batch 의 모든 token 이 끝나면 그 연결의 결과 큐에 넣는다.
    수정(2026-11-06, 검토 2): multiprocessing.Pool → ProcessPoolExecutor(워커가 죽으면 진행 중 token 이 BrokenProcessPool 로 실패하고
    그 micro-batch 의 poll 이 RuntimeError — 영원히 기다리지 않음). 그 뒤 서버는 broken 상태로 모든 submit 을 거부한다(자동 재기동 없음,
    원인 확인 후 재시작). 기본 guide=True: 결과에 ref_traj/ref_epdms/ref_sub/human_* (advantage 단계 ③ guide_target 입력)를 넣는다.
    채점 값은 GRPOScorer.score(blocking)와 같은 함수·같은 입력이라 비트 동일하다(tests/test_score_server.py 로 확인).
  전송: stdlib multiprocessing.connection 유닉스 소켓(AF_UNIX) + authkey(HMAC 인증; 키 파일 0600). 메시지 안의 numpy 배열은
    (dtype, shape, raw bytes) 로(object 배열은 내장 스칼라 목록으로) 바꿔 보내 양쪽 numpy 버전·pickle 경로 차이와 무관하다. 연결마다 스레드 하나, 결과 큐 하나.
  클라이언트(ScoreClient): numpy + stdlib 만 import(nuplan·navsim·grpo_scorer 를 부르지 않는다) → hipad_bw 학습 루프에서 쓴다.

API (클라이언트)
  cl = ScoreClient(socket_path)                      # 키 파일 기본 = socket_path + ".key"
  cl.submit(mb_id, {token: ndarray (K,8,2|3)})       # 즉시 반환. mb_id = 요청 id(연결 안에서 유일, int/str)
  done = cl.poll(block=False, timeout=None)          # [(mb_id, {token: dict}), ...] 끝난 micro-batch 전부(제출 순서 무관)
  cl.wait(mb_ids)                                    # 지정 id 가 모두 끝날 때까지 → {mb_id: {token: dict}}
  cl.status() / cl.close() / cl.shutdown_server()
  결과 dict 는 grpo_scorer.score_token 출력과 같다(subscores, epdms, fail_idx, fail_wp, filtered, cont, frenet, ...)
  + "_timing": (캐시 읽기 s, 채점 s). 채점 중 예외가 나면 그 micro-batch 의 poll 이 RuntimeError(원 traceback 포함)를 던진다
  — 건너뛰지 않는다. 알 수 없는 token·형식 오류는 submit 에서 바로 예외.

서버 실행
  source scripts/navsim/navsim_env.sh
  $NAVSIM_PY tools/navsim/grpo/score_server.py --socket /tmp/<짧은 경로>/grpo.sock --workers 16 \
      [--metric-cache $NAVSIM_EXP_ROOT/metric_cache_navtrain] [--no-margins] [--no-guide]   # 수정(2026-11-06): guide 기본 켬
  준비되면 "[ready]" 한 줄을 출력한다. 소켓 경로는 유닉스 한도(≈107 바이트) 안이어야 한다.
"""
import os

# BLAS 스레드 1 고정은 numpy import 전에 해야 한다(grpo_scorer 의 setdefault 는 numpy 가 먼저 import 되면 효과 없음).
# 실측(11-06): numpy 를 먼저 import 해 OpenBLAS 가 다중 스레드면 같은 입력의 frenet·lat_dev_wp_m 이 호출마다 1 ulp 씩 달라졌다
# (7 token × 4 반복 중 1 회); 1 스레드로 고정하면 0 회.
# 수정(2026-11-06, 검토 2): 위 1 ulp(≤ 1e-15 m) 차이의 원인은 OpenBLAS 스레드가 아니다(_frenet 에 BLAS 호출 없음). numpy 1.23 의
#   np.cos/np.sin 이 출력 버퍼가 입력과 인접(겹침 검사)하냐에 따라 SVML(AVX-512) 경로와 libm 경로를 고르고, 두 경로가 마지막 ulp 에서
#   다르다 — 그래서 할당 이력(앞서 채점한 token·환경 변수 수)에 따라 달라진다. 크기 ≤ 1e-15 m 라 허용(시험은 이 두 키만 1e-12 허용).
#   스레드 1 고정은 다른 BLAS 경로의 결정성을 위해 그대로 둔다.
# 서버로 실행할 때만 고정한다(클라이언트로 import 하는 학습 프로세스의 torch 스레드에는 영향 주지 않음).
if __name__ == "__main__":
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
import argparse
import secrets
import signal
import stat
import sys
import threading
import time
import traceback
from multiprocessing.connection import Client, Listener
from multiprocessing import AuthenticationError  # 수정(2026-11-06): 검토 3
from pathlib import Path

import numpy as np

DEFAULT_CACHE = "/home1/irteam/data-vol1/datasets/navsim/exp/metric_cache_navtrain"
MAX_SOCK_LEN = 100


# ---------------------------------------------------------------- 공통: numpy 직렬화(내장형만 pickle 되게)
def _enc(o):
    if isinstance(o, np.ndarray):
        a = np.ascontiguousarray(o)
        if a.dtype.kind == "O":                                  # 문자열 등 내장 스칼라 배열(예: 충돌 객체 token)
            vals = a.ravel().tolist()
            bad = [type(v).__name__ for v in vals if not isinstance(v, (str, int, float, bool, type(None)))]
            if bad:
                raise TypeError(f"object 배열에 내장 스칼라가 아닌 값 {sorted(set(bad))}")
            return ("__ndobj__", a.shape, vals)
        if a.dtype.kind == "V":
            raise TypeError(f"직렬화 불가 dtype {a.dtype}")
        return ("__nd__", a.dtype.str, a.shape, a.tobytes())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, dict):
        return {k: _enc(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_enc(v) for v in o)
    return o


def _dec(o):
    if isinstance(o, tuple) and len(o) == 4 and o[0] == "__nd__":
        return np.frombuffer(o[3], dtype=np.dtype(o[1])).reshape(o[2]).copy()
    if isinstance(o, tuple) and len(o) == 3 and o[0] == "__ndobj__":
        out = np.empty(len(o[2]), dtype=object)
        out[:] = o[2]
        return out.reshape(o[1])
    if isinstance(o, dict):
        return {k: _dec(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_dec(v) for v in o)
    return o


def _check_sock(path):
    if len(str(path).encode()) > MAX_SOCK_LEN:
        raise ValueError(f"소켓 경로가 너무 길다({len(str(path))} > {MAX_SOCK_LEN}): {path}")


def _read_key(key_file):
    k = Path(key_file).read_bytes()
    if len(k) < 16:
        raise RuntimeError(f"키 파일 {key_file} 이 비었거나 짧다")
    return k


# ---------------------------------------------------------------- 클라이언트 (nuplan 불필요)
class ScoreClient:
    def __init__(self, socket_path, key_file=None, connect_timeout=120.0):
        _check_sock(socket_path)
        key_file = key_file or str(socket_path) + ".key"
        t0 = time.time()
        while True:                                     # 서버 기동 직후 대비: 소켓·키 파일이 생길 때까지 기다림(시간 제한)
            if os.path.exists(socket_path) and os.path.exists(key_file):
                try:
                    self.conn = Client(str(socket_path), family="AF_UNIX", authkey=_read_key(key_file))
                    break
                except (ConnectionRefusedError, FileNotFoundError):
                    pass
            if time.time() - t0 > connect_timeout:
                raise TimeoutError(f"채점 서버 {socket_path} 연결 실패({connect_timeout}s)")
            time.sleep(0.2)
        self.pending = set()
        self._ready = {}

    def _rpc(self, msg):
        self.conn.send(_enc(msg))
        r = _dec(self.conn.recv())
        if not r.get("ok"):
            raise RuntimeError(f"채점 서버 오류({msg.get('op')}): {r.get('error')}")
        return r

    def submit(self, mb_id, batch):
        if mb_id in self.pending or mb_id in self._ready:
            raise ValueError(f"mb_id {mb_id} 가 이미 대기 중")
        if not batch:
            raise ValueError("빈 micro-batch")
        b = {}
        for tok, c in batch.items():
            c = np.asarray(c, np.float64)
            if c.ndim != 3 or c.shape[1] != 8 or c.shape[2] not in (2, 3) or not np.isfinite(c).all():
                raise ValueError(f"{tok}: 후보 형식 {c.shape} 또는 비유한값 — (K,8,2|3) ego 프레임")
            b[str(tok)] = c
        self._rpc(dict(op="submit", mb_id=mb_id, batch=b))
        self.pending.add(mb_id)

    def _fetch(self, block, timeout):
        r = self._rpc(dict(op="poll", block=bool(block), timeout=timeout))
        # 수정(2026-11-06): 검토 2 — 한 번의 poll 에 여러 micro-batch 가 오면 전부 처리한 뒤 실패를 모아 올린다
        #   (전에는 첫 실패에서 바로 raise 해 같은 응답의 나머지 결과·실패가 클라이언트에서 사라졌다).
        errs = []
        for mb_id, res, err in r["results"]:
            self.pending.discard(mb_id)
            if err is not None:
                errs.append((mb_id, err))
            else:
                self._ready[mb_id] = res
        if errs:
            raise RuntimeError(f"micro-batch {[m for m, _ in errs]} 채점 실패(정상 완료분은 다음 poll/wait 가 돌려준다):\n"
                               + "\n".join(f"[mb {m}] {e}" for m, e in errs))

    def poll(self, block=False, timeout=None):
        """끝난 micro-batch 전부 [(mb_id, {token: dict})] — wait() 가 받아 두고 아직 안 돌려준 것 포함."""
        if not (block and self._ready):
            self._fetch(block and not self._ready, timeout)
        out, self._ready = list(self._ready.items()), {}
        return out

    def wait(self, mb_ids, timeout=600.0):
        """지정 id 가 모두 끝날 때까지 → {mb_id: {token: dict}}. 그 사이 끝난 다른 id 는 다음 poll() 이 돌려준다."""
        want, t0 = set(mb_ids), time.time()
        bad = want - self.pending - set(self._ready)
        if bad:
            raise ValueError(f"제출하지 않았거나 이미 받은 id: {sorted(map(str, bad))}")
        while not want <= set(self._ready):
            left = timeout - (time.time() - t0)
            if left <= 0:
                raise TimeoutError(f"wait 시간 초과: 남은 {sorted(map(str, want - set(self._ready)))}")
            self._fetch(True, min(left, 5.0))
        return {m: self._ready.pop(m) for m in want}

    def status(self):
        return self._rpc(dict(op="status"))

    def shutdown_server(self):
        try:
            self._rpc(dict(op="shutdown"))
        finally:
            self.conn.close()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ---------------------------------------------------------------- 서버 (navsim env)
def _take_lock(path):
    """수정(2026-11-06): 검토 4 — <소켓>.lock 비차단 배타 flock. 잡으면 fd(서버 수명 동안 유지), 못 잡으면 RuntimeError."""
    import fcntl
    fd = os.open(path, os.O_RDWR | os.O_CREAT, stat.S_IRUSR | stat.S_IWUSR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError(f"{path} 를 다른 서버가 잡고 있다(같은 소켓 경로로 기동 중이거나 실행 중) — 거부")
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


def _clear_stale_socket(sock, key_file):
    """수정(2026-11-06): 검토 3 — 소켓 파일이 있으면 접속해 본다. 받는 프로세스가 있으면(살아 있는 서버) 거부, 접속 거부
    (ECONNREFUSED = 소유 프로세스가 죽은 잔재)면 소켓·키 파일을 지우고 기록한다. #이전 소켓이 있으면 무조건 거부
    수정(2026-11-06): 검토 4 — 호출 전에 <소켓>.lock 을 잡으므로 잠금을 쓰는 서버끼리는 경쟁이 없다. 접속 확인은 잠금을 모르는
    수정 전 코드 서버(운영 주의: 먼저 내릴 것)를 위한 것으로 남긴다."""
    import socket as _socket
    if not os.path.exists(sock):
        return
    st = os.lstat(sock)
    if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
        raise RuntimeError(f"{sock} 이 내 소유 유닉스 소켓이 아니다 — 지우지 않음")
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    s.settimeout(5.0)                                                   # 수정(2026-11-06): 검토 4 — 멈춘 소유자에 무한 대기 방지
    try:
        s.connect(sock)
    except ConnectionRefusedError:
        for f in (sock, key_file):
            if os.path.exists(f):
                os.unlink(f)
        print(f"[정리] 죽은 서버의 소켓·키 파일 삭제: {sock}", flush=True)
        return
    finally:
        s.close()
    raise RuntimeError(f"소켓 {sock} 에서 서버가 이미 받고 있다(다른 서버 실행 중) — 거부")


class _Conn:
    def __init__(self):
        self.lock = threading.Condition()
        self.done = []          # [(mb_id, res|None, err|None)]
        self.inflight = {}      # mb_id → dict(pending set, out dict, err)


class ScoreServer:
    def __init__(self, socket_path, metric_cache=DEFAULT_CACHE, workers=16, margins=True, key_file=None, guide=True):
        # 수정(2026-11-06): 검토 2 — guide(단계 ③ 안내 궤적) 기본 켬. 전에는 GRPOScorer·작업 tuple 에 빠져 결과에 ref_traj 가 없었고
        #   advantage.guide_target 이 ValueError 를 냈다.
        _check_sock(socket_path)
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            if os.environ.get(v) != "1":
                raise RuntimeError(f"{v}=1 이 numpy import 전에 설정돼야 한다(BLAS 경로 결정성; 수정(2026-11-06) frenet 1 ulp 는 다른 원인 — 위 주석) — "
                                   "score_server.py 를 직접 실행하거나 import 전에 환경변수를 둘 것")
        # 수정(2026-11-06): 검토 4 — 같은 경로 동시 기동 경쟁(진 서버가 키 파일을 덮어써 이긴 서버가 인증 불능) 방지: <소켓>.lock 을
        #   비차단 flock 으로 먼저 잡고 서버 수명 동안 유지. 못 잡으면 즉시 거부. 잠금 파일은 지우지 않는다(지우면 다른 기동과 경쟁).
        self._lock_fd = _take_lock(str(socket_path) + ".lock")
        _clear_stale_socket(socket_path, key_file or str(socket_path) + ".key")   # 수정(2026-11-06): 검토 3 — 죽은 서버 잔재만 정리
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # tools/navsim
        import grpo_scorer as G                                           # nuplan 은 서버에서만
        self.G = G
        from concurrent.futures.process import BrokenProcessPool        # 수정(2026-11-06): 검토 2
        self._Broken = BrokenProcessPool
        self.workers = workers
        self.guide = bool(guide)
        self.scorer = G.GRPOScorer(metric_cache, workers=workers, margins=margins, guide=self.guide)   # 워커 수 명시  # 수정(2026-11-06): guide
        self.ex, self.paths = self.scorer.ex, self.scorer.paths   # 수정(2026-11-06): 검토 2 — ProcessPoolExecutor #이전 self.pool = scorer.pool
        self.broken = None                                         # 수정(2026-11-06): 워커 죽음 기록(이후 submit 거부)
        self.sock = str(socket_path)
        self.key_file = key_file or self.sock + ".key"
        key = secrets.token_bytes(32)
        # 수정(2026-11-06): 검토 4 — bind(Listener) 가 성공한 뒤 키를 임시 파일(0600)에 써서 rename(원자적). bind 가 실패한 기동은 키 파일을
        #   건드리지 않는다. #이전 키 파일 O_TRUNC 쓰기 → Listener
        self.listener = Listener(self.sock, family="AF_UNIX", authkey=key)
        os.chmod(self.sock, stat.S_IRUSR | stat.S_IWUSR)
        tmp = f"{self.key_file}.tmp{os.getpid()}"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "wb") as f:
            f.write(key)
        os.replace(tmp, self.key_file)
        self.stop = threading.Event()
        self.n_tok = 0
        self.t_start = time.time()

    # -- 콜백: executor 관리 스레드(또는 이미 끝났으면 submit 스레드)에서 실행
    def _on_future(self, c, mb_id, tok, fut):
        """수정(2026-11-06): 검토 2 — future 완료 콜백. 예외(BrokenProcessPool 포함)는 그 micro-batch 의 오류로 넘긴다."""
        try:
            e = fut.exception()
        except BaseException as ce:                                   # 취소(CancelledError) 도 오류로
            e = ce
        if e is None:
            self._on_done(c, mb_id, fut.result())
        else:
            if isinstance(e, self._Broken) and self.broken is None:
                self.broken = f"{type(e).__name__}: {e}"
                print(f"[오류] 채점 워커가 죽었다 — 진행 중 token 을 모두 실패 처리하고 이후 submit 을 거부한다: {self.broken}", flush=True)
            self._on_err(c, mb_id, tok, e)

    def _on_done(self, c, mb_id, res):
        tok, out, t_load, t_score = res
        out["_timing"] = np.array([t_load, t_score])
        with c.lock:
            st = c.inflight.get(mb_id)
            if st is None or tok not in st["pending"]:                  # 수정(2026-11-06): 검토 3 — 이미 고아로 실패 처리된 token
                print(f"[경고] 이미 끝난 micro-batch {mb_id} token {tok} 의 늦은 결과 무시(그 micro-batch 는 오류로 보고됨)", flush=True)
                return
            st["out"][tok] = out
            st["pending"].discard(tok)
            self._maybe_finish(c, mb_id)

    def _on_err(self, c, mb_id, tok, e):
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        if e.__cause__ is not None:
            tb = str(e.__cause__) + "\n" + tb                          # 워커 쪽 RemoteTraceback
        with c.lock:
            st = c.inflight.get(mb_id)
            if st is None or tok not in st["pending"]:                  # 수정(2026-11-06): 검토 3 — 중복 실패 통지 무시
                return
            st["err"] = (st["err"] or "") + f"[token {tok}] {tb}"
            st["pending"].discard(tok)
            self._maybe_finish(c, mb_id)

    def _maybe_finish(self, c, mb_id):
        st = c.inflight[mb_id]
        if not st["pending"]:
            del c.inflight[mb_id]
            c.done.append((mb_id, None if st["err"] else st["out"], st["err"]))
            self.n_tok += len(st["out"])
            c.lock.notify_all()

    def _submit(self, c, mb_id, batch):
        unknown = [t for t in batch if t not in self.paths]
        if unknown:
            raise KeyError(f"지표 캐시에 없는 token {unknown[:5]}{' ...' if len(unknown) > 5 else ''}")
        if self.broken is not None:                                     # 수정(2026-11-06): 검토 2 — 죽은 풀에 넣지 않는다
            raise RuntimeError(f"채점 워커 풀이 깨졌다(서버 재시작 필요): {self.broken}")
        # 수정(2026-11-06): 검토 2 — guide 를 작업 tuple 에 넣는다 #이전 (t, path, poses, False)
        jobs = [(t, self.paths[t], self.G._as_poses(cands), False, self.guide) for t, cands in batch.items()]   # 형식 오류는 여기서 예외
        with c.lock:
            if mb_id in c.inflight or any(m == mb_id for m, _, _ in c.done):
                raise ValueError(f"mb_id {mb_id} 중복")
            c.inflight[mb_id] = dict(pending={t for t, *_ in jobs}, out={}, err=None, futs={})   # 수정(2026-11-06): 검토 3 — futs(고아 감시)
        # 수정(2026-11-06): 검토 2 — Pool.apply_async → executor.submit + 완료 콜백. submit 이 도중에 BrokenProcessPool 을 내면
        #   아직 넣지 못한 token 도 같은 오류로 실패 처리(micro-batch 가 pending 으로 남아 영원히 기다리지 않게).
        for i, j in enumerate(jobs):
            tok = j[0]
            try:
                fut = self.ex.submit(self.G._job, j)
            except Exception as e:
                if isinstance(e, self._Broken) and self.broken is None:
                    self.broken = f"{type(e).__name__}: {e}"
                for t in [jj[0] for jj in jobs[i:]]:
                    self._on_err(c, mb_id, t, e)
                return
            with c.lock:                                                # 수정(2026-11-06): 검토 3 — 고아 future 감시용
                if mb_id in c.inflight:
                    c.inflight[mb_id]["futs"][tok] = fut
            fut.add_done_callback(lambda f, m=mb_id, t=tok: self._on_future(c, m, t, f))

    def _reap_orphans(self, c):
        """수정(2026-11-06): 검토 3 — py3.9 terminate_broken 경쟁: 풀이 깨지는 순간 끼어든 submit 의 future 는 아무도 완료하지 않는다.
        풀이 깨졌고(ex._broken) 관리 스레드가 끝났으면 아직 안 끝난 future 의 token 을 명시 오류로 실패 처리한다."""
        brk = getattr(self.ex, "_broken", False)
        mt = getattr(self.ex, "_executor_manager_thread", None)
        if not brk or (mt is not None and mt.is_alive()):
            return
        if self.broken is None:
            self.broken = f"BrokenProcessPool: {brk}"
        with c.lock:
            orphan = [(m, t) for m, st in c.inflight.items() for t in list(st["pending"])
                      if t not in st["futs"] or not st["futs"][t].done()]
        for m, t in orphan:
            self._on_err(c, m, t, RuntimeError(f"워커 풀이 깨진 뒤 완료되지 않은 작업(고아 future): {brk}"))

    def _poll(self, c, block, timeout):
        # 수정(2026-11-06): 검토 3 — 1 s 마다 깨어 고아 future 를 확인(timeout=None 무한 대기 방지) #이전 wait_for(…, timeout=timeout) 한 번
        t_end = None if timeout is None else time.time() + float(timeout)
        self._reap_orphans(c)
        with c.lock:
            if block and not c.done:
                if not c.inflight:
                    raise RuntimeError("poll(block=True) 인데 대기 중 micro-batch 가 없다")
                while not (c.done or self.stop.is_set()):
                    left = 1.0 if t_end is None else min(1.0, t_end - time.time())
                    if left <= 0:
                        break
                    c.lock.wait_for(lambda: bool(c.done) or self.stop.is_set(), timeout=left)
                    if not c.done:
                        c.lock.release()                              # _reap_orphans → _on_err 가 잠금을 다시 잡는다(RLock 이지만 명시적으로)
                        try:
                            self._reap_orphans(c)
                        finally:
                            c.lock.acquire()
            out, c.done = c.done, []
        return out

    def _serve_conn(self, conn):
        c = _Conn()
        try:
            while not self.stop.is_set():
                try:
                    msg = _dec(conn.recv())
                except EOFError:
                    break
                op = msg.get("op")
                try:
                    if op == "submit":
                        self._submit(c, msg["mb_id"], msg["batch"])
                        rep = dict(ok=True)
                    elif op == "poll":
                        rep = dict(ok=True, results=self._poll(c, msg.get("block"), msg.get("timeout")))
                    elif op == "status":
                        with c.lock:
                            rep = dict(ok=True, inflight=len(c.inflight), done_unread=len(c.done), workers=self.workers,
                                       tokens_scored=self.n_tok, uptime_s=time.time() - self.t_start,
                                       guide=self.guide, broken=self.broken,           # 수정(2026-11-06): 검토 2
                                       worker_pids=self.scorer.worker_pids())
                    elif op == "shutdown":
                        rep = dict(ok=True)
                        conn.send(_enc(rep))
                        self.stop.set()
                        try:                                            # accept() 를 깨운다
                            Client(self.sock, family="AF_UNIX", authkey=_read_key(self.key_file)).close()
                        except OSError:
                            pass
                        break
                    else:
                        raise ValueError(f"알 수 없는 op {op}")
                except Exception as e:                                  # 요청 오류는 클라이언트에 예외로 전달(클라이언트가 raise)
                    rep = dict(ok=False, error=f"{type(e).__name__}: {e}")
                conn.send(_enc(rep))
        finally:
            with c.lock:
                if c.inflight:
                    print(f"[경고] 연결 종료 시 미완료 micro-batch {len(c.inflight)} 개(결과 버려짐)", flush=True)
            conn.close()

    def serve_forever(self):
        print(f"[ready] socket={self.sock} workers={self.workers} tokens={len(self.paths)} guide={self.guide}", flush=True)
        threads = []
        try:
            while not self.stop.is_set():
                # 수정(2026-11-06): 검토 3 — 인증 실패·도중 끊긴 접속(예: 다른 기동의 잔재 확인 접속)이 accept 를 예외로 깨서 서버가
                #   죽던 문제. 그 접속만 기록하고 계속 받는다(진행 중 작업과 무관). #이전 conn = self.listener.accept()
                try:
                    conn = self.listener.accept()
                except (AuthenticationError, EOFError, BrokenPipeError, ConnectionResetError) as e:
                    if self.stop.is_set():
                        break
                    print(f"[경고] 접속 거부/중단(인증 실패 또는 끊김): {type(e).__name__}: {e}", flush=True)
                    continue
                if self.stop.is_set():
                    conn.close()
                    break
                th = threading.Thread(target=self._serve_conn, args=(conn,), daemon=True)
                th.start()
                threads.append(th)
        finally:
            self.close()

    def close(self):
        self.stop.set()
        self.listener.close()
        self.scorer.close()
        for f in (self.sock, self.key_file):
            if os.path.exists(f):
                os.unlink(f)
        os.close(self._lock_fd)        # 수정(2026-11-06): 검토 4 — 소켓·키 정리 뒤 잠금 해제(잠금 파일은 남김). 워커는 위 close 에서 이미 종료
        print(f"[closed] 채점 token {self.n_tok}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--metric-cache", default=DEFAULT_CACHE)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-margins", action="store_true")
    # 수정(2026-11-06): 검토 2 — 단계 ③ 안내 궤적 채점 기본 켬(--guide), 끄려면 --no-guide
    ap.add_argument("--guide", dest="guide", action="store_true", default=True)
    ap.add_argument("--no-guide", dest="guide", action="store_false")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))      # finally 에서 풀·소켓 정리
    ScoreServer(a.socket, a.metric_cache, a.workers, margins=not a.no_margins, guide=a.guide).serve_forever()


if __name__ == "__main__":
    main()
