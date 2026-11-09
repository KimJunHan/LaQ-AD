# 수정(2026-11-06): C3 GRPO 설계 §12-② — navtrain 지표 캐시의 PDM-Closed 중심선을 자차 프레임 npz 로 추출 + 사영 도우미(신규 파일).
"""PDM 중심선 추출기 + 사영 도우미 (docs/c3_grpo_design.md §2.1 축 기저, §5, §6 lat_dev, §12-②).

무엇을 저장하나
  지표 캐시 MetricCache.centerline(PDMPath; PDM-Closed 가 경로 차선으로 만든 0.25 m 간격 이산 경로, 공식 채점의 lk·ep 기준선)을
  현재 자차 후륜축 자세 기준 ego 프레임(+x 전진, +y 왼쪽, navsim ego 프레임과 같음)으로 바꾸고,
  자차 원점의 사영점 s_ego 기준 상대 호길이 S_GRID = −10 .. +120 m, 0.5 m 간격(261 점)으로 다시 표본한다.
    · 범위 근거: 4 s 수평선에서 navtrain 최대 진행 ≈ 70–80 m(≈ 20 m/s) + 표본 잡음 여유, 뒤쪽 10 m 는 후진·정지 표본용.
    · 경로 끝(0 또는 길이 L)을 벗어나는 격자 점은 끝 접선 방향으로 직선 연장하고 valid=False 로 표시한다.
  샤드 npz(shard_00000.npz …, 샤드당 SHARD 토큰): tokens (n,) str, xyh (n,261,3) float32 [x, y, heading(연속, unwrap)],
    valid (n,261) bool(원 경로 [0, L] 안; 자차가 경로 끝을 지난 장면은 0 개일 수 있음), s_ego (n,) float64 (원 경로 위 자차 사영 호길이), path_len (n,), ego_d (n,) 자차 원점 부호 횡편차(+왼쪽),
    ego_dh (n,) 사영점 접선 heading(ego 프레임, = 접선 − 자차 heading). [−π, π)  # 수정(2026-11-06): 범위 명시(감기 추가)
  index.json: {token: [shard, row]}, meta.json: 격자·원천·개수.

사영 도우미(numpy 만, nuplan 불필요 → hipad_bw 학습 env 에서 import 가능)
  project(points_xy, cl_xy, s_grid=None, valid=None, max_lat=5.0) → dict(s, d, tangent, out_of_range, foot)
    · 폴리라인 최근접 선분 사영(shapely LineString.project 와 같은 최근접 규칙). s = 호길이(s_grid 를 주면 그 좌표계, 즉 자차 사영 기준
      상대 m), d = 부호 횡편차(+왼쪽), tangent = 사영 선분 방향각(rad).
    · 폴리라인 양 끝을 지나는 점: 끝 접선 직선으로 연장해 s(<시작, >끝), d 를 그 직선 기준으로 준다.
    · out_of_range = 끝을 지남 | |d| > max_lat(기본 5 m). 학습 쪽은 이 마스크로 기저 신뢰도를 판단한다(조용히 버리지 말 것).
  CenterlineStore(out_dir).get(token) → dict(xyh, valid, s_grid, ...) ; .project(token, points_xy) → project(...) 결과.

사용
  source scripts/navsim/navsim_env.sh
  $NAVSIM_PY tools/navsim/grpo/extract_pdm_centerlines.py --out /home1/irteam/data-vol1/src/HiP-AD/data/navsim/grpo/pdm_centerlines_navtrain \
      --workers 16                       # 재개 가능: 검증을 통과한 기존 샤드는 건너뛴다
  ... --limit 500 --out <scratch>        # 처리량 측정용 부분 실행(정렬 token 앞 500 개)
  ... --verify                           # 출력만 검사: 모든 샤드 로드·형식·유한값·index 일치, 무작위 표본을 원 캐시와 다시 대조
시험: tools/navsim/grpo/tests/test_centerlines.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import lzma
import pickle
import sys
import time
from pathlib import Path

import numpy as np

S_MIN, S_MAX, S_STEP = -10.0, 120.0, 0.5
S_GRID = np.round(np.arange(S_MIN, S_MAX + 1e-9, S_STEP), 6)      # (261,)
SHARD = 500
MAX_WORKERS = 16
DEFAULT_CACHE = "/home1/irteam/data-vol1/datasets/navsim/exp/metric_cache_navtrain"
_EPS = 1e-9


# ---------------------------------------------------------------- 사영 도우미 (numpy 전용)
def _seg_arrays(cl_xy):
    v = np.asarray(cl_xy, np.float64)
    seg = v[1:] - v[:-1]
    ln = np.linalg.norm(seg, axis=-1)
    keep = np.concatenate([[True], ln > _EPS])                     # 길이 0 선분 제거(중복 점)
    v = v[keep]
    if len(v) < 2:
        raise ValueError("중심선 점이 2 개 미만(서로 다른 점 기준)")
    seg = v[1:] - v[:-1]
    ln = np.linalg.norm(seg, axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(ln)])
    return v, seg, ln, cum


def project(points_xy, cl_xy, s_grid=None, valid=None, max_lat=5.0):
    """points_xy (..., 2) 를 폴리라인 cl_xy (M, 2) 에 사영. 반환 dict:
    s (...): 호길이(s_grid 를 주면 그 좌표 — 첫 유효 점의 s_grid 값에서 시작), d (...): 부호 횡편차(+왼쪽),
    tangent (...): 사영 선분 방향각, out_of_range (...) bool: 끝을 지나거나 |d| > max_lat, foot (..., 2): 사영점(끝 밖은 연장선 위).
    저장 격자(CenterlineStore)는 원 경로 끝 밖 점이 이미 끝 접선 직선 연장이므로 격자 전체를 폴리라인으로 쓰고,
    valid (M,) 를 주면 유효 구간 밖으로 사영된 점도 out_of_range 로 표시한다(유효 점이 없으면 전부 out_of_range)."""
    cl_xy = np.asarray(cl_xy, np.float64)
    s_off = 0.0 if s_grid is None else float(np.asarray(s_grid)[0])
    v, seg, ln, cum = _seg_arrays(cl_xy)
    p = np.asarray(points_xy, np.float64)
    shp = p.shape[:-1]
    p = p.reshape(-1, 2)
    rel = p[:, None, :] - v[None, :-1, :]                          # (N, S, 2)
    t_raw = (rel * seg[None]).sum(-1) / (ln[None] ** 2)
    t = np.clip(t_raw, 0.0, 1.0)
    foot = v[None, :-1] + t[..., None] * seg[None]
    dist2 = ((p[:, None] - foot) ** 2).sum(-1)
    k = dist2.argmin(-1)                                           # 최근접 선분(동률이면 앞쪽 = shapely 와 같은 규칙)
    n = np.arange(len(p))
    tk = t_raw[n, k]
    last = len(seg) - 1
    # 끝 밖: 첫 선분 t<0 이거나 끝 선분 t>1 이면 접선 연장선에 사영
    before = (k == 0) & (tk < 0)
    after = (k == last) & (tk > 1)
    tt = np.where(before | after, tk, np.clip(tk, 0, 1))
    u = seg[k] / ln[k, None]
    fk = v[k] + tt[:, None] * seg[k]
    s = cum[k] + tt * ln[k]
    w = p - fk
    d = u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]
    tang = np.arctan2(u[:, 1], u[:, 0])
    oor = before | after | (np.abs(d) > max_lat)
    if valid is not None:                                          # 저장 격자: 연장 점 구간도 원 경로 밖
        valid = np.asarray(valid, bool)
        if valid.any():
            sg = np.asarray(s_grid, np.float64) - s_off
            oor = oor | (s < sg[valid].min() - 1e-6) | (s > sg[valid].max() + 1e-6)
        else:
            oor = np.ones_like(oor)
    return dict(s=(s + s_off).reshape(shp), d=d.reshape(shp), tangent=tang.reshape(shp),
                out_of_range=oor.reshape(shp), foot=fk.reshape(shp + (2,)))


class CenterlineStore:
    """추출 결과 읽기(nuplan 불필요). 샤드는 처음 쓸 때 통째로 읽어 둔다."""

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.index = json.loads((self.dir / "index.json").read_text())
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.s_grid = np.asarray(self.meta["s_grid"], np.float64)
        self._shards = {}

    def __contains__(self, tok):
        return tok in self.index

    def __len__(self):
        return len(self.index)

    def _shard(self, i):
        if i not in self._shards:
            with np.load(self.dir / f"shard_{i:05d}.npz") as z:
                self._shards[i] = {k: z[k] for k in z.files}
        return self._shards[i]

    def get(self, tok):
        if tok not in self.index:
            raise KeyError(f"중심선 없음: {tok}")
        sh, r = self.index[tok]
        z = self._shard(sh)
        if z["tokens"][r] != tok:
            raise RuntimeError(f"index 불일치: {tok} → shard {sh} row {r} = {z['tokens'][r]}")
        return dict(xyh=z["xyh"][r], valid=z["valid"][r], s_grid=self.s_grid, s_ego=float(z["s_ego"][r]),
                    path_len=float(z["path_len"][r]), ego_d=float(z["ego_d"][r]), ego_dh=float(z["ego_dh"][r]))

    def project(self, tok, points_xy, max_lat=5.0):
        c = self.get(tok)
        return project(points_xy, c["xyh"][:, :2], self.s_grid, c["valid"], max_lat)


# ---------------------------------------------------------------- 추출 (navsim env)
def load_cache(path):
    # 지표 캐시는 이 서버에서 devkit(run_metric_caching)이 직접 만든 로컬 파일 — devkit 공식 형식(lzma pickle)이라 pickle 이 필요하다.
    with open(path, "rb") as f:
        return pickle.loads(lzma.decompress(f.read()))


def cache_paths(cache_dir):
    md = Path(cache_dir) / "metadata"
    csvs = sorted(p for p in md.iterdir() if p.suffix == ".csv")
    if len(csvs) != 1:
        raise RuntimeError(f"{md}: metadata csv 가 정확히 1개여야 한다({len(csvs)})")
    rows = csvs[0].read_text().splitlines()[1:]
    out = {r.split("/")[-2]: r for r in rows}
    if len(out) != len(rows):
        raise RuntimeError("metadata csv 에 중복 token")
    return out


def to_ego(xy, h, ego_xyh):
    """전역 (N,2),(N,) → ego 프레임(후륜축 원점, heading 0)."""
    x0, y0, h0 = ego_xyh
    c, s = np.cos(h0), np.sin(h0)
    dx, dy = xy[:, 0] - x0, xy[:, 1] - y0
    return np.stack([c * dx + s * dy, -s * dx + c * dy], -1), h - h0


def extract_one(mc):
    """MetricCache → 한 token 의 행(dict). 실패는 예외."""
    path = mc.centerline
    arr = np.asarray(path._states_se2_array, np.float64)          # (M,3), heading unwrap 됨
    prog = np.asarray(path._progress, np.float64)
    if arr.ndim != 2 or len(arr) < 2 or not np.isfinite(arr).all():
        raise ValueError(f"중심선 형식 이상 {arr.shape}")
    ra = mc.ego_state.rear_axle
    xy, h = to_ego(arr[:, :2], arr[:, 2], (ra.x, ra.y, ra.heading))
    pr = project(np.zeros((1, 2)), xy, max_lat=np.inf)
    s_ego, ego_d = float(pr["s"][0]), float(pr["d"][0])
    # 우리 사영의 s 는 길이 0 선분 제거 후 누적 → 원 progress 와 같은 값(중복 점은 길이 0)
    L = float(prog[-1])
    s_abs = s_ego + S_GRID
    valid = (s_abs >= 0.0) & (s_abs <= L)
    out = np.zeros((len(S_GRID), 3))
    sc = np.clip(s_abs, 0.0, L)
    out[:, 0] = np.interp(sc, prog, xy[:, 0])
    out[:, 1] = np.interp(sc, prog, xy[:, 1])
    out[:, 2] = np.interp(sc, prog, h)
    v, seg, ln, _ = _seg_arrays(xy)
    for side, mask, base, u in (("b", s_abs < 0, xy[0], seg[0] / ln[0]), ("a", s_abs > L, xy[-1], seg[-1] / ln[-1])):
        if mask.any():
            ds = s_abs[mask] - (0.0 if side == "b" else L)
            out[mask, 0] = base[0] + ds * u[0]
            out[mask, 1] = base[1] + ds * u[1]
            out[mask, 2] = np.arctan2(u[1], u[0])
    # 연장 점 heading 을 인접 유효 heading 과 같은 가지(2π)로 맞춘다
    if valid.any():
        for mask, ref in ((s_abs < 0, out[valid][0, 2]), (s_abs > L, out[valid][-1, 2])):
            if mask.any():
                out[mask, 2] = ref + (out[mask, 2] - ref + np.pi) % (2 * np.pi) - np.pi
    # 유효 점이 0 개일 수 있다(실측: 자차가 PDM 중심선 끝을 지나 있는 장면, 예 9804c6bbf2715b87 s_ego 28.9 > L 15.2).
    # 그때도 끝 접선 연장 격자를 저장하고 valid 로 표시 — 학습 쪽이 out_of_range 로 판단한다(조용히 빼지 않음).
    # 수정(2026-11-06): 검토 2 — h 는 unwrap 된 heading − 자차 heading 이라 ±2π 배수가 남을 수 있다 → [−π, π) 로 감는다.
    #   ego_dh 는 메타(학습·advantage·채점기 어디서도 읽지 않음, grep 확인) → 기존 추출 결과 재생성 불필요. #이전 감지 않음
    ego_dh = float((np.interp(np.clip(s_ego, 0, L), prog, h) + np.pi) % (2 * np.pi) - np.pi)
    return dict(xyh=out.astype(np.float32), valid=valid, s_ego=s_ego, path_len=L, ego_d=ego_d, ego_dh=ego_dh)


FIELDS = ("xyh", "valid", "s_ego", "path_len", "ego_d", "ego_dh")


def _shard_ok(f, toks):
    try:
        with np.load(f) as z:
            ok = list(z["tokens"]) == list(toks) and z["xyh"].shape == (len(toks), len(S_GRID), 3) \
                and np.isfinite(z["xyh"]).all() and all(k in z.files for k in FIELDS)
        return ok
    except Exception:
        return False


def _do_shard(args):
    i, toks, paths, out_dir = args
    f = Path(out_dir) / f"shard_{i:05d}.npz"
    if f.exists() and _shard_ok(f, toks):
        return i, len(toks), 0.0, True
    t = time.time()
    rows = []
    for tok in toks:
        try:
            rows.append(extract_one(load_cache(paths[tok])))
        except Exception as e:
            raise RuntimeError(f"token {tok} ({paths[tok]}): {type(e).__name__}: {e}") from e
    tmp = f.with_name(f.stem + ".tmp.npz")
    np.savez(tmp, tokens=np.array(toks), **{k: np.stack([r[k] for r in rows]) for k in FIELDS})
    os.replace(tmp, f)
    return i, len(toks), time.time() - t, False


def run_extract(a):
    paths = cache_paths(a.metric_cache)
    toks = sorted(paths)
    if a.limit:
        toks = toks[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shards = [toks[i:i + a.shard] for i in range(0, len(toks), a.shard)]
    meta = dict(source=str(a.metric_cache), n_tokens=len(toks), shard=a.shard, n_shards=len(shards),
                s_grid=S_GRID.tolist(), s_min=S_MIN, s_max=S_MAX, s_step=S_STEP,
                frame="navsim ego (rear axle, +x fwd, +y left) at MetricCache.ego_state",
                field="MetricCache.centerline (PDMPath)", created="2026-11-06")
    mp = out / "meta.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        if {k: old.get(k) for k in ("source", "n_tokens", "shard", "s_grid")} != \
                {k: meta[k] for k in ("source", "n_tokens", "shard", "s_grid")}:
            raise RuntimeError(f"{mp} 의 설정이 이번 실행과 다르다 — 다른 --out 을 쓰거나 지우고 다시")
    mp.write_text(json.dumps(meta))
    from multiprocessing import get_context
    t0 = time.time()
    done = 0
    with get_context("fork").Pool(a.workers) as pool:
        for i, n, dt, skipped in pool.imap_unordered(_do_shard, [(i, s, paths, str(out)) for i, s in enumerate(shards)]):
            done += n
            el = time.time() - t0
            print(f"shard {i:05d} {'건너뜀' if skipped else f'{dt:.1f}s'}  {done}/{len(toks)}  {el:.0f}s  "
                  f"{done / max(el, 1e-9):.1f} tok/s", flush=True)
    index = {}
    for i, s in enumerate(shards):
        for r, tok in enumerate(s):
            index[tok] = [i, r]
    (out / "index.json").write_text(json.dumps(index))
    wall = time.time() - t0
    print(f"[완료] {len(toks)} token, {len(shards)} 샤드, {wall:.1f}s ({len(toks) / wall:.1f} tok/s, 워커 {a.workers}) → {out}")


def _verify_sample(args):
    tok, path, row = args
    ref = extract_one(load_cache(path))
    for k in FIELDS:
        a, b = np.asarray(ref[k]), np.asarray(row[k])
        if k == "xyh":
            a = a.astype(np.float32)
        if not np.array_equal(a, b):
            raise AssertionError(f"{tok}: {k} 재추출 불일치")
    return tok


def run_verify(a):
    out = Path(a.out)
    st = CenterlineStore(out)
    meta = st.meta
    paths = cache_paths(meta["source"])
    toks = sorted(paths)[:meta["n_tokens"]]
    if sorted(st.index) != toks:
        raise AssertionError(f"index token {len(st.index)} != 원천 {len(toks)}")
    n_valid, n_short = [], 0
    i0 = int(np.argmin(np.abs(S_GRID)))
    for i in range(meta["n_shards"]):
        z = st._shard(i)
        exp = toks[i * meta["shard"]:(i + 1) * meta["shard"]]
        if list(z["tokens"]) != exp:
            raise AssertionError(f"shard {i}: token 목록 불일치")
        for k in ("xyh", "s_ego", "path_len", "ego_d", "ego_dh"):
            if not np.isfinite(z[k]).all():
                raise AssertionError(f"shard {i}: {k} 비유한값")
        n_valid.append(z["valid"].sum(1))
        n_short += int((~z["valid"][:, i0:i0 + 121]).any(1).sum())
        st._shards.pop(i)
    n_valid = np.concatenate(n_valid)
    rng = np.random.default_rng(0)
    samp = [toks[j] for j in rng.choice(len(toks), min(a.verify_n, len(toks)), replace=False)]
    from multiprocessing import get_context
    with get_context("fork").Pool(a.workers) as pool:
        for _ in pool.imap_unordered(_verify_sample, [(t, paths[t], st.get(t)) for t in samp]):
            pass
    print(f"[검증 통과] {len(toks)} token, {meta['n_shards']} 샤드; 유효 격자점 수 min {n_valid.min()} "
          f"p1 {np.percentile(n_valid, 1):.0f} 중앙 {np.median(n_valid):.0f} (전체 {len(S_GRID)}), 유효 0 개 token "
          f"{int((n_valid == 0).sum())}, s 0..+60 m 중 일부라도 연장인 token {n_short}; "
          f"무작위 {len(samp)} token 재추출 비트 동일")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric-cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard", type=int, default=SHARD)
    ap.add_argument("--limit", type=int, default=0, help="정렬 token 앞 N 개만(시험·속도 측정)")
    ap.add_argument("--verify", action="store_true", help="추출 대신 출력 검사")
    ap.add_argument("--verify-n", type=int, default=200)
    a = ap.parse_args()
    if a.workers > MAX_WORKERS:
        raise SystemExit(f"워커 상한 {MAX_WORKERS}")
    if os.environ.get("OPENBLAS_CORETYPE") != "SkylakeX":
        raise SystemExit("OPENBLAS_CORETYPE=SkylakeX 필요(scripts/navsim/navsim_env.sh)")
    (run_verify if a.verify else run_extract)(a)


if __name__ == "__main__":
    main()
