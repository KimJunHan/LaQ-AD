# 수정(2026-11-06): §12-② 중심선 추출기·사영 도우미 시험(신규 파일).
"""extract_pdm_centerlines 시험. navsim env 에서 실행(실 캐시 부분은 devkit 필요):
  source scripts/navsim/navsim_env.sh
  $NAVSIM_PY tools/navsim/grpo/tests/test_centerlines.py [--real 40] [--scratch <dir>]
1) 합성: 직선·원호에서 project 의 s·d·접선·끝 연장·out_of_range, extract_one 변환(자차 원점·heading 0)
2) 실 캐시(무작위 N token): 자차 원점 사영 = (s 0, d ego_d), 저장 격자 위 사영 s·d 가 원 PDMPath(shapely project, 공식 _frenet 식)와 일치,
   사람 GT 미래 8 점(MetricCache.human_trajectory, ego 프레임)의 중심선 횡편차 통계 출력
3) 왕복: run_extract → CenterlineStore 가 extract_one 과 같은 값, --verify 통과
실패는 AssertionError(비영 종료).
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extract_pdm_centerlines as E  # noqa: E402


class _FakePath:
    def __init__(self, xyh):
        self._states_se2_array = np.asarray(xyh, np.float64)
        d = np.linalg.norm(np.diff(self._states_se2_array[:, :2], axis=0), axis=-1)
        self._progress = np.concatenate([[0.0], np.cumsum(d)])


def _fake_mc(xyh, ego):
    return SimpleNamespace(centerline=_FakePath(xyh),
                           ego_state=SimpleNamespace(rear_axle=SimpleNamespace(x=ego[0], y=ego[1], heading=ego[2])))


def test_synthetic():
    # 직선 y=0, x 0..50
    cl = np.stack([np.linspace(0, 50, 201), np.zeros(201)], -1)
    pts = np.array([[10, 2.0], [10, -3.0], [-4, 1.0], [60, -1.0], [20, 7.0]])
    r = E.project(pts, cl)
    np.testing.assert_allclose(r["s"], [10, 10, -4, 60, 20], atol=1e-9)
    np.testing.assert_allclose(r["d"], [2, -3, 1, -1, 7], atol=1e-9)
    assert list(r["out_of_range"]) == [False, False, True, True, True], r["out_of_range"]
    np.testing.assert_allclose(r["foot"][2], [-4, 0], atol=1e-9)
    # 원호(반지름 20, 반시계): 안쪽 점은 d>0(왼쪽)
    th = np.linspace(0, np.pi / 2, 400)
    arc = np.stack([20 * np.sin(th), 20 - 20 * np.cos(th)], -1)
    p = np.array([[18 * np.sin(0.5), 20 - 18 * np.cos(0.5)]])
    r = E.project(p, arc)
    assert abs(r["d"][0] - 2.0) < 1e-3 and abs(r["s"][0] - 20 * 0.5) < 1e-2 and abs(r["tangent"][0] - 0.5) < 1e-2, r
    # extract_one: 전역 경로를 회전·이동, 자차를 경로 위 s=30 m, 경로 heading 으로 → 변환 후 원점·heading 0
    h0, o = 0.7, np.array([100.0, -50.0])
    R = np.array([[np.cos(h0), -np.sin(h0)], [np.sin(h0), np.cos(h0)]])
    gxy = arc @ R.T + o
    gh = np.unwrap(np.arctan2(np.gradient(gxy[:, 1]), np.gradient(gxy[:, 0])))
    path = _FakePath(np.concatenate([gxy, gh[:, None]], -1))
    k = int(np.searchsorted(path._progress, 15.0))
    ego = (gxy[k, 0], gxy[k, 1], gh[k])
    row = E.extract_one(_fake_mc(path._states_se2_array, ego))
    i0 = int(np.argmin(np.abs(E.S_GRID)))
    assert np.abs(row["xyh"][i0, :2]).max() < 1e-4 and abs(row["xyh"][i0, 2]) < 2e-2, row["xyh"][i0]
    assert abs(row["ego_d"]) < 1e-6 and abs(row["ego_dh"]) < 2e-2
    # 앞쪽 점은 +x, 원호가 반시계라 +y 로 휜다
    assert row["xyh"][i0 + 20, 0] > 9.5 and row["xyh"][i0 + 20, 1] > 0
    # 경로 끝(L≈31.4) 밖 격자점은 valid False 이고 끝 접선으로 직선 연장
    L = path._progress[-1]
    vb = (row["s_ego"] + E.S_GRID >= 0) & (row["s_ego"] + E.S_GRID <= L)
    assert np.array_equal(row["valid"], vb)
    ext = row["xyh"][~row["valid"] & (E.S_GRID > 0), :2]
    dd = np.linalg.norm(np.diff(ext, axis=0), axis=-1)
    np.testing.assert_allclose(dd, E.S_STEP, atol=1e-4)
    # 저장 격자에 사영: 끝 밖은 out_of_range
    r = E.project(np.array([[0.0, 0.0], [ext[-1, 0], ext[-1, 1]]]), row["xyh"][:, :2], E.S_GRID, row["valid"])
    assert abs(r["s"][0]) < 1e-3 and not r["out_of_range"][0] and r["out_of_range"][1]
    print("[통과] 합성: 직선·원호 사영, 끝 연장, extract_one 프레임 변환")


def test_real(n, seed=0):
    import shapely
    paths = E.cache_paths(E.DEFAULT_CACHE)
    toks = sorted(paths)
    rng = np.random.default_rng(seed)
    samp = [toks[i] for i in rng.choice(len(toks), n, replace=False)]
    ds, oors, errs_s, errs_d, last_s, e0 = [], [], [], [], [], []
    for tok in samp:
        mc = E.load_cache(paths[tok])
        row = E.extract_one(mc)
        cl = row["xyh"][:, :2].astype(np.float64)
        r0 = E.project(np.zeros((1, 2)), cl, E.S_GRID, row["valid"], max_lat=np.inf)
        # 0.5 m 재표본 현(chord) 오차: 곡선에서 자차가 중심선에서 떨어져 있으면 s 가 수 cm 어긋난다(측정값을 출력)
        e0.append((abs(r0["s"][0]), abs(r0["d"][0] - row["ego_d"])))
        assert e0[-1][0] < 0.15 and e0[-1][1] < 0.05, (tok, r0, row["ego_d"])
        hp = np.asarray(mc.human_trajectory.poses, np.float64)[:, :2]       # (8,2) ego 프레임 GT 미래
        r = E.project(hp, cl, E.S_GRID, row["valid"])
        # 원 PDMPath(전역)에 shapely 사영, grpo_scorer._frenet 과 같은 식
        ra = mc.ego_state.rear_axle
        c, s = np.cos(ra.heading), np.sin(ra.heading)
        gx, gy = ra.x + c * hp[:, 0] - s * hp[:, 1], ra.y + s * hp[:, 0] + c * hp[:, 1]
        sp = np.asarray(mc.centerline.project(shapely.points(np.stack([gx, gy], -1))))
        st = mc.centerline.interpolate(sp, as_array=True)
        dref = np.cos(st[:, 2]) * (gy - st[:, 1]) - np.sin(st[:, 2]) * (gx - st[:, 0])
        inside = ~r["out_of_range"]
        errs_s.append(np.abs((sp - row["s_ego"]) - r["s"])[inside])
        errs_d.append(np.abs(dref - r["d"])[inside])
        ds.append(r["d"])
        oors.append(r["out_of_range"])
        last_s.append(r["s"][-1])
    es, ed = np.concatenate(errs_s), np.concatenate(errs_d)
    # 0.5 m 재표본 + float32 → 곡선에서 s 는 수 cm(현 오차), d 는 mm–cm 수준까지 허용, 그 이상은 실패
    assert es.max() < 0.15 and ed.max() < 0.05, (es.max(), ed.max())
    d = np.abs(np.stack(ds))
    oor = np.stack(oors)
    e0 = np.array(e0)
    print(f"[통과] 실 캐시 {n} token: 원 PDMPath 대비 GT 경유점 |Δs| 최대 {es.max():.4f} 중앙 {np.median(es):.4f} m, "
          f"|Δd| 최대 {ed.max():.4f} 중앙 {np.median(ed):.4f} m; 자차 원점 |Δs| 최대 {e0[:, 0].max():.4f}, |Δd| 최대 {e0[:, 1].max():.4f} m")
    print(f"[GT 미래 vs 중심선] |d| 평균 {d.mean():.3f} 중앙 {np.median(d):.3f} p90 {np.percentile(d, 90):.3f} "
          f"max {d.max():.3f} m; 경유점별 평균 {np.round(d.mean(0), 3).tolist()}; out_of_range 비율 {oor.mean():.4f}; "
          f"4 s 끝 s 최대 {max(last_s):.1f} m (격자 끝 {E.S_MAX})")


def test_beyond_end(tok="9804c6bbf2715b87"):
    """실측 사례: 자차가 PDM 중심선 끝을 지난 장면 — 유효 점 0 개여도 끝 접선 연장 격자·out_of_range 로 처리."""
    paths = E.cache_paths(E.DEFAULT_CACHE)
    mc = E.load_cache(paths[tok])
    row = E.extract_one(mc)
    assert row["s_ego"] > row["path_len"] and not row["valid"].any(), (row["s_ego"], row["path_len"])
    hp = np.asarray(mc.human_trajectory.poses, np.float64)[:, :2]
    r = E.project(hp, row["xyh"][:, :2].astype(np.float64), E.S_GRID, row["valid"])
    assert r["out_of_range"].all() and np.isfinite(r["s"]).all() and np.isfinite(r["d"]).all()
    # 연장 직선 = 원 경로 끝 접선: 끝점에서 같은 방향
    a = np.asarray(mc.centerline._states_se2_array)
    u = a[-1, :2] - a[-2, :2]
    ra = mc.ego_state.rear_axle
    h_end = np.arctan2(u[1], u[0]) - ra.heading
    assert abs((row["xyh"][0, 2] - h_end + np.pi) % (2 * np.pi) - np.pi) < 1e-5
    print(f"[통과] 끝 지난 장면 {tok}: s_ego {row['s_ego']:.1f} > L {row['path_len']:.1f}, 유효 0, GT 8 점 모두 out_of_range, "
          f"연장 d {np.round(r['d'], 2).tolist()}")


def test_roundtrip(scratch, n=40):
    out = Path(scratch) / "cl_roundtrip"
    if out.exists():
        for f in out.iterdir():
            f.unlink()
    a = SimpleNamespace(metric_cache=E.DEFAULT_CACHE, out=str(out), workers=4, shard=16, limit=n, verify_n=10)
    E.run_extract(a)
    E.run_extract(a)                      # 재개: 모두 건너뛰어야 한다
    st = E.CenterlineStore(out)
    assert len(st) == n
    paths = E.cache_paths(E.DEFAULT_CACHE)
    for tok in sorted(paths)[:n:7]:
        ref = E.extract_one(E.load_cache(paths[tok]))
        got = st.get(tok)
        assert np.array_equal(got["xyh"], ref["xyh"]) and np.array_equal(got["valid"], ref["valid"])
        assert got["s_ego"] == ref["s_ego"] and got["ego_d"] == ref["ego_d"]
    E.run_verify(a)
    # 손상 샤드 감지: 망가뜨리면 재추출로 복구되어야
    (out / "shard_00000.npz").write_bytes(b"broken")
    E.run_extract(a)
    E.run_verify(a)
    print(f"[통과] 왕복: {n} token 추출 → 재개 건너뜀 → CenterlineStore 값 동일 → --verify 통과, 손상 샤드 재생성")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=int, default=40)
    ap.add_argument("--scratch", default=tempfile.gettempdir())
    a = ap.parse_args()
    if os.environ.get("OPENBLAS_CORETYPE") != "SkylakeX":
        raise SystemExit("OPENBLAS_CORETYPE=SkylakeX 필요")
    test_synthetic()
    test_real(a.real)
    test_beyond_end()
    test_roundtrip(a.scratch)
