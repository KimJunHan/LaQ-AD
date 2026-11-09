# 수정(2026-11-06): C3 §12-① 채점기 확장(M3) 회귀 시험 — 신규 파일.
"""확장 전 채점기(backup/code_20261106/grpo_scorer.py.pre_grpo_ext)와 확장 후(tools/navsim/grpo_scorer.py)를 같은 token·후보로
돌려 기존 출력 키가 비트 동일한지 확인한다. 허용되는 차이는 하나뿐: comfort 첫 실패가 이력 패딩에서 시작한 후보의
fail_idx[:, 6]·fail_wp[:, 6] 이 0 → -2 로 바뀐 것(설계 §4.3; 수정(2026-11-06) 검토 4 — 첫 실패 시점에 후보가 넘은 신호를 PDM 참조도
같은 시점에 모두 넘을 때만 -2, 아니면 종전처럼 0. 종전·검토 2·새 규칙 개수를 출력한다). 그 경우 hc 하위 점수가 0 인지도 확인한다.
또 확장 키의 기본 성질(coll_t ≥ 0 ⇔ 필터 전 nc < 1 은 채점기 안에서 단언됨, 정면/측면·전방·측방 거리 ≥ 전체 거리)을 본다.
  source scripts/navsim/navsim_env.sh
  $NAVSIM_PY tools/navsim/grpo/tests/test_scorer_ext.py --tokens 64 --K 16 --workers 8
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import importlib.machinery
import importlib.util
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools/navsim"))
import grpo_scorer as NEW  # noqa: E402

_loader = importlib.machinery.SourceFileLoader("grpo_scorer_old", str(ROOT / "backup/code_20261106/grpo_scorer.py.pre_grpo_ext"))
_spec = importlib.util.spec_from_loader("grpo_scorer_old", _loader)
OLD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OLD)

OLD_KEYS = ["subscores", "multiplicative", "epdms", "pdms_v1like", "fail_idx", "fail_wp", "filtered", "cont", "frenet",
            "ref_subscores"]


def _init():
    NEW._init_worker(True)
    OLD._init_worker(True)


def _eq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


def _comfort_rules(mc, cands):
    """수정(2026-11-06): 검토 2 — 같은 묶음 채점 상태에서 첫 편안함 실패를 여러 규칙으로 분류해 개수를 비교한다.
    수정(2026-11-06): 검토 3 — prev = 종전(이력 안이면 전부 -2), rev2 = 검토 2 규칙(창 11 + 참조 동시 실패, 폐기),
    rev3 = 검토 3 규칙(실패 신호 비율이 참조와 비트 동일, 폐기). 수정(2026-11-06): 검토 4 — new = 채점기 규칙(반사실: 후보가 넘은
    신호를 참조도 같은 시점에 모두 넘을 때만 -2)."""
    W = NEW._W
    ego0, t0 = mc.ego_state, mc.ego_state.time_point
    st = np.stack([NEW.get_trajectory_as_array(mc.trajectory, W["sampling"], t0)] +
                  [NEW.get_trajectory_as_array(NEW.transform_trajectory(NEW.Trajectory(v.astype(np.float32), W["agent_sampling"]),
                                                                        ego0), W["sampling"], t0) for v in cands])
    sim = W["simulator"].simulate_proposals(st, ego0)
    tr = W["traffic"].simulate_environment(sim[1], mc)
    W["scorer"].score_proposals(sim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
                                mc.map_parameters, tr, mc.past_human_trajectory)
    ratio, n_hist, sig6 = NEW._comfort_signals(W["scorer"]._states, W["scorer"], per_signal=True)   # 수정(2026-11-06): 검토 3
    if ratio is None:
        return None
    bad = ratio >= 1.0
    first = NEW._first_true(bad)
    fi = np.clip(first, 0, bad.shape[1] - 1)
    ref_bad = bad[0, fi]
    in_hist = (first >= 0) & (first < n_hist)
    # 수정(2026-11-06): 검토 3 — 한계를 넘은 신호의 비율이 참조와 모두 비트 동일한가(검토 3 규칙, 비교용)
    same = np.array([all(sig6[p, j, fi[p]] == sig6[0, j, fi[p]] for j in range(6) if sig6[p, j, fi[p]] >= 1.0)
                     for p in range(len(first))])
    # 수정(2026-11-06): 검토 4 — 반사실 규칙: 후보가 넘은 신호를 참조도 같은 시점에 모두 넘는가(채점기와 독립적으로 루프로 재계산)
    ref_also = np.array([all(sig6[0, j, fi[p]] >= 1.0 for j in range(6) if sig6[p, j, fi[p]] >= 1.0)
                         for p in range(len(first))])
    lab = lambda h: np.where(first < 0, -1, np.where(h, -2, np.maximum(first - n_hist, 0)))[1:]
    return dict(prev=lab(in_hist), rev2=lab(in_hist & ((first < n_hist - 11) | ref_bad)), rev3=lab(in_hist & same),
                new=lab(in_hist & ref_also),   # 수정(2026-11-06): 검토 4 #이전 new=lab(in_hist & same)
                ref_bad_diff=(in_hist & ref_bad & ~same)[1:], early_dep=(in_hist & (first < n_hist - 11) & ~same)[1:],
                n_hist=n_hist)


def _job(args):
    tok, path, cands = args
    mc = NEW.load_cache(path)
    o_old = OLD.score_token(mc, cands, True)
    o_new = NEW.score_token(mc, cands, True, guide=True)
    rep = dict(tok=tok, n=len(cands), changed_comfort=0, frenet_ulp=0, frenet_ulp_max=0.0, coll=int((o_new["coll_t"] >= 0).sum()),
               front=int((o_new["coll_front"] & (o_new["coll_t"] >= 0)).sum()), ref_epdms=float(o_new["ref_epdms"]))
    for k in OLD_KEYS:
        if k in ("fail_idx", "fail_wp"):
            a, b = o_old[k].copy(), o_new[k].copy()
            diff = a != b
            if diff[:, :6].any():
                raise AssertionError(f"{tok} {k}: comfort 밖 열이 바뀜")
            d6 = diff[:, 6]
            if not (np.all(a[d6, 6] == 0) and np.all(b[d6, 6] == -2)):
                raise AssertionError(f"{tok} {k}: comfort 차이가 0 → -2 가 아님 {a[d6, 6]} → {b[d6, 6]}")
            if k == "fail_idx":
                rep["changed_comfort"] = int(d6.sum())
                if d6.any() and not np.all(o_new["subscores"][d6, 7] == 0.0) and not o_new["filtered"][6]:
                    raise AssertionError(f"{tok}: -2 후보인데 hc ≠ 0")
            if a.dtype != b.dtype:
                raise AssertionError(f"{tok} {k}: dtype {a.dtype} → {b.dtype}")
        elif k == "frenet" and not _eq(o_old[k], o_new[k]):
            # 수정(2026-11-06): 검토 2 — frenet 은 numpy 1.23 cos/sin 의 SVML/libm 경로 선택(출력 버퍼 인접 여부, 할당 이력 의존)으로
            #   같은 입력에서도 마지막 ulp 가 가끔 다르다(실측 4 회 중 1 회 실패). |차| ≤ 1e-12 m 허용, 개수 보고(test_score_server 와 같은 기준).
            # 수정(2026-11-06): 검토 3 — shape 를 먼저, NaN 위치가 같은지 단언한 뒤 차를 본다(nanmax 는 NaN 위치 차이를 놓친다)
            if o_old[k].shape != o_new[k].shape or o_old[k].dtype != o_new[k].dtype:
                raise AssertionError(f"{tok} frenet: shape/dtype {o_old[k].shape}/{o_old[k].dtype} → {o_new[k].shape}/{o_new[k].dtype}")
            if not np.array_equal(np.isnan(o_old[k]), np.isnan(o_new[k])):
                raise AssertionError(f"{tok} frenet: NaN 위치 다름")
            d = np.abs(np.where(np.isnan(o_old[k]), 0.0, o_old[k] - o_new[k]))
            if d.max() > 1e-12:
                raise AssertionError(f"{tok} frenet: |차| {d.max():.3e} > 1e-12")
            rep["frenet_ulp"] = int((d > 0).sum())
            rep["frenet_ulp_max"] = float(d.max())
        elif not _eq(o_old[k], o_new[k]):
            raise AssertionError(f"{tok} {k}: 확장 전/후 다름 (최대 |차| {np.nanmax(np.abs(np.asarray(o_old[k], float) - np.asarray(o_new[k], float))):.3e}, 다른 원소 {int((np.asarray(o_old[k]) != np.asarray(o_new[k])).sum())})")
    mo = o_new["cont"][:, NEW.CONT.index("min_obj_dist_m")]
    fs = o_new["min_obj_dist_fs_m"]
    ok = np.isnan(fs) | (fs >= mo - 1e-12)
    if not ok.all():
        raise AssertionError(f"{tok}: 전방·측방 거리 < 전체 최소 거리")
    if not np.allclose(o_new["lat_dev_wp_m"], np.abs(o_new["frenet"][..., 1]).mean(-1)):
        raise AssertionError("lat_dev_wp_m")
    rep["fs_gt_all"] = int((fs > mo + 1e-9).sum())
    # 수정(2026-11-06): 검토 2 — comfort 귀속 규칙 비교(재계산 = 채점기 출력 단언)
    cr = _comfort_rules(mc, cands)
    # 수정(2026-11-06): 검토 3 — 값 비교 규칙. rev2_wrong = 검토 2 규칙이 -2 인데 값이 참조와 달라 후보 의존인 것
    z = dict(prev2=0, new2=0, rev2_2=0, to_wp0=0, rev2_wrong=0, early_dep=0, ref_bad_diff=0, rev3_2=0, rev3_vs_new=0)  # 수정(2026-11-06): 검토 4
    rep["n_hist"] = None
    if cr is not None and not o_new["filtered"][6]:
        if not np.array_equal(cr["new"], o_new["fail_idx"][:, 6]):
            raise AssertionError(f"{tok}: comfort 재계산 {cr['new']} != 채점기 {o_new['fail_idx'][:, 6]}")
        z.update(prev2=int((cr["prev"] == -2).sum()), new2=int((cr["new"] == -2).sum()), rev2_2=int((cr["rev2"] == -2).sum()),
                 to_wp0=int(((cr["prev"] == -2) & (cr["new"] == 0)).sum()),
                 rev2_wrong=int(((cr["rev2"] == -2) & (cr["new"] != -2)).sum()),
                 early_dep=int(cr["early_dep"].sum()), ref_bad_diff=int(cr["ref_bad_diff"].sum()),
                 rev3_2=int((cr["rev3"] == -2).sum()), rev3_vs_new=int((cr["rev3"] != cr["new"]).sum()))   # 수정(2026-11-06): 검토 4
        rep["n_hist"] = cr["n_hist"]
    rep["comfort"] = z
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric-cache", default=os.path.join(os.environ.get("NAVSIM_EXP_ROOT", ""), "metric_cache_navtrain"))
    ap.add_argument("--anchors", default=str(ROOT / "data/kmeans/navsim_v2_plan_temp_48.npy"))
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.workers > 16:
        raise ValueError("워커 상한 16")
    rng = np.random.default_rng(a.seed)
    anchors = np.load(a.anchors).astype(np.float64)
    paths = NEW.cache_paths(a.metric_cache)
    toks = sorted(paths)
    toks = [toks[i] for i in rng.choice(len(toks), a.tokens, replace=False)]
    jobs = [(t, paths[t], NEW._as_poses(NEW._bench_cands(rng, anchors, a.K, a.noise))) for t in toks]
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init) as pool:
        reps = list(pool.imap_unordered(_job, jobs))
    n = sum(r["n"] for r in reps)
    print(f"[확장 회귀 통과] {len(reps)} token × K={a.K} = {n} 후보 ({time.time() - t0:.0f}s): 기존 키 {len(OLD_KEYS)} 개 비트 동일(frenet 은 아래 ulp 허용), "
          f"comfort 0 → -2 변경 {sum(r['changed_comfort'] for r in reps)} 후보(모두 hc 0), at-fault 충돌 {sum(r['coll'] for r in reps)} "
          f"(정면 {sum(r['front'] for r in reps)}), 전방·측방 거리 > 전체 거리 {sum(r['fs_gt_all'] for r in reps)} 후보, "
          f"PDM-Closed r* = 0 token {sum(r['ref_epdms'] == 0 for r in reps)}")
    cz = {k: sum(r["comfort"][k] for r in reps) for k in reps[0]["comfort"]}   # 수정(2026-11-06): 검토 2
    print(f"[frenet ulp] 비트 불일치 원소 {sum(r['frenet_ulp'] for r in reps)} 개, 최대 |차| "
          f"{max(r['frenet_ulp_max'] for r in reps):.2e} (허용 1e-12; 나머지 키는 비트 동일)")
    nh = [r["n_hist"] for r in reps if r["n_hist"] is not None]
    print(f"[comfort 귀속(검토 4, 반사실: 참조도 같은 시점·신호 실패면 -2)] 검토 3(비트 비교) -2 {cz['rev3_2']}, 검토 3 과 다른 후보 "
          f"{cz['rev3_vs_new']}")   # 수정(2026-11-06): 검토 4
    print(f"[comfort 귀속] n_hist 분포 {dict(sorted((v, nh.count(v)) for v in set(nh)))}; 종전 -2 {cz['prev2']} → "
          f"새 규칙 -2 {cz['new2']} (경유점 0 으로 재분류 {cz['to_wp0']}); 검토 2 규칙 -2 {cz['rev2_2']} 중 후보 의존(오귀속) "
          f"{cz['rev2_wrong']} [창 밖 이른 시점인데 값이 다름 {cz['early_dep']}, 참조 실패지만 값이 다름 {cz['ref_bad_diff']}]")


if __name__ == "__main__":
    main()
