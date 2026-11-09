# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
"""B2D 2단계 지도 정렬 확산(MapAlignedPlanTruncatedDiffusion) 설정값의 근거 통계. CPU 전용, 파일은 JSON 하나만 쓴다.

  CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=. python tools/b2d_plan_diffusion_stats.py \
      [--out results/b2d_prep/b2d_plan_diffusion_stats.json]

(1) 정규화 범위(norm_scale/norm_offset) — nuScenes 와 같은 DiffusionDrive 규칙: (x-offset)/scale 이 train 분할 GT 누적 궤적의
    [min,max] 를 [-1,1] 에 사상. B2D 2단계가 실제로 쓰는 계획 목표 4 종(temp 2hz, temp 5hz, spat 2m, spat 5m) 전부,
    train 234,769 프레임 전부(부분표본 아님). Bench2DriveDataset(split_group=5)의 get_ego_temporal_trajs /
    get_ego_spatial_trajs(with_fitting=False) 의미를 그대로 재현한다:
      * split_group=5 재배열 목록과 "실제 다음 프레임" 사상(group_cumsum 규칙)을 문자 그대로 구현하고,
        그 사상이 원래 순서의 +1 과 같은지 전수 확인한다.
      * temporal: 재배열 색인 r-2(파이썬 음수 색인 포함)의 folder 가 같아야 채움, folder 바뀌면 중단, 마스크 = 채운 칸.
      * spatial: 같은 folder 의 뒤 프레임 전부, 균일 거리 표집(adj_diff < 0.25·간격), 오프셋 < 0.1 m 이면 마스크 0.
    누적 점: temporal = 마스크 1 인 위치(=오프셋 누적; 마스크는 앞부분 연속), spatial = 마스크 1 인 표집 위치(라이다 좌표 m).
(2) 차로 폭: 지도 Center 선(차로 중심선)에서 같은 도로(road_id)·같은 진행 부호·lane_id 가 1 차이인 이웃 차로 쌍의 중심선 간격
    (A 의 점 → B 의 최근접 점 거리, 쌍별 중앙값) 의 중앙값.
(3) GT 계획 경로(2hz, 6 칸 모두 유효)가 표시선 클래스(Broken/Solid/SolidSolid/Center)를 가로지르는 프레임 비율(train 전부).
    표시선 = 데이터셋 get_map_info(with_connect_lane=True)가 GT 로 넣는 것과 같은 집합(Center 는 전부, 다른 종류는 lane_topos 가 있는 것)
    — 연결은 같은 선분의 이어 붙임이라 교차 여부는 같다. 교차점이 라이다 ROI(|x|<15, |y|<30) 안에 있을 때만 센다(모델이 보는 지도).
"""
import argparse
import collections
import json
import math
import os
import pickle
import time

import numpy as np

TRAIN = "data/infos/b2d_infos_train.pkl"
MAPF = "data/infos/b2d_map_infos.pkl"
SPLIT_GROUP = 5
FUT = 6
SPAT_PTS = 6
MAP_CLASSES = ["Broken", "Solid", "SolidSolid", "Center"]


def reorganize(n, g=SPLIT_GROUP):
    orig = np.concatenate([np.arange(i, n, g) for i in range(g)])
    length = np.array([len(range(i, n, g)) for i in range(g)])
    cums = np.array([0] + list(length[:-1])).cumsum()
    return orig, cums


def next_reorg(r, cums, n, g=SPLIT_GROUP):
    """Bench2DriveDataset.get_ego_temporal_trajs 의 '실제 다음 프레임' 사상(문자 그대로)."""
    diffs = r - cums
    diffs = np.where(diffs < 0, n, diffs)
    gi = int(np.argmin(diffs)); gd = int(diffs[gi])
    if gi >= g - 1:
        gi = 0; gd += 1
    else:
        gi += 1
    return int(cums[gi] + gd)


def mm(a):
    a = np.asarray(a)
    return dict(n=int(len(a)), min=[float(a[:, 0].min()), float(a[:, 1].min())], max=[float(a[:, 0].max()), float(a[:, 1].max())],
                p0_1=np.percentile(a, 0.1, 0).tolist(), p99_9=np.percentile(a, 99.9, 0).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/b2d_prep/b2d_plan_diffusion_stats.json")
    ap.add_argument("--skip_map", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    infos = pickle.load(open(TRAIN, "rb"))
    n = len(infos)
    folder = np.array([i["folder"] for i in infos])
    town = np.array([i["town_name"] for i in infos])
    W = np.stack([np.asarray(i["sensors"]["LIDAR_TOP"]["world2lidar"], np.float64) for i in infos])
    del infos
    Linv_t = np.linalg.inv(W)[:, :3, 3]          # 라이다 원점의 세계 좌표
    print(f"[load] {n} 프레임 {time.time() - t0:.0f}s", flush=True)

    # 재배열(split_group=5)과 다음 프레임 사상
    orig, cums = reorganize(n)
    pos_of = np.empty(n, np.int64); pos_of[orig] = np.arange(n)
    nxt = np.array([next_reorg(r, cums, n) for r in range(n)])
    # 사상 = 원래 순서 +1 인가(범위 안인 것)
    ok_map = sum(1 for r in range(n) if nxt[r] < n and orig[nxt[r]] == orig[r] + 1)
    in_range = int((nxt < n).sum())
    print(f"[map] next=orig+1 {ok_map}/{in_range} (범위 밖 {n - in_range})", flush=True)

    def xy(cur, adj):  # 둘 다 원래 색인
        return W[cur, :2, :3] @ Linv_t[adj] + W[cur, :2, 3]

    pts = {k: [] for k in ("temp_2hz", "temp_5hz", "spat_2m", "spat_5m")}
    fut2 = {}   # 원래 색인 → (6,2) 2hz 위치(6 칸 모두 유효할 때) — (3) 교차 진단용
    for r in range(n):
        cur = orig[r]
        # ---- temporal (2hz interval 5, 5hz interval 2) ----
        chain = [r]
        adj = r
        for _ in range(FUT * 5):
            adj = nxt[adj] if adj < n else adj + 1   # 범위 밖이면 이후는 모두 무효(아래 range 검사)
            chain.append(adj)
        past_ok = folder[orig[r - 2]] == folder[cur]  # 재배열 색인 r-2 (음수면 파이썬 끝에서부터 — 원본 동작)
        for name, itv in (("temp_2hz", 5), ("temp_5hz", 2)):
            lst = chain[: FUT * itv + 1][::itv]
            if not past_ok:
                continue
            got = []
            for j, aj in enumerate(lst):
                if not (0 <= aj < n):
                    continue  # 원본: 범위 밖이면 그 칸만 건너뜀(이후도 범위 밖)
                o = orig[aj]
                if folder[o] != folder[cur]:
                    break
                if j > 0:
                    got.append(xy(cur, o))
            if got:
                pts[name] += got
                if name == "temp_2hz" and len(got) == FUT:
                    fut2[cur] = np.asarray(got)
        # ---- spatial (uniform 2 m, 5 m) ----
        adjp = []
        aj = r
        while True:
            aj = nxt[aj] if aj < n else n
            if 0 <= aj < n and folder[orig[aj]] == folder[cur]:
                adjp.append(orig[aj])
            else:
                break
        if not adjp:
            continue
        P = W[cur, :2, :3] @ Linv_t[np.asarray(adjp)].T + W[cur, :2, 3:4]
        P = P.T
        dist = np.linalg.norm(P, axis=1)
        for name, d in (("spat_2m", 2.0), ("spat_5m", 5.0)):
            sds = [i * d for i in range(1, SPAT_PTS + 1)]
            mask = np.zeros(SPAT_PTS); tr = np.full((SPAT_PTS, 2), -1.0)
            ai = -1
            for si, sd in enumerate(sds):
                pre = sd if si == 0 else sd - sds[si - 1]
                ad = np.abs(sd - dist)
                mi = int(np.argmin(ad))
                if mi > ai and ad[mi] < pre * 0.25:
                    tr[si] = P[mi]; mask[si] = 1; ai = mi
            for i in range(SPAT_PTS):
                if mask[i]:
                    off = tr[i] if i == 0 else tr[i] - tr[i - 1]
                    if np.linalg.norm(off) < 0.1:
                        mask[i] = 0
            pts[name] += [tr[i] for i in range(SPAT_PTS) if mask[i]]
        if r % 20000 == 0:
            print(f"[traj] {r}/{n} {time.time() - t0:.0f}s", flush=True)

    out = dict(created="2026-10-19", source=TRAIN, n_frames=n, split_group=SPLIT_GROUP,
               next_frame_mapping=dict(equals_orig_plus_1=ok_map, in_range=in_range),
               ranges={k: mm(v) for k, v in pts.items()})
    allp = np.concatenate([np.asarray(v) for v in pts.values()])
    out["ranges"]["union"] = mm(allp)
    out["tails"] = {k: dict(n_y_lt_m2=int((np.asarray(v)[:, 1] < -2).sum()), n_abs_x_gt_20=int((np.abs(np.asarray(v)[:, 0]) > 20).sum()),
                            n=len(v)) for k, v in pts.items()}
    lo, hi = allp.min(0), allp.max(0)
    mid = (lo + hi) / 2
    off = np.round(mid) + 0.0   # -0.0 → 0.0
    scale = np.ceil(np.maximum(hi - off, off - lo))
    out["norm"] = dict(rule="offset = round((min+max)/2), scale = ceil(max(max-offset, offset-min)) → [min,max] ⊂ [-1,1] (nuScenes 와 같은 규칙: 정수 반올림)",
                       norm_offset=off.tolist(), norm_scale=scale.tolist())
    # DiffusionDrive 일정: β 선형 1e-4→0.02 / 1000, 학습 t ≤ 49
    betas = np.linspace(1e-4, 0.02, 1000)
    ab = np.cumprod(1 - betas)
    s49 = math.sqrt(1 - ab[49])
    out["schedule"] = dict(sqrt_1m_ab_49=s49, sqrt_1m_ab_8=math.sqrt(1 - ab[8]))
    print(json.dumps(out["ranges"]["union"]), out["norm"], flush=True)

    if not a.skip_map:
        from scipy.spatial import cKDTree
        from shapely.geometry import LineString, box as sbox
        from shapely.strtree import STRtree
        t1 = time.time()
        M = pickle.load(open(MAPF, "rb"))
        print(f"[map] 로드 {time.time() - t1:.0f}s", flush=True)
        # ---- (2) 차로 폭 ----
        pair_med, per_town = [], {}
        for tn, mi in M.items():
            cen = [(tuple(mi["lane_ids"][k]), np.asarray(mi["lane_points"][k], np.float64)[:, :2])
                   for k in range(len(mi["lane_types"])) if mi["lane_types"][k] == "Center"]
            by = collections.defaultdict(list)
            for lid, p in cen:
                if len(p) >= 2:
                    by[lid].append(p)
            tm = []
            for (rd, ln), plist in by.items():
                for dl in (1,):
                    nb = (rd, ln + dl if ln > 0 else ln - dl)
                    if nb not in by or ln == 0:
                        continue
                    A = np.concatenate(plist); B = np.concatenate(by[nb])
                    d, _ = cKDTree(B).query(A)
                    tm.append(float(np.median(d)))
            per_town[tn] = dict(n_pairs=len(tm), median=float(np.median(tm)) if tm else None)
            pair_med += tm
        pm = np.asarray(pair_med)
        pm_f = pm[(pm > 1.5) & (pm < 8.0)]
        out["lane_width"] = dict(rule="같은 road_id·같은 부호·|Δlane_id|=1 인 Center 선 쌍의 점간 최근접거리 중앙값(쌍별) → 쌍 중앙값",
                                 n_pairs=int(len(pm)), median_all=float(np.median(pm)), n_pairs_1_5_to_8m=int(len(pm_f)),
                                 median_1_5_to_8m=float(np.median(pm_f)), p25=float(np.percentile(pm_f, 25)),
                                 p75=float(np.percentile(pm_f, 75)), per_town=per_town)
        print("[lane]", out["lane_width"]["median_1_5_to_8m"], flush=True)
        # ---- (3) 교차 ----
        trees = {}
        for tn, mi in M.items():
            geoms, cls = [], []
            for k, ty in enumerate(mi["lane_types"]):
                if ty not in MAP_CLASSES:
                    continue
                if ty != "Center" and len(mi["lane_topos"][k]) == 0:
                    continue   # get_map_info(with_connect_lane=True) 는 lane_topos 가 빈 표시선을 GT 에 넣지 않는다
                p = np.asarray(mi["lane_points"][k], np.float64)[:, :2]
                if len(p) < 2:
                    continue
                geoms.append(LineString(p)); cls.append(ty)
            trees[tn] = (STRtree(geoms), {id(g): i for i, g in enumerate(geoms)}, cls)   # shapely 1.8: query 가 도형을 돌려준다
        del M
        roi = sbox(-15.0, -30.0, 15.0, 30.0)
        cnt = collections.Counter(); n_eval = 0
        ex = collections.defaultdict(list)
        for k, (cur, f2) in enumerate(fut2.items()):
            tree, gid, cls = trees[town[cur]]
            # 경로(라이다) → 세계: 경로 점 = 해당 프레임 라이다 원점(위에서 그 위치를 라이다로 옮긴 것) — 세계 좌표로 다시 계산
            Winv = np.linalg.inv(W[cur])
            pw = (Winv[:2, :2] @ np.vstack([np.zeros((1, 2)), f2]).T + Winv[:2, 3:4]).T
            path = LineString(pw)
            hit = set()
            for g in tree.query(path):
                gi = gid[id(g)]
                inter = path.intersection(g)
                if inter.is_empty:
                    continue
                ip = np.asarray([c for geom in getattr(inter, "geoms", [inter]) for c in geom.coords], np.float64)[:, :2]
                ipl = (W[cur, :2, :2] @ ip.T + W[cur, :2, 3:4]).T
                if any(roi.contains(__import__("shapely.geometry", fromlist=["Point"]).Point(*q)) for q in ipl):
                    hit.add(cls[gi])
            n_eval += 1
            for c in hit:
                cnt[c] += 1
                if len(ex[c]) < 5:
                    ex[c].append(folder[cur])
            if k % 20000 == 0:
                print(f"[cross] {k}/{len(fut2)} {time.time() - t0:.0f}s", flush=True)
        out["gt_path_crossing"] = dict(
            rule="train 프레임 중 2hz 미래 6 칸 모두 유효한 것 전부. 경로 = 원점 + 6 점 꺾은선, 교차점이 ROI 안일 때만",
            n_frames=n_eval, frames_crossing={c: cnt[c] for c in MAP_CLASSES},
            fraction={c: round(cnt[c] / max(n_eval, 1), 5) for c in MAP_CLASSES},
            examples={c: sorted(set(v)) for c, v in ex.items()})
        print(json.dumps(out["gt_path_crossing"]["fraction"]), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"[done] {a.out} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
