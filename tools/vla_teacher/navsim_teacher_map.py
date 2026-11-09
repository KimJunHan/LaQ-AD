# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
"""교사 GT(v2-sensor) 의 지도 줄·차로변경 판정에 필요한 nuPlan 지도 정보를 프레임별로 뽑는다 (env: navsim, nuplan-devkit).

nuScenes 교사 GT 는 info 의 map_annos {0: 횡단보도, 1: 구분선(divider), 2: 도로 경계(boundary)} (lidar 프레임,
ROI x ±15 m · y ±30 m, +x 오른쪽 · +y 전방) 와 nuScenes 지도 API(lane_v2) 를 쓴다. NAVSIM 은 nuPlan 지도에서 같은 뜻의
선을 만든다(학생 지도 헤드용 navsim_devkit_steps.py map 과는 별개 — 그쪽은 lane_centerline/road_boundary/walkway):

  0 ped_crossing : CROSSWALK 폴리곤 외곽(닫힌 폴리라인 → common.map_context 가 Polygon 으로 다룸)
  1 divider      : LANE(차로, 교차로 연결부 제외)의 좌·우 경계선(BOUNDARIES, id 로 중복 제거) 중 도로 경계와
                   겹치지 않는 선(경계선 점들의 도로 경계까지 거리 중앙값 > 0.3 m). nuScenes 의 lane/road divider 와 같은 뜻.
  2 boundary     : ROADBLOCK ∪ ROADBLOCK_CONNECTOR 폴리곤 합집합의 외곽(주행 가능 영역 경계; 주차장 CARPARK 제외 —
                   nuScenes 경계도 road_segment·lane 만).
좌표: 교사 프레임 = nuScenes lidar 규약(+x 오른쪽, +y 전방). NAVSIM ego(=lidar, lidar2ego 단위행렬) 는 +x 전방 · +y 왼쪽이므로
  x_t = −y_e, y_t = x_e. 원점 = rear-axle(NAVSIM ego 원점). ROI 로 자른다(x ±15, y ±30).

차로변경(위상) 신호: 현재 위치의 LANE(진행방향과 30° 이내) L0 와 미래 8 포즈(4 s) 각각의 LANE 을 찾아, 같은 roadblock 안의 다른
  차로로 옮겨 갔으면 L0 기준 왼쪽/오른쪽(is_left_of/is_right_of)으로 'lane change left/right'. 교차로 연결부(LANE_CONNECTOR)
  안에서는 판정하지 않는다(회전으로 다룸). 결과: dict(decision='left'|'right'|None, L0=id|None, n_lane_pts).
  — 라벨(정답) 쪽 계산이므로 미래 포즈 사용 허용. 입력(프롬프트)에는 쓰지 않는다.

실행:
  CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX /home1/irteam/data-vol1/anaconda3/envs/navsim/bin/python -I \
    tools/vla_teacher/navsim_teacher_map.py --infos <info pkl> --out <out pkl> --workers 8 [--max-logs N]
"""
import argparse
import collections
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

MAPS_ROOT = "/home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/maps"
MAP_VERSION = "nuplan-maps-v1.0"
ROI_X, ROI_Y = 30.0, 60.0          # nuScenes 컨버터 roi_size=(30,60) 과 같음(교사 프레임 x, y 전체 길이)
RADIUS = float(np.hypot(ROI_X / 2, ROI_Y / 2)) + 2.0
DIV_ON_BOUNDARY_M = 0.3
ROAD_CLOSE_M = 0.25
LANE_HEAD_TOL_DEG = 30.0

_APIS = {}


def api_of(name):
    if name not in _APIS:
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
        _APIS[name] = get_maps_api(MAPS_ROOT, MAP_VERSION, name)
    return _APIS[name]


def g2t_matrix(e2g):
    """전역 → 교사 프레임(2D affine 6 계수, shapely 형식).
    yaw 와 x, y 만 쓴다(2D). 4x4 역행렬의 상단 2x2 를 쓰면 피치·롤과 큰 전역 z(수백 m)가 섞여 수 m 어긋난다(실측: 자차 원점이
    도로 폴리곤 밖으로 29% — 이 방식으로 고침)."""
    e2g = np.asarray(e2g, np.float64)
    yaw = float(np.arctan2(e2g[1, 0], e2g[0, 0]))
    c, s = np.cos(yaw), np.sin(yaw)
    Rg2e = np.array([[c, s], [-s, c]])
    R = np.array([[0.0, -1.0], [1.0, 0.0]])          # ego(x 전방, y 왼쪽) → 교사(x 오른쪽, y 전방)
    A = R @ Rg2e
    b = -A @ e2g[:2, 3]
    return [A[0, 0], A[0, 1], A[1, 0], A[1, 1], b[0], b[1]]


def _lines(geom):
    from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, (MultiLineString,)):
        return list(geom.geoms)
    if isinstance(geom, Polygon):
        return [LineString(geom.exterior.coords)] + [LineString(r.coords) for r in geom.interiors]
    if isinstance(geom, MultiPolygon) or hasattr(geom, "geoms"):
        return [l for g in geom.geoms for l in _lines(g)]
    return []


def frame_map(info):
    from shapely import affinity
    from shapely.geometry import box as sbox, Polygon
    from shapely.ops import unary_union
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from nuplan.common.actor_state.state_representation import Point2D
    api = api_of(info["map_location"])
    e2g = np.asarray(info["ego2global"], np.float64)
    M = g2t_matrix(e2g)
    T = lambda g: affinity.affine_transform(g, M)  # noqa: E731
    clip = sbox(-ROI_X / 2, -ROI_Y / 2, ROI_X / 2, ROI_Y / 2)
    objs = api.get_proximal_map_objects(Point2D(e2g[0, 3], e2g[1, 3]), RADIUS,
                                        [L.LANE, L.CROSSWALK, L.ROADBLOCK, L.ROADBLOCK_CONNECTOR])
    # 이웃 roadblock 사이의 미세 틈(경계선이 도로 안을 가로지르는 가짜 경계)을 닫는다: +0.25 m 팽창 → 합집합 → −0.25 m
    #   (실측 20 로그: 회랑 안 경계 거리 중앙값 12.3 → 19.9 m, 1 m 안 0.017 → 0)
    road = unary_union([T(o.polygon).buffer(ROAD_CLOSE_M) for o in objs[L.ROADBLOCK] + objs[L.ROADBLOCK_CONNECTOR]])
    road = road.buffer(-ROAD_CLOSE_M) if not road.is_empty else road
    road_b = road.boundary if not road.is_empty else None
    annos = {0: [], 1: [], 2: []}
    for o in objs[L.CROSSWALK]:
        g = T(o.polygon).intersection(clip)
        for p in ([g] if g.geom_type == "Polygon" else list(getattr(g, "geoms", []))):
            if p.geom_type == "Polygon" and not p.is_empty and p.area > 0:
                annos[0].append(np.asarray(p.exterior.coords, np.float32))
    seen = set()
    for o in objs[L.LANE]:
        for b in (o.left_boundary, o.right_boundary):
            if b.id in seen:
                continue
            seen.add(b.id)
            ls = T(b.linestring)
            if road_b is not None:
                pts = np.asarray(ls.coords)
                from shapely.geometry import Point
                dmed = float(np.median([road_b.distance(Point(p)) for p in pts]))
                if dmed <= DIV_ON_BOUNDARY_M:
                    continue
            for ln in _lines(ls.intersection(clip)):
                if ln.length > 0:
                    annos[1].append(np.asarray(ln.coords, np.float32)[:, :2])
    if road_b is not None:
        for ln in _lines(road_b.intersection(clip)):
            if ln.length > 0:
                annos[2].append(np.asarray(ln.coords, np.float32)[:, :2])
    return annos


def lane_change_signal(info):
    """라벨용 위상 신호. 미래 8 포즈(ego 증분 → 전역)."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    from nuplan.common.actor_state.state_representation import Point2D
    m = np.asarray(info["gt_ego_fut_masks"])
    if not (m == 1).all():
        return dict(decision=None, L0=None, n_lane_pts=0, reason="future_incomplete")
    api = api_of(info["map_location"])
    e2g = np.asarray(info["ego2global"], np.float64)
    loc = np.concatenate([np.zeros((1, 2)), np.asarray(info["gt_ego_fut_trajs"], np.float64).cumsum(0)], 0)  # ego(=lidar)
    yaw0 = float(np.arctan2(e2g[1, 0], e2g[0, 0]))
    R2 = np.array([[np.cos(yaw0), -np.sin(yaw0)], [np.sin(yaw0), np.cos(yaw0)]])   # yaw 만(2D)
    glob = loc @ R2.T + e2g[:2, 3]
    heads = [yaw0]
    for k in range(1, len(glob)):
        d = glob[k] - glob[k - 1]
        heads.append(float(np.arctan2(d[1], d[0])) if np.hypot(*d) > 0.2 else heads[-1])

    def lane_at(p, h):
        pt = Point2D(float(p[0]), float(p[1]))
        best = None
        for ln in api.get_all_map_objects(pt, L.LANE):     # 차로 우선(연결부 폴리곤이 차로 끝과 겹치는 경우가 있다)
            st = ln.baseline_path.get_nearest_pose_from_position(pt)
            dh = abs((st.heading - h + np.pi) % (2 * np.pi) - np.pi)
            if np.degrees(dh) <= LANE_HEAD_TOL_DEG and (best is None or dh < best[0]):
                best = (dh, ln)
        if best is not None:
            return best[1]
        return "connector" if api.get_all_map_objects(pt, L.LANE_CONNECTOR) else None
    L0 = lane_at(glob[0], heads[0])
    if L0 is None or L0 == "connector":
        return dict(decision=None, L0=None, n_lane_pts=0, reason="start_not_in_lane" if L0 is None else "start_in_connector")
    dec, n = None, 0
    for k in range(1, len(glob)):
        lk = lane_at(glob[k], heads[k])
        if lk is None or lk == "connector":
            continue
        n += 1
        if lk.id != L0.id and lk.get_roadblock_id() == L0.get_roadblock_id():
            dec = "left" if lk.is_left_of(L0) else "right" if lk.is_right_of(L0) else dec
    return dict(decision=dec, L0=str(L0.id), n_lane_pts=n, reason="ok")


def work(args):
    log, infos = args
    out = {}
    t0 = time.time()
    for i in infos:
        out[i["token"]] = dict(map_annos=frame_map(i), lane_change=lane_change_signal(i))
    return log, out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-logs", type=int, default=0)
    ap.add_argument("--max-frames-per-log", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    d = pickle.load(open(a.infos, "rb"))
    by = collections.OrderedDict()
    seen = set()
    for i in d["infos"]:
        if i["token"] in seen:          # navtest pkl 은 장면 창이 겹쳐 같은 프레임이 여러 번 있다
            continue
        seen.add(i["token"])
        by.setdefault(i["log_name"], []).append(i)
    jobs = list(by.items())
    if a.max_logs:
        jobs = jobs[:a.max_logs]
    if a.max_frames_per_log:
        jobs = [(k, v[:a.max_frames_per_log]) for k, v in jobs]
    # 같은 도시 로그를 같은 워커로 몰지 않는다(지도 적재 비용은 워커·도시당 1회)
    res, cpu = {}, 0.0
    with Pool(a.workers) as pool:
        for k, (log, out, s) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            res.update(out); cpu += s
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(jobs)} logs, frames {len(res)}, {time.time() - t0:.0f}s", flush=True)
    meta = dict(classes={0: "ped_crossing", 1: "divider", 2: "boundary"}, roi=[ROI_X, ROI_Y],
                frame="teacher(+x right, +y forward, origin rear-axle)", maps_root=MAPS_ROOT,
                div_on_boundary_m=DIV_ON_BOUNDARY_M, road_close_m=ROAD_CLOSE_M, lane_head_tol_deg=LANE_HEAD_TOL_DEG,
                source_infos=a.infos, partial=bool(a.max_logs or a.max_frames_per_log),
                n_frames=len(res), wall_s=round(time.time() - t0, 1), worker_cpu_s=round(cpu, 1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "wb") as f:
        pickle.dump(dict(frames=res, metadata=meta), f, protocol=4)
    os.replace(a.out + ".tmp", a.out)
    print(meta)


if __name__ == "__main__":
    main()
