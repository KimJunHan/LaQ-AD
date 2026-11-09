"""수정(2026-10-09): LaQ-AD NAVSIM — nuplan-devkit 이 필요한 변환 단계(별도 devkit env 에서 실행).

build_navsim_infos.py(hipad_bw) 산출 info pkl 에 다음을 덧붙인다. 하위명령:
  egometa : 자차(nuPlan Pacifica) 제원 + lidar2ego 로 config 에 넣을 custom ego 앵커/크기를 계산해 출력·저장.
  lidar   : OpenScene .pcd → (n,5) float32 .bin [x,y,z,intensity,ring] (HiP-AD LoadPointsFromFile load_dim=5 형식).
  map     : nuPlan 지도 API 로 벡터 지도 GT(map_annos) 생성.

devkit 근거(/home1/irteam/data-vol1/src/navsim_ref @0a380a9):
  - 지도 API: get_maps_api(NUPLAN_MAPS_ROOT, "nuplan-maps-v1.0", map_name) (navsim/common/dataclasses.py:16,441),
    map_name = 프레임 map_location(:494), get_proximal_map_objects(point, radius, layers)
    (navsim/agents/transfuser/transfuser_features.py:241-243), map_object.polygon(:247), baseline_path.linestring(:271),
    전역→국소 변환 = 같은 파일 _geometry_local_coords(:330-349, affine 변환).
  - 사용 레이어는 devkit 이 실제로 쓰는 것만: LANE/INTERSECTION 폴리곤(도로), WALKWAYS 폴리곤, LANE/LANE_CONNECTOR
    중심선(transfuser_config.py:84-89). CROSSWALK 등 devkit 에서 쓰이지 않는 레이어는 소스로 검증할 수 없어 쓰지 않는다.
  - 라이다 적재: LidarPointCloud.from_buffer(bytes, "pcd").points → (6,n) (x,y,z,intensity,ring,lidar_id)
    (dataclasses.py:18,104-109,132).
  - 차량 제원: nuplan.common.actor_state.vehicle_parameters.get_pacifica_parameters (navsim_scenario.py:8,42).

지도 클래스(결정 사항, 사용자 확인 필요): 0 lane_centerline / 1 road_boundary / 2 walkway.
  nuScenes 의 ped_crossing/divider/boundary 와 이름·의미가 다르다(nuPlan 에서 devkit 이 쓰는 레이어로 구성).
"""
import argparse
import os
import pickle

import numpy as np

MAP_CLASSES = ("lane_centerline", "road_boundary", "walkway")


def _load(p):
    d = pickle.load(open(p, "rb"))
    return d, d["infos"]


def _save(d, p):
    tmp = p + ".tmp"
    pickle.dump(d, open(tmp, "wb"))
    os.replace(tmp, p)


# ------------------------------------------------------------------ egometa
def cmd_egometa(args):
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    d, infos = _load(args.infos)
    vp = get_pacifica_parameters()
    l, w, h = float(vp.length), float(vp.width), float(vp.height)
    r2c = float(vp.rear_axle_to_center)
    l2es = np.stack([i["lidar2ego"] for i in infos])
    if np.abs(l2es - l2es[0]).max() > 1e-4:
        raise RuntimeError("프레임마다 lidar2ego 가 다르다 — 고정 자차 앵커 가정이 성립하지 않는다.")
    e2l = np.linalg.inv(l2es[0])
    # 차체 중심 = rear-axle 에서 전방 rear_axle_to_center, 높이 h/2 (ego 프레임: x 전방, z 위, 원점 rear-axle 지면 가정)
    # [미확정] ego 프레임 원점 높이가 지면인지는 데이터로 확인할 것(라이다 점의 지면 높이와 대조).
    c = e2l[:3, :3] @ np.array([r2c, 0.0, h / 2]) + e2l[:3, 3]
    fwd = e2l[:3, :3] @ np.array([1.0, 0.0, 0.0])
    yaw = float(np.arctan2(fwd[1], fwd[0]))
    anchor = [float(c[0]), float(c[1]), float(c[2]), float(np.log(l)), float(np.log(w)), float(np.log(h)),
              float(np.sin(yaw)), float(np.cos(yaw)), 0.0, 0.0, 0.0]
    d["metadata"]["ego_anchor"] = anchor
    d["metadata"]["ego_size"] = [l, w, h]
    d["metadata"]["lidar_forward_axis"] = fwd.tolist()
    _save(d, args.infos)
    print("config 에 그대로 넣을 값:")
    print(f"navsim_ego_anchor = {anchor}")
    print(f"navsim_ego_size = {[l, w, h]}")
    print(f"lidar 프레임에서 자차 전방 축 = {np.round(fwd, 4).tolist()}  (map_roi_size 축 순서 결정에 사용)")


# ------------------------------------------------------------------ lidar
def cmd_lidar(args):
    from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
    d, infos = _load(args.infos)
    done = 0
    for i in infos:
        src = i["lidar_path"]
        if src.endswith(".bin"):
            continue
        rel = os.path.relpath(src, args.sensor_root)
        dst = os.path.join(args.out_root, os.path.splitext(rel)[0] + ".bin")
        if not os.path.exists(dst):
            with open(src, "rb") as fp:
                pts = LidarPointCloud.from_buffer(fp, "pcd").points      # (6, n)
            if pts.shape[0] != 6:
                raise RuntimeError(f"{src}: 점 배열 형태 {pts.shape} — (6,n) 가정 위반(dataclasses.py:108)")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            pts[:5].T.astype(np.float32).tofile(dst + ".tmp")
            os.replace(dst + ".tmp", dst)
        i["lidar_path"] = dst
        done += 1
    _save(d, args.infos)
    print(f"lidar bin {done} -> {args.out_root}")


# ------------------------------------------------------------------ map
def _to_local(geom, x, y, heading):
    """transfuser_features._geometry_local_coords 와 같은 변환(전역 → 원점 (x,y,heading) 국소)."""
    from shapely import affinity
    a, b, dd, e = np.cos(heading), np.sin(heading), -np.sin(heading), np.cos(heading)
    g = affinity.affine_transform(geom, [1, 0, 0, 1, -x, -y])
    return affinity.affine_transform(g, [a, b, dd, e, 0, 0])


def _lines_of(geom):
    from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if isinstance(geom, Polygon):
        return [LineString(geom.exterior.coords)] + [LineString(r.coords) for r in geom.interiors]
    if isinstance(geom, MultiPolygon):
        return [l for p in geom.geoms for l in _lines_of(p)]
    if hasattr(geom, "geoms"):
        return [l for g in geom.geoms for l in _lines_of(g)]
    raise TypeError(type(geom))


def cmd_map(args):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from shapely.geometry import box as sbox
    from shapely.ops import unary_union
    from pyquaternion import Quaternion

    d, infos = _load(args.infos)
    roots = os.environ.get("NUPLAN_MAPS_ROOT")
    if not roots:
        raise RuntimeError("NUPLAN_MAPS_ROOT 환경변수가 필요하다(navsim/common/dataclasses.py:30).")
    roi_x, roi_y = args.roi            # 라이다 프레임 x, y 방향 전체 길이(m)
    clip = sbox(-roi_x / 2, -roi_y / 2, roi_x / 2, roi_y / 2)
    radius = float(np.hypot(roi_x, roi_y))  # 라이다가 rear-axle 에서 떨어져 있어도 ROI 를 덮도록 대각선 길이
    apis = {}
    for i in infos:
        name = i["map_location"]
        if name not in apis:
            apis[name] = get_maps_api(roots, "nuplan-maps-v1.0", name)
        api = apis[name]
        e2g = i["ego2global"]
        x, y = float(e2g[0, 3]), float(e2g[1, 3])
        heading = float(Quaternion(matrix=e2g[:3, :3]).yaw_pitch_roll[0])
        # transfuser_features.py:241 와 같이 StateSE2(...).point 로 질의(dataclasses.py:13 의 import 경로)
        from nuplan.common.actor_state.state_representation import StateSE2
        objs = api.get_proximal_map_objects(StateSE2(x, y, heading).point, radius,
                                            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR,
                                             SemanticMapLayer.INTERSECTION, SemanticMapLayer.WALKWAYS])
        e2l = np.linalg.inv(i["lidar2ego"])
        M = [e2l[0, 0], e2l[0, 1], e2l[1, 0], e2l[1, 1], e2l[0, 3], e2l[1, 3]]  # ego→lidar (xy 평면)
        from shapely import affinity

        def to_lidar(g):
            return affinity.affine_transform(_to_local(g, x, y, heading), M)

        center = [to_lidar(o.baseline_path.linestring)
                  for o in objs[SemanticMapLayer.LANE] + objs[SemanticMapLayer.LANE_CONNECTOR]]
        road = unary_union([to_lidar(o.polygon) for o in objs[SemanticMapLayer.LANE] + objs[SemanticMapLayer.INTERSECTION]])
        walk = unary_union([to_lidar(o.polygon) for o in objs[SemanticMapLayer.WALKWAYS]]) \
            if objs[SemanticMapLayer.WALKWAYS] else None
        annos = {0: [], 1: [], 2: []}
        for lab, geoms in ((0, center), (1, [road.boundary] if not road.is_empty else []),
                           (2, [walk.boundary] if walk is not None and not walk.is_empty else [])):
            for g in geoms:
                for ln in _lines_of(g.intersection(clip)):
                    if ln.length > 0:
                        annos[lab].append(np.asarray(ln.coords, np.float32)[:, :2])
        i["map_annos"] = annos
    d["metadata"]["map_classes"] = MAP_CLASSES
    d["metadata"]["map_roi"] = [roi_x, roi_y]
    _save(d, args.infos)
    print(f"map_annos -> {len(infos)} frames, classes {MAP_CLASSES}, roi {args.roi}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("egometa"); a.add_argument("--infos", required=True)
    b = sub.add_parser("lidar"); b.add_argument("--infos", required=True)
    b.add_argument("--sensor-root", required=True); b.add_argument("--out-root", required=True)
    c = sub.add_parser("map"); c.add_argument("--infos", required=True)
    c.add_argument("--roi", type=float, nargs=2, required=True,
                   help="라이다 프레임 x, y 방향 ROI 전체 길이(m). egometa 의 전방 축을 보고 정한다(nuScenes 는 30 60)")
    args = ap.parse_args()
    {"egometa": cmd_egometa, "lidar": cmd_lidar, "map": cmd_map}[args.cmd](args)


if __name__ == "__main__":
    main()
