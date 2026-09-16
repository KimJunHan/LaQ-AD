#수정: VLA-v0 / nuScenes-mini — Bench2Drive 의 visualize.py 패턴을 nuScenes 용으로 포팅.
# 입력 :
#   --config:  학습/평가에 쓴 config
#   --result:  dist_test --out 으로 덤프된 pkl (per-sample {'img_bbox':{'boxes_3d', 'scores_3d', 'labels_3d'}})
#   --out-dir: 출력 디렉토리. <out>/combine/0000.jpg ... 와 <out>/video.mp4 생성.
# 출력 형식 (가로 concat):
#   [6-cam 가로 stitch (pred 3D box overlay)] | [pred BEV] | [GT BEV]
# 한 sample 당 한 장. 마지막에 video.mp4 (12 fps) 생성.
import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mmdet3d_plugin  # registries
from mmcv import Config
from mmcv.utils import build_from_cfg
from mmdet.datasets import DATASETS
from pyquaternion import Quaternion

# ----------------------------------------------------------------------------
# nuScenes 10-class color (BGR) — devkit 표준에 가까운 색
NUSC_COLORS_BGR = {
    "car":                  (255, 158,   0),
    "truck":                (255, 158,   0),
    "construction_vehicle": (255,  61,  99),
    "bus":                  (255, 127,  80),
    "trailer":              (255, 158,   0),
    "barrier":              (  0,   0,   0),
    "motorcycle":           (255,  61,  99),
    "bicycle":              (220,  20,  60),
    "pedestrian":           (  0,   0, 230),
    "traffic_cone":         (245, 230, 100),
}


def get_box_corners_3d(center, size, yaw):
    """center=(x,y,z), size=(w,l,h)=(dx,dy,dz), yaw (z 축 회전) → (8,3) corners.
    nuScenes 'size' 는 (w, l, h) = (x_extent, y_extent, z_extent)."""
    w, l, h = float(size[0]), float(size[1]), float(size[2])
    # local frame: (±w/2, ±l/2, ±h/2)
    x = np.array([w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2])
    y = np.array([l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2])
    z = np.array([-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2])
    pts = np.stack([x, y, z], axis=1)  # (8, 3)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pts = pts @ R.T
    pts += np.asarray(center)
    return pts


def project_to_image(pts3d, lidar2img):
    """(N, 3) lidar → (N, 2) image pixels. 카메라 뒤(z<=0) 는 None 반환."""
    homo = np.concatenate([pts3d, np.ones((pts3d.shape[0], 1))], axis=1)  # (N,4)
    cam = homo @ lidar2img.T          # (N, 4) homo image
    z = cam[:, 2]
    if (z <= 0.1).any():
        return None, None
    u = cam[:, 0] / z
    v = cam[:, 1] / z
    return np.stack([u, v], axis=1), z


def draw_box_2d_on_img(img, corners_2d, color, thickness=2):
    """8 corners 의 2D 좌표로 큐브 12 엣지 그리기."""
    if corners_2d is None:
        return
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
    ]
    H, W = img.shape[:2]
    pts = corners_2d.astype(int)
    #수정: 화면 밖 엣지를 skip 하던 것이 박스를 끊기게 함 → 모든 엣지 그림(cv2.line 이 자동 clip).
    #  단 투영 발산(좌표 폭주)만 가드. LINE_AA 로 매끈한 연결 실선.
    for i, j in edges:
        p1, p2 = pts[i], pts[j]
        if max(abs(int(p1[0])), abs(int(p1[1])), abs(int(p2[0])), abs(int(p2[1]))) > 20000:
            continue
        cv2.line(img, tuple(p1), tuple(p2), color, thickness, cv2.LINE_AA)


#수정: 카메라 3D box 를 "끊김 없는 연결 실선" 으로 — edge 별 near-plane(z>0.1) 클리핑 후 그림.
#  (기존 project_to_image 는 한 corner라도 뒤에 있으면 박스 전체 skip → 끊김/누락. 여기선 edge 단위로
#   near-plane 에서 잘라 항상 연결되게 그린다. cv2.line 이 화면 경계는 자동 clip.)
def draw_box3d_on_img(img, corners_3d, lidar2img, color, thickness=2, near=0.1):
    H, W = img.shape[:2]
    homo = np.concatenate([corners_3d, np.ones((corners_3d.shape[0], 1))], axis=1)
    cam = homo @ lidar2img.T          # (8,4): [u*z, v*z, z, *]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    def _uv(p):
        z = max(p[2], 1e-3)
        u = float(np.clip(p[0] / z, -1e5, 1e5))
        v = float(np.clip(p[1] / z, -1e5, 1e5))
        return int(u), int(v)

    drew = False
    for i, j in edges:
        a, b = cam[i, :3].copy(), cam[j, :3].copy()
        z1, z2 = a[2], b[2]
        if z1 < near and z2 < near:
            continue                  # 둘 다 카메라 뒤 → skip
        if z1 < near:                 # a 가 뒤 → near-plane 으로 보간 클립
            t = (near - z2) / (z1 - z2)
            a = b + t * (a - b)
        elif z2 < near:
            t = (near - z1) / (z2 - z1)
            b = a + t * (b - a)
        cv2.line(img, _uv(a), _uv(b), color, thickness, cv2.LINE_AA)
        drew = True
    return drew


def draw_traj_on_img(img, traj_xy, lidar2img, color, z_ground=-1.6, thickness=3, add_origin=True):
    """ego future trajectory(lidar x,y)를 카메라 이미지에 ground 높이로 투영 → 연결선 + dot."""
    if traj_xy is None or len(traj_xy) < 1:
        return
    pts = [[float(p[0]), float(p[1]), z_ground] for p in traj_xy]
    if add_origin:
        pts = [[0.0, 0.0, z_ground]] + pts
    pts3d = np.array(pts)
    homo = np.concatenate([pts3d, np.ones((len(pts3d), 1))], axis=1)
    cam = homo @ lidar2img.T
    z = cam[:, 2]
    H, W = img.shape[:2]
    prev = None
    for k in range(len(pts3d)):
        if z[k] <= 0.1:
            prev = None
            continue
        u, v = int(cam[k, 0] / z[k]), int(cam[k, 1] / z[k])
        if prev is not None:
            cv2.line(img, prev, (u, v), color, thickness, cv2.LINE_AA)
        cv2.circle(img, (u, v), 4, color, -1, cv2.LINE_AA)
        prev = (u, v)


def build_scene_index(nusc):
    """token → ordered sample list (scene 내) 매핑."""
    token2scene = {s["token"]: s["scene_token"] for s in nusc.sample}
    scene2tokens = {}
    for scene in nusc.scene:
        toks = []
        st = scene["first_sample_token"]
        while st:
            toks.append(st)
            samp = nusc.get("sample", st)
            st = samp["next"]
        scene2tokens[scene["token"]] = toks
    return token2scene, scene2tokens


def get_future_ego_in_lidar(info, future_infos, n_future=6):
    """info 시점 lidar 좌표계 기준 미래 ego 위치 (N, 3). info 자신 제외."""
    import numpy as np
    from mmdet3d_plugin.datasets.nuscenes_mini_dataset import _quat_to_rot
    # current lidar → global
    lidar2ego_R = _quat_to_rot(info["lidar2ego_rotation"])
    lidar2ego_t = np.asarray(info["lidar2ego_translation"])
    ego2global_R = _quat_to_rot(info["ego2global_rotation"])
    ego2global_t = np.asarray(info["ego2global_translation"])
    global2ego = np.linalg.inv(_compose_4x4(ego2global_R, ego2global_t))
    ego2lidar = np.linalg.inv(_compose_4x4(lidar2ego_R, lidar2ego_t))

    pts_lidar = []
    for fi in future_infos[:n_future]:
        # future ego position in global = future ego2global_translation
        p_global = np.asarray(fi["ego2global_translation"])
        # → current ego frame
        p_ego_homo = global2ego @ np.array([p_global[0], p_global[1], p_global[2], 1.0])
        # → current lidar frame
        p_lidar_homo = ego2lidar @ p_ego_homo
        pts_lidar.append(p_lidar_homo[:3])
    return np.array(pts_lidar) if pts_lidar else np.zeros((0, 3))


def _compose_4x4(R, t):
    import numpy as np
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


#수정: vis 스펙 — nuScenes 전체 map class 를 vector map(polyline)으로 BEV 에 그림.
#  info["map_location"] + NuScenesMap API 로 ego 주변 layer 추출 → global→lidar→canvas 변환.
NUSC_MAP_LAYERS = [
    # (layer, geom, BGR color, thickness)
    ("drivable_area", "polygon", (215, 215, 215), 1),
    ("road_segment",  "polygon", (200, 205, 170), 1),
    ("lane",          "polygon", (195, 195, 195), 1),
    ("ped_crossing",  "polygon", (255, 150,  0),  2),   # blue
    ("walkway",       "polygon", (0, 175,   0),   1),   # green
    ("stop_line",     "polygon", (0,   0, 220),   2),   # red
    ("carpark_area",  "polygon", (180, 180,  0),  1),   # cyan-ish
    ("road_divider",  "line",    (0, 140, 255),   2),   # orange
    ("lane_divider",  "line",    (0, 230, 230),   2),   # yellow
]

_NUSC_MAP_CACHE = {}
def get_nusc_map(dataroot, location):
    if location not in _NUSC_MAP_CACHE:
        from nuscenes.map_expansion.map_api import NuScenesMap
        _NUSC_MAP_CACHE[location] = NuScenesMap(dataroot=dataroot, map_name=location)
    return _NUSC_MAP_CACHE[location]


def draw_vector_map(canvas, nusc_map, lidar2global, point_cloud_range, lidar_to_canvas, margin=10.0):
    """ego 주변 nuScenes map layer 를 global→lidar→canvas 변환해 polyline 으로 그림."""
    try:
        global2lidar = np.linalg.inv(lidar2global)
        ego_g = lidar2global[:3, 3]
        half = max(point_cloud_range[3] - point_cloud_range[0],
                   point_cloud_range[4] - point_cloud_range[1]) / 2 + margin
        box_coords = (ego_g[0] - half, ego_g[1] - half, ego_g[0] + half, ego_g[1] + half)
        avail = set(getattr(nusc_map, "non_geometric_layers", []))
        layers = [l for l in NUSC_MAP_LAYERS if l[0] in avail] or NUSC_MAP_LAYERS
        recs = nusc_map.get_records_in_patch(box_coords, [l[0] for l in layers], mode="intersect")
    except Exception:
        return

    def g2c(coords):
        pts = []
        for x, y in coords:
            p = global2lidar @ np.array([x, y, 0.0, 1.0])
            pts.append(lidar_to_canvas(p[0], p[1]))
        return np.array(pts, dtype=np.int32)

    #수정: vis 스펙 — 연결 실선 + 꼭짓점 dot(원). 점이 너무 촘촘하면 ~n_dot 으로 resample 해서 dot.
    def _line_with_dots(pts, closed, color, th, n_dot=12, r=2):
        if len(pts) < 2:
            return
        cv2.polylines(canvas, [pts], closed, color, th, cv2.LINE_AA)
        if len(pts) > n_dot:
            sel = pts[np.linspace(0, len(pts) - 1, n_dot).astype(int)]
        else:
            sel = pts
        for p in sel:
            cv2.circle(canvas, (int(p[0]), int(p[1])), r, color, -1, cv2.LINE_AA)

    for layer, geom, color, th in layers:
        for tok in recs.get(layer, []):
            try:
                rec = nusc_map.get(layer, tok)
                if geom == "line":
                    ls = nusc_map.extract_line(rec["line_token"])
                    if ls.is_empty:
                        continue
                    _line_with_dots(g2c(list(ls.coords)), False, color, th)
                else:
                    poly_toks = rec.get("polygon_tokens") or [rec.get("polygon_token")]
                    for ptk in poly_toks:
                        if ptk is None:
                            continue
                        poly = nusc_map.extract_polygon(ptk)
                        if poly.is_empty:
                            continue
                        _line_with_dots(g2c(list(poly.exterior.coords)), True, color, th)
            except Exception:
                continue


def draw_bev(boxes, labels, scores, classes, point_cloud_range, canvas_size=400, score_thresh=0.2,
             ego_traj_lidar=None, qwen_traj_lidar=None, hipad_traj_lidar=None,
             draw_velocity=True, color_override=None, nusc_map=None, lidar2global=None):
    """BEV (z 축 위에서 본 모습) 그림. 단위: meter → pixel 매핑."""
    canvas = np.full((canvas_size, canvas_size, 3), 240, dtype=np.uint8)  # 연한 회색 배경
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    span_x = x_max - x_min
    span_y = y_max - y_min
    # ego 중앙 (자차 사각형은 map 그린 뒤에 그림 → 아래 참조)
    cx, cy = canvas_size // 2, canvas_size // 2

    def lidar_to_canvas(x, y):
        # 시계방향 90도 회전 매핑:
        #   lidar +x (forward) → canvas 오른쪽 (큰 u)
        #   lidar +y (left)    → canvas 위쪽   (작은 v)
        u = int(canvas_size * (x - x_min) / span_x)
        v = int(canvas_size * (1 - (y - y_min) / span_y))
        return u, v

    #수정: vector map 을 박스 아래(배경)에 먼저 그림
    if nusc_map is not None and lidar2global is not None:
        draw_vector_map(canvas, nusc_map, lidar2global, point_cloud_range, lidar_to_canvas)

    #수정: vis 스펙 — 자차(ego)를 파란 사각형 차량으로 (map 위에). lidar +x=forward(=canvas 위쪽).
    ego_l, ego_w = 4.5, 1.9
    ego_corners = [(ego_l / 2, ego_w / 2), (ego_l / 2, -ego_w / 2),
                   (-ego_l / 2, -ego_w / 2), (-ego_l / 2, ego_w / 2)]
    ep = np.array([lidar_to_canvas(x, y) for x, y in ego_corners], dtype=np.int32)
    cv2.fillPoly(canvas, [ep], (255, 120, 0))                 # 파란 채움 (BGR)
    cv2.polylines(canvas, [ep], True, (255, 0, 0), 2)         # 파란 외곽선
    cv2.line(canvas, (cx, cy), lidar_to_canvas(ego_l / 2, 0), (255, 0, 0), 2)  # heading

    for box, lab, sc in zip(boxes, labels, scores):
        if sc < score_thresh:
            continue
        if not (0 <= int(lab) < len(classes)):
            continue
        cname = classes[int(lab)]
        color = color_override if color_override is not None else NUSC_COLORS_BGR.get(cname, (128, 128, 128))
        cx_l, cy_l, _ = box[:3]
        w, l = float(box[3]), float(box[4])
        yaw = float(box[6])
        # 사각형 4 corners in lidar (z 무관)
        corners = np.array([
            [ w / 2,  l / 2], [ w / 2, -l / 2],
            [-w / 2, -l / 2], [-w / 2,  l / 2],
        ])
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        corners = corners @ R.T + np.array([cx_l, cy_l])
        pts = np.array([lidar_to_canvas(x, y) for x, y in corners], dtype=int)
        cv2.polylines(canvas, [pts], True, color, 1)
        # heading 표시 — 박스 중심 → 앞쪽 face 중심 line
        front_mid = (corners[0] + corners[1]) / 2
        cv2.line(canvas, lidar_to_canvas(cx_l, cy_l), lidar_to_canvas(*front_mid), color, 1)
        #수정: velocity 화살표 — box[7:9] 사용 (lidar 좌표계 vx,vy m/s). 1초 후 위치 = (cx+vx, cy+vy).
        if draw_velocity and len(box) >= 9:
            vx, vy = float(box[7]), float(box[8])
            if abs(vx) + abs(vy) > 0.1:
                end_x, end_y = cx_l + vx, cy_l + vy
                cv2.arrowedLine(canvas, lidar_to_canvas(cx_l, cy_l),
                                lidar_to_canvas(end_x, end_y), color, 1, tipLength=0.3)

    #수정: trajectory polylines — GT(magenta), Qwen pred(cyan), HiP-AD pred(yellow). 각각 옵션.
    def _draw_traj(pts_lidar, color, label_pos):
        if pts_lidar is None or len(pts_lidar) < 1:
            return
        pts = np.array([lidar_to_canvas(p[0], p[1]) for p in pts_lidar], dtype=int)
        start = (cx, cy)
        full = np.concatenate([np.array([start]), pts])
        cv2.polylines(canvas, [full], False, color, 2)
        cv2.circle(canvas, tuple(pts[-1]), 3, color, -1)
    _draw_traj(ego_traj_lidar,   (255,   0, 255), None)   # GT — magenta
    _draw_traj(qwen_traj_lidar,  (255, 255,   0), None)   # Qwen — cyan (BGR)
    _draw_traj(hipad_traj_lidar, (  0, 220, 220), None)   # HiP-AD — yellow (BGR)

    # 격자 + 텍스트
    cv2.rectangle(canvas, (0, 0), (canvas_size - 1, canvas_size - 1), (180, 180, 180), 1)
    cv2.putText(canvas, f"{int(span_x)}m x {int(span_y)}m", (5, canvas_size - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)
    return canvas


def _wrap_text(text, max_chars):
    """단순 word-wrap → list of lines."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def render_one_sample(info, result, dataset, point_cloud_range, score_thresh=0.2,
                      cam_resize_h=180, token2scene=None, scene2tokens=None,
                      token2info=None, n_future=6,
                      qwen_traj=None, hipad_traj=None, qwen_reason=None):
    """한 frame: 6-cam stitch (pred boxes) | pred BEV | GT BEV."""
    # 6-cam paths (dataset.get_data_info 와 같은 정규화)
    cam_paths = []
    cam_names = list(info["cams"].keys())
    for cam_name, cam in info["cams"].items():
        p = cam["data_path"]
        if "samples/" in p:
            p = "samples/" + p.split("samples/", 1)[1]
        elif "sweeps/" in p:
            p = "sweeps/" + p.split("sweeps/", 1)[1]
        if not os.path.isabs(p):
            p = os.path.join(dataset.data_root, p)
        cam_paths.append(p)

    # build lidar2img per cam
    from mmdet3d_plugin.datasets.nuscenes_mini_dataset import _quat_to_rot, _compose_rt
    lidar2ego = _compose_rt(_quat_to_rot(info["lidar2ego_rotation"]),
                            np.asarray(info["lidar2ego_translation"]))
    #수정: vector map 용 — lidar2global + 해당 location 의 NuScenesMap (실패 시 None → map 생략)
    nusc_map = None
    lidar2global = None
    try:
        ego2global = _compose_rt(_quat_to_rot(info["ego2global_rotation"]),
                                 np.asarray(info["ego2global_translation"]))
        lidar2global = ego2global @ lidar2ego
        loc = info.get("map_location")
        if loc:
            nusc_map = get_nusc_map(dataset.data_root, loc)
    except Exception as _e:
        nusc_map = None
    lidar2img_list = []
    for cam_name, cam in info["cams"].items():
        cam2lidar = _compose_rt(np.asarray(cam["sensor2lidar_rotation"]),
                                np.asarray(cam["sensor2lidar_translation"]))
        lidar2cam = np.linalg.inv(cam2lidar)
        K = np.asarray(cam["cam_intrinsic"])
        Kp = np.eye(4); Kp[:3, :3] = K
        lidar2img_list.append(Kp @ lidar2cam)

    # load 6 images
    cams_loaded = []
    for cp in cam_paths:
        if not os.path.exists(cp):
            cams_loaded.append(np.zeros((900, 1600, 3), dtype=np.uint8))
        else:
            cams_loaded.append(cv2.imread(cp))  # BGR

    # extract pred
    pred = result.get("img_bbox") if isinstance(result, dict) else None
    pred_boxes, pred_scores, pred_labels = np.zeros((0, 9)), np.zeros((0,)), np.zeros((0,), dtype=int)
    if pred is not None:
        b = pred["boxes_3d"]; s = pred["scores_3d"]; l = pred["labels_3d"]
        if torch.is_tensor(b): b = b.cpu().numpy()
        if torch.is_tensor(s): s = s.cpu().numpy()
        if torch.is_tensor(l): l = l.cpu().numpy()
        pred_boxes, pred_scores, pred_labels = b, s, l

    #수정: pred box 단일 강조 색 (orange) — GT 의 클래스별 색과 시각 구분
    PRED_BOX_COLOR = (0, 165, 255)   # BGR orange
    # draw pred boxes on each cam
    for box, sc, lab in zip(pred_boxes, pred_scores, pred_labels):
        if sc < score_thresh:
            continue
        if not (0 <= int(lab) < len(dataset.det_classes)):
            continue
        cname = dataset.det_classes[int(lab)]
        color = PRED_BOX_COLOR
        corners_3d = get_box_corners_3d(box[:3], box[3:6], box[6])
        for i, l2i in enumerate(lidar2img_list):
            #수정: near-plane 클리핑 연결 실선
            draw_box3d_on_img(cams_loaded[i], corners_3d, l2i, color, thickness=2)

    #수정: vis 스펙 — 카메라 GT 3D box 를 BEV 와 "동일한 클래스별 색"(NUSC_COLORS_BGR)으로 투영.
    #  (BEV gt_bev 도 color_override=None → 같은 팔레트 사용 → 카메라↔BEV 클래스 색 일치. car=파랑 등.)
    gt_boxes_cam = info["gt_boxes"]
    gt_names_cam = info["gt_names"]
    for box, gname in zip(gt_boxes_cam, gt_names_cam):
        gcolor = NUSC_COLORS_BGR.get(gname, (0, 255, 0))
        corners_3d = get_box_corners_3d(box[:3], box[3:6], box[6])
        for i, l2i in enumerate(lidar2img_list):
            #수정: near-plane 클리핑 연결 실선
            draw_box3d_on_img(cams_loaded[i], corners_3d, l2i, gcolor, thickness=2)

    #수정: vis 스펙 — ego future trajectory 를 CAM_FRONT 에 투영 (GT magenta / Qwen cyan / HiP-AD yellow)
    ego_traj_lidar = None
    if token2scene is not None and scene2tokens is not None and token2info is not None:
        scene_tok = token2scene.get(info["token"])
        if scene_tok:
            scene_toks = scene2tokens.get(scene_tok, [])
            try:
                cur_idx = scene_toks.index(info["token"])
                future_toks = scene_toks[cur_idx + 1: cur_idx + 1 + n_future]
                future_infos = [token2info[t] for t in future_toks if t in token2info]
                if future_infos:
                    ego_traj_lidar = get_future_ego_in_lidar(info, future_infos, n_future=n_future)
            except ValueError:
                pass
    if "CAM_FRONT" in cam_names:
        _fi = cam_names.index("CAM_FRONT")
        _l2i = lidar2img_list[_fi]
        draw_traj_on_img(cams_loaded[_fi], ego_traj_lidar, _l2i, (255, 0, 255))    # GT magenta
        draw_traj_on_img(cams_loaded[_fi], qwen_traj,      _l2i, (255, 255, 0))     # Qwen cyan
        draw_traj_on_img(cams_loaded[_fi], hipad_traj,     _l2i, (0, 220, 220))     # HiP-AD yellow

    # resize + arrange 6 cams in 2x3 grid (matching nuScenes layout)
    # Top row: FRONT_LEFT FRONT FRONT_RIGHT
    # Bottom row: BACK_LEFT BACK BACK_RIGHT
    desired_order = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                     "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
    name_to_idx = {name: i for i, name in enumerate(cam_names)}
    grid_imgs = []
    for cn in desired_order:
        idx = name_to_idx.get(cn)
        img = cams_loaded[idx] if idx is not None else np.zeros((900, 1600, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        ratio = cam_resize_h / h
        img = cv2.resize(img, (int(w * ratio), cam_resize_h))
        cv2.putText(img, cn, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        grid_imgs.append(img)

    # equalize widths to min
    min_w = min(im.shape[1] for im in grid_imgs)
    grid_imgs = [cv2.resize(im, (min_w, cam_resize_h)) for im in grid_imgs]
    top = np.concatenate(grid_imgs[:3], axis=1)
    bot = np.concatenate(grid_imgs[3:], axis=1)
    cams_stitched = np.concatenate([top, bot], axis=0)   # (2*H, 3*W, 3)

    #수정: Qwen reasoning 을 카메라 다음 별도 panel 로 표시 (cam stitch 와 같은 높이)
    reason_panel_w = 320
    reason_panel = np.full((cams_stitched.shape[0], reason_panel_w, 3), 30, dtype=np.uint8)
    # 헤더
    cv2.putText(reason_panel, "QWEN REASONING", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(reason_panel, (8, 30), (reason_panel_w - 8, 30), (0, 200, 200), 1)
    # 본문 텍스트 wrap
    body = qwen_reason if qwen_reason else "(reasoning not available)"
    max_chars = reason_panel_w // 9
    lines = _wrap_text(body, max_chars=max_chars)
    line_h = 18
    y = 50
    for ln in lines:
        if y + line_h > reason_panel.shape[0] - 5:
            break
        cv2.putText(reason_panel, ln, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)
        y += line_h
    cams_H, cams_W = cams_stitched.shape[:2]
    #수정: ego_traj_lidar 는 위 카메라 단계에서 이미 계산됨 (CAM_FRONT 투영과 공유)

    # BEV
    bev_size = cams_H  # match height
    #수정: GT BEV 만 ego trajectory polyline 그림 (pred plan task 미사용 → pred ego trajectory 불가)
    # GT box 들에 gt_velocity 붙여서 9D 로 (BEV velocity arrow 그릴 수 있게)
    gt_boxes = info["gt_boxes"]
    gt_velocity = info.get("gt_velocity")
    if gt_velocity is not None and gt_boxes.shape[1] == 7:
        gt_boxes = np.concatenate([gt_boxes, gt_velocity], axis=1)
    gt_names = info["gt_names"]
    gt_labels = []
    for n in gt_names:
        gt_labels.append(dataset.det_classes.index(n) if n in dataset.det_classes else -1)
    gt_labels = np.array(gt_labels)
    gt_scores = np.ones(len(gt_labels))

    pred_bev = draw_bev(pred_boxes, pred_labels, pred_scores, dataset.det_classes,
                        point_cloud_range, canvas_size=bev_size, score_thresh=score_thresh,
                        ego_traj_lidar=ego_traj_lidar,        # GT 도 PRED BEV 에 같이 (비교)
                        qwen_traj_lidar=qwen_traj,
                        hipad_traj_lidar=hipad_traj,
                        draw_velocity=True,
                        color_override=PRED_BOX_COLOR,        #수정: pred box 단일 색
                        nusc_map=nusc_map, lidar2global=lidar2global)   #수정: vector map
    gt_bev = draw_bev(gt_boxes, gt_labels, gt_scores, dataset.det_classes,
                      point_cloud_range, canvas_size=bev_size, score_thresh=0.0,
                      ego_traj_lidar=ego_traj_lidar, draw_velocity=True,
                      nusc_map=nusc_map, lidar2global=lidar2global)     #수정: vector map

    # 라벨
    cv2.putText(pred_bev, "PRED BEV", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    cv2.putText(gt_bev,   "GT BEV",   (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

    #수정: vis 스펙 — layout = [cams(+pred/GT box) | pred_bev | gt_bev | reasoning_panel(맨 오른쪽)]
    combined = np.concatenate([cams_stitched, pred_bev, gt_bev, reason_panel], axis=1)
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--result", required=True, help="dist_test --out 으로 덤프한 pkl")
    ap.add_argument("--out-dir", default="vis_nusc")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--interval", type=int, default=1)
    ap.add_argument("--score-thresh", type=float, default=0.2)
    ap.add_argument("--fps", type=int, default=2)
    #수정: 3-trajectory viz
    ap.add_argument("--qwen-ckpt", default=None, help="qwen_traj_train.py 산출 traj_proj.pth")
    ap.add_argument("--qwen-path", default="/workspace/src/HiP-AD/ckpts/qwen2-vl-2b")
    ap.add_argument("--hipad-ckpt", default=None, help="hipad_traj_train.py 산출 hipad_traj.pth")
    ap.add_argument("--hipad-detector-ckpt", default=None, help="HiP-AD iter_*.pth")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = build_from_cfg(cfg.data["val"], DATASETS)
    print(f"dataset: {type(dataset).__name__} | n={len(dataset)}")

    #수정: scene 인덱스를 info pkl 기반으로 직접 빌드 (v1.0-mini API 대신 — full trainval 토큰 커버).
    #  scene 내 토큰은 timestamp 순 정렬 → 미래 ego trajectory lookup 정확.
    try:
        token2info = {i["token"]: i for i in dataset.data_infos}
        import pickle as _pkl
        for k in ("train", "val"):
            try:
                with open(cfg.data[k]["ann_file"], "rb") as f:
                    d = _pkl.load(f)
                for i in d.get("infos", d if isinstance(d, list) else []):
                    token2info.setdefault(i["token"], i)
            except Exception:
                pass
        token2scene = {t: i["scene_token"] for t, i in token2info.items()}
        scene2tokens = {}
        for t, i in token2info.items():
            scene2tokens.setdefault(i["scene_token"], []).append(t)
        for st in scene2tokens:
            scene2tokens[st].sort(key=lambda t: token2info[t]["timestamp"])
        print(f"scene index(info-based): {len(scene2tokens)} scenes, {len(token2info)} tokens")
    except Exception as e:
        print(f"WARN: scene index build 실패 ({type(e).__name__}: {e}) → ego trajectory 생략")
        token2scene = scene2tokens = token2info = None

    with open(args.result, "rb") as f:
        results = pickle.load(f)
    print(f"results: n={len(results)}")
    assert len(results) == len(dataset), f"len mismatch: results {len(results)} vs dataset {len(dataset)}"

    out_dir = Path(args.out_dir)
    combine_dir = out_dir / "combine"
    combine_dir.mkdir(parents=True, exist_ok=True)

    end = args.end if args.end is not None else len(dataset)
    indices = list(range(args.start, end, args.interval))
    pcr = cfg.point_cloud_range

    #수정: Qwen / HiP-AD trajectory predictors 로드 (옵션)
    qwen_pred_fn = None
    if args.qwen_ckpt and os.path.exists(args.qwen_ckpt):
        try:
            import torch
            from mmdet3d_plugin.models.qwen_traj_generator import QwenTrajectoryGenerator
            from transformers import AutoProcessor
            qg = QwenTrajectoryGenerator(qwen_model_path=args.qwen_path, embed_dim=1536,
                                         traj_ts=6, traj_dim=2, freeze_qwen=True,
                                         torch_dtype="bf16", use_lora=False,
                                         gradient_checkpointing=False)
            ck = torch.load(args.qwen_ckpt, map_location="cpu")
            qg.traj_proj.load_state_dict(ck["traj_proj"])
            qg = qg.to("cuda").eval()
            processor = AutoProcessor.from_pretrained(args.qwen_path,
                                                       min_pixels=200 * 28 * 28,
                                                       max_pixels=200 * 28 * 28)
            from PIL import Image as _PIL
            INSTR = ("These are 6 surround-view camera images of an ego vehicle (front, front-left, "
                     "front-right, back-left, back, back-right). Predict the ego vehicle's future "
                     "trajectory for the next 3 seconds as 6 waypoints (0.5s spacing) in lidar frame meters.")
            REASON_INSTR = ("You see 6 surround cameras of an ego vehicle. Look carefully at "
                            "what is specifically visible (other vehicles, pedestrians, traffic "
                            "signs, intersections, lane markings, parked cars, road geometry, "
                            "obstacles). In ONE concise English sentence (under 25 words), "
                            "explain the next driving decision and cite ONE specific visual cue. "
                            "Avoid generic phrasing.")
            def _qwen_fn(info):
                imgs = []
                for cam in ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                            "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]:
                    p = info["cams"][cam]["data_path"]
                    if "samples/" in p:
                        p = "samples/" + p.split("samples/", 1)[1]
                    if not os.path.isabs(p):
                        p = os.path.join(dataset.data_root, p)
                    imgs.append(_PIL.open(p).convert("RGB") if os.path.exists(p)
                                else _PIL.new("RGB", (1600, 900), (0, 0, 0)))
                # 1) trajectory (traj_proj forward)
                msgs = [{"role":"user","content":[{"type":"image","image":im} for im in imgs]+
                                                 [{"type":"text","text":INSTR}]}]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inp = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
                inp = {k_:v.to("cuda") for k_,v in inp.items()}
                with torch.no_grad():
                    pred = qg(qwen_pixel_values=inp["pixel_values"],
                              qwen_input_ids=inp["input_ids"],
                              qwen_attention_mask=inp["attention_mask"],
                              qwen_image_grid_thw=inp.get("image_grid_thw"))
                traj_arr = pred[0].cpu().numpy()  # (T, 2)

                # 2) reasoning text (Qwen.generate)
                reasoning = ""
                try:
                    r_msgs = [{"role":"user","content":[{"type":"image","image":im} for im in imgs]+
                                                       [{"type":"text","text":REASON_INSTR}]}]
                    r_text = processor.apply_chat_template(r_msgs, tokenize=False, add_generation_prompt=True)
                    r_inp = processor(text=[r_text], images=imgs, return_tensors="pt", padding=True)
                    r_inp = {k_:v.to("cuda") for k_,v in r_inp.items()}
                    with torch.no_grad():
                        gen_ids = qg.qwen.generate(
                            **r_inp, max_new_tokens=55,
                            do_sample=True, temperature=0.8, top_p=0.9,
                            repetition_penalty=1.15,
                        )
                    # strip input prefix
                    new_ids = gen_ids[:, r_inp["input_ids"].shape[1]:]
                    reasoning = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
                except Exception as e:
                    reasoning = f"(reasoning err: {type(e).__name__})"
                return traj_arr, reasoning
            qwen_pred_fn = _qwen_fn
            print(f"[viz] Qwen pred loaded from {args.qwen_ckpt}")
        except Exception as e:
            print(f"[viz] Qwen pred 로드 실패: {type(e).__name__}: {e}")

    hipad_pred_fn = None
    if args.hipad_ckpt and os.path.exists(args.hipad_ckpt):
        try:
            import torch
            from mmcv.runner import load_checkpoint
            from mmdet.models import build_detector
            from tools.hipad_traj_train import TrajHead
            detector = build_detector(cfg.model)
            ck_d = args.hipad_detector_ckpt or "work_dirs/hipad_nusc_mini_stage1/iter_50000.pth"
            load_checkpoint(detector, ck_d, map_location="cpu", strict=False)
            detector = detector.to("cuda").eval()
            head = TrajHead(in_dim=cfg.embed_dims, hidden=128, n_future=6, traj_dim=2)
            ck = torch.load(args.hipad_ckpt, map_location="cpu")
            head.load_state_dict(ck["head"])
            head = head.to("cuda").eval()

            # 데이터셋 pipeline 으로 img 텐서 생성
            ds_for_img = build_from_cfg(cfg.data["val"], DATASETS)
            def _hipad_fn(info):
                # find idx for this info in ds_for_img (token match)
                idx_ = None
                for i, di in enumerate(ds_for_img.data_infos):
                    if di["token"] == info["token"]:
                        idx_ = i; break
                if idx_ is None:
                    return None
                data = ds_for_img[idx_]
                img = data["img"]
                if hasattr(img, "data"): img = img.data
                if isinstance(img, list):
                    img = img[0] if len(img) == 1 else torch.stack(img)
                img = img.to("cuda").unsqueeze(0) if img.dim() == 4 else img.to("cuda")
                with torch.no_grad():
                    bs_ = img.shape[0]
                    if img.dim() == 5:
                        num_cams_ = img.shape[1]
                        img_flat = img.flatten(end_dim=1)
                    else:
                        num_cams_ = 1
                        img_flat = img
                    fm = detector.img_backbone(img_flat)
                    if detector.img_neck is not None:
                        fm = list(detector.img_neck(fm))
                    feat = fm[-1].mean(dim=(2, 3))  # (B*num_cams, C)
                    feat = feat.view(bs_, num_cams_, -1).mean(dim=1)  # (B, C)
                    pred = head(feat.float())
                return pred[0].cpu().numpy()
            hipad_pred_fn = _hipad_fn
            print(f"[viz] HiP-AD pred loaded from {args.hipad_ckpt}")
        except Exception as e:
            print(f"[viz] HiP-AD pred 로드 실패: {type(e).__name__}: {e}")

    saved_paths = []
    for k, idx in enumerate(indices):
        info = dataset.data_infos[idx]
        result = results[idx]
        qwen_out = qwen_pred_fn(info) if qwen_pred_fn else None
        if isinstance(qwen_out, tuple):
            qwen_traj, qwen_reason = qwen_out
        else:
            qwen_traj, qwen_reason = qwen_out, None
        hipad_traj = hipad_pred_fn(info) if hipad_pred_fn else None
        img = render_one_sample(info, result, dataset, pcr, score_thresh=args.score_thresh,
                                 token2scene=token2scene, scene2tokens=scene2tokens,
                                 token2info=token2info, n_future=6,
                                 qwen_traj=qwen_traj, hipad_traj=hipad_traj,
                                 qwen_reason=qwen_reason)
        save_path = combine_dir / f"{idx:04d}.jpg"
        cv2.imwrite(str(save_path), img)
        saved_paths.append(str(save_path))
        if k % 10 == 0:
            print(f"  [{k+1}/{len(indices)}] saved {save_path.name}  size={img.shape}")

    # video.mp4 via cv2 VideoWriter (ffmpeg 의존 없이)
    if saved_paths:
        H, W = cv2.imread(saved_paths[0]).shape[:2]
        video_path = out_dir / "video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (W, H))
        for p in saved_paths:
            writer.write(cv2.imread(p))
        writer.release()
        print(f"video saved: {video_path} | {len(saved_paths)} frames @ {args.fps} fps")

    print(f"DONE: {len(saved_paths)} combines in {combine_dir}")


if __name__ == "__main__":
    main()
