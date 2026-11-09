# 수정(2026-10-19): 사용자 결정 — GRPO 대신 Hydra-MDP 방식 계획 점수 학습(LaQ v2 2단계 뒤 적용). 새 파일(기존 파일 수정 없음).
"""모드별 규칙 점수(Hydra-MDP 의 sub-score) — CPU, 벡터화.

Hydra-MDP(Li et al., arXiv 2406.06978)는 후보 궤적마다 규칙 평가기(PDM score)의 하위 점수(NC·DAC·EP·TTC·C)를 정답으로
두고 점수기 머리가 그것을 BCE 로, 사람 궤적 모방을 soft-target CE 로 배운다. nuScenes 개방 루프에는 PDM 시뮬레이터가 없으므로
같은 역할을 하는 점수를 평가기·지도 GT 로 직접 계산한다:

  no_collision (nc)   tools/eval_nusc_planning.py 의 충돌 규칙을 그대로 쓴다(같은 함수 box3d_to_corners·get_yaw 재사용):
                      자차 박스 4.084×1.85(평가기 값), 궤적 접선 yaw(평가기 get_yaw — 이동 0.5 m 미만이면 π/2),
                      중심 0.5 m 전방 이동, 바닥 4 모서리 사각형이 시점 t 의 미래 GT 박스(평가기 _get_fut_boxes)와 겹치면 충돌.
                      GT 궤적도 그 시점에 충돌하면 그 시점은 세지 않는다(평가기 obj_box_col = pc & ~gc).
                      nc = 1 ⇔ 6 시점 어디에서도 (모드 충돌 & ~GT 충돌) 이 없다.
                      겹침 판정은 볼록 사각형 분리축 정리(SAT, 경계 접촉 = 겹침 — shapely intersects 와 같은 정의)로 벡터화하고,
                      평가기 shapely 판정과 같은지는 test_plan_scorer_cpu.py 와 --verify_evaluator(전 표본 대조)로 확인한다.
  drivable (dac)      nuScenes map expansion v1.3 의 drivable_area 다각형(전역 좌표) 합집합. 위와 같은 자차 박스의 4 모서리가
                      시점마다 모두 안에 있으면 그 시점 준수(PDM DAC 와 같은 '모든 모서리' 규칙). lidar→global 은 info 의
                      lidar2ego·ego2global(데이터셋 _T_global 과 같은 식).
                      dac_strict = 6 시점 모두 준수. dac(학습 기본) = GT 가 준수한 시점에서 모드가 어긴 적이 없다
                      (충돌 규칙의 'GT 도 충돌한 시점 제외' 와 같은 원리 — 지도 오차·주차장 등 GT 자체가 밖인 시점은 판단 근거가 없다).
  progress (ep)       PDM EP 와 같은 '최선 안전 모드 대비 경로 진행 비율'. 수정(2026-10-19, 사용자 결정 2): NAVSIM/PDM 처럼 **경로(route)
                      중심선** 위 진행을 잰다. nuScenes 에는 경로가 없으므로 map expansion 의 차선 중심선(lane·lane_connector
                      arcline, 1 m 간격 이산화)과 연결 그래프로 자차 근처에서 시작하는 후보 경로들을 만들고, GT 미래 6 점과 평균 거리가
                      가장 작은 경로를 고른다(GT 는 정답 라벨 계산에만 쓰고 점수기 입력에는 쓰지 않는다). 진행 s_i = 모드 끝점을 그
                      경로에 사영한 호 길이 − 원점(자차) 사영 호 길이. 안전 모드(nc=1 & dac=1; 없으면 전체) 중 최대 s_max 로 나눠
                      [0,1] 로 자른다. s_max < 5 m 이면 전 모드 1(PDM 규칙).
                      반경 안에 맞는 차선 경로가 없으면(또는 GT 와 평균 거리 > route_max_dev) GT 미래 경로(원점→6 점, 마지막 구간
                      반직선 연장)로 대신하고 그 수를 센다. 종전 'GT 경로' 판(--progress gt)은 라벨 통계 비교용으로 남긴다.
  imitation (im)      soft target softmax(-L2̄/τ), L2̄ = 6 시점 L2 평균(평가기 L2 와 같은 시점 거리). τ 는 인자(기본 0.5 m).

미래 GT 가 불완전한 표본(gt_ego_fut_masks 미완)은 평가기와 같이 제외한다 — valid=False 로 표시하고 수를 센다(점수는 NaN).
사용: python tools/plan_scorer/subscores.py --dump <dump_dir> --out <subscores.npz> [--verify_evaluator] [--workers 16] [--progress lane|gt]
      차선 경로 통계만(덤프 없이 GT 로): python tools/plan_scorer/subscores.py --route_stats <config> --split train --out <json>
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import scorer_io  # noqa: E402

EGO_SIZE = (4.084, 1.85, 1.56)  # 평가기 single_coll 값(W,L,H 열 위치 그대로)
EGO_OFFSET = 0.5                # 평가기 check_collision 의 전방 이동
PROGRESS_MIN_M = 5.0            # PDM EP: 최선 진행이 이보다 작으면 전 모드 1


def _load_evaluator():
    """tools/eval_nusc_planning.py 를 경로로 적재(같은 함수를 그대로 쓰기 위해). 패키지 이름 충돌 회피."""
    p = os.path.join(_ROOT, "tools", "eval_nusc_planning.py")
    spec = importlib.util.spec_from_file_location("_eval_nusc_planning", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EVAL = _load_evaluator()


# --------------------------------------------------------------------------------------------- 기하
def get_yaw_batch(traj):
    """평가기 get_yaw 의 배치판. traj (M,T,2) torch float32 → (M,T). 같은 torch 연산 순서라 값이 같다."""
    M, T, _ = traj.shape
    cat = torch.cat([traj.new_zeros((M, 1, 2)), traj], 1)
    yaw = traj.new_zeros((M, T + 1))
    yaw[:, 1:-1] = torch.atan2(cat[:, 2:, 1] - cat[:, :-2, 1], cat[:, 2:, 0] - cat[:, :-2, 0])
    yaw[:, -1] = torch.atan2(cat[:, -1, 1] - cat[:, -2, 1], cat[:, -1, 0] - cat[:, -2, 0])
    yaw = yaw[:, 1:]
    still = torch.linalg.norm(traj[:, -1] - traj[:, 0], dim=-1) < 0.5
    yaw[still] = np.pi / 2
    return yaw


def ego_corners(trajs):
    """trajs (M,T,2) → 평가기와 같은 자차 박스의 바닥 4 모서리 (M,T,4,2) float64.
    평가기: ego=[x,y,0,4.084,1.85,1.56,yaw] (float32 torch) → 중심 += 0.5(cos,sin) → box3d_to_corners[...,[0,3,7,4],:2]."""
    tr = torch.as_tensor(np.asarray(trajs), dtype=torch.float32)
    M, T, _ = tr.shape
    ego = tr.new_zeros((M, T, 7))
    ego[..., :2] = tr
    ego[..., 3:6] = ego.new_tensor(EGO_SIZE)
    ego[..., 6] = get_yaw_batch(tr)
    ego = ego.reshape(M * T, 7)
    ego[:, 0] += EGO_OFFSET * torch.cos(ego[:, 6])
    ego[:, 1] += EGO_OFFSET * torch.sin(ego[:, 6])
    c = EVAL.box3d_to_corners(ego)[:, [0, 3, 7, 4], :2]
    return c.reshape(M, T, 4, 2)


def box_corners(boxes):
    """미래 GT 박스 (n,7) → (n,4,2). 평가기 check_collision 과 같은 함수·모서리 순서."""
    b = np.asarray(boxes, np.float32).reshape(-1, 7)
    if len(b) == 0:
        return np.zeros((0, 4, 2))
    return EVAL.box3d_to_corners(torch.as_tensor(b))[:, [0, 3, 7, 4], :2]


def rect_intersect(A, B):
    """볼록 사각형 겹침(분리축 정리). A (P,4,2), B (Q,4,2) → (P,Q) bool. 경계 접촉도 겹침(shapely intersects 와 같은 정의)."""
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    P, Q = len(A), len(B)
    if P == 0 or Q == 0:
        return np.zeros((P, Q), bool)
    ea = A[:, [1, 2]] - A[:, [0, 1]]            # (P,2,2) 두 변
    eb = B[:, [1, 2]] - B[:, [0, 1]]
    na = np.stack([-ea[..., 1], ea[..., 0]], -1)  # 법선(정규화 불필요: 비교는 같은 축 위)
    nb = np.stack([-eb[..., 1], eb[..., 0]], -1)
    sep = np.zeros((P, Q), bool)
    for axes, per_a in ((na, True), (nb, False)):
        for k in range(2):
            if per_a:
                ax = axes[:, k][:, None, :]             # (P,1,2)
            else:
                ax = axes[:, k][None, :, :]             # (1,Q,2)
            pa = np.einsum("pcd,pqd->pqc", A, np.broadcast_to(ax, (P, Q, 2)))   # (P,Q,4)
            pb = np.einsum("qcd,pqd->pqc", B, np.broadcast_to(ax, (P, Q, 2)))
            sep |= (pa.max(-1) < pb.min(-1)) | (pb.max(-1) < pa.min(-1))
    return ~sep


def collision_steps(trajs, fut_boxes, ego_c=None):
    """trajs (M,T,2), fut_boxes 길이 T 목록 → (M,T) bool. 평가기 single_coll 과 같은 판정(벡터화)."""
    if ego_c is None:
        ego_c = ego_corners(trajs)
    M, T = ego_c.shape[:2]
    if len(fut_boxes) < T:
        raise ValueError(f"미래 박스 시점 {len(fut_boxes)} < {T} — 불완전 GT 표본은 호출 전에 제외해야 한다")
    out = np.zeros((M, T), bool)
    for t in range(T):
        bc = box_corners(fut_boxes[t])
        if len(bc):
            out[:, t] = rect_intersect(ego_c[:, t], bc).any(-1)
    return out


def evaluator_collision_steps(trajs, fut_boxes):
    """대조용: 평가기 single_coll(shapely)을 모드마다 그대로 호출. (M,T) bool."""
    tr = torch.as_tensor(np.asarray(trajs), dtype=torch.float32)
    return np.stack([EVAL.single_coll(tr[m].clone(), fut_boxes).numpy() for m in range(len(tr))])


def l2_steps(trajs, gt):
    """(M,T,2), (T,2) → (M,T). 평가기 L2(시점별 유클리드 거리; x 부호 반전은 거리에 무관)."""
    tr = torch.as_tensor(np.asarray(trajs), dtype=torch.float32)
    g = torch.as_tensor(np.asarray(gt), dtype=torch.float32)
    return torch.sqrt(((tr - g[None]) ** 2).sum(-1)).numpy()


# --------------------------------------------------------------------------------------------- 주행 가능 영역
class DrivableArea:
    """nuScenes map expansion drivable_area 합집합(전역 좌표). 점 판정은 shapely.vectorized.contains."""

    def __init__(self, dataroot="data/nuscenes"):
        self.dataroot = dataroot
        self._maps = {}

    def geom(self, location):
        if location not in self._maps:
            from nuscenes.map_expansion.map_api import NuScenesMap
            from shapely.ops import unary_union
            m = NuScenesMap(dataroot=self.dataroot, map_name=location)
            polys = [m.extract_polygon(pt) for r in m.drivable_area for pt in r["polygon_tokens"]]
            if not polys:
                raise RuntimeError(f"{location}: drivable_area 다각형이 없다")
            self._maps[location] = unary_union(polys)
        return self._maps[location]

    def inside(self, location, xy_global):
        from shapely import vectorized
        xy = np.asarray(xy_global, np.float64).reshape(-1, 2)
        return vectorized.contains(self.geom(location), xy[:, 0], xy[:, 1])


def lidar_to_global_xy(pts, lidar2global):
    """pts (...,2) lidar 평면 점(z=0) → 전역 xy."""
    T = np.asarray(lidar2global, np.float64)
    p = np.asarray(pts, np.float64)
    return p @ T[:2, :2].T + T[:2, 3]


def drivable_steps(corners, lidar2global, location, area):
    """corners (M,T,4,2) lidar → (M,T) bool: 4 모서리 모두 주행 가능 영역 안."""
    M, T = corners.shape[:2]
    g = lidar_to_global_xy(corners.reshape(-1, 2), lidar2global)
    ok = area.inside(location, g).reshape(M, T, 4)
    return ok.all(-1)


# --------------------------------------------------------------------------------------------- 진행·모방
def polyline_arclength(points, path):
    """points (K,2) 를 polyline path (P,2) 에 사영한 호 길이와 거리. 마지막 구간은 위로 열린 반직선(경로 끝을 넘은 진행도 센다).
    가장 가까운 구간을 고른다. 반환 (s (K,), d (K,))."""
    path = np.asarray(path, np.float64)
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    seg = path[1:] - path[:-1]
    ln = np.linalg.norm(seg, axis=-1)
    keep = ln > 1e-6
    if not keep.any():
        return np.zeros(len(pts)), np.linalg.norm(pts - path[0], axis=-1)
    starts = path[:-1][keep]; seg = seg[keep]; ln = ln[keep]
    cum = np.concatenate([[0.0], np.cumsum(ln)[:-1]])
    rel = pts[:, None, :] - starts[None]
    u = (rel * seg[None]).sum(-1) / (ln[None] ** 2)
    hi = np.ones_like(u); hi[:, -1] = np.inf
    u = np.clip(u, 0.0, hi)
    proj = starts[None] + u[..., None] * seg[None]
    d = np.linalg.norm(pts[:, None, :] - proj, axis=-1)
    k = d.argmin(1)
    r = np.arange(len(pts))
    return cum[k] + u[r, k] * ln[k], d[r, k]


def gt_path(gt):
    return np.concatenate([np.zeros((1, 2)), np.asarray(gt, np.float64)], 0)


def path_progress(points, gt):
    """(종전 판, --progress gt) points (M,2) 를 GT 경로(원점→gt, 마지막 구간 반직선 연장)에 사영한 호 길이 (M,).
    정지 GT(방향 정보 없음)는 전방(+y) 반직선."""
    path = gt_path(gt)
    if not (np.linalg.norm(path[1:] - path[:-1], axis=-1) > 1e-6).any():
        return np.asarray(points, np.float64)[:, 1].copy()
    return polyline_arclength(points, path)[0]


def route_progress(points, route):
    """차선 경로(lidar, (P,2)) 위 진행: 끝점 사영 호 길이 − 원점 사영 호 길이."""
    s, _ = polyline_arclength(points, route)
    s0, _ = polyline_arclength(np.zeros((1, 2)), route)
    return s - s0[0]


class LaneRoutes:
    """nuScenes map expansion 차선 중심선 그래프로 '경로(route)' 를 만든다(수정 2026-10-19, 사용자 결정 2).
    후보 = 자차(lidar 원점) 반경 radius 안에 점이 있고 방향 차가 heading_tol 이하인 lane/lane_connector 에서 시작해
    outgoing 연결을 따라 GT 길이 + extra_m 이상 이어지는 경로들(최대 max_paths). 선택 = GT 6 점과 평균 거리 최소.
    평균 거리 > max_dev 이면 '경로 없음'(호출자가 GT 경로로 대신하고 수를 센다)."""

    def __init__(self, dataroot="data/nuscenes", radius=5.0, heading_tol=np.pi / 3, max_dev=2.0, extra_m=30.0,
                 max_paths=256, resolution=1.0):
        self.dataroot = dataroot
        self.radius, self.heading_tol, self.max_dev = radius, heading_tol, max_dev
        self.extra_m, self.max_paths, self.res = extra_m, max_paths, resolution
        self._maps = {}

    def _map(self, location):
        if location not in self._maps:
            from nuscenes.map_expansion.map_api import NuScenesMap
            from scipy.spatial import cKDTree
            m = NuScenesMap(dataroot=self.dataroot, map_name=location)
            toks = [r["token"] for r in m.lane] + [r["token"] for r in m.lane_connector]
            toks = [t for t in toks if t in m.arcline_path_3]
            disc = m.discretize_lanes(toks, self.res)
            pts = {t: np.asarray(disc[t], np.float64) for t in toks if len(disc[t]) >= 2}
            owner, idx, allp = [], [], []
            for t, a in pts.items():
                owner += [t] * len(a); idx += list(range(len(a))); allp.append(a[:, :2])
            out = {t: [o for o in m.connectivity.get(t, {}).get("outgoing", []) if o in pts] for t in pts}
            self._maps[location] = dict(pts=pts, out=out, owner=owner, idx=np.asarray(idx),
                                        tree=cKDTree(np.concatenate(allp, 0)))
        return self._maps[location]

    def candidates(self, location, lidar2global, need_len):
        """전역 좌표 후보 경로 목록 [(P,2), ...]."""
        M = self._map(location)
        T = np.asarray(lidar2global, np.float64)
        ego = T[:2, 3]
        fwd = T[:3, :3] @ np.array([0.0, 1.0, 0.0])          # lidar +y = 전방
        ego_yaw = np.arctan2(fwd[1], fwd[0])
        best = {}
        for j in M["tree"].query_ball_point(ego, self.radius):
            t, k = M["owner"][j], int(M["idx"][j])
            yaw = M["pts"][t][k, 2]
            dyaw = abs((yaw - ego_yaw + np.pi) % (2 * np.pi) - np.pi)
            if dyaw > self.heading_tol:
                continue
            dist = np.linalg.norm(M["pts"][t][k, :2] - ego)
            if t not in best or dist < best[t][1]:
                best[t] = (k, dist)
        paths = []
        for t, (k, _) in best.items():
            seg = M["pts"][t][max(k - 2, 0):, :2]
            stack = [(t, seg, float(np.linalg.norm(np.diff(seg, axis=0), axis=-1).sum()), 0)]
            while stack and len(paths) < self.max_paths:
                tok, pl, L, depth = stack.pop()
                nxt = M["out"].get(tok, [])
                if L >= need_len or not nxt or depth >= 20:
                    paths.append(pl); continue
                for o in nxt:
                    a = M["pts"][o][:, :2]
                    stack.append((o, np.concatenate([pl, a], 0), L + float(np.linalg.norm(np.diff(a, axis=0), axis=-1).sum()),
                                  depth + 1))
        return paths

    def route(self, location, lidar2global, gt):
        """반환: (경로 lidar (P,2) 또는 None, GT 평균 거리, 후보 수)."""
        gt = np.asarray(gt, np.float64)
        need = float(np.linalg.norm(np.diff(gt_path(gt), axis=0), axis=-1).sum()) + self.extra_m
        cands = self.candidates(location, lidar2global, need)
        if not cands:
            return None, np.inf, 0
        Tinv = np.linalg.inv(np.asarray(lidar2global, np.float64))
        z0 = float(np.asarray(lidar2global)[2, 3])
        best, bdev = None, np.inf
        for c in cands:
            h = np.concatenate([c, np.full((len(c), 1), z0), np.ones((len(c), 1))], 1)
            cl = (h @ Tinv.T)[:, :2]
            dev = float(polyline_arclength(gt, cl)[1].mean())
            if dev < bdev:
                best, bdev = cl, dev
        if bdev > self.max_dev:
            return None, bdev, len(cands)
        return best, bdev, len(cands)


def progress_score(s, safe, min_m=PROGRESS_MIN_M):
    """PDM EP: s/max_safe 를 [0,1] 로. 안전 모드가 없으면 전체 최대. 최대 < min_m 이면 전부 1."""
    s = np.maximum(np.asarray(s, np.float64), 0.0)
    ref = s[safe] if np.any(safe) else s
    smax = float(ref.max())
    if smax < min_m:
        return np.ones_like(s)
    return np.clip(s / smax, 0.0, 1.0)


def imitation_target(l2_mean, tau):
    z = -np.asarray(l2_mean, np.float64) / float(tau)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# --------------------------------------------------------------------------------------------- 평가기 지표 집계
def evaluator_metrics(l2_step, col_step):
    """평가기 main() 의 집계와 같다: 시점별 표본 평균 → t 초까지 누적 평균, 1s/2s/3s = 2/4/6 번째, avg = 셋 평균.
    l2_step (N,6), col_step (N,6) — 선택된 궤적의 시점별 L2·(모드 충돌 & ~GT 충돌)."""
    out = {}
    n = len(l2_step)
    for k, v in (("L2", np.asarray(l2_step, np.float64).sum(0) / n),
                 ("obj_box_col", np.asarray(col_step, np.float64).sum(0) / n)):
        cum = [float(v[:j + 1].mean()) for j in range(6)]
        out[k] = {"1s": cum[1], "2s": cum[3], "3s": cum[5], "avg": (cum[1] + cum[3] + cum[5]) / 3}
    out["n_eval"] = n
    return out


# --------------------------------------------------------------------------------------------- 표본 단위
def sample_subscores(modes, gt, gt_mask, fut_boxes, lidar2global, location, area, tau, min_progress=PROGRESS_MIN_M,
                     routes=None, progress="lane"):
    """표본 하나. 반환 dict(값은 numpy). 불완전 GT 이면 valid=False(평가기와 같은 제외).
    progress='lane': 차선 경로 진행(경로 없으면 GT 경로로 대신, route_found=False). 'gt': 종전 GT 경로 진행.
    두 판(ep_lane, ep_gt)을 모두 계산해 저장하고, ep = 선택한 판."""
    if progress not in ("lane", "gt"):
        raise ValueError(progress)
    M, T = modes.shape[:2]
    valid = bool((np.asarray(gt_mask) == 1).all())
    if not valid:
        return {"valid": False}
    if len(fut_boxes) != T:  # 평가기와 같은 단언 — 마스크 완비 표본은 미래 박스 6 시점이 있어야 한다
        raise AssertionError(f"미래 박스 {len(fut_boxes)} != {T} (마스크 완비 표본)")
    gt = np.asarray(gt, np.float32)
    ec = ego_corners(modes)
    gc_ = ego_corners(gt[None])
    pc = collision_steps(modes, fut_boxes, ec)                 # (M,T)
    gc = collision_steps(gt[None], fut_boxes, gc_)[0]          # (T,)
    col = pc & ~gc[None]
    d_m = drivable_steps(ec, lidar2global, location, area)     # (M,T)
    d_g = drivable_steps(gc_, lidar2global, location, area)[0]
    l2 = l2_steps(modes, gt)
    nc = ~col.any(1)
    dac_strict = d_m.all(1)
    dac = ~((~d_m) & d_g[None]).any(1)
    ends = np.asarray(modes, np.float64)[:, -1]
    safe = nc & dac
    s_gt = path_progress(ends, gt)
    ep_gt = progress_score(s_gt, safe, min_progress)
    route, dev, ncand = (None, np.inf, 0) if routes is None else routes.route(location, lidar2global, gt)
    if route is not None:
        s_lane = route_progress(ends, route)
        ep_lane = progress_score(s_lane, safe, min_progress)
    else:  # 반경 안 맞는 차선 경로 없음 → GT 경로로 대신(수를 센다)
        s_lane, ep_lane = s_gt, ep_gt
    im = imitation_target(l2.mean(1), tau)
    return dict(valid=True, nc=nc, dac=dac, dac_strict=dac_strict, ep=(ep_lane if progress == "lane" else ep_gt),
                ep_lane=ep_lane, ep_gt=ep_gt, progress_m=(s_lane if progress == "lane" else s_gt),
                route_found=route is not None, route_dev=dev, route_ncand=ncand, im=im, l2_step=l2, col_step=col,
                gt_col_step=gc, gt_drivable_step=d_g, mode_drivable_step=d_m)


_W = {}


def _worker_init(dataroot, route_kw=None):
    _W["area"] = DrivableArea(dataroot)
    _W["routes"] = LaneRoutes(dataroot, **(route_kw or {}))


def _work(args):
    i, modes, gt, gm, fut, l2g, loc, tau, minp, verify, progress = args
    r = sample_subscores(modes, gt, gm, fut, l2g, loc, _W["area"], tau, minp, _W["routes"], progress)
    if verify and r["valid"]:
        ev_p = evaluator_collision_steps(modes, fut)
        ev_g = evaluator_collision_steps(np.asarray(gt, np.float32)[None], fut)[0]
        ev_col = ev_p & ~ev_g[None]
        r["verify_mismatch"] = int((ev_col != r["col_step"]).sum())
    return i, r


def compute_all(dump, tau, dataroot="data/nuscenes", workers=8, verify=False, min_progress=PROGRESS_MIN_M,
                progress="lane", route_kw=None):
    N = len(dump["idx"])
    M, T = dump["modes"].shape[1:3]
    jobs = [(i, dump["modes"][i], dump["gt_traj"][i], dump["gt_mask"][i], dump["fut_boxes"][i],
             dump["lidar2global"][i], str(dump["map_location"][i]), tau, min_progress, verify, progress) for i in range(N)]
    res = [None] * N
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers, initializer=_worker_init, initargs=(dataroot, route_kw)) as pool:
            for i, r in pool.imap_unordered(_work, jobs, chunksize=32):
                res[i] = r
    else:
        _worker_init(dataroot, route_kw)
        for j in jobs:
            i, r = _work(j); res[i] = r
    valid = np.array([r["valid"] for r in res])
    nanf = lambda shape: np.full(shape, np.nan, np.float32)  # noqa: E731
    out = {"idx": dump["idx"], "valid": valid,
           "nc": nanf((N, M)), "dac": nanf((N, M)), "dac_strict": nanf((N, M)), "ep": nanf((N, M)),
           "ep_lane": nanf((N, M)), "ep_gt": nanf((N, M)), "im": nanf((N, M)),
           "progress_m": nanf((N, M)), "l2_step": nanf((N, M, T)), "col_step": np.zeros((N, M, T), bool),
           "gt_col_step": np.zeros((N, T), bool), "gt_drivable_step": np.zeros((N, T), bool),
           "mode_drivable_step": np.zeros((N, M, T), bool), "route_found": np.zeros(N, bool),
           "route_dev": nanf((N,)), "route_ncand": np.zeros(N, np.int64)}
    mism = 0
    for i, r in enumerate(res):
        if not r["valid"]:
            continue
        for k in ("nc", "dac", "dac_strict", "ep", "ep_lane", "ep_gt", "im", "progress_m", "l2_step",
                  "col_step", "gt_col_step", "gt_drivable_step", "mode_drivable_step", "route_found", "route_dev",
                  "route_ncand"):
            out[k][i] = r[k]
        mism += r.get("verify_mismatch", 0)
    stats = summarize(out, dump)
    stats["tau"] = tau; stats["min_progress_m"] = min_progress; stats["progress"] = progress
    stats["route_params"] = vars(LaneRoutes(dataroot, **(route_kw or {})))
    stats["route_params"].pop("_maps", None)
    if verify:
        stats["verify_evaluator_mismatch_cells"] = int(mism)
        stats["verify_evaluator_checked_cells"] = int(valid.sum() * M * T)
    return out, stats


def _rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


def label_correlations(sc):
    """진행 점수와 모방 점수의 상관(유효 표본 × 모드 펼침). 표본 안 상관(모드 간)의 평균도 함께."""
    v = sc["valid"].astype(bool)
    im = sc["im"][v]
    out = {}
    for k in ("ep_lane", "ep_gt"):
        ep = sc[k][v]
        a, b = ep.reshape(-1), im.reshape(-1)
        out[f"pearson_{k}_vs_im"] = float(np.corrcoef(a, b)[0, 1])
        out[f"spearman_{k}_vs_im"] = float(np.corrcoef(_rank(a), _rank(b))[0, 1])
        per = []
        for e, m in zip(ep, im):
            if e.std() > 1e-9 and m.std() > 1e-9:
                per.append(np.corrcoef(e, m)[0, 1])
        out[f"within_sample_pearson_{k}_vs_im_mean"] = float(np.mean(per)) if per else float("nan")
        out[f"within_sample_n_{k}"] = len(per)
    return out


def summarize(sc, dump):
    v = sc["valid"]
    N = len(v)
    sel = dump["sel"][v]
    r = np.arange(v.sum())
    st = {"n_samples": int(N), "n_valid": int(v.sum()), "n_excluded_incomplete_gt": int((~v).sum())}
    for k in ("nc", "dac", "dac_strict"):
        st[f"mean_{k}_all_modes"] = float(np.nanmean(sc[k][v]))
        st[f"mean_{k}_selected"] = float(sc[k][v][r, sel].mean())
    st["mean_ep_selected"] = float(sc["ep"][v][r, sel].mean())
    st["route_found_rate"] = float(sc["route_found"][v].mean())
    st["route_fallback_gt_count"] = int((~sc["route_found"][v]).sum())
    st["route_dev_median_found"] = float(np.median(sc["route_dev"][v][sc["route_found"][v]])) if sc["route_found"][v].any() else None
    st.update(label_correlations(sc))
    st["gt_drivable_all_steps_rate"] = float(sc["gt_drivable_step"][v].all(1).mean())
    st["gt_drivable_step_rate"] = float(sc["gt_drivable_step"][v].mean())
    st["gt_collision_any_step_rate"] = float(sc["gt_col_step"][v].any(1).mean())
    l2m = sc["l2_step"][v].mean(-1)
    st["selected_l2_mean"] = float(l2m[r, sel].mean())
    st["oracle_min_l2_mean"] = float(l2m.min(1).mean())
    st["metrics_original_selection"] = evaluator_metrics(sc["l2_step"][v][r, sel], sc["col_step"][v][r, sel])
    return st


def _route_stats_job(args):
    l2g, loc, gt = args
    route, dev, nc = _W["routes"].route(loc, l2g, gt)
    return route is not None, dev, nc


def route_stats(config, split, dataroot, workers, route_kw=None):
    """덤프 없이 info 만으로(GT) 차선 경로를 찾는 비율·GT 평균 거리 통계. 미래 GT 완비 표본(평가기 규칙)만."""
    import importlib
    from mmcv import Config
    importlib.import_module("mmdet3d_plugin")
    from mmdet.datasets import build_dataset
    cfg = Config.fromfile(config)
    dcfg = cfg.data.test.copy(); dcfg.pipeline = None
    if split == "train":
        dcfg.ann_file = cfg.data.train.ann_file
    ds = build_dataset(dcfg)
    jobs, n_inc = [], 0
    for i, inf in enumerate(ds.data_infos):
        if not (np.asarray(inf["gt_ego_fut_masks"]) == 1).all():
            n_inc += 1; continue
        gt = np.asarray(inf["gt_ego_fut_trajs"], np.float32).cumsum(0)[:6, :2]
        jobs.append((ds._T_global(inf), inf["map_location"], gt))
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers, initializer=_worker_init, initargs=(dataroot, route_kw)) as pool:
        res = pool.map(_route_stats_job, jobs, chunksize=64)
    found = np.array([r[0] for r in res]); dev = np.array([r[1] for r in res]); nc = np.array([r[2] for r in res])
    return {"split": split, "ann_file": dcfg.ann_file, "n_complete_gt": len(jobs), "n_excluded_incomplete_gt": n_inc,
            "route_found_rate": float(found.mean()), "route_fallback_gt_count": int((~found).sum()),
            "no_candidate_count": int((nc == 0).sum()),
            "dev_too_large_count": int(((nc > 0) & ~found).sum()),
            "route_dev_median_found": float(np.median(dev[found])) if found.any() else None,
            "route_dev_p90_found": float(np.percentile(dev[found], 90)) if found.any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None)
    ap.add_argument("--out", required=True, help="subscores .npz (같은 이름 .json 에 요약) / --route_stats 면 .json")
    ap.add_argument("--tau", type=float, default=0.5, help="모방 soft target 온도(m)")
    ap.add_argument("--min_progress", type=float, default=PROGRESS_MIN_M)
    ap.add_argument("--progress", choices=["lane", "gt"], default="lane",
                    help="학습 라벨 ep 로 쓸 판: lane(차선 경로, 기본) / gt(종전 GT 경로, 통계 비교용). 두 판 모두 저장된다")
    ap.add_argument("--route_radius", type=float, default=5.0)
    ap.add_argument("--route_max_dev", type=float, default=2.0)
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--verify_evaluator", action="store_true",
                    help="전 표본·전 모드에서 평가기 shapely 충돌 판정과 SAT 판정이 같은지 대조(다르면 중단)")
    ap.add_argument("--route_stats", default=None, metavar="CONFIG", help="덤프 없이 차선 경로 통계만(info+GT)")
    ap.add_argument("--split", choices=["train", "val"], default="val")
    a = ap.parse_args()
    route_kw = {"radius": a.route_radius, "max_dev": a.route_max_dev}
    t0 = time.time()
    if a.route_stats:
        st = route_stats(a.route_stats, a.split, a.dataroot, a.workers, route_kw)
        st["route_params"] = route_kw; st["seconds"] = round(time.time() - t0, 1)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(st, open(a.out, "w"), indent=1, ensure_ascii=False)
        print(json.dumps(st, indent=1, ensure_ascii=False))
        return
    if not a.dump:
        raise SystemExit("--dump 또는 --route_stats 가 필요하다")
    dump = scorer_io.load_dump(a.dump)
    sc, stats = compute_all(dump, a.tau, a.dataroot, a.workers, a.verify_evaluator, a.min_progress, a.progress, route_kw)
    stats["dump"] = os.path.abspath(a.dump); stats["seconds"] = round(time.time() - t0, 1)
    if a.verify_evaluator and stats["verify_evaluator_mismatch_cells"] != 0:
        print(json.dumps(stats, indent=1, ensure_ascii=False))
        raise SystemExit(f"[중단] SAT 충돌 판정이 평가기와 {stats['verify_evaluator_mismatch_cells']} 칸 다르다 — 원인 조사 필요")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out[:-4] + ".tmp.npz"
    np.savez(tmp, **sc); os.replace(tmp, a.out)
    json.dump(stats, open(a.out[:-4] + ".json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
