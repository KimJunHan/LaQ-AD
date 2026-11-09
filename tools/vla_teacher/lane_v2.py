# 수정(2026-10-15): 교사 VLM ① 정답 v2 (사용자 승인 4개 개선) — 개선 1: 차로변경 다중 신호.
#   nuScenes map-expansion v1.3(NuScenesMap) 의 lane / lane_connector 중심선(arcline, 0.5 m 이산화)과
#   위상(outgoing/incoming)으로 '인접 차로로의 이동'을 판정한다. 라벨 전용(미래 GT 사용 가능).
#   신호 3개(방향 'left'|'right'|None):
#     (a) topo : t0 차로 L0 의 도달 가능 집합(L0 + incoming 1단 + outgoing BFS 4단 — 교차로 lane_connector·
#                분기·후속 차로는 모두 여기 포함되므로 '차로변경 아님')에 속하지 않는 차로에 마지막 두 시점(t5,t6)이
#                모두 배정되고(배정 차로 중심선 횡오프셋 ≤ ASSIGN_MAX), 그 위치의 L0-체인 기준 횡오프셋이
#                ADJ_MIN..ADJ_MAX(=인접 차로 중심선 간격 범위) 이면 발화. 방향 = 체인 기준 오프셋 부호.
#     (b) divider : v1 규칙(미래 꺾은선이 map_annos 구분선을 가로지름, common.divider_crossing).
#     (c) kin  : 차로(체인) 기준 횡변위 Δd = d_ref(t6) − d_ref(t0), |Δd| ≥ HALF_LANE(1.6 m ≈ 3.2 m 차로폭의 절반)
#                이고, 자차 진행방향이 차로 방향 대비 같은 쪽으로 ≥ HEAD_MIN_DEG 기운 시점이 있으며, 현재 yaw rate
#                (CAN 또는 과거 포즈)가 반대 방향으로 > YAW_OPP(0.1 rad/s, v1 에서 찾은 '회전 꼬리' 구분값)이 아님.
#                체인 기준이 정의되지 않으면(지도 차로 없음) 지도 없는 대체: v1 원호 잔차 |r| ≥ 1.0 m(v1 비회전
#                p98=0.92) + 같은 yaw 가드.
#   판정: 같은 방향 ≥2 신호 → lane change; 1 신호(또는 방향 충돌) → uncertain(답은 go straight, 가중 0.3).
import numpy as np
from pyquaternion import Quaternion

ASSIGN_R = 4.0        # m, 후보 중심선 점 탐색 반경
ASSIGN_MAX = 2.0      # m, 차로 배정 허용 횡오프셋(차로폭 ~3–3.5 m 의 절반 + 여유)
HEAD_TOL_DEG = 40.0   # 자차 진행방향과 차로 방향 차 허용(역방향·교차 차로 배제)
ALONG_TOL = 0.6       # m, 체인 끝을 넘어선 투영(0.5 m 이산화 + 여유) 이면 기준 미정
ADJ_MIN, ADJ_MAX = 2.0, 6.0   # m, 인접 차로 중심선 간격 범위
HALF_LANE = 1.6       # m
HEAD_MIN_DEG = 3.0
YAW_OPP = 0.1         # rad/s
RESID_FALLBACK = 1.0  # m
BFS_DEPTH = 4


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class LaneIndex:
    """한 도시 지도의 차로 중심선 점 색인(KD-tree). map_root/maps/expansion/<loc>.json 을 한 번만 읽는다."""

    def __init__(self, map_root, location, res=0.5):
        from nuscenes.map_expansion.map_api import NuScenesMap
        from scipy.spatial import cKDTree
        self.m = NuScenesMap(dataroot=map_root, map_name=location)
        recs = self.m.lane + self.m.lane_connector
        self.is_conn = {r["token"]: False for r in self.m.lane}
        self.is_conn.update({r["token"]: True for r in self.m.lane_connector})
        disc = self.m.discretize_lanes([r["token"] for r in recs], res)
        self.tokens, pts, lid = [], [], []
        for k, (t, p) in enumerate(disc.items()):
            if len(p) == 0:
                continue
            self.tokens.append(t)
            pts.append(np.asarray(p, np.float64)); lid.append(np.full(len(p), len(self.tokens) - 1))
        self.pts = np.concatenate(pts); self.lid = np.concatenate(lid)
        self.tree = cKDTree(self.pts[:, :2])
        self.tok2i = {t: i for i, t in enumerate(self.tokens)}
        self._out = {}; self._inc = {}

    def outgoing(self, t):
        if t not in self._out:
            self._out[t] = [x for x in self.m.get_outgoing_lane_ids(t) if x in self.tok2i]
        return self._out[t]

    def incoming(self, t):
        if t not in self._inc:
            self._inc[t] = [x for x in self.m.get_incoming_lane_ids(t) if x in self.tok2i]
        return self._inc[t]

    def chain(self, t0):
        """L0 에서 도달 가능한(= 차로변경이 아닌) 차로 집합: L0, incoming 1단, outgoing BFS_DEPTH 단."""
        s = {t0} | set(self.incoming(t0))
        fr = [t0]
        for _ in range(BFS_DEPTH):
            nx = []
            for t in fr:
                for o in self.outgoing(t):
                    if o not in s:
                        s.add(o); nx.append(o)
            fr = nx
        return s

    def _cands(self, xy, psi, allowed=None):
        """반환 [(|d|, d, along, lane_token, rel_heading)] — 진행방향과 맞는 차로별 최근접 중심선 점."""
        idx = self.tree.query_ball_point(xy, ASSIGN_R if allowed is None else ADJ_MAX + 1.0)
        best = {}
        for j in idx:
            li = int(self.lid[j]); t = self.tokens[li]
            if allowed is not None and t not in allowed:
                continue
            qx, qy, qyaw = self.pts[j]
            rel = _wrap(psi - qyaw)
            if abs(np.degrees(rel)) > HEAD_TOL_DEG:
                continue
            dx, dy = xy[0] - qx, xy[1] - qy
            dist2 = dx * dx + dy * dy
            if t not in best or dist2 < best[t][0]:
                c, s = np.cos(qyaw), np.sin(qyaw)
                along = c * dx + s * dy
                d = c * dy - s * dx                 # + = 차로 진행방향 기준 왼쪽
                best[t] = (dist2, d, along, rel)
        return [(abs(v[1]), v[1], v[2], t, v[3]) for t, v in best.items()]

    def assign(self, xy, psi):
        c = [x for x in self._cands(xy, psi) if abs(x[2]) <= ALONG_TOL and x[0] <= ASSIGN_MAX]
        if not c:
            return None
        c.sort(key=lambda x: x[0])
        return c[0]

    def ref_offset(self, xy, psi, chain):
        c = [x for x in self._cands(xy, psi, allowed=chain) if abs(x[2]) <= ALONG_TOL]
        if not c:
            return None
        c.sort(key=lambda x: x[0])
        return c[0]


def lidar_to_global(info, xy_lidar):
    """(N,2) 현재 lidar 프레임 변위(converter: 미래 lidar 원점의 현재 lidar 프레임 좌표) → 전역 xy, 그리고
    lidar +y(전방) 축의 전역 방위각 φ0."""
    R_l2e = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    R_e2g = Quaternion(info["ego2global_rotation"]).rotation_matrix
    Rg = R_e2g @ R_l2e
    o = R_e2g @ np.asarray(info["lidar2ego_translation"]) + np.asarray(info["ego2global_translation"])
    p = np.concatenate([xy_lidar, np.zeros((len(xy_lidar), 1))], 1) @ Rg.T + o
    phi0 = float(np.arctan2(Rg[1, 1], Rg[0, 1]))
    return p[:, :2], phi0, Rg, o


def lane_signals(li, info, xy, theta, yaw_rate0, divider_dir, lat_resid):
    """xy: (7,2) lidar 프레임 경로(원점 포함), theta: (7,) heading_series(+ = 왼쪽).
    yaw_rate0: 현재 yaw rate(+ 왼쪽) 또는 None. divider_dir: 신호 (b). 반환 dict."""
    out = dict(a=None, b=divider_dir, c=None, L0=None, L6=None, L0_is_connector=None,
               d_ref=None, rel_head_deg=None, c_mode=None)
    g, phi0, _, _ = lidar_to_global(info, xy)
    psi = phi0 + theta
    yaw_opp = (lambda side: yaw_rate0 is not None and
               ((side == "left" and yaw_rate0 < -YAW_OPP) or (side == "right" and yaw_rate0 > YAW_OPP)))
    a0 = li.assign(g[0], psi[0]) if li is not None else None
    if a0 is not None:
        L0 = a0[3]; ch = li.chain(L0)
        out["L0"] = L0; out["L0_is_connector"] = li.is_conn[L0]
        refs = [li.ref_offset(g[t], psi[t], ch) for t in range(len(g))]
        out["d_ref"] = [None if r is None else round(r[1], 3) for r in refs]
        out["rel_head_deg"] = [None if r is None else round(float(np.degrees(r[4])), 2) for r in refs]
        asg = [li.assign(g[t], psi[t]) for t in (5, 6)]
        out["L6"] = None if asg[1] is None else asg[1][3]
        # (a) 위상
        r6 = refs[6]
        if all(x is not None and x[3] not in ch for x in asg) and r6 is not None and ADJ_MIN <= abs(r6[1]) <= ADJ_MAX:
            out["a"] = "left" if r6[1] > 0 else "right"
        # (c) 차로 기준 운동학
        if refs[0] is not None and r6 is not None:
            out["c_mode"] = "lane"
            dd = r6[1] - refs[0][1]
            side = "left" if dd > 0 else "right"
            rh = [r[4] for r in refs if r is not None]
            toward = any((np.degrees(h) >= HEAD_MIN_DEG) if side == "left" else (np.degrees(h) <= -HEAD_MIN_DEG) for h in rh)
            if abs(dd) >= HALF_LANE and toward and not yaw_opp(side):
                out["c"] = side
    if out["c_mode"] is None:
        out["c_mode"] = "resid"   # 지도 차로 기준 없음 → 지도 없는 대체(v1 원호 잔차, +x = 오른쪽)
        side = "right" if lat_resid > 0 else "left"
        if abs(lat_resid) >= RESID_FALLBACK and not yaw_opp(side):
            out["c"] = side
    votes = [v for v in (out["a"], out["b"], out["c"]) if v is not None]
    nl, nr = votes.count("left"), votes.count("right")
    out["n_left"], out["n_right"] = nl, nr
    if max(nl, nr) >= 2 and min(nl, nr) == 0:
        out["decision"] = "lane change left" if nl > nr else "lane change right"
    elif votes:
        out["decision"] = "uncertain"
    else:
        out["decision"] = None
    return out
