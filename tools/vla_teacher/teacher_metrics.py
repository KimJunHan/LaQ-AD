# 수정(2026-10-15): 교사 VLM ② LoRA SFT — 교사 출력 채점(형식 무관). eval_teacher.py 와 smoke_sft_cpu.py 가 쓴다.
#   정답(GT)은 정답 텍스트를 다시 파싱하지 않고 레코드의 targets(구조화 값)에서 읽는다 → v1/v2 문법과 무관.
#   예측은 문법 파서(자동 선택: GT 정답을 ≥99% 파싱하는 parse_answer*)로 읽는다.
#   계획(행동) 지표는 tools/eval_nusc_planning.py(SparseDrive 공개 평가기 식)의 계산을 그대로 쓴다:
#     시점별 L2 → t초까지 누적 평균(1s/2s/3s = 2/4/6번째), 충돌 = obj_box_col(GT 궤적도 충돌하는 시점 제외),
#     미래 GT 불완전 표본 제외, 미래 박스 = 테스트 데이터셋 기본 규칙(valid_flag 있으면 그것, 없으면 num_lidar_pts>0).
import glob
import importlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SECTORS = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"]
LON = ["accelerate", "keep speed", "decelerate", "stop"]
LAT = ["go straight", "turn left", "turn right", "lane change left", "lane change right"]
CMD2LAT = {"go straight": "go straight", "turn left": "turn left", "turn right": "turn right"}


# ---------------------------------------------------------------- 파서 선택
def _candidates():
    out = []
    files = sorted(glob.glob(str(HERE / "common*.py"))) + sorted(glob.glob(str(HERE / "grammar*.py")))
    for f in files:
        name = Path(f).stem
        try:
            m = importlib.import_module(name)
        except Exception as e:  # 후보 모듈 import 실패는 기록하고 다음 후보(선택 결과에 사유 남김)
            out.append((f"{name}:<import 실패 {type(e).__name__}: {e}>", None))
            continue
        for a in sorted(dir(m)):
            if a.startswith("parse_answer") and callable(getattr(m, a)):
                out.append((f"{name}:{a}", getattr(m, a)))
    return out


def select_parser(spec, records, n=200, need=0.99):
    """spec='module:function' 이면 그것. 아니면 tools/vla_teacher/{common*,grammar*}.py 의 parse_answer* 중
    GT 정답 n 개를 need 이상 파싱하는 첫 후보. 반환 (이름, 함수, 후보별 GT 파싱률)."""
    if spec:
        mod, fn = spec.split(":")
        f = getattr(importlib.import_module(mod), fn)
        return spec, f, {}
    rates = {}
    sample = records[:n]
    for name, f in _candidates():
        if f is None:
            rates[name] = None
            continue
        ok = 0
        for r in sample:
            try:
                f(r["answer"]); ok += 1
            except Exception:
                pass
        rates[name] = ok / max(len(sample), 1)
        if rates[name] >= need:
            return name, f, rates
    raise RuntimeError(f"GT 정답을 파싱하는 parser 가 없다: {rates} — --parser module:function 으로 지정할 것")


# ---------------------------------------------------------------- 정규화
def _signed_xy(o):
    """(+x 오른쪽, +y 전방) m. v2 = 부호 있는 lat/lon, v1 = 크기 + lat_side/lon_side. 없으면 None."""
    if o.get("lat") is None or o.get("lon") is None:
        return None
    x, y = float(o["lat"]), float(o["lon"])
    if "lat_side" in o:
        x = abs(x) * (1 if o["lat_side"] == "right" else -1)
    if "lon_side" in o:
        y = abs(y) * (1 if o["lon_side"] == "ahead" else -1)
    return x, y


def norm_obj(o, dist_from_xy=False):
    """예측 객체 정규화. 거리: 텍스트에 거리가 있으면(v1) 그 값, 없으면(v2) hypot(lat, lon)."""
    name = o.get("name", o.get("class"))
    xy = _signed_xy(o)
    dist = o.get("dist", o.get("distance"))
    if dist is None or dist_from_xy:
        if xy is None:
            raise ValueError(f"거리도 lat/lon 도 없는 객체: {o}")
        dist = float(np.hypot(*xy))
    vs = o.get("vel_sector")
    sp = o.get("speed")
    moving = vs is not None if "vel_sector" in o else (sp is not None and sp > 0)
    return dict(name=str(name).split(".")[-1], sector=o.get("sector"), dist=float(dist), xy=xy,
                moving=bool(moving), speed=float(sp) if (moving and sp is not None) else 0.0)


def norm_gt_obj(o, dist_from_xy=False):
    """GT 객체(targets). dist_from_xy=True 면 예측과 같은 방식(반올림 lat/lon 의 hypot)으로 거리 계산 —
    v2 처럼 텍스트에 거리가 없을 때 반올림 차이(≤0.07 m)가 오차로 잡히지 않게 한다."""
    moving = o.get("vel_sector") is not None
    xy = _signed_xy(o)
    if xy is None and o.get("x") is not None:
        xy = (round(float(o["x"]), 1), round(float(o["y"]), 1))
    dist = float(np.hypot(*xy)) if (dist_from_xy and xy is not None) else float(o["dist"])
    return dict(name=o["name"], sector=o["sector"], dist=dist, xy=xy, moving=moving,
                speed=float(o["speed"]) if moving else 0.0)


# ---------------------------------------------------------------- 인지
def match_objects(pred, gt, tol_abs=2.0, tol_rel=0.1):
    """일대일 매칭: 같은 클래스·같은 방위·|Δd| ≤ max(tol_abs, tol_rel·d_gt). 비용 |Δd| 헝가리안. 반환 [(pi, gi)]."""
    from scipy.optimize import linear_sum_assignment
    if not pred or not gt:
        return []
    BIG = 1e6
    C = np.full((len(pred), len(gt)), BIG)
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            if p["name"] == g["name"] and p["sector"] == g["sector"]:
                e = abs(p["dist"] - g["dist"])
                if e <= max(tol_abs, tol_rel * g["dist"]):
                    C[i, j] = e
    r, c = linear_sum_assignment(C)
    return [(i, j) for i, j in zip(r, c) if C[i, j] < BIG]


class PerceptionAcc:
    def __init__(self, tol_abs=2.0, tol_rel=0.1, order_margin=0.5):
        self.tol_abs, self.tol_rel, self.margin = tol_abs, tol_rel, order_margin
        self.pos_err = []
        self.n_gt = self.n_pred = self.n_match = 0
        self.d_err, self.v_err, self.mov_ok = [], [], []
        self.ord_ok = self.ord_n = 0
        self.ego_err = []
        self.count_err = []
        self.n_samples = 0

    def add(self, pred_objs, gt_objs, pred_ego=None, gt_ego=None):
        P0 = pred_objs or []
        dxy = bool(P0) and all("dist" not in o for o in P0)
        P = [norm_obj(o) for o in P0]
        G = [norm_gt_obj(o, dist_from_xy=dxy) for o in gt_objs]
        self.n_samples += 1
        self.n_gt += len(G); self.n_pred += len(P)
        self.count_err.append(abs(len(P) - len(G)))
        M = match_objects(P, G, self.tol_abs, self.tol_rel)
        self.n_match += len(M)
        for i, j in M:
            self.d_err.append(abs(P[i]["dist"] - G[j]["dist"]))
            self.v_err.append(abs(P[i]["speed"] - G[j]["speed"]))
            self.mov_ok.append(P[i]["moving"] == G[j]["moving"])
            if P[i]["xy"] is not None and G[j]["xy"] is not None:
                self.pos_err.append(float(np.hypot(P[i]["xy"][0] - G[j]["xy"][0], P[i]["xy"][1] - G[j]["xy"][1])))
        for a in range(len(M)):
            for b in range(a + 1, len(M)):
                (pa, ga), (pb, gb) = M[a], M[b]
                dg = G[ga]["dist"] - G[gb]["dist"]
                if abs(dg) < self.margin:
                    continue
                dp = P[pa]["dist"] - P[pb]["dist"]
                self.ord_n += 1
                self.ord_ok += int(np.sign(dp) == np.sign(dg))
        if pred_ego is not None and gt_ego is not None:
            self.ego_err.append(abs(pred_ego - gt_ego))

    def result(self):
        rec = self.n_match / max(self.n_gt, 1)
        prec = self.n_match / max(self.n_pred, 1)
        f = lambda x: float(np.mean(x)) if len(x) else None  # noqa: E731
        return dict(n_samples=self.n_samples, n_gt=self.n_gt, n_pred=self.n_pred, n_match=self.n_match,
                    recall=rec, precision=prec, f1=(2 * rec * prec / (rec + prec)) if rec + prec else 0.0,
                    dist_mae=f(self.d_err), pos_err_mean=f(self.pos_err), speed_mae=f(self.v_err), moving_state_acc=f(self.mov_ok),
                    depth_order_acc=(self.ord_ok / self.ord_n) if self.ord_n else None, depth_order_pairs=self.ord_n,
                    ego_speed_mae=f(self.ego_err), count_mae=f(self.count_err),
                    match_rule=f"same class & sector & |dd|<=max({self.tol_abs},{self.tol_rel}*d_gt), Hungarian on |dd|")


# ---------------------------------------------------------------- 지도
def _pc_state(pc):
    if pc is None:
        return "none"
    return "here" if pc.get("sector") == "here" else pc.get("sector")


class MapAcc:
    def __init__(self):
        self.rows = []

    def add(self, pred, gt):
        if gt is None:
            return
        if pred is None:
            self.rows.append(dict(valid=False))
            return
        sp, sg = _pc_state(pred.get("ped_crossing")), _pc_state(gt.get("ped_crossing"))
        r = dict(valid=True, state=sp == sg, presence=(sp == "none") == (sg == "none"),
                 div=pred.get("n_divider") == gt.get("n_divider"), bnd=pred.get("n_boundary") == gt.get("n_boundary"),
                 div_err=abs(pred.get("n_divider", 0) - gt.get("n_divider", 0)),
                 bnd_err=abs(pred.get("n_boundary", 0) - gt.get("n_boundary", 0)))
        if sp not in ("none", "here") and sg not in ("none", "here"):
            r["pc_dist_err"] = abs(pred["ped_crossing"]["dist"] - gt["ped_crossing"]["dist"])
        same_dist = True
        if sg not in ("none", "here"):
            same_dist = r.get("pc_dist_err", 1e9) < 0.05
        r["exact"] = r["state"] and r["div"] and r["bnd"] and same_dist
        self.rows.append(r)

    def result(self):
        n = len(self.rows)
        g = lambda k: float(np.mean([r.get(k, False) for r in self.rows])) if n else None  # noqa: E731
        de = [r["pc_dist_err"] for r in self.rows if "pc_dist_err" in r]
        return dict(n=n, line_valid=g("valid"), line_exact=g("exact"), ped_state_acc=g("state"),
                    ped_presence_acc=g("presence"), ped_dist_mae=float(np.mean(de)) if de else None,
                    divider_acc=g("div"), boundary_acc=g("bnd"),
                    divider_mae=float(np.mean([r.get("div_err", 0) for r in self.rows if r["valid"]])) if n else None,
                    boundary_mae=float(np.mean([r.get("bnd_err", 0) for r in self.rows if r["valid"]])) if n else None)


# ---------------------------------------------------------------- 의도
class IntentAcc:
    """GT 'unavailable'(미래 불완전) 제외. 예측 없음/형식 오류 = 'invalid'(오답)."""

    def __init__(self):
        self.rows = []

    def add(self, pred, gt, uncertain=False):
        if gt is None:
            return
        pl = pred.get("longitudinal") if pred else "invalid"
        pa = pred.get("lateral") if pred else "invalid"
        self.rows.append((pl, pa, gt["longitudinal"], gt["lateral"], bool(uncertain)))

    @staticmethod
    def _f1(pairs, classes):
        f1s, rec = [], {}
        for c in classes:
            tp = sum(1 for p, g in pairs if p == c and g == c)
            fp = sum(1 for p, g in pairs if p == c and g != c)
            fn = sum(1 for p, g in pairs if p != c and g == c)
            if tp + fn == 0:
                continue
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn)
            rec[c] = dict(recall=rc, support=tp + fn)
            f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        return float(np.mean(f1s)) if f1s else None, rec

    def _res(self, rows):
        n = len(rows)
        if not n:
            return dict(n=0)
        lon = [(r[0], r[2]) for r in rows]
        lat = [(r[1], r[3]) for r in rows]
        lf1, lrec = self._f1(lon, LON)
        af1, arec = self._f1(lat, LAT)
        return dict(n=n, lon_acc=float(np.mean([p == g for p, g in lon])), lat_acc=float(np.mean([p == g for p, g in lat])),
                    joint_acc=float(np.mean([r[0] == r[2] and r[1] == r[3] for r in rows])),
                    lon_macro_f1=lf1, lat_macro_f1=af1, lon_per_class=lrec, lat_per_class=arec,
                    invalid=int(sum(r[0] == "invalid" for r in rows)))

    def result(self):
        out = self._res(self.rows)
        if any(r[4] for r in self.rows):
            out["certain_only"] = self._res([r for r in self.rows if not r[4]])
        return out


# ---------------------------------------------------------------- 계획(행동) — tools/eval_nusc_planning.py 와 같은 식
def load_planning_eval():
    spec = importlib.util.spec_from_file_location("eval_nusc_planning", str(REPO / "tools/eval_nusc_planning.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _T_global(info):
    from pyquaternion import Quaternion

    def rt(q, t):
        T = np.eye(4); T[:3, :3] = Quaternion(q).rotation_matrix; T[:3, 3] = np.asarray(t); return T
    return rt(info["ego2global_rotation"], info["ego2global_translation"]) @ \
        rt(info["lidar2ego_rotation"], info["lidar2ego_translation"])


def fut_boxes(infos, index, valid_rule="dataset_default"):
    """models/loader/nuscenes/dataset.py NuScenes3DDataset._get_fut_boxes 와 같은 변환.
    valid_rule='dataset_default' = use_valid_flag=None(테스트 config 기본): valid_flag 있으면 그것, 없으면 num_lidar_pts>0."""
    info = infos[index]
    fut_ts = int(np.asarray(info["gt_ego_fut_masks"]).sum())
    out, cur_tok, cur_T = [], info["scene_token"], _T_global(info)
    for i in range(1, fut_ts + 1):
        if index + i >= len(infos):
            break
        fi = infos[index + i]
        if fi["scene_token"] != cur_tok:
            break
        if valid_rule == "dataset_default":
            m = fi["valid_flag"] if "valid_flag" in fi else (fi["num_lidar_pts"] > 0)
        elif valid_rule == "num_lidar_pts":
            m = fi["num_lidar_pts"] > 0
        else:
            raise ValueError(valid_rule)
        fb = np.asarray(fi["gt_boxes"], np.float32)[m].copy()
        if len(fb) == 0:
            out.append(fb); continue
        T = np.linalg.inv(cur_T) @ _T_global(fi)
        fb[:, :3] = fb[:, :3] @ T[:3, :3].T + T[:3, 3]
        yaw = np.stack([np.cos(fb[:, 6]), np.sin(fb[:, 6])], -1) @ T[:2, :2].T
        fb[:, 6] = np.arctan2(yaw[..., 1], yaw[..., 0])
        out.append(fb)
    return out


class PlanAcc:
    """L2 / obj_box_col 누적 평균(SparseDrive). add(pred(6,2) | None, gt(6,2), fut_boxes, gt_col_cache)."""

    def __init__(self, pe):
        import torch
        self.torch, self.pe = torch, pe
        self.L2 = torch.zeros(6); self.col = torch.zeros(6); self.box_col = torch.zeros(6)
        self.n = 0; self.invalid = 0

    def add(self, pred, gt, fut, gt_col=None):
        torch = self.torch
        if pred is None:
            self.invalid += 1
            return gt_col
        p = torch.as_tensor(np.asarray(pred, np.float32))[:6, :2].clone()
        g = torch.as_tensor(np.asarray(gt, np.float32))[:6, :2].clone()
        p[:, 0] = -p[:, 0]; g[:, 0] = -g[:, 0]
        self.L2 += torch.sqrt(((p - g) ** 2).sum(-1))
        p2, g2 = p * torch.tensor([-1., 1.]), g * torch.tensor([-1., 1.])
        if gt_col is None:
            gt_col = self.pe.single_coll(g2, fut)
        pc = self.pe.single_coll(p2, fut)
        self.col += gt_col.float(); self.box_col += (pc & ~gt_col).float(); self.n += 1
        return gt_col

    def result(self):
        out = dict(n_eval=self.n, n_invalid_action=self.invalid)
        if not self.n:
            return out
        for k, v in (("L2", self.L2 / self.n), ("obj_box_col", self.box_col / self.n)):
            cum = [float(v[:j + 1].mean()) for j in range(6)]
            out[k] = {"1s": cum[1], "2s": cum[3], "3s": cum[5], "avg": (cum[1] + cum[3] + cum[5]) / 3}
        return out


# ---------------------------------------------------------------- 복사 기준선(지름길 정량화)
def copy_baseline(rec, codebook=None):
    """레코드에 생성기가 만든 copy-baseline 이 있으면 그것을 쓰고, 없으면 계산:
    의도 = (keep speed, 명령→횡방향), 행동 = 현재 자차 속도로 직진 등속(lidar: +y 전방), 0.5 s × 6.
    반환 (intent dict, traj (6,2), 출처)."""
    cb = None
    for k in ("copy_baseline", "copy_pred", "baseline_copy"):
        if k in rec:
            cb = rec[k]; break
    if cb is None:
        cb = rec.get("targets", {}).get("copy_baseline")
    if isinstance(cb, dict):
        it = cb.get("intent")
        if isinstance(it, str) and ", " in it:
            lo, la = it.rstrip(".").split(", ", 1)
            it = dict(longitudinal=lo, lateral=la)
        traj = None
        for k in ("traj", "fut_traj", "trajectory", "action_traj"):
            if cb.get(k) is not None:
                traj = np.asarray(cb[k], np.float64); break
        if traj is None and cb.get("action_tokens") is not None and codebook is not None:
            traj = codebook.decode(cb["action_tokens"])
        if isinstance(it, dict) and traj is not None:
            return it, traj, "record"
        raise ValueError(f"{rec['token']}: copy_baseline 구조 해석 불가(keys={list(cb.keys())}) — teacher_metrics.copy_baseline 갱신 필요")
    cmd = rec["command"]["text"] if isinstance(rec["command"], dict) else str(rec["command"])
    es = rec.get("ego_status", {})
    v = float(es.get("speed", 0.0)) if es.get("available", True) else 0.0
    traj = np.stack([np.zeros(6), v * 0.5 * np.arange(1, 7)], 1)
    return dict(longitudinal="keep speed", lateral=CMD2LAT[cmd]), traj, "computed"
