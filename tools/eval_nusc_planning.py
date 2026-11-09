"""수정(2026-10-09): nuScenes open-loop 계획(행동) 평가 — HiP-AD Table 3 와 같은 프로토콜.

  근거: HiP-AD Table 3 의 SparseDrive-S 행(0.29/0.58/0.96, Col 0.01/0.05/0.18)이 SparseDrive 논문 값과 같다 →
  SparseDrive 공개 평가기(swc-17/SparseDrive fbadea6 projects/mmdet3d_plugin/datasets/evaluation/planning/planning_eval.py)
  를 계산식 그대로 옮긴다:
    - 시점별 L2(0.5s 간격 6점) → t 초까지 누적 평균, 1s/2s/3s = 2/4/6 번째, avg = 셋의 평균
    - 충돌: 자차 박스(4.084×1.85, UniAD 식 0.5m 전방 오프셋, 궤적 접선 yaw)가 미래 GT 박스와 겹치면 충돌.
      GT 궤적도 충돌하는 시점은 제외(obj_box_col). 같은 누적 평균.
    - 미래 GT 가 불완전한 표본(gt_ego_fut_masks 미완)은 제외.
  미래 박스는 NuScenesMiniDataset._get_fut_boxes(SparseDrive 와 동일 변환)를 쓴다. 예측 = results.pkl 의 plan_temp_2hz (6,2).
  사용: python tools/eval_nusc_planning.py <config> <results.pkl> [--out json]
"""
import argparse, json, pickle, sys
import numpy as np, torch
from shapely.geometry import Polygon
sys.path.insert(0, ".")
from mmcv import Config


def box3d_to_corners(box3d):  # mmdet3d_plugin/datasets/utils.py 와 동일(SparseDrive 동일) — import 부작용 회피용 사본
    YAW, X, Y, Z, W, L, H = 6, 0, 1, 2, 3, 4, 5
    if isinstance(box3d, torch.Tensor):
        box3d = box3d.detach().cpu().numpy()
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box3d[:, None, [W, L, H]] * corners_norm.reshape([1, 8, 3])
    rot_mat = np.eye(3)[None].repeat(len(box3d), 0)
    rot_cos, rot_sin = np.cos(box3d[:, YAW]), np.sin(box3d[:, YAW])
    rot_mat[:, 0, 0] = rot_cos; rot_mat[:, 0, 1] = -rot_sin
    rot_mat[:, 1, 0] = rot_sin; rot_mat[:, 1, 1] = rot_cos
    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
    corners += box3d[:, None, :3]
    return corners


def check_collision(ego_box, boxes):
    if boxes.shape[0] == 0:
        return False
    ego_box = ego_box.clone()
    ego_box[0] += 0.5 * torch.cos(ego_box[6])
    ego_box[1] += 0.5 * torch.sin(ego_box[6])
    ego_c = box3d_to_corners(ego_box.unsqueeze(0))[0, [0, 3, 7, 4], :2]
    cs = box3d_to_corners(boxes)[:, [0, 3, 7, 4], :2]
    ego_poly = Polygon([(p[0], p[1]) for p in ego_c])
    return any(ego_poly.intersects(Polygon([(p[0], p[1]) for p in c])) for c in cs)


def get_yaw(traj):
    if torch.linalg.norm(traj[-1] - traj[0], dim=-1) < 0.5:
        return traj.new_ones(traj.shape[0]) * np.pi / 2
    cat = torch.cat([traj.new_zeros((1, 2)), traj], 0)
    yaw = traj.new_zeros(traj.shape[0] + 1)
    yaw[1:-1] = torch.atan2(cat[2:, 1] - cat[:-2, 1], cat[2:, 0] - cat[:-2, 0])
    yaw[-1] = torch.atan2(cat[-1, 1] - cat[-2, 1], cat[-1, 0] - cat[-2, 0])
    return yaw[1:]


def single_coll(traj, fut_boxes):
    n = traj.shape[0]
    ego = traj.new_zeros((n, 7)); ego[:, :2] = traj
    ego[:, 3:6] = ego.new_tensor([4.084, 1.85, 1.56]); ego[:, 6] = get_yaw(traj)
    out = torch.zeros(n, dtype=torch.bool)
    for t in range(n):
        out[t] = check_collision(ego[t], torch.as_tensor(fut_boxes[t], dtype=torch.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("results"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import importlib; importlib.import_module("mmdet3d_plugin")
    from mmdet.datasets import build_dataset
    cfg = Config.fromfile(a.config)
    ds = build_dataset(cfg.data.test)
    # results.pkl 은 CUDA 텐서를 담고 있다 → GPU 를 쓰지 않도록 역직렬화를 CPU 로 매핑
    import functools
    _tl = torch.load
    torch.load = functools.partial(_tl, map_location="cpu", weights_only=False)
    try:
        res = pickle.load(open(a.results, "rb"))
    finally:
        torch.load = _tl
    assert len(res) == len(ds.data_infos), (len(res), len(ds.data_infos))
    L2 = torch.zeros(6); col = torch.zeros(6); box_col = torch.zeros(6); n = 0; skipped = 0
    for i, info in enumerate(ds.data_infos):
        m = np.asarray(info["gt_ego_fut_masks"], np.float32)
        if not (m == 1).all():
            skipped += 1; continue
        gt = torch.as_tensor(np.asarray(info["gt_ego_fut_trajs"], np.float32)).cumsum(0)[:6, :2].clone()
        r = res[i].get("img_bbox", res[i])
        pred = torch.as_tensor(r["plan_temp_2hz"], dtype=torch.float32)[:6, :2].clone()
        fut = ds._get_fut_boxes(i, info)
        assert len(fut) == 6, f"미래 박스 {len(fut)} != 6 (마스크 완비 표본) idx {i}"
        pred[:, 0] = -pred[:, 0]; gt[:, 0] = -gt[:, 0]          # SparseDrive update() 와 동일
        L2 += torch.sqrt(((pred - gt) ** 2).sum(-1))
        p2, g2 = pred * torch.tensor([-1., 1.]), gt * torch.tensor([-1., 1.])   # evaluate_coll() 의 재반전
        gc = single_coll(g2, fut); pc = single_coll(p2, fut)
        col += gc.float(); box_col += (pc & ~gc).float(); n += 1
    out = {}
    for k, v in (("L2", L2 / n), ("obj_col", col / n), ("obj_box_col", box_col / n)):
        cum = [float(v[:j + 1].mean()) for j in range(6)]
        out[k] = {"1s": cum[1], "2s": cum[3], "3s": cum[5], "avg": (cum[1] + cum[3] + cum[5]) / 3}
    out["n_eval"] = n; out["n_skipped_incomplete_gt"] = skipped
    print(f"표본 {n} (미래 GT 불완전 제외 {skipped})")
    print(f"L2 (m)        1s {out['L2']['1s']:.3f}  2s {out['L2']['2s']:.3f}  3s {out['L2']['3s']:.3f}  avg {out['L2']['avg']:.3f}")
    c = out["obj_box_col"]
    print(f"Collision (%) 1s {100*c['1s']:.3f}  2s {100*c['2s']:.3f}  3s {100*c['3s']:.3f}  avg {100*c['avg']:.3f}")
    print("HiP-AD Table 3: L2 0.28/0.53/0.87 avg 0.56 | Col 0.01/0.05/0.15 avg 0.07 (%)")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
