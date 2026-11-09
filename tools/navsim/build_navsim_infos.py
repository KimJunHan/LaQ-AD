"""수정(2026-10-09): LaQ-AD NAVSIM — OpenScene 로그(navsim_logs/<split>/*.pkl) → HiP-AD 프레임 info pkl.

실행 env: hipad_bw (devkit import 없음 — 로그 pkl 은 numpy/dict 이다). 지도 GT·라이다 변환·자차 제원은 nuplan-devkit 이
필요하므로 tools/navsim/navsim_devkit_steps.py 가 이 산출물에 덧붙인다(별도 env).

devkit 근거(/home1/irteam/data-vol1/src/navsim_ref @0a380a9):
  - 장면 구성: navsim/common/dataloader.py:16-61 filter_scenes — 로그를 num_frames(=history+future) 창으로 frame_interval
    간격 분할(:24-26), 창 길이 미달 제외(:49), has_route 이면 현재 프레임 roadblock_ids 비면 제외(:53),
    장면 token = frame_list[num_history-1]['token'](:57), 같은 token 은 dict 로 덮어씀(:61).
  - 프레임 키(dataclasses.py 가 읽는 것): token, timestamp, cams, lidar_path, ego2global_translation/rotation(:176-177),
    ego_dynamic_state, driving_command(:144,:204), roadblock_ids, map_location(:494), anns{gt_boxes, gt_names,
    gt_velocity_3d, instance_tokens, track_tokens}(Annotations :261-268).
  - 카메라: cams[CAM_F0..CAM_B0]{data_path, sensor2lidar_rotation, sensor2lidar_translation, cam_intrinsic, distortion},
    영상 경로 = sensor_blobs/<split> / data_path (dataclasses.py:79).
  - 계획 출력: Trajectory(time_horizon=4, interval_length=0.5) = 8 포즈, ego rear-axle 국소좌표(dataclasses.py:283-284,358).
  - 자차 포즈: ego2global = rear-axle 포즈(로컬 궤적이 ego2global 로부터 rear-axle 좌표로 계산됨, dataclasses.py:174-190,358).

모델 좌표계: nuScenes 와 같게 "라이다 프레임"에서 학습한다(투영 = K @ inv(sensor2lidar)). 따라서 lidar2ego 가 필요하다.
[데이터로 확정해야 하는 것 — 아니면 즉시 실패]
  (1) 로그 프레임에 lidar2ego_translation / lidar2ego_rotation 키가 있는가(devkit 은 읽지 않아 소스로 검증 불가).
  (2) anns.gt_boxes 가 라이다 프레임인가 ego 프레임인가 → `--probe` 로 두 가설을 카메라 투영으로 대조한 뒤
      `--anno-frame lidar|ego` 를 명시해야 변환한다(기본값 없음).

산출 info(프레임 단위, 시퀀스 순서로 정렬):
  token, scene_token(시간 뱅크 시퀀스 id), log_name, timestamp(us), map_location,
  cams{CAM: data_path(절대), sensor2lidar_rotation, sensor2lidar_translation, cam_intrinsic},
  lidar_path(원본 .pcd; navsim_devkit_steps.py lidar 가 .bin 으로 교체), lidar2ego(4x4), ego2global(4x4),
  ego_status(4: vx, vy, ax, ay — ego_dynamic_state), driving_command(4, one-hot: 좌/직/우/미상),
  gt_boxes(N,7: x,y,z,l,w,h,yaw — 라이다), gt_names, gt_velocity(N,2 — 라이다), instance_inds(N),
  gt_agent_fut_trajs(N,12,2 증분)/masks(N,12), gt_ego_fut_trajs(8,2 증분 — 라이다, rear-axle 경로)/masks(8),
  is_eval_frame(bool), navsim_token(평가 장면 token 또는 None).
"""
import argparse
import os
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import yaml
from pyquaternion import Quaternion

NAVSIM_CAMS = ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0")
EGO_FUT_TS = 8      # 4 s @ 2 Hz (Trajectory 기본 샘플링)
AGENT_FUT_TS = 12   # 6 s @ 2 Hz (nuScenes 모션 지평과 동일)
SEQ_GAP_S = 0.75    # 2 Hz(0.5 s) 연속 프레임만 한 시퀀스로 묶는다


# ---------------------------------------------------------------- scene filter (devkit 동일)
def load_scene_filter(yaml_path):
    cfg = yaml.safe_load(open(yaml_path))
    return dict(num_history=int(cfg["num_history_frames"]), num_future=int(cfg["num_future_frames"]),
                frame_interval=int(cfg["frame_interval"]), has_route=bool(cfg["has_route"]),
                max_scenes=cfg.get("max_scenes"), log_names=cfg.get("log_names"), tokens=cfg.get("tokens"))


def iter_logs(log_dir, sf):
    files = sorted(Path(log_dir).iterdir())
    if sf["log_names"] is not None:
        names = set(sf["log_names"])
        files = [f for f in files if f.name.replace(".pkl", "") in names]
    for f in files:
        yield f.name.replace(".pkl", ""), pickle.load(open(f, "rb"))


def scenes_of_log(frames, sf):
    """devkit filter_scenes 와 같은 규칙. 반환 {scene_token: 창 시작 인덱스} (같은 token 은 뒤가 덮어씀, :61)."""
    num_frames = sf["num_history"] + sf["num_future"]
    want = set(sf["tokens"]) if sf["tokens"] is not None else None
    out = OrderedDict()
    for i in range(0, len(frames), sf["frame_interval"]):
        fl = frames[i:i + num_frames]
        if len(fl) < num_frames:
            continue
        cur = fl[sf["num_history"] - 1]
        if sf["has_route"] and len(cur["roadblock_ids"]) == 0:
            continue
        if want is not None and cur["token"] not in want:
            continue
        out[cur["token"]] = i
    return out


# ---------------------------------------------------------------- geometry
def rt(rot_wxyz, trans):
    T = np.eye(4)
    T[:3, :3] = Quaternion(*rot_wxyz).rotation_matrix
    T[:3, 3] = np.asarray(trans, dtype=np.float64)
    return T


def lidar2ego_of(fr):
    if "lidar2ego_rotation" not in fr or "lidar2ego_translation" not in fr:
        raise KeyError("로그 프레임에 lidar2ego_rotation/lidar2ego_translation 이 없다 — 모델이 라이다 프레임에서 "
                       f"학습하려면 필요하다. 실제 키: {sorted(fr.keys())}. --probe 로 확인 후 변환 경로를 정할 것.")
    return rt(fr["lidar2ego_rotation"], fr["lidar2ego_translation"])


def ego2global_of(fr):
    return rt(fr["ego2global_rotation"], fr["ego2global_translation"])


def apply(T, pts):
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


def boxes_to_lidar(boxes, vel, anno_frame, l2e):
    """boxes (N,7) x,y,z,l,w,h,yaw / vel (N,3) 를 라이다 프레임으로."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7).copy()
    vel = np.asarray(vel, dtype=np.float64).reshape(-1, 3).copy()
    if anno_frame == "lidar":
        return boxes, vel
    if anno_frame != "ego":
        raise ValueError(anno_frame)
    e2l = np.linalg.inv(l2e)
    boxes[:, :3] = apply(e2l, boxes[:, :3])
    d = np.stack([np.cos(boxes[:, 6]), np.sin(boxes[:, 6]), np.zeros(len(boxes))], 1) @ e2l[:3, :3].T
    boxes[:, 6] = np.arctan2(d[:, 1], d[:, 0])
    vel = vel @ e2l[:3, :3].T
    return boxes, vel


def box_frame_to_global(anno_frame, l2e, e2g):
    return e2g @ l2e if anno_frame == "lidar" else e2g


# ---------------------------------------------------------------- per-frame info
def frame_info(frames, idx, log, sensor_root, anno_frame, track_ids):
    fr = frames[idx]
    l2e, e2g = lidar2ego_of(fr), ego2global_of(fr)
    l2g = e2g @ l2e
    g2l = np.linalg.inv(l2g)

    cams = {}
    for cam in NAVSIM_CAMS:
        c = fr["cams"][cam]
        cams[cam] = dict(data_path=os.path.join(sensor_root, c["data_path"]),
                         sensor2lidar_rotation=np.asarray(c["sensor2lidar_rotation"], np.float64),
                         sensor2lidar_translation=np.asarray(c["sensor2lidar_translation"], np.float64),
                         cam_intrinsic=np.asarray(c["cam_intrinsic"], np.float64))

    a = fr["anns"]
    boxes, vel = boxes_to_lidar(a["gt_boxes"], a["gt_velocity_3d"], anno_frame, l2e)
    tracks = list(a["track_tokens"])
    for t in tracks:
        track_ids.setdefault(t, len(track_ids))
    inst = np.array([track_ids[t] for t in tracks], np.int64)

    # 에이전트 미래 궤적(모션 GT): 같은 track_token 의 미래 중심을 현재 라이다 프레임으로, nuScenes 와 같은 증분 형식.
    n = len(boxes)
    fut = np.zeros((n, AGENT_FUT_TS, 2), np.float32)
    fmask = np.zeros((n, AGENT_FUT_TS), np.float32)
    nxt = []
    for k in range(1, AGENT_FUT_TS + 1):
        if idx + k >= len(frames) or frames[idx + k]["timestamp"] - frames[idx + k - 1]["timestamp"] > SEQ_GAP_S * 1e6:
            break
        f2 = frames[idx + k]
        T = g2l @ box_frame_to_global(anno_frame, lidar2ego_of(f2), ego2global_of(f2))
        pos = {t: apply(T, b[None, :3])[0, :2] for t, b in zip(f2["anns"]["track_tokens"],
                                                             np.asarray(f2["anns"]["gt_boxes"]).reshape(-1, 7))}
        nxt.append(pos)
    for i, t in enumerate(tracks):
        prev = boxes[i, :2]
        for k, pos in enumerate(nxt):
            if t not in pos:
                break          # nuScenes 와 같이 연속 구간만 유효(이후 마스크 0)
            fut[i, k] = pos[t] - prev
            fmask[i, k] = 1.0
            prev = pos[t]

    # 자차 미래 궤적: rear-axle(=ego2global 원점) 위치를 현재 라이다 프레임으로, p0(현재 rear-axle) 기준 증분.
    p0 = apply(g2l, e2g[None, :3, 3])[0]
    ego = np.zeros((EGO_FUT_TS, 2), np.float32)
    emask = np.zeros(EGO_FUT_TS, np.float32)
    prev = p0
    for k in range(1, EGO_FUT_TS + 1):
        if idx + k >= len(frames) or frames[idx + k]["timestamp"] - frames[idx + k - 1]["timestamp"] > SEQ_GAP_S * 1e6:
            break
        p = apply(g2l, ego2global_of(frames[idx + k])[None, :3, 3])[0]
        ego[k - 1] = (p - prev)[:2]
        emask[k - 1] = 1.0
        prev = p

    return dict(
        token=fr["token"], log_name=log, timestamp=int(fr["timestamp"]), map_location=fr["map_location"],
        cams=cams, lidar_path=os.path.join(sensor_root, fr["lidar_path"]),
        lidar2ego=l2e, ego2global=e2g,
        ego_status=np.asarray(fr["ego_dynamic_state"], np.float32).reshape(4),
        driving_command=np.asarray(fr["driving_command"], np.float32).reshape(4),
        gt_boxes=boxes.astype(np.float32), gt_names=np.asarray(a["gt_names"]),
        gt_velocity=vel[:, :2].astype(np.float32), instance_inds=inst,
        gt_agent_fut_trajs=fut, gt_agent_fut_masks=fmask,
        gt_ego_fut_trajs=ego, gt_ego_fut_masks=emask,
        is_eval_frame=False, navsim_token=None,
    )


# ---------------------------------------------------------------- probe
def probe(log_dir, sensor_root):
    from PIL import Image
    f = sorted(Path(log_dir).iterdir())[0]
    frames = pickle.load(open(f, "rb"))
    fr = frames[len(frames) // 2]
    print("frame keys:", sorted(fr.keys()))
    print("cam keys  :", sorted(fr["cams"]["CAM_F0"].keys()))
    print("anns keys :", sorted(fr["anns"].keys()))
    has_l2e = "lidar2ego_rotation" in fr and "lidar2ego_translation" in fr
    print("lidar2ego present:", has_l2e)
    c = fr["cams"]["CAM_F0"]
    W, H = Image.open(os.path.join(sensor_root, c["data_path"])).size
    print("CAM_F0 image size (W,H):", (W, H))
    s2l = np.eye(4); s2l[:3, :3] = c["sensor2lidar_rotation"]; s2l[:3, 3] = c["sensor2lidar_translation"]
    K = np.asarray(c["cam_intrinsic"])
    centers = np.asarray(fr["anns"]["gt_boxes"]).reshape(-1, 7)[:, :3]
    hyps = {"lidar": centers}
    if has_l2e:
        hyps["ego"] = apply(np.linalg.inv(lidar2ego_of(fr)), centers)
        l2e = lidar2ego_of(fr)
        print("ego x-axis (forward) in lidar frame:", np.round(np.linalg.inv(l2e)[:3, :3] @ [1, 0, 0], 3))
        print("lidar origin in ego frame          :", np.round(l2e[:3, 3], 3))
    for name, pts in hyps.items():
        cam = apply(np.linalg.inv(s2l), pts)
        front = cam[:, 2] > 1.0
        uv = cam[front, :2] / cam[front, 2:3] @ K[:2, :2].T + K[:2, 2]
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)).sum()
        print(f"hypothesis anno_frame={name}: boxes {len(pts)}, in front of CAM_F0 {front.sum()}, inside image {inside}")
    print("두 가설이 다르면 inside 가 큰 쪽(그리고 영상에서 실제 상자 위치 육안 확인)을 --anno-frame 으로 쓴다.")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="dataset/navsim_logs/<trainval|test>")
    ap.add_argument("--sensor-root", required=True, help="dataset/sensor_blobs/<trainval|test> (data_path 의 기준)")
    ap.add_argument("--scene-filter", help="navsim/planning/script/config/common/train_test_split/scene_filter/<navtrain|navtest>.yaml")
    ap.add_argument("--kind", choices=["train", "test"], help="train=로그 연속 시퀀스, test=장면별 이력 4프레임 시퀀스")
    ap.add_argument("--anno-frame", choices=["lidar", "ego"], help="anns.gt_boxes 좌표계(--probe 로 확정). 기본값 없음")
    ap.add_argument("--out")
    ap.add_argument("--probe", action="store_true", help="키·좌표계 확인만 하고 종료")
    args = ap.parse_args()
    if args.probe:
        return probe(args.logs, args.sensor_root)
    for k in ("scene_filter", "kind", "anno_frame", "out"):
        if getattr(args, k) is None:
            ap.error(f"--{k.replace('_', '-')} 가 필요하다")

    sf = load_scene_filter(args.scene_filter)
    infos, n_scenes, eval_tokens = [], 0, []
    for log, frames in iter_logs(args.logs, sf):
        # devkit 은 로그 리스트 순서 그대로 창을 자른다(dataloader.py:47). 정렬하면 창이 달라질 수 있으므로 정렬하지 않고 검사만 한다.
        ts = [f["timestamp"] for f in frames]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            raise RuntimeError(f"{log}: 로그 프레임이 시간순이 아니다 — 시퀀스/미래 궤적 가정이 깨진다.")
        scenes = scenes_of_log(frames, sf)
        if sf["max_scenes"] is not None:
            raise NotImplementedError("max_scenes 는 devkit 이 로그 순서에 따라 자른다 — 재현 규칙 확인 전 미지원")
        if not scenes:
            continue
        n_scenes += len(scenes)
        track_ids = {}
        h = sf["num_history"]
        if args.kind == "test":
            # 장면마다 이력 4프레임을 하나의 시퀀스로(시간 뱅크 예열), 마지막(현재) 프레임만 평가.
            for tok, start in scenes.items():
                for j in range(h):
                    inf = frame_info(frames, start + j, log, args.sensor_root, args.anno_frame, track_ids)
                    inf["scene_token"] = f"navtest#{tok}"
                    if j == h - 1:
                        assert inf["token"] == tok
                        inf["is_eval_frame"], inf["navsim_token"] = True, tok
                    infos.append(inf)
                eval_tokens.append(tok)
        else:
            # navtrain 장면 창(이력+미래 14프레임)에 속한 프레임의 합집합을 로그 시간순 연속 시퀀스로.
            used = sorted({start + j for start in scenes.values() for j in range(h + sf["num_future"])})
            seg, prev = 0, None
            for i in used:
                if prev is not None and (i != prev + 1 or
                                         frames[i]["timestamp"] - frames[prev]["timestamp"] > SEQ_GAP_S * 1e6):
                    seg += 1
                inf = frame_info(frames, i, log, args.sensor_root, args.anno_frame, track_ids)
                inf["scene_token"] = f"{log}#{seg}"
                infos.append(inf)
                prev = i
    if sf["tokens"] is not None and args.kind == "test":
        missing = set(sf["tokens"]) - set(eval_tokens)
        if missing:
            raise RuntimeError(f"필터 token {len(missing)}개가 로그에서 생성되지 않았다(예: {sorted(missing)[:3]}).")
    meta = dict(kind=args.kind, anno_frame=args.anno_frame, scene_filter=sf, cams=NAVSIM_CAMS,
                ego_fut_ts=EGO_FUT_TS, agent_fut_ts=AGENT_FUT_TS, num_scenes=n_scenes,
                eval_tokens=eval_tokens if args.kind == "test" else None)
    pickle.dump(dict(infos=infos, metadata=meta), open(args.out, "wb"))
    print(f"{args.kind}: scenes {n_scenes}, frames {len(infos)} -> {args.out}")


if __name__ == "__main__":
    main()
