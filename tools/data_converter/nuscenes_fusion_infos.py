# 수정(2026-10-15): B 융합 — 실험 B(HiP-AD + LiDAR + Radar, 언어 없음)용 info 생성기(새 파일).
#   기존 data/infos/nuscenes_infos_{train,val}.pkl 의 info dict 를 그대로 복사하고 두 키를 덧붙여
#   data/infos/nuscenes_fusion_infos_{train,val}.pkl 로 **새로** 저장한다(기존 info 는 덮어쓰지 않는다).
#     'sweeps' : LIDAR_TOP 직전 sweep 최대 max_lidar_sweeps 개(키프레임 제외, 최근→과거 순; mmdet3d obtain_sensor2top 동일 의미)
#     'radars' : {RADAR_*: [키프레임, 직전 sweep, ...] 최대 max_radar_sweeps 개}
#   각 항목: data_path('./data/nuscenes/...', lidar_path 와 같은 규약), sample_data_token, timestamp(us),
#            sensor2lidar_rotation(3x3, 열벡터 규약: p_lidar = R @ p_sensor + t), sensor2lidar_translation(3,),
#            sensor2ego_*/ego2global_*(감사용 원자료).
#   좌표 사슬(devkit LidarPointCloud.from_file_multisweep 과 같은 순서):
#     sensor → ego(센서 시각) → global → ego(LiDAR 키프레임 시각) → LiDAR(키프레임)
#   = inv(L2E_key) @ inv(E2G_key) @ E2G_sweep @ S2E_sweep.  기존 info 의 카메라 sensor2lidar 와 같은 규약임을
#   --validate 에서 카메라로 재계산해 수치 대조한다.
#   파일 존재는 NFS 디렉터리 나열 대신 info 경로별 stat 으로 확인한다(스레드 병렬). 누락이 있으면 목록을 저장하고 보고한다
#   (조용히 빼지 않는다 — 누락 처리 방침은 사용자 결정 사항).
import argparse
import os
import os.path as osp
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pyquaternion import Quaternion

RADAR_CHANNELS = ("RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT")


def _rt(translation, rotation):
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = np.asarray(translation)
    return T


def sensor_entry(nusc, sd_token, lidar2ego_key, ego2global_key, sensor_type, data_prefix):
    sd = nusc.get("sample_data", sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    s2e = _rt(cs["translation"], cs["rotation"])
    e2g = _rt(pose["translation"], pose["rotation"])
    s2l = np.linalg.inv(lidar2ego_key) @ np.linalg.inv(ego2global_key) @ e2g @ s2e
    return dict(
        data_path=osp.join(data_prefix, sd["filename"]),
        type=sensor_type,
        sample_data_token=sd["token"],
        is_key_frame=bool(sd["is_key_frame"]),
        timestamp=sd["timestamp"],
        sensor2ego_translation=cs["translation"],
        sensor2ego_rotation=cs["rotation"],
        ego2global_translation=pose["translation"],
        ego2global_rotation=pose["rotation"],
        sensor2lidar_rotation=s2l[:3, :3].astype(np.float64),
        sensor2lidar_translation=s2l[:3, 3].astype(np.float64),
    ), sd


def build_fusion_fields(nusc, info, max_lidar_sweeps, max_radar_sweeps, data_prefix):
    sample = nusc.get("sample", info["token"])
    l2e = _rt(info["lidar2ego_translation"], info["lidar2ego_rotation"])
    e2g = _rt(info["ego2global_translation"], info["ego2global_rotation"])
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    # 정합성: info 의 lidar_path 와 키프레임 timestamp 가 devkit 표본과 같아야 한다.
    assert osp.basename(info["lidar_path"]) == osp.basename(lidar_sd["filename"]), info["token"]
    assert info["timestamp"] == lidar_sd["timestamp"], info["token"]
    sweeps = []
    sd = lidar_sd
    while len(sweeps) < max_lidar_sweeps and sd["prev"] != "":
        e, sd = sensor_entry(nusc, sd["prev"], l2e, e2g, "lidar", data_prefix)
        sweeps.append(e)
    radars = {}
    for ch in RADAR_CHANNELS:
        lst = []
        tok = sample["data"][ch]
        while len(lst) < max_radar_sweeps and tok != "":
            e, sd = sensor_entry(nusc, tok, l2e, e2g, "radar", data_prefix)
            lst.append(e)
            tok = sd["prev"]
        radars[ch] = lst
    return sweeps, radars


def cam_check(nusc, info, data_prefix):
    """기존 info 의 카메라 sensor2lidar 를 같은 함수로 재계산해 최대 오차를 돌려준다(규약 대조)."""
    l2e = _rt(info["lidar2ego_translation"], info["lidar2ego_rotation"])
    e2g = _rt(info["ego2global_translation"], info["ego2global_rotation"])
    err = 0.0
    for cam, c in info["cams"].items():
        e, _ = sensor_entry(nusc, c["sample_data_token"], l2e, e2g, "camera", data_prefix)
        err = max(err, np.abs(e["sensor2lidar_rotation"] - np.asarray(c["sensor2lidar_rotation"])).max(),
                  np.abs(e["sensor2lidar_translation"] - np.asarray(c["sensor2lidar_translation"])).max())
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--in-dir", default="data/infos")
    ap.add_argument("--out-dir", default="data/infos")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--data-prefix", default="./data/nuscenes")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--max-lidar-sweeps", type=int, default=10)
    ap.add_argument("--max-radar-sweeps", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만(검증용). 0=전체")
    ap.add_argument("--no-exist-check", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="기존 출력 pkl 덮어쓰기 허용(명시적으로만)")  # 수정(2026-10-15)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    t0 = time.time()
    src = osp.join(args.in_dir, f"nuscenes_infos_{args.split}.pkl")
    with open(src, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"]
    if args.limit:
        infos = infos[: args.limit]
    print(f"[load] {src}: {len(infos)} infos ({time.time()-t0:.1f}s)", flush=True)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f"[load] devkit {args.version} ({time.time()-t0:.1f}s)", flush=True)

    out_infos = []
    cam_err = 0.0
    n_sweeps = []
    n_radar = {ch: [] for ch in RADAR_CHANNELS}
    for i, info in enumerate(infos):
        sweeps, radars = build_fusion_fields(nusc, info, args.max_lidar_sweeps, args.max_radar_sweeps,
                                             args.data_prefix)
        if i % 50 == 0:
            cam_err = max(cam_err, cam_check(nusc, info, args.data_prefix))
        new = dict(info)
        new["sweeps"] = sweeps
        new["radars"] = radars
        out_infos.append(new)
        n_sweeps.append(len(sweeps))
        for ch in RADAR_CHANNELS:
            n_radar[ch].append(len(radars[ch]))
        if (i + 1) % 2000 == 0:
            print(f"[meta] {i+1}/{len(infos)} ({time.time()-t0:.1f}s)", flush=True)
    print(f"[meta] done ({time.time()-t0:.1f}s). cam sensor2lidar 재계산 최대오차 {cam_err:.3e}", flush=True)
    print(f"[stat] lidar sweeps/표본: min {min(n_sweeps)} mean {np.mean(n_sweeps):.2f} "
          f"(<{args.max_lidar_sweeps}: {sum(n < args.max_lidar_sweeps for n in n_sweeps)})", flush=True)
    for ch in RADAR_CHANNELS:
        v = n_radar[ch]
        print(f"[stat] {ch}: sweeps/표본 min {min(v)} mean {np.mean(v):.2f}", flush=True)

    missing = {}
    if not args.no_exist_check:
        paths = []
        for info in out_infos:
            paths.append(("LIDAR_TOP_key", info["lidar_path"]))
            for s in info["sweeps"]:
                paths.append(("LIDAR_TOP_sweep", s["data_path"]))
            for ch, lst in info["radars"].items():
                for j, s in enumerate(lst):
                    paths.append((ch + ("_key" if j == 0 else "_sweep"), s["data_path"]))
        uniq = {}
        for k, p in paths:
            uniq.setdefault(p, k)
        plist = list(uniq.items())
        print(f"[exist] stat {len(plist)} 고유 파일 (threads={args.threads})", flush=True)
        done = [0]

        def _chk(item):
            p, k = item
            return k, p, osp.exists(p)

        with ThreadPoolExecutor(args.threads) as ex:
            for n, (k, p, ok) in enumerate(ex.map(_chk, plist, chunksize=256)):
                if not ok:
                    missing.setdefault(k, []).append(p)
                if (n + 1) % 100000 == 0:
                    print(f"[exist] {n+1}/{len(plist)} ({time.time()-t0:.1f}s) 누락 "
                          f"{sum(len(v) for v in missing.values())}", flush=True)
        print(f"[exist] done ({time.time()-t0:.1f}s). 누락: "
              + (", ".join(f"{k}={len(v)}" for k, v in sorted(missing.items())) or "없음"), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out = osp.join(args.out_dir, f"nuscenes_fusion_infos_{args.split}{args.out_suffix}.pkl")
    meta = dict(data.get("metadata", {}))
    meta.update(fusion=dict(max_lidar_sweeps=args.max_lidar_sweeps, max_radar_sweeps=args.max_radar_sweeps,
                            radar_channels=RADAR_CHANNELS, created="2026-10-15", source=src,
                            frame="LIDAR_TOP keyframe; sensor2lidar column convention p_l = R p_s + t",
                            missing_counts={k: len(v) for k, v in missing.items()}))
    # 수정(2026-10-15): 검토 — 기존 assert 는 끝의 "or True" 로 항상 통과(덮어쓰기 방지 무효). 명시적 플래그로 교체.
    if osp.exists(out) and not getattr(args, "overwrite", False):
        raise FileExistsError(f"{out} 가 이미 있다 — 덮어쓰려면 --overwrite")
    with open(out + ".tmp", "wb") as f:
        pickle.dump(dict(infos=out_infos, metadata=meta), f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(out + ".tmp", out)
    if missing:
        with open(out.replace(".pkl", "_missing.txt"), "w") as f:
            for k, v in sorted(missing.items()):
                for p in v:
                    f.write(f"{k}\t{p}\n")
    print(f"[save] {out} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
