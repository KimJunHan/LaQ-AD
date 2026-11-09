# 수정(2026-11-05): navhard_two_stage 평가 준비 — build_navhard_infos.py 산출물 검사(새 파일, CPU, env navsim).
"""navhard info pkl 을 devkit(SceneLoader/AgentInput) 과 대조한다. 실패하면 비 0 종료.

  1) token 집합: info eval_tokens == devkit SceneLoader(navhard_two_stage 필터 + 합성 경로) 의 tokens_stage_one ∪ reactive_tokens_stage_two.
  2) 형식: 시퀀스(scene_token)마다 4 프레임, 마지막만 is_eval_frame, navsim_token == 마지막 token, lidar2ego == I,
     시퀀스 안 timestamp 증가(0.5 s ± 0.1), 1·2단계 scene_token 접두사.
  3) 전 프레임 학생 카메라(CAM_L0·F0·R0) 영상 파일 존재 + 표본 N 개 디코드(1920×1080).
  4) 보정: 내부행렬(fx, fy > 0, |fx−fy| < 1 %, 주점 영상 안 — 원본 보정은 fx≠fy 가 정상), sensor2lidar 회전 직교·det 1, 합성 ↔ 같은 로그 원본 프레임 보정 차이.
  5) devkit 대조(표본 N 개/단계): AgentInput 4 프레임의 ego_velocity·ego_acceleration·driving_command == info ego_status·명령,
     AgentInput 국소 ego_pose(현재 기준) == info ego2global 로 계산한 국소 포즈, 카메라 보정(sensor2lidar·K) == info,
     현재 프레임 CAM_F0 영상 화소 == info data_path 영상 화소.
  6) GT 상자 투영(CAM_F0 앞·영상 안 비율) 원본 vs 합성 — 좌표계 이상 탐지용 참고 수치.
실행: source scripts/navsim/navsim_env.sh && python -I tools/navsim/check_navhard_infos.py --infos <pkl> [--n 40] [--decode 200]
"""
import argparse
import collections
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

CAMS3 = ("CAM_L0", "CAM_F0", "CAM_R0")


def yaw_of(T):
    """devkit 과 같은 yaw: Quaternion(ego2global_rotation).yaw_pitch_roll[0] (dataclasses.py:176-186). 회전행렬 atan2(R10, R00) 는
    pitch·roll 이 있으면 최대 ~7e-3 rad 다르다(원본 로그 프레임 실측)."""
    from pyquaternion import Quaternion
    R = np.asarray(T, np.float64)[:3, :3]
    return float(Quaternion(matrix=R, rtol=1e-6, atol=1e-6).yaw_pitch_roll[0])


def local_pose(T_cur, T):
    """devkit convert_absolute_to_relative_se2_array 와 같은 2D 상대 포즈 (x, y, heading)."""
    yc = yaw_of(T_cur)
    d = T[:2, 3] - T_cur[:2, 3]
    c, s = np.cos(-yc), np.sin(-yc)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], (yaw_of(T) - yc + np.pi) % (2 * np.pi) - np.pi])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--n", type=int, default=40, help="단계별 devkit 대조 표본 수")
    ap.add_argument("--decode", type=int, default=200, help="실제 디코드할 영상 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json")
    a = ap.parse_args()
    from PIL import Image
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    rng = random.Random(a.seed)
    d = pickle.load(open(a.infos, "rb"))
    infos, meta = d["infos"], d["metadata"]
    nh = meta["navhard"]
    rep, fails = {}, []

    def fail(msg):
        fails.append(msg)
        print("FAIL:", msg, flush=True)

    # ---- 2) 형식
    seqs = collections.OrderedDict()
    for k, i in enumerate(infos):
        seqs.setdefault(i["scene_token"], []).append(k)
    ev = []
    for st, ks in seqs.items():
        fr = [infos[k] for k in ks]
        stage = 1 if st.startswith("navhard_s1#") else 2 if st.startswith("navhard_s2#") else None
        if stage is None or any(f["navhard_stage"] != stage for f in fr):
            fail(f"{st}: 단계 표시 이상")
        if len(fr) != 4 or [bool(f["is_eval_frame"]) for f in fr] != [False, False, False, True]:
            fail(f"{st}: 프레임 {len(fr)} / eval 표시 {[f['is_eval_frame'] for f in fr]}")
        if fr[-1]["navsim_token"] != fr[-1]["token"] or st.split("#", 1)[1] != fr[-1]["token"]:
            fail(f"{st}: navsim_token 불일치")
        dts = np.diff([f["timestamp"] for f in fr]) * 1e-6
        if not np.all(np.abs(dts - 0.5) < 0.1):
            # 원본 로그 자체에 프레임 빠짐(1.0 s)이 있는 장면은 devkit 창도 같은 프레임을 쓴다(filter_scenes 는 간격을 보지 않음)
            #   → 1단계만 경고로 센다(navtest info 도 같은 규칙). 합성 장면은 실패.
            if stage == 1 and np.all((np.abs(dts - 0.5) < 0.1) | (np.abs(dts - 1.0) < 0.1)):
                rep["warn_stage1_log_gap_sequences"] = rep.get("warn_stage1_log_gap_sequences", 0) + 1
            else:
                fail(f"{st}: 프레임 간격 {dts}")
        for f in fr:
            if np.abs(np.asarray(f["lidar2ego"]) - np.eye(4)).max() > 1e-9:
                fail(f"{f['token']}: lidar2ego != I")
        ev.append(fr[-1]["token"])
    if ev != list(meta["eval_tokens"]):
        fail("eval_tokens 순서/집합이 시퀀스 마지막 프레임과 다르다")
    rep["sequences"] = len(seqs)
    rep["frames"] = len(infos)
    rep["unique_frame_tokens"] = len({i["token"] for i in infos})

    # ---- 1) devkit token 집합
    os.environ.setdefault("OPENSCENE_DATA_ROOT", "/home1/irteam/data-vol1/datasets/navsim/openscene")
    cfg_dir = os.path.join(os.environ["NAVSIM_DEVKIT_ROOT"], "navsim/planning/script/config/common")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="default_common", overrides=["train_test_split=navhard_two_stage"])
    sf = instantiate(cfg.train_test_split.scene_filter)
    syn = Path(nh["navhard_root"])
    sensor_cfg = SensorConfig(cam_f0=True, cam_l0=True, cam_l1=False, cam_l2=False, cam_r0=True, cam_r1=False,
                              cam_r2=False, cam_b0=False, lidar_pc=False)
    loader = SceneLoader(data_path=Path(nh["logs"]), original_sensor_path=Path(nh["sensor_root"]),
                         synthetic_sensor_path=syn / "sensor_blobs", synthetic_scenes_path=syn / "synthetic_scene_pickles",
                         scene_filter=sf, sensor_config=sensor_cfg)
    s1, s2 = set(loader.tokens_stage_one), set(loader.reactive_tokens_stage_two)
    rep["devkit_stage1"], rep["devkit_stage2"] = len(s1), len(s2)
    if s1 != set(nh["stage1_tokens"]) or s2 != set(nh["stage2_tokens"]):
        fail(f"devkit token 집합 불일치: 1단계 {len(s1 ^ set(nh['stage1_tokens']))}, 2단계 {len(s2 ^ set(nh['stage2_tokens']))}")

    # ---- 3) 영상 존재 + 디코드
    paths = sorted({i["cams"][c]["data_path"] for i in infos for c in CAMS3})
    miss = [p for p in paths if not os.path.exists(p)]
    rep["student_cam_files"], rep["student_cam_missing"] = len(paths), len(miss)
    if miss:
        fail(f"영상 없음 {len(miss)} (예: {miss[:3]})")
    bad = []
    for p in rng.sample(paths, min(a.decode, len(paths))):
        im = Image.open(p)
        im.load()
        if im.size != (1920, 1080):
            bad.append((p, im.size))
    rep["decoded"], rep["decode_bad"] = min(a.decode, len(paths)), len(bad)
    if bad:
        fail(f"영상 크기 이상 {bad[:3]}")

    # ---- 4) 보정
    orig_cal = {}
    for i in infos:
        if not i["synthetic"]:
            orig_cal.setdefault(i["log_name"], i["cams"])
    worst = collections.defaultdict(float)
    for i in infos:
        for c in CAMS3:
            cc = i["cams"][c]
            K, R = np.asarray(cc["cam_intrinsic"]), np.asarray(cc["sensor2lidar_rotation"])
            if not (K[0, 0] > 0 and K[1, 1] > 0 and abs(K[0, 0] - K[1, 1]) < 0.01 * K[0, 0] and 0 < K[0, 2] < 1920 and 0 < K[1, 2] < 1080):
                fail(f"{i['token']} {c}: K 이상 {K.tolist()}")
            if np.abs(R @ R.T - np.eye(3)).max() > 1e-4 or abs(np.linalg.det(R) - 1) > 1e-4:
                fail(f"{i['token']} {c}: 회전 비직교")
            if i["synthetic"] and i["log_name"] in orig_cal:
                o = orig_cal[i["log_name"]][c]
                worst["K"] = max(worst["K"], float(np.abs(K - o["cam_intrinsic"]).max()))
                worst["R"] = max(worst["R"], float(np.abs(R - o["sensor2lidar_rotation"]).max()))
                worst["t"] = max(worst["t"], float(np.abs(np.asarray(cc["sensor2lidar_translation"]) - o["sensor2lidar_translation"]).max()))
    rep["synthetic_vs_original_calib_maxabs"] = dict(worst)

    # ---- 5) devkit 대조
    by_seq = {st: [infos[k] for k in ks] for st, ks in seqs.items()}
    dev = collections.Counter()
    for stage, toks in ((1, nh["stage1_tokens"]), (2, nh["stage2_tokens"])):
        for tok in rng.sample(list(toks), min(a.n, len(toks))):
            fr = by_seq[f"navhard_s{stage}#{tok}"]
            ai = loader.get_agent_input_from_token(tok)
            if len(ai.ego_statuses) != 4:
                fail(f"{tok}: AgentInput 프레임 {len(ai.ego_statuses)}")
                continue
            for k, (es, f) in enumerate(zip(ai.ego_statuses, fr)):
                e = np.concatenate([es.ego_velocity, es.ego_acceleration]).astype(np.float32)
                if not np.array_equal(e, f["ego_status"]):
                    fail(f"{tok}[{k}]: ego_status {e} != info {f['ego_status']}")
                if not np.array_equal(np.asarray(es.driving_command, np.float32), f["driving_command"]):
                    fail(f"{tok}[{k}]: 명령 불일치")
                lp = local_pose(np.asarray(fr[-1]["ego2global"]), np.asarray(f["ego2global"]))
                err = np.abs(lp - np.asarray(es.ego_pose, np.float64))
                err[2] = min(err[2], 2 * np.pi - err[2])
                dev[f"pose_err_s{stage}"] = max(dev[f"pose_err_s{stage}"], float(err.max()))
                if err.max() > 1e-3:
                    fail(f"{tok}[{k}]: 국소 포즈 {lp} != devkit {es.ego_pose}")
                cams = ai.cameras[k]
                for c in CAMS3:
                    dc = getattr(cams, c.lower())
                    ic = f["cams"][c]
                    if not (np.allclose(dc.sensor2lidar_rotation, ic["sensor2lidar_rotation"]) and
                            np.allclose(dc.sensor2lidar_translation, ic["sensor2lidar_translation"]) and
                            np.allclose(dc.intrinsics, ic["cam_intrinsic"])):
                        fail(f"{tok}[{k}] {c}: 보정 불일치")
            img = np.asarray(Image.open(fr[-1]["cams"]["CAM_F0"]["data_path"]))
            if not np.array_equal(img, ai.cameras[-1].cam_f0.image):
                fail(f"{tok}: CAM_F0 화소가 devkit 영상과 다르다")
            dev[f"checked_s{stage}"] += 1
    rep["devkit_compare"] = dict(dev)

    # ---- 6) GT 상자 투영 비율(참고)
    proj = collections.defaultdict(lambda: [0, 0])
    for i in rng.sample(infos, min(3000, len(infos))):
        b = np.asarray(i["gt_boxes"]).reshape(-1, 7)
        if not len(b):
            continue
        c = i["cams"]["CAM_F0"]
        s2l = np.eye(4); s2l[:3, :3] = c["sensor2lidar_rotation"]; s2l[:3, 3] = c["sensor2lidar_translation"]
        pc = (np.linalg.inv(s2l) @ np.c_[b[:, :3], np.ones(len(b))].T).T[:, :3]
        fr_ = pc[:, 2] > 1
        uv = pc[fr_, :2] / pc[fr_, 2:3] @ np.asarray(c["cam_intrinsic"])[:2, :2].T + np.asarray(c["cam_intrinsic"])[:2, 2]
        ins = ((uv[:, 0] >= 0) & (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] < 1080)).sum()
        key = "synthetic" if i["synthetic"] else "original"
        proj[key][0] += int(ins); proj[key][1] += len(b)
    rep["gt_box_in_cam_f0_ratio"] = {k: round(v[0] / max(v[1], 1), 4) for k, v in proj.items()}

    rep["fails"] = len(fails)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    if a.out_json:
        json.dump(dict(rep, fail_examples=fails[:50]), open(a.out_json, "w"), indent=1, ensure_ascii=False)
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
