# 수정(2026-11-05): navhard_two_stage 평가 준비(평가 전용 — navhard 로 학습 금지). 새 파일, 기존 소스 무수정.
"""navhard_two_stage 학생 info pkl (1단계 원본 장면 + 2단계 합성 장면) — navtest info 와 같은 형식(kind="test").

devkit 근거(/home1/irteam/data-vol1/src/navsim_ref @0a380a9):
  - 분할: config/common/train_test_split/navhard_two_stage.yaml (data_split test, reactive_all_mapping 225 묶음)
    + scene_filter/navhard_two_stage.yaml (history 4, future 8, frame_interval 1, has_route, tokens 450,
    reactive_synthetic_initial_tokens 5,462, include_synthetic_scenes True).
  - 1단계 장면 = 로그 창(이력 4 + 미래 8) 의 현재 프레임 token (common/dataloader.py:16-73 filter_scenes; 우리 B.scenes_of_log 와 같은 규칙).
    450 = 225 묶음 × (현재 orig_token, 0.5 s 전 prev_token) — run_pdm_score.py 가 둘 다 1단계로 채점하고 묶음 점수로 합친다.
  - 2단계 장면 = 합성 장면 pickle(synthetic_scene_pickles/<scene_token>.pkl). 채점 token = scene_metadata.initial_token
    (= frames[3].token, dataloader.py:105 키; Scene.load_from_disk dataclasses.py:674-730). 1단계 장면 창의 마지막 프레임
    (현재 + 4 s)을 corresponding_original_scene 으로 갖는 것만 포함(dataloader.py:76-107) ∩ reactive 목록(:167-175).
  - 에이전트 입력(AgentInput): 원본 = 로그 이력 4 프레임(dataclasses.py:160-215), 합성 = pickle 의 4 프레임(get_agent_input :410-435).
    ego_status 속도·가속 = 국소(ego) 좌표 그대로(합성 pickle 의 ego_velocity/ego_acceleration, in_global_frame 은 ego_pose 에만 해당).

info 형식(navtest = build_navsim_infos.py --kind test 와 같은 키, 같은 단위):
  1단계: B.frame_info 그대로(로그 기반, 미래 GT 포함). scene_token = "navhard_s1#<token>", 이력 4 프레임 = 한 시퀀스.
  2단계(합성): pickle 4 프레임 → token, timestamp, map_location(metadata.map_name), cams(8; 영상 = 합성 sensor_blobs),
    lidar2ego = I(NAVSIM 원본과 같음 — 전 프레임 단위행렬, NavsimTeacherDataset 이 검사), ego2global = (x, y, heading) 의 SE(2)
    (z·roll·pitch = 0 — 합성 pickle 에 없다. 한 합성 시퀀스 안 4 프레임이 같은 규약이라 상대 변환(시간 뱅크 정렬)은 일관),
    ego_status = [vx, vy, ax, ay](국소), driving_command, gt_boxes/names/velocity(합성 annotations, 원본과 같은 ego=lidar 프레임),
    미래 GT 없음(합성 장면 num_future_frames = 0) → gt_*_fut_masks = 0. lidar_path = None(navhard 합성 라이다 파일이 배포에 없음
    — 실측 21,848 개 전부 없음; 학생 test 파이프라인은 라이다를 읽지 않는다).
    scene_token = "navhard_s2#<initial_token>" — 합성 장면마다 별도 시퀀스 → 시간 뱅크가 장면 시작에서 반드시 초기화된다
    (SparseDetector._maybe_reset_banks, batch 1). 원본 장면이나 다른 합성 장면(같은 궤적 쌍은 프레임 3 개를 공유)의 기억을 넘기지 않는다.
  공통 추가 키: navhard_stage(1|2), synthetic(bool), is_eval_frame / navsim_token(시퀀스 마지막 = 채점 token).
  metadata: kind="test", eval_tokens(1단계 450 + 2단계 5,462 = 5,912), navhard={stage1_tokens, stage2_tokens, mapping_sha256, ...}.

실행(CPU, env hipad_bw 또는 navsim — numpy·yaml·pyquaternion 만 필요):
  CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX /home1/irteam/data-vol1/anaconda3/envs/hipad_bw/bin/python -I \
    tools/navsim/build_navhard_infos.py --out data/infos/navsim_navhard_20261105/navsim_navhard_two_stage.pkl
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_navsim_infos as B  # noqa: E402  (수정 없이 재사용: scenes_of_log, frame_info, load_scene_filter)

OPENSCENE = "/home1/irteam/data-vol1/datasets/navsim/openscene"
DEVKIT = "/home1/irteam/data-vol1/src/navsim_ref"
SPLIT_YAML = f"{DEVKIT}/navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml"
FILTER_YAML = f"{DEVKIT}/navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"
CAM_KEY = {c: c.lower() for c in B.NAVSIM_CAMS}   # 합성 pickle camera_dict 키 = 소문자(cam_f0 …)


def se2(x, y, h):
    T = np.eye(4)
    c, s = np.cos(h), np.sin(h)
    T[:2, :2] = [[c, -s], [s, c]]
    T[0, 3], T[1, 3] = x, y
    return T


def synthetic_frame_info(fr, log, map_name, syn_sensor_root, track_ids):
    es = fr["ego_status"]
    if not bool(es.get("in_global_frame", False)):
        raise ValueError(f"{fr['token']}: 합성 ego_status.in_global_frame 이 True 가 아니다 — ego_pose 규약 확인 필요")
    pose = np.asarray(es["ego_pose"], np.float64).reshape(3)
    cams = {}
    for cam in B.NAVSIM_CAMS:
        c = fr["camera_dict"][CAM_KEY[cam]]
        cams[cam] = dict(data_path=os.path.join(syn_sensor_root, str(c["data_path"])),
                         sensor2lidar_rotation=np.asarray(c["sensor2lidar_rotation"], np.float64),
                         sensor2lidar_translation=np.asarray(c["sensor2lidar_translation"], np.float64),
                         cam_intrinsic=np.asarray(c["cam_intrinsic"], np.float64))
    a = fr["annotations"]
    boxes = np.asarray(a["boxes"], np.float64).reshape(-1, 7)
    vel = np.asarray(a["velocity_3d"], np.float64).reshape(-1, 3)
    tracks = list(a["track_tokens"])
    for t in tracks:
        track_ids.setdefault(t, len(track_ids))
    n = len(boxes)
    return dict(
        token=fr["token"], log_name=log, timestamp=int(fr["timestamp"]), map_location=map_name,
        cams=cams, lidar_path=None,
        lidar2ego=np.eye(4), ego2global=se2(*pose),
        ego_status=np.concatenate([np.asarray(es["ego_velocity"], np.float32).reshape(2),
                                   np.asarray(es["ego_acceleration"], np.float32).reshape(2)]).astype(np.float32),
        driving_command=np.asarray(es["driving_command"], np.float32).reshape(4),
        gt_boxes=boxes.astype(np.float32), gt_names=np.asarray(a["names"]),
        gt_velocity=vel[:, :2].astype(np.float32), instance_inds=np.array([track_ids[t] for t in tracks], np.int64),
        gt_agent_fut_trajs=np.zeros((n, B.AGENT_FUT_TS, 2), np.float32), gt_agent_fut_masks=np.zeros((n, B.AGENT_FUT_TS), np.float32),
        gt_ego_fut_trajs=np.zeros((B.EGO_FUT_TS, 2), np.float32), gt_ego_fut_masks=np.zeros(B.EGO_FUT_TS, np.float32),
        is_eval_frame=False, navsim_token=None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=f"{OPENSCENE}/navsim_logs/test")
    ap.add_argument("--sensor-root", default=f"{OPENSCENE}/sensor_blobs/test")
    ap.add_argument("--navhard-root", default=f"{OPENSCENE}/navhard_two_stage",
                    help="synthetic_scene_pickles/ 와 sensor_blobs/ 가 있는 폴더")
    ap.add_argument("--split-yaml", default=SPLIT_YAML)
    ap.add_argument("--scene-filter", default=FILTER_YAML)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    if os.path.exists(a.out):
        raise SystemExit(f"[중단] 이미 있음: {a.out}")
    sf = B.load_scene_filter(a.scene_filter)
    fy = yaml.safe_load(open(a.scene_filter))
    if not fy.get("include_synthetic_scenes") or fy.get("synthetic_scene_tokens") is not None or sf["max_scenes"] is not None:
        raise SystemExit("navhard 필터 가정(include_synthetic_scenes True, synthetic_scene_tokens 없음, max_scenes null) 위반")
    reactive = set(fy["reactive_synthetic_initial_tokens"])
    mapping = yaml.safe_load(open(a.split_yaml))["reactive_all_mapping"]
    map_s1 = {t for o, p, _ in mapping for t in (o, p)}
    map_s2 = [t for _, _, pairs in mapping for pr in pairs for t in pr]
    if map_s1 != set(sf["tokens"]) or set(map_s2) != reactive or len(map_s2) != len(reactive):
        raise SystemExit("mapping ↔ scene_filter token 집합 불일치")
    mapping_sha = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()

    # ---------------- 1단계(원본 로그)
    infos, s1_tokens, final_tok = [], [], {}
    log_set = set(sf["log_names"])
    for log, frames in B.iter_logs(a.logs, sf):
        ts = [f["timestamp"] for f in frames]
        if any(b <= c for c, b in zip(ts, ts[1:])):
            raise RuntimeError(f"{log}: 로그 프레임이 시간순이 아니다")
        scenes = B.scenes_of_log(frames, sf)
        nf = sf["num_history"] + sf["num_future"]
        track_ids = {}
        for tok, start in scenes.items():
            final_tok[frames[start + nf - 1]["token"]] = tok
            for j in range(sf["num_history"]):
                inf = B.frame_info(frames, start + j, log, a.sensor_root, "lidar", track_ids)
                inf.update(scene_token=f"navhard_s1#{tok}", navhard_stage=1, synthetic=False)
                if j == sf["num_history"] - 1:
                    assert inf["token"] == tok
                    inf["is_eval_frame"], inf["navsim_token"] = True, tok
                infos.append(inf)
            s1_tokens.append(tok)
    if set(s1_tokens) != set(sf["tokens"]) or len(s1_tokens) != len(sf["tokens"]):
        raise RuntimeError(f"1단계 token 불일치: 생성 {len(s1_tokens)} / 필터 {len(sf['tokens'])}, "
                           f"누락 {sorted(set(sf['tokens']) - set(s1_tokens))[:5]}")

    # ---------------- 2단계(합성)
    syn_dir = Path(a.navhard_root) / "synthetic_scene_pickles"
    syn_sensor = str(Path(a.navhard_root) / "sensor_blobs")
    scenes2 = []
    for p in sorted(syn_dir.iterdir()):
        d = pickle.load(open(p, "rb"))
        m = d["scene_metadata"]
        if m["log_name"] not in log_set or m["corresponding_original_scene"] not in final_tok:
            continue                                          # devkit filter_synthetic_scenes 와 같은 규칙
        if m["initial_token"] not in reactive:
            continue                                          # reactive_tokens_stage_two
        fr = d["frames"]
        if len(fr) != m["num_history_frames"] or m["num_history_frames"] != sf["num_history"] or fr[-1]["token"] != m["initial_token"]:
            raise RuntimeError(f"{p.name}: 합성 장면 프레임 구성 이상(num_history {m['num_history_frames']}, 프레임 {len(fr)})")
        scenes2.append((m["log_name"], int(fr[-1]["timestamp"]), m["initial_token"], p, d))
    scenes2.sort(key=lambda x: (x[0], x[1], x[2]))
    s2_tokens = []
    for log, _, init, p, d in scenes2:
        m = d["scene_metadata"]
        track_ids = {}
        for j, fr in enumerate(d["frames"]):
            inf = synthetic_frame_info(fr, log, m["map_name"], syn_sensor, track_ids)
            inf.update(scene_token=f"navhard_s2#{init}", navhard_stage=2, synthetic=True,
                       synthetic_scene_file=p.name, corresponding_original_scene=m["corresponding_original_scene"],
                       first_stage_token=final_tok[m["corresponding_original_scene"]])
            if j == len(d["frames"]) - 1:
                inf["is_eval_frame"], inf["navsim_token"] = True, init
            infos.append(inf)
        s2_tokens.append(init)
    if set(s2_tokens) != reactive or len(s2_tokens) != len(reactive):
        raise RuntimeError(f"2단계 token 불일치: 생성 {len(s2_tokens)} / reactive {len(reactive)}")
    if set(s1_tokens) & set(s2_tokens):
        raise RuntimeError("1·2단계 token 이 겹친다")

    meta = dict(kind="test", anno_frame="lidar", scene_filter=sf, cams=B.NAVSIM_CAMS, ego_fut_ts=B.EGO_FUT_TS,
                agent_fut_ts=B.AGENT_FUT_TS, num_scenes=len(s1_tokens) + len(s2_tokens),
                eval_tokens=s1_tokens + s2_tokens,
                navhard=dict(split="navhard_two_stage", stage1_tokens=s1_tokens, stage2_tokens=s2_tokens,
                             mapping_sha256=mapping_sha, n_mapping_groups=len(mapping),
                             logs=a.logs, sensor_root=a.sensor_root, navhard_root=a.navhard_root,
                             split_yaml=a.split_yaml, scene_filter_yaml=a.scene_filter,
                             synthetic_ego2global="SE(2) from ego_pose (x, y, heading); z = roll = pitch = 0",
                             synthetic_lidar=None, eval_only=True,
                             built=time.strftime("%Y-%m-%d %H:%M:%S")))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".tmp", "wb") as f:
        pickle.dump(dict(infos=infos, metadata=meta), f, protocol=4)
    os.replace(a.out + ".tmp", a.out)
    nu = len({i["token"] for i in infos})
    print(f"navhard: 1단계 장면 {len(s1_tokens)}, 2단계 장면 {len(s2_tokens)}, 프레임 {len(infos)} (고유 token {nu}), "
          f"{time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
