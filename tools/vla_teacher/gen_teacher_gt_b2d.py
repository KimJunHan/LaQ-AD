# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
#   B2D 교사 VLM 정답(v2-sensor-b2d, 547 토큰 고정) 생성기. nuScenes 판 gen_teacher_gt_v2.py 와 같은 레코드 형식
#   (sft_common / train_sft / answer_grammar 가 그대로 읽는다 — B2D 실행은 b2d_teacher_shim.py 로 common_v2 자리에 common_b2d 를 끼움).
#   대상 = 2 Hz 키프레임(clip 첫 프레임 f0 기준 (frame_idx − f0) % 5 == 0; 대부분 f0 = 0)만(사용자 결정: 학습 캐시는 2 Hz 키프레임만, 그 사이 프레임은 직전 키프레임 출력 재사용).
#   라벨 출처 = 학습과 같은 Bench2DriveDataset.get_data_info(split_group=5 → 0.5 s 간격 미래, num_points>0 박스, 지도 폴리라인
#   with_connect_lane=True, ROI x±15 / y±30) — 학생 GT 와 정의가 같다. 영상은 읽지 않는다(경로만 기록).
#
#   단계(cwd = 저장소 루트, CPU 전용):
#     export CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=.
#     python tools/vla_teacher/gen_teacher_gt_b2d.py extract --split val   [--limit N]   # 데이터셋 → 표본 묶음(raw pkl)
#     python tools/vla_teacher/gen_teacher_gt_b2d.py extract --split train
#     python tools/vla_teacher/gen_teacher_gt_b2d.py conventions --split train         # 좌표·부호 규약 실측(필수 검사)
#     python tools/vla_teacher/gen_teacher_gt_b2d.py explore_ttc                       # TTC_GAP(train 전용 p5)
#     python tools/vla_teacher/build_b2d_action_codebook.py                            # 행동 코드북(train 키프레임)
#     python tools/vla_teacher/gen_teacher_gt_b2d.py answers --out_dir <dir>           # 정답 jsonl + 문법·지시문·특수토큰·stats
import argparse
import collections
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import common_b2d as B  # noqa: E402

RAW_DIR = "results/b2d_prep/teacher_gt_raw"        # 로컬(재생성 가능한 중간 산출물)
CONV_FILE = "results/b2d_prep/b2d_conventions.json"
TTC_FILE = "results/b2d_prep/b2d_ttc_explore.json"
CODEBOOK = "results/b2d_prep/b2d_action_codebook.npz"
TOKENIZER = "/home1/irteam/data-vol1/src/HiP-AD/ckpts/qwen2-vl-2b"
INFOS = {"train": "data/infos/b2d_infos_train.pkl", "val": "data/infos/b2d_infos_val.pkl"}
MAP_FILE = "data/infos/b2d_map_infos.pkl"
CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
KF = 5          # 10 Hz → 2 Hz
TC_TYPES = ("TrafficLight", "StopSign")

_DS = None


def make_dataset(split):
    sys.path.insert(0, REPO)
    import projects.mmdet3d_plugin  # noqa: F401  (레지스트리)
    from projects.mmdet3d_plugin.models.loader.bench2drive.dataset import Bench2DriveDataset
    return Bench2DriveDataset(
        data_root="data/bench2drive", ann_file=INFOS[split], map_file=MAP_FILE, pipeline=None,
        modality=dict(use_camera=True, use_lidar=False, use_radar=False, use_map=False, use_external=False),
        test_mode=True, det_classes=B.DET_CLASSES, map_classes=B.MAP_CLASSES,
        plan_anchor_types=[("temp", "2hz")], serialize_data=True)   # 학습 config 와 같은 split_group=5·map 설정(기본값). 직렬화 = 워커 fork 시 COW 복제 방지(학습과 같은 방식)


def clip_of(folder):
    return folder.split("/")[-1]


def _tc_polys(ds, info, world2lidar):
    """신호등·정지 표지 트리거 영역(world) → 라이다 xy. 자차 50 m 안만(지도 함수와 같은 거리 컷)."""
    m = ds.map_infos[info["town_name"]]
    pts_l, types = m.get("trigger_volumes_points", []), m.get("trigger_volumes_types", [])
    ego_xy = np.linalg.inv(world2lidar)[0:2, 3]
    out, kinds = [], []
    for P, t in zip(pts_l, types):
        if t not in TC_TYPES:
            continue
        P = np.asarray(P, np.float64)
        if P.ndim != 2 or len(P) == 0 or np.min(np.linalg.norm(P[:, :2] - ego_xy, axis=1)) > 50:
            continue
        Q = np.concatenate([P[:, :3] if P.shape[1] >= 3 else np.c_[P[:, :2], np.zeros(len(P))], np.ones((len(P), 1))], 1)
        L = (world2lidar @ Q.T).T[:, :2]
        out.append(L.round(3)); kinds.append(t)
    return out, kinds


def gap_fill_keyframes(frames, f0):
    """빈 격자 칸 보충: 격자 키프레임 g = f0 + 5k 가 infos 에 없고(프레임 누락) 칸 (g, g+5) 안에 실제 프레임이 있으면 그 첫 프레임을
    키프레임으로 삼는다 → 칸 안 프레임의 교사 출력 나이 ≤ 0.4 s 유지(폐쇄루프 상한과 같음). train 에서 2 clip 2 칸(실측)."""
    out = set()
    for c, f in frames:
        g = f0[c] + KF * ((f - f0[c]) // KF)
        if (c, g) in frames:
            continue
        first = min(x for x in range(g + 1, g + KF) if (c, x) in frames)
        out.add((c, first))
    return out


def _one(i):
    ds = _DS
    info = ds.data_infos[i]
    d = ds.get_data_info(i)
    w2l = np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"], np.float64)
    tc, tck = _tc_polys(ds, info, w2l)
    names = list(d["gt_names"])
    ma = {int(k): [np.asarray(p, np.float32) for p in v] for k, v in d["map_infos"].items()}
    cams = []
    for c in CAM_ORDER:
        cams.append(os.path.join("data/bench2drive", info["sensors"][c]["data_path"]))
    return dict(
        idx=i, folder=info["folder"], clip=clip_of(info["folder"]), frame=int(info["frame_idx"]), town=info["town_name"],
        cams=cams,
        boxes=np.asarray(d["gt_bboxes_3d"], np.float32), names=names, ids=np.asarray(d["instance_inds"]),
        agent_fut=np.asarray(d["gt_agent_fut_trajs"], np.float32), agent_mask=np.asarray(d["gt_agent_fut_masks"], np.float32),
        ego_fut=np.asarray(d["gt_ego_fut_trajs"], np.float32), ego_mask=np.asarray(d["gt_ego_fut_masks"], np.float32),
        map=ma, tc=tc, tc_kind=tck,
        cmd_near=int(info["command_near"]), cmd_far=int(info["command_far"]),
        target_far=np.asarray(d["target_point_far"], np.float32), target_near=np.asarray(d["target_point_near"], np.float32),
        ego_vel=np.asarray(info["ego_vel"], np.float64), ego_accel=np.asarray(info["ego_accel"], np.float64),
        ego_rot=np.asarray(info["ego_rotation_rate"], np.float64), ego_size=np.asarray(info["ego_size"], np.float64),
        ego_yaw=float(info["ego_yaw"]), ego_xy=np.asarray(info["ego_translation"][:2], np.float64),
        steer=float(info["steer"]), throttle=float(info["throttle"]), brake=float(info["brake"]),
        lidar2ego=np.asarray(info["sensors"]["LIDAR_TOP"]["lidar2ego"], np.float64),
        lidar2img=[np.asarray(x, np.float64) for x in d["lidar2img"]], camera_names=list(d["camera_names"]),
    )


def cmd_extract(a):
    global _DS
    t0 = time.time()
    _DS = make_dataset(a.split)
    print(f"[extract] {a.split}: 데이터셋 {len(_DS)} 프레임, 적재 {time.time() - t0:.0f}s", flush=True)
    frames = {}
    for i in range(len(_DS)):
        inf = _DS.data_infos[i]
        frames[(clip_of(inf["folder"]), int(inf["frame_idx"]))] = i
    clips = sorted({c for c, _ in frames})
    # 키프레임 규칙(clip 기준): f0 = 그 clip 의 infos 첫 프레임, 키프레임 = f0, f0+5, f0+10, … (대부분 f0 = 0 → frame % 5 == 0).
    #   일부 clip 은 infos 가 frame 1 부터 시작한다(예: val VehicleTurningRoutePedestrian_Town15_Route481_Weather19) — frame 0 기준이면
    #   frame 1..4 의 키프레임이 없다. 폐쇄루프도 경로 첫 틱에 교사를 갱신하므로 clip 첫 프레임 기준이 같은 규칙이다.
    f0 = {}
    for c, f in frames:
        f0[c] = min(f, f0.get(c, f))
    kf = sorted(k for k in frames if (k[1] - f0[k[0]]) % KF == 0)
    kf_fill = gap_fill_keyframes(frames, f0)          # 빈 격자 칸 보충 키프레임(아래 함수 주석)
    kf = sorted(set(kf) | set(kf_fill))
    # 누락 검사: 모든 10 Hz 프레임이 나이 ≤ 0.4 s 의 키프레임(격자 또는 보충)을 갖는가(학생 프레임 → 교사 캐시 대응 보장)
    miss = []       # 격자 키프레임이 infos 에 없는 프레임(기록용)
    uncovered = []  # 보충 후에도 키프레임이 없는 프레임(있으면 실패)
    kset = set(kf)
    for c, f in frames:
        k = f0[c] + KF * ((f - f0[c]) // KF)
        if (c, k) not in frames:
            miss.append((c, f, k))
        if not any((c, f - d) in kset for d in range(KF)):
            uncovered.append((c, f))
    if uncovered:
        raise SystemExit(f"키프레임 없는 프레임 {len(uncovered)}: {uncovered[:5]}")
    n_f0_nonzero = sum(1 for c in clips if f0[c] != 0)
    if a.only_gap_fill:
        kf = sorted(kf_fill)
    if a.limit:
        kf = kf[:a.limit]
    t1 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        out = pool.map(_one, [frames[k] for k in kf], chunksize=16)
    dt = time.time() - t1
    prev = {}
    for s in out:
        s["prev_frame"] = s["frame"] - KF if (s["clip"], s["frame"] - KF) in frames else None
        s["clip_f0"] = f0[s["clip"]]
        prev[(s["clip"], s["frame"])] = s
    for s in out:
        p = prev.get((s["clip"], s["prev_frame"])) if s["prev_frame"] is not None else None
        if s["prev_frame"] is not None and p is None:      # --limit 으로 이전 키프레임을 추출하지 않은 경우 경로만 infos 에서
            inf = _DS.data_infos[frames[(s["clip"], s["prev_frame"])]]
            s["cams_prev"] = [os.path.join("data/bench2drive", inf["sensors"][c]["data_path"]) for c in CAM_ORDER]
            s["prev_ego_vel"] = np.asarray(inf["ego_vel"], np.float64)
        elif p is not None:
            s["cams_prev"] = p["cams"]; s["prev_ego_vel"] = p["ego_vel"]
        else:
            s["cams_prev"] = None; s["prev_ego_vel"] = None
    os.makedirs(RAW_DIR, exist_ok=True)
    path = f"{RAW_DIR}/{a.split}{'_limit%d' % a.limit if a.limit else ''}{'_gapfill' if a.only_gap_fill else ''}.pkl"
    meta = dict(split=a.split, n_frames=len(_DS), n_clips=len(clips), n_keyframes=len(kf), keyframe_rule="clip 첫 프레임 f0 기준 (frame_idx - f0) % 5 == 0", clips_f0_nonzero=n_f0_nonzero,
                missing_keyframe_frames=len(miss), missing_examples=miss[:20],
                gap_fill_keyframes=[list(x) for x in sorted(kf_fill)], extract_seconds=round(dt, 1),
                sec_per_sample=round(dt / max(len(out), 1) * a.workers, 4), workers=a.workers)
    with open(path, "wb") as f:
        pickle.dump(dict(meta=meta, samples=out), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(meta, ensure_ascii=False, indent=1), flush=True)
    print(f"[extract] → {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


def cmd_merge_gap_fill(a):
    """보충 키프레임 raw(<split>_gapfill.pkl)를 본 raw 에 합친다(같은 token 이면 실패). 이전 프레임 경로는 추출 때 infos 에서 채웠다."""
    p, q = f"{RAW_DIR}/{a.split}.pkl", f"{RAW_DIR}/{a.split}_gapfill.pkl"
    A, G = pickle.load(open(p, "rb")), pickle.load(open(q, "rb"))
    have = {(s["clip"], s["frame"]) for s in A["samples"]}
    add = [s for s in G["samples"] if (s["clip"], s["frame"]) not in have]
    if len(add) != len(G["samples"]):
        raise SystemExit("보충 키프레임이 이미 본 raw 에 있다")
    for s in add:
        s["gap_fill"] = True
    A["samples"] = sorted(A["samples"] + add, key=lambda s: (s["clip"], s["frame"]))
    A["meta"]["gap_fill_merged"] = [[s["clip"], s["frame"]] for s in add]
    A["meta"]["n_keyframes"] = len(A["samples"])
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(A, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)
    print(f"[merge] {len(add)} 보충 키프레임 → {p} (총 {len(A['samples'])})")


def load_raw(split):
    p = f"{RAW_DIR}/{split}.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- 규약 실측(좌표축·부호) — 라벨 정의 전에 반드시 확인
def cmd_conventions(a):
    R = load_raw(a.split)["samples"]
    res = {}
    # (1) +x = 오른쪽? 라이다 (+5, 10) 이 CAM_FRONT_RIGHT 에, (−5, 10) 이 CAM_FRONT_LEFT 에 보이는가(투영)
    s = R[len(R) // 2]
    vis = {}
    for name, M in zip(s["camera_names"], s["lidar2img"]):
        for p in ([8.0, 8.0, 0.0], [-8.0, 8.0, 0.0], [0.0, 15.0, 0.0]):
            q = M @ np.array(p + [1.0])
            if q[2] > 0 and 0 <= q[0] / q[2] < 1600 and 0 <= q[1] / q[2] < 900:
                vis.setdefault(str(p[:2]), []).append(name)
    res["projection"] = vis
    res["x_right"] = "CAM_FRONT_RIGHT" in vis.get("[8.0, 8.0]", []) and "CAM_FRONT_LEFT" in vis.get("[-8.0, 8.0]", [])
    res["y_forward"] = "CAM_FRONT" in vis.get("[0.0, 15.0]", [])
    # (2) 자차 크기 일정?  (3) 라이다 → 자차 중심 오프셋(lidar2ego 의 역: 자차 원점이 라이다 좌표에서 어디인가)
    sz = np.array([x["ego_size"] for x in R])
    res["ego_size_unique"] = np.unique(sz.round(4), axis=0).tolist()
    l2e = np.array([x["lidar2ego"] for x in R])
    res["lidar2ego_unique"] = np.unique(l2e.round(4), axis=0).tolist()[:3]
    e2l = np.linalg.inv(l2e[0])
    res["ego_origin_in_lidar_xy"] = e2l[:2, 3].round(4).tolist()
    # (4) 요레이트 부호: rot z vs 미래 0.5 s 진행방향 변화(+ = 왼쪽, heading_series 규약) 상관
    ys, zs, ax, dv, vx = [], [], [], [], []
    for x in R:
        if not (x["ego_mask"] == 1).all() or x["ego_vel"][0] < 2.0:
            continue
        fut = x["ego_fut"].astype(np.float64).cumsum(0)
        xy = np.concatenate([np.zeros((1, 2)), fut], 0)
        th = B.heading_series(xy)
        ys.append(th[1] / 0.5); zs.append(x["ego_rot"][2])
        if x["prev_ego_vel"] is not None:
            ax.append(x["ego_accel"][0]); dv.append((x["ego_vel"][0] - x["prev_ego_vel"][0]) / 0.5)
        vx.append([x["ego_vel"][0], np.linalg.norm(fut[0]) / 0.5])
    ys, zs = np.array(ys), np.array(zs)
    res["yaw_rate_corr_rotz_vs_fut_heading_rate"] = float(np.corrcoef(zs, ys)[0, 1])
    res["yaw_rate_regress_slope"] = float(np.polyfit(ys, zs, 1)[0])
    ax, dv = np.array(ax), np.array(dv)
    res["accel_x_corr_vs_dv_past"] = float(np.corrcoef(ax, dv)[0, 1]) if len(ax) > 2 else None
    res["accel_x_abs_p99"] = float(np.percentile(np.abs(ax), 99)) if len(ax) else None
    res["accel_x_abs_max"] = float(np.abs(ax).max()) if len(ax) else None
    res["dv_past_abs_p99"] = float(np.percentile(np.abs(dv), 99)) if len(dv) else None
    vx = np.array(vx)
    res["speed_vel0_vs_fut0_corr"] = float(np.corrcoef(vx[:, 0], vx[:, 1])[0, 1])
    res["speed_max"] = float(max(x["ego_vel"][0] for x in R))
    res["speed_min"] = float(min(x["ego_vel"][0] for x in R))
    # (5) 객체 속도 = 라이다 좌표? vx,vy vs 첫 0.5 s 증분/0.5 상관
    a1, a2 = [], []
    for x in R:
        for j in range(len(x["boxes"])):
            if x["agent_mask"][j, 0] == 1:
                a1.append(x["boxes"][j, 7:9]); a2.append(x["agent_fut"][j, 0] / 0.5)
    a1, a2 = np.array(a1), np.array(a2)
    res["obj_vel_corr_x"] = float(np.corrcoef(a1[:, 0], a2[:, 0])[0, 1]); res["obj_vel_corr_y"] = float(np.corrcoef(a1[:, 1], a2[:, 1])[0, 1])
    res["obj_vel_nan"] = int((~np.isfinite(a1)).any(1).sum())
    # (6) 박스 yaw 규약: 이동 객체의 방향벡터(cos yaw, sin yaw) 와 이동 방향 내적
    dots = []
    for x in R:
        for j in range(len(x["boxes"])):
            v = x["boxes"][j, 7:9]
            if np.linalg.norm(v) > 2 and x["names"][j] in ("car", "van", "truck"):
                h = np.array([np.cos(x["boxes"][j, 6]), np.sin(x["boxes"][j, 6])])
                dots.append(float(h @ v / np.linalg.norm(v)))
    res["yaw_vs_motion_dot_mean"] = float(np.mean(dots)) if dots else None
    res["yaw_vs_motion_dot_p5"] = float(np.percentile(dots, 5)) if dots else None
    # (7) 지도 개수·명령 분포·목표점 크기
    res["map_counts_max"] = {str(c): int(max(len(x["map"].get(c, [])) for x in R)) for c in range(4)}
    res["cmd_near"] = dict(collections.Counter(x["cmd_near"] for x in R))
    res["target_far_abs_max"] = float(max(np.abs(x["target_far"]).max() for x in R))
    res["tc_kinds"] = dict(collections.Counter(k for x in R for k in x["tc_kind"]))
    res["n"] = len(R)
    os.makedirs(os.path.dirname(CONV_FILE), exist_ok=True)
    json.dump(res, open(CONV_FILE.replace(".json", f"_{a.split}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


def load_conventions():
    p = CONV_FILE.replace(".json", "_train.json")
    if not os.path.exists(p):
        raise SystemExit(f"{p} 없음 — 먼저 conventions --split train")
    c = json.load(open(p))
    if not (c["x_right"] and c["y_forward"]):
        raise SystemExit(f"라이다 축 규약이 +x 오른쪽·+y 전방이 아니다: {c['projection']} — 라벨 정의 재검토 필요")
    return c


def apply_geometry(conv):
    """자차 크기(전 표본 같은 값 검사)·라이다→자차 중심 전방 오프셋을 common_b2d 에 넣는다."""
    sz = conv["ego_size_unique"]
    if len(sz) != 1:
        raise SystemExit(f"자차 크기가 표본마다 다르다: {sz[:5]}")
    w, l = sz[0][0], sz[0][1]
    if abs(l - B.EGO_L) > 1e-3 or abs(w - B.EGO_W) > 1e-3:
        raise SystemExit(f"common_b2d EGO_L/W({B.EGO_L},{B.EGO_W}) 와 infos ego_size({l},{w}) 불일치")
    ox, oy = conv["ego_origin_in_lidar_xy"]
    B.set_ego_geom(oy)   # 자차 원점(박스 중심 가정)의 라이다 전방(+y) 좌표


# ---------------------------------------------------------------- 자차 상태(프롬프트 입력)
def ego_inputs_b2d(s, conv):
    """(speed, accel_long, yaw_rate(+ = 왼쪽)) — 학생 ego_status 와 같은 시뮬레이터 값(현재 프레임만, 미래 미사용).
    speed = ego_vel[0](자차 전방 속도, 음수 미세값(−0.07 등 후진 떨림)은 0 으로 — 고정폭 부호 없는 형식),
    accel_long = ego_accel[0](IMU 전방 가속, 학생 ego_status[1] 과 같은 값; conventions: 과거 0.5 s 속도 차분과 상관 +0.77),
    yaw_rate = ego_rotation_rate[2] × sign(conventions 회귀 기울기) — '+ = 왼쪽'(nuScenes CAN 규약)으로 맞춘다."""
    sgn = 1.0 if conv["yaw_rate_regress_slope"] > 0 else -1.0
    return (max(float(s["ego_vel"][0]), 0.0), float(s["ego_accel"][0]), sgn * float(s["ego_rot"][2])), "sim"


# ---------------------------------------------------------------- TTC 간격 임계값(train 전용)
def _gap_one(s):
    if not B.future_complete(s):
        return []
    v0 = float(s["ego_vel"][0])
    objs, _, _ = B.select_objects(s, v0)
    ego = B.ego_track(s)
    return [B.agent_gap_series(s, r["box_index"], ego).tolist() for r in objs]


def cmd_explore_ttc(a):
    conv = load_conventions(); apply_geometry(conv)
    R = load_raw("train")["samples"]
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(_gap_one, R, chunksize=64)
    G = np.asarray([g for r in res for g in r], np.float64)
    m3 = G[:, :7].min(1)
    out = dict(n_samples=sum(1 for r in res if r), n_objects=len(G),
               min_gap_0_3s_percentiles={p: round(float(np.percentile(m3, p)), 4) for p in (1, 5, 10, 25, 50)},
               p5_rounded=round(float(np.percentile(m3, 5)), 1), buckets={})
    for th in (0.0, 0.5, 1.0, 1.4, round(float(np.percentile(m3, 5)), 1), 2.0):
        c = collections.Counter(B.ttc_bucket(g, th)[0] for g in G)
        out["buckets"][str(th)] = {k: round(c[k] / len(G), 4) for k in ("lt2", "2_4", "4_6", "safe")}
    json.dump(out, open(TTC_FILE, "w"), indent=1)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------- 정답
def r4(x):
    return [[round(float(v), 4) for v in p] for p in x]


_CTX = None   # (표본 목록, 규약, 코드북, TTC 간격) — fork 로 워커에 상속(피클 불가 지역 클래스 회피)


def _answer_one(i):
    R, conv, cb, gap = _CTX
    return build_record(R[i], conv, cb, gap)


def build_record(s, conv, cb, gap):
    ego_vals, ego_src = ego_inputs_b2d(s, conv)
    v0 = ego_vals[0]
    prev_ok = s["cams_prev"] is not None
    objs, n_in, sel_meta = B.select_objects(s, v0)
    B.enrich_objects(s, objs, gap)
    mp_ = B.map_context_sensor(s)
    complete = B.future_complete(s)
    fut = B.fut_cum(s)
    intent = toks = rec = None
    weight = 0.0
    if complete:
        intent = B.intent_labels(s, v0)
        weight = B.UNCERTAIN_WEIGHT if intent["uncertain"] else 1.0
        toks, rec = cb.tokenize(fut)
    intent_ans = None if intent is None else dict(longitudinal=intent["longitudinal"], lateral=intent["lateral"])
    answer = B.build_answer(objs, mp_, intent_ans, toks)
    parts = B.build_prompt_parts(s["cmd_near"], s["target_far"], ego_vals)
    user = B.join_prompt(parts)
    spans = B.answer_spans(answer)
    lm = B.loss_mask_spans(spans, complete, prev_ok)
    cbase = B.copy_baseline(s["cmd_near"], ego_vals, cb)
    if complete:
        cbase["ade_vs_gt"] = round(float(np.linalg.norm(np.asarray(cbase["traj"]) - fut, axis=1).mean()), 4)
    token = f"{s['clip']}/{s['frame']:05d}"
    return dict(
        token=token, scene_token=s["clip"], timestamp=s["frame"] / 10.0, frame_idx=s["frame"], town=s["town"],
        cams=s["cams"], cams_prev=s["cams_prev"] if prev_ok else s["cams"], prev_available=prev_ok,
        prev_token=None if not prev_ok else f"{s['clip']}/{s['prev_frame']:05d}",
        command=dict(id=B.norm_command(s["cmd_near"]), text=B.CMD_TEXT_B2D[B.norm_command(s["cmd_near"])], raw=s["cmd_near"]),
        target_point=[round(float(v), 3) for v in s["target_far"]],
        ego_status=dict(src=ego_src, available=True, speed=round(ego_vals[0], 4), accel_long=round(ego_vals[1], 4),
                        yaw_rate=round(ego_vals[2], 5), raw=dict(ego_vel=s["ego_vel"].tolist(), ego_accel=s["ego_accel"].tolist(),
                                                                ego_rot=s["ego_rot"].tolist())),
        ego_drop_ok=True, command_drop_ok=True,
        prompt=dict(system=B.SYSTEM_PROMPT, user=user), prompt_parts=parts,
        answer=answer, answer_spans=spans, loss_mask_spans=lm,
        modes=dict(ego_in_prompt="current", object_speed="numeric", answer_format=B.ANSWER_FORMAT, dataset="b2d"),
        targets=dict(objects=objs, n_objects_in_range=n_in, object_selection=sel_meta, map=mp_, future_complete=complete,
                     fut_mask=[int(v) for v in s["ego_mask"]], fut_traj=r4(fut), label_v0=round(v0, 4), intent=intent,
                     intent_weight=weight, action_tokens=toks, action_recon_traj=None if rec is None else r4(rec),
                     action_recon_ade=None if rec is None else round(float(np.linalg.norm(rec - fut, axis=1).mean()), 5)),
        copy_baseline=cbase,
    )


def cmd_answers(a):
    from transformers import AutoTokenizer
    import common as C1
    conv = load_conventions(); apply_geometry(conv)
    if not os.path.exists(TTC_FILE):
        raise SystemExit(f"{TTC_FILE} 없음 — 먼저 explore_ttc")
    gap = json.load(open(TTC_FILE))["p5_rounded"]
    B.set_ttc_gap(gap)
    z = np.load(CODEBOOK)
    K = len(z["codebook"])
    B.set_action_k(K)

    class CB(C1.ActionCodebook):           # nuScenes 판은 K == 2041 을 강제하므로 B2D 크기로 다시 만든다(알고리즘 동일)
        def __init__(self, book):
            self.book = np.asarray(book, np.float64); self.K = len(self.book)
    cb = CB(z["codebook"])
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    n_added = tok.add_tokens(B.SPECIAL_TOKENS, special_tokens=True)
    assert n_added == len(B.SPECIAL_TOKENS), (n_added, len(B.SPECIAL_TOKENS))
    out_dir = a.out_dir
    os.makedirs(out_dir, exist_ok=True)
    outs = [f"{out_dir}/teacher_gt_b2d_{s}.jsonl" for s in a.splits]
    if any(os.path.exists(p) for p in outs) and not a.force:
        raise SystemExit(f"[중단] 기존 산출물 있음: {[p for p in outs if os.path.exists(p)]} (--force)")
    with open(f"{out_dir}/special_tokens.txt", "w") as f:
        f.write("\n".join(B.SPECIAL_TOKENS) + "\n")
    with open(f"{out_dir}/grammar.txt", "w") as f:
        f.write(B.GRAMMAR_B2D.replace("TTC_GAP_B2D", f"{gap} m").replace("ACTION_K_B2D", str(K)))
    with open(f"{out_dir}/instruction.txt", "w") as f:
        f.write("[SYSTEM]\n" + B.SYSTEM_PROMPT + "\n\n[USER — 이미지 12장 뒤 텍스트 = prompt_parts 를 '\\n' 로 이음]\n" +
                B.build_instruction() + "\nCommand: <turn left ahead|turn right ahead|go straight ahead|follow the lane|change lane left|change lane right>."
                "   |   " + B.CMD_DROPPED + "\nTarget point: x <+xxxx.x> m, y <+yyyy.y> m.\n"
                "Ego state: speed <vv.v> m/s, longitudinal acceleration <+aa.aa> m/s^2, yaw rate <+w.www> rad/s (positive = turning left)."
                "   |   " + B.EGO_DROPPED + "\n")
    stats = dict(version="v2-sensor-b2d", answer_format=B.ANSWER_FORMAT, conventions=conv, ttc_gap_m=gap, action_K=K,
                 codebook=dict(path=CODEBOOK, val_ade=float(z["val_ade"]), val_fde=float(z["val_fde"])),
                 ego_geom=dict(L=B.EGO_L, W=B.EGO_W, fwd=B.EGO_FWD), special_tokens=len(B.SPECIAL_TOKENS), splits={})
    ans_len = None
    for split, path in zip(a.splits, outs):
        raw = load_raw(split if not a.raw_suffix else f"{split}{a.raw_suffix}")
        R = raw["samples"]
        t0 = time.time()
        recs = [build_record(s, conv, cb, gap) for s in R] if a.workers <= 1 else None
        if recs is None:
            global _CTX
            _CTX = (R, conv, cb, gap)
            with mp.get_context("fork").Pool(a.workers) as pool:
                recs = pool.map(_answer_one, range(len(R)), chunksize=32)
        S = collections.defaultdict(collections.Counter)
        lens = []
        with open(path, "w") as fo:
            for r in recs:
                enc = tok(r["answer"], return_offsets_mapping=True, add_special_tokens=False)
                n = len(enc["input_ids"])
                if ans_len is None:
                    ans_len = n
                if n != ans_len:
                    raise ValueError(f"{r['token']}: 정답 토큰 {n} != {ans_len} (고정 길이 위반)")
                offs = enc["offset_mapping"]
                r["loss_mask_token_spans"] = [[min(k for k, (s0, e0) in enumerate(offs) if e0 > a0),
                                               max(k for k, (s0, e0) in enumerate(offs) if s0 < b0) + 1] for a0, b0 in r["loss_mask_spans"]]
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                lens.append(len(tok(r["prompt"]["system"] + "\n" + r["prompt"]["user"])["input_ids"]))
                t = r["targets"]
                S["future_complete"][str(t["future_complete"])] += 1
                S["prev_available"][str(r["prev_available"])] += 1
                S["command"][r["command"]["text"]] += 1
                S["n_objects"][str(len(t["objects"]))] += 1
                for o in t["objects"]:
                    S["obj_class"][o["name"]] += 1; S["obj_reason"][o["reason"]] += 1; S["obj_ttc"][str(o["ttc"])] += 1
                    S["obj_motion"][o["motion"]] += 1
                if t["intent"] is not None:
                    S["longitudinal"][t["intent"]["longitudinal"]] += 1; S["lateral"][t["intent"]["lateral"]] += 1
                    S["uncertain"][str(t["intent"]["uncertain"])] += 1
                    S["command_vs_lateral"][f"{r['command']['text']} -> {t['intent']['lateral']}"] += 1
                    sig = t["intent"].get("lc_signals")
                    if sig:
                        S["lc_signal_combo"][f"b={sig['b_divider']} c={sig['c_center']} c_ok={sig['c_ok']}"] += 1
                m = t["map"]
                if m is None:
                    S["map"]["na"] += 1
                else:
                    pc = m["ped_crossing"]
                    S["map"]["tc_" + ("none" if pc is None else "on" if pc["sector"] == "here" else pc.get("kind", "near"))] += 1
                    S["map_free"]["capped" if m["free_dist"] >= B.FREE_CAP else "blocked"] += 1
                    for y, d in m["lanes"].items():
                        for k, v in d.items():
                            S["map_lanes_present"][f"y{y}_{k}"] += int(v is not None)
                S["loss_mask"]["samples_with_na"] += int(bool(r["answer_spans"]["na"]))
        n = len(recs)
        ade = [r["targets"]["action_recon_ade"] for r in recs if r["targets"]["action_recon_ade"] is not None]
        stats["splits"][split] = dict(samples=n, clips=len({r["scene_token"] for r in recs}), seconds=round(time.time() - t0, 1),
                                      answer_tokens=ans_len, prompt_tokens_text=dict(min=min(lens), max=max(lens)),
                                      action_recon_ade_m=round(float(np.mean(ade)), 5) if ade else None,
                                      bytes=os.path.getsize(path), **{k: dict(v) for k, v in S.items()})
        print(f"[answers] {split}: {n} → {path} ({os.path.getsize(path) / 1e6:.1f} MB, {time.time() - t0:.0f}s)", flush=True)
    json.dump(stats, open(f"{out_dir}/stats.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in stats.items() if k != "conventions"}, indent=1, ensure_ascii=False)[:6000])


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract"); e.add_argument("--split", required=True, choices=["train", "val"])
    e.add_argument("--limit", type=int, default=0); e.add_argument("--workers", type=int, default=8)
    e.add_argument("--only_gap_fill", action="store_true", help="보충 키프레임만 따로 추출(기존 raw 에 merge_gap_fill 로 합침)")
    mg = sp.add_parser("merge_gap_fill"); mg.add_argument("--split", required=True)
    c = sp.add_parser("conventions"); c.add_argument("--split", default="train")
    t = sp.add_parser("explore_ttc"); t.add_argument("--workers", type=int, default=8)
    w = sp.add_parser("answers"); w.add_argument("--out_dir", required=True)
    w.add_argument("--splits", nargs="+", default=["train", "val"]); w.add_argument("--force", action="store_true")
    w.add_argument("--workers", type=int, default=8); w.add_argument("--raw_suffix", default="")
    a = ap.parse_args()
    os.chdir(REPO)
    dict(extract=cmd_extract, merge_gap_fill=cmd_merge_gap_fill, conventions=cmd_conventions, explore_ttc=cmd_explore_ttc, answers=cmd_answers)[a.cmd](a)


if __name__ == "__main__":
    main()
