# 수정(2026-11-09): 새 파일 — B2D 폐쇄루프 센서·제어 주기 정합 CPU 점검(B-A6). CARLA·GPU 불필요, 소스 수정 없음.
"""에이전트(hipad_b2d_agent.py) 센서 정의·하드코딩 행렬 ↔ 학습 데이터(b2d_infos) 카메라 6 대 내·외부 파라미터, 제어 주기 점검.

  source /home1/irteam/data-vol1/hipad_env.sh
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tools/verify_b2d/check_sensor_pid_b2d.py [--split val|train] [--json out.json]

(1) 에이전트 sensors() 의 (x, y, z, roll, pitch, yaw, w, h, fov) → 핀홀 내부 K(f = w / (2 tan(fov/2))) 와 cam2ego(CARLA 왼손 좌표
    x 앞·y 오른쪽 → 데이터 ego x 앞·y 왼쪽: y → −y, yaw → −yaw) 를 만들어 infos 전 프레임의 intrinsic·cam2ego 와 비교.
(2) 에이전트 하드코딩 LIDAR2CAM·LIDAR2IMG·CAM2IMG·LIDAR2EGO ↔ 데이터셋 get_data_info 식(ego2cam = inv(cam2ego),
    lidar2cam = ego2cam @ lidar2ego, lidar2img = K @ lidar2cam) 을 infos 값으로 계산한 것.
(3) 카메라 순서(학생 입력 = infos sensors 순서 / 교사 = qwen_common 순서), 영상 크기·JPEG 품질(데이터 파일 양자화표 추정 vs 에이전트 20).
(4) 시간: 평가기 fixed_delta·에이전트 frame_rate·디코더 close/open loop Hz·교사 갱신·IMU/GNSS sensor_tick·PID 창·waypoint_time.
파일을 고치지 않는다. 결과 표를 출력하고(--json 이면 저장) 불일치가 있으면 종료 코드 1.
"""
import argparse
import ast
import json
import math
import os
import pickle
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AGENT = os.path.join(REPO, "bench2drive/leaderboard/team_code/hipad_b2d_agent.py")
TEACHER_AGENT = os.path.join(REPO, "bench2drive/leaderboard/team_code/hipad_b2d_laq_teacher_agent.py")
EVALUATOR = os.path.join(REPO, "bench2drive/leaderboard/leaderboard/leaderboard_evaluator.py")
CFG = os.path.join(REPO, "configs/b2d/v2_teacher/stage2/test_e3_nuim.py")
ROWS, BAD = [], []


def row(item, agent, data, ok, note=""):
    ROWS.append(dict(item=item, agent=agent, data=data, ok=bool(ok), note=note))
    if not ok:
        BAD.append(item)


def agent_constants():
    """에이전트 모듈 최상위 상수(np.array 리터럴)만 numpy 로 평가 — carla·mmcv import 없이."""
    tree = ast.parse(open(AGENT).read())
    env = {"np": np}
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("LIDAR2IMG", "LIDAR2CAM", "CAM2IMG", "LIDAR2EGO", "CAMERA"):
                # 저장소 자체 소스(에이전트 파일)의 np.array 리터럴만 평가 — 외부 입력 아님
                out[name] = eval(compile(ast.Expression(node.value), AGENT, "eval"), env)
    return out


def agent_sensors():
    """sensors() 의 리스트 리터럴에서 카메라 dict 를 읽는다(조건부 bev 카메라 제외)."""
    src = open(AGENT).read()
    tree = ast.parse(src)
    cams, other = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "sensors":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    try:
                        d = ast.literal_eval(sub)
                    except ValueError:
                        continue
                    if d.get("type") == "sensor.camera.rgb" and str(d.get("id", "")).startswith("CAM_"):
                        cams[d["id"]] = d
                    elif "id" in d:
                        other[d["id"]] = d
    return cams, other


def k_of(w, h, fov):
    f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


CAM_AXES = np.array([[0, 0, 1.0], [-1, 0, 0], [0, -1, 0]])   # 카메라(x 오른쪽·y 아래·z 앞) → ego(x 앞·y 왼쪽·z 위), yaw 0


def cam2ego_of(d):
    if abs(d.get("roll", 0)) > 1e-9 or abs(d.get("pitch", 0)) > 1e-9:
        raise NotImplementedError("roll/pitch ≠ 0 미지원")
    T = np.eye(4)
    T[:3, :3] = rz(-d["yaw"]) @ CAM_AXES
    T[:3, 3] = [d["x"], -d["y"], d["z"]]
    return T


def jpeg_quality(path):
    """PIL 양자화표(휘도) → IJG 품질 추정(표준 표 비율의 중앙값 역산)."""
    from PIL import Image
    std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29,
                    51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120,
                    101, 72, 92, 95, 98, 112, 100, 103, 99], float)
    im = Image.open(path)
    q = getattr(im, "quantization", None)
    if not q:
        return None, im.size
    t = np.array(q[0], float)
    zz = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42,
          49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]
    natural = np.empty(64); natural[zz] = t   # PIL 은 지그재그 순서
    best = min(range(1, 101), key=lambda Q: np.abs(np.clip(np.floor((std * (5000 / Q if Q < 50 else 200 - 2 * Q) + 50) / 100), 1, 255)
                                                    - natural).sum())
    return best, im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--json")
    a = ap.parse_args()
    C = agent_constants()
    cams, other = agent_sensors()
    # 저장소 자체 산출물(data_converter) — 학습 파이프라인과 같은 파일
    infos = pickle.load(open(os.path.join(REPO, f"data/infos/b2d_infos_{a.split}.pkl"), "rb"))
    infos = infos["infos"] if isinstance(infos, dict) else infos
    names = list(infos[0]["sensors"].keys())
    cam_names = [n for n in names if n.startswith("CAM_")]
    row("학생 카메라 순서(에이전트 CAMERA ↔ infos sensors 순서 = 데이터셋 get_data_info 순회 순서)", C["CAMERA"], cam_names,
        list(C["CAMERA"]) == cam_names)
    # (1) 내부·외부 — infos 전 프레임에서 값이 하나인지 + 에이전트 정의와 같은지
    for c in C["CAMERA"]:
        d = cams[c]
        Ka, Ea = k_of(d["width"], d["height"], d["fov"]), cam2ego_of(d)
        Ks = np.stack([np.asarray(i["sensors"][c]["intrinsic"], float) for i in infos])
        Es = np.stack([np.asarray(i["sensors"][c]["cam2ego"], float) for i in infos])
        uniq_k = float(np.abs(Ks - Ks[0]).max()); uniq_e = float(np.abs(Es - Es[0]).max())
        dk = float(np.abs(Ks - Ka).max()); de = float(np.abs(Es - Ea).max())
        row(f"{c} 내부 K (f, cx, cy)", f"w{d['width']} h{d['height']} fov{d['fov']} → f {Ka[0, 0]:.4f}, ({Ka[0, 2]:.0f},{Ka[1, 2]:.0f})",
            f"f {Ks[0, 0, 0]:.4f}, ({Ks[0, 0, 2]:.0f},{Ks[0, 1, 2]:.0f}); 프레임 간 최대 차 {uniq_k:.2g}", dk < 1e-3, f"최대 차 {dk:.2g}")
        row(f"{c} 외부 cam2ego (x,y,z,yaw)", f"({d['x']}, {d['y']}, {d['z']}, yaw {d['yaw']}) [CARLA]",
            f"t ({Es[0, 0, 3]:.2f}, {Es[0, 1, 3]:.2f}, {Es[0, 2, 3]:.2f}), yaw {math.degrees(math.atan2(Es[0, 1, 2], Es[0, 0, 2])):.1f}° [ego, y 왼쪽]; "
            f"프레임 간 최대 차 {uniq_e:.2g}", de < 1e-3, f"최대 차 {de:.2g}")
    # (2) 하드코딩 행렬 ↔ 데이터셋 식
    l2e = np.asarray(infos[0]["sensors"]["LIDAR_TOP"]["lidar2ego"], float)
    l2e_all = max(float(np.abs(np.asarray(i["sensors"]["LIDAR_TOP"]["lidar2ego"], float) - l2e).max()) for i in infos)
    row("LIDAR2EGO", np.round(C["LIDAR2EGO"][:3, 3], 3).tolist(), np.round(l2e[:3, 3], 3).tolist(),
        np.abs(C["LIDAR2EGO"] - l2e).max() < 1e-6 and l2e_all < 1e-9, f"회전 차 {np.abs(C['LIDAR2EGO'] - l2e).max():.2g}, 프레임 간 {l2e_all:.2g}")
    for c in C["CAMERA"]:
        e2c = np.linalg.inv(np.asarray(infos[0]["sensors"][c]["cam2ego"], float))
        Kp = np.eye(4); Kp[:3, :3] = np.asarray(infos[0]["sensors"][c]["intrinsic"], float)
        l2c = e2c @ l2e
        l2i = Kp @ l2c
        d1 = float(np.abs(C["LIDAR2CAM"][c] - l2c).max())
        d2 = float(np.abs(C["LIDAR2IMG"][c] - l2i).max() / np.abs(l2i).max())
        d3 = float(np.abs(C["CAM2IMG"][c] - Kp).max())
        row(f"{c} LIDAR2CAM / LIDAR2IMG(상대) / CAM2IMG", f"하드코딩", f"데이터셋 식", d1 < 1e-3 and d2 < 1e-6 and d3 < 1e-3,
            f"최대 차 {d1:.2g} / {d2:.2g} / {d3:.2g}")
    # (3) 교사 영상 순서·크기·JPEG
    tsrc = open(TEACHER_AGENT).read()
    tord = re.search(r"TEACHER_CAM_ORDER = \(([^)]*)\)", tsrc).group(1).replace('"', "").replace(" ", "").split(",")
    sys.path.insert(0, os.path.join(REPO, "tools/vla_teacher"))
    gsrc = open(os.path.join(REPO, "tools/vla_teacher/gen_teacher_gt_b2d.py")).read()
    gord = ast.literal_eval(re.search(r"^CAM_ORDER = (\[.*\])", gsrc, re.M).group(1))
    row("교사 영상 순서(에이전트 TEACHER_CAM_ORDER ↔ 교사 정답 생성기 CAM_ORDER)", tord, gord, tord == list(gord))
    p0 = os.path.join(REPO, "data/bench2drive", infos[0]["sensors"]["CAM_FRONT"]["data_path"])
    qs = []
    for i in infos[:: max(1, len(infos) // 20)][:20]:
        for c in ("CAM_FRONT", "CAM_BACK"):
            q, size = jpeg_quality(os.path.join(REPO, "data/bench2drive", i["sensors"][c]["data_path"]))
            qs.append((q, size))
    asrc = open(AGENT).read()
    aq = int(re.search(r"IMWRITE_JPEG_QUALITY\), (\d+)\]", asrc).group(1))
    row("영상 크기·JPEG 품질(에이전트 cv2.imencode ↔ 데이터 파일 양자화표 추정, 40 장)", f"{cams['CAM_FRONT']['width']}×{cams['CAM_FRONT']['height']}, q{aq}",
        f"{sorted(set(s for _, s in qs))}, q {sorted(set(q for q, _ in qs))}", all(s == (1600, 900) for _, s in qs) and all(q == aq for q, _ in qs),
        "에이전트는 RGB 배열을 BGR 로 간주해 인코딩(채널 뒤바뀐 YCbCr 양자화) — 색차 오차 분포가 데이터(BGR 저장 추정)와 다를 수 있음, 아래 위험")
    # (4) 시간
    ev = open(EVALUATOR).read()
    fr = float(re.search(r"frame_rate = ([\d.]+)", ev).group(1))
    afr = int(re.search(r"self\.frame_rate = (\d+)", asrc).group(1))
    from mmcv import Config
    cfg = Config.fromfile(CFG)
    dec = cfg.model.head.onedecoder_head
    row("시뮬레이터 Δt(평가기 fixed_delta = 1/frame_rate) ↔ 에이전트 frame_rate(timestamp = step/20)", f"1/{fr:g} s", f"에이전트 {afr} Hz",
        fr == afr == 20)
    usrc = open(os.path.join(REPO, "models/decoder/unified_decoder.py")).read()   # config 에 없으면 생성자 기본값
    dflt = {k: int(re.search(rf"{k}=(\d+)", usrc).group(1)) for k in ("close_loop_hz", "open_loop_hz")}
    hz = (dec.get("close_loop_hz", dflt["close_loop_hz"]), dec.get("open_loop_hz", dflt["open_loop_hz"]))
    row("디코더 close_loop_hz / open_loop_hz (에이전트 setup 이 20/2 강제)", "20 / 2",
        f"{hz[0]} / {hz[1]} (config {'값' if 'close_loop_hz' in dec else '없음 → unified_decoder 기본값'}), num_stream_groups {dec.get('num_stream_groups')}",
        hz == (20, 2) and dec.get("num_stream_groups") == 1, "학습 데이터 10 Hz, 시간 뱅크 0.5 s 간격(2 Hz) — 에이전트는 with_close_loop=True 로 뱅크 10 벌")
    imu = other.get("IMU", {}); gps = other.get("GPS", {})
    row("IMU / GNSS sensor_tick", f"{imu.get('sensor_tick')} / {gps.get('sensor_tick')} s", "Δt 0.05 s",
        imu.get("sensor_tick") == 0.05, "GNSS 0.01 s < Δt → 매 틱 최신값")
    m = re.search(r"PIDController\((.*?)\)\n", asrc, re.S)
    pid = dict(re.findall(r"(\w+)=([\d.]+)", m.group(1)))
    row("PID waypoint_time ↔ 계획 출력 plan_speed_5hz 간격", f"{pid.get('waypoint_time')} s", "5 Hz = 0.2 s", float(pid["waypoint_time"]) == 0.2,
        f"turn K {pid.get('turn_KP')}/{pid.get('turn_KI')}/{pid.get('turn_KD')} n{pid.get('turn_n')}, speed K {pid.get('speed_KP')}/"
        f"{pid.get('speed_KI')}/{pid.get('speed_KD')} n{pid.get('speed_n')} — 창 10 틱 = 0.5 s(20 Hz), 미분 = 틱 차분(Δt 미포함): HiP-AD 원본 값·원본 20 Hz 그대로")
    row("교사 갱신 주기", "0.5 s (10 틱)", "캐시 키프레임 2 Hz(나이 0–0.4 s)", True, "폐쇄루프는 나이 0.45 s 틱이 추가로 생김(학습에 없음)")
    print(f"\n| 항목 | 에이전트 | 학습 데이터/기대 | 판정 | 비고 |\n|---|---|---|---|---|")
    for r in ROWS:
        print(f"| {r['item']} | {r['agent']} | {r['data']} | {'일치' if r['ok'] else '**불일치**'} | {r['note']} |")
    print(f"\n프레임 {len(infos)} ({a.split}), 불일치 {len(BAD)}: {BAD}")
    if a.json:
        json.dump(dict(split=a.split, n_frames=len(infos), rows=ROWS, bad=BAD), open(a.json, "w"), indent=1, ensure_ascii=False, default=str)
    sys.exit(1 if BAD else 0)


if __name__ == "__main__":
    main()
