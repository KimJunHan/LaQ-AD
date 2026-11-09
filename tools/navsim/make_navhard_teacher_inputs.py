# 수정(2026-11-05): navhard_two_stage 평가 준비 — 교사 캐시 입력(프롬프트) 레코드. 새 파일, 기존 소스 무수정. CPU 전용.
"""navhard info 의 모든 고유 프레임(학생 test 파이프라인이 프레임마다 교사 캐시를 읽는다)에 대해 교사 캐시 추출 입력 레코드를 만든다.
정답(answer/targets)은 만들지 않는다 — 추출기(extract_teacher_cache_navsim.py)는 KEEP_KEYS 입력 필드만 읽는다(strip_to_input).

규칙 = gen_teacher_gt_navsim.py(navtest GT 생성기)와 같다:
  - 영상: 현재 전방 3 캠(CAM_L0·F0·R0) + 직전 프레임(t−0.5 s) 3 캠(prev_first). 직전 프레임이 없으면 현재 영상을 반복하고 prev_available=False.
  - 명령: driving_command one-hot 의 argmax → N.command_line. 자차 줄: speed=|(vx,vy)|, accel_long=ax, yaw rate=(ψ0−ψ−1)/Δt
    (직전 포즈 없으면 'unavailable'). user = C.join_prompt(N.build_prompt_parts(cmd, ego, gap)), gap = navtest GT 의 gap_m(1.2).
  1단계(원본) 프레임: navtest GT(teacher_gt_navtest.jsonl)에 같은 token 이 있으면 그 레코드의 입력 필드를 그대로 쓴다(navtest 캐시와
    같은 입력 → 같은 교사 출력). 없으면 위 규칙으로 로그에서 만든다(직전 프레임 = 로그 k−1, Δt ≤ 0.75 s, 3 캠 파일 존재).
    겹치는 token 은 '규칙으로 만든 것'과 navtest 레코드를 대조해 일치 수를 보고한다(규칙 재현 검사).
  2단계(합성) 프레임: 직전 프레임 = 같은 합성 궤적의 앞 프레임(합성 장면들의 frames[k−1] → frames[k] 관계 합집합; 충돌 시 중단).
    자차 포즈 ψ = 합성 ego_pose heading(전역), 속도·가속 = 합성 ego_status(국소). 앞 프레임이 없는 합성 첫 프레임(2,731 개)은
    자차 줄 unavailable·prev_available False(로그 첫 프레임과 같은 처리) — 원본 로그 영상으로 대체하지 않는다(궤적이 다르다).
교사 배정(분할 파일 fold_split_navsim.json, navtest 와 같은 로그 단위 규칙): 로그 ∈ val.A → teacher_B, ∈ val.B → teacher_A.
  navhard 76 로그 = 전부 navtest 로그(val.A 40 / val.B 36, train 0) → 두 교사 모두 학습에 쓰지 않은 로그(누설 없음).
출력: <out_dir>/teacher_inputs_navhard.jsonl (token 1 행, part 키 포함), teacher_inputs_navhard_stats.json
실행(env hipad_bw, CPU): python -I tools/navsim/make_navhard_teacher_inputs.py --infos <navhard pkl> --out_dir <dir>
"""
import argparse
import collections
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VT = os.path.join(HERE, "..", "vla_teacher")
sys.path.insert(0, VT)
import common_v2 as C  # noqa: E402
import navsim_sft_common as NS  # noqa: E402
import navsim_teacher_common as N  # noqa: E402

SEQ_GAP_US = 0.75e6
KEEP = ("token", "scene_token", "log_name", "timestamp", "is_scene_frame", "teacher_excluded_log", "cams", "cams_prev",
        "prev_available", "prev_token", "command", "ego_status", "command_drop_ok", "ego_drop_ok", "prompt", "prompt_parts")


def yaw(T):
    T = np.asarray(T, np.float64)
    return float(np.arctan2(T[1, 0], T[0, 0]))       # gen_teacher_gt_navsim._yaw 와 같은 식


def cam_paths(info):
    return [info["cams"][c]["data_path"] for c in N.CAMS]


def record(info, prev_info, prev_pose, gap, exclude):
    """prev_info: 직전 프레임 info(영상용, 없으면 None), prev_pose: (timestamp, ego2global) 또는 None(자차 줄용)."""
    es = np.asarray(info["ego_status"], np.float64)
    ego_vals, src = None, "unavailable"
    if np.isfinite(es).all() and prev_pose is not None:
        dt = (info["timestamp"] - prev_pose[0]) * 1e-6
        if 0 < dt <= SEQ_GAP_US * 1e-6:
            dyaw = (yaw(info["ego2global"]) - yaw(prev_pose[1]) + np.pi) % (2 * np.pi) - np.pi
            ego_vals, src = (float(np.hypot(es[0], es[1])), float(es[2]), float(dyaw / dt)), "ego_dynamic_state+pose_past"
    cmd = int(np.argmax(info["driving_command"]))
    parts = N.build_prompt_parts(cmd, ego_vals, gap)
    prev_ok = prev_info is not None
    return dict(
        token=info["token"], scene_token=info["scene_token"], log_name=info["log_name"], timestamp=int(info["timestamp"]),
        is_scene_frame=bool(info.get("is_eval_frame")), teacher_excluded_log=info["log_name"] in exclude,
        cams=cam_paths(info), cams_prev=cam_paths(prev_info) if prev_ok else cam_paths(info),
        prev_available=prev_ok, prev_token=prev_info["token"] if prev_ok else None,
        command=dict(id=cmd, text=parts["command"], onehot=[int(x) for x in info["driving_command"]]),
        ego_status=dict(src=src, available=ego_vals is not None,
                        speed=None if ego_vals is None else round(ego_vals[0], 4),
                        accel_long=None if ego_vals is None else round(ego_vals[1], 4),
                        yaw_rate=None if ego_vals is None else round(ego_vals[2], 5),
                        raw=[float(x) for x in info["ego_status"]]),
        ego_drop_ok=ego_vals is not None, command_drop_ok=cmd != 3,
        prompt=dict(system=N.SYSTEM_PROMPT, user=C.join_prompt(parts)), prompt_parts=parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--logs", default="/home1/irteam/data-vol1/datasets/navsim/openscene/navsim_logs/test")
    ap.add_argument("--sensor-root", default="/home1/irteam/data-vol1/datasets/navsim/openscene/sensor_blobs/test")
    ap.add_argument("--gt_dir", default=NS.NAVSIM_GT_DIR)
    ap.add_argument("--fold_file", default=NS.NAVSIM_FOLD_FILE)
    a = ap.parse_args()
    out = os.path.join(a.out_dir, "teacher_inputs_navhard.jsonl")
    if os.path.exists(out):
        raise SystemExit(f"[중단] 이미 있음: {out}")
    N.configure(NS.NAVSIM_CODEBOOK, NS.NAVSIM_THRESHOLDS)
    gst = json.load(open(os.path.join(a.gt_dir, "stats_navtest.json")))
    gap = float(gst["gap_m"])
    fold = json.load(open(a.fold_file))
    excl = set(NS.exclude_logs(fold))                       # 교사 제외 로그(NuPlanQA-Eval 6 로그) — 분할 파일 기록
    owner = {}
    for T, f in (("B", "A"), ("A", "B")):                  # teacher_B → val.A, teacher_A → val.B (assignment.generate)
        if f"val.{f}" not in fold["assignment"][f"teacher_{T}"]["generate"]:
            raise SystemExit(f"분할 파일 배정이 가정과 다르다: teacher_{T} generate {fold['assignment'][f'teacher_{T}']['generate']}")
        for log in fold["val"][f]["scenes"]:
            owner[log] = (T, f"val.{f}")
    train_logs = set(fold["train"]["A"]["scenes"]) | set(fold["train"]["B"]["scenes"])

    d = pickle.load(open(a.infos, "rb"))
    infos = d["infos"]
    uniq = collections.OrderedDict()
    for i in infos:
        uniq.setdefault(i["token"], i)
        if i.get("is_eval_frame"):
            uniq[i["token"]] = dict(uniq[i["token"]], is_eval_frame=True)
    logs = sorted({i["log_name"] for i in uniq.values()})
    if set(logs) & train_logs or set(logs) - set(owner):
        raise SystemExit(f"[거부] navhard 로그가 교사 학습 분할에 있거나(겹침 {len(set(logs) & train_logs)}) 배정 밖({len(set(logs) - set(owner))})")

    # 합성 직전 프레임 관계
    pred = {}
    for st, grp in _by_seq(infos).items():
        if not grp[0]["synthetic"]:
            continue
        for k in range(1, len(grp)):
            p = pred.setdefault(grp[k]["token"], grp[k - 1]["token"])
            if p != grp[k - 1]["token"]:
                raise SystemExit(f"합성 직전 프레임 충돌: {grp[k]['token']}: {p} vs {grp[k - 1]['token']}")

    # navtest GT 입력 레코드(1단계 겹침용)
    want = {t for t, i in uniq.items() if not i["synthetic"]}
    nt = {}
    with open(os.path.join(a.gt_dir, "teacher_gt_navtest.jsonl")) as fh:
        for ln in fh:
            r = json.loads(ln)
            if r["token"] in want:
                nt[r["token"]] = {k: r[k] for k in KEEP if k in r}

    from build_navsim_infos import frame_info  # noqa: E402  (tools/navsim)
    S = collections.Counter()
    rows = []
    log_cache = {}
    for tok, i in uniq.items():
        if i["synthetic"]:
            p = pred.get(tok)
            pi = uniq[p] if p is not None else None
            r = record(i, pi, None if pi is None else (pi["timestamp"], pi["ego2global"]), gap, excl)
            S["synthetic"] += 1
        else:
            if i["log_name"] not in log_cache:
                fr = pickle.load(open(os.path.join(a.logs, i["log_name"] + ".pkl"), "rb"))
                log_cache = {i["log_name"]: (fr, {f["token"]: k for k, f in enumerate(fr)})}
            fr, idx = log_cache[i["log_name"]]
            k = idx[tok]
            prev_pose, prev_info = None, None
            if k > 0:
                from build_navsim_infos import ego2global_of
                prev_pose = (int(fr[k - 1]["timestamp"]), ego2global_of(fr[k - 1]))
                if i["timestamp"] - fr[k - 1]["timestamp"] <= SEQ_GAP_US:
                    pinf = frame_info(fr, k - 1, i["log_name"], a.sensor_root, "lidar", {})
                    if all(os.path.exists(x) for x in cam_paths(pinf)):
                        prev_info = pinf
            r = record(i, prev_info, prev_pose, gap, excl)
            if tok in nt:
                o = nt[tok]
                S["orig_in_navtest_gt"] += 1
                S["rule_user_text_equal"] += int(o["prompt"]["user"] == r["prompt"]["user"])
                S["rule_prev_available_equal"] += int(o["prev_available"] == r["prev_available"])
                S["rule_cams_equal_basename"] += int([os.path.basename(x) for x in o["cams"] + o["cams_prev"]] ==
                                                     [os.path.basename(x) for x in r["cams"] + r["cams_prev"]])
                r = dict(o, scene_token=i["scene_token"], is_scene_frame=bool(i.get("is_eval_frame")))   # navtest 입력 그대로
                r["input_source"] = "navtest_gt"
            else:
                S["orig_generated"] += 1
                r["input_source"] = "rule"
        T, part = owner[i["log_name"]]
        r.update(part=part, teacher=T, navhard_stage=int(i["navhard_stage"]), synthetic=bool(i["synthetic"]))
        S[f"teacher_{T}"] += 1
        S[f"ego_{r['ego_status']['src']}"] += 1
        S[f"prev_available_{r['prev_available']}"] += 1
        rows.append(r)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(out + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(out + ".tmp", out)
    st = dict(S, rows=len(rows), logs=len(logs), gap_m=gap, infos=os.path.abspath(a.infos), gt_dir=a.gt_dir, fold_file=a.fold_file,
              assignment="teacher_B ← val.A logs, teacher_A ← val.B logs (navtest 와 같은 규칙)")
    json.dump(st, open(os.path.join(a.out_dir, "teacher_inputs_navhard_stats.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(st, indent=1, ensure_ascii=False))


def _by_seq(infos):
    out = collections.OrderedDict()
    for i in infos:
        out.setdefault(i["scene_token"], []).append(i)
    return out


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
