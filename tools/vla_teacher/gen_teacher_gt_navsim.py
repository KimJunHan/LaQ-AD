# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
"""NAVSIM 교사 GT (v2-sensor 형식, Action 8 토큰 = 4 s → 정답 549 토큰) — env: hipad_bw (transformers 토크나이저).

입력: build_navsim_infos(_hist).py 의 info pkl + navsim_teacher_map.py 의 지도 pkl + OpenScene 로그(직전 프레임 포즈).
출력(--out_dir): teacher_gt_<split>.jsonl (프레임 1 행), stats_<split>.json, instruction.txt, special_tokens.txt.
설계·근거: docs/navsim_teacher_gt_design.md, 공용 규칙: navsim_teacher_common.py.

  # TTC 간격 임계값(navtrain 만, nuScenes 와 같은 규칙: 선택 객체 0–3 s 최소 간격의 5 백분위수, 0.1 m 반올림)
  python -I tools/vla_teacher/gen_teacher_gt_navsim.py --explore_ttc --infos <navtrain pkl> --logs <trainval logs> [--max-logs N]
  # 생성
  python -I tools/vla_teacher/gen_teacher_gt_navsim.py --infos <pkl> --map <map pkl> --logs <logs> --split navtrain \
      --gap <m> --out_dir <dir> [--max-logs N]
"""
import argparse
import collections
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import common as C1  # noqa: E402
import common_v2 as C  # noqa: E402
import navsim_teacher_common as N  # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
TOKENIZER = os.path.join(REPO, "ckpts/qwen2-vl-2b")
CODEBOOK = os.path.join(REPO, "data/qwen3d/nusc_action_codebook_k2048.npz")
EXCLUDE = "/home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/qa/nuplanqa/teacher_exclude_logs_navsim.json"
SEQ_GAP_US = 0.75e6
UNCERTAIN_WEIGHT = 0.3

_G = {}


def _init(args):
    _G.update(args)
    if not _G.get("explore"):
        from transformers import AutoTokenizer
        _G["cb"] = N.configure(_G["codebook"], _G["thresholds"])   # 수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건 (4)
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        n = tok.add_tokens(N.SPECIAL_TOKENS, special_tokens=True)
        assert n == len(N.SPECIAL_TOKENS), (n, len(N.SPECIAL_TOKENS))
        _G["tok"] = tok


def _yaw(T):
    T = np.asarray(T, np.float64)
    return float(np.arctan2(T[1, 0], T[0, 0]))


def log_poses(log):
    fr = pickle.load(open(os.path.join(_G["logs"], log + ".pkl"), "rb"))
    from pyquaternion import Quaternion
    out = []
    for f in fr:
        T = np.eye(4)
        T[:3, :3] = Quaternion(*f["ego2global_rotation"]).rotation_matrix
        T[:3, 3] = f["ego2global_translation"]
        out.append((f["token"], int(f["timestamp"]), T))
    return out


def ego_inputs(info, prev_pose):
    """speed = |(vx,vy)|, accel_long = ax(ego_dynamic_state), yaw rate = (ψ0 − ψ−1)/Δt (직전 로그 프레임, 과거만)."""
    es = np.asarray(info["ego_status"], np.float64)
    if not np.isfinite(es).all() or prev_pose is None:
        return None, "unavailable"
    ts_prev, T_prev = prev_pose
    dt = (info["timestamp"] - ts_prev) * 1e-6
    if not (0 < dt <= SEQ_GAP_US * 1e-6):
        return None, "unavailable"
    dyaw = (_yaw(info["ego2global"]) - _yaw(T_prev) + np.pi) % (2 * np.pi) - np.pi
    return (float(np.hypot(es[0], es[1])), float(es[2]), float(dyaw / dt)), "ego_dynamic_state+pose_past"


def r4(a):
    return [[round(float(v), 4) for v in p] for p in a]


def cam_paths(info):
    return [info["cams"][c]["data_path"] for c in N.CAMS]


def work(job):
    log, infos = job
    poses = log_poses(log)
    idx = {t: k for k, (t, _, _) in enumerate(poses)}
    have = _G["have"]
    rows, S = [], collections.defaultdict(collections.Counter)
    gaps = []
    for info in infos:
        k = idx[info["token"]]
        prev_pose = (poses[k - 1][1], poses[k - 1][2]) if k > 0 else None
        prev_tok = poses[k - 1][0] if k > 0 and (info["timestamp"] - poses[k - 1][1]) <= SEQ_GAP_US else None
        ego_vals, ego_src = ego_inputs(info, prev_pose)
        v0 = ego_vals[0] if ego_vals is not None else float(np.hypot(*np.asarray(info["ego_status"], np.float64)[:2]))
        if _G.get("explore"):
            v = N.teacher_view(info, {})
            gaps += N.gap_series_for_explore(v, v0)
            continue
        m = _G["map"][info["token"]]
        v = N.teacher_view(info, m["map_annos"])
        cmd = int(np.argmax(info["driving_command"]))
        objs, n_in, sel_meta = N.select_objects(v, v0)
        N.enrich_objects(v, objs, _G["gap"])
        mp = N.map_context_sensor(v)
        complete = N.future_complete(v)
        fut = N.fut_cum(v)
        intent = toks = rec = None
        weight = 0.0
        if complete:
            lab = N.intent_labels(fut, v0, v["map_annos"].get(1, []))
            lat, uncertain, src = N.lateral_decision(lab, m["lane_change"])
            weight = UNCERTAIN_WEIGHT if uncertain else 1.0
            intent = dict(lab, lateral=lat, lateral_rule=lab["lateral"], uncertain=uncertain, lateral_src=src,
                          lane_topology=m["lane_change"])
            toks, rec = _G["cb"].tokenize(fut)
            S["lateral"][lat] += 1; S["longitudinal"][lab["longitudinal"]] += 1; S["lateral_src"][src] += 1
            S["action_recon_ade_sum"]["m"] += float(np.linalg.norm(rec - fut, axis=1).mean())
            S["action_recon_ade_sum"]["n"] += 1
        else:
            S["future_incomplete"]["n"] += 1
        intent_ans = None if intent is None else dict(longitudinal=intent["longitudinal"], lateral=intent["lateral"])
        answer = N.build_answer(objs, mp, intent_ans, toks)
        parsed, _ = N.parse_answer(answer)                       # 엄격 파서 + 왕복 일치(실패 시 예외 → 즉시 중단)
        parts = N.build_prompt_parts(cmd, ego_vals, _G["gap"])
        user = C.join_prompt(parts)
        spans = N.answer_spans(answer)
        prev_ok = prev_tok is not None and prev_tok in have
        lm = N.loss_mask_spans(spans, complete, prev_ok)
        enc = _G["tok"](answer, return_offsets_mapping=True, add_special_tokens=False)
        if len(enc["input_ids"]) != N.ANSWER_TOKENS:
            raise ValueError(f"{info['token']}: 정답 토큰 {len(enc['input_ids'])} != {N.ANSWER_TOKENS}\n{answer}")
        offs = enc["offset_mapping"]
        n_user = len(_G["tok"](user, add_special_tokens=False)["input_ids"])   # 캐시 context 길이 고정 전제(명령·자차 줄 고정폭)
        if _G.setdefault("n_user", n_user) != n_user:
            raise ValueError(f"{info['token']}: user 텍스트 토큰 {n_user} != {_G['n_user']} (캐시 길이 고정 위반)\n{user[-300:]}")
        tsp = [[min(q for q, (s0, e0) in enumerate(offs) if e0 > a0), max(q for q, (s0, e0) in enumerate(offs) if s0 < b0) + 1]
               for a0, b0 in lm]
        cb_v, cb_w = (0.0, 0.0) if ego_vals is None else (ego_vals[0], ego_vals[2])
        ctrv = C.ctrv_traj(cb_v, cb_w, T=N.N_FUT)
        rows.append(dict(
            token=info["token"], scene_token=info["scene_token"], log_name=log, timestamp=int(info["timestamp"]),
            is_scene_frame=bool(info.get("is_eval_frame")), teacher_excluded_log=log in _G["exclude"],
            cams=cam_paths(info), cams_prev=cam_paths(have[prev_tok]) if prev_ok else cam_paths(info),
            prev_available=prev_ok, prev_token=prev_tok if prev_ok else None,
            command=dict(id=cmd, text=parts["command"], onehot=[int(x) for x in info["driving_command"]]),
            ego_status=dict(src=ego_src, available=ego_vals is not None,
                            speed=None if ego_vals is None else round(ego_vals[0], 4),
                            accel_long=None if ego_vals is None else round(ego_vals[1], 4),
                            yaw_rate=None if ego_vals is None else round(ego_vals[2], 5),
                            raw=[float(x) for x in info["ego_status"]]),
            ego_drop_ok=ego_vals is not None, command_drop_ok=cmd != 3,
            prompt=dict(system=N.SYSTEM_PROMPT, user=user), prompt_parts=parts,
            answer=answer, answer_spans=spans, loss_mask_spans=lm, loss_mask_token_spans=tsp,
            modes=dict(ego_in_prompt="current", object_speed="numeric", answer_format="v2-sensor-navsim",
                       action_tokens=N.ACTION_N, horizon_s=4.0),
            targets=dict(objects=objs, n_objects_in_range=n_in, object_selection=sel_meta, map=mp,
                         future_complete=complete, fut_mask=[int(x) for x in v["gt_ego_fut_masks"]], fut_traj=r4(fut),
                         label_v0=round(v0, 4), intent=intent, intent_weight=weight, action_tokens=toks,
                         action_recon_traj=None if rec is None else r4(rec)),
            copy_baseline=dict(traj=r4(ctrv), ade_vs_gt=None if not complete else
                               round(float(np.linalg.norm(ctrv - fut, axis=1).mean()), 4)),
        ))
        S["objects"][str(len(objs))] += 1
        for r in objs:
            S["obj_class"][r["name"]] += 1; S["obj_ttc"][str(r["ttc"])] += 1; S["obj_reason"][r["reason"]] += 1
        S["map"]["na" if mp is None else "ok"] += 1
        if mp is not None:
            S["map_ped"]["none" if mp["ped_crossing"] is None else mp["ped_crossing"]["sector"] == "here" and "on" or "near"] += 1
            S["map_free"]["capped" if mp["free_dist"] >= C.FREE_CAP else "blocked"] += 1
        S["ego_src"][ego_src] += 1; S["command"][parts["command"]] += 1; S["prev_available"][str(prev_ok)] += 1
    return log, rows, {k: dict(v) for k, v in S.items()}, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--map")
    ap.add_argument("--logs", required=True)
    ap.add_argument("--split", default="navtrain")
    ap.add_argument("--gap", type=float)
    ap.add_argument("--out_dir")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-logs", type=int, default=0)
    ap.add_argument("--explore_ttc", action="store_true")
    ap.add_argument("--codebook", help="NAVSIM 행동 코드북 npz (build_navsim_codebook_intent.py) — 수정(2026-10-19): 권장안 (4)")
    ap.add_argument("--thresholds", help="NAVSIM 의도 임계값 json — 수정(2026-10-19): 권장안 (4)")
    a = ap.parse_args()
    t0 = time.time()
    d = pickle.load(open(a.infos, "rb"))
    have, by = {}, collections.OrderedDict()
    for i in d["infos"]:
        if i["token"] in have:
            continue
        have[i["token"]] = i
        by.setdefault(i["log_name"], []).append(i)
    jobs = list(by.items())
    if a.max_logs:
        jobs = jobs[:a.max_logs]
    excl = set(json.load(open(EXCLUDE))["logs"])
    g = dict(logs=a.logs, have=have, exclude=excl, explore=a.explore_ttc, gap=a.gap, codebook=a.codebook, thresholds=a.thresholds)
    if a.explore_ttc:
        if d["metadata"]["kind"] not in ("train", "train_hist"):
            raise SystemExit("TTC 간격은 navtrain(학습) 통계로만 정한다")
        G = []
        with Pool(a.workers, initializer=_init, initargs=(g,)) as pool:
            for _, _, _, gaps in pool.imap_unordered(work, jobs, chunksize=1):
                G += gaps
        G = np.asarray(G, np.float64)
        m3 = G[:, :7].min(1)
        out = dict(n_objects=len(G), logs=len(jobs), min_gap_0_3s_percentiles={p: round(float(np.percentile(m3, p)), 4)
                                                                                for p in (1, 5, 10, 25, 50)},
                   p5_rounded=round(float(np.percentile(m3, 5)), 1), buckets={})
        for th in (0.0, 1.0, 1.4, out["p5_rounded"], 2.0):
            c = collections.Counter(C.ttc_bucket(x, th)[0] for x in G)
            out["buckets"][str(th)] = {k: round(c[k] / len(G), 4) for k in ("lt2", "2_4", "4_6", "safe")}
        out["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(out, indent=1))
        return
    if a.gap is None or a.map is None or a.out_dir is None or a.codebook is None or a.thresholds is None:
        raise SystemExit("--gap, --map, --out_dir, --codebook, --thresholds 필요")
    N.configure(a.codebook, a.thresholds)   # 주 프로세스(지시문·특수토큰 파일 기록용)
    g["map"] = pickle.load(open(a.map, "rb"))["frames"]
    missing = [t for _, lst in jobs for t in (i["token"] for i in lst) if t not in g["map"]]
    if missing:
        raise SystemExit(f"지도 pkl 에 없는 프레임 {len(missing)} (예: {missing[:3]}) — navsim_teacher_map.py 먼저")
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"teacher_gt_{a.split}.jsonl")
    if os.path.exists(path):
        raise SystemExit(f"[중단] 이미 있음: {path}")
    S = collections.defaultdict(collections.Counter)
    n = 0
    with open(path + ".tmp", "w") as fo, Pool(a.workers, initializer=_init, initargs=(g,)) as pool:
        res = {}
        for k, (log, rows, st, _) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            res[log] = rows
            for kk, vv in st.items():
                S[kk].update(vv)
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(jobs)} logs {time.time() - t0:.0f}s", flush=True)
        for log, _ in jobs:                                  # 결정적 순서
            for r in res[log]:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    os.replace(path + ".tmp", path)
    st = {k: dict(v) for k, v in S.items()}
    ade = st.pop("action_recon_ade_sum", {"m": 0, "n": 0})
    st["action_recon_ade_m"] = round(ade["m"] / max(ade["n"], 1), 4)
    st.update(navsim_config=dict(N.CONFIG), samples=n, logs=len(jobs), partial=bool(a.max_logs), gap_m=a.gap, answer_tokens=N.ANSWER_TOKENS,
              infos=a.infos, map=a.map, seconds=round(time.time() - t0, 1))
    json.dump(st, open(os.path.join(a.out_dir, f"stats_{a.split}.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(a.out_dir, "instruction.txt"), "w") as f:
        f.write("[SYSTEM]\n" + N.SYSTEM_PROMPT + "\n\n[USER]\n" + N.build_instruction(a.gap) + "\n<command line>\n<ego state line>\n")
    with open(os.path.join(a.out_dir, "special_tokens.txt"), "w") as f:
        f.write("\n".join(N.SPECIAL_TOKENS) + "\n")
    print(json.dumps(st, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
