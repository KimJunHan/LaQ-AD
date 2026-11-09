# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
"""build_navsim_infos_hist.py 산출 navtrain info(현재+과거 프레임만) 점검.

검사(전부 통과해야 PASS):
  1. 장면 프레임(is_eval_frame) token 집합 == navtrain 필터 token 103,288 (개수·집합 모두).
  2. 미래 프레임 영상 미참조: 모든 info 프레임의 (로그, 로그 내 번호)가 로그에서 독립적으로 다시 계산한
     "장면 창의 이력 4 프레임(현재 포함) 합집합" 안에 있다. 하나라도 밖이면 실패.
  3. 형식: 프레임 dict 키 집합이 navtest pkl 프레임 키와 같다(+frame_index_in_log).
  4. 영상·라이다 실재: 로그별 os.listdir 로 카메라 8대 + MergedPointCloud 파일 존재(전 프레임).
  5. 유한값, 시퀀스 안 시간 증가, 주행 명령 one-hot.
실행(hipad_bw, CPU):
  CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX /home1/irteam/data-vol1/anaconda3/envs/hipad_bw/bin/python -I \
    tools/navsim/check_navsim_infos_hist.py --infos <navsim_navtrain.pkl> --navtest-infos <navsim_navtest.pkl> --out <json>
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_navsim_infos as B  # noqa: E402

D = "/home1/irteam/data-vol1/nas_mirror/navsim/navsim_data"
LOGS = f"{D}/trainval_navsim_logs/trainval"
BLOBS = f"{D}/trainval_sensor_blobs/trainval"
FILTER = "/home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/splits/navtrain.yaml"
NUM_KEYS = ["lidar2ego", "ego2global", "ego_status", "driving_command", "gt_boxes", "gt_velocity",
            "gt_agent_fut_trajs", "gt_agent_fut_masks", "gt_ego_fut_trajs", "gt_ego_fut_masks"]


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("builtins", "collections", "numpy", "copyreg", "_codecs"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"허용하지 않는 클래스 {module}.{name}")


def allowed_and_files(args):
    """로그에서 독립적으로: 허용 프레임 번호(이력 창 합집합), 미래 전용 프레임 번호, 디렉터리 파일 목록."""
    log, sf = args
    frames = pickle.load(open(os.path.join(LOGS, log + ".pkl"), "rb"))
    sc = B.scenes_of_log(frames, sf)
    h, n = sf["num_history"], sf["num_history"] + sf["num_future"]
    allowed = {s + j for s in sc.values() for j in range(h)}
    window = {s + j for s in sc.values() for j in range(n)}
    tok_of = {i: frames[i]["token"] for i in window}
    files = {}
    for c in B.NAVSIM_CAMS + ("MergedPointCloud",):
        try:
            files[c] = set(os.listdir(os.path.join(BLOBS, log, c)))
        except FileNotFoundError:
            files[c] = set()
    # 참고: 미래 전용 프레임 영상이 디스크에 있는가(공식 배포에 없으므로 대부분 없어야 정상)
    fut_only = sorted(window - allowed)
    fut_img = sum(os.path.basename(frames[i]["cams"]["CAM_F0"]["data_path"]) in files["CAM_F0"] for i in fut_only)
    return log, allowed, tok_of, files, len(fut_only), int(fut_img)


def q(x):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return None
    return dict(min=float(x.min()), p1=float(np.percentile(x, 1)), p50=float(np.percentile(x, 50)),
                p99=float(np.percentile(x, 99)), max=float(x.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--navtest-infos", required=True, help="형식 비교용(navtest pkl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    d = SafeUnpickler(open(a.infos, "rb")).load()
    infos, meta = d["infos"], d["metadata"]
    sf = B.load_scene_filter(FILTER)
    want = set(sf["tokens"])
    R = dict(infos=a.infos, n_frames=len(infos), metadata={k: v for k, v in meta.items() if k != "eval_tokens"})
    fails = []

    # 1. 장면 token
    ev = [i["token"] for i in infos if i["is_eval_frame"]]
    R["scene_frames"] = len(ev)
    R["scene_frames_unique"] = len(set(ev))
    R["scene_tokens_eq_filter"] = set(ev) == want and len(ev) == len(want)
    R["meta_eval_tokens_eq_filter"] = set(meta["eval_tokens"]) == want
    R["navsim_token_eq_token"] = all(i["navsim_token"] == i["token"] for i in infos if i["is_eval_frame"])
    if not (R["scene_tokens_eq_filter"] and R["meta_eval_tokens_eq_filter"] and R["navsim_token_eq_token"]):
        fails.append("scene_tokens")
    R["unique_frame_tokens"] = len({i["token"] for i in infos})
    if R["unique_frame_tokens"] != len(infos):
        fails.append("duplicate_frames")

    # 3. 형식
    dt = SafeUnpickler(open(a.navtest_infos, "rb")).load()
    test_keys = set(dt["infos"][0].keys())
    del dt
    key_sets = collections.Counter(frozenset(i.keys()) for i in infos)
    R["frame_key_sets"] = len(key_sets)
    R["keys_minus_navtest"] = sorted(set().union(*key_sets) - test_keys)
    R["navtest_minus_keys"] = sorted(test_keys - set().union(*key_sets))
    if len(key_sets) != 1 or R["navtest_minus_keys"] or R["keys_minus_navtest"] != ["frame_index_in_log"]:
        fails.append("format")

    # 2·4. 로그에서 독립 재계산
    by_log = collections.defaultdict(list)
    for i in infos:
        by_log[i["log_name"]].append(i)
    R["logs"] = len(by_log)
    if set(by_log) != set(sf["log_names"]):
        R["logs_missing"] = sorted(set(sf["log_names"]) - set(by_log))[:5]
        fails.append("logs")
    outside, tok_mismatch, missing_allowed = 0, 0, 0
    img_ok = collections.Counter(); img_tot = 0; lid_ok = 0; all8 = 0; front3 = 0
    fut_only_frames = fut_only_imgs = 0
    with Pool(a.workers) as pool:
        for k, (log, allowed, tok_of, files, nfo, nfi) in enumerate(
                pool.imap_unordered(allowed_and_files, [(lg, sf) for lg in sorted(by_log)], chunksize=2)):
            fut_only_frames += nfo; fut_only_imgs += nfi
            idxs = set()
            for i in by_log[log]:
                fi = i["frame_index_in_log"]
                idxs.add(fi)
                if fi not in allowed:
                    outside += 1
                if tok_of.get(fi) != i["token"]:
                    tok_mismatch += 1
                img_tot += 1
                good = 0
                for c in B.NAVSIM_CAMS:
                    e = os.path.basename(i["cams"][c]["data_path"]) in files[c]
                    img_ok[c] += e; good += e
                all8 += good == 8
                front3 += all(os.path.basename(i["cams"][c]["data_path"]) in files[c] for c in ("CAM_L0", "CAM_F0", "CAM_R0"))
                lid_ok += os.path.basename(i["lidar_path"]) in files["MergedPointCloud"]
            missing_allowed += len(allowed - idxs)
            if (k + 1) % 200 == 0:
                print(f"  logs {k + 1}/{len(by_log)}", flush=True)
    R["frames_outside_history_windows"] = outside
    R["frames_token_mismatch_vs_log"] = tok_mismatch
    R["history_frames_not_in_infos"] = missing_allowed
    R["future_only_frames_in_logs(ref)"] = fut_only_frames
    R["future_only_frames_with_CAM_F0_on_disk(ref)"] = fut_only_imgs
    R["images_frames_all8_present"] = all8
    R["images_frames_front3_present"] = front3
    R["lidar_pcd_present"] = lid_ok
    R["images_per_cam_present_frac"] = {c: round(img_ok[c] / max(img_tot, 1), 6) for c in B.NAVSIM_CAMS}
    if outside or tok_mismatch or missing_allowed:
        fails.append("history_only")
    if all8 != len(infos) or lid_ok != len(infos):
        fails.append("files_missing")

    # 5. 값
    nonfinite = collections.Counter()
    for i in infos:
        for kk in NUM_KEYS:
            n = int((~np.isfinite(np.asarray(i[kk], np.float64))).sum())
            if n:
                nonfinite[kk] += n
    R["nonfinite_values"] = dict(nonfinite)
    if nonfinite:
        fails.append("nonfinite")
    bad_order, prev = 0, {}
    for i in infos:
        s = i["scene_token"]
        if s in prev and i["timestamp"] <= prev[s]:
            bad_order += 1
        prev[s] = i["timestamp"]
    R["sequence_time_not_increasing"] = bad_order
    if bad_order:
        fails.append("order")
    seqs = collections.Counter(i["scene_token"] for i in infos)
    R["sequences"] = len(seqs)
    R["sequence_len"] = q(list(seqs.values()))
    dc = np.stack([i["driving_command"] for i in infos])
    R["driving_command_onehot"] = bool(np.all(dc.sum(1) == 1) and np.all((dc == 0) | (dc == 1)))
    if not R["driving_command_onehot"]:
        fails.append("command")
    R["driving_command_hist"] = {str(k): int(v) for k, v in zip(*np.unique(dc.argmax(1), return_counts=True))}
    em = np.stack([i["gt_ego_fut_masks"] for i in infos if i["is_eval_frame"]])
    R["scene_frame_ego_fut_full_frac"] = float((em.sum(1) == em.shape[1]).mean())
    nb = np.array([len(i["gt_boxes"]) for i in infos])
    R["boxes_per_frame"] = q(nb)
    names = np.concatenate([i["gt_names"] for i in infos if len(i["gt_names"])])
    R["box_class_hist"] = {str(k): int(v) for k, v in zip(*np.unique(names, return_counts=True))}
    R["has_map_annos"] = sum("map_annos" in i for i in infos)
    R["lidar_path_is_bin"] = sum(i["lidar_path"].endswith(".bin") for i in infos)
    R["FAILS"] = fails
    R["RESULT"] = "PASS" if not fails else "FAIL"
    R["seconds"] = round(time.time() - t0, 1)
    with open(a.out, "w") as fp:
        json.dump(R, fp, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in R.items() if k != "metadata"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
