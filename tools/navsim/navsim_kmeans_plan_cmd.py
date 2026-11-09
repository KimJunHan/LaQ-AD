# 수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건
"""NAVSIM 명령별 계획 앵커 (3, 16, 8, 2) — nuScenes v2(tools/kmeans/kmeans_nusc_plan_cmd.py, 3×16)와 같은 방식(권장안 3).

명령 대응(navsim_ref docs/agents.md:79, 실측 분포로 확인): NAVSIM driving_command one-hot [0 left, 1 straight, 2 right, 3 unknown].
  앵커 묶음 순서 = [left, straight, right] (NAVSIM 0·1·2 그대로). unknown(navtrain 363 프레임, navtest 56 프레임, 0.24%)은
  straight(1) 로 보낸다 — 경로 방향 정보가 없을 때 가장 흔한 명령. 학습·추론에 같은 규칙(NavsimTeacherDataset.cmd3).
  (nuScenes 순서는 [right, left, straight] 이지만 앵커 파일·데이터셋이 같은 순서를 쓰면 되므로 NAVSIM 순서를 유지한다.)
표본: navtrain(현재+과거 info) 모든 프레임 중 미래 8 포즈 완전(앵커 통계 추정 조건 — 학습 표본을 버리는 것이 아님), 라이다=ego 누적 궤적.
KMeans(n_init=10, random_state=0) — kmeans_nusc_plan_cmd.py 와 같다.
"""
import argparse
import json
import os
import pickle

import numpy as np
from sklearn.cluster import KMeans

K_PER_CMD, N_CMD, FUT_TS = 16, 3, 8
NAV2CMD3 = {0: 0, 1: 1, 2: 2, 3: 1}   # unknown → straight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", default="/home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/infos/navsim_20261019/navsim_navtrain.pkl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pickle.load(open(a.infos, "rb"))
    if d["metadata"]["kind"] not in ("train", "train_hist"):
        raise ValueError("navtrain 만")
    by = [[] for _ in range(N_CMD)]
    raw = np.zeros(4, int)
    for i in d["infos"]:
        m = np.asarray(i["gt_ego_fut_masks"])
        if m.min() < 1:
            continue
        c = int(np.argmax(i["driving_command"]))
        raw[c] += 1
        by[NAV2CMD3[c]].append(np.cumsum(np.asarray(i["gt_ego_fut_trajs"], np.float64), 0))
    anchor = np.zeros((N_CMD, K_PER_CMD, FUT_TS, 2), np.float32)
    rep = dict(raw_command_counts=dict(zip(["left", "straight", "right", "unknown"], raw.tolist())), groups={})
    for c, name in enumerate(["left", "straight", "right"]):
        X = np.stack(by[c]).reshape(len(by[c]), -1)
        anchor[c] = KMeans(n_clusters=K_PER_CMD, n_init=10, random_state=0).fit(X).cluster_centers_.reshape(K_PER_CMD, FUT_TS, 2)
        end = anchor[c, :, -1]
        rep["groups"][name] = dict(n=len(X), end_y_left_mean=round(float(end[:, 1].mean()), 2),
                                   end_x_fwd_mean=round(float(end[:, 0].mean()), 2),
                                   gt_end_y_mean=round(float(X.reshape(-1, FUT_TS, 2)[:, -1, 1].mean()), 2))
    np.save(a.out, anchor)
    json.dump(rep, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(json.dumps(rep, indent=1), "->", a.out, anchor.shape)


if __name__ == "__main__":
    main()
