# 수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건
"""NAVSIM 교사 행동 코드북 + 의도 임계값 — 데이터셋별 규칙(nuScenes·B2D 와 같은 절차)을 navtrain 으로 다시 정한다. env: hipad_bw.

표본: navtrain 장면 프레임(103,288 중 미래 8 포즈 완전, 교사 제외 6 로그 제외) — 교사 SFT 가 토큰화하는 바로 그 표본(권장안 6).
      navtest 장면 프레임은 재구성 오차 보고에만 쓴다(선택에 미사용).
코드북(nuScenes tools/qwen3d_nusc/nusc_action_codebook.py 알고리즘 그대로 import — 조각 분해·자차 박스 모서리 거리·K-disk 이분탐색):
  0.5 s 조각 (Δx, Δy, Δθ), 목표 K = 2048(±1%), 자차 크기 = Pacifica 5.176 × 2.297(nuScenes 4.084 × 1.85 대신). 지평 8 조각(4 s).
의도 임계값(common.py 주석의 nuScenes 규칙을 같은 방식으로):
  STOP_V = 0.5 m/s, MIN_PATH = 2 m: 정의값(객체 moving 기준·정지 판정) — 데이터와 무관하게 같다.
  DV_TH  = |현재 속도(ego_dynamic_state) − 첫 0.5 s 포즈 차분 속도|(측정 잡음) 의 p95 를 넘는 가장 작은 0.5 단위 값.
  TURN_TH = {5,10,12.5,15,17.5,20,25,30}° 중 '표준 명령'(미래 끝 횡변위 ±2 m 규칙 — nuScenes 컨버터 명령과 같은 정의, 지평 4 s)과의
            일치율 최대. 참고로 NAVSIM driving_command(경로 기반: 좌/우에 차로변경·급커브 포함)와의 일치율도 기록.
출력: --out_dir/navsim_action_codebook.npz, navsim_intent_thresholds.json
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qwen3d_nusc"))
sys.path.insert(0, HERE)
import nusc_action_codebook as NAC  # noqa: E402
import common as C1  # noqa: E402
import navsim_teacher_common as N  # noqa: E402

EXCLUDE = "/home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/qa/nuplanqa/teacher_exclude_logs_navsim.json"


def scene_futs(pkl, excl):
    d = pickle.load(open(pkl, "rb"))
    out = []
    seen = set()
    for i in d["infos"]:
        if not i.get("is_eval_frame") or i["token"] in seen or i["log_name"] in excl:
            continue
        seen.add(i["token"])
        v = N.teacher_view(i, {})
        if not N.future_complete(v):
            continue
        es = np.asarray(i["ego_status"], np.float64)
        out.append(dict(fut=N.fut_cum(v), v0=float(np.hypot(es[0], es[1])), cmd=int(np.argmax(i["driving_command"]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/infos/navsim_20261019/navsim_navtrain.pkl")
    ap.add_argument("--test", default="/home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/infos/navsim_20261019/navsim_navtest.pkl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=2048)
    a = ap.parse_args()
    t0 = time.time()
    excl = set(json.load(open(EXCLUDE))["logs"])
    tr = scene_futs(a.train, excl)
    te = scene_futs(a.test, set())
    print(f"train 장면(미래 완전·제외 로그 뺌) {len(tr)}, test {len(te)}  {time.time() - t0:.0f}s", flush=True)

    # ---------------- 의도 임계값
    F = np.stack([r["fut"] for r in tr]); V0 = np.array([r["v0"] for r in tr]); CMD = np.array([r["cmd"] for r in tr])
    V1 = np.linalg.norm(F[:, 0], axis=1) / C1.DT
    noise = np.abs(V1 - V0)
    p95 = float(np.percentile(noise, 95))
    dv_th = float(np.ceil((p95 + 1e-9) / 0.5) * 0.5)
    TH, S = [], []
    for f in F:
        xy = np.concatenate([np.zeros((1, 2)), f], 0)
        TH.append(float(np.degrees(C1.heading_series(xy)[-1]))); S.append(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()))
    TH, S = np.array(TH), np.array(S)
    std = np.where(F[:, -1, 0] >= 2.0, "R", np.where(F[:, -1, 0] <= -2.0, "L", "S"))     # 표준 명령(+x = 오른쪽)
    nav = np.where(CMD == 0, "L", np.where(CMD == 2, "R", np.where(CMD == 1, "S", "U")))
    agree, agree_nav = {}, {}
    for t in [5, 10, 12.5, 15, 17.5, 20, 25, 30]:
        lab = np.where((TH >= t) & (S >= C1.MIN_PATH), "L", np.where((TH <= -t) & (S >= C1.MIN_PATH), "R", "S"))
        agree[str(t)] = round(float(np.mean(lab == std)), 4)
        k = nav != "U"
        agree_nav[str(t)] = round(float(np.mean(lab[k] == nav[k])), 4)
    turn_th = float(max(agree, key=lambda k: (agree[k], -float(k))))
    VE = np.linalg.norm(F[:, -1] - F[:, -3], axis=1) / (2 * C1.DT)
    thr = dict(STOP_V=C1.STOP_V, MIN_PATH=C1.MIN_PATH, DV_TH=dv_th, TURN_TH_DEG=turn_th,
               evidence=dict(n_train=len(tr), noise_p50_80_90_95=[round(float(x), 3) for x in np.percentile(noise, [50, 80, 90, 95])],
                             dv_p5_20_50_80_95=[round(float(x), 3) for x in np.percentile(VE - V0, [5, 20, 50, 80, 95])],
                             turn_agree_standard_cmd=agree, turn_agree_navsim_cmd_ref=agree_nav,
                             nuscenes_values=dict(DV_TH=C1.DV_TH, TURN_TH_DEG=C1.TURN_TH_DEG)))
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(thr, open(os.path.join(a.out_dir, "navsim_intent_thresholds.json"), "w"), indent=1)
    print(json.dumps(thr, indent=1), flush=True)

    # ---------------- 코드북
    NAC.EGO_L, NAC.EGO_W = N.EGO_L, N.EGO_W
    seg = np.concatenate([NAC.segments_from_traj(f) for f in F])
    feat = NAC.corners(seg[:, 0], seg[:, 1], seg[:, 2]).reshape(len(seg), -1)
    delta, Cc = NAC.kdisk(feat, a.k)
    idx = [int(np.argmin(np.linalg.norm(feat - c, axis=1))) for c in Cc]
    book = seg[idx]

    class CB(C1.ActionCodebook):
        def __init__(self, b):
            self.book = np.asarray(b, np.float64); self.K = len(self.book)
    cb = CB(book)
    Ft = np.stack([r["fut"] for r in te])
    errs = np.array([np.linalg.norm(cb.tokenize(f)[1] - f, axis=1) for f in Ft])
    old = C1.ActionCodebook(os.path.join(HERE, "..", "..", "data/qwen3d/nusc_action_codebook_k2048.npz"))
    errs_old = np.array([np.linalg.norm(old.tokenize(f)[1] - f, axis=1) for f in Ft])
    res = dict(K=len(book), delta=float(delta), n_train_segments=len(seg), n_train_traj=len(F), n_test_traj=len(Ft),
               test_ade=float(errs.mean()), test_fde=float(errs[:, -1].mean()),
               nusc_codebook_test_ade=float(errs_old.mean()), nusc_codebook_test_fde=float(errs_old[:, -1].mean()),
               seconds=round(time.time() - t0))
    np.savez(os.path.join(a.out_dir, "navsim_action_codebook.npz"), codebook=book, ego_size=np.array([N.EGO_L, N.EGO_W]),
             **{k: v for k, v in res.items()})
    json.dump(res, open(os.path.join(a.out_dir, "navsim_action_codebook.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
