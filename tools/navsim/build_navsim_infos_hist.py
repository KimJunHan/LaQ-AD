# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
"""navtrain info pkl — "현재 + 과거 프레임만" (미래 프레임 영상은 공식 배포에 없으므로 참조하지 않는다).

사용자 결정(2026-10-19 지시): NAVSIM 학습 시퀀스는 현재 + 과거 프레임만 쓴다.
기존 build_navsim_infos.py --kind train 은 장면 창(이력 4 + 미래 10) 합집합 프레임을 info 로 만들어, 공식 배포에 없는
미래 프레임 영상을 참조했다. 이 스크립트는 그 파일을 고치지 않고(공유 파일 수정 금지) 같은 frame_info() 를 import 해서
**장면 창의 이력 4 프레임(현재 포함)의 합집합**만 info 로 만든다.

  - 프레임 dict 형식 = navtest pkl(data/infos/navsim_20261016/navsim_navtest.pkl) 과 같다(같은 frame_info 함수).
  - 정답(GT)은 로그에서 계산하므로 미래 위치(자차 8포즈·에이전트 12스텝)는 그대로 들어간다 — 미래 "영상" 만 안 쓴다.
  - 시퀀스(scene_token): 같은 로그 안에서 프레임 번호가 연속이고 시간 간격 ≤ 0.75 s 인 구간 = 하나의 시퀀스
    "<log>#<seg>". (navtrain 은 frame_interval=1 이라 연속 장면들의 이력 창이 이어져 긴 시퀀스가 된다.)
  - navtrain 장면 token 인 프레임(장면의 현재 프레임)은 is_eval_frame=True, navsim_token=token 으로 표시한다
    (교사 GT·캐시·navtrain 채점에서 장면 프레임을 찾는 데 쓴다. 학습 손실은 모든 프레임에 건다).
  - metadata.kind = "train_hist" (기존 "train" = 창 합집합과 구별), metadata.eval_tokens = navtrain token 목록.

실행(hipad_bw, CPU):
  CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX /home1/irteam/data-vol1/anaconda3/envs/hipad_bw/bin/python -I \
    tools/navsim/build_navsim_infos_hist.py \
    --logs /home1/irteam/data-vol1/nas_mirror/navsim/navsim_data/trainval_navsim_logs/trainval \
    --sensor-root /home1/irteam/data-vol1/nas_mirror/navsim/navsim_data/trainval_sensor_blobs/trainval \
    --scene-filter /home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/splits/navtrain.yaml \
    --anno-frame lidar --workers 8 \
    --out /home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/infos/navsim_20261019/navsim_navtrain.pkl
"""
import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_navsim_infos as B  # noqa: E402  (수정 없이 재사용)


def build_log(args):
    log, log_dir, sensor_root, anno_frame, sf = args
    frames = pickle.load(open(os.path.join(log_dir, log + ".pkl"), "rb"))
    ts = [f["timestamp"] for f in frames]
    if any(b <= a for a, b in zip(ts, ts[1:])):
        raise RuntimeError(f"{log}: 로그 프레임이 시간순이 아니다")
    scenes = B.scenes_of_log(frames, sf)
    if not scenes:
        return log, [], [], 0
    h = sf["num_history"]
    used = sorted({start + j for start in scenes.values() for j in range(h)})   # 이력 4(현재 포함)만
    scene_tok = set(scenes.keys())
    track_ids, infos, seg, prev = {}, [], 0, None
    for i in used:
        if prev is not None and (i != prev + 1 or
                                 frames[i]["timestamp"] - frames[prev]["timestamp"] > B.SEQ_GAP_S * 1e6):
            seg += 1
        inf = B.frame_info(frames, i, log, sensor_root, anno_frame, track_ids)
        inf["scene_token"] = f"{log}#{seg}"
        if inf["token"] in scene_tok:
            inf["is_eval_frame"], inf["navsim_token"] = True, inf["token"]
        inf["frame_index_in_log"] = int(i)
        infos.append(inf)
        prev = i
    return log, infos, list(scenes.keys()), len(scenes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--sensor-root", required=True)
    ap.add_argument("--scene-filter", required=True)
    ap.add_argument("--anno-frame", choices=["lidar", "ego"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-logs", type=int, default=0, help=">0 이면 앞 N 로그만(시간 측정용; 결과는 부분 산출물로 표시)")
    a = ap.parse_args()
    t0 = time.time()
    sf = B.load_scene_filter(a.scene_filter)
    if sf["max_scenes"] is not None:
        raise NotImplementedError("max_scenes 미지원(devkit 재현 규칙 확인 전)")
    names = sorted(set(sf["log_names"]))
    if a.max_logs > 0:
        names = names[:a.max_logs]
    jobs = [(n, a.logs, a.sensor_root, a.anno_frame, sf) for n in names]
    out = {}
    with Pool(a.workers) as pool:
        for k, (log, infos, toks, ns) in enumerate(pool.imap_unordered(build_log, jobs, chunksize=1)):
            out[log] = (infos, toks, ns)
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(jobs)} logs  {time.time() - t0:.0f}s", flush=True)
    infos, eval_tokens, n_scenes = [], [], 0
    for log in names:                      # 로그 이름 순으로 고정(결정적 출력)
        i, t, ns = out[log]
        infos += i
        eval_tokens += t
        n_scenes += ns
    if sf["tokens"] is not None and a.max_logs == 0:
        missing = set(sf["tokens"]) - set(eval_tokens)
        extra = set(eval_tokens) - set(sf["tokens"])
        if missing or extra:
            raise RuntimeError(f"navtrain token 불일치: 누락 {len(missing)}, 초과 {len(extra)}")
    meta = dict(kind="train_hist", anno_frame=a.anno_frame, scene_filter={k: v for k, v in sf.items()
                                                                           if k not in ("log_names", "tokens")},
                cams=B.NAVSIM_CAMS, ego_fut_ts=B.EGO_FUT_TS, agent_fut_ts=B.AGENT_FUT_TS, num_scenes=n_scenes,
                eval_tokens=eval_tokens, frames_used="history+current only (num_history frames per scene window)",
                partial=a.max_logs > 0, builder="tools/navsim/build_navsim_infos_hist.py (2026-10-19)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(dict(infos=infos, metadata=meta), f, protocol=4)
    os.replace(tmp, a.out)
    print(f"train_hist: logs {len(names)}, scenes {n_scenes}, frames {len(infos)}, "
          f"sequences {len({i['scene_token'] for i in infos})} -> {a.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
