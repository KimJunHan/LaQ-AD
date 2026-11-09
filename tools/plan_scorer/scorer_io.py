# 수정(2026-10-19): 사용자 결정 — GRPO 대신 Hydra-MDP 방식 계획 점수 학습(LaQ v2 2단계 뒤 적용). 새 파일(기존 파일 수정 없음).
"""계획 점수기(plan scorer) 덤프 형식 읽기/쓰기 공용 도우미.

덤프 디렉터리 구성(dump_plan_feats.py 가 만든다):
  manifest.json            — config·체크포인트·분할·샤드(장면 범위) 정의. 재개 시 같은지 검사.
  shard_XXXX.npz           — 샤드 하나(연속한 장면 묶음)의 표본별 배열. 원자적 저장(tmp → rename).
  shard_XXXX.state.pt      — 그 샤드를 끝낸 직후의 시간 인스턴스 뱅크 상태(정확한 재개용).

샤드 배열(표본 N 개, 모드 M=16, 시점 T=6):
  idx (N,) int64            데이터셋 색인(= 평가 results.pkl 색인)
  token, scene_token (N,)   문자열
  timestamp (N,) float64
  cmd (N,) int64            운전 명령(0 우, 1 좌, 2 직진; info gt_ego_fut_cmd argmax — 모델 decoder.select 와 같은 출처)
  modes (N,M,T,2) float32   명령 묶음의 16 모드 누적 궤적(lidar, +x 오른쪽, +y 전방, 0.5 s 간격) = results plan_modes
  probs (N,M) float32       = results plan_mode_probs (rescore 반영 softmax)
  raw_logits (N,M) float32  마지막 refine 층 plan_cls_branch 출력(rescore 전)
  sel (N,) int64            원래 선택 모드 = argmax probs (= results plan_temp_2hz)
  feats (N,M,C) float32     마지막 refine 층 plan_cls_branch 입력(= 계획 쿼리 특징 + 앵커 임베딩), C=256
  gt_traj (N,T,2) float32   GT 자차 미래 누적 궤적(info gt_ego_fut_trajs cumsum)
  gt_mask (N,T) float32     info gt_ego_fut_masks
  fut_box_counts (N,T) int64  시점별 미래 GT 박스 수(평가기 _get_fut_boxes 결과). 그 시점 프레임이 없으면 -1
  fut_boxes (K,7) float32   위 박스들을 이어 붙인 것(x,y,z,w,l,h,yaw — 평가기가 쓰는 7 열)
  lidar2global (N,4,4) float64, map_location (N,) 문자열
"""
import glob
import json
import os

import numpy as np

SHARD_KEYS = ("idx", "token", "scene_token", "timestamp", "cmd", "modes", "probs", "raw_logits", "sel", "feats",
              "gt_traj", "gt_mask", "fut_box_counts", "fut_boxes", "lidar2global", "map_location")
PER_SAMPLE_KEYS = tuple(k for k in SHARD_KEYS if k != "fut_boxes")


def shard_path(out_dir, k):
    return os.path.join(out_dir, f"shard_{k:04d}.npz")


def state_path(out_dir, k):
    return os.path.join(out_dir, f"shard_{k:04d}.state.pt")


def save_shard(out_dir, k, rec):
    """rec: SHARD_KEYS 를 모두 가진 dict. 원자적 저장."""
    missing = [key for key in SHARD_KEYS if key not in rec]
    if missing:
        raise KeyError(f"샤드 {k} 에 없는 키: {missing}")
    n = len(rec["idx"])
    for key in PER_SAMPLE_KEYS:
        if len(rec[key]) != n:
            raise ValueError(f"샤드 {k}: {key} 길이 {len(rec[key])} != {n}")
    if int(np.clip(rec["fut_box_counts"], 0, None).sum()) != len(rec["fut_boxes"]):
        raise ValueError(f"샤드 {k}: fut_box_counts 합과 fut_boxes 행 수가 다르다")
    final = shard_path(out_dir, k)
    tmp = final[:-4] + ".tmp.npz"
    np.savez(tmp, **{key: rec[key] for key in SHARD_KEYS})
    os.replace(tmp, final)


def load_dump(dump_dir, require_complete=True):
    """덤프 전체를 표본 순서대로 이어 붙여 반환. fut_boxes 는 표본별 목록(list of list of (n,7))으로 푼다."""
    man_p = os.path.join(dump_dir, "manifest.json")
    if not os.path.exists(man_p):
        raise FileNotFoundError(f"{man_p} 없음 — dump_plan_feats.py 로 먼저 만든다")
    man = json.load(open(man_p))
    n_shards = len(man["shards"])
    paths = [shard_path(dump_dir, k) for k in range(n_shards)]
    have = [p for p in paths if os.path.exists(p)]
    if require_complete and len(have) != n_shards:
        raise RuntimeError(f"{dump_dir}: 샤드 {len(have)}/{n_shards} 만 있다(미완료 덤프). 덤프를 끝까지 재개한 뒤 쓴다.")
    parts = {k: [] for k in PER_SAMPLE_KEYS}
    fut = []
    for p in have:
        z = np.load(p, allow_pickle=False)
        for key in PER_SAMPLE_KEYS:
            parts[key].append(z[key])
        fut.extend(unpack_fut_boxes(z["fut_box_counts"], z["fut_boxes"]))
    out = {k: np.concatenate(v, 0) for k, v in parts.items()}
    out["fut_boxes"] = fut
    out["manifest"] = man
    if require_complete:
        exp = man["num_samples"]
        if len(out["idx"]) != exp:
            raise RuntimeError(f"{dump_dir}: 표본 {len(out['idx'])} != manifest {exp}")
        if not np.array_equal(out["idx"], np.arange(exp)):
            raise RuntimeError(f"{dump_dir}: 표본 색인이 0..N-1 연속이 아니다")
    return out


def pack_fut_boxes(fut_list, T=6):
    """fut_list: 표본 하나의 미래 박스 목록(길이 ≤ T, 각 (n, ≥7)). → counts (T,), boxes (K,7)."""
    counts = np.full(T, -1, np.int64)
    rows = []
    for t, b in enumerate(fut_list[:T]):
        b = np.asarray(b, np.float32)
        b = np.zeros((0, 7), np.float32) if b.size == 0 else b.reshape(len(b), -1)[:, :7]
        counts[t] = len(b)
        rows.append(b)
    boxes = np.concatenate(rows, 0) if rows else np.zeros((0, 7), np.float32)
    return counts, boxes.astype(np.float32)


def unpack_fut_boxes(counts, boxes):
    counts = np.asarray(counts)
    out, o = [], 0
    for row in counts:
        lst = []
        for c in row:
            if c < 0:
                break
            lst.append(boxes[o:o + c]); o += c
        out.append(lst)
    if o != len(boxes):
        raise ValueError("fut_boxes 풀기 실패: 행 수 불일치")
    return out


def file_sig(path):
    """체크포인트·pkl 식별용(크기·수정 시각). 실제 경로(링크 해석) 포함."""
    rp = os.path.realpath(path)
    st = os.stat(rp)
    return {"path": path, "realpath": rp, "size": st.st_size, "mtime": int(st.st_mtime)}


def list_shards(dump_dir):
    return sorted(glob.glob(os.path.join(dump_dir, "shard_*.npz")))
