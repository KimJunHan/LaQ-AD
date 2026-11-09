# 수정(2026-10-16): B2D 교사 QA 준비 — Bench2Drive(B2D) 공통 상수·영상 목록·공식 val 분할(make_qa_data_b2d.py / make_fold_split_b2d.py /
#   check_qa_split_b2d.py / smoke_qa_b2d_cpu.py 가 공유). nuScenes 교사 QA(qa_common.py / make_qa_data.py)와 같은 형식을 쓰되
#   장면 단위 = B2D clip(=route: Base-1000 의 Route 번호는 clip 마다 하나, 겹치는 route 없음 — 확인함).
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

B2D_ROOT = "/home1/irteam/data-vol1/src/HiP-AD/data/bench2drive/v1"     # NAS(느린 NFS) — 전체 재귀 탐색 금지, clip 별 카메라 폴더 listdir 만
OFFICIAL_SPLIT = "/home1/irteam/data-vol1/src/HiP-AD/data/splits/bench2drive_base_train_val_split.json"   # Bench2DriveZoo 공식(val 50 clip)
QA_ROOT = "/home1/irteam/data-vol1/src/HiP-AD/data/nuscenes/qa"          # 내려받은 B2D QA(신뢰하지 않는 데이터 — python -I 로만 읽음)
OUT_DIR = "data/vla_teacher/b2d"                            # 저장소 기준 산출 폴더
QA_DIR_B2D = "data/vla_teacher/b2d/qa"                      # train_qa_full.py --qa_dir 로 넘길 폴더
FOLD_FILE_B2D = "data/vla_teacher/b2d/fold_split_b2d.json"  # train_qa_full.py --fold_file 로 넘길 파일
LISTING_CACHE = "data/vla_teacher/b2d/image_listing_b2d.json"
# 교사 SFT·nuScenes QA 와 같은 카메라 순서(qa_common.CAM_ORDER) ↔ B2D 폴더 이름
CAM_DIRS = ["rgb_front", "rgb_front_left", "rgb_front_right", "rgb_back_left", "rgb_back", "rgb_back_right"]
CAM_NAMES = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
DATASETS_B2D = ("chat_b2d_plus", "b2d_vl_drivecommenter")


def official_val_clips(path=OFFICIAL_SPLIT):
    d = json.load(open(path))
    val = [v.split("/")[-1] for v in d["val"]]
    if len(val) != 50 or len(set(val)) != 50:
        raise RuntimeError(f"공식 val clip 수 이상: {len(val)} (고유 {len(set(val))})")
    return set(val)


def all_clips(root=B2D_ROOT):
    cl = sorted(x for x in os.listdir(root) if "_Town" in x and os.path.isdir(os.path.join(root, x)))
    if len(cl) != 1000:
        raise RuntimeError(f"B2D Base clip 수 {len(cl)} != 1000")
    return cl


def cam_path(clip, cam_dir, frame, root=B2D_ROOT):
    return f"{root}/{clip}/camera/{cam_dir}/{frame}.jpg"


def _list_clip(clip, root):
    out = {}
    for c in CAM_DIRS:
        d = os.path.join(root, clip, "camera", c)
        out[c] = sorted(os.listdir(d)) if os.path.isdir(d) else None
    return clip, out


def image_listing(repo, root=B2D_ROOT, threads=8, refresh=False):
    """clip 별 6 카메라 폴더의 파일 이름 목록(listdir 1 회씩 = 6,000 회). NFS 가 느려 결과를 캐시한다(저장소 data/ 아래, 영상 아님).
    반환 {clip: {cam_dir: [파일 이름] | None(폴더 없음)}}."""
    p = Path(repo) / LISTING_CACHE
    if p.exists() and not refresh:
        d = json.load(open(p))
        return d["listing"]
    clips = all_clips(root)
    t = time.time()
    res = {}
    with ThreadPoolExecutor(threads) as ex:          # I/O 대기용 스레드(프로세스 1 개)
        for i, (clip, lst) in enumerate(ex.map(lambda c: _list_clip(c, root), clips)):
            res[clip] = lst
            if (i + 1) % 100 == 0:
                print(f"[listing] {i + 1}/{len(clips)} clip ({time.time() - t:.0f}s)", flush=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    json.dump(dict(created="2026-10-16", root=root, cam_dirs=CAM_DIRS, n_clips=len(res), listing=res), open(tmp, "w"))
    os.replace(tmp, p)
    return res


def frames_per_clip(listing):
    """clip → rgb_front 의 jpg 프레임 수(fold 분할의 표본 수 단위)."""
    return {c: sum(1 for f in v["rgb_front"] if f.endswith(".jpg")) for c, v in listing.items() if v["rgb_front"] is not None}
