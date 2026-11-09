# 수정(2026-10-16): 교사 QA — NAVSIM 트랙. make_fold_split_navsim.py / make_qa_data_navsim.py / check_qa_split_navsim.py 가 공유.
#   NAVSIM(OpenScene) 로그 pkl → 프레임 색인(프레임 token → 로그·시각·카메라 파일 이름). 내려받은 데이터는 신뢰하지 않으므로
#   이 모듈을 쓰는 스크립트는 python -I 로 실행한다. 로그 pkl 은 numpy 배열을 담은 pickle 이라 표준 pickle 로 읽되,
#   numpy/builtins/collections 밖의 클래스는 거부한다(SafeUnpickler).
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import yaml

NAVSIM_DATA = "/home1/irteam/data-vol1/nas_mirror/navsim/navsim_data"
TRAIN_LOGS = f"{NAVSIM_DATA}/trainval_navsim_logs/trainval"
TRAIN_BLOBS = f"{NAVSIM_DATA}/trainval_sensor_blobs/trainval"
TEST_LOGS = f"{NAVSIM_DATA}/test_navsim_logs/test"
TEST_BLOBS = f"{NAVSIM_DATA}/test_sensor_blobs/test"
NUPLAN_RAW_BLOBS = "/home1/irteam/data-vol1/nas_mirror/Public_datasets/Nuplan/dataset/nuplan-v1.1/sensor_blobs"   # nuPlan 원본(평가 QA 영상 찾기용)
REF = "/tmp/claude-0/-workspace-src/868c57e8-6d51-4ca5-b2e3-25a0b7e11891/scratchpad/ref"
NAVTRAIN_YAML = f"{REF}/navtrain.yaml"
NAVTEST_YAML = f"{REF}/navtest.yaml"
QA_DIR_NAVSIM = "data/vla_teacher/navsim"
FOLD_FILE_NAVSIM = "data/vla_teacher/navsim/fold_split_navsim.json"
EXCLUDE_LOGS = "/home1/irteam/data-vol1/src/HiP-AD/data/nuscenes/qa/nuplanqa/teacher_exclude_logs_navsim.json"
ALL_CAMS = ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0")
# 교사 입력 카메라·해상도 = LaQ NAVSIM Qwen 캐시와 같게(tools/qwen_cache_navsim.py: 전방 3캠 l0·f0·r0, 128*28*28)
QA_CAMS = ("CAM_L0", "CAM_F0", "CAM_R0")
NAVSIM_PIXELS = 128 * 28 * 28
NAVSIM_IMAGE_HW = (1080, 1920)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("builtins", "collections", "numpy", "copyreg", "_codecs"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"허용하지 않는 클래스 {module}.{name}")


def safe_load(path):
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()


def load_filter(path):
    cfg = yaml.safe_load(open(path))
    return dict(num_history=int(cfg["num_history_frames"]), num_future=int(cfg["num_future_frames"]),
                frame_interval=int(cfg["frame_interval"]), has_route=bool(cfg["has_route"]),
                log_names=list(cfg["log_names"]), tokens=list(cfg["tokens"]))


def _index_one(args):
    split, path = args
    frames = safe_load(path)
    log = os.path.basename(path)[:-4]
    out = []
    for i, fr in enumerate(frames):
        if fr["log_name"] != log:
            raise RuntimeError(f"{path}#{i}: log_name {fr['log_name']} != 파일 이름")
        cams = {}
        for c in ALL_CAMS:
            dp = fr["cams"][c]["data_path"]
            parts = dp.split("/")
            if len(parts) != 3 or parts[0] != log or parts[1] != c or not parts[2].endswith(".jpg"):
                raise RuntimeError(f"{path}#{i}: 예상 밖 data_path {dp}")
            cams[c] = parts[2][:-4]
        out.append((fr["token"], int(fr["timestamp"]), i, len(fr["roadblock_ids"]), cams))
    return split, log, out


def build_frame_index(cache=None, workers=6):
    """반환 dict(logs={(split, log): [(token, ts, idx, n_roadblock, {cam: stem})]}). cache 가 있으면 읽고, 없으면 만들어 저장."""
    if cache and os.path.exists(cache):
        return safe_load(cache)
    jobs = [("trainval", str(p)) for p in sorted(Path(TRAIN_LOGS).iterdir()) if p.suffix == ".pkl"]
    jobs += [("test", str(p)) for p in sorted(Path(TEST_LOGS).iterdir()) if p.suffix == ".pkl"]
    t0 = time.time()
    logs = {}
    with Pool(workers) as pool:
        for k, (split, log, fr) in enumerate(pool.imap_unordered(_index_one, jobs, chunksize=2)):
            logs[(split, log)] = fr
            if (k + 1) % 200 == 0:
                print(f"[index] {k + 1}/{len(jobs)} 로그 ({time.time() - t0:.0f}s)", flush=True)
    idx = dict(logs=logs, created=time.strftime("%Y-%m-%d %H:%M:%S"), train_logs_dir=TRAIN_LOGS, test_logs_dir=TEST_LOGS)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(idx, f, protocol=4)
        os.replace(tmp, cache)
    print(f"[index] 로그 {len(logs)} 개, 프레임 {sum(len(v) for v in logs.values())} 개 ({time.time() - t0:.0f}s)", flush=True)
    return idx


def token_maps(idx):
    """반환 (frame token → (split, log, idx, {cam: stem}), 영상 stem → (split, log, frame token, cam))."""
    tok, stem = {}, {}
    for (split, log), frs in idx["logs"].items():
        for t, ts, i, nrb, cams in frs:
            if t in tok:
                raise RuntimeError(f"프레임 token 중복 {t}: {tok[t][:2]} / {(split, log)}")
            tok[t] = (split, log, i, cams)
            for c, s in cams.items():
                if s in stem:
                    raise RuntimeError(f"영상 stem 중복 {s}: {stem[s]} / {(split, log, t, c)}")
                stem[s] = (split, log, t, c)
    return tok, stem


def scene_tokens_by_log(idx, filt, split):
    """devkit filter_scenes 규칙으로 장면 token 을 다시 만들어 로그별로 센다(필터 token 과 정확히 같아야 함)."""
    nh, n = filt["num_history"], filt["num_history"] + filt["num_future"]
    want = set(filt["tokens"])
    out = {}
    for log in filt["log_names"]:
        frs = idx["logs"].get((split, log))
        if frs is None:
            raise RuntimeError(f"{split} 로그 pkl 없음: {log}")
        sc = {}
        for i in range(0, len(frs), filt["frame_interval"]):
            fl = frs[i:i + n]
            if len(fl) < n:
                continue
            if filt["has_route"] and fl[nh - 1][3] == 0:
                continue
            if fl[nh - 1][0] in want:
                sc[fl[nh - 1][0]] = i
        out[log] = sorted(sc)
    got = {t for v in out.values() for t in v}
    if got != want:
        raise RuntimeError(f"장면 token 재구성 불일치: 필터 {len(want)}, 재구성 {len(got)}, 누락 {len(want - got)}, 초과 {len(got - want)}")
    return out
