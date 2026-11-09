# 수정(2026-11-06): 사용자 결정 "NAVSIM MCMC SFT" — 새 파일(기존 파일 무수정). 설계 docs/mcmc_sft_navsim_plan.md §2-1.
"""NAVSIM 2단계 모델의 navtrain 전 프레임(152,495 = 평가 프레임 103,288 + 이력 전용 49,207) 계획 모드 덤프(GPU, 추론만).

왜 tools/test.py --out 이나 tools/plan_scorer/dump_plan_feats.py 를 그대로 쓰지 않는가:
  - test.py --out: single_gpu_test 가 15 만 프레임의 전체 결과(검출·지도·모션 포함)를 메모리에 모은 뒤 한 번에 저장한다
    (navtest 200 프레임 results.pkl 66 MB → navtrain 152,495 프레임 ≈ 50 GB). 또 결과의 plan_mode_probs 는 rescore(충돌 모드 −999)
    뒤 값이라 밀도 π 에 필요한 rescore 전 logit 이 없다. 중간 재개도 안 된다.
  - dump_plan_feats.py: GPU1 고정·분할 이름 train/val(ann_file 접미사 검사)·nuScenes 미래 박스(_get_fut_boxes)·특징 저장 — NAVSIM 에 맞지 않는다.
평가와 같게 만드는 것(tools/test.py 비분산 경로와 같은 순서):
  config custom_imports·plugin 적재, model.pretrained=None, train_cfg=None, data.test(test 파이프라인, test_mode=True, 배치 1),
  set_random_seed(0), fp16 이면 wrap_fp16_model, load_checkpoint(cpu), MMDataParallel(device_ids=[0]), shuffle=False.
  navtrain 은 data.test 설정에 ann_file·teacher_map_files·ego_input_file 만 data.train 의 것으로 바꿔 쓴다(test 파이프라인 = 증강 없음).
  장면 순서 연속 추론(시간 인스턴스 뱅크가 평가와 같게), 이력 전용 프레임도 추론·저장한다(정답 생성은 평가 프레임만 쓰지만 덤프는 전부 —
  건너뛰지 않음, 생성기가 상태별로 셈).
저장(프레임마다): idx, token, scene_token, timestamp, is_eval, navsim_token(이력 프레임 ""), cmd(3 묶음, 데이터셋 get_ann_info 와 같은 값),
  modes (16,8,2) 누적(라이다 = ego, +x 전방, +y 왼쪽) = results plan_modes, probs (16) = plan_mode_probs(rescore 뒤 softmax),
  raw_logits (16) = rescore 전 명령 묶음 logit(plan_decoder.rescore 입력), post_logits (16) = rescore 뒤 logit, rescored (bool),
  sel = argmax probs, plan (8,2) = plan_temp_2hz, gt_traj (8,2) = cumsum(info gt_ego_fut_trajs), gt_disp (8,2) = info 증분 그대로,
  gt_mask (8), ego_status (4) = info ego_status(비유한값 → 0, 데이터셋과 같은 규칙).
검사(매 프레임, 다르면 중단): plan == modes[argmax probs], softmax(post_logits) == probs(1e-5), post − raw ∈ {0, −999}(rescore 규칙).
  # 수정(2026-11-06): 검토 반영 — 샤드마다·끝에서 rescore 호출 수 == 프레임 수 단언(가로채기 미연결이면 raw 에 −999 가 섞여 π 오염). #이전 "rescore 를 안 부른 프레임은 raw == post" 허용
재개: 샤드 = 연속한 장면 묶음. 샤드마다 시간 뱅크 상태(stream_state)와 run_step 을 저장 → 다음 샤드 직전 상태로 정확히 재개.
  # 수정(2026-11-06): 검토 반영 — 상태 파일을 샤드 npz 보다 먼저 쓴다(npz 존재 ⇒ 그 상태 존재).
GPU: CUDA_VISIBLE_DEVICES=3(NAVSIM 전용) 만. 스모크(--smoke_samples > 0)에 한해 0(공용) 허용 — 이때 /tmp/claude-500/gpu0_timing.lock 을
  스스로 잡는다(못 잡으면 중단). 스모크 덤프는 manifest smoke=True 라 생성기가 학습용 정답을 만들지 않는다.
사용: CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. OPENBLAS_CORETYPE=SkylakeX python tools/mcmc_sft/dump_plan_modes_navsim.py \
        configs/navsim/v2_teacher/stage2/test_e3_nuim.py results/hipad_navsim_laq_v2_teacher_stage2_e3_nuim_gtfilt/iter_129060_ema.pth \
        --out results/mcmc_sft_navsim/dump_navtrain_s2e3nuim_gtfilt_ema  # 수정(2026-11-09): GT 필터 2단계 이름(사용 예만) [--workers 8 --scenes_per_shard 500]
"""
import argparse
import copy
import fcntl
import functools
import importlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "tools", "plan_scorer"))
import scorer_io  # noqa: E402  (file_sig 만 쓴다)

GPU0_LOCK = "/tmp/claude-500/gpu0_timing.lock"
KEYS = ("idx", "token", "scene_token", "timestamp", "is_eval", "navsim_token", "cmd", "modes", "probs", "raw_logits",
        "post_logits", "rescored", "sel", "plan", "gt_traj", "gt_disp", "gt_mask", "ego_status")
RESCORE_PENALTY = -999.0  # models/head/motion_planning/plan_decoder.py rescore: score_offset = col * -999


def shard_path(out, k):
    return os.path.join(out, f"shard_{k:04d}.npz")


def state_path(out, k):
    return os.path.join(out, f"shard_{k:04d}.state.pt")


def scene_shards(infos, spp):
    """장면(scene_token) 경계에 맞춘 샤드. 장면이 연속·시간 증가인지 단언(평가 순서 재현 조건)."""
    starts, seen = [0], {infos[0]["scene_token"]}
    for i in range(1, len(infos)):
        if infos[i]["scene_token"] != infos[i - 1]["scene_token"]:
            if infos[i]["scene_token"] in seen:
                raise RuntimeError(f"장면 {infos[i]['scene_token']} 이 연속하지 않다(색인 {i})")
            seen.add(infos[i]["scene_token"]); starts.append(i)
        elif infos[i]["timestamp"] <= infos[i - 1]["timestamp"]:
            raise RuntimeError(f"색인 {i}: 장면 안 시간이 증가하지 않는다")
    starts.append(len(infos))
    return [(starts[s], starts[min(s + spp, len(starts) - 1)]) for s in range(0, len(starts) - 1, spp)]


def load_dump(out, require_complete=True):
    """생성기용: 샤드를 이어 붙인 dict + manifest."""
    man = json.load(open(os.path.join(out, "manifest.json")))
    n = len(man["shards"])
    have = [k for k in range(n) if os.path.exists(shard_path(out, k))]
    if require_complete and len(have) != n:
        raise RuntimeError(f"{out}: 샤드 {len(have)}/{n} — 미완료 덤프")
    parts = {k: [] for k in KEYS}
    for k in have:
        z = np.load(shard_path(out, k), allow_pickle=False)
        for key in KEYS:
            parts[key].append(z[key])
    d = {k: np.concatenate(v, 0) for k, v in parts.items()}
    if require_complete and not np.array_equal(d["idx"], np.arange(man["num_samples"])):
        raise RuntimeError(f"{out}: 색인이 0..N-1 연속이 아니다")
    d["manifest"] = man
    return d


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenes_per_shard", type=int, default=500)
    ap.add_argument("--smoke_samples", type=int, default=0, help=">0 이면 처음 N 프레임만(검증 전용, manifest smoke)")
    ap.add_argument("--max_gpu_mib", type=int, default=0)
    return ap.parse_args()


def main():
    a = parse()
    dev = os.environ.get("CUDA_VISIBLE_DEVICES")
    lock_fd = None
    if dev == "0":
        if a.smoke_samples <= 0:
            raise SystemExit("GPU0(공용)은 스모크(--smoke_samples > 0) 전용 — 본 덤프는 CUDA_VISIBLE_DEVICES=3")
        lock_fd = open(GPU0_LOCK, "a")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"{GPU0_LOCK} 이 잡혀 있다 — GPU0 스모크 보류")
    elif dev != "3":
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={dev} — NAVSIM 덤프는 GPU3 전용(스모크만 GPU0)")
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet.datasets import build_dataloader, build_dataset
    from torch.utils.data import Subset

    if a.max_gpu_mib > 0:
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gpu_mib * 2 ** 20 / tot), 0)
    cfg = Config.fromfile(a.config)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    importlib.import_module(os.path.dirname(cfg.plugin_dir).replace("/", "."))
    from projects.mmdet3d_plugin.models.decoder.temporal_bank.stream_state import (
        bank_state_keys, load_bank_state, save_bank_state)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    ds_cfg = copy.deepcopy(cfg.data.test)
    ds_cfg.test_mode = True
    if ds_cfg.pop("samples_per_gpu", 1) != 1:
        raise SystemExit("data.test.samples_per_gpu != 1 — 평가(배치 1, 순차)와 다르다")
    for k in ("ann_file", "teacher_map_files", "ego_input_file"):
        if k not in cfg.data.train:
            raise SystemExit(f"data.train 에 {k} 가 없다")
        ds_cfg[k] = cfg.data.train[k]
    if os.path.basename(ds_cfg.ann_file) != "navsim_navtrain.pkl":
        raise SystemExit(f"ann_file {ds_cfg.ann_file} — navtrain 전용(navtest·navhard 금지)")
    ds = build_dataset(ds_cfg)
    if ds.metadata.get("kind") != "train_hist":
        raise SystemExit(f"info kind {ds.metadata.get('kind')} != train_hist")
    infos = ds.data_infos
    N_all = len(infos)
    shards = scene_shards(infos, a.scenes_per_shard)
    if a.smoke_samples > 0:
        n_sm = min(a.smoke_samples, N_all)
        shards = [(s0, min(s1, n_sm)) for s0, s1 in shards if s0 < n_sm]

    os.makedirs(a.out, exist_ok=True)
    man = {"config": a.config, "checkpoint": scorer_io.file_sig(a.checkpoint), "split": "navtrain",
           "ann_file": ds_cfg.ann_file, "num_samples": shards[-1][1] if a.smoke_samples else N_all,
           "shards": [list(s) for s in shards], "smoke": bool(a.smoke_samples), "gpu": dev,
           "frame": "lidar = ego (lidar2ego identity, dataset-checked), +x forward, +y left, rear axle origin",
           "tool": "tools/mcmc_sft/dump_plan_modes_navsim.py"}
    man_p = os.path.join(a.out, "manifest.json")
    if os.path.exists(man_p):
        old = json.load(open(man_p))
        diff = [k for k in ("config", "checkpoint", "ann_file", "num_samples", "shards", "smoke") if old.get(k) != man.get(k)]
        if diff:
            raise SystemExit(f"{a.out} 의 기존 manifest 와 다르다({diff}) — 다른 폴더를 쓸 것")
    else:
        json.dump(man, open(man_p, "w"), indent=1, ensure_ascii=False)
    done = [os.path.exists(shard_path(a.out, k)) for k in range(len(shards))]
    if all(done):
        print(f"[완료] {a.out}: 샤드 {len(shards)} 개 모두 있다"); return
    first = done.index(False)
    if any(done[first:]):
        raise SystemExit(f"샤드 {first} 뒤에 이미 있는 샤드 — 순차 덤프 가정 위반")

    set_random_seed(0, deterministic=False)
    from mmdet.models import build_detector
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
    _tl = torch.load
    torch.load = functools.partial(_tl, weights_only=False)
    try:
        load_checkpoint(model, a.checkpoint, map_location="cpu")
    finally:
        torch.load = _tl
    model = MMDataParallel(model, device_ids=[0]).eval()
    det = model.module
    dec = det.head.onedecoder_head
    if not dec.is_init_bank_list:
        dec.init_instance_bank_list()
    pdec = dec.plan_decoder
    if list(pdec.anchor_types) != [("temp", "2hz")]:
        raise RuntimeError(f"앵커 묶음 {pdec.anchor_types} — ('temp','2hz') 하나 가정")
    M, T = dec.ego_fut_mode, dec.ego_fut_ts
    if T != 8:
        raise RuntimeError(f"ego_fut_ts {T} != 8 (NAVSIM 4 s @ 2 Hz)")

    cap = []
    _orig_rescore = pdec.rescore

    def rescore_spy(plan_cls, *args, **kw):  # rescore 전 logit 기록(값은 바꾸지 않는다)
        cap.append(plan_cls.detach().float().clone())
        return _orig_rescore(plan_cls, *args, **kw)
    pdec.rescore = rescore_spy

    def bank_objs():
        return {n: getattr(dec, f"{n}_instance_bank_list")[0] for n in ("ego", "det", "map", "plan") if n in dec.query_select}

    def save_state(k):
        st = {"banks": {n: save_bank_state(b, bank_state_keys(b)) for n, b in bank_objs().items()},
              "run_step": dec.run_step, "last_scene_token": getattr(det, "_last_scene_token", None)}
        p = state_path(a.out, k)
        torch.save(st, p + ".tmp"); os.replace(p + ".tmp", p)

    def load_state(k):
        st = torch.load(state_path(a.out, k), map_location="cuda:0", weights_only=False)
        for n, b in bank_objs().items():
            load_bank_state(b, st["banks"][n])
        dec.run_step = st["run_step"]
        if hasattr(det, "_last_scene_token"):
            det._last_scene_token = st["last_scene_token"]

    if first > 0:
        if not os.path.exists(state_path(a.out, first - 1)):
            raise SystemExit(f"샤드 {first - 1} 뱅크 상태 파일 없음 — 정확한 재개 불가")
        load_state(first - 1)
        print(f"[재개] 샤드 {first} 부터")

    t0 = time.time(); n_done = 0; n_resc = 0
    for k in range(first, len(shards)):
        s0, s1 = shards[k]
        loader = build_dataloader(Subset(ds, list(range(s0, s1))), samples_per_gpu=1, workers_per_gpu=a.workers,
                                  dist=False, shuffle=False)
        rec = {key: [] for key in KEYS}
        for j, data in enumerate(loader):
            i = s0 + j
            info = infos[i]
            cap.clear()
            with torch.no_grad():
                r = model(return_loss=False, rescale=True, **data)[0]["img_bbox"]
            mreg, mcls = pdec._vis_modes
            post = mcls[0].detach().float().cpu().numpy()
            modes = np.asarray(r["plan_modes"], np.float32)
            probs = np.asarray(r["plan_mode_probs"], np.float32)
            plan = np.asarray(r["plan_temp_2hz"], np.float32)
            if modes.shape != (M, T, 2) or probs.shape != (M,) or post.shape != (M,):
                raise RuntimeError(f"색인 {i}: 모양 {modes.shape} {probs.shape} {post.shape}")
            if len(cap) > 1:
                raise RuntimeError(f"색인 {i}: rescore 호출 {len(cap)} 회(1 이하여야 함)")
            raw = cap[0][0].cpu().numpy() if cap else post.copy()
            d = post - raw
            if not np.all((np.abs(d) < 1e-4) | (np.abs(d - RESCORE_PENALTY) < 1e-2)):
                raise RuntimeError(f"색인 {i}: rescore 전후 logit 차가 {{0, −999}} 가 아니다: {d}")
            sm = torch.softmax(torch.as_tensor(post), 0).numpy()
            sel = int(np.argmax(probs))
            if np.abs(sm - probs).max() > 1e-5 or np.abs(plan - modes[sel]).max() > 1e-6:
                raise RuntimeError(f"색인 {i}: post_logits/plan 이 결과와 맞지 않다")
            ann = ds.get_ann_info(i)
            es = np.asarray(info["ego_status"], np.float32)
            disp = np.asarray(info["gt_ego_fut_trajs"], np.float32)
            rec["idx"].append(i); rec["token"].append(info["token"]); rec["scene_token"].append(info["scene_token"])
            rec["timestamp"].append(float(info["timestamp"]) / 1e6); rec["is_eval"].append(bool(info.get("is_eval_frame")))
            rec["navsim_token"].append(info.get("navsim_token") or "")
            rec["cmd"].append(int(np.argmax(ann["gt_ego_fut_cmd"])))
            rec["modes"].append(modes); rec["probs"].append(probs); rec["raw_logits"].append(raw.astype(np.float32))
            rec["post_logits"].append(post.astype(np.float32)); rec["rescored"].append(bool(cap)); rec["sel"].append(sel)
            rec["plan"].append(plan); rec["gt_disp"].append(disp); rec["gt_traj"].append(disp.cumsum(0))
            rec["gt_mask"].append(np.asarray(info["gt_ego_fut_masks"], np.float32))
            rec["ego_status"].append(np.where(np.isfinite(es), es, 0.0).astype(np.float32))
            n_done += 1; n_resc += bool(cap)
        if len(rec["idx"]) != s1 - s0:
            raise RuntimeError(f"샤드 {k}: {len(rec['idx'])} != {s1 - s0}")
        if int(np.sum(rec["rescored"])) != s1 - s0:  # 수정(2026-11-06): 검토 반영 — rescore 가로채기가 모든 프레임에서 불렸는지(안 불리면 raw = rescore 뒤 logit 이라 −999 가 π 에 섞임)
            raise RuntimeError(f"샤드 {k}: rescore 호출 {int(np.sum(rec['rescored']))} != 프레임 {s1 - s0} — 가로채기 미연결 또는 rescore 생략 프레임(밀도 π 오염 방지 위해 중단)")  # 수정(2026-11-06): 검토 반영
        out = {key: np.asarray(v) for key, v in rec.items()}
        for key in ("token", "scene_token", "navsim_token"):
            out[key] = out[key].astype("U64")
        save_state(k)  # 수정(2026-11-06): 검토 반영 — 상태를 샤드 npz 보다 먼저 저장(사이에서 죽으면 npz 없음 → 다음 실행이 샤드 k 를 k−1 상태에서 다시 함). #이전 npz os.replace 뒤에 저장(사이에서 죽으면 상태 k 없이 npz k 만 남아 재개 불능)
        tmp = shard_path(a.out, k)[:-4] + ".tmp.npz"
        np.savez(tmp, **out); os.replace(tmp, shard_path(a.out, k))
        el = time.time() - t0
        print(f"[샤드 {k + 1}/{len(shards)}] {s0}..{s1 - 1} | 누적 {n_done} 프레임 {el:.0f}s ({el / max(n_done, 1):.3f}s/프레임), "
              f"rescore 호출 {n_resc}", flush=True)
    pdec.rescore = _orig_rescore
    if n_resc != n_done:  # 수정(2026-11-06): 검토 반영 — 끝에서 한 번 더: rescore 호출 수 == 처리 프레임 수
        raise RuntimeError(f"rescore 호출 {n_resc} != 처리 프레임 {n_done}")
    print("[완료]", a.out)
    if lock_fd is not None:
        lock_fd.close()


if __name__ == "__main__":
    main()
