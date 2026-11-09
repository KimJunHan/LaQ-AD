# 수정(2026-10-19): 사용자 결정 — GRPO 대신 Hydra-MDP 방식 계획 점수 학습(LaQ v2 2단계 뒤 적용). 새 파일(기존 파일 수정 없음).
"""고정된 2단계 모델로 한 분할(train/val) 전체를 평가와 같은 순서·설정으로 추론하며 계획 점수기 입력을 덤프한다.

평가와 같게 만드는 것(tools/test.py 비분산 경로와 한 줄씩 대응):
  - config 의 plugin 적재(projects.mmdet3d_plugin), model.pretrained=None, train_cfg=None, data.test.test_mode=True
  - set_random_seed(0)(test.py --seed 기본값), fp16 설정이 있으면 wrap_fp16_model, load_checkpoint(map_location=cpu),
    MMDataParallel(device_ids=[0]), 배치 1, shuffle=False(데이터셋 색인 순서 = 장면·시간 순서) → 시간 인스턴스 뱅크 상태가 평가와 같다.
  - train 분할도 data.test 설정(테스트 파이프라인·GT 유효 마스크 규칙)에 ann_file 만 data.train 의 것으로 바꿔 쓴다.
  - 확산 잡음은 PlanTruncatedDiffusion 의 고정 시드 생성기(infer_seed)라 프레임마다 같다(평가와 동일).

계획 쿼리 특징을 얻는 위치(훅):
  models/decoder/unified_decoder.py:753  plan_reg, plan_cls = self.plan_refine[refine_i](plan_instance_feature, plan_anchor, plan_anchor_embed, ...)
  → models/decoder/split_refine/plan_refine.py:104  cls_output = self.plan_cls_branch(align_query)  (align_query: :70, :87)
     align_query = (plan_instance_feature + plan_anchor_embed)  (앵커 묶음 1 개: ("temp","2hz"))  shape (B, 3×16, 256)
  마지막 디코더 층의 plan_refine[-1].plan_cls_branch 에 forward_pre_hook(입력 = 특징)·forward_hook(출력 = rescore 전 logit)을,
  plan_refine[k] 전체에 forward_hook(출력 = 누적 궤적 reg)을 건다. 프레임마다 refine 호출 수(= operation_order 의 'refine' 수)와
  층 순서를 단언하고, 아래 세 가지로 '그 텐서가 맞는지' 매 표본 검사한다(다르면 중단 — 건너뛰지 않음):
    (a) plan_refine[-1] 의 reg 출력[명령 묶음] == 같은 실행의 plan_modes (1e-3 m)
    (b) rescore 로 0 이 되지 않은 모드에서 softmax(plan_cls_branch 출력[명령 묶음]) == plan_mode_probs (1e-4)
    (c) plan_temp_2hz == plan_modes[argmax plan_mode_probs]
  (val) --results 를 주면 덤프 모드·확률을 평가 results.pkl 과 비교해 차이 분포·argmax 뒤집힘 수를 기록한다.
  주의(실측 2026-10-19): GPU 추론은 비트 단위로 재현되지 않는다 — 같은 설정 두 번 실행에서 모드 최대 4.3e-3 m, 특징 1.3e-2 차이.
  원인: mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu:184 정방향 커널이 atomicAdd 로 누적(합산 순서 비결정) → 시간 뱅크로 전파.
  확률이 거의 균등해서 이 작은 차이로도 argmax 가 바뀔 수 있다. 그래서 apply_and_eval.py 는 전/후를 '같은 덤프 실행' 에서 비교한다.

재개: 샤드 = 연속한 장면 묶음(--scenes_per_shard). 샤드를 끝낼 때마다 시간 뱅크 상태(stream_state.save_bank_state 와 같은 키)와
  decoder.run_step 을 shard_XXXX.state.pt 에 저장 → 재시작 시 다음 샤드 직전 상태를 그대로 복원한다(연속 실행과 같은 상태; 남는 차이는 위 atomicAdd 비결정 수준).
  (주의: 현재 모델은 장면이 바뀌어도 계획·자차 뱅크의 이전 특징을 시간 어텐션 키로 쓴다 — mask 는 계산되지만 plan/ego 뱅크에서 쓰이지 않음.
   그래서 장면 경계에서 끊고 새로 시작하면 평가와 달라진다. 상태 저장·복원은 이 때문이다.)

사용: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. OPENBLAS_CORETYPE=SkylakeX python tools/plan_scorer/dump_plan_feats.py \
        <test_config> <checkpoint> --split {train,val} --out <dir> [--results evaluation/<run>/results.pkl] [--workers 8]
      스모크(검증 전용, 실제 학습에 쓰지 못하게 manifest 에 smoke 표시): --smoke_samples 3 --max_gpu_mib 6000
"""
import argparse
import copy
import functools
import importlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import scorer_io  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=None, help="(val) 평가 results.pkl — 덤프가 평가 결과를 재현하는지 대조")
    ap.add_argument("--ref_gross_tol", type=float, default=2.0,
                    help="--results 대조: 표본 하나의 모드 최대차가 이 값(m)을 넘으면 설정 오류로 보고 중단(비결정 차이는 기록만)")
    ap.add_argument("--ref_mean_tol", type=float, default=0.3,
                    help="--results 대조: 표본별 모드 최대차의 평균이 이 값(m)을 넘으면 중단(실측 비결정 평균 ≈ 0.07 m)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenes_per_shard", type=int, default=25)
    ap.add_argument("--smoke_samples", type=int, default=0, help=">0 이면 처음 N 표본만(검증 전용 스모크)")
    ap.add_argument("--max_gpu_mib", type=int, default=0, help=">0 이면 이 프로세스의 GPU 메모리 상한(MiB) — 공유 GPU 보호")
    from mmcv import DictAction
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default=None,
                    help="tools/test.py 에 넘긴 것과 같은 값만 쓸 것(평가와 같은 설정)")
    return ap.parse_args()


def scene_shards(infos, spp):
    """장면 경계에 맞춘 샤드 [(start, end), ...]. 장면이 연속·시간 증가인지 단언."""
    starts = [0]
    seen = {infos[0]["scene_token"]}
    for i in range(1, len(infos)):
        if infos[i]["scene_token"] != infos[i - 1]["scene_token"]:
            if infos[i]["scene_token"] in seen:
                raise RuntimeError(f"장면 {infos[i]['scene_token']} 이 연속하지 않다(색인 {i}) — 평가 순서 재현 불가")
            seen.add(infos[i]["scene_token"]); starts.append(i)
        elif infos[i]["timestamp"] <= infos[i - 1]["timestamp"]:
            raise RuntimeError(f"색인 {i}: 장면 안 시간이 증가하지 않는다")
    starts.append(len(infos))
    return [(starts[s], starts[min(s + spp, len(starts) - 1)]) for s in range(0, len(starts) - 1, spp)]


def main():
    a = parse()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("GPU1 전용(사용자 규칙): CUDA_VISIBLE_DEVICES=1 로 실행한다. GPU0 사용 금지.")
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet.datasets import build_dataset, build_dataloader

    if a.max_gpu_mib > 0:
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gpu_mib * 2 ** 20 / tot), 0)

    cfg = Config.fromfile(a.config)
    if a.cfg_options:
        cfg.merge_from_dict(a.cfg_options)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    # tools/test.py 와 같은 플러그인 경로(projects.mmdet3d_plugin) — 다른 이름으로 적재하면 레지스트리가 이중 등록된다
    importlib.import_module(os.path.dirname(cfg.plugin_dir).replace("/", "."))
    from projects.mmdet3d_plugin.models.decoder.temporal_bank.stream_state import (
        bank_state_keys, save_bank_state, load_bank_state)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    ds_cfg = copy.deepcopy(cfg.data.test)
    ds_cfg.test_mode = True
    if ds_cfg.pop("samples_per_gpu", 1) != 1:
        raise SystemExit("data.test.samples_per_gpu != 1 — 평가(배치 1, 순차)와 다르다")
    if a.split == "train":
        ds_cfg.ann_file = cfg.data.train.ann_file
    if not os.path.basename(ds_cfg.ann_file).endswith(f"_{a.split}.pkl"):
        raise SystemExit(f"ann_file {ds_cfg.ann_file} 가 분할 {a.split} 와 맞지 않는다")
    ds = build_dataset(ds_cfg)
    infos = ds.data_infos
    N_all = len(infos)
    shards = scene_shards(infos, a.scenes_per_shard)
    if a.smoke_samples > 0:  # 스모크: 같은 장면 경계 샤드를 처음 N 표본에서 자른다(재개 경로도 시험 가능)
        n_sm = min(a.smoke_samples, N_all)
        shards = [(s0, min(s1, n_sm)) for s0, s1 in shards if s0 < n_sm]

    os.makedirs(a.out, exist_ok=True)
    man = {
        "config": a.config, "checkpoint": scorer_io.file_sig(a.checkpoint), "split": a.split,
        "ann_file": ds_cfg.ann_file, "num_samples": shards[-1][1] if a.smoke_samples else N_all,
        "shards": shards, "smoke": bool(a.smoke_samples), "cfg_options": dict(a.cfg_options or {}),
        "feature": "plan_refine[-1].plan_cls_branch input (plan_instance_feature + plan_anchor_embed), command group",
    }
    man_p = os.path.join(a.out, "manifest.json")
    if os.path.exists(man_p):
        old = json.load(open(man_p))
        cmp_keys = ("config", "checkpoint", "split", "ann_file", "num_samples", "shards", "smoke", "cfg_options")
        old["shards"] = [tuple(s) for s in old["shards"]]; man_cmp = dict(man, shards=[tuple(s) for s in shards])
        diff = [k for k in cmp_keys if old.get(k) != man_cmp.get(k)]
        if diff:
            raise SystemExit(f"{a.out} 의 기존 manifest 와 다르다({diff}) — 다른 출력 폴더를 쓰거나 기존 덤프를 확인할 것")
    else:
        json.dump(man, open(man_p, "w"), indent=1, ensure_ascii=False)

    done = [os.path.exists(scorer_io.shard_path(a.out, k)) for k in range(len(shards))]
    if all(done):
        print(f"[완료] {a.out}: 샤드 {len(shards)} 개 모두 있다")
        return
    first = done.index(False)
    if any(done[first:]):
        raise SystemExit(f"샤드 {first} 이후에 이미 있는 샤드가 있다 — 순차 덤프 가정 위반. 폴더 상태를 확인할 것")

    ref = None
    if a.results:
        if a.split != "val":
            raise SystemExit("--results 대조는 val(평가와 같은 분할) 전용")
        _tl = torch.load
        torch.load = functools.partial(_tl, map_location="cpu", weights_only=False)
        try:
            import pickle
            res = pickle.load(open(a.results, "rb"))
        finally:
            torch.load = _tl
        if len(res) != N_all:
            raise SystemExit(f"results.pkl 길이 {len(res)} != 데이터셋 {N_all}")
        ref = [(np.asarray(r.get("img_bbox", r)["plan_modes"]), np.asarray(r.get("img_bbox", r)["plan_mode_probs"]))
               for r in res]
        del res

    # ---- 모델(tools/test.py 비분산 경로와 같은 순서) ----
    set_random_seed(0, deterministic=False)
    from mmdet.models import build_detector
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
    torch_load = torch.load
    torch.load = functools.partial(torch_load, weights_only=False)
    try:
        load_checkpoint(model, a.checkpoint, map_location="cpu")
    finally:
        torch.load = torch_load
    model = MMDataParallel(model, device_ids=[0]).eval()
    det = model.module
    dec = det.head.onedecoder_head
    if not dec.is_init_bank_list:
        dec.init_instance_bank_list()
    n_refine = list(dec.operation_order).count("refine")
    if len(dec.plan_refine) != n_refine:
        raise RuntimeError(f"plan_refine {len(dec.plan_refine)} != refine 연산 {n_refine}")
    if len(dec.plan_anchor_types) != 1:
        raise RuntimeError(f"앵커 묶음 {dec.plan_anchor_types} — 이 덤프는 묶음 1 개(temp,2hz) 가정(align_query = 그 묶음 특징)")
    C, M, T = dec.ego_fut_cmd, dec.ego_fut_mode, dec.ego_fut_ts

    calls = []
    hooks = []
    for k, pr in enumerate(dec.plan_refine):
        hooks.append(pr.plan_cls_branch.register_forward_pre_hook(
            lambda mod, inp, k=k: calls.append(("feat", k, inp[0].detach().float()))))
        hooks.append(pr.plan_cls_branch.register_forward_hook(
            lambda mod, inp, out, k=k: calls.append(("cls", k, out.detach().float()))))
        hooks.append(pr.register_forward_hook(
            lambda mod, inp, out, k=k: calls.append(("reg", k, out[0].detach().float()))))

    def bank_objs():
        return {n: getattr(dec, f"{n}_instance_bank_list")[0] for n in ("ego", "det", "map", "plan")
                if n in dec.query_select}

    def save_state(k):
        st = {"banks": {n: save_bank_state(b, bank_state_keys(b)) for n, b in bank_objs().items()},
              "run_step": dec.run_step, "last_scene_token": getattr(det, "_last_scene_token", None)}
        p = scorer_io.state_path(a.out, k)
        torch.save(st, p + ".tmp"); os.replace(p + ".tmp", p)

    def load_state(k):
        st = torch.load(scorer_io.state_path(a.out, k), map_location="cuda:0", weights_only=False)
        for n, b in bank_objs().items():
            load_bank_state(b, st["banks"][n])
        dec.run_step = st["run_step"]
        if hasattr(det, "_last_scene_token"):
            det._last_scene_token = st["last_scene_token"]

    if first > 0:
        if not os.path.exists(scorer_io.state_path(a.out, first - 1)):
            raise SystemExit(f"샤드 {first - 1} 의 뱅크 상태 파일이 없다 — 정확한 재개 불가")
        load_state(first - 1)
        print(f"[재개] 샤드 {first} 부터(샤드 {first - 1} 직후 뱅크 상태 복원)")

    from torch.utils.data import Subset
    stats = {"max_mode_diff": 0.0, "max_prob_diff": 0.0, "max_ref_mode_diff": 0.0, "max_ref_prob_diff": 0.0,
             "sum_ref_mode_diff": 0.0, "n_ref": 0, "n_ref_argmax_flip": 0}
    t0 = time.time(); n_done = 0
    for k in range(first, len(shards)):
        s0, s1 = shards[k]
        loader = build_dataloader(Subset(ds, list(range(s0, s1))), samples_per_gpu=1,
                                  workers_per_gpu=a.workers, dist=False, shuffle=False)
        rec = {key: [] for key in scorer_io.PER_SAMPLE_KEYS}
        fut_all = []
        for j, data in enumerate(loader):
            i = s0 + j
            info = infos[i]
            calls.clear()
            with torch.no_grad():
                r = model(return_loss=False, rescale=True, **data)[0]["img_bbox"]
            feats = [c for c in calls if c[0] == "feat"]
            clss = [c for c in calls if c[0] == "cls"]
            regs = [c for c in calls if c[0] == "reg"]
            if [c[1] for c in feats] != list(range(n_refine)) or len(clss) != n_refine or len(regs) != n_refine:
                raise RuntimeError(f"색인 {i}: plan_refine 호출 순서 {[c[1] for c in feats]} — 층 {n_refine} 개 순서대로가 아니다")
            cmd = int(np.argmax(np.asarray(info["gt_ego_fut_cmd"])))
            f = feats[-1][2]; cl = clss[-1][2]; rg = regs[-1][2]
            if f.shape != (1, C * M, f.shape[-1]) or cl.shape != (1, C * M, 1) or rg.shape != (1, C * M, T * 2):
                raise RuntimeError(f"색인 {i}: 훅 텐서 모양 {f.shape} {cl.shape} {rg.shape}")
            feat16 = f.view(C, M, -1)[cmd].cpu().numpy()
            raw16 = cl.view(C, M)[cmd].cpu().numpy()
            reg16 = rg.view(C, M, T, 2)[cmd].cpu().numpy()
            modes = np.asarray(r["plan_modes"], np.float32)
            probs = np.asarray(r["plan_mode_probs"], np.float32)
            plan = np.asarray(r["plan_temp_2hz"], np.float32)
            # (a)(b)(c) 텐서 위치 검사
            dm = float(np.abs(reg16 - modes).max())
            nz = probs > 0
            sm = torch.softmax(torch.as_tensor(raw16[nz]), 0).numpy()
            dp = float(np.abs(sm - probs[nz] / probs[nz].sum()).max()) if nz.any() else 0.0
            sel = int(np.argmax(probs))
            if dm > 1e-3 or dp > 1e-4 or not nz.any() or np.abs(plan - modes[sel]).max() > 1e-6:
                raise RuntimeError(f"색인 {i}: 훅 텐서가 결과와 맞지 않다(모드 차 {dm:.2e}, 확률 차 {dp:.2e}) — 훅 위치 재확인")
            stats["max_mode_diff"] = max(stats["max_mode_diff"], dm)
            stats["max_prob_diff"] = max(stats["max_prob_diff"], dp)
            if ref is not None:
                # 수정(2026-10-19): 실측 — 같은 설정 두 번 실행도 첫 프레임 4e-3 m, 한 장면(40 프레임) 끝 무렵 모드 최대 0.27 m 까지
                #   벌어진다(원인: ops/src/deformable_aggregation_cuda.cu:184 정방향 atomicAdd 누적 순서 비결정 → 시간 뱅크로 증폭).
                #   그래서 평가 결과와의 차이는 '측정·기록' 하고, 설정 오류 수준(표본 > ref_gross_tol, 평균 > ref_mean_tol)만 중단한다.
                rdm = float(np.abs(ref[i][0] - modes).max()); rdp = float(np.abs(ref[i][1] - probs).max())
                stats["max_ref_mode_diff"] = max(stats["max_ref_mode_diff"], rdm)
                stats["max_ref_prob_diff"] = max(stats["max_ref_prob_diff"], rdp)
                stats["sum_ref_mode_diff"] += rdm; stats["n_ref"] += 1
                stats["n_ref_argmax_flip"] += int(int(np.argmax(ref[i][1])) != int(np.argmax(probs)))
                if rdm > a.ref_gross_tol:
                    raise RuntimeError(f"색인 {i}: 덤프가 평가 results.pkl 과 {rdm:.3f} m 다르다(> {a.ref_gross_tol} m) — "
                                       "비결정 수준을 넘는다. 평가 설정·순서·체크포인트 확인(건너뛰지 않음)")
            fut = ds._get_fut_boxes(i, info)  # 평가기와 같은 함수·같은 GT 유효 마스크 규칙
            cnt, boxes = scorer_io.pack_fut_boxes(fut, T)
            rec["idx"].append(i); rec["token"].append(info["token"]); rec["scene_token"].append(info["scene_token"])
            rec["timestamp"].append(float(info["timestamp"]) / 1e6); rec["cmd"].append(cmd)
            rec["modes"].append(modes); rec["probs"].append(probs); rec["raw_logits"].append(raw16); rec["sel"].append(sel)
            rec["feats"].append(feat16.astype(np.float32))
            rec["gt_traj"].append(np.asarray(info["gt_ego_fut_trajs"], np.float32).cumsum(0)[:T, :2])
            rec["gt_mask"].append(np.asarray(info["gt_ego_fut_masks"], np.float32)[:T])
            rec["fut_box_counts"].append(cnt); fut_all.append(boxes)
            rec["lidar2global"].append(ds._T_global(info)); rec["map_location"].append(info["map_location"])
            n_done += 1
        out = {key: np.asarray(v) for key, v in rec.items()}
        out["token"] = out["token"].astype("U64"); out["scene_token"] = out["scene_token"].astype("U64")
        out["map_location"] = out["map_location"].astype("U64")
        out["fut_boxes"] = np.concatenate(fut_all, 0) if fut_all else np.zeros((0, 7), np.float32)
        scorer_io.save_shard(a.out, k, out)
        save_state(k)
        el = time.time() - t0
        print(f"[샤드 {k + 1}/{len(shards)}] 색인 {s0}..{s1 - 1} 저장 | 누적 {n_done} 표본 {el:.0f}s "
              f"({el / max(n_done, 1):.3f}s/표본) | 검사 최대차 {json.dumps(stats)}", flush=True)
    for h in hooks:
        h.remove()
    if stats["n_ref"]:
        stats["mean_ref_mode_diff"] = stats["sum_ref_mode_diff"] / stats["n_ref"]
        if stats["mean_ref_mode_diff"] > a.ref_mean_tol:
            raise RuntimeError(f"results.pkl 과 평균 모드 차 {stats['mean_ref_mode_diff']:.3f} m > {a.ref_mean_tol} — 비결정 수준을 넘는다. "
                               "평가 설정·순서·체크포인트 확인")
    json.dump(stats, open(os.path.join(a.out, f"check_stats_from_shard{first}.json"), "w"), indent=1)
    print("[완료]", a.out, json.dumps(stats))


if __name__ == "__main__":
    main()
