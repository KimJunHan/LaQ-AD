# 수정(2026-11-09): 사용자 결정 "6단계 양자화: PTQ(학습 후 양자화) 정확도 표 먼저, 저하가 잡음 바닥을 넘을 때만 QAD".
#   설계 docs/stage5_6_feasibility_20261106.md §B.5. 새 도구 — 모델 소스(models/, projects/)는 고치지 않고, 만들어진 모델에
#   forward-pre-hook(활성 가짜 양자화)과 가중치 제자리 가짜 양자화(양자화→역양자화 값으로 교체)를 덧씌운다.
#   가짜 양자화 = 정확도 시뮬레이션일 뿐 지연 이득을 뜻하지 않는다(연산은 여전히 fp32).
#
#   흐름(추론 전용, 학습 없음):
#     1) config·체크포인트로 모델 A 를 만든다(tools/test.py 와 같은 절차: plugin 등록, fp16 설정이면 wrap_fp16_model, load_checkpoint).
#     2) --quant 가 none 이 아니면: 백본 BN 을 합성곱에 접고(fuse_conv_bn, 배포 시 TensorRT 와 같은 순서), 대상 Conv2d/Linear 에
#        관측 hook 을 달아 nuScenes **train** 에서 고른 보정 표본(무작위 장면 32 개 × 연속 8 프레임 = 256, seed 고정; val 사용 금지)으로
#        입력 |x| 최댓값(max 보정)을 모은다. 클립마다 시간 뱅크를 비운다.
#     3) 보정 상태가 val 추론에 새지 않도록 **모델 B 를 새로 만들어** 체크포인트를 다시 읽고, 같은 접기·가중치 양자화·활성 양자화 hook 을 단다.
#     4) val(또는 --limit N 앞 N 표본) 순차 추론(mmdet single_gpu_test) → results.pkl. 지표 계산은 이 도구에서 하지 않는다
#        (스크립트가 tools/eval_from_pkl.py(= test.py 의 dataset.evaluate 경로)·tools/eval_nusc_planning.py 로 CPU 에서 따로 한다).
#
#   양자화 방식(사전에 고정, val 로 고르지 않음):
#     int8 : 가중치 출력 채널별 대칭(scale = max|w|/127, 정수 [-127,127]), 활성 텐서별 대칭(scale = amax/127, amax = 보정 max).
#     fp8  : torch.float8_e4m3fn 캐스팅 왕복. 가중치·활성 모두 텐서별 scale = amax/448(e4m3 최댓값), 캐스팅 전 ±448 로 자름
#            (PyTorch 의 fp8 캐스팅은 넘치면 포화가 아니라 NaN 이므로).
#     대상(--scope): bb = img_backbone + img_neck 의 Conv2d/Linear 전부(줄기 7×7 포함).
#                    bb_dec = bb + head 의 nn.Linear(교사 융합 QwenTokenAttn 하위 전체 제외 — 교사 교차 어텐션은 fp32/원 정밀도 유지).
#     DAF(deformable aggregation CUDA 연산) 자체는 Linear/Conv 가 아니므로 손대지 않는다(fp32 그대로). DAF 앞뒤의 Linear(오프셋·가중치·출력 사영)는 bb_dec 에서 양자화된다.
#     보정 중 한 번도 호출되지 않은 대상 모듈은 양자화하지 않고 목록으로 남긴다(조용히 넘기지 않음: stats json 의 never_called).
#   정밀도(--precision): fp32 = config 의 fp16 을 끈다(백본·FPN 도 fp32). mixed = config 그대로(학습 정밀도: 백본 fp16 auto_fp16, FPN bf16).
#
#   사용:
#     python tools/quant/ptq_fakequant.py <config> <ckpt> --out <dir>/results.pkl --precision fp32 --quant int8 --scope bb
#        [--limit 50] [--calib-clips 32 --calib-len 8 --calib-seed 0] [--stats <json>]
import argparse
import copy
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_orig_torch_load = torch.load
def _torch_load_compat(*a, **kw):  # tools/test.py 와 같은 shim(레거시 체크포인트)
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)
torch.load = _torch_load_compat

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
import mmcv
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import single_gpu_test, set_random_seed
from mmdet.datasets import build_dataset
from mmdet.datasets import build_dataloader as build_dataloader_origin
from mmdet.models import build_detector

FP8_MAX = 448.0


# ---------------------------------------------------------------- 가짜 양자화 함수
def fq_int8(x, scale):
    """대칭 INT8 가짜 양자화. scale 은 x 에 브로드캐스트 가능한 텐서."""
    return torch.clamp(torch.round(x / scale), -127, 127) * scale


def fq_fp8(x, scale):
    """e4m3 가짜 양자화(캐스팅 왕복)."""
    return (x / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(x.dtype) * scale


def quantize_weight_(mod, kind):
    w = mod.weight.data
    wf = w.float()
    if kind == "int8":
        amax = wf.abs().flatten(1).amax(1).clamp_min(1e-12).view(-1, *([1] * (w.dim() - 1)))  # 출력 채널별
        wq = fq_int8(wf, amax / 127.0)
    else:
        amax = wf.abs().amax().clamp_min(1e-12)
        wq = fq_fp8(wf, amax / FP8_MAX)
    rel = float((wq - wf).norm() / wf.norm().clamp_min(1e-12))
    mod.weight.data.copy_(wq.to(w.dtype))
    return rel


class ActQuant:
    """입력 활성 텐서별 가짜 양자화 forward-pre-hook. mode = 'observe' | 'quant'."""

    def __init__(self, name, kind):
        self.name, self.kind = name, kind
        self.mode = "observe"
        self.amax = 0.0
        self.calls = 0
        self.nonfinite = 0
        self.zero_pass = 0  # amax=0 모듈에 전부 0 입력이 와서 그대로 통과한 횟수
        # 수정(2026-11-09): 조정자 지시 — ego DAF 비0 입력 원인 조사(train 전용 탐침). probe=True 이면 amax=0 모듈의 비0 입력을
        #   멈추지 않고 기록만 하고 양자화 없이 통과시킨다(탐침 실행에서만; val 추론에서는 쓰지 않는다).
        self.probe = False
        self.probe_nonzero = 0
        self.probe_amax = 0.0

    def __call__(self, mod, args):
        x = args[0]
        if not torch.is_tensor(x) or not x.is_floating_point():
            raise TypeError(f"{self.name}: 첫 입력이 부동소수 텐서가 아니다({type(x)})")
        self.calls += 1
        if self.mode == "observe":
            xf = x.detach().float()
            fin = torch.isfinite(xf)
            if not bool(fin.all()):
                self.nonfinite += 1  # 보고용(보정 입력 자체에 비유한 값이 있으면 원인 조사 대상)
                raise FloatingPointError(f"{self.name}: 보정 입력에 비유한 값")
            self.amax = max(self.amax, float(xf.abs().amax()))
            return None
        s = self.amax / (127.0 if self.kind == "int8" else FP8_MAX)
        if s <= 0:
            # 보정에서 입력이 늘 0 이었던 모듈(예: ego DAF 출력 사영 — 자차 앵커 키포인트가 영상 밖이라 DAF 출력이 0).
            #   0 의 양자화는 0 이므로 입력이 실제로 전부 0 이면 그대로 둔다(항등과 같음). 0 이 아닌 입력이 오면 scale 을 정할 수 없어 멈춘다.
            if bool((x == 0).all()):
                self.zero_pass += 1
                return None
            if self.probe:
                self.probe_nonzero += 1
                self.probe_amax = max(self.probe_amax, float(x.detach().float().abs().amax()))
                return None
            raise ValueError(f"{self.name}: 보정 amax 가 0 인데 0 이 아닌 입력 — 양자화 scale 을 정할 수 없다")
        xf = x.float()
        xq = fq_int8(xf, s) if self.kind == "int8" else fq_fp8(xf, s)
        return (xq.to(x.dtype),) + tuple(args[1:])


# ---------------------------------------------------------------- 모델·데이터
def import_plugin(cfg):
    import importlib
    if cfg.get("plugin", False):
        plugin_dir = cfg.get("plugin_dir", "projects/mmdet3d_plugin/")
        importlib.import_module(osp.dirname(plugin_dir).replace("/", "."))


def build_model(cfg, ckpt):
    c = copy.deepcopy(cfg)
    c.model.train_cfg = None
    model = build_detector(c.model, test_cfg=c.get("test_cfg"))
    if c.get("fp16", None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, ckpt, map_location="cpu")
    model.eval()
    return model


def target_modules(model, scope):
    """(이름, 모듈) 목록. 교사 융합(QwenTokenAttn) 하위는 제외."""
    out = []
    for part in ("img_backbone", "img_neck"):
        sub = getattr(model, part, None)
        if sub is None:
            continue
        for n, m in sub.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                out.append((f"{part}.{n}", m))
    excluded = []
    if scope == "bb_dec":
        teacher_prefixes = [n for n, m in model.head.named_modules() if type(m).__name__ == "QwenTokenAttn"]
        for n, m in model.head.named_modules():
            if isinstance(m, nn.Linear):
                if any(n == p or n.startswith(p + ".") for p in teacher_prefixes):
                    excluded.append(f"head.{n}")
                else:
                    out.append((f"head.{n}", m))
        mha = [f"head.{n}" for n, m in model.head.named_modules() if isinstance(m, nn.MultiheadAttention)]
        if mha:  # nn.MultiheadAttention 의 in_proj 는 Linear 모듈이 아니어서 대상에서 빠진다 — 숨기지 않고 보고
            excluded += [f"{n}(nn.MultiheadAttention in_proj)" for n in mha]
    return out, excluded


def reset_banks(model):
    n = 0
    for m in model.modules():
        if "temporal_bank" in type(m).__module__ and hasattr(m, "reset"):
            m.reset(); n += 1
    if n == 0:
        raise RuntimeError("시간 뱅크를 하나도 찾지 못했다(뱅크 초기화 불가)")
    return n


def calib_indices(train_infos, clips, length, seed):
    """무작위 장면 clips 개 × 장면 안 연속 length 프레임(train 전용)."""
    scenes = {}
    for i, info in enumerate(train_infos):
        scenes.setdefault(info["scene_token"], []).append(i)
    rng = np.random.default_rng(seed)
    toks = sorted(t for t, ix in scenes.items() if len(ix) >= length)
    pick = rng.choice(len(toks), size=clips, replace=False)
    out = []
    for p in pick:
        ix = scenes[toks[p]]
        s = int(rng.integers(0, len(ix) - length + 1))
        clip = ix[s:s + length]
        assert all(b - a == 1 for a, b in zip(clip, clip[1:])), "train infos 가 장면 안에서 연속 정렬돼 있지 않다"
        out.append(clip)
    return out


def calibrate(model, cfg, hooks, args):
    tcfg = copy.deepcopy(cfg.data.test)
    tcfg.ann_file = args.calib_ann
    tcfg.test_mode = True
    ds = build_dataset(tcfg)
    clips = calib_indices(ds.data_infos, args.calib_clips, args.calib_len, args.calib_seed)
    flat = [i for c in clips for i in c]
    val_tokens = set()
    if osp.exists(cfg.data.test.ann_file):
        import pickle
        d = pickle.load(open(cfg.data.test.ann_file, "rb"))
        val_tokens = {x["token"] for x in (d["infos"] if isinstance(d, dict) else d)}
    leak = [ds.data_infos[i]["token"] for i in flat if ds.data_infos[i]["token"] in val_tokens]
    if leak:
        raise RuntimeError(f"보정 표본에 val token {len(leak)} 개 — val 사용 금지")
    sub = torch.utils.data.Subset(ds, flat)
    dl = build_dataloader_origin(sub, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)
    dp = MMDataParallel(model, device_ids=[0])
    t0 = time.time()
    with torch.no_grad():
        for k, data in enumerate(dl):
            if k % args.calib_len == 0:
                reset_banks(model)
            dp(return_loss=False, rescale=True, **data)
    reset_banks(model)
    print(f"[ptq] 보정 {len(flat)} 표본({args.calib_clips} 클립 × {args.calib_len}) {time.time() - t0:.0f}s", flush=True)
    return {"clips_first_token": [ds.data_infos[c[0]]["token"] for c in clips], "n": len(flat), "ann_file": args.calib_ann}


def attach(model, scope, kind):
    if hasattr(model, "img_backbone"):
        model.img_backbone = fuse_conv_bn(model.img_backbone)
    mods, excluded = target_modules(model, scope)
    hooks = {}
    for n, m in mods:
        h = ActQuant(n, kind)
        m.register_forward_pre_hook(h)
        hooks[n] = (m, h)
    return hooks, excluded


EGO_OUT = "head.onedecoder_head.ego_deformable.{}.output_proj"


def ego_layers(model):
    dec = model.head.onedecoder_head
    return list(dec.ego_deformable)


class EgoProbe:
    """ego DAF 층별 탐침: output_proj 입력(DAF 출력)의 비0 여부·amax 와, 키포인트가 영상 안(깊이>1e-5, 정규화 u,v∈[0,1])에 드는 수."""

    def __init__(self, model):
        self.layers = ego_layers(model)
        self.cur = {}
        self.rows = []
        for i, daf in enumerate(self.layers):
            daf.register_forward_pre_hook(self._kps_hook(i))
            daf.output_proj.register_forward_pre_hook(self._out_hook(i))

    def _kps_hook(self, i):
        def f(mod, args):
            inst, anchor, anchor_embed, fmaps, metas = args[:5]
            with torch.no_grad():
                kps = mod.kps_generator(anchor, anchor_embed, inst)            # (B, A, P, 3)
                P = metas["projection_mat"]                                    # (B, C, 4, 4)
                ext = torch.cat([kps, torch.ones_like(kps[..., :1])], -1)
                p = torch.matmul(P[:, :, None, None], ext[:, None, ..., None]).squeeze(-1)  # (B,C,A,P,4)
                z = p[..., 2]
                uv = p[..., :2] / z.clamp(min=1e-5)[..., None]
                wh = metas.get("image_wh")
                if wh is not None:
                    uv = uv / wh[:, :, None, None]
                inside = (z > 1e-5) & (uv >= 0).all(-1) & (uv <= 1).all(-1)
                front = (z > 1e-5)
                self.cur[i] = dict(n_inside=int(inside.sum()), cams_inside=sorted({int(c) for c in inside[0].nonzero()[:, 0].tolist()}),
                                   n_in_front=int(front.sum()), n_pts=int(kps.shape[2]),
                                   v_max_in_front=float(uv[..., 1][front].max()) if bool(front.any()) else None,
                                   anchor_xyz=[round(float(v), 3) for v in anchor[0, 0, :3]],
                                   kps_z_range=[round(float(kps[..., 2].min()), 3), round(float(kps[..., 2].max()), 3)],
                                   kps_x_range=[round(float(kps[..., 0].min()), 3), round(float(kps[..., 0].max()), 3)])
            return None
        return f

    def _out_hook(self, i):
        def f(mod, args):
            x = args[0].detach().float()
            self.cur[i].update(nonzero=int((x != 0).sum()), amax=float(x.abs().amax()))
            return None
        return f

    def end_frame(self, token, k):
        self.rows.append(dict(token=token, k=k, layers=[self.cur.get(i) for i in range(len(self.layers))]))
        self.cur = {}


def probe_ego(cfg, args, stats):
    """수정(2026-11-09): 조정자 지시 (1)(3) — train 전용. 무작위 장면 N 개의 전체 프레임을 장면 순서대로(장면 시작마다 뱅크 초기화) 돌려
    ego_deformable.{0..5}.output_proj 입력의 비0 빈도·amax 와 키포인트 투영을 기록한다. val 은 쓰지 않는다."""
    tcfg = copy.deepcopy(cfg.data.test); tcfg.ann_file = args.calib_ann; tcfg.test_mode = True
    ds = build_dataset(tcfg)
    scenes = {}
    for i, info in enumerate(ds.data_infos):
        scenes.setdefault(info["scene_token"], []).append(i)
    toks = sorted(scenes)
    rng = np.random.default_rng(args.probe_seed)
    pick = sorted(rng.choice(len(toks), size=min(args.probe_ego_scenes, len(toks)), replace=False).tolist())
    idx = [i for p in pick for i in scenes[toks[p]]]
    starts = {scenes[toks[p]][0] for p in pick}
    import pickle
    d = pickle.load(open(cfg.data.test.ann_file, "rb"))
    val_tokens = {x["token"] for x in (d["infos"] if isinstance(d, dict) else d)}
    if any(ds.data_infos[i]["token"] in val_tokens for i in idx):
        raise RuntimeError("탐침 표본에 val token — val 사용 금지")
    hooks = {}
    if args.quant != "none":   # 양자화 모델(같은 보정 256 → 평가 모델과 같은 상태)에서의 탐침
        mA = build_model(cfg, args.checkpoint).cuda()
        hA, _ = attach(mA, args.scope, args.quant)
        stats["calib"] = calibrate(mA, cfg, hA, args)
        amax = {n: h.amax for n, (m, h) in hA.items()}
        del mA, hA; torch.cuda.empty_cache()
        model = build_model(cfg, args.checkpoint)
        hooks, _ = attach(model, args.scope, args.quant)
        for n, (m, h) in hooks.items():
            h.amax = amax[n]; h.mode = "quant"; h.probe = True
            quantize_weight_(m, args.quant)
    else:
        model = build_model(cfg, args.checkpoint)
    probe = EgoProbe(model)
    model = model.cuda()
    dl = build_dataloader_origin(torch.utils.data.Subset(ds, idx), samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
                                 dist=False, shuffle=False)
    dp = MMDataParallel(model, device_ids=[0])
    t0 = time.time()
    with torch.no_grad():
        for k, data in enumerate(dl):
            if idx[k] in starts:
                reset_banks(model)
            dp(return_loss=False, rescale=True, **data)
            probe.end_frame(ds.data_infos[idx[k]]["token"], k)
            if k % 500 == 0:
                print(f"[probe] {k}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
    L = len(probe.layers)
    summ = {}
    for i in range(L):
        rs = [r["layers"][i] for r in probe.rows]
        nz = [r for r in rs if r["nonzero"] > 0]
        summ[EGO_OUT.format(i)] = dict(
            frames=len(rs), nonzero_frames=len(nz), amax=max([r["amax"] for r in rs] + [0.0]),
            frames_kps_inside=sum(1 for r in rs if r["n_inside"] > 0),
            nonzero_and_inside=sum(1 for r in nz if r["n_inside"] > 0),
            nonzero_not_inside=sum(1 for r in nz if r["n_inside"] == 0),
            inside_but_zero=sum(1 for r in rs if r["n_inside"] > 0 and r["nonzero"] == 0),
            max_in_front_pts=max(r["n_in_front"] for r in rs), n_pts=rs[0]["n_pts"])
    stats.update(probe_scenes=len(pick), probe_frames=len(idx), probe_sec=time.time() - t0, summary=summ,
                 nonzero_rows=[r for r in probe.rows if any(l["nonzero"] > 0 for l in r["layers"])],
                 example_zero_rows=probe.rows[:3])
    if hooks:
        stats["probe_quant_nonzero_pass"] = {n: dict(n=h.probe_nonzero, amax=h.probe_amax) for n, (m, h) in hooks.items() if h.probe_nonzero}
    print(json.dumps(summ, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True, help="results.pkl 경로")
    ap.add_argument("--precision", choices=["fp32", "mixed"], required=True)
    ap.add_argument("--quant", choices=["none", "int8", "fp8"], default="none")
    ap.add_argument("--scope", choices=["bb", "bb_dec"], default="bb")
    ap.add_argument("--limit", type=int, default=0, help=">0 이면 val 앞 N 표본만(경로 검증용)")
    ap.add_argument("--calib-ann", default="data/infos/nuscenes_infos_train.pkl")
    ap.add_argument("--calib-clips", type=int, default=32)
    ap.add_argument("--calib-len", type=int, default=8)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--stats", default=None, help="양자화 통계 json(기본 <out 폴더>/ptq_stats.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    # 수정(2026-11-09): 조정자 지시 — ego DAF 출력 사영 처리. 탐침(train 전용)과 그 결과로 amax 를 정하는 규칙.
    ap.add_argument("--probe-ego-scenes", type=int, default=0, help=">0 이면 val 추론 대신 train 장면 N 개 탐침만(--out 은 탐침 json 경로로 .json)")
    ap.add_argument("--probe-seed", type=int, default=1)
    ap.add_argument("--ego-amax-json", nargs="*", default=None,
                    help="탐침 json 들. ego_deformable.*.output_proj 의 amax = max(보정 256 amax, 탐침 비0 사례 amax). 규칙은 docs/ptq_nuscenes_20261109.md")
    ap.add_argument("--exclude-ego-out", action="store_true", help="ego_deformable.*.output_proj 를 양자화 범위에서 제외(fp32 유지)")
    args = ap.parse_args()
    if args.probe_ego_scenes > 0:
        if not args.out.endswith(".json") or osp.exists(args.out):
            raise ValueError("탐침 --out 은 새 .json")
        cfg = Config.fromfile(args.config); import_plugin(cfg)
        if args.precision == "fp32":
            cfg.fp16 = None
        cfg.model.pretrained = None; cfg.data.test.test_mode = True; cfg.data.test.pop("samples_per_gpu", None)
        set_random_seed(args.seed, deterministic=False)
        cfg.work_dir = osp.join("./work_dirs", cfg.get("run_name") or "ptq"); cfg.data.test.work_dir = cfg.work_dir
        st = dict(mode="probe_ego", precision=args.precision, quant=args.quant, scope=args.scope, probe_seed=args.probe_seed,
                  ann_file=args.calib_ann, started=time.strftime("%Y-%m-%d %H:%M:%S"))
        mmcv.mkdir_or_exist(osp.dirname(osp.abspath(args.out)))
        probe_ego(cfg, args, st)
        st["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        json.dump(st, open(args.out, "w"), indent=1, ensure_ascii=False)
        print(f"[probe] → {args.out}"); return
    if not args.out.endswith(".pkl"):
        raise ValueError("--out 은 .pkl")
    if osp.exists(args.out):
        raise FileExistsError(f"{args.out} 가 이미 있다(덮어쓰지 않음)")
    if "train" not in osp.basename(args.calib_ann):
        raise ValueError("보정은 train 분할만(val 사용 금지)")

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    import_plugin(cfg)
    if args.precision == "fp32":
        cfg.fp16 = None  # 백본 auto_fp16·FPN bf16 끔 → 전부 fp32
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.data.test.pop("samples_per_gpu", None)
    set_random_seed(args.seed, deterministic=False)  # tools/test.py 와 같음
    if cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs", cfg.get("run_name") or osp.splitext(osp.basename(args.config))[0])
    cfg.data.test.work_dir = cfg.work_dir
    out_dir = osp.dirname(osp.abspath(args.out))
    mmcv.mkdir_or_exist(out_dir)
    stats_path = args.stats or osp.join(out_dir, "ptq_stats.json")
    stats = dict(config=args.config, checkpoint=args.checkpoint, precision=args.precision, quant=args.quant,
                 scope=args.scope, limit=args.limit, fp16_cfg=str(cfg.get("fp16", None)),
                 torch=torch.__version__, started=time.strftime("%Y-%m-%d %H:%M:%S"))

    hooks = {}
    if args.quant != "none":
        # ---- 보정(모델 A)
        mA = build_model(cfg, args.checkpoint).cuda()
        hA, excluded = attach(mA, args.scope, args.quant)
        stats["calib"] = calibrate(mA, cfg, hA, args)
        amax = {n: h.amax for n, (m, h) in hA.items()}
        calls = {n: h.calls for n, (m, h) in hA.items()}
        del mA, hA
        torch.cuda.empty_cache()
        # ---- 평가 모델 B(새로 만들어 보정 상태 누출 차단)
        model = build_model(cfg, args.checkpoint)
        hooks, excluded_b = attach(model, args.scope, args.quant)
        assert set(hooks) == set(amax) and excluded_b == excluded, "보정·평가 모델의 대상 모듈이 다르다"
        ego_names = [n for n in hooks if n.startswith("head.onedecoder_head.ego_deformable.") and n.endswith(".output_proj")]
        if args.exclude_ego_out:
            for n in ego_names:
                m, h = hooks.pop(n); m._forward_pre_hooks.clear(); excluded.append(n + "(ego DAF 출력, 규칙상 제외)")
        elif args.ego_amax_json:
            ov = {}
            for pj in args.ego_amax_json:
                pr = json.load(open(pj))
                for n, v in pr["summary"].items():
                    ov[n] = max(ov.get(n, 0.0), v["amax"])
                for n, v in pr.get("probe_quant_nonzero_pass", {}).items():
                    ov[n] = max(ov.get(n, 0.0), v["amax"])
            stats["ego_amax_override"] = {n: dict(calib256=amax[n], probe=ov.get(n, 0.0), used=max(amax[n], ov.get(n, 0.0))) for n in ego_names}
            for n in ego_names:
                amax[n] = max(amax[n], ov.get(n, 0.0))
        never = sorted(n for n in hooks if calls[n] == 0)
        wrel = {}
        for n, (m, h) in list(hooks.items()):
            if calls[n] == 0:
                h.mode = "off"  # 아래에서 hook 을 무력화
                continue
            h.amax = amax[n]; h.mode = "quant"
            wrel[n] = quantize_weight_(m, args.quant)
        for n in never:  # 호출되지 않은 모듈: 양자화하지 않음(목록 보고)
            m, h = hooks.pop(n)
            m._forward_pre_hooks.clear()
        stats.update(n_quantized=len(hooks), never_called=never, excluded=excluded,
                     act_amax=amax, calib_calls=calls, weight_rel_err=wrel,
                     n_by_part={p: sum(1 for n in hooks if n.startswith(p)) for p in ("img_backbone", "img_neck", "head")})
        print(f"[ptq] 양자화 모듈 {len(hooks)} {stats['n_by_part']} / 미호출 {len(never)} / 제외 {len(excluded)}", flush=True)
    else:
        model = build_model(cfg, args.checkpoint)

    # ---- val 추론
    ds = build_dataset(cfg.data.test)
    evalset = torch.utils.data.Subset(ds, list(range(args.limit))) if args.limit > 0 else ds
    dl = build_dataloader_origin(evalset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)
    for n, (m, h) in hooks.items():
        h.calls = 0
    model = MMDataParallel(model, device_ids=[0])
    t0 = time.time()
    outputs = single_gpu_test(model, dl, False, None)
    stats["infer_sec"] = time.time() - t0
    stats["n_samples"] = len(outputs)
    if hooks:
        zero = [n for n, (m, h) in hooks.items() if h.calls == 0]
        stats["val_uncalled_quantized"] = zero
        stats["val_calls"] = {n: h.calls for n, (m, h) in hooks.items()}
        stats["val_zero_pass"] = {n: h.zero_pass for n, (m, h) in hooks.items() if h.zero_pass}
        stats["amax_zero_modules"] = sorted(n for n, (m, h) in hooks.items() if h.amax == 0)
    print(f"\n[ptq] 추론 {len(outputs)} 표본 {stats['infer_sec']:.0f}s → {args.out}", flush=True)
    mmcv.dump(outputs, args.out)
    stats["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(stats, open(stats_path, "w"), indent=1, ensure_ascii=False)
    print(f"[ptq] 통계 {stats_path}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
