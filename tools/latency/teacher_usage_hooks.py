# 수정(2026-11-04): 교사 답 구간 사용도 측정 — forward hook 만 쓰는 추론 전용 진단. 모델 코드는 바꾸지 않는다.
#   QwenTokenAttn.forward(query, qwen_hidden, qwen_mask, layer_idx=, kv=, qwen_type=) 호출마다(디코더 6 계층) 입력·출력을 잡아
#   같은 가중치로 어텐션을 fp32 로 다시 계산한다:
#     P = softmax(q k^T / sqrt(d_h))  (머리 8 개, 쿼리 N × 토큰 L)
#     A = W_o concat_h(P V) + b_o  →  폴더 f 기여 A_f = W_o concat_h(P[:, :, f] V_f)   (Σ_f A_f + b_o = A)
#     출력 = query + g ⊙ A  (g = 계층·과제 구간·채널 게이트)
#   보고(계층 × 쿼리 종류 det/map/plan/ego × 폴더):
#     mass   = 머리 평균 어텐션 질량 Σ_{l∈f} P (쿼리 평균), mass/token = mass / L_f
#     rel    = ||g ⊙ A_f|| / ||query||  (쿼리 평균) — 게이트를 거친 폴더별 잔차 기여의 상대 크기
#     total  = ||g ⊙ A|| / ||query||, 재구성 오차 ||query + g⊙A − 출력|| / ||g⊙A||(검사)
#     gate   = |g| 평균·최대(계층·구간별)
#   부분집합: val 장면 순서에서 9 장면마다 1 장면(장면 전체 프레임, 시간 순서 — 시간 은행이 유효하도록), 결정적.
#   사용: python tools/latency/teacher_usage_hooks.py <config> <ckpt> --out results/latency/teacher_usage/<name>.json
import argparse
import json
import os
import os.path as osp
import sys
import time

import torch

_REPO = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.chdir(_REPO)
_orig_torch_load = torch.load


def _torch_load_compat(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_torch_load(*a, **kw)


torch.load = _torch_load_compat

from mmcv import Config  # noqa: E402
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmcv.runner import load_checkpoint, wrap_fp16_model  # noqa: E402
from mmdet.apis import set_random_seed  # noqa: E402
from mmdet.datasets import build_dataset  # noqa: E402
from mmdet.datasets import build_dataloader as build_dataloader_origin  # noqa: E402
from mmdet.models import build_detector  # noqa: E402

FOLDERS = ("context", "detection", "tracking", "motion", "map", "plan", "action")


class Acc:
    def __init__(self):
        self.s = {}

    def add(self, key, val, n=1):
        a = self.s.setdefault(key, [0.0, 0])
        a[0] += float(val); a[1] += n

    def mean(self):
        return {k: v[0] / max(v[1], 1) for k, v in self.s.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene-step", type=int, default=9)
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    cfg = Config.fromfile(a.config)
    import importlib
    importlib.import_module("projects.mmdet3d_plugin")
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    set_random_seed(0, deterministic=False)
    cfg.data.test.work_dir = "./work_dirs/latency_teacher_usage"
    dataset = build_dataset(cfg.data.test)
    infos = dataset.data_infos
    scenes = []
    for i, info in enumerate(infos):
        if not scenes or scenes[-1][0] != info["scene_token"]:
            scenes.append((info["scene_token"], []))
        scenes[-1][1].append(i)
    chosen = scenes[::a.scene_step]
    idx = [i for _, ii in chosen for i in sorted(ii, key=lambda j: infos[j]["timestamp"])]
    if a.max_frames:
        idx = idx[:a.max_frames]
    print(f"[usage] scenes {len(chosen)}/{len(scenes)} frames {len(idx)}", flush=True)
    sub = torch.utils.data.Subset(dataset, idx)
    loader = build_dataloader_origin(sub, samples_per_gpu=1, workers_per_gpu=8, dist=False, shuffle=False)

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, a.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0]).eval()
    dec = model.module.head.onedecoder_head if hasattr(model.module.head, "onedecoder_head") else None
    qta = None
    for m in model.modules():
        if type(m).__name__ == "QwenTokenAttn":
            qta = m
            break
    assert qta is not None
    mha = qta.attn.attn  # FlashMHA
    H = mha.num_heads
    seg_bounds = list(qta.seg_bounds)
    acc = Acc()
    info_once = {}

    def hook(mod, args, kwargs, output):
        query = args[0]
        kv = kwargs["kv"]
        li = int(kwargs.get("layer_idx", 0))
        qt = kwargs.get("qwen_type")
        with torch.no_grad():
            q32, kv32 = query.float(), kv.float()
            W, b = mha.in_proj_weight.float(), mha.in_proj_bias.float()
            Wq, Wk, Wv = W.chunk(3); bq, bk, bv = b.chunk(3)
            B, N, D = q32.shape
            L = kv32.shape[1]
            dh = D // H
            qh = (q32 @ Wq.T + bq).view(B, N, H, dh).transpose(1, 2)        # B H N dh
            kh = (kv32 @ Wk.T + bk).view(B, L, H, dh).transpose(1, 2)       # B H L dh
            vh = (kv32 @ Wv.T + bv).view(B, L, H, dh).transpose(1, 2)
            P = torch.softmax(qh @ kh.transpose(-1, -2) / dh ** 0.5, -1)    # B H N L
            Wo, bo = mha.out_proj.weight.float(), mha.out_proj.bias.float()
            # 게이트 (N, D) — forward 와 같은 규칙
            gl = mod.gate[min(li, mod.gate.shape[0] - 1)].float()
            g = q32.new_zeros(N, D)
            prev = 0
            for si, end in enumerate(seg_bounds + [N]):
                end = min(int(end), N)
                if end > prev:
                    g[prev:end] = (mod.gate_sep[li][mod.sep_gate_segs.index(si)].float()
                                   if si in getattr(mod, "sep_gate_segs", []) else gl[si])
                prev = end
            # 쿼리 구간: det [0,900) map [900,1000) plan [1000,N-1) ego [N-1]  (query_select det,map,plan,ego; ego 1 개)
            segs = {"det": (0, seg_bounds[0]), "map": (seg_bounds[0], seg_bounds[1]),
                    "plan": (seg_bounds[1], N - 1), "ego": (N - 1, N)}
            qn = q32.norm(dim=-1)                                            # B N
            A_tot = torch.zeros_like(q32)
            t0 = qt[0]
            for fi, f in enumerate(FOLDERS):
                sel = (t0 == fi)
                nf = int(sel.sum())
                if nf == 0:
                    continue
                Pf = P[..., sel]                                             # B H N Lf
                mass = Pf.sum(-1).mean(1)                                    # B N (머리 평균)
                ctx = (Pf @ vh[:, :, sel]).transpose(1, 2).reshape(B, N, D)  # B N D
                Af = ctx @ Wo.T
                A_tot += Af
                rel = (g * Af).norm(dim=-1) / qn                             # B N
                for sname, (s0, s1) in segs.items():
                    if s1 <= s0:
                        continue
                    acc.add((li, sname, f, "mass"), mass[:, s0:s1].mean())
                    acc.add((li, sname, f, "rel"), rel[:, s0:s1].mean())
                    acc.add((li, sname, f, "max_mass"), mass[:, s0:s1].max())
                info_once.setdefault(f"L_{f}", nf)
            A_tot = A_tot + bo
            gA = g * A_tot
            rec = (query.float() + gA - output.float()).norm(dim=-1) / gA.norm(dim=-1).clamp_min(1e-12)
            for sname, (s0, s1) in segs.items():
                if s1 <= s0:
                    continue
                acc.add((li, sname, "ALL", "rel"), (gA.norm(dim=-1) / qn)[:, s0:s1].mean())
                acc.add((li, sname, "ALL", "recon_err"), rec[:, s0:s1].mean())
                acc.add((li, sname, "ALL", "qnorm"), qn[:, s0:s1].mean())
                acc.add((li, sname, "ALL", "bias_rel"), ((g * bo).norm(dim=-1) / qn)[:, s0:s1].mean())
            info_once.setdefault("N", N); info_once.setdefault("L", L)

    h = qta.register_forward_hook(hook, with_kwargs=True)
    t0 = time.time()
    n = 0
    with torch.no_grad():
        for data in loader:
            model(return_loss=False, rescale=True, **data)
            n += 1
            if n % 100 == 0:
                print(f"[usage] {n}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
    h.remove()

    # 게이트 통계
    gates = {}
    G = qta.gate.detach().float()
    for li in range(G.shape[0]):
        for si, sname in enumerate(["det", "map", "plan_ego"]):
            gg = G[li, si]
            gates[f"{li}/{sname}"] = dict(mean_abs=float(gg.abs().mean()), max_abs=float(gg.abs().max()),
                                          l2=float(gg.norm()))
    if getattr(qta, "sep_gate_segs", None):
        gates["sep_gate_segs"] = list(qta.sep_gate_segs)
    tte = getattr(qta, "token_type_embed", None)
    out = dict(config=a.config, checkpoint=a.checkpoint, frames=len(idx), scenes=len(chosen),
               scene_tokens=[s for s, _ in chosen], info=info_once, gates=gates,
               type_embed_norm=None if tte is None else [float(x) for x in tte.detach().float().norm(dim=-1)],
               stats={"|".join(map(str, k)): v for k, v in acc.mean().items()})
    os.makedirs(osp.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[usage] 저장 {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("fork")
    main()
