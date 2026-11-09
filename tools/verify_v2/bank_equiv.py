# 수정(2026-10-15): 실험 C — 스트림 그룹 GPU 시험(GPU1 전용, 소량 메모리). 사용(저장소 루트):
#   CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tools/verify_v2/bank_equiv.py c <config> <out.pt> [steps]   # (c) G=1 기록(트리 간 비교용)
#   CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tools/verify_v2/bank_equiv.py b <v2 config>                 # (b) 그룹 k 뱅크 = 그룹 k 단독
#   python tools/verify_v2/bank_equiv.py cmp a.pt b.pt                                                      # (c) 비교
#   (c): 같은 시드·같은 표본으로 학습 모드 forward+backward 를 연속 프레임 steps 회(파라미터 고정, 옵티마이저 없음) 돌려
#        손실값·기울기·뱅크 상태를 저장한다. 정리 전 스냅숏 트리와 현재 트리에서 각각 돌려 비교(같은 트리 반복 = 잡음 기준).
#   (b): 드롭아웃 등 확률 연산을 끄기 위해 하위 모듈은 eval, 통합 디코더만 training=True 로 두고(그룹 분기만 학습 모드),
#        G=2·B=2 로 micro-batch 6 개(k=0,1,0,1,0,1)를 순서대로 처리한 뒤의 그룹별 뱅크 상태·출력이
#        같은 모델을 G=1 로 그룹 k 의 micro-batch 만 처리했을 때와 비트 단위로 같은지 본다.
import os, sys, random, hashlib
import numpy as np
import torch

_orig_load = torch.load
def _load(*a, **kw):  # tools/train.py 와 같은 호환 shim(로컬 신뢰 가중치의 레거시 .tar 포맷)
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _load

assert os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "0"), "GPU1 전용: CUDA_VISIBLE_DEVICES=1"


def daf_torch(mc_ms_feat, spatial_shape, scale_start_index, loc, weights):
    """메모리 절약: 기울기 필요 시 재계산(checkpoint) — 값은 같다."""
    if torch.is_grad_enabled() and (mc_ms_feat.requires_grad or loc.requires_grad or weights.requires_grad):
        from torch.utils.checkpoint import checkpoint
        return checkpoint(_daf_torch, mc_ms_feat, spatial_shape, scale_start_index, loc, weights, use_reentrant=False)
    return _daf_torch(mc_ms_feat, spatial_shape, scale_start_index, loc, weights)


def _daf_torch(mc_ms_feat, spatial_shape, scale_start_index, loc, weights):
    """시험 전용 결정적 deformable aggregation(순수 torch, grid_sample). CUDA 커널(atomicAdd 누적 → 실행마다 합산 순서가 달라
    비트 재현 불가)과 같은 수식: bilinear(align_corners=False, 범위 밖 0), loc∉(0,1) 는 기여 0, 그룹별 가중합.
    합산 순서만 달라 커널과 ~1e-6 상대 오차가 있을 수 있다(여기선 같은 트리 안의 비교에만 쓴다)."""
    import torch.nn.functional as F
    bs, A, P, Ncam, _ = loc.shape
    L, G = weights.shape[-2:]
    C = mc_ms_feat.shape[-1]
    out = mc_ms_feat.new_zeros(bs, A, C)
    ss = spatial_shape.tolist(); st = scale_start_index.tolist()
    for cam in range(Ncam):
        lw, lh = loc[:, :, :, cam, 0], loc[:, :, :, cam, 1]
        valid = ((lw > 0) & (lw < 1) & (lh > 0) & (lh < 1)).to(mc_ms_feat.dtype)  # [bs,A,P]
        grid = torch.stack([lw * 2 - 1, lh * 2 - 1], -1)  # [bs,A,P,2]
        for l in range(L):
            h, w = ss[cam][l]; s0 = st[cam][l]
            feat = mc_ms_feat[:, s0:s0 + h * w].reshape(bs, h, w, C).permute(0, 3, 1, 2)
            smp = F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)  # [bs,C,A,P]
            wgt = weights[:, :, :, cam, l, :] * valid[..., None]  # [bs,A,P,G]
            wgt = wgt.repeat_interleave(C // G, dim=-1).permute(0, 3, 1, 2)  # [bs,C,A,P]
            out = out + (smp * wgt).sum(-1).permute(0, 2, 1)
    return out


def setup(cfg_path, bs, G, workers=0, deterministic_daf=True):
    from mmcv import Config
    import importlib
    importlib.import_module("projects.mmdet3d_plugin")
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from projects.mmdet3d_plugin.models.loader.builder import build_dataloader
    cfg = Config.fromfile(cfg_path)
    cfg.data.samples_per_gpu = bs
    if G > 1:
        cfg.data.num_stream_groups = G
        cfg.model.head.onedecoder_head.num_stream_groups = G
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ds = build_dataset(cfg.data.train)
    kw = dict(num_stream_groups=G) if G > 1 else {}
    dl = build_dataloader(ds, bs, workers, 1, dist=False, seed=0, runner_type="IterBasedRunner", **kw)
    if deterministic_daf:
        agg = sys.modules['projects.mmdet3d_plugin.models.decoder.deformable_aggregation.aggregation']
        agg.DAF = daf_torch
    model = build_detector(cfg.model)
    model.init_weights()
    from mmcv.parallel import MMDataParallel
    model = MMDataParallel(model.cuda(), device_ids=[0])
    return cfg, dl, model


def dec_of(model):
    return model.module.head.onedecoder_head


def bank_snapshot(dec):
    out = {}
    for name in ("ego", "det", "map", "plan"):
        if name not in dec.query_select:
            continue
        b = getattr(dec, f"{name}_instance_bank")
        for k, v in vars(b).items():
            if k.startswith("_") or k == "training":
                continue
            if torch.is_tensor(v):
                out[f"{name}.{k}"] = v.detach().clone().cpu()
            elif k == "metas" and isinstance(v, dict):
                out[f"{name}.metas.timestamp"] = v["timestamp"].detach().clone().cpu()
                out[f"{name}.metas.tokens"] = tuple(m.get("token") for m in v["img_metas"])
            elif isinstance(v, (int, float)) or v is None:
                out[f"{name}.{k}"] = v
    return out


def run_c(cfg_path, out_path, steps):
    cfg, dl, model = setup(cfg_path, bs=2, G=1)
    model.train()
    it = iter(dl)
    rec = []
    for s in range(steps):
        data = next(it)
        torch.manual_seed(100 + s)
        model.zero_grad(set_to_none=True)
        outs = model.train_step(data, None)
        outs["loss"].backward()
        grads = {n: p.grad.detach().clone().cpu() for n, p in model.module.named_parameters() if p.grad is not None}
        rec.append(dict(log=dict(outs["log_vars"]), loss=outs["loss"].item(), grads=grads,
                        bank=bank_snapshot(dec_of(model)),
                        tokens=[m["token"] for m in data["img_metas"].data[0]]))
        print(s, "loss", f"{outs['loss'].item():.6f}", {k: round(v, 4) for k, v in outs["log_vars"].items() if k in ("det_loss_cls", "det_loss_box", "det_loss_cns", "map_loss_line")}, flush=True)
    torch.save(rec, out_path)
    print("peak mem GB", torch.cuda.max_memory_allocated() / 2**30)


def cmp(a_path, b_path):
    a, b = torch.load(a_path), torch.load(b_path)
    for s, (x, y) in enumerate(zip(a, b)):
        assert x["tokens"] == y["tokens"], "표본이 다르다"
        dl = max(abs(x["log"][k] - y["log"][k]) for k in x["log"])
        keys = set(x["log"]) ^ set(y["log"])
        gk = set(x["grads"]) ^ set(y["grads"])
        gd = max((x["grads"][k] - y["grads"][k]).abs().max().item() for k in x["grads"] if k in y["grads"])
        gn = max(x["grads"][k].abs().max().item() for k in x["grads"])
        bd = 0.0
        for k in x["bank"]:
            u, v = x["bank"][k], y["bank"].get(k)
            if torch.is_tensor(u):
                bd = max(bd, (u.double() - v.double()).abs().max().item() if u.numel() else 0.0)
            else:
                assert u == v, (k, u, v)
        print(f"step {s}: loss 차 max {dl:.3e} (loss {x['loss']:.6f} vs {y['loss']:.6f}, 키 차집합 {keys}), "
              f"grad 차 max {gd:.3e} (grad max {gn:.3e}, 키 차집합 {len(gk)}), bank 차 max {bd:.3e}")


def run_b(cfg_path):
    G, B, M = 2, 2, 6
    cfg, dl, model = setup(cfg_path, bs=B, G=G)
    it = iter(dl)
    batches = [next(it) for _ in range(M)]
    ks = [{m["stream_group"] for m in bt["img_metas"].data[0]} for bt in batches]
    print("micro-batch 그룹 태그", ks)
    from mmcv.parallel import scatter
    dec = dec_of(model)
    model.eval()
    dec.training = True  # 그룹 분기만 학습 모드(하위 연산은 eval → 확률 연산 없음)

    def fwd(bt):
        d = scatter(bt, [torch.cuda.current_device()])[0]
        img = d.pop("img")
        with torch.no_grad():
            m = model.module
            fm = m.extract_feat(img)
            d = m._maybe_load_qwen_cache(d)
            det, mp, ego, plan, mo = m.head(img, fm, d)
        return dict(det_cls=det["classification"][-1].cpu(), det_box=det["prediction"][-1].cpu(),
                    inst=det.get("instance_id").cpu() if det.get("instance_id") is not None else None,
                    map_pts=mp["prediction"][-1].cpu() if mp is not None and mp.get("prediction") else None)

    def reset_all():
        dec._stream_states = None
        for n in ("ego", "det", "map", "plan"):
            if n in dec.query_select:
                getattr(dec, f"{n}_instance_bank").reset()
                getattr(dec, f"{n}_instance_bank").mask = None  # 자차 뱅크 reset 은 mask 를 비우지 않는다(미사용 속성) — 비교 공정성

    bank_state_keys = sys.modules['projects.mmdet3d_plugin.models.decoder.temporal_bank.stream_state'].bank_state_keys

    def snap_bank():
        snap = {}
        for n in ("ego", "det", "map", "plan"):
            if n not in dec.query_select:
                continue
            bnk = getattr(dec, f"{n}_instance_bank")
            for kk in bank_state_keys(bnk):
                v = getattr(bnk, kk, None)
                if torch.is_tensor(v):
                    snap[f"{n}.{kk}"] = v.detach().clone().cpu()
                elif kk == "metas" and isinstance(v, dict):
                    snap[f"{n}.metas.ts"] = v["timestamp"].cpu().clone()
                    snap[f"{n}.metas.tok"] = tuple(m.get("token") for m in v["img_metas"])
                else:
                    snap[f"{n}.{kk}"] = v
        return snap

    def solo(k):
        reset_all(); dec.num_stream_groups = 1
        outs, states = {}, {}
        for t in range(k, M, G):
            outs[t] = fwd(batches[t]); states[t] = snap_bank()
        return outs, states

    def dmax(u, v):
        """(구조 일치 여부, 실수 텐서 최대 절대차, 정수 텐서 불일치 원소 수)"""
        if torch.is_tensor(u) != torch.is_tensor(v):
            return False, float("inf"), 0
        if not torch.is_tensor(u):
            return u == v, 0.0, 0
        if u.shape != v.shape or u.dtype != v.dtype:
            return False, float("inf"), 0
        if u.is_floating_point():
            return True, (u.double() - v.double()).abs().max().item() if u.numel() else 0.0, 0
        return True, 0.0, int((u != v).sum())

    def compare(oa, sa, ob, sb):
        st_ok, fmax, imis = True, 0.0, 0
        for kk in set(sa) | set(sb):
            a_, b_ = sa.get(kk, "MISSING"), sb.get(kk, "MISSING")
            if kk.endswith("metas.ts") or kk.endswith("metas.tok") or kk.endswith("prev_id"):
                same = (torch.equal(a_, b_) if torch.is_tensor(a_) and torch.is_tensor(b_) else a_ == b_)
                st_ok &= bool(same)
                continue
            ok_, f_, i_ = dmax(a_, b_)
            st_ok &= ok_; fmax = max(fmax, f_); imis += i_
        out_f = max(dmax(oa[x], ob[x])[1] for x in oa if oa[x] is not None and oa[x].is_floating_point())
        out_i = sum(dmax(oa[x], ob[x])[2] for x in oa if oa[x] is not None and not oa[x].is_floating_point())
        return st_ok, max(fmax, out_f), imis + out_i

    # A: G=2 로 6 개 순서대로. forward 직후 뱅크 속성 = 방금 저장된 그룹 k 상태(같은 참조)이므로 그대로 찍는다.
    reset_all(); dec.num_stream_groups = G
    A_out, A_state = [], []
    for t, bt in enumerate(batches):
        A_out.append(fwd(bt))
        A_state.append(snap_bank())
        k = t % G
        assert all(dec._stream_states[k][n][kk] is getattr(getattr(dec, f"{n}_instance_bank"), kk, None)
                   for n in dec._stream_states[k] for kk in dec._stream_states[k][n]), "저장본과 뱅크 속성이 다르다"
    ok = True
    for k in range(G):
        o1, s1 = solo(k)
        o2, s2 = solo(k)  # 같은 처리 반복 = GPU 잡음 기준(deformable 집약 forward 가 atomicAdd 라 비트 재현 불가 — SD 원본 커널과 같음)
        for t in range(k, M, G):
            st_a, fa, ia = compare(A_out[t], A_state[t], o1[t], s1[t])
            st_f, ff, if_ = compare(o2[t], s2[t], o1[t], s1[t])
            print(f"그룹 {k} micro-batch {t}: [A(G=2 교차) vs 단독] 구조·시각·토큰 일치={st_a} 최대차 {fa:.2e} 정수불일치 {ia} | "
                  f"[단독 반복 잡음] 일치={st_f} 최대차 {ff:.2e} 정수불일치 {if_}")
            ok &= st_a and fa <= max(10 * ff, 1e-4)
    # 대조: G=1 로 6 개를 순서대로(종전 방식 = 그룹 섞임) 처리 — 뱅크가 다른 그룹 프레임을 물려받아 크게 달라야 한다(시험의 검출력)
    reset_all(); dec.num_stream_groups = 1
    mixed_o, mixed_s = [], []
    for bt in batches:
        mixed_o.append(fwd(bt)); mixed_s.append(snap_bank())
    for t in (2, 4):
        st_m, fm_, im = compare(mixed_o[t], mixed_s[t], A_out[t], A_state[t])
        print(f"대조 micro-batch {t}: [G=1 섞어 처리 vs A] 구조·시각 일치={st_m} 최대차 {fm_:.2e} 정수불일치 {im} (크게 달라야 함)")
    print("peak mem GB", torch.cuda.max_memory_allocated() / 2**30)
    print("ALL OK" if ok else "FAIL")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "c":
        run_c(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 3)
    elif mode == "b":
        run_b(sys.argv[2])
    elif mode == "cmp":
        cmp(sys.argv[2], sys.argv[3])
