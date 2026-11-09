# 수정(2026-10-19): 사용자 결정 — 언어 토큰 자리에 LiDAR/레이더 토큰만 바꿔 넣는 비교(인코더 함께 학습)
#   CPU 전용 시험(GPU·학습 없음). 사용(저장소 루트):
#     OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=. CUDA_VISIBLE_DEVICES="" python tools/verify_v2/test_sensor_tokens_cpu.py [--no-full]
#   (1) config 8 개 Config.fromfile + 핵심 값(검출기 type, 사영 입력 1536→256, 유형 1 종, operation_order 의 qwen_attn 유지,
#       교사 config 와 같은 optimizer·runner·fp16·배치·스트림 그룹).
#   (2) SensorBEVTokenizer(LiDAR·레이더, config 그대로 빌드): 가짜 점 → 토큰 (B, 2304, 256), 칸 크기 2.133 m·칸 중심 좌표,
#       위치 임베딩(행마다 다름·배치 공통·이웃 칸이 먼 칸보다 가까움), 점 위치 → 토큰 번호 대응(행 우선 iy*48+ix),
#       격자 끝 점(±51.2 m 직전) 처리, 모든 인코더 파라미터에 기울기.
#   (3) 토큰 → 실제 QwenTokenAttn(config 로 빌드한 모델의 모듈): 게이트 0 이면 출력 = 쿼리(카메라 단독과 같음)·게이트에만 기울기,
#       게이트 ≠0 이면 교차 어텐션을 거쳐 인코더까지 기울기.
#   (4) 전체 모델(실제 val 표본 1 개, NAS 데이터): 학습 모드 forward_train(배치 1) — 손실 유한, 역전파 후 인코더 기울기
#       (게이트 0: 0 / 게이트 무작위: ≠0), 평가 모드 simple_test — 6 계층 모두 2,304 토큰으로 융합 호출.
#   flash-attn·deformable aggregation 은 CUDA 전용이라 test_teacher_cache_cpu.py 와 같은 torch 기준 구현으로 바꿔 돌린다.
import argparse
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch
import torch.nn.functional as F

_orig_load = torch.load
def _load(*a, **kw):  # tools/train.py 와 같은 호환 shim
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _load

import projects.mmdet3d_plugin  # noqa: F401  레지스트리
from mmcv import Config
from mmcv.parallel import collate, DataContainer as DC
from mmdet.models import build_backbone, build_detector

from projects.mmdet3d_plugin.models.decoder.temporal_self_attn import flash_attention as fa

CFG = {(m, s, k): f"configs/nuscenes/token_{m}/stage{s}/{k}.py" for m in ("lidar", "radar") for s in (1, 2) for k in ("train", "test")}
TEACHER = {(s, k): f"configs/nuscenes/v2_teacher/stage{s}/{k}.py" for s in (1, 2) for k in ("train", "test")}
TEACHER_L = 1832 + 168 + 40 + 240 + 68 + 13 + 8   # 교사 캐시 _meta.json 의 폴더별 L 합(2026-10-19 확인)
FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}", flush=True)
    if not cond:
        FAIL.append(name)


def flash_ref_forward(self, q, kv, causal=False, key_padding_mask=None):
    """FlashAttention.forward 와 같은 수식(softmax(QK^T/sqrt(d))V). fp32 계산."""
    assert not causal
    k, v = kv.unbind(2)
    q_, k_, v_ = (x.transpose(1, 2).float() for x in (q, k, v))
    mask = None if key_padding_mask is None else key_padding_mask[:, None, None, :].bool()
    p = self.dropout_p if self.training else 0.0
    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask, dropout_p=p, scale=self.softmax_scale)
    return out.transpose(1, 2).to(q.dtype), None


def daf_torch(mc_ms_feat, spatial_shape, scale_start_index, loc, weights):
    bs, A, P, Ncam, _ = loc.shape
    L, G = weights.shape[-2:]
    C = mc_ms_feat.shape[-1]
    out = mc_ms_feat.new_zeros(bs, A, C)
    ss = spatial_shape.tolist(); st = scale_start_index.tolist()
    for cam in range(Ncam):
        lw, lh = loc[:, :, :, cam, 0], loc[:, :, :, cam, 1]
        valid = ((lw > 0) & (lw < 1) & (lh > 0) & (lh < 1)).to(mc_ms_feat.dtype)
        grid = torch.stack([lw * 2 - 1, lh * 2 - 1], -1)
        for l in range(L):
            h, w = ss[cam][l]; s0 = st[cam][l]
            feat = mc_ms_feat[:, s0:s0 + h * w].reshape(bs, h, w, C).permute(0, 3, 1, 2)
            smp = F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            wgt = weights[:, :, :, cam, l, :] * valid[..., None]
            wgt = wgt.repeat_interleave(C // G, dim=-1).permute(0, 3, 1, 2)
            out = out + (smp * wgt).sum(-1).permute(0, 2, 1)
    return out


fa.FlashAttention.forward = flash_ref_forward
sys.modules["projects.mmdet3d_plugin.models.decoder.deformable_aggregation.aggregation"].DAF = daf_torch


def fake_points(m, B, n, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(B):
        xy = (torch.rand(n, 2, generator=g) * 2 - 1) * 51.19
        z = torch.rand(n, 1, generator=g) * 6 - 4
        D = 5 if m == "lidar" else 7
        extra = torch.randn(n, D - 3, generator=g)
        out.append(torch.cat([xy, z, extra], 1))
    return out


# ---------------- (1) config ----------------
def test_configs():
    print("(1) config 8 개", flush=True)
    for (m, s, k), p in CFG.items():
        c = Config.fromfile(p)
        t = Config.fromfile(TEACHER[(s, k)])
        qa, qt = c.model.head.onedecoder_head.qwen_token_attn, t.model.head.onedecoder_head.qwen_token_attn
        oo = c.model.head.onedecoder_head.operation_order
        same = all(c.get(x) == t.get(x) for x in ("optimizer", "optimizer_config", "lr_config", "runner", "fp16",
                                                     "checkpoint_config", "evaluation", "num_iters_per_epoch"))
        same &= c.data.samples_per_gpu == t.data.samples_per_gpu and c.data.num_stream_groups == t.data.num_stream_groups
        same &= oo == t.model.head.onedecoder_head.operation_order and oo.count("qwen_attn") == 6
        same &= {kk: v for kk, v in qa.items() if kk not in ("qwen_hidden_dim", "num_token_types", "token_type_names")} == \
                {kk: v for kk, v in qt.items() if kk not in ("qwen_hidden_dim", "num_token_types", "token_type_names")}
        ok = (c.model.type == "SensorTokenSparseDetector" and qa.qwen_hidden_dim == 256 and qa.num_token_types == 1
              and c.model.qwen_cache_dir is None and c.run_name == f"hipad_nusc_token_{m}_stage{s}")
        if s == 2:
            ok &= c.load_from == f"./results/hipad_nusc_token_{m}_stage1/latest.pth"
        check(f"{p}", ok and same, f"run={c.run_name}, lr={c.optimizer.lr}, batch={c.data.samples_per_gpu}×{c.optimizer_config.cumulative_iters}, "
                                   f"max_iters={c.runner.max_iters}, fp16={dict(c.fp16)}")


# ---------------- (2) 토큰 만들기 ----------------
def test_tokenizer(m):
    print(f"(2) SensorBEVTokenizer — {m}", flush=True)
    cfg = Config.fromfile(CFG[(m, 1, "train")])
    torch.manual_seed(0)
    tk = build_backbone(cfg.model.sensor_token_encoder)
    tk.train()
    n_param = sum(p.numel() for p in tk.parameters())
    B = 2
    pts = fake_points(m, B, 20000 if m == "lidar" else 1500, seed=1)
    # 격자 끝(±51.2 m 직전) 점: fp32 에서 마지막 칸 안으로 들어가야 한다
    edge = torch.tensor([[51.2 - 1e-4, 51.2 - 1e-4, 0.0], [-51.2, -51.2, 0.0]])
    pts[0] = torch.cat([pts[0], torch.cat([edge, torch.zeros(2, pts[0].shape[1] - 3)], 1)], 0)
    tok = tk(pts)
    check("토큰 형상 (B, 48·48=2304, 256)", tuple(tok.shape) == (B, 2304, 256), f"{tuple(tok.shape)}, 교사 {TEACHER_L} 대비 {2304 / TEACHER_L - 1:+.1%}")
    check("칸 크기 2.133 m", abs(tk.cell_size[0] - 102.4 / 48) < 1e-9 and abs(tk.cell_size[1] - 102.4 / 48) < 1e-9, f"{tk.cell_size}")
    cm = tk.cell_centers_metric()
    check("칸 중심: 첫 토큰 (-50.13,-50.13), 둘째 (-48.0,-50.13)(x 가 빠른 축), 마지막 (50.13,50.13)",
          torch.allclose(cm[0], torch.tensor([-51.2 + 1.0667, -51.2 + 1.0667]), atol=1e-3)
          and torch.allclose(cm[1], torch.tensor([-51.2 + 3 * 1.0667, -51.2 + 1.0667]), atol=1e-3)
          and torch.allclose(cm[-1], torch.tensor([51.2 - 1.0667, 51.2 - 1.0667]), atol=1e-3),
          f"{cm[0].tolist()}, {cm[1].tolist()}, {cm[-1].tolist()}")
    pe = tk.pos_embed().detach()
    d = torch.cdist(pe, pe)
    d.fill_diagonal_(float("inf"))
    check("위치 임베딩: 2304 행 모두 서로 다름", float(d.min()) > 1e-6, f"최소 행 간 거리 {float(d.min()):.3e}")
    d_near = float((pe[0] - pe[1]).norm()); d_far = float((pe[0] - pe[2303]).norm())
    check("위치 임베딩: 이웃 칸이 반대편 칸보다 가깝다", d_near < d_far, f"이웃 {d_near:.3f} < 반대편 {d_far:.3f}")
    # 점 위치 → 토큰 번호: 점 하나만 있는 표본에서 캔버스의 기둥 위치가 토큰 칸 (iy//r, ix//r) 와 같다
    enc = tk.bev_encoder
    x, y = 12.3, -30.7
    p1 = torch.zeros(1, pts[0].shape[1]); p1[0, 0], p1[0, 1] = x, y
    with torch.no_grad():
        enc.eval()   # 점 1 개라 BatchNorm1d 학습 모드(배치 통계)를 쓸 수 없다 — 위치 대응만 본다
        canvas = enc.scatter_canvas([p1])
        enc.train()
    nz = torch.nonzero(canvas[0].abs().sum(0))
    r = enc.H // 48
    iy, ix = int(nz[0, 0]) // r, int(nz[0, 1]) // r
    exp_ix, exp_iy = int((x + 51.2) // (102.4 / 48)), int((y + 51.2) // (102.4 / 48))
    k = iy * 48 + ix
    check("점 (12.3, -30.7) → 토큰 칸 (iy,ix) 와 칸 중심이 점에서 칸 절반 이내", (iy, ix) == (exp_iy, exp_ix)
          and abs(float(cm[k, 0]) - x) <= 1.0667 + 1e-3 and abs(float(cm[k, 1]) - y) <= 1.0667 + 1e-3,
          f"기둥 1 개 → 칸 ({iy},{ix}), 토큰 번호 {k}, 칸 중심 {cm[k].tolist()}")
    loss = (tok * torch.randn_like(tok)).sum()
    loss.backward()
    nograd = [n for n, p in tk.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0]
    check(f"모든 인코더 파라미터({n_param / 1e6:.2f} M)에 기울기", not nograd, f"없음: {nograd[:5]}")
    return tk


# ---------------- (3) 토큰 → QwenTokenAttn ----------------
def test_attn(m):
    print(f"(3) 토큰 → QwenTokenAttn(config 모델의 실제 모듈) — {m}", flush=True)
    cfg = Config.fromfile(CFG[(m, 1, "train")])
    torch.manual_seed(0)
    det = build_detector(cfg.model)
    det.init_weights()
    det.train()
    qta = det.head.onedecoder_head.qwen_token_attn
    check("사영 첫 층 Linear(256→512), 유형 임베딩 (1,256) 0", tuple(qta.qwen_token_proj[0].weight.shape) == (512, 256)
          and tuple(qta.token_type_embed.shape) == (1, 256) and float(qta.token_type_embed.abs().sum()) == 0)
    B, N = 2, 1000 + 6 * 7  # det 900 + map 100 + 나머지(plan/ego) 구간
    query = torch.randn(B, N, 256)
    data = {det.sensor_token_encoder.points_key: fake_points(m, B, 5000 if m == "lidar" else 800, seed=3)}
    data = det._maybe_load_qwen_cache(data)
    check("qwen_hidden (B,2304,256) fp32 / qwen_mask 전부 1 / qwen_type 전부 0",
          tuple(data["qwen_hidden"].shape) == (B, 2304, 256) and data["qwen_hidden"].dtype == torch.float32
          and bool(data["qwen_mask"].bool().all()) and int(data["qwen_type"].abs().sum()) == 0)
    out = qta(query, data["qwen_hidden"], data["qwen_mask"], layer_idx=2, qwen_type=data["qwen_type"])
    check("게이트 0 → 출력 == 쿼리(step 0 은 카메라 단독과 같다)", torch.equal(out, query))
    out.pow(2).sum().backward()
    enc_g = sum(float(p.grad.abs().sum()) for p in det.sensor_token_encoder.parameters() if p.grad is not None)
    check("게이트 0 → 게이트에 기울기 ≠0, 인코더 기울기 = 0(교사 런의 사영 MLP 와 같은 출발)",
          float(qta.gate.grad.abs().sum()) > 0 and enc_g == 0.0, f"|∇gate|={float(qta.gate.grad.abs().sum()):.3e}, |∇enc|={enc_g}")
    det.zero_grad(set_to_none=True)
    with torch.no_grad():
        qta.gate.normal_(0, 0.1)
    data = det._maybe_load_qwen_cache({det.sensor_token_encoder.points_key: fake_points(m, B, 5000 if m == "lidar" else 800, seed=3)})
    out = qta(query, data["qwen_hidden"], data["qwen_mask"], layer_idx=2, qwen_type=data["qwen_type"])
    out.pow(2).sum().backward()
    nograd = [n for n, p in det.sensor_token_encoder.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0]
    check("게이트 ≠0 → 교차 어텐션을 거쳐 인코더 모든 파라미터에 기울기, 유형 임베딩에도",
          not nograd and float(qta.token_type_embed.grad.abs().sum()) > 0, f"없음: {nograd[:5]}")


# ---------------- (4) 전체 모델, 실제 표본 ----------------
def unwrap(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, DC):
            out[k] = v.data[0]
        elif isinstance(v, list) and v and isinstance(v[0], DC):
            out[k] = [x.data[0] for x in v]
        else:
            out[k] = v
    return out


def test_full(m):
    print(f"(4) 전체 모델(CPU, val 표본 1, 실제 영상·{m} 점) — {m}", flush=True)
    from mmdet.datasets import build_dataset
    cfg = Config.fromfile(CFG[(m, 1, "train")])
    torch.manual_seed(0)
    det = build_detector(cfg.model)
    det.init_weights()
    qta = det.head.onedecoder_head.qwen_token_attn
    key = det.sensor_token_encoder.points_key
    # 학습 파이프라인을 val info 에 적용(28k 학습 info 를 읽지 않으려고; 표본 형식 동일)
    tr = cfg.data.train.copy()
    tr.ann_file = "data/infos/nuscenes_fusion_infos_val.pkl"
    t0 = time.time()
    ds = build_dataset(tr)
    sample = ds[dict(idx=100, aug_config=ds.get_augmentation(), stream_group=0)]
    print(f"    표본 준비 {time.time() - t0:.1f} s", flush=True)
    d = unwrap(collate([sample], samples_per_gpu=1))
    npts = d[key][0].shape[0]
    check(f"학습 파이프라인 {key}: (N,{5 if m == 'lidar' else 7}) 텐서 리스트", isinstance(d[key], list)
          and d[key][0].shape[1] == (5 if m == "lidar" else 7) and npts > 0, f"N={npts}")
    img = d.pop("img")
    calls = {"n": 0, "L": set()}
    orig = qta.forward
    def spy(query, qh, *a, **kw):
        calls["n"] += 1; calls["L"].add(tuple(qh.shape))
        return orig(query, qh, *a, **kw)
    qta.forward = spy
    det.train()
    det.use_grid_mask = False   # GridMask 는 .cuda() 를 부르는 학습용 영상 가림(무작위) — CPU 시험에서만 끈다(토큰 경로와 무관)
    torch.Tensor.pin_memory = lambda self, *a, **k: self   # 학습용 카메라 가중치 dropout 이 pin_memory(CUDA) 를 부른다 — CPU 시험에서만 항등
    with torch.no_grad():
        qta.gate.normal_(0, 0.1)   # 1 step 뒤처럼 게이트가 움직인 상태(인코더까지 기울기 확인용)
    t0 = time.time()
    losses = det.forward_train(img, **d)
    total = sum(v for k, v in losses.items() if "loss" in k and torch.is_tensor(v))
    total.backward()
    print(f"    forward+backward {time.time() - t0:.1f} s", flush=True)
    bad = [k for k, v in losses.items() if torch.is_tensor(v) and not torch.isfinite(v).all()]
    check("학습 모드: 손실 모두 유한", not bad, f"total={float(total):.3f}, 항목 {len(losses)} 개, 비유한 {bad}")
    check("학습 모드: 6 계층 융합 호출, 토큰 (1,2304,256)", calls["n"] == 6 and calls["L"] == {(1, 2304, 256)}, str(calls))
    nograd = [n for n, p in det.sensor_token_encoder.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0]
    check("학습 모드(게이트 ≠0): 디코더 손실 → 인코더 모든 파라미터 기울기", not nograd, f"없음: {nograd[:5]}")
    # 평가 모드
    te = cfg.data.test.copy()
    det.eval(); calls["n"] = 0; calls["L"] = set()
    ds_t = build_dataset(Config.fromfile(CFG[(m, 1, "test")]).data.test)
    dt = unwrap(collate([ds_t[0]], samples_per_gpu=1))
    img_t = dt.pop("img")
    if isinstance(img_t, list):
        img_t = img_t[0]
    with torch.no_grad():
        out = det.simple_test(img_t, **dt)
    r = out[0]["img_bbox"]
    check("평가 모드 simple_test: 6 계층 융합 호출, 결과 키 생성", calls["n"] == 6 and calls["L"] == {(1, 2304, 256)} and len(r) > 0,
          f"결과 키 {sorted(r)[:6]}…")
    qta.forward = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-full", action="store_true", help="(4) 전체 모델 생략")
    a = ap.parse_args()
    assert not torch.cuda.is_available(), "CPU 전용 시험: CUDA_VISIBLE_DEVICES=\"\""
    test_configs()
    for m in ("lidar", "radar"):
        test_tokenizer(m)
        test_attn(m)
    if not a.no_full:
        for m in ("lidar", "radar"):
            test_full(m)
    print("결과:", "전부 통과" if not FAIL else f"실패 {len(FAIL)}: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
