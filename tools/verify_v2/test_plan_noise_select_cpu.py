# 수정(2026-10-23): 사용자 결정 — 잡음 세기 확률 선택(①, t_sampling="gauss") + 지도 확신도 선택적 가우시안(②, selective=True) CPU 검사.
#   (1) 기본값(uniform, selective=False) = 10-19 모듈과 비트 동일(학습·평가, 지도 있음/없음)
#   (2) ① 학습 t 분포 = 이산 가우시안(중심 infer_t, σ t_std, 범위 0..trunc_steps−1), 표본 20만 개 경험 분포와 비교
#   (3) ② 평가: z 는 종전과 같은 값, 점수 1.0 차선 → 전 점 회전 타원(selective 끈 것과 동일), 점수 0.4 차선 → 회전 비율 ≈ 0.4,
#       같은 입력 두 번 → 비트 동일(고정 시드), 지도 없음 → 종전과 동일
#   (4) 기울기: 앵커로 가는 기울기 = sqrt(ᾱ_t) (선택 여부와 무관)
#   (5) config: 본 2단계(gauss·selective 켬)·비교 2단계(DiffusionDrive 원래 PlanTruncatedDiffusion) 빌드, 두 모듈 state_dict 키·모양 동일
#   사용: CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tools/verify_v2/test_plan_noise_select_cpu.py
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch

import projects.mmdet3d_plugin  # noqa: F401
import tools.plan_diffusion_map_register  # noqa: F401
from mmcv import Config
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from projects.mmdet3d_plugin.models.decoder.noise_t.diffusion import PlanTruncatedDiffusion
from projects.mmdet3d_plugin.models.decoder.noise_t.plan_diffusion_map import MapAlignedPlanTruncatedDiffusion

FAIL = []
KW = dict(embed_dims=256, ego_fut_ts=6, num_train_timesteps=1000, trunc_steps=50, infer_t=8, beta_start=1e-4,
          beta_end=0.02, norm_scale=(13.0, 29.0), norm_offset=(-1.0, 27.0), infer_seed=0)
NEW = dict(score_thr=0.4, lane_class_ids=(1, 2), boundary_class_ids=(2,), search_radius=3.5,
           boundary_margin=0.1, clip_to_boundary=True)


def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not cond:
        FAIL.append(name)


def anchors(B, N=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    step = torch.stack([torch.randn(B, N, 6, generator=g) * 0.3, torch.rand(B, N, 6, generator=g) * 4 + 0.5], -1)
    return step.cumsum(2).reshape(B, N, 12)


def make_map(lines, B=1, M=100, P=20, C=3):
    pts = torch.zeros(B, M, P, 2); sc = torch.zeros(B, M, C)
    for i, (c, s, xy) in enumerate(lines):
        pts[:, i] = torch.as_tensor(xy, dtype=torch.float32); sc[:, i, c] = s
    return dict(pts=pts, scores=sc)


def line(p0, p1, P=20):
    p0, p1 = np.asarray(p0, np.float32), np.asarray(p1, np.float32)
    return p0[None] + np.linspace(0, 1, P, dtype=np.float32)[:, None] * (p1 - p0)[None]


def build(**kw):
    torch.manual_seed(123)
    m = MapAlignedPlanTruncatedDiffusion(**KW, **NEW, **kw)
    with torch.no_grad():
        for p in m.t_mlp.parameters():
            p.normal_(0, 0.05)
    return m


def diag_map(score, B=2):
    # 비스듬한 차선(45°) 여러 줄 → 대부분 경유점이 반경 3.5 m 안에 차선을 가진다. 경계 없음(잘라내기 무관).
    return make_map([(1, score, line((-40 + 4 * k, 0), (10 + 4 * k, 50))) for k in range(20)], B=B)


def test_default_identity():
    print("(1) 기본값 = 종전 모듈 비트 동일", flush=True)
    ref, new = build(), build()   # new 는 기본값(uniform, selective=False) — 10-19 동작 그대로여야 한다
    new.load_state_dict(ref.state_dict())
    A = anchors(2)
    for mode in ("train", "eval"):
        getattr(ref, mode)(); getattr(new, mode)()
        ok = []
        for mm in (None, diag_map(0.9)):
            torch.manual_seed(5); xa, ea = ref(A, map_memory=mm)
            torch.manual_seed(5); xb, eb = new(A, map_memory=mm)
            ok.append(torch.equal(xa, xb) and torch.equal(ea, eb))
        check(f"{mode}: 지도 없음/있음 출력 비트 동일", all(ok), str(ok))
    old = PlanTruncatedDiffusion(**KW); old.load_state_dict(new.state_dict(), strict=True)
    old.train(); new.train()
    torch.manual_seed(9); xo, _ = old(A)
    torch.manual_seed(9); xn, _ = new(A, map_memory=None)
    check("학습·지도 없음: 부모 PlanTruncatedDiffusion 과 비트 동일(t 추출 난수 소비 동일)", torch.equal(xo, xn))


def test_t_dist():
    print("(2) ① 학습 t 분포", flush=True)
    m = build(t_sampling="gauss", t_std=8.0)
    p = m.t_probs.double()
    ts = torch.arange(50, dtype=torch.float64)
    w = torch.exp(-0.5 * ((ts - 8) / 8) ** 2); w /= w.sum()
    check("t_probs = 정규화된 이산 가우시안(중심 8, σ 8, 0..49)", torch.allclose(p, w, atol=1e-7), f"합 {p.sum():.6f}")
    torch.manual_seed(0)
    n = 200000
    t = m._sample_t(n, "cpu")
    check("t 범위 0..49", int(t.min()) >= 0 and int(t.max()) <= 49, f"{int(t.min())}..{int(t.max())}")
    h = torch.bincount(t, minlength=50).double() / n
    tv = 0.5 * (h - p).abs().sum().item()
    check("경험 분포 ≈ 이론 분포(총변동 거리 < 0.01)", tv < 0.01, f"TV {tv:.4f}")
    check("t ≥ 10 비율 ≈ 0.497(균등 0.8)", abs(h[10:].sum().item() - p[10:].sum().item()) < 0.01,
          f"경험 {h[10:].sum():.3f} / 이론 {p[10:].sum():.3f}")
    check("t_probs 는 비저장 버퍼(state_dict 불변)", "t_probs" not in m.state_dict())
    m2 = build(t_sampling="gauss")
    check("t_std 기본값 = infer_t", m2.t_std == 8.0)
    try:
        build(t_sampling="lognormal"); check("알 수 없는 t_sampling 거부", False)
    except AssertionError:
        check("알 수 없는 t_sampling 거부(AssertionError)", True)


def test_selective():
    print("(3) ② 선택적 가우시안(평가, 고정 시드)", flush=True)
    A = anchors(2, seed=3)
    base = build(); base.eval()
    sel = build(selective=True); sel.load_state_dict(base.state_dict()); sel.eval()
    x0, _ = base(A, map_memory=None); x1, _ = sel(A, map_memory=None)
    check("지도 없음 → 선택 켜도 종전과 비트 동일", torch.equal(x0, x1))
    full_b, _ = base(A, map_memory=diag_map(1.0)); full_s, _ = sel(A, map_memory=diag_map(1.0))
    check("점수 1.0 차선 → 모든 점 회전 타원(선택 끈 것과 비트 동일)", torch.equal(full_b, full_s))
    iso, _ = base(A, map_memory=None)
    mm = diag_map(0.4)
    al_b, _ = base(A, map_memory=mm); al_s, _ = sel(A, map_memory=mm)
    rot = ~torch.isclose(al_b, iso).all(-1) if False else None
    B, N = A.shape[:2]
    pb, ps, pi = (x.reshape(B, N, 6, 2) for x in (al_b, al_s, iso))
    changed = ~(pb == pi).all(-1)                      # 선택 없이 회전된 점(차선 반경 안)
    same_as_rot = (ps == pb).all(-1) & changed
    same_as_iso = (ps == pi).all(-1) & changed
    frac = same_as_rot.sum().item() / max(changed.sum().item(), 1)
    check("점수 0.4 → 회전된 점 비율 ≈ 0.4", abs(frac - 0.4) < 0.05, f"{frac:.3f} ({changed.sum().item()} 점)")
    check("회전 안 된 점 = 종전 전방축 잡음(eps = z) 그대로", bool((same_as_rot | same_as_iso)[changed].all()))
    al_s2, _ = sel(A, map_memory=mm)
    check("같은 입력 두 번 → 비트 동일(고정 시드)", torch.equal(al_s, al_s2))
    # 학습 모드: 선택 비율
    sel.train(); base.train()
    torch.manual_seed(1); xs, _ = sel(A, map_memory=mm)
    check("학습 모드 출력 유한", bool(torch.isfinite(xs).all()))


def test_grad():
    print("(4) 기울기 경로", flush=True)
    m = build(t_sampling="gauss", selective=True); m.eval()
    A = anchors(1, seed=4).requires_grad_(True)
    x, _ = m(A, map_memory=make_map([(1, 0.7, line((-40 + 4 * k, 0), (10 + 4 * k, 50))) for k in range(20)]))
    x.sum().backward()
    want = m.sqrt_ab[m.infer_t].item()
    # 경계 없음 → 잘라내기 없음 → 앵커 기울기는 모든 성분에서 sqrt(ᾱ_t)
    check("앵커 기울기 = sqrt(ᾱ_t)", torch.allclose(A.grad, torch.full_like(A.grad, want), atol=1e-6), f"{want:.6f}")


def test_configs():
    print("(5) config", flush=True)
    from projects.mmdet3d_plugin.models.decoder.noise_t.plan_diffusion_map import MapAlignedPlanTruncatedDiffusion as M
    mods = {}
    for f in ("configs/nuscenes/v2_teacher/stage2/train.py", "configs/nuscenes/v2_teacher/stage2/test.py",
              "configs/nuscenes/v2_teacher/stage2/train_noise_dd.py", "configs/nuscenes/v2_teacher/stage2/test_noise_dd.py"):
        c = Config.fromfile(f)
        pd = c.model.head.onedecoder_head.plan_diffusion
        mods[f] = build_from_cfg(pd, PLUGIN_LAYERS)
        print(f"    {f}: {pd['type']} {dict((k, pd[k]) for k in ('t_sampling', 't_std', 'selective') if k in pd)}")
    a, b = mods["configs/nuscenes/v2_teacher/stage2/train.py"], mods["configs/nuscenes/v2_teacher/stage2/train_noise_dd.py"]
    check("본 2단계 = MapAligned + gauss(σ 8) + selective", isinstance(a, M) and a.t_sampling == "gauss" and a.t_std == 8.0 and a.selective)
    check("비교 2단계 = DiffusionDrive 원래(PlanTruncatedDiffusion, 지도 미사용)", type(b) is PlanTruncatedDiffusion
          and not getattr(b, "uses_map_memory", False))
    same = all(type(m) is type(mods[k.replace("test", "train")]) for k, m in mods.items())
    check("test config 모듈 = train config 모듈", same)
    for k in ("trunc_steps", "infer_t", "infer_seed"):
        check(f"두 config 의 {k} 같음", getattr(a, k) == getattr(b, k), f"{getattr(a, k)} / {getattr(b, k)}")
    check("두 config 의 정규화·일정 버퍼 같음", torch.equal(a.norm_scale, b.norm_scale) and torch.equal(a.norm_offset, b.norm_offset)
          and torch.equal(a.sqrt_ab, b.sqrt_ab))
    sa, sb = a.state_dict(), b.state_dict()
    check("두 모듈 state_dict 키·모양 동일(1단계 체크포인트 그대로 로드)",
          sa.keys() == sb.keys() and all(sa[k].shape == sb[k].shape for k in sa))
    c = Config.fromfile("configs/nuscenes/v2_teacher/stage2/train_noise_dd.py")
    c0 = Config.fromfile("configs/nuscenes/v2_teacher/stage2/train.py")
    d0, d1 = c0.pretty_text.splitlines(), c.pretty_text.splitlines()
    import difflib
    diff = [l for l in difflib.unified_diff(d0, d1, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    check("본·비교 config 차이는 plan_diffusion 블록·run_name 뿐", all(any(s in l for s in ("plan_diffusion", "type=", "score_thr", "boundary",
          "search_radius", "lane_class_ids", "t_sampling", "t_std", "selective", "embed_dims", "num_train_timesteps", "norm_",
          "infer_", "beta_", "ego_fut_ts", "trunc_steps", "clip_to_boundary", "#")) for l in diff
          if not l[1:].startswith("run_name")), "\n".join(diff[:20]))   # 수정(2026-10-23): 검사 1차 D1 — 비교 런 이름은 일부러 다름
    check("비교 config run_name = hipad_nusc_laq_v2_teacher_stage2_noise_dd", c.run_name == "hipad_nusc_laq_v2_teacher_stage2_noise_dd", c.run_name)
    check("load_from 같음(같은 1단계 체크포인트)", c.load_from == c0.load_from, c.load_from)


def main():
    assert not torch.cuda.is_available(), "CPU 전용 시험: CUDA_VISIBLE_DEVICES=\"\""
    test_default_identity(); test_t_dist(); test_selective(); test_grad(); test_configs()
    print("결과:", "모두 통과" if not FAIL else f"실패 {len(FAIL)}: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
