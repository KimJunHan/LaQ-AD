# 수정(2026-10-19): 사용자 결정 — 지도 정렬 선택적 Gaussian 잡음(지도 밖 잡음 제거)
#   CPU 전용 시험(GPU·학습 없음). 사용(저장소 루트):
#     OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=. CUDA_VISIBLE_DEVICES="" python tools/verify_v2/test_plan_diffusion_map_cpu.py [--no-full]
#   (1) 지도 없음(None / 점수 미달 / 반경 밖) → 종전 PlanTruncatedDiffusion 과 비트 동일(학습·평가, 같은 시드).
#   (2) 수정(2026-10-19): 사용자 결정 — 기존 잡음 타원을 차선 방향으로 회전(새 비율 값 없음): (2a) 결정적 — 전방축 차선 Σ_n = I 비트 정확, 30/45/90/135° 에서 m 좌표 고유방향 = (u, n)·σ = (29, 13) m,
#       eps = chol(Σ_n) z. (2b) 모듈 전체 — 전방축 차선은 같은 시드에서 등방(종전)과 비트 동일, 기운 차선 통계, 반경 밖 비트 동일.
#   (3) 경계: 잡음 변위가 경계를 넘는 경유점이 하나도 없음, 넘지 않는 경유점은 잘라내기 끈 결과와 비트 동일,
#       잘린 점 = 평균점 + f·변위(교차점에서 0.1 m 안쪽). 꺾인 경계(여러 선분)에서도.
#   (4) 평가 결정성(고정 시드), 배치 독립(다른 표본의 지도가 바뀌어도 내 출력 불변), 기울기(t_mlp·앵커 = 종전과 같음),
#       fp16 지도 입력·autocast 안에서도 fp32 로 동작.
#   (5) PlanMapMemoryInstanceBank: 지도 과제 반환값·상태가 InstanceBank 와 같음, 직전 예측을 자차 이동으로 옮김,
#       시간 차 > 2 s / 다른 장면 → 지도 없음, 스트림 그룹 저장 키에 기억 포함, get() 의 reset 이 기억을 지우지 않음.
#   (6) config: 2단계 6 개 Config.fromfile(새 type·등록 모듈·종전 확산 값 유지), 1단계 config 파일 해시 불변(작업 전 해시와 비교).
#   (7) 전체 모델(CPU, token_lidar 2단계 config = v2_teacher 2단계 상속, 실제 val 연속 2 프레임): 첫 프레임 지도 기억 없음 →
#       둘째 프레임 디코더가 지도 기억을 잡음 모듈에 넘김, 손실 유한·역전파, 평가 모드 2 프레임.
import argparse
import hashlib
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch

_orig_load = torch.load
def _load(*a, **kw):  # tools/train.py 와 같은 호환 shim
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _load

import projects.mmdet3d_plugin  # noqa: F401  레지스트리
import tools.plan_diffusion_map_register  # noqa: F401  새 모듈 등록(config custom_imports 와 같은 이름)
from mmcv import Config
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

from projects.mmdet3d_plugin.models.decoder.noise_t.diffusion import PlanTruncatedDiffusion
from projects.mmdet3d_plugin.models.decoder.noise_t.plan_diffusion_map import MapAlignedPlanTruncatedDiffusion
from projects.mmdet3d_plugin.models.decoder.temporal_bank.instance_bank import InstanceBank
from projects.mmdet3d_plugin.models.decoder.temporal_bank.plan_map_memory_bank import PlanMapMemoryInstanceBank
from projects.mmdet3d_plugin.models.decoder.temporal_bank.stream_state import bank_state_keys

FAIL = []
KW = dict(embed_dims=256, ego_fut_ts=6, num_train_timesteps=1000, trunc_steps=50, infer_t=8, beta_start=1e-4,
          beta_end=0.02, norm_scale=(13.0, 29.0), norm_offset=(-1.0, 27.0), infer_seed=0)
# 수정(2026-10-19): 사용자 결정 — 기존 잡음 타원을 차선 방향으로 회전(새 비율 값 없음) — anisotropy_ratio 제거
NEW = dict(score_thr=0.4, lane_class_ids=(1, 2), boundary_class_ids=(2,), search_radius=3.5,
           boundary_margin=0.1, clip_to_boundary=True)

# 작업 시작 전(2026-10-19) 1단계 config 해시 — 이번 변경이 1단계를 건드리지 않았음을 확인
STAGE1_SHA256 = {
    # 수정(2026-10-26): 사용자 지시 소스 정리 — diag_fp16·fusion_lidar·fusion_radar·stage1/test_v2.py 는 archive/cleanup_20261026 로 옮겨 감시 목록에서 뺀다.
    "configs/nuscenes/stage1/_base_common.py": "ecf9bb452a83429d1563609b87337a536c5e660f9a0dab4ddb6f5afbb45a938f",
    "configs/nuscenes/stage1/train_v2.py": "49e468c8033cbbe5b2c9ca04ace60f847d90cb5d2b29641cac4e3dc7136d1959",  # 수정(2026-10-23): 10-20 FPN 수정(bf16 neck·img_neck lr ×0.5, 사용자 승인) 뒤 해시 — 현재 1단계 학습(10-20 22:04 시작)이 읽은 파일
    "configs/nuscenes/token_lidar/stage1/test.py": "c59385d436be8e075f0d5d52b00cd0188376b820744b23613071bc8cea095adf",
    "configs/nuscenes/token_lidar/stage1/train.py": "df2ab94aa2aa820a56ac511dd27dd8409b67c14a9a9f0fb25e066768c782a282",
    "configs/nuscenes/token_radar/stage1/test.py": "c59385d436be8e075f0d5d52b00cd0188376b820744b23613071bc8cea095adf",
    "configs/nuscenes/token_radar/stage1/train.py": "3c969a3f45d3b10e54952b26eeeefebaec231ca1de6e1b117a987eff8c18750f",
    "configs/nuscenes/v2_teacher/stage1/test.py": "aca0f81ae74ce7d15df5a06d76d66c346022d28892c7a3780b6a8fe92aaa34a5",
    # 수정(2026-10-19): 사용자 결정 — 기준 해시 갱신. 종전 3ea20fe2…(= HiP-AD/backup/configs_pre_v2_sync_20261016/configs_full_before/
    #   nuscenes/v2_teacher/stage1/train.py 와 같은 해시) 대비 차이는 오늘 15:12 의 주석 2 줄(1·11행, GRPO 제외 서술)뿐임을 diff 로 확인.
    "configs/nuscenes/v2_teacher/stage1/train.py": "227575936386bac0269b31b5c92f4e243b66b6536a19e3c850935ce1dade1342",
}
STAGE2 = [f"configs/nuscenes/{e}/stage2/{k}.py" for e in ("v2_teacher", "token_lidar", "token_radar") for k in ("train", "test")]


def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}", flush=True)
    if not cond:
        FAIL.append(name)


def pair(**new):
    torch.manual_seed(123)
    old = PlanTruncatedDiffusion(**KW)
    kw = dict(NEW); kw.update(new)
    nm = MapAlignedPlanTruncatedDiffusion(**KW, **kw)
    with torch.no_grad():   # t_mlp 마지막 층은 0 초기화 → 비교가 의미 있도록 무작위 값
        for p in old.t_mlp.parameters():
            p.normal_(0, 0.05)
    nm.load_state_dict(old.state_dict())
    return old, nm


def anchors(B, N=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    step = torch.stack([torch.randn(B, N, 6, generator=g) * 0.3, torch.rand(B, N, 6, generator=g) * 4 + 0.5], -1)
    return step.cumsum(2).reshape(B, N, 12)


def make_map(lines, B=1, M=100, P=20, C=3):
    """lines: [(cls, score, (P,2) 점)] → map_memory dict (나머지 인스턴스 점수 0)."""
    pts = torch.zeros(B, M, P, 2); sc = torch.zeros(B, M, C)
    for i, (c, s, xy) in enumerate(lines):
        pts[:, i] = torch.as_tensor(xy, dtype=torch.float32)
        sc[:, i, c] = s
    return dict(pts=pts, scores=sc)


def line(p0, p1, P=20):
    p0, p1 = np.asarray(p0, np.float32), np.asarray(p1, np.float32)
    return p0[None] + np.linspace(0, 1, P, dtype=np.float32)[:, None] * (p1 - p0)[None]


def mean_m(nm, anchor, t):
    B, N, _ = anchor.shape
    a = (anchor.reshape(B, N, 6, 2) - nm.norm_offset) / nm.norm_scale
    return nm.sqrt_ab[t].view(-1, 1, 1, 1) * a * nm.norm_scale + nm.norm_offset


# ---------------- (1) 지도 없음 → 종전과 비트 동일 ----------------
def test_identity():
    print("(1) 지도 없음 → 종전 모듈과 비트 동일", flush=True)
    old, nm = pair()
    A = anchors(3)
    far = make_map([(1, 0.9, line((200, 0), (200, 50))), (2, 0.9, line((-200, 0), (-200, 50)))], B=3)
    low = make_map([(1, 0.39, line((0, 0), (0, 50))), (2, 0.39, line((0.5, 0), (0.5, 50)))], B=3)
    for mode in ("eval", "train"):
        getattr(old, mode)(); getattr(nm, mode)()
        outs = []
        for mm in (None, make_map([], B=3), low, far):
            torch.manual_seed(7); xo, eo = old(A)
            torch.manual_seed(7); xn, en = nm(A, map_memory=mm)
            outs.append(torch.equal(xo, xn) and torch.equal(eo, en))
        check(f"{mode}: map None / 빈 지도 / 점수 0.39(<0.4) / 반경 밖 지도 → x_t·시점 임베딩 비트 동일", all(outs), str(outs))


# ---------------- (2) 차선 방향 회전 잡음 ----------------
# 수정(2026-10-19): 사용자 결정 — 기존 잡음 타원을 차선 방향으로 회전(새 비율 값 없음)
#   (같은 날 앞선 "m 좌표 비율 1.5 · 정규화 trace 2" 검사를 대체). 기대: m 좌표 공분산 Σ_m = 29² u uᵀ + 13² n nᵀ
#   (기존 등방 잡음의 m 좌표 타원 diag(13², 29²) 를 회전만 한 것), 전방축 차선이면 등방과 비트 동일.
SX, SY = KW["norm_scale"]


def _cov(nm, u):
    """_aniso_cov 의 정규화 공간 Σ_n 과 m 좌표 Σ_m = S Σ_n S (S = diag(scale)), float64."""
    a, b, c = nm._aniso_cov(torch.as_tensor(u, dtype=torch.float64)[None])
    Sn = torch.tensor([[a.item(), b.item()], [b.item(), c.item()]], dtype=torch.float64)
    S = torch.diag(nm.norm_scale.double())
    return Sn, S @ Sn @ S


def test_cov_geometry():
    print("(2a) 공분산 기하(결정적): 0° = 등방(Σ_n = I 정확), 기운 차선 = 기존 타원(29·13 m)을 회전", flush=True)
    _, nm = pair()
    check("anisotropy_ratio 인자 없음(넘기면 TypeError)", not hasattr(nm, "anisotropy_ratio"))
    try:
        MapAlignedPlanTruncatedDiffusion(**KW, **NEW, anisotropy_ratio=1.5); bad = False
    except TypeError:
        bad = True
    check("anisotropy_ratio=1.5 를 넘기면 TypeError(조용히 무시하지 않음)", bad)
    for uy in (1.0, -1.0, 7.3, -0.25):   # 전방축 차선(길이·부호 무관)
        a, b, c = nm._aniso_cov(torch.tensor([[0.0, uy]]))
        check(f"전방축 u=(0,{uy}): Σ_n = I 비트 정확", a.item() == 1.0 and b.item() == 0.0 and c.item() == 1.0,
              f"a={a.item()!r}, b={b.item()!r}, c={c.item()!r}")
    for deg in (30.0, 45.0, 90.0, 135.0):
        th = math.radians(deg)
        u = np.array([math.sin(th), math.cos(th)])             # +y(전방)에서 deg 만큼 오른쪽
        n = np.array([-u[1], u[0]])
        Sn, Sm = _cov(nm, u)
        Sn_neg, _ = _cov(nm, -u)
        ut, nt = torch.as_tensor(u), torch.as_tensor(n)
        exp = SY ** 2 * torch.outer(ut, ut) + SX ** 2 * torch.outer(nt, nt)
        ev, evec = torch.linalg.eigh(Sm)                        # 오름차순: (작은 = 가로 n, 큰 = 차선 u)
        dir_ok = abs(abs(float(evec[:, 1] @ ut)) - 1) < 1e-9 and abs(abs(float(evec[:, 0] @ nt)) - 1) < 1e-9
        check(f"{deg:5.1f}°: Σ_m = 29²uuᵀ + 13²nnᵀ, 고유방향 = (u, n), 표준편차 = (29, 13) m, u·−u 같은 Σ, det Σ_n = 1",
              torch.allclose(Sm, exp, rtol=1e-12, atol=1e-9) and dir_ok
              and abs(math.sqrt(ev[1]) - SY) < 1e-9 and abs(math.sqrt(ev[0]) - SX) < 1e-9
              and torch.allclose(Sn, Sn_neg, rtol=0, atol=1e-15) and abs(float(torch.det(Sn)) - 1) < 1e-12,
              f"σ = {math.sqrt(ev[1]):.6f}, {math.sqrt(ev[0]):.6f} m, 정규화 trace = {float(torch.trace(Sn)):.4f}(각도에 따라 변함; 0° 는 2)")
    # eps = chol(Σ_n) z: 알려진 z 로 하삼각 인자 확인
    th = math.radians(45.0); u = np.array([math.sin(th), math.cos(th)], np.float32)
    lane = make_map([(1, 0.9, line(-10 * u, 60 * u))])
    z = torch.randn(1, 1, 6, 2, generator=torch.Generator().manual_seed(4))
    mean = torch.as_tensor(u)[None, None, None] * torch.linspace(5, 30, 6)[None, None, :, None] + 0.5 * torch.as_tensor([u[1], -u[0]])
    eps = nm._aniso_eps(z, mean, lane)
    Sn, _ = _cov(nm, u.astype(np.float64))
    L = torch.linalg.cholesky(Sn).float()
    check("eps = chol(Σ_n)·z (torch.linalg.cholesky 와 같은 하삼각 인자)", torch.allclose(eps, z @ L.T, atol=1e-6),
          f"max|Δ|={float((eps - z @ L.T).abs().max()):.1e}")


def test_straight_lane():
    print("(2b) 모듈 전체: 전방축 차선 = 등방과 비트 동일(같은 시드), 기운 차선 통계", flush=True)
    old, nm = pair(clip_to_boundary=False)
    # 전방축 차선(구분선 x = 1.5, 경유점이 반경 안): 학습·평가 모두 같은 시드에서 종전(등방) 모듈과 비트 동일
    B, N = 64, 48
    A0 = anchors(B, seed=21).reshape(B, N, 6, 2)
    A0[..., 0] = 0.0; A0 = A0.reshape(B, N, 12)                 # 평균점 x ≈ 0 → x=1.5 구분선에서 반경 3.5 m 안
    for mm_name, xline in (("x=1.5", 1.5), ("x=-2.0", -2.0)):
        fwd = {k: v.repeat(B, *([1] * (v.dim() - 1))) for k, v in make_map([(1, 0.9, line((xline, -5), (xline, 80)))]).items()}
        for mode in ("eval", "train"):
            getattr(old, mode)(); getattr(nm, mode)()
            torch.manual_seed(31); xo, eo = old(A0)
            torch.manual_seed(31); xn, en = nm(A0, map_memory=fwd)
            mu = mean_m(nm, A0, torch.full((B,), nm.infer_t, dtype=torch.long))
            inside = int(((mu[..., 0] - xline).abs() < 3.5).sum())
            check(f"전방축 차선 {mm_name} {mode}: 출력 x_t·시점 임베딩이 등방(종전) 모듈과 비트 동일",
                  torch.equal(xo, xn) and torch.equal(eo, en) and inside > 0, f"반경 안 경유점 {inside} 개")
    old.eval(); nm.eval()
    for deg in (30.0, 45.0, 90.0, 135.0):
        th = math.radians(deg); u = np.array([math.sin(th), math.cos(th)], np.float32)
        nvec = np.array([-u[1], u[0]], np.float32)
        lane = make_map([(1, 0.9, line(-60 * u, 60 * u))])
        B, N = 400, 48
        s_along = torch.linspace(2, 50, N * 6).reshape(N, 6)
        pts = torch.as_tensor(u)[None, None] * s_along[..., None] + 1.5 * torch.as_tensor(nvec)
        A = pts.reshape(1, N, 12).repeat(B, 1, 1)
        mm = {k: v.repeat(B, *([1] * (v.dim() - 1))) for k, v in lane.items()}
        xn, _ = nm(A, map_memory=mm)
        t = torch.full((B,), nm.infer_t, dtype=torch.long)
        mu = mean_m(nm, A, t)
        s1 = float(nm.sqrt_1mab[nm.infer_t])
        dm = ((xn.reshape(B, N, 6, 2) - mu) / s1).reshape(-1, 2).double()   # 정규화 단위 잡음당 m 변위
        uu, nv = torch.as_tensor(u).double(), torch.as_tensor(nvec).double()
        pa, pc = dm @ uu, dm @ nv
        corr = float((pa * pc).mean() / (pa.pow(2).mean().sqrt() * pc.pow(2).mean().sqrt()))
        sa, sc = float(pa.pow(2).mean().sqrt()), float(pc.pow(2).mean().sqrt())
        check(f"{deg:.0f}° m 좌표(통계): 차선·가로 무상관, 차선 방향 σ ≈ 29 m, 가로 σ ≈ 13 m (정규화 단위 잡음당)",
              abs(corr) < 0.02 and abs(sa / SY - 1) < 0.02 and abs(sc / SX - 1) < 0.02,
              f"상관 {corr:+.4f}, σ 차선 {sa:.2f} m / 가로 {sc:.2f} m (× sqrt(1−ᾱ_t)={s1:.4f} → {sa * s1:.3f} / {sc * s1:.3f} m, t={nm.infer_t}, 표본 {dm.shape[0]})")
    # 선택적: 일부 경유점만 반경 안(기운 차선: 반경 밖은 종전과 비트 동일, 반경 안은 달라짐)
    A2 = anchors(2, seed=3)
    th = math.radians(30.0); u2 = np.array([math.sin(th), math.cos(th)], np.float32)
    side = make_map([(1, 0.9, line(np.array([3.0, 0.0]) - 10 * u2, np.array([3.0, 0.0]) + 80 * u2))], B=2)
    xo2, _ = old(A2); xn2, _ = nm(A2, map_memory=side)
    mu2 = mean_m(nm, A2, torch.full((2,), nm.infer_t, dtype=torch.long)).double()
    p0 = torch.tensor([3.0, 0.0], dtype=torch.float64); ud = torch.as_tensor(u2).double()
    w = mu2 - p0
    tt = (w @ ud).clamp(-10, 80)
    dist = (w - tt[..., None] * ud).norm(dim=-1)
    far = dist > 3.5 + 1e-3
    near = dist < 3.5 - 1e-3
    same = (xo2.reshape(2, 48, 6, 2) == xn2.reshape(2, 48, 6, 2)).all(-1)
    check("반경(3.5 m) 밖 경유점은 종전과 비트 동일, 반경 안 경유점은 달라짐(30° 차선)",
          bool(same[far].all()) and bool((~same[near]).all()) and int(far.sum()) > 0 and int(near.sum()) > 0,
          f"밖 {int(far.sum())} 개, 안 {int(near.sum())} 개")


# ---------------- (3) 경계 넘는 잡음 제거 ----------------
def _seg_cross(p, q, a, b):
    d, e, ap = q - p, b - a, a - p
    den = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]
    s = (ap[..., 0] * e[..., 1] - ap[..., 1] * e[..., 0]) / den
    v = (ap[..., 0] * d[..., 1] - ap[..., 1] * d[..., 0]) / den
    return (den.abs() > 1e-9) & (s >= 0) & (s <= 1) & (v >= 0) & (v <= 1)


def test_boundary():
    print("(3) 도로 경계 넘는 잡음 제거", flush=True)
    for name, bpts in (("곧은 경계 x=1.0", line((1.0, -5), (1.0, 65))),
                       ("꺾인 경계", np.concatenate([line((1.5, -5), (1.5, 20), 10), line((1.5, 22), (6.0, 65), 10)]))):
        old, nm = pair(clip_to_boundary=True)
        _, nm_off = pair(clip_to_boundary=False)
        for m in (nm, nm_off):
            m.infer_t = 49; m.eval()                            # 큰 잡음(학습 최대 t)으로 많이 넘게
        B = 64
        A = anchors(B, seed=5)
        A = A.reshape(B, 48, 6, 2); A[..., 0] = A[..., 0].clamp(-3, 0.5); A = A.reshape(B, 48, 12)  # 평균점이 경계 안쪽
        mm = {k: v.repeat(B, *([1] * (v.dim() - 1))) for k, v in make_map([(2, 0.9, bpts)]).items()}
        x_on, _ = nm(A, map_memory=mm); x_off, _ = nm_off(A, map_memory=mm)
        mu = mean_m(nm, A, torch.full((B,), 49, dtype=torch.long)).reshape(-1, 2)
        q_on, q_off = x_on.reshape(-1, 2), x_off.reshape(-1, 2)
        a = torch.as_tensor(bpts[:-1]); b = torch.as_tensor(bpts[1:])
        cr_on = _seg_cross(mu[:, None], q_on[:, None], a[None], b[None]).any(-1)
        cr_off = _seg_cross(mu[:, None], q_off[:, None], a[None], b[None]).any(-1)
        unchanged = (q_on == q_off).all(-1)
        check(f"{name}: 잘라내기 후 경계를 넘는 경유점 0 개(잘라내기 전 {int(cr_off.sum())} 개 / {mu.shape[0]})",
              int(cr_on.sum()) == 0 and int(cr_off.sum()) > 0)
        check(f"{name}: 넘지 않던 경유점은 비트 동일, 넘던 경유점만 바뀜",
              bool(unchanged[~cr_off].all()) and bool((~unchanged[cr_off]).all()))
        # 잘린 점은 평균점→원래 잡음점 선분 위, 교차점에서 0.1 m 안쪽
        d_off = q_off - mu; d_on = q_on - mu
        f = (d_on.norm(dim=-1) / d_off.norm(dim=-1).clamp_min(1e-12))[cr_off]
        col = (d_on[cr_off, 0] * d_off[cr_off, 1] - d_on[cr_off, 1] * d_off[cr_off, 0]).abs().max().item()
        # 교차점까지 거리 − |잘린 변위| ≈ 0.1 m (변위가 0.1 m 보다 짧은 경우 0 으로 붙음)
        ok_margin = True
        idx = torch.nonzero(cr_off).squeeze(1)
        for i in idx[:200].tolist():
            p, q = mu[i], q_off[i]
            hit = _seg_cross(p[None], q[None], a, b)
            d = q - p; e = (b - a)[hit]; ap = (a - p)[hit]
            den = d[0] * e[:, 1] - d[1] * e[:, 0]
            s = ((ap[:, 0] * e[:, 1] - ap[:, 1] * e[:, 0]) / den).min()
            dist_cross = s * d.norm()
            exp = max(dist_cross.item() - 0.1, 0.0)
            ok_margin &= abs(d_on[i].norm().item() - exp) < 1e-3
        check(f"{name}: 잘린 점 = 평균점 + f·변위(같은 방향), 교차점에서 0.1 m 안쪽",
              bool(((f >= 0) & (f <= 1)).all()) and col < 1e-3 and ok_margin, f"f 범위 [{f.min():.3f}, {f.max():.3f}]")


# ---------------- (4) 결정성·배치 독립·기울기·정밀도 ----------------
def test_misc():
    print("(4) 결정성 · 배치 독립 · 기울기 · 정밀도", flush=True)
    old, nm = pair()
    nm.eval()
    A = anchors(2, seed=9)
    # 수정(2026-10-19): 사용자 결정 — 기존 잡음 타원을 차선 방향으로 회전(새 비율 값 없음): 전방축(x 고정) 차선은 이제 등방과 비트 동일이라
    #   지도를 바꿔도 잡음이 안 바뀐다 → 배치 독립 검사가 의미 있도록 차선을 기울인다(x 가 y 에 따라 변함).
    mp = make_map([(1, 0.9, line((0.5, 0), (12.0, 60))), (2, 0.9, line((2.0, 0), (13.5, 60)))], B=2)
    x1, e1 = nm(A, map_memory=mp); x2, e2 = nm(A, map_memory=mp)
    check("평가 모드: 같은 입력 두 번 → 비트 동일(고정 시드 생성기)", torch.equal(x1, x2) and torch.equal(e1, e2))
    mp2 = {k: v.clone() for k, v in mp.items()}
    mp2["scores"][1] = 0.0            # 표본 1 의 지도만 제거
    x3, _ = nm(A, map_memory=mp2)
    check("배치 독립: 표본 1 의 지도만 바꾸면 표본 0 출력 불변, 표본 1 만 바뀜",
          torch.equal(x3[0], x1[0]) and not torch.equal(x3[1], x1[1]))
    # 기울기: 학습 모드, 같은 시드 → t·z 같음
    old.train(); nm.train()
    for mm_name, mm in (("지도 없음", None), ("지도 있음(이방성+경계)", {k: v.repeat(2, *([1] * (v.dim() - 1)))[:2] for k, v in mp.items()})):
        Ao = A.clone().requires_grad_(True); An = A.clone().requires_grad_(True)
        old.zero_grad(); nm.zero_grad()
        torch.manual_seed(11); xo, eo = old(Ao)
        torch.manual_seed(11); xn, en = nm(An, map_memory=mm)
        w = torch.randn_like(xo); we = torch.randn_like(eo)
        ((xo * w).sum() + (eo * we).sum()).backward()
        ((xn * w).sum() + (en * we).sum()).backward()
        g_ok = all(torch.allclose(po.grad, pn.grad) and pn.grad.abs().sum() > 0
                   for po, pn in zip(old.t_mlp.parameters(), nm.t_mlp.parameters()))
        a_ok = torch.allclose(Ao.grad, An.grad, atol=1e-6)
        check(f"기울기({mm_name}): t_mlp 기울기 = 종전, 앵커 기울기 = 종전(sqrt(ᾱ_t))", g_ok and a_ok)
    # 정밀도: fp16 지도 입력 + autocast(bf16) 안에서도 fp32 계산
    nm.eval()
    mp16 = {k: v.half() for k, v in mp.items()}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        x4, e4 = nm(A, map_memory=mp16)
    check("fp16 지도 입력 · autocast 안: x_t fp32·유한, fp32 지도 결과와 같음(지도 값 fp16 반올림 차 이내)",
          x4.dtype == torch.float32 and bool(torch.isfinite(x4).all()) and torch.allclose(x4, x1, atol=5e-3),
          f"최대 차 {float((x4 - x1).abs().max()):.2e} m")


# ---------------- (5) 지도 기억 뱅크 ----------------
def _metas(ts, T, scenes):
    return dict(timestamp=torch.tensor(ts, dtype=torch.float64),
                img_metas=[dict(T_global=Ti, T_global_inv=np.linalg.inv(Ti), scene_token=s) for Ti, s in zip(T, scenes)])


def _pose(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]; T[0, 3], T[1, 3] = x, y
    return T


def test_bank():
    print("(5) PlanMapMemoryInstanceBank", flush=True)
    cfg = Config.fromfile("configs/nuscenes/v2_teacher/stage2/train.py").model.head.onedecoder_head.map_instance_bank
    check("config 의 지도 뱅크 type = PlanMapMemoryInstanceBank", cfg.type == "PlanMapMemoryInstanceBank")
    torch.manual_seed(0); nb = build_from_cfg(cfg, PLUGIN_LAYERS)
    ocfg = cfg.copy(); ocfg.type = "InstanceBank"
    torch.manual_seed(0); ob = build_from_cfg(ocfg, PLUGIN_LAYERS)
    check("파라미터·state_dict 키 동일(체크포인트 호환)",
          list(nb.state_dict()) == list(ob.state_dict()) and all(torch.equal(nb.state_dict()[k], ob.state_dict()[k]) for k in nb.state_dict()))
    B = 2
    T0 = [_pose(0, 0, 0.0), _pose(100, 50, 0.3)]
    T1 = [_pose(0, 3.0, 0.0), _pose(100 + 3 * math.cos(0.3 + math.pi / 2), 50 + 3 * math.sin(0.3 + math.pi / 2), 0.3 + 0.05)]
    m0 = _metas([0.0, 0.0], T0, ["s0", "s1"]); m1 = _metas([0.5, 0.5], T1, ["s0", "s1"])
    feat = torch.randn(B, 100, 256); anc = torch.randn(B, 100, 40) * 10; cls = torch.randn(B, 100, 3)
    same = True
    for bank in (nb, ob):
        bank.reset()
    outs = []
    for bank in (nb, ob):
        r0 = bank.get(B, m0); bank.cache(feat, anc, cls, m0); r1 = bank.get(B, m1)
        u = bank.update(feat, anc, cls)
        outs.append((r0, r1, u))
    flat = lambda r: [x for x in r if torch.is_tensor(x)]
    for (a0, a1, au), (b0, b1, bu) in [tuple(outs)]:
        same = all(torch.equal(x, y) for x, y in zip(flat(a0) + flat(a1) + list(au), flat(b0) + flat(b1) + list(bu)))
        same &= all((x is None) == (y is None) for x, y in zip(a0 + a1, b0 + b1))
    st_same = all(getattr(nb, k) is None and getattr(ob, k) is None or (torch.is_tensor(getattr(nb, k)) and torch.equal(getattr(nb, k), getattr(ob, k)))
                  for k in ("cached_feature", "cached_anchor", "confidence"))
    check("지도 과제 get/update 반환값·상태 = InstanceBank", same and st_same)
    mem = nb.plan_map_memory
    # 기대값: 직전 좌표 p → 전역 → 현재: T_inv(cur) @ T(prev) @ p
    p = anc.reshape(B, 100, 20, 2)
    exp = []
    for i in range(B):
        M = np.linalg.inv(T1[i]) @ T0[i]
        exp.append(p[i] @ torch.tensor(M[:2, :2].T, dtype=torch.float32) + torch.tensor(M[:2, 3], dtype=torch.float32))
    exp = torch.stack(exp)
    check("직전 프레임 지도 예측을 자차 이동(회전+이동)으로 현재 좌표에 옮김, 점수 = sigmoid(로짓)",
          mem is not None and torch.allclose(mem["pts"], exp, atol=1e-4) and torch.allclose(mem["scores"], cls.sigmoid()))
    nb.cache(feat, anc, cls, m1)
    nb.get(B, _metas([3.0, 1.0], T1, ["s0", "s1"]))
    sc = nb.plan_map_memory["scores"]
    check("시간 차 > 2 s 인 표본만 지도 없음(점수 0)", float(sc[0].abs().sum()) == 0 and float(sc[1].abs().sum()) > 0)
    nb.cache(feat, anc, cls, _metas([3.0, 1.0], T1, ["s0", "s1"]))
    nb.get(B, _metas([3.5, 1.5], T1, ["s0", "OTHER"]))
    sc = nb.plan_map_memory["scores"]
    check("다른 장면(scene_token) 표본만 지도 없음", float(sc[1].abs().sum()) == 0 and float(sc[0].abs().sum()) > 0)
    nb.get(3, _metas([4.0] * 3, [T1[0]] * 3, ["s0"] * 3))
    check("배치 크기가 바뀌면 지도 없음(None)", nb.plan_map_memory is None)
    keys = bank_state_keys(nb)
    check("스트림 그룹 저장 키에 지도 기억 포함", all(k in keys for k in PlanMapMemoryInstanceBank._MEM_KEYS), str(keys))
    nb.reset(); nb.get(B, m0)
    check("기억 없이 시작 → None(등방 잡음)", nb.plan_map_memory is None)


# ---------------- (6) config ----------------
def test_configs():
    print("(6) config", flush=True)
    for p in STAGE2:
        c = Config.fromfile(p)
        od = c.model.head.onedecoder_head
        pd = od.plan_diffusion
        ok = pd.type == "MapAlignedPlanTruncatedDiffusion" and od.map_instance_bank.type == "PlanMapMemoryInstanceBank"
        ok &= all(pd[k] == v for k, v in NEW.items() if not isinstance(v, tuple)) and \
            all(tuple(pd[k]) == v for k, v in NEW.items() if isinstance(v, tuple))
        ok &= all((tuple(pd[k]) if isinstance(pd[k], (list, tuple)) else pd[k]) == v for k, v in KW.items())
        ok &= "tools.plan_diffusion_map_register" in c.custom_imports.imports
        if "token_" in p:
            ok &= "tools.sensor_tokens_register" in c.custom_imports.imports
        check(p, ok, f"imports={c.custom_imports.imports}")
    bad = []
    for p, h in STAGE1_SHA256.items():
        if hashlib.sha256(open(p, "rb").read()).hexdigest() != h:
            bad.append(p)
    check(f"1단계 config {len(STAGE1_SHA256)} 개 파일 해시 = 작업 전", not bad, str(bad))
    for p in ("configs/nuscenes/v2_teacher/stage1/train.py", "configs/nuscenes/token_lidar/stage1/train.py"):
        c = Config.fromfile(p)
        od = c.model.head.onedecoder_head
        check(f"{p}: 지도 뱅크 InstanceBank, 확산 모듈 종전 그대로",
              od.map_instance_bank.type == "InstanceBank" and (od.get("plan_diffusion") is None or od.plan_diffusion.type == "PlanTruncatedDiffusion"))


# ---------------- (7) 전체 모델 ----------------
def test_full():
    print("(7) 전체 모델(CPU, token_lidar 2단계 config, val 연속 2 프레임)", flush=True)
    import tools.verify_v2.test_sensor_tokens_cpu as sens   # flash-attn·deformable aggregation torch 기준 구현으로 교체(같은 방식)
    from mmcv.parallel import collate
    from mmdet.models import build_detector
    from mmdet.datasets import build_dataset
    cfg = Config.fromfile("configs/nuscenes/token_lidar/stage2/train.py")
    cfg.model.head.onedecoder_head.plan_diffusion.score_thr = 0.0   # 무작위 초기화 모델은 지도 점수가 낮다 → 기하 경로를 실제로 타게
    torch.manual_seed(0)
    det = build_detector(cfg.model); det.init_weights()
    od = det.head.onedecoder_head
    check("디코더: 잡음 모듈 MapAligned + 지도 뱅크 PlanMapMemory", isinstance(od.plan_diffusion, MapAlignedPlanTruncatedDiffusion)
          and isinstance(od.map_instance_bank, PlanMapMemoryInstanceBank) and od.plan_diffusion_uses_map)
    seen = []
    orig = od.plan_diffusion.forward
    def spy(anchor, map_memory=None):
        # 점수 > 0 인 인스턴스 수(None = 기억 없음 → 0). 다른 장면·시간 차 초과는 점수 0 으로 들어온다.
        seen.append(0 if map_memory is None else int((map_memory["scores"].max(-1).values > 0).sum()))
        return orig(anchor, map_memory=map_memory)
    od.plan_diffusion.forward = spy
    tr = cfg.data.train.copy(); tr.ann_file = "data/infos/nuscenes_fusion_infos_val.pkl"
    ds = build_dataset(tr)
    det.train(); det.use_grid_mask = False
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    scenes = []
    for idx in (100, 101):
        t0 = time.time()
        d = sens.unwrap(collate([ds[dict(idx=idx, aug_config=ds.get_augmentation(), stream_group=0)]], samples_per_gpu=1))
        img = d.pop("img")
        scenes.append(d["img_metas"][0]["scene_token"])
        losses = det.forward_train(img, **d)
        total = sum(v for k, v in losses.items() if "loss" in k and torch.is_tensor(v))
        total.backward(); det.zero_grad()
        bad = [k for k, v in losses.items() if torch.is_tensor(v) and not torch.isfinite(v).all()]
        check(f"학습 idx {idx}: 손실 유한·역전파", not bad, f"total={float(total):.3f}, {time.time() - t0:.1f} s")
    check("학습: 첫 프레임 지도 기억 없음 → 둘째 프레임(같은 장면) 디코더가 지도 기억(100 인스턴스)을 넘김",
          scenes[0] == scenes[1] and seen == [0, 100], f"seen={seen}")
    det.eval(); seen.clear()
    # 평가 첫 프레임에는 위 학습 프레임(다른 장면)의 기억이 뱅크에 남아 있다 → 장면 검사로 점수 0(= 지도 없음)이어야 한다
    ds_t = build_dataset(Config.fromfile("configs/nuscenes/token_lidar/stage2/test.py").data.test)
    for i in (0, 1):
        dt = sens.unwrap(collate([ds_t[i]], samples_per_gpu=1))
        img_t = dt.pop("img")
        img_t = img_t[0] if isinstance(img_t, list) else img_t
        with torch.no_grad():
            out = det.simple_test(img_t, **dt)
    check("평가: 첫 프레임(남은 기억은 다른 장면) 지도 없음 → 둘째 프레임 지도 기억 사용, 결과 생성", seen == [0, 100] and len(out[0]["img_bbox"]) > 0,
          f"seen={seen}")
    od.plan_diffusion.forward = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-full", action="store_true", help="(7) 전체 모델 생략")
    a = ap.parse_args()
    assert not torch.cuda.is_available(), "CPU 전용 시험: CUDA_VISIBLE_DEVICES=\"\""
    test_identity()
    test_cov_geometry()   # 수정(2026-10-19): 사용자 결정 — 기존 잡음 타원을 차선 방향으로 회전(새 비율 값 없음)
    test_straight_lane()
    test_boundary()
    test_misc()
    test_bank()
    test_configs()
    if not a.no_full:
        test_full()
    print("결과:", "전부 통과" if not FAIL else f"실패 {len(FAIL)}: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
