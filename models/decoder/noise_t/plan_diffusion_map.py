# 수정(2026-10-19): 사용자 결정 — 지도 정렬 선택적 Gaussian 잡음(지도 밖 잡음 제거)
#   PlanTruncatedDiffusion(./diffusion.py)의 하위 클래스. 바뀌는 것은 "잡음 eps 의 분포" 하나뿐이다.
#   시점 t 뽑기(학습 U{0..trunc_steps-1}, 추론 infer_t 고정)·일정(β 선형)·정규화((x-offset)/scale)·고정 시드 생성기·
#   시점 임베딩(t_mlp)·출력 형식((B,N,T*2) m, (B,1,C))은 부모와 같다.
#
#   [지도] 직전 프레임에 모델 자신이 예측한 지도(detach)를 자차 이동으로 현재 좌표에 옮긴 것
#     (temporal_bank/plan_map_memory_bank.py 의 PlanMapMemoryInstanceBank.plan_map_memory). 정답 지도는 쓰지 않는다
#     → 학습·평가가 같은 규칙. 지도가 없는 표본(시퀀스 첫 프레임 등)은 부모와 같은 등방 잡음.
#
#   (1) 지도 정렬 이방성 잡음("선택적 Gaussian splatting"): 경유점마다(정규화 전 m 좌표의 평균점 m = sqrt(ab_t)·a 를 되돌린 점)
#       점수 ≥ score_thr 인 차선 구분선(divider)·도로 경계(boundary) 선분 중 search_radius(m) 안의 가장 가까운 선분을 찾는다.
#       수정(2026-10-19): 사용자 결정 — 기존 잡음 타원을 차선 방향으로 회전(새 비율 값 없음)
#       (같은 날 1차 지적 3 의 "m 좌표 비율 anisotropy_ratio·정규화 trace 2" 설계를 대체한다. 그 설계는 앞쪽 차선에서 가로 잡음이
#        등방보다 커졌다: t=49 가로 1σ 2.60 m vs 등방 2.21 m.)
#       기존(DiffusionDrive 식) 잡음은 정규화 공간에서 등방 N(0, I₂) 이고, m 좌표에서는 축에 붙은 타원이다:
#       표준편차 = norm_scale ⊙ sqrt(1−ᾱ_t) (가로 x 13 m, 전방 y 29 m — 정규화 단위 잡음당). 이 두 크기를 그대로 두고 **방향만**
#       돌린다: 긴 축(29 m)을 차선 접선 u(m 좌표 단위 벡터)에, 짧은 축(13 m)을 법선 n 에 맞춘다.
#         Σ_m = s_y² u uᵀ + s_x² n nᵀ,  s_y = norm_scale[1], s_x = norm_scale[0],  n = (−u_y, u_x)  (주축이 m 좌표에서 정확히 수직)
#         Σ_n = D Σ_m D,  D = diag(1/norm_scale)  (m 좌표 변위 = scale ⊙ 정규화 변위)
#         eps = chol(Σ_n) z  (2×2 닫힌 꼴 Cholesky, 하삼각),  z ~ N(0, I) (부모와 같은 방법·같은 시드).
#       새 하이퍼파라미터 없음(조정값 없음). 접선 부호는 무관(u 와 −u 가 같은 Σ).
#       차선이 정확히 전방축(u = (0, ±1))이면 Σ_n = I 가 정확히 성립하고 eps = z 와 비트 동일(등방과 같은 출력).
#       정규화 공간 trace(Σ_n) = ρ²u_x² + u_y² + u_y² + u_x²/ρ² (ρ = s_y/s_x = 29/13) 는 차선 각도에 따라 2(0°) ~ 5.18(90°) 로 변한다
#       (m 좌표 분산 합 s_x² + s_y² 은 각도와 무관하게 등방과 같다).
#       가까운 선분이 없으면 eps = z (부모와 비트 동일). → 지도가 없으면 출력이 부모와 완전히 같다.
#   (2) 지도 밖 잡음 제거: 경유점의 잡음 변위 d = x_t − m (m 좌표)가 점수 ≥ score_thr 인 도로 경계 선분을 가로지르면
#       가장 먼저 만나는 교차점(변위 비율 s)에서 boundary_margin(m)만큼 안쪽(변위 방향 반대로)까지만 남긴다:
#       x_t ← m + max(s − margin/|d|, 0)·d. 다시 뽑지 않는다(표본 건너뛰기 아님). 즉 잡음 분포의 지지 집합을 "평균점에서
#       경계를 넘지 않고 닿는 영역" 으로 정의한 것이다. 경계를 넘지 않는 경유점은 x_t 가 그대로(비트 동일)다.
#       방향 구분은 하지 않는다: 평균점이 이미 경계 바깥이면 잡음은 그 쪽에 머문다(경계를 어느 방향으로도 넘지 않는다).
#       평가에도 같은 규칙(고정 시드 잡음 → 같은 잘라내기).
#   기울기: 기하 계산(가까운 선분·교차 비율)은 모두 detach. 잘린 점 = m + f·(x_t − m) 이고 x_t − m 은 앵커에 의존하지 않으므로
#     앵커로 가는 기울기는 부모와 같다(sqrt(ab_t)). t_mlp 경로는 부모 그대로.
#   (3) 수정(2026-10-23): 사용자 결정 — 잡음 세기 확률 선택 + 지도 확신도에 따른 선택적 가우시안(①+②)
#     ① t_sampling="gauss": 학습 시점 t 를 균등 U{0..trunc_steps-1} 대신 추론 시점 infer_t 를 중심으로 한 이산 가우시안
#        p(t) ∝ exp(−(t − infer_t)² / (2 t_std²)), t ∈ {0..trunc_steps−1} (범위 밖 확률 0, 정규화) 에서 뽑는다(torch.multinomial,
#        다시 뽑기 없음). 추론은 항상 infer_t 인데 균등 추출은 학습의 ~80% 를 t ≥ 10 에 쓴다 → 학습·추론 잡음 세기를 맞춘다.
#        t_std 기본값 = infer_t(8): t 4~12 에 49.8 %, t ≥ 10 은 49.7 %(균등 80 %), t ≥ 25 는 2.3 %(강한 잡음 복구 능력 유지), 평균 t 10.1. 선례: EDM(Karras et al.,
#        NeurIPS 2022) 로그 정규 σ 추출, SD3(Esser et al., ICML 2024) 로짓 정규 t 추출. t_sampling="uniform"(기본) 이면 종전과 비트 동일.
#     ② selective=True: 경유점마다 가까운 차선 인스턴스의 예측 점수 c(sigmoid, ≥ score_thr)를 "이 지도를 믿을 확률" 로 쓴다.
#        확률 c 로 차선 회전 타원(1), 확률 1−c 로 종전 전방축 타원(eps = z) 을 쓴다(점마다 베르누이, 혼합 가우시안
#        Σ = c·Σ_aligned + (1−c)·Σ_ego). 새 조정값 없음(점수 그대로). 학습: torch.rand, 평가: 별도 고정 시드 생성기(infer_seed + 1)
#        (z 는 종전과 같은 값, 배치 크기와 무관). selective=False(기본) 이면 종전과 비트 동일. (2) 경계 잘라내기는 그대로 모든 점에 적용.
#   (4) 수정(2026-11-01): 사용자 지시 — 전진축 옵션 forward_axis(NAVSIM 좌표 규약용). 종전 (1) 은 "norm_scale[1] = 전진(긴) 축"
#     을 가정했다(nuScenes·B2D 계획 좌표 = 라이다 +x 오른쪽·+y 전방). NAVSIM 계획 좌표(라이다 = ego, lidar2ego 단위행렬)는
#     +x 전방·+y 왼쪽이라(models/loader/navsim/dataset_v2_teacher.py, CAM_F0 광축 = 라이다 +x, navtest 4 s 끝 변위 x 평균 19.5 m·|y| 2.6 m)
#     전진축 표준편차가 norm_scale[0](34)·횡축이 norm_scale[1](24)이다. 종전 식을 그대로 쓰면 차선 접선에 횡축 크기(24)·법선에 전진축
#     크기(34)가 붙어, 앞으로 곧은 차선에서 잡음 타원이 등방(DiffusionDrive) 대비 90° 돌아간다(전진 24·횡 34).
#     forward_axis = 전진축의 좌표 번호. 1(기본) = 종전 식·종전 연산 그대로(nuScenes·B2D 비트 동일). 0 = 접선에 s_f = norm_scale[0],
#     법선에 s_l = norm_scale[1]:  Σ_m = s_f² û ûᵀ + s_l² n nᵀ → p = s_f D û = (û_x, κ û_y), q = s_l D n = (n_x/κ, n_y), κ = s_0/s_1.
#     û = (±1, 0)(차선이 정확히 전진축)이면 Σ_n = I 가 정확하다(eps = z, 등방과 같음) — 종전 축의 성질을 그대로 옮긴 것.
#     (2) 경계 자르기·지도 기억·시점 추출·selective 의 eps = z 대안은 좌표축과 무관해 바뀌지 않는다.
#   정밀도: 디코더는 fp32 구간(백본·FPN 만 fp16, stage2/train_v2.py)이고, 여기서는 지도 입력도 float32 로 바꿔 계산한다.
import torch

from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

from .diffusion import PlanTruncatedDiffusion, timestep_embedding

__all__ = ["MapAlignedPlanTruncatedDiffusion"]


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@PLUGIN_LAYERS.register_module()
class MapAlignedPlanTruncatedDiffusion(PlanTruncatedDiffusion):
    uses_map_memory = True  # 디코더가 지도 기억(map_memory)을 넘겨야 함을 알리는 표시

    def __init__(self, *args, score_thr=0.4, lane_class_ids=(1, 2), boundary_class_ids=(2,), search_radius=3.5,
                 boundary_margin=0.1, clip_to_boundary=True, t_sampling="uniform", t_std=None, selective=False, forward_axis=1,
                 infer_noise_broadcast=False, **kwargs):  # 수정(2026-11-09): C3 GRPO 설계 §9.1 — 추론 잡음 배치 broadcast 플래그(기본 False = 종전)
        # 수정(2026-10-19): 사용자 결정 — 기존 잡음 타원을 차선 방향으로 회전(새 비율 값 없음) — anisotropy_ratio 인자 제거(넘기면 TypeError)
        super().__init__(*args, **kwargs)
        assert search_radius > 0 and boundary_margin >= 0
        self.score_thr = float(score_thr)
        self.lane_class_ids = tuple(int(c) for c in lane_class_ids)
        self.boundary_class_ids = tuple(int(c) for c in boundary_class_ids)
        self.search_radius = float(search_radius)
        self.boundary_margin = float(boundary_margin)
        self.clip_to_boundary = bool(clip_to_boundary)
        # 수정(2026-10-23): 사용자 결정 — 잡음 세기 확률 선택(①) + 지도 확신도 선택적 가우시안(②)
        assert t_sampling in ("uniform", "gauss"), t_sampling
        self.t_sampling = t_sampling
        self.t_std = float(self.infer_t if t_std is None else t_std)
        assert self.t_std > 0
        ts = torch.arange(self.trunc_steps, dtype=torch.float64)
        w = torch.exp(-0.5 * ((ts - self.infer_t) / self.t_std) ** 2)
        self.register_buffer("t_probs", (w / w.sum()).float(), persistent=False)   # 버퍼(비저장) → state_dict 불변
        self.selective = bool(selective)
        # 수정(2026-11-01): 사용자 지시 — 전진축 옵션(NAVSIM 은 0 = +x 전진). 1(기본) = 종전(+y 전진, nuScenes·B2D) 비트 동일.
        assert forward_axis in (0, 1), forward_axis
        self.forward_axis = int(forward_axis)
        # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — 추론 고정 시드 잡음(z, r) 캐시·클래스 색인 장치 텐서 캐시(파이썬 속성, state_dict 불변)
        self._infer_noise_cache = {}
        self._class_idx_cache = {}
        # 수정(2026-11-09): C3 GRPO 설계 §9.1 "확산 잡음" — True 면 추론(eval) 잡음 z·r 을 배치 1 모양 (1, N, T, 2)·(1, N, T) 로 뽑아
        #   배치 B 에 broadcast 한다 → 배치 8 의 모든 표본이 배치 1 forward_test 와 같은 잡음을 받는다(종전: (B, N, T, 2) 를 한 시드에서 뽑아
        #   두 번째 이후 표본이 다른 잡음). False(기본) 면 종전 동작·값 그대로. 학습(train) 분기는 이 플래그와 무관. 파이썬 속성이라
        #   state_dict 불변 — GRPO 루프가 실행 중 켠다.
        self.infer_noise_broadcast = bool(infer_noise_broadcast)

    def _sample_t(self, B, device):
        """수정(2026-10-23): 학습 시점 t. uniform = 종전(torch.randint, 같은 난수 소비), gauss = infer_t 중심 이산 가우시안."""
        if self.t_sampling == "uniform":
            return torch.randint(0, self.trunc_steps, (B,), device=device)
        return torch.multinomial(self.t_probs.to(device), B, replacement=True)

    # ---------- 지도 → 선분 ----------
    def _segments(self, map_memory, class_ids):
        """점수 ≥ thr 인 인스턴스의 선분. 반환 A, E(=B−A) (B, S, 2), valid (B, S), 선분이 속한 인스턴스 점수 (B, S). 해당 없음 → None."""
        pts = map_memory["pts"].float()            # (B, M, P, 2)
        sc = map_memory["scores"].float()          # (B, M, C)
        # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — 리스트 색인(매 호출 색인 텐서 H2D) → 장치별 캐시 색인 텐서 index_select(같은 원소)
        ck = (tuple(class_ids), sc.device)
        if ck not in self._class_idx_cache:
            self._class_idx_cache[ck] = torch.tensor(list(class_ids), dtype=torch.long, device=sc.device)
        inst = sc.index_select(-1, self._class_idx_cache[ck]).max(-1).values  # (B, M) 해당 클래스 최대 점수
        keep = inst >= self.score_thr                                        # (B, M)
        # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — 고정 모양 — n_keep = int(keep.sum(1).max().item())(동기화·데이터 의존 모양)
        #   과 통과 인스턴스 앞으로 모으기(정렬·gather)를 없애고 모든 M 인스턴스를 그대로 두되 keep 을 valid 에 넣는다.
        #   종전과 결과가 같은 이유: 종전은 통과 인스턴스를 원래 순서대로(안정 정렬) 앞에 모아 n_keep 개만 남겼고, 남은 비통과 자리는
        #   valid=False 였다. 이 판에서 비통과 인스턴스의 선분도 valid=False → _aniso_eps 의 d2=inf, _clip_boundary 의 hit=False.
        #   min/argmin 은 유효 선분 집합이 같고 원래 순서가 보존되므로(동률이면 앞 색인) 같은 선분을 고른다. n_keep=0 이면 종전은
        #   None(eps=z, 자르기 없음)이고, 이 판은 has·crossed 가 모두 False 라 torch.where 로 같은 값(기울기도 같음)이 나온다.
        #   M=0(인스턴스 없음)은 모양만으로 판정(동기화 없음) — 종전과 같이 None.
        if keep.shape[1] == 0:
            return None
        A, E = pts[:, :, :-1], pts[:, :, 1:] - pts[:, :, :-1]                 # (B, K, P-1, 2)
        valid = keep[..., None] & (E.norm(dim=-1) > 1e-6)                      # 길이 0 선분 제외
        B = pts.shape[0]
        conf = inst[..., None].expand(-1, -1, A.shape[2])                     # 선분마다 인스턴스 점수
        return A.reshape(B, -1, 2), E.reshape(B, -1, 2), valid.reshape(B, -1), conf.reshape(B, -1)

    # ---------- (1) 이방성 잡음 ----------
    def _aniso_eps(self, z, mean_m, map_memory, r=None):
        """z (B,N,T,2) ~ N(0,I), mean_m (B,N,T,2) m, r (B,N,T) ~ U[0,1)(selective 일 때 선택용). 반환 eps (B,N,T,2) 정규화 공간."""
        seg = self._segments(map_memory, self.lane_class_ids)
        if seg is None:
            return z
        A, E, valid, conf = seg
        B = z.shape[0]
        P = mean_m.reshape(B, -1, 2)                                            # (B, K, 2)
        W = P[:, :, None] - A[:, None]                                          # (B, K, S, 2)
        ee = (E * E).sum(-1).clamp_min(1e-12)                                   # (B, S)
        tt = ((W * E[:, None]).sum(-1) / ee[:, None]).clamp(0.0, 1.0)           # (B, K, S)
        d2 = (W - tt[..., None] * E[:, None]).pow(2).sum(-1)
        d2 = d2.masked_fill(~valid[:, None], float("inf"))
        dmin, idx = d2.min(-1)                                                  # (B, K)
        has = dmin <= self.search_radius ** 2
        if self.selective:   # 수정(2026-10-23): ② 확률 c(가까운 차선 점수)로만 회전 타원, 나머지는 종전 전방축 타원
            c = torch.gather(conf, 1, idx)                                      # (B, K)
            has = has & (r.reshape(B, -1).to(c.dtype) < c)
        u = torch.gather(E, 1, idx[..., None].expand(-1, -1, 2))                # (B, K, 2) 선분 방향(m)
        # 수정(2026-10-19): 사용자 결정 — 기존 잡음 타원을 차선 방향으로 회전(새 비율 값 없음) — m 좌표 공분산 → 정규화 공간 Σ_n → eps = chol(Σ_n) z
        a, b, c = self._aniso_cov(u)                                            # Σ_n = [[a, b], [b, c]]  (B, K)
        l11 = a.sqrt()
        l21 = b / l11
        l22 = (c - l21 * l21).clamp_min(0.0).sqrt()                             # Σ_n 은 양정치(det = 1) → 음수 아님(반올림 보호)
        zf = z.reshape(B, -1, 2).to(u.dtype)
        aniso = torch.stack([l11 * zf[..., 0], l21 * zf[..., 0] + l22 * zf[..., 1]], -1).to(z.dtype)
        eps = torch.where(has[..., None], aniso, z.reshape(B, -1, 2))
        return eps.reshape(z.shape)

    def _aniso_cov(self, u):
        """수정(2026-10-19): 사용자 결정 — 기존 잡음 타원을 차선 방향으로 회전(새 비율 값 없음).
        u (..., 2) 차선 선분 방향(m, 길이·부호 무관). 반환 정규화 공간 공분산 Σ_n = D Σ_m D 의 성분 (a=Σ_xx, b=Σ_xy, c=Σ_yy).
        Σ_m = s_y² û ûᵀ + s_x² n nᵀ (û = u/|u|, n = (−û_y, û_x)), s = norm_scale, D = diag(1/s).
        계산은 Σ_n = p pᵀ + q qᵀ,  p = s_y·D û = (ρ û_x, û_y),  q = s_x·D n = (n_x, n_y/ρ),  ρ = s_y/s_x 로 한다(같은 식).
        û = (0, ±1) 이면 p = (0, ±1), q = (∓1, 0) 이 반올림 없이 나와 Σ_n = I 가 정확하다. det(Σ_n) = 1 (모든 각도)."""
        uh = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        sc = self.norm_scale.to(u.dtype)
        if self.forward_axis == 1:   # 수정(2026-11-01): 사용자 지시 — 전진축 옵션. 1 = 종전 식·연산 그대로(nuScenes·B2D 비트 동일)
            rho = sc[1] / sc[0]
            px, py = rho * uh[..., 0], uh[..., 1]                               # p = s_y D û
            qx, qy = -uh[..., 1], uh[..., 0] / rho                              # q = s_x D n,  n = (−û_y, û_x)
        else:                        # 수정(2026-11-01): 사용자 지시 — 0 = +x 전진(NAVSIM): 접선에 s_x(전진), 법선에 s_y(횡)
            kap = sc[0] / sc[1]
            px, py = uh[..., 0], kap * uh[..., 1]                               # p = s_x D û
            qx, qy = -uh[..., 1] / kap, uh[..., 0]                              # q = s_y D n,  n = (−û_y, û_x)
        a = px * px + qx * qx
        b = px * py + qx * qy
        c = py * py + qy * qy
        return a, b, c

    # ---------- (2) 경계 넘는 잡음 제거 ----------
    def _clip_boundary(self, mean_m, x_m, map_memory):
        """mean_m, x_m (B,N,T,2) m(기울기 없음). 반환 f (B,N,T) ∈ [0,1] 남길 변위 비율, crossed (B,N,T) bool. 경계 없음 → None."""
        seg = self._segments(map_memory, self.boundary_class_ids)
        if seg is None:
            return None
        A, E, valid, _ = seg
        B = mean_m.shape[0]
        P = mean_m.reshape(B, -1, 2)
        D = (x_m - mean_m).reshape(B, -1, 2)                                    # 잡음 변위 (B, K, 2)
        AP = A[:, None] - P[:, :, None]                                         # (B, K, S, 2)
        den = _cross(D[:, :, None], E[:, None])                                 # (B, K, S)
        ok = den.abs() > 1e-9                                                   # 평행(교차점 하나로 정해지지 않음) 제외
        den_s = torch.where(ok, den, torch.ones_like(den))
        s = _cross(AP, E[:, None]) / den_s                                      # 변위 위 비율
        v = _cross(AP, D[:, :, None]) / den_s                                   # 경계 선분 위 비율
        hit = ok & valid[:, None] & (s >= 0) & (s <= 1) & (v >= 0) & (v <= 1)
        s_min = torch.where(hit, s, torch.full_like(s, float("inf"))).min(-1).values   # 가장 먼저 만나는 교차
        crossed = torch.isfinite(s_min)
        dn = D.norm(dim=-1).clamp_min(1e-12)
        f = (torch.where(crossed, s_min, torch.ones_like(s_min)) - self.boundary_margin / dn).clamp(0.0, 1.0)
        shp = mean_m.shape[:-1]
        return f.reshape(shp), crossed.reshape(shp)

    def forward(self, anchor, map_memory=None):
        """anchor: (B, N, T*2) 누적 궤적(lidar, m). map_memory: PlanMapMemoryInstanceBank.plan_map_memory(None = 지도 없음).
        반환: 오염된 앵커 (B, N, T*2), 시점 임베딩 (B, 1, C)."""
        B, N, D = anchor.shape
        a = (anchor.reshape(B, N, self.ego_fut_ts, 2) - self.norm_offset) / self.norm_scale
        r = None
        if self.training:
            t = self._sample_t(B, anchor.device)   # 수정(2026-10-23): ① 잡음 세기 확률 선택(uniform 이면 종전 그대로)
            z = torch.randn_like(a)
            if self.selective and map_memory is not None:
                r = torch.rand(a.shape[:-1], device=anchor.device, dtype=a.dtype)
        else:
            t = torch.full((B,), self.infer_t, device=anchor.device, dtype=torch.long)
            # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — 고정 시드 생성기 결과는 (모양, dtype, 장치, 시드)로만 정해진다
            #   → 처음 한 번 종전과 같은 호출로 만들어 캐시(매 프레임 생성기 생성·시드·난수 커널 제거, 값 동일). z·r 은 제자리 수정되지 않는다.
            shp = (1,) + tuple(a.shape[1:]) if self.infer_noise_broadcast else tuple(a.shape)  # 수정(2026-11-09): C3 GRPO §9.1 broadcast 플래그
            nk = (shp, a.dtype, anchor.device, self.infer_seed)  # 수정(2026-11-09): C3 GRPO §9.1 — 모양 키를 shp 로. #이전 (tuple(a.shape), ...)
            if nk not in self._infer_noise_cache:
                assert not (anchor.is_cuda and torch.cuda.is_current_stream_capturing()), \
                    "MapAlignedPlanTruncatedDiffusion: 추론 잡음 캐시는 CUDA Graph 캡처 전에 만들어야 한다"
                g = torch.Generator(device=anchor.device); g.manual_seed(self.infer_seed)
                z = torch.randn(shp, generator=g, device=anchor.device, dtype=a.dtype)  # 수정(2026-11-09): C3 GRPO §9.1. #이전 torch.randn(a.shape, ...)
                # 수정(2026-10-23): 무결점 검사 1차 지적 D2 — z 다음에 같은 생성기에서 뽑으면 r 의 시작 위치가 배치 크기에 따라 바뀐다
                #   (표본 0 의 r 이 B 에 의존). 별도 생성기(시드 infer_seed + 1)로 뽑아 z 처럼 표본 순서만으로 정해지게 한다. z 는 종전과 같은 값.
                gr = torch.Generator(device=anchor.device); gr.manual_seed(self.infer_seed + 1)
                r_all = torch.rand(shp[:-1], generator=gr, device=anchor.device, dtype=a.dtype)  # 수정(2026-11-09): C3 GRPO §9.1. #이전 torch.rand(a.shape[:-1], ...)
                self._infer_noise_cache[nk] = (z, r_all)
            z, r_all = self._infer_noise_cache[nk]
            if self.infer_noise_broadcast:  # 수정(2026-11-09): C3 GRPO §9.1 — (1, …) → (B, …) broadcast(복사 없는 expand; 제자리 수정 없음)
                z, r_all = z.expand(a.shape), r_all.expand(a.shape[:-1])
            if self.selective and map_memory is not None:
                r = r_all
        sa = self.sqrt_ab[t].view(B, 1, 1, 1)
        s1 = self.sqrt_1mab[t].view(B, 1, 1, 1)
        if map_memory is None:
            x_t = sa * a + s1 * z
            x_t = x_t * self.norm_scale + self.norm_offset
        else:
            mean_n = sa * a
            mean_m = mean_n * self.norm_scale + self.norm_offset                # 잡음의 평균점(m), 기울기 있음
            with torch.no_grad():
                eps = self._aniso_eps(z, mean_m.detach().float(), map_memory, r)
            x_t = (mean_n + s1 * eps) * self.norm_scale + self.norm_offset
            if self.clip_to_boundary:
                with torch.no_grad():
                    clip = self._clip_boundary(mean_m.detach().float(), x_t.detach().float(), map_memory)
                if clip is not None:
                    f, crossed = clip
                    clipped = mean_m + f[..., None].to(x_t.dtype) * (x_t - mean_m)
                    x_t = torch.where(crossed[..., None], clipped, x_t)
        x_t = x_t.reshape(B, N, D)
        t_emb = self.t_mlp(timestep_embedding(t, self.embed_dims)).unsqueeze(1)
        return x_t, t_emb
