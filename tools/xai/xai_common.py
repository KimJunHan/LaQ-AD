# 수정(2026-10-20): 사용자 승인 — 설명가능성 4가지(학습 변경 없음)
"""LaQ-AD 설명가능성(XAI) 공용 도구. 새 파일이며 기존 소스·config·패키지 __init__ 은 하나도 바꾸지 않는다.

이 모듈이 하는 일(모두 추론·분석 전용 — 학습 경로에는 들어가지 않는다):
  1) TokenLayout  : 교사 캐시 7 폴더(K/V 2,369 토큰)의 토큰별 의미 — 폴더, context 의 카메라/지시문 부분, 객체 칸(slot 0..7),
                    plan 의 자유거리 토큰, action 의 의도/행동 토큰. 폴더 _meta.json 의 answer_positions 와 표본의 qwen_type 으로 만든다.
  2) explicit_cross_attn : QwenTokenAttn.attn(MultiheadFlashAttention→FlashMHA) 과 같은 수식을 fp32 로 직접 계산해
                    어텐션 가중치 P(B,H,N,L) 와 잔차 없는 출력 A 를 함께 돌려준다(flash 커널은 가중치를 돌려주지 않는다).
  3) XaiHook      : 모델의 QwenTokenAttn 인스턴스 하나에 **인스턴스 속성으로만** forward/_attn_only 를 덮어써(클래스·파일 불변)
                    (a) 계층별 가중치·게이트 주입량 기록, (b) 교사 토큰 가리기(key padding mask), (c) 언어 항 0(=g⊙A 제거)을 한다.
                    uninstall() 로 원래 메서드를 되돌린다.
  4) 추론 공통    : tools/plan_scorer/dump_plan_feats.py 와 같은 순서로 모델·데이터 구성(평가와 같은 설정), 장면 경계 샤드, 시간 뱅크
                    상태 저장·복원(재개·반사실 비교용), GPU1 전용·메모리 상한 검사.
  5) 의도 라벨    : 교사 GT 의도 라벨을 만든 함수(tools/vla_teacher/gen_teacher_gt_v2.py 189~250 행)를 같은 인자로 다시 불러
                    임의 궤적(학생 계획)에 같은 정의의 의도를 붙인다.
"""
import copy
import functools
import importlib
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
VT = os.path.join(REPO, "tools", "vla_teacher")
PS = os.path.join(REPO, "tools", "plan_scorer")
for _p in (VT, PS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TAG = "# 수정(2026-10-20): 사용자 승인 — 설명가능성 4가지(학습 변경 없음)"
FOLDERS = ("context", "detection", "tracking", "motion", "map", "plan", "action")
CACHE_ROOT = os.path.join(REPO, "data/nuscenes/laq_cache")
INDEX_FILES = [os.path.join(CACHE_ROOT, "_index/teacher_A.jsonl"), os.path.join(CACHE_ROOT, "_index/teacher_B.jsonl")]
GT_DIR = os.path.join(REPO, "data/vla_teacher/v2")
GT_VAL = os.path.join(GT_DIR, "teacher_gt_val.jsonl")
FOLD_FILE = os.path.join(GT_DIR, "fold_split.json")
CODEBOOK = os.path.join(REPO, "data/qwen3d/nusc_action_codebook_k2048.npz")
INFOS_VAL = os.path.join(REPO, "data/infos/nuscenes_infos_val.pkl")
MAP_ROOT = os.path.join(REPO, "data/nuscenes")
CAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")  # 캐시 context 영상 블록 순서
CAM_SHORT = ("F", "FL", "FR", "BL", "B", "BR")
TOKENS_PER_CAM, N_SLOTS, SLOT_LEN = 180, 8, 57      # tools/vla_teacher/laq_cache_layout.py 와 같은 값(아래 TokenLayout 이 메타로 검사)
GPU1_BUDGET_MIB = 88000                              # 사용자 지시: nvidia-smi -i 1 사용량 + 필요량 ≤ 88000 MiB 일 때만 GPU 스모크


# =========================================================================================== 1) 토큰 배치
class TokenLayout:
    """K/V 토큰(이어 붙인 7 폴더) → 해석 단위(segment) 소속 행렬.

    segment 이름(겹칠 수 있음 — 폴더 합과 부분 합을 같이 본다):
      폴더 7                : context, detection, tracking, motion, map, plan, action
      context 부분 7        : ctx_img_F/FL/FR/BL/B/BR (현재 프레임 카메라별 180 토큰), ctx_text (지시문·명령·자차 상태 줄)
      객체 칸 8             : slot0..slot7 = 그 칸의 detection 21 + tracking 5 + motion 30 + plan(RISK) 1 토큰(교사 답 k 번째 객체)
      기타 3                : plan_free(<free>DD.D 5), intent(2), action_tok(6)
    """

    def __init__(self, qwen_type, cache_root=CACHE_ROOT):
        qt = np.asarray(qwen_type).reshape(-1).astype(np.int64)
        self.L = len(qt)
        metas = {}
        for f in FOLDERS:
            mp = os.path.join(cache_root, f, "_meta.json")
            metas[f] = json.load(open(mp)) if os.path.exists(mp) else {}
        self.folder_idx = {}
        for k, f in enumerate(FOLDERS):
            idx = np.flatnonzero(qt == k)
            if len(idx) == 0:
                raise ValueError(f"TokenLayout: 폴더 {f} 토큰이 없다(qwen_type)")
            if not np.array_equal(idx, np.arange(idx[0], idx[0] + len(idx))):
                raise ValueError(f"TokenLayout: 폴더 {f} 토큰이 연속하지 않다")
            mL = metas[f].get("L")
            if mL is not None and int(mL) != len(idx):
                raise ValueError(f"TokenLayout: 폴더 {f} 길이 {len(idx)} != _meta.json L {mL}")
            self.folder_idx[f] = idx
        seg = {f: self.folder_idx[f] for f in FOLDERS}
        ctx = self.folder_idx["context"]
        n_img = TOKENS_PER_CAM * len(CAMS)
        if len(ctx) <= n_img:
            raise ValueError(f"TokenLayout: context {len(ctx)} <= 영상 토큰 {n_img}")
        for c, s in enumerate(CAM_SHORT):
            seg[f"ctx_img_{s}"] = ctx[c * TOKENS_PER_CAM:(c + 1) * TOKENS_PER_CAM]
        seg["ctx_text"] = ctx[n_img:]
        # 객체 칸: 각 폴더 토큰의 '정답 시작 기준 위치'(answer_positions) → 칸 번호 = (pos - 첫 칸 시작) // 57
        slot_of = {}
        det_pos = metas["detection"].get("answer_positions")
        if det_pos is None:
            raise ValueError("TokenLayout: detection/_meta.json 에 answer_positions 가 없다")
        p0 = int(det_pos[0])
        for f in ("detection", "tracking", "motion", "plan"):
            pos = np.asarray(metas[f]["answer_positions"], np.int64)
            if len(pos) != len(self.folder_idx[f]):
                raise ValueError(f"TokenLayout: {f} answer_positions {len(pos)} != 토큰 {len(self.folder_idx[f])}")
            sl = (pos - p0) // SLOT_LEN
            inside = (pos >= p0) & (sl < N_SLOTS)
            slot_of[f] = np.where(inside, sl, -1)
        for k in range(N_SLOTS):
            seg[f"slot{k}"] = np.concatenate([self.folder_idx[f][slot_of[f] == k] for f in ("detection", "tracking", "motion", "plan")])
        for f in ("detection", "tracking", "motion"):
            if (slot_of[f] < 0).any():
                raise ValueError(f"TokenLayout: {f} 토큰 중 칸 밖 위치가 있다")
        seg["plan_free"] = self.folder_idx["plan"][slot_of["plan"] < 0]
        act = self.folder_idx["action"]
        seg["intent"], seg["action_tok"] = act[:2], act[2:]
        if len(seg["plan_free"]) != 5 or len(act) != 8 or any(len(seg[f"slot{k}"]) != SLOT_LEN for k in range(N_SLOTS)):
            raise ValueError("TokenLayout: plan_free/action/칸 길이가 laq_cache_layout 과 다르다")
        self.seg = seg
        self.names = list(seg)
        self.qwen_type = qt

    def matrix(self, torch_mod=None, device=None, dtype=None):
        """(L, n_seg) 0/1 소속 행렬. torch_mod 를 주면 torch 텐서."""
        M = np.zeros((self.L, len(self.names)), np.float32)
        for j, n in enumerate(self.names):
            M[self.seg[n], j] = 1.0
        if torch_mod is None:
            return M
        return torch_mod.as_tensor(M, device=device, dtype=dtype or torch_mod.float32)

    def keep_mask(self, mask_segments):
        """가릴 segment 이름 목록 → (L,) bool, True = 남김(유효)."""
        keep = np.ones(self.L, bool)
        for n in mask_segments:
            keep[self.seg[n]] = False
        if not keep.any():
            raise ValueError("keep_mask: 모든 토큰을 가렸다(softmax 정의 불가) — 언어 전체 제거는 'gate_off' 조건을 쓴다")
        return keep


# =========================================================================================== 2) 명시적 교차 어텐션
def explicit_cross_attn(mfa, q, kv, keep=None):
    """MultiheadFlashAttention(mfa) 의 교차 어텐션을 fp32 로 직접 계산. 반환 (A (B,N,E) fp32, P (B,H,N,L) fp32).

    수식(= models/decoder/temporal_self_attn/flash_attention.py FlashMHA.forward + FlashAttention + MultiheadFlashAttention.forward, eval):
      [Q;K;V] = 입력 @ in_proj_weight 의 세 조각ᵀ + in_proj_bias 조각,  머리 H 개로 나눔(d = E/H)
      P = softmax(Q Kᵀ / √d)      (flash softmax_scale=None → 1/√headdim, 인과 마스크 없음, eval 이라 dropout 0)
      A = out_proj(P V)            (identity=0 으로 부르므로 잔차 없음 — QwenTokenAttn._attn_only 와 같음; proj_drop·dropout_layer 는 eval 항등)
    keep: (B,L) bool, True = 유효 키(가려진 키는 logit -inf).
    """
    import torch
    import torch.nn.functional as F
    fm = mfa.attn
    H, E = fm.num_heads, fm.embed_dim
    d = E // H
    with torch.autocast(device_type=q.device.type, enabled=False):
        W = fm.in_proj_weight.float()
        b = fm.in_proj_bias.float() if fm.in_proj_bias is not None else None
        wq, wk, wv = W.chunk(3)
        bq, bk, bv = (None, None, None) if b is None else b.chunk(3)
        B, N, _ = q.shape
        L = kv.shape[1]
        Q = F.linear(q.float(), wq, bq).view(B, N, H, d).transpose(1, 2)
        K = F.linear(kv.float(), wk, bk).view(B, L, H, d).transpose(1, 2)
        Vv = F.linear(kv.float(), wv, bv).view(B, L, H, d).transpose(1, 2)
        logits = torch.matmul(Q, K.transpose(-1, -2)) * (1.0 / math.sqrt(d))
        if keep is not None:
            logits = logits.masked_fill(~keep.bool()[:, None, None, :], float("-inf"))
        P = torch.softmax(logits, dim=-1)
        ctx = torch.matmul(P, Vv).transpose(1, 2).reshape(B, N, E)
        A = F.linear(ctx, fm.out_proj.weight.float(), None if fm.out_proj.bias is None else fm.out_proj.bias.float())
    return A, P


def gate_rows(qta, layer_idx, N):
    """QwenTokenAttn.forward 와 같은 규칙으로 계층 layer_idx 의 쿼리별 게이트 (N, E) 를 만든다(값 복사, 그래프 없음)."""
    import torch
    li = min(int(layer_idx), qta.gate.shape[0] - 1)
    g = torch.zeros(N, qta.embed_dims, device=qta.gate.device, dtype=torch.float32)
    prev = 0
    for si, end in enumerate(list(qta.seg_bounds) + [N]):
        end = min(int(end), N)
        if end > prev:
            g[prev:end] = (qta.gate_sep[li][qta.sep_gate_segs.index(si)] if si in qta.sep_gate_segs else qta.gate[li][si]).detach().float()
        prev = end
    return g


def static_gate_table(qta):
    """계층 × 쿼리 구간(det, map, plan/ego) 의 |g| 평균·최대 — 학습된 파라미터 그 자체(입력 무관)."""
    L = qta.gate.shape[0]
    names = ["det", "map", "plan_ego"][:qta.num_segs]
    out = {"segments": names, "mean_abs": [], "max_abs": []}
    for li in range(L):
        ma, mx = [], []
        for si in range(qta.num_segs):
            g = (qta.gate_sep[li][qta.sep_gate_segs.index(si)] if si in qta.sep_gate_segs else qta.gate[li][si]).detach().float()
            ma.append(float(g.abs().mean())); mx.append(float(g.abs().max()))
        out["mean_abs"].append(ma); out["max_abs"].append(mx)
    return out


# =========================================================================================== 3) 훅(인스턴스 덮어쓰기)
class XaiHook:
    """QwenTokenAttn 인스턴스에 설치하는 기록·개입 훅. 클래스·파일은 바꾸지 않는다(인스턴스 속성만, uninstall 로 복원).

    설정(각 forward 전에 바꿀 수 있음):
      impl      : 'flash'(기본, 모델이 실제 쓰는 커널로 A 계산) | 'explicit'(fp32 직접 계산 — CPU·검증용)
      keep      : None 또는 (L,) bool numpy — False 인 교사 토큰을 key padding mask 로 가린다(모든 계층·모든 쿼리)
      force_pad : True 면 keep 이 전부 True 여도 key_padding_mask 를 넘겨 flash 의 unpad 경로를 탄다(경로 동등성 검사용)
      zero_lang : True 면 A=0 → 출력 = query (게이트 언어 항 제거 = '언어 없음' 반사실)
      record    : True 면 계층마다 명시적 가중치를 계산해 rows_sel 쿼리 행을 segment 로 모아 기록
    기록(self.rec, forward 마다 clear()):
      layers        : 호출된 layer_idx 순서
      agg[l]        : (n_rows, n_seg) 머리 평균 가중치의 segment 합 (rows = plan C×M + ego)
      tok[l]        : (n_rows, L) 머리 평균 가중치(plan/ego 행) — 표본 처리 후 선택 행만 남긴다
      ent[l]        : (n_rows,) 엔트로피(nats, 머리 평균 가중치 기준)
      det_agg/map_agg[l] : (n_seg,) 검출 900 / 지도 100 쿼리 평균
      inj[l]        : {det,map,plan,ego}: mean_q ‖g⊙A‖/‖query‖  (언어가 쿼리를 실제로 얼마나 바꿨나)
      a_diff[l]     : flash A 와 명시적 A 의 최대 절대차·상대 노름차(impl='flash' 이고 record 일 때)
    """

    def __init__(self, qta, plan_rows, ego_rows, det_rows, map_rows):
        if qta.route_type_ids is not None:
            raise NotImplementedError("XaiHook: route_segs(구간별 폴더 경로 지정) 설정은 지원하지 않는다 — 현재 LaQ v2 config 는 None")
        self.qta = qta
        self.rows = list(plan_rows) + list(ego_rows)
        self.n_plan = len(plan_rows)
        self.det_rows, self.map_rows = det_rows, map_rows
        self.impl, self.keep, self.force_pad, self.zero_lang, self.record = "flash", None, False, False, False
        self.seg_mat = None
        self.rec = None
        self._cur_layer = None
        self._orig = None
        self.clear()

    def clear(self):
        self.rec = dict(layers=[], agg={}, tok={}, ent={}, det_agg={}, map_agg={}, inj={}, a_diff={})

    def install(self):
        if self._orig is not None:
            raise RuntimeError("XaiHook: 이미 설치됨")
        qta = self.qta
        self._orig = (qta.__dict__.get("forward"), qta.__dict__.get("_attn_only"))
        orig_forward = type(qta).forward.__get__(qta)
        orig_attn = type(qta)._attn_only.__get__(qta)
        hook = self

        def forward(query, qwen_hidden, qwen_mask=None, layer_idx=0, qwen_type=None, kv=None):
            hook._cur_layer = int(layer_idx)
            out = orig_forward(query, qwen_hidden, qwen_mask=qwen_mask, layer_idx=layer_idx, qwen_type=qwen_type, kv=kv)
            if hook.record:
                hook._record_injection(query, out)
            return out

        def _attn_only(q, kv, **kw):
            import torch
            keep = None
            if hook.keep is not None:
                keep = torch.as_tensor(hook.keep, device=kv.device, dtype=torch.bool)[None].expand(kv.shape[0], -1)
            if "key_padding_mask" in kw:   # 실제 패딩(현 캐시는 없음)과 결합
                keep = kw["key_padding_mask"].bool() if keep is None else (keep & kw["key_padding_mask"].bool())
            if keep is None and hook.force_pad:
                keep = torch.ones(kv.shape[:2], device=kv.device, dtype=torch.bool)
            if hook.zero_lang:
                A = torch.zeros_like(q)
                if hook.record:   # 가중치는 기록(언어를 빼도 '무엇을 보려 했나'는 같은 수식)
                    _, P = explicit_cross_attn(qta.attn, q, kv, keep)
                    hook._record_weights(P, None, None)
                return A
            if hook.impl == "explicit":
                A, P = explicit_cross_attn(qta.attn, q, kv, keep)
                A = A.to(q.dtype)
                if hook.record:
                    hook._record_weights(P, None, None)
                return A
            kw2 = {k: v for k, v in kw.items() if k != "key_padding_mask"}
            if keep is not None:
                kw2["key_padding_mask"] = keep    # flash-attn unpad_input 규약: True = 유효
            A = orig_attn(q, kv, **kw2)
            if hook.record:
                Ae, P = explicit_cross_attn(qta.attn, q, kv, keep)
                hook._record_weights(P, Ae, A)
            return A

        qta.forward = forward
        qta._attn_only = _attn_only
        return self

    def uninstall(self):
        if self._orig is None:
            return
        qta = self.qta
        for name, old in zip(("forward", "_attn_only"), self._orig):
            if old is None:
                qta.__dict__.pop(name, None)
            else:
                setattr(qta, name, old)
        self._orig = None

    # ---- 기록
    def _record_weights(self, P, Ae, A):
        import torch
        l = self._cur_layer
        Pm = P.mean(1)[0]                           # (N, L) 머리 평균, 배치 1 가정(추론)
        if P.shape[0] != 1:
            raise RuntimeError("XaiHook: 기록은 배치 1 추론 전용")
        rows = torch.as_tensor(self.rows, device=Pm.device)
        Pr = Pm.index_select(0, rows)               # (n_rows, L)
        S = self.seg_mat.to(Pm.device)
        self.rec["layers"].append(l)
        self.rec["agg"][l] = (Pr @ S).cpu()
        self.rec["tok"][l] = Pr.half().cpu()
        self.rec["ent"][l] = (-(Pr.clamp_min(1e-30).log() * Pr).sum(-1)).cpu()
        if self.det_rows is not None:
            self.rec["det_agg"][l] = (Pm[self.det_rows[0]:self.det_rows[1]].mean(0) @ S).cpu()
        if self.map_rows is not None:
            self.rec["map_agg"][l] = (Pm[self.map_rows[0]:self.map_rows[1]].mean(0) @ S).cpu()
        if Ae is not None and A is not None:
            d = (Ae - A.float())
            self.rec["a_diff"][l] = (float(d.abs().max()), float(d.norm() / A.float().norm().clamp_min(1e-12)))

    def _record_injection(self, query, out):
        import torch
        l = self._cur_layer
        d = (out.float() - query.float())[0].norm(dim=-1)
        qn = query.float()[0].norm(dim=-1).clamp_min(1e-12)
        r = d / qn
        res = {}
        if self.det_rows is not None:
            res["det"] = float(r[self.det_rows[0]:self.det_rows[1]].mean())
        if self.map_rows is not None:
            res["map"] = float(r[self.map_rows[0]:self.map_rows[1]].mean())
        res["plan"] = float(r[self.rows[0]:self.rows[0] + self.n_plan].mean())
        res["ego"] = float(r[self.rows[self.n_plan:]].mean()) if len(self.rows) > self.n_plan else float("nan")
        self.rec["inj"][l] = res


# =========================================================================================== 4) 추론 공통
def require_gpu1(need_mib=0, allow_cpu=False):
    """GPU1 전용 규칙. CUDA_VISIBLE_DEVICES=1 이 아니면 중단(GPU0 금지). need_mib>0 이면 GPU1 사용량+필요량 ≤ 88000 MiB 검사."""
    if allow_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("GPU1 전용(사용자 규칙): CUDA_VISIBLE_DEVICES=1 로 실행한다. GPU0 사용 금지.")
    if need_mib > 0:
        used = gpu1_used_mib()
        if used + need_mib > GPU1_BUDGET_MIB:
            raise SystemExit(f"GPU1 사용량 {used} MiB + 필요 {need_mib} MiB > {GPU1_BUDGET_MIB} MiB — 실행하지 않는다")
    return "cuda"


def gpu1_used_mib():
    o = subprocess.run(["nvidia-smi", "-i", "1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout.strip()
    return int(o.splitlines()[0])


def load_cfg(config, cfg_options=None):
    from mmcv import Config
    cfg = Config.fromfile(config)
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    # tools/test.py 와 같은 플러그인 경로(projects.mmdet3d_plugin) — 다른 이름으로 적재하면 레지스트리가 이중 등록된다
    importlib.import_module(os.path.dirname(cfg.plugin_dir).replace("/", "."))
    return cfg


def build_test_dataset(cfg, ann_file=None):
    from mmdet.datasets import build_dataset
    ds_cfg = copy.deepcopy(cfg.data.test)
    ds_cfg.test_mode = True
    if ds_cfg.pop("samples_per_gpu", 1) != 1:
        raise SystemExit("data.test.samples_per_gpu != 1 — 평가(배치 1, 순차)와 다르다")
    if ann_file:
        ds_cfg.ann_file = ann_file
    return build_dataset(ds_cfg), ds_cfg


def build_model(cfg, checkpoint, device="cuda"):
    """tools/test.py 비분산 경로(= dump_plan_feats.py) 와 같은 순서. device='cpu' 는 단위 시험 전용."""
    import torch
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet.models import build_detector
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    set_random_seed(0, deterministic=False)
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None and device != "cpu":
        wrap_fp16_model(model)
    if checkpoint:
        tl = torch.load
        torch.load = functools.partial(tl, weights_only=False)
        try:
            load_checkpoint(model, checkpoint, map_location="cpu")
        finally:
            torch.load = tl
    if device == "cpu":
        return model.eval(), model
    from mmcv.parallel import MMDataParallel
    m = MMDataParallel(model, device_ids=[0]).eval()
    return m, m.module


def decoder_rows(dec, N=None):
    """통합 쿼리(det 900, map 100, plan C×M, ego 1 — query_select 순서) 행 범위. N 을 주면 합이 맞는지 검사."""
    order = list(dec.query_select)
    n = {"det": dec.det_instance_bank.num_anchor if "det" in order else 0,
         "map": dec.map_instance_bank.num_anchor if "map" in order else 0,
         "plan": dec.ego_fut_cmd * dec.ego_fut_mode if "plan" in order else 0,
         "ego": 1 if "ego" in order else 0}
    rng, s = {}, 0
    for k in order:
        rng[k] = (s, s + n[k]); s += n[k]
    if N is not None and s != N:
        raise RuntimeError(f"decoder_rows: 쿼리 합 {s} != N {N} ({n})")
    qta = dec.qwen_token_attn
    if "det" in rng and "map" in rng and list(qta.seg_bounds) != [rng["det"][1], rng["map"][1]]:
        raise RuntimeError(f"seg_bounds {qta.seg_bounds} 가 쿼리 배치 {rng} 와 맞지 않다")
    return rng


class BankState:
    """시간 인스턴스 뱅크 + run_step + 장면 토큰 저장·복원(dump_plan_feats.py 와 같은 키). 반사실: 표본 직전 상태로 되돌려 다시 추론."""

    def __init__(self, det_model):
        from projects.mmdet3d_plugin.models.decoder.temporal_bank.stream_state import bank_state_keys
        self.det = det_model
        self.dec = det_model.head.onedecoder_head
        if not self.dec.is_init_bank_list:
            self.dec.init_instance_bank_list()
        self.keys = {n: bank_state_keys(b) for n, b in self.banks().items()}

    def banks(self):
        return {n: getattr(self.dec, f"{n}_instance_bank_list")[0] for n in ("ego", "det", "map", "plan")
                if n in self.dec.query_select}

    def save(self):
        from projects.mmdet3d_plugin.models.decoder.temporal_bank.stream_state import save_bank_state
        return {"banks": {n: save_bank_state(b, self.keys[n]) for n, b in self.banks().items()},
                "run_step": self.dec.run_step, "last_scene_token": getattr(self.det, "_last_scene_token", None)}

    def load(self, st):
        from projects.mmdet3d_plugin.models.decoder.temporal_bank.stream_state import load_bank_state
        for n, b in self.banks().items():
            load_bank_state(b, st["banks"][n])
        self.dec.run_step = st["run_step"]
        if hasattr(self.det, "_last_scene_token"):
            self.det._last_scene_token = st["last_scene_token"]


def scene_shards(infos, spp):
    from dump_plan_feats import scene_shards as _ss   # tools/plan_scorer — 장면 연속·시간 증가 단언 포함
    return _ss(infos, spp)


def qwen_type_from_data(data):
    """dataloader 표본(DataContainer) → qwen_type numpy (L,)."""
    qt = data["qwen_type"]
    qt = qt.data[0] if hasattr(qt, "data") else qt
    qt = qt[0] if isinstance(qt, (list, tuple)) else qt
    return np.asarray(qt).reshape(-1)


def atomic_savez(path, **arrs):
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


def check_manifest(out_dir, man, keys):
    p = os.path.join(out_dir, "manifest.json")
    if os.path.exists(p):
        old = json.load(open(p))
        norm = lambda m: json.loads(json.dumps(m))  # noqa: E731  (튜플→리스트)
        diff = [k for k in keys if norm(old.get(k)) != norm(man.get(k))]
        if diff:
            raise SystemExit(f"{out_dir} 의 기존 manifest 와 다르다({diff}) — 다른 출력 폴더를 쓰거나 기존 결과를 확인할 것")
    else:
        os.makedirs(out_dir, exist_ok=True)
        json.dump(man, open(p, "w"), indent=1, ensure_ascii=False)


# =========================================================================================== 5) 교사 색인·GT·의도 라벨
def read_jsonl_by_token(files, want=None, part_prefix=None):
    out = {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                if not line.endswith("\n"):
                    break   # 쓰는 중인 마지막 줄
                t = line[11:43] if line.startswith('{"token": "') else json.loads(line)["token"]
                if want is not None and t not in want:
                    continue
                e = json.loads(line)
                if part_prefix and not str(e.get("part", "")).startswith(part_prefix):
                    continue
                if t in out:
                    raise SystemExit(f"토큰 {t} 가 두 번 있다({f})")
                e["_file"] = os.path.basename(f)
                out[t] = e
    return out


def load_val_index():
    return read_jsonl_by_token(INDEX_FILES, part_prefix="val.")


def load_gt_val():
    return read_jsonl_by_token([GT_VAL])


class IntentLabeler:
    """교사 GT 의도 라벨과 **같은 함수·같은 인자** 로 임의 궤적의 의도를 붙인다.

    출처 = tools/vla_teacher/gen_teacher_gt_v2.py (GT 생성, 189~250 행) — 이 클래스는 그 블록을 그대로 옮긴 것이다:
      lab = common.intent_labels(fut, v0, info['map_annos'][1])          # 종방향 + 회전(v1 규칙)
      회전이 아니고 경로 길이 ≥ MIN_PATH 이면 lane_v2.lane_signals(LaneIndex(map_root, loc), info, xy, heading_series(xy),
        현재 yaw rate, lab['divider_crossing'], lab['lat_resid']) 의 decision 이 'lane change *' 면 그것, 아니면 go straight
        (decision == 'uncertain' 이면 go straight + uncertain 표시)
    v0 = targets.label_v0 (CAN 속도, 결측 시 포즈 차분 — GT 와 같은 값), yaw rate = 레코드 ego_status.yaw_rate(ego_in_prompt='current'
    의 ego_cur 와 같은 값, 없으면 None).
    임계값(common.py 25~55 행, train 통계로 고정): STOP_V 0.5 m/s, DV_TH 1.0 m/s, TURN_TH_DEG 12.5°, MIN_PATH 2.0 m,
      lane_v2: ASSIGN_R 4.0 m, ASSIGN_MAX 2.0 m, HEAD_TOL 40°, HALF_LANE 1.6 m, HEAD_MIN 3°, YAW_OPP 0.1 rad/s, RESID_FALLBACK 1.0 m.
    """

    def __init__(self, map_root=MAP_ROOT):
        import common as C1
        import common_v2 as C
        import lane_v2 as LV
        self.C1, self.C, self.LV = C1, C, LV
        self.map_root = map_root
        self.lane_idx = {}

    def thresholds(self):
        C1, LV = self.C1, self.LV
        t = dict(STOP_V=C1.STOP_V, DV_TH=C1.DV_TH, TURN_TH_DEG=C1.TURN_TH_DEG, MIN_PATH=C1.MIN_PATH, DT=C1.DT)
        for k in ("ASSIGN_R", "ASSIGN_MAX", "HEAD_TOL_DEG", "ALONG_TOL", "HALF_LANE", "HEAD_MIN_DEG", "YAW_OPP", "RESID_FALLBACK",
                  "ADJ_MIN", "ADJ_MAX", "BFS_DEPTH"):
            if hasattr(LV, k):
                t[k] = getattr(LV, k)
        t["source"] = "tools/vla_teacher/common.py:25-55 (intent_labels) + tools/vla_teacher/lane_v2.py:19-28 (lane_signals); " \
                      "decision rule = tools/vla_teacher/gen_teacher_gt_v2.py:229-250"
        return t

    def label(self, traj, info, rec):
        """traj (6,2) lidar(+x 오른쪽, +y 전방) 누적 궤적 → dict(longitudinal, lateral, uncertain, ...)."""
        C1, C, LV = self.C1, self.C, self.LV
        fut = np.asarray(traj, np.float64)[:6, :2]
        v0 = float(rec["targets"]["label_v0"])
        lab = C1.intent_labels(fut, v0, info["map_annos"].get(1, []))
        lat_v1 = lab["lateral"]
        uncertain = False
        lat = lat_v1
        if not lat_v1.startswith("turn") and lab["path_len"] >= C1.MIN_PATH:
            loc = info["map_location"]
            if loc not in self.lane_idx:
                self.lane_idx[loc] = LV.LaneIndex(self.map_root, loc)
            xy = np.concatenate([np.zeros((1, 2)), fut], 0)
            es = rec.get("ego_status") or {}
            yr = es.get("yaw_rate") if es.get("available", False) else None
            sig = LV.lane_signals(self.lane_idx[loc], info, xy, C.heading_series(xy), yr, lab["divider_crossing"], lab["lat_resid"])
            dec = sig["decision"]
            lat = dec if (dec or "").startswith("lane change") else "go straight"
            uncertain = dec == "uncertain"
        return dict(lab, lateral=lat, lateral_v1=lat_v1, uncertain=uncertain)


def teacher_intent(gen_text):
    """교사 생성 답 → (intent dict | None, parsed). common_v2.parse_answer(엄격 파서) 사용."""
    import common_v2 as C
    p = C.parse_answer(gen_text)
    return p["intent"], p


def load_results_cpu(path):
    import pickle
    import torch
    tl = torch.load
    torch.load = functools.partial(tl, map_location="cpu", weights_only=False)  # results.pkl 의 CUDA 텐서 → CPU(평가기와 같은 방식)
    try:
        return pickle.load(open(path, "rb"))
    finally:
        torch.load = tl


def load_infos(path=INFOS_VAL):
    import pickle
    d = pickle.load(open(path, "rb"))
    return d["infos"] if isinstance(d, dict) else d


def planning_eval_module():
    spec = importlib.util.spec_from_file_location("eval_nusc_planning", os.path.join(REPO, "tools/eval_nusc_planning.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def plan_metrics(pe, pred, gt, gt_mask, fut_boxes, gt_col=None):
    """평가기(tools/eval_nusc_planning.py) 규칙: 시점별 L2, 충돌 = 자차 상자 vs 미래 GT 상자(GT 궤적도 충돌한 시점 제외).
    미래 GT 불완전이면 (None, None, None). 반환 (l2 (6,), col (6,) bool, gt_col)."""
    import torch
    if not (np.asarray(gt_mask)[:6] == 1).all() or len(fut_boxes) != 6:
        return None, None, None
    p = torch.as_tensor(np.asarray(pred, np.float32))[:6, :2].clone()
    g = torch.as_tensor(np.asarray(gt, np.float32))[:6, :2].clone()
    l2 = torch.sqrt(((p - g) ** 2).sum(-1)).numpy()   # x 부호 반전은 L2 에 무관(평가기와 같은 값)
    if gt_col is None:
        gt_col = pe.single_coll(g, fut_boxes)          # 평가기: x 반전 후 다시 반전(p*[-1,1]) → 원 좌표
    pc = pe.single_coll(p, fut_boxes)
    return l2, (pc & ~gt_col).numpy(), gt_col
