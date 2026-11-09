# 수정(2026-11-09): C3 GRPO 설계 v4.1 §9.1(롤아웃 모드)·§12-⑤⑥ 구현 — 신규 파일(기존 파일 무수정).
"""GRPO 롤아웃 경로 공용 모듈(학습 루프·덤프·G2 시험이 같이 쓴다). hipad_bw env(GPU).

전제: docs/grpo_model_flag_20261109.diff(모델 플래그 3 개)가 적용된 소스. 적용 전이면 build_rl_model 이 즉시 실패한다.

모델(§9.1)
  - 모델 config(MODEL_CFG, 예: 2단계 train_e3_nuim 계열 — 경로는 호출자가 준다) + 디코더 플래그
    force_stream_groups=True, plan_grad_isolation=True, plan_diffusion.infer_noise_broadcast=True.
  - plan_sft_key 는 bc_target="gt"(기본, 설계 §8 "2단계 계획 손실")면 None 으로 덮는다. "sft" 면 config 값을 그대로 둔다.
  - model.eval() 전체, fp16 이면 wrap_fp16_model(tools/test.py 와 같음), 체크포인트는 엄격 적재
    (빠진 키 없음, 남는 키는 비저장 버퍼만 — ema_to_ckpt.py 와 같은 규칙).
  - 호출: extract_feat(no_grad) → _maybe_extract_bev → _maybe_load_qwen_cache → head(img, feats, data). simple_test 를 쓰지 않는다.
데이터(§9.1·§9.2)
  - data.train 설정(navtrain, test_mode False, GroupInBatchSampler, 스트림 그룹 6) + 파이프라인에서
    PhotoMetricDistortion·MultiScaleDepthMapGenerator·LoadPointsFromPcdNavsim(·LoadPointsFromFile)·VectorizeMap 제거, Collect 키 교체.
  - 증강 설정은 test_mode 분기 값(결정적)을 쓴다: NAVSIM data_aug_conf 는 학습 분기도 같은 값이지만 numpy RNG 를 소비하므로
    get_augmentation 을 test 분기로 감싼다(값 동일, RNG 소비 없음 — 재개 재현성).
선택(§9.1): 명령 묶음 16 모드 → rescore(no_grad, 배포 select 와 같은 입력) → 상위 M = rescore 후 logit 내림차순, 같으면 원시 logit
  내림차순(§12.1(4)). 최종 궤적 = rescore 후 argmax(torch.argmax, 배포와 같음).
좌표: NAVSIM lidar2ego = 단위행렬(NavsimTeacherDataset 이 전 프레임 검사) → 모델 좌표 = ego 프레임 = 채점 좌표.
"""
import copy
import hashlib
import importlib
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.abspath(__file__)).parent   # 심볼릭 링크를 풀지 않는다(스모크 오버레이에서 그 트리의 models 를 쓰도록)
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import ar1  # noqa: E402

T = 8
N_MODE = 16
RL_DROP = ("PhotoMetricDistortionMultiViewImage", "MultiScaleDepthMapGenerator", "LoadPointsFromPcdNavsim", "LoadPointsFromFile",
           "VectorizeMap")
RL_KEYS = ["img", "qwen_hidden", "qwen_mask", "qwen_type", "timestamp", "projection_mat", "image_wh", "ego_status",
           "gt_ego_fut_cmd", "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz", "ego_input"]
RL_META = ["T_global", "T_global_inv", "timestamp", "scene_token", "token", "stream_group"]
TRAINABLE = (r"^head\.onedecoder_head\.plan_refine\.", r"^head\.onedecoder_head\.plan_anchor_encoder\.",
             r"^head\.onedecoder_head\.plan_deformable\.", r"^head\.onedecoder_head\.plan_diffusion\.t_mlp\.",
             r"^head\.onedecoder_head\.layers\.\d+\.plan\.", r"^head\.onedecoder_head\.qwen_token_attn\.gate_sep$")
DEFAULT_CENTERLINES = "data/navsim/grpo/pdm_centerlines_navtrain"
DEFAULT_METRIC_CACHE = "/home1/irteam/data-vol1/datasets/navsim/exp/metric_cache_navtrain"


# ---------------------------------------------------------------- config·모델
def load_model_cfg(path, bc_target="gt", flags=True):
    from mmcv import Config
    cfg = Config.fromfile(path)
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    importlib.import_module(os.path.dirname(cfg.plugin_dir).replace("/", "."))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    dh = cfg.model.head.onedecoder_head
    if flags:
        dh.force_stream_groups = True
        dh.plan_grad_isolation = True
        pd = dh.get("plan_diffusion", None)
        if pd is None:
            raise RuntimeError("모델 config 에 plan_diffusion 이 없다 — 확산 계획기 전제(§1) 위반")
        pd.infer_noise_broadcast = True
    if bc_target == "gt":
        dh.plan_sft_key = None
        cfg.data.train.pop("plan_sft_target_file", None)
    elif bc_target != "sft":
        raise ValueError(f"bc_target {bc_target} — gt|sft")
    return cfg


def _plain_state_dict(ck):
    sd = ck.get("state_dict", ck)
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def strict_load(model, ckpt_path):
    """빠진 키 없음, 남는 키는 비저장 버퍼만(값은 config 로 만든 상수와 같아야 함)."""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _plain_state_dict(ck)
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys:
        raise RuntimeError(f"{ckpt_path}: 빠진 키 {len(res.missing_keys)} 개 {res.missing_keys[:8]}")
    persistent = set(model.state_dict().keys())
    buffers = dict(model.named_buffers())
    bad = [k for k in res.unexpected_keys if k in persistent or k not in buffers]
    if bad:
        raise RuntimeError(f"{ckpt_path}: 모델에 없는 키 {len(bad)} 개 {bad[:8]}")
    for k in res.unexpected_keys:
        if not torch.equal(sd[k].float().cpu(), buffers[k].float().cpu()):
            raise RuntimeError(f"{ckpt_path}: 비저장 버퍼 {k} 저장값이 config 값과 다르다")
    return ck.get("meta", {}) if isinstance(ck, dict) else {}


def build_rl_model(cfg, ckpt_path, device="cuda:0"):
    """→ det(MMDet3D detector, eval, device). 플래그 미적용 소스면 즉시 실패."""
    import torch
    from mmcv.runner import wrap_fp16_model
    from mmdet.models import build_detector
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    dec = model.head.onedecoder_head
    need = [("force_stream_groups", dec), ("plan_grad_isolation", dec), ("infer_noise_broadcast", dec.plan_diffusion)]
    miss = [n for n, o in need if not hasattr(o, n)]
    if miss:
        raise RuntimeError(f"모델 소스에 GRPO 플래그 {miss} 가 없다 — docs/grpo_model_flag_20261109.diff 적용 필요")
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
    meta = strict_load(model, ckpt_path)
    model = model.to(device).eval()
    if not dec.is_init_bank_list:
        dec.init_instance_bank_list()
    pdec = dec.plan_decoder
    if list(pdec.anchor_types) != [("temp", "2hz")] or pdec.num_group != 1:
        raise RuntimeError(f"앵커 묶음 {pdec.anchor_types}, num_group {pdec.num_group} — ('temp','2hz') 하나 가정")
    if dec.ego_fut_ts != T or dec.ego_fut_mode != N_MODE or pdec.ego_fut_cmd != 3:
        raise RuntimeError(f"ego_fut_ts {dec.ego_fut_ts}, mode {dec.ego_fut_mode}, cmd {pdec.ego_fut_cmd} — 8/16/3 가정")
    if not pdec.with_rescore:
        raise RuntimeError("with_rescore=False — 배포 선택 규칙(§9.1)과 다르다")
    if dec.num_stream_groups <= 1:
        raise RuntimeError("num_stream_groups ≤ 1 — 스트림 그룹 6 가정(§9.2)")
    model._grpo_meta = meta
    return model


def select_trainable(model):
    """§9.7 학습 대상만 requires_grad True. 나머지(지도 뱅크 instance_feature 등 feat_grad 포함) 전부 False."""
    names = []
    pats = [re.compile(p) for p in TRAINABLE]
    hit = {p.pattern: 0 for p in pats}
    for n, p in model.named_parameters():
        ok = False
        for pt in pats:
            if pt.search(n):
                ok = True
                hit[pt.pattern] += 1
        p.requires_grad_(ok)
        if ok:
            names.append(n)
    empty = [k for k, v in hit.items() if v == 0]
    if empty:
        raise RuntimeError(f"학습 대상 패턴에 걸린 매개변수가 없다: {empty}")
    return names


# ---------------------------------------------------------------- 데이터
def rl_dataset_cfg(cfg, ann_file=None, bc_target="gt"):
    ds = copy.deepcopy(cfg.data.train)
    pipe = [copy.deepcopy(p) for p in ds.pipeline if p["type"] not in RL_DROP]
    col = [p for p in pipe if p["type"] == "Collect"]
    if len(col) != 1:
        raise RuntimeError("RL 파이프라인에 Collect 가 하나여야 한다")
    keys = list(RL_KEYS) + (["gt_ego_plan_sft_2hz"] if bc_target == "sft" else [])
    col[0]["keys"] = keys
    col[0]["meta_keys"] = list(RL_META)
    ds.pipeline = pipe
    if ann_file is not None:
        ds.ann_file = ann_file
    if os.path.basename(ds.ann_file) != "navsim_navtrain.pkl":
        raise RuntimeError(f"ann_file {ds.ann_file} — GRPO 롤아웃은 navtrain 전용(navtest·navhard 금지)")
    return ds


class SeqSubset:
    """GroupInBatchSampler 용 장면 부분집합(덤프의 G1 1k·내부 보류 집합). 고른 장면의 시퀀스 그룹만 남긴다."""

    def __init__(self, ds, keep_idx):
        self.ds = ds
        self.indices = np.asarray(sorted(keep_idx), np.int64)
        old = np.asarray(ds.flag)[self.indices]
        _, self.flag = np.unique(old, return_inverse=True)
        self.flag = self.flag.astype(np.int64)
        self.keep_consistent_seq_aug = ds.keep_consistent_seq_aug
        self.data_infos = [ds.data_infos[i] for i in self.indices]

    def get_augmentation(self):
        return self.ds.get_augmentation()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        item = dict(item)
        item["idx"] = int(self.indices[item["idx"]])
        return self.ds[item]


def token_hash(s):
    return int.from_bytes(hashlib.sha256(str(s).encode()).digest()[:8], "little") >> 1


def scene_select(infos, rule):
    """rule: 'all' | 'hash_first:N'(장면 해시 오름차순 첫 N 장면) | 'hash_mod:m:r'(해시 % m == r 인 장면) → 프레임 색인 목록."""
    if rule == "all":
        return None
    by_scene = {}
    for i, inf in enumerate(infos):
        by_scene.setdefault(inf["scene_token"], []).append(i)
    if rule.startswith("hash_first:"):
        n = int(rule.split(":")[1])
        sc = sorted(by_scene, key=token_hash)[:n]
    elif rule.startswith("hash_mod:"):
        _, m, r = rule.split(":")
        sc = [s for s in by_scene if token_hash(s) % int(m) == int(r)]
    else:
        raise ValueError(f"scene_select 규칙 {rule}")
    return sorted(i for s in sc for i in by_scene[s])


def build_rl_dataset(cfg, ann_file=None, bc_target="gt", select="all"):
    from mmdet.datasets import build_dataset
    ds = build_dataset(rl_dataset_cfg(cfg, ann_file, bc_target))
    if ds.metadata.get("kind") != "train_hist":
        raise RuntimeError(f"info kind {ds.metadata.get('kind')} != train_hist")
    orig = ds.get_augmentation

    def det_aug():                                   # test 분기 값(결정적) — 위 docstring
        ds.test_mode = True
        try:
            return orig()
        finally:
            ds.test_mode = False
    ds.get_augmentation = det_aug
    keep = scene_select(ds.data_infos, select)
    return ds if keep is None else SeqSubset(ds, keep)


def build_rl_loader(ds, batch, groups, workers, seed, fast_forward=0):
    from projects.mmdet3d_plugin.models.loader.builder import build_dataloader
    ld = build_dataloader(ds, batch, workers, num_gpus=1, dist=False, seed=seed, runner_type="IterBasedRunner",
                          num_stream_groups=groups, persistent_workers=workers > 0)
    if fast_forward:
        ld.batch_sampler.fast_forward(fast_forward)
    return ld


def frame_table(infos):
    """프레임 token → (navsim_token 또는 None, is_eval, scene_token, log_name)."""
    return {inf["token"]: (inf.get("navsim_token") if inf.get("is_eval_frame") else None, bool(inf.get("is_eval_frame")),
                           inf["scene_token"], inf.get("log_name")) for inf in infos}


def metric_cache_tokens(cache_dir):
    """navsim_score_vocab_navtrain.cache_paths 와 같은 규칙(순수 파이썬)."""
    md = Path(cache_dir) / "metadata"
    csvs = sorted(p for p in md.iterdir() if p.suffix == ".csv")
    if len(csvs) != 1:
        raise RuntimeError(f"{md}: metadata csv 가 정확히 1개여야 한다({len(csvs)})")
    return {r.split("/")[-2] for r in csvs[0].read_text().splitlines()[1:]}


def to_device(data, device):
    from mmcv.parallel import scatter
    import torch
    dev = torch.device(device)
    return scatter(data, [dev.index if dev.index is not None else 0])[0]


# ---------------------------------------------------------------- 순전파·선택
def forward_heads(det, data, grad):
    """→ (model_outs, data). extract_feat 는 no_grad, head 는 grad 여부 지정."""
    import torch
    img = data["img"]
    with torch.no_grad():
        feats = det.extract_feat(img)
    data = det._maybe_extract_bev(data)
    data = det._maybe_load_qwen_cache(data)
    with torch.set_grad_enabled(grad):
        outs = det.head(img, feats, data)
    return outs, data


def plan_bundle(det, outs, data):
    """명령 묶음 16 모드: raw16 (B,16), mu16 (B,16,8,2) 누적(그래프 유지), cmd (B,). decode/select 와 같은 변환."""
    import torch
    pdec = det.head.onedecoder_head.plan_decoder
    plan_out = outs[3]
    pred, cls = plan_out["prediction"][-1], plan_out["classification"][-1]
    B = cls.shape[0]
    cls = cls.reshape(B, pdec.ego_fut_cmd, -1)
    reg = pred.reshape(B, pdec.ego_fut_cmd, -1, T, 2).cumsum(dim=-2)
    cmd = data["gt_ego_fut_cmd"].argmax(dim=-1)
    bi = torch.arange(B, device=cls.device)
    return cls[bi, cmd], reg[bi, cmd], cmd


def rescore16(det, outs, raw16, mu16):
    """배포 select 의 rescore 와 같은 입력(no_grad). → rescore 후 logit (B,16)."""
    import torch
    pdec = det.head.onedecoder_head.plan_decoder
    det_out, motion_out = outs[0], outs[4]
    if not (len(det_out["prediction"]) > 0 and len(motion_out["prediction"]) > 0):
        raise RuntimeError("det/motion 출력이 없다 — rescore 불가(배포 규칙과 다름)")
    with torch.no_grad():
        det_cls = det_out["classification"][-1].sigmoid()
        det_anchors = det_out["prediction"][-1]
        det_conf = det_cls.max(dim=-1).values
        motion_cls = motion_out["classification"][-1].sigmoid()
        motion_reg = motion_out["prediction"][-1].cumsum(-2)
        resc, _ = pdec.rescore(raw16.detach(), mu16.detach(), motion_cls, motion_reg, det_anchors, det_conf,
                               ego_fut_mode=mu16.shape[1])
    return resc


def top_modes(resc, raw, M=4):
    """(16,) numpy → 상위 M 색인: rescore 후 내림차순, 같으면 원시 logit 내림차순, 그래도 같으면 색인 오름차순."""
    order = np.lexsort((np.arange(len(resc)), -np.asarray(raw, np.float64), -np.asarray(resc, np.float64)))
    return order[:M]


def frame_metas(data):
    metas = data["img_metas"]
    if isinstance(metas, list) and metas and isinstance(metas[0], (list, tuple)):
        metas = metas[0]
    return metas


def policy_candidates(mu16, top, tok, cl, s_grid, sig, run_seed, step, G=8, axis="lane"):
    """μ16 (16,8,2) numpy(detach) → (cands (16+M·G,8,2), E_top (M,8,2,2), eta (M,G,8,2), info)."""
    if axis == "lane":
        E16, info = ar1.axis_basis(mu16, cl["xyh"][:, :2].astype(np.float64), s_grid=s_grid, valid=cl["valid"])
    elif axis == "ego":
        E16, info = ar1.ego_axis_basis(mu16), None
    else:
        raise ValueError(axis)
    eta = np.stack([ar1.sample_eta(run_seed, step, tok, int(q), G) for q in top])
    tau = np.stack([ar1.sample_tau(mu16[q], E16[q], sig, eta[m]) for m, q in enumerate(top)])
    cands = np.concatenate([mu16, tau.reshape(-1, T, 2)], 0)
    return cands, E16, eta, info
