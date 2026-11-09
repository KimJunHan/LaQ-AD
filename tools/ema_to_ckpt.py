# 수정(2026-10-30): EMASidecarHook(mmdet3d_plugin/apis/ema_sidecar_hook.py)이 저장한 ema_iter_<N>.pth 를 평가용 체크포인트로 만든다.
#   같은 N 의 iter_<N>.pth 를 읽어 state_dict 의 부동소수 텐서만 EMA 값으로 바꾸고(키·모양·dtype 일치 검사), 정수 버퍼·meta 는
#   원 체크포인트 그대로 둔다. optimizer 상태는 뺀다(평가 전용). 원 파일은 읽기만 한다.
#   사용: python tools/ema_to_ckpt.py <work_dir> <N> [--out <path>]   (기본 출력 <work_dir>/iter_<N>_ema.pth)
import argparse
import os.path as osp

import torch

ap = argparse.ArgumentParser()
ap.add_argument("work_dir")
ap.add_argument("iter", type=int)
ap.add_argument("--out", default=None)
# 수정(2026-10-31): 2단계 스모크에서 발견 — mmcv 1.7 save_checkpoint(get_state_dict)는 persistent=False 버퍼(확산 일정 sqrt_ab 등 상수)도
#   체크포인트에 넣지만, EMASidecarHook 은 model.state_dict()(비저장 버퍼 제외)만 추적한다 → 체크포인트에만 있는 부동소수 키가 생긴다.
#   이 키들은 평가 시 load_state_dict 가 읽지 않는다(모델 state_dict 에 없음). --config 를 주면 모델을 만들어 그 키들이 정말
#   비저장 버퍼인지 확인하고(아니면 멈춤), 값은 원 체크포인트 그대로 둔다. --config 없이 그런 키가 있으면 멈춘다.
ap.add_argument("--config", default=None, help="체크포인트에만 있는 키가 비저장 버퍼인지 확인할 학습 config")
a = ap.parse_args()
ck = torch.load(osp.join(a.work_dir, f"iter_{a.iter}.pth"), map_location="cpu")
em = torch.load(osp.join(a.work_dir, f"ema_iter_{a.iter}.pth"), map_location="cpu")
assert em["iter"] == a.iter == ck["meta"]["iter"] + 1, (em["iter"], a.iter, ck["meta"].get("iter"))  # mmcv: iter_<N>.pth 의 meta iter = N-1
sd = ck["state_dict"]
fl_all = [k for k, v in sd.items() if v.is_floating_point()]
ek = list(em["state_dict"].keys())
extra = [k for k in fl_all if k not in set(ek)]
fl = [k for k in fl_all if k in set(ek)]
assert fl == ek, "EMA 키가 체크포인트의 부동소수 키와 다르다(EMA 에만 있는 키 또는 순서 차이)"
if extra:
    assert a.config, f"체크포인트에만 있는 부동소수 키 {len(extra)}개 — --config 로 비저장 버퍼인지 확인해야 한다: {extra[:5]}"
    import importlib, os, sys
    sys.path.insert(0, os.getcwd())
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmdet.models import build_detector
    cfg = Config.fromfile(a.config)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    importlib.import_module(os.path.dirname(cfg.plugin_dir).replace("/", "."))
    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    persistent = set(model.state_dict().keys())
    buffers = dict(model.named_buffers())
    bad = [k for k in extra if k in persistent or k not in buffers]
    assert not bad, f"체크포인트에만 있는 키 중 비저장 버퍼가 아닌 것: {bad}"
    assert set(ek) == {k for k, v in model.state_dict().items() if v.is_floating_point()}, "EMA 키가 모델의 부동소수 state_dict 와 다르다"
    for k in extra:
        assert torch.equal(sd[k].float(), buffers[k].float()), f"비저장 버퍼 {k} 의 저장값이 config 로 만든 값과 다르다"
    print(f"[ema_to_ckpt] 비저장 버퍼 {len(extra)}개는 원 값 유지(평가 시 읽지 않음, 값은 config 로 만든 상수와 같음): {extra}")
for k in fl:
    e = em["state_dict"][k]
    assert e.shape == sd[k].shape, (k, e.shape, sd[k].shape)
    assert torch.isfinite(e).all(), f"EMA 값이 유한하지 않다: {k}"
    sd[k] = e.to(sd[k].dtype)
meta = dict(ck["meta"])
meta["ema"] = dict(source=osp.abspath(osp.join(a.work_dir, f"ema_iter_{a.iter}.pth")), n_updates=em["n_updates"],
                   decay=em["decay"], ramp_steps=em["ramp_steps"], update_every=em["update_every"])
out = a.out or osp.join(a.work_dir, f"iter_{a.iter}_ema.pth")
# 수정(2026-10-31): 독립 검토 지적 — 쓰는 도중 끊기면 깨진 파일이 남아 재실행 때 변환을 건너뛴다 → 임시 파일 + os.replace(원자적 교체)
import os
torch.save(dict(meta=meta, state_dict=sd), out + ".tmp")
os.replace(out + ".tmp", out)
print(f"[ema_to_ckpt] {out}: 부동소수 텐서 {len(fl)}개를 EMA 로 교체 (갱신 {em['n_updates']} 회)")
