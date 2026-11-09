# 수정(2026-11-09): C3 GRPO 검토 B (6) — 평가 전 적재 검사. 신규 파일(기존 파일 무수정).
"""평가(tools/test.py) 전에 체크포인트가 평가 config 의 모델에 **엄격히** 맞는지 CPU 로 확인한다(rollout.strict_load 와 같은 규칙:
빠진 키 0, 남는 키는 값이 config 상수와 같은 비저장 버퍼만, 모양 불일치는 load_state_dict 가 예외).
tools/test.py 의 load_checkpoint(strict=False)가 조용히 넘기는 키 누락(예: GRPO 학습 대상 일부만 저장된 체크포인트)을 막는다.
사용: CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tools/navsim/grpo/check_load.py --config $TEST_CFG --ckpt final_ema.pth
종료 코드 0 = 통과, 1 = 실패(사유 출력).
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import rollout as RO  # noqa: E402


def check(config, ckpt):
    from mmcv.runner import wrap_fp16_model
    from mmdet.models import build_detector
    cfg = RO.load_model_cfg(config, bc_target="gt", flags=False)
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
    meta = RO.strict_load(model, ckpt)
    return dict(ok=True, config=config, ckpt=os.path.abspath(ckpt), n_params=len(model.state_dict()),
                meta_keys=sorted(meta) if isinstance(meta, dict) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    os.chdir(ROOT)
    try:
        r = check(a.config, a.ckpt)
    except (RuntimeError, KeyError) as e:
        print(json.dumps(dict(ok=False, error=f"{type(e).__name__}: {e}"), ensure_ascii=False))
        sys.exit(1)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
