"""추론 결과 pkl 로부터 open-loop 지표만 재산출한다.

수정: 추론(5172 샘플, ~30분)은 이미 끝나 results.pkl 에 저장돼 있는데
      포맷/평가 단계만 실패한 경우(예: motion 없는 stage1 의 trajs_3d KeyError),
      추론을 다시 돌리는 것은 순수 낭비다. tools/test.py 의 평가 단계만 떼어 재실행한다.
      test.py 의 eval 경로(cfg.evaluation → dataset.evaluate)와 동일한 인자를 쓴다.

사용법:
    python tools/eval_from_pkl.py <config> <results.pkl> [--eval bbox]
"""
import argparse

import mmcv
from mmcv import Config, DictAction
from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="results.pkl 로부터 평가 지표만 재산출")
    parser.add_argument("config", help="학습에 쓴 config 경로")
    parser.add_argument("pkl", help="tools/test.py --out 으로 저장된 결과 pkl")
    parser.add_argument("--eval", type=str, nargs="+", default=["bbox"])
    #수정(2026-08-25): test.py 와 동일하게 --cfg-options 를 받는다.
    #  없으면 evaluation.jsonfile_prefix 가 config 기본값("val/")로 남아 산출물이
    #  evaluation/<run>/ 이 아닌 엉뚱한 곳에 떨어진다. test.py 로 돌린 평가와 출력 경로를
    #  맞춰야 같은 run 의 결과로 비교·보관할 수 있다.
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # config 의 plugin 을 등록해야 커스텀 dataset/모듈이 레지스트리에 잡힌다 (test.py 와 동일)
    if cfg.get("plugin", False):
        import importlib
        import os.path as osp
        import sys

        plugin_dir = cfg.get("plugin_dir", "projects/mmdet3d_plugin/")
        sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
        module_path = plugin_dir.rstrip("/").replace("/", ".")
        importlib.import_module(module_path)

    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    outputs = mmcv.load(args.pkl)
    print(f"[eval] {args.pkl} 로드 완료 — {len(outputs)} 샘플")

    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.eval))
    print(f"[eval] eval_kwargs = {eval_kwargs}")

    results_dict = dataset.evaluate(outputs, **eval_kwargs)
    print(results_dict)


if __name__ == "__main__":
    main()
