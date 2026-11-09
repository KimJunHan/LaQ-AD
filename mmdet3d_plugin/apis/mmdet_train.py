# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    HOOKS,
    DistSamplerSeedHook,
    EpochBasedRunner,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
    get_dist_info,
)
from mmcv.utils import build_from_cfg

from mmdet.core import EvalHook

from mmdet.datasets import build_dataset, replace_ImageToTensor
from mmdet.utils import get_root_logger
import time
import os.path as osp
from ..models.loader.builder import build_dataloader  #수정(2026-10-15): 경로 이동 datasets/builder.py → models/loader/builder.py (상대 import 유지)
from ..core.evaluation.eval_hooks import (  #수정(2026-09-15): 절대->상대. 패키지가 두 이름(projects.*, 최상위)으로 불려도 이중 등록이 없도록
    CustomDistEvalHook,
)
from ..models.loader.builder import custom_build_dataset  #수정(2026-10-15): 경로 이동(상대 import 유지)
from ..models.loader.samplers import GroupInBatchSampler  # 수정(2026-10-15): 재개 시 샘플러 상태 복원


def custom_train_detector(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
    meta=None,
):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders

    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # assert len(dataset)==1s
    if "imgs_per_gpu" in cfg.data:
        logger.warning(
            '"imgs_per_gpu" is deprecated in MMDet V2.0. '
            'Please use "samples_per_gpu" instead'
        )
        if "samples_per_gpu" in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f"={cfg.data.imgs_per_gpu} is used in this experiments"
            )
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f"{cfg.data.imgs_per_gpu} in this experiments"
            )
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    if "runner" in cfg:
        runner_type = cfg.runner["type"]
    else:
        runner_type = "EpochBasedRunner"
    # 수정(2026-10-15): 실험 C — 스트림 그룹(cfg.data.num_stream_groups, 기본 1 = 종전). G>1 이면 optimizer step 1회가
    #   그룹 0..G-1 을 정확히 한 번씩 담도록 기울기 누적 횟수와 같아야 하고, 디코더도 같은 G 로 뱅크 상태를 나눠야 한다.
    num_stream_groups = int(cfg.data.get("num_stream_groups", 1))
    if num_stream_groups > 1:
        cum = cfg.optimizer_config.get("cumulative_iters", 1)
        assert runner_type == "IterBasedRunner", "num_stream_groups>1 은 GroupInBatchSampler(IterBasedRunner) 전용"
        assert cum == num_stream_groups, \
            f"data.num_stream_groups={num_stream_groups} 는 optimizer_config.cumulative_iters={cum} 와 같아야 한다"
        dec_g = cfg.model.head.onedecoder_head.get("num_stream_groups", 1)
        assert dec_g == num_stream_groups, \
            f"model.head.onedecoder_head.num_stream_groups={dec_g} ≠ data.num_stream_groups={num_stream_groups}"
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            nonshuffler_sampler=dict(
                type="DistributedSampler"
            ),  # dict(type='DistributedSampler'),
            runner_type=runner_type,
            num_stream_groups=num_stream_groups,  # 수정(2026-10-15): 실험 C
            # 수정: config/cfg-options 의 persistent_workers·prefetch_factor 를 DataLoader 까지 전달
            #       (기존엔 미전달=no-op). batch16 worker RAM 크래시 대응: prefetch_factor=1 로 버퍼 절반.
            persistent_workers=cfg.data.get("persistent_workers", False),
            prefetch_factor=cfg.data.get("prefetch_factor", 2),
        )
        for ds in dataset
    ]

    # 수정(2026-10-30): 4-GPU DDP 1단계 — mmdet ResNet 의 with_cp 는 재진입(reentrant) torch.utils.checkpoint 라
    #   DDP(find_unused_parameters=True)에서 "Expected to mark a variable ready only once"(img_backbone.layer4.2.bn3.weight)로 멈춘다.
    #   config backbone_cp_non_reentrant=True 일 때만 같은 함수를 use_reentrant=False 로 부르게 한다. 순전파·역전파 수식·재계산 시
    #   BN running stats 2회 갱신·RNG 보존은 재진입 방식과 같다(체크포인트 구현 방식만 다름). 1 GPU 기존 config 는 해당 없음.
    if cfg.get("backbone_cp_non_reentrant", False):
        import functools
        import torch.utils.checkpoint as _tcp
        import mmdet.models.backbones.resnet as _mmdet_resnet

        class _NonReentrantCP:
            checkpoint = staticmethod(functools.partial(_tcp.checkpoint, use_reentrant=False))

        _mmdet_resnet.cp = _NonReentrantCP
        logger.info("backbone_cp_non_reentrant: mmdet ResNet with_cp → torch.utils.checkpoint(use_reentrant=False)")

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )

    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids
        )

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if "runner" not in cfg:
        cfg.runner = {
            "type": "EpochBasedRunner",
            "max_epochs": cfg.total_epochs,
        }
        warnings.warn(
            "config is now expected to have a `runner` section, "
            "please set `runner` in your config.",
            UserWarning,
        )
    else:
        if "total_epochs" in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
        ),
    )

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        # 수정(2026-10-16): fp16 넘침 원인 수정 — 종전 Fp16OptimizerHook(**optimizer_config) 은 (1) optimizer_config 에
        #   type="GradientCumulativeOptimizerHook"·cumulative_iters 가 있으면 TypeError 로 시작조차 못 했고(기울기 누적 미지원),
        #   (2) torch>=1.6 에서 GradScaler.step 이 inf/NaN 기울기 step 을 조용히 건너뛴다(정적 scale 이어도). 같은 계산
        #   (×loss_scale → backward → unscale → clip → step)에 기울기 누적을 지원하고, 넘침 시 건너뛰지 않고 멈추는 훅으로 바꾼다.
        from .fp16_strict_hook import StrictGradientCumulativeFp16OptimizerHook
        _oc = {k: v for k, v in cfg.optimizer_config.items() if k != "type"}
        _t = cfg.optimizer_config.get("type", None)
        assert _t in (None, "OptimizerHook", "GradientCumulativeOptimizerHook"), f"fp16 와 함께 쓸 수 없는 optimizer hook: {_t}"
        optimizer_config = StrictGradientCumulativeFp16OptimizerHook(
            **_oc, **fp16_cfg, distributed=distributed
        )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )

    # register profiler hook
    # trace_config = dict(type='tb_trace', dir_name='work_dir')
    # profiler_config = dict(on_trace_ready=trace_config)
    # runner.register_profiler_hook(profiler_config)

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            assert False
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline
            )
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            nonshuffler_sampler=dict(type="DistributedSampler"),
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_cfg["jsonfile_prefix"] = osp.join(
            "val",
            cfg.work_dir,
            time.ctime().replace(" ", "_").replace(":", "_"),
        )
        eval_hook = CustomDistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get("custom_hooks", None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(
            custom_hooks, list
        ), f"custom_hooks expect list type, but got {type(custom_hooks)}"
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), (
                "Each item in custom_hooks expects dict type, but got "
                f"{type(hook_cfg)}"
            )
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop("priority", "NORMAL")
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
        # 수정(2026-11-09): 학습 속도 근본 수정(docs/navsim_s1_divergence_20261109.md §8 원인 1) — mmcv runner.resume 은
        #   체크포인트를 map_location=cuda 로 읽어 Adam/AdamW 의 state["step"](스칼라)까지 GPU 로 올린다. torch 의 비-fused·
        #   비-capturable 경로는 step 을 CPU 텐서로 만들고 bias correction 에서 step.item() 을 부르므로, GPU 에 있으면 파라미터마다
        #   동기화가 생긴다(1,588 그룹 × 2 ≈ 3,180 회/step, step 당 약 1.1 s). torch 기본 배치(CPU)로 되돌린다 — 값·dtype 동일.
        #   fused·capturable 그룹은 torch 가 step 을 파라미터 장치에 두는 것이 맞으므로 건드리지 않는다.
        if isinstance(runner.optimizer, torch.optim.Optimizer):
            _moved = 0
            for _g in runner.optimizer.param_groups:
                if _g.get("fused") or _g.get("capturable"):
                    continue
                for _p in _g["params"]:
                    _st = runner.optimizer.state.get(_p)
                    if _st and torch.is_tensor(_st.get("step")) and _st["step"].device.type != "cpu":
                        _st["step"] = _st["step"].cpu()
                        _moved += 1
            if _moved:
                logger.info(f"resume: optimizer state['step'] {_moved} 개를 CPU 로 옮김(비-fused·비-capturable, torch 기본 배치)")
        # 수정(2026-10-15): 재개 시 샘플러 상태 복원 — runner.resume 이 ckpt meta 의 iter K(micro-iter)를 읽은 직후,
        #   runner.run 이 IterLoader(iter(dataloader)) 로 worker prefetch 를 시작하기 전에 GroupInBatchSampler 를
        #   K 배치만큼 인덱스만 앞으로 감는다 → 재개 첫 배치 = 무중단 런의 iter K 배치(G=1·G>1 공통). 새 런(resume 없음)은 무변경.
        if runner.iter > 0:
            for dl in data_loaders:
                if isinstance(getattr(dl, "batch_sampler", None), GroupInBatchSampler):
                    dl.batch_sampler.fast_forward(runner.iter)
                    logger.info(f"GroupInBatchSampler fast-forwarded {runner.iter} batches for resume")
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
