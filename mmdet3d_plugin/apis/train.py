# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

# 수정(2026-10-15): 소스 정리 — 쓰이지 않던 train_model(mmdet train_detector 래퍼) 제거.
from .mmdet_train import custom_train_detector


def custom_train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
    meta=None,
):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ["EncoderDecoder3D"]:
        assert False
    else:
        custom_train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta,
        )

