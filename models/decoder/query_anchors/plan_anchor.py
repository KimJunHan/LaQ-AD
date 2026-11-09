# 수정(2026-10-15, 2차): 아키텍처 그림의 query anchors 상자 — 계획 앵커("cmd anchors 3×16") 적재.
#   PlanningInstanceBank.__init__ 에 있던 k-means 앵커 파일 적재·펼침 코드를 그대로 옮겼다(값·순서 동일).
#   nuScenes: nusc_plan_temp_3x16.npy = (명령 3, 모드 16, T, 2) → (48, T*2). 앵커 유형(temp/spat/speed)별로 이어 붙인다.
import numpy as np

__all__ = ["load_plan_anchors"]


def load_plan_anchors(anchor_paths, anchor_types, anchor_scales=None):
    if anchor_scales is None:
        anchor_scales = [1.0 for _ in range(len(anchor_types))]

    assert anchor_paths is not None
    assert len(anchor_types) == len(anchor_scales)

    anchor_dict = dict()
    if isinstance(anchor_paths, str):
        for anchor_type in anchor_types:
            anchor_dict[anchor_type] = anchor_paths
    elif isinstance(anchor_paths, list):
        for anchor_type, anchor_path in zip(anchor_types, anchor_paths):
            anchor_dict[anchor_type] = anchor_path
    elif isinstance(anchor_paths, dict):
        anchor_dict = anchor_paths
    else:
        raise NotImplementedError

    anchors = []
    for anchor_type, anchor_scale in zip(anchor_types, anchor_scales):
        anchor = np.load(anchor_dict[anchor_type])

        if len(anchor.shape) == 3:  # for planning
            anchor = anchor.reshape(anchor.shape[0], -1)
        elif len(anchor.shape) == 4:  # (cmd, mode, T, 2) → (cmd*mode, T*2)
            anchor = anchor.reshape(anchor.shape[0] * anchor.shape[1], -1)

        anchor = anchor * anchor_scale
        anchors.append(anchor)

    return np.concatenate(anchors, axis=0)
