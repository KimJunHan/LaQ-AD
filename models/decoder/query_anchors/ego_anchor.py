# 수정(2026-10-15, 2차): 아키텍처 그림의 query anchors 상자 — 자차 앵커(11차원 박스 1개). EgoInstanceBank 에서 옮겼다(값 동일).
import numpy as np

__all__ = ["ego_anchor"]


def ego_anchor(anchor_type='nus', custom_anchor=None):
    nus_ego_anchor = [[0, 0.5, -1.84 + 1.56/2, np.log(4.08), np.log(1.73), np.log(1.56), 1, 0, 0, 0, 0]]
    b2d_ego_anchor = [[0, 0.5, -1.84 + 1.49/2, np.log(4.89), np.log(1.84), np.log(1.49), 1, 0, 0, 0, 0]]
    if anchor_type == 'nus':
        return nus_ego_anchor
    elif anchor_type == 'b2d':
        return b2d_ego_anchor
    elif anchor_type == 'custom':
        #수정(2026-10-09): NAVSIM 자차(nuPlan Pacifica)·라이다 배치는 nus/b2d 와 다르다. 값은 데이터 변환기
        #   (tools/navsim/navsim_devkit_steps.py egometa)가 nuplan 차량 제원 + lidar2ego 로 계산해 출력한다.
        if custom_anchor is None or len(custom_anchor) != 11:
            raise ValueError("anchor_type='custom' 은 11차원 custom_anchor 가 필요하다.")
        return [list(custom_anchor)]
    raise NotImplementedError(f"anchor_type={anchor_type}")
