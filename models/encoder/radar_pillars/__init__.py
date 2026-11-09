# 수정(2026-10-15): B 융합 — Radar 기둥 BEV 인코더(카메라+Radar 실험).
#   구조는 LiDAR 와 같은 PillarBEVEncoder(동적 기둥 + PFN + 2D 백본)를 쓰고, config 에서 입력 7열
#   (x,y,z,rcs,vx_comp,vy_comp,time_lag)·거친 기둥(0.8 m)·얕은 백본으로 둔다. 레지스트리 type 이름만 따로 둔다.
from mmdet.models import BACKBONES

from ..lidar_pointpillars.pillar_bev import PillarBEVEncoder


@BACKBONES.register_module()
class RadarPillarBEVEncoder(PillarBEVEncoder):
    def __init__(self, in_channels=7, points_key="radar_points", **kwargs):
        super().__init__(in_channels=in_channels, points_key=points_key, **kwargs)
