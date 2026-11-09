# 수정(2026-10-15): B 융합 — 카메라+LiDAR / 카메라+Radar 실험용 데이터셋(새 파일).
#   NuScenes3DDataset 를 그대로 상속하고, tools/data_converter/nuscenes_fusion_infos.py 가 만든
#   data/infos/nuscenes_fusion_infos_{train,val}.pkl 의 'sweeps'(LiDAR 직전 sweep)·'radars'(5 레이더 × sweep) 를
#   파이프라인 입력 dict 로 넘기기만 한다. 실제 점 읽기는 config 가 넣은 transform 만 한다
#   (loader/pipelines/loading_points.py) — 카메라+Radar 런은 LiDAR sweep 을 읽지 않는다.
from mmdet.datasets import DATASETS

from .dataset import NuScenes3DDataset


@DATASETS.register_module()
class NuScenes3DFusionDataset(NuScenes3DDataset):
    def get_data_info(self, index):
        input_dict = super().get_data_info(index)
        info = self.data_infos[index]
        if "sweeps" not in info or "radars" not in info:
            raise KeyError(
                "NuScenes3DFusionDataset: info 에 'sweeps'/'radars' 가 없다. ann_file 을 "
                "data/infos/nuscenes_fusion_infos_*.pkl 로 지정할 것(tools/data_converter/nuscenes_fusion_infos.py).")
        input_dict["lidar_sweeps"] = info["sweeps"]
        input_dict["radar_sweeps"] = info["radars"]
        return input_dict
