# 수정(2026-10-15): 경로 이동 mmdet3d_plugin/datasets/pipelines → models/loader/pipelines. 미사용 qa_loading(LoadDriveLMQA, LoadQwenInput)은 archive 로 이동.
from .transform import (
    InstanceNameFilter,
    BEVObjectRangeFilter,
    CircleObjectRangeFilter,
    NormalizeMultiviewImage,
    NuScenesSparse4DAdaptor,
    MultiScaleDepthMapGenerator,
    B2DMultiScaleDepthMapGenerator,
    ObjectHeightRangeFilter,  # 수정(2026-11-09): NAVSIM GT z 범위 필터(docs/navsim_gt_filter_20261109.md)
    CameraFOVObjectFilter,    # 수정(2026-11-09): NAVSIM GT 카메라 시야 필터(docs/navsim_gt_filter_20261109.md)
)
from .augment import (
    ResizeCropFlipImage,
    BBoxRotation,
    PhotoMetricDistortionMultiViewImage,
)
from .loading import (
    LoadPointsFromFile,
    B2DLoadPointsFromFile,
    LoadMultiViewImageFromFiles,
)
# 수정(2026-10-15, 2차): VL 토큰 캐시 읽기는 encoder/vl_tokens_cache 에 있다(아키텍처 그림 기준). 여기서 다시 내보낸다.
from ...encoder.vl_tokens_cache.loading import LoadQwenCacheNusc
from ...encoder.vl_tokens_cache.loading import LoadTeacherCacheNusc  # 수정(2026-10-16): 교사 캐시 7 폴더 · 토큰 유형 임베딩

from .vectorize import VectorizeMap, VectorizePloyLine
# 수정(2026-10-15): B 융합 — LiDAR/Radar 다중 sweep 점 읽기(새 파일, 융합 config 만 사용)
from .loading_points import (
    LoadPointsFromMultiSweepsFusion,
    LoadRadarPointsMultiSweeps,
    PointsRangeFilterFusion,
    FusionPointsToTensor,
)

__all__ = [
    "InstanceNameFilter",
    "ResizeCropFlipImage",
    "BBoxRotation",
    "CircleObjectRangeFilter",
    "MultiScaleDepthMapGenerator",
    "NormalizeMultiviewImage",
    "PhotoMetricDistortionMultiViewImage",
    "NuScenesSparse4DAdaptor",
    "LoadMultiViewImageFromFiles",
    "LoadPointsFromFile",
    "VectorizeMap",
    "VectorizePloyLine",
    "B2DLoadPointsFromFile",
    "B2DMultiScaleDepthMapGenerator",
    "LoadQwenCacheNusc",
    "LoadTeacherCacheNusc",  # 수정(2026-10-16): 교사 캐시 7 폴더 · 토큰 유형 임베딩
    "ObjectHeightRangeFilter",  # 수정(2026-11-09)
    "CameraFOVObjectFilter",    # 수정(2026-11-09)
]
