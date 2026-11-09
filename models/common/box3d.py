# 수정(2026-10-15): 경로 이동 mmdet3d_plugin/core/box3d.py → models/common/box3d.py (소스 정리).
#   3D 박스 상태 벡터 인덱스 상수. mmdet3d_plugin/core/box3d.py 는 이 파일을 다시 내보낸다(하위 호환).
X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))  # undecoded
CNS, YNS = 0, 1  # centerness and yawness indices in quality
YAW = 6  # decoded

__all__ = ["X", "Y", "Z", "W", "L", "H", "SIN_YAW", "COS_YAW", "VX", "VY", "VZ", "CNS", "YNS", "YAW"]
