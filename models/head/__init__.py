# 수정(2026-10-15, 2차): 헤드 — SparseHead(과제별 손실·후처리 디스패치)와 아키텍처 그림의 출력 상자.
#   detection_tracking/  검출 decode(+추적 instance id 출력)·정합 target·손실
#   online_map/          지도 decode·target/assigner·match cost·손실
#   motion_planning/     모션 decode·target, 계획 decode(재채점·명령 선택)·target
from .sparse_head import SparseHead
from .detection_tracking import *
from .online_map import *
from .motion_planning import *
