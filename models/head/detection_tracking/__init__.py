# 수정(2026-10-15, 2차): detection / tracking — 추적 ID 는 decoder/temporal_bank InstanceBank.get_instance_id 가 만들고
#   SparseBox3DDecoder 가 instance_ids 로 내보낸다(평가 시 loader/nuscenes/track_relink.py 재연결).
from .decoder import SparseBox3DDecoder
from .target import SparseBox3DTarget
from .losses import SparseBox3DLoss
