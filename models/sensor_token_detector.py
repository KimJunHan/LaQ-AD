# 수정(2026-10-19): 사용자 결정 — 언어 토큰 자리에 LiDAR/레이더 토큰만 바꿔 넣는 비교(인코더 함께 학습)
#   SparseDetector(models/detector.py, 수정 없음)를 상속한 새 검출기. 달라지는 곳은 한 군데:
#     LaQ v2 교사 경로에서는 파이프라인(LoadTeacherCacheNusc)이 data['qwen_hidden'/'qwen_mask'/'qwen_type'] 를 넣고
#     SparseDetector._maybe_load_qwen_cache 가 모양만 정리한다. 여기서는 그 메서드를 바꿔, 같은 자리에
#     점 → SensorBEVTokenizer(기둥 BEV 인코더 + 칸 위치 임베딩) 로 만든 센서 토큰을 넣는다.
#       qwen_hidden (B, L=2304, C=256) fp32 — 인코더는 모델 안에서 계산되므로 디코더 손실의 기울기가 인코더까지 흐른다.
#       qwen_mask   (B, L) uint8 전부 1 — 교사와 같이 고정 길이·패딩 없음(QwenTokenAttn 은 전부 1 이면 마스크를 넘기지 않는다).
#       qwen_type   (B, L) int64 전부 0 — 센서 토큰 1 종(0 초기화 유형 임베딩 1 행).
#   이후 디코더(operation_order 의 qwen_attn)·QwenTokenAttn·손실·후처리는 교사 런과 같은 코드를 그대로 지난다.
#   기존 융합(B 융합) 경로(bev_encoders → fusion_bev_feats → 변형 집약)는 쓰지 않는다 — 켜면 즉시 실패.
#   정밀도: 인코더는 auto_fp16 구간(extract_feat, 영상 백본·FPN) 밖에서 돌므로 fp32 다(디코더·언어 모듈과 같은 취급).
import torch
from mmdet.models import DETECTORS, build_backbone

from .detector import SparseDetector

__all__ = ["SensorTokenSparseDetector"]


@DETECTORS.register_module()
class SensorTokenSparseDetector(SparseDetector):
    def __init__(self, sensor_token_encoder, **kwargs):
        if kwargs.get("qwen_cache_dir"):
            raise ValueError("SensorTokenSparseDetector: qwen_cache_dir 는 None 이어야 한다(토큰은 센서에서 만든다).")
        if kwargs.get("bev_encoders"):
            raise ValueError("SensorTokenSparseDetector: bev_encoders(BEV 융합 경로)와 함께 쓸 수 없다.")
        super().__init__(**kwargs)
        self.sensor_token_encoder = build_backbone(sensor_token_encoder)

    def _maybe_load_qwen_cache(self, data):
        """부모의 '언어 토큰 준비' 자리에 센서 토큰을 넣는다(forward_train·simple_test 모두 이 메서드를 부른다)."""
        for k in ("qwen_hidden", "qwen_mask", "qwen_type"):
            if k in data:
                raise RuntimeError(f"SensorTokenSparseDetector: 입력에 {k} 가 있다 — 파이프라인에서 언어 캐시 읽기를 뺄 것.")
        key = self.sensor_token_encoder.points_key
        if key not in data:
            raise KeyError(f"SensorTokenSparseDetector: 입력에 '{key}' 가 없다(파이프라인 Collect keys 확인).")
        pts = data[key]
        if isinstance(pts, torch.Tensor):  # 배치 1 이 텐서로 풀린 경우(부모 _maybe_extract_bev 와 같은 처리)
            pts = [pts]
        tokens = self.sensor_token_encoder([p.float() for p in pts])   # (B, L, C)
        B, L = tokens.shape[:2]
        data["qwen_hidden"] = tokens
        data["qwen_mask"] = torch.ones(B, L, dtype=torch.uint8, device=tokens.device)
        data["qwen_type"] = torch.zeros(B, L, dtype=torch.long, device=tokens.device)
        return data
