# 수정(2026-10-19): 사용자 결정 — 언어 토큰 자리에 LiDAR/레이더 토큰만 바꿔 넣는 비교(인코더 함께 학습)
#   encoder/__init__.py 는 이 패키지를 import 하지 않는다(기존 파일 불변). config 의 custom_imports 로만 등록한다.
from .bev_tokenizer import SensorBEVTokenizer

__all__ = ["SensorBEVTokenizer"]
