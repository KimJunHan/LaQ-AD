# 수정(2026-10-15, 2차): Qwen2-VL-2B (VLM). transformers 의존성이 무거워 패키지 import 시 아무것도 불러오지 않는다.
#   qwen_online.py         : 동결 Qwen2-VL-2B 온라인 인코더(18층 은닉) — 캐시 생성(tools/qwen_cache_*.py)·B2D 폐쇄루프 에이전트가 파일 경로로 적재
#   qwen_traj_generator.py : VLA-v0 궤적 생성기(학습 경로 미사용, 시각화 도구가 직접 import)
