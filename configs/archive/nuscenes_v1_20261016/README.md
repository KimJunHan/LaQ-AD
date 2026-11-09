# nuScenes v1 config 보관 (2026-10-16)

사용자 지시(2026-10-16): "v1 학습 config 는 제거해도 좋다(성능 안 나옴)". v2 학습 config 가 기준이다.
이 폴더의 파일은 **이미 보고된 v1 결과(results/hipad_nusc_laq_stage{1,2}*, evaluation/hipad_nusc_laq_stage{1,2}*)의 기록**으로만 남긴다. 새 학습에 쓰지 않는다.

- v1 학습 config 본문은 `configs/nuscenes/stage{1,2}/_base_common.py` (종전 `train.py`, 내용 같음 — 맨 위 주석 2줄만 추가). v2 학습 config 가 이 파일을 `_base_` 로 상속한다.
- 여기 파일들의 `_base_` 는 위 파일을 가리키도록 경로만 고쳤다. 고친 뒤 `Config.fromfile` 결과가 옮기기 전과 같음을 확인했다.
- 옮긴 파일: stage1/test.py, stage1/train_2gpu.py, stage2/test.py, stage2/nodiff_train.py, stage2/nodiff_test.py
