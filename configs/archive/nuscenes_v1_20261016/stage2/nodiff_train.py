# 수정(2026-10-15): 경로 이동 configs/nuscenes/hipad_nusc_laq_stage2_nodiff_train.py → configs/nuscenes/stage2/nodiff_train.py
#   run_name: 결과 디렉터리 이름(results/<run_name>, evaluation/<run_name>). tools/train.py·test.py 가 --work-dir 미지정 시 쓴다.
# 수정(2026-10-12): 사용자 결정 — 확산 단독 기여 분리용 2단계(확산만 끔).
#   대표 모델 2단계(hipad_nusc_laq_stage2_train.py)와 같은 1단계 가중치·명령 조건 3x16·계획 lr x10·배치 8x6·일정이고,
#   차이는 plan_diffusion=None(언어 조건 잘린 확산 제거) 하나뿐이다. 출력: results/hipad_nusc_laq_stage2_nodiff.
_base_ = ["../../../nuscenes/stage2/_base_common.py"]  # 수정(2026-10-16): v1 실험 config 제거 — 기록 보관용으로 이동, 종전 ./train.py(= 현 nuscenes/stage2/_base_common.py, 내용 동일)를 가리킴. 실행용 아님

model = dict(head=dict(onedecoder_head=dict(plan_diffusion=None)))

# 수정(2026-10-15): 결과 디렉터리 이름(파일명이 train.py 로 바뀌어 파일명 기반 기본값을 쓸 수 없다).
run_name = "hipad_nusc_laq_stage2_nodiff"
