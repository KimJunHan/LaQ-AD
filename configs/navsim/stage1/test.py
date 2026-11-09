# 수정(2026-10-15): 경로 이동 configs/navsim/hipad_navsim_laq_stage1_test.py → configs/navsim/stage1/test.py
# 수정(2026-10-09): LaQ-AD NAVSIM stage1 테스트 config (navtest 추론 → EPDMS).
#   학습 config 를 그대로 상속한다(모델·데이터 설정이 학습과 어긋나지 않게). test 분할 = navtest(장면별 이력 4프레임 시퀀스).
#   1) 추론(hipad_bw, GPU0, 단일 GPU·launcher 없음·batch 1 — 시간 뱅크 순차성 필수):
#        PYTHONPATH=. python tools/test.py configs/navsim/hipad_navsim_laq_stage1_test.py results/hipad_navsim_laq_stage1/latest.pth \
#            --eval bbox    # → NavsimDataset.evaluate 가 evaluation/navsim/navsim_pred.pkl 저장
#   2) EPDMS 채점(devkit env): bash tools/navsim/run_navsim_eval.sh evaluation/navsim/navsim_pred.pkl evaluation/hipad_navsim_laq_stage1
#   stage1 은 계획 손실 0(인지만 학습)이라 EPDMS 는 참고용이다. 보고 수치는 stage2.
_base_ = ["./train.py"]

data = dict(samples_per_gpu=1, test=dict(samples_per_gpu=1))  # 장면 경계 초기화는 batch 1 만 허용(SparseDetector._maybe_reset_banks)
evaluation = dict(jsonfile_prefix="evaluation/hipad_navsim_laq_stage1")
