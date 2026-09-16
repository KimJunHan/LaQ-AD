#!/bin/bash
#수정: stage1(det+map+motion) B2D open-loop 평가(2026-07-18, 사용자 지시).
#      stage1 완주 후 stage2 전환 '전에' latest.pth 를 평가해 지표를 evaluation/ 에 기록.
#      eval_mode 는 standalone test.py 에 영향 없음(EvalHook 전용) → --eval bbox 로 det/map/motion 자동 산출.
#      GPU0 단독. 결과 pkl + 로그 저장. 실패해도 파이프라인이 stage2 로 진행하도록 호출측에서 tolerant 처리.
set -u
cd /workspace/src/HiP-AD
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
PY=/opt/anaconda3/envs/hipad_bw/bin/python
CFG=projects/configs/hipad_b2d_full_stage1.py
CKPT=results/hipad_b2d_full_stage1/latest.pth
OUTDIR=evaluation/hipad_b2d_full_stage1
mkdir -p "$OUTDIR"

if [ ! -f "$CKPT" ]; then echo "[eval] $CKPT 없음 — 평가 skip"; exit 1; fi
echo "[$(date)] STAGE1 EVAL 시작: $CKPT"
$PY tools/test.py "$CFG" "$CKPT" \
  --eval bbox \
  --out "$OUTDIR/results.pkl" \
  2>&1 | tee "$OUTDIR/eval_$(date +%m%d_%H%M).log"
rc=${PIPESTATUS[0]}
echo "[$(date)] STAGE1 EVAL 종료 rc=$rc (결과: $OUTDIR/)"
exit $rc
