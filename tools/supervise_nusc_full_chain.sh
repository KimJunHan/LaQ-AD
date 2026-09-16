#!/bin/bash
# 수정(2026-08-11): nuScenes full 체인 감독 — stage1 학습 -> stage1 평가 -> stage2 학습.
#
#   사용자 지시: "stage1 학습 진행하고 stage1 평가 진행하고 stage2 학습하자".
#
#   [단독 소유 원칙] 이 스크립트가 체인의 유일한 실행 주체다. 2026-08-11 에 감독 스크립트와
#   수동 실행이 겹쳐 같은 results_nusc.json 에 동시 write 하여 JSON 이 깨진 전례가 있다.
#   평가 스크립트에 flock 을 넣어 막아두었지만, 애초에 중복 실행을 만들지 않는다.
#
#   [데이터] 학습·평가 모두 full 데이터셋만 사용한다(사용자 지시, 무조건).
#            nuScenes train 28,130 / val 6,019. 축소본(mini/smoke64/scene40) 사용 금지.
#
#   사용법: TAG=hipad_nusc_vla nohup bash tools/supervise_nusc_full_chain.sh <stage1_train_pid> &
#     TAG 기본값 hipad_nusc_vla. configs/<TAG>_stage{1,2}.py 와 results/<TAG>_stage{1,2} 를 쓴다.
#
#   ⚠ PID 는 반드시 실제 python 프로세스여야 한다. nohup 을 감싼 bash 래퍼 PID 를 주면
#     래퍼가 먼저 종료되면서 감독이 즉시 오발동한다(2026-08-11 실제 발생).
set -u
cd /workspace/src/HiP-AD
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0   # GPU1 사용 금지(사용자 지시)
PY=/opt/anaconda3/envs/hipad_bw/bin/python

S1_PID=${1:?"stage1 학습 프로세스 PID 를 인자로 줄 것"}
S1_DIR=results/${TAG:-hipad_nusc_vla}_stage1
S2_DIR=results/${TAG:-hipad_nusc_vla}_stage2
S1_EVAL=evaluation/${TAG:-hipad_nusc_vla}_stage1
LOG=results/nusc_full_chain.log

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "체인 감독 시작. stage1 학습 PID=$S1_PID 종료 대기"
while kill -0 "$S1_PID" 2>/dev/null; do sleep 60; done
say "stage1 학습 프로세스 종료"

# --- stage1 완주 검증: 마지막 체크포인트가 실제로 만들어졌는지 확인한다.
#     중간에 죽은 경우 평가/stage2 로 넘어가면 안 된다(잘못된 수치가 기록으로 남는다).
if [ ! -f "$S1_DIR/latest.pth" ]; then
  say "중단: $S1_DIR/latest.pth 없음 — stage1 이 체크포인트를 남기지 못했다. 체인 종료"
  exit 1
fi
LAST_ITER=$(grep -ao "Iter \[[0-9]*/351600\]" "$S1_DIR"/*.log | tail -1 | grep -o "[0-9]*" | head -1)
say "stage1 마지막 기록 iter=${LAST_ITER:-unknown}/351600"
if [ "${LAST_ITER:-0}" -lt 351600 ]; then
  say "중단: stage1 이 351600 iter 를 완주하지 못했다(=${LAST_ITER}). 원인 확인 후 재개할 것"
  exit 1
fi

# --- stage1 평가 (det/map/track). motion 은 stage2 에서만 산출된다.
say "stage1 평가 시작 -> $S1_EVAL"
mkdir -p "$S1_EVAL"
exec 9>"$S1_EVAL/.eval.lock"
if ! flock -n 9; then
  say "중단: $S1_EVAL 평가가 이미 실행 중이다(중복 실행 방지)"
  exit 2
fi
$PY tools/test.py configs/${TAG:-hipad_nusc_vla}_stage1.py "$S1_DIR/latest.pth" \
  --eval bbox --out "$S1_EVAL/results.pkl" \
  --cfg-options evaluation.jsonfile_prefix="$S1_EVAL" \
  >> "$S1_EVAL/eval.log" 2>&1
RC=$?
flock -u 9
say "stage1 평가 종료 rc=$RC (결과: $S1_EVAL/)"
if [ $RC -ne 0 ]; then
  say "중단: stage1 평가 실패. 지표를 확인하지 못한 채 stage2 로 넘어가지 않는다"
  exit 3
fi
grep -aE "^  (mAP|NDS|mATE|mASE|mAOE|mAVE|mAAE):" "$S1_EVAL/eval.log" | tail -8 | tee -a "$LOG"
grep -a "amota\|mAP_normal" "$S1_EVAL/eval.log" | tail -4 | tee -a "$LOG"

# --- stage2 학습 (stage1 가중치 로드, +motion, 계획·자차 손실 켬)
say "stage2 학습 시작 -> $S2_DIR"
mkdir -p "$S2_DIR"
$PY tools/train.py configs/${TAG:-hipad_nusc_vla}_stage2.py \
  --no-validate --work-dir "$S2_DIR" --seed 0 \
  --cfg-options load_from="$S1_DIR/latest.pth" \
  >> "$S2_DIR/launch.log" 2>&1
say "stage2 학습 종료 rc=$? (다음: stage2 평가는 별도로 판단)"
