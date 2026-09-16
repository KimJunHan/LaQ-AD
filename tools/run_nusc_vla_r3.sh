#!/bin/bash
# 수정(2026-08-26): 896x448 고해상도 fine-tune 드라이버 (Run1).
#   stage1 100epoch 체크포인트에서 이어 학습하며, 이번 run 에서 바꾸는 것은 해상도 하나뿐이다.
#   목표는 HiP-AD Table 4(det mAP 0.424) 초과. 근거·설정은 configs/hipad_nusc_vla_r3.py 헤더 참조.
#   stage1/stage2 드라이버와 동일하게 latest.pth 자동 재개 + 재시도 + 메모리 모니터를 둔다.
set -u
cd /workspace/src/HiP-AD
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0   # GPU1 사용 금지(사용자 지시)
PY=/opt/anaconda3/envs/hipad_bw/bin/python
CFG=configs/hipad_nusc_vla_r3.py
WD=results/hipad_nusc_vla_r3
MAX_RETRY=20

mkdir -p "$WD"

$PY tools/monitor_mem.py --interval 10 --match train.py \
    --log results/nusc_vla_r3_mem_monitor.log > results/nusc_vla_r3_mem_console.log 2>&1 &
MON_PID=$!
bash tools/reclaim_page_cache.sh 1200 > /dev/null 2>&1 &
RECLAIM_PID=$!
echo "[$(date)] mem monitor pid=$MON_PID / page-cache reclaim pid=$RECLAIM_PID"
trap 'kill $MON_PID $RECLAIM_PID 2>/dev/null' EXIT

try=0
while : ; do
  args=(--no-validate --work-dir "$WD" --seed 0)
  [ $# -gt 0 ] && args+=(--cfg-options "$@")  #수정(Run3): 체인이 load_from 등을 넘길 수 있게
  if [ -f "$WD/latest.pth" ]; then
    echo "[$(date)] RESUME $CFG from $WD/latest.pth (try $try)"
    args+=(--resume-from "$WD/latest.pth")
  else
    echo "[$(date)] START $CFG (fresh, try $try)"
  fi
  $PY tools/train.py "$CFG" "${args[@]}"
  rc=$?
  [ $rc -eq 0 ] && { echo "[$(date)] STAGE1 DONE rc=0"; break; }
  try=$((try+1))
  if [ $try -gt $MAX_RETRY ]; then
    echo "[$(date)] 재시도 $MAX_RETRY 회 초과 — 중단 rc=$rc"; exit $rc; fi
  echo "[$(date)] 비정상 종료 rc=$rc → 20초 후 latest.pth 에서 자동 재시도 ($try/$MAX_RETRY)"
  sleep 20
done

echo "[$(date)] ===== NUSC VLA STAGE1 EVAL 대기 — 사용자 확인 후 tools/eval_nusc_stage.sh 실행 ====="
