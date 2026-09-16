#!/bin/bash
# 수정(2026-08-03): stage2 단독 재개용 러너.
#   run_b2d_full_stage12.sh 는 stage1 → stage1 eval → stage2 순서라 그대로 쓰면
#   이미 완주한 stage1(iter 363600)과 30분짜리 stage1 eval 을 다시 타게 된다.
#   stage1 은 results/hipad_b2d_full_stage1/latest.pth 로 끝났고 평가도 별도로 돌렸으므로
#   stage2 만 떼어 실행한다. 재시도/resume 로직은 run_b2d_full_stage12.sh 의 run_stage 와 동일.
#
# 배경: 2026-07-23 stage2 가 iter 57750 에서 호스트 RAM 고갈로 죽었다.
#   - ckpt interval 이 60600(첫 저장 ≈28.8h)이라 저장본이 0개 → 이틀치 전소실. 이번엔 5000(≈2.4h).
#   - 사망 원인 미확정(train_rss 는 평평한데 시스템 used 만 100초에 +61GB).
#     → Shmem/SUnreclaim 계측을 추가한 monitor_mem.py 를 동반 실행해 소비처를 관측으로 특정한다.
#   - DataLoader 설정(worker 6 / prefetch 1)은 실패한 그 실행과 동일하게 유지한다.
#     조건을 바꾸면 재현되더라도 원인 규명이 불가능해지기 때문.
set -u
cd /workspace/src/HiP-AD
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0   # 수정: GPU1 사용 금지(사용자 지시) — GPU0 단독
PY=/opt/anaconda3/envs/hipad_bw/bin/python
OUT=results
WD="$OUT/hipad_b2d_full_stage2"
CFG=configs/hipad_b2d_full_stage2.py
MAX_RETRY=20
WORKERS=${WORKERS:-6}
PREFETCH=${PREFETCH:-1}
CKPT_INTERVAL=${CKPT_INTERVAL:-5000}

mkdir -p "$WD"

# 선행 프로세스(예: 진행 중인 open-loop 평가)가 끝날 때까지 대기 — RAM/NFS 경합 방지.
WAIT_PID=${WAIT_PID:-0}
if [ "$WAIT_PID" != "0" ] && kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "[$(date)] 선행 프로세스 pid=$WAIT_PID 종료 대기 중..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[$(date)] 선행 프로세스 종료 확인 — stage2 시작"
fi

# 메모리 계측 모니터 동반 실행(학습을 죽이지 않는 관찰 전용)
$PY tools/monitor_mem.py --interval 10 --match train.py \
    --log "$WD/mem_monitor.log" > "$OUT/stage2_mem_console.log" 2>&1 &
MON_PID=$!
echo "[$(date)] mem monitor pid=$MON_PID → $WD/mem_monitor.log"
trap 'kill $MON_PID 2>/dev/null' EXIT

try=0
while : ; do
  args=(--no-validate --work-dir "$WD" --seed 0 \
    --cfg-options data.workers_per_gpu=$WORKERS data.prefetch_factor=$PREFETCH \
    checkpoint_config.interval=$CKPT_INTERVAL \
    load_from="./$OUT/hipad_b2d_full_stage1/latest.pth")
  if [ -f "$WD/latest.pth" ]; then
    echo "[$(date)] RESUME stage2 from $WD/latest.pth (try $try)"
    args+=(--resume-from "$WD/latest.pth")
  else
    echo "[$(date)] START  stage2 (fresh, stage1 가중치에서 시작, try $try)"
  fi
  $PY tools/train.py "$CFG" "${args[@]}"
  rc=$?
  [ $rc -eq 0 ] && { echo "[$(date)] STAGE2 DONE"; exit 0; }
  try=$((try+1))
  if [ $try -gt $MAX_RETRY ]; then
    echo "[$(date)] stage2 재시도 $MAX_RETRY 회 초과 — 중단 rc=$rc"; exit $rc; fi
  echo "[$(date)] stage2 비정상 종료 rc=$rc → 20초 후 latest.pth 에서 자동 재시도 ($try/$MAX_RETRY)"
  sleep 20
done
