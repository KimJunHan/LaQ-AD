#!/bin/bash
# 수정(2026-08-24): nuScenes VLA stage1(ReLaQ-AD 언어-쿼리 융합) 학습 재개 드라이버.
#
#   [배경] 2026-08-21 12:29, iter 332950/351600(94.7%) 지점에서 학습이 파이썬 트레이스백
#   없이 갑자기 끊겼다. 같은 날 15:30 경 호스트가 재부팅된 흔적(uptime)이 있고 로그에
#   예외·NaN·OOM 기록이 전혀 없다 → 모델/데이터 문제가 아니라 외부 종료(호스트 재시작)다.
#   손실 11.0 부근 정상, latest.pth(iter_330504) 가중치·옵티마이저 상태 모두 finite 확인.
#
#   [설계] 같은 사고가 또 나도 사람 개입 없이 latest.pth 에서 자동 재개하도록 감독 루프를
#   둔다. run_nusc_stage12.sh 와 동일한 구조이며, 다음 조사용 증거를 남기기 위해
#   메모리 모니터를 동반한다(이번엔 모니터가 없어서 사후 원인 특정이 불가능했다).
#   페이지 캐시 회수기(reclaim_page_cache.sh)도 같이 띄운다 — Qwen 토큰 캐시 114GB 가
#   재사용되지 않고 페이지 캐시만 밀어내므로 주기적으로 반납해야 RAM 압박이 없다.
#
#   설정은 원 실행과 동일하게 유지한다(비교 타당성): GPU0 단독, fp32, batch 8 × 누적 8,
#   workers_per_gpu=8(config 기본값 — override 하지 않는다), seed 0.
#
#   사용법: nohup bash tools/run_nusc_vla_stage1.sh > results/nusc_vla_stage1_resume.log 2>&1 &
set -u
cd /workspace/src/HiP-AD
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0   # GPU1 사용 금지(사용자 지시)
PY=/opt/anaconda3/envs/hipad_bw/bin/python
CFG=configs/hipad_nusc_vla_stage1.py
WD=results/hipad_nusc_vla_stage1
MAX_RETRY=20

mkdir -p "$WD"

$PY tools/monitor_mem.py --interval 10 --match train.py \
    --log results/nusc_vla_mem_monitor.log > results/nusc_vla_mem_console.log 2>&1 &
MON_PID=$!
bash tools/reclaim_page_cache.sh 1200 > /dev/null 2>&1 &
RECLAIM_PID=$!
echo "[$(date)] mem monitor pid=$MON_PID / page-cache reclaim pid=$RECLAIM_PID"
trap 'kill $MON_PID $RECLAIM_PID 2>/dev/null' EXIT

try=0
while : ; do
  args=(--no-validate --work-dir "$WD" --seed 0)
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
