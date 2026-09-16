#!/bin/bash
# 수정(2026-08-25): nuScenes VLA stage2(= stage1 + motion) 학습 드라이버.
#
#   [설계] stage1 드라이버와 동일한 감독 구조 — latest.pth 자동 재개 + 최대 20회 재시도 +
#   메모리 모니터 + 페이지 캐시 회수기. 2026-08-21 호스트 재시작으로 학습이 통째로 멈춘
#   전례가 있어 무인 복구가 필수다.
#
#   [주의 — 과거 stage2 인지 성능 하락] 2026-08-11 진단: (1) lr 을 2e-4 로 되돌려 stage1
#   수렴점을 파괴, (2) 모션 기울기가 전체의 75% 를 차지해 인지를 잠식 → AMOTA -23.7%.
#   config(hipad_nusc_vla_stage2.py)에 이미 수정 반영: 기준 lr 2e-5 + 신규 모션 모듈만
#   lr_mult=10, loss_motion_reg 0.2->0.1(지평 정규화). 그래도 3-iter 스모크에서
#   motion_loss_reg 25.8 / ego_loss_status 10.7 로 여전히 큰 비중이므로,
#   det_loss_cls 가 stage1 최종값(~1.82) 아래로 회복하는지 학습 중 계속 확인할 것.
#
#   사용법: nohup bash tools/run_nusc_vla_stage2.sh > results/nusc_vla_stage2_resume.log 2>&1 &
set -u
cd /workspace/src/HiP-AD
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0   # GPU1 사용 금지(사용자 지시)
PY=/opt/anaconda3/envs/hipad_bw/bin/python
CFG=configs/hipad_nusc_vla_stage2.py
WD=results/hipad_nusc_vla_stage2
MAX_RETRY=20

mkdir -p "$WD"

$PY tools/monitor_mem.py --interval 10 --match train.py \
    --log results/nusc_vla_s2_mem_monitor.log > results/nusc_vla_s2_mem_console.log 2>&1 &
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
