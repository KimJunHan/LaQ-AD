#!/bin/bash
# 수정(2026-08-03): nuScenes full trainval 학습 드라이버 — stage1 → (성공 시) stage2 자동 연결.
#   목적: HiP-AD 논문 Table 4(검출 mAP/NDS · 매핑 mAP · 추적 AMOTA · 모션 minADE) 재현.
#   출력은 results/ 에 저장(스토리지 정책). work_dirs 미사용 — 과거 nuScenes ckpt 가
#   work_dirs 에 있다가 전부 소실된 전례가 있어 반드시 results/ 로 떨어뜨린다.
#   재시작 복구(resume) 지원: results/<name>/latest.pth 있으면 이어서 학습.
#   GPU0 단독(GPU1 사용 금지), fp32, eff batch 48(HiP-AD upstream 정렬).
set -u
cd /workspace/src/HiP-AD
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
PY=/opt/anaconda3/envs/hipad_bw/bin/python
OUT=results
MAX_RETRY=20
WORKERS=${WORKERS:-6}      # RAM/swap 압박 방지 상한
PREFETCH=${PREFETCH:-1}

# 메모리 계측 모니터(Shmem/SUnreclaim 포함) 동반 — 관찰 전용, 학습을 죽이지 않는다.
mkdir -p "$OUT"
$PY tools/monitor_mem.py --interval 10 --match train.py \
    --log "$OUT/nusc_mem_monitor.log" > "$OUT/nusc_mem_console.log" 2>&1 &
MON_PID=$!
echo "[$(date)] mem monitor pid=$MON_PID → $OUT/nusc_mem_monitor.log"
trap 'kill $MON_PID 2>/dev/null' EXIT

run_stage () {
  local cfg="$1" wd="$2"; shift 2
  local extra=("$@")
  local try=0
  mkdir -p "$wd"
  while : ; do
    local args=(--no-validate --work-dir "$wd" --seed 0 \
      --cfg-options data.workers_per_gpu=$WORKERS data.prefetch_factor=$PREFETCH "${extra[@]}")
    if [ -f "$wd/latest.pth" ]; then
      echo "[$(date)] RESUME $cfg from $wd/latest.pth (try $try)"
      args+=(--resume-from "$wd/latest.pth")
    else
      echo "[$(date)] START  $cfg (fresh, try $try)"
    fi
    $PY tools/train.py "$cfg" "${args[@]}"
    local rc=$?
    [ $rc -eq 0 ] && return 0
    try=$((try+1))
    if [ $try -gt $MAX_RETRY ]; then
      echo "[$(date)] $cfg 재시도 $MAX_RETRY 회 초과 — 중단 rc=$rc"; return $rc; fi
    echo "[$(date)] $cfg 비정상 종료 rc=$rc → 20초 후 latest.pth 에서 자동 재시도 ($try/$MAX_RETRY)"
    sleep 20
  done
}

echo "[$(date)] ===== NUSC STAGE1 (det+map, 100ep, 351600 iter) ====="
run_stage configs/hipad_trainval_nusc_stage1.py "$OUT/hipad_nusc_stage1"
rc=$?
if [ $rc -ne 0 ]; then echo "[$(date)] STAGE1 FAILED rc=$rc — abort"; exit 1; fi
if [ ! -f "$OUT/hipad_nusc_stage1/latest.pth" ]; then
  echo "[$(date)] STAGE1 latest.pth 없음 — abort"; exit 1; fi
echo "[$(date)] STAGE1 DONE"

# 수정(2026-08-04, 사용자 지시): stage2 전환 '전에' stage1 을 반드시 평가한다.
#   stage1 ckpt 로 Table 4 의 det mAP/NDS · map mAP · track AMOTA 3개 열이 나온다.
#   여기서 지표를 남기지 않으면 stage2 결과가 나빠도 어느 단계 문제인지 분리할 수 없다.
#   평가가 실패해도 stage2 는 계속 진행한다(ckpt 는 보존되므로 수동 재평가 가능).
echo "[$(date)] ===== NUSC STAGE1 EVAL ====="
bash tools/eval_nusc_stage.sh stage1 || \
  echo "[$(date)] STAGE1 EVAL 실패/부분실패 — 로그 확인 후 수동 재평가 가능. stage2 계속 진행."

echo "[$(date)] ===== NUSC STAGE2 (+motion, 10ep, 70320 iter) ====="
# 수정: config 의 load_from 이 work_dirs 하드코딩이라 results 경로로 override
run_stage configs/hipad_trainval_nusc_stage2.py "$OUT/hipad_nusc_stage2" \
  load_from="./$OUT/hipad_nusc_stage1/latest.pth"
rc=$?
if [ $rc -ne 0 ]; then echo "[$(date)] STAGE2 FAILED rc=$rc"; exit 1; fi
echo "[$(date)] STAGE2 DONE"

# 수정(2026-08-04): stage2 평가까지 해야 Table 4 4개 열(+motion minADE)이 전부 채워진다.
echo "[$(date)] ===== NUSC STAGE2 EVAL ====="
bash tools/eval_nusc_stage.sh stage2 || \
  echo "[$(date)] STAGE2 EVAL 실패/부분실패 — 로그 확인 후 수동 재평가 가능."
echo "[$(date)] ALL DONE — nuScenes stage1+eval+stage2+eval 완료 (출력: $OUT/, evaluation/)"
