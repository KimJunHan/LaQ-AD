#!/bin/bash
# 수정: bench2drive base 모델 full 학습 드라이버 — stage1 → (성공 시) stage2 자동 연결.
# 출력(로그+ckpt)은 results/ 에 저장(사용자 지시 2026-06-26). work_dirs 미사용.
# 재시작 복구(resume) 지원: results/<name>/latest.pth 있으면 이어서 학습.
# GPU0 단독, fp32, full b2d(242403 frame). worker=6 고정(RAM/swap OOM 방지).
set -u
cd /workspace/src/HiP-AD
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
PY=/opt/anaconda3/envs/hipad_bw/bin/python
OUT=results

# 수정: DataLoader worker 크래시(RAM 압박) 재발 방지 —
#   ① prefetch_factor=1 (기본2→1): worker당 prefetch 버퍼를 절반으로 → batch16 RAM 스파이크를 안정적이던 batch8 수준으로.
#   ② 크래시 시 latest.pth 에서 자동 재시도(최대 MAX_RETRY): 20시간 방치 재발 방지.
MAX_RETRY=20
run_stage () {
  local cfg="$1" wd="$2"; shift 2
  local extra=("$@")
  local try=0
  while : ; do
    local args=(--no-validate --work-dir "$wd" --seed 0 \
      --cfg-options data.workers_per_gpu=${WORKERS:-6} data.prefetch_factor=${PREFETCH:-2} checkpoint_config.interval=${CKPT_INTERVAL:-5000} "${extra[@]}")  # 수정: worker 6 고정(사용자 지시 2026-07-15). prefetch_factor 1→2 (2026-07-16): data_infos 직렬화로 RAM 76GB 여유 확보 → I/O-bound data_time(~3s) 을 prefetch 로 숨김. RAM 추가 ~1-5GB 로 안전. persistent_workers=True 로 6워커 유지. 수정(2026-08-03): checkpoint_config.interval 60600→5000(≈2.6h). stage2 iter57750 OOM kill 때 60600 첫저장 전이라 ckpt 0개로 진척 전소실 → 재시도 래퍼가 latest.pth 에서 실제로 이어받게 촘촘히 저장. max_keep_ckpts=3 유지.
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

echo "[$(date)] ===== STAGE1 ====="
run_stage configs/hipad_b2d_full_stage1.py "$OUT/hipad_b2d_full_stage1"
rc=$?
if [ $rc -ne 0 ]; then echo "[$(date)] STAGE1 FAILED rc=$rc — abort (stage2 미실행)"; exit 1; fi
if [ ! -f "$OUT/hipad_b2d_full_stage1/latest.pth" ]; then
  echo "[$(date)] STAGE1 latest.pth 없음 — abort"; exit 1; fi
echo "[$(date)] STAGE1 DONE"

# 수정(2026-07-18): stage2 전환 '전에' stage1 open-loop 평가(사용자 지시). 평가 실패해도 stage2 는 진행.
echo "[$(date)] ===== STAGE1 EVAL ====="
bash tools/eval_stage1.sh || echo "[$(date)] STAGE1 EVAL 실패/부분실패 — 로그 확인 후 수동 재평가 가능(latest.pth 보존됨). stage2 계속 진행."

echo "[$(date)] ===== STAGE2 ====="
# stage2 config 의 load_from(work_dirs 하드코딩) 을 results 경로로 override
run_stage configs/hipad_b2d_full_stage2.py "$OUT/hipad_b2d_full_stage2" \
  load_from="./$OUT/hipad_b2d_full_stage1/latest.pth"
rc=$?
if [ $rc -ne 0 ]; then echo "[$(date)] STAGE2 FAILED rc=$rc"; exit 1; fi
echo "[$(date)] STAGE2 DONE — bench2drive base 모델 stage1+stage2 학습 완료 (출력: $OUT/)"
