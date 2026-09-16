#!/bin/bash
# 수정(2026-08-04): 실행 중이던 런처 교체용 감시자.
#
# 배경: 2026-08-03 21:54 에 띄운 `bash tools/run_nusc_stage12.sh`(pid 140599)가
#   스크립트 파일을 **열어둔 채** 돌고 있는데, 이후 그 파일을 수정하면서 inode 가 교체돼
#   해당 bash 의 fd 255 가 `(deleted)` 상태의 구버전을 가리키게 됐다.
#   bash 는 스크립트를 바이트 오프셋 기준으로 이어 읽으므로, 구버전을 잡은 런처는
#   **새로 넣은 "stage1 EVAL" 단계를 절대 실행하지 않는다**(stage1 → 곧바로 stage2).
#   사용자 지시는 "stage1 끝나면 평가하고 stage2" 이므로 런처를 교체해야 한다.
#
# 방식: 학습 프로세스(train.py)는 그대로 두고 구 런처만 죽인 뒤, 이 감시자가
#   train.py 종료를 기다렸다가 디스크의 **현재 버전** run_nusc_stage12.sh 를 실행한다.
#   그 스크립트는 latest.pth 가 있으면 stage1 을 resume 하므로:
#     - stage1 이 이미 완주했으면 즉시 끝나고 → EVAL → stage2 로 진행
#     - 중간에 죽은 것이면 이어서 학습 후 → EVAL → stage2
#   어느 쪽이든 사용자 지시대로 "평가 후 stage2" 가 보장된다.
set -u
cd /workspace/src/HiP-AD
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
PY=/opt/anaconda3/envs/hipad_bw/bin/python
OUT=results

TRAIN_PID=${TRAIN_PID:?TRAIN_PID 를 지정해야 한다}

# 구 런처를 죽이면 그쪽 EXIT trap 이 모니터를 함께 죽인다 → 감시 공백이 생기지 않도록
# 이 감시자가 자체 모니터를 띄운다(학습 종료 후 파이프라인이 자기 모니터를 다시 띄우므로 그때 정리).
$PY tools/monitor_mem.py --interval 10 --match train.py \
    --log "$OUT/nusc_mem_monitor.log" > "$OUT/nusc_mem_console_supervisor.log" 2>&1 &
MON_PID=$!
echo "[$(date)] supervisor 시작 — train pid=$TRAIN_PID 감시, mem monitor pid=$MON_PID"

while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "[$(date)] train pid=$TRAIN_PID 종료 확인 → 파이프라인(현재 버전) 실행"

kill $MON_PID 2>/dev/null   # 파이프라인이 자체 모니터를 띄운다

exec bash tools/run_nusc_stage12.sh
