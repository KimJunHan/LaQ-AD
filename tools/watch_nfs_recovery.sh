#!/bin/bash
#수정: NAS/NFS hang 복구 감시(2026-07-16). hard 마운트라 NAS 회복 시 학습이 손실0으로 자가재개.
#      NAS 에 부하를 더하지 않도록 무거운 read 대신 학습 로그 진행/프로세스 생존만 감시.
set -u
RL="$1"          # 학습 로그 경로
NAS=222.111.164.37
last_mtime=$(stat -c %Y "$RL" 2>/dev/null || echo 0)
echo "[$(date +%H:%M:%S)] watch 시작 — 학습 로그 진행/프로세스/NFS 회복 감시"
while : ; do
  sleep 60
  # 1) 학습 프로세스 살아있나
  if ! pgrep -f 'tools/train.py' >/dev/null; then
    echo "[$(date +%H:%M:%S)] ALERT: train.py 프로세스 사라짐 — 크래시/종료. 재시작 필요."
    exit 2
  fi
  # 2) 로그가 진행됐나 = NFS 회복 + 자가재개
  m=$(stat -c %Y "$RL" 2>/dev/null || echo 0)
  if [ "$m" -gt "$last_mtime" ]; then
    echo "[$(date +%H:%M:%S)] RECOVERED: 학습 로그 진행 재개 — NFS 복구 확인, 손실0 자가재개."
    tail -1 "$RL" | grep -oE "Iter \[[0-9/]+\].*data_time: [0-9.]+"
    exit 0
  fi
  # 3) 워커 D-state(rpc_wait) 여부 요약 (가벼움)
  dcnt=$(ps -eo stat,cmd | grep 'tools/train.py' | grep -c '^D')
  echo "[$(date +%H:%M:%S)] 대기중 — 로그 정지(마지막 $(date -d @$m +%H:%M:%S)), D-state 워커 ${dcnt}개 (NFS 여전히 hang)"
done
