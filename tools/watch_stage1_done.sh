#!/bin/bash
#수정(2026-07-18): stage1 완료 임박 감시. iter>=THRESH 도달 시 알림(=런처 재시작 타이밍),
#      또는 학습 프로세스 사망 시 알림. 가볍게 로그만 파싱(NAS 부하 없음).
set -u
RL="$1"
THRESH="${2:-358000}"
echo "[$(date +%m-%d_%H:%M)] stage1 완료감시 시작 — iter>=$THRESH 또는 프로세스 사망 시 알림"
while : ; do
  sleep 300
  if ! pgrep -f 'tools/train.py' >/dev/null; then
    echo "[$(date +%m-%d_%H:%M)] ALERT: train.py 사라짐 — 크래시/완료. 확인 필요."
    exit 2
  fi
  it=$(grep -oE "Iter \[[0-9]+/" "$RL" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  [ -z "$it" ] && continue
  if [ "$it" -ge "$THRESH" ]; then
    echo "[$(date +%m-%d_%H:%M)] stage1 완료임박: iter $it >= $THRESH. 런처 재시작(eval-gate 반영) 타이밍."
    exit 0
  fi
  echo "[$(date +%m-%d_%H:%M)] iter $it (임계 $THRESH 미도달) — 대기"
done
