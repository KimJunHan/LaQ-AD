#!/bin/bash
# 수정(2026-11-04): 교사 답 구간 진단 대기열 — 추론 단계(results.pkl 미생성) 작업이 MAXINF 개 미만이면 다음 작업을 띄운다.
#   사용: queue_teacher_subset.sh <jobfile>   (줄마다: name cfg ckpt [인자...])
cd /home1/irteam/data-vol1/src/HiP-AD_refactor
MAXINF=${MAXINF:-5}
D=results/latency/teacher_subset
while read -r name rest; do
  [ -z "$name" ] && continue
  [[ "$name" == \#* ]] && continue
  while true; do
    n=0
    for p in $D/*.pid; do
      pid=$(cat $p); nm=$(basename $p .pid)
      if kill -0 $pid 2>/dev/null && [ ! -f $D/$nm/results.pkl ]; then n=$((n+1)); fi
    done
    [ $n -lt $MAXINF ] && break
    sleep 30
  done
  echo "$(date +%T) start $name $rest"
  R=tools/latency/run_teacher_subset_one.sh; [[ $name == s2_* ]] && R=tools/latency/run_teacher_subset_plan.sh; setsid nohup $R $name $rest > /dev/null 2>&1 &
  echo $! > $D/$name.pid
  sleep 20
done < "$1"
echo "$(date +%T) queue done"
