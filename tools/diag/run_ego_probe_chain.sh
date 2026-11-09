#!/bin/bash
# 진단(2026-11-05): ego_probe 연쇄 실행(GPU1, fp32). 사용: run_ego_probe_chain.sh <stream_name> "<name> <cfg> <ckpt> [args]" ...
source /home1/irteam/data-vol1/hipad_env.sh
export CUDA_VISIBLE_DEVICES=1
cd /home1/irteam/data-vol1/src/HiP-AD_refactor
s=$1; shift
for job in "$@"; do
  set -- $job; name=$1; cfg=$2; ckpt=$3; shift 3
  python tools/diag/ego_probe.py $cfg $ckpt --out results/diag/probe_$name.npz "$@" > results/diag/probe_$name.log 2>&1
  echo "$name rc=$?" >> results/diag/chain_$s.log
done
