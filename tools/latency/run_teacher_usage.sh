#!/bin/bash
# 수정(2026-11-04): 교사 답 구간 사용도 측정(GPU0). 사용: run_teacher_usage.sh <name> <cfg> <ckpt>
source /home1/irteam/data-vol1/hipad_env.sh
export CUDA_VISIBLE_DEVICES=0 HIPAD_ALLOW_GPU0=1
mkdir -p results/latency/teacher_usage
python tools/latency/teacher_usage_hooks.py $2 $3 --out results/latency/teacher_usage/$1.json > results/latency/teacher_usage/$1.log 2>&1
echo "rc=$?" >> results/latency/teacher_usage/$1.log
