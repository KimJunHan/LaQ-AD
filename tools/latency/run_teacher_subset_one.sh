#!/bin/bash
# 수정(2026-11-04): 교사 답 구간 진단 1회 실행(GPU0). 사용: run_teacher_subset_one.sh <name> <cfg> <ckpt> [eval_teacher_subset.py 인자...]
source /home1/irteam/data-vol1/hipad_env.sh
export CUDA_VISIBLE_DEVICES=0 HIPAD_ALLOW_GPU0=1
name=$1; cfg=$2; ckpt=$3; shift 3
out=results/latency/teacher_subset/$name
mkdir -p $out
python tools/latency/eval_teacher_subset.py $cfg $ckpt --out-dir $out "$@" > $out/run.log 2>&1
echo "rc=$?" >> $out/run.log
