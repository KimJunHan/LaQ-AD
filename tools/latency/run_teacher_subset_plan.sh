#!/bin/bash
# 수정(2026-11-04): 2단계 계획 진단 1회(GPU0) — 추론만(eval_teacher_subset.py --no-eval) 뒤 tools/eval_nusc_planning.py 로 L2·충돌.
#   사용: run_teacher_subset_plan.sh <name> <cfg> <ckpt> [eval_teacher_subset.py 인자...]
source /home1/irteam/data-vol1/hipad_env.sh
export CUDA_VISIBLE_DEVICES=0 HIPAD_ALLOW_GPU0=1
name=$1; cfg=$2; ckpt=$3; shift 3
out=results/latency/teacher_subset/$name
mkdir -p $out
python tools/latency/eval_teacher_subset.py $cfg $ckpt --out-dir $out --no-eval "$@" > $out/run.log 2>&1 && \
CUDA_VISIBLE_DEVICES="" python tools/eval_nusc_planning.py $cfg $out/results.pkl --out $out/planning_nusc.json >> $out/run.log 2>&1
echo "rc=$?" >> $out/run.log
