#!/usr/bin/env bash
# 수정(2026-11-04): 교사 온라인 지연 측정 일괄 실행 — 새 파일. 설정마다 flock 으로 GPU0 측정 직렬화.
#   사용: bash tools/latency/run_teacher_latency.sh [N] [WARMUP] [설정 이름 ...]
set -euo pipefail
source /home1/irteam/data-vol1/hipad_env.sh
export CUDA_VISIBLE_DEVICES=0 HIPAD_ALLOW_GPU0=1
N=${1:-30}; W=${2:-2}; shift 2 || true
LOCK=/tmp/claude-500/gpu0_timing.lock
LOG=docs/latency/logs; mkdir -p "$LOG"
declare -A CFG=(
  [hf_ref]="--engine hf --save_ref"
  [f1_cpu_refwd]="--engine fast --grammar cpu --hidden refwd --decode eager"
  [f2_gpu_refwd]="--engine fast --grammar gpu --hidden refwd --decode eager"
  [f3_gpu_inloop]="--engine fast --grammar gpu --hidden inloop --decode eager"
  [f4_graph_step]="--engine fast --hidden inloop --decode graph_step"
  [f5_graph_all]="--engine fast --hidden inloop --decode graph_all"
  [f6_graph_vis_pre]="--engine fast --hidden inloop --decode graph_all --vision perimg_graph --prefill graph"
  [f7_prep_reuse]="--engine fast --hidden inloop --decode graph_all --vision perimg_graph --prefill graph --prep threads --reuse_prev 1"
  [n1_fa2_vision]="--engine fast --hidden inloop --decode graph_all --vision fa2_graph --prefill graph --prep threads --reuse_prev 1"
  [n2_lm_last]="--engine fast --hidden inloop --decode graph_all --vision perimg_graph --prefill graph --prep threads --reuse_prev 1 --lm_head last"
)
ORDER=(hf_ref f1_cpu_refwd f2_gpu_refwd f3_gpu_inloop f4_graph_step f5_graph_all f6_graph_vis_pre f7_prep_reuse n1_fa2_vision n2_lm_last)
[ $# -gt 0 ] && ORDER=("$@")
for t in "${ORDER[@]}"; do
  echo "== $t $(date +%T) gpu0_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 | wc -l)"
  flock "$LOCK" timeout 1800 python tools/latency/teacher_latency.py ${CFG[$t]} --n "$N" --warmup "$W" --tag "$t" \
    > "$LOG/$t.log" 2>&1 || echo "!! $t 실패(exit $?) — $LOG/$t.log"
  grep -E "^\[(summary|equiv|hid_vs_ref|per_folder_ms|capture)" "$LOG/$t.log" || true
done
