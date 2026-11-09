#!/usr/bin/env bash
# 수정(2026-11-05): 유휴 GPU0 재측정 일괄 — 새 파일. 설정마다 flock 직렬화, 측정 전·중·후 GPU0 프로세스 기록,
#   남의 프로세스가 보이면 그 설정을 다시 잰다(최대 3 회). 사용: bash tools/latency/run_idle_20261105.sh [job ...]
set -uo pipefail
cd /home1/irteam/data-vol1/src/HiP-AD_refactor
source /home1/irteam/data-vol1/hipad_env.sh
B=/home1/irteam/data-vol1/anaconda3/envs/hipad_bw/bin
export CC=$B/x86_64-conda-linux-gnu-cc CXX=$B/x86_64-conda-linux-gnu-g++
export CUDA_VISIBLE_DEVICES=0 HIPAD_ALLOW_GPU0=1 PYTHONPATH=.
export TORCHINDUCTOR_CACHE_DIR=/home1/irteam/data-vol1/src/HiP-AD_refactor/tools/latency/build/inductor_cache
LOCK=/tmp/claude-500/gpu0_timing.lock
LOG=docs/latency/logs_idle_20261105; SO=results/latency/idle_20261105; mkdir -p "$LOG" "$SO"
TL="python tools/latency/teacher_latency.py --n 30 --warmup 2"
TC="python tools/latency/teacher_context_latency.py --n 30 --warmup 2"
SB="python tools/latency/student_bench.py --warmup 30 --n 300"
NUIM="--config configs/nuscenes/v2_teacher/stage2/test_e3_nuim.py --checkpoint results/hipad_nusc_laq_v2_teacher_stage2_e3_nuim/iter_35160_ema.pth"
declare -A J=(
  [ctx_eager]="$TC --tag idle_ctx_eager"
  [ctx_eager_reuse]="$TC --tag idle_ctx_eager_reuse --prep threads --reuse_prev 1"
  [ctx_graph]="$TC --tag idle_ctx_graph --vision perimg_graph --prefill graph --prep threads"
  [ctx_graph_reuse]="$TC --tag idle_ctx_graph_reuse --vision perimg_graph --prefill graph --prep threads --reuse_prev 1"
  [ctx_graph_reuse_kv]="$TC --tag idle_ctx_graph_reuse_kv --vision perimg_graph --prefill graph --prep threads --reuse_prev 1 --ctx_kv slice"
  [hf_ref]="$TL --tag idle_hf_ref --engine hf"
  [f4_graph_step]="$TL --tag idle_f4_graph_step --engine fast --hidden inloop --decode graph_step"
  [n2_lm_last]="$TL --tag idle_n2_lm_last --engine fast --hidden inloop --decode graph_all --vision perimg_graph --prefill graph --prep threads --reuse_prev 1 --lm_head last"
  [s_eager_fp16]="$SB --variant eager --out $SO/e3_eager_fp16.json"
  [s_graph_fp16]="$SB --variant graph_post_cw --out $SO/e3_graph_post_cw_fp16.json"
  [s_eager_fp32]="$SB --variant eager --fp32 --out $SO/e3_eager_fp32.json"
  [s_graph_fp32]="$SB --variant graph_post_cw --fp32 --out $SO/e3_graph_post_cw_fp32.json"
  [n_eager_fp16]="$SB $NUIM --variant eager --out $SO/nuim_eager_fp16.json"
  [n_graph_fp16]="$SB $NUIM --variant graph_post_cw --out $SO/nuim_graph_post_cw_fp16.json"
  [n_eager_fp32]="$SB $NUIM --variant eager --fp32 --out $SO/nuim_eager_fp32.json"
  [n_graph_fp32]="$SB $NUIM --variant graph_post_cw --fp32 --out $SO/nuim_graph_post_cw_fp32.json"
)
ORDER=(ctx_eager ctx_eager_reuse ctx_graph ctx_graph_reuse ctx_graph_reuse_kv hf_ref f4_graph_step n2_lm_last
       s_eager_fp16 s_graph_fp16 s_eager_fp32 s_graph_fp32 n_eager_fp16 n_graph_fp16 n_eager_fp32 n_graph_fp32)
[ $# -gt 0 ] && ORDER=("$@")
apps() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader -i 0; }
for t in "${ORDER[@]}"; do
  for try in 1 2 3; do
    exec 9>"$LOCK"; flock 9
    P="$LOG/$t.procs"
    { echo "# try $try $(date '+%F %T') before:"; apps; } >> "$P"
    ${J[$t]} > "$LOG/$t.log" 2>&1 &
    CPID=$!
    # 측정 중 2 초마다 GPU0 프로세스 기록
    while kill -0 $CPID 2>/dev/null; do apps | sed "s/^/$(date +%T) /" >> "$P.during"; sleep 2; done
    wait $CPID; rc=$?
    { echo "# after (rc=$rc):"; apps; } >> "$P"
    flock -u 9
    foreign=$(awk '{print $2}' "$P.during" 2>/dev/null | tr -d ',' | sort -u | grep -v -x "$CPID" | grep -v '^$' || true)
    echo "== $t try $try rc=$rc cpid=$CPID foreign=[${foreign//$'\n'/ }]"
    mv "$P.during" "$P.during.try$try"
    [ -z "$foreign" ] && break
    echo "!! $t: 남의 GPU0 프로세스 발견 → 재측정" | tee -a "$P"
  done
  grep -E "^\[(summary|equiv|hid_vs_ref|per_folder_ms|capture)|median .* p90" "$LOG/$t.log" | tail -4
done
echo DONE > "$LOG/DONE"
