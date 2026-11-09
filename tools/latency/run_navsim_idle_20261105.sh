#!/usr/bin/env bash
# 수정(2026-11-05): NAVSIM 유휴 GPU0 지연 측정 일괄 — 새 파일(run_idle_20261105.sh 의 NAVSIM 판). 설정마다 flock 직렬화, 측정 전·중·후 GPU0 프로세스 기록,
#   남의 프로세스가 보이면 그 설정을 다시 잰다(최대 3 회). 사용: bash tools/latency/run_navsim_idle_20261105.sh [job ...]
set -uo pipefail
cd /home1/irteam/data-vol1/src/HiP-AD_refactor
source /home1/irteam/data-vol1/hipad_env.sh
B=/home1/irteam/data-vol1/anaconda3/envs/hipad_bw/bin
export CC=$B/x86_64-conda-linux-gnu-cc CXX=$B/x86_64-conda-linux-gnu-g++
export CUDA_VISIBLE_DEVICES=0 HIPAD_ALLOW_GPU0=1 PYTHONPATH=.
export TORCHINDUCTOR_CACHE_DIR=/home1/irteam/data-vol1/src/HiP-AD_refactor/tools/latency/build/inductor_cache
LOCK=/tmp/claude-500/gpu0_timing.lock
LOG=docs/latency/navsim/logs_idle_20261105; SO=results/latency/navsim_idle_20261105; mkdir -p "$LOG" "$SO"
ST=$SO/samples_navtest_A.json; SC=$SO/samples_navtrain_cache_A.json
TL="python tools/latency/teacher_latency_navsim.py --n 30 --warmup 2"
TC="python tools/latency/teacher_context_latency.py --dataset navsim --n 30 --warmup 2"
SB="python tools/latency/student_bench_navsim.py --warmup 30 --n 300"
C1=configs/navsim/v2_teacher/stage1/test.py; C2=configs/navsim/v2_teacher/stage2/test.py
N2="--engine fast --hidden inloop --decode graph_all --vision perimg_graph --prefill graph --prep threads --reuse_prev 1 --lm_head last"
declare -A J=(
  [hf_ref]="$TL --samples $ST --tag nav_hf_ref --engine hf --save_ref"
  [ctx_eager]="$TC --samples $ST --tag nav_ctx_eager"
  [ctx_eager_reuse]="$TC --samples $ST --tag nav_ctx_eager_reuse --prep threads --reuse_prev 1"
  [ctx_graph]="$TC --samples $ST --tag nav_ctx_graph --vision perimg_graph --prefill graph --prep threads"
  [ctx_graph_reuse]="$TC --samples $ST --tag nav_ctx_graph_reuse --vision perimg_graph --prefill graph --prep threads --reuse_prev 1"
  [n2_lm_last]="$TL --samples $ST --tag nav_n2_lm_last $N2"
  [c_n2_lm_last]="$TL --samples $SC --tag navtrain_n2_lm_last --ref_tag none_navtrain $N2"
  [c_ctx_graph_reuse]="$TC --samples $SC --tag navtrain_ctx_graph_reuse --ref_tag none_navtrain --vision perimg_graph --prefill graph --prep threads --reuse_prev 1"
  [s1_eager_fp16]="$SB --config $C1 --variant eager --out $SO/s1_eager_fp16.json"
  [s1_graph_fp16]="$SB --config $C1 --variant graph_post_cw --out $SO/s1_graph_post_cw_fp16.json"
  [s1_eager_fp32]="$SB --config $C1 --variant eager --fp32 --out $SO/s1_eager_fp32.json"
  [s1_graph_fp32]="$SB --config $C1 --variant graph_post_cw --fp32 --out $SO/s1_graph_post_cw_fp32.json"
  [s2_eager_fp16]="$SB --config $C2 --variant eager --out $SO/s2_eager_fp16.json"
  [s2_graph_fp16]="$SB --config $C2 --variant graph_post_cw --out $SO/s2_graph_post_cw_fp16.json"
  [s2_eager_fp32]="$SB --config $C2 --variant eager --fp32 --out $SO/s2_eager_fp32.json"
  [s2_graph_fp32]="$SB --config $C2 --variant graph_post_cw --fp32 --out $SO/s2_graph_post_cw_fp32.json"
  [eq_s1_graph]="$SB --config $C1 --variant graph_post_cw --equiv 40 --det_daf --out $SO/equiv_s1_graph_detdaf.json"
  [eq_s2_graph]="$SB --config $C2 --variant graph_post_cw --equiv 40 --det_daf --out $SO/equiv_s2_graph_detdaf.json"
  [eq_s2_eager_floor]="$SB --config $C2 --variant eager --equiv 40 --out $SO/equiv_s2_eager_floor.json"
)
ORDER=(hf_ref ctx_eager ctx_eager_reuse ctx_graph ctx_graph_reuse n2_lm_last c_n2_lm_last c_ctx_graph_reuse
       s1_eager_fp16 s1_graph_fp16 s1_eager_fp32 s1_graph_fp32 s2_eager_fp16 s2_graph_fp16 s2_eager_fp32 s2_graph_fp32
       eq_s1_graph eq_s2_graph eq_s2_eager_floor)
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
