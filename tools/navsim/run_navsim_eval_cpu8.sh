#!/bin/bash
# 수정(2026-10-19): 사용자 승인 — NAVSIM 권장안 8건
#   run_navsim_eval.sh(수정 안 함)와 같은 공식 1단계 EPDMS 채점(run_pdm_score_one_stage.py + navsim_laq_agent) 래퍼. 다른 점:
#     - CPU 상한: worker.threads_per_node=8 (devkit 기본 null = CPU 전부; ray_distributed_no_torch.yaml:4)
#     - 기본 info = data/infos/navsim_20261019/navsim_navtest.pkl (→ navsim_20261016, 장면 12,146)
#     - 환경변수 기본값을 이 컨테이너의 실제 경로로 채운다(NAS eval_prep_20261019 의 정상 링크·경로 바꾼 캐시 목록)
#     - 지표 캐시를 **절대 새로 만들지 않는다**: 캐시 목록(metadata/*.csv)이 없거나 navtest token 이 하나라도 빠지면 즉시 실패
#     - DRY=1 이면 hydra 조립 결과(--cfg job)만 출력하고 끝낸다(채점 없음)
#   env: /home1/irteam/data-vol1/anaconda3/envs/navsim (navsim 패키지 editable 설치). 사용:
#     bash tools/navsim/run_navsim_eval_cpu8.sh <navsim_pred.pkl> <out_dir> [expected_infos.pkl]
set -euo pipefail
PRED=${1:?usage: run_navsim_eval_cpu8.sh <navsim_pred.pkl> <out_dir> [expected_infos.pkl]}
OUT=${2:?out_dir}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
INFOS=${3:-$REPO/data/infos/navsim_20261019/navsim_navtest.pkl}
PREP=/home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/eval_prep_20261019
PY=${NAVSIM_PY:-/home1/irteam/data-vol1/anaconda3/envs/navsim/bin/python}
export CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-/home1/irteam/data-vol1/src/navsim_ref}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-$PREP/openscene}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-/home1/irteam/data-vol1/nas_mirror/navsim/download/maps}
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-$PREP}
THREADS=${THREADS:-8}
[ "$THREADS" -le 8 ] || { echo "THREADS=$THREADS > 8 (상한)"; exit 1; }
CACHE=${NAVSIM_METRIC_CACHE:-$PREP/metric_cache_navtest_v2_remap}

# 지표 캐시 확인(재생성 금지): metadata csv 의 token 이 navtest 필터 token 을 모두 포함해야 한다
ls "$CACHE"/metadata/*.csv >/dev/null 2>&1 || { echo "[중단] 지표 캐시 목록 없음: $CACHE/metadata/*.csv (자동 재생성하지 않음)"; exit 1; }
"$PY" -I - "$CACHE" "$NAVSIM_DEVKIT_ROOT" <<'EOF'
import sys, glob, os, yaml
cache, dk = sys.argv[1], sys.argv[2]
fs = sorted(glob.glob(os.path.join(cache, "metadata", "*.csv")))
# devkit MetricCacheLoader 와 같이 첫 csv 만 읽는다(dataloader.py:313-317: 첫 줄 헤더, token = 경로의 뒤에서 두 번째)
paths = open(fs[0]).read().splitlines()[1:]
toks = {p.split("/")[-2] for p in paths}
filt = yaml.safe_load(open(os.path.join(dk, "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml")))
want = set(filt["tokens"])
miss = want - toks
probe = [p for p in paths[:20] if not os.path.exists(p)]
if miss or probe:
    sys.exit(f"[중단] 캐시 누락 token {len(miss)}, 경로 없음(앞 20 중) {len(probe)} — 재생성하지 않음")
print(f"[캐시 확인] {cache}: navtest token {len(want)} 전부 있음")
EOF

mkdir -p "$OUT"
ARGS=("hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$HERE/navsim_hydra]"
      train_test_split=navtest agent=navsim_laq_agent "agent.pred_path=$(realpath "$PRED")"
      experiment_name=navsim_laq "metric_cache_path=$CACHE" "output_dir=$(realpath "$OUT")"
      "worker.threads_per_node=$THREADS")
if [ "${DRY:-0}" = "1" ]; then
  PYTHONPATH="$HERE:${PYTHONPATH:-}" "$PY" "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py" "${ARGS[@]}" --cfg job
  exit 0
fi
PYTHONPATH="$HERE:${PYTHONPATH:-}" "$PY" "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py" "${ARGS[@]}" \
  2>&1 | tee "$OUT/navsim_eval.log"
CSV=$(ls -t "$OUT"/*.csv | head -1)
"$PY" "$HERE/navsim_check_pdm_csv.py" --csv "$CSV" --expected-tokens "$INFOS"
