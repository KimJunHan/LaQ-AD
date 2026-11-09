#!/bin/bash
# 수정(2026-10-09): LaQ-AD NAVSIM navtest EPDMS 채점 (devkit env 에서 실행; hipad_bw 아님).
#   입력: NavsimDataset.evaluate 가 쓴 navsim_pred.pkl ({scene token: (8,3) ego rear-axle 국소 포즈})
#   공식 진입점: $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py (navtest = 1단계 분할)
#   지표 캐시가 없으면 먼저 만든다(scripts/evaluation/run_metric_caching.sh 와 동일 명령).
#   devkit 러너는 실패/캐시누락 장면을 빼고 평균을 내므로(run_pdm_score_one_stage.py:256-262,288-291)
#   navsim_check_pdm_csv.py 로 navtest 전 장면 valid 를 강제한다 — 아니면 점수를 보고하지 않는다.
#   필요 환경변수: NAVSIM_DEVKIT_ROOT, OPENSCENE_DATA_ROOT, NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION, NAVSIM_EXP_ROOT
set -euo pipefail
PRED=${1:?usage: run_navsim_eval.sh <navsim_pred.pkl> <out_dir> [expected_infos.pkl]}
OUT=${2:?out_dir}
INFOS=${3:-data/infos/navsim_navtest.pkl}
HERE=$(cd "$(dirname "$0")" && pwd)
: "${NAVSIM_DEVKIT_ROOT:?}" "${OPENSCENE_DATA_ROOT:?}" "${NUPLAN_MAPS_ROOT:?}" "${NAVSIM_EXP_ROOT:?}"
CACHE=${NAVSIM_METRIC_CACHE:-$NAVSIM_EXP_ROOT/metric_cache_navtest}
mkdir -p "$OUT"
if [ ! -d "$CACHE" ]; then
  python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split=navtest metric_cache_path="$CACHE"
fi
PYTHONPATH="$HERE:${PYTHONPATH:-}" python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py" \
  "hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$HERE/navsim_hydra]" \
  train_test_split=navtest agent=navsim_laq_agent agent.pred_path="$(realpath "$PRED")" \
  experiment_name=navsim_laq metric_cache_path="$CACHE" output_dir="$OUT" \
  2>&1 | tee "$OUT/navsim_eval.log"
CSV=$(ls -t "$OUT"/*.csv | head -1)
python "$HERE/navsim_check_pdm_csv.py" --csv "$CSV" --expected-tokens "$INFOS"
