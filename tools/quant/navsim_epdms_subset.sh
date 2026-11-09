#!/bin/bash
# 수정(2026-11-09): N-A13 — navtest **부분집합**(경로 검증용 --limit-scenes) EPDMS 채점. 새 파일.
#   tools/navsim/run_navsim_eval_cpu8.sh(전체 navtest, 체인과 같은 경로)와 같은 공식 1단계 채점(run_pdm_score_one_stage + navsim_laq_agent,
#   워커 8, 지표 캐시 재생성 없음)에 장면 필터 token 만 부분 info 의 eval_tokens 로 덮어쓴다(results/smoke_navsim_prep_20261106 스모크와 같은 방식).
#   전체 navtest 는 이 스크립트가 아니라 run_navsim_eval_cpu8.sh 를 쓴다. 호출 전 scripts/navsim/navsim_env.sh 를 source 한 셸이어야 한다.
#   사용: bash tools/quant/navsim_epdms_subset.sh <navsim_pred.pkl> <out_dir> <부분 info pkl>
set -euo pipefail
PRED=${1:?pred}; OUT=${2:?out_dir}; SUB=${3:?subset infos}
: "${NAVSIM_PY:?navsim_env.sh 를 source 할 것}" "${NAVSIM_DEVKIT_ROOT:?}" "${NAVSIM_METRIC_CACHE:?}"
REPO=$(cd "$(dirname "$0")/../.." && pwd); TOOLS=$REPO/tools/navsim
export CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX
TOK=$("$NAVSIM_PY" -I -c "import pickle,sys;print(','.join(pickle.load(open(sys.argv[1],'rb'))['metadata']['eval_tokens']))" "$SUB")
mkdir -p "$OUT"
PYTHONPATH="$TOOLS" "$NAVSIM_PY" "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py" \
  "hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$TOOLS/navsim_hydra]" \
  train_test_split=navtest "train_test_split.scene_filter.tokens=[$TOK]" agent=navsim_laq_agent \
  "agent.pred_path=$(realpath "$PRED")" experiment_name=navsim_laq_ptq_subset "metric_cache_path=$NAVSIM_METRIC_CACHE" \
  "output_dir=$(realpath "$OUT")" worker.threads_per_node=8 > "$OUT/navsim_eval.log" 2>&1
CSV=$(ls -t "$OUT"/*.csv | head -1)
"$NAVSIM_PY" "$TOOLS/navsim_check_pdm_csv.py" --csv "$CSV" --expected-tokens "$SUB" | tee "$OUT/epdms.txt"
