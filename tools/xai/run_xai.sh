#!/usr/bin/env bash
# 수정(2026-10-20): 사용자 승인 — 설명가능성 4가지(학습 변경 없음)
# LaQ-AD 설명가능성 전체 실행(2단계 학습이 끝난 뒤). GPU 단계는 GPU1 전용(GPU0 금지), 나머지는 CPU.
#   사용: bash tools/xai/run_xai.sh <run_name> [config] [checkpoint]
#     기본 run_name=hipad_nusc_laq_v2_teacher_stage2, config=configs/nuscenes/v2_teacher/stage2/test.py,
#     checkpoint=results/<run>/latest.pth, 평가 결과 evaluation/<run>/results.pkl (설명–행동 일치에 필요)
#   출력: results/<run>_xai/{attention,counterfactual,scorer_subscores.npz}  (샤드·뱅크 상태 — 재개 가능)
#         evaluation/<run>_xai/{rationale,consistency,figures,attention_summary.*,counterfactual_summary.*}
#   환경변수: SKIP_GPU=1 (CPU 단계만), WORKERS(기본 4), XAI_SCORER_DUMP / XAI_SCORER(계획 점수기 val 덤프·모델, 있으면 패널)
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=${1:-hipad_nusc_laq_v2_teacher_stage2}
CFG=${2:-configs/nuscenes/v2_teacher/stage2/test.py}
CKPT=${3:-results/${RUN}/latest.pth}
RES=evaluation/${RUN}/results.pkl
OUT_R=results/${RUN}_xai
OUT_E=evaluation/${RUN}_xai
PY=/home1/irteam/data-vol1/anaconda3/envs/hipad_bw/bin/python
export PYTHONPATH=. OPENBLAS_CORETYPE=SkylakeX
WORKERS=${WORKERS:-4}
mkdir -p "$OUT_R" "$OUT_E"

echo "[1/6] 교사 설명문·정확도 (CPU)"
CUDA_VISIBLE_DEVICES="" $PY tools/xai/xai_rationale.py --out_dir "$OUT_E/rationale"

echo "[2/6] 설명–행동 일치 (CPU, 정의 재현 검사 포함)"
CUDA_VISIBLE_DEVICES="" $PY tools/xai/xai_consistency.py --gt_sanity --out_dir "$OUT_E/consistency_gt_sanity"
[ -f "$RES" ] || { echo "평가 결과 $RES 없음 — tools/test.py 로 먼저 평가할 것"; exit 1; }
CUDA_VISIBLE_DEVICES="" $PY tools/xai/xai_consistency.py --results "$RES" --out_dir "$OUT_E/consistency"

if [ "${SKIP_GPU:-0}" != "1" ]; then
  [ -f "$CKPT" ] || { echo "체크포인트 $CKPT 없음"; exit 1; }
  echo "[3/6] 어텐션 기록 (GPU1, ~0.33 s/표본 → val 6,019 ≈ 35 분)"
  CUDA_VISIBLE_DEVICES=1 $PY tools/xai/xai_attention.py "$CFG" "$CKPT" --out "$OUT_R/attention" --workers "$WORKERS" --need_mib 12000
  echo "[4/6] 교사 토큰 가리기 반사실 (GPU1, 15 조건 ~3.9 s/표본 → val ≈ 6.5 시간)"
  CUDA_VISIBLE_DEVICES=1 $PY tools/xai/xai_counterfactual.py run "$CFG" "$CKPT" --out "$OUT_R/counterfactual" --workers "$WORKERS" --need_mib 12000
fi
cp "$OUT_R/attention/summary.json" "$OUT_E/attention_summary.json"; cp "$OUT_R/attention/summary.md" "$OUT_E/attention_summary.md"
cp "$OUT_R/counterfactual/summary.json" "$OUT_E/counterfactual_summary.json"; cp "$OUT_R/counterfactual/summary.md" "$OUT_E/counterfactual_summary.md"

echo "[5/6] 계획 점수기 후보별 하위 점수(있을 때만)"
SC_ARGS=()
if [ -n "${XAI_SCORER_DUMP:-}" ] && [ -n "${XAI_SCORER:-}" ]; then
  CUDA_VISIBLE_DEVICES="" $PY tools/xai/xai_counterfactual.py scorer --val_dump "$XAI_SCORER_DUMP" --scorer "$XAI_SCORER" --out "$OUT_R/scorer_subscores.npz"
  SC_ARGS=(--scorer_npz "$OUT_R/scorer_subscores.npz")
else
  echo "  XAI_SCORER_DUMP/XAI_SCORER 미지정 — 패널 (d) 생략"
fi

echo "[6/6] 장면 그림 (CPU, 가리기 효과가 큰 표본 8 개)"
CUDA_VISIBLE_DEVICES="" $PY tools/xai/fig_xai_scene.py --attn_dir "$OUT_R/attention" --cf_dir "$OUT_R/counterfactual" \
  --select max_effect --n 8 --results "$RES" "${SC_ARGS[@]}" --out_dir "$OUT_E/figures"
echo "[완료] $OUT_R , $OUT_E"
