#!/bin/bash
# 수정(2026-08-04): nuScenes open-loop 평가 — HiP-AD Table 4 지표 산출.
#   사용자 지시: stage1 완주 후 곧바로 stage2 로 넘어가지 말고 반드시 평가부터 한다.
#   stage1 ckpt 로도 Table 4 의 4개 열 중 3개(det mAP/NDS · map mAP · track AMOTA)가 나온다
#   — tracking 은 학습 태스크가 아니라 temporal instance bank 의 instance_ids 를 track ID 로
#   쓰는 후처리 평가이기 때문. motion 은 stage2(task_select 에 motion)에서만 산출된다.
#
# 사용법: bash tools/eval_nusc_stage.sh <stage1|stage2>
set -u
cd /workspace/src/HiP-AD
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0   # GPU1 사용 금지(사용자 지시)
PY=/opt/anaconda3/envs/hipad_bw/bin/python

STAGE=${1:-stage1}
# 수정(2026-08-25): run 이름을 인자로 받도록 일반화.
#   [이유] 경로가 baseline(hipad_nusc_<stage>)에 하드코딩되어 있어 VLA run
#   (results/hipad_nusc_vla_stage1, config configs/hipad_nusc_vla_stage1.py)을
#   평가할 수 없었다. 스크립트를 복제하면 잠금·출력 규약이 갈라지므로 파라미터화한다.
#   인자 미지정 시 동작은 종전과 완전히 동일하다(하위 호환).
#   사용법: bash tools/eval_nusc_stage.sh <stage1|stage2> [run_name] [config_path]
RUN=${2:-hipad_nusc_${STAGE}}
CFG=${3:-configs/hipad_trainval_nusc_${STAGE}.py}
#수정(2026-08-25): 4번째 인자로 추가 --cfg-options 를 받는다(공백 구분 문자열).
#  100epoch 체크포인트 하나로 도는 ablation(예: qwen 캐시 디렉터리를 뒤섞은 것으로 교체)에
#  필요하다. 미지정 시 종전과 동일하게 동작한다.
EXTRA_OPTS=${4:-}
CKPT=results/${RUN}/latest.pth
OUTDIR=evaluation/${RUN}
mkdir -p "$OUTDIR"
if [ ! -f "$CFG" ]; then echo "[eval] config $CFG 없음 — 중단"; exit 1; fi

if [ ! -f "$CKPT" ]; then echo "[eval] $CKPT 없음 — 평가 skip"; exit 1; fi

# 수정(2026-08-11): 동시 실행 방지 잠금.
#   근본 원인 — 감독 스크립트(학습 종료 자동 실행)와 수동 실행이 겹치면 두 프로세스가
#   같은 $OUTDIR/results_nusc.json 에 동시에 write 하여 JSON 이 구조적으로 깨진다
#   (실측: "detection_name": ""translation": ... 처럼 값 중간에 다른 레코드가 끼어듦).
#   출력 경로가 stage 로 고정되어 있어 발생하는 문제이므로, 경로를 바꾸는 대신
#   같은 stage 평가가 둘 이상 돌지 못하게 배타 잠금을 건다.
exec 9>"$OUTDIR/.eval.lock"
if ! flock -n 9; then
  echo "[eval] $OUTDIR 에 대한 평가가 이미 실행 중이다 — 중복 실행 중단(출력 파손 방지)"
  exit 2
fi
echo "[$(date)] NUSC ${RUN} EVAL 시작: ckpt=$CKPT cfg=$CFG"
$PY tools/test.py "$CFG" "$CKPT" \
  --eval bbox \
  --out "$OUTDIR/results.pkl" \
  --cfg-options evaluation.jsonfile_prefix="$OUTDIR" $EXTRA_OPTS \
  2>&1 | tee "$OUTDIR/eval_$(date +%m%d_%H%M).log"
rc=${PIPESTATUS[0]}
echo "[$(date)] NUSC ${RUN} EVAL 종료 rc=$rc (결과: $OUTDIR/)"
exit $rc
