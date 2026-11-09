#!/bin/bash
# 수정(2026-10-09): LaQ-AD Bench2Drive 폐쇄루프 평가 — run_evaluation_multi_hipad.sh 복제본.
#   차이: 에이전트 = team_code/hipad_b2d_laq_agent.py(온라인 Qwen 주입), config = configs/b2d/stage2/test.py, ckpt = results/hipad_b2d_laq_stage2/latest.pth,
#         GPU1 기본(2026-10-15 사용자 규칙: GPU1 전용·GPU0 금지), 첫 실행은 1개 route 분할만.
#   전제: carla 클라이언트 모듈(cp310) + 외부에서 기동된 CARLA 0.9.15 서버(포트별). 아래 기동 전 점검 참조.
#   ⚠ 폐쇄루프 평가 기동은 사용자 승인 사항(CLAUDE.md).
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
CONFIG_NAME=hipad_b2d_laq_stage2  # 수정(2026-10-09): ckpt results/<name>/, 결과 evaluation/<name>/
# 수정(2026-10-15): config 경로 이동 configs/b2d/hipad_b2d_laq_stage2_test.py → configs/b2d/stage2/test.py, 트리 루트는 스크립트 위치로 계산
HIPAD_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
TEAM_AGENT=bench2drive/leaderboard/team_code/hipad_b2d_laq_agent.py
TEAM_CONFIG=$HIPAD_ROOT/configs/b2d/stage2/test.py+$HIPAD_ROOT/results/$CONFIG_NAME/latest.pth+$CONFIG_NAME
PLANNER_TYPE=traj
BASE_ROUTES=bench2drive/leaderboard/data/splits16/bench2drive220
SAVE_PATH=evaluation/$CONFIG_NAME
BASE_CHECKPOINT_ENDPOINT=evaluation/$CONFIG_NAME/$CONFIG_NAME
export LAQ_QWEN_PATH=${LAQ_QWEN_PATH:-$HIPAD_ROOT/ckpts/qwen2-vl-2b}
GPU_RANK_LIST=(${GPU_RANK_LIST:-1})  # 수정(2026-10-15): 사용자 규칙 변경 — GPU1 전용, GPU0 금지
TASK_LIST=(${TASK_LIST:-0})       # 전체: TASK_LIST="0 1 ... 15" GPU_RANK_LIST="1 1 ... 1"
# 수정(2026-10-09): 기동 전 점검 — 조건이 안 되면 조용히 진행하지 않고 즉시 실패한다.
#   (1) 파이썬: 주행 모델·Qwen 이 깔린 hipad_bw env 로 고정(run_evaluation.sh 는 PATH 의 python3 를 쓴다).
#   (2) carla 클라이언트 모듈: run_evaluation.sh 가 만드는 PYTHONPATH 그대로 import 가능해야 한다.
#       CARLA 0.9.15 배포의 py3.7 egg 는 py3.10 에서 import 불가 → PythonAPI/carla/dist 의 cp310 wheel 설치 필요(승인 사항).
#   (3) CARLA 서버: 로컬 수정된 leaderboard_evaluator.py 는 서버를 직접 띄우지 않고 이미 떠 있는 서버(포트 30000+200*i)에
#       접속한다("local host carla running"). 해당 포트가 열려 있지 않으면 중단.
export PATH=/home1/irteam/data-vol1/anaconda3/envs/hipad_bw/bin:$PATH
export B2D_CARLA_ROOT=${B2D_CARLA_ROOT:-/home1/irteam/data-vol1/src/carla}
CR=$B2D_CARLA_ROOT
PYTHONPATH="$PYTHONPATH:$CR/PythonAPI:$CR/PythonAPI/carla:$CR/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" \
    python3 -c "import carla" 2>/dev/null || { echo "[중단] carla 모듈 import 실패 (B2D_CARLA_ROOT=$CR). CARLA 0.9.15 PythonAPI(cp310) 설치 필요"; exit 1; }
for ((i=0; i<${#TASK_LIST[@]}; i++ )); do
    P=$((BASE_PORT + i * 200))
    (exec 3<>/dev/tcp/127.0.0.1/$P) 2>/dev/null || { echo "[중단] CARLA 서버가 포트 $P 에 없음 — CarlaUE4.sh -carla-rpc-port=$P 를 먼저 띄울 것"; exit 1; }
done
mkdir -p "$SAVE_PATH"   # 수정(2026-10-09): 점검 통과 후에만 생성
for ((i=0; i<${#TASK_LIST[@]}; i++ )); do
    PORT=$((BASE_PORT + i * 200)); TM_PORT=$((BASE_TM_PORT + i * 200))
    ROUTES="${BASE_ROUTES}_${TASK_LIST[$i]}.xml"
    CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.json"
    GPU_RANK=${GPU_RANK_LIST[$i]}
    [ "$GPU_RANK" = "0" ] && { echo "GPU0 사용 금지(2026-10-15 사용자 규칙)"; exit 1; }
    bash -e bench2drive/leaderboard/scripts/run_evaluation.sh \
        $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG \
        $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK
    sleep 10
done

# 수정(2026-10-09): 집계 — 220 route 가 다 모였을 때만 점수를 낸다(실패 route 제외 금지).
echo "집계: python tools/eval_b2d_laq_closedloop.py $SAVE_PATH --write-merged   (부분 점검은 --expect N --subset)"
