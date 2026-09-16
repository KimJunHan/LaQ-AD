#!/bin/bash
# 수정: VLA-v0 — Bench2Drive mini-10 데이터셋 전체를 시각화한다.
#   각 scenario: visualize.py(per-cam 3D-bbox 렌더) → combine_views.py(6cam+top-down 합성).
#   출력: vis_b2d/<scenario>/camera|combine + combine.mp4
# visualize.py 는 cwd 기준 ./maps, ./leaderboard/data/weather.xml 를 요구하므로
# 두 심볼릭 링크를 가진 RUN_DIR 에서 실행한다.
set -u

HIP=/workspace/src/HiP-AD
DATA="$HIP/data/bench2drive/v1"
TOOLS="$HIP/bench2drive/tools"
RUN_DIR=/tmp/b2dvis            # ./maps, ./leaderboard 링크가 있는 작업 디렉토리
OUT="$HIP/vis_b2d"

# RUN_DIR 준비 (maps, leaderboard 링크)
mkdir -p "$RUN_DIR"
[ -e "$RUN_DIR/maps" ]        || ln -s "$HIP/data/bench2drive/maps"   "$RUN_DIR/maps"
[ -e "$RUN_DIR/leaderboard" ] || ln -s "$HIP/bench2drive/leaderboard" "$RUN_DIR/leaderboard"

# mini-10 (bench2drive/docs/bench2drive_mini_10.json 기준)
SCENARIOS=(
  AccidentTwoWays_Town12_Route1444_Weather0
  Accident_Town03_Route156_Weather0
  ConstructionObstacle_Town05_Route68_Weather8
  ControlLoss_Town11_Route401_Weather11
  DynamicObjectCrossing_Town02_Route13_Weather6
  HardBreakRoute_Town01_Route30_Weather3
  OppositeVehicleTakingPriority_Town13_Route600_Weather2
  ParkedObstacle_Town10HD_Route371_Weather7
  VehicleTurningRoute_Town15_Route443_Weather1
  YieldToEmergencyVehicle_Town04_Route165_Weather7
)

cd "$RUN_DIR" || exit 1
N=${#SCENARIOS[@]}
i=0
for s in "${SCENARIOS[@]}"; do
  i=$((i+1))
  src="$DATA/$s"
  if [ ! -d "$src" ]; then
    echo "[$i/$N] SKIP (missing): $s"
    continue
  fi
  # Town 번호 추출: Town12 -> 12, Town10HD -> 10HD
  town=$(echo "$s" | sed -nE 's/.*Town([0-9A-Za-z]+)_.*/\1/p')
  echo "=================================================================="
  echo "[$i/$N] $s  (Town=$town)"
  echo "=================================================================="
  # 1) per-cam 3D-bbox 렌더 (출력 기본값 = $OUT)
  python "$TOOLS/visualize.py" -f "$src" -m "$town" -o "$OUT"
  # 2) 6cam + top-down 합성
  python "$TOOLS/combine_views.py" -s "$OUT/$s" --fps 10
done

echo "ALL DONE: mini-10 visualization → $OUT"
