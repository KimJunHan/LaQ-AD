#!/bin/bash
# 수정(2026-11-09): 새 파일 — B2D 폐쇄루프(CARLA 0.9.15) 실행 스크립트: 교사 에이전트 + E3·nuImages·E2 학생(stage2 test_e3_nuim).
#   옛 run_evaluation_b2d_laq.sh(옛 CARLA 경로·GPU1·옛 에이전트·옛 config)는 그대로 두고 새로 만든다.
#   ⚠ 폐쇄루프 평가 기동은 사용자 승인 사항(CLAUDE.md). 먼저 DRYRUN=1 로 점검만 할 것(CARLA·GPU 를 쓰지 않음).
#
#   구성: CARLA 서버(start_carla_gpu.sh, -graphicsadapter=$GPU, 이 호스트는 adapter 번호 = nvidia-smi 번호 — 11-04 확인)
#         + leaderboard_evaluator(HiP-AD 사본 — 미리 띄운 서버에 접속하는 판) + 에이전트 hipad_b2d_laq_teacher_agent.py
#         + 교사 빌더 LAQ_TEACHER_BUILDER=laq_teacher_builder_b2d:build (병합 교사, 캐시 _meta 대조).
#   파이썬 = hipad_bw(hipad_env.sh) + carla cp310 모듈(/home1/irteam/data-vol1/carla/pylib_cp310). cwd = 저장소(config plugin_dir 상대경로).
#   경로 집합: ROUTE_SET=dev10(10 경로, 먼저) → b2d220(220 경로; dev10 완료 표식이 있어야 시작, FORCE_220=1 로만 우회).
#   220 분할: NUM_SPLITS=N 이면 경로 순서대로 연속 N 조각(s<k>of<N>.xml), SPLITS="0 3 5" 로 이번에 돌릴 조각 지정(기본 전부),
#             PARALLEL=P 개를 동시에(각각 CARLA 서버 1 개 + 에이전트 1 개). 조각별 checkpoint json 으로 재개(RESUME).
#   포트: 슬롯 s 의 시작 후보 = BASE_PORT + 200·s (RPC P, 스트리밍 P+1, 보조 P+2), TM = BASE_TM_PORT + 200·s.
#         후보가 사용 중(ss -ltn)이거나 다른 실행이 잠갔으면(flock $LOCK_DIR/carla_port_P.lock) +200 씩 옮긴다. 평가기도 TM 포트는 find_free_port 로 다시 고른다.
#   서버·평가기 충돌(시뮬레이터 crash) 시: 서버 프로세스 그룹 TERM(30 s 뒤 KILL)·포트 해제 확인 → 새 서버 → 같은 checkpoint 로 재개.
#     재시작 전 b2d_ckpt_tool.py prep: checkpoint 백업(logs/n<N>/*.ckpt_before_<실행시각>_try<a>.json — 덮어쓰일 기록 보존) +
#     평가기가 재개 대상으로 보지 않는 'Agent couldn't be set up'·'sensors were invalid' 경로로 progress 되돌림(건너뛰기 방지, *.resume.log).
#     MAX_RESTARTS = 진전(최종 기록 수 증가) 없는 재시작 허용 횟수. 같은 RUN_NAME·조각 동시 실행은 flock(locks/)으로 막는다.
#   결과: evaluation/$RUN_NAME/<set>/{routes,ckpt,logs}/n<N>/ (조각 수별), save/, score_n<N>/. 집계는 이번 NUM_SPLITS 조각 0..N-1 의
#     checkpoint 만 복사해 읽는다(tools/eval_b2d_laq_closedloop.py, 분모 = 경로 수 고정, 기반시설 실패가 있으면 점수 안 냄; 실패 시 DONE 안 씀).
#     AGG_ONLY=1 = 실행 없이 집계만.
#
#   예상 소요(추정, 미측정 — 첫 Dev10 에서 다시 잴 것):
#     경로당 시뮬레이션 ≈ 60–100 s(1,200–2,000 틱). 교사 = 0.5 s 마다 HF generate 547 토큰 + 2 차 순전파 ≈ 9.4 s/호출(NAVSIM 같은 경로 H200 배치 1 실측,
#     docs/latency/navsim_idle_20261105.md §1b; 복제 배치 16 은 문법 처리 ×16 으로 ≈ 11 s 추정) → 120–200 호출 ≈ 20–37 분.
#     렌더링 6 캠 1.31 FPS(GPU2 공유 실측) → 15–25 분. 학생 ≈ 0.1 s/틱 → 2–4 분. 합 ≈ 40–65 분/경로.
#     Dev10 ≈ 7–11 h(PARALLEL=1) / ≈ 3–5 h(PARALLEL=3). 220 ≈ 6–10 일(PARALLEL=1) / ≈ 2.5–4 일(PARALLEL=3–4; 렌더링·교사가 같은 GPU 를 나눠 써 선형 단축 아님).
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ts() { date '+%F %T'; }
die() { echo "[$(ts)] [중단] $*" >&2; exit 1; }
log() { echo "[$(ts)] $*"; }

# ---------------------------------------------------------------- 변수
ROUTE_SET=${ROUTE_SET:-dev10}                     # dev10 | b2d220 | <routes xml 경로>
GPU=${GPU:-2}                                     # B2D 전용 GPU2(CLAUDE.md 배정)
RUN_NAME=${RUN_NAME:-hipad_b2d_laq_v2_teacher_stage2_e3_nuim}
CFG=${CFG:-$R/configs/b2d/v2_teacher/stage2/test_e3_nuim.py}
CKPT=${CKPT:-$R/results/hipad_b2d_laq_v2_teacher_stage2_e3_nuim/iter_176064_ema.pth}   # 체크포인트는 아직 없음 — 변수로 바꿔 쓴다
export LAQ_TEACHER_FOLD=${LAQ_TEACHER_FOLD:-}    # A|B (필수 — 폐쇄루프 교사 fold 는 사용자 결정 B-C12(b))
export LAQ_TEACHER_MODEL=${LAQ_TEACHER_MODEL:-${LAQ_TEACHER_FOLD:+$R/results/vla_teacher/b2d/b2d_merged_fold${LAQ_TEACHER_FOLD}_qa}}
export LAQ_TEACHER_LAYER=${LAQ_TEACHER_LAYER:-18}
export LAQ_TEACHER_DTYPE=${LAQ_TEACHER_DTYPE:-bf16}
export LAQ_TEACHER_ATTN=${LAQ_TEACHER_ATTN:-sdpa}
export LAQ_TEACHER_CACHE_ROOT=${LAQ_TEACHER_CACHE_ROOT:-$R/data/bench2drive/laq_cache}
export LAQ_TEACHER_GEN_BATCH=${LAQ_TEACHER_GEN_BATCH:-}   # 비우면 캐시 _meta 의 batch(캐시와 같은 배치 크기로 복제)
export LAQ_TEACHER_FWD_BATCH=${LAQ_TEACHER_FWD_BATCH:-}
export LAQ_TEACHER_BUILDER=laq_teacher_builder_b2d:build
NUM_SPLITS=${NUM_SPLITS:-1}
SPLITS=${SPLITS:-all}
PARALLEL=${PARALLEL:-1}
BASE_PORT=${BASE_PORT:-30000}
BASE_TM_PORT=${BASE_TM_PORT:-50000}
PORT_STRIDE=200
NEED_GB_PER_TASK=${NEED_GB_PER_TASK:-40}          # CARLA ≤ 9 GB(실측) + 교사 2B bf16·복제 배치 + 학생 — 추정
MAX_RESTARTS=${MAX_RESTARTS:-5}
SERVER_WAIT_S=${SERVER_WAIT_S:-300}
FORCE_220=${FORCE_220:-0}
DRYRUN=${DRYRUN:-0}                               # 1 = 점검만(CARLA·GPU 안 씀, 결과는 임시 폴더)
CARLA_ROOT_0915=/home1/irteam/data-vol1/carla/CARLA_0.9.15
CARLA_PYLIB=/home1/irteam/data-vol1/carla/pylib_cp310
START_CARLA=/home1/irteam/data-vol1/carla/tools/start_carla_gpu.sh
LOCK_DIR=${LOCK_DIR:-/home1/irteam/data-vol1/carla/.port_locks}
DEV10_XML=/home1/irteam/data-vol1/src/Bench2Drive/leaderboard/data/drivetransformer_bench2drive_dev10.xml
B2D220_XML=$R/bench2drive/leaderboard/data/bench2drive220.xml
TEAM_AGENT=$R/bench2drive/leaderboard/team_code/hipad_b2d_laq_teacher_agent.py
EVALUATOR=$R/bench2drive/leaderboard/leaderboard/leaderboard_evaluator.py

CKPT_TOOL=$R/bench2drive/leaderboard/scripts/b2d_ckpt_tool.py
AGG_ONLY=${AGG_ONLY:-0}                           # 1 = 실행 없이 집계만(이미 끝난 조각들; DRYRUN=1 과 함께면 DRY_DIR 안에서)

# ---------------------------------------------------------------- 점검(조건이 안 되면 진행하지 않음)
case "$GPU" in
    2) ;;
    0) [ "${HIPAD_ALLOW_GPU0:-0}" = 1 ] || die "GPU0 은 공용 — HIPAD_ALLOW_GPU0=1 일 때만" ;;
    *) [ "${B2D_ALLOW_GPU:-}" = "$GPU" ] || die "GPU$GPU 는 B2D 배정이 아님(B2D=GPU2). 꼭 써야 하면 B2D_ALLOW_GPU=$GPU (사용자 확인 뒤)" ;;
esac
case "$LAQ_TEACHER_FOLD" in A|B) ;; *) die "LAQ_TEACHER_FOLD=A|B 를 정해야 한다(폐쇄루프 교사 fold — 결정 B-C12(b))" ;; esac
[ -f "$CFG" ] || die "config 없음: $CFG"
for v in "$CFG" "$CKPT" "$RUN_NAME"; do case "$v" in *+*) die "'+' 가 든 경로/이름은 쓸 수 없다(TEAM_CONFIG 구분자): $v" ;; esac; done
case "$ROUTE_SET" in
    dev10) XML=$DEV10_XML; SET_TAG=dev10; EXPECT=10 ;;
    b2d220) XML=$B2D220_XML; SET_TAG=b2d220; EXPECT=220 ;;
    *.xml) XML=$ROUTE_SET; SET_TAG=custom_$(basename "$ROUTE_SET" .xml); EXPECT= ;;
    *) die "ROUTE_SET=$ROUTE_SET (dev10 | b2d220 | <xml>)" ;;
esac
[ -f "$XML" ] || die "경로 파일 없음: $XML"
[[ "$NUM_SPLITS" =~ ^[1-9][0-9]*$ ]] && [[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]] && [[ "$MAX_RESTARTS" =~ ^[0-9]+$ ]] \
    || die "NUM_SPLITS·PARALLEL 은 양의 정수, MAX_RESTARTS 는 0 이상 정수"
# 교사 캐시 대조: 기본은 필수. LAQ_TEACHER_CACHE_ROOT=none 이면 빌더가 대조를 생략(경고)한다 — 스크립트도 같은 뜻으로 허용(비권장).
cache_ok=1
if [ "$LAQ_TEACHER_CACHE_ROOT" = none ]; then
    log "[경고] LAQ_TEACHER_CACHE_ROOT=none — 교사 캐시 _meta 대조 생략(학습 캐시와 설정 불일치를 못 잡음, 스모크 전용)"
elif [ ! -d "$LAQ_TEACHER_CACHE_ROOT/context" ]; then
    cache_ok=0
fi
if [ "$DRYRUN" = 1 ]; then
    OUT=${DRY_DIR:-$(mktemp -d -t b2d_cl_dry_XXXX)}/$RUN_NAME/$SET_TAG
    log "DRYRUN — 결과 폴더 대신 $OUT (evaluation/ 는 건드리지 않음)"
    [ -f "$CKPT" ] || log "[경고] 학생 체크포인트 없음: $CKPT (DRYRUN 이라 계속)"
    [ -f "$LAQ_TEACHER_MODEL/teacher_meta.json" ] || log "[경고] 병합 교사 없음: $LAQ_TEACHER_MODEL (DRYRUN 이라 계속)"
    [ $cache_ok = 1 ] || log "[경고] 교사 캐시 없음: $LAQ_TEACHER_CACHE_ROOT (DRYRUN 이라 계속)"
else
    OUT=$R/evaluation/$RUN_NAME/$SET_TAG
    if [ "$AGG_ONLY" != 1 ]; then
        [ -f "$CKPT" ] || die "학생 체크포인트 없음: $CKPT"
        [ -f "$LAQ_TEACHER_MODEL/teacher_meta.json" ] || die "병합 교사 없음: $LAQ_TEACHER_MODEL"
        [ $cache_ok = 1 ] || die "교사 캐시 없음: $LAQ_TEACHER_CACHE_ROOT/context (_meta 대조용). 대조 없이 쓰려면 LAQ_TEACHER_CACHE_ROOT=none — 비권장"
        [ -f "$START_CARLA" ] || die "CARLA 기동 스크립트 없음: $START_CARLA"
        [ -x "$CARLA_ROOT_0915/CarlaUE4.sh" ] || die "CARLA 0.9.15 없음: $CARLA_ROOT_0915"
    fi
fi
if [ "$SET_TAG" = b2d220 ] && [ "$FORCE_220" != 1 ] && [ "$AGG_ONLY" != 1 ]; then
    [ -f "$R/evaluation/$RUN_NAME/dev10/DONE" ] || [ "$DRYRUN" = 1 ] \
        || die "Dev10 완료 표식 없음($R/evaluation/$RUN_NAME/dev10/DONE) — ROUTE_SET=dev10 을 먼저(우회는 FORCE_220=1)"
fi

# 실행 환경(hipad_bw + carla cp310). hipad_env.sh 는 cd 를 저장소로 한다.
source /home1/irteam/data-vol1/hipad_env.sh
cd "$R" || die "cd $R"
PY=$(command -v python)
export CARLA_ROOT=$CARLA_ROOT_0915
export PYTHONPATH=$CARLA_PYLIB:$CARLA_ROOT/PythonAPI/carla:$R/bench2drive/leaderboard:$R/bench2drive/leaderboard/team_code:$R/bench2drive/scenario_runner:$R${PYTHONPATH:+:$PYTHONPATH}
export SCENARIO_RUNNER_ROOT=$R/bench2drive/scenario_runner LEADERBOARD_ROOT=$R/bench2drive/leaderboard CHALLENGE_TRACK_CODENAME=SENSORS
export IS_BENCH2DRIVE=True PLANNER_TYPE=traj OPENBLAS_CORETYPE=SkylakeX TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID              # CUDA 번호 = nvidia-smi 번호(= CARLA -graphicsadapter 번호)
TEAM_CONFIG="$CFG+$CKPT+$RUN_NAME"

# 조각 수마다 하위 폴더(n<N>) — 다른 NUM_SPLITS 로 돌린 checkpoint 와 섞여 집계되지 않게.
NT=n$NUM_SPLITS
RDIR=$OUT/routes/$NT CDIR=$OUT/ckpt/$NT LDIR=$OUT/logs/$NT
mkdir -p "$RDIR" "$CDIR" "$LDIR" "$OUT/save" "$OUT/locks" || die "mkdir $OUT"
split_xml() { echo "$RDIR/${SET_TAG}_s$1of$NUM_SPLITS.xml"; }
split_ck()  { echo "$CDIR/${SET_TAG}_s$1of$NUM_SPLITS.json"; }
ALL_K=($(seq 0 $((NUM_SPLITS - 1))))             # 숫자 순서(glob 사전순 s10 < s2 문제를 피함)
ALL_XML=(); for k in "${ALL_K[@]}"; do ALL_XML+=("$(split_xml "$k")"); done

# ---------------------------------------------------------------- 경로 분할(연속 조각, 재실행 시 같은 내용인지 확인)
CUDA_VISIBLE_DEVICES="" "$PY" - "$XML" "$NUM_SPLITS" "$RDIR" "$SET_TAG" "${EXPECT:-0}" <<'EOF' || die "경로 분할 실패"
import os, sys
import xml.etree.ElementTree as ET
xml, n, out, tag, expect = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
tree = ET.parse(xml)
root = tree.getroot()
routes = list(root.iter("route"))
ids = [r.get("id") for r in routes]
if len(set(ids)) != len(ids):
    sys.exit(f"route id 중복: {xml}")
if expect and len(ids) != expect:
    sys.exit(f"{xml}: 경로 {len(ids)} != 기대 {expect}")
if n > len(routes):
    sys.exit(f"NUM_SPLITS {n} > 경로 수 {len(routes)}")
b = [len(routes) * k // n for k in range(n + 1)]
seen = []
for k in range(n):
    new = ET.Element(root.tag, root.attrib)
    for r in routes[b[k]:b[k + 1]]:
        new.append(r)
    data = ET.tostring(new, encoding="unicode")
    p = os.path.join(out, f"{tag}_s{k}of{n}.xml")
    if os.path.exists(p):
        if open(p).read() != data:
            sys.exit(f"{p} 가 이미 있고 내용이 다르다(분할을 바꾸면 재개 checkpoint 와 어긋남) — 새 RUN_NAME 을 쓰거나 직접 확인")
    else:
        open(p + ".tmp", "w").write(data); os.replace(p + ".tmp", p)
    seen += [r.get("id") for r in routes[b[k]:b[k + 1]]]
if seen != ids:
    sys.exit("분할 합집합 ≠ 원본")
print(f"[분할] {xml}: {len(ids)} 경로 → {n} 조각 {[b[k + 1] - b[k] for k in range(n)]}")
EOF

# ---------------------------------------------------------------- 사전 점검(파이썬 import·config·경로 파싱, CPU)
PRE_LOG="$LDIR/preflight_$(date +%Y%m%d_%H%M%S).log"
CUDA_VISIBLE_DEVICES="" "$PY" - "$CFG" "$XML" "${ALL_XML[@]}" > "$PRE_LOG" 2>&1 <<'EOF'

import sys
cfg_path, xml, splits = sys.argv[1], sys.argv[2], sys.argv[3:]
import carla
print(f"[점검] carla {carla.__file__}")
import leaderboard.leaderboard_evaluator  # noqa: F401
from leaderboard.utils.route_parser import RouteParser
cfgs = RouteParser.parse_routes_file(xml, "")
print(f"[점검] 경로 파싱 {len(cfgs)} (마을 {sorted({str(c.town) for c in cfgs})})")
parts = [[c.name for c in RouteParser.parse_routes_file(x, "")] for x in splits]
assert sum(parts, []) == [c.name for c in cfgs], "분할 xml 을 평가기 파서로 읽은 경로 합(조각 번호 순) ≠ 원본"
print(f"[점검] 분할 xml {len(splits)} 개(조각 0..{len(splits) - 1} 순)를 평가기 파서로 읽음: {[len(q) for q in parts]}")
import hipad_b2d_laq_teacher_agent as A  # noqa: F401  (기반 에이전트·mmcv·torch import 포함)
import laq_teacher_builder_b2d as TB
TB._mods()
print(f"[점검] 에이전트·교사 빌더 import OK ({A.get_entry_point()}, LAQ_TEACHER_BUILDER 모듈 {TB.__file__})")
from mmcv import Config
cfg = Config.fromfile(cfg_path)
dec = cfg.model.head.onedecoder_head
assert dec.get("num_stream_groups") == 1, f"num_stream_groups {dec.get('num_stream_groups')} != 1 (폐쇄루프 뱅크와 불가)"
e2 = dec.get("ego_input_embed")
assert e2 is not None and int(e2["in_dims"]) == 4, f"E2 ego_input_embed {e2}"
assert any(p["type"] == "Collect" for p in cfg.inference_only_pipeline), "inference_only_pipeline 없음"
print(f"[점검] config OK: E2 {e2}, num_stream_groups 1, fp16 {cfg.get('fp16')}")
EOF
pre_rc=$?
grep -E "^\[점검\]|Error|Traceback|assert" "$PRE_LOG"
[ $pre_rc = 0 ] || die "사전 점검 실패(rc $pre_rc) — $PRE_LOG"

# ---------------------------------------------------------------- 조각 목록·포트·checkpoint 도구
if [ "$SPLITS" = all ]; then SPL=("${ALL_K[@]}"); else SPL=($SPLITS); fi
for k in "${SPL[@]}"; do [[ "$k" =~ ^[0-9]+$ ]] && [ "$k" -lt "$NUM_SPLITS" ] || die "SPLITS 의 $k 가 범위 밖(0..$((NUM_SPLITS - 1)))"; done
declare -A SLOT_PID
cleanup() {   # 조각 서브셸에 TERM → 각 서브셸 trap 이 자기 평가기와 자기 CARLA 프로세스 그룹만 끝낸다
    for p in "${SLOT_PID[@]}"; do kill -TERM "$p" 2>/dev/null; done
    wait
}
trap 'cleanup; exit 130' INT TERM

# >>> funcs (tools/verify_b2d/test_run_script_ckpt_b2d.py 가 이 구간을 그대로 떼어 가짜 서버·평가기로 시험한다)
port_busy() { ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]($1|$(($1 + 1))|$(($1 + 2))|$2)\$"; }
ck_tool() { CUDA_VISIBLE_DEVICES="" "$PY" "$CKPT_TOOL" "$@"; }
PORT_FREE_WAIT_S=${PORT_FREE_WAIT_S:-120}
MAP_LOAD_WAIT_S=${MAP_LOAD_WAIT_S:-20}
SIM_FAIL_LIMIT=${SIM_FAIL_LIMIT:-3}       # 같은 경로 연속 'Simulation crashed' 이 횟수면 그 조각 중단(서버 일시 문제 여지를 둠)
AGENT_FAIL_LIMIT=${AGENT_FAIL_LIMIT:-2}   # 같은 경로 연속 'Agent crashed'·'Agent couldn't be set up' 이 횟수면 중단(에이전트 쪽은 재시작으로 잘 안 풀림)

stop_server() {  # $1 PGID(비면 kill 생략) $2 RPC 포트 $3 TM 포트 — TERM → 30 s 안에 그룹이 안 사라지면 KILL → 포트 해제 확인(항상, 최대 PORT_FREE_WAIT_S)
    local pg=$1 w=0
    if [ -n "$pg" ]; then
        kill -TERM -- -"$pg" 2>/dev/null
        while kill -0 -- -"$pg" 2>/dev/null; do
            sleep 1; w=$((w + 1))
            if [ $w -ge 30 ]; then log "서버 그룹 $pg 가 30 s 뒤에도 남음 — KILL"; kill -KILL -- -"$pg" 2>/dev/null; sleep 3; break; fi
        done
        kill -0 -- -"$pg" 2>/dev/null && log "[경고] 서버 그룹 $pg KILL 뒤에도 남음"
    fi
    w=0
    while port_busy "$2" "$3"; do
        [ $w = 0 ] && log "포트 $2(+1,+2)/TM $3 사용 중 — 해제 대기(최대 ${PORT_FREE_WAIT_S}s)"
        sleep 2; w=$((w + 2))
        [ $w -ge "$PORT_FREE_WAIT_S" ] && { log "[중단] 포트 $2(+1,+2)/TM $3 이 ${PORT_FREE_WAIT_S}s 뒤에도 사용 중 — ss -ltnp 로 확인"; return 1; }
    done
    return 0
}

run_split() {  # $1 조각 번호, $2 슬롯 — 서버 기동·평가·재시작. 자기 서버 프로세스 그룹만 죽인다.
    local k=$1 slot=$2 P TMP j=0
    local xml ck L
    xml=$(split_xml "$k"); ck=$(split_ck "$k"); L="$LDIR/${SET_TAG}_s${k}of${NUM_SPLITS}"
    # 같은 RUN_NAME·조각 동시 실행 방지(fd 8; 서버·평가기에는 넘기지 않음)
    exec 8>"$OUT/locks/${NT}_s$k.lock"
    flock -n 8 || { log "[s$k] 같은 RUN_NAME·조각($OUT, $NT s$k)이 이미 실행 중 — 중단"; return 1; }
    while :; do   # 포트 + 잠금(fd 9)
        P=$((BASE_PORT + PORT_STRIDE * (slot + j))); TMP=$((BASE_TM_PORT + PORT_STRIDE * (slot + j)))
        [ $j -gt 50 ] && { log "[s$k] 빈 포트 없음"; return 1; }
        if [ "$DRYRUN" = 1 ]; then port_busy $P $TMP || break
        else
            mkdir -p "$LOCK_DIR"; exec 9>"$LOCK_DIR/carla_port_$P.lock"
            if flock -n 9 && ! port_busy $P $TMP; then break; fi
            exec 9>&-
        fi
        j=$((j + PARALLEL))
    done
    log "[s$k] 포트 RPC $P(+1,+2) TM $TMP, 경로 $(grep -c '<route ' "$xml"), checkpoint $ck"
    if [ "$DRYRUN" = 1 ]; then log "[s$k] DRYRUN — 서버·평가기 기동 안 함"; return 0; fi
    local a=0 noprog=0 st pg= epid= rc ok0 ok1 bk pp prc cur_fail rep rid kind key n lim
    local rts; rts=$(date +%Y%m%d_%H%M%S)   # 백업 이름에 실행 시각 — 다음 실행이 이전 백업을 덮어쓰지 않게
    local -A streak=() nstreak=()           # "route_id 종류" → 연속 기반시설 실패 횟수(바로 앞 시도까지)
    trap '[ -n "$epid" ] && kill -TERM "$epid" 2>/dev/null; [ -n "$pg" ] && kill -KILL -- -"$pg" 2>/dev/null; exit 143' TERM INT
    while :; do
        # 재시작 전(가장 먼저): 원본 바이트 보존(파싱 전) + 평가기가 건너뛸 기반시설 실패 경로로 progress 되돌림(바꾼 내용은 로그)
        bk="$L.ckpt_before_${rts}_try$a.json"; pp="$L.ckpt_prepped_${rts}_try$a.json"
        ck_tool prep "$ck" "$bk" 2>&1 | tee -a "$L.resume.log"; prc=${PIPESTATUS[0]}
        [ "$prc" = 0 ] || { log "[s$k][중단] checkpoint prep 실패(rc $prc; 잘린/깨진 json?) — 평가기 안 띄움. 원본 바이트 $bk, 로그 $L.resume.log"; return 1; }
        st=$(ck_tool state "$ck" 2>>"$L.resume.log") || { log "[s$k][중단] checkpoint 읽기 실패($ck) — 평가기 안 띄움, $L.resume.log 확인"; return 1; }
        [ "$st" = done ] && { log "[s$k] 완료"; return 0; }
        [ $noprog -gt "$MAX_RESTARTS" ] && break
        rm -f "$pp"; [ -f "$ck" ] && cp "$ck" "$pp"
        ok0=$(ck_tool ok "$ck" 2>>"$L.resume.log")
        [[ "$ok0" =~ ^[0-9]+$ ]] || { log "[s$k][중단] 최종 기록 수(ok0)를 못 읽음('$ok0') — 평가기 안 띄움"; return 1; }
        log "[s$k] 시도 $a(진전 없는 재시작 $noprog/$MAX_RESTARTS): $st"
        stop_server "" $P $TMP || return 1   # 기동 전: 포트가 비었는지 항상 확인(다른 프로세스가 잡았으면 대기, 안 풀리면 중단)
        pg=$(bash "$START_CARLA" "$GPU" "$P" "$L.carla_try$a.log" 8>&- 9>&-) || { log "[s$k] CARLA 기동 실패"; pg=; noprog=$((noprog + 1)); a=$((a + 1)); sleep 30; continue; }
        local w=0; until (exec 3<>/dev/tcp/127.0.0.1/$P) 2>/dev/null; do
            sleep 5; w=$((w + 5)); [ $w -ge "$SERVER_WAIT_S" ] && break
        done
        if [ $w -ge "$SERVER_WAIT_S" ]; then
            log "[s$k] CARLA 포트 $P 가 ${SERVER_WAIT_S}s 안에 안 열림 — 서버 종료 후 재시도"
            stop_server "$pg" $P $TMP || return 1; pg=; noprog=$((noprog + 1)); a=$((a + 1)); continue
        fi
        sleep "$MAP_LOAD_WAIT_S"   # 맵 적재 여유(업스트림 evaluator 의 기동 후 대기와 같은 용도)
        ROUTES="$xml" SAVE_PATH="$OUT/save" CHECKPOINT_ENDPOINT="$ck" ROUTES_SUBSET="${SET_TAG}_s${k}" \
        LAQ_TEACHER_LOG="$L.teacher.jsonl" CUDA_VISIBLE_DEVICES=$GPU \
            "$PY" "$EVALUATOR" --routes="$xml" --repetitions=1 --track=SENSORS --checkpoint="$ck" \
            --debug-checkpoint="$L.live.txt" --agent="$TEAM_AGENT" --agent-config="$TEAM_CONFIG" --debug=0 --record= \
            --resume=True --port=$P --traffic-manager-port=$TMP --gpu-rank=$GPU >> "$L.eval.log" 2>&1 8>&- 9>&- &
        epid=$!; wait "$epid"; rc=$?; epid=
        log "[s$k] 평가기 종료 rc=$rc — 서버 그룹 $pg 종료"
        stop_server "$pg" $P $TMP || return 1; pg=
        ck_tool diff "$bk" "$ck" 2>&1 | tee -a "$L.overwritten.log"
        # 같은 경로가 연속 시도에서 기반시설 실패면 재시작으로 안 풀리는 원인 — 한도: Simulation crashed 3 회, Agent crashed·Agent couldn't be set up(·센서 무효) 2 회
        cur_fail=$(ck_tool fails "$pp" "$ck" 2>>"$L.resume.log") || { log "[s$k][중단] checkpoint 읽기 실패(시도 뒤) — $L.resume.log"; return 1; }
        nstreak=(); rep=
        while read -r rid kind; do
            [ -n "$rid" ] || continue
            key="$rid $kind"; n=$(( ${streak[$key]:-0} + 1 )); nstreak[$key]=$n
            if [ "$kind" = sim ]; then lim=$SIM_FAIL_LIMIT; else lim=$AGENT_FAIL_LIMIT; fi
            [ "$n" -ge "$lim" ] && rep="$rep $rid($kind ${n}회)"
        done <<< "$cur_fail"
        streak=(); for key in "${!nstreak[@]}"; do streak[$key]=${nstreak[$key]}; done
        if [ -n "$rep" ]; then
            log "[s$k][중단] 같은 경로가 연속 기반시설 실패 한도 도달:$rep (한도 sim $SIM_FAIL_LIMIT·agent $AGENT_FAIL_LIMIT; 상태는 $ck, 원인은 $L.eval.log 의 해당 경로 traceback) — 원인 수정 전 재시작 안 함(건너뛰지 않음)"
            echo "[$(date '+%F %T')] 연속 기반시설 실패로 중단:$rep (시도 $a)" >> "$L.resume.log"
            return 1
        fi
        [ -n "$cur_fail" ] && log "[s$k] 이번 시도 기반시설 실패: $(echo $cur_fail)"
        ok1=$(ck_tool ok "$ck" 2>>"$L.resume.log")
        [[ "$ok1" =~ ^[0-9]+$ ]] || { log "[s$k][중단] 최종 기록 수(ok1)를 못 읽음('$ok1') — $L.resume.log"; return 1; }
        if [ "$ok1" -gt "$ok0" ]; then log "[s$k] 최종 기록 $ok0 → $ok1"; else noprog=$((noprog + 1)); log "[s$k] 진전 없음(최종 기록 $ok1) — 진전 없는 재시작 $noprog"; fi
        a=$((a + 1))
    done
    log "[s$k] 진전 없는 재시작 $MAX_RESTARTS 회 초과, 미완료: $st — 로그 $L.eval.log·$L.resume.log 확인(건너뛰지 않음, 집계 안 함)"; return 1
}
# <<< funcs

rc_all=0
if [ "$AGG_ONLY" != 1 ]; then
    # ------------------------------------------------------------ GPU 메모리
    if [ "$DRYRUN" != 1 ]; then
        read -r tot used < <(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ',')
        free_gb=$(( (tot - used) / 1024 ))
        np=$(( ${#SPL[@]} < PARALLEL ? ${#SPL[@]} : PARALLEL ))
        [ "$free_gb" -ge $((NEED_GB_PER_TASK * np)) ] || die "GPU$GPU 여유 ${free_gb} GiB < 필요 $((NEED_GB_PER_TASK * np)) GiB (동시 $np × $NEED_GB_PER_TASK) — 다른 작업 확인(nvidia-smi)"
        log "GPU$GPU 여유 ${free_gb} GiB, 동시 $np"
    fi
    log "시작: $SET_TAG ($XML) 조각 ${SPL[*]} / $NUM_SPLITS, 동시 $PARALLEL, GPU$GPU, 학생 $CKPT, 교사 $LAQ_TEACHER_MODEL(fold $LAQ_TEACHER_FOLD, layer $LAQ_TEACHER_LAYER), 결과 $OUT"
    for k in "${SPL[@]}"; do
        while :; do   # 빈 슬롯 찾기
            for s in $(seq 0 $((PARALLEL - 1))); do
                p=${SLOT_PID[$s]:-}
                if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then
                    [ -n "$p" ] && { wait "$p" || rc_all=1; }
                    run_split "$k" "$s" & SLOT_PID[$s]=$!; break 2
                fi
            done
            sleep 10
        done
        [ "$DRYRUN" = 1 ] || sleep 30   # 서버 동시 기동을 피함
    done
    for s in "${!SLOT_PID[@]}"; do wait "${SLOT_PID[$s]}" || rc_all=1; done
    [ "$DRYRUN" = 1 ] && { log "DRYRUN 끝(rc $rc_all) — 결과 폴더 $OUT"; exit $rc_all; }
fi

# ---------------------------------------------------------------- 집계(전 조각 완료일 때만, 이번 NUM_SPLITS 의 조각 0..N-1 파일만)
all_done=1
for k in "${ALL_K[@]}"; do
    st=$(ck_tool state "$(split_ck "$k")"); [ "$st" = done ] || { all_done=0; log "[집계] 조각 s$k 미완료: $st"; }
done
if [ $all_done = 1 ] && [ -n "$EXPECT" ]; then
    AGG=$OUT/score_$NT
    rm -rf "$AGG.tmp"; mkdir -p "$AGG.tmp" || die "mkdir $AGG.tmp"
    for k in "${ALL_K[@]}"; do cp "$(split_ck "$k")" "$AGG.tmp/" || die "복사 실패 s$k"; done
    sub=; [ "$EXPECT" != 220 ] && sub=--subset
    set -o pipefail
    "$PY" tools/eval_b2d_laq_closedloop.py "$AGG.tmp" --expect "$EXPECT" $sub --write-merged 2>&1 | tee "$AGG.tmp/score.txt"
    agg_rc=$?
    set +o pipefail
    if [ $agg_rc = 0 ]; then
        rm -rf "$AGG"; mv "$AGG.tmp" "$AGG"; cp "$AGG/score.txt" "$OUT/score.txt"
        log "집계 OK — $AGG (조각 ${#ALL_K[@]} 개 파일만)"
        [ "$SET_TAG" = dev10 ] && { date '+%F %T' > "$OUT/DONE"; log "Dev10 완료 표식 $OUT/DONE"; }
    else
        log "[집계 실패] rc $agg_rc — 점수·DONE 을 쓰지 않음(내용 $AGG.tmp/score.txt)"
        rc_all=1
    fi
else
    log "미완료 조각 있음 또는 사용자 xml — 집계 생략. 남은 조각을 SPLITS 로 다시 돌릴 것"
    [ $all_done = 1 ] || rc_all=1
fi
exit $rc_all
