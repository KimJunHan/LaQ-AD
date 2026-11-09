# 신규(2026-11-05): 사용자 지시 — 기여 3(실패 국소화 + 붕괴 인지 GRPO) 용 NAVSIM CPU 채점 래퍼. 기존 파일 무수정.
"""NAVSIM GRPO 채점기 — 장면(token)마다 후보 궤적 K 개를 공식 PDM 규칙으로 채점하고, GRPO advantage 설계(docs/c3_grpo_design.md)에
필요한 값(하위 점수, EPDMS(EC 제외)/PDMS 유사값, 실패 유형별 첫 실패 시점, 연속 보상 성분)을 돌려준다. CPU 전용, 워커 ≤ 16.
# 수정(2026-11-05): 사용자 결정 "채점 워커 16" — 상한 8 → 16(실측 K=48 여유 켬 7.84 장면/s, 끔 11.6; 8 워커 대비 2배 아님).

채점 규칙은 tools/navsim/navsim_score_vocab_navtrain.py 와 같다(공식 pdm_score 와 --verify 로 대조된 방식):
  [PDM 참조, 후보 1..K] 한 묶음 시뮬레이션·채점 → 진행도만 후보마다 "참조와 둘" 규칙으로 재정규화, 사람 궤적 벌점 필터 적용,
  EC(두 프레임 확장 편안함)는 정의되지 않으므로 가중치 0(공식 compute_final_scores 의 NaN 처리와 같음).
  배경 차량 non-reactive(log replay) → 자차 후보와 무관하므로 한 묶음 채점이 후보별 단독 채점과 같다.

입력: {token: (K, 8, 3) 또는 (K, 8, 2)} — navsim ego 프레임(+x 전진, +y 왼쪽, 후륜축 원점), 0.5 s 간격 8 점(4 s) 누적 위치(+heading).
  (K,8,2) 이면 heading 은 navsim_score_vocab_navtrain.heading_from_xy 규칙(이동 < 0.05 m 이면 직전 heading)으로 만든다.
출력: {token: dict} — 배열 첫 축 = 후보 K. (시뮬레이션 시점 = 0.1 s 간격 0..40, 0 = 현재)
  subscores (K,8) float64       : SUBSCORES 순서 [nc, dac, ddc, tlc, ep, ttc, lk, hc] — 공식과 비트 동일(검증 대상)
  multiplicative (K,)           : nc·dac·ddc·tlc
  epdms (K,)                    : EPDMS(EC 가중치 0) = multiplicative · (5ep+5ttc+2lk+2hc)/14 — 공식 값과 비트 동일(검증 대상)
  pdms_v1like (K,)              : nc·dac·(5ep+5ttc+2hc)/12 — v1 PDMS 근사(v1 comfort 는 이력 미포함이라 공식 v1 과 다를 수 있음. 참고용)
  fail_idx (K,7) int            : FAIL_TYPES 순서 [collision, ttc, offroad, direction, red_light, lane_keeping, comfort] 의 첫 실패
                                  시뮬레이션 시점(0..40), 없으면 -1. 사람 궤적 필터로 지워진 지표는 -1(filtered 에 표시).
                                  collision/ttc = 공식 _collision_time_idcs/_ttc_time_idcs, offroad = 첫 NON_DRIVABLE 시점,
                                  direction = 1 s 창 역주행 진행 누적이 2 m 이상이 되는 첫 시점, red_light = 첫 적신호 교차,
                                  lane_keeping = 2 s 연속 이탈 구간의 시작 시점, comfort = 6개 편안함 신호 중 첫 한계 초과 시점
                                  (이력 패딩 구간에서 시작하면 0; 수정(2026-11-06) 아래 comfort fail_idx 규칙).
  fail_wp (K,7) int             : fail_idx 를 계획 경유점 번호(0..7, 경유점 j = 시점 5(j+1))로 — ceil(idx/5)-1, idx 0 → 0, 없으면 -1
  filtered (7,) bool            : 사람 궤적 필터로 해당 지표가 1 로 바뀌었는지(장면 공통)
  cont (K, C) float64           : CONT 순서의 연속 성분 —
      progress_m           중심선 진행 거리(m, 공식 _progress_raw)
      ref_progress_m       PDM 참조 진행 거리(m, 후보 공통)
      min_obj_dist_m       0..4 s 자차 다각형과 모든 객체(적신호 제외) 최소 거리(m, 0 = 접촉) — 휴리스틱 여유
      drivable_margin_m    0..4 s 자차 네 모서리의 주행 가능 영역 경계까지 부호 거리 최솟값(+ 안쪽, − 바깥)
      offroad_steps        NON_DRIVABLE 시점 수(0..41)
      oncoming_max_m       1 s 창 역주행 진행 누적 최댓값(m; ddc 1/0.5/0 경계 2·6 m)
      lat_dev_mean_m       교차로 밖 시점의 중심선 횡 이탈 평균(m; 교차로만이면 0)
      lat_dev_max_m        같은 최댓값
      comfort_ratio_max    6개 편안함 신호의 |값|/한계 최댓값(≥ 1 이면 hc 실패와 같다; 이력 없으면 NaN)
      ttc_time_s           첫 TTC 실패 시각(s), 없으면 4.0
      collision_time_s     첫 at-fault 충돌 시각(s), 없으면 4.0
  frenet (K,8,3) float64        : 입력 경유점의 PDM 중심선 기준 (s 진행 m, d 부호 횡 오프셋 m(+왼쪽), 접선 각 − 자차 heading rad)
                                  → 설계 (b) 차선 정렬 축 분해의 "정답 지도" 변형용(주 설계는 모델 예측 지도 축).
  ref_subscores (8,)            : PDM 참조 궤적의 하위 점수(묶음 채점 값, 진행도 = 묶음 정규화 — 참고용)
  # 수정(2026-11-06): C3 설계 v4 §5·§6·§7·§12-① 확장 키(기존 키 값은 comfort −2 를 빼고 그대로):
  coll_t (K,) int               : 첫 at-fault 충돌 시뮬레이션 시점(pdm_scorer.py:363–418 루프 재현, proposal_collided_track_ids 누적
                                  포함), 없으면 -1. 매 호출에서 공식 _collision_time_idcs·nc 값과 같은지 단언한다. 사람 필터 전 원시 값
                                  (filtered[0] 이면 학습 쪽은 nc 실패로 보지 않는다).
  coll_token (K,) str / coll_obj_type (K,) str / coll_ctype (K,) str / coll_agent (K,) bool / coll_nc (K,) float
                                : 그 첫 충돌의 객체 토큰, TrackedObjectType 이름, 공식 CollisionType 이름, 에이전트 여부(nc 0)
                                  / 비에이전트(nc 0.5), 재현 nc 값(전체 시점 최소, 공식과 비트 동일 단언). 없으면 "", "", "", False, 1.0
                                  수정(2026-11-06, 검토 2) 주의: coll_token·coll_obj_type·coll_ctype·coll_agent·coll_front 는 **첫 at-fault
                                  사건**(coll_t 시점, 같은 시점이면 질의 순서상 첫 객체) 기준이고 coll_nc 는 공식처럼 **모든 사건의 최소**다.
                                  그래서 첫 사건이 비에이전트(coll_agent False)라도 뒤 사건이 에이전트면 coll_nc = 0 일 수 있다.
  coll_front (K,) bool          : 기하 정면 판정 — 그 시점 자차 앞 범퍼 선분 FL–FR 이 객체 다각형과 교차하면 True(정면), 아니면 측면.
                                  비에이전트(nc 0.5)도 같은 규칙. 충돌 없으면 False.
  min_obj_dist_fs_m (K,)        : min_obj_dist_m 과 같되, 각 시점에서 다각형 전체가 자차 뒤 범퍼선(RL–RR) 뒤에 있는 객체를 뺀 최소 거리
                                  (§6 c 성분; margins=False 면 NaN, 대상 객체가 없으면 NaN)
  lat_dev_wp_m (K,)             : 입력 경유점 8 개의 PDM 중심선 |횡편차| 평균(m, frenet[..., 1]) — §6 c 성분
  comfort fail_idx              : 이력 패딩 구간에서 시작한 첫 편안함 실패는 -2(귀속 불가; 종전 0). fail_wp 도 -2.
                                  수정(2026-11-06, 검토 4): 반사실 기준 — 이력 구간 첫 실패 시점에 후보가 한계를 넘은 신호를 PDM 참조
                                  (proposal 0)도 같은 시점에 모두 넘으면 -2(미래와 무관한 이력 기원), 아니면 0(경유점 0 귀속). 위치 조건 없음.
                                  한계: 참조 자신의 미래가 만든 실패(드묾)도 -2. 기록·진단용(advantage 주 실패에 쓰이지 않음).
                                  #이전(검토 3) 실패 신호 비율이 참조와 비트 동일 #검토 2 이력 끝 11 표본 창 + 참조 동시 실패
  guide=True 일 때(§7 단계 ③ 안내 궤적):
  ref_traj (8,3)                : PDM-Closed 참조 궤적을 0.5 s × 8 로 표본한 ego 프레임 (x, y, heading)
  ref_epdms / ref_sub (8,)      : ref_traj 를 후보로 공식 규칙(참조와 2-묶음 진행도, 사람 필터, EC 0)으로 채점한 r* 와 하위 점수
  human_traj (8,3) / human_epdms / human_sub (8,)
                                : 사람 미래 궤적(MetricCache.human_trajectory, ego 프레임)과 같은 규칙의 점수(합성 장면이면 NaN)

사용(라이브러리):
  from tools.navsim.grpo_scorer import GRPOScorer
  with GRPOScorer(metric_cache_dir, workers=8) as sc:
      out = sc.score({token: cands})        # cands (K,8,3) np.ndarray, ego 프레임

검증·속도 측정(CLI):
  source scripts/navsim/navsim_env.sh
  $NAVSIM_PY tools/navsim/grpo_scorer.py --metric-cache $NAVSIM_EXP_ROOT/metric_cache_navtrain \
      --tokens 200 --K 16 --workers 8 --verify 4
  --verify N: 처음 N token 의 후보 전부를 공식 pdm_score 로 하나씩 채점해 subscores·multiplicative·epdms 가 비트 동일한지,
  그리고 후보마다 [참조, 후보] 2-묶음 score_proposals 로 다시 구한 실패 시점(collision/ttc/offroad 등)이 같은지 확인, 다르면 즉시 실패.
환경: OPENBLAS_CORETYPE=SkylakeX 필수(scripts/navsim/navsim_env.sh 주석 — 이 CPU 의 기본 OpenBLAS 커널이 틀린 값을 낸다).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # 공유 서버: 워커당 BLAS 스레드 1
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import signal  # 수정(2026-11-06): 검토 2 — 워커 SIGTERM 기본 처리
import copy  # 수정(2026-11-06): at-fault 충돌 루프 재현(proposal_collided_track_ids 복사)
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed  # 수정(2026-11-06): 검토 2 — 워커 죽음이 BrokenProcessPool 로 드러나게
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from navsim_score_vocab_navtrain import (AGENT, HUMAN_SKIP, PROPOSAL, cache_paths,  # noqa: E402
                                        heading_from_xy, load_cache)

import shapely  # noqa: E402
from nuplan.common.actor_state.state_representation import TimePoint  # noqa: E402
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES  # noqa: E402  # 수정(2026-11-06): at-fault 루프 재현
from nuplan.planning.metrics.utils.collision_utils import CollisionType  # noqa: E402  # 수정(2026-11-06): 같은 이유
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters  # noqa: E402
from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402

from navsim.common.dataclasses import Trajectory  # noqa: E402
from navsim.common.enums import SceneFrameType  # noqa: E402
from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score, transform_trajectory  # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.scoring import pdm_comfort_metrics as CM  # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig  # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type  # noqa: E402  # 수정(2026-11-06): at-fault 루프 재현
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator  # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (  # noqa: E402
    ego_states_to_state_array)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (  # noqa: E402
    BBCoordsIndex, EgoAreaIndex, MultiMetricIndex, StateIndex, WeightedMetricIndex)
from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents  # noqa: E402

SUBSCORES = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
             "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
             "history_comfort"]
FAIL_TYPES = ["collision", "ttc", "offroad", "direction", "red_light", "lane_keeping", "comfort"]
FAIL_METRIC = ["no_at_fault_collisions", "time_to_collision_within_bound", "drivable_area_compliance",
               "driving_direction_compliance", "traffic_light_compliance", "lane_keeping", "history_comfort"]
CONT = ["progress_m", "ref_progress_m", "min_obj_dist_m", "drivable_margin_m", "offroad_steps", "oncoming_max_m",
        "lat_dev_mean_m", "lat_dev_max_m", "comfort_ratio_max", "ttc_time_s", "collision_time_s"]
MAX_WORKERS = 16  # 수정(2026-11-05): 사용자 결정 "채점 워커 16" #이전 8
N_SIM = PROPOSAL["num_poses"] + 1          # 41
# 수정(2026-11-06): 검토 3 — 검토 2 의 COMFORT_REACH(시점 창 11) 규칙 폐기(이력 앞 가장자리 savgol interp 의존을 놓침). 값 비교 규칙은
#   _derived 편안함 주석. #이전 COMFORT_REACH = 11
DRIVABLE_LAYERS = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.DRIVABLE_AREA,
                   SemanticMapLayer.CARPARK_AREA]
_VP = get_pacifica_parameters()


# ---------------------------------------------------------------- worker
_W = {}


def _set_pdeathsig(ppid):
    """수정(2026-11-06): 검토 3 — 부모(채점기·서버 프로세스)가 SIGKILL 등으로 죽으면 워커도 SIGKILL(리눅스 prctl PR_SET_PDEATHSIG).
    ProcessPoolExecutor 워커는 부모가 죽어도 남는다(종전 Pool 대비 회귀). prctl 전에 부모가 이미 죽었으면 바로 끝낸다.
    주의: PDEATHSIG 는 fork 한 '스레드' 기준 — GRPOScorer 는 생성 스레드에서 워커를 모두 띄우므로 그 스레드가 프로세스 수명 동안 살아
    있어야 한다(서버·CLI 는 주 스레드)."""
    import ctypes
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:                   # 1 = PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) 실패")
    if os.getppid() != ppid:
        os._exit(1)


def _init_worker(margins, ppid=None):  # 수정(2026-11-06): 검토 3 — ppid 주면 부모 죽음 시 워커 종료
    if ppid is not None:
        _set_pdeathsig(ppid)
    # 수정(2026-11-06): 검토 2 — fork 워커는 부모(score_server)의 SIGTERM 처리기(sys.exit)를 물려받는다. 그러면 executor 가 깨진 풀을
    #   정리하려고 보내는 SIGTERM 이 작업 안의 SystemExit 로 잡혀 워커가 살아남고, 관리 스레드의 join 이 영원히 멈춘다(실측).
    #   워커에서는 기본 처리(즉시 종료)로 되돌린다.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    sampling = TrajectorySampling(**PROPOSAL)
    _W.update(sampling=sampling, agent_sampling=TrajectorySampling(**AGENT), simulator=PDMSimulator(sampling),
              scorer=PDMScorer(sampling, PDMScorerConfig(human_penalty_filter=True)),
              traffic=LogReplayTrafficAgents(sampling), margins=margins)


def _comfort_edge_dependence(n_hist, trials=5, seed=0):
    """수정(2026-11-06): 검토 3 — 패딩 길이 n_hist + 41 에서 후보 미래(시점 > n_hist; n_hist = 공통 현재 상태)를 바꿀 때 신호별로
    값이 바뀌는 가장 이른 시점(실측). 저크는 savgol interp 앞 가장자리 때문에 0(이력 전 구간)이다 — 시험 1 이 단언."""
    rng = np.random.default_rng(seed)
    n = n_hist + N_SIM
    ts = np.arange(n) * 0.1
    sig = lambda p: dict(lon_acc=CM._extract_ego_acceleration(p, "x", _VP), lat_acc=CM._extract_ego_acceleration(p, "y", _VP),
                         jerk=CM._extract_ego_jerk(p, "magnitude", ts, _VP), lon_jerk=CM._extract_ego_jerk(p, "x", ts, _VP),
                         yaw_acc=CM._extract_ego_yaw_rate(p, ts, deriv_order=2, poly_order=3),
                         yaw_rate=CM._extract_ego_yaw_rate(p, ts))
    out = {}
    for _ in range(trials):
        base = rng.normal(size=(1, n, StateIndex.size()))
        q = base.copy()
        q[:, n_hist + 1:] += rng.normal(size=q[:, n_hist + 1:].shape)
        s0, s1 = sig(base), sig(q)
        for k in s0:
            ch = np.nonzero(s0[k][0] != s1[k][0])[0]
            out[k] = min(out.get(k, n), int(ch.min()))
    return out


def _first_true(mask):
    """mask (K, T) bool → (K,) 첫 True 위치, 없으면 -1."""
    has = mask.any(-1)
    return np.where(has, mask.argmax(-1), -1)


def _comfort_signals(states, scorer, per_signal=False):  # 수정(2026-11-06): 검토 3 — per_signal 이면 (K,6,T) 신호값도
    """공식 _calculate_history_comfort 와 같은 이력 패딩 후 6개 신호의 |값|/한계 비율 (K, T_pad), 이력 길이. 이력 없으면 None."""
    hp = scorer._human_past_trajectory
    if hp is None:
        return (None, 0, None) if per_signal else (None, 0)   # 수정(2026-11-06): 검토 3
    s_us, e_us = hp.start_time.time_us, hp.end_time.time_us
    h_us = np.clip(np.arange(s_us, e_us, int(0.1 * 1e6), dtype=np.int64), s_us, e_us)
    hist = ego_states_to_state_array(hp.get_state_at_times([TimePoint(t) for t in h_us[:-1]]))
    n_pad = len(hist) + states.shape[1]
    pad = np.zeros((states.shape[0], n_pad, StateIndex.size()), np.float64)
    pad[:, :len(hist)] = hist
    pad[:, len(hist):] = states
    ts = np.arange(0, n_pad).astype(np.float64) * scorer.proposal_sampling.interval_length
    lon_a = CM._extract_ego_acceleration(pad, "x", _VP)
    lat_a = CM._extract_ego_acceleration(pad, "y", _VP)
    jerk = CM._extract_ego_jerk(pad, "magnitude", ts, _VP)
    lon_j = CM._extract_ego_jerk(pad, "x", ts, _VP)
    yaw_acc = CM._extract_ego_yaw_rate(pad, ts, deriv_order=2, poly_order=3)
    yaw_rate = CM._extract_ego_yaw_rate(pad, ts)
    # _within_bound 는 (min < v < max) 엄격 부등호 → 비율 ≥ 1 이 실패와 같다
    r6 = np.stack([np.maximum(lon_a / CM.MAX_LON_ACCEL, lon_a / CM.MIN_LON_ACCEL),   # 수정(2026-11-06): 검토 3 — 신호별 보존
                   np.abs(lat_a) / CM.MAX_ABS_LAT_ACCEL, np.abs(jerk) / CM.MAX_ABS_MAG_JERK,
                   np.abs(lon_j) / CM.MAX_ABS_LON_JERK, np.abs(yaw_acc) / CM.MAX_ABS_YAW_ACCEL,
                   np.abs(yaw_rate) / CM.MAX_ABS_YAW_RATE], 0)
    r = r6.max(0)
    if per_signal:                                                    # 수정(2026-11-06): 검토 3
        return r, len(hist), np.moveaxis(r6, 0, 1)                     # (K, 6, T) 신호별 |값|/한계
    return r, len(hist)


def _at_fault_events(scorer):
    """수정(2026-11-06): C3 §5·§12-① — pdm_scorer.py:363–418 _calculate_no_at_fault_collision 루프를 그대로 재현해 proposal 마다
    첫 at-fault 충돌(시점, 토큰, 유형)과 기하 정면 여부를 돌려준다. 재현 nc·시점이 공식 내부 상태와 다르면 즉시 오류."""
    obs = scorer._observation
    P = scorer._num_proposals
    areas = scorer._ego_areas
    coords = scorer._ego_coords
    collided = {p: copy.deepcopy(obs.collided_track_ids) for p in range(P)}
    nc = np.ones(P, dtype=np.float64)
    t_first = np.full(P, np.inf)
    ev = [("", "", "", False, False)] * P                     # token, obj_type, ctype, agent, front
    for t in range(scorer.proposal_sampling.num_poses + 1):
        polys = scorer._ego_polygons[:, t]
        inter = obs[t].query(polys, predicate="intersects")
        if len(inter) == 0:
            continue
        for p, g in zip(inter[0], inter[1]):
            tok = obs[t].tokens[g]
            if (obs.red_light_token in tok) or (tok in collided[p]):
                continue
            multi_or_nd = areas[p, t, EgoAreaIndex.MULTIPLE_LANES] or areas[p, t, EgoAreaIndex.NON_DRIVABLE_AREA]
            obj = obs.unique_objects[tok]
            ctype = get_collision_type(scorer._states[p, t], scorer._ego_polygons[p, t], obj, obs[t][tok])
            front_or_stopped = ctype in [CollisionType.ACTIVE_FRONT_COLLISION, CollisionType.STOPPED_TRACK_COLLISION]
            lateral = ctype == CollisionType.ACTIVE_LATERAL_COLLISION
            if front_or_stopped or (multi_or_nd and lateral):
                agent = obj.tracked_object_type in AGENT_TYPES
                nc[p] = np.minimum(nc[p], 0.0 if agent else 0.5)
                if t < t_first[p]:
                    bumper = shapely.LineString([coords[p, t, BBCoordsIndex.FRONT_LEFT], coords[p, t, BBCoordsIndex.FRONT_RIGHT]])
                    ev[p] = (tok, obj.tracked_object_type.name, ctype.name, bool(agent), bool(bumper.intersects(obs[t][tok])))
                    t_first[p] = t
            else:
                collided[p].append(tok)
    ref_t = scorer._collision_time_idcs
    same_t = np.all((np.isinf(ref_t) & np.isinf(t_first)) | (ref_t == t_first))
    if not same_t or not np.array_equal(nc, scorer._multi_metrics[MultiMetricIndex.NO_COLLISION]):
        raise AssertionError(f"at-fault 루프 재현 불일치: 시점 {t_first} vs 공식 {ref_t}, nc {nc} vs "
                             f"{scorer._multi_metrics[MultiMetricIndex.NO_COLLISION]}")
    return dict(coll_t=np.where(np.isfinite(t_first), t_first, -1).astype(np.int64),
                coll_token=np.array([e[0] for e in ev], dtype=object), coll_obj_type=np.array([e[1] for e in ev], dtype=object),
                coll_ctype=np.array([e[2] for e in ev], dtype=object), coll_agent=np.array([e[3] for e in ev], bool),
                coll_front=np.array([e[4] for e in ev], bool), coll_nc=nc)


def _derived(scorer, mc, margins):
    """묶음 score_proposals 직후 scorer 내부 상태에서 실패 시점·연속 성분을 만든다(모든 proposal, 참조 포함)."""
    cfg = scorer._config
    P = scorer._num_proposals
    areas = scorer._ego_areas                                    # (P, 41, 3)
    coords = scorer._ego_coords                                  # (P, 41, 5, 2)
    center = coords[:, :, BBCoordsIndex.CENTER]                  # (P, 41, 2)
    dmap = scorer._drivable_area_map
    obs = scorer._observation
    dt = scorer.proposal_sampling.interval_length

    fail = np.full((P, len(FAIL_TYPES)), -1, np.int64)
    ci, ti = scorer._collision_time_idcs, scorer._ttc_time_idcs
    fail[:, 0] = np.where(np.isfinite(ci), ci, -1)
    fail[:, 1] = np.where(np.isfinite(ti), ti, -1)
    fail[:, 2] = _first_true(areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA])

    # 교차로 안 여부(ddc·lk 공통, 공식과 같은 is_in_layer 호출)
    in_inter = np.array([[dmap.is_in_layer(shapely.Point(*center[p, t]), SemanticMapLayer.INTERSECTION)
                          for t in range(N_SIM)] for p in range(P)])

    # 주행 방향(공식 _calculate_driving_direction_compliance 와 같은 누적)
    onc = np.zeros((P, N_SIM))
    onc[:, 1:] = np.linalg.norm(center[:, 1:] - center[:, :-1], axis=-1)
    onc[~areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC] | in_inter] = 0.0
    hz = int(cfg.driving_direction_horizon / dt)
    win = np.stack([onc[:, max(0, t - hz):t + 1].sum(-1) for t in range(N_SIM)], -1)
    fail[:, 3] = _first_true(win >= cfg.driving_direction_compliance_threshold)

    # 적신호(공식 _calculate_traffic_light_compliance 와 같은 질의)
    for t in range(N_SIM):
        inter = obs[t].query(scorer._ego_polygons[:, t], predicate="intersects")
        for p, g in zip(inter[0], inter[1]):
            if fail[p, 4] < 0 and obs[t].tokens[g].startswith(obs.red_light_token):
                fail[p, 4] = t

    # 차선 유지(공식: 교차로 시점은 건너뛰고 연속 카운트 유지, 2 s 연속 이탈 → 실패). 시작 시점 = 연속 구간 첫 시점
    lat = shapely.distance(shapely.points(center.reshape(-1, 2)), scorer._centerline.linestring).reshape(P, N_SIM)
    need = int(np.ceil(cfg.lane_keeping_horizon_window / dt))
    for p in range(P):
        run, start = 0, -1
        for t in range(N_SIM):
            if in_inter[p, t]:
                continue
            if lat[p, t] > cfg.lane_keeping_deviation_limit:
                if run == 0:
                    start = t
                run += 1
            else:
                run = 0
            if run >= need:
                fail[p, 5] = start
                break
    lat_ok = np.where(in_inter, np.nan, lat)

    # 편안함
    ratio, n_hist, sig6 = _comfort_signals(scorer._states, scorer, per_signal=True)   # 수정(2026-11-06): 검토 3 — 신호값 비교용
    if ratio is not None:
        bad = ratio >= 1.0
        first = _first_true(bad)
        # 수정(2026-11-06): 검토 3 — 시점 창 판정 폐기. savgol 기본 mode='interp' 는 배열 앞 가장자리 창 표본을 한 다항식으로 맞추므로
        #   실제 n_hist(13–16)에서 저크·종방향 저크는 이력 전 구간이 후보 미래에 의존한다(실측, _comfort_edge_dependence).
        # 수정(2026-11-06): 검토 4 — 판정 기준을 '값의 비트 동일'(의존 여부)에서 반사실(실패 원인)로 바꿈: 이력 구간 첫 실패 시점에서
        #   후보가 한계를 넘은 신호(비율 ≥ 1)를 PDM 참조(proposal 0, 같은 묶음에서 채점된 PDM-Closed 행)도 같은 시점에 모두 넘으면
        #   "미래를 바꿔도 남는 실패" → 이력 기원(-2), 하나라도 참조는 넘지 않으면 후보 미래가 만든 실패 → 경유점 0 귀속(0). 위치 조건 없음.
        #   (검토 B: 비트 비교는 정지 궤적 포함 모든 후보가 같은 시점·신호로 실패하는 이력 기원 실패를 savgol 가장자리 미세 차이로 0 판정.)
        #   한계: 참조 자신의 미래가 그 실패를 만든 경우(드묾)도 -2 가 된다(참조는 반사실 대용일 뿐).
        #   용도: 기록·진단용. advantage.primary_failure 는 comfort 열을 읽지 않는다(test_scorer_robust 7 이 단언).
        #   #이전(검토 3) same_ref = np.all((rp < 1.0) | (rp == r0), axis=-1)  #검토 2: 창 11 + ref_bad  #그 전: 이력 안이면 전부 -2
        in_hist = (first >= 0) & (first < n_hist)
        fi = np.clip(first, 0, ratio.shape[1] - 1)
        rp = sig6[np.arange(P), :, fi]                                       # (P, 6) 후보 p 의 그 시점 신호별 비율
        r0 = sig6[0][:, fi].T                                               # (P, 6) 참조의 같은 시점(후보 p 의 fi) 신호별 비율  # 수정(2026-11-06): 검토 4 — 모양 명시
        ref_also = np.all((rp < 1.0) | (r0 >= 1.0), axis=-1)                # (P,) 후보 실패 신호를 참조도 모두 넘음
        hist_only = in_hist & ref_also
        fail[:, 6] = np.where(first < 0, -1, np.where(hist_only, -2, np.maximum(first - n_hist, 0)))
        comfort_max = ratio.max(-1)
    else:
        comfort_max = np.full(P, np.nan)

    # 연속 여유: 객체 거리, 주행 가능 영역 부호 거리
    min_obj = np.full(P, np.inf)
    min_obj_fs = np.full(P, np.inf)   # 수정(2026-11-06): C3 §6 — 전방·측방 객체만(뒤 범퍼선 뒤 객체 제외)
    marg = np.full(P, np.nan)
    if margins:
        for t in range(N_SIM):
            om = obs[t]
            keep = [i for i, tok in enumerate(om.tokens) if obs.red_light_token not in tok]
            if not keep:
                continue
            geoms = np.asarray(om._geometries, dtype=object)[keep]
            dmat = shapely.distance(scorer._ego_polygons[:, t][:, None], geoms[None, :])   # 수정(2026-11-06): 행렬 보존(아래 전방·측방 거리용)
            d = dmat.min(-1)
            min_obj = np.minimum(min_obj, d)
            # 수정(2026-11-06): 객체 꼭짓점의 (v − RL)·전진 단위벡터 최댓값 < 0 이면 다각형 전체가 뒤 범퍼선 뒤
            vxy, vidx = shapely.get_coordinates(geoms, return_index=True)
            rl = coords[:, t, BBCoordsIndex.REAR_LEFT]
            fwd = coords[:, t, BBCoordsIndex.FRONT_LEFT] - rl
            fwd = fwd / np.linalg.norm(fwd, axis=-1, keepdims=True)
            dots = ((vxy[None] - rl[:, None]) * fwd[:, None]).sum(-1)              # (P, nV)
            cnt = np.bincount(vidx, minlength=len(geoms))
            if (cnt == 0).any() or (np.diff(vidx) < 0).any():
                raise ValueError("객체 다각형 꼭짓점 없음/순서 이상 — 뒤 범퍼선 판정 불가")
            mx = np.maximum.reduceat(dots, np.concatenate([[0], np.cumsum(cnt)[:-1]]), axis=1)   # (P, G)
            min_obj_fs = np.minimum(min_obj_fs, np.where(mx < 0.0, np.inf, dmat).min(-1))
        idcs = dmap.get_indices_of_map_type(DRIVABLE_LAYERS)
        if idcs:
            area = shapely.union_all(np.asarray(dmap._geometries, dtype=object)[idcs])
            bnd = area.boundary
            corners = shapely.points(coords[:, :, :4].reshape(-1, 2))   # FL, RL, RR, FR (CENTER 제외)
            dist = shapely.distance(corners, bnd)
            inside = shapely.contains_xy(area, coords[:, :, :4, 0].reshape(-1), coords[:, :, :4, 1].reshape(-1))
            marg = np.where(inside, dist, -dist).reshape(P, -1).min(-1)
    cont = dict(min_obj_dist_m=np.where(np.isfinite(min_obj), min_obj, np.nan), drivable_margin_m=marg,
                offroad_steps=areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA].sum(-1).astype(np.float64),
                oncoming_max_m=win.max(-1),
                lat_dev_mean_m=np.nan_to_num(np.nanmean(np.where(np.isnan(lat_ok).all(-1, keepdims=True), 0, lat_ok), -1)),
                lat_dev_max_m=np.nan_to_num(np.nanmax(np.where(np.isnan(lat_ok).all(-1, keepdims=True), 0, lat_ok), -1)),
                comfort_ratio_max=comfort_max,
                ttc_time_s=np.where(np.isfinite(ti), ti * dt, 4.0), collision_time_s=np.where(np.isfinite(ci), ci * dt, 4.0))
    ext = _at_fault_events(scorer)   # 수정(2026-11-06): C3 §5 첫 at-fault 충돌·기하 정면/측면
    ext["min_obj_dist_fs_m"] = np.where(np.isfinite(min_obj_fs), min_obj_fs, np.nan)
    return fail, cont, ext


def _frenet(cands, ego0, centerline):
    """입력 경유점(ego 프레임) → (s, d, 접선−자차 heading). 중심선 = 공식 PDM centerline."""
    x0, y0, h0 = ego0.rear_axle.x, ego0.rear_axle.y, ego0.rear_axle.heading
    c, s = np.cos(h0), np.sin(h0)
    gx = x0 + c * cands[..., 0] - s * cands[..., 1]
    gy = y0 + s * cands[..., 0] + c * cands[..., 1]
    pts = shapely.points(np.stack([gx, gy], -1).reshape(-1, 2))
    sp = centerline.project(pts)
    st = centerline.interpolate(np.asarray(sp), as_array=True)     # (N, 3) x, y, θ
    tx, ty = np.cos(st[:, 2]), np.sin(st[:, 2])
    d = tx * (gy.reshape(-1) - st[:, 1]) - ty * (gx.reshape(-1) - st[:, 0])
    rel = (st[:, 2] - h0 + np.pi) % (2 * np.pi) - np.pi
    return np.stack([np.asarray(sp), d, rel], -1).reshape(cands.shape[0], cands.shape[1], 3)


def _as_poses(c):
    c = np.asarray(c, np.float64)
    if c.ndim != 3 or c.shape[1] != 8 or c.shape[2] not in (2, 3):
        raise ValueError(f"후보 형식 {c.shape} — (K, 8, 2|3) 이어야 한다(ego 프레임, 0.5 s × 8)")
    if c.shape[2] == 2:
        c = np.stack([heading_from_xy(t) for t in c])
    if not np.isfinite(c).all():
        raise ValueError("후보에 비유한값")
    return c


def _ego_frame_poses(states, ego0):
    """수정(2026-11-06): 전역 상태 배열 (41, StateIndex) → 0.5 s × 8 ego 프레임 (x, y, heading) (8, 3)."""
    ra = ego0.rear_axle
    c, s_ = np.cos(ra.heading), np.sin(ra.heading)
    st = states[5::5][:8]
    dx, dy = st[:, StateIndex.X] - ra.x, st[:, StateIndex.Y] - ra.y
    h = (st[:, StateIndex.HEADING] - ra.heading + np.pi) % (2 * np.pi) - np.pi
    return np.stack([c * dx + s_ * dy, -s_ * dx + c * dy, h], -1)


def _guide_scores(mc, human):
    """수정(2026-11-06): C3 §7 단계 ③ — PDM-Closed 참조(0.5 s × 8)와 사람 궤적을 후보로 [참조, ref8, human8] 묶음 채점.
    EPDMS 규칙은 score_token 후보와 같다(참조와 2-묶음 진행도 정규화, 사람 필터 human 행, EC 0). 주 묶음과 분리해 기존 출력에 영향 없음."""
    sampling, sim, scorer, traffic = _W["sampling"], _W["simulator"], _W["scorer"], _W["traffic"]
    ego0 = mc.ego_state
    t0 = ego0.time_point
    pdm_states = get_trajectory_as_array(mc.trajectory, sampling, t0)
    ref8 = _ego_frame_poses(pdm_states, ego0)
    trajs = [ref8]
    if mc.human_trajectory is not None:
        trajs.append(np.asarray(mc.human_trajectory.poses, np.float64))
    cs = [get_trajectory_as_array(transform_trajectory(Trajectory(v.astype(np.float32), _W["agent_sampling"]), ego0),
                                  sampling, t0) for v in trajs]
    simulated = sim.simulate_proposals(np.stack([pdm_states] + cs, 0), ego0)
    tracks = traffic.simulate_environment(simulated[1], mc)
    res = scorer.score_proposals(simulated, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
                                 mc.map_parameters, tracks, mc.past_human_trajectory)
    progress_raw = scorer._progress_raw.copy()
    multi = scorer._multi_metrics.prod(axis=0)
    cfg = scorer._config
    wa = cfg.weighted_metrics_array.copy()
    wa[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = 0.0
    out = []
    for k in range(len(trajs)):
        r = res[k + 1].iloc[0]
        row = {m: float(r[m]) for m in SUBSCORES}
        pair_raw = np.array([progress_raw[0], progress_raw[k + 1]])
        norm = (pair_raw * np.array([multi[0], multi[k + 1]])).max()
        row["ego_progress"] = float(np.clip(pair_raw[1] / norm, 0.0, 1.0)) if norm > cfg.progress_distance_threshold else 1.0
        if human is not None:
            for c in human.columns:
                if c not in HUMAN_SKIP and c in row and human[c].iloc[0] == 0:
                    row[c] = 1.0
        sub = np.array([row[m] for m in SUBSCORES])
        wm = np.zeros(len(wa))
        wm[WeightedMetricIndex.PROGRESS] = sub[4]
        wm[WeightedMetricIndex.TTC] = sub[5]
        wm[WeightedMetricIndex.LANE_KEEPING] = sub[6]
        wm[WeightedMetricIndex.HISTORY_COMFORT] = sub[7]
        out.append((sub, float(sub[0] * sub[1] * sub[2] * sub[3] * float((wm * wa).sum() / wa.sum()))))
    g = dict(ref_traj=ref8, ref_sub=out[0][0], ref_epdms=out[0][1])
    if len(trajs) > 1:
        g.update(human_traj=trajs[1], human_sub=out[1][0], human_epdms=out[1][1])
    else:
        g.update(human_traj=np.full((8, 3), np.nan), human_sub=np.full(8, np.nan), human_epdms=np.nan)
    return g


def score_token(mc, cands, margins=True, guide=False):
    """한 장면의 후보 (K,8,3) 채점 → dict(모듈 docstring 의 출력).
    수정(2026-11-06): guide=True 면 §7 단계 ③ 안내 궤적(PDM-Closed 0.5 s × 8, 사람)과 그 점수를 더한다(별도 묶음)."""
    sampling, sim, scorer, traffic = _W["sampling"], _W["simulator"], _W["scorer"], _W["traffic"]
    ego0 = mc.ego_state
    t0 = ego0.time_point
    K = len(cands)
    pdm_states = get_trajectory_as_array(mc.trajectory, sampling, t0)
    cs = [get_trajectory_as_array(transform_trajectory(Trajectory(v.astype(np.float32), _W["agent_sampling"]), ego0),
                                  sampling, t0) for v in cands]
    simulated = sim.simulate_proposals(np.stack([pdm_states] + cs, 0), ego0)
    tracks = traffic.simulate_environment(simulated[1], mc)          # log replay: 자차 무관
    res = scorer.score_proposals(simulated, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
                                 mc.map_parameters, tracks, mc.past_human_trajectory)
    progress_raw = scorer._progress_raw.copy()
    multi = scorer._multi_metrics.prod(axis=0)
    cfg = scorer._config
    fail, cont, ext = _derived(scorer, mc, margins)                  # 사람 궤적 채점 전에(내부 상태가 덮이기 전)  # 수정(2026-11-06): ext 추가

    filtered = np.zeros(len(FAIL_TYPES), bool)
    human = None
    if cfg.human_penalty_filter and mc.scene_type == SceneFrameType.ORIGINAL:
        hs = get_trajectory_as_array(transform_trajectory(mc.human_trajectory, ego0), sampling, t0)
        hsim = sim.simulate_proposals(hs[None], ego0)
        htr = traffic.simulate_environment(hsim[0], mc)
        human = scorer.score_proposals(hsim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
                                       mc.map_parameters, htr)[0]
        for i, m in enumerate(FAIL_METRIC):
            filtered[i] = bool(human[m].iloc[0] == 0)

    sub = np.zeros((K, len(SUBSCORES)))
    for k in range(K):
        r = res[k + 1].iloc[0]
        row = {m: float(r[m]) for m in SUBSCORES}
        pair_raw = np.array([progress_raw[0], progress_raw[k + 1]])
        norm = (pair_raw * np.array([multi[0], multi[k + 1]])).max()
        row["ego_progress"] = float(np.clip(pair_raw[1] / norm, 0.0, 1.0)) if norm > cfg.progress_distance_threshold else 1.0
        if human is not None:
            for c in human.columns:
                if c not in HUMAN_SKIP and c in row and human[c].iloc[0] == 0:
                    row[c] = 1.0
        sub[k] = [row[m] for m in SUBSCORES]
    mp = sub[:, 0] * sub[:, 1] * sub[:, 2] * sub[:, 3]
    wa = cfg.weighted_metrics_array.copy()
    wa[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = 0.0
    wm = np.zeros((K, len(wa)))
    wm[:, WeightedMetricIndex.PROGRESS] = sub[:, 4]
    wm[:, WeightedMetricIndex.TTC] = sub[:, 5]
    wm[:, WeightedMetricIndex.LANE_KEEPING] = sub[:, 6]
    wm[:, WeightedMetricIndex.HISTORY_COMFORT] = sub[:, 7]
    epdms = np.array([mp[k] * float((wm[k] * wa).sum() / wa.sum()) for k in range(K)])
    pdms_v1 = sub[:, 0] * sub[:, 1] * (5 * sub[:, 4] + 5 * sub[:, 5] + 2 * sub[:, 7]) / 12.0

    fail_c = fail[1:].copy()
    fail_c[:, filtered] = -1
    fail_wp = np.where(fail_c < 0, -1, np.clip(np.ceil(fail_c / 5.0).astype(np.int64) - 1, 0, 7))
    fail_wp[fail_c == -2] = -2   # 수정(2026-11-06): 이력 기원 편안함 실패 표시 유지
    cont_arr = np.stack([progress_raw[1:], np.full(K, progress_raw[0])] +
                        [cont[c][1:] for c in CONT[2:]], -1)
    ref = res[0].iloc[0]
    fr = _frenet(cands, ego0, mc.centerline)
    out = dict(subscores=sub, multiplicative=mp, epdms=epdms, pdms_v1like=pdms_v1, fail_idx=fail_c, fail_wp=fail_wp,
               filtered=filtered, cont=cont_arr, frenet=fr,
               ref_subscores=np.array([float(ref[m]) for m in SUBSCORES]))
    # 수정(2026-11-06): C3 확장 키(후보 행만; 0 번 = PDM 참조 제외)
    out.update({k: v[1:] for k, v in ext.items()})
    out["lat_dev_wp_m"] = np.abs(fr[..., 1]).mean(-1)
    if guide:
        out.update(_guide_scores(mc, human))
    return out


def _official(mc, cand):
    """검증: 공식 pdm_score 단독 호출 → (subscores 8, multiplicative, epdms(EC 0))."""
    r, _ = pdm_score(mc, Trajectory(cand.astype(np.float32), _W["agent_sampling"]), _W["sampling"], _W["simulator"],
                     _W["scorer"], _W["traffic"])
    r = r.iloc[0]
    w = r["weighted_metrics_array"].copy()
    w[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = 0.0
    return (np.array([float(r[m]) for m in SUBSCORES]), float(r["multiplicative_metrics_prod"]),
            float(r["multiplicative_metrics_prod"]) * float((r["weighted_metrics"] * w).sum() / w.sum()))


def _pair_fail(mc, cand):
    """검증: [참조, 후보] 2-묶음 score_proposals 로 구한 후보의 실패 시점(사람 필터 전, 원시)."""
    sampling, sim, scorer, traffic = _W["sampling"], _W["simulator"], _W["scorer"], _W["traffic"]
    ego0, t0 = mc.ego_state, mc.ego_state.time_point
    st = np.stack([get_trajectory_as_array(mc.trajectory, sampling, t0),
                   get_trajectory_as_array(transform_trajectory(Trajectory(cand.astype(np.float32),
                                                                           _W["agent_sampling"]), ego0), sampling, t0)])
    simulated = sim.simulate_proposals(st, ego0)
    tracks = traffic.simulate_environment(simulated[1], mc)
    scorer.score_proposals(simulated, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
                           mc.map_parameters, tracks, mc.past_human_trajectory)
    hc_raw = scorer._weighted_metrics[WeightedMetricIndex.HISTORY_COMFORT, 1]
    fail, cont, ext = _derived(scorer, mc, _W["margins"])   # 수정(2026-11-06): ext(at-fault 사건) 추가
    return fail[1], {c: cont[c][1] for c in CONT[2:]}, hc_raw, {k: v[1] for k, v in ext.items()}


def _job(args):
    tok, path, cands, verify = args[:4]
    guide = args[4] if len(args) > 4 else False   # 수정(2026-11-06): 단계 ③ 안내 궤적 채점 선택
    t = time.time()
    mc = load_cache(path)
    t_load = time.time() - t
    t = time.time()
    out = score_token(mc, cands, _W["margins"], guide=guide)
    t_score = time.time() - t
    if verify:
        for k in range(len(cands)):
            sub, mp, ep = _official(mc, cands[k])
            if not (np.array_equal(out["subscores"][k], sub) and out["multiplicative"][k] == mp and out["epdms"][k] == ep):
                raise AssertionError(f"{tok} 후보 {k}: 묶음 {out['subscores'][k]}/{out['multiplicative'][k]}/{out['epdms'][k]}"
                                     f" != 공식 {sub}/{mp}/{ep}")
            pf, pc, hc_raw, pe = _pair_fail(mc, cands[k])   # 수정(2026-11-06): pe = 2-묶음 at-fault 사건
            pf[out["filtered"]] = -1
            if not np.array_equal(pf, out["fail_idx"][k]):
                raise AssertionError(f"{tok} 후보 {k}: 실패 시점 묶음 {out['fail_idx'][k]} != 2-묶음 {pf}")
            for i, c in enumerate(CONT[2:]):
                a, b = out["cont"][k, i + 2], pc[c]
                if not (a == b or (np.isnan(a) and np.isnan(b))):
                    raise AssertionError(f"{tok} 후보 {k}: 연속 성분 {c} 묶음 {a} != 2-묶음 {b}")
            # 수정(2026-11-06): 확장 키도 2-묶음과 같아야 한다(첫 at-fault 사건·정면 판정·전방·측방 거리)
            for c in ("coll_t", "coll_token", "coll_obj_type", "coll_ctype", "coll_agent", "coll_front", "coll_nc",
                      "min_obj_dist_fs_m"):
                a, b = out[c][k], pe[c]
                if not (a == b or (isinstance(a, float) and np.isnan(a) and np.isnan(b))):
                    raise AssertionError(f"{tok} 후보 {k}: {c} 묶음 {a} != 2-묶음 {b}")
            if not out["filtered"][0] and (out["coll_t"][k] >= 0) != (out["subscores"][k, 0] < 1.0):
                raise AssertionError(f"{tok} 후보 {k}: coll_t {out['coll_t'][k]} / nc {out['subscores'][k, 0]} 불일치")
            if not out["filtered"][0] and out["coll_t"][k] >= 0 and out["coll_nc"][k] != out["subscores"][k, 0]:
                raise AssertionError(f"{tok} 후보 {k}: 재현 nc {out['coll_nc'][k]} != 공식 {out['subscores'][k, 0]}")
            # 내부 일관성: 실패 시점 유무 ↔ 하위 점수(필터 전 원시 값 기준)
            cr = out["cont"][k, CONT.index("comfort_ratio_max")]
            if not np.isnan(cr) and ((cr >= 1.0) != (hc_raw == 0.0)):
                raise AssertionError(f"{tok} 후보 {k}: 편안함 비율 {cr} 과 hc {hc_raw} 불일치")
            for i, m in enumerate(FAIL_METRIC):
                if out["filtered"][i] or m in ("no_at_fault_collisions",):
                    continue   # nc 는 0.5(비에이전트 객체)도 있어 시점 유무만 아래에서 확인
                if (out["fail_idx"][k, i] != -1) != (out["subscores"][k, SUBSCORES.index(m)] < 1.0):  # 수정(2026-11-06): -2(이력 기원)도 실패
                    raise AssertionError(f"{tok} 후보 {k}: {FAIL_TYPES[i]} 시점 {out['fail_idx'][k, i]} vs "
                                         f"{m}={out['subscores'][k, SUBSCORES.index(m)]}")
            if not out["filtered"][0] and (out["fail_idx"][k, 0] >= 0) != (out["subscores"][k, 0] < 1.0):
                raise AssertionError(f"{tok} 후보 {k}: collision 시점/nc 불일치")
        if guide:   # 수정(2026-11-06): 안내 궤적 점수 = 공식 pdm_score 단독 채점과 비트 동일
            for nm in ("ref", "human"):
                if np.isnan(out[f"{nm}_epdms"]):
                    continue
                sub, mp, ep = _official(mc, out[f"{nm}_traj"])
                if not (np.array_equal(out[f"{nm}_sub"], sub) and out[f"{nm}_epdms"] == ep):
                    raise AssertionError(f"{tok} {nm}: 안내 {out[f'{nm}_sub']}/{out[f'{nm}_epdms']} != 공식 {sub}/{ep}")
    return tok, out, t_load, t_score


# ---------------------------------------------------------------- API
class GRPOScorer:
    """여러 장면을 CPU 워커(≤ 16, 수정(2026-11-05): 사용자 결정 "채점 워커 16")로 나눠 채점. score() 는 {token: (K,8,2|3)} → {token: dict}."""

    def __init__(self, metric_cache, workers=8, margins=True, guide=False):  # 수정(2026-11-06): guide(단계 ③ 안내 궤적)
        if workers > MAX_WORKERS:
            raise ValueError(f"워커 상한 {MAX_WORKERS}(공유 서버)")
        if os.environ.get("OPENBLAS_CORETYPE") != "SkylakeX":
            raise RuntimeError("OPENBLAS_CORETYPE=SkylakeX 필요(scripts/navsim/navsim_env.sh)")
        for v in ("NUPLAN_MAPS_ROOT", "NUPLAN_MAP_VERSION"):
            if v not in os.environ:
                raise RuntimeError(f"환경변수 {v} 필요")
        self.paths = cache_paths(metric_cache)
        self.guide = guide
        self.workers = workers
        # 수정(2026-11-06): 검토 2 — multiprocessing.Pool 은 워커가 죽으면(OOM·SIGKILL) 그 작업을 잃고 imap/apply_async 가 영원히
        #   기다린다. ProcessPoolExecutor 는 죽음을 감지해 대기 중 future 전부에 BrokenProcessPool 을 건다(조용히 버리지 않음).
        #   fork 방식은 첫 submit 에서 워커 전부를 띄우므로 여기서 한 번 띄워 둔다(서버의 스레드가 생기기 전에 fork).
        #   #이전 self.pool = get_context("fork").Pool(workers, initializer=_init_worker, initargs=(margins,))
        # 수정(2026-11-06): 검토 3 — initargs 에 부모 pid(워커 PDEATHSIG) #이전 initargs=(margins,)
        self.ex = ProcessPoolExecutor(workers, mp_context=get_context("fork"), initializer=_init_worker,
                                      initargs=(margins, os.getpid()))
        self.ex.submit(os.getpid).result()
        if len(self.ex._processes) != workers:
            raise RuntimeError(f"워커 {len(self.ex._processes)}/{workers} 개만 떴다")
        self.last_timing = None

    def worker_pids(self):
        """수정(2026-11-06): 살아 있는 워커 pid(시험·서버 status 용).
        수정(2026-11-06): 검토 3 — _processes 는 관리 스레드가 바꾸고 shutdown 뒤 None 이 된다 → 복사 재시도·None 처리·죽은 pid 제외."""
        for _ in range(5):
            procs = getattr(self.ex, "_processes", None)
            if not procs:
                return []
            try:
                items = list(procs.items())
                break
            except RuntimeError:                                       # 순회 중 크기 변경
                continue
        else:
            raise RuntimeError("워커 목록 복사 실패(계속 변경 중)")
        return sorted(pid for pid, p in items if p.is_alive())

    def score(self, batch, verify=()):
        jobs = [(t, self.paths[t], _as_poses(c), t in verify, self.guide) for t, c in batch.items()]  # 수정(2026-11-06): guide
        out, tl, ts = {}, [], []
        # 수정(2026-11-06): 검토 2 — imap_unordered → submit/as_completed. 워커 죽음·예외는 result() 에서 그대로 올라온다.
        futs = [self.ex.submit(_job, j) for j in jobs]
        try:
            for f in as_completed(futs):
                tok, o, a, b = f.result()
                out[tok] = o
                tl.append(a)
                ts.append(b)
        except BaseException:
            for f in futs:                 # 남은 작업을 취소하고 원 예외를 그대로 올린다(부분 결과를 돌려주지 않음)
                f.cancel()
            raise
        if len(out) != len(jobs):
            raise RuntimeError(f"채점 결과 {len(out)}/{len(jobs)} token")
        self.last_timing = (np.array(tl), np.array(ts))
        return out

    def close(self):
        self.ex.shutdown(wait=True, cancel_futures=True)   # 수정(2026-11-06): 검토 3 — 대기 작업 취소 #이전 shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ---------------------------------------------------------------- CLI: 검증 + 속도
def _bench_cands(rng, anchors, K, noise):
    idx = rng.choice(len(anchors), K, replace=K > len(anchors))
    c = anchors[idx].copy()
    sc = np.linspace(0.15, 1.0, 8)[None, :, None]                   # 먼 경유점일수록 큰 섭동
    c = c + rng.normal(0, noise, c.shape) * sc
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric-cache", required=True)
    ap.add_argument("--anchors", default=str(Path(__file__).resolve().parents[2] / "data/kmeans/navsim_v2_plan_temp_48.npy"),
                    help="속도·검증용 후보 원천(ego=라이다 프레임 앵커 (48,8,2))")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--noise", type=float, default=1.5, help="앵커 섭동 표준편차(m, 4 s 끝 기준)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", type=int, default=4)
    ap.add_argument("--no-margins", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="결과 npz 저장 경로(선택)")
    ap.add_argument("--guide", action="store_true", help="수정(2026-11-06): 단계 ③ 안내 궤적(PDM-Closed·사람) 채점·검증")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    anchors = np.load(a.anchors).astype(np.float64)
    with GRPOScorer(a.metric_cache, a.workers, margins=not a.no_margins, guide=a.guide) as sc:  # 수정(2026-11-06): guide
        toks = sorted(sc.paths)
        toks = [toks[i] for i in rng.choice(len(toks), a.tokens, replace=False)]
        batch = {t: _bench_cands(rng, anchors, a.K, a.noise) for t in toks}
        vset = set(toks[:a.verify])
        t = time.time()
        out = sc.score({k: batch[k] for k in toks[:a.verify]}, verify=vset) if vset else {}
        if vset:
            print(f"[검증 통과] {len(vset)} token × {a.K} 후보: subscores·multiplicative·epdms 공식 pdm_score 와 비트 동일, "
                  f"실패 시점·연속 성분 2-묶음과 동일 ({time.time() - t:.1f}s)", flush=True)
            # 수정(2026-11-06): at-fault 재현 시점 = 공식 _collision_time_idcs (매 호출 단언) — 검증 표본의 충돌 건수 보고
            ct = np.concatenate([out[k]["coll_t"] for k in vset])
            fr = np.concatenate([out[k]["coll_front"] for k in vset])
            ag = np.concatenate([out[k]["coll_agent"] for k in vset])
            print(f"[검증 at-fault] 충돌 후보 {(ct >= 0).sum()}/{len(ct)} (token {sum((out[k]['coll_t'] >= 0).any() for k in vset)}): "
                  f"시점·nc 공식과 일치, 정면 {(fr & (ct >= 0)).sum()} · 측면 {(~fr & (ct >= 0)).sum()}, "
                  f"비에이전트(nc 0.5) {(~ag & (ct >= 0)).sum()}; comfort -2 {sum((out[k]['fail_idx'][:, 6] == -2).sum() for k in vset)}"
                  + (f"; 안내 궤적 공식과 비트 동일(ref r* {[round(float(out[k]['ref_epdms']), 3) for k in vset]})" if a.guide else ""),
                  flush=True)
        rest = {k: batch[k] for k in toks[a.verify:]}
        t = time.time()
        out.update(sc.score(rest))
        wall = time.time() - t
        tl, ts = sc.last_timing
    n = len(rest) * a.K
    print(f"[속도] {len(rest)} token × K={a.K} = {n} 후보, 워커 {a.workers}, 벽시계 {wall:.1f}s → {n / wall:.1f} 후보/s "
          f"({len(rest) / wall:.2f} token/s); 워커당 캐시 읽기 중앙 {np.median(tl):.3f}s, 채점 중앙 {np.median(ts):.3f}s/token")
    ep = np.stack([out[t]["epdms"] for t in toks])
    fail = np.stack([out[t]["fail_idx"] for t in toks])
    zero_std = (ep.std(1) == 0).mean()
    print(f"[분포] EPDMS 평균 {ep.mean():.3f}; 그룹(K={a.K}) 보상 표준편차 0 비율 {zero_std:.3f}; "
          + ", ".join(f"{f} {(fail[..., i] >= 0).mean():.3f}" for i, f in enumerate(FAIL_TYPES)))
    if a.save:
        np.savez(a.save, tokens=np.array(toks), epdms=ep, fail_idx=fail,
                 subscores=np.stack([out[t]["subscores"] for t in toks]), cont=np.stack([out[t]["cont"] for t in toks]),
                 subscore_names=np.array(SUBSCORES), fail_types=np.array(FAIL_TYPES), cont_names=np.array(CONT))


if __name__ == "__main__":
    main()
