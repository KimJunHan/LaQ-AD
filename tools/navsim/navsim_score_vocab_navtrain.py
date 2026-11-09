# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
"""NAVSIM — 고정 후보 궤적 집합(어휘, K 개)을 장면마다 PDM 채점해 EPDMS 세부 점수 (N, K, M) 를 저장한다.

용도: Hydra 방식 채점기(scorer) 학습 정답. Hydra-MDP 와 같이 "고정 어휘를 오프라인으로 한 번 채점" 한다.
  - 우리 계획 앵커(navsim_kmeans.py 의 navsim_plan_temp_K.npy)를 어휘로 쓰면, 모델 출력(앵커 + 정제 오프셋)은 채점하지 않는다.
    온라인 모델 출력 채점은 학습 중 CPU 시뮬레이션이 필요해 비현실적이다(장면당 수 초) — 오프라인 어휘 채점으로 고정.
실행 env: navsim (navsim 패키지 설치 필요). CPU 전용. 결과는 결정적(같은 캐시·어휘 → 같은 값).

공식 채점과의 일치 (navsim_ref @0a380a9 근거):
  - 공식 1개 궤적 채점 = navsim/evaluate/pdm_score.py:pdm_score — [PDM 참조, 예측] 2개를 한 묶음으로 시뮬레이션·채점(:146-172).
  - 진행도(ego_progress)만 묶음 안 최댓값으로 정규화된다(pdm_scorer.py:_aggregate_pdm_scores, :225-233). 나머지 지표는 후보별 독립.
  - 배경 차량은 non_reactive(LogReplayTrafficAgents) — 자차 궤적과 무관(log_replay_traffic_agents.py:30-56).
  ⇒ [PDM 참조, 후보 1..K] 를 한 번에 시뮬레이션·채점한 뒤, 진행도를 후보마다 "참조와 둘만 있을 때" 규칙으로 다시 정규화하면
    공식 pdm_score 를 후보마다 따로 부른 것과 같은 값이다(--verify N 이 처음 N 장면에서 후보 전부를 공식 함수와 대조해 다르면 즉시 실패).
  - 사람 궤적 벌점 필터(human_penalty_filter=True, 기본 scorer yaml)도 pdm_score.py:178-219 와 같은 규칙으로 후보마다 적용.
  - 두 프레임 확장 편안함(EC)은 직전 장면에서 "선택된" 궤적이 있어야 정의된다(run_pdm_score_one_stage.py:129-227) → 어휘
    채점에서는 정의되지 않으므로 저장하지 않고, score 는 EC 가중치 0 으로 계산(compute_final_scores 의 NaN 처리 :177-189 와 같음).
저장(shard 마다 npz): tokens (N,), vocab (K,8,3), metric_names (M,), scores (N,K,M) float32,
  M = [no_at_fault_collisions, drivable_area_compliance, driving_direction_compliance, traffic_light_compliance,
       ego_progress, time_to_collision_within_bound, lane_keeping, history_comfort, multiplicative_metrics_prod, score_no_ec]
실패 처리: 어떤 장면이든 예외가 나면 그 token 과 함께 즉시 중단한다(건너뛰지 않음). 이미 끝난 shard 파일은 다시 계산하지 않는다(재개).

예:
  CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX NUPLAN_MAPS_ROOT=/home1/irteam/data-vol1/nas_mirror/navsim/download/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \
  /home1/irteam/data-vol1/anaconda3/envs/navsim/bin/python tools/navsim/navsim_score_vocab_navtrain.py \
    --metric-cache /home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/eval_prep_20261019/metric_cache_navtrain_v2_remap \
    --vocab data/kmeans/navsim_plan_temp_48.npy --vocab-frame lidar --lidar2ego-identity \
    --out /home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/eval_prep_20261019/vocab_scores_navtrain_K48 \
    --workers 8 --shard-size 500 --verify 20
"""
import argparse
import json
import lzma
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

try:
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from navsim.common.dataclasses import Trajectory
    from navsim.common.enums import SceneFrameType
    from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_score, transform_trajectory
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
    from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
except ImportError as e:  # 설치 전에는 바로 멈춘다(조용한 대체 없음)
    sys.exit(f"[navsim_score_vocab_navtrain] navsim 패키지를 import 할 수 없다({e}). "
             "env 'navsim' 에 `pip install -e /home1/irteam/data-vol1/src/navsim_ref` 설치(사용자 승인 사항) 후 실행할 것.")

METRICS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
           "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
           "history_comfort", "multiplicative_metrics_prod", "score_no_ec"]
MULTI = METRICS[:4]
HUMAN_SKIP = {"multiplicative_metrics_prod", "weighted_metrics", "weighted_metrics_array", "pdm_score"}

# 공식 기본 설정(config/common/default_common.yaml:31-35, pdm_scoring/scorer/pdm_scorer.yaml, agent 4 s @ 0.5 s)
PROPOSAL = dict(num_poses=40, interval_length=0.1)
AGENT = dict(time_horizon=4, interval_length=0.5)


# ---------------------------------------------------------------- vocab
def heading_from_xy(xy, min_step=0.05):
    """(T,2) 누적 위치 → (T,3). NavsimDataset.plan_lidar_to_ego_poses 와 같은 규칙(이동 < min_step 이면 직전 heading)."""
    out = np.zeros((len(xy), 3), np.float64)
    prev_xy, prev_h = np.zeros(2), 0.0
    for t in range(len(xy)):
        d = xy[t] - prev_xy
        h = float(np.arctan2(d[1], d[0])) if np.hypot(*d) >= min_step else prev_h
        out[t] = (xy[t, 0], xy[t, 1], h)
        prev_xy, prev_h = xy[t], h
    return out


def load_vocab(path, frame, lidar2ego_identity):
    v = np.load(path).astype(np.float64)
    if v.ndim != 3 or v.shape[1] != 8 or v.shape[2] not in (2, 3):
        raise ValueError(f"어휘 형식 {v.shape} — (K, 8, 2|3) 이어야 한다(4 s @ 0.5 s, 누적 위치)")
    if frame == "lidar" and not lidar2ego_identity:
        raise ValueError("라이다 프레임 어휘는 lidar2ego 가 단위행렬임을 확인한 경우에만(--lidar2ego-identity) 그대로 쓴다 "
                         "(navtest/navtrain 로그 점검: 최대 편차 0.0). 아니면 ego 프레임으로 변환해 줄 것.")
    if v.shape[2] == 2:
        v = np.stack([heading_from_xy(t) for t in v])
    if not np.isfinite(v).all():
        raise ValueError("어휘에 비유한값")
    return v


# ---------------------------------------------------------------- cache
def cache_paths(cache_dir):
    md = Path(cache_dir) / "metadata"
    csvs = sorted(p for p in md.iterdir() if p.suffix == ".csv")
    if len(csvs) != 1:
        raise RuntimeError(f"{md}: metadata csv 가 정확히 1개여야 한다({len(csvs)})")
    rows = csvs[0].read_text().splitlines()[1:]
    out = {r.split("/")[-2]: r for r in rows}
    if len(out) != len(rows):
        raise RuntimeError("metadata csv 에 중복 token")
    return out


def load_cache(path):
    with open(path, "rb") as f:
        return pickle.loads(lzma.decompress(f.read()))


# ---------------------------------------------------------------- scoring
_W = {}


def _init_worker(vocab):
    sampling = TrajectorySampling(**PROPOSAL)
    _W.update(vocab=vocab, sampling=sampling, agent_sampling=TrajectorySampling(**AGENT),
              simulator=PDMSimulator(sampling),
              scorer=PDMScorer(sampling, PDMScorerConfig(human_penalty_filter=True)),
              traffic=LogReplayTrafficAgents(sampling))


def score_token_batched(mc):
    """(K, M) — [참조, 후보들] 한 묶음 채점 후 진행도를 후보별 2-묶음 규칙으로 재정규화."""
    vocab, sampling, sim, scorer, traffic = _W["vocab"], _W["sampling"], _W["simulator"], _W["scorer"], _W["traffic"]
    ego0 = mc.ego_state
    t0 = ego0.time_point
    pdm_states = get_trajectory_as_array(mc.trajectory, sampling, t0)
    cand = [get_trajectory_as_array(transform_trajectory(Trajectory(v.astype(np.float32), _W["agent_sampling"]), ego0),
                                    sampling, t0) for v in vocab]
    states = np.stack([pdm_states] + cand, 0)
    simulated = sim.simulate_proposals(states, ego0)
    tracks = traffic.simulate_environment(simulated[1], mc)  # log replay: 자차 무관
    res = scorer.score_proposals(simulated, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
                                 mc.map_parameters, tracks, mc.past_human_trajectory)
    progress_raw = scorer._progress_raw.copy()           # (1+K,) 정규화 전 진행도(m)
    multi = scorer._multi_metrics.prod(axis=0)           # (1+K,)
    cfg = scorer._config
    wa = cfg.weighted_metrics_array.copy()

    human = None
    if cfg.human_penalty_filter and mc.scene_type == SceneFrameType.ORIGINAL:
        ht = transform_trajectory(mc.human_trajectory, ego0)
        hs = get_trajectory_as_array(ht, sampling, t0)
        hsim = sim.simulate_proposals(hs[None], ego0)
        htr = traffic.simulate_environment(hsim[0], mc)
        human = scorer.score_proposals(hsim, mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map,
                                       mc.map_parameters, htr)[0]

    out = np.zeros((len(vocab), len(METRICS)), np.float64)
    for k in range(len(vocab)):
        r = res[k + 1].iloc[0]
        row = {m: float(r[m]) for m in METRICS[:8]}
        # 진행도: 공식 2-묶음 [참조, 후보] 의 정규화(pdm_scorer.py:_aggregate_pdm_scores)
        pair_raw = np.array([progress_raw[0], progress_raw[k + 1]])
        pair_masked = pair_raw * np.array([multi[0], multi[k + 1]])
        norm = pair_masked.max()
        row["ego_progress"] = float(np.clip(pair_raw[1] / norm, 0.0, 1.0)) if norm > cfg.progress_distance_threshold else 1.0
        if human is not None:
            for c in human.columns:
                if c not in HUMAN_SKIP and c in row and human[c].iloc[0] == 0:
                    row[c] = 1.0
        mp = row[MULTI[0]] * row[MULTI[1]] * row[MULTI[2]] * row[MULTI[3]]
        wm = np.zeros(len(wa))
        wm[WeightedMetricIndex.PROGRESS] = row["ego_progress"]
        wm[WeightedMetricIndex.TTC] = row["time_to_collision_within_bound"]
        wm[WeightedMetricIndex.LANE_KEEPING] = row["lane_keeping"]
        wm[WeightedMetricIndex.HISTORY_COMFORT] = row["history_comfort"]
        w = wa.copy()
        w[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = 0.0   # EC 미정의 → 가중치 0 (compute_final_scores NaN 규칙)
        row["multiplicative_metrics_prod"] = mp
        row["score_no_ec"] = mp * float((wm * w).sum() / w.sum())
        out[k] = [row[m] for m in METRICS]
    return out


def score_token_official(mc, k):
    """검증용: 공식 pdm_score 를 후보 k 하나로 호출(2-묶음)."""
    traj = Trajectory(_W["vocab"][k].astype(np.float32), _W["agent_sampling"])
    r, _ = pdm_score(mc, traj, _W["sampling"], _W["simulator"], _W["scorer"], _W["traffic"])
    r = r.iloc[0]
    w = r["weighted_metrics_array"].copy()
    w[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = 0.0
    score = r["multiplicative_metrics_prod"] * float((r["weighted_metrics"] * w).sum() / w.sum())
    return np.array([float(r[m]) for m in METRICS[:9]] + [score])


def _job(args):
    tok, path, verify = args
    t = time.time()
    mc = load_cache(path)
    t_load = time.time() - t
    t = time.time()
    s = score_token_batched(mc)
    t_score = time.time() - t
    if verify:
        for k in range(len(_W["vocab"])):
            ref = score_token_official(mc, k)
            if not np.allclose(s[k], ref, atol=1e-9, equal_nan=True):
                raise AssertionError(f"{tok} 후보 {k}: 묶음 채점 {s[k]} != 공식 {ref}")
    return tok, s.astype(np.float32), t_load, t_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric-cache", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--vocab-frame", choices=["ego", "lidar"], required=True)
    ap.add_argument("--lidar2ego-identity", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="시간 측정용: 앞의 N token 만(학습 정답 생성에는 쓰지 말 것)")
    ap.add_argument("--verify", type=int, default=20, help="처음 N token 은 후보 전부를 공식 pdm_score 와 대조")
    a = ap.parse_args()
    if a.workers > 16:
        raise ValueError("CPU worker 상한 16")
    for v in ("NUPLAN_MAPS_ROOT", "NUPLAN_MAP_VERSION"):
        if v not in os.environ:
            raise RuntimeError(f"환경변수 {v} 필요")
    vocab = load_vocab(a.vocab, a.vocab_frame, a.lidar2ego_identity)
    paths = cache_paths(a.metric_cache)
    toks = sorted(paths)
    if a.limit:
        toks = toks[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(dict(
        tag="# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)", metric_cache=a.metric_cache, vocab=a.vocab,
        vocab_frame=a.vocab_frame, K=len(vocab), metrics=METRICS, n_tokens=len(toks), proposal=PROPOSAL,
        traffic="non_reactive(log_replay)", human_penalty_filter=True, ec="미정의(가중치 0)"), ensure_ascii=False, indent=1))
    shards = [toks[i:i + a.shard_size] for i in range(0, len(toks), a.shard_size)]
    verify_set = set(toks[:a.verify])
    t_all = time.time()
    tl, ts = [], []
    with Pool(a.workers, initializer=_init_worker, initargs=(vocab,)) as pool:
        for si, sh in enumerate(shards):
            f = out / f"shard_{si:05d}.npz"
            if f.exists():
                continue
            res = dict()
            for tok, s, t_load, t_score in pool.imap_unordered(_job, [(t, paths[t], t in verify_set) for t in sh]):
                res[tok] = s
                tl.append(t_load)
                ts.append(t_score)
            np.savez(str(f) + ".tmp.npz", tokens=np.array(sh), vocab=vocab, metric_names=np.array(METRICS),
                     scores=np.stack([res[t] for t in sh]))
            os.replace(str(f) + ".tmp.npz", f)
            print(f"[{si + 1}/{len(shards)}] {len(sh)} token, 경과 {time.time() - t_all:.0f}s, "
                  f"캐시 읽기 중앙 {np.median(tl):.2f}s, 채점 중앙 {np.median(ts):.2f}s/token(K={len(vocab)})", flush=True)
    print("완료", out)


if __name__ == "__main__":
    main()
