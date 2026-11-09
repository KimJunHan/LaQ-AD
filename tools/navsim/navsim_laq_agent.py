"""수정(2026-10-09): LaQ-AD NAVSIM — EPDMS 채점용 "사전 계산 궤적" 에이전트(devkit env 에서 실행).

왜 에이전트인가: navtest(1단계) EPDMS 의 공식 진입점은 navsim/planning/script/run_pdm_score_one_stage.py 이고,
이 스크립트는 제출 파일이 아니라 AbstractAgent 를 instantiate 해 장면마다 compute_trajectory 를 부른다(:57-92).
run_pdm_score_from_submission.py 는 2단계 분할의 reactive_all_mapping 으로 점수를 집계하므로(:327) navtest 용이 아니다.
HiP-AD 추론(hipad_bw env)은 무겁고 env 가 달라, 추론 결과 {scene token: (8,3)} 를 먼저 만들고(NavsimDataset.evaluate)
이 에이전트가 token 으로 꺼내 돌려준다.
  - requires_scene=True: 러너가 compute_trajectory(agent_input, scene) 를 부른다(:88-90). scene 에서 쓰는 것은
    scene.scene_metadata.initial_token(=장면 token, dataclasses.py:495) 하나뿐이다 — 미래 정답(scene frames)은 읽지 않는다.
  - 센서 불필요: SensorConfig.build_no_sensors() (dataclasses.py:832).
  - token 누락은 KeyError → 러너가 그 장면을 valid=False 로 기록 → navsim_check_pdm_csv.py 가 전체를 실패 처리한다.
"""
import pickle

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory


class NavsimLaqAgent(AbstractAgent):
    def __init__(self, trajectory_sampling: TrajectorySampling, pred_path: str):
        super().__init__(trajectory_sampling=trajectory_sampling, requires_scene=True)
        self._pred_path = pred_path
        self._preds = None

    def name(self) -> str:
        return "navsim_laq_agent"

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def initialize(self) -> None:
        with open(self._pred_path, "rb") as f:
            self._preds = pickle.load(f)
        n = self._trajectory_sampling.num_poses
        for tok, p in self._preds.items():
            p = np.asarray(p)
            if p.shape != (n, 3) or not np.isfinite(p).all():
                raise ValueError(f"{tok}: 궤적 형식 오류 {p.shape} (기대 {(n, 3)})")

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        tok = scene.scene_metadata.initial_token
        if tok not in self._preds:
            raise KeyError(f"사전 계산 궤적에 장면 {tok} 가 없다")
        return Trajectory(np.asarray(self._preds[tok], dtype=np.float32), self._trajectory_sampling)
