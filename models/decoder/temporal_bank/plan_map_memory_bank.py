# 수정(2026-10-19): 사용자 결정 — 지도 정렬 선택적 Gaussian 잡음(지도 밖 잡음 제거)
#   지도 인스턴스 뱅크(InstanceBank)에 "직전 프레임에 모델이 예측한 지도(선 + 클래스 점수)" 기억을 더한 하위 클래스.
#   models/decoder/noise_t/plan_diffusion_map.py 의 MapAlignedPlanTruncatedDiffusion 이 계획 앵커에 잡음을 넣을 때 쓴다.
#
#   [왜 직전 프레임인가] 계획 잡음은 디코더 맨 앞(첫 층 이전)에서 넣는다. 그 시점에 현재 프레임 지도 예측은 아직 없다.
#     nuScenes 2단계 config 는 지도 뱅크의 시간 기억이 꺼져 있다(num_temp_instances=0 → InstanceBank.cache 가 아무것도 저장하지
#     않는다). 그래서 이 클래스가 마지막 층의 지도 예측(점 20 개 × (x, y) m, 라이다 좌표 + 클래스 로짓)을 따로 떼어(detach) 저장하고,
#     다음 프레임 get() 에서 자차 이동(T_global_inv(현재) @ T_global(직전))으로 현재 좌표에 옮긴다. 지도는 정지 물체라 옮긴 값이
#     그대로 유효하다. 학습·평가 모두 같은 규칙(정답 지도는 쓰지 않음 → 학습/평가 불일치 없음).
#   [언제 기억이 없나] 시퀀스 첫 프레임, 배치 크기가 바뀐 경우, 시간 차 > max_time_interval(2 s, 다른 뱅크와 같은 규칙),
#     scene_token 이 다른 경우 → 해당 표본은 지도 없음(잡음 모듈이 등방 잡음으로 되돌린다 = "선택적").
#   [지도 과제 동작] get/update/cache 의 반환값·상태는 InstanceBank 와 완전히 같다(기억은 별도 속성). 파라미터 추가 없음
#     → state_dict·체크포인트 호환(1단계 체크포인트를 그대로 읽는다).
#   [스트림 그룹] 기억 속성은 reset() 이 대입하므로 temporal_bank/stream_state.py 가 그룹별로 자동 저장·복원한다.
#     단 InstanceBank.get() 이 기억이 없을 때마다 reset() 을 부르므로, get() 에서 기억 속성만 보존했다가 되돌린다.
import numpy as np
import torch
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

from .instance_bank import InstanceBank
from ...common.sync_free import to_device_async  # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — 장면 동일 여부·T 비동기 H2D

__all__ = ["PlanMapMemoryInstanceBank"]


@PLUGIN_LAYERS.register_module()
class PlanMapMemoryInstanceBank(InstanceBank):
    _MEM_KEYS = ("plan_map_pts", "plan_map_scores", "plan_map_ts", "plan_map_T_global", "plan_map_scene")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.anchor_handler is None:
            raise ValueError("PlanMapMemoryInstanceBank: 직전 지도 예측을 현재 좌표로 옮기려면 anchor_handler(SparsePoint3DKeyPointsGenerator)가 필요하다.")
        self.plan_map_memory = None

    def reset(self):
        # super() 대신 직접 호출: stream_state.bank_state_keys 가 type(bank).reset(빈 객체) 로 상태 이름을 얻는다.
        InstanceBank.reset(self)
        self.plan_map_pts = None       # (B, M, P*2) 직전 프레임 마지막 층 지도 예측(라이다 m), detach
        self.plan_map_scores = None    # (B, M, C) sigmoid 클래스 점수, detach
        self.plan_map_ts = None        # (B,) 직전 프레임 timestamp
        self.plan_map_T_global = None  # list[np.ndarray 4x4] 직전 프레임 T_global
        self.plan_map_scene = None     # list[scene_token or None]

    @staticmethod
    def _scenes(metas):
        return [m.get("scene_token") for m in metas["img_metas"]]

    def get(self, batch_size, metas=None):
        mem = {k: getattr(self, k, None) for k in self._MEM_KEYS}
        out = super().get(batch_size, metas)
        for k, v in mem.items():   # InstanceBank.get 의 reset() 이 지운 기억만 되돌린다(지도 과제 상태는 그대로)
            setattr(self, k, v)
        self.plan_map_memory = self._project_memory(batch_size, metas)
        return out

    def _project_memory(self, batch_size, metas):
        """직전 프레임 지도 예측을 현재 라이다 좌표로 옮긴다. 반환 dict(pts=(B,M,P,2), scores=(B,M,C)) 또는 None."""
        if self.plan_map_pts is None or self.plan_map_pts.shape[0] != batch_size or metas is None:
            return None
        ts = metas["timestamp"]
        dt = (ts - self.plan_map_ts).abs()
        valid = dt <= self.max_time_interval
        cur_scene = self._scenes(metas)
        # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — torch.tensor(리스트, device=)·new_tensor(numpy) 의 pageable H2D(호스트 동기화) → 고정 메모리 비동기 복사.
        #   같은 값·dtype(bool / plan_map_pts.dtype, CPU 에서 변환). 목적지가 CPU 면 to_device_async 가 고정 메모리를 쓰지 않는다.
        same_scene = to_device_async([(a is None or b is None or a == b) for a, b in zip(cur_scene, self.plan_map_scene)],
                                     valid.device, dtype=torch.bool)
        valid = valid & same_scene
        T = to_device_async(np.stack([
            x["T_global_inv"] @ self.plan_map_T_global[i] for i, x in enumerate(metas["img_metas"])]),
            self.plan_map_pts.device, dtype=self.plan_map_pts.dtype)
        pts = self.anchor_handler.anchor_projection(self.plan_map_pts, [T])[0]
        B, M, _ = pts.shape
        scores = self.plan_map_scores * valid.to(self.plan_map_scores.dtype)[:, None, None]  # 무효 표본 = 점수 0 = 지도 없음
        return dict(pts=pts.reshape(B, M, -1, 2), scores=scores)

    def cache(self, instance_feature, anchor, confidence, metas=None, feature_maps=None):
        super().cache(instance_feature, anchor, confidence, metas, feature_maps)
        self.plan_map_pts = anchor.detach().float()
        self.plan_map_scores = confidence.detach().float().sigmoid()
        self.plan_map_ts = metas["timestamp"].detach()
        self.plan_map_T_global = [np.asarray(m["T_global"]) for m in metas["img_metas"]]
        self.plan_map_scene = self._scenes(metas)
