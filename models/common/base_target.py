# 수정(2026-10-15): 경로 이동 models/base_target.py → models/common/base_target.py 및 정리.
#   노이즈 쿼리(denoising, DN) 생성·갱신·캐시(get_dn_anchors / update_dn / cache_dn)는 원본 Sparse4D 헤드
#   (Sparse4DHead, archive 로 이동) 전용이었다. 통합 디코더(SparseOneDecoder)는 DN 을 쓰지 않으므로
#   dn_metas 는 항상 None 이었다. 모든 config 가 num_dn_groups=0 / num_temp_dn_groups=0 을 넘기므로 키는 받되,
#   DN 을 켜려는 설정은 조용히 무시하지 않고 즉시 실패시킨다.
from abc import ABC, abstractmethod


__all__ = ["BaseTargetWithDenoising"]


class BaseTargetWithDenoising(ABC):
    def __init__(self, num_dn_groups=0, num_temp_dn_groups=0):
        super(BaseTargetWithDenoising, self).__init__()
        if num_dn_groups > 0 or num_temp_dn_groups > 0:
            raise ValueError(
                f"num_dn_groups={num_dn_groups}, num_temp_dn_groups={num_temp_dn_groups}: 노이즈 쿼리(DN) 경로는 "
                "2026-10-15 소스 정리에서 제거됐다(Sparse4DHead 전용, archive/refactor_20261015 참조).")
        self.num_dn_groups = num_dn_groups
        self.num_temp_dn_groups = num_temp_dn_groups

    @abstractmethod
    def sample(self, cls_pred, box_pred, cls_target, box_target):
        """
        Perform Hungarian matching between predictions and ground truth,
        returning the matched ground truth corresponding to the predictions
        along with the corresponding regression weights.
        """
