# 수정(2026-10-15): 실험 C(LaQ 교정 레시피) — 스트림 그룹별 시간 인스턴스 뱅크 상태 저장·복원(새 파일).
#
#   [왜] SparseDrive/HiP-AD 원본은 GPU 8 개 × batch 8 = optimizer step 당 독립 시퀀스 64 개를 본다(각 GPU 의 뱅크가 자기
#   8 개 스트림만 들고 있다). 우리는 1 GPU 에서 micro-batch 16 × 누적 4 라, 종전 샘플러는 같은 16 개 시퀀스의 연속 4 프레임을
#   한 step 에 넣었다(독립 스트림 16 개). GroupInBatchSampler(num_stream_groups=G) 가 64 개 스트림을 G 개 그룹으로 나눠
#   micro-step 마다 그룹 k 만 내보내고, 디코더는 아래 함수로 micro-step 직전에 그룹 k 의 뱅크 상태를 복원하고 직후에 저장한다
#   → 그룹 k 의 뱅크는 "그룹 k 만 처리했을 때" 와 똑같은 상태를 본다(= GPU k 의 뱅크).
#
#   [무엇을 저장하나] 각 뱅크 클래스의 reset() 이 대입하는 속성 전부(= 프레임 간에 이어지는 상태의 정의)와 get() 이
#   대입하는 mask. 현재: InstanceBank(검출·지도) = cached_feature, cached_anchor, metas, mask, confidence, temp_confidence,
#   instance_id, prev_id / PlanningInstanceBank = cached_feature, cached_anchor, confidence, metas, mask /
#   EgoInstanceBank = cached_feature, cached_anchor, metas (+ get 의 mask). reset() 에 새 상태가 추가되면 자동으로 포함된다.
#   텐서는 참조만 옮긴다(복사 없음) — 뱅크는 cache() 에서 항상 새 텐서를 대입하고(detach) 제자리 수정은 그 새 텐서에만
#   하므로(검출 뱅크의 zero_velocity 대입은 cache() 안에서 막 만든 topk 결과에 한다) 다른 그룹의 저장본을 건드리지 않는다.
#   모듈·파라미터는 복제하지 않는다(deepcopy 없음) → state_dict·옵티마이저·메모리(파라미터) 불변. 상태는 파이썬 속성이라
#   체크포인트에도 들어가지 않는다(종전 뱅크 캐시와 같음).
import torch

__all__ = ["bank_state_keys", "save_bank_state", "load_bank_state", "empty_bank_state"]


class _Recorder:
    pass


def bank_state_keys(bank):
    """bank 의 reset() 이 대입하는 속성 이름 + 'mask'(get 에서 대입). reset 을 빈 객체에 적용해 이름을 얻는다."""
    rec = _Recorder()
    type(bank).reset(rec)
    keys = set(vars(rec))
    keys.add("mask")
    return tuple(sorted(keys))


def save_bank_state(bank, keys):
    return {k: getattr(bank, k, None) for k in keys}


def load_bank_state(bank, state):
    for k, v in state.items():
        setattr(bank, k, v)


def empty_bank_state(bank, keys):
    """reset() 직후 상태(그룹이 처음 쓰일 때). bank 자체는 건드리지 않는다."""
    rec = _Recorder()
    type(bank).reset(rec)
    st = {k: None for k in keys}
    st.update(vars(rec))
    return st
