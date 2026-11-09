# 수정(2026-11-09): 학습 속도 근본 수정 — optimizer 파라미터 그룹 병합(docs/navsim_s1_divergence_20261109.md §8 원인 1).
#
# [왜 필요한가] mmcv DefaultOptimizerConstructor 는 paramwise_cfg 가 있으면 파라미터마다 그룹을 하나씩 만든다
#   (NAVSIM 1단계 1,588 파라미터 → 그룹 1,588 개). torch AdamW 는 그룹마다 따로 step 을 돌므로
#   (1) foreach 커널 묶음이 그룹 수만큼 쪼개져 step 당 CPU 발사 비용이 크고(합성 재현 약 0.1 s/step),
#   (2) 재개한 런에서는 state["step"] 이 GPU 에 올라와 파라미터마다 step.item() 동기화가 2 회씩 생긴다(약 3,180 회/step, 1.1 s/step).
# [무엇을 하나] mmcv 규칙(custom_keys·lr_mult·decay_mult·bias/norm 규칙)으로 파라미터별 그룹을 그대로 만든 뒤,
#   하이퍼파라미터(lr, weight_decay, betas, eps, amsgrad, fused 등 params 외 모든 키)가 완전히 같은 그룹끼리 합친다.
#   파라미터마다 받는 lr·weight_decay·betas·eps 는 바뀌지 않는다(같은 값끼리만 합침). 그룹 순서 = 처음 나타난 순서,
#   그룹 안 파라미터 순서 = mmcv 순회 순서.
# [켜는 법] 새 config 에서만: optimizer = dict(constructor="GroupedOptimizerConstructor", fused=True, ...).
#   기존 config(constructor 미지정)는 mmcv 기본 생성기 그대로라 동작이 바뀌지 않는다.
# [호환] 그룹 수가 바뀌므로 기존(파라미터별 그룹) 체크포인트의 optimizer 상태로는 재개할 수 없다
#   (torch load_state_dict 가 "different number of parameter groups" 로 멈춘다 — 조용히 넘어가지 않음).
#   이 생성기로 시작한 런의 재개는 같은 config 면 같은 그룹이 다시 만들어지므로 문제없다. load_from(가중치만)은 무관.
# [LR·EMA] mmcv LrUpdaterHook 은 그룹마다 initial_lr 을 두고 같은 비율을 곱하므로 그룹 수와 무관하다.
#   EMASidecarHook 은 optimizer 를 읽지 않는다.
from collections import OrderedDict

from mmcv.runner.optimizer import OPTIMIZER_BUILDERS, OPTIMIZERS, DefaultOptimizerConstructor
from mmcv.utils import build_from_cfg, print_log


def _hashable(v):
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    return v


@OPTIMIZER_BUILDERS.register_module()
class GroupedOptimizerConstructor(DefaultOptimizerConstructor):
    """DefaultOptimizerConstructor 와 같은 파라미터별 하이퍼파라미터, 단 같은 값의 그룹을 하나로 합친다."""

    def __call__(self, model):
        per_param = super().__call__(model)  # mmcv 규칙 그대로 + torch 가 그룹마다 기본값을 채운 상태
        buckets = OrderedDict()
        for g in per_param.param_groups:
            hp = {k: v for k, v in g.items() if k != "params"}
            key = tuple(sorted((k, _hashable(v)) for k, v in hp.items()))
            if key not in buckets:
                buckets[key] = (hp, [])
            buckets[key][1].extend(g["params"])
        groups = [dict(hp, params=ps) for hp, ps in buckets.values()]

        optimizer_cfg = self.optimizer_cfg.copy()
        optimizer_cfg["params"] = groups
        opt = build_from_cfg(optimizer_cfg, OPTIMIZERS)

        # 병합 전후 파라미터별 하이퍼파라미터가 같은지 확인(다르면 즉시 실패).
        before = {id(p): g for g in per_param.param_groups for p in g["params"]}
        after = {id(p): g for g in opt.param_groups for p in g["params"]}
        assert before.keys() == after.keys() and len(before) == sum(len(g["params"]) for g in opt.param_groups), \
            "GroupedOptimizerConstructor: 병합 전후 파라미터 집합이 다르다"
        for pid, gb in before.items():
            ga = after[pid]
            for k, v in gb.items():
                if k != "params":
                    assert _hashable(ga.get(k)) == _hashable(v), f"GroupedOptimizerConstructor: 파라미터 하이퍼파라미터 {k} 불일치"
        summary = ", ".join(f"[lr {g['lr']:g}, wd {g.get('weight_decay', 0):g}, {len(g['params'])} params]" for g in opt.param_groups)
        print_log(f"GroupedOptimizerConstructor: {len(per_param.param_groups)} groups -> {len(opt.param_groups)} groups "
                  f"({type(opt).__name__}, fused={opt.defaults.get('fused')}): {summary}", logger="root")
        del per_param
        return opt
