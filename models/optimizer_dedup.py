#수정: VLA-v0 — Qwen2-VL 의 tied weight (input_embed ↔ lm_head) 가 frozen 상태로 두 번 등록되어
#  mmcv DefaultOptimizerConstructor 가 'appears in more than one parameter group' 던지는 문제 해결.
#  로직: DefaultOptimizerConstructor 의 add_params 가 frozen param 도 group 에 추가 (bypass_duplicate
#  체크 이전에 continue) → tied weight 같은 frozen param 이 두 번 등록됨.
#  이 constructor 는 id() 로 dedup 한 뒤 optimizer 에 넘김. requires_grad=False 인 param 은 아예 제외.
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmcv.runner.optimizer.default_constructor import DefaultOptimizerConstructor
from mmcv.utils import build_from_cfg


@OPTIMIZER_BUILDERS.register_module()
class DedupOptimizerConstructor(DefaultOptimizerConstructor):
    """Default constructor + frozen-param 제거 + id() 중복 제거."""

    def __call__(self, model):
        if hasattr(model, "module"):
            model = model.module
        optimizer_cfg = self.optimizer_cfg.copy()
        # paramwise_cfg 비어있으면 default behavior (all params at base lr)
        if not self.paramwise_cfg:
            optimizer_cfg["params"] = [p for p in model.parameters() if p.requires_grad]
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

        params = []
        self.add_params(params, model)

        # dedup by id, drop frozen
        seen = set()
        deduped = []
        for grp in params:
            new_params = []
            for p in grp["params"]:
                if not p.requires_grad:
                    continue
                if id(p) in seen:
                    continue
                seen.add(id(p))
                new_params.append(p)
            if new_params:
                new_grp = {k: v for k, v in grp.items() if k != "params"}
                new_grp["params"] = new_params
                deduped.append(new_grp)
        optimizer_cfg["params"] = deduped
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
