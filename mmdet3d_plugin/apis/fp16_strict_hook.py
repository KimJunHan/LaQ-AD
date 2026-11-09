# 수정(2026-10-16): fp16 넘침 원인 수정 — mmcv 정적 loss scale 의 "조용한 step 건너뛰기" 제거.
#
# [왜 필요한가] mmcv 1.7.1 + torch>=1.6 의 Fp16OptimizerHook / GradientCumulativeFp16OptimizerHook 은
#   fp16 = dict(loss_scale=32.) 이어도 내부적으로 torch GradScaler(init_scale=32) 를 쓰고 매 step
#   GradScaler.step() → _maybe_opt_step() 를 부른다. 이 함수는 기울기에 inf/NaN 이 하나라도 있으면
#   optimizer.step() 을 **아무 기록 없이 건너뛴다**(torch/amp/grad_scaler.py:355). 스케일은 update(32.) 로
#   다시 32 로 고정되므로 "정적" 이지만, 건너뛰기 자체는 동적 loss scaling 과 똑같이 일어난다.
#   (HiP-AD·SparseDrive 원본도 같은 mmcv 경로라 원본 학습에서도 넘친 step 은 조용히 버려졌다.)
#   프로젝트 절대 규칙(건너뛰기 금지)에 어긋나므로, 같은 수치 방식(×32 정적 스케일, fp32 주 가중치,
#   unscale 후 clip)은 그대로 두고 **넘침이 생기면 그 자리에서 멈추고 원인 위치를 알린다.**
#
# [fp32 결과에 영향] 없음 — fp16=None 이면 이 훅을 쓰지 않는다. fp16 학습에서도 넘침이 없는 step 은
#   mmcv 훅과 계산이 완전히 같다(scale → backward → unscale_ → clip → step → update(32)).
import math

import torch
from mmcv.runner import HOOKS
from mmcv.runner.hooks.optimizer import GradientCumulativeFp16OptimizerHook


def _nonfinite_report(model, limit=20):
    bad = []
    for n, p in model.named_parameters():
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
            g = p.grad.detach()
            bad.append(f"{n}: inf {int(torch.isinf(g).sum())}, nan {int(torch.isnan(g).sum())} / {g.numel()}")
            if len(bad) >= limit:
                break
    return bad


@HOOKS.register_module()
class StrictGradientCumulativeFp16OptimizerHook(GradientCumulativeFp16OptimizerHook):
    """mmcv GradientCumulativeFp16OptimizerHook 와 같은 계산, 단 넘침 시 건너뛰지 않고 즉시 실패.

    - 매 micro-iter: 손실이 유한한지 확인(아니면 손실 항목별 값과 함께 실패).
    - optimizer step 직전: unscale_ 한 기울기가 전부 유한한지 확인(아니면 비유한 파라미터 이름과 함께 실패).
    loss_scale 은 float(정적)만 허용한다 — 'dynamic'/dict 는 스케일을 낮추며 step 을 버리는 방식이라 받지 않는다.
    """

    def __init__(self, *args, **kwargs):
        ls = kwargs.get("loss_scale", 512.)
        if not isinstance(ls, float):
            raise ValueError(f"StrictGradientCumulativeFp16OptimizerHook: loss_scale 은 정적 float 만 허용(받은 값 {ls!r})")
        super().__init__(*args, **kwargs)

    def after_train_iter(self, runner) -> None:
        if not self.initialized:
            self._init(runner)

        raw = runner.outputs["loss"]
        if not bool(torch.isfinite(raw.detach())):
            lv = {k: v for k, v in runner.outputs.get("log_vars", {}).items()}
            raise FloatingPointError(
                f"[fp16 strict] iter {runner.iter}: 손실이 유한하지 않다 → 건너뛰지 않고 멈춘다. 손실 항목: {lv}")
        loss = raw / self._get_loss_factor(runner)
        self.loss_scaler.scale(loss).backward()

        if self.every_n_iters(runner, self.cumulative_iters) or self.is_last_iter(runner):
            self.loss_scaler.unscale_(runner.optimizer)
            # GradScaler.unscale_ 가 기록한 넘침 여부(장치별) — step 이 이것을 보고 건너뛰므로 여기서 먼저 막는다.
            st = self.loss_scaler._per_optimizer_states[id(runner.optimizer)]
            found = sum(float(v.item()) for v in st["found_inf_per_device"].values())
            if found:
                bad = _nonfinite_report(runner.model)
                raise FloatingPointError(
                    f"[fp16 strict] iter {runner.iter}: ×{self.loss_scaler.get_scale():g} 스케일 기울기에 inf/NaN → "
                    f"건너뛰지 않고 멈춘다. 비유한 기울기 파라미터(최대 20): {bad}")
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    runner.log_buffer.update({"grad_norm": float(grad_norm)}, runner.outputs["num_samples"])
            self.loss_scaler.step(runner.optimizer)  # 위에서 넘침 없음을 확인했으므로 항상 실제 step
            self.loss_scaler.update(self._scale_update_param)
            runner.meta.setdefault("fp16", {})["loss_scaler"] = self.loss_scaler.state_dict()
            runner.model.zero_grad()
            runner.optimizer.zero_grad()
