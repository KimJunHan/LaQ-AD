# 수정: NaN 근본원인 진단용 훅 (일회성). batch16 stage1 이 iter~41350 에서 focal-loss NaN 발산.
# forward 훅으로 어느 모듈 activation 이 먼저 폭발/비유한이 되는지 잡고, 매 iter BN running_var·top activation 추이를 로깅.
import torch
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class NanDiagHook(Hook):
    def __init__(self, watch_from=40800, topk=5):
        self.watch_from = watch_from
        self.topk = topk
        self._act = {}          # module name -> max abs finite activation (이번 iter)
        self._nonfinite = []    # 이번 iter 비유한 출력 낸 모듈 (실행 순서)
        self._detail = {}       # 비유한 모듈의 입력/weight 상세
        self._handles = []

    def before_run(self, runner):
        model = runner.model
        for name, m in model.named_modules():
            if len(list(m.children())) != 0:
                continue  # leaf 모듈만
            self._handles.append(m.register_forward_hook(self._make_hook(name)))
        runner.logger.info(f"[NanDiag] forward 훅 {len(self._handles)}개 등록, watch_from={self.watch_from}")

    def _make_hook(self, name):
        def hook(module, inp, out):
            t = out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, (list, tuple)) and len(out) and isinstance(out[0], torch.Tensor) else None)
            if t is None or not t.is_floating_point():
                return
            finite = torch.isfinite(t)
            if not bool(finite.all()):
                self._nonfinite.append(name)
                mx = t[finite].abs().max().item() if bool(finite.any()) else float('inf')
                # 결정적 진단: 이 모듈의 입력·weight 가 이미 비유한인지(=원인이 상류/weight) vs 유한인데 출력만 터짐(=이 op)
                it = inp[0] if isinstance(inp, (list, tuple)) and len(inp) and isinstance(inp[0], torch.Tensor) else None
                in_fin = bool(torch.isfinite(it).all()) if it is not None else None
                in_mx = it[torch.isfinite(it)].abs().max().item() if (it is not None and bool(torch.isfinite(it).any())) else None
                w = getattr(module, 'weight', None)
                w_fin = bool(torch.isfinite(w).all()) if w is not None else None
                w_mx = w.abs().max().item() if (w is not None and w_fin) else (float('inf') if w is not None else None)
                self._detail[name] = dict(in_finite=in_fin, in_max=in_mx, w_finite=w_fin, w_max=w_mx)
            else:
                mx = t.abs().max().item()
            self._act[name] = mx
        return hook

    def before_train_iter(self, runner):
        self._act = {}
        self._nonfinite = []
        self._detail = {}

    def after_train_iter(self, runner):
        it = runner.iter
        loss = runner.outputs.get('loss', None) if hasattr(runner, 'outputs') else None
        loss_bad = loss is not None and (not torch.isfinite(loss).all())

        # BN running_var 최대치
        bn_top = sorted(
            [(n, b.max().item()) for n, b in runner.model.named_buffers() if n.endswith('running_var')],
            key=lambda x: -x[1])[:3]
        act_top = sorted(self._act.items(), key=lambda x: -x[1])[:self.topk]

        if self._nonfinite or loss_bad:
            runner.logger.info("=" * 60)
            runner.logger.info(f"[NanDiag] ★ 비유한 감지! iter={it} loss_bad={loss_bad}")
            runner.logger.info(f"[NanDiag] 처음 비유한 출력 모듈(실행순): {self._nonfinite[:8]}")
            first = self._nonfinite[0] if self._nonfinite else None
            if first:
                runner.logger.info(f"[NanDiag] ★첫 모듈 '{first}' 상세: {self._detail.get(first)}  (in_finite=True&out=inf → 이 op/weight 문제 / in_finite=False → 상류·데이터 문제)")
            runner.logger.info(f"[NanDiag] top activation: {[(n, round(v,1)) for n,v in act_top]}")
            runner.logger.info(f"[NanDiag] top BN running_var: {[(n, round(v,1)) for n,v in bn_top]}")
            runner.logger.info("=" * 60)
            # 낭비 방지: 즉시 중단
            for h in self._handles:
                h.remove()
            raise SystemExit("[NanDiag] 비유한 감지로 진단 종료")

        if it >= self.watch_from:
            runner.logger.info(
                f"[NanDiag] iter={it} top_act={[(n.split('.')[-1], round(v,1)) for n,v in act_top[:3]]} "
                f"bn_var={[(n.split('.')[-2], round(v,1)) for n,v in bn_top[:2]]}")
