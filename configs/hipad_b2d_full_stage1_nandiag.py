# 수정: NaN 근본원인 진단 (일회성). base stage1 상속 + NanDiagHook + 촘촘 로깅.
_base_ = ["./hipad_b2d_full_stage1.py"]
log_config = dict(interval=1, hooks=[dict(type="TextLoggerHook", by_epoch=False)])
custom_hooks = [dict(type="NanDiagHook", watch_from=40800, topk=6)]
