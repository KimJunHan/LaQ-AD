# 수정(2026-11-05): 사용자 결정 "NAVSIM 1단계를 GPU1+GPU3 DDP 로 이어 간다" — 1 GPU(16 × 누적 4) 체크포인트를 DDP(GPU 2 × 16 × 누적 2)
#   재개용으로 옮긴다. 가중치·옵티마이저(AdamW 상태, step 수)·EMA 값은 바이트 그대로이고 "iter 번호"만 optimizer step 기준으로 바꾼다.
#     1 GPU: micro-iter = 4 × step  →  DDP: micro-iter = 2 × step.  iter_<N>.pth(N = 4s) → iter_<2s>.pth.
#     mmcv 규칙 그대로: iter_<M>.pth 의 meta.iter = M − 1 (재개 시 runner.iter = M − 1; mmcv save_checkpoint 와 같은 규칙, 모든 재개와 동일).
#     EMA: ema_iter_<N>.pth → ema_iter_<2s>.pth, iter = 2s, update_every 4 → 2, n_updates·decay·ramp·state_dict 그대로.
#   검사(위반 시 중단, 출력 없음): N % 4 == 0(step 경계), 원 meta.iter == N − 1, EMA iter == N·update_every == 4·decay/ramp 가 두 config 와 일치,
#     두 config 의 step 길이(max_iters / 누적)가 같음, 출력 폴더에 이미 파일이 있으면 덮어쓰지 않음.
#   샘플러: 재개 시 mmdet_train.py 가 runner.iter(= 2s − 1) 배치만큼 GroupInBatchSampler.fast_forward → rank r 그룹 k 슬롯
#     (전역 (r×2+k)×16+j) 이 1 GPU 그룹 k'=r×2+k 슬롯과 같은 장면 순서를 같은 위치(슬롯당 s 프레임 소비 직후)에서 이어 간다.
#   사용: python tools/navsim/convert_1gpu_to_ddp_ckpt.py <1gpu work_dir> <N> <ddp work_dir> <1gpu config> <ddp config>
import os
import sys

import torch
from mmcv import Config

src, N, dst, cfg1, cfg2 = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
c1 = Config.fromfile(cfg1, import_custom_modules=False)
c2 = Config.fromfile(cfg2, import_custom_modules=False)
a1, a2 = c1.optimizer_config.cumulative_iters * c1.num_gpus, c2.optimizer_config.cumulative_iters * c2.num_gpus
k1, k2 = c1.optimizer_config.cumulative_iters, c2.optimizer_config.cumulative_iters
assert c1.num_gpus == 1 and c2.num_gpus == 2 and c1.batch_size * k1 == c2.batch_size * k2 * c2.num_gpus, "유효 배치가 다르다"
assert c1.runner.max_iters // k1 == c2.runner.max_iters // k2, f"총 step 수 다름 {c1.runner.max_iters // k1} vs {c2.runner.max_iters // k2}"
assert c1.lr_config.warmup_iters // k1 == c2.lr_config.warmup_iters // k2 and c1.optimizer == c2.optimizer and c1.lr_config.policy == c2.lr_config.policy
h1, h2 = c1.custom_hooks[0], c2.custom_hooks[0]
assert (h1.decay, h1.ramp_steps, h1.update_every, h2.decay, h2.ramp_steps, h2.update_every) == (h2.decay, h2.ramp_steps, k1, h1.decay, h1.ramp_steps, k2)
assert N % k1 == 0, f"N={N} 이 step 경계(누적 {k1} 의 배수)가 아니다"
s = N // k1
M = s * k2
os.makedirs(dst, exist_ok=True)
outs = [os.path.join(dst, f"iter_{M}.pth"), os.path.join(dst, f"ema_iter_{M}.pth")]
for p in outs:
    if os.path.exists(p):
        raise SystemExit(f"[거부] 이미 있다: {p}")

ck = torch.load(os.path.join(src, f"iter_{N}.pth"), map_location="cpu", weights_only=False)
assert ck["meta"]["iter"] == N - 1, f"meta.iter {ck['meta']['iter']} != {N - 1}"
steps = {int(v["step"]) for v in ck["optimizer"]["state"].values() if "step" in v}
ck["meta"]["iter"] = M - 1
ck["meta"]["converted_from"] = dict(src=os.path.abspath(os.path.join(src, f"iter_{N}.pth")), src_iter=N, opt_step=s,
                                    rule="1gpu iter 4s -> ddp2 iter 2s (tools/navsim/convert_1gpu_to_ddp_ckpt.py)")
ema = torch.load(os.path.join(src, f"ema_iter_{N}.pth"), map_location="cpu", weights_only=False)
assert ema["iter"] == N and ema["update_every"] == k1 and (ema["decay"], ema["ramp_steps"]) == (h1.decay, float(h1.ramp_steps))
ema["iter"] = M
ema["update_every"] = k2
for obj, p in zip((ck, ema), outs):
    torch.save(obj, p + ".tmp")
    os.replace(p + ".tmp", p)
lk = os.path.join(dst, "latest.pth")
if os.path.lexists(lk):
    os.remove(lk)
os.symlink(f"iter_{M}.pth", lk)
print(f"1gpu iter_{N} (step {s}) -> ddp iter_{M}; AdamW step 값 {sorted(steps)[:3]}; EMA 갱신 {ema['n_updates']} 회 → {outs}")
