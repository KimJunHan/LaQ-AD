# 수정(2026-10-30): 사용자 결정 "EMA 는 가중치 같이 저장" — 학습은 그대로 두고 지수 이동 평균(EMA) 가중치를 따로 유지해
#   체크포인트를 저장할 때마다 같은 폴더에 ema_iter_<N>.pth 로 함께 저장한다(iter_<N>.pth 와 같은 N).
#
# [학습에 영향] 없음 — 모델 파라미터·버퍼·state_dict 키·옵티마이저를 건드리지 않는다(읽기만). rank 0 만 동작.
#   mmcv EMAHook 을 쓰지 않는 이유: (1) EMA 를 모델 버퍼로 등록해 state_dict 키가 바뀌고(평가 config 로 로드 시 키 불일치),
#   (2) 재개 시 runner.resume 이 before_run 보다 먼저 돌아 EMA 가 사라지며, (3) epoch 훅에서 파라미터를 맞바꿔 IterBasedRunner 에서
#   학습 가중치 자체가 EMA 로 바뀔 수 있다.
# [대상] 모델 state_dict 의 부동소수 텐서 전부(파라미터 + BN running_mean/var). 정수 버퍼(num_batches_tracked)는 평가용 변환
#   (tools/ema_to_ckpt.py)에서 같은 iter 의 원 체크포인트 값을 쓴다.
# [갱신] optimizer step 마다 1회(update_every = 기울기 누적 수), decay_t = decay × (1 − exp(−t / ramp_steps)) (YOLOX 방식 워밍업).
#   EMA ← decay_t × EMA + (1 − decay_t) × 현재값, fp32.
# [재개] runner.iter > 0 이면 work_dir/ema_iter_<runner.iter + 1>.pth(재개한 iter_<N>.pth 와 같은 N)가 반드시 있어야 한다
#   (없으면 멈춤 — EMA 를 조용히 새로 시작하지 않음).
import math
import os
import os.path as osp

import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, CheckpointHook, Hook, get_dist_info


@HOOKS.register_module()
class EMASidecarHook(Hook):
    def __init__(self, decay=0.9998, ramp_steps=2000, update_every=1, max_keep=3):
        assert 0.0 < decay < 1.0 and ramp_steps > 0 and int(update_every) >= 1
        self.decay = float(decay)
        self.ramp_steps = float(ramp_steps)
        self.update_every = int(update_every)
        self.max_keep = int(max_keep)
        self.active = False

    def _model(self, runner):
        m = runner.model
        return m.module if is_module_wrapper(m) else m

    def before_run(self, runner):
        rank, _ = get_dist_info()
        self.active = rank == 0
        if not self.active:
            return
        ck = [h for h in runner.hooks if isinstance(h, CheckpointHook)]
        assert len(ck) == 1 and not ck[0].by_epoch, "EMASidecarHook: iter 기준 CheckpointHook 이 하나 있어야 한다"
        self.ckpt_hook = ck[0]
        sd = self._model(runner).state_dict()  # detach 된 텐서지만 파라미터·버퍼와 저장 공간을 공유(제자리 갱신이 그대로 보인다)
        self.keys = [k for k, v in sd.items() if v.is_floating_point()]
        self.live = [sd[k] for k in self.keys]
        self.ema = [v.detach().clone().float() for v in self.live]
        self.n = 0
        if runner.iter > 0:
            # mmcv IterBasedRunner 는 iter_<N>.pth 의 meta 에 iter=N-1 을 적고(save_checkpoint: filename = iter+1, meta iter = iter)
            #   재개 시 runner.iter = N-1 로 시작한다 → 같은 N 의 EMA 파일을 읽는다(모델 가중치와 같은 시점).
            n_file = runner.iter + 1
            p = osp.join(runner.work_dir, f"ema_iter_{n_file}.pth")
            if not osp.isfile(p):
                raise RuntimeError(f"[EMA] 재개(runner.iter {runner.iter}) 의 EMA 파일이 없다: {p} — EMA 를 새로 시작하지 않고 멈춘다")
            st = torch.load(p, map_location="cpu")
            assert st["iter"] == n_file and list(st["state_dict"].keys()) == self.keys, f"[EMA] {p} 의 iter·키가 맞지 않다"
            assert (st["decay"], st["ramp_steps"], st["update_every"]) == (self.decay, self.ramp_steps, self.update_every), \
                f"[EMA] {p} 의 설정이 지금과 다르다"
            for e, k in zip(self.ema, self.keys):
                e.copy_(st["state_dict"][k])
            self.n = int(st["n_updates"])
            runner.logger.info(f"[EMA] {p} 에서 재개 (갱신 {self.n} 회)")

    @torch.no_grad()
    def after_train_iter(self, runner):
        if not self.active:
            return
        if (runner.iter + 1) % self.update_every == 0:  # 기울기 누적 훅과 같은 경계(every_n_iters)
            self.n += 1
            d = self.decay * (1.0 - math.exp(-self.n / self.ramp_steps))
            cur = [v.float() for v in self.live]
            torch._foreach_mul_(self.ema, d)
            torch._foreach_add_(self.ema, cur, alpha=1.0 - d)
        h = self.ckpt_hook
        if self.every_n_iters(runner, h.interval) or (h.save_last and self.is_last_iter(runner)):
            self._save(runner, runner.iter + 1)

    def _save(self, runner, it):
        p = osp.join(runner.work_dir, f"ema_iter_{it}.pth")
        tmp = p + ".tmp"
        torch.save(dict(iter=it, n_updates=self.n, decay=self.decay, ramp_steps=self.ramp_steps,
                        update_every=self.update_every,
                        state_dict={k: e.detach().cpu().clone() for k, e in zip(self.keys, self.ema)}), tmp)
        os.replace(tmp, p)
        runner.logger.info(f"[EMA] 저장 {p} (갱신 {self.n} 회)")
        if self.max_keep > 0:
            olds = sorted(int(f[9:-4]) for f in os.listdir(runner.work_dir)
                          if f.startswith("ema_iter_") and f.endswith(".pth") and f[9:-4].isdigit())
            for o in olds[:-self.max_keep]:
                try:
                    os.remove(osp.join(runner.work_dir, f"ema_iter_{o}.pth"))
                except FileNotFoundError:
                    pass
