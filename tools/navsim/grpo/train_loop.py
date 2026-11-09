# 수정(2026-11-09): C3 GRPO 설계 v4.1 §8·§9·§12-⑥⑥′ 자체 학습 루프(M7) — 신규 파일(기존 파일 무수정).
"""NAVSIM GRPO 학습 루프(mmcv runner 없음). hipad_bw env, GPU 1 장. 채점 서버는 navsim env 자식 프로세스로 띄운다.

전제: docs/grpo_model_flag_20261109.diff 적용(모델 플래그). 미적용이면 rollout.build_rl_model 이 즉시 실패.

1 step(§9.2·§9.3) = micro-batch 6 × 8 프레임(스트림 그룹 6, GroupInBatchSampler).
  ① 6 개 micro-batch 를 데이터로더에서 먼저 받아 RL 장면 수 n_rl(평가 프레임 ∧ 지표 캐시 있음)을 센다 → N = n_rl × M × G(§12.1(3)).
  ② micro-batch 마다: grad 켠 롤아웃(eval 모드, 격리 켬) → 명령 묶음 16 모드 μθ·원시 logit → rescore(no_grad) → 상위 M → 차선축
     AR(1) 표본(η = default_rng([seed, step, token_hash, mode])) → 후보 [16 ξ=0, M×G] 를 채점 서버에 비동기 제출.
     L_BC(2단계 계획 손실, 롤아웃 출력)는 바로 계산해 둔다.
  ③ 점수가 도착한 micro-batch 부터 L = L_RL + L_g + λ_BC·L_BC + λ_cls·L_cls 를 만들어 즉시 backward(누적), 그래프를 놓는다.
     L_RL = ar1.loss_rl_torch(N = step 전체 정책 표본 수). L_g·L_BC·L_cls 는 micro-batch 안 프레임 평균 / 6.
     L_cls = 명령 묶음 16 원시 logit 의 교차 엔트로피(목표 softmax(EPDMS/0.1)), rescore −999 프레임 제외(§12.1(5)).
  ④ clip 1.0 → AdamW(lr 1e-5, wd 1e-4, 학습 대상만, lr_mult 1) → cosine(warmup 100) → EMA(d = decay·(1 − exp(−n/ramp))).
변형: Full·B1·B2·R·P·A-a·A-b·A-d(advantage.scene_advantage, R 은 ego 축 기저), C1(L_cls + L_BC, ξ=0 16 모드만 채점),
  C2(L_BC 만, 감시용으로 16 모드만 채점).
감시(§9.8 G3): 매 step RL 프레임의 ξ=0 최종 궤적(rescore 후 argmax)의 속도(끝 변위/4), ep(subscores[:,4]), 정지(끝 변위 < 2 m ∧
  ref_progress_m > 5 m), KL(μ_ref, θ 측 기저). 50 step 마다 이동 창(step < 500: 처음부터, 이후 직전 500 step)으로 B0 대비 판정,
  step ≥ 250 부터. 정지율 +5 %p 초과 또는 평균 ep −0.05 초과면 G3_STOP.json + 체크포인트·final_ema 저장 후 종료 코드 3.
  50 step 마다 항별(L_RL, L_g, L_BC, L_cls) 매개변수 기울기 노름, 출력 공간 축별·혁신별 L_RL 기울기 노름(ar1.grad_mu_np), clip 발동 기록.
모드
  기본 학습 | --measure-sigma-ref K(G2-5: 갱신 없이 K step γ = ‖g_RL‖/‖g_BC+g_cls‖ 중앙값 → σ_ref 규칙, sigma_ref.json)
  | --hold-all-graphs(G2-7: 6 개 그래프를 모두 든 최악 순서, micro-batch 별 최대 메모리 기록) | --no-isolation(G2-7 비교용).
재개: work_dir/latest.pth(학습 대상 가중·AdamW·EMA·step·스트림 뱅크 상태·torch/numpy/python RNG·감시 기록) + 샘플러 fast_forward.
  NAVSIM 증강은 결정적이고(rollout.build_rl_dataset) 샘플러 skip/flip 확률이 0 이라 numpy RNG 는 GridMask(eval 무효)만 쓴다 —
  데이터로더 선행 읽기로 RNG 소비 순서가 달라져도 값은 같다.
GPU: CUDA_VISIBLE_DEVICES 는 정확히 한 장. 2 금지. 0 은 HIPAD_ALLOW_GPU0=1 이고 스모크·G2 모드(--smoke/--measure-sigma-ref/
  --hold-all-graphs)일 때만. 본 런은 GPU3(NAVSIM 전용).
산출: work_dir/{log.jsonl, latest.pth, final.pth(온라인), final_ema.pth(test.py 형식: meta + 전체 state_dict, 학습 대상만 EMA),
  monitor.jsonl, DONE 또는 G3_STOP.json}.
"""
import argparse
import ast
import fcntl
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(os.path.abspath(__file__)).parent   # 심볼릭 링크를 풀지 않는다(스모크 오버레이에서 그 트리의 models 를 쓰도록)
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import advantage as AD  # noqa: E402
import ar1  # noqa: E402
import rollout as RO  # noqa: E402
from score_server import ScoreClient  # noqa: E402

RL_VARIANTS = AD.VARIANTS
ALL_VARIANTS = RL_VARIANTS + ("C1", "C2")
SUB_EP = 4
CONT_REF_PROGRESS = 1          # grpo_scorer.CONT 의 "ref_progress_m" 색인. navsim env 밖이라 import 하지 않고, 시작 시 check_cont_index() 가
#                                 tools/navsim/grpo_scorer.py 소스의 CONT 리터럴을 AST 로 읽어 색인이 맞는지 검사한다(어긋나면 중단).
SCORER_SRC = ROOT / "tools" / "navsim" / "grpo_scorer.py"

GRPO_DEFAULT = dict(variant="Full", seed=0, steps=3000, M=4, G=8, micro_batch=8, accum=6, workers=12, lr=1e-5, wd=1e-4,
                    warmup=100, warmup_ratio=0.1, clip=1.0, ema_decay=0.99708, ema_ramp=137, lambda_bc=0.1, lambda_cls=0.5,
                    rho=ar1.RHO, ckpt_interval=250, log_interval=10, monitor_interval=50, g3_start=250, g3_window=500,
                    g3_stop_pp=0.05, g3_ep_drop=0.05, bc_target="gt", score_workers=16)


# ---------------------------------------------------------------- 보조
def check_cont_index(src=None, name="ref_progress_m", idx=None):
    """grpo_scorer.CONT(연속 지표 열 이름 목록)에서 name 의 색인이 idx(기본 CONT_REF_PROGRESS)와 같은지 소스 AST 로 검사."""
    idx = CONT_REF_PROGRESS if idx is None else idx
    tree = ast.parse(Path(src or SCORER_SRC).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONT" for t in node.targets):
            cont = ast.literal_eval(node.value)
            if name not in cont or cont.index(name) != idx:
                raise SystemExit(f"grpo_scorer.CONT 에서 {name} 색인 {cont.index(name) if name in cont else None} ≠ {idx} — "
                                 f"train_loop.CONT_REF_PROGRESS 를 고칠 것")
            return cont
    raise SystemExit(f"{src or SCORER_SRC}: CONT 정의를 찾지 못했다")


def acquire_lock(path):
    """비차단 배타 flock. 성공하면 열린 파일 객체(프로세스 수명 동안 유지), 이미 잠겨 있으면 SystemExit."""
    fp = open(path, "a")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fp.close()
        raise SystemExit(f"{path} 가 잠겨 있다 — 같은 work_dir 에서 다른 train_loop 가 실행 중")
    fp.seek(0); fp.truncate(); fp.write(f"pid {os.getpid()} {time.strftime('%F %T')}\n"); fp.flush()
    return fp


def truncate_jsonl(path, keep, backup_tag):
    """이어하기: path(jsonl) 에서 keep(rec) 가 False 인 줄과 깨진 줄을 잘라 path.<backup_tag>.dropped 에 백업. → (남김, 잘라냄) 줄 수."""
    path = Path(path)
    if not path.exists():
        return 0, 0
    kept, dropped = [], []
    for ln in path.read_text().splitlines(keepends=True):
        if not ln.strip():
            continue
        try:
            ok = keep(json.loads(ln))
        except (json.JSONDecodeError, TypeError, KeyError):
            ok = False                      # 쓰는 중 끊긴 줄
        (kept if ok else dropped).append(ln if ln.endswith("\n") else ln + "\n")
    if dropped:
        bk = path.with_name(f"{path.name}.{backup_tag}.dropped")
        with open(bk, "a") as f:
            f.writelines(dropped)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("".join(kept))
        os.replace(tmp, path)
    return len(kept), len(dropped)


def log_line(fp, rec):
    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fp.flush()


def lr_at(step, g):
    """선형 warmup(warmup_ratio → 1) 뒤 cosine(→ 0). step 은 0 부터."""
    if step < g["warmup"]:
        return g["lr"] * (g["warmup_ratio"] + (1 - g["warmup_ratio"]) * step / g["warmup"])
    t = (step - g["warmup"]) / max(1, g["steps"] - g["warmup"])
    return g["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def check_gpu(a):
    dev = os.environ.get("CUDA_VISIBLE_DEVICES")
    if dev is None or dev.strip() == "" or "," in dev:
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={dev!r} — 정확히 한 장")
    if dev == "2":
        raise SystemExit("GPU2 는 Bench2Drive 전용 — 사용 금지")
    g2 = a.smoke or a.measure_sigma_ref > 0 or a.hold_all_graphs
    if dev == "0":
        if not (os.environ.get("HIPAD_ALLOW_GPU0") == "1" and g2):
            raise SystemExit("GPU0(공용)은 HIPAD_ALLOW_GPU0=1 + 스모크/G2 모드 전용")
    elif dev != "3" and os.environ.get("GRPO_ALLOW_GPU") != dev:
        raise SystemExit(f"GPU{dev} — NAVSIM 은 GPU3 전용. 병렬(GPU1 등)은 GRPO_ALLOW_GPU={dev} 로 명시(사용자 승인 사항)")
    return dev


def load_json(p):
    return json.loads(Path(p).read_text())


def resolve_gates(a, g):
    """κ·σ_ref·G0 커널·B0 기준·μ_ref 를 관문 산출물(또는 명시 인자)에서 읽는다. 기본값으로 조용히 넘어가지 않는다."""
    gd = Path(a.gates_dir) if a.gates_dir else None
    out = {}
    if a.kappa is not None:
        out["kappa"] = a.kappa
    else:
        j = load_json(gd / "g1.json")
        if j.get("decision") != "go":
            raise SystemExit(f"G1 결정 {j.get('decision')} — no-go 면 학습하지 않는다")
        out["kappa"] = float(j["kappa"])
    if a.sigma_ref is not None:
        out["sigma_ref"] = a.sigma_ref
    elif a.measure_sigma_ref > 0:
        out["sigma_ref"] = ar1.SIGMA_REF
    else:
        out["sigma_ref"] = float(load_json(gd / "sigma_ref.json")["sigma_ref"])
    kp = a.kernel or (str(gd / "g0_kernel.json") if gd else None)
    out["kernel_path"] = None if a.no_kernel else kp
    out["kernel"] = None if a.no_kernel else AD.load_kernel(kp)
    bp = a.b0_stats or (str(gd / "b0_stats.json") if gd else None)
    out["b0"] = None if a.no_b0 else load_json(bp)
    mp = a.mu_ref or (str(gd / "mu_ref.npz") if gd else None)
    out["mu_ref_path"] = None if a.no_mu_ref else mp
    if g["variant"] in RL_VARIANTS and out["kernel"] is None and not a.smoke and a.measure_sigma_ref <= 0:
        raise SystemExit("본 RL 런은 G0 커널 필요(--no-kernel 은 스모크·G2 전용)")
    if out["b0"] is None and not (a.smoke or a.measure_sigma_ref > 0 or a.hold_all_graphs):
        raise SystemExit("본 런은 B0 기준(b0_stats.json) 필요 — G3 감시")
    return out


class MuRef:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.idx = {str(t): i for i, t in enumerate(z["tokens"])}
        self.mu = z["mu16"]

    def get(self, tok):
        i = self.idx.get(tok)
        if i is None:
            raise KeyError(f"μ_ref 에 {tok} 없음 — μ_ref 덤프는 navtrain RL 프레임 전부여야 한다")
        return self.mu[i].astype(np.float64)


# ---------------------------------------------------------------- 채점 서버
class Server:
    def __init__(self, sock, metric_cache, workers, log_path, timeout=900):
        self.sock = sock
        if len(sock) > 100:
            raise SystemExit(f"소켓 경로가 길다({len(sock)}): {sock}")
        self.client = None
        self.log = open(log_path, "a")
        off = self.log.tell()            # 이번 기동 전 로그 크기 — 이전 기동의 [ready] 줄을 잘못 읽지 않도록 이후 내용만 본다
        cmd = (f'source scripts/navsim/navsim_env.sh && exec "$NAVSIM_PY" tools/navsim/grpo/score_server.py '
               f'--socket {sock} --workers {int(workers)} --metric-cache {metric_cache}')
        self.p = subprocess.Popen(["bash", "-c", cmd], cwd=str(ROOT), stdout=self.log, stderr=subprocess.STDOUT)
        try:
            wait_ready(self.p, log_path, off, f"[ready] socket={sock}", timeout)
        except SystemExit:
            self.close()
            raise
        self.client = ScoreClient(sock)

    def close(self):
        try:
            if getattr(self, "client", None) is not None:
                self.client.shutdown_server()
        except Exception as e:  # 종료 경로: 아래에서 프로세스를 직접 끝낸다(기록)
            print(f"[경고] 채점 서버 shutdown 요청 실패: {e}", flush=True)
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.client = None


def wait_ready(p, log_path, offset, marker, timeout, poll_s=1.0):
    """log_path 의 offset 이후 내용에 marker 가 나올 때까지 기다린다. 그 전에 p 가 끝나면(p.poll()) 또는 시간 초과면 SystemExit."""
    t0 = time.time()
    while True:
        with open(log_path, "rb") as f:
            f.seek(offset)
            txt = f.read().decode(errors="ignore")
        if marker in txt:
            if p.poll() is not None:
                raise SystemExit(f"채점 서버가 준비 표시 뒤 종료(코드 {p.returncode}) — {log_path}")
            return
        if p.poll() is not None:
            raise SystemExit(f"채점 서버 종료(코드 {p.returncode}) — {log_path}")
        if time.time() - t0 > timeout:
            raise SystemExit(f"채점 서버 기동 시간 초과({timeout}s) — {log_path}")
        time.sleep(poll_s)


# ---------------------------------------------------------------- 본체
class Trainer:
    def __init__(self, a, g, gates):
        import torch
        self.torch = torch
        self.a, self.g, self.gates = a, g, gates
        self.dev = "cuda:0"
        self.work = Path(a.work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self._lock = acquire_lock(self.work / ".train.lock")     # 같은 work_dir 동시 실행 금지(GPU1·3 병렬 체인 보호)
        self.variant = g["variant"]
        self.rl = self.variant in RL_VARIANTS
        self.use_cls = self.variant != "C2"
        self.M, self.G, self.B, self.accum = g["M"], g["G"], g["micro_batch"], g["accum"]
        self.sig = ar1.sigma_table(gates["kappa"])
        self.sigma_ref = gates["sigma_ref"]
        self.axis = "ego" if self.variant == "R" else "lane"
        if a.max_gpu_mib > 0:
            tot = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gpu_mib * 2 ** 20 / tot), 0)
        cfg = RO.load_model_cfg(a.model_config, bc_target=g["bc_target"])
        if a.no_isolation:
            cfg.model.head.onedecoder_head.plan_grad_isolation = False
        self.cfg = cfg
        from mmdet.apis import set_random_seed
        set_random_seed(g["seed"], deterministic=False)
        self.det = RO.build_rl_model(cfg, a.init_ckpt, self.dev)
        self.dec = self.det.head.onedecoder_head
        if self.dec.num_stream_groups != self.accum:
            raise SystemExit(f"누적 {self.accum} ≠ 디코더 스트림 그룹 {self.dec.num_stream_groups}(§9.2: 같아야 step = 그룹 6 개 × 1 프레임)")
        self.train_names = RO.select_trainable(self.det)
        pm = dict(self.det.named_parameters())
        self.params = [pm[n] for n in self.train_names]
        self.opt = torch.optim.AdamW(self.params, lr=g["lr"], weight_decay=g["wd"])
        self.ema = [p.detach().clone().float() for p in self.params]
        self.n_ema = 0
        self.ds = RO.build_rl_dataset(cfg, ann_file=a.ann_file, bc_target=g["bc_target"], select=a.scene_select)
        self.table = RO.frame_table(self.ds.data_infos)
        self.cache_toks = RO.metric_cache_tokens(a.metric_cache)
        from extract_pdm_centerlines import CenterlineStore      # 모든 변형: KL 감시 기저를 차선축으로 통일(C1·C2·R 포함)
        self.store = CenterlineStore(a.centerlines)
        self.mu_ref = MuRef(gates["mu_ref_path"]) if gates["mu_ref_path"] else None
        self.step = 0
        self.mon = []          # [(step, stop, ep, speed, kl)]
        self.stats = dict(dup_tokens=0)
        if (self.work / "latest.pth").exists() and not a.no_resume:
            self._resume()
        self.loader = RO.build_rl_loader(self.ds, self.B, self.accum, g["workers"], g["seed"],
                                         fast_forward=self.step * self.accum)
        if getattr(self, "_np_state", None) is not None:
            np.random.set_state(self._np_state)
        self.it = iter(self.loader)
        # 이어하기: 체크포인트 step 이후 기록을 잘라낸다(백업). log.jsonl 의 step = 그 step 시작 시 번호(끝나지 않은 step 은 < self.step 이
        #   아니다), monitor.jsonl 의 step = 끝난 step 수(체크포인트 저장보다 먼저 써지므로 ≤ self.step 은 체크포인트에 반영됨).
        #   처음 시작(self.step = 0)이면 이전 기록 전부를 백업으로 옮긴다.
        tag = time.strftime("%m%d_%H%M%S")
        for name, keep in (("log.jsonl", lambda r: r["step"] < self.step), ("monitor.jsonl", lambda r: r["step"] <= self.step)):
            nk, nd = truncate_jsonl(self.work / name, keep, f"resume{self.step}_{tag}")
            if nd:
                print(f"[이어하기] {name}: {nd} 줄 잘라 백업(남김 {nk})", flush=True)
        self.log = open(self.work / "log.jsonl", "a")
        self.monlog = open(self.work / "monitor.jsonl", "a")
        self.mem_peak = 0.0

    # ---------------- 체크포인트
    def _ckpt_obj(self):
        torch = self.torch
        return dict(step=self.step, variant=self.variant, grpo=self.g, gates={k: v for k, v in self.gates.items()
                                                                               if k in ("kappa", "sigma_ref", "kernel_path", "mu_ref_path")},
                    trainable={n: p.detach().cpu() for n, p in zip(self.train_names, self.params)},
                    opt=self.opt.state_dict(), ema=[e.cpu() for e in self.ema], n_ema=self.n_ema,
                    stream_states=self.dec._stream_states, run_step=self.dec.run_step,
                    rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(), numpy=np.random.get_state(),
                             py=random.getstate()),
                    mon=self.mon, stats=self.stats, init_ckpt=os.path.abspath(self.a.init_ckpt),
                    model_config=os.path.abspath(self.a.model_config))

    def save(self, name="latest.pth"):
        p = self.work / name
        self.torch.save(self._ckpt_obj(), str(p) + ".tmp")
        os.replace(str(p) + ".tmp", p)

    def _resume(self):
        torch = self.torch
        st = torch.load(self.work / "latest.pth", map_location="cpu", weights_only=False)
        if st["variant"] != self.variant or st["grpo"] != self.g:
            raise SystemExit("latest.pth 의 변형·설정이 지금과 다르다 — 다른 work_dir 를 쓸 것")
        if st["init_ckpt"] != os.path.abspath(self.a.init_ckpt):
            raise SystemExit(f"시작 체크포인트가 다르다: {st['init_ckpt']}")
        pm = dict(self.det.named_parameters())
        with torch.no_grad():
            for n, v in st["trainable"].items():
                pm[n].copy_(v)
        self.opt.load_state_dict(st["opt"])
        self.ema = [e.to(self.dev) for e in st["ema"]]
        self.n_ema = st["n_ema"]
        def to_dev(o):                                  # 뱅크 상태는 dict·list·tuple 안에 텐서가 섞여 있다 — 재귀로 옮긴다
            if isinstance(o, torch.Tensor):
                return o.to(self.dev)
            if isinstance(o, dict):
                return {k: to_dev(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return type(o)(to_dev(v) for v in o)
            return o
        self.dec._stream_states = to_dev(st["stream_states"])
        self.dec.run_step = st["run_step"]
        torch.set_rng_state(st["rng"]["torch"])
        torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        random.setstate(st["rng"]["py"])
        self._np_state = st["rng"]["numpy"]
        self.mon, self.stats, self.step = st["mon"], st["stats"], st["step"]
        print(f"[재개] step {self.step}", flush=True)

    def write_mem_summary(self):
        """G2-7 판정 입력: micro-batch 최대 메모리(MiB)와 기준 분모(상한 --max-gpu-mib 가 있으면 그것, 없으면 장치 전체)."""
        tot = self.torch.cuda.get_device_properties(0).total_memory / 2 ** 20
        denom = min(tot, self.a.max_gpu_mib) if self.a.max_gpu_mib > 0 else tot
        rec = dict(peak_mib=self.mem_peak, device_total_mib=tot, cap_mib=self.a.max_gpu_mib or None, denom_mib=denom,
                   frac=self.mem_peak / denom, isolation=not self.a.no_isolation, hold_all_graphs=bool(self.a.hold_all_graphs),
                   steps=self.step)
        (self.work / "mem_summary.json").write_text(json.dumps(rec, indent=1))

    def final_ema(self):
        torch = self.torch
        sd = {k: v.detach().cpu().clone() for k, v in self.det.state_dict().items()}
        for n, e in zip(self.train_names, self.ema):
            sd[n] = e.cpu().to(sd[n].dtype)
        meta = dict(grpo=self.g, variant=self.variant, step=self.step, n_ema=self.n_ema, ema_decay=self.g["ema_decay"],
                    ema_ramp=self.g["ema_ramp"], init_ckpt=os.path.abspath(self.a.init_ckpt), gates=self._ckpt_obj()["gates"])
        torch.save(dict(meta=meta, state_dict=sd), str(self.work / "final_ema.pth") + ".tmp")
        os.replace(str(self.work / "final_ema.pth") + ".tmp", self.work / "final_ema.pth")
        sd2 = {k: v.detach().cpu().clone() for k, v in self.det.state_dict().items()}
        torch.save(dict(meta=meta, state_dict=sd2), str(self.work / "final.pth") + ".tmp")
        os.replace(str(self.work / "final.pth") + ".tmp", self.work / "final.pth")

    # ---------------- micro-batch
    def _frames(self, batch):
        metas = batch["img_metas"].data[0]
        out = []
        for b, m in enumerate(metas):
            nav, is_eval, scene, _ = self.table[m["token"]]
            rl = nav is not None and nav in self.cache_toks
            out.append((b, nav if rl else None))
        return out

    def forward_mb(self, k, batch, frames, N):
        torch = self.torch
        data = RO.to_device(batch, self.dev)
        outs, data = RO.forward_heads(self.det, data, grad=True)
        raw16, mu16, _ = RO.plan_bundle(self.det, outs, data)
        resc = RO.rescore16(self.det, outs, raw16, mu16)
        L_bc = None
        if self.g["lambda_bc"] > 0:
            lp = self.dec.loss_plan(outs[0], outs[1], outs[4], outs[3], data)
            L_bc = sum(lp.values())
        mu_np = mu16.detach().double().cpu().numpy()
        raw_np = raw16.detach().double().cpu().numpy()
        resc_np = resc.detach().double().cpu().numpy()
        sub, fr, seen = {}, [], set()
        for b, nav in frames:
            if nav is None:
                continue
            if nav in seen:
                self.stats["dup_tokens"] += 1          # 같은 micro-batch 안 같은 장면(드묾): 두 번째는 BC 만(기록)
                continue
            seen.add(nav)
            top = RO.top_modes(resc_np[b], raw_np[b], self.M)
            if self.rl:
                cl = self.store.get(nav) if self.axis == "lane" else None
                cands, E16, eta, info = RO.policy_candidates(
                    mu_np[b], top, nav, cl, None if cl is None else self.store.s_grid, self.sig, self.g["seed"], self.step,
                    self.G, axis=self.axis)
            else:
                cands, E16, eta, info = mu_np[b], None, None, None
            sub[nav] = cands
            fr.append(dict(b=b, tok=nav, top=top, E16=E16, eta=eta, oor=0 if info is None else int(info["out_of_range"].sum())))
        if sub:
            self.server.client.submit((self.step, k), sub)
        # 계측(G2-7)
        mem = torch.cuda.max_memory_allocated() / 2 ** 20
        return dict(k=k, mu16=mu16, raw16=raw16, L_bc=L_bc, mu_np=mu_np, raw_np=raw_np, resc_np=resc_np, frames=fr,
                    N=N, submitted=bool(sub), mem_peak_mib=mem)

    def finish_mb(self, pm, res, record):
        """점수 → 손실 → backward(누적). record=True 면 항별 기울기 노름 계측. 반환 기록 dict."""
        torch = self.torch
        B, accum, M, G = self.B, self.accum, self.M, self.G
        rec = dict(n_rl=len(pm["frames"]), stage=[0, 0, 0, 0], cls_valid=0, lg_n=0, oor=0, mon=[])
        mu_l, E_l, eta_l, A_l = [], [], [], []
        lg = mu16 = None
        mu16 = pm["mu16"]
        lg_terms, cls_terms = [], []
        gmu_axis, gmu_innov = np.zeros(2), np.zeros(8)
        for f in pm["frames"]:
            out = res[f["tok"]]
            b, top = f["b"], f["top"]
            resc = pm["resc_np"][b]
            sel = int(np.argmax(resc))
            mu_b = pm["mu_np"][b]
            ep = float(out["subscores"][sel, SUB_EP])
            endd = float(np.linalg.norm(mu_b[sel, -1]))
            stop = endd < 2.0 and float(out["cont"][sel, CONT_REF_PROGRESS]) > 5.0
            kl = float("nan")
            if self.mu_ref is not None:
                if f["E16"] is not None and self.axis == "lane":
                    Esel = f["E16"][sel]
                else:                                            # R·C1·C2: 감시 KL 은 차선축 기저(런 사이 비교 가능)
                    cl = self.store.get(f["tok"])
                    Esel = ar1.axis_basis(mu_b[sel], cl["xyh"][:, :2].astype(np.float64), s_grid=self.store.s_grid,
                                          valid=cl["valid"])[0]
                kl = float(ar1.kl_np(mu_b[sel], self.mu_ref.get(f["tok"])[sel], Esel, self.sig))
            rec["mon"].append((self.step, bool(stop), ep, endd / 4.0, kl, float(out["epdms"][sel])))
            rec["oor"] += f["oor"]
            tgt, valid = None, False
            if self.rl:
                sc = AD.build_scene(f["tok"], out, M=M, G=G, mu16=mu_b, top_modes=top)
                r = AD.scene_advantage(sc, self.variant, self.gates["kernel"], rescored_logits=resc)
                for s in r["stage"]:
                    rec["stage"][int(s) - 1] += 1
                A = r["A"]
                if np.any(A != 0):
                    mu_l.append(mu16[b, torch.as_tensor(top, device=mu16.device)].repeat_interleave(G, 0))
                    E_l.append(np.repeat(f["E16"][top], G, 0))
                    eta_l.append(f["eta"].reshape(M * G, 8, 2))
                    A_l.append(A.reshape(M * G, 8, 2))
                    if record:
                        Er = np.repeat(f["E16"][top], G, 0)
                        Ar = A.reshape(M * G, 8, 2)
                        et = f["eta"].reshape(M * G, 8, 2)
                        for ax in range(2):
                            A1 = np.zeros_like(Ar); A1[..., ax] = Ar[..., ax]
                            gmu_axis[ax] += (ar1.grad_mu_np(Er, self.sig, et, A1, self.sigma_ref, pm["N"]) ** 2).sum()
                        for kk in range(8):
                            A1 = np.zeros_like(Ar); A1[:, kk] = Ar[:, kk]
                            gmu_innov[kk] += (ar1.grad_mu_np(Er, self.sig, et, A1, self.sigma_ref, pm["N"]) ** 2).sum()
                Lg = r["Lg"]
                if Lg is not None and Lg["q_star"] >= 0 and Lg["weight"] > 0:
                    tau = torch.as_tensor(Lg["tau_star"], dtype=mu16.dtype, device=mu16.device)
                    lg_terms.append(Lg["weight"] * (mu16[b, Lg["q_star"]] - tau).abs().sum() / 8.0)
                    rec["lg_n"] += 1
                tgt, valid = r["cls_target"], r["cls_valid"]
            elif self.use_cls:
                tgt, valid, _ = AD.cls_target(np.asarray(out["epdms"])[:16], rescored_logits=resc)
            if self.use_cls and valid:
                t = torch.as_tensor(tgt, dtype=pm["raw16"].dtype, device=mu16.device)
                cls_terms.append(-(t * torch.log_softmax(pm["raw16"][b], -1)).sum())
                rec["cls_valid"] += 1
        terms = {}
        if mu_l:
            terms["rl"] = ar1.loss_rl_torch(torch.cat(mu_l), np.concatenate(E_l), self.sig, np.concatenate(eta_l),
                                            np.concatenate(A_l), sigma_ref=self.sigma_ref, N=pm["N"], rho=self.g["rho"])
        if lg_terms:
            terms["g"] = torch.stack(lg_terms).sum() / (B * accum)
        if pm["L_bc"] is not None:
            terms["bc"] = self.g["lambda_bc"] * pm["L_bc"] / accum
        if cls_terms:
            terms["cls"] = self.g["lambda_cls"] * torch.stack(cls_terms).sum() / (B * accum)
        rec["loss"] = {k: float(v.detach()) for k, v in terms.items()}
        if record:
            rec["gmu_axis_sq"], rec["gmu_innov_sq"] = gmu_axis.tolist(), gmu_innov.tolist()
        return terms, rec

    def backward(self, terms, acc=None):
        """acc 가 dict 면 항별 기울기를 따로 모은다(계측 step·σ_ref 측정). 아니면 합을 한 번에 backward."""
        torch = self.torch
        live = {k: v for k, v in terms.items() if v.requires_grad}
        if not live:
            return
        if acc is None:
            sum(live.values()).backward()
            return
        names = list(live)
        for i, k in enumerate(names):
            gs = torch.autograd.grad(live[k], self.params, retain_graph=i < len(names) - 1, allow_unused=True)
            buf = acc.setdefault(k, [torch.zeros_like(p) for p in self.params])
            for j, gi in enumerate(gs):
                if gi is not None:
                    buf[j] += gi
                    if not self.a.measure_sigma_ref:
                        if self.params[j].grad is None:
                            self.params[j].grad = gi.clone()
                        else:
                            self.params[j].grad += gi

    # ---------------- step
    def run_step(self, record):
        torch = self.torch
        t0 = time.time()
        batches = [next(self.it) for _ in range(self.accum)]
        frames = [self._frames(bt) for bt in batches]
        n_rl = 0
        for fr in frames:
            seen = set()
            for _, nav in fr:
                if nav is not None and nav not in seen:
                    seen.add(nav); n_rl += 1
        N = max(1, n_rl * self.M * self.G)
        for p in self.params:
            p.grad = None
        pend, recs, acc = {}, [], ({} if record else None)
        t_wait = 0.0
        mems = []

        def drain(block, want=None):
            nonlocal t_wait
            tw = time.time()
            got = self.server.client.wait(want) if want is not None else dict(self.server.client.poll(block=block))
            t_wait += time.time() - tw
            for mb_id, res in got.items():
                k = mb_id[1]
                pm = pend.pop(k)
                terms, rec = self.finish_mb(pm, res, record)
                self.backward(terms, acc)
                recs.append(rec)
                del pm

        for k, (bt, fr) in enumerate(zip(batches, frames)):
            torch.cuda.reset_peak_memory_stats()
            pm = self.forward_mb(k, bt, fr, N)
            mems.append(pm["mem_peak_mib"])
            if not pm["submitted"]:                      # RL 프레임 없음 → BC 만
                terms = {"bc": self.g["lambda_bc"] * pm["L_bc"] / self.accum} if pm["L_bc"] is not None else {}
                self.backward(terms, acc)
                recs.append(dict(n_rl=0, stage=[0, 0, 0, 0], cls_valid=0, lg_n=0, oor=0, mon=[],
                                 loss={k2: float(v.detach()) for k2, v in terms.items()}))
                continue
            pend[k] = pm
            del pm
            if not self.a.hold_all_graphs:
                drain(False)
        if pend:
            drain(True, want=[(self.step, k) for k in sorted(pend)])
        peak_all = torch.cuda.max_memory_allocated() / 2 ** 20
        out = dict(step=self.step, n_rl=n_rl, N=N, t_step=time.time() - t0, t_wait_score=t_wait, mem_mb_peak_mib=mems,
                   mem_peak_after_mib=peak_all)
        for key in ("rl", "g", "bc", "cls"):
            out[f"loss_{key}"] = float(sum(r["loss"].get(key, 0.0) for r in recs))
        st = np.sum([r["stage"] for r in recs], 0)
        out["stage_counts"] = st.tolist()
        out["cls_valid"] = int(sum(r["cls_valid"] for r in recs))
        out["lg_n"] = int(sum(r["lg_n"] for r in recs))
        out["basis_oor"] = int(sum(r["oor"] for r in recs))
        for r in recs:
            self.mon.extend(r["mon"])
        if record:
            out["gnorm_term"] = {k: float(torch.sqrt(sum((x.float() ** 2).sum() for x in v))) for k, v in acc.items()}
            out["gmu_axis"] = np.sqrt(np.sum([r.get("gmu_axis_sq", [0, 0]) for r in recs], 0)).tolist()
            out["gmu_innov"] = np.sqrt(np.sum([r.get("gmu_innov_sq", [0] * 8) for r in recs], 0)).tolist()
        return out, acc

    def optimizer_step(self, out):
        torch = self.torch
        tot = torch.nn.utils.clip_grad_norm_(self.params, self.g["clip"])
        out["grad_norm"] = float(tot)
        out["clipped"] = bool(float(tot) > self.g["clip"])
        if not math.isfinite(float(tot)):
            raise RuntimeError(f"step {self.step}: 기울기 노름 비유한({tot}) — 원인 조사 필요(건너뛰지 않음)")
        lr = lr_at(self.step, self.g)
        for gr in self.opt.param_groups:
            gr["lr"] = lr
        out["lr"] = lr
        self.opt.step()
        self.n_ema += 1
        d = self.g["ema_decay"] * (1.0 - math.exp(-self.n_ema / self.g["ema_ramp"]))
        with torch.no_grad():
            for e, p in zip(self.ema, self.params):
                e.mul_(d).add_(p.detach().float(), alpha=1 - d)

    def g3(self):
        """→ (판정 dict, 중단 여부)."""
        s = self.step                       # 끝난 step 수(호출 시점에 이미 1 증가)
        lo = 0 if s < self.g["g3_window"] else s - self.g["g3_window"]
        w = [m for m in self.mon if m[0] >= lo]
        if not w:
            return dict(step=s, n=0), False
        stop = float(np.mean([m[1] for m in w]))
        ep = float(np.mean([m[2] for m in w]))
        sp = float(np.mean([m[3] for m in w]))
        kls = [m[4] for m in w if np.isfinite(m[4])]
        rec = dict(step=s, window_from=lo, n=len(w), stop_rate=stop, ep_mean=ep, speed_mean=sp,
                   epdms_mean=float(np.mean([m[5] for m in w])), kl_mean=float(np.mean(kls)) if kls else None)
        halt = False
        b0 = self.gates["b0"]
        if b0 is not None:
            rec["d_stop"] = stop - b0["stop_rate"]
            rec["d_ep"] = ep - b0["ep_mean"]
            if s >= self.g["g3_start"] and (rec["d_stop"] > self.g["g3_stop_pp"] or rec["d_ep"] < -self.g["g3_ep_drop"]):
                halt = True
        rec["halt"] = halt
        return rec, halt

    # ---------------- 실행
    def train(self):
        a, g = self.a, self.g
        steps = a.max_steps if a.max_steps else g["steps"]
        if a.measure_sigma_ref > 0:
            return self.measure_sigma(a.measure_sigma_ref)
        while self.step < steps:
            record = (self.step % g["monitor_interval"] == 0)
            out, _ = self.run_step(record)
            self.mem_peak = max(self.mem_peak, max(out["mem_mb_peak_mib"]))
            self.optimizer_step(out)
            if self.step % g["log_interval"] == 0 or record or a.smoke or a.hold_all_graphs:
                log_line(self.log, out)
                print(f"[step {self.step}] rl {out['loss_rl']:.4g} g {out['loss_g']:.4g} bc {out['loss_bc']:.4g} "
                      f"cls {out['loss_cls']:.4g} |g| {out['grad_norm']:.3g} n_rl {out['n_rl']} "
                      f"{out['t_step']:.1f}s (채점 대기 {out['t_wait_score']:.1f}s) mem {max(out['mem_mb_peak_mib']):.0f}MiB",
                      flush=True)
            self.step += 1
            if self.step % g["monitor_interval"] == 0:
                rec, halt = self.g3()
                log_line(self.monlog, rec)
                if halt:
                    self.save()
                    self.final_ema()
                    (self.work / "G3_STOP.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
                    print(f"[G3 중단] {rec}", flush=True)
                    return 3
            if self.step % g["ckpt_interval"] == 0:
                self.save()
        self.save()
        self.final_ema()
        self.write_mem_summary()
        if not a.smoke and not a.max_steps:
            (self.work / "DONE").write_text(json.dumps(dict(step=self.step, stats=self.stats)))
        return 0

    def measure_sigma(self, K):
        """G2-5: 갱신 없이 K step, γ = ‖g_RL‖ / ‖g_BC + g_cls‖(clip 전, 학습 대상 전체)."""
        torch = self.torch
        gam, rows = [], []
        for _ in range(self.a.sigma_warmup):            # 스트림 앞머리(이력 전용 프레임) 지나기: 순전파·뱅크 갱신만, 갱신·측정 없음
            out, _ = self.run_step(False)
            for p in self.params:
                p.grad = None
            print(f"[σ_ref 예열 step {self.step}] n_rl {out['n_rl']}", flush=True)
            self.step += 1
        for _ in range(K):
            out, acc = self.run_step(True)
            if out["n_rl"] == 0:
                raise RuntimeError(f"σ_ref 측정 step {self.step} 에 RL 장면이 없다 — --sigma-warmup 을 늘릴 것(0 을 중앙값에 넣지 않음)")
            def nrm(keys):
                vs = [acc[k] for k in keys if k in acc]
                if not vs:
                    return 0.0
                sq = 0.0
                for j in range(len(self.params)):
                    sq += float((sum(v[j] for v in vs).float() ** 2).sum())
                return math.sqrt(sq)
            g_rl = nrm(["rl"])
            g_bc = nrm(["bc", "cls"])
            gm = g_rl / g_bc if g_bc > 0 else float("nan")
            gam.append(gm)
            out.update(g_rl=g_rl, g_bccls=g_bc, gamma=gm)
            rows.append(out)
            log_line(self.log, out)
            print(f"[σ_ref 측정 step {self.step}] ‖g_RL‖ {g_rl:.4g} ‖g_BC+cls‖ {g_bc:.4g} γ {gm:.4g}", flush=True)
            self.step += 1
        med = float(np.nanmedian(gam))
        if 0.5 <= med <= 2.0:
            s2 = self.sigma_ref ** 2
            rule = "keep"
        else:
            s2 = self.sigma_ref ** 2 * 2.0 ** round(math.log2(1.0 / med))
            rule = f"scale 2^{round(math.log2(1.0 / med))}"
        res = dict(variant=self.variant, steps=K, warmup=self.a.sigma_warmup, gammas=gam, gamma_median=med, sigma_ref_in=self.sigma_ref,
                   sigma_ref=math.sqrt(s2), rule=rule, kappa=self.gates["kappa"], init_ckpt=os.path.abspath(self.a.init_ckpt),
                   note="G2-5 규칙(§2.3 성질 3): B1 기준 값만 σ_ref 로 쓴다. Full 은 기록용")
        (self.work / f"sigma_ref_{self.variant}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        if self.variant == "B1":
            (self.work / "sigma_ref.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"[σ_ref] γ 중앙값 {med:.4g} → σ_ref {res['sigma_ref']:.4g} ({rule})", flush=True)
        return 0


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("grpo_config", help="configs/navsim/v2_teacher/grpo/<run>.py (grpo dict 만)")
    ap.add_argument("--model-config", required=True, help="모델 config(MODEL_CFG, 2단계 계열)")
    ap.add_argument("--init-ckpt", required=True, help="시작 체크포인트(GRPO_INIT_CKPT, 사용자 결정 11-06: MCMC SFT seed 0)")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--gates-dir", default=None, help="g1.json, sigma_ref.json, g0_kernel.json, b0_stats.json, mu_ref.npz")
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--sigma-ref", type=float, default=None)
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--b0-stats", default=None)
    ap.add_argument("--mu-ref", default=None)
    ap.add_argument("--no-kernel", action="store_true")
    ap.add_argument("--no-b0", action="store_true")
    ap.add_argument("--no-mu-ref", action="store_true")
    ap.add_argument("--ann-file", default=None)
    ap.add_argument("--scene-select", default="all")
    ap.add_argument("--metric-cache", default=RO.DEFAULT_METRIC_CACHE)
    ap.add_argument("--centerlines", default=RO.DEFAULT_CENTERLINES)
    ap.add_argument("--score-workers", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-gpu-mib", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--measure-sigma-ref", type=int, default=0)
    ap.add_argument("--sigma-warmup", type=int, default=5, help="σ_ref 측정 전 예열 step(측정에서 제외)")
    ap.add_argument("--hold-all-graphs", action="store_true")
    ap.add_argument("--no-isolation", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="grpo 키=값 덮어쓰기(스모크 전용: micro_batch, accum, workers 등)")
    return ap.parse_args()


def load_grpo(path, sets, smoke):
    from mmcv import Config
    c = Config.fromfile(path)
    g = dict(GRPO_DEFAULT)
    unknown = set(c.grpo) - set(g)
    if unknown:
        raise SystemExit(f"알 수 없는 grpo 키 {unknown}")
    g.update(c.grpo)
    for s in sets:
        if not smoke:
            raise SystemExit("--set 은 스모크·G2 전용(본 런 설정은 config 로만)")
        k, v = s.split("=", 1)
        if k not in g:
            raise SystemExit(f"--set 키 {k}")
        g[k] = type(g[k])(v) if not isinstance(g[k], bool) else v in ("1", "true", "True")
    if g["variant"] not in ALL_VARIANTS:
        raise SystemExit(f"variant {g['variant']}")
    return g


def main():
    a = parse()
    check_gpu(a)
    if a.smoke and a.max_steps == 0:
        a.max_steps = 2
    if a.smoke and a.max_steps > 20:
        raise SystemExit("스모크는 20 step 이하")
    os.chdir(ROOT)
    g = load_grpo(a.grpo_config, a.set, a.smoke or a.measure_sigma_ref > 0 or a.hold_all_graphs)
    if a.measure_sigma_ref > 0 and g["variant"] not in ("B1", "Full"):
        raise SystemExit("σ_ref 측정은 B1(규칙)·Full(기록)만")
    check_cont_index()
    gates = resolve_gates(a, g)
    tr = Trainer(a, g, gates)
    work_rel = os.path.relpath(tr.work, ROOT)
    sock = os.path.join(work_rel, "score.sock")
    tr.server = Server(sock, a.metric_cache, a.score_workers or g["score_workers"], str(tr.work / f"score_server_gpu{os.environ.get('CUDA_VISIBLE_DEVICES')}.log"))

    def _term(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, _term)
    try:
        code = tr.train()
    finally:
        tr.server.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
