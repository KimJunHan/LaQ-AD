# 수정(2026-10-19): 사용자 결정 — GRPO 대신 Hydra-MDP 방식 계획 점수 학습(LaQ v2 2단계 뒤 적용). 새 파일(기존 파일 수정 없음).
"""Hydra-MDP 방식 계획 점수기 학습(고정된 2단계 모델의 덤프 위에서, train 분할만 사용).

입력(모드마다): 마지막 층 계획 쿼리 특징(256) + 모드 궤적(12, 표준화) + 운전 명령 + 원 모델 신호 2 개
  (rescore 전 logit, 예측 운동과의 충돌로 rescore 되어 확률 0 이 된 모드 표시 — 둘 다 추론 시 모델이 내는 값이라 GT 누설 없음).
  표본 안 모드 평균을 문맥으로 덧붙인다(모드끼리 비교가 필요한 모방 softmax 용).
출력(모드마다): imitation logit(16 모드 softmax), no-collision·drivable·progress logit(시그모이드).
손실(Hydra-MDP 식 그대로): CE(soft target softmax(-L2/τ), softmax(imi)) + Σ_k BCE(σ(logit_k), y_k), k ∈ {nc, dac, ep}.
선택(Hydra-MDP 식): score = w_im·log softmax(imi) + w_nc·log σ(nc) + w_dac·log σ(dac) + w_ep·log σ(ep) 의 argmax.
  가중치는 **train 분할 안의 보류 장면(held-out, 기본 10 %, 장면 단위)** 에서만 고른다(val 은 쓰지 않음):
  수정(2026-10-19, 사용자 결정 1): 기준 = 보류 장면에서 평가기 규칙의 L2 avg 최소(충돌 조건 없음; 후보별 충돌률은 계산해 보고만).
  수정(2026-10-19, 사용자 결정 3): 가중치를 고른 뒤 train 전체(100 % 장면)로 같은 설정·에폭 재학습한 모델이 최종 scorer.pt
  (Hydra-MDP 와 같이 전체 학습 집합으로 학습). 90 % 모델은 scorer_split90.pt 로 남긴다.
미래 GT 가 불완전한 표본(평가기 제외 규칙)은 학습·선택 모두에서 제외하고 수를 기록한다.
결정성: 고정 시드, torch.use_deterministic_algorithms(True), CPU 스레드 수 고정, fp32.

사용: PYTHONPATH=. OPENBLAS_CORETYPE=SkylakeX python tools/plan_scorer/train_scorer.py \
        --dump <train_dump_dir> --subscores <train_subscores.npz> --out <dir> [--epochs 30]
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import scorer_io  # noqa: E402
from subscores import evaluator_metrics  # noqa: E402

HEADS = ("im", "nc", "dac", "ep")


# --------------------------------------------------------------------------------------------- 모델
class PlanScorer(nn.Module):
    def __init__(self, feat_dim=256, traj_dim=12, hidden=256, num_cmd=3, aux_dim=2):
        super().__init__()
        self.feat_proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, hidden))
        self.traj_enc = nn.Sequential(nn.Linear(traj_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.cmd_emb = nn.Embedding(num_cmd, hidden)
        self.aux_proj = nn.Linear(aux_dim, hidden)
        self.ctx = nn.Linear(hidden, hidden)
        self.trunk = nn.Sequential(nn.ReLU(), nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, len(HEADS))

    def forward(self, feat, traj, cmd, aux):
        h = self.feat_proj(feat) + self.traj_enc(traj) + self.cmd_emb(cmd)[:, None] + self.aux_proj(aux)
        c = self.ctx(h.mean(1, keepdim=True)).expand_as(h)
        o = self.head(self.trunk(torch.cat([h, c], -1)))
        return {k: o[..., j] for j, k in enumerate(HEADS)}


def build_inputs(dump, norm=None):
    """덤프 → 모델 입력 텐서(fp32). norm 이 None 이면 이 자료로 표준화 통계를 만든다(학습용 부분에서만 호출)."""
    traj = dump["modes"].reshape(len(dump["modes"]), dump["modes"].shape[1], -1).astype(np.float64)
    raw = dump["raw_logits"].astype(np.float64)
    if norm is None:
        norm = {"traj_mean": traj.reshape(-1, traj.shape[-1]).mean(0).tolist(),
                "traj_std": (traj.reshape(-1, traj.shape[-1]).std(0) + 1e-6).tolist(),
                "raw_mean": float(raw.mean()), "raw_std": float(raw.std() + 1e-6)}
    t = (traj - np.asarray(norm["traj_mean"])) / np.asarray(norm["traj_std"])
    aux = np.stack([(raw - norm["raw_mean"]) / norm["raw_std"], (dump["probs"] <= 0).astype(np.float64)], -1)
    X = {"feat": torch.as_tensor(dump["feats"], dtype=torch.float32),
         "traj": torch.as_tensor(t, dtype=torch.float32),
         "cmd": torch.as_tensor(dump["cmd"], dtype=torch.long),
         "aux": torch.as_tensor(aux, dtype=torch.float32)}
    return X, norm


def take(d, m):
    return {k: (v[m] if isinstance(v, (np.ndarray, torch.Tensor)) else [v[i] for i in np.flatnonzero(m)])
            for k, v in d.items() if k != "manifest"}


def loss_fn(out, y):
    l_im = -(y["im"] * F.log_softmax(out["im"], -1)).sum(-1).mean()
    parts = {"im": l_im}
    for k in ("nc", "dac", "ep"):
        parts[k] = F.binary_cross_entropy_with_logits(out[k], y[k])
    return sum(parts.values()), parts


def selection_score(out, w):
    s = w["im"] * F.log_softmax(out["im"], -1)
    for k in ("nc", "dac", "ep"):
        s = s + w[k] * F.logsigmoid(out[k])
    return s


@torch.no_grad()
def predict(model, X, bs=4096):
    model.eval()
    outs = {k: [] for k in HEADS}
    n = len(X["cmd"])
    for s in range(0, n, bs):
        o = model(X["feat"][s:s + bs], X["traj"][s:s + bs], X["cmd"][s:s + bs], X["aux"][s:s + bs])
        for k in HEADS:
            outs[k].append(o[k])
    return {k: torch.cat(v, 0) for k, v in outs.items()}


def set_determinism(seed, threads):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def train_model(X, Y, epochs, lr, wd, bs, seed, hidden=256, log=None, X_val=None, Y_val=None):
    """fp32 결정적 학습. 반환: 모델, 에폭별 기록."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = PlanScorer(feat_dim=X["feat"].shape[-1], traj_dim=X["traj"].shape[-1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(X["cmd"])
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.05,
                                                anneal_strategy="cos")
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = {k: 0.0 for k in ("loss",) + HEADS}; nb = 0
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            out = model(X["feat"][b], X["traj"][b], X["cmd"][b], X["aux"][b])
            loss, parts = loss_fn(out, {k: Y[k][b] for k in HEADS})
            if not torch.isfinite(loss):
                raise FloatingPointError(f"에폭 {ep}: 손실 비유한 — 입력/목표 점검(건너뛰지 않음)")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot["loss"] += float(loss); nb += 1
            for k in HEADS:
                tot[k] += float(parts[k])
        rec = {"epoch": ep + 1, **{f"train_{k}": v / nb for k, v in tot.items()}}
        if X_val is not None:
            o = predict(model, X_val)
            vl, vp = loss_fn(o, Y_val)
            rec.update({"heldout_loss": float(vl), **{f"heldout_{k}": float(v) for k, v in vp.items()}})
        hist.append(rec)
        if log:
            log(json.dumps(rec))
    return model, hist


def targets(sc):
    return {k: torch.as_tensor(np.asarray(sc[k], np.float32)) for k in HEADS}


def metrics_for_selection(sc, sel):
    r = np.arange(len(sel))
    return evaluator_metrics(sc["l2_step"][r, sel], sc["col_step"][r, sel])


def weight_grid(values_im=(1.0,), values=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)):
    for wi, wn, wd, we in itertools.product(values_im, values, values, values):
        yield {"im": wi, "nc": wn, "dac": wd, "ep": we}


def choose_weights(out, sc, base_sel):
    """수정(2026-10-19, 사용자 결정 1): 보류 장면(train 의 10 %)에서 L2 avg 가 최소인 가중치 조합을 고른다(충돌 조건 없음).
    같은 L2 면 충돌이 작은 것. 후보마다 충돌률 avg 도 계산해 보고만 한다(선택에는 쓰지 않음)."""
    base = metrics_for_selection(sc, base_sel)
    rows = []
    for w in weight_grid():
        sel = selection_score(out, w).argmax(-1).numpy()
        m = metrics_for_selection(sc, sel)
        rows.append({"w": w, "L2_avg": m["L2"]["avg"], "col_avg": m["obj_box_col"]["avg"], "metrics": m})
    best = min(rows, key=lambda r: (r["L2_avg"], r["col_avg"]))
    return best, base, rows


def scene_holdout(scene_tokens, frac, seed):
    scenes = np.array(sorted(set(scene_tokens.tolist())))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(scenes))
    n_h = max(1, int(round(frac * len(scenes))))
    held = set(scenes[perm[:n_h]].tolist())
    return np.array([s in held for s in scene_tokens])


def load_split(dump_dir, sub_path):
    dump = scorer_io.load_dump(dump_dir)
    if dump["manifest"].get("smoke"):
        raise SystemExit(f"{dump_dir} 는 스모크 덤프다 — 실제 학습/평가에 쓰지 않는다")
    z = np.load(sub_path, allow_pickle=False)
    sc = {k: z[k] for k in z.files}
    if not np.array_equal(sc["idx"], dump["idx"]):
        raise SystemExit(f"{sub_path} 와 {dump_dir} 의 표본 순서가 다르다")
    return dump, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="train 분할 덤프 폴더")
    ap.add_argument("--subscores", required=True, help="train 분할 subscores .npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--holdout_frac", type=float, default=0.1)
    ap.add_argument("--final_fit", choices=["full", "split90"], default="full",
                    help="최종 점수기: full = 가중치 선택 뒤 train 전체로 재학습(Hydra-MDP, 기본) / split90 = 90 % 모델 그대로")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    set_determinism(a.seed, a.threads)
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "train_log.txt"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    dump, sc = load_split(a.dump, a.subscores)
    if dump["manifest"]["split"] != "train":
        raise SystemExit("점수기 학습은 train 분할 덤프만 쓴다(val 금지)")
    valid = sc["valid"].astype(bool)
    log(f"표본 {len(valid)}: 학습 사용 {valid.sum()}, 미래 GT 불완전 제외 {(~valid).sum()} (평가기와 같은 규칙)")
    held = scene_holdout(dump["scene_token"], a.holdout_frac, a.seed)
    fit_m = valid & ~held; hold_m = valid & held
    log(f"장면 단위 분할: 학습 {fit_m.sum()} 표본 / 보류 {hold_m.sum()} 표본 (보류 장면 비율 {a.holdout_frac})")
    d_fit, d_hold = take(dump, fit_m), take(dump, hold_m)
    s_fit, s_hold = take(sc, fit_m), take(sc, hold_m)
    X_fit, norm = build_inputs(d_fit)
    X_hold, _ = build_inputs(d_hold, norm)
    Y_fit, Y_hold = targets(s_fit), targets(s_hold)

    t0 = time.time()
    # (1) 90 % 장면으로 학습 → 보류 10 % 에서 가중치 선택(L2 최소)
    model, hist = train_model(X_fit, Y_fit, a.epochs, a.lr, a.wd, a.bs, a.seed, a.hidden, log, X_hold, Y_hold)
    out = predict(model, X_hold)
    best, base, rows = choose_weights(out, s_hold, d_hold["sel"])
    oracle = metrics_for_selection(s_hold, s_hold["l2_step"].mean(-1).argmin(-1))
    log(f"보류 장면 원 모델 선택: L2 avg {base['L2']['avg']:.4f} / 충돌 avg {100 * base['obj_box_col']['avg']:.4f}%")
    log("보류 장면 후보(L2 낮은 순 상위 10, 충돌은 보고만):")
    for r in sorted(rows, key=lambda r: (r["L2_avg"], r["col_avg"]))[:10]:
        log(f"  w={r['w']}  L2 avg {r['L2_avg']:.4f}  충돌 avg {100 * r['col_avg']:.4f}%")
    log(f"선택 가중치 {best['w']}: L2 avg {best['L2_avg']:.4f} / 충돌 avg {100 * best['col_avg']:.4f}%")
    log(f"보류 장면 최근접 모드(상한): L2 avg {oracle['L2']['avg']:.4f}")
    meta = {"args": vars(a), "hidden": a.hidden, "feat_dim": int(X_fit["feat"].shape[-1]),
            "traj_dim": int(X_fit["traj"].shape[-1]), "train_dump_manifest": dump["manifest"],
            "subscores": os.path.abspath(a.subscores), "weights": best["w"]}
    torch.save(dict(meta, state_dict=model.state_dict(), norm=norm, fit="split90"), os.path.join(a.out, "scorer_split90.pt"))
    # (2) 수정(2026-10-19, 사용자 결정 3): Hydra-MDP 처럼 train 전체(100 % 장면, 유효 표본)로 같은 설정·에폭 재학습 → 최종 점수기.
    #     표준화 통계도 전체로 다시 만든다. 가중치는 (1) 에서 고른 값을 그대로 쓴다.
    full_hist = None
    if a.final_fit == "full":
        d_all, s_all = take(dump, valid), take(sc, valid)
        X_all, norm_all = build_inputs(d_all)
        log(f"최종 재학습: train 전체 유효 표본 {valid.sum()} (장면 100 %), 같은 설정 {a.epochs} 에폭")
        set_determinism(a.seed, a.threads)
        model_f, full_hist = train_model(X_all, targets(s_all), a.epochs, a.lr, a.wd, a.bs, a.seed, a.hidden, log)
        torch.save(dict(meta, state_dict=model_f.state_dict(), norm=norm_all, fit="full"), os.path.join(a.out, "scorer.pt"))
    else:
        torch.save(dict(meta, state_dict=model.state_dict(), norm=norm, fit="split90"), os.path.join(a.out, "scorer.pt"))
    rep = {"n_total": int(len(valid)), "n_valid": int(valid.sum()), "n_excluded_incomplete_gt": int((~valid).sum()),
           "n_fit": int(fit_m.sum()), "n_heldout": int(hold_m.sum()), "history_split90": hist, "history_full": full_hist,
           "final_fit": a.final_fit, "weight_rule": "heldout L2 avg 최소(충돌은 보고만)",
           "heldout_original_selection": base, "heldout_chosen": {k: best[k] for k in ("w", "L2_avg", "col_avg", "metrics")},
           "heldout_oracle_nearest": oracle,
           "grid": [{k: r[k] for k in ("w", "L2_avg", "col_avg")} for r in rows], "seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(a.out, "train_report.json"), "w"), indent=1, ensure_ascii=False)
    log(f"저장: {os.path.join(a.out, 'scorer.pt')}")


if __name__ == "__main__":
    main()
