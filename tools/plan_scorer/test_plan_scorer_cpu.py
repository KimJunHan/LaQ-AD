# 수정(2026-10-19): 사용자 결정 — GRPO 대신 Hydra-MDP 방식 계획 점수 학습(LaQ v2 2단계 뒤 적용). 새 파일(기존 파일 수정 없음).
"""계획 점수기 CPU 시험(GPU 사용 안 함).

  1. 기하: get_yaw_batch·ego_corners 가 평가기 get_yaw·check_collision 의 상자와 같은 값인지
  2. 충돌: SAT 판정 == 평가기 single_coll(shapely) — 무작위 장면 수백 개 + 손으로 만든 경우(정면 박스, GT 도 충돌한 시점 제외)
  3. 주행 가능: 합성 다각형에서 모서리 규칙·GT 기준 제외 규칙
  4. 진행: 경로 사영·반직선 연장·안전 모드 정규화·5 m 규칙
  5. 모방 soft target, 평가기 집계 재현(평가기 main 의 누적 평균 식과 비교)
  6. 덤프 형식 저장/읽기(샤드·미래 박스 묶기)
  7. 학습: 작은 합성 자료에서 손실 감소·결정성(두 번 실행 동일)·가중치 선택(보류 L2 최소), 명령행(90/10 선택 → train 전체 재학습)
  3b. 차선 경로 진행(수정 2026-10-19): 사영 호 길이·원점 기준·경로 없음 대신·gt 판
  8. 적용: results 사본에 계획 교체(원본 파일 불변), 다른 모드 거부
  9. (선택) 실제 nuScenes val info 로 평가기 대조 — GT 주변 합성 모드에서 SAT == shapely, GT 주행 가능 비율
사용: PYTHONPATH=. OPENBLAS_CORETYPE=SkylakeX CUDA_VISIBLE_DEVICES="" python tools/plan_scorer/test_plan_scorer_cpu.py [--real 300]
"""
import argparse
import os
import pickle
import sys
import tempfile

import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import scorer_io  # noqa: E402
import subscores as S  # noqa: E402
import train_scorer as TS  # noqa: E402
import apply_and_eval as AE  # noqa: E402

E = S.EVAL
OK = []


def ok(name):
    OK.append(name); print(f"[통과] {name}", flush=True)


def rand_traj(rng, M, T=6):
    v = rng.uniform(0, 12, (M, 1)); k = rng.uniform(-0.15, 0.15, (M, 1))
    t = np.arange(1, T + 1)[None] * 0.5
    head = np.pi / 2 + k * t * v / 5
    step = np.stack([np.cos(head), np.sin(head)], -1) * (v[..., None] * 0.5)
    tr = np.cumsum(step, 1) + rng.normal(0, 0.05, (M, T, 2))
    tr[: max(1, M // 8)] *= 0.02  # 일부는 정지(이동 0.5 m 미만 → yaw π/2 규칙)
    return tr.astype(np.float32)


def rand_boxes(rng, n):
    b = np.zeros((n, 7), np.float32)
    b[:, 0] = rng.uniform(-12, 12, n); b[:, 1] = rng.uniform(-5, 35, n); b[:, 2] = rng.uniform(-2, 0, n)
    b[:, 3] = rng.uniform(0.5, 2.5, n); b[:, 4] = rng.uniform(0.5, 6, n); b[:, 5] = rng.uniform(1, 3, n)
    b[:, 6] = rng.uniform(-np.pi, np.pi, n)
    return b


def test_geometry():
    rng = np.random.RandomState(0)
    tr = rand_traj(rng, 64)
    yb = S.get_yaw_batch(torch.as_tensor(tr))
    for m in range(len(tr)):
        ye = E.get_yaw(torch.as_tensor(tr[m]))
        assert (yb[m] - ye).abs().max() < 1e-6, (m, yb[m], ye)  # 벡터 atan2 는 1 ulp 차이 가능
    c = S.ego_corners(tr)
    for m in range(0, 64, 7):
        for t in range(6):
            eb = torch.zeros(7); eb[:2] = torch.as_tensor(tr[m, t]); eb[3:6] = torch.tensor([4.084, 1.85, 1.56])
            eb[6] = E.get_yaw(torch.as_tensor(tr[m]))[t]
            eb[0] += 0.5 * torch.cos(eb[6]); eb[1] += 0.5 * torch.sin(eb[6])
            ce = E.box3d_to_corners(eb.unsqueeze(0))[0, [0, 3, 7, 4], :2]
            assert np.abs(ce - c[m, t]).max() < 1e-5
    ok("기하(get_yaw·자차 상자) == 평가기")


def test_collision_equivalence():
    rng = np.random.RandomState(1)
    cells = 0; hits = 0
    for scene in range(300):
        tr = rand_traj(rng, 16)
        fut = [rand_boxes(rng, rng.randint(0, 25)) for _ in range(6)]
        mine = S.collision_steps(tr, fut)
        ref = S.evaluator_collision_steps(tr, fut)
        assert np.array_equal(mine, ref), (scene, np.argwhere(mine != ref))
        cells += mine.size; hits += mine.sum()
    assert hits > 50, hits
    # 손으로 만든 경우: 직진 10 m/s, 1.5 s 시점(t=2) 전방 15 m 에 박스 → t=2 충돌
    tr = np.stack([np.zeros(6), np.arange(1, 7) * 5.0], -1)[None].astype(np.float32)
    fut = [np.zeros((0, 7), np.float32) for _ in range(6)]
    fut[2] = np.array([[0, 15, 0, 1.8, 4.5, 1.6, 0]], np.float32)
    col = S.collision_steps(tr, fut)
    assert col[0].tolist() == [False, False, True, False, False, False], col
    # 옆 차선(4 m 옆) 은 충돌 아님
    fut[2] = np.array([[4.0, 15, 0, 1.8, 4.5, 1.6, 0]], np.float32)
    assert not S.collision_steps(tr, fut).any()
    ok(f"충돌 SAT == 평가기 shapely ({cells} 칸, 충돌 {hits}) + 정면/옆 차선 경우")


class _FakeArea:
    """lidar 좌표 = 전역 좌표(항등 변환)에서 x∈[-5,5] 띠(도로) 만 주행 가능."""

    def inside(self, location, xy):
        xy = np.asarray(xy).reshape(-1, 2)
        return (xy[:, 0] > -5) & (xy[:, 0] < 5)


def test_subscores_synthetic():
    I = np.eye(4)
    gt = np.stack([np.zeros(6), np.arange(1, 7) * 5.0], -1).astype(np.float32)
    modes = np.stack([gt,                                   # 0 GT 와 같음
                      gt + np.array([0, -2.0], np.float32),  # 1 약간 느림
                      gt * np.array([1, 0.5], np.float32),   # 2 절반 속도
                      gt + np.array([6.0, 0], np.float32),   # 3 도로 밖(x=6)
                      ], 0).astype(np.float32)
    modes = np.concatenate([modes, np.repeat(gt[None], 12, 0)], 0)
    fut = [np.zeros((0, 7), np.float32) for _ in range(6)]
    fut[5] = np.array([[0, 15, 0, 1.8, 4.5, 1.6, 0]], np.float32)  # t=5(3 s) 에 15 m 지점: 절반 속도 모드(2)만 충돌
    r = S.sample_subscores(modes, gt, np.ones(6), fut, I, "x", _FakeArea(), tau=0.5)
    assert r["valid"]
    assert r["nc"][:4].tolist() == [True, True, False, True], r["nc"][:4]
    assert r["dac"][:4].tolist() == [True, True, True, False] and r["dac_strict"][3] == False  # noqa: E712
    # 진행: 안전 모드(0,1 …) 최대 30 m → 모드 1 = 28/30, 모드 2(충돌, 15 m) = 0.5, 모드 3 = 30 m 사영 → 1
    assert abs(r["ep"][0] - 1.0) < 1e-9 and abs(r["ep"][1] - 28 / 30) < 1e-6 and abs(r["ep"][2] - 0.5) < 1e-6, r["ep"][:4]
    assert abs(r["im"].sum() - 1) < 1e-9 and r["im"].argmax() == 0
    # GT 도 충돌한 시점은 제외: GT 경로 위(t=1, 10 m)에 박스 → GT·모드 0 모두 충돌 → 모드 0 은 nc 유지
    fut2 = [b.copy() for b in fut]; fut2[1] = np.array([[0, 10, 0, 1.8, 4.5, 1.6, 0]], np.float32)
    r2 = S.sample_subscores(modes, gt, np.ones(6), fut2, I, "x", _FakeArea(), tau=0.5)
    assert r2["gt_col_step"][1] and r2["nc"][0]
    # GT 가 도로 밖인 시점은 dac 판단에서 제외(dac_strict 는 위반)
    gt_off = gt + np.array([6.0, 0], np.float32)
    r3 = S.sample_subscores(modes, gt_off, np.ones(6), fut, I, "x", _FakeArea(), tau=0.5)
    assert r3["dac"].all() and not r3["dac_strict"][3] and r3["dac_strict"][0]  # 모드 3 은 GT 와 같이 도로 밖
    # 불완전 GT → 제외
    assert S.sample_subscores(modes, gt, np.array([1, 1, 1, 0, 0, 0]), fut[:3], I, "x", _FakeArea(), 0.5) == {"valid": False}
    # 정지 GT(최대 진행 < 5 m) → 전 모드 progress 1
    st = np.zeros((16, 6, 2), np.float32); st[:, :, 1] = np.linspace(0, 1, 6)[None] * np.arange(16)[:, None] * 0.2
    r4 = S.sample_subscores(st, np.zeros((6, 2), np.float32), np.ones(6), [np.zeros((0, 7))] * 6, I, "x", _FakeArea(), 0.5)
    assert np.allclose(r4["ep"], 1.0)
    # 반직선 연장: GT 보다 더 간 모드는 진행이 더 크다
    s = S.path_progress(np.array([[0, 40.0], [0, 30.0], [3, 20.0]]), gt)
    assert abs(s[0] - 40) < 1e-9 and abs(s[1] - 30) < 1e-9 and abs(s[2] - 20) < 1e-9
    ok("하위 점수 합성 경우(nc·dac·dac_strict·ep·im, GT 충돌/도로밖 제외, 불완전 GT, 정지)")


def test_aggregate():
    rng = np.random.RandomState(2)
    l2 = rng.uniform(0, 3, (50, 6)); col = rng.rand(50, 6) < 0.1
    m = S.evaluator_metrics(l2, col)
    L2 = torch.as_tensor(l2, dtype=torch.float32).sum(0) / 50
    cum = [float(L2[:j + 1].mean()) for j in range(6)]
    assert abs(m["L2"]["avg"] - (cum[1] + cum[3] + cum[5]) / 3) < 1e-5
    ok("평가기 집계(누적 평균) 재현")


def make_fake_dump(rng, N, M=16, T=6, C=32, scenes=10):
    gt = np.cumsum(np.stack([rng.normal(0, 0.2, (N, T)), rng.uniform(1, 5, (N, T))], -1), 1).astype(np.float32)
    modes = (gt[:, None] + rng.normal(0, 1.5, (N, M, T, 2))).astype(np.float32)
    best = np.sqrt(((modes - gt[:, None]) ** 2).sum(-1)).mean(-1).argmin(1)
    feats = rng.normal(0, 1, (N, M, C)).astype(np.float32)
    feats[np.arange(N), best, 0] += 3.0  # 학습 가능한 신호: 최근접 모드의 특징 0 번이 크다
    raw = rng.normal(0, 1, (N, M)).astype(np.float32)
    probs = torch.softmax(torch.as_tensor(raw), -1).numpy()
    return {"idx": np.arange(N), "token": np.array([f"t{i}" for i in range(N)]),
            "scene_token": np.array([f"s{i * scenes // N}" for i in range(N)]), "timestamp": np.arange(N) * 0.5,
            "cmd": rng.randint(0, 3, N), "modes": modes, "probs": probs, "raw_logits": raw, "sel": probs.argmax(1),
            "feats": feats, "gt_traj": gt, "gt_mask": np.ones((N, T), np.float32),
            "fut_box_counts": np.zeros((N, T), np.int64), "fut_boxes": [[] for _ in range(N)],  # load_dump 과 같은 풀린 형식
            "lidar2global": np.repeat(np.eye(4)[None], N, 0), "map_location": np.array(["x"] * N)}, best


def fake_subscores(dump, rng):
    N, M, T = dump["modes"].shape[:3]
    l2 = np.sqrt(((dump["modes"] - dump["gt_traj"][:, None]) ** 2).sum(-1))
    col = rng.rand(N, M, T) < 0.02
    im = np.stack([S.imitation_target(l2[i].mean(-1), 0.5) for i in range(N)])
    return {"idx": dump["idx"], "valid": np.ones(N, bool), "nc": (~col.any(-1)).astype(np.float32),
            "dac": np.ones((N, M), np.float32), "ep": rng.uniform(0, 1, (N, M)).astype(np.float32),
            "im": im.astype(np.float32), "l2_step": l2.astype(np.float32), "col_step": col}


def test_io(tmp):
    rng = np.random.RandomState(3)
    d, _ = make_fake_dump(rng, 40)
    fut = [[rand_boxes(rng, rng.randint(0, 4)) for _ in range(6 if i % 5 else 3)] for i in range(40)]
    out = os.path.join(tmp, "dump"); os.makedirs(out)
    import json
    json.dump({"split": "train", "num_samples": 40, "shards": [[0, 25], [25, 40]], "smoke": False,
               "checkpoint": {"realpath": "x"}}, open(os.path.join(out, "manifest.json"), "w"))
    for k, (a, b) in enumerate([(0, 25), (25, 40)]):
        rec = {key: d[key][a:b] for key in scorer_io.PER_SAMPLE_KEYS if key != "fut_box_counts"}
        packed = [scorer_io.pack_fut_boxes(f) for f in fut[a:b]]
        rec["fut_box_counts"] = np.stack([p[0] for p in packed]); rec["fut_boxes"] = np.concatenate([p[1] for p in packed])
        scorer_io.save_shard(out, k, rec)
    back = scorer_io.load_dump(out)
    assert np.array_equal(back["modes"], d["modes"]) and len(back["fut_boxes"]) == 40
    for i in range(40):
        assert len(back["fut_boxes"][i]) == len(fut[i])
        for t in range(len(fut[i])):
            assert np.array_equal(back["fut_boxes"][i][t], fut[i][t])
    ok("덤프 샤드 저장/읽기(미래 박스 묶기 포함)")


def test_training(tmp):
    rng = np.random.RandomState(4)
    d, best = make_fake_dump(rng, 600)
    sc = fake_subscores(d, rng)
    held = TS.scene_holdout(d["scene_token"], 0.2, 0)
    assert held.any() and (~held).any()
    assert len(set(d["scene_token"][held]) & set(d["scene_token"][~held])) == 0
    fit, hold = ~held, held
    X, norm = TS.build_inputs(TS.take(d, fit)); Xh, _ = TS.build_inputs(TS.take(d, hold), norm)
    Y, Yh = TS.targets(TS.take(sc, fit)), TS.targets(TS.take(sc, hold))
    res = []
    for _ in range(2):
        TS.set_determinism(0, 2)
        m, hist = TS.train_model(X, Y, epochs=15, lr=3e-3, wd=1e-4, bs=64, seed=0, hidden=64, X_val=Xh, Y_val=Yh)
        res.append((m, hist))
    h = res[0][1]
    assert h[-1]["train_loss"] < h[0]["train_loss"] * 0.9, (h[0]["train_loss"], h[-1]["train_loss"])
    sd0, sd1 = res[0][0].state_dict(), res[1][0].state_dict()
    assert all(torch.equal(sd0[k], sd1[k]) for k in sd0), "같은 시드 두 번 학습 결과가 다르다(비결정)"
    out = TS.predict(res[0][0], Xh)
    best_w, base, rows = TS.choose_weights(out, TS.take(sc, hold), d["sel"][hold])
    assert best_w["L2_avg"] == min(r["L2_avg"] for r in rows)  # 사용자 결정 1: 보류 L2 최소(충돌 조건 없음)
    assert all("col_avg" in r for r in rows)                   # 충돌률은 후보마다 계산(보고용)
    sel = TS.selection_score(out, best_w["w"]).argmax(-1).numpy()
    acc = (sel == best[hold]).mean(); acc0 = (d["sel"][hold] == best[hold]).mean()
    assert acc > acc0 + 0.2, (acc, acc0)
    assert best_w["L2_avg"] <= base["L2"]["avg"]
    ckp = os.path.join(tmp, "scorer.pt")
    torch.save({"state_dict": res[0][0].state_dict(), "norm": norm, "weights": best_w["w"], "hidden": 64,
                "feat_dim": X["feat"].shape[-1], "traj_dim": X["traj"].shape[-1],
                "train_dump_manifest": {"split": "train", "checkpoint": {"realpath": "x"}}}, ckp)
    ok(f"학습(손실 {h[0]['train_loss']:.3f}→{h[-1]['train_loss']:.3f}, 결정성, 최근접 모드 적중 {acc0:.2f}→{acc:.2f}, "
       f"보류 L2 {base['L2']['avg']:.3f}→{best_w['L2_avg']:.3f})")
    return d, ckp


class _AllArea:
    """어디나 주행 가능(진행 점수만 보는 시험용)."""

    def inside(self, location, xy):
        return np.ones(len(np.asarray(xy).reshape(-1, 2)), bool)


class _FakeRoutes:
    """lidar 좌표에서 주어진 polyline 을 경로로 돌려준다(없으면 None)."""

    def __init__(self, route):
        self.r = route

    def route(self, location, l2g, gt):
        return (None, np.inf, 0) if self.r is None else (self.r, 0.0, 1)


def test_lane_progress():
    I = np.eye(4)
    # 경로: 원점 뒤 5 m 에서 시작해 전방(+y) 으로 가다 20 m 지점부터 오른쪽(+x)으로 꺾인다
    route = np.array([[0, -5.0], [0, 20.0], [30.0, 20.0]])
    pts = np.array([[0, 10.0], [10.0, 21.0], [0, -1.0]])
    s = S.route_progress(pts, route)
    assert abs(s[0] - 10) < 1e-9 and abs(s[1] - 30) < 1e-9 and abs(s[2] + 1) < 1e-9, s  # 원점 사영을 0 으로
    gt = np.stack([np.zeros(6), np.arange(1, 7) * 3.0], -1).astype(np.float32)   # GT 18 m 직진
    modes = np.repeat(gt[None], 16, 0).copy()
    modes[1] = np.stack([np.arange(1, 7) * 2.0, np.minimum(np.arange(1, 7) * 4.0, 20.0)], -1)  # 경로를 따라 우회전, 더 멀리
    fut = [np.zeros((0, 7), np.float32)] * 6
    r = S.sample_subscores(modes.astype(np.float32), gt, np.ones(6), fut, I, "x", _AllArea(), 0.5,
                           routes=_FakeRoutes(route), progress="lane")
    assert r["route_found"] and np.allclose(r["ep"], r["ep_lane"])
    # 차선 경로 기준: 모드 1 끝점 (12,20) → 20 + 12 = 32 m 로 최대 → 1, GT(18 m) → 18/32
    assert abs(r["ep_lane"][1] - 1) < 1e-9 and abs(r["ep_lane"][0] - 18 / 32) < 1e-6, r["ep_lane"][:2]
    # 종전 GT 경로 기준은 다르다(모드 1 이 GT 반직선에서 벗어남)
    assert not np.allclose(r["ep_lane"], r["ep_gt"])
    r2 = S.sample_subscores(modes.astype(np.float32), gt, np.ones(6), fut, I, "x", _AllArea(), 0.5,
                            routes=_FakeRoutes(None), progress="lane")
    assert not r2["route_found"] and np.allclose(r2["ep"], r2["ep_gt"])  # 경로 없음 → GT 경로로 대신
    r3 = S.sample_subscores(modes.astype(np.float32), gt, np.ones(6), fut, I, "x", _AllArea(), 0.5,
                            routes=_FakeRoutes(route), progress="gt")
    assert np.allclose(r3["ep"], r3["ep_gt"])
    ok("차선 경로 진행(사영 호 길이, 원점 기준, 경로 없음 → GT 경로 대신, gt 판 선택)")


def test_train_cli(tmp):
    """train_scorer.py 명령행: 90/10 가중치 선택(L2 최소) → train 전체 재학습 최종 모델(scorer.pt fit=full)."""
    import json, subprocess
    rng = np.random.RandomState(6)
    d, _ = make_fake_dump(rng, 400, scenes=20)
    sc = fake_subscores(d, rng)
    out = os.path.join(tmp, "cli_dump"); os.makedirs(out)
    json.dump({"split": "train", "num_samples": 400, "shards": [[0, 400]], "smoke": False,
               "checkpoint": {"realpath": "x"}}, open(os.path.join(out, "manifest.json"), "w"))
    rec = {k: d[k] for k in scorer_io.PER_SAMPLE_KEYS}
    rec["fut_boxes"] = np.zeros((0, 7), np.float32)
    scorer_io.save_shard(out, 0, rec)
    sp = os.path.join(tmp, "cli_sub.npz"); np.savez(sp, **sc)
    so = os.path.join(tmp, "cli_scorer")
    cmd = [sys.executable, os.path.join(_HERE, "train_scorer.py"), "--dump", out, "--subscores", sp, "--out", so,
           "--epochs", "3", "--hidden", "32", "--threads", "2", "--holdout_frac", "0.2"]
    subprocess.run(cmd, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), stdout=subprocess.DEVNULL)
    rep = json.load(open(os.path.join(so, "train_report.json")))
    ck = torch.load(os.path.join(so, "scorer.pt"), weights_only=False)
    ck90 = torch.load(os.path.join(so, "scorer_split90.pt"), weights_only=False)
    assert ck["fit"] == "full" and ck90["fit"] == "split90" and ck["weights"] == ck90["weights"]
    assert rep["heldout_chosen"]["L2_avg"] == min(r["L2_avg"] for r in rep["grid"])
    assert rep["history_full"] is not None and len(rep["history_full"]) == 3
    assert not all(torch.equal(ck["state_dict"][k], ck90["state_dict"][k]) for k in ck["state_dict"])
    ok(f"학습 명령행(보류 L2 최소 가중치 {ck['weights']}, train 전체 재학습 최종 모델, 90 % 모델 별도 보관)")


def test_apply(tmp, d, ckp):
    model, ck = AE.load_scorer(ckp)
    new_sel, _ = AE.rescore_dump(model, ck, d)
    N = len(new_sel)
    results = [{"img_bbox": {"plan_modes": torch.as_tensor(d["modes"][i]), "plan_mode_probs": torch.as_tensor(d["probs"][i]),
                             "plan_temp_2hz": torch.as_tensor(d["modes"][i][d["sel"][i]]), "boxes_3d": torch.zeros(3, 10)}}
               for i in range(N)]
    src = os.path.join(tmp, "results.pkl"); pickle.dump(results, open(src, "wb"))
    before = open(src, "rb").read()
    res = AE.load_results(src)
    st = AE.replace_plans(res, new_sel, d["modes"], d["probs"])
    changed = st["changed_vs_original_plan"]
    dst = os.path.join(tmp, "out", "results_plan_scorer.pkl")
    AE.write_results_copy(res, src, dst)
    assert open(src, "rb").read() == before, "원본 results.pkl 이 바뀌었다"
    new = pickle.load(open(dst, "rb"))
    for i in range(N):
        r = new[i]["img_bbox"]
        assert torch.equal(r["plan_temp_2hz"], torch.as_tensor(d["modes"][i][new_sel[i]]))
        assert torch.equal(r["plan_temp_2hz_orig"], torch.as_tensor(d["modes"][i][d["sel"][i]]))
        assert torch.equal(r["plan_modes"], torch.as_tensor(d["modes"][i])) and r["plan_scorer_sel"] == int(new_sel[i])
    assert changed == int((new_sel != d["sel"]).sum())
    try:
        AE.write_results_copy(res, src, src); raise AssertionError("원본 덮어쓰기를 막지 못했다")
    except SystemExit:
        pass
    bad = [dict(img_bbox=dict(r["img_bbox"])) for r in AE.load_results(src)]
    bad[3]["img_bbox"]["plan_modes"] = bad[3]["img_bbox"]["plan_modes"] + 3.0  # 비결정 수준(≤0.27 m)을 크게 넘는 차
    try:
        AE.replace_plans(bad, new_sel, d["modes"], d["probs"]); raise AssertionError("다른 모드를 거부하지 못했다")
    except RuntimeError:
        pass
    shifted = [dict(img_bbox=dict(r["img_bbox"])) for r in AE.load_results(src)]
    for r in shifted:
        r["img_bbox"]["plan_modes"] = r["img_bbox"]["plan_modes"] + 0.5  # 표본마다는 2 m 안이지만 평균 0.5 m > 0.3 m
    try:
        AE.replace_plans(shifted, new_sel, d["modes"], d["probs"]); raise AssertionError("평균 차 상한을 지키지 못했다")
    except RuntimeError:
        pass
    ok(f"적용(사본에만 교체 {changed}/{N}, 원본 불변, 원본 경로 거부, 모드 불일치 거부 — 표본·평균 상한)")


def test_real(n):
    """실제 nuScenes val info: 평가기 데이터셋(_get_fut_boxes)으로 GT 주변 합성 모드에서 SAT == shapely, GT 주행 가능 비율."""
    from mmcv import Config
    import importlib
    importlib.import_module("mmdet3d_plugin")  # 평가기와 같은 적재 방식(평가기 main 과 동일)
    from mmdet.datasets import build_dataset
    cfg = Config.fromfile("configs/archive/nuscenes_v1_20261016/stage2/test.py")
    dcfg = cfg.data.test.copy(); dcfg.pipeline = None
    ds = build_dataset(dcfg)
    area = S.DrivableArea("data/nuscenes")
    routes = S.LaneRoutes("data/nuscenes")
    rng = np.random.RandomState(5)
    cand = [i for i, inf in enumerate(ds.data_infos) if (np.asarray(inf["gt_ego_fut_masks"]) == 1).all()]
    pick = sorted(rng.choice(cand, size=min(n, len(cand)), replace=False).tolist())
    cells = hits = 0; gt_ok = []; gt_col = 0; found = []; gt_prog_err = []
    for i in pick:
        inf = ds.data_infos[i]
        gt = np.asarray(inf["gt_ego_fut_trajs"], np.float32).cumsum(0)[:6, :2]
        fut = ds._get_fut_boxes(i, inf)
        modes = (gt[None] + rng.normal(0, 1.5, (16, 6, 2)) * np.linspace(0.3, 1, 6)[None, :, None]).astype(np.float32)
        modes[0] = gt
        mine = S.collision_steps(modes, fut); ref = S.evaluator_collision_steps(modes, fut)
        assert np.array_equal(mine, ref), (i, np.argwhere(mine != ref))
        cells += mine.size; hits += int(mine.sum())
        r = S.sample_subscores(modes, gt, inf["gt_ego_fut_masks"], fut, ds._T_global(inf), inf["map_location"], area, 0.5,
                               routes=routes, progress="lane")
        gt_ok.append(r["gt_drivable_step"].all()); gt_col += int(r["gt_col_step"].any()); found.append(r["route_found"])
        if r["route_found"]:  # GT 끝점의 차선 경로 진행 ≈ GT 경로 길이(경로가 GT 를 따라가면)
            glen = float(np.linalg.norm(np.diff(S.gt_path(gt), axis=0), axis=-1).sum())
            gt_prog_err.append(abs(r["progress_m"][0] - glen))
        assert r["nc"][0] and r["dac"][0]  # GT 와 같은 모드는 정의상 nc·dac 1
    ok(f"실제 val {len(pick)} 표본: SAT == 평가기 shapely ({cells} 칸, 충돌 {hits}); "
       f"GT 6 시점 모두 주행 가능 {np.mean(gt_ok):.3f}; GT 자체 충돌 표본 {gt_col}; 차선 경로 찾음 {np.mean(found):.3f}, "
       f"GT 진행 − GT 길이 중앙값 {np.median(gt_prog_err):.2f} m")
    assert np.mean(found) > 0.9 and np.median(gt_prog_err) < 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=int, default=0, help=">0 이면 실제 val info 표본 수만큼 평가기 대조(데이터 필요)")
    a = ap.parse_args()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as tmp:
        test_geometry(); test_collision_equivalence(); test_subscores_synthetic(); test_lane_progress()
        test_aggregate(); test_io(tmp)
        d, ckp = test_training(tmp)
        test_train_cli(tmp)
        test_apply(tmp, d, ckp)
        if a.real > 0:
            test_real(a.real)
    print(f"전부 통과: {len(OK)} 항목")


if __name__ == "__main__":
    main()
