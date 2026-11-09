# 수정(2026-11-06): C3 GRPO M2(advantage.py) 단위 시험 — 신규 파일. 합성 점수만 쓴다(nuplan 불필요).
"""  source /home1/irteam/data-vol1/hipad_env.sh && python tools/navsim/grpo/tests/test_advantage.py   (pytest 가능)"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import advantage as AD  # noqa: E402

M, G = 4, 8


def _scene(r, fail=None, wp=None, front=None, mo=None, lat=None, det=None, guide=None, mu16=None):
    r = np.asarray(r, np.float64).reshape(M, G)
    fail = np.full((M, G, 7), -1) if fail is None else fail
    wp = np.where(fail < 0, fail, np.clip(np.ceil(fail / 5) - 1, 0, 7)).astype(int) if wp is None else wp
    pol = dict(epdms=r, fail_idx=fail, fail_wp=wp, coll_front=np.zeros((M, G), bool) if front is None else front,
               min_obj_dist_fs_m=np.full((M, G), 5.0) if mo is None else mo,
               lat_dev_wp_m=np.full((M, G), 0.2) if lat is None else lat)
    return dict(token="t", policy=pol, det_epdms=np.linspace(0, 1, 16) if det is None else det,
                mu16=np.zeros((16, 8, 2)) if mu16 is None else mu16, guide=guide or {}, lidar2ego=None)


def test_each_group_one_stage_random():
    rng = np.random.default_rng(0)
    for _ in range(500):
        kind = rng.integers(4)
        if kind == 0:
            r = rng.choice([0, 0.5, 1.0], (M, G))
        elif kind == 1:
            r = np.repeat(rng.choice([0, 0.7, 1.0], (M, 1)), G, 1)
        elif kind == 2:
            r = np.zeros((M, G))
        else:
            r = np.full((M, G), 0.9)
        sc = _scene(r, guide=dict(ref_traj=np.zeros((8, 3)), ref_epdms=1.0, human_traj=np.zeros((8, 3)), human_epdms=1.0))
        out = AD.scene_advantage(sc)
        st = out["stage"]
        assert st.shape == (M,) and set(st.tolist()) <= {1, 2, 3, 4}
        for g in range(M):
            var = r[g].std() > 1e-6
            assert (st[g] == 1) == var
            if not var:
                exp = 2 if r.std() > 1e-6 else (3 if r.mean() <= 1e-6 else 4)
                assert st[g] == exp, (r, st)
        if (st == 3).any() or (st == 4).any():
            assert len(set(st.tolist())) == 1                       # ③④ 는 장면 전체
        assert (out["Lg"] is not None) == (st[0] == 3)              # ③ 이면 장면당 L_g 하나
        assert np.all(out["A"][st == 2] == 0) and np.all(out["A"][st == 3] == 0)


def test_det_samples_excluded():
    r = np.zeros((M, G))
    r[0, :] = 1.0   # 앵커 내 붕괴, 장면 분산 있음 → ②
    for det in (np.zeros(16), np.ones(16), np.linspace(0, 1, 16)):
        out = AD.scene_advantage(_scene(r, det=det))
        assert list(out["stage"]) == [2, 2, 2, 2]
    # 정책 32 가 모두 0 이고 ξ=0 은 좋아도 ③
    out = AD.scene_advantage(_scene(np.zeros((M, G)), det=np.ones(16),
                                    guide=dict(ref_traj=np.zeros((8, 3)), ref_epdms=0.9, human_traj=np.zeros((8, 3)),
                                               human_epdms=1.0)))
    assert list(out["stage"]) == [3, 3, 3, 3]


def test_rloo_and_weight_sums():
    rng = np.random.default_rng(1)
    r = rng.uniform(0, 1, (M, G))
    fail = np.full((M, G, 7), -1)
    kinds = rng.integers(0, 6, (M, G))
    for g in range(M):
        for i in range(G):
            t = int(rng.integers(1, 41))
            col = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4}[kinds[g, i]]
            if rng.uniform() < 0.8:
                fail[g, i, col] = t
            if rng.uniform() < 0.3:
                fail[g, i, 6] = -2                                   # 이력 기원 comfort — 주 실패 아님
    front = kinds == 0
    out = AD.scene_advantage(_scene(r, fail=fail, front=front))
    assert list(out["stage"]) == [1] * M
    for g in range(M):
        for i in range(G):
            exp = r[g, i] - np.mean(np.delete(r[g], i))
            assert abs(out["Ahat"][g, i] - exp) < 1e-12
    np.testing.assert_allclose(out["w"].sum(-1), 8.0, atol=1e-12)
    np.testing.assert_allclose(out["u"].sum(-1), 2.0, atol=1e-12)
    np.testing.assert_allclose(out["A"], out["Ahat"][..., None, None] * out["w"][..., None] * out["u"][..., None, :])
    np.testing.assert_allclose((out["w"][..., None] * out["u"][..., None, :]).sum((-1, -2)), 16.0)
    # 실패 없는 표본은 균등
    nof = np.array([[f is None for f in row] for row in out["ftype"]])
    assert np.all(out["w"][nof] == 1) and np.all(out["u"][nof] == 1)
    # RLOO 합은 0
    np.testing.assert_allclose(out["Ahat"].sum(-1), 0, atol=1e-12)


def test_primary_failure_and_tables():
    f = np.full(7, -1)
    f[1], f[2], f[0] = 5, 12, 12            # ttc 5(곱 지표 아님), offroad 12, collision 12 → 같은 시점이면 collision 먼저
    wp = np.where(f < 0, -1, np.ceil(f / 5) - 1).astype(int)
    assert AD.primary_failure(f, wp, True) == ("collision_front", 2)
    assert AD.primary_failure(f, wp, False) == ("collision_side", 2)
    f2 = np.full(7, -1)
    f2[1], f2[5], f2[6] = 20, 3, 1
    assert AD.primary_failure(f2, np.where(f2 < 0, -1, np.ceil(f2 / 5) - 1).astype(int), False) == ("ttc", 3)
    f3 = np.full(7, -1)
    f3[5], f3[6] = 3, -2
    assert AD.primary_failure(f3, f3, False) == (None, -1)
    w, u, _ = AD.weights("red_light", 4)
    assert tuple(u) == (2.0, 0.0) and abs(w.sum() - 8) < 1e-12
    pw = AD.prior_w(3)
    raw = np.array([0.2 + 0.8 * 0.8 ** (3 - k) if k <= 3 else (1.0 if k <= 5 else 0.2) for k in range(8)])
    np.testing.assert_allclose(pw, raw * 8 / raw.sum())
    np.testing.assert_allclose(AD.b2_w(2), [8 / 3] * 3 + [0] * 5)
    for v in ("B1", "P"):
        w, u, _ = AD.weights("offroad", 3, v)
        assert np.all(w == 1) and np.all(u == 1)
    w, u, _ = AD.weights("offroad", 3, "A-a")
    assert np.all(w == 1) and tuple(u) == (0.6, 1.4)
    w, u, _ = AD.weights("offroad", 3, "A-b")
    assert np.all(u == 1) and not np.all(w == 1)


def test_stage4_c_rloo_anchor_8():
    r = np.full((M, G), 0.8)
    rng = np.random.default_rng(2)
    mo = rng.uniform(0, 10, (M, G))
    mo[1, 3] = np.nan                                                # 전방·측방 객체 없음 → +∞
    lat = rng.uniform(0, 1, (M, G))
    out = AD.scene_advantage(_scene(r, mo=mo, lat=lat))
    assert list(out["stage"]) == [4] * M
    c = 0.5 * np.tanh(np.where(np.isnan(mo), np.inf, mo) / 2) - 0.5 * np.tanh(lat / 0.5)
    assert abs(c[1, 3] - (0.5 - 0.5 * np.tanh(lat[1, 3] / 0.5))) < 1e-12
    for g in range(M):
        for i in range(G):
            assert abs(out["Ahat"][g, i] - 0.1 * (c[g, i] - np.mean(np.delete(c[g], i)))) < 1e-12   # 앵커 내 8
    assert out["log"]["c_nan_as_inf"] == 1 and np.all(out["w"] == 1) and np.all(out["u"] == 1)
    for v in ("B1", "A-d", "P"):
        o = AD.scene_advantage(_scene(r, mo=mo, lat=lat), variant=v)
        assert np.all(o["A"] == 0)


def test_stage3_guide():
    mu16 = np.zeros((16, 8, 2))
    for q in range(16):
        mu16[q, :, 0] = np.arange(1, 9) * (q + 1) * 0.5
    ref = np.stack([np.arange(1, 9) * 3.0, np.zeros(8), np.zeros(8)], -1)          # q = 5 와 같음
    hum = np.stack([np.arange(1, 9) * 1.0, np.zeros(8), np.zeros(8)], -1)          # q = 1
    g = dict(ref_traj=ref, ref_epdms=0.6, human_traj=hum, human_epdms=0.9)
    out = AD.scene_advantage(_scene(np.zeros((M, G)), guide=g, mu16=mu16))
    Lg = out["Lg"]
    assert Lg["source"] == "pdm" and Lg["q_star"] == 5 and abs(Lg["weight"] - 0.1 * 0.6) < 1e-12
    g2 = dict(g, ref_epdms=0.0)
    Lg2 = AD.scene_advantage(_scene(np.zeros((M, G)), guide=g2, mu16=mu16))["Lg"]
    assert Lg2["source"] == "human" and Lg2["q_star"] == 1 and abs(Lg2["weight"] - 0.09) < 1e-12
    LgP = AD.scene_advantage(_scene(np.zeros((M, G)), guide=g, mu16=mu16), variant="P")["Lg"]
    assert abs(LgP["weight"] - 0.5 * 0.1 * 0.6) < 1e-12                           # P: 전부 실패 그룹 가중 0.5
    for v in ("B1", "A-d"):
        assert AD.scene_advantage(_scene(np.zeros((M, G)), guide=g, mu16=mu16), variant=v)["Lg"] is None
    g3 = dict(ref_traj=ref, ref_epdms=0.0, human_traj=np.full((8, 3), np.nan), human_epdms=np.nan)
    Lg3 = AD.scene_advantage(_scene(np.zeros((M, G)), guide=g3, mu16=mu16))["Lg"]
    assert Lg3["source"] == "none" and Lg3["weight"] == 0.0
    try:
        AD.scene_advantage(_scene(np.zeros((M, G)), guide={}))
        raise AssertionError("안내 궤적 없이 ③ 은 오류여야 함")
    except ValueError:
        pass


def test_cls_target():
    e = np.linspace(0, 1, 16)
    t, ok, why = AD.cls_target(e)
    assert ok and why == "no_rescore_info" and abs(t.sum() - 1) < 1e-12 and t.argmax() == 15
    np.testing.assert_allclose(t, np.exp(e / 0.1) / np.exp(e / 0.1).sum())
    # 채점 안 된 후보(epdms −999)
    e2 = e.copy()
    e2[3] = -999.0
    t2, ok2, why2 = AD.cls_target(e2)
    assert not ok2 and why2 == "epdms_invalid" and np.all(t2 == 0)
    # §12.1(5): rescore 후 logit 에 −999 가 하나라도 있으면 무효(예외 아님)
    lg = np.linspace(-1, 1, 16)
    t3, ok3, why3 = AD.cls_target(e, rescored_logits=lg)
    assert ok3 and why3 == "ok" and np.allclose(t3, t)
    for j in (0, 7, 15):
        lg2 = lg.copy()
        lg2[j] = -999.0
        t4, ok4, why4 = AD.cls_target(e, rescored_logits=lg2)
        assert not ok4 and why4 == "rescore_sentinel" and np.all(t4 == 0)
        m = np.zeros(16, bool)
        m[j] = True
        assert AD.cls_target(e, sentinel_mask=m)[1] is False
    assert AD.cls_target(e, sentinel_mask=np.zeros(16, bool))[1] is True
    try:
        AD.cls_target(e, rescored_logits=np.zeros(8))
        raise AssertionError("모양 오류는 거부")
    except ValueError:
        pass


def test_cls_invalid_keeps_advantage():
    """−999 프레임도 advantage·단계는 그대로(학습 쪽 순전파·L_BC 유지), cls_valid 만 False."""
    rng = np.random.default_rng(5)
    r = rng.uniform(0, 1, (M, G))
    lg = np.linspace(-1, 1, 16)
    lg_bad = lg.copy()
    lg_bad[2] = -999.0
    a = AD.scene_advantage(_scene(r), rescored_logits=lg)
    b = AD.scene_advantage(_scene(r), rescored_logits=lg_bad)
    assert a["cls_valid"] and not b["cls_valid"] and b["log"]["cls_reason"] == "rescore_sentinel"
    np.testing.assert_array_equal(a["A"], b["A"])
    np.testing.assert_array_equal(a["stage"], b["stage"])
    lg_bat = AD.batch_log([a, b])
    assert lg_bat["cls_invalid"] == 1 and lg_bat["cls_rescore_sentinel"] == 1


def test_ego_to_lidar_3x3_4x4():
    psi, t = 0.3, np.array([1.2, -0.4])
    c, s_ = np.cos(psi), np.sin(psi)
    M4 = np.eye(4)
    M4[:2, :2] = [[c, -s_], [s_, c]]
    M4[:2, 3] = t
    M4[2, 3] = 1.8
    M3 = np.eye(3)
    M3[:2, :2] = M4[:2, :2]
    M3[:2, 2] = t
    p_l = np.random.default_rng(0).normal(size=(8, 2))
    p_e = p_l @ M4[:2, :2].T + t
    np.testing.assert_allclose(AD.ego_to_lidar(p_e, M4), p_l, atol=1e-12)
    np.testing.assert_allclose(AD.ego_to_lidar(p_e, M3), p_l, atol=1e-12)
    np.testing.assert_allclose(AD.ego_to_lidar(np.c_[p_e, np.zeros(8)], M3), p_l, atol=1e-12)
    try:
        AD.ego_to_lidar(p_e, np.eye(2))
        raise AssertionError("2×2 는 거부")
    except ValueError:
        pass


def test_g0_kernel_json():
    shape = np.full(15, 0.1)
    shape[7] = 1.0      # Δ = 0 에 집중
    shape[6] = 0.5
    d = dict(label="예비", w0=0.2, types={"offroad": dict(status="measured", n=60, shape=shape.tolist(),
                                                           u=dict(lon=0.5, lat=1.5)),
                                         "ttc": dict(status="prior", n=10)})
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(d, f)
    k = AD.load_kernel(f.name)
    w, u, src = AD.weights("offroad", 4, kernel=k)
    raw = np.maximum(0.2, shape[np.arange(8) - 4 + 7])
    np.testing.assert_allclose(w, raw * 8 / raw.sum())
    assert tuple(u) == (0.5, 1.5) and src == "g0/g0"
    w, u, src = AD.weights("ttc", 4, kernel=k)
    assert src == "prior/table" and tuple(u) == (1.4, 0.6)


def test_batch_log():
    r = np.random.default_rng(3).uniform(0, 1, (M, G))
    res = [AD.scene_advantage(_scene(r)), AD.scene_advantage(_scene(np.full((M, G), 0.8)))]
    lg = AD.batch_log(res)
    assert abs(lg["stage1_frac"] - 0.5) < 1e-12 and abs(lg["stage4_frac"] - 0.5) < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print(f"PASS {f.__name__}", flush=True)
    print(f"모든 시험 통과 ({len(fns)})")
