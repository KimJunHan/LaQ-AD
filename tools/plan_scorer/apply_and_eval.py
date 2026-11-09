# 수정(2026-10-19): 사용자 결정 — GRPO 대신 Hydra-MDP 방식 계획 점수 학습(LaQ v2 2단계 뒤 적용). 새 파일(기존 파일 수정 없음).
"""학습된 계획 점수기로 val 표본마다 계획 모드를 다시 고르고, results.pkl 의 **사본**에 계획을 바꿔 넣은 뒤
tools/eval_nusc_planning.py(평가기 그대로)로 평가해 전/후 표를 출력한다(같은 체크포인트).

왜 사본이 둘인가(실측 2026-10-19): GPU 추론이 비트 단위로 재현되지 않는다(같은 설정 두 번 실행에서 첫 프레임 4e-3 m,
한 장면 끝 무렵 모드 최대 0.27 m, 선택 모드가 40 프레임 중 2 번 다름 — ops/src/deformable_aggregation_cuda.cu:184 정방향 atomicAdd 가 원인, 시간 뱅크로 증폭). 원 모델 확률이 거의 균등해 이 차이로도 argmax 가 바뀐다.
그래서 '점수기 효과' 는 같은 덤프 실행 안에서 비교한다:
  (1) 원본 results.pkl(평가 실행)            — 참고: 기존 보고 수치. 원본은 읽기만 한다.
  (2) 덤프 실행의 원 모델 선택(argmax 확률)   — 사본 results_dump_argmax.pkl
  (3) 덤프 실행의 점수기 선택                 — 사본 results_plan_scorer.pkl
  전/후 = (2) → (3). (1)↔(2) 차이가 비결정 크기다.
사본의 각 표본: plan_temp_2hz ← 덤프 plan_modes[선택], plan_modes·plan_mode_probs ← 덤프 값, 원래 값은 *_orig 키로 보관,
  plan_scorer_sel 추가. 덤프 모드와 원본 plan_modes 차이는 기록하고, 표본 2 m 초과 또는 평균 0.3 m 초과(다른 체크포인트·순서 수준)면 중단.
점수기는 GT 를 보지 않으므로 미래 GT 불완전 표본도 다시 고른다. 평가기는 그 표본을 평가에서 제외한다(평가기 규칙 그대로).
--val_subscores 를 주면 덤프 기반 지표(하위 점수와 같은 규칙)를 (2)(3) 평가기 결과와 대조한다(다르면 중단).
사용: PYTHONPATH=. OPENBLAS_CORETYPE=SkylakeX python tools/plan_scorer/apply_and_eval.py \
        --config <test_config> --val_dump <dir> --scorer <scorer.pt> --results evaluation/<run>/results.pkl \
        --out_dir evaluation/<run>_plan_scorer [--val_subscores <val_subscores.npz>]
"""
import argparse
import copy
import functools
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import scorer_io  # noqa: E402
from train_scorer import PlanScorer, build_inputs, predict, selection_score, metrics_for_selection  # noqa: E402

GROSS_TOL_M = 2.0   # 표본 하나의 모드 최대차 상한(실측 비결정: 같은 설정 두 번 실행에서 한 장면 안 최대 0.27 m)
MEAN_TOL_M = 0.3    # 표본별 최대차 평균 상한(실측 비결정 평균 ≈ 0.07 m)


def load_scorer(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = PlanScorer(feat_dim=ck["feat_dim"], traj_dim=ck["traj_dim"], hidden=ck["hidden"])
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def rescore_dump(model, ck, dump):
    X, _ = build_inputs(dump, ck["norm"])
    out = predict(model, X)
    return selection_score(out, ck["weights"]).argmax(-1).numpy(), out


def load_results(path):
    _tl = torch.load
    torch.load = functools.partial(_tl, map_location="cpu", weights_only=False)  # results.pkl 의 CUDA 텐서 → CPU(평가기와 같은 방식)
    try:
        return pickle.load(open(path, "rb"))
    finally:
        torch.load = _tl


def replace_plans(results, sel, dump_modes, dump_probs, gross_tol=GROSS_TOL_M):
    """results(메모리 사본)를 바꾼다: plan_temp_2hz ← dump_modes[sel]. 반환: 통계 dict."""
    if len(results) != len(sel):
        raise ValueError(f"results {len(results)} != 선택 {len(sel)}")
    st = {"changed_vs_original_plan": 0, "max_mode_diff": 0.0, "mean_mode_diff": 0.0}
    for i, r in enumerate(results):
        d = r.get("img_bbox", r)
        modes_o = torch.as_tensor(d["plan_modes"]).float()
        modes_d = torch.as_tensor(np.asarray(dump_modes[i])).float()
        diff = float((modes_o - modes_d).abs().max())
        if diff > gross_tol:
            raise RuntimeError(f"색인 {i}: 덤프 모드와 results plan_modes 차 {diff:.3f} m > {gross_tol} — 같은 체크포인트·순서가 아니다")
        st["max_mode_diff"] = max(st["max_mode_diff"], diff); st["mean_mode_diff"] += diff / len(sel)
        old = torch.as_tensor(d["plan_temp_2hz"]).float()
        new = modes_d[int(sel[i])].clone()
        d["plan_temp_2hz_orig"] = old
        d["plan_modes_orig"] = modes_o
        d["plan_mode_probs_orig"] = torch.as_tensor(d["plan_mode_probs"]).float()
        d["plan_modes"] = modes_d
        d["plan_mode_probs"] = torch.as_tensor(np.asarray(dump_probs[i])).float()
        d["plan_scorer_sel"] = int(sel[i])
        d["plan_temp_2hz"] = new
        st["changed_vs_original_plan"] += int(not torch.allclose(new, old, atol=1e-2))
    if st["mean_mode_diff"] > MEAN_TOL_M:
        raise RuntimeError(f"덤프와 results 의 평균 모드 차 {st['mean_mode_diff']:.3f} m > {MEAN_TOL_M} — 같은 체크포인트·순서가 아니다")
    return st


def write_results_copy(results, src_path, dst_path):
    if os.path.realpath(dst_path) == os.path.realpath(src_path):
        raise SystemExit("사본 경로가 원본 results.pkl 과 같다 — 원본은 덮어쓰지 않는다")
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    tmp = dst_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, dst_path)


def run_evaluator(config, results_path, out_json):
    cmd = [sys.executable, os.path.join(_ROOT, "tools", "eval_nusc_planning.py"), config, results_path, "--out", out_json]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=_ROOT)
    return json.load(open(out_json))


def table(rows):
    lines = ["| 선택 방식 | L2 1s | L2 2s | L2 3s | L2 avg | 충돌 1s % | 충돌 2s % | 충돌 3s % | 충돌 avg % | 표본 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in rows:
        L, C = m["L2"], m["obj_box_col"]
        lines.append(f"| {name} | {L['1s']:.3f} | {L['2s']:.3f} | {L['3s']:.3f} | {L['avg']:.3f} | "
                     f"{100 * C['1s']:.3f} | {100 * C['2s']:.3f} | {100 * C['3s']:.3f} | {100 * C['avg']:.3f} | {m['n_eval']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="평가에 쓴 test config(평가기 데이터셋 구성용)")
    ap.add_argument("--val_dump", required=True)
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--results", required=True, help="원본 results.pkl (읽기 전용)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_subscores", default=None, help="있으면 덤프 기반 지표를 평가기 결과와 대조(다르면 중단)")
    a = ap.parse_args()

    dump = scorer_io.load_dump(a.val_dump)
    man = dump["manifest"]
    if man["split"] != "val" or man.get("smoke"):
        raise SystemExit(f"{a.val_dump}: val 전체 덤프가 아니다(split={man['split']}, smoke={man.get('smoke')})")
    model, ck = load_scorer(a.scorer)
    if ck["train_dump_manifest"]["split"] != "train":
        raise SystemExit("점수기가 train 분할 덤프로 학습되지 않았다")
    if ck["train_dump_manifest"]["checkpoint"]["realpath"] != man["checkpoint"]["realpath"]:
        raise SystemExit("점수기 학습 덤프와 val 덤프의 2단계 체크포인트가 다르다")
    new_sel, _ = rescore_dump(model, ck, dump)
    base_sel = dump["sel"]
    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, "scorer_sel.npy"), new_sel)

    original = load_results(a.results)
    orig_sel = np.array([int(np.argmax(np.asarray(r.get("img_bbox", r)["plan_mode_probs"]))) for r in original])
    paths, st = {}, {}
    for name, sel in (("dump_argmax", base_sel), ("plan_scorer", new_sel)):
        res = copy.deepcopy(original)
        st[name] = replace_plans(res, sel, dump["modes"], dump["probs"])
        paths[name] = os.path.join(a.out_dir, f"results_{name}.pkl")
        write_results_copy(res, a.results, paths[name])
        del res
        print(f"사본 저장: {paths[name]} {st[name]}", flush=True)
    del original

    m_orig = run_evaluator(a.config, a.results, os.path.join(a.out_dir, "planning_original_results.json"))
    m_base = run_evaluator(a.config, paths["dump_argmax"], os.path.join(a.out_dir, "planning_before.json"))
    m_new = run_evaluator(a.config, paths["plan_scorer"], os.path.join(a.out_dir, "planning_after.json"))
    rows = [("(참고) 원본 평가 실행 results.pkl", m_orig), ("전: 덤프 실행 원 모델 선택(argmax 확률)", m_base),
            ("후: 덤프 실행 점수기 선택(Hydra-MDP 방식)", m_new)]
    summ = {"original_results": m_orig, "before": m_base, "after": m_new, "n": int(len(new_sel)),
            "changed_selection_vs_dump_argmax": int((new_sel != base_sel).sum()),
            "argmax_flip_original_vs_dump": int((orig_sel != base_sel).sum()),
            "copy_stats": st, "weights": ck["weights"], "scorer": os.path.abspath(a.scorer),
            "results_original": os.path.abspath(a.results), "results_copies": paths, "checkpoint": man["checkpoint"]}
    if a.val_subscores:
        z = np.load(a.val_subscores, allow_pickle=False)
        if not np.array_equal(z["idx"], dump["idx"]):
            raise SystemExit("val subscores 와 덤프 순서가 다르다")
        v = z["valid"].astype(bool)
        sc = {"l2_step": z["l2_step"][v], "col_step": z["col_step"][v]}
        nearest = z["l2_step"][v].mean(-1).argmin(-1)
        own_b = metrics_for_selection(sc, base_sel[v]); own_a = metrics_for_selection(sc, new_sel[v])
        oracle = metrics_for_selection(sc, nearest)
        rows.append(("(참고) GT 최근접 모드 — 상한, GT 사용", oracle))
        summ.update({"dump_rule_before": own_b, "dump_rule_after": own_a, "oracle_nearest": oracle,
                     "selected_is_nearest_before": float((base_sel[v] == nearest).mean()),
                     "selected_is_nearest_after": float((new_sel[v] == nearest).mean())})
        for nm, x, y in (("before", own_b, m_base), ("after", own_a, m_new)):
            dl = abs(x["L2"]["avg"] - y["L2"]["avg"]); dc = abs(x["obj_box_col"]["avg"] - y["obj_box_col"]["avg"])
            summ[f"dump_vs_evaluator_{nm}_absdiff"] = {"L2_avg": dl, "col_avg": dc}
            if dl > 1e-4 or dc > 1e-6 or x["n_eval"] != y["n_eval"]:
                raise SystemExit(f"[중단] 덤프 기반 지표({nm})가 평가기와 다르다(L2 {dl:.2e}, 충돌 {dc:.2e}, "
                                 f"표본 {x['n_eval']} vs {y['n_eval']}) — 하위 점수 계산 규칙 재확인")
    t = table(rows)
    print(t)
    print(f"점수기가 선택을 바꾼 표본 {summ['changed_selection_vs_dump_argmax']}/{len(new_sel)}; "
          f"원본 평가 실행과 덤프 실행의 argmax 가 다른 표본(비결정) {summ['argmax_flip_original_vs_dump']}")
    open(os.path.join(a.out_dir, "before_after.md"), "w").write(t + "\n")
    json.dump(summ, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
