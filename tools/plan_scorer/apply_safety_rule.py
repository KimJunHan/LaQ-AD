# 수정(2026-11-01): 사용자 결정("받아두자") — Hydra-MDP 점수기 선택에 사전 선언한 안전 규칙 1 개를 추가 비교 행으로 붙인다.
#   새 파일(기존 파일 수정 없음). 원래 Hydra-MDP 선택 경로(apply_and_eval.py)는 그대로 두고, 그 결과와 나란히 비교만 한다.
"""안전 규칙: 점수기는 '모델 스스로 자기 예측 물체와 충돌한다고 판정한 후보 궤적' 을 고르지 못한다(추론 시 신호, GT 없음).

고정한 규칙(자유 선택값 없음 — 전부 모델 코드의 기존 규칙). 수정(2026-11-01, 독립 검토 2건): 이 규칙은 이전·현 2단계의 Hydra val
  충돌 악화를 본 뒤 만든 사후 분석이다(사전 선언 아님). 조정값이 없고 val 은 한 번만 채점했다.
  - 충돌 판정 = 모델 자체의 rescore(models/head/motion_planning/plan_decoder.py SparsePlanDecoder.rescore, config with_rescore=True)
    가 logit 에 -999 를 더한 모드. 덤프의 probs(rescore 반영 softmax, fp32)에서 정확히 0 이 된다(실측: 0 이 아닌 최소 확률
    val 0.0035 / train 0.0032 — 경계가 깨끗하다). 판정 기준 `probs <= 0` 은 train_scorer.build_inputs 의 aux 특징과 같은 식.
    rescore 의 기하·임계값(코드 기본값 그대로): 예측 검출 신뢰도 >= 0.15 인 물체만, 예측 운동 상위 1 모드, 시점 6 개(3 s, 0.5 s 간격),
    자차 박스 ego_size × 1.1, 궤적 접선 yaw, 박스 모서리 포함 검사(평가기의 shapely 교차가 아닌 거친 근사).
    수정(2026-11-01, 독립 검토): 코드의 '전방 0.5 m 오프셋'(plan_decoder.py:304-305)은 flatten 뒤 행 0·1 을 바꾸는
    SparseDrive 원본 버그라 실제로는 적용되지 않는다 — 규칙 설명에 넣지 않는다. 명령으로 고른 묶음(temp,2hz)의 16 모드.
  - 규칙 = 판정된 모드를 제외하고 Hydra-MDP 선택 점수(같은 가중치) argmax. 판정되지 않은 모드가 있으면 그중 최고 점수.
  - 대체(전 모드 판정): 원 모델 선택(argmax plan_mode_probs = 덤프 sel). 단, 모델 rescore 자체가 '전 모드 충돌이면 -999 를
    더하지 않는다'(plan_decoder.py: col[all_col] = False) 규칙이라 덤프에서는 전 모드 판정 표본이 생길 수 없다(실측 train·val 0).
    그 경우(전 모드 충돌)는 판정 0 개로 보이므로 규칙이 작동하지 않고 점수기 선택이 그대로 남는다 — 모델 자체의 GT 없는 관례.
    전 모드 충돌 표본을 따로 가려내려면 rescore 의 all_col 을 덤프하는 GPU 재덤프가 필요하다(이 스크립트는 하지 않음).
  - 보류 장면(train 10 %, train_scorer.scene_holdout 과 같은 시드·비율) 결과는 가중치를 고른 그 90 % 모델(scorer_split90.pt)로,
    val 결과는 최종 scorer.pt 로 낸다. 규칙에 조정할 값이 없어서 보류 장면에서 고른 것도 없다(보고만). val 은 한 번만 채점.
사용: CUDA_VISIBLE_DEVICES="" PYTHONPATH=. OPENBLAS_CORETYPE=SkylakeX python tools/plan_scorer/apply_safety_rule.py \
        --config <test_config> --dump_root results/<run>_plan_scorer --results evaluation/<run>/results.pkl \
        --ref_dir evaluation/<run>_plan_scorer --out_dir evaluation/<run>_plan_scorer_safety
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import scorer_io  # noqa: E402
from train_scorer import (build_inputs, predict, selection_score, metrics_for_selection, scene_holdout, take,  # noqa: E402
                          load_split)
from apply_and_eval import load_scorer, load_results, replace_plans, write_results_copy, run_evaluator, table  # noqa: E402

RULE = ("점수기는 모델 자체 rescore 가 자기 예측 물체(검출 신뢰도 >= 0.15, 예측 운동 상위 1 모드, 3 s)와 충돌한다고 판정한 모드"
        "(probs <= 0)를 고르지 못한다 — 판정 제외 후 Hydra 점수 argmax; 전 모드 판정이면 원 모델 argmax(실제로는 rescore 규칙상 0 건)")


def safety_select(score, probs, fallback_sel):
    """score (N,M) Hydra 선택 점수, probs (N,M) rescore 반영 확률, fallback_sel (N,) 원 모델 선택. 반환: 선택, 통계."""
    # 수정(2026-11-01, 독립 검토): NaN·inf 가 있으면 판정·argmax 가 조용히 틀어지므로 멈춘다
    if not (np.isfinite(score).all() and np.isfinite(probs).all()):
        raise RuntimeError("점수 또는 probs 에 NaN/inf")
    flag = np.asarray(probs) <= 0
    hydra = score.argmax(-1)
    sel = np.where(flag, -np.inf, score).argmax(-1)
    all_f = flag.all(-1)
    sel[all_f] = np.asarray(fallback_sel)[all_f]
    st = {"n": int(len(sel)), "n_samples_with_flag": int(flag.any(-1).sum()),
          "n_hydra_picked_flagged": int(flag[np.arange(len(sel)), hydra].sum()),
          "n_changed_by_rule": int((sel != hydra).sum()), "n_fallback_all_flagged": int(all_f.sum())}
    if flag[np.arange(len(sel)), sel][~all_f].any():
        raise RuntimeError("규칙 적용 뒤에도 판정된 모드가 선택됐다")
    return sel, hydra, st


def heldout_part(dump_root, scorer_dir):
    """train 보류 장면(가중치를 고른 그 장면)에서 같은 행을 계산. train_report 의 수치를 재현하는지 대조한다."""
    dump, sc = load_split(os.path.join(dump_root, "dump_train"), os.path.join(dump_root, "subscores_train.npz"))
    model, ck = load_scorer(os.path.join(scorer_dir, "scorer_split90.pt"))
    rep = json.load(open(os.path.join(scorer_dir, "train_report.json")))
    a = ck["args"]
    valid = sc["valid"].astype(bool)
    hold_m = valid & scene_holdout(dump["scene_token"], a["holdout_frac"], a["seed"])
    if int(hold_m.sum()) != rep["n_heldout"]:
        raise SystemExit(f"보류 표본 수 {hold_m.sum()} != train_report {rep['n_heldout']}")
    d, s = take(dump, hold_m), take(sc, hold_m)
    X, _ = build_inputs(d, ck["norm"])
    score = selection_score(predict(model, X), ck["weights"]).numpy()
    sel, hydra, st = safety_select(score, d["probs"], d["sel"])
    rows = [("전: 원 모델 선택(argmax 확률)", metrics_for_selection(s, d["sel"])),
            ("후: 점수기 선택(Hydra-MDP 방식)", metrics_for_selection(s, hydra)),
            ("후: 점수기 선택 + 안전 규칙", metrics_for_selection(s, sel)),
            ("(참고) GT 최근접 모드 — 상한, GT 사용", metrics_for_selection(s, s["l2_step"].mean(-1).argmin(-1)))]
    for (nm, m), ref in ((rows[0], rep["heldout_original_selection"]["L2"]["avg"]), (rows[1], rep["heldout_chosen"]["L2_avg"])):
        if abs(m["L2"]["avg"] - ref) > 1e-6:
            raise SystemExit(f"보류 장면 '{nm}' L2 {m['L2']['avg']:.6f} 가 train_report {ref:.6f} 를 재현하지 못한다")
    return rows, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dump_root", required=True, help="results/<run>_plan_scorer (dump_train, dump_val, subscores_*.npz, scorer/)")
    ap.add_argument("--results", required=True, help="원본 results.pkl (읽기 전용)")
    ap.add_argument("--ref_dir", required=True, help="기존 apply_and_eval 출력(읽기 전용) — 같은 Hydra 선택인지 대조, 사본 재평가")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if os.path.realpath(a.out_dir) == os.path.realpath(a.ref_dir):
        raise SystemExit("out_dir 는 기존 출력과 다른 폴더여야 한다")
    os.makedirs(a.out_dir, exist_ok=True)
    scorer_dir = os.path.join(a.dump_root, "scorer")

    # (1) 보류 장면(train) — 규칙 고정 뒤 보고만
    h_rows, h_st = heldout_part(a.dump_root, scorer_dir)
    print(table(h_rows), "\n보류 장면 규칙 통계", h_st, flush=True)

    # (2) val — 한 번만 채점
    dump = scorer_io.load_dump(os.path.join(a.dump_root, "dump_val"))
    if dump["manifest"]["split"] != "val" or dump["manifest"].get("smoke"):
        raise SystemExit("val 전체 덤프가 아니다")
    model, ck = load_scorer(os.path.join(scorer_dir, "scorer.pt"))
    if ck["train_dump_manifest"]["split"] != "train":  # 수정(2026-11-01, 독립 검토): apply_and_eval.py 와 같은 검사
        raise SystemExit("점수기가 train 덤프로 학습되지 않았다")
    if ck["train_dump_manifest"]["checkpoint"]["realpath"] != dump["manifest"]["checkpoint"]["realpath"]:
        raise SystemExit("점수기 학습 덤프와 val 덤프의 체크포인트가 다르다")
    X, _ = build_inputs(dump, ck["norm"])
    score = selection_score(predict(model, X), ck["weights"]).numpy()
    sel, hydra, v_st = safety_select(score, dump["probs"], dump["sel"])
    if not np.array_equal(hydra, np.load(os.path.join(a.ref_dir, "scorer_sel.npy"))):
        raise SystemExit("Hydra 선택이 기존 scorer_sel.npy 와 다르다 — 같은 점수기·덤프가 아니다")
    np.save(os.path.join(a.out_dir, "scorer_safety_sel.npy"), sel)
    res = load_results(a.results)
    cp_st = replace_plans(res, sel, dump["modes"], dump["probs"])
    p_safe = os.path.join(a.out_dir, "results_plan_scorer_safety.pkl")
    write_results_copy(res, a.results, p_safe)
    del res
    ev = lambda p, n: run_evaluator(a.config, p, os.path.join(a.out_dir, n))  # noqa: E731
    m_orig = ev(a.results, "planning_original_results.json")
    m_base = ev(os.path.join(a.ref_dir, "results_dump_argmax.pkl"), "planning_before.json")
    m_hyd = ev(os.path.join(a.ref_dir, "results_plan_scorer.pkl"), "planning_after_hydra.json")
    m_safe = ev(p_safe, "planning_after_hydra_safety.json")
    z = np.load(os.path.join(a.dump_root, "subscores_val.npz"), allow_pickle=False)
    if not np.array_equal(z["idx"], dump["idx"]):
        raise SystemExit("val subscores 와 덤프 순서가 다르다")
    v = z["valid"].astype(bool)
    sc = {"l2_step": z["l2_step"][v], "col_step": z["col_step"][v]}
    own = metrics_for_selection(sc, sel[v])
    dl, dc = abs(own["L2"]["avg"] - m_safe["L2"]["avg"]), abs(own["obj_box_col"]["avg"] - m_safe["obj_box_col"]["avg"])
    if dl > 1e-4 or dc > 1e-6 or own["n_eval"] != m_safe["n_eval"]:
        raise SystemExit(f"[중단] 덤프 기반 지표(안전 규칙)가 평가기와 다르다(L2 {dl:.2e}, 충돌 {dc:.2e})")
    v_rows = [("(참고) 원본 평가 실행 results.pkl", m_orig), ("전: 덤프 실행 원 모델 선택(argmax 확률)", m_base),
              ("후: 덤프 실행 점수기 선택(Hydra-MDP 방식)", m_hyd), ("후: 덤프 실행 점수기 선택 + 안전 규칙", m_safe),
              ("(참고) GT 최근접 모드 — 상한, GT 사용", metrics_for_selection(sc, z["l2_step"][v].mean(-1).argmin(-1)))]
    v_st_eval = {"changed_by_rule_eval_samples": int((sel[v] != hydra[v]).sum()),
                 "hydra_picked_flagged_eval_samples": int((dump["probs"][v][np.arange(v.sum()), hydra[v]] <= 0).sum())}

    md = ["# Hydra-MDP 점수기 + 안전 규칙(2026-11-01)", "", f"규칙(고정): {RULE}", "",
          "고정값(모두 모델 코드 기존 규칙, 보류·val 에서 고른 값 없음): 판정 = 덤프 probs <= 0 (= rescore -999), "
          "rescore 기본값 score_thresh 0.15 · num_motion_mode 1 · 시점 6(3 s) · dim_scale 1.1 (rescore 의 offset 0.5 m 는 원본 버그로 미적용); "
          "Hydra 가중치는 기존 그대로 " + json.dumps(ck["weights"]) + "; 대체 = 원 모델 argmax(전 모드 판정 시, rescore 규칙상 발생 불가).", "",
          f"## val ({m_safe['n_eval']} 평가 표본, 최종 scorer.pt)"  # 수정(2026-11-01, 독립 검토): 고정 문자열 대신 실제 표본 수, "", table(v_rows), "",
          f"- 규칙이 선택을 바꾼 표본: 전체 {v_st['n_changed_by_rule']}/{v_st['n']} (평가 표본 {v_st_eval['changed_by_rule_eval_samples']}/{int(v.sum())}); "
          f"대체(전 모드 판정) {v_st['n_fallback_all_flagged']} 건; 판정 모드가 하나라도 있는 표본 {v_st['n_samples_with_flag']}",
          "", "## train 보류 장면(가중치를 고른 10 % 장면, scorer_split90.pt)", "",
          "원본 평가 실행 행은 없다(train 은 평가 실행이 없고 덤프 실행 하나뿐).", "", table(h_rows), "",
          f"- 규칙이 선택을 바꾼 표본: {h_st['n_changed_by_rule']}/{h_st['n']}; 대체(전 모드 판정) {h_st['n_fallback_all_flagged']} 건; "
          f"판정 모드가 하나라도 있는 표본 {h_st['n_samples_with_flag']}", ""]
    open(os.path.join(a.out_dir, "before_after.md"), "w").write("\n".join(md))
    summ = {"rule": RULE, "weights": ck["weights"], "val": dict(zip(["original_results", "before", "after_hydra",
                                                                     "after_hydra_safety", "oracle_nearest"], [r[1] for r in v_rows])),
            "val_rule_stats": dict(v_st, **v_st_eval), "val_copy_stats": cp_st,
            "heldout": dict(zip(["before", "after_hydra", "after_hydra_safety", "oracle_nearest"], [r[1] for r in h_rows])),
            "heldout_rule_stats": h_st, "dump_vs_evaluator_safety_absdiff": {"L2_avg": dl, "col_avg": dc},
            "results_copy": os.path.abspath(p_safe), "ref_dir": os.path.abspath(a.ref_dir)}
    json.dump(summ, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
