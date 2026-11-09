# 수정(2026-10-20): 사용자 승인 — 설명가능성 4가지(학습 변경 없음)
"""교사 설명문(rationale)과 그 정확도 — val 전체, CPU 전용(모델 추론 없음).

설명문 = 교사 VLM 이 실제로 생성한 547 토큰 답(색인 data/nuscenes/laq_cache/_index/teacher_{A,B}.jsonl 의 gen_text)을
  AutoVLA 4단 형식(1 장면 서술 · 2 주요 객체 · 3 의도 근거 · 4 최종 행동)으로 풀어 쓴 글.
  글·비교는 기존 그림 스크립트(/home1/irteam/data-vol1/src/HiP-AD/tools/visualization/nusc_fig_teacher_reasoning.py 의 compare·build_items)를
  수정 없이 불러 쓴다 → 논문·PPT 그림과 같은 문장·같은 판정.
정확도(GT 대비)는 교사 평가와 같은 함수(tools/vla_teacher/eval_teacher.py score / score_baselines, teacher_metrics):
  객체: 같은 클래스·방위·|Δd| ≤ max(2 m, 0.1 d) 헝가리안 일대일 매칭 → 재현율·정밀도·거리/속도 MAE·깊이 순서
  지도: 횡단보도 상태·거리, 차선 구분선·도로 경계 개수, 줄 전체 일치
  의도: 종·횡 정확도(미래 완비 표본, GT 'unavailable' 제외), 행동: 코드북 복호 궤적의 L2·충돌(평가기와 같은 식)
  + 표본별 판정(compare): 객체 일치 수, 지도 항목 일치(허용 1 m), 의도 일치, 행동 궤적 L2(1/2/3 s)
완비 검사: val 6,019 표본 모두 색인에 있고(파싱·위치 검사 통과), 교사 배정이 fold_split.json 과 같아야 한다(아니면 중단).

산출(--out_dir): rationale.jsonl(표본별 글·판정), metrics.json(전체·교사 A/B 별), summary.md
사용: CUDA_VISIBLE_DEVICES="" PYTHONPATH=. OPENBLAS_CORETYPE=SkylakeX python tools/xai/xai_rationale.py --out_dir evaluation/<run>_xai/rationale
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xai_common as X  # noqa: E402

FIG_DIR = "/home1/irteam/data-vol1/src/HiP-AD/tools/visualization"


def import_fig_module():
    """nusc_fig_teacher_reasoning 은 import 시 os.chdir(HiP-AD) 를 한다 → 작업 폴더를 되돌린다."""
    cwd = os.getcwd()
    if FIG_DIR not in sys.path:
        sys.path.insert(0, FIG_DIR)
    try:
        import nusc_fig_teacher_reasoning as F
    finally:
        os.chdir(cwd)
    return F


def items_to_lines(items):
    out = []
    for st, txt in items:
        if st in ("row", "rowh"):
            out.append(" | ".join(str(x) for x in txt[:3]) + ("" if st == "rowh" or txt[3] else "   [differs from GT]"))
        elif st == "gap":
            out.append("")
        elif st == "gt":
            out.append("  >> " + txt.strip())   # 그림에서 빨간 글(GT 와 다름)
        else:
            out.append(txt)
    return out


def sample_summary(cmp_):
    o, m, it, ac = cmp_["objects"], cmp_["map"], cmp_["intent"], cmp_["action"]
    s = dict(obj_n_teacher=o["n_teacher"], obj_n_gt=o["n_gt"], obj_matched=o["n_matched"], obj_precision=o["precision"],
             obj_recall=o["recall"], intent_lon_same=it["lon_same"], intent_lat_same=it["lat_same"],
             action_l2_123=ac.get("l2_at_1_2_3s_vs_gt_future"), action_ade=ac.get("ade_vs_gt_future"))
    if "ped_state_same" in m:
        s.update(map_ped_ok=m["ped_ok"], map_div_same=m["n_divider_same"], map_bnd_same=m["n_boundary_same"],
                 map_lanes_same_frac=m["lanes_same"] / m["lanes_total"], map_free_ok=m["free_dist_ok"])
    ms = o["matches"]
    s["obj_motion_same_frac"] = float(np.mean([x["motion_same"] for x in ms])) if ms else None
    s["obj_risk_same_frac"] = float(np.mean([x["ttc_same"] for x in ms])) if ms else None
    s["obj_fut3_err_mean"] = float(np.mean([x["fut3_err"] for x in ms if x["fut3_err"] is not None])) \
        if any(x["fut3_err"] is not None for x in ms) else None
    return s


def agg_samples(rows):
    def m(k):
        v = [r["summary"][k] for r in rows if r["summary"].get(k) is not None]
        return (float(np.mean(v)) if v else None), len(v)
    keys = ["obj_precision", "obj_recall", "obj_motion_same_frac", "obj_risk_same_frac", "obj_fut3_err_mean", "map_ped_ok",
            "map_div_same", "map_bnd_same", "map_lanes_same_frac", "map_free_ok", "intent_lon_same", "intent_lat_same", "action_ade"]
    out = {}
    for k in keys:
        v, n = m(k)
        out[k] = dict(mean=v, n=n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--infos", default=X.INFOS_VAL)
    ap.add_argument("--max_samples", type=int, default=0, help="개발 전용(0 = val 전체 — 보고는 전체만)")
    ap.add_argument("--no_text", action="store_true", help="설명문 생략(지표만)")
    a = ap.parse_args()
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    F = import_fig_module()
    import common_v2 as V
    import eval_teacher as E
    from common import ActionCodebook

    infos = X.load_infos(a.infos)
    toks = [i["token"] for i in infos]
    idx = X.load_val_index()
    gts = X.load_gt_val()
    miss = [t for t in toks if t not in idx]
    extra = [t for t in idx if t not in set(toks)]
    if miss or extra or len(gts) != len(toks):
        raise SystemExit(f"val 완비 아님: 색인 누락 {len(miss)}, 색인에만 있음 {len(extra)}, GT {len(gts)} / infos {len(toks)}")
    bad = [t for t in toks if not (idx[t].get("parse_ok") and idx[t].get("layout_ok") and idx[t].get("grammar_ok"))]
    if bad:
        raise SystemExit(f"파싱/위치/문법 검사 실패 표본 {len(bad)} (예: {bad[:3]})")
    fold = json.load(open(X.FOLD_FILE))
    cb = ActionCodebook(X.CODEBOOK)
    N = len(toks) if not a.max_samples else min(a.max_samples, len(toks))
    toks = toks[:N]
    rows = []
    for k, t in enumerate(toks):
        e, rec = idx[t], gts[t]
        part, exp_T = F.fold_of(rec["scene_token"], fold)
        if e["teacher"] != exp_T or e["part"] != part:
            raise SystemExit(f"{t}: 색인 교사 {e['teacher']}/{e['part']} != fold 배정 {exp_T}/{part}")
        Tp, Gp = V.parse_answer(e["gen_text"]), V.parse_answer(rec["answer"])
        cmp_, To, Go, M, tt = F.compare(Tp, Gp, rec, cb)
        row = dict(idx=k, token=t, scene_token=rec["scene_token"], teacher=e["teacher"], part=part, command=rec["command"]["text"],
                   future_complete=bool(rec["targets"]["future_complete"]), summary=sample_summary(cmp_),
                   teacher_intent=Tp["intent"], gt_intent=Gp["intent"], teacher_action_traj=None if tt is None else np.round(tt, 3).tolist())
        if not a.no_text:
            row["rationale"] = items_to_lines(F.build_items(rec, Tp, Gp, cmp_, To, Go, M, tt, e, None, paper=False))
        rows.append(row)
        if (k + 1) % 1000 == 0:
            print(f"[글·판정 {k + 1}/{N}] {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(a.out_dir, "rationale.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 교사 평가와 같은 지표(eval_teacher.score) — 전체 + 교사별
    infos_index = {i["token"]: j for j, i in enumerate(infos)}
    recs = [gts[t] for t in toks]
    gens = {t: dict(text=idx[t]["gen_text"], ended=True) for t in toks}
    args = types.SimpleNamespace(dist_tol_abs=2.0, dist_tol_rel=0.1, fut_box_rule="dataset_default")
    pe = X.planning_eval_module()
    E.check_gt_alignment(recs, infos, infos_index)
    cache = {}
    parser = lambda s: V.parse_answer(s)  # noqa: E731  (색인 생성 때 쓴 엄격 파서)
    res = dict(n_val=len(infos), n=len(toks), complete=(N == len(infos)), parser="common_v2.parse_answer",
               match_rule="same class & sector & |dd|<=max(2,0.1*d_gt), Hungarian (teacher_metrics.match_objects)",
               overall=E.score(recs, gens, parser, infos_index, infos, pe, cb, args, cache))
    print(f"[지표 전체] {time.time() - t0:.0f}s", flush=True)
    for T in ("A", "B"):
        rT = [r for r in recs if idx[r["token"]]["teacher"] == T]
        res[f"teacher_{T}"] = E.score(rT, gens, parser, infos_index, infos, pe, cb, args, cache)
    res["baselines"] = E.score_baselines(recs, infos_index, infos, pe, cb, args, cache)
    res["per_sample_judgement_mean"] = agg_samples(rows)
    res["per_sample_judgement_tolerances"] = dict(map_dist_m=F.TOL_M, speed_mps=F.TOL_SPEED, fut3_m=F.TOL_FUT)
    json.dump(res, open(os.path.join(a.out_dir, "metrics.json"), "w"), indent=1, ensure_ascii=False)
    write_md(res, os.path.join(a.out_dir, "summary.md"))
    print(open(os.path.join(a.out_dir, "summary.md")).read())
    print(f"[완료] {a.out_dir} ({time.time() - t0:.0f}s)")


def write_md(r, path):
    f = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
    pc = lambda x: "-" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    L = [f"# 교사 설명문 정확도 (val {r['n']}/{r['n_val']}{'' if r['complete'] else ' — 부분, 보고용 아님'})", "",
         "| 집합 | 객체 재현율 | 정밀도 | 거리 MAE(m) | 속도 MAE(m/s) | 깊이순서 | 지도 줄 정확 | 교차보도 | 의도 종 | 의도 횡 | 의도 결합 | 행동 L2 avg | 행동 Col avg |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in ("overall", "teacher_A", "teacher_B"):
        s = r[k]; p, mp, it, pl = s["perception"], s["map"], s["intent"], s["planning_valid_only"]
        L.append(f"| {k} | {pc(p['recall'])} | {pc(p['precision'])} | {f(p['dist_mae'], 2)} | {f(p['speed_mae'], 2)} | "
                 f"{pc(p['depth_order_acc'])} | {pc(mp['line_exact'])} | {pc(mp['ped_state_acc'])} | {pc(it.get('lon_acc'))} | "
                 f"{pc(it.get('lat_acc'))} | {pc(it.get('joint_acc'))} | {f(pl.get('L2', {}).get('avg'))} | "
                 f"{pc(pl.get('obj_box_col', {}).get('avg'))} |")
    b = r["baselines"]["copy"]
    L.append(f"| copy 기준선(명령+등속) | | | | | | | | {pc(b['intent'].get('lon_acc'))} | {pc(b['intent'].get('lat_acc'))} | "
             f"{pc(b['intent'].get('joint_acc'))} | {f(b['planning'].get('L2', {}).get('avg'))} | {pc(b['planning'].get('obj_box_col', {}).get('avg'))} |")
    L += ["", "## 표본별 판정 평균(그림 글과 같은 규칙)", "", "| 항목 | 평균 | n |", "|---|---|---|"]
    for k, v in r["per_sample_judgement_mean"].items():
        L.append(f"| {k} | {f(v['mean'])} | {v['n']} |")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
