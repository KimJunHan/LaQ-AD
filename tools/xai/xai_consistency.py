# 수정(2026-10-20): 사용자 승인 — 설명가능성 4가지(학습 변경 없음)
"""설명–행동 일치도(explanation–action agreement), CPU 전용.

질문: 교사(언어 모델)가 말한 의도("감속하며 직진")와 학생(LaQ-AD)이 실제로 낸 최종 계획 궤적이 같은 의도인가?

같은 정의로 비교하기 위해 학생 궤적의 의도는 **교사 GT 의도 라벨을 만든 함수 그대로** 붙인다(xai_common.IntentLabeler):
  tools/vla_teacher/gen_teacher_gt_v2.py 229~250 행(common.intent_labels + lane_v2.lane_signals 결정 규칙)을 같은 입력으로 호출.
  임계값(common.py 25~55 행, train 통계로 고정 — val 미사용):
    종방향  v_end = |p6 − p4| / 1.0 s, Δv = v_end − v0 (v0 = 현재 자차 속도, GT 레코드 targets.label_v0)
            stop: v_end < 0.5 m/s · accelerate: Δv ≥ +1.0 m/s · decelerate: Δv ≤ −1.0 m/s · 그 외 keep speed
    횡방향  경로 길이 < 2 m → go straight · θ_end ≥ +12.5° → turn left · ≤ −12.5° → turn right
            그 외 lane_v2 세 신호(지도 차로 위상·차선 구분선 교차·차로 기준 운동학, 2 표 이상 같은 쪽·반대 0) → lane change left/right,
            결정이 'uncertain' 이면 go straight(+uncertain 표시), 아니면 go straight
교사 의도 = 색인 _index/teacher_{A,B}.jsonl 의 생성 원문(gen_text)을 엄격 파서(common_v2.parse_answer)로 읽은 Intent 줄.

산출(--out_dir):
  per_sample.jsonl     표본별 교사/학생/GT 의도·학생 라벨 근거값(v_end, Δv, θ_end, 경로 길이, 구분선 교차, uncertain)·일치 여부
  disagreements.jsonl  교사↔학생 불일치 표본(횡방향 불일치 먼저, 그다음 종방향) — 그림·사례 분석용
  metrics.json / summary.md   일치율(종·횡·결합), 혼동행렬, 명령별, 미래 완비/확실 표본 부분집합, 참고 삼각형(학생↔GT, 교사↔GT)
--gt_sanity : 학생 대신 GT 미래 궤적을 같은 함수로 라벨링해 GT 의도와 비교(정의 재현 검사 — 구성상 ~100 % 여야 한다).

사용(CPU):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. OPENBLAS_CORETYPE=SkylakeX python tools/xai/xai_consistency.py \
      --results evaluation/<run>/results.pkl --out_dir evaluation/<run>_xai/consistency
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. OPENBLAS_CORETYPE=SkylakeX python tools/xai/xai_consistency.py --gt_sanity --out_dir <dir>
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xai_common as X  # noqa: E402

LON = ["accelerate", "keep speed", "decelerate", "stop"]
LAT = ["go straight", "turn left", "turn right", "lane change left", "lane change right"]


def confusion(pairs, classes):
    idx = {c: k for k, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), np.int64)
    for a, b in pairs:
        M[idx[a], idx[b]] += 1
    return M.tolist()


def rates(rows, ka, kb):
    """rows: dict 목록. ka/kb: 비교할 두 의도 키(예: 'teacher', 'student'). 둘 다 있는 행만."""
    R = [r for r in rows if r.get(ka) is not None and r.get(kb) is not None]
    n = len(R)
    if not n:
        return dict(n=0)
    lon = [(r[ka]["longitudinal"], r[kb]["longitudinal"]) for r in R]
    lat = [(r[ka]["lateral"], r[kb]["lateral"]) for r in R]
    out = dict(n=n, lon_agree=float(np.mean([a == b for a, b in lon])), lat_agree=float(np.mean([a == b for a, b in lat])),
               joint_agree=float(np.mean([a[0] == a[1] and b[0] == b[1] for a, b in zip(lon, lat)])),
               lon_confusion=dict(rows=ka, cols=kb, classes=LON, M=confusion(lon, LON)),
               lat_confusion=dict(rows=ka, cols=kb, classes=LAT, M=confusion(lat, LAT)))
    # 횡방향 '방향 반대'(좌↔우) — 가장 심한 불일치
    opp = sum(1 for a, b in lat if ("left" in a and "right" in b) or ("right" in a and "left" in b))
    out["lat_opposite_direction"] = opp / n
    # 클래스별 재현(교사 의도가 c 일 때 학생도 c)
    for name, pairs, cls in (("lon", lon, LON), ("lat", lat, LAT)):
        out[f"{name}_per_class"] = {c: dict(support=sum(1 for a, _ in pairs if a == c),
                                            agree=(sum(1 for a, b in pairs if a == c and b == c) /
                                                   max(sum(1 for a, _ in pairs if a == c), 1)))
                                    for c in cls if any(a == c for a, _ in pairs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="학생 results.pkl(평가 순서 = infos 순서). --gt_sanity 면 불필요")
    ap.add_argument("--infos", default=X.INFOS_VAL)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gt_sanity", action="store_true", help="학생 대신 GT 미래 궤적을 라벨링(정의 재현 검사)")
    ap.add_argument("--max_samples", type=int, default=0, help="개발 전용(0 = val 전체)")
    a = ap.parse_args()
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    if not a.gt_sanity and not a.results:
        raise SystemExit("--results 가 필요하다(또는 --gt_sanity)")
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    infos = X.load_infos(a.infos)
    gts = X.load_gt_val()
    idx = X.load_val_index()
    toks = [i["token"] for i in infos]
    miss_gt = [t for t in toks if t not in gts]
    miss_idx = [t for t in toks if t not in idx]
    if miss_gt or miss_idx:
        raise SystemExit(f"val 표본 누락 — GT {len(miss_gt)}, 교사 색인 {len(miss_idx)} (전체 {len(toks)}). 색인이 끝난 뒤 실행한다")
    res = None
    if not a.gt_sanity:
        res = X.load_results_cpu(a.results)
        if len(res) != len(infos):
            raise SystemExit(f"results.pkl 길이 {len(res)} != infos {len(infos)}")
    lab = X.IntentLabeler()
    N = len(infos) if not a.max_samples else min(a.max_samples, len(infos))
    rows = []
    n_teacher_na = 0
    for i in range(N):
        info, t = infos[i], toks[i]
        rec, e = gts[t], idx[t]
        if not e.get("parse_ok") or not e.get("layout_ok"):
            raise SystemExit(f"{t}: 교사 답 파싱/위치 검사 실패 표본 — 색인 확인")
        ti, _ = X.teacher_intent(e["gen_text"])
        n_teacher_na += int(ti is None)
        complete = bool(rec["targets"].get("future_complete"))
        gi = rec["targets"].get("intent")
        g = None if (not complete or gi in (None, "unavailable")) else dict(longitudinal=gi["longitudinal"], lateral=gi["lateral"])
        if a.gt_sanity:
            if not complete:
                continue
            traj = np.asarray(rec["targets"]["fut_traj"], np.float64)
        else:
            r = res[i].get("img_bbox", res[i])
            traj = np.asarray(r["plan_temp_2hz"], np.float64)[:6, :2]
        s = lab.label(traj, info, rec)
        row = dict(idx=i, token=t, scene_token=info["scene_token"], command=rec["command"]["text"], teacher_id=e["teacher"],
                   future_complete=complete, gt_uncertain=bool(gi.get("uncertain")) if isinstance(gi, dict) else None,
                   teacher=None if ti is None else dict(longitudinal=ti["longitudinal"], lateral=ti["lateral"]),
                   student=dict(longitudinal=s["longitudinal"], lateral=s["lateral"]), gt=g,
                   student_evidence={k: s[k] for k in ("v0", "v_end", "dv", "theta_end_deg", "path_len", "lat_resid",
                                                      "divider_crossing", "lateral_v1", "uncertain")},
                   traj_end=[round(float(traj[-1, 0]), 2), round(float(traj[-1, 1]), 2)])
        if row["teacher"] is not None:
            row["agree_lon"] = row["teacher"]["longitudinal"] == row["student"]["longitudinal"]
            row["agree_lat"] = row["teacher"]["lateral"] == row["student"]["lateral"]
        rows.append(row)
        if (i + 1) % 1000 == 0:
            print(f"[{i + 1}/{N}] {time.time() - t0:.0f}s", flush=True)

    out = dict(mode="gt_sanity" if a.gt_sanity else "student", results=a.results, infos=a.infos, n_val=len(infos), n_rows=len(rows),
               n_teacher_intent_na=n_teacher_na, thresholds=lab.thresholds(),
               definition="student intent = same labeling function as teacher GT intent (xai_common.IntentLabeler)")
    if a.gt_sanity:
        R = [r for r in rows if r["gt"] is not None]
        out["gt_traj_vs_gt_label"] = rates([dict(r, lbl=r["student"]) for r in R], "gt", "lbl")
        out["gt_traj_vs_gt_label_certain"] = rates([dict(r, lbl=r["student"]) for r in R if not r["gt_uncertain"]], "gt", "lbl")
        bad = [r for r in R if r["gt"] != r["student"]]
        out["n_mismatch"] = len(bad)
        out["mismatch_examples"] = bad[:20]
    else:
        out["teacher_vs_student_all"] = rates(rows, "teacher", "student")
        fc = [r for r in rows if r["future_complete"]]
        out["teacher_vs_student_future_complete"] = rates(fc, "teacher", "student")
        out["teacher_vs_student_certain"] = rates([r for r in fc if not r["gt_uncertain"] and not r["student_evidence"]["uncertain"]],
                                                  "teacher", "student")
        out["by_command"] = {c: rates([r for r in rows if r["command"] == c], "teacher", "student")
                             for c in ("go straight", "turn left", "turn right")}
        out["reference_student_vs_gt"] = rates(fc, "gt", "student")
        out["reference_teacher_vs_gt"] = rates(fc, "gt", "teacher")
        # 교사가 맞았을 때 / 틀렸을 때 학생 일치
        tc = [r for r in fc if r["teacher"] is not None and r["gt"] is not None]
        out["agree_when_teacher_correct"] = rates([r for r in tc if r["teacher"] == r["gt"]], "teacher", "student")
        out["agree_when_teacher_wrong"] = rates([r for r in tc if r["teacher"] != r["gt"]], "teacher", "student")
        dis = [r for r in rows if r["teacher"] is not None and not (r["agree_lon"] and r["agree_lat"])]
        opp = lambda r: ("left" in r["teacher"]["lateral"] and "right" in r["student"]["lateral"]) or \
            ("right" in r["teacher"]["lateral"] and "left" in r["student"]["lateral"])  # noqa: E731
        dis.sort(key=lambda r: (not opp(r), r["agree_lat"], r["agree_lon"], r["idx"]))
        with open(os.path.join(a.out_dir, "disagreements.jsonl"), "w") as f:
            for r in dis:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out["n_disagreements"] = len(dis)
    with open(os.path.join(a.out_dir, "per_sample.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(out, open(os.path.join(a.out_dir, "metrics.json"), "w"), indent=1, ensure_ascii=False)
    write_md(out, os.path.join(a.out_dir, "summary.md"))
    print(open(os.path.join(a.out_dir, "summary.md")).read())
    print(f"[완료] {a.out_dir} ({time.time() - t0:.0f}s)")


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def write_md(o, path):
    L = [f"# 설명–행동 일치도 ({o['mode']})", "", f"표본 {o['n_rows']} / val {o['n_val']} · 교사 의도 없음(<na>) {o['n_teacher_intent_na']}",
         "", "의도 정의 = 교사 GT 라벨 함수 그대로: " + o["thresholds"]["source"],
         f"임계값: stop v_end<{o['thresholds']['STOP_V']} m/s · |Δv|≥{o['thresholds']['DV_TH']} m/s · 회전 ±{o['thresholds']['TURN_TH_DEG']}° · "
         f"최소 경로 {o['thresholds']['MIN_PATH']} m · 차로변경 = lane_v2 세 신호 다수결", "",
         "| 비교 | n | 종방향 일치 | 횡방향 일치 | 결합 일치 | 좌우 반대 |", "|---|---|---|---|---|---|"]
    keys = ["gt_traj_vs_gt_label", "gt_traj_vs_gt_label_certain"] if o["mode"] == "gt_sanity" else \
        ["teacher_vs_student_all", "teacher_vs_student_future_complete", "teacher_vs_student_certain", "agree_when_teacher_correct",
         "agree_when_teacher_wrong", "reference_student_vs_gt", "reference_teacher_vs_gt"]
    for k in keys:
        r = o.get(k) or {}
        L.append(f"| {k} | {r.get('n', 0)} | {pct(r.get('lon_agree'))} | {pct(r.get('lat_agree'))} | {pct(r.get('joint_agree'))} | "
                 f"{pct(r.get('lat_opposite_direction'))} |")
    if o["mode"] != "gt_sanity":
        L += ["", "| 명령 | n | 종방향 | 횡방향 | 결합 |", "|---|---|---|---|---|"]
        for c, r in o["by_command"].items():
            L.append(f"| {c} | {r.get('n', 0)} | {pct(r.get('lon_agree'))} | {pct(r.get('lat_agree'))} | {pct(r.get('joint_agree'))} |")
        L += ["", f"불일치 표본 {o['n_disagreements']} → disagreements.jsonl (좌우 반대 먼저)"]
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
