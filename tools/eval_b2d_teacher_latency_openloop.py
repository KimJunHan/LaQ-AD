# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
"""B2D 교사 출력 지연(나이) 개방루프 비교 — 지연 0 vs 0.4 s (사용자 결정 2026-10-16: "2 Hz 갱신·직전 출력 재사용,
지연 0 vs 0.4초 오픈 루프 비교로 판단").

[배경] 학습·폐쇄루프 규칙(수정 2026-10-19: clip 첫 프레임 기준): clip 별 f0 = infos 의 첫 frame_idx, 위상 p(f) = (f − f0) % 5,
  keyframe kf(f) = f0 + 5·((f − f0)//5) (나이 (f−kf)×0.1 = 0~0.4 s) — models/encoder/vl_tokens_cache/loading_b2d.py b2d_keyframes 와 같은 규칙.
  교사 캐시는 keyframe(위상 0)에만 있다. 그래서 keyframe 위에서는 "0.4 s 묵은 출력"이 존재하지 않는다(가장 가까운 과거
  keyframe 은 0.5 s 전). '지연 0 vs 0.4 s'를 정확히 재려면 비교 대상 프레임에 "새(나이 0) 교사 출력"이 있어야 한다.

[비교 정의 — 측정 전에 고정]
  D1 (정확한 정의, 권장): 평가 프레임 = val 의 위상 4 프레임 전부(공식 val 50 clip).
      A = 그 프레임 자신의 교사 출력(나이 0)   ← val 전용 추가 캐시 필요(위상 4 프레임 약 2.5천 개, GPU 교사 추출).
      B = 학습 규칙 그대로 kf(f)=f−4 의 출력(나이 0.4 s).
      학생 모델·체크포인트·나머지 입력은 같다. 다른 위상 프레임은 점수에 넣지 않는다(시간 메모리는 위상별 흐름이라
      그 프레임들이 위상 4 흐름의 메모리에 섞이지 않는다 — B2D 데이터셋 split_group=5 재배열: 그룹 하나 = clip 안 같은 위상).
  D2 (추가 캐시 없이 가능한 상한): 평가 프레임 = val keyframe 중 같은 clip 에 f−5 가 있는 것.
      A = kf 자신의 출력(나이 0), B = f−5 keyframe 출력(나이 0.5 s — 0.4 s 보다 나쁜 쪽의 상한).
      D2 에서 '작은 차이'면 0.4 s 도 작다고 볼 수 있지만, 크다고 나오면 D1 이 필요하다.
  부가(기술 통계): 학습 규칙 그대로 돌린 한 번의 결과에서 나이 구간(0, 0.1, 0.2, 0.3, 0.4 s = 위상)별 지표와
      (구간 k − 구간 0) 차이. 프레임 내용도 위상별로 다르므로 인과 비교가 아니라 참고값이다.
  지표: 기존 B2D 개방루프 평가(models/head/sparse_head.py → dataset.evaluate)의 표본별 metric_results
      plan_L2_{1,2,3}s(m), plan_obj_box_col_{1,2,3}s(비율). fut_valid_flag 가 참인 프레임만(평가기와 같은 규칙).
      L2_avg = 세 시점 평균, Col_avg = 세 시점 평균(%).
  통계: 짝지은 차이 Δ = B − A(같은 프레임). clip 단위 짝지은 bootstrap(clip 을 복원 추출, 10,000 회, seed 0),
      재표본의 프레임 가중 평균 Δ 의 2.5 / 97.5 백분위 = 95 % CI.

[작은 차이 기준 — 측정 전에 고정(이 파일 상수가 기록)]
  DELTA_L2 = 0.05 m, DELTA_COL = 0.10 %p (L2_avg, Col_avg 각각).
  판정: 두 지표 모두 Δ 의 95 % CI 가 [−δ, +δ] 안 → "작은 차이"(동등성; 2 Hz 갱신 채택 유지).
        어느 하나라도 CI 하한 > +δ → "의미 있게 나빠짐"(갱신 주기 재검토).
        그 밖 → "판정 불가"(표본·잡음 부족 — val 전체가 이미 쓰였으므로 결론 보류, 사용자 보고).
  근거: (1) 개방루프 L2 는 표마다 소수 둘째 자리(cm 단위)로 보고되고, 논문에서 개선이라 주장할 최소 차이를 0.05 m 로
        보면 그보다 작은 지연 효과는 결론을 바꾸지 않는다. (2) 0.05 m 는 B2D 평균 속도 수 m/s 에서 수 cm/틱 수준의 자차 이동량과
        같은 크기라 '한 틱 어긋남'보다 작은 영향이다. (3) 충돌률은 B2D 개방루프에서 1 % 안팎의 작은 값이라 0.1 %p(상대 약 10 %)
        를 상한으로 둔다. 이 값들은 측정 전에 사용자 확인을 받아야 한다(열린 질문).
  0.4 s 가 아니라 D2(0.5 s)로 판정할 때도 같은 δ 를 쓴다.

[사용]
  # 0) (GPU, 나중) 결과 pkl 2 개 만들기 — 캐시 보기(view) 폴더를 만들어 같은 config 의 teacher_cache_root 만 바꿔 tools/test.py --out
  python tools/eval_b2d_teacher_latency_openloop.py build_view --definition D1 --cond A \
      --cache_root data/bench2drive/laq_cache --extra_root data/bench2drive/laq_cache_val_phase4 --out_root /tmp/view_D1_A
  # 1) 판정
  python tools/eval_b2d_teacher_latency_openloop.py score --definition D1 --result_a a.pkl --result_b b.pkl
  # 2) CPU 시험(합성 지표, 실제 val clip·프레임 구조 사용)
  python tools/eval_b2d_teacher_latency_openloop.py dry_run
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

DELTA_L2 = 0.05          # m   — 측정 전 고정(머리말 근거)
DELTA_COL = 0.10         # %p  — 측정 전 고정
N_BOOT = 10000
SEED = 0
STRIDE = 5               # 10 Hz → 2 Hz keyframe
SPLIT_GROUP = 5          # Bench2DriveDataset 기본값(configs/b2d/* 가 덮어쓰지 않음) — 결과 pkl 순서 복원에 사용
VAL_INFOS = "data/infos/b2d_infos_val.pkl"
FOLDERS = ("context", "detection", "tracking", "motion", "map", "plan", "action")


def clip_of(info):
    return os.path.basename(str(info["folder"]).rstrip("/"))


def dataset_order(infos, split_group=SPLIT_GROUP):
    """Bench2DriveDataset._split_data_infos 와 같은 재배열(i::split_group 이어 붙임) → 결과 pkl 의 i 번째 = 반환 목록 i 번째."""
    if split_group <= 0:
        return list(infos)
    out = []
    for i in range(split_group):
        out.extend(infos[i::split_group])
    return out


def frame_table(infos_ordered):
    """[(clip, frame_idx)] + clip 별 프레임 집합."""
    rows = [(clip_of(x), int(x["frame_idx"])) for x in infos_ordered]
    per = {}
    for c, f in rows:
        per.setdefault(c, set()).add(f)
    return rows, per


def clip_f0(per_clip):
    """clip → 첫 frame_idx (loading_b2d.b2d_keyframes 의 f0)."""
    return {c: min(fs) for c, fs in per_clip.items()}


def phase(c, f, f0):
    return (f - f0[c]) % STRIDE


def kf_of(c, f, f0):
    return f0[c] + STRIDE * ((f - f0[c]) // STRIDE)


def per_frame_metrics(results):
    """tools/test.py 결과 list → (L2_avg m, Col_avg %, valid) 배열."""
    L2, COL, V = [], [], []
    for r in results:
        m = r["metric_results"] if "metric_results" in r else r
        v = bool(m["fut_valid_flag"])
        L2.append(np.mean([float(m[f"plan_L2_{k}s"]) for k in (1, 2, 3)]))
        COL.append(100.0 * np.mean([float(m[f"plan_obj_box_col_{k}s"]) for k in (1, 2, 3)]))
        V.append(v)
    return np.asarray(L2), np.asarray(COL), np.asarray(V)


def population(rows, per_clip, definition):
    """평가 프레임 마스크(정의별, 측정과 무관하게 데이터 구조로만 정해짐)."""
    sel = []
    f0 = clip_f0(per_clip)
    for c, f in rows:
        p = phase(c, f, f0)
        if definition == "D1":
            sel.append(p == 4 and (f - 4) in per_clip[c])            # B(나이 0.4) 의 keyframe 이 val 에 있어야 함
        elif definition == "D2":
            sel.append(p == 0 and (f - STRIDE) in per_clip[c])
        else:
            raise ValueError(definition)
    return np.asarray(sel)


def paired_bootstrap(clips, delta, n_boot=N_BOOT, seed=SEED):
    """clips: 프레임별 clip 이름, delta: 프레임별 B−A. clip 복원 추출 → 프레임 가중 평균의 95 % CI."""
    uc = sorted(set(clips))
    idx = {c: i for i, c in enumerate(uc)}
    ci = np.asarray([idx[c] for c in clips])
    s = np.bincount(ci, weights=delta, minlength=len(uc))
    n = np.bincount(ci, minlength=len(uc)).astype(np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uc), size=(n_boot, len(uc)))
    boot = s[pick].sum(1) / n[pick].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(mean=float(delta.mean()), ci95=[float(lo), float(hi)], n_frames=int(len(delta)), n_clips=len(uc),
                n_boot=n_boot, seed=seed)


def decide(r_l2, r_col):
    def small(r, d):
        return -d <= r["ci95"][0] and r["ci95"][1] <= d

    def worse(r, d):
        return r["ci95"][0] > d
    if small(r_l2, DELTA_L2) and small(r_col, DELTA_COL):
        return "작은 차이(동등) — 2 Hz 갱신·재사용 유지"
    if worse(r_l2, DELTA_L2) or worse(r_col, DELTA_COL):
        return "의미 있게 나빠짐 — 갱신 주기 재검토"
    return "판정 불가(CI 가 기준을 걸침)"


def compare(rows, per_clip, a, b, definition):
    L2a, COLa, Va = a
    L2b, COLb, Vb = b
    pop = population(rows, per_clip, definition)
    if (Va != Vb).any():
        raise RuntimeError("A·B 의 fut_valid_flag 가 다르다 — 같은 GT·같은 프레임이어야 한다(결과 pkl 순서/분할 확인)")
    m = pop & Va
    clips = [rows[i][0] for i in np.nonzero(m)[0]]
    r_l2 = paired_bootstrap(clips, (L2b - L2a)[m])
    r_col = paired_bootstrap(clips, (COLb - COLa)[m])
    return dict(definition=definition, n_population=int(pop.sum()), n_valid=int(m.sum()),
                A=dict(L2_avg=float(L2a[m].mean()), Col_avg_pct=float(COLa[m].mean())),
                B=dict(L2_avg=float(L2b[m].mean()), Col_avg_pct=float(COLb[m].mean())),
                delta_L2=r_l2, delta_Col_pct=r_col, thresholds=dict(L2_m=DELTA_L2, Col_pp=DELTA_COL),
                verdict=decide(r_l2, r_col))


def _per(rows):
    per = {}
    for c, f in rows:
        per.setdefault(c, set()).add(f)
    return per


def age_buckets(rows, a):
    """학습 규칙 한 번의 결과에서 나이(위상 (f−f0)%5 × 0.1 s)별 지표와 (구간 k − 구간 0) 의 clip bootstrap(같은 clip 안 평균 차) — 참고값."""
    L2, COL, V = a
    out = {}
    f0 = clip_f0(_per(rows))
    ph = np.asarray([phase(c, f, f0) for c, f in rows])
    clips = np.asarray([c for c, _ in rows])
    for k in range(STRIDE):
        m = (ph == k) & V
        out[f"age_{k / 10:.1f}s"] = dict(n=int(m.sum()), L2_avg=float(L2[m].mean()), Col_avg_pct=float(COL[m].mean()))
    # clip 별 (구간 k 평균 − 구간 0 평균) → clip bootstrap
    rng = np.random.default_rng(SEED)
    uc = sorted(set(clips[V]))
    for k in range(1, STRIDE):
        d = []
        for c in uc:
            m0 = (clips == c) & (ph == 0) & V
            mk = (clips == c) & (ph == k) & V
            if m0.any() and mk.any():
                d.append(L2[mk].mean() - L2[m0].mean())
        d = np.asarray(d)
        boot = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(1)
        out[f"age_{k / 10:.1f}s"]["dL2_vs_age0_clipmean"] = dict(mean=float(d.mean()),
                                                                ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])],
                                                                n_clips=int(len(d)))
    return out


def load_results(p, n_expect):
    with open(p, "rb") as f:
        res = pickle.load(f)
    if isinstance(res, dict) and "bbox_results" in res:
        res = res["bbox_results"]
    if len(res) != n_expect:
        raise RuntimeError(f"{p}: 결과 {len(res)} != val 프레임 {n_expect} (전체 val 이어야 한다)")
    return res


def load_rows(val_infos):
    with open(val_infos, "rb") as f:
        infos = pickle.load(f)
    rows, per = frame_table(dataset_order(infos))
    return rows, per


# ---------------------------------------------------------------- 캐시 보기(view) 폴더 — GPU 평가 전 준비(심볼릭 링크)
def build_view(a):
    """학생 로더는 프레임 f 에 대해 키 '<clip>/<kf(f):05d>' 파일을 읽는다(학습 규칙). 조건별로 그 키가 가리킬 원본을 바꾼 링크 폴더를 만든다.
      D1-A: 키 kf → extra_root 의 '<clip>/<kf+4:05d>'(위상 4 프레임 자신의 출력)   (다른 위상 프레임은 점수 제외)
      D1-B, D2-A, 규칙: 키 kf → cache_root 의 같은 키(= 원래 캐시; 링크 없이 cache_root 를 그대로 써도 됨)
      D2-B: 키 kf → cache_root 의 '<clip>/<kf−5:05d>'(0.5 s 전 keyframe). 원본이 없으면(clip 첫 keyframe) 링크하지 않고 목록에 기록
            → 그 키를 읽는 프레임은 population(D2) 밖이지만, 로더는 파일이 없으면 즉시 실패하므로 원래 캐시로 링크한다(점수 제외)."""
    src_root, out_root = a.cache_root, a.out_root
    if os.path.exists(out_root) and os.listdir(out_root):
        raise SystemExit(f"[거부] {out_root} 비어 있지 않음")
    rows, per = load_rows(a.val_infos)
    plan, n_fallback = {}, 0
    for c, fs in per.items():
        f0c = min(fs)
        for kf in range(f0c, max(fs) + 1, STRIDE):               # loading_b2d.b2d_keyframes 와 같은 키프레임 목록
            key = f"{c}/{kf:05d}"
            if a.definition == "D1" and a.cond == "A":
                if (kf + 4) in fs:
                    plan[key] = (a.extra_root, f"{c}/{kf + 4:05d}")
                else:
                    plan[key] = (src_root, key); n_fallback += 1
            elif a.definition == "D2" and a.cond == "B":
                if (kf - STRIDE) in fs:
                    plan[key] = (src_root, f"{c}/{kf - STRIDE:05d}")
                else:
                    plan[key] = (src_root, key); n_fallback += 1
            else:
                plan[key] = (src_root, key)
    for fo in FOLDERS:
        for key, (root, sk) in plan.items():
            src = os.path.abspath(os.path.join(root, fo, sk + ".pt"))
            if not os.path.exists(src):
                raise FileNotFoundError(f"원본 캐시 없음: {src}")
            dst = os.path.join(out_root, fo, key + ".pt")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)
        meta = os.path.join(src_root, fo, "_meta.json")
        if os.path.exists(meta):
            os.symlink(os.path.abspath(meta), os.path.join(out_root, fo, "_meta.json"))
    # 색인: 원본 색인 줄을 키 이름만 바꿔 복사(layout_ok·constrained_decoding 등은 원본 값 그대로, source 기록)
    import glob
    idx = {}
    for root in {r for r, _ in plan.values()}:
        for fp in glob.glob(os.path.join(root, "_index", "teacher_*.jsonl")):
            for ln in open(fp):
                if ln.strip():
                    e = json.loads(ln); idx[(root, e["token"])] = e
    os.makedirs(os.path.join(out_root, "_index"), exist_ok=True)
    with open(os.path.join(out_root, "_index", "teacher_view.jsonl"), "w") as fo:
        for key, (root, sk) in plan.items():
            e = idx.get((root, sk))
            if e is None:
                raise RuntimeError(f"색인에 {sk} 없음({root})")
            fo.write(json.dumps(dict(e, token=key, view_source=f"{root}:{sk}"), ensure_ascii=False) + "\n")
    json.dump(dict(definition=a.definition, cond=a.cond, n_keys=len(plan), n_fallback_unscored=n_fallback,
                   cache_root=src_root, extra_root=a.extra_root), open(os.path.join(out_root, "_view.json"), "w"), indent=1)
    print(f"[view] {out_root}: 키 {len(plan)}, 점수 제외 키(원래 캐시 연결) {n_fallback}")


def score(a):
    rows, per = load_rows(a.val_infos)
    A = per_frame_metrics(load_results(a.result_a, len(rows)))
    B = per_frame_metrics(load_results(a.result_b, len(rows)))
    out = dict(compare=compare(rows, per, A, B, a.definition))
    if a.result_rule:
        out["age_buckets_rule"] = age_buckets(rows, per_frame_metrics(load_results(a.result_rule, len(rows))))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)


def dry_run(a):
    """합성 지표(실제 val clip·프레임 구조)로 bootstrap·판정 경로를 CPU 에서 실행. 숫자는 의미 없음."""
    rows, per = load_rows(a.val_infos)
    n = len(rows)
    rng = np.random.default_rng(123)
    clip_eff = {c: rng.normal(0, 0.15) for c in per}             # clip 별 난이도
    base = np.asarray([0.8 + clip_eff[c] for c, _ in rows]) + rng.normal(0, 0.3, n)
    base = np.abs(base)
    colA = (rng.random(n) < 0.01).astype(float)
    valid = rng.random(n) > 0.03
    res = {}
    for name, eff_l2, eff_col in (("no_effect", 0.0, 0.0), ("small_effect_0.01m", 0.01, 0.0), ("large_effect_0.15m", 0.15, 0.005)):
        L2b = base + eff_l2 + rng.normal(0, 0.05, n)
        colB = np.clip(colA + (rng.random(n) < eff_col), 0, 1)
        A = (base, 100 * colA, valid)
        B = (L2b, 100 * colB, valid)
        res[name] = {d: compare(rows, per, A, B, d) for d in ("D1", "D2")}
    res["age_buckets_demo"] = age_buckets(rows, (base, 100 * colA, valid))
    for name in ("no_effect", "small_effect_0.01m", "large_effect_0.15m"):
        for d in ("D1", "D2"):
            r = res[name][d]
            print(f"[dry] {name:20s} {d}: 표본 {r['n_valid']:5d}/{r['n_population']:5d}, ΔL2 {r['delta_L2']['mean']:+.4f} "
                  f"CI [{r['delta_L2']['ci95'][0]:+.4f}, {r['delta_L2']['ci95'][1]:+.4f}] m, ΔCol "
                  f"{r['delta_Col_pct']['mean']:+.3f} CI [{r['delta_Col_pct']['ci95'][0]:+.3f}, {r['delta_Col_pct']['ci95'][1]:+.3f}] %p "
                  f"→ {r['verdict']}")
    print(f"[dry] val 프레임 {n}, clip {len(per)}, 나이별 표본 " +
          ", ".join(f"{k}:{v['n']}" for k, v in res["age_buckets_demo"].items()))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build_view", "score", "dry_run"):
        p = sub.add_parser(name)
        p.add_argument("--val_infos", default=VAL_INFOS)
        p.add_argument("--out", default=None)
        if name == "build_view":
            p.add_argument("--definition", choices=("D1", "D2"), required=True)
            p.add_argument("--cond", choices=("A", "B"), required=True)
            p.add_argument("--cache_root", default="data/bench2drive/laq_cache")
            p.add_argument("--extra_root", default="data/bench2drive/laq_cache_val_phase4")
            p.add_argument("--out_root", required=True)
        if name == "score":
            p.add_argument("--definition", choices=("D1", "D2"), required=True)
            p.add_argument("--result_a", required=True)
            p.add_argument("--result_b", required=True)
            p.add_argument("--result_rule", default=None, help="학습 규칙 그대로의 전체 val 결과(나이별 참고표)")
    a = ap.parse_args(argv)
    dict(build_view=build_view, score=score, dry_run=dry_run)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
