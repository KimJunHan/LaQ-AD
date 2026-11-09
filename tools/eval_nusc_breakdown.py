"""수정(2026-10-09): nuScenes 평가 분해 — 날씨·조명별(BEVCar Table IV 형식) / 거리 구간별(BEVCar Table III 형식).

  프로토콜은 논문 Table I 과 같다(재현 검증을 먼저 수행):
    - 검출: nuScenes devkit DetectionEval(detection_cvpr_2019) 의 GT/예측 적재·필터(add_center_dist, 클래스별 거리 필터)
      를 그대로 쓰고, 부분집합에 대해 devkit accumulate/calc_ap 로 클래스 AP(거리 임계 0.5/1/2/4 m 평균)를 낸다.
    - 지도: 프로젝트 VectorEvaluate(chamfer 0.5/1.0/1.5) 의 GT 수집·표본별 매칭(_evaluate_single_worker)·AP 계산을 그대로 쓴다.
      매칭이 표본 단위이므로 표본 부분집합(날씨)은 전체 매칭 결과를 그 표본만 모아 집계한 것과 같다.
  분할·필터 규칙(보고서에 명시):
    - 날씨·조명: scene['description'] 소문자에 'night' 포함 → Night, 아니고 'rain' 포함 → Rain, 그 외 → Day
      (night 와 rain 이 함께 있는 장면은 Night 로 배정하고 개수를 보고).
    - Vehicle = car, truck, bus, trailer, construction_vehicle 클래스 AP 평균(+ car 단독). 부분집합에 GT 가 없는 클래스는
      AP 가 정의되지 않으므로(devkit 은 0 반환) 평균에서 제외하고 그 사실을 기록.
    - 거리: 자차 좌표계 박스 중심 수평거리(devkit ego_dist)로 GT·예측 모두 [lo, hi) 구간 필터. devkit 클래스 거리 필터(50 m 등) 유지.
    - 지도 거리: 폴리라인 점들의 평균(중심) 거리로 GT·예측 모두 구간 배정 후 재매칭. 지도 ROI 가 30×60 m 라 35 m 너머는 비어 있다.
  사용: CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tools/eval_nusc_breakdown.py
"""
import argparse, json, os, importlib, time
from functools import partial
from multiprocessing import get_context
import numpy as np

VEH = ["car", "truck", "bus", "trailer", "construction_vehicle"]
RANGES = [(0, 50), (0, 20), (20, 35), (35, 50)]


def det_ap(gt, pred, cls, cfg):
    from nuscenes.eval.detection.algo import accumulate, calc_ap
    npos = sum(1 for b in gt.all if b.detection_name == cls)
    if npos == 0:
        return None
    aps = [calc_ap(accumulate(gt, pred, cls, cfg.dist_fcn_callable, d), cfg.min_recall, cfg.min_precision)
           for d in cfg.dist_ths]
    return float(np.mean(aps))


def sub_boxes(eb, tokens, rng=None):
    from nuscenes.eval.common.data_classes import EvalBoxes
    out = EvalBoxes()
    for t in tokens:
        bx = eb[t]
        if rng is not None:
            bx = [b for b in bx if rng[0] <= b.ego_dist < rng[1]]
        out.add_boxes(t, bx)
    return out


def veh_summary(gt, pred, cfg):
    per = {c: det_ap(gt, pred, c, cfg) for c in VEH}
    valid = [v for v in per.values() if v is not None]
    return dict(vehicle_AP=float(np.mean(valid)) if valid else None, car_AP=per["car"],
                per_class=per, classes_without_gt=[c for c, v in per.items() if v is None])


def map_ap_from_tpfp(tpfp_by_cls, num_gts, thresholds, average_precision):
    res = {}
    for label, lst in tpfp_by_cls.items():
        if num_gts[label] == 0:
            res[label] = None; continue
        aps = []
        for thr in thresholds:
            arr = np.vstack([x[thr] for x in lst]) if lst else np.zeros((0, 3))
            o = np.argsort(-arr[:, -1]); tp = np.cumsum(arr[o, 0]); fp = np.cumsum(arr[o, 1])
            eps = np.finfo(np.float32).eps
            aps.append(average_precision(tp / max(num_gts[label], eps), tp / np.maximum(tp + fp, eps), "area"))
        res[label] = float(np.mean(aps))
    valid = [v for v in res.values() if v is not None]
    return dict(per_class=res, mAP=float(np.mean(valid)) if valid else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reference/hipad_nusc_vla_s2r9.py")  # 수정(2026-10-15): config 경로 이동
    ap.add_argument("--eval-dir", default="evaluation/hipad_nusc_vla_s2r9")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out_dir = os.path.join(a.eval_dir, "paper_diag"); os.makedirs(out_dir, exist_ok=True)
    importlib.import_module("mmdet3d_plugin")
    from mmcv import Config
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.detection.config import config_factory
    cfg_m = Config.fromfile(a.config)
    t0 = time.time()

    # ---------------- 검출 ----------------
    nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuscenes", verbose=False)
    dcfg = config_factory("detection_cvpr_2019")
    ev = DetectionEval(nusc, config=dcfg, result_path=os.path.join(a.eval_dir, "results_nusc.json"),
                       eval_set="val", output_dir=os.path.join(out_dir, "_det_tmp"), verbose=False)
    gt_all, pr_all = ev.gt_boxes, ev.pred_boxes
    tokens = list(gt_all.sample_tokens)
    full = {c: det_ap(gt_all, pr_all, c, dcfg) for c in dcfg.class_names}
    det_full_map = float(np.mean([v for v in full.values()]))
    print(f"[재현] 검출 mAP {det_full_map:.4f} (보고값 0.4206)  {time.time()-t0:.0f}s", flush=True)

    # 날씨 분할
    split, both = {}, 0
    for t in tokens:
        desc = nusc.get("scene", nusc.get("sample", t)["scene_token"])["description"].lower()
        night, rain = "night" in desc, "rain" in desc
        both += int(night and rain)
        split[t] = "Night" if night else ("Rain" if rain else "Day")
    weather = {"rule": "description 소문자 'night'→Night, 아니면 'rain'→Rain, 그 외 Day; night&rain 은 Night",
               "night_and_rain_samples": both, "det_full_mAP": det_full_map}
    for w in ("Day", "Rain", "Night"):
        tk = [t for t in tokens if split[t] == w]
        scenes = {nusc.get("sample", t)["scene_token"] for t in tk}
        weather[w] = dict(n_samples=len(tk), n_scenes=len(scenes),
                          **veh_summary(sub_boxes(gt_all, tk), sub_boxes(pr_all, tk), dcfg))
        print(f"[날씨] {w}: 표본 {len(tk)} 장면 {len(scenes)} vehicle {weather[w]['vehicle_AP']}", flush=True)

    rng_res = {"rule": "자차 좌표 박스 중심 수평거리(ego_dist)로 GT·예측 모두 [lo,hi) 필터, devkit 클래스 거리 필터 유지"}
    for lo, hi in RANGES:
        rng_res[f"{lo}-{hi}"] = veh_summary(sub_boxes(gt_all, tokens, (lo, hi)), sub_boxes(pr_all, tokens, (lo, hi)), dcfg)
        print(f"[거리] {lo}-{hi}: vehicle {rng_res[f'{lo}-{hi}']['vehicle_AP']}", flush=True)
    del ev

    # ---------------- 지도 ----------------
    from mmdet3d_plugin.models.loader.evaluation.map.vector_eval import VectorEvaluate, _evaluate_single_worker
    from mmdet3d_plugin.models.loader.evaluation.map.AP import average_precision
    mev = VectorEvaluate(cfg_m.eval_config, n_workers=a.workers)
    gts = mev.gts
    preds = json.load(open(os.path.join(a.eval_dir, "submission_vector.json")))["results"]
    labels = list(mev.id2cat.keys()); thr = mev.thresholds
    pool = get_context("spawn").Pool(a.workers)
    fn = partial(_evaluate_single_worker, thresholds=thr, metric="chamfer")

    def per_sample_tpfp(tok_list, vfilter=None):
        """표본별·클래스별 tp/fp/score 와 GT 수. vfilter(vec)->bool 이면 GT·예측 모두 그 조건으로 거른다."""
        jobs, keys, ngt = [], [], {l: 0 for l in labels}
        for t in tok_list:
            g = gts[t]; p = preds.get(t, {"vectors": [], "scores": [], "labels": []})
            for l in labels:
                pv = [(v, s) for v, s, lb in zip(p["vectors"], p["scores"], p["labels"]) if lb == l
                      and (vfilter is None or vfilter(np.asarray(v)))]
                gv = [v for v in g[l] if vfilter is None or vfilter(np.asarray(v))]
                ngt[l] += len(gv)
                jobs.append(([x[0] for x in pv], [x[1] for x in pv], gv)); keys.append((t, l))
        outs = pool.starmap(fn, jobs, chunksize=64)
        by = {}
        for (t, l), o in zip(keys, outs):
            by.setdefault(t, {})[l] = o
        return by, ngt

    by_tok, _ = per_sample_tpfp(tokens)
    def agg(tok_list, by, ngt_override=None):
        tpfp = {l: [by[t][l] for t in tok_list] for l in labels}
        ngt = ngt_override or {l: sum(len(gts[t][l]) for t in tok_list) for l in labels}
        return map_ap_from_tpfp(tpfp, ngt, thr, average_precision), ngt
    m_full, _ = agg(tokens, by_tok)
    print(f"[재현] 지도 mAP {m_full['mAP']:.4f} (보고값 0.5491)  {time.time()-t0:.0f}s", flush=True)
    weather["map_full_mAP"] = m_full["mAP"]
    for w in ("Day", "Rain", "Night"):
        tk = [t for t in tokens if split[t] == w]
        weather[w]["map"], _ = agg(tk, by_tok)
        print(f"[날씨] {w}: map {weather[w]['map']['mAP']:.4f}", flush=True)

    # 좌표계·범위 확인(지도 거리 구간 배정 근거)
    sample_pts = np.concatenate([np.asarray(v) for t in tokens[:200] for l in labels for v in gts[t][l]] or [np.zeros((1, 2))])
    rng_res["map_rule"] = "폴리라인 점 평균(중심)의 원점 거리로 GT·예측 모두 [lo,hi) 배정 후 재매칭; ROI 30x60 m"
    rng_res["map_gt_coord_abs_max"] = np.abs(sample_pts).max(0).tolist()
    for lo, hi in RANGES:
        vf = (lambda lo, hi: (lambda v: lo <= float(np.linalg.norm(v.mean(0))) < hi))(lo, hi)
        by_r, ngt_r = per_sample_tpfp(tokens, vf)
        rng_res[f"{lo}-{hi}"]["map"], _ = agg(tokens, by_r, ngt_r)
        rng_res[f"{lo}-{hi}"]["map_num_gts"] = ngt_r
        print(f"[거리] {lo}-{hi}: map {rng_res[f'{lo}-{hi}']['map']['mAP']}", flush=True)
    pool.close()

    json.dump(weather, open(os.path.join(out_dir, "weather.json"), "w"), indent=1)
    json.dump(rng_res, open(os.path.join(out_dir, "range.json"), "w"), indent=1)
    f = lambda x: "–" if x is None else f"{100*x:.1f}"
    md = ["# Run10 평가 분해 (CPU, tools/eval_nusc_breakdown.py)", "",
          f"재현: 검출 mAP {det_full_map:.4f} (보고 0.4206), 지도 mAP {m_full['mAP']:.4f} (보고 0.5491)", "",
          "## 날씨·조명 (Vehicle AP / Map mAP, %)", "| | Day | Rain | Night |", "|---|---|---|---|",
          "| 표본(장면) | " + " | ".join(f"{weather[w]['n_samples']} ({weather[w]['n_scenes']})" for w in ("Day", "Rain", "Night")) + " |",
          "| Vehicle | " + " | ".join(f(weather[w]['vehicle_AP']) for w in ("Day", "Rain", "Night")) + " |",
          "| Car | " + " | ".join(f(weather[w]['car_AP']) for w in ("Day", "Rain", "Night")) + " |",
          "| Map | " + " | ".join(f(weather[w]['map']['mAP']) for w in ("Day", "Rain", "Night")) + " |", "",
          "## 거리 구간 (%)", "| | " + " | ".join(f"{lo}-{hi} m" for lo, hi in RANGES) + " |", "|---|" + "---|" * len(RANGES),
          "| Vehicle | " + " | ".join(f(rng_res[f'{lo}-{hi}']['vehicle_AP']) for lo, hi in RANGES) + " |",
          "| Car | " + " | ".join(f(rng_res[f'{lo}-{hi}']['car_AP']) for lo, hi in RANGES) + " |",
          "| Map | " + " | ".join(f(rng_res[f'{lo}-{hi}']['map']['mAP']) for lo, hi in RANGES) + " |"]
    open(os.path.join(out_dir, "breakdown.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md)); print(f"총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
