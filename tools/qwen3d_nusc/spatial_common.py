# 수정(2026-09-21): Qwen 3D 파인튜닝 공용 모듈 — 프롬프트·캡션 파서·매끄러운 검증가능 보상.
#   보상 설계 근거: SpaceR(절대좌표 보상)·Smooth-Operator(SNRA, 연속 수치 보상) 2025 레시피.
#   파서는 make_spatial_captions.py 의 결정적 포맷과 1:1 대응.
import os, re, json, pickle
from PIL import Image
import numpy as np

SPATIAL_INSTRUCTION = (
    "These are 6 surround-view camera images of an autonomous vehicle in order: "
    "front, front-left, front-right, back-left, back, back-right. "
    "Report the ego speed, then list surrounding objects nearest first as "
    "'<sector> <distance>m <class> moving <direction> <speed>m/s' or "
    "'<sector> <distance>m <class> stationary'. Sectors: front, front-right, right, "
    "back-right, back, back-left, left, front-left."
)

CAMS = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT"]

def load_six_cams(info, data_root):  # tools/qwen_cache.py 와 동일 규약(캐시와 입력 정합 유지)
    imgs = []
    for cam in CAMS:
        p = info["cams"][cam]["data_path"]
        if "samples/" in p: p = "samples/" + p.split("samples/", 1)[1]
        if not os.path.isabs(p): p = os.path.join(data_root, p)
        # 수정(2026-10-02): 누락 이미지를 검은 화면으로 대체하면 깊이 지도는 실제 라이다
        #   값을 요구해 학습 불가 타깃이 되고, 캐시 생성 시엔 '내용 없는 토큰에 정답 깊이'가
        #   붙어 카메라 전용 파이프라인이 가질 수 없는 정보가 샌다. 조용한 skip 금지 규칙에
        #   따라 즉시 실패시킨다.
        if not os.path.exists(p):
            raise FileNotFoundError(f"카메라 이미지 누락: {p} (조용한 대체 금지)")
        imgs.append(Image.open(p).convert("RGB"))
    return imgs

def load_infos_by_token(pkl_path):
    with open(pkl_path, "rb") as f:
        infos = pickle.load(f)["infos"]
    return {i["token"]: i for i in infos}

_SEC = r"(front-right|front-left|back-right|back-left|front|back|right|left)"
_OBJ = re.compile(_SEC + r"\s+(\d+(?:\.\d+)?)m\s+([A-Za-z_]+)\s+(?:moving\s+" + _SEC +
                  r"\s+(\d+(?:\.\d+)?)m/s|stationary)")
_EGO = re.compile(r"[Ee]go speed\s+(\d+(?:\.\d+)?)\s*m/s")

def parse_caption(text):
    """반환: (ego_speed|None, [{sector,dist,name,vel_sector|None,speed}])"""
    m = _EGO.search(text)
    ego = float(m.group(1)) if m else None
    objs = []
    for g in _OBJ.finditer(text):
        objs.append(dict(sector=g.group(1), dist=float(g.group(2)), name=g.group(3),
                         vel_sector=g.group(4), speed=float(g.group(5)) if g.group(5) else 0.0))
    return ego, objs

SECTORS = ["front","front-right","right","back-right","back","back-left","left","front-left"]
def _sec_dist(a, b):  # 원형 8방위 거리 0~4
    d = abs(SECTORS.index(a) - SECTORS.index(b)); return min(d, 8 - d)

def smooth_reward(gen_text, gt, sigma_d=2.0, sigma_v=1.0):
    """SNRA 식 연속 보상. gt = jsonl 의 {ego_speed, objects:[...]}. 반환 [0,1] 근방 스칼라.

    수정(2026-09-28, 근본수정): 1차 GRPO 가 보상은 올리면서(0.413→0.442) 정작 목표인
      거리·속도 정밀도를 악화시켰다(거리 MAE 3.62→4.29m, 속도 1.30→1.48). 원인은 보상
      설계 — (i) sigma_d=5m 가 너무 관대해 3.6m 와 4.3m 오차의 보상 차이가 13% 뿐인 반면
      (ii) 누락 GT 는 0 점이라 "더 많이 나열"하는 쪽의 이득이 훨씬 컸다. 즉 정책은 정확도
      대신 재현율을 샀다. 이는 정책의 실패가 아니라 목적함수 오설계이므로 목적함수를 고친다.
      (a) sigma_d 5.0->2.0, sigma_v 2.0->1.0 으로 정확도 기울기를 세우고
      (b) 예측 객체 중 GT 와 맞은 비율(정밀도) 항을 0.30 가중으로 신설해 남발을 직접 벌한다.

    항: 포맷 / 자차속도 / GT 커버리지(r_match) / 예측 정밀도(r_prec).
    """
    ego, objs = parse_caption(gen_text)
    gt_objs = gt["objects"]
    r_fmt = 1.0 if (ego is not None and objs) else 0.0
    r_ego = float(np.exp(-abs((ego or 0.0) - gt["ego_speed"]) / sigma_v)) if ego is not None else 0.0
    used = {}; scores = []
    for g in gt_objs:
        best, bi = 0.0, None
        for i, p in enumerate(objs):
            if i in used or p["name"] != g["name"].split(".")[-1]: continue
            s = (np.exp(-abs(p["dist"] - g["dist"]) / sigma_d)
                 * np.exp(-_sec_dist(p["sector"], g["sector"]) / 2.0)
                 * (0.5 + 0.5 * np.exp(-abs(p["speed"] - g["speed"]) / sigma_v)))
            if s > best: best, bi = s, i
        if bi is not None: used[bi] = best
        scores.append(best)
    r_match = float(np.mean(scores)) if scores else 0.0
    # 정밀도: 예측 객체 중 GT 와 "일대일로" 충분히 잘 맞은 것의 비율.
    #   수정(2026-09-28): GT 재사용을 허용하면 같은 클래스 근처 거리를 남발해도 만점이 되어
    #   억제가 되지 않는다(검증에서 확인) → r_match 의 일대일 매칭 결과를 그대로 쓰고
    #   매칭 품질 0.3 이상만 인정한다.
    r_prec = (sum(1 for q in used.values() if q >= 0.3) / len(objs)) if objs else 0.0
    return 0.05 * r_fmt + 0.10 * r_ego + 0.55 * r_match + 0.30 * r_prec

def metric_errors(gen_text, gt):
    """검증용: 매칭된 쌍의 거리 MAE·속도 MAE·방위 일치율·매칭 재현율."""
    _, objs = parse_caption(gen_text)
    used = set(); de, ve, sec_ok, n = [], [], 0, 0
    for g in gt["objects"]:
        best, bi = None, None
        for i, p in enumerate(objs):
            if i in used or p["name"] != g["name"].split(".")[-1]: continue
            if best is None or abs(p["dist"] - g["dist"]) < abs(best["dist"] - g["dist"]):
                best, bi = p, i
        if bi is None: continue
        used.add(bi); n += 1
        de.append(abs(best["dist"] - g["dist"])); ve.append(abs(best["speed"] - g["speed"]))
        sec_ok += (best["sector"] == g["sector"])
    N = max(1, len(gt["objects"]))
    return dict(dist_mae=float(np.mean(de)) if de else None,
                speed_mae=float(np.mean(ve)) if ve else None,
                sector_acc=sec_ok / max(1, n), recall=n / N)
