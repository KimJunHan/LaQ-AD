# 수정: VLA-v0 — Bench2Drive stage2 결과를 hipad_vis.png 스타일로 시각화.
#   레이아웃: [ 6-cam 2x3 (front cam 에 ego plan 투영) | PRED BEV | GT BEV ]
#   PRED BEV : 예측 박스 + 예측 agent motion 궤적 + 예측 ego plan
#   GT   BEV : GT 박스   + GT  agent 미래 궤적   + GT  ego 미래 경로
# 입력 : config + dump_infer.py 로 만든 results.pkl (per-sample img_bbox).
# 좌표 : 모두 현재 frame LIDAR 프레임. lidar2world = inv(world2lidar).
import argparse, os, pickle, sys
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import cv2, numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mmdet3d_plugin  # noqa  registries
from mmcv import Config
from mmdet.datasets import build_dataset

# B2D 9 det class
B2D_CLASSES = ["car","van","truck","bicycle","traffic_sign","traffic_cone","traffic_light","pedestrian","others"]
CLS_COLOR = {  # BGR
    "car":(0,200,0), "van":(0,200,120), "truck":(0,160,200), "bicycle":(220,20,60),
    "traffic_sign":(0,0,230), "traffic_cone":(0,140,255), "traffic_light":(0,255,255),
    "pedestrian":(255,0,0), "others":(150,150,150),
}
EGO_PLAN_COLOR = (255, 0, 255)    # magenta = PRED ego 궤적
GT_EGO_COLOR = (0, 200, 0)        # 수정: green = GT ego 궤적 (pred 와 비교용, 모든 패널에 함께)
AGENT_TRAJ_COLOR = (255, 0, 0)    # 수정: 파란색 (BGR) — 주변차량 궤적 가시성 개선
# 수정: 맵 4 클래스 모두 구분 표기 (BGR). 파랑(agent)/빨강(box)/보라(ego)와 안 겹치게.
MAP_COLOR = {
    0: (170, 170, 170),   # Broken     - light gray
    1: (60, 60, 60),      # Solid      - dark gray
    2: (130, 130, 0),     # SolidSolid - teal
    3: (0, 150, 0),       # Center     - green
}


def inv(T): return np.linalg.inv(np.asarray(T))


def lidar_to_canvas(x, y, pcr, W, H):
    """pcr=30(x)x60(y) 비율 보존: x->가로(u), y->세로(v, 위가 forward)."""
    x_min, y_min, _, x_max, y_max, _ = pcr
    u = int(W * (x - x_min) / (x_max - x_min))
    v = int(H * (1 - (y - y_min) / (y_max - y_min)))
    return u, v


def draw_bev(boxes, labels, scores, classes, pcr, H=400, score_thr=0.3,
             ego_trajs=None, agent_trajs=None, title="", maps=None, map_legend=None):
    """ego-centric BEV. boxes:(N,>=7) lidar [x,y,z,w,l,h,yaw,...].
    maps: list of (poly_xy(N,2) lidar, label) — 차선."""
    x_min, _, _, x_max, _, _ = pcr
    y_min, y_max = pcr[1], pcr[4]
    W = max(1, int(round(H * (x_max - x_min) / (y_max - y_min))))  # 종횡비 보존
    cv = np.full((H, W, 3), 245, np.uint8)
    cx, cy = lidar_to_canvas(0, 0, pcr, W, H)

    # 수정: 맵 vector = 점(dot)을 한 줄로 연결 (hipad_mapdata.png). 가는 선 + 각 점에 dot.
    if maps:
        for poly, lab in maps:
            if poly is None or len(poly) < 1:
                continue
            mcol = MAP_COLOR.get(int(lab), (120, 120, 120))
            pts = np.array([lidar_to_canvas(p[0], p[1], pcr, W, H) for p in poly], np.int32)
            if len(pts) >= 2:
                cv2.polylines(cv, [pts], False, mcol, 1, cv2.LINE_AA)  # 점들을 한 줄로 연결
            for u, v in pts:
                cv2.circle(cv, (int(u), int(v)), 1, mcol, -1)          # dot 유지

    # 수정: ego = 빨간 박스 (원점, +y=forward). 크기 ~ (w 1.85, l 4.9) m.
    ew, el = 1.85, 4.9
    ego_c = np.array([[ ew/2, el/2],[ ew/2,-el/2],[-ew/2,-el/2],[-ew/2, el/2]])
    ego_pts = np.array([lidar_to_canvas(px, py, pcr, W, H) for px, py in ego_c], np.int32)
    cv2.fillPoly(cv, [ego_pts], (0, 0, 255))          # red fill
    cv2.line(cv, (cx, cy), lidar_to_canvas(0, el/2, pcr, W, H), (0, 0, 150), 1)  # heading

    for bi in range(len(boxes)):
        if scores is not None and scores[bi] < score_thr:
            continue
        lab = int(labels[bi])
        if not (0 <= lab < len(classes)):
            continue
        cname = classes[lab]
        color = (0, 0, 255)  # 수정: 주변차량/보행자 박스 = 빨강 hollow (hipad_mapdata.png 참조)
        x, y = float(boxes[bi][0]), float(boxes[bi][1])
        w, l, yaw = float(boxes[bi][3]), float(boxes[bi][4]), float(boxes[bi][6])
        corners = np.array([[ w/2, l/2],[ w/2,-l/2],[-w/2,-l/2],[-w/2, l/2]])
        c, s = np.cos(yaw), np.sin(yaw)
        corners = corners @ np.array([[c,-s],[s,c]]).T + np.array([x, y])
        pts = np.array([lidar_to_canvas(px, py, pcr, W, H) for px, py in corners], np.int32)
        cv2.polylines(cv, [pts], True, color, 1)
        fm = (corners[0] + corners[1]) / 2
        cv2.line(cv, lidar_to_canvas(x, y, pcr, W, H), lidar_to_canvas(*fm, pcr, W, H), color, 1)

    # agent 미래 궤적 (각 (T,2) lidar 절대좌표)
    if agent_trajs:
        for tr in agent_trajs:
            if tr is None or len(tr) < 1:
                continue
            pts = np.array([lidar_to_canvas(p[0], p[1], pcr, W, H) for p in tr], np.int32)
            cv2.polylines(cv, [pts], False, AGENT_TRAJ_COLOR, 1, cv2.LINE_AA)
            cv2.circle(cv, tuple(pts[-1]), 2, AGENT_TRAJ_COLOR, -1)

    # 수정: ego 궤적 = pred(분홍) + gt(초록) 둘 다 표기. ego_trajs: list of (traj(T,2), color).
    if ego_trajs:
        for traj, col in ego_trajs:
            if traj is None or len(traj) < 1:
                continue
            pts = np.array([lidar_to_canvas(0,0,pcr,W,H)] +
                           [lidar_to_canvas(p[0], p[1], pcr, W, H) for p in traj], np.int32)
            cv2.polylines(cv, [pts], False, col, 2, cv2.LINE_AA)
            cv2.circle(cv, tuple(pts[-1]), 3, col, -1)

    cv2.rectangle(cv, (0,0), (W-1,H-1), (180,180,180), 1)
    if title:
        cv2.putText(cv, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40,40,40), 2, cv2.LINE_AA)
    # 수정: 맵 클래스 범례 (모든 클래스 라벨 표기) — 우상단
    if map_legend:
        ly = 38
        for name, col in map_legend:
            cv2.line(cv, (W-72, ly-4), (W-58, ly-4), col, 2, cv2.LINE_AA)
            cv2.putText(cv, name, (W-54, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (40,40,40), 1, cv2.LINE_AA)
            ly += 13
    return cv


def box_corners_3d(center, size, yaw):
    w, l, h = float(size[0]), float(size[1]), float(size[2])
    x = np.array([w/2,w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2])
    y = np.array([l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2,l/2])
    z = np.array([-h/2,-h/2,-h/2,-h/2,h/2,h/2,h/2,h/2])
    pts = np.stack([x,y,z],1)
    c, s = np.cos(yaw), np.sin(yaw)
    pts = pts @ np.array([[c,-s,0],[s,c,0],[0,0,1]]).T + np.asarray(center)
    return pts


def project(pts3d, lidar2img):
    homo = np.concatenate([pts3d, np.ones((len(pts3d),1))],1)
    cam = homo @ lidar2img.T
    z = cam[:,2]
    if (z <= 0.1).any():
        return None
    return np.stack([cam[:,0]/z, cam[:,1]/z],1)


def draw_box_img(img, c2d, color, t=2):
    if c2d is None: return
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    H, W = img.shape[:2]; p = c2d.astype(int)
    for i,j in edges:
        a,b = p[i],p[j]
        if (0<=a[0]<W and 0<=a[1]<H) or (0<=b[0]<W and 0<=b[1]<H):
            cv2.line(img, tuple(a), tuple(b), color, t)


CAM_ORDER = ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT"]


def draw_traj_on_cam(img, lidar2img, traj, color, z=-1.6, t=3):
    """ego 궤적(lidar xy)을 카메라 이미지에 투영해 폴리라인으로."""
    if traj is None or len(traj) < 2:
        return
    p3 = np.concatenate([np.asarray(traj), np.full((len(traj), 1), z)], 1)
    p2 = project(p3, lidar2img)
    if p2 is None or len(p2) < 2:
        return
    for a in range(len(p2) - 1):
        cv2.line(img, tuple(p2[a].astype(int)), tuple(p2[a+1].astype(int)), color, t, cv2.LINE_AA)


def get_gt_futures(infos_scene, idx_in_scene, n_future=6, stride=5):
    """현재 frame lidar 기준 ego 미래 경로 + agent별 미래 궤적(dict id->(T,2)).
    #수정: B2D 10Hz. stride=5 -> 0.5s 간격 n_future 점 = pred temporal plan(plan_temp_2hz, 3s)과 시평 매칭."""
    cur = infos_scene[idx_in_scene]
    w2l_cur = np.asarray(cur["sensors"]["LIDAR_TOP"]["world2lidar"])
    fut = infos_scene[idx_in_scene+stride: idx_in_scene+1+n_future*stride: stride]
    # ego: 각 미래 frame ego world 위치 = inv(world2lidar_t) 의 translation 은 lidar원점의 world.
    ego = []
    for f in fut:
        l2w_t = inv(f["sensors"]["LIDAR_TOP"]["world2lidar"])
        p_world = l2w_t[:3, 3]
        p_cur = (w2l_cur @ np.array([*p_world, 1.0]))[:3]
        ego.append(p_cur[:2])
    ego = np.array(ego) if ego else None
    # agents: id 매칭. 각 미래 frame gt_boxes 는 그 frame lidar 좌표 -> world -> 현재 lidar.
    agent = {}
    cur_ids = list(cur["gt_ids"])
    for aid in cur_ids:
        pts = []
        for f in fut:
            ids = list(f["gt_ids"])
            if aid not in ids:
                break
            b = np.asarray(f["gt_boxes"])[ids.index(aid)]
            l2w_t = inv(f["sensors"]["LIDAR_TOP"]["world2lidar"])
            p_world = (l2w_t @ np.array([b[0], b[1], b[2], 1.0]))[:3]
            p_cur = (w2l_cur @ np.array([*p_world, 1.0]))[:3]
            pts.append(p_cur[:2])
        if len(pts) >= 2:
            agent[aid] = np.array(pts)
    return ego, list(agent.values())


QWEN_TRAJ_COLOR = (255, 255, 0)   # 수정: cyan = Qwen VLA 예측 ego 궤적


def _wrap_text(text, max_chars):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def make_qwen_panel(reason, H, W=300):
    """수정: Qwen VLA reasoning 우측 패널 (카메라+BEV 끝에 붙임)."""
    panel = np.full((H, W, 3), 30, np.uint8)
    cv2.putText(panel, "QWEN VLA", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.line(panel, (8, 32), (W - 8, 32), (0, 200, 200), 1)
    y, lh = 52, 19
    for ln in _wrap_text(reason or "(reasoning not available)", max_chars=W // 9):
        if y + lh > H - 6:
            break
        cv2.putText(panel, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        y += lh
    return panel


def build_qwen_predictor(qwen_ckpt, qwen_path, data_root, device="cuda"):
    """visualize_nusc.py 미러링 — 6 B2D cam → (traj (6,2) lidar, reasoning str)."""
    import torch as _t
    from mmdet3d_plugin.models.qwen_traj_generator import QwenTrajectoryGenerator
    from transformers import AutoProcessor
    from PIL import Image as _PIL
    qg = QwenTrajectoryGenerator(qwen_model_path=qwen_path, embed_dim=1536, traj_ts=6,
                                 traj_dim=2, freeze_qwen=True, torch_dtype="bf16",
                                 use_lora=False, gradient_checkpointing=False)
    ck = _t.load(qwen_ckpt, map_location="cpu")
    qg.traj_proj.load_state_dict(ck["traj_proj"])
    qg = qg.to(device).eval()
    proc = AutoProcessor.from_pretrained(qwen_path, min_pixels=200*28*28, max_pixels=200*28*28)
    INSTR = ("These are 6 surround-view camera images of an ego vehicle (front, front-left, "
             "front-right, back-left, back, back-right). Predict the ego vehicle's future "
             "trajectory for the next 3 seconds as 6 waypoints (0.5s spacing) in lidar frame meters.")
    REASON = ("You see 6 surround cameras of an ego vehicle. In ONE concise English sentence "
              "(under 25 words), explain the next driving decision and cite ONE specific visual "
              "cue (other vehicles, pedestrians, signs, lane, intersection).")

    def _imgs(info):
        out = []
        for c in CAM_ORDER:
            p = info["sensors"][c]["data_path"]
            p = os.path.join(data_root, p) if not os.path.isabs(p) else p
            out.append(_PIL.open(p).convert("RGB") if os.path.exists(p)
                       else _PIL.new("RGB", (1600, 900), (0, 0, 0)))
        return out

    def _fn(info):
        imgs = _imgs(info)
        m = [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs] +
              [{"type": "text", "text": INSTR}]}]
        txt = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[txt], images=imgs, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        with _t.no_grad():
            traj = qg(qwen_pixel_values=inp["pixel_values"], qwen_input_ids=inp["input_ids"],
                      qwen_attention_mask=inp["attention_mask"],
                      qwen_image_grid_thw=inp.get("image_grid_thw"))[0].float().cpu().numpy()
        reason = ""
        try:
            rm = [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs] +
                   [{"type": "text", "text": REASON}]}]
            rtxt = proc.apply_chat_template(rm, tokenize=False, add_generation_prompt=True)
            rinp = proc(text=[rtxt], images=imgs, return_tensors="pt", padding=True)
            rinp = {k: v.to(device) for k, v in rinp.items()}
            with _t.no_grad():
                gen = qg.qwen.generate(**rinp, max_new_tokens=50, do_sample=True,
                                       temperature=0.7, top_p=0.9, repetition_penalty=1.15)
            reason = proc.batch_decode(gen[:, rinp["input_ids"].shape[1]:],
                                       skip_special_tokens=True)[0].strip()
        except Exception as e:
            reason = f"(reason err: {type(e).__name__})"
        return traj, reason
    print(f"[viz] Qwen VLA loaded from {qwen_ckpt}")
    return _fn


def get_gt_map(info, map_infos, map_element_class, pcr, max_dist=50.0):
    """GT 맵 차선 -> [(poly_xy lidar, label)].
    #수정: get_map_info 가 쓰는 lane_points 는 일부 프레임에서 lidar 변환이 깨짐(범위 밖).
            거리필터에 쓰는 lane_sample_points 가 변환이 정상이므로 그걸 geometry 로도 사용."""
    mi = map_infos[info["town_name"]]
    w2l = np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"])
    ego_xy = inv(w2l)[:2, 3]
    sps = mi["lane_sample_points"]; types = mi["lane_types"]
    out = []
    for idx in range(len(sps)):
        sp = sps[idx]
        if np.min(np.linalg.norm(sp[:, :2] - ego_xy, axis=-1)) >= max_dist:
            continue
        lt = types[idx]
        if lt not in map_element_class:
            continue
        h = np.concatenate([sp[:, :3], np.ones((len(sp), 1))], 1)
        pil = (w2l @ h.T).T
        mask = ((pil[:,0] > pcr[0]) & (pil[:,0] < pcr[3]) &
                (pil[:,1] > pcr[1]) & (pil[:,1] < pcr[4]))
        # 범위 안 연속 구간만 분리해서 폴리라인
        m = mask.astype(int); d = np.diff(m)
        starts = list(np.where(d == 1)[0] + 1); ends = list(np.where(d == -1)[0] + 1)
        if mask[0]: starts = [0] + starts
        if mask[-1]: ends = ends + [len(mask)]
        for s, e in zip(starts, ends):
            seg = pil[s:e, :2]
            if len(seg) > 1:
                out.append((seg, map_element_class[lt]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/configs/hipad_b2d_stage2.py")
    ap.add_argument("--result", required=True, help="dump_infer.py results.pkl")
    ap.add_argument("--scene", required=True, help="folder basename, e.g. AccidentTwoWays_Town12_Route1444_Weather0")
    ap.add_argument("--out-dir", default="vis_b2d")  #수정: 폴더 명명 규칙 — B2D 시각화는 vis_b2d
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--interval", type=int, default=1)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--map-thr", type=float, default=0.4, help="pred 맵 vector score 임계값")
    ap.add_argument("--bev-size", type=int, default=400)
    ap.add_argument("--plan-key", default="plan_temp_2hz")  # 수정: temporal 3s — 정지 시 짧게, 논문 길이
    ap.add_argument("--fps", type=int, default=10)
    # 수정: VLA — Qwen2-VL 6cam → 궤적(traj_proj) + reasoning. visualize_nusc.py 미러링.
    ap.add_argument("--qwen-ckpt", default=None, help="qwen_traj_train_b2d.py 산출 traj_proj.pth")
    ap.add_argument("--qwen-path", default="/workspace/src/HiP-AD/ckpts/qwen2-vl-2b")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    ds = build_dataset(cfg.data["val"])
    pcr = cfg.point_cloud_range
    classes = getattr(ds, "det_classes", B2D_CLASSES)
    with open(args.result, "rb") as f:
        results = pickle.load(f)
    assert len(results) == len(ds.data_infos), f"{len(results)} vs {len(ds.data_infos)}"

    # scene 의 (글로벌 idx, info) 리스트 (frame_idx 순)
    scene_items = [(i, info) for i, info in enumerate(ds.data_infos)
                   if str(info["folder"]).split("/")[-1] == args.scene]
    scene_items.sort(key=lambda t: t[1]["frame_idx"])
    if not scene_items:
        raise SystemExit(f"scene not found in val: {args.scene}")
    scene_infos = [it[1] for it in scene_items]
    print(f"scene {args.scene}: {len(scene_items)} frames")

    out_dir = Path(args.out_dir or f"vis_b2d/{args.scene}_bev")
    (out_dir / "combine").mkdir(parents=True, exist_ok=True)
    end = args.end if args.end is not None else len(scene_items)
    sel = list(range(args.start, end, args.interval))

    # 수정: Qwen VLA predictor (옵션)
    qwen_fn = None
    if args.qwen_ckpt and os.path.exists(args.qwen_ckpt):
        qwen_fn = build_qwen_predictor(args.qwen_ckpt, args.qwen_path, ds.data_root)

    saved = []
    for k in sel:
        gidx, info = scene_items[k]
        res = results[gidx]["img_bbox"]
        data_root = ds.data_root
        # ---- 6 cam + lidar2img ----
        w2l = np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"]); l2w = inv(w2l)
        cams = {}
        for cn in CAM_ORDER:
            s = info["sensors"][cn]
            p = s["data_path"]
            p = os.path.join(data_root, p) if not os.path.isabs(p) else p
            img = cv2.imread(p) if os.path.exists(p) else np.zeros((900,1600,3), np.uint8)
            K = np.eye(4); K[:3,:3] = np.asarray(s["intrinsic"])
            lidar2cam = np.asarray(s["world2cam"]) @ l2w
            cams[cn] = (img, K @ lidar2cam)
        # pred boxes
        pb = res["boxes_3d"].numpy() if torch.is_tensor(res["boxes_3d"]) else np.asarray(res["boxes_3d"])
        ps = res["scores_3d"].numpy() if torch.is_tensor(res["scores_3d"]) else np.asarray(res["scores_3d"])
        pl = res["labels_3d"].numpy() if torch.is_tensor(res["labels_3d"]) else np.asarray(res["labels_3d"])
        # pred agent motion: trajs_3d (N,modes,T,2), pick best mode
        tr = res.get("trajs_3d"); trs = res.get("trajs_score")
        pred_agent_trajs = []
        if tr is not None:
            tr = tr.numpy() if torch.is_tensor(tr) else np.asarray(tr)
            trs = trs.numpy() if torch.is_tensor(trs) else np.asarray(trs)
            best = trs.argmax(-1)
            for n in range(len(pb)):
                if ps[n] < args.score_thr: continue
                pred_agent_trajs.append(tr[n, best[n]])   # (T,2) lidar 절대
        # pred ego plan
        plan = res.get(args.plan_key)
        plan = (plan.numpy() if torch.is_tensor(plan) else np.asarray(plan)) if plan is not None else None

        # draw pred boxes on cams
        for cn,(img,l2i) in cams.items():
            for n in range(len(pb)):
                if ps[n] < args.score_thr: continue
                lab=int(pl[n]);
                if not (0<=lab<len(classes)): continue
                col = CLS_COLOR.get(classes[lab],(0,165,255))
                c3 = box_corners_3d(pb[n][:3], pb[n][3:6], pb[n][6])
                draw_box_img(img, project(c3,l2i), col, 2)

        # 수정: GT futures 를 grid 빌드 전에 계산 (front cam 에 pred+gt 궤적 둘 다 그리려고)
        gt_ego, gt_agents = get_gt_futures(scene_infos, k, n_future=6)
        # CAM_FRONT 에 pred(분홍) + gt(초록) ego 궤적 둘 다 투영
        # 수정: Qwen VLA 예측 (옵션) — 궤적 + reasoning
        qwen_traj, qwen_reason = (qwen_fn(info) if qwen_fn else (None, None))

        img_f, l2i_f = cams["CAM_FRONT"]
        draw_traj_on_cam(img_f, l2i_f, gt_ego, GT_EGO_COLOR, t=4)      # gt 먼저(두껍게)
        draw_traj_on_cam(img_f, l2i_f, plan, EGO_PLAN_COLOR, t=2)      # pred 위로(얇게)
        draw_traj_on_cam(img_f, l2i_f, qwen_traj, QWEN_TRAJ_COLOR, t=2)  # Qwen(cyan)

        # 6-cam grid
        ch = 180
        grid_cells=[]
        for cn in CAM_ORDER:
            im = cams[cn][0]; h,w=im.shape[:2]
            im = cv2.resize(im, (int(w*ch/h), ch))
            cv2.putText(im, cn, (5,18), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            grid_cells.append(im)
        mw = min(c.shape[1] for c in grid_cells)
        grid_cells=[cv2.resize(c,(mw,ch)) for c in grid_cells]
        top=np.concatenate(grid_cells[:3],1); bot=np.concatenate(grid_cells[3:],1)
        cam_grid=np.concatenate([top,bot],0)
        gh=cam_grid.shape[0]

        # ---- GT boxes (gt_ego/gt_agents 는 위에서 계산됨) ----
        gt_boxes = np.asarray(info["gt_boxes"]); gt_names=info["gt_names"]
        gt_labels=np.array([classes.index(n) if n in classes else -1 for n in gt_names])
        gt_scores=np.ones(len(gt_labels))

        # ---- maps ----
        # PRED map: results 'vectors' (lidar xy) + scores + labels
        pred_map = []
        if "vectors" in res and "scores" in res and "labels" in res:
            mvec = res["vectors"]; msc = np.asarray(res["scores"]); mlb = np.asarray(res["labels"])
            for vi in range(len(mvec)):
                if msc[vi] >= args.map_thr:
                    pred_map.append((np.asarray(mvec[vi]), int(mlb[vi])))
        # GT map: lane_sample_points 로 직접 추출 (get_map_info 의 lane_points 변환 버그 회피)
        gt_map = get_gt_map(info, ds.map_infos, ds.map_element_class, pcr)

        # ---- BEV panels ----
        # 수정: pred(분홍)+gt(초록) ego 궤적을 pred BEV·gt BEV 양쪽에 모두 표기 (비교용)
        ego_both = [(gt_ego, GT_EGO_COLOR), (plan, EGO_PLAN_COLOR)]  # gt 먼저, pred(분홍) 위로
        if qwen_traj is not None:
            ego_both.append((qwen_traj, QWEN_TRAJ_COLOR))           # Qwen VLA 궤적(cyan)
        map_legend = [(mc, MAP_COLOR.get(li, (120,120,120)))
                      for li, mc in enumerate(getattr(ds, "map_classes", ["Broken","Solid","SolidSolid","Center"]))]
        pred_bev = draw_bev(pb, pl, ps, classes, pcr, H=gh, score_thr=args.score_thr,
                            ego_trajs=ego_both, agent_trajs=pred_agent_trajs, title="pred", maps=pred_map,
                            map_legend=map_legend)
        gt_bev = draw_bev(gt_boxes, gt_labels, gt_scores, classes, pcr, H=gh, score_thr=0.0,
                          ego_trajs=ego_both, agent_trajs=gt_agents, title="gt", maps=gt_map,
                          map_legend=map_legend)
        # command label
        # CARLA RoadOption: -1 VOID, 1 LEFT, 2 RIGHT, 3 STRAIGHT, 4 LANEFOLLOW, 5/6 CHANGELANE L/R
        cmd = {-1:"VOID",1:"LEFT",2:"RIGHT",3:"STRAIGHT",4:"LANE FOLLOW",5:"CHANGE LEFT",6:"CHANGE RIGHT"}.get(
            int(info.get("command_near", -1)), str(info.get("command_near","")))
        cv2.putText(pred_bev, cmd, (6, gh-10), cv2.FONT_HERSHEY_SIMPLEX,0.4,(80,80,80),1, cv2.LINE_AA)

        # 수정: Qwen VLA = 카메라+BEV 끝(우측)에 패널로 붙임
        panels = [cam_grid, pred_bev, gt_bev]
        if qwen_fn is not None:
            panels.append(make_qwen_panel(qwen_reason, gh))
        combined=np.concatenate(panels, 1)
        sp=out_dir/"combine"/f"{info['frame_idx']:05}.jpg"
        cv2.imwrite(str(sp), combined); saved.append(str(sp))
        if len(saved)%20==1:
            print(f"  [{len(saved)}/{len(sel)}] {sp.name} {combined.shape}")

    if saved:
        H,W=cv2.imread(saved[0]).shape[:2]
        vp=out_dir/"combine.mp4"
        vw=cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,(W,H))
        for p in saved: vw.write(cv2.imread(p))
        vw.release(); print(f"video: {vp} ({len(saved)}f {W}x{H})")
    print(f"DONE: {len(saved)} -> {out_dir}")


if __name__ == "__main__":
    main()
