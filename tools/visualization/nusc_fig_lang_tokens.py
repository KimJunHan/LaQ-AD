"""수정(2026-10-14): 발표 자료용 — "언어 모델이 영상에서 뽑아 디코더에 넘기는 데이터"와 그 데이터를
   검출·지도·추적 쿼리가 어디서 읽는지를 실측으로 시각화한다(사용자 지시).

  [넘기는 데이터] Qwen2-VL-2B 18층 은닉 상태 (L=1159, 1536). 6캠 각각 280×504 로 줄여 14px 패치 → 2×2 병합
    → 카메라당 10×18=180 토큰(영상 1080) + 시스템·지시문 텍스트 79 토큰. (layout 모드가 캐시 길이와 대조해 확인)
  [읽는 방식] 디코더 6계층마다 QwenTokenAttn: 쿼리(Q) × 사영된 토큰(K/V) 교차 어텐션 → 계층·과제·채널별 게이트.
    여기서는 FlashMHA 입력을 훅으로 잡아 softmax(QK^T/√d) 를 다시 계산하고, 그 출력이 실제 모듈 출력과
    일치하는지(상대오차) 검사한 뒤에만 저장한다.
  모드:
    extract : 장면 1개를 앞 프레임부터 순서대로 추론(추적 상태 예열) → 마지막 3프레임의 어텐션·예측 저장 (GPU)
    render  : → docs/ppt/figures/lang_tokens_{data,det,map,track}.png + _stats.json
    inject  : 계층·과제별 언어 주입 크기 ||F'−F||/||F|| (장면 3개) → _inject.json
    probe   : 영상 토큰의 객체 범주 정보(토큰 유사도 AUC, val 60표본) → _probe.json
  사용:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/visualization/nusc_fig_lang_tokens.py extract --ckpt <stage ckpt>
    PYTHONPATH=. python tools/visualization/nusc_fig_lang_tokens.py render
"""
import argparse, json, os, pickle, sys
import numpy as np

sys.path.insert(0, ".")
INFO = "data/infos/nuscenes_infos_val.pkl"
SCENE_JSON = "data/nuscenes/v1.0-trainval/scene.json"
CACHE = "data/nuscenes/qwen_l18"
QWEN = "ckpts/qwen2-vl-2b"
CFG = "configs/nuscenes/stage1/test.py"  # 수정(2026-10-15): config 경로 이동
OUT_PKL = "evaluation/hipad_nusc_laq_stage1/paper_diag/lang_tokens.pkl"
OUT_FIG = "docs/ppt/figures/lang_tokens"
QCAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]  # Qwen 입력 순서(qwen_cache.py)
CAMS_TOP = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
CAMS_BOT = ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
INSTR = ("These are 6 surround-view camera images of an autonomous vehicle (front, "
         "front-left, front-right, back-left, back, back-right). Describe the salient "
         "objects, lane geometry, traffic state, and any driving-relevant cues.")
DET_END, MAP_END = 900, 1000   # qwen_token_attn.seg_bounds
CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
           "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
MAP_NAMES = ["ped_crossing", "divider", "boundary"]


def img_path(info, cam):
    return "data/nuscenes/samples/" + info["cams"][cam]["data_path"].split("samples/", 1)[1]


def token_layout(info):
    """프로세서만 돌려 1159 토큰의 구성을 얻는다(모델 forward 없음). 캐시 길이와 다르면 실패."""
    import torch
    from transformers import AutoProcessor
    from PIL import Image
    imgs = [Image.open(img_path(info, c)).convert("RGB") for c in QCAMS]
    p = AutoProcessor.from_pretrained(QWEN, min_pixels=200 * 28 * 28, max_pixels=200 * 28 * 28)
    msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": INSTR}]}]
    x = p(text=[p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)], images=imgs, return_tensors="pt")
    ids = x["input_ids"][0]
    pad = p.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    grid = x["image_grid_thw"].tolist()
    L = len(ids)
    hid = torch.load(os.path.join(CACHE, info["token"] + ".pt"), map_location="cpu")["hidden"]
    assert hid.shape[0] == L, (hid.shape, L)
    img_idx = (ids == pad).nonzero().flatten().numpy()
    gh, gw = grid[0][1] // 2, grid[0][2] // 2       # 2×2 병합 후 격자
    assert all(g == grid[0] for g in grid) and len(img_idx) == 6 * gh * gw
    cam_tok = {c: img_idx[k * gh * gw:(k + 1) * gh * gw] for k, c in enumerate(QCAMS)}
    txt_idx = (ids != pad).nonzero().flatten().numpy()
    words = [p.tokenizer.decode([int(ids[i])]) for i in txt_idx]
    return dict(L=L, gh=gh, gw=gw, cam_tok=cam_tok, txt_idx=txt_idx, txt_words=words,
                resized=(grid[0][1] * 14, grid[0][2] * 14), hidden=hid.float().numpy())


def pick_scene(infos, desc, min_hist):
    """규칙: 낮(설명에 night/rain 없음) · 이전 프레임 ≥ min_hist · 30 m 내 GT ≥ 12 · 지도 요소 ≥ 8 중 30 m 내 GT 최다."""
    pos = {}
    best = None
    for i, inf in enumerate(infos):
        pos.setdefault(inf["scene_token"], []).append(i)
        d = desc[inf["scene_token"]].lower()
        if "night" in d or "rain" in d or len(pos[inf["scene_token"]]) <= min_hist:
            continue
        m = np.asarray(inf["valid_flag"])
        b = np.asarray(inf["gt_boxes"])[m]
        n = int((np.linalg.norm(b[:, :2], axis=1) < 30).sum()) if len(b) else 0
        nm = sum(len(v) for v in inf["map_annos"].values())
        if n >= 12 and nm >= 8 and (best is None or n > best[0]):
            best = (n, i)
    i = best[1]
    sc = [j for j in pos[infos[i]["scene_token"]] if j <= i]
    return sc[-(min_hist + 1):]


# ------------------------------------------------------------------ extract
def _patch_decoders(head, store):
    """det/map decode 와 같은 계산에 쿼리 번호(query_idx)만 더 돌려준다(원 소스는 바꾸지 않음)."""
    import torch
    from types import MethodType
    det, mp = head.det_decoder, head.map_decoder
    det_orig, map_orig = det.decode, mp.decode

    def det_decode(self, cls_scores, box_preds, instance_id=None, quality=None, output_idx=-1):
        out = det_orig(cls_scores, box_preds, instance_id, quality, output_idx)
        s = cls_scores[output_idx].sigmoid()
        squeeze = instance_id is not None
        if squeeze:
            s = s.max(-1)[0].unsqueeze(-1)
        nc = s.shape[-1]
        sc, ind = s.flatten(1).topk(self.num_output, dim=1, sorted=self.sorted)
        mask = sc >= self.score_threshold if self.score_threshold is not None else torch.ones_like(sc, dtype=torch.bool)
        if quality is not None and quality[output_idx] is not None:
            cns = torch.gather(quality[output_idx][..., 0], 1, ind // nc)   # CNS=0 (det/decoder.py)
            sc = sc * cns.sigmoid()
            sc, o = torch.sort(sc, 1, descending=True)
            mask, ind = torch.gather(mask, 1, o), torch.gather(ind, 1, o)
        qi = (ind[0] // nc)[mask[0]].cpu().numpy()
        assert len(qi) == len(out[0]["scores_3d"]) and np.allclose(sc[0][mask[0]].cpu().numpy(), out[0]["scores_3d"].numpy())
        store["det_q"] = qi
        return out

    def map_decode(self, cls_scores, pts_preds, instance_id=None, quality=None, output_idx=-1):
        out = map_orig(cls_scores, pts_preds, instance_id, quality, output_idx)
        s = cls_scores[-1].sigmoid()
        nc = s.shape[-1]
        sc, ind = s.flatten(1).topk(s.shape[1], dim=1)
        mask = sc >= self.score_threshold if self.score_threshold is not None else torch.ones_like(sc, dtype=torch.bool)
        qi = (ind[0] // nc)[mask[0]].cpu().numpy()
        assert len(qi) == len(out[0]["scores"])
        store["map_q"] = qi
        return out

    det.decode = MethodType(det_decode, det)
    mp.decode = MethodType(map_decode, mp)


def cmd_extract(args):
    import torch, functools
    torch.load = functools.partial(torch.load, weights_only=False)
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate, scatter
    from mmcv.runner import load_checkpoint
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    import importlib
    cfg = Config.fromfile(CFG)
    importlib.import_module(cfg.plugin_dir.rstrip("/").replace("/", "."))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    ds = build_dataset(cfg.data.test)
    infos = pickle.load(open(INFO, "rb"))["infos"]
    assert [d["token"] for d in ds.data_infos] == [d["token"] for d in infos]
    desc = {s["token"]: s["description"] for s in json.load(open(SCENE_JSON))}
    frames = pick_scene(infos, desc, args.hist)
    print("scene:", desc[infos[frames[-1]]["scene_token"]], "frames", frames)

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model = MMDataParallel(model.cuda().eval(), device_ids=[0])
    head = model.module.head.onedecoder_head if hasattr(model.module.head, "onedecoder_head") else model.module.head
    store = {}
    _patch_decoders(head, store)
    qmods = [m for n, m in model.named_modules() if n.endswith("qwen_token_attn.attn.attn")]
    assert len(qmods) == 1, len(qmods)
    fmha = qmods[0]
    rec = {"attn": [], "err": []}

    def hook(mod, inp, kw, out):
        q, k, v = kw["q"], kw["k"], kw["v"]
        W, b = mod.in_proj_weight.float(), mod.in_proj_bias.float()
        E, H = mod.embed_dim, mod.num_heads
        qp = (q.float() @ W[:E].T + b[:E]).view(1, -1, H, E // H).transpose(1, 2)
        kp = (k.float() @ W[E:2 * E].T + b[E:2 * E]).view(1, -1, H, E // H).transpose(1, 2)
        vp = (v.float() @ W[2 * E:].T + b[2 * E:]).view(1, -1, H, E // H).transpose(1, 2)
        a = torch.softmax(qp @ kp.transpose(-1, -2) / (E // H) ** 0.5, -1)          # (1,H,N,L)
        o = mod.out_proj((a @ vp).transpose(1, 2).reshape(1, -1, E))
        rec["err"].append(float((o - out[0]).norm() / out[0].norm()))
        rec["attn"].append(a[0].mean(0).half().cpu().numpy())                       # 헤드 평균 (N,L)

    h = fmha.register_forward_hook(hook, with_kwargs=True)
    saved = []
    for k, i in enumerate(frames):
        rec["attn"].clear(); rec["err"].clear()
        data = scatter(collate([ds[i]], samples_per_gpu=1), [torch.cuda.current_device()])[0]
        with torch.no_grad():
            res = model.module(return_loss=False, rescale=True, **data)
        assert len(rec["attn"]) == 6, len(rec["attn"])
        print(f"frame {i}: 재계산 상대오차 max {max(rec['err']):.2e}")
        assert max(rec["err"]) < 2e-2, rec["err"]      # bf16 코어 대비 허용오차
        if k >= len(frames) - 3:
            r = res[0]["img_bbox"]
            saved.append(dict(idx=i, info=infos[i], attn=np.stack(rec["attn"]),
                              det=dict(boxes=r["boxes_3d"].numpy(), scores=r["scores_3d"].numpy(),
                                       labels=r["labels_3d"].numpy(), ids=(r["instance_ids"].cpu().numpy() if "instance_ids" in r else np.zeros(0, int)),
                                       q=store["det_q"]),
                              map=dict(vectors=[np.asarray(v) for v in r["vectors"]], scores=np.asarray(r["scores"]),
                                       labels=np.asarray(r["labels"]), q=store["map_q"])))
    # 카메라 일치 통계 — 낮(선택 장면)·비·밤 장면 전체 프레임. 신뢰 박스(≥box_thr)마다
    #   (1) 어텐션 질량 최대 카메라 == 박스가 보이는 카메라 비율 (무작위 1/6)
    #   (2) 박스 카메라에 실린 영상 토큰 질량 비율 (균등이면 1/6)
    lay = token_layout(infos[frames[-1]])
    img_all = np.concatenate([lay["cam_tok"][c] for c in QCAMS])
    sc_pos = {}
    for j, inf in enumerate(infos):
        sc_pos.setdefault(inf["scene_token"], []).append(j)
    want = [infos[frames[-1]]["scene_token"]]
    for kw_ in ("rain", "night"):
        want.append(next(t for t in sc_pos if kw_ in desc[t].lower() and (kw_ == "night" or "night" not in desc[t].lower())))
    cam_stats = []
    for st in want:
        for i in sc_pos[st]:
            rec["attn"].clear(); rec["err"].clear()
            data = scatter(collate([ds[i]], samples_per_gpu=1), [torch.cuda.current_device()])[0]
            with torch.no_grad():
                r = model.module(return_loss=False, rescale=True, **data)[0]["img_bbox"]
            assert max(rec["err"]) < 2e-2, rec["err"]
            Ag = np.stack(rec["attn"]).astype(np.float32).mean(0)
            k_ = r["scores_3d"].numpy() >= args.box_thr
            for b, q in zip(r["boxes_3d"].numpy()[k_], store["det_q"][k_]):
                cb = cam_of_box(b, infos[i])
                if cb is None:
                    continue
                m = np.array([Ag[q][lay["cam_tok"][c]].sum() for c in QCAMS]); m = m / m.sum()
                cam_stats.append((desc[st], i, QCAMS.index(cb[1]), m))     # (장면, 프레임, 객체 카메라, 카메라별 질량)
        print(desc[st], "누적 박스", len(cam_stats))
    h.remove()
    gates = {n: p.detach().cpu().numpy() for n, p in model.module.named_parameters() if "qwen_token_attn.gate" in n}
    os.makedirs(os.path.dirname(OUT_PKL), exist_ok=True)
    pickle.dump(dict(frames=saved, layout=lay, gates=gates, ckpt=args.ckpt, cam_stats=cam_stats,
                     scene=desc[infos[frames[-1]]["scene_token"]]), open(OUT_PKL, "wb"))
    print("->", OUT_PKL)


# ------------------------------------------------------------------ render
def lidar2img(cam):
    c2l = np.eye(4); c2l[:3, :3] = np.asarray(cam["sensor2lidar_rotation"]); c2l[:3, 3] = cam["sensor2lidar_translation"]
    K = np.eye(4); K[:3, :3] = np.asarray(cam["cam_intrinsic"])
    return K @ np.linalg.inv(c2l)


def box_corners(b):
    x, y, z, l, w, h, yaw = b[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    xs = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
    ys = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    zs = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * h / 2   # z = 기하 중심(컨버터 locs = nuScenes box.center). 바닥으로 두면 h/2 위로 뜬다
    return np.stack([x + c * xs - s * ys, y + s * xs + c * ys, z + zs], 1)


def project(pts, cam):
    hom = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ lidar2img(cam).T
    ok = hom[:, 2] > 0.5
    return hom[:, :2] / np.maximum(hom[:, 2:3], 1e-3), ok


def cam_of_box(b, info):
    """박스 중심이 화면 안에 들어가는 카메라(중심 깊이 최소)."""
    best = None
    for c in QCAMS:
        uv, ok = project(b[None, :3], info["cams"][c])
        if ok[0] and 0 <= uv[0, 0] < 1600 and 0 <= uv[0, 1] < 900:
            dep = (np.concatenate([b[:3], [1]]) @ lidar2img(info["cams"][c]).T)[2]
            if best is None or dep < best[0]:
                best = (dep, c, uv[0])
    return best


class Mosaic:
    def __init__(self, info, down=4, crop=(0.0, 1.0)):
        from PIL import Image
        self.d, self.c0, self.c1 = down, crop[0], crop[1]
        self.tw, self.th = 1600 // down, int(900 * (crop[1] - crop[0])) // down
        self.gap = 4
        self.M = np.full((2 * self.th + self.gap, 3 * self.tw + 2 * self.gap, 3), 255, np.uint8)
        self.off = {}
        for r, names in enumerate((CAMS_TOP, CAMS_BOT)):
            for c, n in enumerate(names):
                img = Image.open(img_path(info, n)).convert("RGB")
                img = img.crop((0, int(900 * self.c0), 1600, int(900 * self.c1))).resize((self.tw, self.th))
                ox, oy = c * (self.tw + self.gap), r * (self.th + self.gap)
                self.M[oy:oy + self.th, ox:ox + self.tw] = np.asarray(img)
                self.off[n] = (ox, oy)

    def uv(self, cam, uv):
        ox, oy = self.off[cam]
        return np.stack([uv[:, 0] / self.d + ox, (uv[:, 1] - 900 * self.c0) / self.d + oy], 1)

    def grid_rgba(self, cam_vals, cmap=None, alpha=0.55):
        """카메라별 (gh,gw) 값(또는 RGB) → 모자이크 크기 RGBA 오버레이(토큰 칸을 원영상 영역에 그대로 펼침)."""
        from PIL import Image
        H, W = self.M.shape[:2]
        O = np.zeros((H, W, 4), np.float32)
        for n, v in cam_vals.items():
            gh, gw = v.shape[:2]
            rgba = cmap(v) if v.ndim == 2 else np.concatenate([v, np.ones(v.shape[:2] + (1,))], -1)
            if v.ndim == 2:
                rgba[..., 3] = alpha * np.clip(v, 0, 1) ** 0.8
            else:
                rgba[..., 3] = alpha
            full_h = int(round(self.th / (self.c1 - self.c0)))
            up = np.asarray(Image.fromarray((rgba * 255).astype(np.uint8)).resize((self.tw, full_h), Image.NEAREST)) / 255.
            y0 = int(round(full_h * self.c0))
            ox, oy = self.off[n]
            O[oy:oy + self.th, ox:ox + self.tw] = up[y0:y0 + self.th]
        return O


def cam_grid(vec, lay):
    return {c: vec[lay["cam_tok"][c]].reshape(lay["gh"], lay["gw"]) for c in QCAMS}


def mark_cam(ax, mos):
    for n, (ox, oy) in mos.off.items():
        ax.text(ox + 3, oy + 3, n.replace("CAM_", ""), color="white", fontsize=5.5, va="top",
                bbox=dict(facecolor="black", alpha=0.5, pad=0.6, lw=0))


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_box(ax, p, col, lw=0.9, dx=0):
    for e in EDGES:
        ax.plot(p[list(e), 0] + dx, p[list(e), 1], color=col, lw=lw)


def attn_stats(A, rows, L):
    a = A[:, rows].astype(np.float64)                         # (6,n,L)
    H = -(a * np.log(a + 1e-12)).sum(-1) / np.log(L)
    iu = np.triu_indices(a.shape[1], 1)
    return dict(norm_entropy=[float(x) for x in H.mean(1)],
                inter_query_corr=[float(np.corrcoef(a[l])[iu].mean()) for l in range(a.shape[0])],
                max_over_uniform=[float(x) for x in (a.max(-1) * L).mean(1)])


def cmd_render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, font_manager as fm
    from matplotlib.colors import LinearSegmentedColormap
    fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fm.fontManager.addfont(fp)
    plt.rcParams.update({"font.family": fm.FontProperties(fname=fp).get_name(), "axes.unicode_minus": False})
    D = pickle.load(open(OUT_PKL, "rb"))
    lay, F = D["layout"], D["frames"]
    f = F[-1]; info = f["info"]; L = lay["L"]
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    img_all = np.concatenate([lay["cam_tok"][c] for c in QCAMS])
    stats = dict(L=L, n_img_tok=len(img_all), n_txt_tok=len(lay["txt_idx"]), grid=[lay["gh"], lay["gw"]],
                 resized=list(lay["resized"]))

    # ===== 1) 넘기는 데이터: 18층 영상 토큰 은닉 상태의 주성분 3개 → RGB =====
    #   torch SVD(float64) 사용 — 이 환경의 numpy 1.23.5 LAPACK svd/eigh 가 이 행렬에서 float32·float64 모두 NaN 을
    #   돌려줬다(실측). torch 결과는 유한하고 재구성 상대오차 2.5e-15 로 검증했다.
    import torch
    Hd = lay["hidden"].astype(np.float64)
    #   토큰 위치 성분 제거: 같은 위치(카메라·격자 칸)의 val 표본 평균을 뺀다. 빼지 않으면 1주성분이 모든 카메라의
    #   같은 열(맨 오른쪽)에 같은 색을 칠하는 위치 성분이 된다(실측) — 영상 내용이 아니다.
    vinf = pickle.load(open(INFO, "rb"))["infos"]
    toks = [v["token"] for v in vinf[::max(len(vinf) // args.n_pos, 1)]][:args.n_pos]
    pos_mean = np.mean([torch.load(os.path.join(CACHE, t + ".pt"), map_location="cpu")["hidden"].float().numpy()[img_all]
                        for t in toks], 0).astype(np.float64)
    stats["pos_mean_n"] = len(toks)
    R = Hd[img_all] - pos_mean
    Xt = torch.from_numpy(R - R.mean(0))
    _, S, Vh = torch.linalg.svd(Xt, full_matrices=False)
    P, S = (Xt @ Vh[:3].T).numpy(), S.numpy()
    lo, hi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)
    P = np.clip((P - lo) / (hi - lo), 0, 1)
    assert np.isfinite(P).all()
    rgb = np.zeros((L, 3)); rgb[img_all] = P
    stats["pca_var3"] = float((S[:3] ** 2).sum() / (S ** 2).sum())
    mos = Mosaic(info)
    # 수정(2026-10-15 밤): 사용자 지시 — 주성분 RGB(고정 의미 없음) 대신 '토큰 → 칸별 깊이 복원'(색 = m, turbo 0–50 m) 으로 칠한다.
    #   6캠 선형 탐침(nusc_vl_depth_probe.py --all_cams: train 300 학습, val 100 평가) 산출물. 주성분 통계(pca_var3)는 기록만 유지.
    DZ = np.load("docs/ppt/figures/vl_depth_probe_6cam.npz"); DJ = json.load(open("docs/ppt/figures/vl_depth_probe_6cam.json"))
    assert info["token"] == "c62b6ac878934607be0c524a02f1692b", info["token"]   # 탐침 그림 표본과 같은 표본
    stats["depth_probe_6cam"] = DJ
    pca = mos.grid_rgba({c: cm.turbo(np.minimum(DZ["fig_pred_depth"][k], 50.0) / 50.0)[..., :3] for k, c in enumerate(QCAMS)}, alpha=1.0)
    gray = mos.M.mean(-1, keepdims=True).repeat(3, -1) / 255.
    blend = np.where(pca[..., 3:] > 0, 0.80 * pca[..., :3] + 0.20 * gray, 1.0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 2.75), gridspec_kw=dict(wspace=0.03))
    ax[0].imshow(mos.M); ax[1].imshow(blend)
    for a_ in ax:
        a_.axis("off"); mark_cam(a_, mos)
    # 토큰 격자선(10×18) — 한 칸 = 토큰 1개
    for n, (ox, oy) in mos.off.items():
        full_h = mos.th
        for r in range(1, lay["gh"]):
            ax[1].axhline(oy + r * full_h / lay["gh"], xmin=ox / blend.shape[1], xmax=(ox + mos.tw) / blend.shape[1], color="white", lw=0.15, alpha=0.6)
        for c in range(1, lay["gw"]):
            ax[1].plot([ox + c * mos.tw / lay["gw"]] * 2, [oy, oy + full_h], color="white", lw=0.15, alpha=0.6)
    fig.savefig(OUT_FIG + "_data.png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    # ===== 2) 쿼리별 어텐션 =====
    det, mp = f["det"], f["map"]
    keep = det["scores"] >= args.box_thr
    mkeep = mp["scores"] >= args.map_thr
    mq = mp["q"][mkeep] + DET_END
    stats["det_attn"] = attn_stats(f["attn"], det["q"][keep], L)
    stats["map_attn"] = attn_stats(f["attn"], mq, L)
    # 카메라 일치율 — 저장한 3프레임 전체의 신뢰 박스
    hit = n_eval = 0; img_mass = []
    for g in F:
        Ag = g["attn"].astype(np.float32).mean(0); k_ = g["det"]["scores"] >= args.box_thr
        for b, q in zip(g["det"]["boxes"][k_], g["det"]["q"][k_]):
            img_mass.append(float(Ag[q][img_all].sum()))
            cb = cam_of_box(b, g["info"])
            if cb is None:
                continue
            mass = {c: Ag[q][lay["cam_tok"][c]].sum() for c in QCAMS}
            n_eval += 1; hit += max(mass, key=mass.get) == cb[1]
    A = f["attn"].astype(np.float32).mean(0)
    # 장면 3개(낮·비·밤) 전 프레임 통계 + 같은 프레임 안에서 객체-카메라 대응만 섞은 기준선(200회)
    cs = D["cam_stats"]
    oc = np.array([c[2] for c in cs]); M = np.stack([c[3] for c in cs]); fr = np.array([c[1] for c in cs])
    rng = np.random.default_rng(0); hb, ob = [], []
    for _ in range(200):
        pp = oc.copy()
        for fi in np.unique(fr):
            ix = np.nonzero(fr == fi)[0]; pp[ix] = oc[rng.permutation(ix)]
        hb.append((M.argmax(1) == pp).mean()); ob.append(M[np.arange(len(pp)), pp].mean())
    stats["scenes_cam"] = dict(scenes=sorted(set(c[0] for c in cs)), n_box=len(cs),
                               hit=float((M.argmax(1) == oc).mean()), hit_shuffled=float(np.mean(hb)), hit_shuffled_sd=float(np.std(hb)),
                               own_mass=float(M[np.arange(len(oc)), oc].mean()), own_mass_shuffled=float(np.mean(ob)),
                               p_perm=float(np.mean(np.array(hb) >= (M.argmax(1) == oc).mean())))
    stats.update(det_cam_hit=hit / max(n_eval, 1), det_cam_n=n_eval, det_cam_hit_k=hit, uniform_cam_hit=1 / 6,
                 det_img_mass=float(np.mean(img_mass)), map_img_mass=float(A[mq][:, img_all].sum(1).mean()),
                 img_tok_ratio=len(img_all) / L, n_det=int(keep.sum()), n_map=int(mkeep.sum()))
    objs = []
    for b, s, l, q, tid in zip(det["boxes"][keep], det["scores"][keep], det["labels"][keep], det["q"][keep],
                               det["ids"][keep] if len(det["ids"]) else [-1] * int(keep.sum())):
        cb = cam_of_box(b, info)
        if cb is not None:
            objs.append(dict(b=b, s=float(s), l=int(l), q=int(q), id=int(tid), cam=cb[1], d=float(np.linalg.norm(b[:2]))))
    objs.sort(key=lambda o: o["d"])
    sel, used = [], set()
    for o in objs:
        if o["cam"] not in used:
            sel.append(o); used.add(o["cam"])
        if len(sel) == 3:
            break
    stats["det_examples"] = [dict(cls=CLASSES[o["l"]], cam=o["cam"], dist=round(o["d"], 1), score=round(o["s"], 2),
                                  top_cam=max(QCAMS, key=lambda c: A[o["q"]][lay["cam_tok"][c]].sum())) for o in sel]
    colors = ["#ff3b30", "#ffcc00", "#00e0ff"]

    def heat(v, col, alpha):
        return mos.grid_rgba(cam_grid(v / v[img_all].max(), lay), cmap=LinearSegmentedColormap.from_list("c", [col, col]), alpha=alpha)

    # (a) 검출
    fig, a_ = plt.subplots(figsize=(5.2, 2.75))
    a_.imshow(mos.M)
    for o, col in zip(sel, colors):
        a_.imshow(heat(A[o["q"]], col, 0.85))
        uv, ok = project(box_corners(o["b"]), info["cams"][o["cam"]])
        if ok.all():
            draw_box(a_, mos.uv(o["cam"], uv), col, 1.0)
    a_.axis("off"); mark_cam(a_, mos)
    fig.savefig(OUT_FIG + "_det.png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    # (b) 지도 — 클래스별 평균
    fig, a_ = plt.subplots(figsize=(5.2, 2.75))
    a_.imshow(mos.M)
    mcol = {0: "#00e0ff", 1: "#ffcc00", 2: "#ff3b30"}
    stats["map_by_class"] = {}
    for lab in range(3):
        rows = mq[mp["labels"][mkeep] == lab]
        stats["map_by_class"][MAP_NAMES[lab]] = int(len(rows))
        if len(rows):
            a_.imshow(heat(A[rows].mean(0), mcol[lab], 0.75))
    a_.axis("off"); mark_cam(a_, mos)
    fig.savefig(OUT_FIG + "_map.png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    # (c) 추적 — 같은 ID 쿼리의 연속 3프레임
    tid = None
    for o in sel + objs:
        if o["id"] >= 0 and all(len(g["det"]["ids"]) and (g["det"]["ids"][g["det"]["scores"] >= args.box_thr] == o["id"]).any() for g in F):
            tid = o; break
    if tid is not None:
        fig, axs = plt.subplots(1, 3, figsize=(10, 1.9), gridspec_kw=dict(wspace=0.02))
        trk = []
        for k, (g, a_) in enumerate(zip(F, axs)):
            j = int(np.nonzero(g["det"]["ids"] == tid["id"])[0][0])
            q, bb = int(g["det"]["q"][j]), g["det"]["boxes"][j]
            m2 = Mosaic(g["info"])
            cname = (cam_of_box(bb, g["info"]) or (0, tid["cam"]))[1]
            Ag = g["attn"].astype(np.float32).mean(0)[q]
            mass = {c: float(Ag[lay["cam_tok"][c]].sum()) for c in QCAMS}
            trk.append(dict(query=q, cam=cname, cam_mass=round(mass[cname], 3), top_cam=max(mass, key=mass.get)))
            O = m2.grid_rgba({cname: cam_grid(Ag / Ag[img_all].max(), lay)[cname]}, cmap=cm.inferno, alpha=0.85)
            ox, oy = m2.off[cname]
            base = m2.M[oy:oy + m2.th, ox:ox + m2.tw].astype(np.float32) / 255
            ov = O[oy:oy + m2.th, ox:ox + m2.tw]
            a_.imshow(base * (1 - ov[..., 3:]) + ov[..., :3] * ov[..., 3:])
            uv, ok = project(box_corners(bb), g["info"]["cams"][cname])
            if ok.all():
                draw_box(a_, m2.uv(cname, uv) - [ox, oy], "#34c759", 1.0)
            a_.text(3, 3, f"t{k - len(F) + 1}  {cname.replace('CAM_', '')}  query {q}", color="white", fontsize=6.5, va="top",
                    bbox=dict(facecolor="black", alpha=0.55, pad=0.8, lw=0))
            a_.axis("off")
        fig.savefig(OUT_FIG + "_track.png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
        stats["track"] = dict(id=tid["id"], cls=CLASSES[tid["l"]], frames=trk)
    g = D["gates"][[k for k in D["gates"] if k.endswith("gate")][0]]          # (6,3,256)
    gm = np.abs(g).mean(-1)
    stats["gate_abs_mean"] = dict(det=gm[:, 0].round(4).tolist(), map=gm[:, 1].round(4).tolist(), plan_ego=gm[:, 2].round(4).tolist())
    json.dump(dict(stats, scene=D["scene"], ckpt=D["ckpt"], token=info["token"]),
              open(OUT_FIG + "_stats.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in stats.items()}, ensure_ascii=False))


def cmd_inject(args):
    """언어 주입 크기 실측 — 디코더 계층 l·과제 구간별로 언어 융합이 쿼리 특징을 얼마나 바꾸는가.
    QwenTokenAttn 의 입력 F 와 출력 F' 를 훅으로 잡아 ||F'−F|| / ||F|| 를 쿼리마다 계산(게이트를 0 으로 바꾸는 등
    출력을 건드리는 비교는 하지 않는다 — 원 모델 그대로의 순전파에서 잰다). 장면 3개(낮·비·밤) 전 프레임 평균,
    검출은 전체 900 쿼리와 신뢰 박스(≥box_thr) 쿼리를 따로, 지도는 전체 100 쿼리와 점수 ≥map_thr 쿼리를 따로 낸다."""
    import torch, functools
    torch.load = functools.partial(torch.load, weights_only=False)
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate, scatter
    from mmcv.runner import load_checkpoint
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    import importlib
    cfg = Config.fromfile(CFG)
    importlib.import_module(cfg.plugin_dir.rstrip("/").replace("/", "."))
    cfg.model.pretrained = None; cfg.model.train_cfg = None; cfg.data.test.test_mode = True
    ds = build_dataset(cfg.data.test)
    infos = pickle.load(open(INFO, "rb"))["infos"]
    assert [d["token"] for d in ds.data_infos] == [d["token"] for d in infos]
    desc = {s["token"]: s["description"] for s in json.load(open(SCENE_JSON))}
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model = MMDataParallel(model.cuda().eval(), device_ids=[0])
    head = model.module.head.onedecoder_head if hasattr(model.module.head, "onedecoder_head") else model.module.head
    store = {}
    _patch_decoders(head, store)
    qa = [m for n, m in model.named_modules() if n.endswith("qwen_token_attn") and type(m).__name__ == "QwenTokenAttn"]
    assert len(qa) == 1
    rec = []

    def hook(mod, inp, kw, out):
        F = (kw.get("query") if "query" in kw else inp[0]).float()[0]
        r = ((out.float()[0] - F).norm(dim=-1) / F.norm(dim=-1).clamp_min(1e-6)).cpu().numpy()
        rec.append(r)                                                               # (N,) 계층 순서대로 6번

    h = qa[0].register_forward_hook(hook, with_kwargs=True)
    recA = []

    def hookA(mod, inp, kw, out):   # MultiheadFlashAttention 은 identity + A 를 돌려준다 → ‖A‖/‖F‖
        F = kw["query"].float()[0]
        o = (out[0] if isinstance(out, tuple) else out).float()[0]
        recA.append(((o - F).norm(dim=-1) / F.norm(dim=-1).clamp_min(1e-6)).cpu().numpy())

    hA = qa[0].attn.register_forward_hook(hookA, with_kwargs=True)
    accA = []
    sc_pos = {}
    for j, inf in enumerate(infos):
        sc_pos.setdefault(inf["scene_token"], []).append(j)
    frames = pick_scene(infos, desc, args.hist)
    want = [infos[frames[-1]]["scene_token"]]
    for kw_ in ("rain", "night"):
        want.append(next(t for t in sc_pos if kw_ in desc[t].lower() and (kw_ == "night" or "night" not in desc[t].lower())))
    acc = {k: [] for k in ("det_all", "det_conf", "map_all", "map_conf", "plan_ego")}
    for st in want:
        for i in sc_pos[st]:
            rec.clear(); recA.clear()
            data = scatter(collate([ds[i]], samples_per_gpu=1), [torch.cuda.current_device()])[0]
            with torch.no_grad():
                r = model.module(return_loss=False, rescale=True, **data)[0]["img_bbox"]
            R = np.stack(rec)                                                       # (6, N)
            RA = np.stack(recA)
            accA.append([RA[:, :DET_END].mean(1), RA[:, DET_END:MAP_END].mean(1)])
            assert R.shape[0] == 6, R.shape
            dq = store["det_q"][r["scores_3d"].numpy() >= args.box_thr]
            mq = store["map_q"][np.asarray(r["scores"]) >= args.map_thr] + DET_END
            acc["det_all"].append(R[:, :DET_END].mean(1)); acc["map_all"].append(R[:, DET_END:MAP_END].mean(1))
            acc["plan_ego"].append(R[:, MAP_END:].mean(1))
            if len(dq): acc["det_conf"].append(R[:, dq].mean(1))
            if len(mq): acc["map_conf"].append(R[:, mq].mean(1))
        print(desc[st], "프레임 누적", len(acc["det_all"]))
    h.remove(); hA.remove()
    g = [p for n, p in model.module.named_parameters() if n.endswith("qwen_token_attn.gate")][0].detach().cpu().numpy()
    out = dict(ckpt=args.ckpt, scenes=[desc[t] for t in want], n_frames=len(acc["det_all"]), N=int(R.shape[1]),
               rel_change={k: np.mean(v, 0).round(5).tolist() for k, v in acc.items() if v},
               attn_over_feat={"det_all": np.mean([a[0] for a in accA], 0).round(4).tolist(),
                               "map_all": np.mean([a[1] for a in accA], 0).round(4).tolist()},
               gate_abs_mean={"det": np.abs(g[:, 0]).mean(-1).round(5).tolist(), "map": np.abs(g[:, 1]).mean(-1).round(5).tolist(),
                              "plan_ego": np.abs(g[:, 2]).mean(-1).round(5).tolist()})
    json.dump(out, open(OUT_FIG + "_inject.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def cmd_probe(args):
    """영상 토큰에 객체 범주 정보가 있는가 — val 60표본, 토큰 칸 라벨(GT 박스 투영 사각형이 칸 면적 30% 이상:
    차량/보행자/배경). 기준 칸과 **다른 카메라** 칸들의 코사인 유사도로 같은 범주를 가려내는 AUC(0.5=무정보).
    원 은닉 상태와 위치 평균(val 120표본, 서로 겹치지 않음)을 뺀 것을 함께 잰다."""
    import torch
    from sklearn.metrics import roc_auc_score
    lay = pickle.load(open(OUT_PKL, "rb"))["layout"]
    img_all = np.concatenate([lay["cam_tok"][c] for c in QCAMS])
    gh, gw = lay["gh"], lay["gw"]; CW, CH = 1600 / gw, 900 / gh
    vinf = pickle.load(open(INFO, "rb"))["infos"]
    ld = lambda t: torch.load(os.path.join(CACHE, t + ".pt"), map_location="cpu")["hidden"].float().numpy()[img_all]
    pm = np.mean([ld(v["token"]) for v in vinf[::50][:120]], 0)
    VEH = ("car", "truck", "bus", "trailer", "construction_vehicle")

    def cells(info):
        lab = np.zeros((6, gh, gw), int); m = np.asarray(info["valid_flag"])
        for b, n in zip(np.asarray(info["gt_boxes"])[m], np.asarray(info["gt_names"])[m]):
            cls = 1 if n in VEH else 2 if n == "pedestrian" else 0
            if not cls:
                continue
            for ci, c in enumerate(QCAMS):
                uv, ok = project(box_corners(b), info["cams"][c])
                if not ok.all():
                    continue
                (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
                xs = np.clip(np.minimum(x1, (np.arange(gw) + 1) * CW) - np.maximum(x0, np.arange(gw) * CW), 0, None)
                ys = np.clip(np.minimum(y1, (np.arange(gh) + 1) * CH) - np.maximum(y0, np.arange(gh) * CH), 0, None)
                cov = np.outer(ys, xs) / (CW * CH) > 0.3
                lab[ci][cov] = np.maximum(lab[ci][cov], cls)
        return lab.reshape(-1)

    res = {k: [] for k in ("raw_vehicle", "raw_pedestrian", "posfree_vehicle", "posfree_pedestrian")}
    for v in vinf[25::100][:60]:
        H = ld(v["token"]); lab = cells(v)
        for tag, X in (("raw", H), ("posfree", H - pm)):
            Xn = X / np.linalg.norm(X, axis=1, keepdims=True); S = Xn @ Xn.T
            for cls, nm in ((1, "vehicle"), (2, "pedestrian")):
                for i in np.nonzero(lab == cls)[0]:
                    oc = np.arange(len(lab)) // (gh * gw) != i // (gh * gw); y = lab[oc] == cls
                    if 2 <= y.sum() < len(y) - 2:
                        res[f"{tag}_{nm}"].append(roc_auc_score(y, S[i, oc]))
    out = {k: dict(n=len(v), auc=float(np.mean(v)), se=float(np.std(v) / np.sqrt(len(v)))) for k, v in res.items()}
    json.dump(out, open(OUT_FIG + "_probe.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "render", "probe", "inject"])
    ap.add_argument("--ckpt", default="results/hipad_nusc_laq_stage1/latest.pth")
    ap.add_argument("--hist", type=int, default=6)          # 추적 상태 예열 프레임 수
    ap.add_argument("--box-thr", type=float, default=0.35)
    ap.add_argument("--map-thr", type=float, default=0.4)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--n-pos", type=int, default=120)       # 위치 평균에 쓸 val 표본 수
    a = ap.parse_args()
    {"extract": cmd_extract, "render": cmd_render, "probe": cmd_probe, "inject": cmd_inject}[a.mode](a)
