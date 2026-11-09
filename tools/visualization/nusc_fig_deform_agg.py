"""수정(2026-10-15): 발표 자료용 — 쿼리와 ResNet-50+FPN 영상 특징이 가변형 집약(DeformableFeatureAggregation)에서
   어떻게 합쳐지는지를 실제 순전파로 시각화한다(사용자 지시). 모델 코드는 바꾸지 않고 훅으로만 읽는다.

  [훅] det_deformable[l] / map_deformable[l] 의 forward 입력 (instance_feature, anchor, anchor_embed, feature_maps, metas)
       를 잡아 모듈 자신의 메서드로 다시 계산한다:
         key_points = kps_generator(anchor, anchor_embed, instance_feature)   (forward 와 같은 호출)
         weights    = _get_weights(instance_feature, anchor_embed, metas)     (B,N,cams,levels,pts,groups)
         points_2d  = project_points(key_points, projection_mat, image_wh)    (정규화 영상 좌표)
       재계산 출력(DAF → output_proj → residual cat)이 실제 모듈 출력과 같은지(상대오차) 검사한 뒤에만 저장한다.
  [프레임] nusc_fig_lang_tokens.py 와 같은 장면 선택 규칙(pick_scene) — 장면 처음부터 대상 프레임까지 순서대로
       추론(시간 뱅크 예열). 세부 텐서는 대상 프레임만, 요약 통계(수준·카메라·영상 안 비율)는 장면 전 프레임.
  모드:
    extract : (GPU) → docs/ppt/figures/deform_agg_real.pkl
    render  : → docs/ppt/figures/deform_agg_real.png, deform_agg_map.png, deform_agg_fpn.png, deform_agg_stats.json
  사용(GPU1 전용 지시):
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tools/visualization/nusc_fig_deform_agg.py extract
    PYTHONPATH=. python tools/visualization/nusc_fig_deform_agg.py render
"""
import argparse, json, os, pickle, sys
import numpy as np

sys.path.insert(0, ".")
CFG = "configs/nuscenes/stage1/test.py"  # 수정(2026-10-15): config 경로 이동
CKPT = "results/hipad_nusc_laq_stage1/iter_175800.pth"
INFO = "data/infos/nuscenes_infos_val.pkl"
SCENE_JSON = "data/nuscenes/v1.0-trainval/scene.json"
OUT_PKL = "docs/ppt/figures/deform_agg_real.pkl"
OUT_FIG = "docs/ppt/figures/deform_agg"
CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
           "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
MAP_NAMES = ["ped_crossing", "divider", "boundary"]
LAYERS = (0, 5)   # 디코더 1층(단일 프레임)·6층(마지막)


# ------------------------------------------------------------------ extract
def cmd_extract(args):
    import torch, functools, subprocess
    free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                                        "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "")]).decode().split()[0])
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "GPU1 전용(사용자 지시)"
    assert free >= 15000, f"GPU1 여유 {free} MiB < 15 GB → 중단"
    torch.load = functools.partial(torch.load, weights_only=False)
    from mmcv import Config
    from mmcv.parallel import MMDataParallel, collate, scatter
    from mmcv.runner import load_checkpoint
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    import importlib
    from tools.visualization.nusc_fig_lang_tokens import pick_scene, _patch_decoders
    cfg = Config.fromfile(CFG)
    importlib.import_module(cfg.plugin_dir.rstrip("/").replace("/", "."))
    from projects.mmdet3d_plugin.ops import deformable_aggregation_function as DAF, feature_maps_format
    cfg.model.pretrained = None; cfg.model.train_cfg = None; cfg.data.test.test_mode = True
    ds = build_dataset(cfg.data.test)
    infos = pickle.load(open(INFO, "rb"))["infos"]
    assert [d["token"] for d in ds.data_infos] == [d["token"] for d in infos]
    desc = {s["token"]: s["description"] for s in json.load(open(SCENE_JSON))}
    frames = pick_scene(infos, desc, args.hist)
    tgt = frames[-1]
    scene = [j for j, inf in enumerate(infos) if inf["scene_token"] == infos[tgt]["scene_token"] and j <= tgt]
    print("scene:", desc[infos[tgt]["scene_token"]], "frames", scene[0], "..", tgt)

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model = MMDataParallel(model.cuda().eval(), device_ids=[0])
    head = model.module.head.onedecoder_head
    store = {}
    _patch_decoders(head, store)
    rec = {}

    def mk_hook(task, l):
        def hook(mod, inp, out):
            inst, anchor, emb, fmaps, metas = inp
            kp = mod.kps_generator(anchor, emb, inst)
            w = mod._get_weights(inst, emb, metas)                         # (B,N,cams,lv,pts,G)
            p2 = mod.project_points(kp, metas["projection_mat"], metas.get("image_wh"))   # (B,cams,N,pts,2)
            bs, N = inst.shape[:2]
            pts = p2.permute(0, 2, 3, 1, 4).reshape(bs, N, mod.num_pts, mod.num_cams, 2)
            wd = w.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(bs, N, mod.num_pts, mod.num_cams, mod.num_levels, mod.num_groups)
            f = DAF(*fmaps, pts, wd).reshape(bs, N, mod.embed_dims)
            o = torch.cat([mod.output_proj(f), inst], -1) if mod.residual_mode == "cat" else mod.output_proj(f) + inst
            err = float((o - out).norm() / out.norm())
            assert err < 1e-4, (task, l, err)
            rec[(task, l)] = dict(kp=kp[0], w=w[0], pts=pts[0], anchor=anchor[0], err=err,
                                  agg=f[0], inst=inst[0], spatial=fmaps[1], metas=metas)
            if task == "det" and l == LAYERS[-1]:
                rec["fmaps"] = fmaps
        return hook

    hs = []
    for l in LAYERS:
        hs.append(head.det_deformable[l].register_forward_hook(mk_hook("det", l)))
        hs.append(head.map_deformable[l].register_forward_hook(mk_hook("map", l)))

    def summarize(r, q):
        """선택 쿼리들의 가중치 분해(그룹 평균 → 그룹마다 합 1). 영상 밖 투영점은 커널이 0 을 더한다."""
        w = r["w"][q].float().mean(-1)                                    # (n,cams,lv,pts)
        valid = ((r["pts"][q] > 0) & (r["pts"][q] < 1)).all(-1)           # (n,pts,cams)
        valid = valid.permute(0, 2, 1)[:, :, None, :].float()             # (n,cams,1,pts)
        return dict(level=w.sum((1, 3)).mean(0).cpu().numpy(), inimg=(w * valid).sum((1, 2, 3)).cpu().numpy(),
                    pts=w.sum((1, 2)).mean(0).cpu().numpy(), cam_max=w.sum((2, 3)).max(1).values.cpu().numpy())

    summ = {"det": [], "map": []}
    saved = None
    for i in scene:
        rec.clear()
        data = scatter(collate([ds[i]], samples_per_gpu=1), [torch.cuda.current_device()])[0]
        with torch.no_grad():
            res = model.module(return_loss=False, rescale=True, **data)
            r = res[0]["img_bbox"]
            dsc, msc = r["scores_3d"].numpy(), np.asarray(r["scores"])
            dq = store["det_q"][dsc >= args.box_thr]; mq = store["map_q"][msc >= args.map_thr]
            if len(dq):
                summ["det"].append(summarize(rec[("det", LAYERS[-1])], torch.as_tensor(dq).cuda()))
            if len(mq):
                summ["map"].append(summarize(rec[("map", LAYERS[-1])], torch.as_tensor(mq).cuda()))
            if i == tgt:
                inv = feature_maps_format(rec["fmaps"], inverse=True)[0]       # 수준별 (1,cams,C,H,W)
                fnorm = [x[0].norm(dim=1).cpu().numpy() for x in inv]        # (cams,H,W) 채널 L2 크기
                img = data["img"][0].cpu().numpy()                              # (cams,3,H,W) 정규화
                saved = dict(idx=i, token=infos[i]["token"], cams=list(infos[i]["cams"].keys()), img=img,
                             fnorm=fnorm, spatial=rec[("det", LAYERS[-1])]["spatial"].cpu().numpy(),
                             det=dict(boxes=r["boxes_3d"].numpy(), scores=dsc, labels=r["labels_3d"].numpy(), q=store["det_q"]),
                             map=dict(vectors=[np.asarray(v) for v in r["vectors"]], scores=msc,
                                      labels=np.asarray(r["labels"]), q=store["map_q"]),
                             proj=rec[("det", LAYERS[-1])]["metas"]["projection_mat"][0].cpu().numpy(),
                             image_wh=rec[("det", LAYERS[-1])]["metas"]["image_wh"][0].cpu().numpy(),
                             layers={f"{t}{l}": dict(kp=v["kp"].cpu().numpy(), w=v["w"].half().cpu().numpy(),
                                                     pts=v["pts"].cpu().numpy(), anchor=v["anchor"].cpu().numpy(), err=v["err"],
                                                     inst_norm=v["inst"].norm(dim=-1).cpu().numpy(),
                                                     agg_norm=v["agg"].norm(dim=-1).cpu().numpy())
                                     for (t, l), v in ((k, v) for k, v in rec.items() if isinstance(k, tuple))})
        print(f"frame {i}: det {len(dq)} map {len(mq)} 재계산 오차 max "
              f"{max(v['err'] for k, v in rec.items() if isinstance(k, tuple)):.1e}")
    for h in hs:
        h.remove()
    agg = {t: {k: np.stack([s[k] for s in v]) if k in ("level", "pts") else np.concatenate([s[k] for s in v])
               for k in v[0]} for t, v in summ.items()}
    saved.update(scene=desc[infos[tgt]["scene_token"]], ckpt=args.ckpt, n_scene_frames=len(scene), summary=agg,
                 cfg=dict(det_pts=head.det_deformable[0].num_pts, map_pts=head.map_deformable[0].num_pts,
                          groups=head.det_deformable[0].num_groups, levels=head.det_deformable[0].num_levels,
                          cams=head.det_deformable[0].num_cams))
    pickle.dump(saved, open(OUT_PKL, "wb"))
    print("->", OUT_PKL, "GPU 최대 메모리 %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))


# ------------------------------------------------------------------ render
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
LV_COL = ["#ff3b30", "#ffcc00", "#34c759", "#00c8ff"]   # stride 4 / 8 / 16 / 32
CAM_ORDER = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]


def anchor_corners(a):
    """SparseBox3DKeyPointsGenerator 와 같은 기하: 국소 (±w/2,±l/2,±h/2) → R(yaw) → +중심. z = 기하 중심."""
    s = np.exp(a[3:6])
    loc = np.array([[1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]]) * 0.5 * s
    R = np.array([[a[7], -a[6], 0], [a[6], a[7], 0], [0, 0, 1]])
    return loc @ R.T + a[:3]


def proj(pts, P, wh):
    h = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ P.T
    return h[:, :2] / np.maximum(h[:, 2:3], 1e-5), h[:, 2] > 0.1


def cmd_render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    fp = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fm.fontManager.addfont(fp)
    plt.rcParams.update({"font.family": fm.FontProperties(fname=fp).get_name(), "axes.unicode_minus": False, "font.size": 8})
    D = pickle.load(open(OUT_PKL, "rb"))
    cams = D["cams"]; W, H = [int(x) for x in D["image_wh"][0]]
    mean, std = np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375])     # img_norm_cfg(to_rgb)
    imgs = np.clip(D["img"].transpose(0, 2, 3, 1) * std + mean, 0, 255).astype(np.uint8)
    L5d, L5m = D["layers"][f"det{LAYERS[-1]}"], D["layers"][f"map{LAYERS[-1]}"]
    det, mp = D["det"], D["map"]
    st = dict(scene=D["scene"], ckpt=D["ckpt"], token=D["token"], n_scene_frames=D["n_scene_frames"], cfg=D["cfg"],
              spatial=D["spatial"][0].tolist(), image_wh=[W, H], recompute_err={k: v["err"] for k, v in D["layers"].items()})

    # 쿼리 선택: 신뢰 박스 중 가까운 순, 중심이 보이는 카메라가 서로 다른 3개
    sel = []
    used = set()
    for j in np.argsort(np.linalg.norm(det["boxes"][:, :2], axis=1)):
        if det["scores"][j] < args.box_thr:
            continue
        q = int(det["q"][j]); a = L5d["anchor"][q]
        best = None
        for c in range(len(cams)):
            uv, ok = proj(a[None, :3], D["proj"][c], None)
            if ok[0] and 0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H:
                best = c if best is None else best
        if best is None or best in used:
            continue
        used.add(best); sel.append(dict(q=q, cam=best, cls=CLASSES[det["labels"][j]], score=float(det["scores"][j]),
                                        dist=float(np.linalg.norm(det["boxes"][j, :2]))))
        if len(sel) == args.n_det:
            break
    msel = []
    for j in np.argsort(-mp["scores"]):
        if mp["scores"][j] >= args.map_thr and len(msel) < args.n_map and MAP_NAMES[mp["labels"][j]] not in [m["cls"] for m in msel]:
            msel.append(dict(j=int(j), q=int(mp["q"][j]), cls=MAP_NAMES[mp["labels"][j]], score=float(mp["scores"][j])))
    qcol = ["#ff2d55", "#ff9500", "#af52de"]

    def draw_query(ax_of, L, q, wmax, marker, col, box=None):
        w = L["w"][q].astype(np.float32).mean(-1)                      # (cams,lv,pts) 그룹 평균
        pts = L["pts"][q]                                               # (pts,cams,2)
        out = []
        for c in range(len(cams)):
            ax = ax_of.get(c)
            if ax is None:
                continue
            for lv in range(w.shape[1]):
                uv = pts[:, c] * [W, H]
                ok = (pts[:, c] > 0).all(-1) & (pts[:, c] < 1).all(-1)
                ww = w[c, lv]
                k = ok & (ww > 0.002 * wmax)
                ax.scatter(uv[k, 0], uv[k, 1], s=4 + 140 * ww[k] / wmax, c=LV_COL[lv], marker=marker, alpha=0.85,
                           lw=0.4, edgecolors="k", zorder=5)
            if box is not None:
                cp, okc = proj(anchor_corners(box), D["proj"][c], None)
                if okc.all() and ((cp[:, 0] > 0) & (cp[:, 0] < W)).any():
                    for e in EDGES:
                        ax.plot(cp[list(e), 0], cp[list(e), 1], color=col, lw=1.0, zorder=4)
        return out

    # ---- 그림 1: 6캠(네트워크 입력 704×256) 위 키포인트·가중치 ----
    fig, axs = plt.subplots(2, 3, figsize=(12, 3.05), gridspec_kw=dict(wspace=0.015, hspace=0.04))
    ax_of = {}
    for k, n in enumerate(CAM_ORDER):
        c = cams.index(n); a = axs.flat[k]; ax_of[c] = a
        a.imshow(imgs[c]); a.set_xlim(0, W); a.set_ylim(H, 0); a.axis("off")
        a.text(4, 4, n.replace("CAM_", ""), color="white", fontsize=6.5, va="top", bbox=dict(fc="black", alpha=0.5, pad=0.6, lw=0))
    st["det_examples"] = []
    for s_, col in zip(sel, qcol):
        w = L5d["w"][s_["q"]].astype(np.float32).mean(-1)
        draw_query(ax_of, L5d, s_["q"], w.max(), "o", col, box=L5d["anchor"][s_["q"]])
        valid = ((L5d["pts"][s_["q"]] > 0) & (L5d["pts"][s_["q"]] < 1)).all(-1).T[:, None, :]
        s_.update(col=col, inimg=float((w * valid).sum()), level=w.sum((0, 2)).round(3).tolist(),
                  fixed=float(w[..., :7].sum()), learn=float(w[..., 7:].sum()),
                  top_cam=cams[int(w.sum((1, 2)).argmax())], cam_share=float(w.sum((1, 2)).max()), wmax=float(w.max()))
        st["det_examples"].append({k: v for k, v in s_.items()})
    for s_ in sel:   # 쿼리 이름표
        a = ax_of[s_["cam"]]
        uv, _ = proj(L5d["anchor"][s_["q"]][None, :3], D["proj"][s_["cam"]], None)
        a.text(uv[0, 0], max(uv[0, 1] - 38, 10), f"{s_['cls']} {s_['dist']:.0f} m", color="white", fontsize=6.5, ha="center",
               bbox=dict(fc=s_["col"], alpha=0.85, pad=0.8, lw=0), zorder=6)
    hd = [plt.Line2D([], [], ls="", marker="o", color=c_, mec="k", mew=0.4, ms=5, label=f"stride {s} (FPN {i + 1})")
          for i, (c_, s) in enumerate(zip(LV_COL, (4, 8, 16, 32)))]
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.legend(handles=hd, loc="upper center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(OUT_FIG + "_real.png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    # ---- 그림 1b: 지도 쿼리(신뢰 상위, 범주별 1개) — 같은 수준 색, 흰 선 = 예측 polyline(지면 z=-1.84 m 투영) ----
    fig, axs = plt.subplots(2, 3, figsize=(12, 3.05), gridspec_kw=dict(wspace=0.015, hspace=0.04))
    ax_of = {}
    for k, n in enumerate(CAM_ORDER):
        c = cams.index(n); a = axs.flat[k]; ax_of[c] = a
        a.imshow(imgs[c]); a.set_xlim(0, W); a.set_ylim(H, 0); a.axis("off")
        a.text(4, 4, n.replace("CAM_", ""), color="white", fontsize=6.5, va="top", bbox=dict(fc="black", alpha=0.5, pad=0.6, lw=0))
    for m_, mk in zip(msel, ("s", "D")):
        w = L5m["w"][m_["q"]].astype(np.float32).mean(-1)
        draw_query(ax_of, L5m, m_["q"], w.max(), mk, None)
        valid = ((L5m["pts"][m_["q"]] > 0) & (L5m["pts"][m_["q"]] < 1)).all(-1).T[:, None, :]
        m_.update(inimg=float((w * valid).sum()), level=w.sum((0, 2)).round(3).tolist(),
                  top_cam=cams[int(w.sum((1, 2)).argmax())], cam_share=float(w.sum((1, 2)).max()), wmax=float(w.max()))
        v = mp["vectors"][m_["j"]]
        v3 = np.concatenate([v, np.full((len(v), 1), -1.84023)], 1)
        for c, a in ax_of.items():
            uv, ok = proj(v3, D["proj"][c], None)
            uv = np.where(ok[:, None], uv, np.nan)
            a.plot(uv[:, 0], uv[:, 1], color="white", lw=1.6, zorder=4)
            a.plot(uv[:, 0], uv[:, 1], color="k", lw=0.5, ls="--", zorder=4)
    st["map_examples"] = msel
    hd = [plt.Line2D([], [], ls="", marker="s", color=c_, mec="k", mew=0.4, ms=5, label=f"stride {s}")
          for c_, s in zip(LV_COL, (4, 8, 16, 32))]
    hd += [plt.Line2D([], [], ls="", marker=mk, color="w", mec="k", ms=5, label=f"{m_['cls']} 쿼리")
           for m_, mk in zip(msel, ("s", "D"))]
    hd.append(plt.Line2D([], [], color="0.3", lw=1.6, label="예측 polyline"))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.legend(handles=hd, loc="upper center", ncol=7, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(OUT_FIG + "_map.png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    # ---- 그림 2: 한 카메라의 FPN 4수준 특징 크기 + 수준별 가중치 몫 ----
    c0 = sel[0]["cam"] if sel else cams.index("CAM_FRONT")
    fig, axs = plt.subplots(1, 5, figsize=(12, 1.55), gridspec_kw=dict(wspace=0.04, width_ratios=[1, 1, 1, 1, 1]))
    axs[0].imshow(imgs[c0]); axs[0].set_title(f"입력 {W}×{H} ({cams[c0].replace('CAM_', '')})", fontsize=7.5)
    for lv in range(4):
        f = D["fnorm"][lv][c0]
        axs[lv + 1].imshow(f, cmap="magma", vmin=np.percentile(f, 1), vmax=np.percentile(f, 99), interpolation="nearest")
        axs[lv + 1].set_title(f"FPN {lv + 1}: stride {2 ** (lv + 2)} → {f.shape[0]}×{f.shape[1]}×256", fontsize=7.5,
                              color=LV_COL[lv] if lv != 1 else "#b8860b")
    for a in axs:
        a.axis("off")
    fig.savefig(OUT_FIG + "_fpn.png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

    S = D["summary"]
    st["scene_summary"] = {t: dict(n_query=int(len(S[t]["inimg"])), level=S[t]["level"].mean(0).round(4).tolist(),
                                   inimg_mean=float(S[t]["inimg"].mean()), cam_max_mean=float(S[t]["cam_max"].mean()),
                                   pts=S[t]["pts"].mean(0).round(4).tolist() if t == "det" else None)
                           for t in ("det", "map")}
    st["scene_summary"]["det"]["fixed7"] = float(S["det"]["pts"].mean(0)[:7].sum())
    st["norms_last"] = dict(det_agg=float(np.median(L5d["agg_norm"][[s_["q"] for s_ in sel]])) if sel else None,
                            det_inst=float(np.median(L5d["inst_norm"][[s_["q"] for s_ in sel]])) if sel else None)
    json.dump(st, open(OUT_FIG + "_stats.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(st, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "render"])
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--hist", type=int, default=6)
    ap.add_argument("--box-thr", type=float, default=0.35)
    ap.add_argument("--map-thr", type=float, default=0.4)
    ap.add_argument("--n-det", type=int, default=3)
    ap.add_argument("--n-map", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=220)
    a = ap.parse_args()
    {"extract": cmd_extract, "render": cmd_render}[a.mode](a)
