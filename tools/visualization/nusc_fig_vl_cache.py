"""수정(2026-10-15): 발표 자료용 — 언어 모델(Qwen2-VL-2B)에서 뽑은 VL 토큰 오프라인 캐시(data/nuscenes/qwen_l18/<token>.pt)가
   어떤 데이터로 구성되는지 실측·시각화한다(사용자 지시). CPU 전용(모델 forward 없음, 프로세서·토크나이저만 사용).

  [측정]
    - 캐시 파일 1개의 형식(keys·shape·dtype·mask)과 파일 크기, 표본 파일들의 크기 분포(전체 34,149개는 NAS 라 걷지 않고 표본으로 추정)
    - Qwen 프로세서를 다시 돌려 1,159 토큰의 배치(시스템·역할 표지 / 카메라별 vision 시작·끝 / 영상 패드 / 지시문 / 응답 표지)를
      캐시 길이와 대조해 확인하고, 텍스트 토큰을 실제로 디코드한다.
    - 토큰별 L2 노름(0번 토큰 = 상수 '싱크' 토큰), 카메라별 10×18 영상 토큰 노름 지도(6캠 영상 위 오버레이)
    - 표본 간 같은 위치 토큰의 코사인 유사도(텍스트는 거의 상수, 영상은 표본마다 다름), 같은 표본 안 카메라 간 유사도
  [주의] 이 환경의 numpy LAPACK svd/eigh 는 NaN 을 낸다 → 선형대수는 torch 로만 한다(여기서는 내적·노름만 사용).
  출력: docs/ppt/figures/vl_cache.png (4패널), vl_cache_strip.png (토큰 띠만), vl_cache.json (수치)
  사용: CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tools/visualization/nusc_fig_vl_cache.py
"""
import argparse, json, os, pickle, sys
import numpy as np
import torch

sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nusc_fig_lang_tokens import QCAMS, INSTR, QWEN, CACHE, img_path, Mosaic, mark_cam  # 같은 지시문·카메라 순서·모자이크

OUT = "docs/ppt/figures/vl_cache"
TOKEN = "c62b6ac878934607be0c524a02f1692b"   # lang_tokens 그림과 같은 표본(val, 'Intersection, peds, cars, scooter, parked bicycle')
SHORT = {"CAM_FRONT": "F", "CAM_FRONT_LEFT": "FL", "CAM_FRONT_RIGHT": "FR", "CAM_BACK_LEFT": "BL", "CAM_BACK": "B", "CAM_BACK_RIGHT": "BR"}


def load_infos(p):
    d = pickle.load(open(p, "rb"))
    return d["infos"] if isinstance(d, dict) else d


def layout(info):
    """프로세서만 돌려 토큰 배치를 얻는다(모델 forward 없음)."""
    from transformers import AutoProcessor
    from PIL import Image
    imgs = [Image.open(img_path(info, c)).convert("RGB") for c in QCAMS]
    p = AutoProcessor.from_pretrained(QWEN, min_pixels=200 * 28 * 28, max_pixels=200 * 28 * 28)
    msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": INSTR}]}]
    prompt = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    x = p(text=[prompt], images=imgs, return_tensors="pt")
    ids = x["input_ids"][0].tolist()
    tk = p.tokenizer
    pad, vs, ve = (tk.convert_tokens_to_ids(t) for t in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"))
    typ = np.empty(len(ids), dtype=object)
    img_idx = [i for i, v in enumerate(ids) if v == pad]
    first_img, last_img = img_idx[0], img_idx[-1]
    for i, v in enumerate(ids):
        if v == pad:
            typ[i] = "image"
        elif v in (vs, ve):
            typ[i] = "marker"
        elif i < first_img:
            typ[i] = "header"
        else:
            typ[i] = "instr"
    # 마지막 '<|im_end|>\n<|im_start|>assistant\n' 5개는 응답 표지
    im_end = tk.convert_tokens_to_ids("<|im_end|>")
    last_end = max(i for i, v in enumerate(ids) if v == im_end)
    for i in range(last_end, len(ids)):
        typ[i] = "reply"
    grid = x["image_grid_thw"].tolist()
    gh, gw = grid[0][1] // 2, grid[0][2] // 2
    cam_tok = {c: np.array(img_idx[k * gh * gw:(k + 1) * gh * gw]) for k, c in enumerate(QCAMS)}
    words = [tk.decode([v]) for v in ids]
    return dict(L=len(ids), typ=typ, ids=ids, words=words, cam_tok=cam_tok, gh=gh, gw=gw, grid=grid,
                resized=(grid[0][1] * 14, grid[0][2] * 14), prompt=prompt,
                pixel_values=list(x["pixel_values"].shape), src_size=list(imgs[0].size))


def ranges(mask):
    idx = np.nonzero(mask)[0]
    out, s = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            out.append((int(s), int(a))); s = b
    out.append((int(s), int(idx[-1])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=60, help="표본 간 유사도용 val 표본 수(등간격)")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    vinf = load_infos("data/infos/nuscenes_infos_val.pkl")
    tinf = load_infos("data/infos/nuscenes_infos_train.pkl")
    info = next(v for v in vinf if v["token"] == TOKEN)
    lay = layout(info)
    L = lay["L"]

    # ---- 파일 형식 ----
    fp = os.path.join(CACHE, TOKEN + ".pt")
    d = torch.load(fp, map_location="cpu")
    assert d["hidden"].shape[0] == L, (d["hidden"].shape, L)
    fmt = {k: dict(shape=list(v.shape), dtype=str(v.dtype).replace("torch.", "")) for k, v in d.items()}
    mask_all_one = bool(d["attn_mask"].bool().all())
    H = d["hidden"].float()
    nrm = H.norm(dim=1).numpy()

    # ---- 파일 크기: 표본만(NAS) ----
    samp = [v["token"] for v in tinf[::100]] + [v["token"] for v in vinf[::50]]
    sizes = [os.path.getsize(os.path.join(CACHE, t + ".pt")) for t in samp]
    n_total = len(tinf) + len(vinf)
    fsz = int(np.median(sizes))

    # ---- 토큰 구성 ----
    typ = lay["typ"]
    comp = {}
    for t in ("header", "marker", "image", "instr", "reply"):
        m = np.array([x == t for x in typ])
        comp[t] = dict(n=int(m.sum()), ranges=ranges(m))
    cam_ranges = {c: [int(v[0]), int(v[-1])] for c, v in lay["cam_tok"].items()}
    text_m = np.array([x != "image" for x in typ])
    img_m = ~text_m

    # ---- 표본 간 유사도 ----
    toks = [v["token"] for v in vinf[::max(len(vinf) // args.n_samples, 1)]][:args.n_samples]
    if TOKEN not in toks:
        toks[0] = TOKEN
    X = torch.stack([torch.load(os.path.join(CACHE, t + ".pt"), map_location="cpu")["hidden"].float() for t in toks])  # (N,L,D)
    assert X.shape[1] == L
    Xn = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    N = Xn.shape[0]
    G = torch.einsum("npd,mpd->pnm", Xn, Xn)                       # (L,N,N)
    off = ~torch.eye(N, dtype=torch.bool)
    xs_cos = G[:, off].mean(1).numpy()                              # 위치별 표본 간 평균 코사인
    norms_all = X.norm(dim=-1)                                      # (N,L)
    sink = dict(norm_mean=float(norms_all[:, 0].mean()), norm_sd=float(norms_all[:, 0].std()),
                other_median=float(norms_all[:, 1:].median()), xs_cos=float(xs_cos[0]))
    hdr = np.array([x == "header" for x in typ]); hdr[0] = False
    post = np.array([x in ("instr", "reply") for x in typ])
    mk = np.array([x == "marker" for x in typ])
    xs = dict(header=[float(xs_cos[hdr].min()), float(xs_cos[hdr].max())],
              instr_reply=[float(xs_cos[post].min()), float(xs_cos[post].max())], instr_reply_mean=float(xs_cos[post].mean()),
              marker=[float(xs_cos[mk].min()), float(xs_cos[mk].max())],
              image=[float(xs_cos[img_m].min()), float(xs_cos[img_m].max())], image_mean=float(xs_cos[img_m].mean()),
              image_median=float(np.median(xs_cos[img_m])))
    # 같은 표본 안 카메라 간: 같은 격자 칸끼리 코사인의 평균(표본 평균) + 이 표본
    cams = QCAMS
    C = torch.zeros(6, 6); C0 = torch.zeros(6, 6)
    for a in range(6):
        for b in range(6):
            ia, ib = lay["cam_tok"][cams[a]], lay["cam_tok"][cams[b]]
            cab = (Xn[:, ia] * Xn[:, ib]).sum(-1)                   # (N,180)
            C[a, b] = cab.mean(); C0[a, b] = cab[toks.index(TOKEN)].mean()
    offc = ~torch.eye(6, dtype=torch.bool)
    # 같은 카메라·같은 칸의 표본 간 유사도 vs 같은 표본·같은 칸의 카메라 간 유사도
    same_cam_xs = float(np.mean(xs_cos[img_m]))
    cam_cam = dict(mean_offdiag_all=float(C[offc].mean()), min_offdiag_all=float(C[offc].min()), max_offdiag_all=float(C[offc].max()),
                   mean_offdiag_sample=float(C0[offc].mean()), matrix_sample=C0.tolist(), matrix_all=C.tolist())
    # 영상 토큰 노름(이 표본)
    img_norm = dict(median=float(np.median(nrm[img_m])), p99=float(np.percentile(nrm[img_m], 99)), max=float(nrm[img_m].max()))

    stats = dict(token=TOKEN, file=fp, format=fmt, mask_all_one=mask_all_one, file_bytes=os.path.getsize(fp),
                 size_sample_n=len(sizes), size_sample_unique=sorted(set(sizes)), n_train=len(tinf), n_val=len(vinf), n_total=n_total,
                 est_total_bytes=fsz * n_total, L=L, layer=18, n_layers=28, hidden_dim=int(H.shape[1]),
                 composition=comp, cam_ranges=cam_ranges, grid=[lay["gh"], lay["gw"]], patch_grid=lay["grid"][0][1:],
                 resized_hw=list(lay["resized"]), src_wh=lay["src_size"], pixel_values=lay["pixel_values"],
                 prompt=lay["prompt"], instr_words=[w for w, t in zip(lay["words"], typ) if t == "instr"],
                 norm=dict(token0=float(nrm[0]), median=float(np.median(nrm)), max_except0=float(nrm[1:].max())),
                 sink=sink, img_norm=img_norm, cross_sample=dict(n=N, **xs), cam_cam=cam_cam, same_cam_xs=same_cam_xs)

    # ================= 그림 =================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm, cm
    from matplotlib.patches import Rectangle
    fpath = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fm.fontManager.addfont(fpath)
    plt.rcParams.update({"font.family": fm.FontProperties(fname=fpath).get_name(), "axes.unicode_minus": False, "font.size": 8})
    TC = {"header": "#7F7F7F", "marker": "#C00000", "image": "#2F5597", "instr": "#C55A11", "reply": "#7030A0"}
    TN = {"header": "시스템·역할 표지", "marker": "영상 시작·끝 표지", "image": "영상 토큰", "instr": "지시문", "reply": "응답 표지"}
    camcol = ["#4472C4", "#5B9BD5", "#2F5597", "#8FAADC", "#1F3864", "#6F8FC8"]

    def strip(ax, with_norm=True):
        x = np.arange(L)
        # 띠: 구간별 사각형
        for t, cinfo in comp.items():
            for s, e in cinfo["ranges"]:
                ax.add_patch(Rectangle((s - 0.5, 0), e - s + 1, 1, fc=TC[t], ec="none", transform=ax.get_xaxis_transform(), alpha=0.18, zorder=0))
        for k, c in enumerate(QCAMS):
            s, e = cam_ranges[c]
            ax.text((s + e) / 2, 0.97, f"{SHORT[c]} 10×18", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color=TC["image"])
        if with_norm:
            cols = [TC[t] for t in typ]
            ax.scatter(x, nrm, s=2.0, c=cols, lw=0, zorder=3)
            ax.set_yscale("log"); ax.set_ylim(15, 8000)
            ax.set_ylabel("토큰 L2 노름 (log)")
            ax.annotate(f"0번 토큰(싱크) {nrm[0]:,.0f}", xy=(0, nrm[0]), xytext=(40, 700), fontsize=7, color="#404040",
                        arrowprops=dict(arrowstyle="-", color="#404040", lw=0.6))
            ax.axhline(np.median(nrm), color="#404040", lw=0.6, ls="--")
            ax.text(L - 2, np.median(nrm) * 0.78, f"중앙값 {np.median(nrm):.0f}", ha="right", va="top", fontsize=7, color="#404040")
        ax.set_xlim(-2, L + 1)
        ax.set_xlabel("토큰 위치 (0 … 1,158)")
        ticks = [0, 15, 195, 377, 559, 741, 923, 1105, 1158]
        ax.set_xticks(ticks); ax.tick_params(axis="x", labelsize=6.5)
        hs = [plt.Rectangle((0, 0), 1, 1, fc=TC[t], alpha=0.5) for t in TC]
        ax.legend(hs, [f"{TN[t]} {comp[t]['n']}" for t in TC], ncol=5, fontsize=7, loc="upper center",
                  bbox_to_anchor=(0.5, 1.2), frameon=False, handlelength=1.2, columnspacing=1.2)

    # (1) 토큰 띠만(구성 슬라이드용)
    fig, ax = plt.subplots(figsize=(11.5, 2.0))
    strip(ax, with_norm=True)
    fig.savefig(OUT + "_strip.png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

    # (2) 4패널
    fig = plt.figure(figsize=(11.5, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.35, 1], height_ratios=[1.55, 1], wspace=0.12, hspace=0.38)
    # (a) 6캠 + 영상 토큰 노름
    ax = fig.add_subplot(gs[0, 0])
    mos = Mosaic(info, down=4)
    ax.imshow(mos.M)
    vmin, vmax = np.percentile(nrm[img_m], 2), np.percentile(nrm[img_m], 98)
    vals = {c: np.clip((nrm[lay["cam_tok"][c]].reshape(lay["gh"], lay["gw"]) - vmin) / (vmax - vmin), 0, 1) for c in QCAMS}
    ax.imshow(mos.grid_rgba(vals, cmap=cm.inferno, alpha=0.75))
    ax.axis("off"); mark_cam(ax, mos)
    ax.set_title(f"(a) 영상 토큰 L2 노름 — 캠당 10×18 칸 (밝을수록 큼, {vmin:.0f}~{vmax:.0f} = 2–98 백분위)", fontsize=8, loc="left")
    # (b) 표본 간 코사인 (위치별)
    ax = fig.add_subplot(gs[1, 0])
    xx = np.arange(L)
    ax.scatter(xx, xs_cos, s=2.0, c=[TC[t] for t in typ], lw=0)
    ax.set_ylim(min(0.0, xs_cos.min() - 0.02), 1.02); ax.set_xlim(-2, L + 1)
    ax.set_ylabel("표본 간 코사인"); ax.set_xlabel("토큰 위치")
    ax.axhline(1.0, color="#BFBFBF", lw=0.5)
    ax.text(1095, 0.75, f"지시문·응답 {xs['instr_reply'][0]:.3f}~{xs['instr_reply'][1]:.3f}", ha="right", va="top", fontsize=7, color=TC["instr"])
    ax.text(560, xs["image_median"] - 0.25, f"영상 평균 {xs['image_mean']:.2f}", ha="center", va="top", fontsize=7, color=TC["image"])
    ax.set_title(f"(b) 같은 위치 토큰의 표본 간 유사도 (val {N}표본, 쌍 평균)", fontsize=8, loc="left")
    # (c) 카메라 간 (이 표본)
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(C0.numpy(), cmap="Blues", vmin=float(C0[offc].min()) - 0.05, vmax=1.0)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels([SHORT[c] for c in QCAMS], fontsize=7); ax.set_yticklabels([SHORT[c] for c in QCAMS], fontsize=7)
    for a in range(6):
        for b in range(6):
            v = float(C0[a, b])
            ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.8 else "black")
    ax.set_title("(c) 같은 표본 카메라 간 (같은 칸끼리)", fontsize=8, loc="left")
    # (d) 위치 유형별 분포
    ax = fig.add_subplot(gs[1, 1])
    data = [xs_cos[hdr], xs_cos[mk], xs_cos[post], xs_cos[img_m]]
    bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True)
    for patch, t in zip(bp["boxes"], ("header", "marker", "instr", "image")):
        patch.set_facecolor(TC[t]); patch.set_alpha(0.5)
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["머리", "표지", "지시", "영상"], fontsize=7)
    ax.set_ylim(min(0.0, xs_cos.min() - 0.02), 1.02)
    ax.set_title("(d) 유형별 표본 간 코사인", fontsize=8, loc="left")
    fig.savefig(OUT + ".png", dpi=args.dpi, bbox_inches="tight", pad_inches=0.03); plt.close(fig)

    json.dump(stats, open(OUT + ".json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: stats[k] for k in ("format", "file_bytes", "size_sample_unique", "est_total_bytes", "norm", "sink", "cross_sample", "img_norm")}, ensure_ascii=False, indent=1))
    print("cam_cam", {k: v for k, v in cam_cam.items() if not k.startswith("matrix")}, "same_cam_xs", same_cam_xs)
    print({t: c["ranges"] for t, c in comp.items() if t != "image"}, cam_ranges)


if __name__ == "__main__":
    main()
