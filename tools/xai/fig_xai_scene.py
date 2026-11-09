# 수정(2026-10-20): 사용자 승인 — 설명가능성 4가지(학습 변경 없음)
"""장면 하나의 설명가능성 그림(영문 글자) — 네 분석을 한 장에:
  왼쪽 위   전방 3 캠
  왼쪽 아래 BEV: GT 궤적, 최종 계획, 후보 16 개(계획 점수기 점수 — 없으면 모델 확률 — 색), 교사 행동 궤적·교사 객체 상자
            (계획 쿼리가 가장 많이 본 객체 칸은 굵게 + 칸 번호), [--results 를 주면 우리 검출 상자·지도]
  가운데    (a) 선택된 계획 쿼리의 폴더별 어텐션(계층 평균, 균등 기준 표시)  (b) 객체 칸별 어텐션(교사가 적은 객체 이름)
            (c) 반사실: 조건별 3 s 계획 변위(noop 잡음 바닥 점선)·선택 모드 바뀜·ΔL2   (d) [점수기 있으면] 상위 후보 하위 점수
  오른쪽    교사 설명문(AutoVLA 4 단, 교사 실제 생성 답, GT 와 다른 곳 빨강) + 설명–행동 일치(교사 의도 vs 학생 계획 의도)
입력은 xai_attention.py / xai_counterfactual.py 샤드(같은 토큰이 둘 다 있어야 함), 색인·GT(설명문), 선택적으로 점수기 npz·results.pkl.
사용(CPU):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. OPENBLAS_CORETYPE=SkylakeX python tools/xai/fig_xai_scene.py \
     --attn_dir results/<run>_xai/attention --cf_dir results/<run>_xai/counterfactual [--token <t> | --select max_effect --n 4] \
     [--scorer_npz results/<run>_xai/scorer_subscores.npz] [--results evaluation/<run>/results.pkl] --out_dir evaluation/<run>_xai/figures
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xai_common as X  # noqa: E402
from xai_rationale import import_fig_module  # noqa: E402

COND_LABEL = {"noop": "re-run (noise)", "noop_pad": "pad path", "gate_off": "no language", "mask_context": "context",
              "mask_detection": "detection", "mask_tracking": "tracking", "mask_motion": "motion", "mask_map": "map",
              "mask_plan": "plan(risk)", "mask_action": "action", "mask_ctx_img": "ctx: images", "mask_ctx_text": "ctx: prompt text",
              "mask_slot_top1": "top-1 object", "mask_slot_top3": "top-3 objects"}


def find_sample(d, token):
    man = json.load(open(os.path.join(d, "manifest.json")))
    for p in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        z = np.load(p)
        hit = np.flatnonzero(z["token"] == token)
        if len(hit):
            j = int(hit[0])
            return man, {k: z[k][j] for k in z.files}
    return man, None


def all_tokens(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        z = np.load(p)
        out += [(str(t), z) for t in z["token"]]
    return out


def select_tokens(a):
    if a.token:
        return a.token
    toks_a = {str(t) for p in glob.glob(os.path.join(a.attn_dir, "shard_*.npz")) for t in np.load(p)["token"]}
    man = json.load(open(os.path.join(a.cf_dir, "manifest.json")))
    conds = man["conditions"]
    mi = [c for c, n in enumerate(conds) if n.startswith("mask_") or n == "gate_off"]
    cand = []
    for p in sorted(glob.glob(os.path.join(a.cf_dir, "shard_*.npz"))):
        z = np.load(p)
        for j, t in enumerate(z["token"]):
            if str(t) in toks_a:
                eff = float(np.nanmax(z["disp_3s"][j][mi])) if a.select == "max_effect" else -j
                cand.append((eff, str(t)))
    cand.sort(reverse=True)
    if not cand:
        raise SystemExit("attention·counterfactual 둘 다에 있는 표본이 없다")
    return [t for _, t in cand[:a.n]]


def bar(ax, labels, vals, ref=None, color="#3182ce", title="", fs=8, highlight=None, fmt="{:.1f}%", scale=100.0):
    y = np.arange(len(labels))[::-1]
    cols = [("#dd6b20" if (highlight is not None and k in highlight) else color) for k in range(len(labels))]
    ax.barh(y, np.asarray(vals) * scale, color=cols, height=0.7)
    if ref is not None:
        for yy, r in zip(y, ref):
            ax.plot([r * scale] * 2, [yy - 0.38, yy + 0.38], color="#1a202c", lw=1.0, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=fs)
    ax.tick_params(axis="x", labelsize=fs * 0.85)
    for yy, v in zip(y, vals):
        ax.text(v * scale, yy, " " + fmt.format(v * scale), va="center", fontsize=fs * 0.8)
    ax.set_title(title, fontsize=fs * 1.05, loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def render(token, a, F, gts, idx, infos_index, infos, scorer, results):
    import common_v2 as V
    from common import ActionCodebook
    plt, fam = F.setup_fonts(False)
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D
    man_a, A = find_sample(a.attn_dir, token)
    man_c, Cf = find_sample(a.cf_dir, token)
    if A is None or Cf is None:
        raise SystemExit(f"{token}: attention({A is not None})·counterfactual({Cf is not None}) 샤드에 없다")
    rec, e = gts[token], idx[token]
    cb = ActionCodebook(X.CODEBOOK)
    Tp, Gp = V.parse_answer(e["gen_text"]), V.parse_answer(rec["answer"])
    cmp_, To, Go, M, tt = F.compare(Tp, Gp, rec, cb)
    items = F.build_items(rec, Tp, Gp, cmp_, To, Go, M, tt, e, None, paper=False)
    seg = man_a["segments"]; S = {n: j for j, n in enumerate(seg)}
    sel = int(A["sel"])
    agg = A["agg_modes"][:, sel].mean(0)                      # (n_seg,) 계층 평균, 선택 모드
    folders = list(X.FOLDERS)
    sizes = man_a.get("segment_sizes", {})
    uni = [sizes[f] / man_a["L"] for f in folders] if sizes else None
    nobj = int(A["n_teacher_obj"])
    slot_vals = [float(agg[S[f"slot{k}"]]) for k in range(X.N_SLOTS)]
    rank = [k for k in np.argsort(slot_vals)[::-1] if k < nobj]
    top = set(rank[:1])
    # 학생 의도(설명–행동 일치)
    i_val = infos_index[token]
    plan = np.asarray(Cf["plan"][0], np.float64)
    s_int = X.IntentLabeler().label(plan, infos[i_val], rec)
    t_int = Tp["intent"]

    fig = plt.figure(figsize=(20, 9.6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[0.74, 0.62, 1.0], height_ratios=[0.36, 0.64], wspace=0.10, hspace=0.06,
                  left=0.005, right=0.995, top=0.95, bottom=0.03)
    ax_cam, ax_bev = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    src = None
    if results is not None:
        pr = results[i_val].get("img_bbox", results[i_val])
        src = dict(pred=pr, l2=None)
    F.draw_left(fig, ax_cam, ax_bev, rec, To, tt, src, fs=9, lw=1.1)
    # 후보 16 개(점수기 점수 또는 모델 확률로 색), 최종 계획, 주목 객체 칸
    import matplotlib.cm as cm
    modes = np.asarray(Cf["base_modes"], np.float64)
    sc_lbl = "model prob."
    vals = np.asarray(Cf["probs"][0], np.float64)
    sp = None
    if scorer is not None:
        hit = np.flatnonzero(scorer["token"] == token)
        if len(hit):
            sp = int(hit[0]); vals = np.exp(scorer["score"][sp] - scorer["score"][sp].max()); sc_lbl = "scorer score (exp, rel.)"
    v = (vals - vals.min()) / max(vals.max() - vals.min(), 1e-9)
    for m_ in np.argsort(v):
        ax_bev.plot(*F.P.path(modes[m_]).T, color=cm.viridis(v[m_]), lw=0.9, alpha=0.85, zorder=3)
    ax_bev.plot(*F.P.path(plan).T, color=F.C_PLAN, lw=2.4, zorder=7)
    for k in rank[:3]:
        o = To[k]
        if o["heading"] is None:
            continue
        b = np.array([o["lat"], o["lon"], 0, o["length"], o["width"], 0, np.deg2rad(F.HEAD_YAW[o["heading"]])])
        c = F.box_bev(b)
        ax_bev.plot(c[:, 0], c[:, 1], color="#dd6b20", lw=2.4 if k in top else 1.4, zorder=8)
        ax_bev.text(o["lat"] + 0.8, o["lon"] + 0.8, f"#{k}", color="#dd6b20", fontsize=9, fontweight="bold", zorder=9)
    h, l = ax_bev.get_legend_handles_labels()
    h += [Line2D([], [], color=F.C_PLAN, lw=2.4), Line2D([], [], color=cm.viridis(0.9), lw=1.0), Line2D([], [], color="#dd6b20", lw=2.0)]
    l += ["Final plan", f"16 candidates ({sc_lbl})", "Most-attended teacher objects"]
    ax_bev.legend(h, l, loc="lower right", fontsize=8, framealpha=0.85, edgecolor="none")
    gt_ok = bool(Cf["complete"])
    l2 = Cf["l2"][0]
    ax_bev.text(0.02, 0.98, f"cmd: {rec['command']['text']} | plan L2@3s " + (f"{l2[5]:.2f} m" if gt_ok else "n/a"),
                transform=ax_bev.transAxes, fontsize=9, va="top")
    # 가운데 열
    mid = GridSpecFromSubplotSpec(4 if sp is not None else 3, 1, subplot_spec=gs[:, 1], hspace=0.55,
                                  height_ratios=[1.0, 1.0, 1.35] + ([0.9] if sp is not None else []))
    ax1 = fig.add_subplot(mid[0])
    bar(ax1, folders, [float(agg[S[f]]) for f in folders], ref=uni, title="(a) Plan query attention per teacher folder "
        "(layer mean; dotted = uniform)")
    ax2 = fig.add_subplot(mid[1])
    lab = [f"#{k} " + (F.obj_short(To[k]) if k < nobj else "(empty)") for k in range(X.N_SLOTS)]
    bar(ax2, lab, slot_vals, title="(b) Attention per object slot (57 tokens each)", highlight=top)
    ax3 = fig.add_subplot(mid[2])
    conds = man_c["conditions"]
    ci = [c for c, n in enumerate(conds) if n != "base"]
    d3 = [float(Cf["disp_3s"][c]) if np.isfinite(Cf["disp_3s"][c]) else 0.0 for c in ci]
    labs = []
    for c in ci:
        n = conds[c]
        tag = " *" if Cf["sel"][c] != Cf["sel"][0] and Cf["sel"][c] >= 0 else ""
        dl = Cf["l2"][c].mean() - Cf["l2"][0].mean() if gt_ok and np.isfinite(Cf["l2"][c]).all() else np.nan
        labs.append(f"{COND_LABEL.get(n, n)}{tag}" + ("" if not np.isfinite(dl) else f" ({dl:+.2f})"))
    noise = float(Cf["disp_3s"][conds.index("noop")]) if "noop" in conds else None
    bar(ax3, labs, d3, title="(c) Masking teacher tokens: plan shift at 3 s [m]\n    (* = selected candidate changed; ( ) = ΔL2 to GT, avg)",
        fmt="{:.2f}", scale=1.0, color="#805ad5")
    if noise is not None:
        ax3.axvline(noise, color="#1a202c", ls=":", lw=1.0)
    if sp is not None:
        ax4 = fig.add_subplot(mid[3]); ax4.axis("off")
        order = np.argsort(-scorer["score"][sp])[:4]
        rows = [["cand.", "imit.", "no-coll.", "drivable", "progress", "score"]]
        for m_ in order:
            rows.append([f"{m_}{' (chosen)' if m_ == order[0] else ''}", f"{scorer['im'][sp][m_]:.2f}", f"{scorer['nc'][sp][m_]:.2f}",
                         f"{scorer['dac'][sp][m_]:.2f}", f"{scorer['ep'][sp][m_]:.2f}", f"{scorer['score'][sp][m_]:.2f}"])
        tb = ax4.table(cellText=rows, loc="center", cellLoc="center")
        tb.auto_set_font_size(False); tb.set_fontsize(8)
        ax4.set_title("(d) Why this candidate: predicted sub-scores (plan scorer)", fontsize=8.4, loc="left", fontweight="bold")
    # 오른쪽: 설명문 + 일치
    ax = fig.add_subplot(gs[:, 2]); ax.axis("off")
    agree = None if t_int is None else (t_int["longitudinal"] == s_int["longitudinal"], t_int["lateral"] == s_int["lateral"])
    items = list(items)
    items.append(("h", "5. Explanation–action agreement (same intent definition as teacher labels)"))
    items.append(("rowh", ("", "Teacher says", "Our plan does")))
    if t_int is not None:
        items.append(("row", ("Longitudinal", F.LON.get(t_int["longitudinal"], t_int["longitudinal"]),
                              F.LON.get(s_int["longitudinal"], s_int["longitudinal"]), agree[0])))
        items.append(("row", ("Lateral", F.LAT.get(t_int["lateral"], t_int["lateral"]), F.LAT.get(s_int["lateral"], s_int["lateral"]), agree[1])))
    F.render_text(fig, ax, items, 12.0, fam, char_w=0.55)
    sm = " (SMOKE — not a result)" if (man_a.get("smoke") or man_c.get("smoke")) else ""
    fig.suptitle(f"LaQ-AD explainability · {token} · teacher {e['teacher']}{sm}", fontsize=10, x=0.005, ha="left", y=0.995)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"xai_{token}.png")
    fig.savefig(out, dpi=a.dpi); plt.close(fig)
    meta = dict(token=token, out=out, sel=sel, n_teacher_obj=nobj, slot_rank=[int(k) for k in rank], folder_attn={f: float(agg[S[f]]) for f in folders},
                cf_disp_3s={conds[c]: float(Cf["disp_3s"][c]) for c in ci}, cf_sel_changed={conds[c]: bool(Cf["sel"][c] != Cf["sel"][0]) for c in ci},
                teacher_intent=t_int, student_intent=dict(longitudinal=s_int["longitudinal"], lateral=s_int["lateral"]),
                checkpoint_attn=man_a["checkpoint"], checkpoint_cf=man_c["checkpoint"], smoke=bool(sm))
    json.dump(meta, open(out[:-4] + ".json", "w"), indent=1, ensure_ascii=False)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn_dir", required=True)
    ap.add_argument("--cf_dir", required=True)
    ap.add_argument("--token", nargs="+", default=None)
    ap.add_argument("--select", choices=["max_effect", "first"], default="max_effect")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--scorer_npz", default=None)
    ap.add_argument("--results", default=None, help="학생 results.pkl(우리 검출 상자·지도 그리기, 선택)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dpi", type=int, default=110)
    a = ap.parse_args()
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    F = import_fig_module()
    toks = select_tokens(a)
    gts = X.read_jsonl_by_token([X.GT_VAL], want=set(toks))
    idx = X.read_jsonl_by_token(X.INDEX_FILES, want=set(toks))
    infos = X.load_infos()
    infos_index = {i["token"]: k for k, i in enumerate(infos)}
    scorer = None
    if a.scorer_npz and os.path.exists(a.scorer_npz):
        scorer = dict(np.load(a.scorer_npz, allow_pickle=False))
    results = X.load_results_cpu(a.results) if a.results else None
    for t in toks:
        render(t, a, F, gts, idx, infos_index, infos, scorer, results)


if __name__ == "__main__":
    main()
