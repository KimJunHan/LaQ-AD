# 수정(2026-10-15): 교사 VLM ① 정답 v2 (사용자 승인 4개 개선) — 개선 1 차로변경 라벨 감사용 contact sheet.
#   확정 'lane change' 100개 + 무작위 'go straight'(확정, uncertain 아님) 50개 (+ 참고: uncertain 25개)를 train∪val 에서
#   seed 0 으로 뽑아 [전방 3 카메라 | BEV(지도 차로 중심선·구분선·경로)] 를 한 줄씩, 5 표본/장으로 그린다.
#   이미지는 표본당 전방 3장만 읽는다(NAS IO 최소화). 판정은 사람이(또는 Claude 가 육안으로) 한다 → audit_lc_index.json.
#   사용: CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tools/vla_teacher/audit_lc_v2.py [--map-root data/nuscenes]
import argparse, json, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_v2 as C  # noqa: E402
import lane_v2 as LV  # noqa: E402

OUT = "data/vla_teacher/v2"
S = 0.3
CW, CH = int(1600 * S), int(900 * S)
BW = CH                       # BEV 정사각
BEV_X, BEV_Y = (-20.0, 20.0), (-8.0, 32.0)   # lidar m (x 오른쪽, y 전방)


def bev_px(p):
    u = (p[..., 0] - BEV_X[0]) / (BEV_X[1] - BEV_X[0]) * BW
    v = (BEV_Y[1] - p[..., 1]) / (BEV_Y[1] - BEV_Y[0]) * BW
    return np.stack([u, v], -1)


def draw_bev(rec, info, li):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (BW, BW), (250, 250, 250)); d = ImageDraw.Draw(im)
    _, _, Rg, o = LV.lidar_to_global(info, np.zeros((1, 2)))
    sig = (rec["targets"]["intent"] or {}).get("lc_signals") or {}
    chain = li.chain(sig["L0"]) if sig.get("L0") else set()
    idx = li.tree.query_ball_point(o[:2], 45.0)
    by = {}
    for j in idx:
        by.setdefault(int(li.lid[j]), []).append(j)
    for lid, js in by.items():
        t = li.tokens[lid]
        js = sorted(js)
        g = li.pts[js, :2]
        loc = (np.concatenate([g - o[:2], np.zeros((len(g), 1))], 1) @ Rg)[:, :2]   # 전역 → lidar (Rgᵀ(p−o))
        col = (230, 40, 40) if t == sig.get("L6") else ((60, 110, 255) if t in chain else (175, 175, 175))
        pts = [tuple(p) for p in bev_px(loc)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=3 if col != (175, 175, 175) else 1)
    for lab, col in [(1, (255, 150, 0)), (2, (0, 0, 0)), (0, (0, 170, 170))]:
        for pl in info["map_annos"].get(lab, []):
            pts = [tuple(p) for p in bev_px(np.asarray(pl, np.float64)[:, :2])]
            if len(pts) > 1:
                d.line(pts, fill=col, width=2)
    xy = np.concatenate([np.zeros((1, 2)), np.asarray(rec["targets"]["fut_traj"])], 0)
    pp = bev_px(xy)
    d.line([tuple(p) for p in pp], fill=(0, 160, 0), width=3)
    for p in pp:
        d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(0, 120, 0))
    e = bev_px(np.array([[0.0, 0.0]]))[0]
    d.rectangle([e[0] - 4, e[1] - 8, e[0] + 4, e[1] + 8], outline=(0, 0, 0))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-root", default="data/nuscenes")
    ap.add_argument("--n-lc", type=int, default=100)
    ap.add_argument("--n-straight", type=int, default=50)
    ap.add_argument("--n-uncertain", type=int, default=25)
    a = ap.parse_args()
    from PIL import Image, ImageDraw, ImageFont
    recs, infos = [], {}
    for split in ["train", "val"]:
        with open(f"{OUT}/teacher_gt_{split}.jsonl") as f:
            for l in f:
                r = json.loads(l); r["split"] = split
                it = r["targets"]["intent"]
                if it is not None:
                    recs.append(r)
        infos.update({i["token"]: i for i in pickle.load(open(f"data/infos/nuscenes_infos_{split}.pkl", "rb"))["infos"]})
    rng = np.random.default_rng(0)
    lc = [r for r in recs if r["targets"]["intent"]["lateral"].startswith("lane change")]
    st = [r for r in recs if r["targets"]["intent"]["lateral"] == "go straight" and not r["targets"]["intent"]["uncertain"]
          and r["targets"]["intent"]["path_len"] >= 2.0]
    un = [r for r in recs if r["targets"]["intent"]["uncertain"]]
    groups = {"lc": [lc[k] for k in rng.choice(len(lc), min(a.n_lc, len(lc)), replace=False)],
              "straight": [st[k] for k in rng.choice(len(st), a.n_straight, replace=False)],
              "uncertain": [un[k] for k in rng.choice(len(un), min(a.n_uncertain, len(un)), replace=False)]}
    lis = {}
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    index = {}
    for g, lst in groups.items():
        index[g] = []
        for s0 in range(0, len(lst), 5):
            chunk = lst[s0:s0 + 5]
            sheet = Image.new("RGB", (3 * CW + BW, len(chunk) * (CH + 40)), "white")
            for k, r in enumerate(chunk):
                info = infos[r["token"]]
                loc = info["map_location"]
                if loc not in lis:
                    lis[loc] = LV.LaneIndex(a.map_root, loc)
                y0 = k * (CH + 40)
                cams = dict(zip(C.QWEN_CAMS, r["cams"]))
                for c, cam in enumerate(["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]):
                    sheet.paste(Image.open(cams[cam]).convert("RGB").resize((CW, CH)), (c * CW, y0))
                sheet.paste(draw_bev(r, info, lis[loc]), (3 * CW, y0))
                it = r["targets"]["intent"]; sg = it.get("lc_signals") or {}
                n = s0 + k
                txt = (f"[{g} #{n}] {r['split']} {r['token'][:10]}  label={it['lateral']}{' (uncertain)' if it['uncertain'] else ''}"
                       f"  v1={it['lateral_v1']}  cmd={r['command']['text']}  a={sg.get('a_topology')} b={sg.get('b_divider')} "
                       f"c={sg.get('c_kinematic')}({sg.get('c_mode')})  th_end={it['theta_end_deg']}deg  "
                       f"yaw0={r['ego_status']['yaw_rate']}  d_ref0/6={(sg.get('d_ref') or [None]*7)[0]}/{(sg.get('d_ref') or [None]*7)[-1]}")
                ImageDraw.Draw(sheet).text((5, y0 + CH + 4), txt, fill="black", font=font)
                index[g].append(dict(n=n, sheet=f"{OUT}/audit_lc_{g}_{s0 // 5:02d}.png", token=r["token"], split=r["split"],
                                     label=it["lateral"], signals=[sg.get("a_topology"), sg.get("b_divider"), sg.get("c_kinematic")]))
            sheet.save(f"{OUT}/audit_lc_{g}_{s0 // 5:02d}.png")
            print("saved", f"{OUT}/audit_lc_{g}_{s0 // 5:02d}.png", flush=True)
    with open(f"{OUT}/audit_lc_index.json", "w") as f:
        json.dump(index, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
