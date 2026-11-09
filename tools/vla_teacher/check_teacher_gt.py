# 수정(2026-10-15): 교사 VLM ① 정답 자동 생성 — 산출물 검증.
#   (a) 파서 왕복: parse_answer(answer) == targets (전 샘플) + 구 parse_caption/smooth_reward 호환
#   (b) 누설: prompt 는 명령 + 현재 ego_status 로만 재생성되어야 하고 가변부는 두 줄뿐
#   (c) 5 샘플 카메라 투영 contact sheet (lidar2img = K @ inv(cam2lidar), nusc_fig_lang_tokens.py 와 같은 규약)
#   (d) train/val 장면 분리  (e) 전체 개수 train 28,130 / val 6,019  (+ 카메라 이미지 실재)
#   사용: CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tools/vla_teacher/check_teacher_gt.py
import json, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from spatial_common import smooth_reward  # noqa: E402  (tools/qwen3d_nusc — common 이 경로 추가)

OUT = "data/vla_teacher"
EXPECT = dict(train=28130, val=6019)


def load(split):
    with open(f"{OUT}/teacher_gt_{split}.jsonl") as f:
        return [json.loads(l) for l in f]


def check_roundtrip(recs, cb):
    bad = []; rew = []; old_ok = 0
    for r in recs:
        t = r["targets"]
        try:
            p = C.parse_answer(r["answer"])
        except ValueError as e:
            bad.append((r["token"], str(e))); continue
        exp_objs = [dict(sector=o["sector"], dist=o["dist"], name=o["name"], vel_sector=o["vel_sector"],
                         speed=o["speed"] if o["vel_sector"] else None, lat=o["lat"], lat_side=o["lat_side"],
                         lon=o["lon"], lon_side=o["lon_side"]) for o in t["objects"]]
        ok = (p["ego_speed"] == t["ego_speed"] and p["objects"] == exp_objs)
        m = t["map"]
        if m is None:
            ok &= p["map"] is None
        else:
            pc = m["ped_crossing"]
            exp_pc = None if pc is None else (dict(dist=None, sector="here") if pc["sector"] == "here" else pc)
            ok &= p["map"] == dict(ped_crossing=exp_pc, n_divider=m["n_divider"], n_boundary=m["n_boundary"])
        it = t["intent"]
        ok &= p["intent"] == (None if it is None else dict(longitudinal=it["longitudinal"], lateral=it["lateral"]))
        ok &= p["action"] == t["action_tokens"]
        ok &= (it is None) == (t["action_tokens"] is None) == (not t["future_complete"])
        if t["action_tokens"] is not None:
            ok &= all(0 <= k < cb.K for k in t["action_tokens"])
            ok &= np.allclose(cb.decode(t["action_tokens"]), np.asarray(t["action_recon_traj"]), atol=1e-3)
        # 구 파서 호환(인지 줄): 방위·거리·클래스·이동방향, moving 이면 속도
        ego, objs = C.parse_caption(r["answer"])
        oo = (ego == t["ego_speed"] and len(objs) == len(t["objects"]) and all(
            a["sector"] == b["sector"] and a["dist"] == b["dist"] and a["name"] == b["name"]
            and a["vel_sector"] == b["vel_sector"] and (b["vel_sector"] is None or a["speed"] == b["speed"])
            for a, b in zip(objs, t["objects"])))
        old_ok += oo
        if t["objects"]:
            rew.append(smooth_reward(r["answer"], t))
        if not ok:
            bad.append((r["token"], "targets 불일치"))
    return bad, old_ok, (float(np.mean(rew)), float(np.min(rew)))


def check_leak(recs):
    bad = []
    for r in recs:
        es = r["ego_status"]["raw"]
        u = r["prompt"]["user"]
        ok = u == C.build_user_prompt(r["command"]["id"], es) and r["prompt"]["system"] == C.SYSTEM_PROMPT
        m = C.PROMPT_VAR_RE.search(u)
        ok &= m is not None and u[:m.start()] == C.INSTRUCTION
        ok &= r["command"]["text"] == C.CMD_TEXT[r["command"]["id"]]
        if not ok:
            bad.append(r["token"])
    return bad


# ------------------------------------------------------------ (c) 투영
def lidar2img(cam):  # tools/visualization/nusc_fig_lang_tokens.py 와 동일 식(inv 는 강체 역행렬을 해석적으로 — LAPACK 회피)
    R = np.asarray(cam["sensor2lidar_rotation"], np.float64); t = np.asarray(cam["sensor2lidar_translation"], np.float64)
    l2c = np.eye(4); l2c[:3, :3] = R.T; l2c[:3, 3] = -R.T @ t        # = inv(cam2lidar)
    K = np.eye(4); K[:3, :3] = np.asarray(cam["cam_intrinsic"])
    return K @ l2c


def box_corners(b):  # nusc_fig_lang_tokens.py 와 동일
    x, y, z, l, w, h, yaw = b[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    xs = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
    ys = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    zs = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * h / 2
    return np.stack([x + c * xs - s * ys, y + s * xs + c * ys, z + zs], 1)


def project(pts, cam):
    hom = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ lidar2img(cam).T
    return hom[:, :2] / np.maximum(hom[:, 2:3], 1e-3), hom[:, 2]


def contact_sheet(rec, info, path):
    from PIL import Image, ImageDraw, ImageFont
    layout = [["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"], ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]]
    S = 0.4; W, H = int(1600 * S), int(900 * S)
    paths = dict(zip(C.QWEN_CAMS, rec["cams"]))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14); fsm = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = fsm = ImageFont.load_default()
    al = rec["answer"].split("\n")
    lines = al[0].replace("; ", ";\n").split("\n") + al[1:]   # 인지 줄만 객체별로 나눔(#k = 객체 번호)
    sheet = Image.new("RGB", (3 * W, 2 * H + 16 * (len(lines) + 2)), "white")
    cols = [(255, 60, 60), (60, 200, 60), (60, 120, 255), (255, 170, 0), (200, 0, 200), (0, 200, 200)]
    tiles = {}
    for rr, row in enumerate(layout):
        for cc, cam in enumerate(row):
            im = Image.open(paths[cam]).convert("RGB").resize((W, H))
            ImageDraw.Draw(im).text((5, 5), cam, fill=(255, 255, 0), font=font)
            tiles[cam] = (im, cc * W, rr * H)
    for k, o in enumerate(rec["targets"]["objects"]):
        b = np.asarray(o["box"])
        best = None
        for cam in C.QWEN_CAMS:   # 중심이 화면 안에 있는 카메라 중 깊이 최소(nusc_fig_lang_tokens.cam_of_box 와 같은 규칙)
            uv, dep = project(b[None, :3], info["cams"][cam])
            if dep[0] > 0.5 and 0 <= uv[0, 0] < 1600 and 0 <= uv[0, 1] < 900 and (best is None or dep[0] < best[0]):
                best = (dep[0], cam)
        if best is None:
            continue
        cam = best[1]; im = tiles[cam][0]; d = ImageDraw.Draw(im)
        uv, dep = project(box_corners(b), info["cams"][cam]); uv = uv * S
        col = cols[k % len(cols)]
        if (dep > 0.5).all():
            for a, e in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                d.line([tuple(uv[a]), tuple(uv[e])], fill=col, width=2)
        cu, _ = project(b[None, :3], info["cams"][cam])
        d.text((cu[0, 0] * S, cu[0, 1] * S), f"#{k+1} {o['name'][:5]} {o['dist']}m {o['sector']}", fill=col, font=fsm)
    for cam, (im, x, y) in tiles.items():
        sheet.paste(im, (x, y))
    d = ImageDraw.Draw(sheet)
    d.text((5, 2 * H + 2), f"token {rec['token']}  cmd={rec['command']['text']}  " + rec["prompt"]["user"].split("\n")[-1],
           fill="black", font=fsm)
    for i, l in enumerate(lines):
        d.text((5, 2 * H + 18 + 16 * i), (f"#{i} " if 0 < i <= len(rec['targets']['objects']) else "") + l,
               fill="black", font=fsm)
    sheet.save(path)


def main():
    cb = C.ActionCodebook("data/qwen3d/nusc_action_codebook_k2048.npz")
    report = {}
    scenes = {}
    for split in ["train", "val"]:
        recs = load(split)
        bad, old_ok, rew = check_roundtrip(recs, cb)
        leak = check_leak(recs)
        missing_img = sum(1 for r in recs for p in r["cams"] if not os.path.exists(p))
        infos = pickle.load(open(f"data/infos/nuscenes_infos_{split}.pkl", "rb"))["infos"]
        order_ok = [r["token"] for r in recs] == [i["token"] for i in infos]
        scenes[split] = {r["scene_token"] for r in recs}
        n_inc = sum(1 for r in recs if not r["targets"]["future_complete"])
        n_inc_marked = sum(1 for r in recs if not r["targets"]["future_complete"]
                           and r["answer"].endswith("Intent: unavailable.\nAction: unavailable."))
        report[split] = dict(n=len(recs), expected=EXPECT[split], count_ok=len(recs) == EXPECT[split],
                             unique_tokens=len({r["token"] for r in recs}), order_matches_infos=order_ok,
                             roundtrip_fail=len(bad), roundtrip_fail_examples=bad[:3],
                             old_parse_caption_ok=old_ok, gt_answer_smooth_reward_mean_min=rew,
                             leak_fail=len(leak), missing_images=missing_img,
                             incomplete=n_inc, incomplete_marked_unavailable=n_inc_marked)
        print(split, json.dumps(report[split], ensure_ascii=False))
        if split == "val":
            by_tok = {i["token"]: i for i in infos}
            # 다양한 경우: 회전/정지/차로변경/이동 객체 많은 장면/미래 불완전
            want = []
            rng = np.random.default_rng(0)
            for cond in [lambda r: r["targets"]["intent"] and r["targets"]["intent"]["lateral"] == "turn left",
                         lambda r: r["targets"]["intent"] and r["targets"]["intent"]["longitudinal"] == "stop",
                         lambda r: r["targets"]["intent"] and r["targets"]["intent"]["lateral"].startswith("lane change"),
                         lambda r: sum(o["vel_sector"] is not None for o in r["targets"]["objects"]) >= 5,
                         lambda r: not r["targets"]["future_complete"]]:
                cand = [r for r in recs if cond(r)]
                want.append(cand[int(rng.integers(len(cand)))])
            for k, r in enumerate(want):
                contact_sheet(r, by_tok[r["token"]], f"{OUT}/check_{k}.png")
            report["contact_sheets"] = [dict(file=f"{OUT}/check_{k}.png", token=r["token"]) for k, r in enumerate(want)]
    report["scene_overlap_train_val"] = len(scenes["train"] & scenes["val"])
    print("scene overlap train∩val:", report["scene_overlap_train_val"])
    with open(f"{OUT}/check_report.json", "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
