# 수정(2026-10-15): 교사 VLM ① 정답 v2 (사용자 승인 4개 개선) — v2 산출물 검증(v1 과 같은 항목 + v2 항목).
#   수정(2026-10-15): 정답 형식 v2-sensor — (i) 전 표본 정답 토큰 수 동일(실제 토크나이저+특수토큰), loss_mask_token_spans 재계산,
#       '<na>' 가 전부 손실 제외 구간 안, (j) 수치 정합(infos 독립 재계산: 위치·크기·진행방위·속력·미래 변위·t0 TTC·단면 횡거리·자유거리 범위),
#       (k) TTC 구간·지도 필드 가용률 보고
#   (a) 파서 왕복: parse_answer(answer) == targets, format_parsed(parse_answer(answer)) == answer (전 샘플)
#   (b) 누설: prompt 는 명령 + 현재 CAN / 과거 포즈(t−1,t−2 — infos 에서 독립 재계산)만으로 재생성, 가변부는 두 줄뿐,
#       past_tokens 는 같은 장면의 직전 샘플(시간 엄격 감소)
#   (c) 5 샘플 카메라 투영 contact sheet(lidar2img = K @ inv(cam2lidar), v1 check 와 같은 함수)
#   (d) 장면 분리 (e) 개수 28,130 / 6,019, infos 순서 일치, 카메라 경로 = v1(v1 에서 실재 검사 완료 → NAS stat 재실행 생략)
#   (f) 의도 분포 v1 대비 (g) 행동 토큰 = v1, val 복원 ADE (h) 특수토큰 파일 = common_v2.SPECIAL_TOKENS
#   사용: CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=. python tools/vla_teacher/check_teacher_gt_v2.py
import collections, json, os, pickle, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_v2 as C  # noqa: E402
from check_teacher_gt import project, box_corners  # noqa: E402  (v1 투영 함수 재사용)
from gen_teacher_gt import cam_paths  # noqa: E402

OUT = "data/vla_teacher/v2"
V1 = "data/vla_teacher"
EXPECT = dict(train=28130, val=6019)
VAR_RE = re.compile(r"\nCommand: (go straight|turn left|turn right)\.(?:\n"
                    # 수정(2026-10-15): B안 고정폭 — 속도 두 자리(\d\d\.\d), 결측은 common_v2.EGO_DROPPED 전문
                    r"Ego state: (?:" + re.escape(C.EGO_DROPPED[len("Ego state: "):]) + r"|speed \d\d\.\d m/s, longitudinal acceleration [+-]\d\.\d\d m/s\^2, "
                    r"yaw rate [+-]\d+\.\d\d\d rad/s \(positive = turning left\)\.))?$")


def expected_parse(t, osp):
    # 수정(2026-10-15): v2-sensor 필드 포함
    objs = [dict(name=o["name"], sector=o["sector"], lat=o["lat"], lon=o["lon"], length=o["length"], width=o["width"],
                 heading=o["heading"], motion=o["motion"], vel_sector=o["vel_sector"],
                 speed=o["speed"] if osp == "numeric" else None, fut=o["fut"], ttc=o["ttc"]) for o in t["objects"]]
    m = t["map"]
    if m is None:
        mp = None
    else:
        pc = m["ped_crossing"]
        mp = dict(ped_crossing=None if pc is None else (dict(dist=None, sector="here") if pc["sector"] == "here" else pc),
                  n_divider=m["n_divider"], n_boundary=m["n_boundary"], lanes=m["lanes"], free_dist=m["free_dist"])
    it = t["intent"]
    return dict(objects=objs, map=mp,
                intent=None if it is None else dict(longitudinal=it["longitudinal"], lateral=it["lateral"]),
                action=t["action_tokens"])


def check_roundtrip(recs, cb):
    bad = []
    for r in recs:
        t = r["targets"]
        osp = r["modes"]["object_speed"]
        try:
            p = C.parse_answer(r["answer"], osp)
        except ValueError as e:
            bad.append((r["token"], str(e))); continue
        ok = r["modes"].get("answer_format") == C.ANSWER_FORMAT
        ok &= p == expected_parse(t, osp)
        ok &= C.format_parsed(p, osp) == r["answer"]
        # 손실 마스킹 구간: '<na>' 대체형 + 미래 불완전 → Intent/Action 줄 + 이전 프레임 없음 → 속력 숫자 (정렬·병합)
        sp = C.answer_spans(r["answer"], osp)
        ok &= r["answer_spans"] == sp
        exp_lm = C.loss_mask_spans(sp, t["future_complete"], r["prev_available"])
        ok &= r["loss_mask_spans"] == exp_lm and len(r["loss_mask_token_spans"]) == len(exp_lm)
        ok &= all(r["answer"][a0:b0].startswith("<na>") for a0, b0 in sp["na"])
        # '<na>' 는 손실 제외 구간 밖에 나타나면 안 된다(미상 값을 감독하지 않는다)
        cov = np.zeros(len(r["answer"]), bool)
        for a0, b0 in r["loss_mask_spans"]:
            cov[a0:b0] = True
        ok &= all(cov[m.start():m.end()].all() for m in re.finditer(r"<na>", r["answer"]))
        if not t["future_complete"]:
            a0, b0 = sp["lines"]["intent"]; ok &= r["answer"][a0:b0] == "Intent: <na><_>"
            ok &= all(o["ttc"] is None for o in t["objects"])
        else:
            ok &= all(o["ttc"] is not None for o in t["objects"])
        ok &= all(re.fullmatch(r"\d\d\.\d", r["answer"][a0:b0]) for a0, b0 in sp["speed"])
        ok &= len(t["objects"]) <= C.MAX_OBJECTS
        ok &= [o["dist"] for o in t["objects"]] == sorted(o["dist"] for o in t["objects"])
        it = t["intent"]
        ok &= (it is None) == (t["action_tokens"] is None) == (not t["future_complete"])
        if it is None:
            ok &= t["intent_weight"] == 0.0
        else:
            ok &= t["intent_weight"] == (0.3 if it["uncertain"] else 1.0)
            ok &= (not it["uncertain"]) or it["lateral"] == "go straight"
            ok &= np.allclose(cb.decode(t["action_tokens"]), np.asarray(t["action_recon_traj"]), atol=1e-3)
        ok &= all(0 <= k < cb.K for k in r["copy_baseline"]["action_tokens"])
        if not ok:
            bad.append((r["token"], "targets/왕복 불일치"))
    return bad


def _corners2d(x, y, l, w, yaw):
    """독립 구현(평가기 box3d_to_corners 와 같은 식): 바닥 사각형 4 꼭짓점."""
    c, s_ = np.cos(yaw), np.sin(yaw)
    loc = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5]]) * [l, w]
    return np.stack([x + c * loc[:, 0] - s_ * loc[:, 1], y + s_ * loc[:, 0] + c * loc[:, 1]], 1)


def check_sensor(recs, by_tok):
    """수정(2026-10-15): v2-sensor 수치 정합 — infos 에서 독립 재계산해 정답 값과 비교. 반환 (실패 목록, 통계)."""
    from shapely.geometry import Polygon, LineString
    bad = collections.Counter(); ex = {}
    st = collections.Counter()
    for r in recs:
        info = by_tok[r["token"]]
        B = np.asarray(info["gt_boxes"], np.float64); F = np.asarray(info["gt_agent_fut_trajs"], np.float64)
        M = np.asarray(info["gt_agent_fut_masks"]); V = np.asarray(info["gt_velocity"], np.float64)
        def fail(k):
            bad[k] += 1; ex.setdefault(k, r["token"])
        for o in r["targets"]["objects"]:
            j = o["box_index"]; b = B[j]
            if not np.allclose(b[:7], o["box"], atol=1e-3):
                fail("box_index")
            if abs(o["lat"] - b[0]) > 0.051 or abs(o["lon"] - b[1]) > 0.051:
                fail("offset_vs_gt_boxes")
            if abs(o["length"] - b[3]) > 0.051 or abs(o["width"] - b[4]) > 0.051:
                fail("size_vs_gt_boxes")
            # 진행방위: yaw → 시계방향 방위(전방 0°) 를 독립 계산
            if o["name"] == "traffic_cone":
                if o["heading"] is not None:
                    fail("cone_heading")
            else:
                brg = (90.0 - np.degrees(b[6])) % 360
                sec = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"][int(((brg + 22.5) % 360) // 45)]
                if sec != o["heading"]:
                    fail("heading_vs_yaw")
            fin = np.isfinite(V[j]).all()
            if fin != (o["motion"] != "na"):
                fail("motion_na_vs_nan")
            if fin:
                spd = float(np.hypot(*V[j]))
                if abs(o["speed"] - spd) > 0.051 or (o["motion"] == "moving") != (spd >= 0.5):
                    fail("speed_vs_gt_velocity")
            cum = np.cumsum(F[j], 0)
            for h, pfut in zip((1, 3, 5), o["fut"]):
                valid = bool((M[j, :h + 1] == 1).all())
                if valid != (pfut is not None):
                    fail("fut_mask")
                elif valid and (abs(pfut[0] - cum[h, 0]) > 0.051 or abs(pfut[1] - cum[h, 1]) > 0.051):
                    fail("fut_vs_gt_agent_fut_trajs")
                st[f"fut_{h}_valid"] += int(valid); st["fut_slots"] += 1
            if o["ttc"] is not None:
                # t=0 간격(독립 구현) ≤ 임계값 − 1 cm 이면 반드시 lt2 (그 이상은 미래 시각이 정함)
                eg = Polygon(_corners2d(0.0, C.EGO_FWD, C.EGO_L, C.EGO_W, np.pi / 2))
                g0 = eg.distance(Polygon(_corners2d(b[0], b[1], b[3], b[4], b[6])))
                if g0 <= C.TTC_GAP - 0.01 and o["ttc"] != "lt2":
                    fail("ttc_t0")
                if o["ttc"] == "safe" and g0 <= C.TTC_GAP:
                    fail("ttc_safe_but_close")
                st["ttc_" + o["ttc"]] += 1
        mp = r["targets"]["map"]
        if mp is not None:
            ma = info["map_annos"]
            for y, d in mp["lanes"].items():
                line = LineString([(-50, float(y)), (50, float(y))])
                for key, cls in (("div", 1), ("bnd", 2)):
                    xs = []
                    for pl in ma.get(cls, []):
                        pl = np.asarray(pl, np.float64)
                        if len(pl) < 2:
                            continue
                        g = LineString(pl).intersection(line)
                        if g.is_empty:
                            continue
                        geoms = getattr(g, "geoms", [g])
                        for gg in geoms:
                            xs += [c[0] for c in gg.coords]
                    L = [-x for x in xs if x < 0]; R = [x for x in xs if x >= 0]
                    for side, vals in (("_l", L), ("_r", R)):
                        v = d[key + side]
                        if (v is None) != (not vals) or (v is not None and abs(v - min(vals)) > 0.051):
                            fail("lane_offset_vs_shapely")
                        if v is not None and v > 15.05:
                            fail("lane_offset_out_of_roi")
            if not (0.0 <= mp["free_dist"] <= C.FREE_CAP):
                fail("free_range")
            st["map_samples"] += 1
    return dict(bad), ex, dict(st)


def check_token_lengths(recs, tok):
    """수정(2026-10-15): 실제 토크나이저(+특수토큰)로 모든 정답 토큰 수 + loss_mask_token_spans 재계산 일치."""
    lens = collections.Counter(); bad_tsp = 0
    for r in recs:
        enc = tok(r["answer"], return_offsets_mapping=True, add_special_tokens=False)
        lens[len(enc["input_ids"])] += 1
        offs = enc["offset_mapping"]
        tsp = [[min(k for k, (s0, e0) in enumerate(offs) if e0 > a0), max(k for k, (s0, e0) in enumerate(offs) if s0 < b0) + 1]
               for a0, b0 in r["loss_mask_spans"]]
        bad_tsp += int(tsp != r["loss_mask_token_spans"])
    return dict(lens), bad_tsp


def check_leak(recs, by_tok):
    bad = []; src = collections.Counter()
    for r in recs:
        info = by_tok[r["token"]]
        es = np.asarray(info["ego_status"], np.float64)       # infos 에서 직접(jsonl 값 신뢰하지 않음)
        past = [by_tok[t] for t in r["ego_status"]["past_tokens"]]
        ok = all(p["scene_token"] == info["scene_token"] for p in past)
        ok &= all(a["timestamp"] > b["timestamp"] for a, b in zip([info] + past, past))
        mode = r["modes"]["ego_in_prompt"]
        # 이전 프레임 카메라 = 직전 샘플(같은 장면) 또는 첫 샘플이면 현재 복제
        prev = past[0] if past else None
        ok &= r["prev_available"] == (prev is not None)
        ok &= r["cams_prev"] == cam_paths(prev if prev is not None else info)
        if mode == "none":
            vals, s = None, "not_in_prompt"
        elif mode == "current" and C.can_available(es):
            vals, s = (C.ego_speed_from_status(es), float(es[0]), float(es[5])), "can"
        else:
            pp = C.ego_from_past_poses(info, past)
            vals, s = (None, "unavailable") if pp is None else ((pp["speed"], pp["accel_long"], pp["yaw_rate"]), "pose_past")
        src[s] += 1
        ok &= s == r["ego_status"]["src"]
        parts = C.build_prompt_parts(int(np.argmax(info["gt_ego_fut_cmd"])), vals, mode, r["modes"]["object_speed"])
        ok &= parts == r["prompt_parts"] and C.join_prompt(parts) == r["prompt"]["user"]
        ok &= r["prompt"]["system"] == C.SYSTEM_PROMPT
        u = r["prompt"]["user"]; m = VAR_RE.search(u)
        ok &= m is not None and u[:m.start()] == parts["instruction"]
        ok &= (mode == "none") == (m.group(0).count("\n") == 1)
        ok &= r["ego_drop_ok"] == (vals is not None)
        if not ok:
            bad.append(r["token"])
    return bad, dict(src)


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
    # 수정(2026-10-15): v2-sensor 정답은 길어서 객체별 요약으로 표시
    lines = ["Objects:"] + [f"  #{k+1} {o['name']} ({o['lat']:+.1f},{o['lon']:+.1f}) l{o['length']} w{o['width']} head={o['heading']} "
                            f"{o['motion']} {o['vel_sector'] or ''} {o['speed']} fut={o['fut']} ttc={o['ttc']}   ({o['reason']}, {o['dist']}m)"
                            for k, o in enumerate(rec["targets"]["objects"])] + [l[:220] for l in al[1:]]
    sheet = Image.new("RGB", (3 * W, 2 * H + 16 * (len(lines) + 2)), "white")
    cols = [(255, 60, 60), (60, 200, 60), (60, 120, 255), (255, 170, 0), (200, 0, 200), (0, 200, 200)]
    tiles = {}
    for rr, row in enumerate(layout):
        for cc, cam in enumerate(row):
            im = Image.open(paths[cam]).convert("RGB").resize((W, H))
            ImageDraw.Draw(im).text((5, 5), cam, fill=(255, 255, 0), font=font)
            tiles[cam] = (im, cc * W, rr * H)
    for k, o in enumerate(rec["targets"]["objects"]):
        b = np.asarray(o["box"]); best = None
        for cam in C.QWEN_CAMS:
            uv, dep = project(b[None, :3], info["cams"][cam])
            if dep[0] > 0.5 and 0 <= uv[0, 0] < 1600 and 0 <= uv[0, 1] < 900 and (best is None or dep[0] < best[0]):
                best = (dep[0], cam)
        if best is None:
            continue
        cam = best[1]; d = ImageDraw.Draw(tiles[cam][0])
        uv, dep = project(box_corners(b), info["cams"][cam]); uv = uv * S
        col = cols[k % len(cols)]
        if (dep > 0.5).all():
            for a, e in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                d.line([tuple(uv[a]), tuple(uv[e])], fill=col, width=2)
        cu, _ = project(b[None, :3], info["cams"][cam])
        d.text((cu[0, 0] * S, cu[0, 1] * S), f"#{k+1} {o['name'][:5]} {o['lat']:+.1f},{o['lon']:+.1f}", fill=col, font=fsm)
    for cam, (im, x, y) in tiles.items():
        sheet.paste(im, (x, y))
    d = ImageDraw.Draw(sheet)
    d.text((5, 2 * H + 2), f"token {rec['token']}  {rec['prompt_parts']['command']}  {rec['prompt_parts']['ego']}  prev={rec['prev_available']}"
           f"  [ego src={rec['ego_status']['src']}]", fill="black", font=fsm)
    for i, l in enumerate(lines):
        d.text((5, 2 * H + 18 + 16 * i), l, fill="black", font=fsm)
    sheet.save(path)


def main():
    cb = C.ActionCodebook("data/qwen3d/nusc_action_codebook_k2048.npz")
    report = {}; scenes = {}
    with open(f"{OUT}/special_tokens.txt") as f:
        report["special_tokens_file_ok"] = f.read().split("\n")[:-1] == C.SPECIAL_TOKENS
    report["special_tokens_n"] = len(C.SPECIAL_TOKENS)
    st2 = json.load(open(f"{OUT}/stats.json")); st1 = json.load(open(f"{V1}/stats.json"))
    # 수정(2026-10-15): 실제 토크나이저(ckpts/qwen2-vl-2b) + 특수토큰으로 정답 길이 전수 측정
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("ckpts/qwen2-vl-2b")
    assert tok.add_tokens(C.SPECIAL_TOKENS, special_tokens=True) == len(C.SPECIAL_TOKENS)
    all_lens = collections.Counter()
    for split in ["train", "val"]:
        with open(f"{OUT}/teacher_gt_{split}.jsonl") as f:
            recs = [json.loads(l) for l in f]
        infos = pickle.load(open(f"data/infos/nuscenes_infos_{split}.pkl", "rb"))["infos"]
        by_tok = {i["token"]: i for i in infos}
        v1_cams = {}
        with open(f"{V1}/teacher_gt_{split}.jsonl") as f:
            for l in f:
                r = json.loads(l); v1_cams[r["token"]] = (r["cams"], r["targets"]["action_tokens"])
        bad = check_roundtrip(recs, cb)
        leak, src = check_leak(recs, by_tok)
        lens, bad_tsp = check_token_lengths(recs, tok)          # 수정(2026-10-15)
        all_lens.update(lens)
        sbad, sex, sst = check_sensor(recs, by_tok)              # 수정(2026-10-15)
        scenes[split] = {r["scene_token"] for r in recs}
        ade = [r["targets"]["action_recon_ade"] for r in recs if r["targets"]["action_recon_ade"] is not None]
        s2, s1 = st2["splits"][split], st1["splits"][split]
        report[split] = dict(
            n=len(recs), expected=EXPECT[split], count_ok=len(recs) == EXPECT[split],
            unique_tokens=len({r["token"] for r in recs}),
            order_matches_infos=[r["token"] for r in recs] == [i["token"] for i in infos],
            roundtrip_fail=len(bad), roundtrip_fail_examples=bad[:3],
            leak_fail=len(leak), ego_status_src=src,
            cams_equal_v1=sum(r["cams"] == v1_cams[r["token"]][0] for r in recs),
            action_tokens_equal_v1=sum(r["targets"]["action_tokens"] == v1_cams[r["token"]][1] for r in recs),
            action_recon_ade=round(float(np.mean(ade)), 5),
            prev_available=sum(r["prev_available"] for r in recs),
            loss_masked_samples=sum(bool(r["loss_mask_spans"]) for r in recs),
            modes=st2["modes"],
            incomplete=sum(1 for r in recs if not r["targets"]["future_complete"]),
            incomplete_marked_na=sum(1 for r in recs if not r["targets"]["future_complete"]
                                     and r["answer"].endswith("Intent: <na><_>\nAction: <na>" + "<_>" * 5)),
            # 수정(2026-10-15): v2-sensor 검사
            answer_format=sorted({r["modes"].get("answer_format") for r in recs}, key=str),
            answer_token_lengths=lens, answer_len_fixed=len(lens) == 1, loss_mask_token_spans_mismatch=bad_tsp,
            sensor_fail=sbad, sensor_fail_examples=sex, sensor_counts=sst,
            obj_ttc=s2.get("obj_ttc"), obj_ttc_first_t=s2.get("obj_ttc_first_t"), obj_motion=s2.get("obj_motion"),
            obj_heading_na=s2.get("obj_heading_na"), obj_fut_available=s2.get("obj_fut_available"),
            empty_slots=s2.get("empty_slots"), map_field=s2.get("map_field"), map_free=s2.get("map_free"),
            loss_mask_stats=s2.get("loss_mask"),
            lateral_v1=s1["intent"]["lateral"], lateral_v2=s2["lateral"],
            longitudinal_v1=s1["intent"]["longitudinal"], longitudinal_v2=s2["longitudinal"],
            uncertain=s2["uncertain"].get("True", 0),
            answer_tokens_v1=s2["qwen2vl_tokens"]["answer_v1_action_tokens_added"],
            answer_tokens_v2=s2["qwen2vl_tokens"]["answer_v2_special_tokens_added"])
        print(split, json.dumps(report[split], ensure_ascii=False), flush=True)
        if split == "val":
            rng = np.random.default_rng(0); want = []
            for cond in [lambda r: r["targets"]["intent"] and r["targets"]["intent"]["lateral"].startswith("lane change"),
                         lambda r: r["targets"]["intent"] and r["targets"]["intent"]["uncertain"],
                         lambda r: any(o["reason"] == "path" for o in r["targets"]["objects"]),
                         lambda r: sum(o["reason"] in ("approach", "vru") for o in r["targets"]["objects"]) >= 4,
                         lambda r: not r["targets"]["future_complete"]]:
                cand = [r for r in recs if cond(r)]
                want.append(cand[int(rng.integers(len(cand)))])
            for k, r in enumerate(want):
                contact_sheet(r, by_tok[r["token"]], f"{OUT}/check_{k}.png")
            report["contact_sheets"] = [dict(file=f"{OUT}/check_{k}.png", token=r["token"]) for k, r in enumerate(want)]
    report["scene_overlap_train_val"] = len(scenes["train"] & scenes["val"])
    report["answer_token_lengths_all"] = dict(all_lens)   # 수정(2026-10-15): train+val 전체가 한 값이어야 한다
    report["answer_len_fixed_all"] = len(all_lens) == 1
    print("scene overlap train∩val:", report["scene_overlap_train_val"], " special tokens ok:", report["special_tokens_file_ok"])
    with open(f"{OUT}/check_report.json", "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
