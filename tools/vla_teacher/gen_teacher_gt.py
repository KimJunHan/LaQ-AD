# 수정(2026-10-15): 교사 VLM ① 정답 자동 생성 — nuScenes infos → data/vla_teacher/teacher_gt_{train,val}.jsonl.
#   CPU 전용(CUDA_VISIBLE_DEVICES=""). 기존 data/ 파일은 덮어쓰지 않는다(출력 파일이 있으면 중단; --force 로만 덮어씀,
#   그것도 data/vla_teacher/ 안에 한정).
#   사용(cwd = 저장소 루트):
#     CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tools/vla_teacher/gen_teacher_gt.py --explore   # 임계값 근거 통계(train)
#     CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tools/vla_teacher/gen_teacher_gt.py             # 생성 + stats.json
import argparse, collections, json, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT_DIR = "data/vla_teacher"
CODEBOOK = "data/qwen3d/nusc_action_codebook_k2048.npz"
TOKENIZER = "ckpts/qwen2-vl-2b"


def load_infos(split):
    with open(f"data/infos/nuscenes_infos_{split}.pkl", "rb") as f:
        d = pickle.load(f)
    return d["infos"] if isinstance(d, dict) else d


def cam_paths(info):
    """tools/qwen_cache.py · spatial_common.load_six_cams 와 같은 규약: data_root(data/nuscenes) 상대 'samples/...'."""
    out = []
    for cam in C.QWEN_CAMS:
        p = info["cams"][cam]["data_path"]
        if "samples/" in p:
            p = "samples/" + p.split("samples/", 1)[1]
        out.append(os.path.join("data/nuscenes", p))
    return out


def pose_speeds(infos):
    """CAN 결측 장면용 자차 속도(라벨 전용): ego2global 위치의 후진 차분(장면 첫 프레임은 전진 차분).
    반환 {token: (speed, src)}."""
    by_scene = collections.defaultdict(list)
    for i in infos:
        by_scene[i["scene_token"]].append(i)
    out = {}
    for sc, lst in by_scene.items():
        lst.sort(key=lambda r: r["timestamp"])
        for k, i in enumerate(lst):
            if k > 0:
                a, b, src = lst[k - 1], i, "pose_backward"
            elif len(lst) > 1:
                a, b, src = i, lst[1], "pose_forward"
            else:
                out[i["token"]] = (0.0, "none"); continue
            dp = np.asarray(b["ego2global_translation"][:2]) - np.asarray(a["ego2global_translation"][:2])
            dt = (b["timestamp"] - a["timestamp"]) * 1e-6
            out[i["token"]] = (float(np.linalg.norm(dp) / dt), src)
    return out


def explore():
    """임계값 근거(train 만). common.py 주석의 수치를 재현한다."""
    infos = load_infos("train")
    V0, V1, VE, TH, S, R, CMD, LC = [], [], [], [], [], [], [], []
    for i in infos:
        if not C.future_complete(i):
            continue
        es = np.asarray(i["ego_status"], np.float64)
        f = C.fut_cum(i)
        lab = C.intent_labels(f, C.ego_speed_from_status(es), i["map_annos"].get(1, []))
        TH.append(lab["theta_end_deg"]); S.append(lab["path_len"]); R.append(lab["lat_resid"])
        if lab["lateral"].startswith("lane change") and C.can_available(es):
            LC.append((lab["lateral"].split()[-1], float(es[5])))
        CMD.append(int(np.argmax(i["gt_ego_fut_cmd"])))
        if C.can_available(es):
            V0.append(C.ego_speed_from_status(es)); V1.append(np.linalg.norm(f[0]) / C.DT); VE.append(lab["v_end"])
    V0, V1, VE, TH, S, R, CMD = map(np.array, (V0, V1, VE, TH, S, R, CMD))
    noise = np.abs(V1 - V0)
    print("측정 잡음 |CAN − 첫 0.5s 포즈 속도| p50/80/90/95:", np.percentile(noise, [50, 80, 90, 95]).round(3))
    print("Δv=v_end−v0 p5/20/50/80/95:", np.percentile(VE - V0, [5, 20, 50, 80, 95]).round(3))
    cmdlab = np.where(CMD == 0, "R", np.where(CMD == 1, "L", "S"))
    for t in [5, 10, 12.5, 15, 17.5, 20, 25, 30]:
        lab = np.where((TH >= t) & (S >= C.MIN_PATH), "L", np.where((TH <= -t) & (S >= C.MIN_PATH), "R", "S"))
        print(f"TURN_TH {t:>5}°: 명령 일치율 {np.mean(lab == cmdlab):.4f}")
    m = (np.abs(TH) < C.TURN_TH_DEG) & (S >= C.MIN_PATH)
    print(f"|r| (비회전 |θ|<{C.TURN_TH_DEG}°·주행 s≥{C.MIN_PATH}m, n={m.sum()}) p90/95/98/99:",
          np.percentile(np.abs(R[m]), [90, 95, 98, 99]).round(3))
    # 구분선 교차(차로변경) 표본의 현재 yaw rate(+ = 왼쪽) — 방향 부호 검증(왼쪽 변경이면 대체로 +)
    for side in ["left", "right"]:
        w = np.array([y for x, y in LC if x == side])
        if len(w):
            print(f"lane change {side}: n={len(w)}, 현재 yaw rate p10/50/90 =", np.percentile(w, [10, 50, 90]).round(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--force", action="store_true", help="data/vla_teacher/ 안의 기존 산출물 덮어쓰기 허용")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    a = ap.parse_args()
    if a.explore:
        explore(); return
    os.makedirs(OUT_DIR, exist_ok=True)
    outs = [f"{OUT_DIR}/teacher_gt_{s}.jsonl" for s in a.splits] + [f"{OUT_DIR}/instruction.txt", f"{OUT_DIR}/stats.json"]
    exist = [p for p in outs if os.path.exists(p)]
    if exist and not a.force:
        raise SystemExit(f"[중단] 기존 산출물 있음(덮어쓰지 않음): {exist}")

    cb = C.ActionCodebook(CODEBOOK)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tok_act = AutoTokenizer.from_pretrained(TOKENIZER)
    n_added = tok_act.add_tokens([f"<action_{k}>" for k in range(cb.K)], special_tokens=True)
    assert n_added == cb.K, n_added

    with open(f"{OUT_DIR}/instruction.txt", "w") as f:
        f.write("[SYSTEM]\n" + C.SYSTEM_PROMPT + "\n\n[USER — 이미지 6장 뒤에 오는 텍스트]\n" + C.INSTRUCTION +
                "\nCommand: <go straight|turn left|turn right>.\n"
                "Ego state: speed <v> m/s, longitudinal acceleration <+a> m/s^2, yaw rate <+w> rad/s "
                "(positive = turning left).   |   Ego state: unavailable.   (CAN 결측 장면)\n")

    stats = dict(thresholds=dict(obj_range_m=C.OBJ_RANGE, max_objects=C.MAX_OBJECTS, moving_th_mps=C.MOVING_TH,
                                 valid_rule="num_lidar_pts>0 & gt_names in 10 det classes & xy dist<=55m",
                                 stop_v_mps=C.STOP_V, dv_th_mps=C.DV_TH, turn_th_deg=C.TURN_TH_DEG,
                                 lane_change_rule="path crosses map divider (map_annos label 1), non-turn", min_path_m=C.MIN_PATH,
                                 ped_crossing_range_m=C.PED_CROSS_RANGE),
                 codebook=dict(path=CODEBOOK, K=cb.K), splits={})
    for split in a.splits:
        infos = load_infos(split)
        pspd = pose_speeds(infos)
        cls_cnt = collections.Counter(); dist_hist = np.zeros(11, int); nobj = []
        n_in_range_trunc = 0; nonfinite_vel = 0
        long_cnt = collections.Counter(); lat_cnt = collections.Counter(); joint = collections.Counter()
        cmd_cnt = collections.Counter(); cmd_vs_lat = collections.Counter()
        tok_use = np.zeros(cb.K, int); ade, fde = [], []
        inc = collections.Counter(); can_missing = 0; spd_src = collections.Counter()
        map_cnt = collections.Counter(); len_raw, len_act, len_prompt = [], [], []
        moving_cnt = collections.Counter()
        path = f"{OUT_DIR}/teacher_gt_{split}.jsonl"
        with open(path, "w") as fo:
            for info in infos:
                es = np.asarray(info["ego_status"], np.float64)
                cmd = int(np.argmax(info["gt_ego_fut_cmd"]))
                can = C.can_available(es)
                if can:
                    v0, src = C.ego_speed_from_status(es), "can"
                else:
                    can_missing += 1
                    v0, src = pspd[info["token"]]
                spd_src[src] += 1
                objs, n_in = C.object_rows(info)
                n_in_range_trunc += int(n_in > C.MAX_OBJECTS)
                mp = C.map_context(info)
                complete = C.future_complete(info)
                mask = np.asarray(info["gt_ego_fut_masks"], np.float64)
                fut = C.fut_cum(info)
                if complete:
                    intent = C.intent_labels(fut, v0, info["map_annos"].get(1, []))
                    toks, rec = cb.tokenize(fut)
                    e = np.linalg.norm(rec - fut, axis=1)
                    ade.append(e.mean()); fde.append(e[-1])
                    for k in toks: tok_use[k] += 1
                else:
                    intent = toks = rec = None
                    inc[int(mask.sum())] += 1
                ego_speed_ans = round(v0, 1)
                answer = C.build_answer(ego_speed_ans, objs, mp, intent, toks)
                user = C.build_user_prompt(cmd, es)
                rec_d = dict(
                    token=info["token"], scene_token=info["scene_token"], timestamp=int(info["timestamp"]),
                    cams=cam_paths(info),
                    command=dict(id=cmd, text=C.CMD_TEXT[cmd], onehot=[int(v) for v in info["gt_ego_fut_cmd"]]),
                    ego_status=dict(available=can, speed=round(C.ego_speed_from_status(es), 3) if can else None,
                                    accel_long=round(float(es[0]), 4) if can else None,
                                    accel_lat=round(float(es[1]), 4) if can else None,
                                    yaw_rate=round(float(es[5]), 5) if can else None,
                                    steer=round(float(es[9]), 5) if can else None,
                                    raw=[float(v) for v in es]),  # 원값(float32 정확값) — 반올림하면 프롬프트 재생성이 어긋남(누설 검사로 발견)
                    prompt=dict(system=C.SYSTEM_PROMPT, user=user),
                    answer=answer,
                    targets=dict(
                        ego_speed=ego_speed_ans, ego_speed_src=src, objects=objs, n_objects_in_range=n_in,
                        map=mp, future_complete=complete, fut_mask=[int(v) for v in mask],
                        fut_traj=[[round(float(v), 4) for v in p] for p in fut],
                        intent=intent,
                        action_tokens=toks,
                        action_recon_traj=None if rec is None else [[round(float(v), 4) for v in p] for p in rec],
                        action_recon_ade=None if rec is None else round(float(e.mean()), 5),
                    ),
                )
                fo.write(json.dumps(rec_d, ensure_ascii=False) + "\n")
                # 통계
                nobj.append(len(objs))
                for r in objs:
                    cls_cnt[r["name"]] += 1
                    dist_hist[min(int(r["dist"] // 5), 10)] += 1
                    nonfinite_vel += int(not r["vel_finite"])
                    moving_cnt["moving" if r["vel_sector"] else "stationary"] += 1
                cmd_cnt[C.CMD_TEXT[cmd]] += 1
                if intent:
                    long_cnt[intent["longitudinal"]] += 1; lat_cnt[intent["lateral"]] += 1
                    joint[f"{intent['longitudinal']} | {intent['lateral']}"] += 1
                    cmd_vs_lat[f"cmd={C.CMD_TEXT[cmd]} -> {intent['lateral']}"] += 1
                if mp is None:
                    map_cnt["unavailable"] += 1
                else:
                    pc = mp["ped_crossing"]
                    map_cnt["no_ped_crossing" if pc is None else ("on_ped_crossing" if pc["sector"] == "here" else "ped_crossing_near")] += 1
                len_raw.append(len(tok(answer)["input_ids"]))
                len_act.append(len(tok_act(answer)["input_ids"]))
                len_prompt.append(len(tok(C.SYSTEM_PROMPT + "\n" + user)["input_ids"]))
        n = len(infos); nc = n - sum(inc.values())
        used = tok_use[tok_use > 0]
        p = used / used.sum() if len(used) else used

        def pct(x):
            x = np.asarray(x); return dict(mean=round(float(x.mean()), 2), p50=float(np.percentile(x, 50)),
                                           p95=float(np.percentile(x, 95)), max=int(x.max()), min=int(x.min()))
        stats["splits"][split] = dict(
            samples=n, scenes=len({i["scene_token"] for i in infos}),
            future_complete=nc, future_incomplete=sum(inc.values()),
            future_incomplete_by_valid_steps={str(k): v for k, v in sorted(inc.items())},
            can_missing_samples=can_missing, ego_speed_source=dict(spd_src),
            objects=dict(per_class=dict(cls_cnt.most_common()), total=int(sum(cls_cnt.values())),
                         per_sample=pct(nobj), samples_zero_objects=int(sum(1 for v in nobj if v == 0)),
                         samples_truncated_at_12=n_in_range_trunc, nonfinite_velocity_set_to_0=nonfinite_vel,
                         moving_vs_stationary=dict(moving_cnt),
                         dist_hist_5m={f"{5*k}-{5*k+5}" if k < 10 else "50-55": int(v) for k, v in enumerate(dist_hist)}),
            map=dict(map_cnt),
            command=dict(cmd_cnt),
            intent=dict(longitudinal=dict(long_cnt), lateral=dict(lat_cnt), joint=dict(joint.most_common()),
                        command_vs_lateral=dict(sorted(cmd_vs_lat.items()))),
            action=dict(recon_ade_m=round(float(np.mean(ade)), 5), recon_fde_m=round(float(np.mean(fde)), 5),
                        recon_fde_p95_m=round(float(np.percentile(fde, 95)), 5),
                        tokens_used=int((tok_use > 0).sum()), tokens_unused=int((tok_use == 0).sum()),
                        usage_entropy_bits=round(float(-(p * np.log2(p)).sum()), 3),
                        top20=[[int(k), int(tok_use[k])] for k in np.argsort(-tok_use)[:20]]),
            qwen2vl_tokens=dict(answer_raw_tokenizer=pct(len_raw),
                                answer_with_action_tokens_added=pct(len_act),
                                prompt_text_only=pct(len_prompt)),
        )
        print(f"[{split}] {n} 샘플 -> {path}  (미래 불완전 {sum(inc.values())}, CAN 결측 {can_missing})")
    with open(f"{OUT_DIR}/stats.json", "w") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    print(json.dumps(stats, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
