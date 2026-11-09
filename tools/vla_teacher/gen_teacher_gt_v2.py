# 수정(2026-10-15): 교사 VLM ① 정답 v2 (사용자 승인 4개 개선) — nuScenes infos → data/vla_teacher/v2/.
#   v1(gen_teacher_gt.py / data/vla_teacher/*.jsonl)은 그대로 둔다(재현 가능). CPU 전용.
#   NAS IO 최소화: infos(.pkl, 로컬) + map-expansion json(도시당 1회) 만 읽는다. 이미지는 읽지 않는다.
#   사용(cwd = 저장소 루트):
#     export CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=.
#     python tools/vla_teacher/gen_teacher_gt_v2.py [--map-root data/nuscenes] [--force]
#   산출물: teacher_gt_{train,val}.jsonl, instruction.txt, grammar.txt, special_tokens.txt, stats.json
import argparse, collections, json, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_v2 as C  # noqa: E402
import common as C1  # noqa: E402
import lane_v2 as LV  # noqa: E402
from gen_teacher_gt import load_infos, cam_paths, pose_speeds  # noqa: E402  (v1 함수 재사용 — 동일 규약)

OUT_DIR = "data/vla_teacher/v2"
V1_DIR = "data/vla_teacher"
CODEBOOK = "data/qwen3d/nusc_action_codebook_k2048.npz"
TOKENIZER = "ckpts/qwen2-vl-2b"
UNCERTAIN_WEIGHT = 0.3

# 수정(2026-10-15): 정답 형식 v2-sensor(고정 길이) 문법으로 교체 — 이전 가변 길이 v2 문법은 백업(scratchpad v2_preIntegrate_backup)·README 에 기록.
GRAMMAR = r"""# 교사 VLM 정답 v2-sensor 문법 (common_v2.parse_answer 가 엄격히 강제; format_parsed 로 왕복 정확 일치)
# 모든 표본의 정답 토큰 수가 같다(ckpts/qwen2-vl-2b 토크나이저 + special_tokens.txt; check_teacher_gt_v2.py 가 전수 검사).
answer   := L1 "\n" L2 "\n" L3 "\n" L4                        (정확히 4줄, 끝 개행 없음)
L1       := "Objects: " OBJ{k} EMPTY{8-k}                       (k = 0..8, 채운 칸은 거리 오름차순, 공백 없음)
OBJ      := CLS SEC SN SN UN UN HEAD MOTION FUT FUT FUT RISK    (57 토큰; object_speed=dir_only 면 53)
            SN 1 = 횡 x(+ 오른쪽), SN 2 = 종 y(+ 전방) — lidar 프레임 박스 중심 m
            UN 1 = 길이 l, UN 2 = 폭 w (m, gt_boxes[3], [4])
HEAD     := <hF>|<hFR>|<hR>|<hBR>|<hB>|<hBL>|<hL>|<hFL>  | "<na>"   (박스 yaw 의 자차 기준 진행방위; traffic_cone = <na>)
MOTION   := ("<static>" | SEC) UN  | "<na>" "<_>"x4             (moving = 속력 >= 0.5 m/s: 이동방향 섹터 + 속력; static 도 실제 속력;
                                                                 GT 속도 NaN = <na>. dir_only: ("<static>"|SEC) | "<na>")
FUT      := SN SN | "<na>" "<_>"x9                              (1 s / 2 s / 3 s 누적 변위 dx, dy m; GT 미래 마스크 미완 = <na>)
RISK     := <ttc_lt2>|<ttc_2_4>|<ttc_4_6>|<ttc_safe> | "<na>"   (GT 미래 기반 간격<=TTC_GAP 첫 시각; 자차 미래 불완전 = <na>)
EMPTY    := "<none>" "<_>"x56                                   (dir_only: x52)
L2       := "Map: " ( MAP | "<na>" "<_>"x67 )                   (68 토큰)
MAP      := PC "<div>" I2 "<bnd>" I2 AT0 AT10 AT20 "<free>" UN
PC       := "<pc>" SEC UN | "<pc_on>" "<_>"x5 | "<pc_none>" "<_>"x5   (20 m 안 최근접 횡단보도)
ATy      := "<aty>" LANE LANE LANE LANE                          (y = 0, 10, 20 m 전방 단면: 왼 divider, 오른 divider,
                                                                 왼 boundary, 오른 boundary 까지 |x| m)
LANE     := UN | "<none>" "<_>"x3                               (그쪽에 교점 없음, ROI |x|<=15 m)
"<free>" := 자차 폭 회랑(|x|<=0.925 m) 전방 0..30 m 에서 처음 만나는 boundary/객체 박스의 y (없으면 30.0)
L3       := "Intent: " ( LON LAT | "<na><_>" )
L4       := "Action: " ( ACT{6} | "<na>" "<_>"x5 )
CLS      := <car>|<truck>|<bus>|<trailer>|<cv>|<ped>|<moto>|<bike>|<cone>|<barrier>
SEC      := <F>|<FR>|<R>|<BR>|<B>|<BL>|<L>|<FL>                (0 deg = 전방, 시계방향 45 deg 구간)
LON      := <accelerate>|<keep_speed>|<decelerate>|<stop>
LAT      := <go_straight>|<turn_left>|<turn_right>|<lane_change_left>|<lane_change_right>
ACT      := <action_K>, K = 0..2040 (선행 0 없음)
SN       := ("+"|"-") D D "." D       ("-00.0" 금지: 0 은 "+00.0"; |값| <= 99.9, 범위 밖은 생성 실패 — 자르지 않음)
UN       := D D "." D                 (0 <= 값 <= 99.9)
I2       := D D
손실 제외(loss_mask_spans): "<na>" 대체형 전부, 미래 불완전 표본의 Intent/Action 줄, 이전 프레임 없는 표본의 MOTION 속력 UN.
"""


def scene_past(infos):
    """{token: [t−1, t−2] info}(같은 장면, 시간 역순). 미래 info 는 넣지 않는다."""
    by = collections.defaultdict(list)
    for i in infos:
        by[i["scene_token"]].append(i)
    out = {}
    for lst in by.values():
        lst.sort(key=lambda r: r["timestamp"])
        for k, i in enumerate(lst):
            out[i["token"]] = [lst[k - 1 - j] for j in range(2) if k - 1 - j >= 0]
    return out


def ego_inputs(info, past, mode="current"):
    """자차 상태 값과 출처. mode='current': CAN 우선, 결측이면 과거 포즈, 과거 없으면 unavailable.
    mode='past_pose': 항상 과거 포즈(과거 없으면 unavailable). ('none' 은 호출측에서 프롬프트에 넣지 않음)"""
    es = np.asarray(info["ego_status"], np.float64)
    if mode != "past_pose" and C.can_available(es):
        return (C.ego_speed_from_status(es), float(es[0]), float(es[5])), "can", None
    pp = C.ego_from_past_poses(info, past)
    if pp is None:
        return None, "unavailable", None
    return (pp["speed"], pp["accel_long"], pp["yaw_rate"]), "pose_past", pp["n_past"]


def r4(a):
    return [[round(float(v), 4) for v in p] for p in a]


def pct(x):
    x = np.asarray(x, np.float64)
    return dict(mean=round(float(x.mean()), 2), p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)),
                max=float(x.max()), min=float(x.min()))


def _ttc_explore_one(info):
    """미래 완전 표본의 선택 객체별 간격 시계열(13 시각). 라벨 규칙과 같은 v0·선택·자차 궤적."""
    es = np.asarray(info["ego_status"], np.float64)
    if not C.future_complete(info):
        return []
    v0 = C.ego_speed_from_status(es) if C.can_available(es) else _PSPD[info["token"]][0]
    objs, _, _ = C.select_objects(info, v0)
    ego = C.ego_track(info)
    return [C.agent_gap_series(info, r["box_index"], ego).tolist() for r in objs]


_PSPD = {}


def explore_ttc():
    """수정(2026-10-15): TTC_GAP 근거 — train 전체(val 미사용). 선택 객체별 0–3 s 최소 간격 분포와 후보 임계값별 구간 분포."""
    global _PSPD
    import multiprocessing as mp_
    infos = load_infos("train")
    _PSPD = pose_speeds(infos)
    with mp_.get_context("fork").Pool(8) as pool:   # GPU1 학습 데이터로더와 CPU 공유 → 8 프로세스
        res = pool.map(_ttc_explore_one, infos, chunksize=64)
    G = np.asarray([g for r in res for g in r], np.float64)
    m3 = G[:, :7].min(1)
    out = dict(n_samples_future_complete_with_objects=sum(1 for r in res if r), n_objects=len(G),
               min_gap_0_3s_percentiles={p: round(float(np.percentile(m3, p)), 4) for p in (1, 5, 10, 25, 50)},
               p5_rounded=round(float(np.percentile(m3, 5)), 1), buckets={})
    for th in (0.0, 0.5, 1.0, round(float(np.percentile(m3, 5)), 1), 2.0):
        c = collections.Counter(C.ttc_bucket(g, th)[0] for g in G)
        out["buckets"][str(th)] = {k: round(c[k] / len(G), 4) for k in ("lt2", "2_4", "4_6", "safe")}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="data/vla_teacher/v2/ 안의 기존 산출물 덮어쓰기 허용")
    ap.add_argument("--map-root", default="data/nuscenes", help="maps/expansion/<loc>.json 을 담은 루트(NuScenesMap dataroot)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    # 수정(2026-10-15): 검토 반영 — 입력 설계 선택을 플래그로(재생성 수 분). 기본값 = 사용자 결정(current, numeric).
    ap.add_argument("--ego_in_prompt", choices=C.EGO_MODES, default="current")
    ap.add_argument("--object_speed", choices=C.SPEED_MODES, default="numeric")
    ap.add_argument("--out_dir", default=OUT_DIR, help="다른 모드 변형은 별도 폴더로(예: data/vla_teacher/v2_egonone)")
    # 수정(2026-10-15): v2-sensor TTC 간격 임계값 근거(train 전체, val 미사용) — 출력만 하고 파일은 쓰지 않는다.
    ap.add_argument("--explore_ttc", action="store_true")
    a = ap.parse_args()
    if a.explore_ttc:
        return explore_ttc()
    if C.TTC_GAP is None:
        raise SystemExit("common_v2.TTC_GAP 미설정 — --explore_ttc 결과로 고정할 것")
    OUT_DIR = a.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    outs = [f"{OUT_DIR}/teacher_gt_{s}.jsonl" for s in a.splits] + [f"{OUT_DIR}/stats.json"]
    exist = [p for p in outs if os.path.exists(p)]
    if exist and not a.force:
        raise SystemExit(f"[중단] 기존 산출물 있음(덮어쓰지 않음): {exist}")

    cb = C.ActionCodebook(CODEBOOK)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    n_added = tok.add_tokens(C.SPECIAL_TOKENS, special_tokens=True)
    assert n_added == len(C.SPECIAL_TOKENS), (n_added, len(C.SPECIAL_TOKENS))
    tok_v1 = AutoTokenizer.from_pretrained(TOKENIZER)
    tok_v1.add_tokens([f"<action_{k}>" for k in range(cb.K)], special_tokens=True)   # v1 통계와 같은 조건

    with open(f"{OUT_DIR}/special_tokens.txt", "w") as f:
        f.write("\n".join(C.SPECIAL_TOKENS) + "\n")
    with open(f"{OUT_DIR}/grammar.txt", "w") as f:
        f.write(GRAMMAR)
    with open(f"{OUT_DIR}/instruction.txt", "w") as f:
        f.write("[SYSTEM]\n" + C.SYSTEM_PROMPT + "\n\n[USER — 이미지 6장 뒤 텍스트 = prompt_parts 를 '\\n' 로 이음]\n" +
                C.build_instruction(a.ego_in_prompt, a.object_speed) + "\nCommand: <go straight|turn left|turn right>.   |   " + C.CMD_DROPPED +
                "   (SFT 드롭아웃)\nEgo state: speed <vv.v> m/s, longitudinal acceleration <+a.aa> m/s^2, yaw rate <+w.www> rad/s "  # 수정(2026-10-15): 고정폭
                "(positive = turning left).   |   " + C.EGO_DROPPED + "   (결측 또는 SFT 드롭아웃)\n")

    lane_idx = {}
    stats = dict(version=2, answer_format=C.ANSWER_FORMAT,  # 수정(2026-10-15): v2-sensor
                 modes=dict(ego_in_prompt=a.ego_in_prompt, object_speed=a.object_speed, answer_format=C.ANSWER_FORMAT,
                                       inputs="cams_prev(t-0.5s, 6) + cams(t0, 6) + command" +
                                              ("" if a.ego_in_prompt == "none" else " + ego line")),
                 thresholds=dict(max_objects=C.MAX_OBJECTS, corridor_half_m=C.CORRIDOR_HALF, ttc_max_s=C.TTC_MAX,
                                 closing_min_mps=C.CLOSING_MIN, vru_range_m=C.VRU_RANGE,
                                 straight_min_len_m=C.STRAIGHT_MIN_LEN, uncertain_intent_weight=UNCERTAIN_WEIGHT,
                                 # 수정(2026-10-15): v2-sensor 규칙 상수
                                 sensor=dict(ttc_gap_m=C.TTC_GAP, ttc_times_s=C.TTC_T, ego_box=[C.EGO_L, C.EGO_W, C.EGO_FWD],
                                             fut_steps=list(C.FUT_STEPS), lane_y_m=list(C.LANE_Y), free_cap_m=C.FREE_CAP,
                                             free_half_w_m=C.FREE_HALF_W, slot_tokens=C.slot_tokens(a.object_speed),
                                             map_line_tokens=C.MAP_LINE_FIELDS),
                                 lane=dict(assign_r=LV.ASSIGN_R, assign_max=LV.ASSIGN_MAX, head_tol_deg=LV.HEAD_TOL_DEG,
                                           adj_min=LV.ADJ_MIN, adj_max=LV.ADJ_MAX, half_lane=LV.HALF_LANE,
                                           head_min_deg=LV.HEAD_MIN_DEG, yaw_opp=LV.YAW_OPP,
                                           resid_fallback=LV.RESID_FALLBACK, bfs_depth=LV.BFS_DEPTH),
                                 v1_unchanged="obj filter / map_context / longitudinal / turn / action tokenization"),
                 codebook=dict(path=CODEBOOK, K=cb.K), special_tokens=len(C.SPECIAL_TOKENS), splits={})
    for split in a.splits:
        infos = load_infos(split)
        past = scene_past(infos)
        pspd = pose_speeds(infos)                    # v1 의 라벨용 v0(CAN 결측 시) — 종방향 라벨을 v1 과 동일하게 유지
        v1_path = f"{V1_DIR}/teacher_gt_{split}.jsonl"
        v1 = {}
        if os.path.exists(v1_path):
            with open(v1_path) as f:
                for l in f:
                    r = json.loads(l); v1[r["token"]] = (r["targets"]["intent"], r["targets"]["action_tokens"], r["answer"])
        S = collections.defaultdict(collections.Counter)
        ade, ade_copy, ade_copy_tok, len_v2, len_v1, len_prompt, nobj = [], [], [], [], [], [], []
        pp_err = collections.defaultdict(list)
        ans_len = [None]   # 수정(2026-10-15): 고정 길이 기준(분할별 첫 표본)
        path = f"{OUT_DIR}/teacher_gt_{split}.jsonl"
        with open(path, "w") as fo:
            for info in infos:
                tkn = info["token"]
                es = np.asarray(info["ego_status"], np.float64)
                cmd = int(np.argmax(info["gt_ego_fut_cmd"]))
                ego_cur, cur_src, cur_np = ego_inputs(info, past[tkn], "current")   # copy-baseline·라벨 가드용(항상)
                if a.ego_in_prompt == "none":
                    ego_vals, ego_src, n_past = None, "not_in_prompt", None
                else:
                    ego_vals, ego_src, n_past = ego_inputs(info, past[tkn], a.ego_in_prompt)
                prev = past[tkn][0] if past[tkn] else None
                S["ego_status_src"][ego_src if n_past is None else f"{ego_src}_{n_past}"] += 1
                # 과거 포즈 추정 정확도(CAN 있는 표본에서 비교 — 정보 출처 검증용 통계)
                if cur_src == "can" and past[tkn]:
                    pe = C.ego_from_past_poses(info, past[tkn])
                    pp_err["speed"].append(pe["speed"] - ego_cur[0]); pp_err["yaw_rate"].append(pe["yaw_rate"] - ego_cur[2])
                    if pe["n_past"] == 2:
                        pp_err["accel_long"].append(pe["accel_long"] - ego_cur[1])
                v0 = C.ego_speed_from_status(es) if C.can_available(es) else pspd[tkn][0]   # 라벨용(v1 과 동일)
                objs, n_in, sel_meta = C.select_objects(info, v0)
                C.enrich_objects(info, objs)          # 수정(2026-10-15): v2-sensor 필드(크기·진행방위·운동·미래·TTC)
                mp = C.map_context_sensor(info)       # 수정(2026-10-15): + 단면 횡거리·자유거리
                complete = C.future_complete(info)
                mask = np.asarray(info["gt_ego_fut_masks"], np.float64)
                fut = C.fut_cum(info)
                intent = toks = rec = None; weight = 0.0
                if complete:
                    lab = C1.intent_labels(fut, v0, info["map_annos"].get(1, []))
                    lat_v1 = lab["lateral"]
                    sig = None
                    if not lat_v1.startswith("turn") and lab["path_len"] >= C1.MIN_PATH:
                        loc = info["map_location"]
                        if loc not in lane_idx:
                            lane_idx[loc] = LV.LaneIndex(a.map_root, loc)
                        xy = np.concatenate([np.zeros((1, 2)), fut], 0)
                        yr = None if ego_cur is None else ego_cur[2]   # 라벨 쪽 가드(입력 아님)
                        sig = LV.lane_signals(lane_idx[loc], info, xy, C.heading_series(xy), yr,
                                              lab["divider_crossing"], lab["lat_resid"])
                        dec = sig["decision"]
                        lat = dec if (dec or "").startswith("lane change") else "go straight"
                        uncertain = dec == "uncertain"
                    else:
                        lat, uncertain = lat_v1, False
                    weight = UNCERTAIN_WEIGHT if uncertain else 1.0
                    intent = dict(lab, lateral=lat, lateral_v1=lat_v1, uncertain=uncertain,
                                  lc_signals=None if sig is None else dict(
                                      a_topology=sig["a"], b_divider=sig["b"], c_kinematic=sig["c"],
                                      c_mode=sig["c_mode"], decision=sig["decision"], L0=sig["L0"], L6=sig["L6"],
                                      L0_is_connector=sig["L0_is_connector"], d_ref=sig["d_ref"],
                                      rel_head_deg=sig["rel_head_deg"]))
                    toks, rec = cb.tokenize(fut)
                    e = np.linalg.norm(rec - fut, axis=1); ade.append(e.mean())
                    # 통계
                    S["longitudinal"][lab["longitudinal"]] += 1; S["lateral"][lat] += 1
                    S["lateral_v1"][lat_v1] += 1
                    S["lateral_v1_to_v2"][f"{lat_v1} -> {lat}{' (uncertain)' if uncertain else ''}"] += 1
                    if sig is not None:
                        S["lc_signal_combo"]["a%d b%d c%d" % (sig["a"] is not None, sig["b"] is not None, sig["c"] is not None)] += 1
                        S["lc_c_mode"][sig["c_mode"]] += 1
                        S["lc_L0_found"][str(sig["L0"] is not None)] += 1
                    S["uncertain"][str(uncertain)] += 1
                    S["command_vs_lateral"][f"cmd={C.CMD_TEXT[cmd]} -> {lat}"] += 1
                    if tkn in v1:
                        S["v1_compare"]["longitudinal_same"] += int(v1[tkn][0]["longitudinal"] == lab["longitudinal"])
                        S["v1_compare"]["action_tokens_same"] += int(v1[tkn][1] == toks)
                        S["v1_compare"]["turn_same"] += int(lat_v1 == v1[tkn][0]["lateral"])
                        S["v1_compare"]["n"] += 1
                else:
                    S["future_incomplete_by_valid_steps"][str(int(mask.sum()))] += 1
                cbase = C.copy_baseline(cmd, ego_cur, cb)   # 입력 모드와 무관하게 '현재 자차 상태만' 기준선
                cbase["ego_src"] = cur_src
                if complete:
                    ct = np.asarray(cbase["traj"]); ade_copy.append(np.linalg.norm(ct - fut, axis=1).mean())
                    ade_copy_tok.append(np.linalg.norm(cb.decode(cbase["action_tokens"]) - fut, axis=1).mean())
                    cbase["ade_vs_gt"] = round(float(ade_copy[-1]), 4)
                    cbase["intent_match"] = dict(longitudinal=cbase["intent"]["longitudinal"] == intent["longitudinal"],
                                                 lateral=cbase["intent"]["lateral"] == intent["lateral"])
                    S["copy_intent_match"]["lon"] += int(cbase["intent_match"]["longitudinal"])
                    S["copy_intent_match"]["lat"] += int(cbase["intent_match"]["lateral"])
                    S["copy_intent_match"]["n"] += 1
                intent_ans = None if intent is None else dict(longitudinal=intent["longitudinal"], lateral=intent["lateral"])
                answer = C.build_answer(objs, mp, intent_ans, toks, a.object_speed)
                parts = C.build_prompt_parts(cmd, ego_vals, a.ego_in_prompt, a.object_speed)
                user = C.join_prompt(parts)
                spans = C.answer_spans(answer, a.object_speed)
                # 손실 마스킹 구간: 미래 불완전 → Intent·Action 줄('<na>' 를 학습하지 않음),
                #   이전 프레임 없음(장면 첫 샘플, 현재 프레임 복제) → 객체 속력 숫자(관측 불가)
                #   수정(2026-10-15): + v2-sensor '<na>' 대체형(GT 값 미상) 전부. 정렬·병합된 구간.
                lm = C.loss_mask_spans(spans, complete, prev is not None)
                enc = tok(answer, return_offsets_mapping=True, add_special_tokens=False)
                offs = enc["offset_mapping"]
                # 수정(2026-10-15): 고정 길이 강제 — 첫 표본 길이와 다르면 즉시 실패(캐시 길이 고정 전제)
                if ans_len[0] is None:
                    ans_len[0] = len(enc["input_ids"])
                elif len(enc["input_ids"]) != ans_len[0]:
                    raise ValueError(f"{tkn}: 정답 토큰 수 {len(enc['input_ids'])} != {ans_len[0]} (고정 길이 위반)\n{answer}")
                tsp = [[min(k for k, (s0, e0) in enumerate(offs) if e0 > a0), max(k for k, (s0, e0) in enumerate(offs) if s0 < b0) + 1]
                       for a0, b0 in lm]
                rd = dict(
                    token=tkn, scene_token=info["scene_token"], timestamp=int(info["timestamp"]),
                    cams=cam_paths(info),
                    cams_prev=cam_paths(prev if prev is not None else info),   # 장면 첫 샘플: 현재 프레임 복제
                    prev_available=prev is not None,
                    prev_token=None if prev is None else prev["token"],
                    command=dict(id=cmd, text=C.CMD_TEXT[cmd], onehot=[int(v) for v in info["gt_ego_fut_cmd"]]),
                    ego_status=dict(src=ego_src, n_past=n_past, available=ego_vals is not None,
                                    speed=None if ego_vals is None else round(ego_vals[0], 4),
                                    accel_long=None if ego_vals is None else round(ego_vals[1], 4),
                                    yaw_rate=None if ego_vals is None else round(ego_vals[2], 5),
                                    raw_can=[float(v) for v in es],
                                    current=dict(src=cur_src, n_past=cur_np,
                                                 values=None if ego_cur is None else [round(v, 5) for v in ego_cur]),
                                    past_tokens=[p["token"] for p in past[tkn]]),
                    ego_drop_ok=ego_vals is not None,   # ego 줄이 있고 값이 있을 때만(SFT 무작위 블랭크 허용)     # SFT 가 자차 상태 줄을 무작위로 'unavailable' 로 바꿔도 되는가
                    command_drop_ok=True,
                    prompt=dict(system=C.SYSTEM_PROMPT, user=user),
                    prompt_parts=parts,
                    answer=answer,
                    answer_spans=spans,
                    loss_mask_spans=lm,                 # answer 문자 구간 [s,e) — SFT 에서 손실 제외
                    loss_mask_token_spans=tsp,          # answer 단독 토큰화(특수토큰 등록 Qwen2-VL) 기준 [s,e)
                    modes=dict(ego_in_prompt=a.ego_in_prompt, object_speed=a.object_speed,
                               answer_format=C.ANSWER_FORMAT),   # 수정(2026-10-15)
                    targets=dict(
                        objects=objs, n_objects_in_range=n_in, object_selection=sel_meta, map=mp,
                        future_complete=complete, fut_mask=[int(v) for v in mask], fut_traj=r4(fut),
                        label_v0=round(v0, 4), intent=intent, intent_weight=weight,
                        action_tokens=toks,
                        action_recon_traj=None if rec is None else r4(rec),
                        action_recon_ade=None if rec is None else round(float(e.mean()), 5)),
                    copy_baseline=cbase,
                )
                fo.write(json.dumps(rd, ensure_ascii=False) + "\n")
                nobj.append(len(objs))
                for r in objs:
                    S["obj_reason"][r["reason"]] += 1; S["obj_class"][r["name"]] += 1
                    # 수정(2026-10-15): v2-sensor 분포·가용률
                    S["obj_ttc"][str(r["ttc"])] += 1
                    if r["ttc_t"] is not None:
                        S["obj_ttc_first_t"][str(r["ttc_t"])] += 1
                    S["obj_motion"][r["motion"]] += 1
                    S["obj_heading_na"][str(r["heading"] is None)] += 1
                    for h, p in zip(("1s", "2s", "3s"), r["fut"]):
                        S["obj_fut_available"][f"{h}_{p is not None}"] += 1
                S["empty_slots"][str(C.MAX_OBJECTS - len(objs))] += 1
                if mp is None:
                    S["map_field"]["map_na"] += 1
                else:
                    S["map_field"]["n"] += 1
                    pcs = mp["ped_crossing"]
                    S["map_field"]["ped_" + ("none" if pcs is None else "on" if pcs["sector"] == "here" else "near")] += 1
                    for y, d in mp["lanes"].items():
                        for k, v in d.items():
                            S["map_field"][f"y{y}_{k}_present"] += int(v is not None)
                    S["map_free"]["capped" if mp["free_dist"] >= C.FREE_CAP else "blocked"] += 1
                S["loss_mask"]["na_spans"] += len(spans["na"])
                S["loss_mask"]["samples_with_na"] += int(bool(spans["na"]))
                S["obj_selection"]["selected_not_in_nearest8"] += sel_meta["n_selected_not_in_nearest8"]
                S["obj_selection"]["samples_truncated"] += int(n_in > C.MAX_OBJECTS)
                S["obj_selection"]["path_src_" + sel_meta["path_src"]] += 1
                S["obj_selection"]["path_objects_dropped"] += max(0, sel_meta["n_tier"]["path"] - C.MAX_OBJECTS)
                len_v2.append(len(enc["input_ids"]))
                S["loss_mask"]["intent_action_masked"] += int(not complete)
                S["loss_mask"]["speed_masked_samples"] += int(prev is None and bool(spans["object_speed"]))
                S["prev_frame"][str(prev is not None)] += 1
                if tkn in v1:
                    len_v1.append(len(tok_v1(v1[tkn][2])["input_ids"]))
                len_prompt.append(len(tok(C.SYSTEM_PROMPT + "\n" + user)["input_ids"]))
        n = len(infos)
        stats["splits"][split] = dict(
            samples=n, scenes=len({i["scene_token"] for i in infos}),
            future_complete=S["uncertain"]["True"] + S["uncertain"]["False"],
            **{k: dict(v) for k, v in S.items()},
            objects_per_sample=pct(nobj),
            pose_past_vs_can_error=({k: dict(n=len(v), mean=round(float(np.mean(v)), 4),
                                             abs_p50=round(float(np.percentile(np.abs(v), 50)), 4),
                                             abs_p95=round(float(np.percentile(np.abs(v), 95)), 4)) for k, v in pp_err.items()}),
            action=dict(recon_ade_m=round(float(np.mean(ade)), 5)),
            copy_baseline=dict(ctrv_ade_m=round(float(np.mean(ade_copy)), 4),
                               ctrv_tokenized_ade_m=round(float(np.mean(ade_copy_tok)), 4)),
            qwen2vl_tokens=dict(answer_v2_special_tokens_added=pct(len_v2),
                                answer_v1_action_tokens_added=pct(len_v1) if len_v1 else None,
                                prompt_v2_text_only=pct(len_prompt)),
        )
        print(f"[{split}] {n} -> {path}", json.dumps(stats["splits"][split]["lateral"], ensure_ascii=False))
    with open(f"{OUT_DIR}/stats.json", "w") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    print(json.dumps(stats, indent=1, ensure_ascii=False)[:8000])


if __name__ == "__main__":
    main()
