# 수정(2026-11-09): 새 파일 — B2D 폐쇄루프 온라인 교사 빌더(team_code/laq_teacher_builder_b2d.py) 시험.
"""온라인 교사 빌더가 교사 캐시 생성(extract_teacher_cache_b2d.py)과 같은 프롬프트·전처리·layer 토큰 추출을 하는지 검사한다.

CPU(기본, GPU·CARLA 불필요):
  source /home1/irteam/data-vol1/hipad_env.sh
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tools/verify_b2d/test_teacher_builder_b2d.py [--smoke_dir D] [--train_infos]

  (A) 프롬프트: 교사 정답 jsonl 전 표본(train 47,346 + val 2,583)의 기록 필드(명령 raw·목표점·자차 원값)로 빌더 user_text 를
      만들면 record["prompt"]["user"](= 캐시 프롬프트)와 문자 단위로 같다. 단 jsonl 의 목표점은 소수 3 자리로 반올림 저장돼 있어
      x.x5 경계(±0.0005) 표본은 1 자리 반올림이 달라질 수 있다 → 그 경우만 '반올림 경계'로 따로 세고, 그 밖의 불일치는 0 이어야 한다.
  (B) 시뮬레이터 입력 → 프롬프트: b2d_infos_val(.pkl, --train_infos 면 train 도)의 원값을 에이전트 tick 형식으로 바꿔
      (pos = (x, y), target_xy = 명령 far 점, compass = θ — converter 역변환) target_point_from_tick 이 데이터셋 target_point_far 식과
      비트 단위로 같고, 키프레임 전부에서 빌더 프롬프트 = 캐시 프롬프트(문자 단위, 반올림 경계 없음)인지.
  (C) 소형 무작위 교사(tools/vla_teacher/smoke_b2d_teacher_chain_cpu.py 가 만든 SFT→병합→캐시 2 표본, CPU fp32, layer 2)로
      빌더 __call__ 출력 = 학생 로더(LoadTeacherCacheB2D) 가 같은 키프레임에서 읽은 텐서(qwen_hidden/qwen_mask/qwen_type).
      허용 오차: CPU fp32·배치 1 대 배치 1 → **완전 일치(최대 절대차 0, 생성 토큰 547/547 동일)**.
  (D) 계약: ego_line 반환·불일치 예외·모르는 키워드 거부·캐시 _meta 대조(층·fold·체크포인트 불일치 거부)·build() 재사용.
  (E) 복제 배치(gen 2 / fwd 2): 복제 행 생성 토큰이 서로 같은지(CPU 정보용, 캐시 대비 차이는 기록만).

GPU(--gpu, 사용자 승인 뒤 GPU2 에서 수 분): 실제 병합 교사·실제 캐시로 N 표본을 비교한다.
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python tools/verify_b2d/test_teacher_builder_b2d.py --gpu --model <병합 교사> --fold A [--n 8]
  허용 오차(bf16, 캐시 = 배치 16 생성 + 4 묶음 순전파): context 폴더 = 영상·텍스트 입력만 의존 → 토큰별 cos 최소 ≥ 0.9999
  (NAVSIM 실측은 비트 동일). 답 폴더 6 개 = 생성 토큰이 547/547 같을 때 cos 최소 ≥ 0.999. 토큰이 갈리면(배치 크기에 따른 bf16 동률
  뒤집힘, NAVSIM 배치 1 에서 0/30 일치) 표본별 첫 불일치 위치·폴더 cos 를 보고하고 FAIL 로 표시한다 — 복제 배치(LAQ_TEACHER_GEN_BATCH
  = 캐시 batch)로 일치가 회복되는지가 이 모드의 주 질문이다(--gen_batch 1 과 기본값을 각각 돌려 비교).
"""
import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEAM = os.path.join(REPO, "bench2drive", "leaderboard", "team_code")
VLA = os.path.join(REPO, "tools", "vla_teacher")
sys.path.insert(0, TEAM)
sys.path.insert(0, REPO)
import laq_teacher_builder_b2d as TB  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not cond:
        FAILS.append(name)
    return cond


def raises(fn, exc=Exception):
    try:
        fn()
    except exc as e:
        return f"{type(e).__name__}: {e}"
    return None


def iter_gt(path, keys=("token", "command", "target_point", "ego_status", "prompt", "cams", "cams_prev", "prev_available")):
    with open(path) as fh:
        for ln in fh:
            r = json.loads(ln)
            yield {k: r.get(k) for k in keys}


def ego_from_raw(es):
    """gen_teacher_gt_b2d.ego_inputs_b2d 와 같은 식(sgn = +1, b2d_conventions_train.json 기울기 > 0)."""
    raw = es["raw"]
    return (max(float(raw["ego_vel"][0]), 0.0), float(raw["ego_accel"][0]), 1.0 * float(raw["ego_rot"][2]))


def near_r1_boundary(v3):
    a = abs(float(v3)) * 10.0
    return abs((a - math.floor(a)) - 0.5) <= 0.0051


# ---------------------------------------------------------------- (A)
def test_prompt_records():
    print("(A) 교사 정답 jsonl 전 표본: 기록 필드 → 빌더 user_text == 캐시 프롬프트", flush=True)
    E = TB._mods()
    tot = same = boundary = 0
    other = []
    for split in ("train", "val"):
        p = E.BS.B2D_GT_DIR / E.BS.B2D_SPLIT_FILES[split]
        for r in iter_gt(p):
            tot += 1
            tp = np.asarray(r["target_point"], np.float32)
            try:
                text, line = TB.user_text(r["command"]["raw"], tp, ego_from_raw(r["ego_status"]))
            except Exception as e:  # noqa: BLE001
                other.append((r["token"], f"{type(e).__name__}: {e}"))
                continue
            if text == r["prompt"]["user"]:
                same += 1
                continue
            a, b = text.split("\n"), r["prompt"]["user"].split("\n")
            diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            if (len(a) == len(b) and len(diff) == 1 and a[diff[0]].startswith("Target point:")
                    and b[diff[0]].startswith("Target point:") and any(near_r1_boundary(v) for v in tp)):
                boundary += 1
            else:
                other.append((r["token"], diff))
    check(f"user 텍스트 일치 {same}/{tot}, 반올림 경계 {boundary}, 그 밖 불일치 {len(other)}",
          tot == 49929 and len(other) == 0 and same + boundary == tot, str(other[:3]))


# ---------------------------------------------------------------- (B)
def dataset_target_far(info):
    """models/loader/bench2drive/dataset.py get_data_info(948–960) 의 target_point_far 식."""
    theta_to_lidar = -(info['ego_yaw'] - np.pi / 2)
    rotation_matrix = np.array([[np.cos(theta_to_lidar), -np.sin(theta_to_lidar)],
                                [np.sin(theta_to_lidar), np.cos(theta_to_lidar)]])
    command_far_xy = info['command_far_xy'] - info['ego_translation'][:2]
    return np.array(rotation_matrix @ command_far_xy, dtype=np.float32)


def tick_from_info(info):
    """converter 역변환: ego_translation = (x, −y), command_far_xy = (x_far, −y_far), ego_yaw = −θ + π/2 (θ = 나침반)
    → 에이전트 tick: pos = gps_to_location ≈ (x, y), target_xy = 경로 점 (x, y), compass = θ."""
    et, cf = np.asarray(info["ego_translation"], np.float64), np.asarray(info["command_far_xy"], np.float64)
    return dict(pos=np.array([et[0], -et[1]]), target_xy=np.array([cf[0], -cf[1]]), compass=np.pi / 2 - float(info["ego_yaw"]),
                command=int(info["command_near"]))


# infos pkl 은 저장소 자체 산출물(data_converter) — 학습 파이프라인과 같은 파일을 읽는다(외부 입력 아님).
def test_tick_infos(splits):
    print(f"(B) 시뮬레이터 입력(infos {splits}) → target_point·프롬프트", flush=True)
    E = TB._mods()
    for split in splits:
        infos = pickle.load(open(E.BS.B2D_INFOS[split], "rb"))
        infos = infos["infos"] if isinstance(infos, dict) else infos
        by_key = {}
        n_bit = n_nan = 0
        bad = []
        for info in infos:
            if not np.isfinite(info["ego_yaw"]):
                n_nan += 1
            a = TB.target_point_from_tick(tick_from_info(info))
            b = dataset_target_far(info)
            if a.tobytes() == b.tobytes():
                n_bit += 1
            elif len(bad) < 3:
                bad.append((info["folder"], info["frame_idx"], a.tolist(), b.tolist()))
            by_key[(info["folder"].split("/")[-1], int(info["frame_idx"]))] = info
        check(f"{split}: target_point_from_tick == 데이터셋 target_point_far 비트 일치 {n_bit}/{len(infos)} (yaw 비유한 {n_nan})",
              n_bit == len(infos), str(bad))
        tot = same = 0
        miss = []
        for r in iter_gt(E.BS.B2D_GT_DIR / E.BS.B2D_SPLIT_FILES[split]):
            clip, _, kf = r["token"].rpartition("/")
            info = by_key.get((clip, int(kf)))
            if info is None:
                miss.append(r["token"]); continue
            tick = tick_from_info(info)
            ego = (max(float(info["ego_vel"][0]), 0.0), float(info["ego_accel"][0]), float(info["ego_rotation_rate"][2]))
            text, _ = TB.user_text(TB.command_value(tick["command"]), TB.target_point_from_tick(tick), ego)
            tot += 1
            same += int(text == r["prompt"]["user"])
        check(f"{split}: infos → tick → 빌더 프롬프트 == 캐시 프롬프트 {same}/{tot} (infos 에 없는 키프레임 {len(miss)})",
              tot > 0 and same == tot and not miss, str(miss[:3]))
        del infos, by_key


# ---------------------------------------------------------------- (C)(D)(E)
def ensure_smoke(smoke_dir):
    m = os.path.join(smoke_dir, "b2d_merged_foldA_qa")
    c = os.path.join(smoke_dir, "bench2drive", "laq_cache")
    if not (os.path.exists(os.path.join(m, "teacher_meta.json")) and os.path.isdir(os.path.join(c, "_index"))):
        print(f"  소형 교사·캐시 생성: smoke_b2d_teacher_chain_cpu.py → {smoke_dir}", flush=True)
        env = dict(os.environ, SMOKE_DIR=smoke_dir, CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO)
        p = subprocess.run([sys.executable, os.path.join(VLA, "smoke_b2d_teacher_chain_cpu.py")], cwd=REPO, env=env,
                           capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout[-3000:] + p.stderr[-3000:])
            raise SystemExit("소형 교사 체인 스모크 실패")
    return m, c


def student_loader(cache_root, hidden_dim):
    E = TB._mods()
    _ld, clip_of, _kf, tl = E.student_loader(cache_root)
    from projects.mmdet3d_plugin.models.encoder.vl_tokens_cache.loading_b2d import LoadTeacherCacheB2D
    return LoadTeacherCacheB2D(**dict(tl, hidden_dim=hidden_dim))


def cached_records(cache_root, teacher="A"):
    E = TB._mods()
    idx = E.read_all_index(__import__("pathlib").Path(cache_root))
    items, _, _ = E.assigned(teacher, str(E.BS.B2D_FOLD_FILE))
    recs = {r["token"]: r for _, r in items if r["token"] in idx}
    return [(t, recs[t], idx[t]) for t in sorted(recs)]


def tick_for_record(r):
    """compass 0, pos (0,0) 이면 target_point = (tx, −ty·(−1)) — 기록 목표점이 그대로 학생/교사 목표점이 되도록 만든 tick."""
    tp = np.asarray(r["target_point"], np.float64)
    return dict(pos=np.zeros(2), target_xy=np.array([tp[0], -tp[1]]), compass=0.0, command=r["command"]["raw"])


def split_imgs(r):
    S = sys.modules["sft_common"]
    imgs = S.load_images(r, True, "prev_first", 6)
    return imgs[6:], imgs[:6]


def test_cache_parity(smoke_dir):
    import torch
    print("(C) 소형 교사(CPU fp32, layer 2): 빌더 출력 == 캐시 → 학생 로더 텐서", flush=True)
    model, cache = ensure_smoke(smoke_dir)
    os.environ.pop("LAQ_TEACHER_GEN_BATCH", None)
    T = TB.OnlineTeacherB2D(model, "A", device="cpu", layer=2, dtype="fp32", attn="sdpa", cache_root=cache)
    check("캐시 _meta 대조 통과·복제 배치 = 캐시 batch(1)", T.cache_check is not None and (T.gen_batch, T.fwd_batch) == (1, 1),
          f"gen {T.gen_batch} fwd {T.fwd_batch}")
    ld = student_loader(cache, T.D)
    recs = cached_records(cache)
    check("캐시 표본 2 개(샤드 0/2·1/2)", len(recs) == 2, str([t for t, _, _ in recs]))
    for tok, r, e in recs:
        cams, prev = split_imgs(r)
        tick = tick_for_record(r)
        ego = ego_from_raw(r["ego_status"])
        line = TB._mods().B.ego_line(*ego)
        text, _ = TB.user_text(r["command"]["raw"], TB.target_point_from_tick(tick), ego)
        check(f"{tok}: tick 경로 프롬프트 == 캐시 프롬프트", text == r["prompt"]["user"])
        out = T(0.0, cams=cams, cams_prev=prev, prev_available=r["prev_available"], tick_data=tick, ego=ego, ego_line=line)
        clip, kf = tok.rsplit("/", 1)
        ref = ld(dict(folder=f"v1/{clip}", frame_idx=int(kf) + 3, token="orig"))
        h, m, ty = (ref[k].data for k in ("qwen_hidden", "qwen_mask", "qwen_type"))
        d = float((out["qwen_hidden"].float() - h.float()).abs().max())
        check(f"{tok}: 생성 토큰 == 캐시 색인 gen_ids (547)", out["teacher_info"]["gen_ids"] == e["gen_ids"])
        check(f"{tok}: qwen_hidden {tuple(out['qwen_hidden'].shape)} {out['qwen_hidden'].dtype} == 로더 {tuple(h.shape)} (최대 절대차 {d})",
              out["qwen_hidden"].dtype == torch.bfloat16 and torch.equal(out["qwen_hidden"], h))
        check(f"{tok}: qwen_mask·qwen_type 동일(uint8·int64, 폴더 순서 0..6)",
              out["qwen_mask"].dtype == torch.uint8 and out["qwen_type"].dtype == torch.long
              and torch.equal(out["qwen_mask"], m) and torch.equal(out["qwen_type"], ty))
        check(f"{tok}: ego_line 반환 = 넘긴 줄", out["ego_line"] == line)
    return T, recs, model, cache


def test_contract(T, recs, model, cache):
    print("(D) 계약·설정 대조", flush=True)
    tok, r, _ = recs[0]
    cams, prev = split_imgs(r)
    tick = tick_for_record(r)
    ego = ego_from_raw(r["ego_status"])
    B = TB._mods().B
    wrong = B.ego_line(ego[0] + 1.0, ego[1], ego[2])
    msg = raises(lambda: T(0.0, cams=cams, cams_prev=prev, prev_available=True, tick_data=tick, ego=ego, ego_line=wrong))
    check("넘긴 ego_line 이 ego 와 다르면 예외", msg is not None and "자차 줄 불일치" in msg, str(msg)[:120])
    msg = raises(lambda: T(0.0, cams=cams, cams_prev=prev, prev_available=True, tick_data=tick, ego=ego, ego_line=None, command=4))
    check("모르는 키워드(command) 거부", msg is not None and "TypeError" in msg, str(msg)[:120])
    msg = raises(lambda: T(0.0, cams=cams[:5], cams_prev=prev, prev_available=True, tick_data=tick, ego=ego, ego_line=None))
    check("영상 5 장 거부", msg is not None and "영상 수" in msg)
    msg = raises(lambda: TB.OnlineTeacherB2D(model, "A", device="cpu", layer=1, dtype="fp32", cache_root=cache))
    check("캐시 layer(2) 와 다른 층(1) 거부", msg is not None and "layer" in msg, str(msg)[:160])
    msg = raises(lambda: TB.OnlineTeacherB2D(model, "B", device="cpu", layer=2, dtype="fp32", cache_root=cache))
    check("meta fold(A) 와 다른 LAQ_TEACHER_FOLD(B) 거부", msg is not None and "fold" in msg, str(msg)[:160])
    msg = raises(lambda: TB.OnlineTeacherB2D(model, "A", device="cpu", layer=2, dtype="bf16", cache_root=cache))
    check("캐시 dtype(fp32) 와 다른 bf16 거부", msg is not None and "dtype" in msg, str(msg)[:160])
    # 체크포인트 불일치: 캐시 _meta 사본의 checkpoint_sha 를 바꿔 거부되는지
    tmp = tempfile.mkdtemp(prefix="tb_meta_")
    for k in TB.TEACHER_CACHE_FOLDERS:
        os.makedirs(os.path.join(tmp, k))
        m = json.load(open(os.path.join(cache, k, "_meta.json")))
        m["teachers"]["A"]["checkpoint_sha"] = "0" * 64
        json.dump(m, open(os.path.join(tmp, k, "_meta.json"), "w"))
    msg = raises(lambda: TB.OnlineTeacherB2D(model, "A", device="cpu", layer=2, dtype="fp32", cache_root=tmp))
    check("캐시를 만든 교사와 다른 체크포인트 거부", msg is not None and "checkpoint_sha" in msg, str(msg)[:160])
    # build(): 환경변수 → 같은 설정이면 같은 객체
    os.environ.update(LAQ_TEACHER_MODEL=model, LAQ_TEACHER_FOLD="A", LAQ_TEACHER_LAYER="2", LAQ_TEACHER_DTYPE="fp32",
                      LAQ_TEACHER_CACHE_ROOT=cache)
    a = TB.build(device="cpu")
    b = TB.build(device="cpu")
    check("build() 재사용(경로마다 setup 재호출 시 재적재 없음)", a is b)
    os.environ["LAQ_TEACHER_LAYER"] = "1"
    msg = raises(lambda: TB.build(device="cpu"))
    check("같은 프로세스에서 다른 설정 build 거부", msg is not None and "하나만" in msg)
    for k in ("LAQ_TEACHER_MODEL", "LAQ_TEACHER_FOLD", "LAQ_TEACHER_LAYER", "LAQ_TEACHER_DTYPE", "LAQ_TEACHER_CACHE_ROOT"):
        os.environ.pop(k)
    # 에이전트 스케줄러 호출 형태(teacher_fn(t, **inputs)) 그대로
    from teacher_refresh import TeacherRefreshScheduler
    sch = TeacherRefreshScheduler(T, mode="fresh")
    line = B.ego_line(*ego)
    o1, i1 = sch.step(0.0, scene_id="r", cams=cams, cams_prev=prev, prev_available=True, tick_data=tick, ego=ego, ego_line=line)
    o2, i2 = sch.step(0.05, scene_id="r", cams=cams, cams_prev=prev, prev_available=True, tick_data=tick, ego=ego, ego_line=None)
    check("스케줄러 경유 호출: 첫 틱 갱신·다음 틱 재사용", i1["refreshed"] and not i2["refreshed"] and o1 is o2 and o1["ego_line"] == line)


def test_replica(model, cache, recs):
    print("(E) 복제 배치 gen 2 / fwd 2 (CPU, 정보용)", flush=True)
    import torch
    logp = os.path.join(tempfile.mkdtemp(), "teacher.jsonl")
    T2 = TB.OnlineTeacherB2D(model, "A", device="cpu", layer=2, dtype="fp32", cache_root=cache, gen_batch=2, fwd_batch=2,
                             log_path=logp)
    tok, r, e = recs[0]
    cams, prev = split_imgs(r)
    ego = ego_from_raw(r["ego_status"])
    out = T2(0.0, cams=cams, cams_prev=prev, prev_available=True, tick_data=tick_for_record(r), ego=ego, ego_line=None)
    check("복제 행 생성 토큰 서로 같음", out["teacher_info"]["replica_gen_diff"] == 0)
    lrec = [json.loads(x) for x in open(logp)]
    check("교사 호출 jsonl 에 replica_gen_diff 기록", len(lrec) == 1 and lrec[0].get("replica_gen_diff") == 0, str(lrec)[:160])
    ld = student_loader(cache, T2.D)
    clip, kf = tok.rsplit("/", 1)
    h = ld(dict(folder=f"v1/{clip}", frame_idx=int(kf), token="o"))["qwen_hidden"].data
    print(f"    캐시 대비: 토큰 {'같음' if out['teacher_info']['gen_ids'] == e['gen_ids'] else '다름'}, "
          f"최대 절대차 {float((out['qwen_hidden'].float() - h.float()).abs().max()):.3g} (CPU 커널 기준, 판정 아님)", flush=True)


# ---------------------------------------------------------------- GPU 모드(사용자 승인 뒤)
def run_gpu(a):
    import torch
    E = TB._mods()
    cache = a.cache_root
    T = TB.OnlineTeacherB2D(a.model, a.fold, device="cuda", layer=a.layer, dtype="bf16", cache_root=cache,
                            gen_batch=a.gen_batch or None)
    ld = student_loader(cache, T.D)
    idx = E.read_all_index(__import__("pathlib").Path(cache))
    items, _, _ = E.assigned(a.fold, str(E.BS.B2D_FOLD_FILE))
    sel = [r for _, r in items if r["token"] in idx and idx[r["token"]]["teacher"] == a.fold][:a.n]
    rows = []
    for r in sel:
        cams, prev = split_imgs(r)
        text, line = TB.user_text(r["command"]["raw"], np.asarray(r["target_point"], np.float32), ego_from_raw(r["ego_status"]))
        if text != r["prompt"]["user"]:
            text = r["prompt"]["user"]   # 반올림 경계 표본 — 캐시 프롬프트 그대로(이 모드는 수치 비교가 목적)
        out = T.infer(prev + cams, text, return_folders=True)
        clip, kf = r["token"].rsplit("/", 1)
        ref = ld(dict(folder=f"v1/{clip}", frame_idx=int(kf), token="o"))["qwen_hidden"].data[0].float()
        got = out["qwen_hidden"][0].float().cpu()
        g, c = out["teacher_info"]["gen_ids"], idx[r["token"]]["gen_ids"]
        first = next((i for i, (x, y) in enumerate(zip(g, c)) if x != y), None)
        row = dict(token=r["token"], same_tokens=sum(int(x == y) for x, y in zip(g, c)), first_diff=first, folders={})
        o = 0
        for k in TB.TEACHER_CACHE_FOLDERS:
            L = T.Ls[k]
            cs = torch.nn.functional.cosine_similarity(got[o:o + L], ref[o:o + L], dim=-1)
            row["folders"][k] = dict(cos_min=round(float(cs.min()), 5), maxabs=round(float((got[o:o + L] - ref[o:o + L]).abs().max()), 4))
            o += L
        ok = (row["folders"]["context"]["cos_min"] >= 0.9999 and row["same_tokens"] == 547
              and all(row["folders"][k]["cos_min"] >= 0.999 for k in TB.TEACHER_CACHE_FOLDERS[1:]))
        row["pass"] = ok
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    check(f"GPU: {sum(r['pass'] for r in rows)}/{len(rows)} 표본 허용 오차 안(context cos≥0.9999, 토큰 547/547, 답 cos≥0.999), "
          f"gen_batch {T.gen_batch} fwd {T.fwd_batch}", all(r["pass"] for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke_dir", default=None)
    ap.add_argument("--train_infos", action="store_true", help="(B) 에 train infos(2.4 GB) 도 포함")
    ap.add_argument("--skip_records", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--fold", choices=["A", "B"])
    ap.add_argument("--cache_root", default=os.path.join(REPO, "data", "bench2drive", "laq_cache"))
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--gen_batch", type=int, default=0)
    a = ap.parse_args()
    if a.gpu:
        if not (a.model and a.fold):
            raise SystemExit("--gpu 는 --model --fold 필요")
        run_gpu(a)
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
            raise SystemExit("CPU 모드는 CUDA_VISIBLE_DEVICES='' 로 실행")
        smoke = a.smoke_dir or tempfile.mkdtemp(prefix="tb_smoke_")
        if not a.skip_records:
            test_prompt_records()
            test_tick_infos(["val"] + (["train"] if a.train_infos else []))
        T, recs, model, cache = test_cache_parity(smoke)
        test_contract(T, recs, model, cache)
        test_replica(model, cache, recs)
    print("ALL PASS" if not FAILS else f"FAIL {FAILS}", flush=True)
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
