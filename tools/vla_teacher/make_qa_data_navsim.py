# 수정(2026-10-16): 교사 QA — NAVSIM 트랙. 공개 NAVSIM/nuPlan QA → fold 별 학습/hold-out 표본(jsonl). make_qa_data.py 와 같은 표본 형식.
"""NuPlanQA(train) + ReCogDrive NAVSIM QA → data/vla_teacher/navsim/fold{A,B}_{train,holdout}.jsonl (+ qa_manifest.json)
NuPlanQA-Eval(test 1,801 문항 전체) → data/vla_teacher/navsim/nuplanqa_eval_navsim.jsonl (평가 전용, 학습에 쓰지 않음)

실행(내려받은 JSON/pkl 은 신뢰하지 않는 데이터 — 반드시 python -I):
  CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX python -I tools/vla_teacher/make_qa_data_navsim.py --index_cache <pkl>

표본 형식(make_qa_data.py 와 같음 — train_qa_full.py / qa_common.QAFrameDataset 가 그대로 읽는다):
  id="<dataset>:<NAVSIM 프레임 token>:<chunk>", dataset, token(NAVSIM 프레임 token), scene_token(= NAVSIM 로그 이름 — fold 단위),
  fold, part, chunk, n_chunks, cams(전방 3캠 CAM_L0·CAM_F0·CAM_R0 의 NAVSIM sensor_blobs 절대경로), turns[{q,a,type,src}],
  n_tokens(영상 3×캠당 토큰 + 텍스트), n_answer_tokens.
  카메라·해상도는 LaQ NAVSIM Qwen 캐시(tools/qwen_cache_navsim.py)와 같다: 전방 3캠, min=max_pixels=128*28*28.
  → train_qa_full.py 에 --num_cams 3 --pixels 100352 를 준다(manifest 의 n_cams/pixels 와 대조).

프레임 대응(NuPlanQA): nuplan_metadata.pkl 의 sample_token → 카메라 8대 영상 파일 이름(stem = nuPlan 영상 token) →
  NAVSIM 로그 pkl 의 같은 stem 을 가진 프레임. 8대 중 맞는 stem 은 모두 같은 프레임·같은 카메라 이름이어야 한다(아니면 실패).
  ReCogDrive: image 경로의 <log>/CAM_F0/<stem>.jpg 를 같은 방식으로 대응하고 로그 이름이 같은지 확인.
포함 조건(학습 표본): navtrain 로그(navtrain.yaml log_names) 의 프레임 & 전방 3캠 영상이 NAVSIM sensor_blobs 에 실재 &
  teacher_exclude_logs(NuPlanQA-Eval 프레임이 있는 6 로그) 밖. 나머지는 이유별로 세어 manifest.excluded 에 기록(조용한 제외 없음).
  navtest(test 로그) 프레임이 나오면 즉시 실패(유출).
"""
import argparse
import collections
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import navsim_qa_common as N  # noqa: E402
import qa_common as Q  # noqa: E402
import sft_common as S  # noqa: E402

QA_ROOT = "/home1/irteam/data-vol1/src/HiP-AD/data/nuscenes/qa"
SOURCES = {
    "nuplanqa": dict(
        files=["nuplanqa/nuplanqa_train.json", "nuplanqa/nuplanqa_eval_test.json", "nuplanqa/nuplan_metadata.pkl"],
        paper="S.-Y. Park, C. Cui, Y. Ma, A. Moradipari, R. Gupta, K. Han, Z. Wang, 'NuPlanQA: A Large-Scale Dataset and "
              "Benchmark for Multi-View Driving Scene Understanding in Multi-Modal Large Language Models', arXiv:2503.12772 (2025)",
        note="train json = 학습 후보, eval_test json = 평가 전용(학습 금지)"),
    "recogdrive_navsim": dict(
        files=["recogdrive_navsim_b2d/Navsim_ReCogDrive/dataset_navsim_recogdrive.jsonl"],
        paper="Y. Li et al., 'ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving', arXiv:2506.08052 (2025)",
        note="navtrain 프레임 QA(전방 CAM_F0 1장 기준 질문). Navsim_Traj(궤적 예측 대화)는 QA 가 아니라 쓰지 않음"),
}
DATASETS_NAVSIM = ("nuplanqa", "recogdrive_navsim")
# 원본 답이 생성 오류로 깨진 QA(ReCogDrive): 문장 중간에 끊기고 "📐"/빈 줄이 수백~수천 줄 반복(예: id3126 t0 4,463자 중 1,707줄 반복).
#   정답으로 쓸 수 없으므로 그 턴만 학습에서 뺀다(같은 프레임의 나머지 턴은 유지). 규칙이 결정적이고 뺀 목록은 manifest 에 전부 기록.
#   잘라 쓰지 않는다(잘린 문장을 정답으로 가르치게 됨).
CORRUPT_GLYPH = "\U0001F4D0"   # 📐


def corrupt_reason(ans):
    lines = [l.strip() for l in ans.split("\n")]
    run = best = 1
    for i in range(1, len(lines)):
        run = run + 1 if lines[i] and lines[i] == lines[i - 1] else 1
        best = max(best, run)
    if CORRUPT_GLYPH in ans:
        return f"생성 오류 글리프 U+1F4D0 {ans.count(CORRUPT_GLYPH)}회, 같은 줄 최대 연속 {best}"
    if best >= 3:
        return f"같은 줄 {best}회 연속 반복"
    return None


# ---------------------------------------------------------------- NuPlanQA 메타(nuplan 패키지 없음 → 허용 목록 밖 클래스는 빈 껍데기)
class _Stub(list):
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, st):
        self.__dict__.update(st if isinstance(st, dict) else {})


class _MetaUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("builtins", "collections", "numpy", "datetime", "copyreg", "_codecs"):
            return super().find_class(module, name)
        return type(name, (_Stub,), {})        # nuplan 클래스(Translation/Rotation 등): 쓰지 않는 필드라 내용 무시


def load_nuplanqa_meta(path):
    m = _MetaUnpickler(open(path, "rb")).load()
    out = {}
    for k, v in m.items():
        cams = {}
        for c, d in v["cams"].items():
            cams[c] = os.path.splitext(os.path.basename(d["data_path"]))[0]
        out[k] = dict(cams=cams, scene_token=v.get("scene_token"), timestamp=v.get("timestamp"))
    return out


def map_by_stems(cam_stems, stem_map):
    """{cam: stem} → (split, log, frame token) 또는 None. 맞는 stem 이 서로 다른 프레임/카메라를 가리키면 예외."""
    hits = {}
    for c, s in cam_stems.items():
        h = stem_map.get(s)
        if h is None:
            continue
        if h[3] != c:
            raise RuntimeError(f"카메라 이름 불일치: stem {s} 은 NuPlanQA {c}, NAVSIM {h[3]}")
        hits[c] = h[:3]
    if not hits:
        return None, 0
    vals = set(hits.values())
    if len(vals) != 1:
        raise RuntimeError(f"stem 들이 서로 다른 NAVSIM 프레임을 가리킨다: {hits}")
    return vals.pop(), len(hits)


# ---------------------------------------------------------------- 길이·chunk(make_qa_data.py 와 같은 규칙, 영상 수·캠당 토큰만 인자로)
def tokens_per_cam(pixels, hw=N.NAVSIM_IMAGE_HW):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    h, w = smart_resize(hw[0], hw[1], factor=28, min_pixels=pixels, max_pixels=pixels)
    return (h // 28) * (w // 28), (h, w)


def text_len(tok, turns, n_img, tpc):
    ids, lab = Q.tokenize_pieces(tok, Q.conversation_pieces(dict(turns=turns), n_images=n_img), image_tokens_per_cam=tpc)
    return len(ids), sum(1 for x in lab if x != -100)


def chunk_turns(tok, turns, max_len, n_img, tpc):
    head = len(Q.tokenize_pieces(tok, Q.conversation_pieces(dict(turns=[]), n_images=n_img), image_tokens_per_cam=tpc)[0]) \
        + n_img * (tpc + 2)
    out, cur, cur_len = [], [], head
    for t in turns:
        tl = len(tok("<|im_start|>user\n" + t["q"] + "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]) \
            + len(tok(t["a"] + "<|im_end|>", add_special_tokens=False)["input_ids"]) + 1
        if head + tl > max_len:
            raise RuntimeError(f"QA 하나가 max_seq_tokens {max_len} 초과: {t['src']} ({head + tl})")
        if cur and cur_len + tl > max_len:
            out.append(cur); cur, cur_len = [], head
        cur.append(t); cur_len += tl
    if cur:
        out.append(cur)
    return out


def listdir_set(p):
    try:
        return set(os.listdir(p))
    except FileNotFoundError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_root", default=QA_ROOT)
    ap.add_argument("--out", default=N.QA_DIR_NAVSIM)
    ap.add_argument("--fold_file", default=N.FOLD_FILE_NAVSIM)
    ap.add_argument("--index_cache", required=True, help="NAVSIM 프레임 색인 캐시(make_fold_split_navsim.py 와 같은 파일)")
    ap.add_argument("--max_seq_tokens", type=int, default=6144,
                    help="한 대화 최대 길이. nuScenes 4096(영상 1,080) 과 달리 NAVSIM 은 영상 3캠이라 짧지만 ReCogDrive 3D 검출 답 하나가 최대 ~4,800 토큰 → 잘라내지 않으려면 6144 필요")
    ap.add_argument("--pixels", type=int, default=N.NAVSIM_PIXELS)
    ap.add_argument("--nuplan_raw_scan", type=int, default=1,
                    help="1 = NuPlanQA-Eval 중 NAVSIM 에 없는 프레임의 영상을 nuPlan 원본 sensor_blobs(로그별 CAM_F0 listdir)에서 찾는다")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out) if os.path.isabs(a.out) else S.REPO / a.out
    if out.exists() and any(out.glob("fold*_*.jsonl")) and not a.force:
        raise SystemExit(f"[거부] {out} 에 이미 산출물 — 덮어쓰려면 --force")
    raw = Path(a.qa_root)

    # ---- 분할·필터·색인
    fd, fpath = S.load_fold_split(a.fold_file)
    if fd.get("benchmark") != "navsim":
        raise SystemExit(f"{fpath}: NAVSIM 분할 파일이 아니다")
    part_of = {}
    for f in S.FOLDS:
        ho = set(fd["holdout"][f]["scenes"])
        for lg in fd["train"][f]["scenes"]:
            part_of[lg] = (f, "holdout" if lg in ho else "train")
    val_logs = set(fd["val"]["A"]["scenes"]) | set(fd["val"]["B"]["scenes"])
    excl_logs = set(fd["teacher_exclude_logs"]["logs"])
    ftr, fte = N.load_filter(N.NAVTRAIN_YAML), N.load_filter(N.NAVTEST_YAML)
    navtrain_logs, navtest_logs = set(ftr["log_names"]), set(fte["log_names"])
    navtest_tokens = set(fte["tokens"])
    idx = N.build_frame_index(a.index_cache, a.workers)
    tok_map, stem_map = N.token_maps(idx)
    test_logs_all = {lg for (sp, lg) in idx["logs"] if sp == "test"}
    print(f"[index] 프레임 {len(tok_map)}, 영상 stem {len(stem_map)} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 영상 실재(로그별 listdir, 전방 3캠)
    img_sets = {}

    def has_images(log, frame_token):
        if log not in img_sets:
            img_sets[log] = {c: listdir_set(os.path.join(N.TRAIN_BLOBS, log, c)) for c in N.QA_CAMS}
        cams = tok_map[frame_token][3]
        return all(img_sets[log][c] is not None and cams[c] + ".jpg" in img_sets[log][c] for c in N.QA_CAMS)

    def cam_paths(log, frame_token):
        cams = tok_map[frame_token][3]
        return [os.path.join(N.TRAIN_BLOBS, log, c, cams[c] + ".jpg") for c in N.QA_CAMS]

    counts = collections.Counter()
    groups = collections.OrderedDict()
    excluded_examples = collections.defaultdict(list)

    def route(ds, n_qa, hit, src):
        """반환 (log, frame token) 또는 None(이유를 센다)."""
        if hit is None:
            counts[(ds, "excluded:not_in_navsim_logs")] += n_qa; excluded_examples[(ds, "not_in_navsim_logs")].append(src); return None
        split, log, ftok = hit
        if split == "test" or log in navtest_logs or log in val_logs or ftok in navtest_tokens:
            raise RuntimeError(f"{ds} {src}: navtest/test 로그 프레임({log}) — 유출, 원본 확인 필요")
        if log not in navtrain_logs:
            counts[(ds, "excluded:trainval_log_not_in_navtrain")] += n_qa
            excluded_examples[(ds, "trainval_log_not_in_navtrain")].append(src); return None
        if log in excl_logs:
            counts[(ds, "excluded:teacher_exclude_logs(NuPlanQA-Eval)")] += n_qa
            excluded_examples[(ds, "teacher_exclude_logs")].append(src); return None
        if log not in part_of:
            raise RuntimeError(f"{ds} {src}: navtrain 로그 {log} 가 fold 분할에 없다")
        if not has_images(log, ftok):
            counts[(ds, "excluded:front3_images_missing_on_disk")] += n_qa
            excluded_examples[(ds, "front3_images_missing_on_disk")].append(f"{src} -> {log}/{ftok}"); return None
        return log, ftok

    # ---- NuPlanQA
    meta = load_nuplanqa_meta(raw / "nuplanqa/nuplan_metadata.pkl")
    nq_train = json.load(open(raw / "nuplanqa/nuplanqa_train.json"))
    nq_eval = json.load(open(raw / "nuplanqa/nuplanqa_eval_test.json"))
    eval_tokens = {x["sample_token"] for x in nq_eval}
    hit_cache = {}
    nq_frame_hits = collections.Counter()
    for i, x in enumerate(nq_train):
        st = x["sample_token"]
        if st in eval_tokens:
            raise RuntimeError(f"NuPlanQA train#{i} sample_token {st} 가 eval 에도 있다")
        if len(x["QA"]) != 1:
            raise RuntimeError(f"NuPlanQA train#{i}: QA {len(x['QA'])}개(1개 가정)")
        qa = x["QA"][0]
        if not qa["q"].strip() or not qa["a"].strip():
            raise RuntimeError(f"NuPlanQA train#{i}: 빈 Q/A — 검토 필요")
        if st not in meta:
            counts[("nuplanqa", "excluded:sample_token_not_in_metadata")] += 1
            excluded_examples[("nuplanqa", "sample_token_not_in_metadata")].append(f"train#{i}"); continue
        if st not in hit_cache:
            hit_cache[st] = map_by_stems(meta[st]["cams"], stem_map)
            nq_frame_hits[hit_cache[st][1]] += 1
        r = route("nuplanqa", 1, hit_cache[st][0], f"train#{i}")
        if r is None:
            continue
        log, ftok = r
        counts[("nuplanqa", "used_qa")] += 1
        g = groups.setdefault(("nuplanqa", ftok), dict(log=log, turns=[], nuplanqa_tokens=set()))
        g["turns"].append(dict(q=qa["q"], a=qa["a"], type=x["task"], src=f"train#{i}"))
        g["nuplanqa_tokens"].add(st)

    # ---- ReCogDrive NAVSIM
    rc_path = raw / SOURCES["recogdrive_navsim"]["files"][0]
    corrupt_list = []
    with open(rc_path) as fh:
        for ln, line in enumerate(fh):
            if not line.strip():
                continue
            x = json.loads(line)
            ims = x["image"] if isinstance(x["image"], list) else [x["image"]]
            if len(ims) != 1:
                raise RuntimeError(f"ReCogDrive line {ln}: 영상 {len(ims)}장(1장 가정)")
            parts = ims[0].split("/")
            log, cam, stem = parts[-3], parts[-2], os.path.splitext(parts[-1])[0]
            conv = x["conversations"]
            if len(conv) % 2 or any(c["from"] != ("human" if k % 2 == 0 else "gpt") for k, c in enumerate(conv)):
                raise RuntimeError(f"ReCogDrive line {ln}: human/gpt 교대가 아니다")
            turns = []
            n_corrupt = 0
            for k in range(0, len(conv), 2):
                q, ans = conv[k]["value"], conv[k + 1]["value"]
                if k == 0:
                    if not q.startswith("<image>\n"):
                        raise RuntimeError(f"ReCogDrive line {ln}: 첫 질문이 <image>\\n 로 시작하지 않는다")
                    q = q[len("<image>\n"):]
                if "<image>" in q or "<image>" in ans:
                    raise RuntimeError(f"ReCogDrive line {ln}: 두 번째 <image> 자리표시")
                if not q.strip() or not ans.strip():
                    raise RuntimeError(f"ReCogDrive line {ln} turn {k // 2}: 빈 Q/A — 검토 필요")
                why = corrupt_reason(ans)
                if why:
                    corrupt_list.append(dict(src=f"id{x['id']}#t{k // 2}", reason=why, answer_chars=len(ans), answer_head=ans[:120]))
                    n_corrupt += 1
                    continue
                if any(0x4E00 <= ord(ch) <= 0x9FFF for ch in ans):
                    counts[("recogdrive_navsim", "note:answer_contains_cjk(kept)")] += 1
                turns.append(dict(q=q, a=ans, type=f"recogdrive_q{k // 2}", src=f"id{x['id']}#t{k // 2}"))
            h = stem_map.get(stem)
            if h is not None and (h[1] != log or h[3] != cam):
                raise RuntimeError(f"ReCogDrive line {ln}: 경로 {log}/{cam} 와 NAVSIM 색인 {h[1]}/{h[3]} 불일치")
            counts[("recogdrive_navsim", "excluded:corrupt_answer_turn")] += n_corrupt
            r = route("recogdrive_navsim", len(turns), None if h is None else h[:3], f"id{x['id']}")
            if r is None:
                continue
            log, ftok = r
            counts[("recogdrive_navsim", "used_qa")] += len(turns)
            g = groups.setdefault(("recogdrive_navsim", ftok), dict(log=log, turns=[], nuplanqa_tokens=set()))
            g["turns"] += turns
    print(f"[qa] 프레임 그룹 {len(groups)} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 표본(chunk)·길이
    proc = S.make_processor(S.BASE_QWEN, a.pixels)
    tok = proc.tokenizer
    tpc, hw_resized = tokens_per_cam(a.pixels)
    n_img = len(N.QA_CAMS)
    recs = {(f, p): [] for f in S.FOLDS for p in ("train", "holdout")}
    lens = collections.defaultdict(list)
    n_chunks_split = collections.Counter()
    for (ds, ftok), g in groups.items():
        f, part = part_of[g["log"]]
        chunks = chunk_turns(tok, g["turns"], a.max_seq_tokens, n_img, tpc)
        if sum(len(c) for c in chunks) != len(g["turns"]):
            raise AssertionError("chunk QA 개수 불일치")
        if len(chunks) > 1:
            n_chunks_split[ds] += 1
        for ci, c in enumerate(chunks):
            r = dict(id=f"{ds}:{ftok}:{ci}", dataset=ds, token=ftok, scene_token=g["log"], fold=f, part=part,
                     chunk=ci, n_chunks=len(chunks), cams=cam_paths(g["log"], ftok), turns=c)
            if g["nuplanqa_tokens"]:
                r["nuplanqa_sample_tokens"] = sorted(g["nuplanqa_tokens"])
            L, n_sup = text_len(tok, c, n_img, tpc)
            if L > a.max_seq_tokens:
                raise AssertionError(f"{r['id']}: {L} > {a.max_seq_tokens}")
            r["n_tokens"], r["n_answer_tokens"] = L, n_sup
            recs[(f, part)].append(r)
            lens[(ds, f, part)].append(L)
    print(f"[len] 완료 ({time.time() - t0:.0f}s)", flush=True)

    # ---- NuPlanQA-Eval(1,801 전부, 평가 전용)
    raw_index = {}
    if a.nuplan_raw_scan:
        need = [x for x in nq_eval if x["sample_token"] in meta and map_by_stems(meta[x["sample_token"]]["cams"], stem_map)[0] is None]
        if need:
            want = {meta[x["sample_token"]]["cams"]["CAM_F0"] for x in need}
            logs = sorted(os.listdir(N.NUPLAN_RAW_BLOBS))
            for k, lg in enumerate(logs):
                s = listdir_set(os.path.join(N.NUPLAN_RAW_BLOBS, lg, "CAM_F0")) or set()
                for fn in s:
                    st = fn[:-4]
                    if st in want:
                        raw_index[st] = lg
                if (k + 1) % 200 == 0:
                    print(f"[raw] {k + 1}/{len(logs)} 로그 listdir, 찾음 {len(raw_index)}/{len(want)}", flush=True)
    ev_recs, ev_counts = [], collections.Counter()
    for i, x in enumerate(nq_eval):
        st = x["sample_token"]
        qa = x["QA"][0]
        if len(x["QA"]) != 1 or not qa["q"].strip() or not qa["a"].strip():
            raise RuntimeError(f"NuPlanQA eval#{i}: QA 형식/빈 답")
        r = dict(id=f"nuplanqa_eval:{i}", dataset="nuplanqa_eval", sample_token=st, task=x["task"], split=x.get("split"),
                 turns=[dict(q=qa["q"], a=qa["a"], type=x["task"], src=f"eval_test#{i}")],
                 navsim=None, cams=None, image_source=None)
        if st not in meta:
            ev_counts["no_metadata"] += 1
        else:
            hit, _ = map_by_stems(meta[st]["cams"], stem_map)
            cams = meta[st]["cams"]
            if hit is not None:
                split, log, ftok = hit
                r["navsim"] = dict(split=split, log=log, frame_token=ftok, navtrain_log=log in navtrain_logs,
                                   navtest_log=log in navtest_logs, teacher_exclude_log=log in excl_logs)
                if split == "test":
                    ev_counts["in_navsim_test_logs"] += 1
                blob = N.TRAIN_BLOBS if split == "trainval" else N.TEST_BLOBS
                ps = [os.path.join(blob, log, c, tok_map[ftok][3][c] + ".jpg") for c in N.QA_CAMS]
                if all(os.path.exists(p) for p in ps):
                    r["cams"], r["image_source"] = ps, "navsim_sensor_blobs"
            if r["cams"] is None:
                lg = r["navsim"]["log"] if r["navsim"] else raw_index.get(cams["CAM_F0"])
                if lg is not None:
                    ps = [os.path.join(N.NUPLAN_RAW_BLOBS, lg, c, cams[c] + ".jpg") for c in N.QA_CAMS]
                    if all(os.path.exists(p) for p in ps):
                        r["cams"], r["image_source"] = ps, "nuplan_raw_sensor_blobs"
                        r["nuplan_log"] = lg
            ev_counts["in_navsim_frames" if hit is not None else "not_in_navsim_frames"] += 1
        ev_counts[f"images:{r['image_source'] or 'not_found'}"] += 1
        ev_recs.append(r)

    # ---- 저장
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for (f, p), rs in recs.items():
        rs.sort(key=lambda r: r["id"])
        fp = Q.qa_file(f, p, str(out))
        with open(fp, "w") as fh:
            for r in rs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[f"fold{f}_{p}"] = dict(path=os.path.relpath(fp, S.REPO), sha256=S.file_sha256(fp), n_samples=len(rs))
    evp = out / "nuplanqa_eval_navsim.jsonl"
    with open(evp, "w") as fh:
        for r in ev_recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {}
    for (ds, f, p), L in sorted(lens.items()):
        rs = [r for r in recs[(f, p)] if r["dataset"] == ds]
        L = np.asarray(L)
        stats[f"{ds}/fold{f}_{p}"] = dict(
            samples=len(rs), frames=len({r["token"] for r in rs}), logs=len({r["scene_token"] for r in rs}),
            qa=sum(len(r["turns"]) for r in rs), answer_tokens=int(sum(r["n_answer_tokens"] for r in rs)),
            seq_tokens_total=int(L.sum()), len_min=int(L.min()), len_p50=int(np.percentile(L, 50)),
            len_p90=int(np.percentile(L, 90)), len_p99=int(np.percentile(L, 99)), len_max=int(L.max()))
    src_info = {ds: dict(paper=v["paper"], note=v["note"],
                         files={fn: dict(sha256=S.file_sha256(raw / fn), bytes=os.path.getsize(raw / fn)) for fn in v["files"]})
                for ds, v in SOURCES.items()}
    manifest = dict(
        created="2026-10-16", tag="# 수정(2026-10-16): 교사 QA — NAVSIM 트랙", benchmark="navsim", sources=src_info,
        raw_dir=str(raw), fold_file=fpath, fold_sha256=fd["sha256"], max_seq_tokens=a.max_seq_tokens,
        pixels=a.pixels, n_cams=n_img, cams=list(N.QA_CAMS), tokens_per_cam=tpc, resized_hw=list(hw_resized),
        image_hw_assumed=list(N.NAVSIM_IMAGE_HW),
        assignment=dict(teacher_A=dict(train="train.A − holdout.A", holdout="holdout.A"),
                        teacher_B=dict(train="train.B − holdout.B", holdout="holdout.B"),
                        excluded="navtest 로그 전체 + teacher_exclude_logs 6 로그(NuPlanQA-Eval)"),
        counts={"|".join(k): v for k, v in sorted(counts.items())},
        nuplanqa_frames_by_matched_stems=dict(sorted(nq_frame_hits.items())),
        excluded_examples={"|".join(k): v[:5] for k, v in excluded_examples.items()},
        frames_split_into_chunks=dict(n_chunks_split), files=files, stats=stats, excluded_corrupt_answers=corrupt_list,
        eval_file=dict(path=os.path.relpath(evp, S.REPO), sha256=S.file_sha256(evp), n=len(ev_recs), counts=dict(ev_counts)),
        image_listdir_logs=len(img_sets), seconds=round(time.time() - t0, 1))
    with open(out / "qa_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    print(json.dumps(dict(counts=manifest["counts"], chunks=manifest["frames_split_into_chunks"], files=files, stats=stats,
                          eval=manifest["eval_file"], tpc=tpc, frames_hits=manifest["nuplanqa_frames_by_matched_stems"]),
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
