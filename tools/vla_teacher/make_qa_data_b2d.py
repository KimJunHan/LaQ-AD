# 수정(2026-10-16): B2D 교사 QA 준비 — 공개 Bench2Drive QA 주석 → fold 별 학습/hold-out + val 표본(jsonl) 생성(CPU).
"""Chat-B2D(ORION, ChatB2D-plus) + Bench2Drive-VL(DriveCommenter) → data/vla_teacher/b2d/qa/fold{A,B}_{train,holdout}.jsonl, val.jsonl

실행(내려받은 주석은 신뢰하지 않는 데이터 — 반드시 python -I, GPU 없이):
  OPENBLAS_CORETYPE=SkylakeX CUDA_VISIBLE_DEVICES="" python -I tools/vla_teacher/make_fold_split_b2d.py   # 먼저 1 회
  OPENBLAS_CORETYPE=SkylakeX CUDA_VISIBLE_DEVICES="" python -I tools/vla_teacher/make_qa_data_b2d.py

형식: nuScenes 교사 QA(make_qa_data.py → qa_common.QAFrameDataset / train_qa_full.py)와 똑같은 레코드
  {id, dataset, token, scene_token, fold, part, chunk, n_chunks, cams(6 장, CAM_ORDER 순), turns[{q,a,type,src}], n_tokens, n_answer_tokens}
  * token = "<clip>/<frame 5자리>", scene_token = clip 이름(= route).
  * cams = B2D v1/<clip>/camera/rgb_{front,front_left,front_right,back_left,back,back_right}/<frame>.jpg (원본 1600×900).
    Qwen 입력 해상도는 nuScenes QA 와 같은 processor 설정(sft_common.PIXELS = 200×28×28 → 캠당 180 토큰)을 그대로 쓴다.
    (B2D 640×352 는 주행 모델(HiP-AD) 입력 규칙이고 Qwen 입력과는 별개.)
장면 배정(유출 방지):
  * 공식 Bench2DriveZoo val 50 clip 의 QA 는 학습 파일에 절대 넣지 않고 val.jsonl 에만(평가용, fold = val.A / val.B 반쪽 표시).
  * teacher_A: 학습 = train.A − holdout.A, hold-out = holdout.A. teacher_B 도 같은 규칙. train.B clip 의 QA 는 teacher_A 파일에 없음.
출처 선택(중복 방지):
  * chat_b2d_plus = ChatB2D-plus.zip. 원본 chat-B2D.zip 의 QA 는 모든 프레임에서 plus 의 앞부분과 글자 단위로 같다(검사해서 기록) —
    plus 는 프레임마다 'What actions should the car be taking?' 1 개를 더한 상위 집합이라 원본은 따로 넣지 않는다.
  * b2d_vl_drivecommenter = Bench2Drive-VL(DriveCommenter) 999 clip 전부(upstream 에 1 clip 없음 — 기록).
  * recogdrive Bench2drive_QA 는 Bench2Drive-VL 재포장(중복)이라 넣지 않는다. Bench2drive_Traj·reasonplan 은 이번 QA 단계 대상 아님.
표본: (데이터셋, 프레임) 하나의 QA 전부 → 다중 턴 대화. --max_seq_tokens 를 넘으면 잘라내지 않고 QA 순서대로 여러 대화(chunk)로 나눈다.
이상 표본: 빈 Q/A, 문자열 아님, Qwen 특수 토큰 문자열('<|') 포함, 영상 누락, 원본 파일 해시 불일치, QA 하나가 최대 길이 초과 →
  조용히 버리지 않고 excluded_report.json 에 원인·위치를 전부 적는다(학습에 쓸 수 없는 것만 제외).
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import b2d_qa_common as B  # noqa: E402
import qa_common as Q  # noqa: E402
import sft_common as S  # noqa: E402

CHAT_DIR = f"{B.QA_ROOT}/chat_b2d"
CHAT_ZIP, CHAT_ZIP_PREFIX = "ChatB2D-plus.zip", "data/chat-B2D/"
CHAT_ORIG_ZIP, CHAT_ORIG_PREFIX = "chat-B2D.zip", "chat-B2D/"
DC_DIR = f"{B.QA_ROOT}/b2d_vl_drivecommenter"
DC_SUMS = f"{B.QA_ROOT}/b2d_vl_drivecommenter.sha256sums.txt"
MANIFEST_DL = f"{B.QA_ROOT}/qa_manifest_b2d_navsim.json"
DC_CATS = ("perception", "prediction", "planning", "behaviour")
SOURCES = {
    "chat_b2d_plus": dict(
        url="https://huggingface.co/datasets/poleyzdk/Chat-B2D (revision 5a449d7fc720ba563b7107f84a2b33c1d66c242c)",
        paper="H. Fu et al., 'ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action "
              "Generation', ICCV 2025 (arXiv:2503.19755)",
        licence="HF card Apache-2.0; images = Bench2Drive terms"),
    "b2d_vl_drivecommenter": dict(
        url="https://huggingface.co/datasets/Telkwevr/Bench2Drive-VL-base (revision 02f8d3b0f7e16eddc0258b24b962548ee7233e66)",
        paper="Jia et al., 'Bench2Drive-VL: Benchmarks for Closed-Loop Autonomous Driving with Vision-Language Models', "
              "arXiv:2604.01259 (2026)",
        licence="HF card Apache-2.0 vs GitHub README CC-BY-NC-ND (충돌 → 비상업·재배포 금지로 취급)"),
}


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def text_problem(x):
    """학습 정답/질문으로 쓸 수 없는 이유(없으면 None)."""
    if not isinstance(x, str):
        return f"문자열 아님({type(x).__name__})"
    if not x.strip():
        return "빈 문자열"
    if "<|" in x:
        return "Qwen 특수 토큰 표기 '<|' 포함(대화 틀을 깨뜨림)"
    return None


# ---------------------------------------------------------------- Chat-B2D(plus)
def load_chat(bad):
    """반환 {(split, clip, frame): [turn]}, 원본 포함 검사 결과."""
    zp = zipfile.ZipFile(f"{CHAT_DIR}/{CHAT_ZIP}")
    zo = zipfile.ZipFile(f"{CHAT_DIR}/{CHAT_ORIG_ZIP}")
    names = sorted(n for n in zp.namelist() if n.endswith(".json"))
    onames = {n[len(CHAT_ORIG_PREFIX):] for n in zo.namelist() if n.endswith(".json")}
    if any(not n.startswith(CHAT_ZIP_PREFIX) for n in zp.namelist()):
        raise RuntimeError("ChatB2D-plus.zip 에 예상 밖 경로")
    out, sub = {}, collections.Counter()
    for n in names:
        rel = n[len(CHAT_ZIP_PREFIX):]
        parts = rel.split("/")
        if len(parts) != 3 or parts[0] not in ("train", "val") or not parts[2][:-5].isdigit() or len(parts[2]) != 10:
            bad.append(dict(dataset="chat_b2d_plus", where=n, cause="경로 형식 이상(split/clip/NNNNN.json 아님)"))
            continue
        split, clip, frame = parts[0], parts[1], parts[2][:-5]
        d = json.loads(zp.read(n))
        turns = []
        for i, t in enumerate(d):
            src = f"{split}/{clip}/{frame}#{i}"
            if not (isinstance(t, list) and len(t) == 2 and isinstance(t[0], dict) and isinstance(t[1], dict)
                    and t[0].get("from") == "human" and t[1].get("from") == "gpt"):
                bad.append(dict(dataset="chat_b2d_plus", where=src, cause="human/gpt 한 쌍 형식 아님"))
                continue
            q, a = t[0].get("value"), t[1].get("value")
            pq, pa = text_problem(q), text_problem(a)
            if pq or pa:
                bad.append(dict(dataset="chat_b2d_plus", where=src, cause=f"Q: {pq}" if pq else f"A: {pa}"))
                continue
            turns.append(dict(q=q, a=a, type="chat", src=src))
        # 원본 chat-B2D 의 QA 가 plus 의 앞부분과 같은지(같으면 원본을 따로 넣을 필요 없음)
        if rel in onames:
            o = json.loads(zo.read(CHAT_ORIG_PREFIX + rel))
            po = [(t[0]["value"], t[1]["value"]) for t in o]
            pp = [(t[0]["value"], t[1]["value"]) for t in d]
            sub["orig_prefix_of_plus" if pp[:len(po)] == po else "orig_not_prefix"] += 1
            sub["orig_qa"] += len(po)
            sub["plus_extra_qa"] += len(pp) - len(po)
        else:
            sub["plus_only_frame"] += 1
        out[(split, clip, frame)] = turns
    sub["orig_only_frame"] = len(onames - {n[len(CHAT_ZIP_PREFIX):] for n in names})
    return out, dict(sub)


# ---------------------------------------------------------------- Bench2Drive-VL DriveCommenter
_SUMS = None


def _dc_clip(clip):
    """한 clip 의 프레임 json 전부(작업 프로세스). 반환 (clip, {frame: [turn]}, bad, stats)."""
    d = Path(DC_DIR) / clip
    frames, bad, st = {}, [], collections.Counter()
    for fn in sorted(os.listdir(d)):
        where = f"{clip}/{fn}"
        if not (fn.endswith(".json") and len(fn) == 10 and fn[:5].isdigit()):
            bad.append(dict(dataset="b2d_vl_drivecommenter", where=where, cause="프레임 json 이름 형식 아님"))
            continue
        raw = open(d / fn, "rb").read()
        exp = _SUMS.get(f"b2d_vl_drivecommenter/{clip}/{fn}")
        if exp is None or hashlib.sha256(raw).hexdigest() != exp:
            bad.append(dict(dataset="b2d_vl_drivecommenter", where=where, cause="sha256sums 와 해시 불일치/목록에 없음"))
            continue
        j = json.loads(raw)
        frame = fn[:5]
        ip = j.get("image_paths") or {}
        wrong = [(cn, ip.get(cn, "")) for cn, cd in zip(B.CAM_NAMES, B.CAM_DIRS)   # 원본이 가리키는 영상 = 우리 영상(clip·카메라·프레임)
                 if not str(ip.get(cn, "")).endswith(f"{clip}/camera/{cd}/{frame}.jpg")]
        if wrong:
            st["image_path_mismatch_frames"] += 1
            bad.append(dict(dataset="b2d_vl_drivecommenter", where=where,
                            cause=f"image_paths 가 다른 영상을 가리킴 — 이 프레임 QA 전부 제외(영상-답 불일치) {wrong}"))
            continue
        extra = set(j.get("QA", {})) - set(DC_CATS)
        if extra:
            bad.append(dict(dataset="b2d_vl_drivecommenter", where=where, cause=f"알 수 없는 QA 범주 {sorted(extra)}(사용 안 함)"))
        turns = []
        for cat in DC_CATS:
            for i, qa in enumerate(j.get("QA", {}).get(cat, [])):
                src = f"{clip}/{frame}/{cat}#{i}"
                q, a = qa.get("Q"), qa.get("A")
                pq, pa = text_problem(q), text_problem(a)
                if pq or pa:
                    bad.append(dict(dataset="b2d_vl_drivecommenter", where=src, cause=f"Q: {pq}" if pq else f"A: {pa}",
                                    q=q if isinstance(q, str) else None))
                    continue
                if qa.get("C") not in (None, "", []):
                    st["C_nonempty"] += 1     # 원본 'C'(맥락) 필드 — 질문에 넣지 않음(개수만 기록)
                turns.append(dict(q=q, a=a, type=cat, src=src))
                st[f"qa_{cat}"] += 1
        frames[frame] = turns
    return clip, frames, bad, dict(st)


def _init_worker(sums):
    global _SUMS
    _SUMS = sums


def load_dc(workers, bad):
    sums = {}
    for line in open(DC_SUMS):
        h, p = line.rstrip("\n").split("  ", 1)
        sums[p] = h
    clips = sorted(c for c in os.listdir(DC_DIR) if (Path(DC_DIR) / c).is_dir() and not c.startswith("."))
    out, st = {}, collections.Counter()
    t = time.time()
    with Pool(workers, initializer=_init_worker, initargs=(sums,)) as pool:
        for k, (clip, frames, b, s) in enumerate(pool.imap_unordered(_dc_clip, clips, chunksize=4)):
            for fr, turns in frames.items():
                out[(clip, fr)] = turns
            bad += b
            st.update(s)
            if (k + 1) % 100 == 0:
                print(f"[dc] {k + 1}/{len(clips)} clip ({time.time() - t:.0f}s)", flush=True)
    return out, clips, dict(st)


# ---------------------------------------------------------------- 토큰 길이(정확 — qa_common.tokenize_pieces 와 같은 조각 합)
def batch_lens(tok, texts, bs=20000):
    out = np.zeros(len(texts), dtype=np.int32)
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], add_special_tokens=False)["input_ids"]
        out[i:i + bs] = [len(e) for e in enc]
    return out


def head_plain(q):
    return "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n"


IMG = "<|vision_start|><|image_pad|><|vision_end|>" * S.NUM_CAMS


def head_first_len(tok, q):
    # 첫 턴 머리말(영상 자리표시 6 개 포함). <|image_pad|> 는 특수 토큰 1 개 → 캠당 TOKENS_PER_CAM 개로 펼치면 +(180−1)
    return len(tok("<|im_start|>user\n" + IMG + q + "<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]) \
        + S.NUM_CAMS * (S.TOKENS_PER_CAM - 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=B.QA_DIR_B2D)
    ap.add_argument("--fold_file", default=B.FOLD_FILE_B2D)
    ap.add_argument("--max_seq_tokens", type=int, default=4096, help="nuScenes QA 와 같은 기본값(영상 1,080 + 텍스트)")
    ap.add_argument("--workers", type=int, default=6, help="DriveCommenter json 읽기 프로세스 수(CPU 공유 — ≤ 6 권장)")
    ap.add_argument("--n_verify", type=int, default=3000, help="길이 계산을 qa_common.tokenize_pieces 로 직접 대조할 무작위 표본 수")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    os.environ.setdefault("RAYON_NUM_THREADS", "6")       # 빠른 토크나이저 스레드 수 제한(CPU 공유)
    out = Path(a.out) if os.path.isabs(a.out) else S.REPO / a.out
    if out.exists() and any(out.glob("*.jsonl")) and not a.force:
        raise SystemExit(f"[거부] {out} 에 이미 산출물 — 덮어쓰려면 --force")
    t0 = time.time()

    # 분할
    fd, fpath = S.load_fold_split(a.fold_file)
    off_val = B.official_val_clips()
    clips_all = set(B.all_clips())
    trA, trB = set(fd["train"]["A"]["scenes"]), set(fd["train"]["B"]["scenes"])
    vaA, vaB = set(fd["val"]["A"]["scenes"]), set(fd["val"]["B"]["scenes"])
    hoA, hoB = set(fd["holdout"]["A"]["scenes"]), set(fd["holdout"]["B"]["scenes"])
    if (vaA | vaB) != off_val or (trA | trB) != clips_all - off_val or trA & trB or vaA & vaB:
        raise RuntimeError("fold_split_b2d 의 train/val clip 이 공식 split 과 다르다")
    part_of = {}
    for s in trA - hoA: part_of[s] = ("A", "train")
    for s in hoA: part_of[s] = ("A", "holdout")
    for s in trB - hoB: part_of[s] = ("B", "train")
    for s in hoB: part_of[s] = ("B", "holdout")
    for s in vaA: part_of[s] = ("A", "val")
    for s in vaB: part_of[s] = ("B", "val")

    listing = B.image_listing(S.REPO)
    img_sets = {c: {cd: set(v[cd]) if v[cd] is not None else set() for cd in B.CAM_DIRS} for c, v in listing.items()}

    bad = []
    # 내려받은 원본 해시(내려받기 manifest 와 대조)
    dl = json.load(open(MANIFEST_DL))
    dl_chat = {f["path"]: f["sha256"] for e in dl["downloaded"] if e["name"] == "chat_b2d" for f in e["files"]}
    src_info = {}
    for z in (CHAT_ZIP, CHAT_ORIG_ZIP):
        h = sha256_file(f"{CHAT_DIR}/{z}")
        if dl_chat.get(f"chat_b2d/{z}") != h:
            raise RuntimeError(f"{z}: sha256 {h} != 내려받기 manifest {dl_chat.get('chat_b2d/' + z)}")
        src_info.setdefault("chat_b2d_plus", dict(SOURCES["chat_b2d_plus"], files={}))["files"][z] = dict(
            sha256=h, bytes=os.path.getsize(f"{CHAT_DIR}/{z}"))
    sums_h = sha256_file(DC_SUMS)
    dl_dc = [e for e in dl["downloaded"] if e["name"] == "b2d_vl_drivecommenter"][0]
    if dl_dc["files"]["sha256_of_sha256sums_file"] != sums_h:
        raise RuntimeError("DriveCommenter sha256sums 파일 해시가 내려받기 manifest 와 다르다")
    src_info["b2d_vl_drivecommenter"] = dict(SOURCES["b2d_vl_drivecommenter"], sha256sums_file=DC_SUMS, sha256sums_sha256=sums_h,
                                             per_file_check="모든 프레임 json 을 읽을 때 sha256sums 와 대조")

    print("[load] Chat-B2D(plus) …", flush=True)
    chat, chat_sub = load_chat(bad)
    if chat_sub.get("orig_not_prefix") or chat_sub.get("orig_only_frame"):
        raise RuntimeError(f"원본 chat-B2D 가 plus 의 부분이 아닌 프레임 있음: {chat_sub} — 원본도 따로 넣을지 검토 필요")
    # Chat-B2D 자체 train/val 폴더 == 공식 split 확인
    ch_train = {c for (sp, c, _) in chat if sp == "train"}
    ch_val = {c for (sp, c, _) in chat if sp == "val"}
    chat_split_check = dict(train_clips=len(ch_train), val_clips=len(ch_val), val_eq_official=ch_val == off_val,
                            train_in_official_val=len(ch_train & off_val), train_eq_official_train=ch_train == clips_all - off_val)
    if ch_val != off_val or ch_train & off_val:
        raise RuntimeError(f"Chat-B2D 폴더 split 이 공식과 다름: {chat_split_check}")
    print(f"[load] Chat-B2D {len(chat)} 프레임, {time.time() - t0:.0f}s, 원본 포함 검사 {chat_sub}", flush=True)

    print("[load] DriveCommenter …", flush=True)
    dc, dc_clips, dc_st = load_dc(a.workers, bad)
    dc_missing = sorted(clips_all - set(dc_clips))
    dc_extra = sorted(set(dc_clips) - clips_all)
    if dc_extra:
        raise RuntimeError(f"DriveCommenter 에 B2D 에 없는 clip {dc_extra}")
    print(f"[load] DriveCommenter {len(dc)} 프레임 / {len(dc_clips)} clip, 없는 clip {dc_missing}, {time.time() - t0:.0f}s", flush=True)

    # (데이터셋, 프레임) 묶음 + 영상 확인
    groups = collections.OrderedDict()
    counts = collections.Counter()
    img_missing = []
    items = [("chat_b2d_plus", c, f, turns) for (sp, c, f), turns in sorted(chat.items())] + \
            [("b2d_vl_drivecommenter", c, f, turns) for (c, f), turns in sorted(dc.items())]
    for ds, clip, frame, turns in items:
        if clip not in part_of:
            raise RuntimeError(f"{ds}: clip {clip} 가 분할 어디에도 없다")
        miss = [cd for cd in B.CAM_DIRS if f"{frame}.jpg" not in img_sets[clip][cd]]
        if miss:
            img_missing.append(dict(dataset=ds, where=f"{clip}/{frame}", cams_missing=miss, n_qa=len(turns)))
            bad.append(dict(dataset=ds, where=f"{clip}/{frame}", cause=f"영상 없음 {miss} — 이 프레임 QA {len(turns)}개 제외(대체 금지)"))
            counts[(ds, "excluded_image_missing", "qa")] += len(turns)
            continue
        if not turns:
            bad.append(dict(dataset=ds, where=f"{clip}/{frame}", cause="쓸 수 있는 QA 0개인 프레임(위 항목들로 전부 제외됨)"))
            continue
        f, part = part_of[clip]
        counts[(ds, f"fold{f}_{part}" if part != "val" else f"val{f}", "qa")] += len(turns)
        groups[(ds, f"{clip}/{frame}")] = dict(scene=clip, fold=f, part=part, frame=frame, turns=turns)
    print(f"[group] {len(groups)} (데이터셋, 프레임) 묶음, {time.time() - t0:.0f}s", flush=True)

    # 토큰 길이(배치 계산 — 조각별 토큰화의 합 = qa_common.tokenize_pieces 길이)
    proc = S.make_processor(S.BASE_QWEN)
    tok = proc.tokenizer
    sys_len = len(tok(f"<|im_start|>system\n{S.QWEN_DEFAULT_SYSTEM}<|im_end|>\n", add_special_tokens=False)["input_ids"])
    nl_len = len(tok("\n", add_special_tokens=False)["input_ids"])
    flat = [t for g in groups.values() for t in g["turns"]]
    flat_ds = np.asarray([B.DATASETS_B2D.index(ds) for (ds, _), g in groups.items() for _ in g["turns"]], dtype=np.int8)
    print(f"[tok] 턴 {len(flat)}개 토큰화 …", flush=True)
    Lh = batch_lens(tok, [head_plain(t["q"]) for t in flat])
    La = batch_lens(tok, [t["a"] + "<|im_end|>" for t in flat])
    print(f"[tok] 완료 {time.time() - t0:.0f}s", flush=True)
    first_cache = {}

    def first_len(q):
        if q not in first_cache:
            first_cache[q] = head_first_len(tok, q)
        return first_cache[q]

    recs = collections.defaultdict(list)
    lens = collections.defaultdict(list)
    n_chunks_split = collections.Counter()
    too_long = []
    k = 0
    for (ds, token), g in groups.items():
        n = len(g["turns"])
        idx = range(k, k + n)
        k += n
        chunks, cur, cur_len = [], [], sys_len
        for t, j in zip(g["turns"], idx):
            tl_plain = int(Lh[j]) + int(La[j]) + nl_len
            if cur and cur_len + tl_plain <= a.max_seq_tokens:
                cur.append((t, j)); cur_len += tl_plain
                continue
            if cur:
                chunks.append((cur, cur_len))
            tl_first = first_len(t["q"]) + int(La[j]) + nl_len
            if sys_len + tl_first > a.max_seq_tokens:
                too_long.append(dict(dataset=ds, where=t["src"], length=sys_len + tl_first))
                bad.append(dict(dataset=ds, where=t["src"], cause=f"QA 하나만으로 max_seq_tokens {a.max_seq_tokens} 초과 "
                                f"({sys_len + tl_first}) — 잘라내지 않고 제외"))
                cur, cur_len = [], sys_len
                continue
            cur, cur_len = [(t, j)], sys_len + tl_first
        if cur:
            chunks.append((cur, cur_len))
        if len(chunks) > 1:
            n_chunks_split[ds] += 1
        for ci, (c, L) in enumerate(chunks):
            r = dict(id=f"{ds}:{token}:{ci}", dataset=ds, token=token, scene_token=g["scene"], fold=g["fold"], part=g["part"],
                     chunk=ci, n_chunks=len(chunks),
                     cams=[B.cam_path(g["scene"], cd, g["frame"]) for cd in B.CAM_DIRS],
                     turns=[t for t, _ in c], n_tokens=int(L), n_answer_tokens=int(sum(int(La[j]) for _, j in c)))
            if L > a.max_seq_tokens:
                raise AssertionError(f"{r['id']}: {L} > {a.max_seq_tokens}")
            key = ("val", None) if g["part"] == "val" else (g["fold"], g["part"])
            recs[key].append(r)
            lens[(ds, key)].append(int(L))
    print(f"[chunk] 완료 {time.time() - t0:.0f}s, 나뉜 프레임 {dict(n_chunks_split)}, 한 QA 초과 {len(too_long)}", flush=True)

    # 길이 계산 직접 대조(무작위 표본 — qa_common.tokenize_pieces 로 다시 계산)
    rng = random.Random(0)
    allr = [r for v in recs.values() for r in v]
    ver = rng.sample(allr, min(a.n_verify, len(allr))) + sorted(allr, key=lambda r: -r["n_tokens"])[:50]
    mism = 0
    for r in ver:
        ids, lab = Q.tokenize_pieces(tok, Q.conversation_pieces(r))
        if len(ids) != r["n_tokens"] or sum(1 for x in lab if x != -100) != r["n_answer_tokens"]:
            mism += 1
    if mism:
        raise AssertionError(f"배치 길이 계산 != tokenize_pieces: {mism}/{len(ver)}")
    print(f"[verify] 길이 대조 {len(ver)} 표본 모두 일치", flush=True)

    # 쓰기
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    names = {(f, p): f"fold{f}_{p}" for f in S.FOLDS for p in ("train", "holdout")}
    names[("val", None)] = "val"
    for key, name in names.items():
        rs = sorted(recs.get(key, []), key=lambda r: r["id"])
        fp = out / f"{name}.jsonl"
        with open(fp, "w") as fh:
            for r in rs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[name] = dict(path=os.path.relpath(fp, S.REPO), sha256=sha256_file(fp), n_samples=len(rs), bytes=os.path.getsize(fp))
        print(f"[write] {fp} {len(rs)} 표본", flush=True)

    stats = {}
    for (ds, key), L in sorted(lens.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        rs = [r for r in recs[key] if r["dataset"] == ds]
        L = np.asarray(L)
        na = np.asarray([r["n_answer_tokens"] for r in rs])
        stats[f"{ds}/{names[key]}"] = dict(
            samples=len(rs), frames=len({r["token"] for r in rs}), clips=len({r["scene_token"] for r in rs}),
            qa=sum(len(r["turns"]) for r in rs), answer_tokens=int(na.sum()),
            seq_tokens_total=int(L.sum()), len_min=int(L.min()), len_p50=int(np.percentile(L, 50)), len_mean=round(float(L.mean()), 1),
            len_p90=int(np.percentile(L, 90)), len_p99=int(np.percentile(L, 99)), len_max=int(L.max()),
            answer_tok_per_sample_p50=int(np.percentile(na, 50)), answer_tok_per_sample_max=int(na.max()))
    # 턴(QA) 단위 답 길이 통계
    turn_stats = {}
    for ds in B.DATASETS_B2D:
        m = np.nonzero(flat_ds == B.DATASETS_B2D.index(ds))[0]
        if len(m):
            la, lq = La[m] - 1, Lh[m]
            turn_stats[ds] = dict(n_qa=len(m), answer_tok_p50=int(np.percentile(la, 50)), answer_tok_p99=int(np.percentile(la, 99)),
                                  answer_tok_max=int(la.max()), question_head_tok_p50=int(np.percentile(lq, 50)),
                                  question_head_tok_max=int(lq.max()))

    cause_ct = collections.Counter((b["dataset"], b["cause"].split(" — ")[0].split(":")[0]) for b in bad)
    excl = dict(
        created="2026-10-16", tag="# 수정(2026-10-16): B2D 교사 QA 준비",
        malformed_or_unusable=dict(n=len(bad), by_cause={"|".join(k): v for k, v in sorted(cause_ct.items())}, items=bad),
        image_missing_frames=img_missing,
        not_converted_sources={
            "chat_b2d/chat-B2D.zip": f"원본 Chat-B2D — 모든 프레임에서 ChatB2D-plus 의 앞부분과 같음(검사 {chat_sub}). 중복이라 plus 만 사용",
            "recogdrive_navsim_b2d/Bench2drive_QA": "Bench2Drive-VL(DriveCommenter) 재포장(내려받기 manifest) — 중복이라 사용 안 함",
            "recogdrive_navsim_b2d/Bench2drive_Traj": "궤적 답(행동 학습용) — 이번 QA 단계 대상 아님",
            "reasonplan_pdr_b2d": "이번 QA 단계 대상 아님(작업 범위: chat_b2d + b2d_vl_drivecommenter)"},
        dc_missing_clips=dict(clips=dc_missing, cause="upstream Bench2Drive-VL-base 에 이 clip 폴더가 없음(999/1000)"),
        official_val_routed_to_val_file={"|".join(k): v for k, v in sorted(counts.items()) if k[1].startswith("val")})
    json.dump(excl, open(out / "excluded_report.json", "w"), indent=1, ensure_ascii=False)

    manifest = dict(
        created="2026-10-16", tag="# 수정(2026-10-16): B2D 교사 QA 준비", benchmark="b2d", sources=src_info,
        fold_file=fpath, fold_sha256=fd["sha256"], official_split=B.OFFICIAL_SPLIT, b2d_root=B.B2D_ROOT,
        max_seq_tokens=a.max_seq_tokens, pixels=S.PIXELS, tokens_per_cam=S.TOKENS_PER_CAM,
        cams="current 6 (teacher SFT order: " + ", ".join(B.CAM_DIRS) + ")",
        assignment=dict(teacher_A=dict(train="train.A − holdout.A", holdout="holdout.A"),
                        teacher_B=dict(train="train.B − holdout.B", holdout="holdout.B"),
                        val="공식 val 50 clip 전부 → val.jsonl(학습 파일에서 제외, fold 필드 = val 반쪽)"),
        chat_split_check=chat_split_check, chat_orig_subset_check=chat_sub, dc_stats=dc_st, dc_missing_clips=dc_missing,
        counts={"|".join(k): v for k, v in sorted(counts.items())}, frames_split_into_chunks=dict(n_chunks_split),
        n_excluded=len(bad), excluded_report="excluded_report.json", length_verified_samples=len(ver),
        files=files, stats=stats, turn_stats=turn_stats, wall_s=round(time.time() - t0, 1))
    json.dump(manifest, open(out / "qa_manifest.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(dict(counts=manifest["counts"], chunks=manifest["frames_split_into_chunks"], files=files, stats=stats,
                          turn_stats=turn_stats, n_excluded=len(bad), by_cause=excl["malformed_or_unusable"]["by_cause"]),
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
