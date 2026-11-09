# 수정(2026-10-16): 교사 QA 전체 학습 단계 — 공개 QA 주석 → fold 별 학습/hold-out 표본(jsonl) 생성(CPU, 수 분).
"""NuScenes-QA(Qian et al., AAAI 2024) + DriveLM-nuScenes v1.1(Sima et al., ECCV 2024) → data/vla_teacher/qa/fold{A,B}_{train,holdout}.jsonl

실행(내려받은 JSON 은 신뢰하지 않는 데이터 — 반드시 python -I):
  OPENBLAS_CORETYPE=SkylakeX python -I tools/vla_teacher/make_qa_data.py

장면 배정(유출 방지 — train_sft.py --fold 와 정확히 같은 장면 집합):
  * QA 의 sample_token → 공식 nuScenes v1.0-trainval sample.json 의 scene_token(데이터셋이 주는 scene 키와도 대조).
  * nuScenes 공식 val 150 장면(nuscenes-devkit splits.val) 의 QA 는 전부 제외. fold_split.json 의 val.A ∪ val.B 와 같은지 검사.
  * teacher_A: 학습 = train.A − holdout.A, hold-out(체크포인트 선택) = holdout.A. teacher_B 도 같은 규칙(train.B 쪽).
    train.B 장면의 QA 는 teacher_A 에 절대 들어가지 않는다(교차 적합: teacher_A 가 train.B 캐시를 만든다).
표본: (데이터셋, 키프레임) 하나의 QA 전부 → 다중 턴 대화 1개. 시퀀스(영상 1,080 + 텍스트)가 --max_seq_tokens 를 넘으면
  잘라내지 않고 QA 순서대로 여러 대화(chunk)로 나눈다 — 모든 QA 가 그대로 들어간다(개수 검사).
빈 답: 원본에 답이 비어 있는 QA 는 감독할 정답이 없다. EMPTY_ANSWER_KNOWN 에 원본 위치·이유를 적은 것만 학습 대상에서 빼고
  manifest 에 기록한다. 목록 밖의 빈 답이 나오면 즉시 실패(조용한 제외 금지).
"""
import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import qa_common as Q  # noqa: E402
import sft_common as S  # noqa: E402

NUSC_META = "/home1/irteam/data-vol1/src/HiP-AD/data/nuscenes/v1.0-trainval"
SOURCES = {
    "nuscenes_qa": dict(
        files=["nuscenes_qa/questions/NuScenes_train_questions.json", "nuscenes_qa/questions/NuScenes_val_questions.json"],
        url="https://drive.google.com/drive/folders/1jIkICT23wZWZYPrWCa0x-ubjpClSzOuU (README of https://github.com/qiantianwen/NuScenes-QA)",
        paper="T. Qian, J. Chen, L. Zhuo, Y. Jiao, Y.-G. Jiang, 'NuScenes-QA: A Multi-modal Visual Question Answering Benchmark "
              "for Autonomous Driving Scenario', AAAI 2024 (arXiv:2305.14836)",
        licence="repository: MIT; annotation licence not stated separately; images = nuScenes terms of use (CC BY-NC-SA 4.0)"),
    "drivelm_nus": dict(
        files=["drivelm_nuscenes/v1_1_train_nus.json"],
        url="https://drive.google.com/file/d/1CvTPwChKvfnvrZ1Wr0ZNVqtibkkNeGgt (challenge/README.md of "
            "https://github.com/OpenDriveLab/DriveLM; same file as HF OpenDriveLab/DriveLM v1_1_train_nus.json, 192,961,829 B)",
        paper="C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beisswenger, P. Luo, A. Geiger, H. Li, 'DriveLM: "
              "Driving with Graph Visual Question Answering', ECCV 2024 (arXiv:2312.14150)",
        licence="language data CC BY-NC-SA 4.0 (DriveLM README / HF card); images = nuScenes terms of use"),
}
# 원본에 답이 빈 QA(검토한 것만). (데이터셋, 원본 scene 키, frame 토큰, 범주, 범주 안 순번) → 이유
EMPTY_ANSWER_KNOWN = {
    ("drivelm_nus", "f55c8c62235146d895f611176fd0a5dc", "ff0d87987502472982ed3647c81236d9", "planning", 18):
        "DriveLM v1.1 원본 'A' 가 빈 문자열(질문: priority of the objects ... in descending order) — 감독할 정답 없음",
}


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def official_scene_maps():
    from nuscenes.utils import splits
    scenes = json.load(open(f"{NUSC_META}/scene.json"))
    samples = json.load(open(f"{NUSC_META}/sample.json"))
    name2tok = {s["name"]: s["token"] for s in scenes}
    s2scene = {s["token"]: s["scene_token"] for s in samples}
    val = {name2tok[n] for n in splits.val}
    train = {name2tok[n] for n in splits.train}
    if len(val) != 150 or len(train) != 700 or val & train:
        raise RuntimeError(f"공식 split 이상: val {len(val)} train {len(train)} 겹침 {len(val & train)}")
    return s2scene, train, val


def cams_by_token(data_dir):
    """교사 SFT 와 같은 영상 경로(현재 6 캠, 같은 순서). train/val 정답 파일 둘 다(검사용)."""
    out = {}
    for split in ("train", "val"):
        for r in S.load_records(data_dir, split):
            cams = r["cams"]
            names = [p.split("/samples/")[1].split("/")[0] for p in cams]
            if names != Q.CAM_ORDER:
                raise RuntimeError(f"{r['token']}: 카메라 순서 {names} != {Q.CAM_ORDER}")
            out[r["token"]] = (cams, r["scene_token"])
    return out


def load_nuscenes_qa(raw):
    """반환 [(token, scene_key=None, turn dict, 원본 split)]"""
    items = []
    for rel in SOURCES["nuscenes_qa"]["files"]:
        d = json.load(open(Path(raw) / rel))
        split = d["info"]["split"]
        for i, q in enumerate(d["questions"]):
            if q["split"] != split:
                raise RuntimeError(f"NuScenes-QA {rel}#{i}: split {q['split']} != 파일 {split}")
            items.append((q["sample_token"], None, dict(q=q["question"], a=q["answer"], type=q["template_type"],
                                                        hop=q["num_hop"], src=f"{split}#{i}"), split))
    return items


def load_drivelm(raw, cam_map):
    items, excluded, path_bad = [], [], 0
    d = json.load(open(Path(raw) / SOURCES["drivelm_nus"]["files"][0]))
    for sc, v in d.items():
        for ft, f in v["key_frames"].items():
            ip = f["image_paths"]
            if ft in cam_map:   # DriveLM 이 가리키는 영상 = 우리 영상(파일 이름 기준)
                ours = {os.path.basename(p) for p in cam_map[ft][0]}
                if {os.path.basename(p) for p in ip.values()} != ours:
                    path_bad += 1
            for cat, lst in f["QA"].items():
                for i, qa in enumerate(lst):
                    key = ("drivelm_nus", sc, ft, cat, i)
                    if not qa["A"].strip() or not qa["Q"].strip():
                        if key not in EMPTY_ANSWER_KNOWN:
                            raise RuntimeError(f"목록에 없는 빈 Q/A: {key} {qa!r} — 검토 후 EMPTY_ANSWER_KNOWN 에 추가할지 결정")
                        excluded.append(dict(key=list(key), reason=EMPTY_ANSWER_KNOWN[key], q=qa["Q"]))
                        continue
                    items.append((ft, sc, dict(q=qa["Q"], a=qa["A"], type=cat, src=f"{sc}/{ft}/{cat}#{i}"), "train"))
    if path_bad:
        raise RuntimeError(f"DriveLM image_paths 가 우리 영상 파일과 다른 프레임 {path_bad}개")
    return items, excluded


def text_len(tok, turns, n_img_tok):
    rec = dict(turns=turns)
    ids, lab = Q.tokenize_pieces(tok, Q.conversation_pieces(rec))
    return len(ids), sum(1 for x in lab if x != -100)


def chunk_turns(tok, turns, max_len):
    """QA 순서를 지키며 길이 ≤ max_len 인 대화들로 나눈다(탐욕). 한 QA 만으로 넘으면 실패(잘라내지 않음)."""
    head = len(Q.tokenize_pieces(tok, Q.conversation_pieces(dict(turns=[])))[0]) + S.NUM_CAMS * (S.TOKENS_PER_CAM + 2)
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=Q.QA_RAW)
    ap.add_argument("--out", default=Q.QA_DIR)
    ap.add_argument("--data_dir", default="data/vla_teacher/v2", help="교사 정답 파일(영상 경로·장면 대조용)")
    ap.add_argument("--fold_file", default=S.FOLD_FILE)
    ap.add_argument("--max_seq_tokens", type=int, default=4096,
                    help="한 대화 최대 길이(영상 1,080 + 텍스트). 넘는 프레임은 여러 대화로 나눈다(잘라내지 않음)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out) if os.path.isabs(a.out) else S.REPO / a.out
    if out.exists() and any(out.glob("fold*_*.jsonl")) and not a.force:
        raise SystemExit(f"[거부] {out} 에 이미 산출물 — 덮어쓰려면 --force")

    s2scene, off_train, off_val = official_scene_maps()
    fd, fpath = S.load_fold_split(a.fold_file)
    trA, trB = set(fd["train"]["A"]["scenes"]), set(fd["train"]["B"]["scenes"])
    vaA, vaB = set(fd["val"]["A"]["scenes"]), set(fd["val"]["B"]["scenes"])
    hoA, hoB = set(fd["holdout"]["A"]["scenes"]), set(fd["holdout"]["B"]["scenes"])
    if (trA | trB) != off_train or (vaA | vaB) != off_val:
        raise RuntimeError("fold_split 의 train/val 장면이 공식 nuScenes split 과 다르다")
    part_of = {}
    for s in trA - hoA: part_of[s] = ("A", "train")
    for s in hoA: part_of[s] = ("A", "holdout")
    for s in trB - hoB: part_of[s] = ("B", "train")
    for s in hoB: part_of[s] = ("B", "holdout")

    cam_map = cams_by_token(S.resolve_data_dir(a.data_dir))
    for t, (_, sc) in cam_map.items():
        if s2scene.get(t) != sc:
            raise RuntimeError(f"교사 정답 파일 장면과 공식 sample.json 장면이 다름: {t}")

    raw = Path(a.raw)
    src_info = {}
    for ds, info in SOURCES.items():
        src_info[ds] = dict(url=info["url"], paper=info["paper"], licence=info["licence"],
                            files={f: dict(sha256=sha256(raw / f), bytes=os.path.getsize(raw / f)) for f in info["files"]})
    items = load_nuscenes_qa(raw)
    dl_items, excluded = load_drivelm(raw, cam_map)
    items_by_ds = dict(nuscenes_qa=items, drivelm_nus=dl_items)

    proc = S.make_processor(S.BASE_QWEN)
    tok = proc.tokenizer
    groups = collections.OrderedDict()
    counts = collections.Counter()
    for ds, its in items_by_ds.items():
        for token, sc_key, turn, split in its:
            sc = s2scene.get(token)
            if sc is None:
                raise RuntimeError(f"{ds}: sample_token {token} 가 v1.0-trainval sample.json 에 없다")
            if sc_key is not None and sc_key != sc:
                raise RuntimeError(f"{ds}: scene 키 {sc_key} != 공식 {sc} (token {token})")
            if sc in off_val:
                counts[(ds, "excluded_val_scene", split)] += 1
                continue
            if sc not in part_of:
                raise RuntimeError(f"{ds}: 장면 {sc} 가 train/val 어디에도 없다")
            if split != "train":
                raise RuntimeError(f"{ds}: 원본 {split} 파일 QA 가 nuScenes train 장면에 있다({token})")
            f, part = part_of[sc]
            counts[(ds, f"fold{f}_{part}", "qa")] += 1
            groups.setdefault((ds, token), dict(scene=sc, fold=f, part=part, turns=[]))["turns"].append(turn)

    recs = {(f, p): [] for f in S.FOLDS for p in ("train", "holdout")}
    lens = collections.defaultdict(list)
    n_chunks_split = collections.Counter()
    for (ds, token), g in groups.items():
        if token not in cam_map:
            raise RuntimeError(f"{ds}: {token} 영상 경로 없음")
        chunks = chunk_turns(tok, g["turns"], a.max_seq_tokens)
        if sum(len(c) for c in chunks) != len(g["turns"]):
            raise AssertionError("chunk QA 개수 불일치")
        if len(chunks) > 1:
            n_chunks_split[ds] += 1
        for ci, c in enumerate(chunks):
            r = dict(id=f"{ds}:{token}:{ci}", dataset=ds, token=token, scene_token=g["scene"], fold=g["fold"], part=g["part"],
                     chunk=ci, n_chunks=len(chunks), cams=cam_map[token][0], turns=c)
            L, n_sup = text_len(tok, c, S.TOKENS_PER_CAM)
            if L > a.max_seq_tokens:
                raise AssertionError(f"{r['id']}: {L} > {a.max_seq_tokens}")
            r["n_tokens"], r["n_answer_tokens"] = L, n_sup
            recs[(g["fold"], g["part"])].append(r)
            lens[(ds, g["fold"], g["part"])].append(L)

    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for (f, p), rs in recs.items():
        rs.sort(key=lambda r: r["id"])
        fp = Q.qa_file(f, p, str(out))
        with open(fp, "w") as fh:
            for r in rs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[f"fold{f}_{p}"] = dict(path=os.path.relpath(fp, S.REPO), sha256=sha256(fp), n_samples=len(rs))

    stats = {}
    for (ds, f, p), L in sorted(lens.items()):
        rs = [r for r in recs[(f, p)] if r["dataset"] == ds]
        L = np.asarray(L)
        stats[f"{ds}/fold{f}_{p}"] = dict(
            samples=len(rs), frames=len({r["token"] for r in rs}), scenes=len({r["scene_token"] for r in rs}),
            qa=sum(len(r["turns"]) for r in rs), answer_tokens=int(sum(r["n_answer_tokens"] for r in rs)),
            seq_tokens_total=int(L.sum()), len_min=int(L.min()), len_p50=int(np.percentile(L, 50)),
            len_p90=int(np.percentile(L, 90)), len_p99=int(np.percentile(L, 99)), len_max=int(L.max()))
    manifest = dict(
        created="2026-10-16", tag=Q.TAG, sources=src_info, raw_dir=str(raw), fold_file=fpath, fold_sha256=fd["sha256"],
        nuscenes_meta=NUSC_META, max_seq_tokens=a.max_seq_tokens, pixels=S.PIXELS, cams="current 6 (teacher SFT order)",
        assignment=dict(teacher_A=dict(train="train.A − holdout.A", holdout="holdout.A"),
                        teacher_B=dict(train="train.B − holdout.B", holdout="holdout.B"), excluded="nuScenes val 150 장면 전체"),
        counts={"|".join(k): v for k, v in sorted(counts.items())}, frames_split_into_chunks=dict(n_chunks_split),
        excluded_empty_answer=excluded, files=files, stats=stats)
    json.dump(manifest, open(out / "qa_manifest.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(dict(counts=manifest["counts"], chunks=manifest["frames_split_into_chunks"], files=files, stats=stats,
                          excluded_empty=len(excluded)), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
