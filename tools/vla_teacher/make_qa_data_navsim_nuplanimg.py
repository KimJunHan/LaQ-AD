# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)
"""NuPlanQA 를 nuPlan 원본 영상으로 다시 만든 NAVSIM 교사 QA (make_qa_data_navsim.py 의 변형 — 기존 파일은 바꾸지 않음).

달라진 점(사용자 결정 2026-10-19: "NuPlanQA 는 nuPlan 원본 영상 사용"):
  - NuPlanQA 표본의 영상 = nuPlan 원본 sensor_blobs(/home1/irteam/data-vol1/nas_mirror/Public_datasets/Nuplan/dataset/nuplan-v1.1/sensor_blobs/<log>/<cam>/<stem>.jpg)
    의 전방 3캠(CAM_L0, CAM_F0, CAM_R0). 영상 이름(stem)은 nuplan_metadata.pkl 의 data_path 파일 이름 그대로 = NuPlanQA 가 실제로
    질문을 만든 그 영상. (종전: NAVSIM 2 Hz 프레임의 OpenScene 영상으로 대응 → 2 Hz 격자 밖 프레임 724k 문항 제외, 일부는 다른 시각 영상)
  - 원본 영상의 로그는 로그별 listdir(전방 3캠)로 만든 stem 색인으로 찾는다(NFS 재귀 탐색 없음). 3캠 stem 이 모두 같은 로그·같은
    카메라 폴더에 있어야 사용. 3캠이 서로 다른 로그를 가리키면 즉시 실패.
  - 표본 단위 = NuPlanQA sample_token(= 원본 프레임). id="nuplanqa:<sample_token>:<chunk>", token=sample_token,
    scene_token=로그 이름(fold 단위 — fold_split_navsim.json 그대로), image_source="nuplan_raw_sensor_blobs",
    navsim_frame_token = 같은 영상을 가진 NAVSIM 2 Hz 프레임 token(있으면, 없으면 None).
  - fold 배정: 로그 단위, 기존 분할 파일 그대로. navtrain 이 아닌 로그·분할에 없는 로그·navtest 로그·제외 6 로그는 이유별로 센다.
  - ReCogDrive 표본: 종전 산출물(--old_dir)의 레코드를 **그대로 복사**(재생성하지 않음; 같은 분할·같은 규칙이고 영상은 NAVSIM
    현재 프레임 100% 실재라 재생성해도 같다). 종전 manifest 의 ReCogDrive 수와 일치 검사.
  - NuPlanQA-Eval 1,801 문항 전부 원본 영상으로(종전 122 개는 OpenScene 영상이었음).

실행(내려받은 JSON/pkl 은 신뢰하지 않는 데이터 — 반드시 python -I):
  CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX python -I tools/vla_teacher/make_qa_data_navsim_nuplanimg.py \
    --index_cache <navsim_frame_index.pkl> --raw_index_cache <nuplan_raw_front3_index.pkl> \
    --out /home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/teacher/navsim_nuplanimg_20261019
"""
import argparse
import collections
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import make_qa_data_navsim as M  # noqa: E402
import navsim_qa_common as N  # noqa: E402
import qa_common as Q  # noqa: E402
import sft_common as S  # noqa: E402

RAW = N.NUPLAN_RAW_BLOBS
OLD_DIR = "/home1/irteam/data-vol1/nas_mirror/nuscenes_original_junhan/phd_thesis/navsim/teacher/navsim"
TAG = "# 수정(2026-10-19): 사용자 승인 — NAVSIM 사전 준비(CPU)"


# ---------------------------------------------------------------- nuPlan 원본 전방 3캠 색인(로그별 listdir)
def _list_log(log):
    """반환 (log, {cam: 정렬 uint64 stem 배열 또는 None}, 영상 아닌 파일 목록).
    NAS 에는 과거 복사 도중 남은 임시 사본(예: <stem>.jpg.fF7a6A5e, 원본 .jpg 와 크기 같음)이 섞여 있다 — 영상 목록에서는 빼고
    이름을 전부 기록한다(원본 .jpg 가 없으면 그 프레임은 '영상 없음' 으로 셈)."""
    out, extra = {}, []
    for c in N.QA_CAMS:
        try:
            names = os.listdir(os.path.join(RAW, log, c))
        except FileNotFoundError:
            out[c] = None
            continue
        st = []
        for fn in names:
            if not fn.endswith(".jpg") or len(fn) != 20:
                extra.append(f"{log}/{c}/{fn}")
                continue
            st.append(int(fn[:16], 16))
        out[c] = np.unique(np.asarray(st, np.uint64))
    return log, out, extra


def build_raw_index(cache, workers):
    if cache and os.path.exists(cache):
        return N.safe_load(cache)
    t0 = time.time()
    logs = sorted(os.listdir(RAW))
    print(f"[raw] 로그 {len(logs)} ({time.time() - t0:.0f}s)", flush=True)
    res, extras = {}, []
    with Pool(workers) as pool:
        for k, (lg, d, ex) in enumerate(pool.imap_unordered(_list_log, logs, chunksize=1)):
            res[lg] = d
            extras += ex
            if (k + 1) % 100 == 0:
                print(f"[raw] {k + 1}/{len(logs)} ({time.time() - t0:.0f}s)", flush=True)
    idx = dict(logs=res, root=RAW, cams=list(N.QA_CAMS), created=time.strftime("%Y-%m-%d %H:%M:%S"),
               non_image_files=sorted(extras))
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache + ".tmp", "wb") as f:
            pickle.dump(idx, f, protocol=4)
        os.replace(cache + ".tmp", cache)
    print(f"[raw] 색인 완료 {len(res)} 로그, 영상 {sum(len(v) for d in res.values() for v in d.values() if v is not None)} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return idx


class RawLookup:
    """stem(hex16) → 로그. 카메라별 전역 정렬 배열 + searchsorted."""

    def __init__(self, ridx):
        self.logs = sorted(ridx["logs"])
        self.per = ridx["logs"]
        self.glob = {}
        for c in N.QA_CAMS:
            keys, ids = [], []
            for i, lg in enumerate(self.logs):
                a = self.per[lg][c]
                if a is None or not len(a):
                    continue
                keys.append(a); ids.append(np.full(len(a), i, np.int32))
            k = np.concatenate(keys); v = np.concatenate(ids)
            o = np.argsort(k, kind="stable")
            k, v = k[o], v[o]
            dup = np.nonzero(k[1:] == k[:-1])[0]
            if len(dup):
                raise RuntimeError(f"{c}: 같은 stem 이 여러 로그에 {len(dup)}개 (예 {int(k[dup[0]]):016x})")
            self.glob[c] = (k, v)

    def log_of(self, cam, stem):
        k, v = self.glob[cam]
        x = np.uint64(int(stem, 16))
        j = np.searchsorted(k, x)
        if j < len(k) and k[j] == x:
            return self.logs[v[j]]
        return None

    def has(self, log, cam, stem):
        a = self.per.get(log, {}).get(cam)
        if a is None:
            return False
        x = np.uint64(int(stem, 16))
        j = np.searchsorted(a, x)
        return j < len(a) and a[j] == x

    def resolve(self, cams):
        """{cam: stem} → (log, n_front_found, paths 또는 None). 전방 3캠 stem 이 다른 로그를 가리키면 예외."""
        found = {c: self.log_of(c, cams[c]) for c in N.QA_CAMS if c in cams}
        logs = {lg for lg in found.values() if lg is not None}
        if len(logs) > 1:
            raise RuntimeError(f"전방 3캠 stem 이 서로 다른 원본 로그를 가리킨다: {found}")
        if not logs:
            return None, 0, None
        lg = logs.pop()
        n = sum(1 for c in N.QA_CAMS if c in cams and self.has(lg, c, cams[c]))
        paths = [os.path.join(RAW, lg, c, cams[c] + ".jpg") for c in N.QA_CAMS] if n == len(N.QA_CAMS) else None
        return lg, n, paths


def image_sizes(paths):
    from PIL import Image
    c = collections.Counter()
    for p in paths:
        c[Image.open(p).size] += 1
    return {f"{w}x{h}": n for (w, h), n in c.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_root", default=M.QA_ROOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--old_dir", default=OLD_DIR, help="종전 산출물(ReCogDrive 레코드를 그대로 복사)")
    ap.add_argument("--fold_file", default=N.FOLD_FILE_NAVSIM)
    ap.add_argument("--index_cache", required=True, help="NAVSIM 프레임 색인 캐시(make_fold_split_navsim.py 와 같은 파일)")
    ap.add_argument("--raw_index_cache", required=True, help="nuPlan 원본 전방 3캠 색인 캐시(없으면 생성)")
    ap.add_argument("--raw_index_only", action="store_true")
    ap.add_argument("--max_seq_tokens", type=int, default=6144)
    ap.add_argument("--pixels", type=int, default=N.NAVSIM_PIXELS)
    ap.add_argument("--size_probe", type=int, default=300, help="원본 영상 크기 확인 표본 수(카메라별)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    ridx = build_raw_index(a.raw_index_cache, a.workers)
    if a.raw_index_only:
        return
    out = Path(a.out)
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
    idx = N.build_frame_index(a.index_cache, 4)
    tok_map, stem_map = N.token_maps(idx)
    rl = RawLookup(ridx)
    raw_logs = set(rl.logs)
    print(f"[index] NAVSIM 프레임 {len(tok_map)}, 원본 로그 {len(raw_logs)} (navtest 로그와 겹침 {len(raw_logs & navtest_logs)}, "
          f"navtrain 로그 중 원본 있음 {len(navtrain_logs & raw_logs)}/{len(navtrain_logs)}) ({time.time() - t0:.0f}s)", flush=True)

    counts = collections.Counter()
    frame_counts = collections.Counter()
    excluded_examples = collections.defaultdict(list)
    groups = collections.OrderedDict()

    def navsim_frame_of(cams):
        hit, n = M.map_by_stems(cams, stem_map)
        return hit, n

    # ---- NuPlanQA train
    meta = M.load_nuplanqa_meta(raw / "nuplanqa/nuplan_metadata.pkl")
    nq_train = json.load(open(raw / "nuplanqa/nuplanqa_train.json"))
    nq_eval = json.load(open(raw / "nuplanqa/nuplanqa_eval_test.json"))
    eval_tokens = {x["sample_token"] for x in nq_eval}
    route_cache = {}

    def route(st):
        """sample_token → (reason, log, paths, navsim_hit). reason None = 사용."""
        if st in route_cache:
            return route_cache[st]
        cams = meta[st]["cams"]
        lg, n, paths = rl.resolve(cams)
        nav, _ = navsim_frame_of(cams)
        if nav is not None and lg is not None and nav[1] != lg:
            raise RuntimeError(f"{st}: 원본 로그 {lg} != NAVSIM 로그 {nav[1]}")
        if lg is None:
            if nav is not None and (nav[0] == "test" or nav[1] in navtest_logs):
                r = ("navtest_log", nav[1], None, nav)
            else:
                r = ("not_found_in_nuplan_raw", None, None, nav)
        elif lg in navtest_logs or lg in val_logs or (nav is not None and nav[2] in navtest_tokens):
            r = ("navtest_log", lg, None, nav)
        elif lg not in navtrain_logs:
            r = ("trainval_log_not_in_navtrain", lg, None, nav)
        elif lg in excl_logs:
            r = ("teacher_exclude_logs(NuPlanQA-Eval)", lg, None, nav)
        elif lg not in part_of:
            r = ("log_not_in_fold_split", lg, None, nav)
        elif paths is None:
            r = ("front3_images_missing_on_disk", lg, None, nav)
        else:
            r = (None, lg, paths, nav)
        route_cache[st] = r
        frame_counts[r[0] or "used"] += 1
        return r

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
            excluded_examples[("nuplanqa", "sample_token_not_in_metadata")].append(f"train#{i}")
            continue
        why, lg, paths, nav = route(st)
        if why is not None:
            counts[("nuplanqa", f"excluded:{why}")] += 1
            excluded_examples[("nuplanqa", why)].append(f"train#{i} -> {lg}/{st}")
            continue
        counts[("nuplanqa", "used_qa")] += 1
        g = groups.setdefault(st, dict(log=lg, paths=paths, nav=nav, turns=[]))
        g["turns"].append(dict(q=qa["q"], a=qa["a"], type=x["task"], src=f"train#{i}"))
        if (i + 1) % 200000 == 0:
            print(f"[nuplanqa] {i + 1}/{len(nq_train)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[nuplanqa] 프레임 그룹 {len(groups)}, 프레임 이유별 {dict(frame_counts)} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 원본 영상 크기(카메라별 표본) → 캠당 토큰
    rng = np.random.RandomState(0)
    keys = list(groups)
    pick = [keys[j] for j in rng.choice(len(keys), min(a.size_probe, len(keys)), replace=False)]
    sizes = {c: image_sizes([groups[k]["paths"][ci] for k in pick]) for ci, c in enumerate(N.QA_CAMS)}
    print(f"[size] 원본 영상 크기 {sizes}", flush=True)
    if any(set(v) != {f"{N.NAVSIM_IMAGE_HW[1]}x{N.NAVSIM_IMAGE_HW[0]}"} for v in sizes.values()):
        raise RuntimeError(f"원본 영상 크기가 {N.NAVSIM_IMAGE_HW} 가 아니다: {sizes} — 캠당 토큰 재계산 필요")

    # ---- 표본(chunk)·길이 — NuPlanQA
    proc = S.make_processor(S.BASE_QWEN, a.pixels)
    tok = proc.tokenizer
    tpc, hw_resized = M.tokens_per_cam(a.pixels)
    n_img = len(N.QA_CAMS)
    recs = {(f, p): [] for f in S.FOLDS for p in ("train", "holdout")}
    lens = collections.defaultdict(list)
    n_chunks_split = collections.Counter()
    for k, (st, g) in enumerate(groups.items()):
        f, part = part_of[g["log"]]
        chunks = M.chunk_turns(tok, g["turns"], a.max_seq_tokens, n_img, tpc)
        if sum(len(c) for c in chunks) != len(g["turns"]):
            raise AssertionError("chunk QA 개수 불일치")
        if len(chunks) > 1:
            n_chunks_split["nuplanqa"] += 1
        for ci, c in enumerate(chunks):
            r = dict(id=f"nuplanqa:{st}:{ci}", dataset="nuplanqa", token=st, scene_token=g["log"], fold=f, part=part,
                     chunk=ci, n_chunks=len(chunks), cams=g["paths"], turns=c, nuplanqa_sample_tokens=[st],
                     image_source="nuplan_raw_sensor_blobs", navsim_frame_token=g["nav"][2] if g["nav"] else None)
            L, n_sup = M.text_len(tok, c, n_img, tpc)
            if L > a.max_seq_tokens:
                raise AssertionError(f"{r['id']}: {L} > {a.max_seq_tokens}")
            r["n_tokens"], r["n_answer_tokens"] = L, n_sup
            recs[(f, part)].append(r)
            lens[("nuplanqa", f, part)].append(L)
        if (k + 1) % 20000 == 0:
            print(f"[len] {k + 1}/{len(groups)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[len] NuPlanQA 완료 ({time.time() - t0:.0f}s)", flush=True)

    # ---- ReCogDrive: 종전 레코드 그대로 복사
    old_dir = Path(a.old_dir)
    old_man = json.load(open(old_dir / "qa_manifest.json"))
    if old_man["fold_sha256"] != fd["sha256"]:
        raise RuntimeError("종전 산출물의 분할 해시가 지금 분할 파일과 다르다 — ReCogDrive 레코드 재사용 불가")
    if old_man["pixels"] != a.pixels or old_man["max_seq_tokens"] != a.max_seq_tokens or old_man["tokens_per_cam"] != tpc:
        raise RuntimeError("종전 산출물의 pixels/max_seq_tokens/tokens_per_cam 이 다르다")
    rc_qa = 0
    for f in S.FOLDS:
        for p in ("train", "holdout"):
            old, op = Q.load_qa(f, p, str(old_dir))
            if S.file_sha256(op) != old_man["files"][f"fold{f}_{p}"]["sha256"]:
                raise RuntimeError(f"{op}: 종전 manifest 해시 불일치")
            n = 0
            for r in old:
                if r["dataset"] != "recogdrive_navsim":
                    continue
                if r["fold"] != f or r["part"] != p or part_of.get(r["scene_token"]) != (f, p):
                    raise RuntimeError(f"{r['id']}: 종전 레코드 fold/part 불일치")
                recs[(f, p)].append(r); lens[("recogdrive_navsim", f, p)].append(r["n_tokens"])
                rc_qa += len(r["turns"]); n += 1
            st_old = old_man["stats"][f"recogdrive_navsim/fold{f}_{p}"]
            if n != st_old["samples"]:
                raise RuntimeError(f"ReCogDrive fold{f}_{p} 표본 {n} != 종전 manifest {st_old['samples']}")
    if rc_qa != old_man["counts"]["recogdrive_navsim|used_qa"]:
        raise RuntimeError(f"ReCogDrive QA {rc_qa} != 종전 {old_man['counts']['recogdrive_navsim|used_qa']}")
    for k, v in old_man["counts"].items():
        if k.startswith("recogdrive_navsim|"):
            counts[tuple(k.split("|", 1))] = v
    print(f"[recogdrive] 종전 레코드 복사 QA {rc_qa} ({time.time() - t0:.0f}s)", flush=True)

    # ---- NuPlanQA-Eval(1,801 전부, 원본 영상)
    ev_recs, ev_counts = [], collections.Counter()
    for i, x in enumerate(nq_eval):
        st = x["sample_token"]
        qa = x["QA"][0]
        if len(x["QA"]) != 1 or not qa["q"].strip() or not qa["a"].strip():
            raise RuntimeError(f"NuPlanQA eval#{i}: QA 형식/빈 답")
        r = dict(id=f"nuplanqa_eval:{i}", dataset="nuplanqa_eval", sample_token=st, task=x["task"], split=x.get("split"),
                 turns=[dict(q=qa["q"], a=qa["a"], type=x["task"], src=f"eval_test#{i}")],
                 navsim=None, cams=None, image_source=None, nuplan_log=None)
        if st not in meta:
            ev_counts["no_metadata"] += 1
        else:
            cams = meta[st]["cams"]
            lg, n, paths = rl.resolve(cams)
            hit, _ = navsim_frame_of(cams)
            if hit is not None:
                split, log, ftok = hit
                if lg is not None and log != lg:
                    raise RuntimeError(f"eval#{i}: 원본 로그 {lg} != NAVSIM 로그 {log}")
                r["navsim"] = dict(split=split, log=log, frame_token=ftok, navtrain_log=log in navtrain_logs,
                                   navtest_log=log in navtest_logs, teacher_exclude_log=log in excl_logs)
            if paths is not None:
                r["cams"], r["image_source"], r["nuplan_log"] = paths, "nuplan_raw_sensor_blobs", lg
                ev_counts["in_teacher_exclude_logs" if lg in excl_logs else "NOT_in_teacher_exclude_logs"] += 1
            ev_counts["in_navsim_frames" if hit is not None else "not_in_navsim_frames"] += 1
        ev_counts[f"images:{r['image_source'] or 'not_found'}"] += 1
        ev_recs.append(r)
    if ev_counts["images:not_found"]:
        raise RuntimeError(f"NuPlanQA-Eval 원본 영상 못 찾음 {ev_counts['images:not_found']}")

    # ---- 저장
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for (f, p), rs in recs.items():
        rs.sort(key=lambda r: r["id"])
        fp = Q.qa_file(f, p, str(out))
        with open(fp, "w") as fh:
            for r in rs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[f"fold{f}_{p}"] = dict(path=str(fp), sha256=S.file_sha256(fp), n_samples=len(rs))
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
                for ds, v in M.SOURCES.items() if ds == "nuplanqa"}
    src_info["recogdrive_navsim"] = dict(old_man["sources"]["recogdrive_navsim"],
                                         reused_from=dict(dir=str(old_dir), manifest_sha256=S.file_sha256(old_dir / "qa_manifest.json")))
    manifest = dict(
        created="2026-10-19", tag=TAG, benchmark="navsim", variant="nuplanqa_original_nuplan_images",
        sources=src_info, raw_dir=str(raw), nuplan_raw_blobs=RAW, fold_file=fpath, fold_sha256=fd["sha256"],
        max_seq_tokens=a.max_seq_tokens, pixels=a.pixels, n_cams=n_img, cams=list(N.QA_CAMS), tokens_per_cam=tpc,
        resized_hw=list(hw_resized), image_hw_assumed=list(N.NAVSIM_IMAGE_HW), nuplan_raw_image_sizes_probe=sizes,
        assignment=dict(teacher_A=dict(train="train.A − holdout.A", holdout="holdout.A"),
                        teacher_B=dict(train="train.B − holdout.B", holdout="holdout.B"),
                        excluded="navtest 로그 전체 + teacher_exclude_logs 6 로그(NuPlanQA-Eval) + navtrain 밖 로그"),
        counts={"|".join(k): v for k, v in sorted(counts.items())},
        nuplanqa_frames_by_reason=dict(frame_counts),
        excluded_examples={"|".join(k): v[:5] for k, v in excluded_examples.items()},
        frames_split_into_chunks=dict(n_chunks_split), files=files, stats=stats,
        excluded_corrupt_answers=old_man.get("excluded_corrupt_answers"),
        eval_file=dict(path=str(evp), sha256=S.file_sha256(evp), n=len(ev_recs), counts=dict(ev_counts)),
        raw_index=dict(cache=a.raw_index_cache, logs=len(raw_logs), non_image_files=len(ridx.get("non_image_files", [])),
                       non_image_files_examples=ridx.get("non_image_files", [])[:10]), seconds=round(time.time() - t0, 1))
    with open(out / "qa_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    print(json.dumps(dict(counts=manifest["counts"], frames=frame_counts, chunks=manifest["frames_split_into_chunks"],
                          files=files, eval=manifest["eval_file"], tpc=tpc, sizes=sizes), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
