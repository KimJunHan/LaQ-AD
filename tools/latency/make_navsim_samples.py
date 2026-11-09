# 수정(2026-11-05): NAVSIM 교사 지연 측정용 표본 목록 — 새 파일(기존 소스 수정 없음, CPU 전용).
"""표본 집합 두 개를 JSON 으로 만든다(teacher_latency_navsim.py / teacher_context_latency_navsim.py --samples).
  navtest       : 교사 T 배정 navtest part(teacher_A → val.B) GT 행, 파일 순서에서 고정 간격(len/pool) 앞 n 개. 캐시 없음(추출 중).
  navtrain_cache: 교사 T 색인(_index/teacher_T.s*of*.jsonl, 완료 표본)의 행, 고정 간격 앞 n 개 + GT navtrain 입력 레코드
                  → 캐시 토큰·은닉과 비교 가능.
입력 레코드는 extract_teacher_cache_navsim.strip_to_input(프롬프트 입력 필드만) 그대로. --warm: 표본 영상 파일을 한 번 읽어
페이지 캐시에 올린다(NAS 첫 읽기 지연을 CPU 전처리 시간에서 뺀다 — 배치 상황의 카메라 입력은 메모리에 있다).
사용: python tools/latency/make_navsim_samples.py --set navtest --teacher A --out results/latency/navsim_idle_20261105/samples_navtest_A.json
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/vla_teacher"))
import extract_teacher_cache_navsim as X  # noqa: E402
import navsim_sft_common as NS  # noqa: E402
import sft_common as S  # noqa: E402

CACHE = REPO / "data/navsim/laq_cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["navtest", "navtrain_cache"])
    ap.add_argument("--teacher", default="A", choices=["A", "B"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d, fp = S.load_fold_split(NS.NAVSIM_FOLD_FILE)
    parts = d["assignment"][f"teacher_{a.teacher}"]["generate"]
    want = [p for p in parts if p.startswith("val." if a.set == "navtest" else "train.")]
    assert len(want) == 1, want
    part = want[0]
    split, f = part.split(".")
    logs = set(d[split][f]["scenes"])
    gt = Path(NS.NAVSIM_GT_DIR) / NS.NAVSIM_SPLIT_FILES[split]
    if a.set == "navtest":
        rows = []
        for ln in open(gt):
            if ln.strip():
                r = json.loads(ln)
                if r["log_name"] in logs:
                    rows.append(r)
        step = max(1, len(rows) // max(a.pool, a.n))
        pick = rows[::step][:a.n]
        samples = [dict(entry=dict(token=r["token"], part=part, gen_ids=None), rec=X.strip_to_input(r)) for r in pick]
        n_pool = len(rows)
    else:
        idx = []
        for p in sorted(glob.glob(str(CACHE / "_index" / f"teacher_{a.teacher}.s*of*.jsonl"))):
            for ln in open(p):
                if ln.strip():
                    e = json.loads(ln)
                    if e["part"] == part and e.get("layout_ok"):
                        idx.append(e)
        step = max(1, len(idx) // max(a.pool, a.n))
        pick = idx[::step][:a.n]
        need = {e["token"]: e for e in pick}
        recs = {}
        for ln in open(gt):   # 2.2 GB — token 이 줄 앞에 있으므로 앞부분으로 거른 뒤에만 파싱
            if not ln.startswith('{"token": "'):
                continue
            t = ln[11:ln.index('"', 11)]
            if t in need:
                r = json.loads(ln)
                if r["log_name"] not in logs:
                    raise RuntimeError(f"{t}: 로그 {r['log_name']} 가 {part} 에 없다")
                recs[t] = X.strip_to_input(r)
        missing = [t for t in need if t not in recs]
        if missing:
            raise RuntimeError(f"GT 에 없는 색인 token {missing[:5]}")
        samples = [dict(entry=dict(token=e["token"], part=part, gen_ids=e["gen_ids"]), rec=recs[e["token"]]) for e in pick]
        n_pool = len(idx)
    if a.warm:
        nb = 0
        for s in samples:
            for p in S.cam_paths(s["rec"], True, "prev_first", NS.NAVSIM_NUM_CAMS):
                with open(p, "rb") as fh:
                    nb += len(fh.read())
        print(f"[warm] {len(samples)} 표본 영상 {nb / 2 ** 20:.1f} MiB 읽음", flush=True)
    doc = dict(set=a.set, teacher=a.teacher, part=part, gt_file=str(gt), fold_file=fp, n_pool=n_pool, stride=step,
               samples=samples)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(doc, open(a.out, "w"), ensure_ascii=False)
    print(f"[samples] {a.set} 교사 {a.teacher} part {part}: 후보 {n_pool}, 간격 {step}, {len(samples)} 개 -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
