"""수정(2026-10-26): 사용자 결정 "클라우드 서버로 옮긴다, 학습·평가 자료 전체를 HF 에서 받는다" + "PRO 구독 후 올림" —
   교사 토큰 캐시(data/nuscenes/laq_cache, 34,149 표본 × 7 폴더 ≈ 249 GB)를 비공개 데이터 저장소 laq-ad-nuscenes-v2 의
   teacher/laq_cache_shards/ 에 묶음(tar)으로 올린다.
   묶는 이유: 파일이 약 24만 개라 HF 권장 한도(저장소당 10만 파일, 폴더당 1만 파일)를 넘는다. bf16 값이라 압축 이득이 없어 비압축 tar.
   - 표본 토큰을 정렬해 SHARD 개씩 자른다. 묶음 i = 그 표본들의 <폴더>/<token>.pt 7 개씩(폴더 구조 그대로).
   - meta.tar = 폴더별 _meta.json, _index/, README.md.
   - 로컬 여유(≈160 GB)가 캐시보다 작아 묶음 하나씩 만들고 → sha256 → 올림 → 지운다. MANIFEST.json 에 묶음별 sha256·표본 수를 남긴다.
   - 다시 실행하면 이미 올라간 묶음(저장소 sha256 일치)은 건너뛴다(중단 뒤 이어 올리기).
   풀기: tools/hf_download_laq_cache.py (클라우드).
   사용: python tools/hf_upload_laq_cache.py --stage <로컬 대기 폴더> [--shard 500] [--only 0]
"""
import argparse, hashlib, json, os, sys, tarfile, time
from huggingface_hub import HfApi

CACHE = "/home1/irteam/data-vol1/src/HiP-AD/data/nuscenes/laq_cache"
FOLDERS = ["context", "detection", "tracking", "motion", "map", "plan", "action"]
PREFIX = "teacher/laq_cache_shards"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(16 << 20)):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--shard", type=int, default=500)
    ap.add_argument("--only", type=int, default=None, help="이 번호 묶음만(시험용)")
    ap.add_argument("--repo", default=None)
    a = ap.parse_args()
    api = HfApi()
    repo = a.repo or f"{api.whoami()['name']}/laq-ad-nuscenes-v2"
    os.makedirs(a.stage, exist_ok=True)

    tokens = sorted(n[:-3] for n in os.listdir(f"{CACHE}/context") if n.endswith(".pt"))
    for f in FOLDERS:   # 모든 폴더에 같은 표본이 있는지 먼저 확인(빠진 파일을 조용히 넘기지 않는다)
        have = {n[:-3] for n in os.listdir(f"{CACHE}/{f}") if n.endswith(".pt")}
        if have != set(tokens):
            sys.exit(f"폴더 {f} 표본 불일치: context {len(tokens)} vs {len(have)}")
    shards = [tokens[i:i + a.shard] for i in range(0, len(tokens), a.shard)]
    print(f"표본 {len(tokens)}, 묶음 {len(shards)} (묶음당 {a.shard})", flush=True)

    remote = {}
    try:
        info = api.dataset_info(repo, files_metadata=True)
        remote = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
    except Exception as e:
        sys.exit(f"저장소 확인 실패: {e}")

    man_path = f"{a.stage}/MANIFEST.json"
    man = json.load(open(man_path)) if os.path.exists(man_path) else {"shard_size": a.shard, "num_samples": len(tokens),
                                                                     "folders": FOLDERS, "shards": {}}

    def put(local, name, extra):
        h = sha256(local)
        rp = f"{PREFIX}/{name}"
        if remote.get(rp) == h:
            print(f"  {name}: 이미 있음(sha 일치) — 건너뜀", flush=True)
        else:
            t0 = time.time()
            api.upload_file(path_or_fileobj=local, path_in_repo=rp, repo_id=repo, repo_type="dataset",
                            commit_message=f"laq_cache {name}")
            print(f"  {name}: 올림 {os.path.getsize(local) / 1e9:.2f} GB, {time.time() - t0:.0f}s", flush=True)
        man["shards"][name] = dict(sha256=h, bytes=os.path.getsize(local), **extra)
        json.dump(man, open(man_path, "w"), indent=1)
        os.remove(local)

    if a.only is None:
        mt = f"{a.stage}/meta.tar"
        with tarfile.open(mt, "w") as t:
            for f in FOLDERS:
                t.add(f"{CACHE}/{f}/_meta.json", arcname=f"{f}/_meta.json")
            t.add(f"{CACHE}/_index", arcname="_index")
            t.add(f"{CACHE}/README.md", arcname="README.md")
        put(mt, "meta.tar", dict(samples=0))

    for i, toks in enumerate(shards):
        if a.only is not None and i != a.only:
            continue
        name = f"shard_{i:04d}.tar"
        if man["shards"].get(name) and remote.get(f"{PREFIX}/{name}") == man["shards"][name]["sha256"]:
            print(f"  {name}: 이미 있음 — 건너뜀", flush=True)
            continue
        local = f"{a.stage}/{name}"
        with tarfile.open(local, "w") as t:
            for tok in toks:
                for f in FOLDERS:
                    t.add(f"{CACHE}/{f}/{tok}.pt", arcname=f"{f}/{tok}.pt")
        put(local, name, dict(samples=len(toks), first=toks[0], last=toks[-1]))

    if a.only is None:
        api.upload_file(path_or_fileobj=man_path, path_in_repo=f"{PREFIX}/MANIFEST.json", repo_id=repo, repo_type="dataset",
                        commit_message="laq_cache MANIFEST")
        print("완료:", repo, PREFIX, flush=True)


if __name__ == "__main__":
    main()
