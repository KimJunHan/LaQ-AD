"""수정(2026-10-26): 클라우드 이전 — tools/hf_upload_laq_cache.py 로 올린 교사 토큰 캐시 묶음을 내려받아 원래 구조
   <out>/{context,detection,tracking,motion,map,plan,action}/<token>.pt, _index/, README.md, 폴더별 _meta.json 으로 푼다.
   묶음마다 MANIFEST.json 의 sha256 과 대조한 뒤 풀고, 다 풀면 표본 수(폴더별 34,149)와 positions_sha256 을 검사한다.
   이미 다 풀린 묶음은 건너뛴다(<out>/.done 기록). 사용: python tools/hf_download_laq_cache.py --out data/nuscenes/laq_cache
"""
import argparse, hashlib, json, os, sys, tarfile
from huggingface_hub import HfApi, hf_hub_download

PREFIX = "teacher/laq_cache_shards"
FOLDERS = ["context", "detection", "tracking", "motion", "map", "plan", "action"]
POS_SHA = "bb44d62d06c6176216aa52a90a44c9f2f5b3a3f2d52c9ff84d16707fec331fca"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(16 << 20)):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--tmp", default=None, help="묶음 임시 저장 폴더(기본 <out>/_dl)")
    a = ap.parse_args()
    api = HfApi()
    repo = a.repo or f"{api.whoami()['name']}/laq-ad-nuscenes-v2"
    tmp = a.tmp or f"{a.out}/_dl"
    os.makedirs(tmp, exist_ok=True)
    man = json.load(open(hf_hub_download(repo, f"{PREFIX}/MANIFEST.json", repo_type="dataset", local_dir=tmp)))
    done_p = f"{a.out}/.done"
    done = set(open(done_p).read().split()) if os.path.exists(done_p) else set()
    for name, m in sorted(man["shards"].items()):
        if name in done:
            continue
        p = hf_hub_download(repo, f"{PREFIX}/{name}", repo_type="dataset", local_dir=tmp)
        if sha256(p) != m["sha256"]:
            sys.exit(f"{name} sha256 불일치 — 다시 받을 것")
        with tarfile.open(p) as t:
            t.extractall(a.out)
        os.remove(p)
        with open(done_p, "a") as f:
            f.write(name + "\n")
        print(f"{name} 풀기 완료", flush=True)
    n = man["num_samples"]
    for f in FOLDERS:
        k = sum(1 for x in os.listdir(f"{a.out}/{f}") if x.endswith(".pt"))
        meta = json.load(open(f"{a.out}/{f}/_meta.json"))
        ok = k == n and meta.get("positions_sha256") == POS_SHA
        print(f"{f}: {k}/{n} 표본, positions_sha256 {'일치' if ok else '불일치'}", flush=True)
        if not ok:
            sys.exit(1)
    print("교사 캐시 복원 완료:", a.out)


if __name__ == "__main__":
    main()
