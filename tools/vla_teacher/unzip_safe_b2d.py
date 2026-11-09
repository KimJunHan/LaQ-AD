# 수정(2026-10-16): B2D 교사 QA 준비 — Chat-B2D zip 을 NAS 새 폴더에 안전하게 푸는 도구(재개 가능). 실행: python -I tools/vla_teacher/unzip_safe_b2d.py <zip> <새 폴더>
# 신뢰하지 않는 zip 을 지정 폴더에 안전하게 풀기(절대경로·..·symlink 거부). NFS 지연 때문에 스레드 8 개로 병렬 쓰기.
# 이 스크립트(이전 단일 스레드판)가 직접 풀던 폴더의 재개만: 이미 있는 파일은 크기·CRC 가 zip 과 같으면 두고,
# 다르면 truncated(이전 프로세스 중단 시 쓰던 파일)인 경우만 목록에 남기고 마지막에 다시 씀 — 그 외 다르면 중단.
import os, sys, zipfile, time, zlib, threading
from concurrent.futures import ThreadPoolExecutor
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
root = os.path.realpath(dst)
zf = zipfile.ZipFile(src)
infos = zf.infolist()
for i in infos:
    p = os.path.realpath(os.path.join(root, i.filename))
    if not (p == root or p.startswith(root + os.sep)):
        sys.exit(f"[거부] 위험 경로 {i.filename}")
    if (i.external_attr >> 16) & 0o170000 == 0o120000:
        sys.exit(f"[거부] symlink {i.filename}")
dirs = {os.path.dirname(os.path.join(root, i.filename)) for i in infos if not i.is_dir()}
for d in sorted(dirs):
    os.makedirs(d, exist_ok=True)
lock = threading.Lock(); st = {"new": 0, "kept": 0, "rewritten_truncated": 0}; t = time.time()
local = threading.local()
def zfh():
    if not hasattr(local, "z"):
        local.z = zipfile.ZipFile(src)
    return local.z
def one(i):
    p = os.path.join(root, i.filename)
    data = zfh().read(i.filename)
    if os.path.exists(p):
        old = open(p, "rb").read()
        if len(old) == i.file_size and (zlib.crc32(old) & 0xffffffff) == i.CRC:
            k = "kept"
        elif len(old) < i.file_size and data[:len(old)] == old:
            with open(p, "wb") as fo:
                fo.write(data)
            k = "rewritten_truncated"
            print("[truncated→rewrite]", p, len(old), i.file_size, flush=True)
        else:
            raise RuntimeError(f"[중단] 기존 파일이 zip 과 다름(덮어쓰지 않음): {p}")
    else:
        with open(p, "xb") as fo:
            fo.write(data)
        k = "new"
    with lock:
        st[k] += 1
        n = sum(st.values())
        if n % 20000 == 0:
            print(n, st, round(time.time() - t), flush=True)
files = [i for i in infos if not i.is_dir()]
with ThreadPoolExecutor(8) as ex:
    list(ex.map(one, files))
print("done", src, len(files), st, round(time.time() - t), flush=True)
