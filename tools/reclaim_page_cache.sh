#!/bin/bash
# 수정(2026-08-21): 학습을 중단하지 않고 페이지 캐시를 주기적으로 반납한다.
#
#   [배경] 학습 서버의 main memory 점유가 과다하다는 지적. 조사 결과 원인은 프로세스가
#   아니라 페이지 캐시였다. 컨테이너 cgroup 68.8 GB 중
#       anon(실제 프로세스 메모리)  16.4 GB   <- 학습 본체 + 워커 8개 + IDE
#       file(페이지 캐시)           49.4 GB   <- 이 중 inactive_file 47.7 GB
#   VLA 학습이 매 표본마다 3.4 MB 짜리 Qwen 토큰 캐시를 읽는데, 전체가 114 GB 여서
#   페이지 캐시에 다 들어가지 않는다. 즉 캐시가 재사용되지 못하고 계속 밀려나기만 하므로
#   붙잡아 둘 이득이 없다.
#
#   [왜 이 방법인가]
#   - 정리할 레거시/고아 프로세스는 없었다(좀비 0, 중복 학습 0). 죽일 대상이 애초에 없다.
#   - /sys/fs/cgroup/memory.reclaim 은 이 컨테이너에서 읽기 전용이라 쓸 수 없다.
#   - posix_fadvise(POSIX_FADV_DONTNEED)는 권한이 필요 없고 해당 파일의 캐시만 반납한다.
#     데이터는 디스크에 그대로 있으므로 필요하면 다시 읽힌다 — 손실이 없다.
#   실측: 34,149개 파일에 30초, 페이지 캐시 50.3 -> 17.2 GB (33 GB 회수),
#         학습은 무중단이고 data_time 0.045초로 I/O 지연 변화 없음.
#
#   사용법: nohup bash tools/reclaim_page_cache.sh [주기초=1200] &
set -u
cd /workspace/src/HiP-AD
PY=/opt/anaconda3/envs/hipad_bw/bin/python
INTERVAL=${1:-1200}
LOG=results/page_cache_reclaim.log

while true; do
  $PY - <<'PYEOF' >> "$LOG" 2>&1
import os, glob, time, datetime
def cg(k):
    return int([l.split()[1] for l in open('/sys/fs/cgroup/memory.stat') if l.startswith(k + ' ')][0])
before = cg('file')
n = 0
# Qwen 토큰 캐시가 점유의 대부분이다. nuScenes 원본 영상은 여러 표본이 공유하므로
# 캐시 적중률이 높아 그대로 둔다.
for p in glob.glob('data/nuscenes/qwen/*.pt'):
    try:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += 1
        finally:
            os.close(fd)
    except Exception:
        pass
after = cg('file')
print(f"[{datetime.datetime.now():%F %T}] {n}개 처리, page cache "
      f"{before/2**30:.2f} -> {after/2**30:.2f} GB (회수 {(before-after)/2**30:.2f} GB), "
      f"anon {cg('anon')/2**30:.2f} GB")
PYEOF
  sleep "$INTERVAL"
done
