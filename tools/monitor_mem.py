#!/usr/bin/env python3
#수정: RAM/swap 폭발 감시용 경량 모니터 — 학습과 병행 실행(사용자 요청 2026-07-15).
#      전체 RAM/swap 사용량과 학습 프로세스(python train.py + DataLoader 워커)의 합산 RSS를
#      주기적으로 로그. 임계치 초과 시 경고 표시(학습을 죽이지는 않음 — 관찰 전용).
"""
사용법:
    # 학습 시작 후, 백그라운드로 감시 (5초 간격, 로그 파일 지정)
    python tools/monitor_mem.py --interval 5 --log work_dirs/<cfg>/mem_monitor.log

    # 특정 프로세스명만 합산 (기본: 'train.py' 매칭)
    python tools/monitor_mem.py --match train.py --warn-ram 110 --warn-swap 400

컬럼: time  ram_used/total(GB)  ram_avail(GB)  swap_used/total(MB)  train_rss(GB)  nproc
"""
import argparse
import os
import time


def read_meminfo():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            info[k.strip()] = int(v.strip().split()[0])  # kB
    total = info["MemTotal"] / 1024 / 1024
    avail = info["MemAvailable"] / 1024 / 1024
    used = total - avail
    swap_total = info["SwapTotal"] / 1024
    swap_free = info["SwapFree"] / 1024
    swap_used = swap_total - swap_free
    # 수정: 2026-07-23 stage2 사망(iter 57750) 원인규명용 계측.
    #       당시 train_rss 는 107GB 로 평평했는데 시스템 used 만 100초 만에 +61GB 폭증했다.
    #       = 할당 주체가 학습 프로세스의 RSS 가 아니라는 뜻이므로, RSS 에 안 잡히는
    #       Shmem(/dev/shm — DataLoader 워커간 텐서 전달용 tmpfs, 상한 63GB) 과
    #       회수불가 슬랩(SUnreclaim)을 같이 찍어 진짜 소비처를 특정한다.
    shmem = info.get("Shmem", 0) / 1024 / 1024
    sunreclaim = info.get("SUnreclaim", 0) / 1024 / 1024
    return total, used, avail, swap_total, swap_used, shmem, sunreclaim  # GB,GB,GB,MB,MB,GB,GB


def train_rss(match):
    """match 문자열을 cmdline 에 포함하는 프로세스들의 RSS 합(GB), 프로세스 수."""
    total_kb = 0
    n = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode(errors="ignore")
            if match not in cmd:
                continue
            with open(f"/proc/{pid}/statm") as f:
                rss_pages = int(f.read().split()[1])
            total_kb += rss_pages * (os.sysconf("SC_PAGE_SIZE") // 1024)
            n += 1
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total_kb / 1024 / 1024, n  # GB, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--match", default="train.py")
    ap.add_argument("--log", default=None)
    ap.add_argument("--warn-ram", type=float, default=110.0, help="RAM used(GB) 경고 임계치")
    ap.add_argument("--warn-swap", type=float, default=200.0, help="swap used(MB) 경고 임계치")
    args = ap.parse_args()

    logf = open(args.log, "a", buffering=1) if args.log else None

    def emit(s):
        print(s, flush=True)
        if logf:
            logf.write(s + "\n")

    emit(f"# monitor start match={args.match!r} interval={args.interval}s "
         f"warn_ram={args.warn_ram}GB warn_swap={args.warn_swap}MB")
    emit("# time              ram_used/tot(GB)  avail(GB)  swap_used/tot(MB)  "
         "train_rss(GB)  shmem(GB)  sunrecl(GB)  nproc  flags")
    try:
        while True:
            total, used, avail, swap_total, swap_used, shmem, sunreclaim = read_meminfo()
            trss, nproc = train_rss(args.match)
            flags = []
            if used >= args.warn_ram:
                flags.append("RAM_HIGH")
            if swap_used >= args.warn_swap:
                flags.append("SWAP_HIGH")
            # 수정: RSS 로 설명되지 않는 증가분을 즉시 보이게 한다(사망원인 후보 특정용).
            if shmem >= 10.0:
                flags.append("SHMEM_HIGH")
            ts = time.strftime("%H:%M:%S")
            emit(f"{ts}  {used:6.1f}/{total:5.1f}      {avail:6.1f}    "
                 f"{swap_used:6.0f}/{swap_total:5.0f}        {trss:6.1f}     "
                 f"{shmem:6.1f}     {sunreclaim:6.1f}     "
                 f"{nproc:3d}   {','.join(flags)}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        emit("# monitor stopped")


if __name__ == "__main__":
    main()
