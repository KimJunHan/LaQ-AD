# 수정(2026-11-09): N-A16(docs/pipeline_gap_navsim_b2d_20261109.md §1.6) — NAVSIM navtest 학생 프레임당 추론 시간(배치 1). 새 파일.
#   사용자 결정(11-09): 연산시간은 NAVSIM test 에서만 잰다(nuScenes 제외).
#   규약(기존 tools/latency/student_bench_navsim.py 와 같음): 한 프레임 = DataLoader 표본(CPU, 이미 전처리됨) → 결과 dict 까지,
#     앞뒤 torch.cuda.synchronize, 데이터 로딩·전처리 제외. navtest 로더 순서 그대로(장면 창 4 프레임: 첫 프레임 = 뱅크 초기화 프레임).
#     예열 --warmup 프레임 제외, 측정 --n 프레임의 median / p90 / mean. 정상(장면 안) / 초기화(장면 첫 프레임) / 전체를 따로 낸다.
#   변형(--variant): eager = 원본 모델 / graph_post_cw = student_bench_navsim.make_navsim_fast(CUDA Graph, 기존 도구 그대로 import).
#   정밀도(--precision): mixed = config 그대로(백본 fp16 auto_fp16, FPN bf16) / fp32.
#   PTQ(--quant): tools/quant/ptq_navsim.py 와 같은 navtrain 보정 + 가짜 양자화. **가짜 양자화는 연산을 원 정밀도로 하므로 배포 INT8 의
#     지연이 아니다**(양자화·역양자화 hook 만큼 오히려 느려진다). 실제 INT8 지연은 TensorRT 엔진이 필요(이 환경에 tensorrt 없음,
#     사용자 메모: 학생 TRT 는 1단계 확정 뒤 재확인). 결과 json 에 deploy_latency=False 로 표시한다.
#   부하 기록: 측정 동안 1 s 간격으로 물리 GPU 의 SM/메모리 클럭·최대 클럭·사용률·메모리·온도·전력·스로틀 사유와 연산 프로세스 목록,
#     측정 전후 CPU loadavg. 다른 프로세스가 보이면 foreign_pids 에 남긴다(--require-idle 이면 시작 전에 멈춘다).
#   교사 주기(docs/latency/teacher_segment_usage.md §4, navsim_idle_20261105.md §3): 교사를 장면 안 P 프레임마다 한 번 돌리면
#     프레임당 평균 ≈ 교사/P + 학생. --teacher-json 이름=경로(teacher_latency_*.json, summary.wall) 를 주면 P 별 추정 표를 붙인다
#     (교사 값은 별도 측정값을 읽기만 한다 — 같은 창에서 잰 값이 아니면 결과에 그 날짜를 남긴다).
#   GPU0 공용 — 이 스크립트는 반드시 flock /tmp/claude-500/gpu0_timing.lock 안에서 돌린다(scripts/navsim/run_latency_navsim_20261109.sh).
import argparse
import json
import os
import os.path as osp
import subprocess
import sys
import threading
import time
import types

import numpy as np

_HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, osp.join(osp.dirname(_HERE), "quant"))
import student_common as common  # noqa: E402  (저장소 경로·torch.load shim)
import torch  # noqa: E402

PHYS = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
Q_GPU = "clocks.sm,clocks.mem,clocks.max.sm,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks_throttle_reasons.active"


def smi(args):
    return subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True).stdout.strip()


def gpu_snapshot():
    v = [x.strip() for x in smi([f"--query-gpu={Q_GPU}", "--format=csv,noheader,nounits", "-i", PHYS]).split(",")]
    keys = Q_GPU.split(",")
    apps = [r.strip() for r in smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits", "-i", PHYS]).splitlines()
            if r.strip()]
    return dict(t=time.time(), **dict(zip(keys, v)), apps=apps)


class Monitor(threading.Thread):
    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period, self.rows, self.stop_ev = period, [], threading.Event()

    def run(self):
        while not self.stop_ev.is_set():
            self.rows.append(gpu_snapshot())
            self.stop_ev.wait(self.period)

    def summary(self, me):
        def num(k):
            a = []
            for r in self.rows:
                try:
                    a.append(float(r[k]))
                except (ValueError, KeyError):
                    pass
            return dict(min=min(a), median=float(np.median(a)), max=max(a)) if a else None
        foreign = sorted({int(a.split(",")[0]) for r in self.rows for a in r["apps"]} - {me})
        return dict(n_samples=len(self.rows), sm_clock_mhz=num("clocks.sm"), mem_clock_mhz=num("clocks.mem"),
                    max_sm_clock_mhz=num("clocks.max.sm"), util_pct=num("utilization.gpu"), mem_used_mib=num("memory.used"),
                    temp_c=num("temperature.gpu"), power_w=num("power.draw"),
                    throttle_reasons=sorted({r.get("clocks_throttle_reasons.active") for r in self.rows}),
                    foreign_pids=foreign, foreign_apps_examples=sorted({a for r in self.rows for a in r["apps"]
                                                                        if int(a.split(",")[0]) != me})[:10])


def build(a):
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmdet.datasets import build_dataset, replace_ImageToTensor
    from mmdet.datasets import build_dataloader as build_dataloader_origin
    import ptq_navsim as PN
    cfg = Config.fromfile(a.config)
    PN.F.import_plugin(cfg)
    if a.precision == "fp32":
        cfg.fp16 = None
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.data.test.pop("samples_per_gpu", None)
    info = {}
    if a.quant != "none":
        qa = types.SimpleNamespace(checkpoint=a.checkpoint, scope=a.scope, quant=a.quant, exclude_ego_out=a.exclude_ego_out,
                                   ego_amax_json=a.ego_amax_json)
        stem = osp.splitext(osp.abspath(a.out))[0]
        cal = PN.write_window_ann(cfg, a.calib_windows, a.calib_seed, stem + "_calib_navtrain_windows.pkl", "ptq_calib")
        st = {"calib": cal}
        model, hooks = PN.build_quant_model(cfg, qa, st, cal["ann_file"])
        info = dict(calib=cal, n_quantized=st["n_quantized"], n_by_part=st["n_by_part"], calib_sec=st["calib_sec"])
    else:
        model = PN.F.build_model(cfg, a.checkpoint)
    PN.check_reset_on_scene(model)
    if a.limit_scenes > 0:
        cfg.data.test.ann_file = PN.write_test_subset(cfg.data.test.ann_file, a.limit_scenes,
                                                      osp.splitext(osp.abspath(a.out))[0] + f"_navtest_first{a.limit_scenes}.pkl")
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    ds = build_dataset(cfg.data.test)
    dl = build_dataloader_origin(ds, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    model = MMDataParallel(model.cuda(), device_ids=[0]).eval()
    return cfg, ds, dl, model, info


def teacher_table(specs, periods, student):
    out = []
    for spec in specs or []:
        name, path = spec.split("=", 1)
        d = json.load(open(path))
        w = d["summary"]["wall"]
        for P in periods:
            out.append(dict(teacher=name, teacher_json=path, teacher_measured=d.get("time") or d.get("date") or time.ctime(osp.getmtime(path)),
                            P=P, teacher_wall_median_ms=w["median"], teacher_wall_mean_ms=w["mean"],
                            est_frame_ms_median_sum=w["median"] / P + student["frame_ms_steady"]["median_ms"],
                            est_frame_ms_mean=w["mean"] / P + student["frame_ms_all"]["mean_ms"],
                            est_hz_mean=1000.0 / (w["mean"] / P + student["frame_ms_all"]["mean_ms"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True, help="결과 json(있으면 멈춤)")
    ap.add_argument("--variant", choices=["eager", "graph_post_cw"], default="eager")
    ap.add_argument("--precision", choices=["mixed", "fp32"], default="mixed")
    ap.add_argument("--quant", choices=["none", "int8", "fp8"], default="none")
    ap.add_argument("--scope", choices=["bb", "bb_dec"], default="bb")
    ap.add_argument("--calib-windows", type=int, default=64)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--ego-amax-json", nargs="*", default=None)
    ap.add_argument("--exclude-ego-out", action="store_true")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--limit-scenes", type=int, default=0, help=">0 이면 navtest 앞 N 장면만 데이터셋으로(warmup+n ≤ 4N 이어야 함)")
    ap.add_argument("--teacher-json", nargs="*", default=None, help="이름=teacher_latency_*.json (교사 주기 추정표)")
    ap.add_argument("--periods", default="1,2,4,8")
    ap.add_argument("--require-idle", action="store_true", help="시작 전 물리 GPU 에 다른 연산 프로세스가 있으면 멈춘다")
    ap.add_argument("--max-gpu-gb", type=float, default=20.0)
    a = ap.parse_args()
    if osp.exists(a.out):
        raise FileExistsError(a.out)
    if a.max_gpu_gb > 0:
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gpu_gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory), 0)
    me = os.getpid()
    before = gpu_snapshot()
    foreign0 = [x for x in before["apps"] if int(x.split(",")[0]) != me]
    if a.require_idle and foreign0:
        raise SystemExit(f"[중단] GPU{PHYS} 에 다른 프로세스: {foreign0} (--require-idle)")
    load0 = os.getloadavg()
    cfg, ds, dl, model, qinfo = build(a)
    if a.warmup + a.n > len(ds):
        raise ValueError(f"warmup+n {a.warmup + a.n} > 데이터셋 프레임 {len(ds)}")
    fs = None
    if a.variant == "eager":
        def run(data):
            return model(return_loss=False, rescale=True, **data)
    else:
        from student_bench_navsim import make_navsim_fast
        from student_equiv import VARIANTS
        fs = make_navsim_fast(model, **VARIANTS[a.variant])
        run = fs
    mon = Monitor(); mon.start()
    steady, resets, first = [], [], []
    torch.cuda.reset_peak_memory_stats()
    prev = None
    it = iter(dl)
    t_start = time.time()
    with torch.no_grad():
        for i in range(a.warmup + a.n):
            data = next(it)
            sc = data["img_metas"].data[0][0]["scene_token"]
            is_reset = sc != prev
            prev = sc
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run(data)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            if fs is not None and fs.last_reset != is_reset:
                raise RuntimeError("장면 경계 판정 불일치")
            if i >= a.warmup:
                (resets if is_reset else steady).append(dt)
            elif i < 3:
                first.append(dt)
    mon.stop_ev.set(); mon.join()
    res = dict(config=a.config, checkpoint=a.checkpoint, variant=a.variant, precision=a.precision, quant=a.quant, scope=a.scope,
               deploy_latency=(a.quant == "none"),
               note=("가짜 양자화(연산은 원 정밀도) — 배포 INT8 지연 아님" if a.quant != "none" else "원 정밀도 학생(배치 1)"),
               quant_info=qinfo, warmup=a.warmup, n=a.n, n_dataset=len(ds), gpu=torch.cuda.get_device_name(0), phys_gpu=PHYS,
               torch=torch.__version__, cudnn_benchmark=torch.backends.cudnn.benchmark,
               frame_ms_steady=common.summarize(steady), frame_ms_reset=common.summarize(resets) if resets else None,
               frame_ms_all=common.summarize(steady + resets), n_steady=len(steady), n_reset=len(resets), first_frames_ms=first,
               build_timings_s=getattr(fs, "timings", None), peak_allocated_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
               measure_wall_s=time.time() - t_start, gpu_before=before, gpu_during=mon.summary(me), gpu_after=gpu_snapshot(),
               cpu_loadavg_before=load0, cpu_loadavg_after=os.getloadavg(), ncpu=os.cpu_count(),
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    res["teacher_period"] = teacher_table(a.teacher_json, [int(p) for p in a.periods.split(",")], res)
    os.makedirs(osp.dirname(osp.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False, default=str)
    s, r, g = res["frame_ms_steady"], res["frame_ms_reset"], res["gpu_during"]
    print(f"[latency] {a.variant} {a.precision} quant={a.quant}/{a.scope}: 정상 median {s['median_ms']:.2f} p90 {s['p90_ms']:.2f} (n {len(steady)})"
          f" | 초기화 {('%.2f / %.2f (n %d)' % (r['median_ms'], r['p90_ms'], len(resets))) if r else '-'}"
          f" | 전체 mean {res['frame_ms_all']['mean_ms']:.2f} | SM {g['sm_clock_mhz']} | 다른 프로세스 {g['foreign_pids']}"
          f" | load {load0[0]:.0f}→{res['cpu_loadavg_after'][0]:.0f}", flush=True)
    for t in res["teacher_period"]:
        print(f"  교사 {t['teacher']} P={t['P']}: 프레임당 평균 ≈ {t['est_frame_ms_mean']:.1f} ms ({t['est_hz_mean']:.2f} Hz)", flush=True)
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
