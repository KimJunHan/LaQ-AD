# 수정(2026-11-05): NAVSIM 학생 추론 지연(batch 1, navtest) — 새 파일(기존 소스·student_fast.py 수정 없음).
"""NAVSIM 학생(configs/navsim/v2_teacher/stage{1,2}/test.py) 지연: eager vs FastStudent(graph_post_cw) 의 NAVSIM 판.

NAVSIM 에서 다른 것
  1) reset_bank_on_scene_change=True(navtest 장면 창 경계에서 시간 뱅크 초기화). FastStudent 는 이를 거부하므로
     NavsimFastStudent 가: 장면이 바뀐 프레임 = 원본 eager(인스턴스 패치를 잠시 떼고 → SparseDetector._maybe_reset_banks 가
     뱅크를 비우고 첫 프레임 경로) → 그 뒤 뱅크 상태를 고정 버퍼에 다시 복사, 같은 장면의 다음 프레임부터 graph replay.
     시간은 '초기화 프레임(eager)'과 '정상 프레임'을 따로 낸다(연속 주행 배치에서는 정상 프레임만 있다).
  2) 학습된 NAVSIM 학생 체크포인트가 아직 없다 → --checkpoint none: 생성자 기본 초기화 가중치(예비 측정).
     모양은 가중치와 무관하다. 데이터 의존 경로는 FastStudent 조건 검사(score_threshold=None 등)로 없음이 보장된 경우만 graph.
  3) navtest 교사 캐시가 아직 없다(GPU3 추출 중) → 색인에 없는 token 은 --donor 의 navtrain 캐시 텐서(같은 모양·실제 분포)로 대신
     읽는다(LatencyTeacherCacheDonor). 캐시 파일은 읽기만 한다. 대체된 프레임 수를 기록한다.
사용: CUDA_VISIBLE_DEVICES=0 HIPAD_ALLOW_GPU0=1 flock /tmp/claude-500/gpu0_timing.lock \
      python tools/latency/student_bench_navsim.py --config configs/navsim/v2_teacher/stage2/test.py --variant graph_post_cw --out x.json
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import student_common as common  # noqa: E402
import torch  # noqa: E402

DONOR = {"n": 0, "token": None}


def register_donor_loader():
    from mmdet.datasets.builder import PIPELINES
    LoadTeacherCacheNusc = PIPELINES.get("LoadTeacherCacheNusc")   # 플러그인이 등록한 클래스(모듈 경로 중복 import 방지)

    @PIPELINES.register_module(force=True)
    class LatencyTeacherCacheDonor(LoadTeacherCacheNusc):
        def __init__(self, donor_token=None, **kw):
            super().__init__(**kw)
            self.donor = donor_token
            if self.index is None or donor_token not in self.index:
                raise RuntimeError(f"donor {donor_token} 가 교사 캐시 색인에 없다")

        def __call__(self, input_dict):
            tok = input_dict.get("token")
            if tok in self.index:
                return super().__call__(input_dict)
            input_dict["token"] = self.donor
            try:
                out = super().__call__(input_dict)
            finally:
                input_dict["token"] = tok
            DONOR["n"] += 1
            return out


def build(config, checkpoint, fp32, donor):
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.datasets import build_dataset, replace_ImageToTensor
    from mmdet.datasets import build_dataloader as build_dataloader_origin
    from mmdet.models import build_detector
    cfg = Config.fromfile(config)
    common.import_plugin(cfg, config)
    register_donor_loader()
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.data.test.pop("samples_per_gpu", 1)
    n_sw = 0
    for st in cfg.data.test.pipeline:
        if st["type"] == "LoadTeacherCacheNusc":
            st["type"] = "LatencyTeacherCacheDonor"; st["donor_token"] = donor; n_sw += 1
    if n_sw != 1:
        raise RuntimeError(f"test pipeline 의 LoadTeacherCacheNusc {n_sw} 개(1 기대)")
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader_origin(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    cfg.model.train_cfg = None
    torch.manual_seed(0)
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16 = cfg.get("fp16", None) is not None and not fp32
    if fp16:
        wrap_fp16_model(model)
    if checkpoint != "none":
        load_checkpoint(model, checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0]).eval()
    return cfg, dataset, loader, model, fp16


def make_navsim_fast(model, **kw):
    from student_fast import FastStudent

    class NavsimFastStudent(FastStudent):
        def _check_supported(self):
            det, od = self.det, self.od
            # 1단계(계획 확산·motion 없음, 지도 뱅크 = 일반 InstanceBank num_temp 0): 부모 검사 중 motion·확산 항목만 빼고 같은 검사
            self.stage1 = getattr(od, "plan_diffusion", None) is None
            if self.stage1:
                if od.with_close_loop or od.training:
                    raise NotImplementedError("개방루프 추론만 지원")
                if det.bev_encoders is not None:
                    raise NotImplementedError("bev_encoders(B 융합) 미지원")
                if od.det_decoder.score_threshold is not None or od.map_decoder.score_threshold is not None:
                    raise NotImplementedError("후처리 score_threshold 가 있으면 출력 개수가 데이터 의존이다")
                if od.qwen_token_attn is None or od.qwen_token_attn.route_type_ids is not None:
                    raise NotImplementedError("route_segs 경로 미지원")
                assert od.query_select == ["det", "map", "plan", "ego"] and od.task_select == ["det", "map", "plan", "ego"]
                assert hasattr(od.map_instance_bank, "plan_map_pts") is False and od.map_instance_bank.num_temp_instances == 0
                if self.post_graph:
                    raise NotImplementedError("1단계는 후처리 graph(_post_tensors: motion·계획 확산 출력 가정) 미지원 — post_graph=False")
                return
            keep = det.reset_bank_on_scene_change
            det.reset_bank_on_scene_change = False   # 초기화는 아래 __call__ 이 eager 로 처리 — 나머지 조건 검사는 그대로
            try:
                super()._check_supported()
            finally:
                det.reset_bank_on_scene_change = keep

        def _state_attrs(self):
            st = super()._state_attrs()
            if self.stage1:   # 1단계 지도 뱅크에는 계획용 지도 기억이 없다
                st = [(o, a) for o, a in st if o is not self.od.map_instance_bank]
            return st

        def _install_patches(self):
            mods = list(self.det.modules())
            before = {id(m): dict(vars(m)) for m in mods}
            od = self.od
            if self.stage1:
                # 부모 패치 함수는 계획 확산 패치까지 한 번에 건다 → 빈 자리 객체에 걸게 하고 버린다. 지도 뱅크 패치(계획 지도 기억용)는
                # 아래에서 일반 InstanceBank 판으로 바꾼다. det·교사 토큰 사영 패치는 부모 것 그대로.
                had = "plan_diffusion" in od.__dict__
                old = od.__dict__.get("plan_diffusion")
                object.__setattr__(od, "plan_diffusion", types.SimpleNamespace())
                try:
                    super()._install_patches()
                finally:
                    if had:
                        object.__setattr__(od, "plan_diffusion", old)
                    else:
                        del od.__dict__["plan_diffusion"]
                mb = od.map_instance_bank
                del mb.__dict__["get"], mb.__dict__["cache"]
                ti = {}

                def map_get_plain(bank, batch_size, metas=None):
                    # instance_bank.py:82-111, num_temp_instances=0 → cached_anchor 가 늘 None → else 분기: reset() +
                    # new_tensor([default]*bs)(H2D) → 같은 값의 미리 만든 상수(캡처 안 H2D 복사 금지).
                    instance_feature = torch.tile(bank.instance_feature[None], (batch_size, 1, 1))
                    anchor = torch.tile(bank.anchor[None], (batch_size, 1, 1))
                    bank.reset()
                    key = (instance_feature.dtype, batch_size)
                    if key not in ti:
                        assert not torch.cuda.is_current_stream_capturing()
                        ti[key] = instance_feature.new_tensor([bank.default_time_interval] * batch_size)
                    return instance_feature, anchor, bank.cached_feature, bank.cached_anchor, ti[key]
                mb.get = types.MethodType(map_get_plain, mb)
            else:
                super()._install_patches()
            self._patched = []
            for m in mods:
                b = before[id(m)]
                for k, v in vars(m).items():
                    if callable(v) and not isinstance(v, (torch.nn.Module, torch.Tensor)) and (k not in b or b[k] is not v):
                        self._patched.append((m, k, b.get(k, _MISSING), v))
            if self.verbose:
                print(f"[NavsimFast] 인스턴스 패치 {[(type(m).__name__, k) for m, k, _, _ in self._patched]}", flush=True)

        def _set_patches(self, on):
            for m, k, old, new in self._patched:
                if on:
                    setattr(m, k, new)
                elif old is _MISSING:
                    delattr(m, k)
                else:
                    setattr(m, k, old)

        def _resync_state(self, metas):
            for obj, attr, buf in self.state:
                v = getattr(obj, attr)
                if not torch.is_tensor(v):
                    v = torch.tensor(v, device=self.dev, dtype=torch.long)
                if v.shape != buf.shape or v.dtype != buf.dtype:
                    raise RuntimeError(f"초기화 뒤 상태 {type(obj).__name__}.{attr} {tuple(v.shape)} {v.dtype} != 버퍼 "
                                       f"{tuple(buf.shape)} {buf.dtype}")
                buf.copy_(v)
                setattr(obj, attr, buf)
            self.S["prev_ts"].copy_(self.od.det_instance_bank.metas["timestamp"])
            for b in (self.od.det_instance_bank, self.od.ego_instance_bank, self.od.plan_instance_bank):
                b.metas = {"timestamp": self.S["prev_ts"]}
            self.prev_metas = metas

        def __call__(self, data):
            metas, _ = self._unwrap(data)
            scene = metas[0]["scene_token"]
            reset = self.frame == 0 or scene != self._scene
            self._scene = scene
            if reset:
                self.n_reset += 1
                if not self._built:
                    out = self._eager_first(data)
                else:
                    self._set_patches(False)
                    try:
                        out = self._eager_first(data)
                    finally:
                        self._set_patches(True)
                    self._resync_state(metas)
            else:
                if not self._built:
                    self._build(data)
                    self._built = True
                out = self._run(data)
            self.last_reset = reset
            self.frame += 1
            return out

    od = model.module.head.onedecoder_head
    if getattr(od, "plan_diffusion", None) is None:
        kw = dict(kw, post_graph=False)   # 1단계: 후처리는 원본 post_process 를 graph 출력 버퍼에 eager 로
    fs = NavsimFastStudent(model, **kw)
    fs.effective_kwargs = kw
    fs.n_reset, fs._scene, fs.last_reset = 0, None, False
    return fs


_MISSING = object()


def flat(o, pre=""):
    """출력 dict → {경로: numpy}(비교용)."""
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat(v, f"{pre}.{k}"))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            out.update(flat(v, f"{pre}[{i}]"))
    elif torch.is_tensor(o):
        out[pre] = o.detach().float().cpu().numpy()
    elif isinstance(o, np.ndarray) and o.dtype.kind in "fiub":
        out[pre] = o.astype(np.float64)
    return out


def main():
    from student_equiv import VARIANTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["eager", "graph_post_cw", "graph_post"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default="none")
    ap.add_argument("--donor", default=None, help="navtrain 교사 캐시 token(기본: 교사 A 색인 첫 행)")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--equiv", type=int, default=0, help=">0: 같은 가중치의 원본 eager 모델과 앞 N 프레임 출력 비교(시간 측정 없음)")
    ap.add_argument("--det_daf", action="store_true", help="--equiv 전용: 결정적 deformable aggregation(student_fast._DetDAF) — 비트 비교")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.det_daf:
        if not a.equiv:
            raise SystemExit("--det_daf 는 --equiv 전용(지연 측정에 쓰지 않는다)")
        import student_fast
        student_fast._DetDAF.on = True
    if a.donor is None:
        p = "data/navsim/laq_cache/_index/teacher_A.s0of2.jsonl"
        a.donor = json.loads(open(p).readline())["token"]
    load_before = common.gpu_other_procs()
    cfg, dataset, loader, model, fp16 = build(a.config, a.checkpoint, a.fp32, a.donor)
    fs = None
    if a.variant == "eager":
        def run(data):
            return model(return_loss=False, rescale=True, **data)
    else:
        fs = make_navsim_fast(model, **VARIANTS[a.variant])
        run = fs
    res = {"variant": a.variant, "config": a.config, "checkpoint": a.checkpoint, "donor": a.donor, "fp16_wrapped": fp16,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "warmup": a.warmup, "n": a.n,
           "n_dataset": len(dataset), "effective_kwargs": getattr(fs, "effective_kwargs", None)}
    if a.equiv:
        _, _, _, ref, _ = build(a.config, a.checkpoint, a.fp32, a.donor)
        ref.module.load_state_dict(model.module.state_dict())
        if a.det_daf:   # eager 변형도 DAF 가 결정적 경로로 가도록 모듈 수준 디스패치 설치(원본 동작은 _Active 꺼짐이면 그대로)
            import student_fast
            student_fast.install_global_patches(model.module)
        it = iter(loader)
        worst, rows = {}, []
        with torch.no_grad():
            for i in range(a.equiv):
                data = next(it)
                o1 = flat(run(data)); o2 = flat(ref(return_loss=False, rescale=True, **data))
                if set(o1) != set(o2):
                    raise RuntimeError(f"출력 키 다름: {sorted(set(o1) ^ set(o2))[:10]}")
                d = {k: (float(np.abs(o1[k] - o2[k]).max()) if o1[k].shape == o2[k].shape and o1[k].size else
                         (0.0 if o1[k].shape == o2[k].shape else float("inf"))) for k in o1}
                for k, v in d.items():
                    worst[k] = max(worst.get(k, 0.0), v)
                rows.append(dict(i=i, reset=bool(fs.last_reset) if fs else None, max=max(d.values()) if d else 0.0))
        res.update(equiv_frames=a.equiv, equiv_worst=worst, equiv_rows=rows, donor_substituted=DONOR["n"])
        print(f"[equiv] {a.variant}: 최대 차 {max(worst.values()):.3g} (키 {max(worst, key=worst.get)}), "
              f"reset {sum(r['reset'] for r in rows if r['reset'] is not None)}", flush=True)
    else:
        it = iter(loader)
        steady, resets, first = [], [], []
        torch.cuda.reset_peak_memory_stats()
        prev_scene = None
        with torch.no_grad():
            for i in range(a.warmup + a.n):
                data = next(it)
                sc = data["img_metas"].data[0][0]["scene_token"]
                is_reset = sc != prev_scene
                prev_scene = sc
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
        allf = steady + resets
        res.update(frame_ms_steady=common.summarize(steady), frame_ms_reset=common.summarize(resets) if resets else None,
                   frame_ms_all=common.summarize(allf), n_steady=len(steady), n_reset=len(resets), first_frames_ms=first,
                   build_timings_s=fs.timings if fs is not None else None, donor_substituted=DONOR["n"],
                   peak_allocated_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
        s, r = res["frame_ms_steady"], res["frame_ms_reset"]
        print(f"{a.variant} fp16={fp16}: 정상 median {s['median_ms']:.2f} p90 {s['p90_ms']:.2f} (n {len(steady)}) | 초기화 "
              f"{('median %.2f p90 %.2f (n %d)' % (r['median_ms'], r['p90_ms'], len(resets))) if r else '-'} | 전체 median "
              f"{res['frame_ms_all']['median_ms']:.2f} | build {res['build_timings_s']} | donor {DONOR['n']}", flush=True)
    res["gpu0_load_before"] = load_before
    res["gpu0_load_after"] = common.gpu_other_procs()
    res["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False, default=str)
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
