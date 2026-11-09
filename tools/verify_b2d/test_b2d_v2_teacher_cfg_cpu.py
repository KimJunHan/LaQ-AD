# 수정(2026-10-19): 사용자 승인 — Bench2Drive 사전 준비(CPU)
"""B2D v2_teacher config·교사 캐시 로더(LoadTeacherCacheB2D)·2단계 지도 정렬 확산 CPU 검사.

  CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=. python tools/verify_b2d/test_b2d_v2_teacher_cfg_cpu.py [--full]

(1) 기존 파일 해시 불변(B2D base config 4 개, loading.py, plan_diffusion_map.py)
(2) Config.fromfile 4 개: 등록 모듈·로더 type·qwen_type·run_name·epoch(12/18)·load_from·test num_stream_groups=1·
    2단계 확산 값 = results/b2d_prep/b2d_plan_diffusion_stats.json 에서 유도한 값
(3) LoadTeacherCacheB2D 단위 시험(가짜 캐시): 키프레임 사상·나이·원래 token 보존·키프레임 누락/색인 누락/layout_ok 거짓 → 예외
(4) 모델 빌드(CPU): v2_teacher 1·2단계, B2D base 2단계. state_dict 키 차이 = 유형 임베딩(1·2단계) + plan_diffusion(2단계) 만
(5) 확산 모듈: B2D 계획 앵커(480×12) 그대로 forward(학습·평가, 지도 기억 있음/없음), 모양·유한값·클립 끔 확인
(6) --full: val pkl·지도 pkl 로 학습 데이터셋을 만들고(가짜 교사 캐시) 같은 스트림 연속 2 프레임 forward_train·backward(CPU, 수 분)
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch

_orig_load = torch.load
def _load(*a, **kw):  # tools/train.py 와 같은 호환 shim
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _load

import projects.mmdet3d_plugin  # noqa: F401,E402
import tools.plan_diffusion_map_register  # noqa: F401,E402
import tools.b2d_teacher_register  # noqa: F401,E402
from mmcv import Config  # noqa: E402

from projects.mmdet3d_plugin.models.encoder.vl_tokens_cache.loading_b2d import (LoadTeacherCacheB2D, b2d_clip_name,
                                                                               b2d_keyframes, b2d_keyframe_for)  # noqa: E402
from projects.mmdet3d_plugin.models.decoder.noise_t.plan_diffusion_map import MapAlignedPlanTruncatedDiffusion  # noqa: E402

CFG = {s: {k: f"configs/b2d/v2_teacher/stage{s}/{k}.py" for k in ("train", "test")} for s in (1, 2)}
BASE_SHA = {
    "configs/b2d/stage1/train.py": "824724677ddc97fc32031a963ab841ef24560ee9dd016b4e8f57af00eca082b4",
    "configs/b2d/stage2/train.py": "bfa3431f98550e9a8c46abfa95ccece3141980cde645ff8e2320ec8ec3b8d08e",
    "configs/b2d/stage1/test.py": "edcf3810701f37dc5f5e417fbd279afbe8259694f304986ab8b0ac49ee8e539f",
    "configs/b2d/stage2/test.py": "c12632ed3336b6bc07978d8283d51906652b925055ae5027e0e6a83e5541cd32",
    "models/encoder/vl_tokens_cache/loading.py": "564a12d62e9c25328ed3a7f78ab493307c44a96f3f929f40c765bd77754a4aee",
    # 수정(2026-10-19): 코디네이터 중앙 API 변경(anisotropy_ratio 제거, 타원 회전 규칙) 반영 후 해시 — 이 작업에서는 수정하지 않음
    "models/decoder/noise_t/plan_diffusion_map.py": "1781b0eee834e6b905399d770b1384629b800d9b5933e11b0a979cac62412889",
}
STATS = "results/b2d_prep/b2d_plan_diffusion_stats.json"
FOLDERS = ["context", "detection", "tracking", "motion", "map", "plan", "action"]
META = dict(prompt_mode="B", teacher_chain_verified=True, constrained_decoding="answer_grammar_v1")
RES = []


def check(name, ok, msg=""):
    RES.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {msg}" if msg else ""), flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---------------------------------------------------------------- 가짜 교사 캐시
def make_fake_cache(root, keys, lens=None, layout_ok=True, index_keys=None):
    lens = lens or {f: 3 + i for i, f in enumerate(FOLDERS)}
    for f in FOLDERS:
        os.makedirs(os.path.join(root, f), exist_ok=True)
        json.dump(dict(L=lens[f], **META), open(os.path.join(root, f, "_meta.json"), "w"))
        for k in keys:
            p = os.path.join(root, f, k + ".pt")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            torch.save(dict(hidden=torch.randn(lens[f], 1536).to(torch.bfloat16) * 0.01,
                            attn_mask=torch.ones(lens[f], dtype=torch.uint8)), p)
    os.makedirs(os.path.join(root, "_index"), exist_ok=True)
    with open(os.path.join(root, "_index", "teacher_A.jsonl"), "w") as fh:
        for k in (index_keys if index_keys is not None else keys):
            fh.write(json.dumps(dict(token=k, teacher="A", layout_ok=layout_ok, constrained_decoding="answer_grammar_v1")) + "\n")
    return lens


# ---------------------------------------------------------------- (1)(2)
def test_files_and_configs():
    print("(1) 기존 파일 해시 불변", flush=True)
    for p, h in BASE_SHA.items():
        check(p, sha(p) == h)
    print("(2) Config.fromfile", flush=True)
    st = json.load(open(STATS)) if os.path.exists(STATS) else None
    check("통계 파일 있음", st is not None, STATS)
    for s in (1, 2):
        for k in ("train", "test"):
            c = Config.fromfile(CFG[s][k])
            od = c.model.head.onedecoder_head
            tl = [p for p in c.data.train.pipeline if p["type"].startswith("LoadTeacherCache")]
            tv = [p for p in c.data.test.pipeline if p["type"].startswith("LoadTeacherCache")]
            col = [p for p in c.data.train.pipeline if p["type"] == "Collect"][0]
            colt = [p for p in c.data.test.pipeline if p["type"] == "Collect"][0]
            ok = ("tools.b2d_teacher_register" in c.custom_imports.imports and len(tl) == 1 and len(tv) == 1
                  and tl[0]["type"] == "LoadTeacherCacheB2D" and tl[0]["cache_root"] == "data/bench2drive/laq_cache"
                  and not any(p["type"] == "LoadQwenCacheNusc" for p in c.data.train.pipeline + c.data.test.pipeline)
                  and "qwen_type" in col["keys"] and "qwen_type" in colt["keys"]
                  and od.qwen_token_attn.num_token_types == 7 and list(od.qwen_token_attn.token_type_names) == FOLDERS
                  and od.qwen_token_attn.route_segs is None and c.run_name == f"hipad_b2d_laq_v2_teacher_stage{s}")
            ep = c.runner.max_iters / c.num_iters_per_epoch
            ok &= ep == (12 if s == 1 else 18)
            ok &= c.optimizer.lr == 2e-4
            if k == "test":
                ok &= od.num_stream_groups == 1
            else:
                ok &= od.num_stream_groups == c.cumulative_iters
            if s == 2:
                ok &= c.load_from == "./results/hipad_b2d_laq_v2_teacher_stage1/latest.pth"
                pd = od.plan_diffusion
                ok &= pd.type == "MapAlignedPlanTruncatedDiffusion" and od.map_instance_bank.type == "PlanMapMemoryInstanceBank"
                ok &= "tools.plan_diffusion_map_register" in c.custom_imports.imports
                if st is not None:
                    ok &= list(pd.norm_scale) == st["norm"]["norm_scale"] and list(pd.norm_offset) == st["norm"]["norm_offset"]
            else:
                ok &= od.get("plan_diffusion") is None and od.map_instance_bank.type == "InstanceBank"
            check(f"{CFG[s][k]}", ok, f"epoch {ep:g}, batch {c.batch_size}×누적 {c.cumulative_iters}, run {c.run_name}")
    c = Config.fromfile(CFG[2]["train"])
    pd = c.model.head.onedecoder_head.plan_diffusion
    print(f"     2단계 확산: {dict(pd)}", flush=True)


# ---------------------------------------------------------------- (3)
def test_loader(tmp):
    print("(3) LoadTeacherCacheB2D 단위 시험(가짜 캐시)", flush=True)
    clip = "AccidentTwoWays_Town12_Route1102_Weather10"
    check("키프레임 규칙: f0=0 → 0,5,10 / f0=1 → 1,6,11", b2d_keyframes([0, 1, 2, 9, 12]) == [0, 5, 10]
          and b2d_keyframes([1, 2, 3, 13]) == [1, 6, 11])
    check("사상: kf ≤ f 최대, 나이 > 4 프레임이면 None", [b2d_keyframe_for([1, 6, 11], f) for f in (0, 1, 5, 6, 14, 16)] ==
          [None, 1, 1, 6, 11, None])
    root = os.path.join(tmp, "cache_ok")
    clip1 = "VehicleTurningRoutePedestrian_Town15_Route481_Weather19"   # f0 = 1 인 실제 val 클립 이름
    lens = make_fake_cache(root, [f"{clip}/00000", f"{clip}/00005", f"{clip1}/00001", f"{clip1}/00006"])
    ld = LoadTeacherCacheB2D(cache_root=root, folders=FOLDERS, hidden_dim=1536, meta_expect={f: META for f in FOLDERS})
    d = ld(dict(folder="v1/" + clip, frame_idx=3, token=f"v1/{clip}_3"))
    L = sum(lens.values())
    check("frame 3 → 키프레임 0, 나이 0.3 s, 모양 (1, L, 1536), 유형 번호 0..6, token 원래 값",
          d["teacher_key"] == f"{clip}/00000" and abs(d["teacher_age"] - 0.3) < 1e-9 and tuple(d["qwen_hidden"].data.shape) == (1, L, 1536)
          and d["qwen_type"].data.unique().tolist() == list(range(7)) and d["token"] == f"v1/{clip}_3",
          f"L={L}")
    d5 = ld(dict(folder="v1/" + clip, frame_idx=5, token="x"))
    check("frame 5 → 키프레임 5, 나이 0", d5["teacher_key"] == f"{clip}/00005" and d5["teacher_age"] == 0.0)
    d1 = ld(dict(folder="v1/" + clip1, frame_idx=5, token="x"))
    check("f0=1 클립 frame 5 → 키프레임 1, 나이 0.4 s", d1["teacher_key"] == f"{clip1}/00001" and abs(d1["teacher_age"] - 0.4) < 1e-9)
    for fr in (0,):
        try:
            ld(dict(folder="v1/" + clip1, frame_idx=fr, token="x")); check("f0=1 클립 frame 0 → 예외", False)
        except RuntimeError as e:
            check("f0=1 클립 frame 0(앞 키프레임 없음) → 예외", True, str(e)[:70])
    try:
        ld(dict(folder="v1/NoSuchClip_Town01_Route1_Weather1", frame_idx=0, token="x")); check("색인에 없는 클립 → 예외", False)
    except RuntimeError as e:
        check("색인에 없는 클립 → 예외", True, str(e)[:60])
    try:
        ld(dict(folder="v1/" + clip, frame_idx=12, token="x")); check("키프레임 10 누락 → 예외", False)
    except RuntimeError as e:   # 색인 키프레임 [0,5] → frame 12 는 나이 0.7 s → 대체하지 않고 예외
        check("키프레임 10 누락(frame 12, 마지막 키프레임 5 는 0.7 s 전) → 예외", "키프레임" in str(e), str(e)[:80])
    root2 = os.path.join(tmp, "cache_noidx_file")
    make_fake_cache(root2, [f"{clip}/00000"], index_keys=[f"{clip}/00000", f"{clip}/00005"])
    ld2 = LoadTeacherCacheB2D(cache_root=root2, folders=FOLDERS, hidden_dim=1536)
    try:
        ld2(dict(folder="v1/" + clip, frame_idx=7, token="x")); check("색인은 있으나 파일 누락 → 예외", False)
    except FileNotFoundError as e:
        check("색인은 있으나 파일 누락 → FileNotFoundError", True, str(e)[:80])
    root3 = os.path.join(tmp, "cache_bad")
    make_fake_cache(root3, [f"{clip}/00000"], layout_ok=False)
    ld3 = LoadTeacherCacheB2D(cache_root=root3, folders=FOLDERS, hidden_dim=1536)
    try:
        ld3(dict(folder="v1/" + clip, frame_idx=1, token="x")); check("layout_ok=False → 예외", False)
    except ValueError as e:
        check("layout_ok=False → ValueError", True, str(e)[:60])
    try:
        ld(dict(token="x")); check("folder/frame_idx 없음 → 예외", False)
    except RuntimeError:
        check("folder/frame_idx 없음 → RuntimeError", True)


# ---------------------------------------------------------------- (4)(5)
def build(path, cache_root=None):
    from mmdet.models import build_detector
    c = Config.fromfile(path)
    torch.manual_seed(0)
    m = build_detector(c.model); m.init_weights()
    return c, m


def test_models():
    print("(4) 모델 빌드(CPU)·state_dict 키 비교", flush=True)
    import tools.verify_v2.test_sensor_tokens_cpu  # noqa: F401  flash-attn·deformable CPU 기준 구현(같은 방식)
    _, base2 = build("configs/b2d/stage2/train.py")
    _, base1 = build("configs/b2d/stage1/train.py")
    _, v1 = build(CFG[1]["train"])
    c2, v2 = build(CFG[2]["train"])
    k = lambda m: set(m.state_dict().keys())  # noqa: E731
    add1, rem1 = k(v1) - k(base1), k(base1) - k(v1)
    add2, rem2 = k(v2) - k(base2), k(base2) - k(v2)
    check("1단계: 추가 키 = 유형 임베딩만, 제거 0", not rem1 and all("type_embed" in x or "token_type" in x for x in add1) and add1,
          f"추가 {sorted(add1)}")
    check("2단계: 추가 키 = 유형 임베딩 + plan_diffusion 만, 제거 0",
          not rem2 and all(("type_embed" in x or "token_type" in x) or ".plan_diffusion." in x for x in add2),
          f"추가 {sorted(add2)}")
    p1, p2 = sum(p.numel() for p in v1.parameters()), sum(p.numel() for p in v2.parameters())
    print(f"     파라미터 수: v2_teacher 1단계 {p1:,}, 2단계 {p2:,}", flush=True)
    od = v2.head.onedecoder_head
    check("2단계 디코더: MapAligned 잡음 + PlanMapMemory 지도 뱅크 + 지도 기억 사용 표시",
          isinstance(od.plan_diffusion, MapAlignedPlanTruncatedDiffusion) and od.plan_diffusion_uses_map
          and type(od.map_instance_bank).__name__ == "PlanMapMemoryInstanceBank")
    print("(5) 확산 모듈 — B2D 계획 앵커 그대로", flush=True)
    pdm = od.plan_diffusion
    anc = od.plan_instance_bank.anchor.detach()
    check("계획 앵커 (480, 12) = 10 유형 × 48", tuple(anc.shape) == (480, 12), str(tuple(anc.shape)))
    st = json.load(open(STATS))
    a2 = anc.reshape(-1, 6, 2)
    lo, hi = (pdm.norm_offset - pdm.norm_scale), (pdm.norm_offset + pdm.norm_scale)
    inside = float(((a2 >= lo) & (a2 <= hi)).all(-1).float().mean())
    check("앵커 점이 정규화 범위 [offset±scale] 안(참고)", inside > 0.99, f"안쪽 비율 {inside:.4f}, 앵커 x [{float(a2[...,0].min()):.1f},{float(a2[...,0].max()):.1f}] y [{float(a2[...,1].min()):.1f},{float(a2[...,1].max()):.1f}]")
    B = 2
    A = anc[None].repeat(B, 1, 1)
    # 가짜 지도 기억: 100 인스턴스 × 20 점, 4 클래스 점수(앞 B 표본 중 하나는 점수 0 = 지도 없음)
    xs = torch.linspace(-30, 30, 20)
    # 수정(2026-10-19): 새 규칙(타원 크기 유지·긴 축만 차선 접선으로 회전)에서는 전방(+y)과 나란한 차선이면 회전이 없어 등방과 같다.
    #   회전 효과를 보이려고 가짜 차선을 45° 로 기울인다(지도 있는 표본 ≠ 지도 없는 표본 검사용).
    c45 = 0.70710678
    pts = torch.stack([torch.stack([c45 * xs + (i % 10 - 5) * 3.5, c45 * xs], -1) for i in range(100)])[None].repeat(B, 1, 1, 1)
    sc = torch.full((B, 100, 4), 0.9); sc[1] = 0.0
    mem = dict(pts=pts, scores=sc)
    for train in (True, False):
        pdm.train(train)
        torch.manual_seed(1)
        x, te = pdm(A.clone(), map_memory=mem)
        torch.manual_seed(1)
        x0, _ = pdm(A.clone(), map_memory=None)
        ok = tuple(x.shape) == (B, 480, 12) and tuple(te.shape) == (B, 1, 256) and torch.isfinite(x).all()
        ok &= torch.allclose(x[1], x0[1])        # 점수 0 표본 = 지도 없음 = 등방(부모와 같음)
        ok &= not torch.allclose(x[0], x0[0])    # 지도 있는 표본은 이방성
        check(f"{'학습' if train else '평가'}: 모양·유한값, 지도 없는 표본 = 등방(동일), 지도 있는 표본 ≠", bool(ok))
    check("클립 끔(clip_to_boundary=False, boundary_class_ids=())", (not pdm.clip_to_boundary) and pdm.boundary_class_ids == ())
    return c2, v2


# ---------------------------------------------------------------- (6)
def test_full(tmp):
    print("(6) 실제 B2D val 데이터(가짜 교사 캐시)로 2단계 forward_train·backward 연속 2 프레임", flush=True)
    import time
    from mmcv.parallel import collate
    from mmdet.datasets import build_dataset
    import tools.verify_v2.test_sensor_tokens_cpu as sens
    import pickle
    infos = pickle.load(open("data/infos/b2d_infos_val.pkl", "rb"))
    by = {}
    for i in infos[:400]:
        by.setdefault(b2d_clip_name(i["folder"]), []).append(i["frame_idx"])
    keys = sorted(f"{c}/{k:05d}" for c, fs in by.items() for k in b2d_keyframes(fs))
    root = os.path.join(tmp, "cache_full")
    make_fake_cache(root, keys)
    del infos
    c = Config.fromfile(CFG[2]["train"])
    c.model.head.onedecoder_head.plan_diffusion.score_thr = 0.0   # 무작위 초기 모델의 지도 점수는 낮다 → 기하 경로를 타게
    for p in c.data.train.pipeline:
        if p["type"] == "LoadTeacherCacheB2D":
            p["cache_root"] = root
    tr = c.data.train.copy(); tr.ann_file = "data/infos/b2d_infos_val.pkl"
    t0 = time.time()
    ds = build_dataset(tr)
    print(f"     데이터셋 {len(ds)} 표본, {time.time() - t0:.0f}s", flush=True)
    from mmdet.models import build_detector
    torch.manual_seed(0)
    det = build_detector(c.model); det.init_weights()
    od = det.head.onedecoder_head
    seen = []
    orig = od.plan_diffusion.forward
    def spy(anchor, map_memory=None):
        seen.append(0 if map_memory is None else int((map_memory["scores"].max(-1).values > 0).sum()))
        return orig(anchor, map_memory=map_memory)
    od.plan_diffusion.forward = spy
    det.train(); det.use_grid_mask = False
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    sc = []
    for idx in (20, 21):
        t1 = time.time()
        item = ds[dict(idx=idx, aug_config=ds.get_augmentation(), stream_group=0)]
        d = sens.unwrap(collate([item], samples_per_gpu=1))
        img = d.pop("img")
        sc.append((d["img_metas"][0]["scene_token"], float(d["img_metas"][0]["timestamp"])))
        losses = det.forward_train(img, **d)
        total = sum(v for k, v in losses.items() if "loss" in k and torch.is_tensor(v))
        total.backward(); det.zero_grad()
        bad = [k for k, v in losses.items() if torch.is_tensor(v) and not torch.isfinite(v).all()]
        check(f"학습 idx {idx}: 손실 유한·역전파", not bad, f"total={float(total):.3f}, {time.time() - t1:.1f}s, {sc[-1]}")
    check("같은 클립 0.5 s 간격 → 둘째 프레임에 지도 기억 전달", sc[0][0] == sc[1][0] and abs(sc[1][1] - sc[0][1] - 0.5) < 1e-6
          and seen[0] == 0 and seen[1] == 100, f"seen={seen}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    assert not torch.cuda.is_available(), "CPU 전용 시험: CUDA_VISIBLE_DEVICES=\"\""
    tmp = tempfile.mkdtemp(prefix="b2d_v2t_", dir=os.environ.get("B2D_TEST_TMP"))
    try:
        test_files_and_configs()
        test_loader(tmp)
        test_models()
        if a.full:
            test_full(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(RES)
    print(f"\n결과: {n_ok}/{len(RES)} PASS" + ("" if n_ok == len(RES) else "  ← FAIL 있음"))
    sys.exit(0 if n_ok == len(RES) else 1)


if __name__ == "__main__":
    main()
