"""수정(2026-10-09): LaQ-AD NAVSIM — Qwen2-VL-2B 18층 은닉 토큰 오프라인 캐시(전방 3캠 — 2026-10-09 SparseDriveV2 설정으로 변경, 이전 8캠).

공용 인코더 models/encoder/vlm_qwen2vl/qwen_online.py(지시문·화소예산·계층을 nuScenes/B2D 와 공유하는 모듈)를 num_views=8 로 쓴다.
nuScenes 캐시와 같은 문체의 지시문을 8방향으로 늘린 것만 다르다(시점 수가 다르므로 같은 문구를 쓸 수 없다).
  [확인 필요] 카메라 이름→방향 대응(L0=전좌, L1=좌, L2=후좌, R0=전우, R1=우, R2=후우)은 nuPlan 센서 배치 관례에 따른
  가정이다. devkit 소스는 방향을 서술하지 않는다 — 데이터 마운트 후 영상으로 확인하고 다르면 NAVSIM_VIEW_NAMES 를 고친다.

화소예산(사용자 결정 2026-10-09): AutoVLA(backup/reference/AutoVLA.pdf 부록 multi-view setup) 와 같게
  max_pixels = min_pixels = 128*28*28 = 100,352 ("reduced to 28 × 28 × 128 pixels", 원 종횡비 유지).
  transformers smart_resize 기준 1920×1080 → 420×224/캠, 병합 후 시각 토큰 120개/캠(코디네이터 확인).
  nuScenes/B2D 캐시(200*28*28 → 504×280, 180 토큰/캠)와 다르다 — 벤치마크별 설정이며 공용 인코더 기본값은 바꾸지 않는다.
  [AutoVLA 와 다른 점] AutoVLA 는 3캠(front, front-left, front-right) × 4프레임(2 Hz)을 쓴다. 우리는 단일 시각 전방 3캠(SparseDriveV2 와 동일 카메라)을 쓴다
  (사용자가 달리 정하기 전까지). 논문 비교 시 이 차이를 밝힐 것.

저장: <out-dir>/<frame token>.pt {'hidden': (L,1536) bf16, 'attn_mask': (L,) uint8}. 이력 프레임도 추론에 쓰이므로
info 의 모든 프레임 token 을 처리한다(중복 token 은 1회). 첫 표본의 L 을 <out-dir>/_meta.txt 에 기록하고 이후 다르면 실패.
용량 추정: 시각 8×120=960 + 문자 ≈ 1.05k 토큰 × 1536 × 2B ≈ 3.2MB/표본(추정, 실제 L 은 _meta.txt) → NAS 경로에 생성.

예) CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/qwen_cache_navsim.py \
      --ann-files data/infos/navsim_navtrain.pkl data/infos/navsim_navtest.pkl --out-dir <NAS>/navsim_qwen_l18 --shard 0/1
"""
import argparse
import importlib.util as _ilu
import os
import pickle
import time
from pathlib import Path

import torch
from PIL import Image

_spec = _ilu.spec_from_file_location(
    "qwen_online", str(Path(__file__).resolve().parents[1] / "models" / "encoder" / "vlm_qwen2vl" / "qwen_online.py"))  # 수정(2026-10-15): 경로 이동
_qo = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_qo)

NAVSIM_CAMS = ("CAM_L0", "CAM_F0", "CAM_R0")  # 수정(2026-10-09): SparseDriveV2 NAVSIM 설정(사용자 결정) — 전방 3캠 l0·f0·r0, 1프레임 (swc-17/SparseDriveV2 sparsedrive_config.py:63)
# 이전(8캠): NAVSIM_CAMS = ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0")
NAVSIM_VIEW_NAMES = ("front-left", "front", "front-right")  # 수정(2026-10-09): NAVSIM_CAMS 순서와 일치
NAVSIM_PIXELS = 128 * 28 * 28   # 수정(2026-10-09): AutoVLA NAVSIM 설정(사용자 결정)
NAVSIM_INSTRUCTION = (
    "These are 3 front-facing camera images of an autonomous vehicle (" + ", ".join(NAVSIM_VIEW_NAMES) + "). "
    "Describe the salient objects, lane geometry, traffic state, and any driving-relevant cues."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-path", default="/home1/irteam/data-vol1/src/HiP-AD/ckpts/qwen2-vl-2b")
    ap.add_argument("--ann-files", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True, help="NAS 경로 권장(용량 수백 GB)")
    ap.add_argument("--layer", type=int, default=_qo.LAYER)
    ap.add_argument("--pixels", type=int, default=NAVSIM_PIXELS, help="Qwen min=max_pixels (AutoVLA 128*28*28)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seen, todo = set(), []
    for af in args.ann_files:
        for i in pickle.load(open(af, "rb"))["infos"]:
            if i["token"] not in seen:
                seen.add(i["token"]); todo.append(i)
    si, sn = (int(x) for x in args.shard.split("/"))
    todo = todo[si::sn]
    print(f"unique frames (shard {si}/{sn}): {len(todo)}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    meta_p = os.path.join(args.out_dir, "_meta.txt")
    known_L = int(open(meta_p).read().split()[1]) if os.path.exists(meta_p) else None

    enc = _qo.QwenOnlineEncoder(args.qwen_path, layer=args.layer, device=args.device, pixels=args.pixels,
                                instruction=NAVSIM_INSTRUCTION, num_views=len(NAVSIM_CAMS))
    t0, saved = time.time(), 0
    for k, info in enumerate(todo):
        out = os.path.join(args.out_dir, f"{info['token']}.pt")
        if os.path.exists(out):
            continue   # 재개: 결정적 산출이라 재계산과 동일
        imgs = []
        for cam in NAVSIM_CAMS:
            p = info["cams"][cam]["data_path"]
            if not os.path.exists(p):
                raise FileNotFoundError(f"NAVSIM 영상 누락: {p}")   # 검은 영상 대체 금지
            imgs.append(Image.open(p).convert("RGB"))
        hidden, mask = enc.encode(imgs)
        L = int(hidden.shape[1])
        if known_L is None:
            known_L = L
            with open(meta_p, "w") as f:
                f.write(f"L {L}\nlayer {args.layer}\npixels {args.pixels}\ninstruction {NAVSIM_INSTRUCTION}\n")
            print(f"token length L = {L}", flush=True)
        elif L != known_L:
            raise RuntimeError(f"{info['token']}: L={L} != {known_L} — 화소예산/지시문 고정 가정 위반")
        torch.save({"hidden": hidden[0].cpu().contiguous(), "attn_mask": mask[0].cpu()}, out + ".tmp")
        os.replace(out + ".tmp", out)
        saved += 1
        if (k + 1) % 100 == 0:
            el = time.time() - t0
            print(f"[{k+1}/{len(todo)}] saved {saved} ETA {el/(k+1)*(len(todo)-k-1)/3600:.2f}h", flush=True)
    print(f"DONE saved {saved} / {len(todo)}", flush=True)


if __name__ == "__main__":
    main()
