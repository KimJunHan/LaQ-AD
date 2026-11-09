"""수정(2026-10-09): LaQ-AD Bench2Drive — Qwen2-VL-2B 18층 은닉 토큰 오프라인 캐시 생성.

nuScenes Run10 캐시(data/nuscenes/qwen_l18, tools/qwen_cache.py --layers 18)와 같은 지시문·화소예산·계층을
models/encoder/vlm_qwen2vl/qwen_online.py 의 단일 인코더로 만든다. 폐쇄루프 에이전트도 같은 인코더를 쓰므로 학습/추론 입력이 일치한다.

저장: <out-dir>/<token>.pt  {'hidden': (L,1536) bf16, 'attn_mask': (L,) uint8}
  token = f"{folder}_{frame_idx}" (models/loader/bench2drive/dataset.py get_data_info 의 token 과 동일, folder 에 '/' 포함 → 하위 폴더 생성)
용량: 표본당 ~3.7MB(L≈1159 가정) x 242,403(train)+val → 약 0.9TB. NAS 에 생성할 것.
GPU 1장(CUDA_VISIBLE_DEVICES=0), 기존 실측 nuScenes 34,149 표본 ≈ 11h 기준 B2D 전체 ≈ 3.4일 추정(샤딩 가능).

사용 예(학습 중인 GPU 와 겹치지 않을 때, 사용자 승인 후):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tools/qwen_cache_b2d.py \
      --ann-files data/infos/b2d_infos_train.pkl data/infos/b2d_infos_val.pkl \
      --out-dir data/bench2drive_qwen_l18 --shard 0/1
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# models/ 패키지를 통째로 import 하면 mmdet3d_plugin 전체(상대 import)가 끌려오므로 파일 단위로 적재한다.
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "qwen_online", str(Path(__file__).resolve().parents[1] / "models" / "encoder" / "vlm_qwen2vl" / "qwen_online.py"))  # 수정(2026-10-15): 경로 이동
_qo = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_qo)
QwenOnlineEncoder, B2D_CAM_ORDER = _qo.QwenOnlineEncoder, _qo.B2D_CAM_ORDER


def token_of(info):
    # models/loader/bench2drive/dataset.py get_data_info 와 같은 규칙(수정 2026-10-09)
    return info.get("sample_token") or f"{info['folder']}_{info['frame_idx']}"


def load_b2d_cams(info, data_root):
    imgs = []
    for cam in B2D_CAM_ORDER:
        p = os.path.join(data_root, info["sensors"][cam]["data_path"])
        if not os.path.exists(p):
            # 검은 영상 대체 금지: 일부 표본만 다른 입력을 받으면 캐시가 결정적이지 않다.
            raise FileNotFoundError(f"B2D 영상 누락: {p}")
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-path", default="/home1/irteam/data-vol1/src/HiP-AD/ckpts/qwen2-vl-2b")
    ap.add_argument("--data-root", default="data/bench2drive")
    ap.add_argument("--ann-files", nargs="+",
                    default=["data/infos/b2d_infos_train.pkl", "data/infos/b2d_infos_val.pkl"])
    ap.add_argument("--out-dir", default="data/bench2drive_qwen_l18")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--shard", default="0/1", help='"i/n": 토큰을 stride n 으로 i 번째만 처리')
    ap.add_argument("--device", default="cuda")
    #수정(2026-10-09): 학습 기동 전 캐시 완비 확인(GPU·Qwen 적재 없음). 누락이 있으면 몇 시간 학습 후 실패하는 대신 지금 실패한다.
    ap.add_argument("--verify", action="store_true", help="생성하지 않고 전 token 의 캐시 존재·형식만 검사")
    args = ap.parse_args()

    infos = []
    for af in args.ann_files:
        d = pickle.load(open(af, "rb"))
        infos.extend(d["infos"] if isinstance(d, dict) else d)
    seen, uniq = set(), []
    for i in infos:
        t = token_of(i)
        if t not in seen:
            seen.add(t); uniq.append(i)
    if args.verify:
        missing = [token_of(i) for i in uniq if not os.path.exists(os.path.join(args.out_dir, f"{token_of(i)}.pt"))]
        print(f"verify: tokens {len(uniq)} missing {len(missing)} (예: {missing[:3]})", flush=True)
        if not missing and uniq:
            d = torch.load(os.path.join(args.out_dir, f"{token_of(uniq[0])}.pt"), map_location="cpu")
            L = d["hidden"].shape[0]
            print(f"  형식 표본: hidden {tuple(d['hidden'].shape)} {d['hidden'].dtype}, layer {args.layer} 가정, L={L}")
            if d["hidden"].shape[1] != 1536 or d["hidden"].dtype != torch.bfloat16:
                sys.exit("형식 불일치")
        sys.exit(1 if missing else 0)
    si, sn = (int(x) for x in args.shard.split("/"))
    uniq = uniq[si::sn]
    print(f"unique tokens (shard {si}/{sn}): {len(uniq)}", flush=True)

    enc = QwenOnlineEncoder(args.qwen_path, layer=args.layer, device=args.device)
    t0, saved, existed = time.time(), 0, 0
    for k, info in enumerate(uniq):
        out = os.path.join(args.out_dir, f"{token_of(info)}.pt")
        if os.path.exists(out):
            # 재개용: 이미 완성된 파일만 건너뛴다(내용이 결정적이므로 재계산 결과와 동일).
            existed += 1
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        hidden, mask = enc.encode(load_b2d_cams(info, args.data_root))
        tmp = out + ".tmp"
        torch.save({"hidden": hidden[0].cpu().contiguous(), "attn_mask": mask[0].cpu()}, tmp)
        os.replace(tmp, out)   # 중단 시 반쪽 파일이 남지 않도록 원자적 교체
        saved += 1
        if (k + 1) % 100 == 0:
            el = time.time() - t0
            print(f"[{k+1}/{len(uniq)}] saved {saved} existed {existed} "
                  f"elapsed {el/3600:.2f}h ETA {el/(k+1)*(len(uniq)-k-1)/3600:.2f}h", flush=True)
    print(f"DONE saved {saved} existed {existed} total {len(uniq)}", flush=True)


if __name__ == "__main__":
    main()
