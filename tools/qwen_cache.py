"""수정: VLA-v1 — 각 sample (train + val) 에 대해 Qwen2-VL hidden states pre-compute → 디스크 저장.
이후 학습/평가 시 Qwen 메모리 점유 없이 token K/V 사용 가능.

저장 형식 : work_dirs/qwen_cache/<token>.pt  (token 별 1 파일)
각 파일 dict: {'hidden': torch.bfloat16 tensor (L, 1536), 'attn_mask': (L,)}

용량 추정: L≈1100, 1536 dim, bf16 → ~3.4 MB/sample.
  train 323 + val 81 = 404 → ~1.4 GB 디스크.
"""
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import torch
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#수정(2026-09-15): 한 파일 안에서 mmdet3d_plugin 과 projects.mmdet3d_plugin 두 이름이
#  섞여 있으면 같은 패키지가 두 번 로드되어 레지스트리 이중 등록으로 죽는다. 한 이름으로 통일.
import projects.mmdet3d_plugin  # noqa
from projects.mmdet3d_plugin.datasets.nuscenes_mini_dataset import NuScenesMiniDataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


INSTRUCTION = (
    "These are 6 surround-view camera images of an autonomous vehicle (front, "
    "front-left, front-right, back-left, back, back-right). Describe the salient "
    "objects, lane geometry, traffic state, and any driving-relevant cues."
)


def load_six_cams(info, data_root):
    imgs = []
    for cam in ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]:
        c = info["cams"][cam]
        p = c["data_path"]
        if "samples/" in p:
            p = "samples/" + p.split("samples/", 1)[1]
        if not os.path.isabs(p):
            p = os.path.join(data_root, p)
        imgs.append(Image.open(p).convert("RGB") if os.path.exists(p)
                    else Image.new("RGB", (1600, 900), (0, 0, 0)))
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-path", default="/workspace/src/HiP-AD/ckpts/qwen2-vl-2b")
    ap.add_argument("--data-root", default="data/nuscenes")
    ap.add_argument("--ann-files", nargs="+",
                    default=["data/nuscenes/nuscenes_infos_train.pkl",
                             "data/nuscenes/nuscenes_infos_val.pkl"])
    ap.add_argument("--out-dir", default="work_dirs/qwen_cache")
    ap.add_argument("--min-pixels", type=int, default=200 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=200 * 28 * 28)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-samples", type=int, default=None)
    #수정: VLA-FULL — 2-GPU 병렬용 stride 샤딩. "i/n" → dedup 된 uniq 를 n 등분 중 i 번째만 처리.
    #  skip-existing 와 합쳐져 여러 인스턴스 안전 병렬(샤드 겹쳐도 idempotent).
    ap.add_argument("--shard", default="0/1", help='"i/n": uniq 토큰을 stride n 으로 i 번째만 처리')
    ap.add_argument("--layers", default="", help="저장할 hidden_states 계층들(쉼표). 빈값=마지막 계층(종전 동작)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Qwen + processor
    print(f"Loading Qwen from {args.qwen_path} ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.qwen_path, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(
        args.qwen_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    print("Qwen loaded.")

    # 모든 ann 파일 합쳐서 한 번씩 처리
    all_infos = []
    for af in args.ann_files:
        if not os.path.exists(af):
            print(f"skip missing {af}"); continue
        ds = NuScenesMiniDataset(data_root=args.data_root, ann_file=af,
                                  pipeline=None, modality=dict(use_camera=True, use_lidar=False))
        all_infos.extend(ds.data_infos)
    print(f"total samples (train+val merged, dedup by token): {len(all_infos)}")
    # token 기준 dedup
    seen = set(); uniq = []
    for i in all_infos:
        if i["token"] not in seen:
            seen.add(i["token"]); uniq.append(i)
    print(f"unique tokens: {len(uniq)}")

    #수정: VLA-FULL — stride 샤딩. uniq[i::n] (max-samples 보다 먼저 적용)
    _si, _sn = (int(x) for x in args.shard.split("/"))
    if _sn > 1:
        uniq = uniq[_si::_sn]
        print(f"shard {_si}/{_sn} → {len(uniq)} tokens")

    #수정(Run9): --layers "18,14,22" 형식. 미지정 시 -1(마지막) 로 종전과 동일.
    LAYERS = [int(x) for x in (args.layers.split(",") if getattr(args, "layers", "") else ["-1"])]
    OUT_DIRS = []
    for li in LAYERS:
        d = args.out_dir if li == -1 else f"{args.out_dir.rstrip('/')}_l{li}"
        os.makedirs(d, exist_ok=True)
        OUT_DIRS.append(d)
    print("target layers:", LAYERS, "->", OUT_DIRS)

    n_max = args.max_samples or len(uniq)
    t0 = time.time()
    skipped = 0; saved = 0
    for k, info in enumerate(uniq[:n_max]):
        if all(os.path.exists(os.path.join(d, f"{info['token']}.pt")) for d in OUT_DIRS):
            skipped += 1
            continue
        imgs = load_six_cams(info, args.data_root)
        msgs = [{"role": "user", "content":
                    [{"type": "image", "image": im} for im in imgs] +
                    [{"type": "text", "text": INSTRUCTION}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[text], images=imgs, return_tensors="pt", padding=True)
        inp = {k_: v.to(args.device) for k_, v in inp.items()}
        with torch.no_grad():
            out = model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                pixel_values=inp["pixel_values"].to(torch.bfloat16),
                image_grid_thw=inp.get("image_grid_thw"),
                output_hidden_states=True,
                return_dict=True,
            )
        #수정(2026-09-15, Run9): 마지막 계층 대신 지정 계층(들) 저장.
        #  근거 — VLM 분석 연구(arXiv 2606.06890): 시각적으로 접지된 신호는 중간 계층에서
        #  가장 강하고 마지막 계층에서는 언어 prior 에 눌린다(28계층 7B 기준 14~19층).
        #  output_hidden_states=True 라 전 계층이 이미 계산돼 있어 여러 계층을 같은 비용으로
        #  저장한다. hidden_states[0]=임베딩이므로 계층 k 는 index k.
        m = inp["attention_mask"].squeeze(0).cpu().to(torch.uint8)
        for li, odir in zip(LAYERS, OUT_DIRS):
            hp = os.path.join(odir, f"{info['token']}.pt")
            if os.path.exists(hp):
                continue
            h = out.hidden_states[li].squeeze(0).cpu().to(torch.bfloat16).contiguous()
            torch.save({"hidden": h, "attn_mask": m}, hp)
        saved += 1
        if (k + 1) % 10 == 0:
            el = time.time() - t0
            eta = el / (k + 1) * (n_max - k - 1)
            print(f"  [{k+1}/{n_max}] saved {saved}, skipped {skipped}, elapsed {el:.0f}s, ETA {eta:.0f}s")
    print(f"DONE. saved {saved}, skipped {skipped}, total time {time.time()-t0:.0f}s, dir={args.out_dir}")


if __name__ == "__main__":
    main()
