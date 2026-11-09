# 수정(2026-10-15): 교사 VLM ② LoRA SFT — 병합 교사 모델의 layer-18 은닉상태 캐시(기존 data/nuscenes/qwen_l18 과 같은 형식).
"""교사 캐시 추출.

형식: <out_dir>/<token>.pt = {'hidden': (L, 1536) bf16, 'attn_mask': (L,) uint8} — tools/qwen_cache.py 와 동일.
      L = 1,159 (6 캠 × 180 영상 토큰 + 템플릿/지시문 79) — 기존 캐시와 같은 토큰 배치를 검사(--expect_len)한다.
경로: 기본 data/nuscenes/qwen_teacher_l18/ (기존 qwen_l18 는 절대 덮어쓰지 않는다 — 같은 경로면 거부).
입력: 레코드(teacher_gt_{train,val}.jsonl, train+val 34,149 토큰 = 기존 캐시 토큰 집합)의 6 캠(F,FL,FR,BL,B,BR — qwen_cache 와 같은 순서),
      processor min=max_pixels=200·28·28, bf16, eager attention(기존 캐시와 같은 수치 경로).

프롬프트 선택(--prompt):
  cache(기본·권장) : tools/qwen_cache.py 의 INSTRUCTION(글자 그대로, AST 로 읽고 qwen_online.py 사본과 일치 검사), system 없음
                     (= Qwen 기본 "You are a helpful assistant."). 이유:
                     (1) 토큰 배치 1,159 가 기존과 동일 → 학생 융합 코드·마스크·위치 가정 무변경, A/B 비교가 '가중치만' 다르다.
                     (2) 인과 마스크라 영상 토큰(1,080개)의 은닉상태는 그 앞 접두(system+'<|im_start|>user\\n')와 영상에만
                         의존하고 뒤따르는 지시문과 무관하다. SFT 를 --system qwen_default(기본)로 학습하면 이 접두가
                         SFT·생성·캐시에서 비트 단위로 같아, 교사가 배운 영상 표현이 캐시에 그대로 실린다.
                     (3) 교사 지시문에는 명령·자차상태 텍스트가 있어 캐시에 넣으면 학생이 이미 받는 입력이 중복되고
                         (지름길 강화), 길이가 ~1,700 으로 바뀌어 기존 캐시와 비교가 깨진다.
                     한계: 지시문 뒤 79 개 텍스트 토큰은 교사가 학습 때 본 지시문과 달라 분포가 약간 이동할 수 있다.
  teacher          : 교사 학습 프롬프트 전체(학습 meta 의 system, 이전+현재 12 캠, 명령·자차상태) — 길이가 달라
                     --expect_len 0 필요. 어느 모드든 정답은 넣지 않는다(teacher-forcing 금지; 입력은 생성 프롬프트까지).
  cache_ego        : 수정(2026-10-15) — cache 지시문 뒤에 '\n' + 자차 상태 줄(레코드 prompt_parts['ego'], common_v2 고정폭 — 캐시 문맥 +41 토큰, L = 1,200 실측)을 붙인다.
                     영상·지시문 접두는 cache 와 같아 인과 마스크상 그 은닉상태는 cache 와 같고, 자차 정보는 뒤 텍스트 토큰에 실린다.
                     길이 L = 1,159 + Δ (Δ 는 첫 표본에서 측정한 뒤 전 표본 동일 검사 — 다르면 즉시 실패).
  학생 융합 기본 = cache_ego (사용자 결정 2026-10-15, B안: 교사 토큰에 언어 경유 자차 상태 포함. 계획 결과는 w/ ego status 범주).
  cache(자차 상태 없음, A)는 소거 비교용으로 유지 — 종전 기본값(10-15 오후)이었다.
  주의: 교사는 SFT 때 현재 프레임 토큰 앞에 이전 프레임 6 캠(prev_first)을 보았다. cache 모드는 현재 6 캠만 넣으므로
        현재 프레임 영상 토큰의 문맥(앞쪽 접두)이 학습 때와 다르다 — frame_order=cur_first 로 학습하면 현재 프레임 토큰의
        접두가 cache 모드와 비트 단위로 같아진다(열린 결정).
위치(--rope): 기본은 학습 meta 의 rope(기본 mrope). 기존 qwen_l18 은 transformers 4.46.3 forward 가 M-RoPE 를 만들지 않아
      사실상 1d 위치로 계산되었다(tools/qwen_cache.py / qwen_online.py). M-RoPE 로 학습한 교사는 M-RoPE 로 추출해야
      학습 때 표현과 일치한다. 온라인 추론(qwen_online.QwenOnlineEncoder)에 교사를 쓰려면 같은 위치 규칙을 맞춰야 한다.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_common import (NUM_CAMS, REPO, build_messages, build_user_text, load_images, load_meta, load_records,  # noqa: E402
                        make_processor, original_cache_instruction, position_ids_for, require_gpu_env, resolve_data_dir,
                        system_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="병합 교사 모델 디렉터리")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--out_dir", default=None, help="기본: cache_ego → qwen_teacher_ego_l18, 그 외 → qwen_teacher_l18")  # 수정(2026-10-15): 모드별 분리
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--prompt", choices=["cache_ego", "cache", "teacher"], default="cache_ego")  # 수정(2026-10-15): B안 기본
    ap.add_argument("--rope", choices=["meta", "mrope", "1d"], default="meta")
    ap.add_argument("--expect_len", type=int, default=-1,
                    help="-1 = 모드 기본(cache 1159, cache_ego 첫 표본 측정 후 고정), 0 = 검사 안 함(--prompt teacher)")  # 수정(2026-10-15)
    ap.add_argument("--ref_dir", default="data/nuscenes/qwen_l18", help="형식 대조용 기존 캐시(읽기 전용)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max_samples", type=int, default=0, help="스모크 전용")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    dev = require_gpu_env(allow_cpu=(a.device == "cpu"))
    import torch
    from transformers import Qwen2VLForConditionalGeneration

    if a.out_dir is None:  # 수정(2026-10-15): 자차 상태 유무가 다른 캐시를 한 디렉터리에 섞지 않는다
        a.out_dir = "data/nuscenes/qwen_teacher_ego_l18" if a.prompt == "cache_ego" else "data/nuscenes/qwen_teacher_l18"
    if a.expect_len == -1:
        a.expect_len = {"cache": 1159, "cache_ego": 1200, "teacher": 0}[a.prompt]  # cache_ego 1,200 = 1,159 + 41(전 표본 실측)
    out = Path(a.out_dir) if os.path.isabs(a.out_dir) else REPO / a.out_dir
    ref = Path(a.ref_dir) if os.path.isabs(a.ref_dir) else REPO / a.ref_dir
    if out.resolve() == ref.resolve() or out.name == "qwen_l18":
        raise SystemExit(f"[거부] 출력 {out} 이 기존 캐시와 같다 — 새 디렉터리를 쓸 것")
    out.mkdir(parents=True, exist_ok=True)
    meta = load_meta(a.model)
    rope = meta.get("rope", "mrope") if a.rope == "meta" else a.rope
    data_dir = resolve_data_dir(meta.get("data_dir"))
    recs = []
    for s in a.splits:
        recs += load_records(data_dir, s)
    seen, uniq = set(), []
    for r in recs:
        if r["token"] not in seen:
            seen.add(r["token"]); uniq.append(r)
    si, sn = (int(x) for x in a.shard.split("/"))
    uniq = uniq[si::sn]
    if a.max_samples:
        uniq = uniq[:a.max_samples]

    instr = original_cache_instruction()
    model = Qwen2VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                            attn_implementation="eager")
    model.to("cpu" if dev == "cpu" else "cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    proc = make_processor(a.model, meta["pixels"])
    ref_files = sorted(ref.glob("*.pt"))[:1] if ref.exists() else []
    ref_d = torch.load(ref_files[0], map_location="cpu") if ref_files else None
    print(f"[cache] {len(uniq)} 토큰 → {out} | layer {a.layer} | prompt {a.prompt} | rope {rope} | ref {ref_files[:1]}", flush=True)

    t0, saved, skipped = time.time(), 0, 0
    for k, r in enumerate(uniq):
        hp = out / f"{r['token']}.pt"
        if hp.exists():
            skipped += 1          # 재개: 이미 만든 파일(원자적 저장이라 완결본만 존재)
            continue
        imgs = load_images(r)                     # 현재 프레임 6 캠(F,FL,FR,BL,B,BR)
        if a.prompt in ("cache", "cache_ego"):
            utext = instr
            if a.prompt == "cache_ego":  # 수정(2026-10-15): B안 — 자차 상태 줄을 지시문 뒤에(교사 SFT 프롬프트와 같은 순서: 영상 → 텍스트)
                ego = (r.get("prompt_parts") or {}).get("ego")
                if not ego or not ego.startswith("Ego state:"):
                    raise RuntimeError(f"{r['token']}: 자차 상태 줄 없음(v2 레코드, ego_in_prompt=current 필요)")
                utext = instr + "\n" + ego
            msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs] +
                     [{"type": "text", "text": utext}]}]
        else:
            if meta.get("use_prev"):
                imgs = load_images(r, True, meta.get("frame_order", "prev_first"))
            msgs = build_messages(system_text(r, meta["system"]), build_user_text(r), len(imgs))
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # 정답을 붙여 teacher-forcing 하지 않는다: 입력은 생성 프롬프트('<|im_start|>assistant\n' 로 끝남)뿐.
        if not text.endswith("<|im_start|>assistant\n"):
            raise RuntimeError("캐시 입력이 생성 프롬프트로 끝나지 않는다(정답 혼입 의심)")
        inp = proc(text=[text], images=imgs, return_tensors="pt", padding=True)
        inp = {kk: v.to(model.device) for kk, v in inp.items()}
        pos = position_ids_for(model, inp["input_ids"], inp["image_grid_thw"], inp["attention_mask"], rope)
        with torch.no_grad():
            o = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                      pixel_values=inp["pixel_values"].to(torch.bfloat16), image_grid_thw=inp["image_grid_thw"],
                      position_ids=pos, output_hidden_states=True, return_dict=True)
        h = o.hidden_states[a.layer].squeeze(0).cpu().to(torch.bfloat16).contiguous()
        m = inp["attention_mask"].squeeze(0).cpu().to(torch.uint8)
        if a.expect_len is None:  # 수정(2026-10-15): cache_ego 길이는 첫 표본에서 확정하고 이후 전 표본이 같아야 한다
            a.expect_len = int(h.shape[0])
            print(f"[cache_ego] L = {a.expect_len} (cache 1159 + {a.expect_len - 1159})", flush=True)
        if a.expect_len and h.shape[0] != a.expect_len:
            raise RuntimeError(f"{r['token']}: 길이 {h.shape[0]} != {a.expect_len} (토큰 배치가 기존 캐시와 다르다)")
        if ref_d is not None and k == 0 and a.prompt == "cache_ego":  # 수정(2026-10-15): 영상 접두 + 지시문 1,159 토큰까지는 형식 동일
            assert ref_d["hidden"].shape[1] == h.shape[1] and ref_d["hidden"].dtype == h.dtype
        if ref_d is not None and k == 0 and a.prompt == "cache":
            assert ref_d["hidden"].shape == h.shape and ref_d["hidden"].dtype == h.dtype, (ref_d["hidden"].shape, h.shape)
            assert ref_d["attn_mask"].dtype == m.dtype and ref_d["attn_mask"].shape == m.shape
        tmp = hp.with_suffix(".pt.tmp")
        torch.save({"hidden": h, "attn_mask": m}, tmp)
        os.replace(tmp, hp)
        saved += 1
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  [{k + 1}/{len(uniq)}] saved {saved} skipped {skipped} {el / max(saved, 1):.2f}s/표본 "
                  f"ETA {(len(uniq) - k - 1) * el / max(saved, 1) / 3600:.1f}h", flush=True)
    print(f"DONE saved {saved} skipped {skipped} {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
