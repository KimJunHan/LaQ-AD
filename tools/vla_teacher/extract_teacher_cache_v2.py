# 수정(2026-10-16): 교사 생성 답 캐시 7 폴더 — 교사가 '스스로 생성한 답'(GT 아님)의 은닉 상태를 7 폴더로 나눠 저장.
"""LaQ-AD 교사 캐시 추출 v2 (data/nuscenes/laq_cache, README.md 와 PPT '교사 설계 과정 ③' 설계).

흐름(표본마다):
  1) 프롬프트 = 교사 SFT/GRPO 학습과 같은 입력: system(학습 meta) + 이전·현재 6 캠(12 장, meta frame_order) + user 텍스트
     (지시문·명령·고정폭 자차 상태 줄, sft_common.build_user_text, dropout 없음). 정답 관련 필드는 레코드에서 먼저 지운다
     (strip_gt) — GT 는 이 스크립트 어디에서도 읽지 않는다.
  2) 탐욕 생성(do_sample=False), 정확히 547 토큰(min_new_tokens = max_new_tokens = 547, 종료 토큰 억제) — 배치·왼쪽 패딩,
     시각 인코더는 영상 1 장씩(model.visual.forward 교체: 결과 동일, 배치 전체 시각 어텐션 OOM 회피 — sft_common.embed_inputs 와 같은 이유).
  3) 프롬프트 + 생성 답 전체를 한 번 순전파(sft_common.lm_forward: 학습과 같은 임베딩·M-RoPE 경로)해 layer(기본 18) 은닉 상태를 얻고
     laq_cache_layout 의 고정 위치로 잘라 7 폴더에 저장: {'hidden': (L_폴더, D) bf16, 'attn_mask': (L_폴더,) uint8(전부 1)}.
  4) 표본별 색인 _index/teacher_<T>.jsonl: 생성 원문·id, 파싱 성공 여부, 위치 검사, 순전파-생성 일치율, 종료 예측.
     파싱 실패·위치 불일치 표본도 버리지 않는다(고정 위치로 저장하고 플래그만 남김).
  수정(2026-10-16): 설계 검토 반영 — 2) 의 생성은 문법 제약(answer_grammar.py: TRAIN GT 위치별 허용 집합 ∩ 문법 상태)으로
     한다 → 생성 547 토큰이 위치마다 문법 단위 하나라 고정 위치 자르기가 어긋날 수 없다. 색인에 grammar_ok·layout_ok·
     constrained_decoding 을 남기고, layout_ok=False 가 생기면 그 자리에서 중단(버그), --verify 는 layout_bad>0 이면 실패
     (하드 게이트), 학생 로더(LoadTeacherCacheNusc)는 layout_ok 가 아닌 표본을 거부(예외). 실행 전 QA → SFT → GRPO 전 단계
     fold·분할 해시를 sft_common.verify_teacher_chain 으로 검사하고, 폴더 _meta.json 에 prompt_mode="B"(자차 상태 줄 포함)·
     teacher_chain_verified·constrained_decoding 과 교사별 chain 을 기록한다(config teacher_meta_expect 가 요구).
교사 배정: data/vla_teacher/v2/fold_split.json 의 assignment.teacher_<T>.generate (A: train.B + val.B, B: train.A + val.A).
재개: 색인 줄이 있고 7 파일 모두 모양·dtype 이 맞는 표본만 건너뛴다(mmap 으로 헤더만 읽음). 색인은 있는데 파일이 틀리면 즉시 실패.
쓰기: 같은 디렉터리 임시 파일 → fsync → os.replace(원자적). 기존 data/nuscenes/qwen_l18 은 경로 검사로 쓰기 거부.
검증: --verify (두 교사 배정 합 = train+val 34,149 표본이 7 폴더·색인에 정확히 한 번씩 있는지; --deep 이면 모든 파일 모양까지).

실행(GPU1 전용 — sft_common.require_gpu_env 가 GPU0 거부):
  CUDA_VISIBLE_DEVICES=1 python tools/vla_teacher/extract_teacher_cache_v2.py --teacher A --model <teacher_A 병합 모델>
  CUDA_VISIBLE_DEVICES=1 python tools/vla_teacher/extract_teacher_cache_v2.py --teacher B --model <teacher_B 병합 모델>
  CUDA_VISIBLE_DEVICES="" python tools/vla_teacher/extract_teacher_cache_v2.py --verify [--deep]
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laq_cache_layout as LY  # noqa: E402
import answer_grammar as AG  # noqa: E402  # 수정(2026-10-16): 설계 검토 반영 — 문법 제약 생성
from sft_common import (FOLD_FILE, REPO, build_messages, build_user_text, file_sha256, load_fold_split,  # noqa: E402
                        load_images, load_meta, load_records, lm_forward, make_processor, position_ids_for,
                        require_gpu_env, resolve_data_dir, system_text)
from sft_common import exact_fp32_attention, verify_teacher_chain  # noqa: E402  # 수정(2026-10-16): 설계 검토 반영

TAG = "수정(2026-10-16): 교사 생성 답 캐시 7 폴더"
PROMPT_MODE = "B"   # 수정(2026-10-16): 설계 검토 반영 — 사용자 결정(2026-10-15) B안: 프롬프트(=context 폴더)에 자차 상태 줄 포함
META_FIXED_KEYS = ("L", "hidden_dim", "layer", "answer_tokens", "positions_sha256",   # 수정(2026-10-16): 설계 검토 반영 — 한 폴더에 섞이면 안 되는 값
                   "prompt_mode", "constrained_decoding", "answer_grammar_sha256")
OUT_ROOT = "/home1/irteam/data-vol1/src/HiP-AD/data/nuscenes/laq_cache"
PROTECTED = ("qwen_l18",)                     # 기존 캐시 디렉터리 이름 — 출력 경로에 들어 있으면 거부
GT_KEYS = ("answer", "answer_spans", "loss_mask_spans", "loss_mask_token_spans", "targets", "copy_baseline",
           "intent_weight", "intent_uncertain", "future_complete")


def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def strip_gt(r):
    """정답·정답 유도 필드를 지운 사본(캐시 경로에서 GT 를 읽을 수 없게)."""
    return {k: v for k, v in r.items() if k not in GT_KEYS}


def out_root_checked(p):
    p = Path(p).resolve()
    if any(part in PROTECTED for part in p.parts):
        raise SystemExit(f"[거부] 출력 {p} 이 기존 캐시(qwen_l18) 경로 — 덮어쓰기 금지")
    return p


# ---------------------------------------------------------------- 배정
def parse_parts(parts):
    out = []
    for s in parts:
        split, fold = s.split(".")
        if split not in ("train", "val") or fold not in ("A", "B"):
            raise ValueError(f"배정 항목 형식 오류: {s}")
        out.append((split, fold))
    return out


def assigned_records(data_dir, fold_file, teacher, assign="fold"):
    """반환 [(part 문자열, 레코드)] (part 순서 → 파일 순서). 분할 파일의 장면·표본 수와 데이터가 정확히 같아야 한다.
    수정(2026-10-19): 사용자 승인 D4 — assign="val_all" 이면 교사 한 명이 val 전체(val.A + val.B)를 만든다(교사별 검증 캐시).
      두 교사 모두 val 장면으로 학습하지 않았으므로(fold 는 train·holdout 만 나눔) 어느 교사가 만들어도 val 은 보지 않은 데이터다."""
    d, fp = load_fold_split(fold_file)
    if assign == "val_all":
        parts = [("val", "A"), ("val", "B")]
    else:
        parts = parse_parts(d["assignment"][f"teacher_{teacher}"]["generate"])
    cache, out = {}, []
    for split, fold in parts:
        if split not in cache:
            cache[split] = load_records(data_dir, split)
        scenes = set(d[split][fold]["scenes"])
        rs = [r for r in cache[split] if r["scene_token"] in scenes]
        if len(rs) != d[split][fold]["n_samples"]:
            raise RuntimeError(f"{split}.{fold}: 표본 {len(rs)} != 분할 파일 {d[split][fold]['n_samples']}")
        out += [(f"{split}.{fold}", r) for r in rs]
    toks = [r["token"] for _, r in out]
    if len(set(toks)) != len(toks):
        raise RuntimeError("배정 표본 token 중복")
    return out, d, fp


# ---------------------------------------------------------------- 저장·검사
def atomic_torch_save(obj, path):
    import torch
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def check_file(path, L, D, deep=False):
    """반환 None(정상) 또는 문제 문자열. deep=False 는 mmap 으로 헤더(모양·dtype)만 읽는다."""
    import torch
    try:
        d = torch.load(path, map_location="cpu", weights_only=True, mmap=not deep)
    except Exception as e:  # 파일 손상 = 보고(호출자가 실패 처리)
        return f"읽기 실패 {type(e).__name__}: {e}"
    if set(d) != {"hidden", "attn_mask"}:
        return f"키 {sorted(d)}"
    h, m = d["hidden"], d["attn_mask"]
    if tuple(h.shape) != (L, D) or h.dtype != torch.bfloat16:
        return f"hidden {tuple(h.shape)} {h.dtype} != ({L}, {D}) bf16"
    if tuple(m.shape) != (L,) or m.dtype != torch.uint8:
        return f"attn_mask {tuple(m.shape)} {m.dtype}"
    if deep and (not bool(torch.isfinite(h.float()).all()) or not bool((m == 1).all())):
        return "비유한 값 또는 mask != 1"
    return None


def read_index(path):
    out = {}
    if path.exists():
        with open(path) as f:
            for ln in f:
                if ln.strip():
                    e = json.loads(ln)
                    out[e["token"]] = e
    return out


def update_meta(folder_dir, common, teacher, tinfo):
    """폴더 _meta.json 갱신(잠금 + 원자적 교체). 같은 교사의 checkpoint 가 이미 다르게 기록돼 있으면 거부."""
    folder_dir.mkdir(parents=True, exist_ok=True)
    mp = folder_dir / "_meta.json"
    with open(folder_dir / ".meta.lock", "w") as lk:
        fcntl.lockf(lk, fcntl.LOCK_EX)
        cur = json.load(open(mp)) if mp.exists() else {}
        for k in META_FIXED_KEYS:   # 수정(2026-10-16): 설계 검토 반영
            if k in cur and k in common and cur[k] != common[k]:
                raise SystemExit(f"[거부] {mp}: 기존 {k}={cur[k]} != 이번 {common[k]} — 다른 설정을 한 폴더에 섞지 않는다")
        old = cur.get("teachers", {}).get(teacher)
        if old and old.get("checkpoint_sha") != tinfo.get("checkpoint_sha"):
            raise SystemExit(f"[거부] {mp}: teacher_{teacher} 가 다른 체크포인트({old.get('checkpoint')})로 이미 기록됨")
        cur.update(common)
        cur.setdefault("teachers", {})[teacher] = {**(old or {}), **tinfo}
        cur["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
        tmp = mp.with_name(mp.name + f".tmp{os.getpid()}")
        json.dump(cur, open(tmp, "w"), indent=1, ensure_ascii=False)
        os.replace(tmp, mp)


# ---------------------------------------------------------------- 모델
def patch_visual_per_image(model):
    """generate 가 부르는 model.visual 을 영상 1 장씩 돌리도록 교체(시각 어텐션은 영상별 독립 → 결과 동일)."""
    import torch
    vis = model.visual
    orig = vis.forward

    def fwd(hidden_states, grid_thw):
        sizes = [int(t * h * w) for t, h, w in grid_thw.tolist()]
        if sum(sizes) != hidden_states.shape[0]:
            raise ValueError(f"패치 수 {sum(sizes)} != {hidden_states.shape[0]}")
        return torch.cat([orig(p, grid_thw=g[None]) for p, g in zip(hidden_states.split(sizes), grid_thw)], 0)
    vis.forward = fwd
    vis._laq_orig_forward = orig
    return model


def checkpoint_sha(model_dir):
    """체크포인트 식별 해시: config + teacher_meta + 가중치 파일 이름·크기(대용량 전체 해시는 생략, 크기·이름으로 구분)."""
    h = hashlib.sha256()
    md = Path(model_dir)
    for f in sorted(md.iterdir()):
        if f.is_file() and (f.suffix in (".json",) or f.name.endswith(".safetensors") or f.name.endswith(".bin")):
            h.update(f.name.encode()); h.update(str(f.stat().st_size).encode())
            if f.suffix == ".json":
                h.update(f.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------- 표본 후처리
def analyse_answer(tok, gen_ids, template, grammar):
    """생성 id → 색인용 플래그. 위치는 항상 template(고정) 을 쓰고, 여기서는 '맞는지'만 기록한다.
    # 수정(2026-10-16): 설계 검토 반영: layout_ok = 길이 547 ∧ 문법 제약 수용(위치마다 문법 단위 하나 → 고정 위치 자르기가 맞음) ∧ 파싱 성공 ∧
    문자 구간 → 토큰 위치가 템플릿과 같음. 제약 생성이면 항상 참이어야 하고, 거짓이면 --verify 가 실패한다(학생 로더도 거부)."""
    import common_v2 as V
    e = dict(n_new=len(gen_ids))
    g_ok, g_bad = grammar.accepts([int(i) for i in gen_ids])
    e["grammar_ok"], e["grammar_bad_pos"] = bool(g_ok), int(g_bad)
    text, offs = LY.token_char_offsets(tok, gen_ids)
    e["gen_text"] = text
    e["length_ok"] = len(gen_ids) == LY.ANSWER_TOKENS
    try:
        V.parse_answer(text)
        e["parse_ok"], e["parse_error"] = True, None
    except Exception as ex:   # 파싱 실패는 버리지 않고 기록(설계 결정)
        e["parse_ok"], e["parse_error"] = False, f"{type(ex).__name__}: {ex}"[:300]
    e["canonical"] = tok(text, add_special_tokens=False)["input_ids"] == [int(i) for i in gen_ids]
    lay = False
    if e["parse_ok"] and offs is not None and e["grammar_ok"] and e["length_ok"]:
        try:
            pos = LY.folder_token_positions_from_text(text, offs)
            lay = all(pos[k] == template[k] for k in LY.ANSWER_FOLDERS)
        except ValueError:
            lay = False
    e["layout_ok"] = lay
    e["parse_failure"] = not (e["parse_ok"] and e["layout_ok"] and e["length_ok"])
    e["constrained_decoding"] = AG.CONSTRAINT_NAME   # 수정(2026-10-16): 설계 검토 반영 — 학생 로더가 이 값과 layout_ok 를 검사
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=["A", "B"])
    ap.add_argument("--model", help="병합 교사 모델 디렉터리(teacher_meta.json + processor 포함)")
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--data_dir", default=None, help="기본: 교사 meta 의 data_dir")
    ap.add_argument("--fold_file", default=FOLD_FILE)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--batch", type=int, default=16, help="생성 배치(10-16 실측 batch 16 ≈ 1.33 s/표본)")
    ap.add_argument("--fwd_batch", type=int, default=4, help="은닉 상태 순전파 하위 배치")
    ap.add_argument("--answer_tokens", type=int, default=LY.ANSWER_TOKENS)
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")  # 수정(2026-10-16): 설계 검토 반영
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--max_samples", type=int, default=0, help="스모크 전용: 배정 목록 앞 N 개")
    ap.add_argument("--device", default=None, help="'cpu' = CPU 스모크(CUDA_VISIBLE_DEVICES='')")
    ap.add_argument("--assign", choices=["fold", "val_all"], default="fold",   # 수정(2026-10-19): 사용자 승인 D4
                    help="fold = 분할 파일 배정(학습 캐시), val_all = 이 교사 하나로 val 전체(교사별 검증 캐시, --out_root 별도 필수)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--deep", action="store_true", help="--verify: 모든 파일을 실제로 읽어 모양·유한성 검사")
    a = ap.parse_args()
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    root = out_root_checked(a.out_root)
    if a.assign == "val_all":   # 수정(2026-10-19): 사용자 승인 D4 — 교사별 검증 캐시는 학습 캐시와 다른 폴더에만 쓴다
        if a.teacher is None:
            raise SystemExit("--assign val_all 은 --teacher 필요")
        if root == Path(OUT_ROOT).resolve() or f"valonly_teacher{a.teacher}" not in root.name:
            raise SystemExit(f"[거부] --assign val_all 출력은 '...valonly_teacher{a.teacher}' 폴더여야 한다(학습 캐시 {OUT_ROOT} 와 분리): {root}")
    if a.verify:
        sys.exit(verify(a, root))
    if a.teacher is None or a.model is None:
        raise SystemExit("--teacher 와 --model 필요(검증은 --verify)")
    if a.answer_tokens != LY.ANSWER_TOKENS:
        raise SystemExit(f"--answer_tokens 는 문법 고정 길이 {LY.ANSWER_TOKENS} 이어야 한다")
    run(a, root)


def run(a, root):
    dev = require_gpu_env(allow_cpu=(a.device == "cpu"))
    import torch
    from transformers import Qwen2VLForConditionalGeneration

    meta = load_meta(a.model)
    # 수정(2026-10-16): 설계 검토 반영 — QA → SFT → GRPO(→ 병합) 전 단계 fold·분할 해시·base 출처 검사(어긋나면 SystemExit)
    chain = verify_teacher_chain(meta, a.teacher, a.fold_file)
    print(f"[chain] {json.dumps(chain, ensure_ascii=False)}", flush=True)
    fold_path = Path(a.fold_file) if os.path.isabs(a.fold_file) else REPO / a.fold_file
    fdoc, _ = load_fold_split(a.fold_file)
    if not meta.get("use_prev"):
        raise SystemExit("[거부] 교사 meta use_prev=False — 설계는 이전+현재 12 장 프롬프트")
    data_dir = resolve_data_dir(a.data_dir or meta.get("data_dir"))
    items, fdoc, _ = assigned_records(data_dir, a.fold_file, a.teacher, a.assign)
    n_assigned = len(items)
    if a.max_samples:
        items = items[:a.max_samples]
    items = [(p, strip_gt(r)) for p, r in items]
    for _, r in items:   # 수정(2026-10-16): 설계 검토 반영 — B안(자차 상태 줄 포함) 프롬프트인지 표본마다 확인(섞이면 즉시 실패)
        ego = (r.get("prompt_parts") or {}).get("ego")
        if not ego or ego not in build_user_text(r).split("\n"):
            raise SystemExit(f"[거부] {r['token']}: user 프롬프트에 자차 상태 줄이 없다 — 캐시 프롬프트 모드 {PROMPT_MODE} 위반")
    frame_order = meta.get("frame_order", "prev_first")
    cur_block = range(6, 12) if frame_order == "prev_first" else range(0, 6)

    proc = make_processor(a.model, meta["pixels"])
    tok = proc.tokenizer
    tok.padding_side = "left"
    if len(tok) != meta["vocab_after"]:
        raise SystemExit(f"토크나이저 크기 {len(tok)} != 학습 {meta['vocab_after']}")
    template = LY.answer_template(tok)
    grammar = AG.load_grammar(tok, data_dir=data_dir)   # 수정(2026-10-16): 설계 검토 반영 — TRAIN GT 위치별 허용 집합 ∩ 문법 상태
    print(f"[grammar] {grammar.info}", flush=True)
    ids_of = tok.convert_tokens_to_ids
    IMG, VEND, IMEND, EOT = ids_of("<|image_pad|>"), ids_of("<|vision_end|>"), ids_of("<|im_end|>"), ids_of("<|endoftext|>")

    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=dt, attn_implementation=a.attn)
    model.to("cpu" if dev == "cpu" else "cuda").eval()
    if dev != "cpu":   # 수정(2026-10-16): 설계 검토 반영 — fp32 이면 SDPA math 커널 고정·자기검사(bf16 은 'unchanged')
        print(f"[attn] {exact_fp32_attention(dt)}", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    patch_visual_per_image(model)
    D = int(model.config.hidden_size)
    if not (1 <= a.layer <= model.config.num_hidden_layers):
        raise SystemExit(f"--layer {a.layer} 범위 밖(1..{model.config.num_hidden_layers})")
    rope = meta.get("rope", "mrope")
    if rope != "mrope":
        raise SystemExit("[거부] 교사 rope != mrope — generate() 는 M-RoPE 를 쓰므로 학습·생성·순전파 위치가 어긋난다")

    # 첫 표본으로 context 길이 확정(이후 전 표본 같아야 함)
    r0 = items[0][1]
    sys0 = system_text(r0, meta["system"])
    probe = proc.apply_chat_template(build_messages(sys0, build_user_text(r0), 12), tokenize=False, add_generation_prompt=True)
    if not probe.endswith("<|im_start|>assistant\n"):
        raise RuntimeError("프롬프트가 생성 프롬프트로 끝나지 않는다(정답 혼입 의심)")
    # processor 가 <|image_pad|> 를 영상마다 180 개로 펼치므로 여기선 텍스트만으로 user 구간 길이를 잰다(표본마다 실제 위치로 재검사)
    n_user = len(tok(build_user_text(r0), add_special_tokens=False)["input_ids"])
    L_ctx = LY.NUM_CAMS * LY.TOKENS_PER_CAM + n_user
    Ls = dict(context=L_ctx, **{k: len(template[k]) for k in LY.ANSWER_FOLDERS})

    pos_doc = {k: template[k] for k in LY.ANSWER_FOLDERS}
    pos_sha = sha_text(json.dumps(pos_doc, sort_keys=True) + f"|ctx{L_ctx}|{frame_order}")
    gram = data_dir / "grammar.txt"
    spt = data_dir / "special_tokens.txt"
    instr_set = sorted({(r.get("prompt_parts") or {}).get("instruction", "") for _, r in items})
    tinfo = dict(
        checkpoint=str(Path(a.model).resolve()), checkpoint_sha=checkpoint_sha(a.model),
        teacher_meta_sha256=file_sha256(Path(a.model) / "teacher_meta.json"),
        fold=a.teacher, assign=a.assign, n_assigned=n_assigned,   # 수정(2026-10-19): 사용자 승인 D4
        generate=(["val.A", "val.B"] if a.assign == "val_all" else fdoc["assignment"][f"teacher_{a.teacher}"]["generate"]),
        fold_file=str(fold_path), fold_sha256=fdoc["sha256"],
        system_sha256=sha_text(sys0), instruction_sha256=[sha_text(s) for s in instr_set],
        grammar_sha256=file_sha256(gram) if gram.exists() else None,
        special_tokens_sha256=file_sha256(spt) if spt.exists() else None,
        script_sha256=file_sha256(__file__), layout_sha256=file_sha256(LY.__file__),
        chain=chain, prompt_mode=PROMPT_MODE, answer_grammar=grammar.info,   # 수정(2026-10-16): 설계 검토 반영
        generation=dict(do_sample=False, min_new_tokens=a.answer_tokens, max_new_tokens=a.answer_tokens,
                        batch=a.batch, padding_side="left", visual="per_image", attn=a.attn, dtype=a.dtype,
                        rope=rope, frame_order=frame_order, system=meta["system"],
                        constrained_decoding=AG.CONSTRAINT_NAME),   # 수정(2026-10-16): 설계 검토 반영
        started=datetime.datetime.now().isoformat(timespec="seconds"))
    desc = {"context": f"현재 프레임 6 캠 영상 토큰 1,080(블록 {list(cur_block)}) + user 텍스트 {n_user}(지시문·명령·자차 상태), 시퀀스 순서",
            "detection": "칸별 CLS·SEC·x·y·길이·폭·HEAD (21 × 8)", "tracking": "칸별 MOTION (5 × 8)",
            "motion": "칸별 1/2/3 s 변위 (30 × 8)", "map": "Map 줄 68 토큰(<free> 포함)",
            "plan": "칸별 RISK 8 + <free>DD.D 5 (지도 줄과 공유)", "action": "Intent 2 + Action 6"}
    for k in LY.FOLDERS:
        common = dict(folder=k, L=Ls[k], hidden_dim=D, dtype="bfloat16", layer=a.layer, answer_tokens=a.answer_tokens,
                      positions_sha256=pos_sha, description=desc[k], format="{'hidden': (L, D) bf16, 'attn_mask': (L,) uint8}",
                      answer_positions=(template[k] if k != "context" else None), source="teacher-generated answer (GT 아님)",
                      created_by=TAG,
                      # 수정(2026-10-16): 설계 검토 반영 — config teacher_meta_expect 가 이 세 키를 요구한다
                      prompt_mode=PROMPT_MODE, ego_in_prompt=True, teacher_chain_verified=True,
                      constrained_decoding=AG.CONSTRAINT_NAME, answer_grammar_sha256=grammar.info["sha256"])
        update_meta(root / k, common, a.teacher, tinfo)

    idx_dir = root / "_index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    idx_path = idx_dir / f"teacher_{a.teacher}.jsonl"
    index = read_index(idx_path)
    todo, n_skip = [], 0
    for part, r in items:
        t = r["token"]
        files = [root / k / f"{t}.pt" for k in LY.FOLDERS]
        if t in index:
            if index[t].get("teacher") != a.teacher:
                raise SystemExit(f"[거부] 색인 {t} 의 교사 {index[t].get('teacher')} != {a.teacher}")
            bad = [(f.name, f.parent.name, check_file(f, Ls[f.parent.name], D)) for f in files if not f.exists() or
                   check_file(f, Ls[f.parent.name], D)]
            if bad:
                raise SystemExit(f"[실패] 색인에 완료로 기록된 {t} 의 파일 이상: "
                                 f"{[(d, (m or '없음')) for _, d, m in bad]} — 손상 원인 확인 후 수동 조치(자동 덮어쓰기 안 함)")
            n_skip += 1
            continue
        todo.append((part, r))   # 색인 없음 = 미완(파일 일부가 있으면 같은 교사로 다시 만들어 교체)
    print(f"[laq] teacher_{a.teacher} 배정 {n_assigned} (이번 {len(items)}) | 완료 건너뜀 {n_skip} | 할 일 {len(todo)} | "
          f"L {Ls} | layer {a.layer} | D {D} | {dev} {a.dtype} {a.attn} | -> {root}", flush=True)

    fidx = open(idx_path, "a")
    t0, done, n_fail = time.time(), 0, 0
    gen_kw = dict(do_sample=False, min_new_tokens=a.answer_tokens, max_new_tokens=a.answer_tokens,
                  eos_token_id=[IMEND, EOT], pad_token_id=tok.pad_token_id, use_cache=True)
    from transformers import LogitsProcessorList   # 수정(2026-10-16): 설계 검토 반영
    for s in range(0, len(todo), a.batch):
        chunk = todo[s:s + a.batch]
        texts, imgs = [], []
        for _, r in chunk:
            sysm = system_text(r, meta["system"])
            if sysm != sys0:
                raise RuntimeError(f"{r['token']}: system 문장이 표본마다 다르다")
            texts.append(proc.apply_chat_template(build_messages(sysm, build_user_text(r), 12), tokenize=False,
                                                  add_generation_prompt=True))
            imgs.extend(load_images(r, True, frame_order))   # 영상 없으면 FileNotFoundError(대체 금지)
        enc = proc(text=texts, images=imgs, return_tensors="pt", padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        enc["pixel_values"] = enc["pixel_values"].to(dt)
        P = enc["input_ids"].shape[1]
        with torch.no_grad():   # 수정(2026-10-16): 설계 검토 반영 — 문법 제약 생성(행마다 상태)
            out = model.generate(**enc, **gen_kw, logits_processor=LogitsProcessorList(
                [AG.make_logits_processor(grammar, P, len(chunk))]))
        gen = out[:, P:]
        if gen.shape[1] != a.answer_tokens:
            raise RuntimeError(f"생성 길이 {gen.shape[1]} != {a.answer_tokens} (강제 고정 길이 위반)")
        full = torch.cat([enc["input_ids"], gen], 1)
        am = torch.cat([enc["attention_mask"], torch.ones_like(gen)], 1)
        results = []
        for b0 in range(0, len(chunk), a.fwd_batch):
            sl = slice(b0, b0 + a.fwd_batch)
            g = enc["image_grid_thw"]
            per = [int(t * h * w) for t, h, w in g.tolist()]
            i0, i1 = 12 * b0, 12 * min(b0 + a.fwd_batch, len(chunk))
            pv = enc["pixel_values"][sum(per[:i0]):sum(per[:i1])]
            batch = dict(input_ids=full[sl], attention_mask=am[sl], pixel_values=pv, image_grid_thw=g[i0:i1])
            cap = {}
            hk = model.model.layers[a.layer - 1].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0]))
            with torch.no_grad():
                o = lm_forward(model, batch, rope)
            hk.remove()
            H = cap["h"]
            last = o.last_hidden_state
            for j in range(H.shape[0]):
                bi = b0 + j
                ctx_pos, (ua, ub) = LY.context_positions(full[bi, :P].tolist(), IMG, VEND, IMEND, cur_block)
                if len(ctx_pos) != L_ctx:
                    raise RuntimeError(f"{chunk[bi][1]['token']}: context {len(ctx_pos)} != {L_ctx} (프롬프트 길이 비고정)")
                if tok.decode(full[bi, ua:ub].tolist()) != build_user_text(chunk[bi][1]):
                    raise RuntimeError(f"{chunk[bi][1]['token']}: context 텍스트 구간이 user 텍스트와 다르다")
                with torch.no_grad():   # 순전파 ↔ 생성 일치(같은 위치 로짓 argmax = 생성 토큰) · 547 뒤 종료 예측
                    lg = model.lm_head(last[j, P - 1:P - 1 + a.answer_tokens + 1])
                    am_ = lg.argmax(-1)
                    # 수정(2026-10-16): 설계 검토 반영: 일치는 '제약 안 argmax' 로 잰다(생성과 같은 규칙). 제약 없는 argmax 일치율은 따로
                    #   기록 = 모델이 스스로 문법을 지킨 비율(제약이 실제로 바꾼 위치의 반대).
                    gi, gv = grammar.gather_index(grammar.replay_keys(gen[bi].tolist()), lg.device)
                    amc = gi.gather(1, lg[:a.answer_tokens].gather(1, gi).masked_fill(~gv, float("-inf")).argmax(-1, keepdim=True))[:, 0]
                agree = float((amc == gen[bi]).float().mean())
                free_agree = float((am_[:a.answer_tokens] == gen[bi]).float().mean())
                would_end = int(am_[-1]) in (IMEND, EOT)
                hs = {"context": H[j, ctx_pos]}
                for k in LY.ANSWER_FOLDERS:
                    hs[k] = H[j, [P + q for q in template[k]]]
                results.append((bi, {k: v.to("cpu", torch.bfloat16).contiguous() for k, v in hs.items()}, agree, would_end,
                                free_agree))
        for bi, hs, agree, would_end, free_agree in results:
            part, r = chunk[bi]
            e = dict(token=r["token"], teacher=a.teacher, part=part, scene_token=r["scene_token"])
            e.update(analyse_answer(tok, gen[bi].tolist(), template, grammar))
            e.update(fwd_gen_agree=round(agree, 5), unconstrained_agree=round(free_agree, 5),  # 수정(2026-10-16): 설계 검토 반영
                     would_end_after=would_end, gen_ids=gen[bi].tolist())
            for k in LY.FOLDERS:
                if tuple(hs[k].shape) != (Ls[k], D):
                    raise RuntimeError(f"{r['token']} {k}: {tuple(hs[k].shape)} != ({Ls[k]}, {D})")
                atomic_torch_save({"hidden": hs[k], "attn_mask": torch.ones(Ls[k], dtype=torch.uint8)},
                                  root / k / f"{r['token']}.pt")
            fidx.write(json.dumps(e, ensure_ascii=False) + "\n")   # 7 파일 저장 뒤에만 색인 = 완료 표시
            fidx.flush(); os.fsync(fidx.fileno())
            done += 1; n_fail += int(e["parse_failure"])
            if not e["layout_ok"]:   # 수정(2026-10-16): 설계 검토 반영 — 제약 생성에서 위치 불일치는 버그: 저장은 했지만(색인에 표시) 즉시 중단해 원인 확인
                raise RuntimeError(f"{r['token']}: layout_bad (grammar_ok={e['grammar_ok']} pos={e['grammar_bad_pos']}, "
                                   f"parse={e['parse_error']}) — 문법 제약 생성인데 위치가 어긋났다. 원인 확인 전 진행 금지")
        el = time.time() - t0
        print(f"  [{s + len(chunk)}/{len(todo)}] {el / max(done, 1):.2f}s/표본 파싱실패 {n_fail} "
              f"ETA {(len(todo) - done) * el / max(done, 1) / 3600:.2f}h", flush=True)
    fidx.close()
    index = read_index(idx_path)
    mine = [index[r["token"]] for _, r in items if r["token"] in index]
    summ = dict(n_done=len(mine), n_parse_failure=sum(e["parse_failure"] for e in mine),
                n_parse_error=sum(not e["parse_ok"] for e in mine), n_layout_bad=sum(not e.get("layout_ok", False) for e in mine),
                n_grammar_bad=sum(not e.get("grammar_ok", False) for e in mine),   # 수정(2026-10-16): 설계 검토 반영
                unconstrained_agree_mean=(sum(e.get("unconstrained_agree", 0.0) for e in mine) / max(len(mine), 1)),
                n_noncanonical=sum(not e["canonical"] for e in mine), n_would_end=sum(e["would_end_after"] for e in mine),
                fwd_gen_agree_min=min((e["fwd_gen_agree"] for e in mine), default=None),
                finished=datetime.datetime.now().isoformat(timespec="seconds"))
    for k in LY.FOLDERS:
        update_meta(root / k, dict(folder=k, L=Ls[k], hidden_dim=D, layer=a.layer, answer_tokens=a.answer_tokens,
                                   positions_sha256=pos_sha, prompt_mode=PROMPT_MODE, constrained_decoding=AG.CONSTRAINT_NAME,
                                   answer_grammar_sha256=grammar.info["sha256"]), a.teacher, {**tinfo, **summ})
    print(f"DONE teacher_{a.teacher} 이번 저장 {done} ({time.time() - t0:.0f}s) | 요약 {summ}", flush=True)


def verify(a, root):
    """두 교사 배정 합 전체(또는 --teacher/--max_samples 로 좁힌 집합)가 7 폴더·색인에 정확히 있는지. 반환 종료 코드."""
    data_dir = resolve_data_dir(a.data_dir)
    exp = {}
    for T in (["A", "B"] if a.teacher is None else [a.teacher]):
        items, _, _ = assigned_records(data_dir, a.fold_file, T, a.assign)
        if a.max_samples:
            items = items[:a.max_samples]
        for p, r in items:
            if r["token"] in exp:
                raise RuntimeError(f"{r['token']} 가 두 교사에 배정됨")
            exp[r["token"]] = T
    print(f"[verify] 기대 표본 {len(exp)} (A {sum(v == 'A' for v in exp.values())}, B {sum(v == 'B' for v in exp.values())})", flush=True)
    problems = {}
    metas = {}
    for k in LY.FOLDERS:
        mp = root / k / "_meta.json"
        if not mp.exists():
            problems.setdefault("meta_missing", []).append(k); continue
        metas[k] = json.load(open(mp))
    idx = {}
    for T in ("A", "B"):
        for t, e in read_index(root / "_index" / f"teacher_{T}.jsonl").items():
            if t in idx:
                problems.setdefault("index_dup_across_teachers", []).append(t)
            idx[t] = e
    for t, T in exp.items():
        if t not in idx:
            problems.setdefault("index_missing", []).append(t)
        elif idx[t]["teacher"] != T:
            problems.setdefault("wrong_teacher", []).append(t)
    sizes = {}
    for k in LY.FOLDERS:
        if k not in metas:
            continue
        L, D = metas[k]["L"], metas[k]["hidden_dim"]
        present = {p.stem for p in (root / k).glob("*.pt")}
        extra = present - set(exp) if (a.teacher is None or a.assign == "val_all") and not a.max_samples else set()
        if extra:
            problems.setdefault(f"{k}_extra", []).extend(sorted(extra))
        for t in exp:
            f = root / k / f"{t}.pt"
            if t not in present:
                problems.setdefault(f"{k}_missing", []).append(t); continue
            if a.deep:
                m = check_file(f, L, D, deep=True)
            else:   # 빠른 검사: 첫 파일은 헤더 검사, 나머지는 같은 크기인지(torch.save 크기는 모양·키가 같으면 일정)
                sz = f.stat().st_size
                if k not in sizes:
                    m = check_file(f, L, D)
                    sizes[k] = sz if m is None else None
                else:
                    m = None if sz == sizes[k] else (check_file(f, L, D) or None)
            if m:
                problems.setdefault(f"{k}_bad", []).append(f"{t}: {m}")
    pf = [e for t, e in idx.items() if t in exp]
    # 수정(2026-10-16): 설계 검토 반영 — layout_bad(위치 불일치)·문법 제약 아닌 색인 줄 = 하드 게이트(하나라도 있으면 실패)
    lay_bad = [e["token"] for e in pf if not e.get("layout_ok", False)]
    unconstrained = [e["token"] for e in pf if e.get("constrained_decoding") != AG.CONSTRAINT_NAME]
    stats = dict(n_index=len(pf), parse_failure=sum(e["parse_failure"] for e in pf),
                 parse_error=sum(not e["parse_ok"] for e in pf), layout_bad=len(lay_bad),
                 grammar_bad=sum(not e.get("grammar_ok", False) for e in pf), not_constrained=len(unconstrained),
                 noncanonical=sum(not e["canonical"] for e in pf), would_end=sum(e["would_end_after"] for e in pf))
    print(f"[verify] 색인 통계 {stats}", flush=True)
    if lay_bad:
        problems["layout_bad"] = lay_bad
    if unconstrained:
        problems["index_not_constrained"] = unconstrained
    for k, m in metas.items():
        for kk, vv in (("prompt_mode", PROMPT_MODE), ("teacher_chain_verified", True),
                       ("constrained_decoding", AG.CONSTRAINT_NAME)):
            if m.get(kk) != vv:
                problems.setdefault("meta_key_mismatch", []).append(f"{k}.{kk}={m.get(kk)!r} (기대 {vv!r})")
    if problems:
        for k, v in problems.items():
            print(f"[verify] 문제 {k}: {len(v)} (예: {v[:3]})", flush=True)
        print("[verify] FAIL", flush=True)
        return 1
    print(f"[verify] OK — {len(exp)} 표본 × 7 폴더 완전(deep={a.deep})", flush=True)
    return 0


if __name__ == "__main__":
    main()
