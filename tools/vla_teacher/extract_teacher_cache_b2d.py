# 수정(2026-10-31): B2D 교사 체인 — extract_teacher_cache_v2.py(nuScenes, 수정 안 함)의 Bench2Drive 판(샤드 = NAVSIM 판과 같은 방식).
"""LaQ-AD B2D 교사 캐시 추출(학생 config configs/b2d/v2_teacher/stage{1,2} 의 data/bench2drive/laq_cache, LoadTeacherCacheB2D).

흐름·저장 형식·색인·재개·원자적 쓰기·문법 제약 생성·순전파 검사는 extract_teacher_cache_v2.py 와 같다(그 파일의 도우미를 import).
B2D 정답(v2-sensor-b2d)은 줄·필드·토큰 수가 nuScenes 와 같아(547, 7 폴더 위치 동일) b2d_teacher_shim 으로 common_v2 자리에
common_b2d 를 끼운 뒤 nuScenes 도우미(laq_cache_layout·answer_grammar·extract_teacher_cache_v2)를 그대로 쓴다. B2D 에서 다른 것만:
  1) 입력 = 교사 학습과 같은 프롬프트: system(학습 meta, qwen_default) + 직전 키프레임(0.5 s)·현재 6 캠(12 장, prev_first) + user 텍스트
     (지시문·명령·목표점·자차 상태 줄 = GT 생성기의 prompt.user 그대로, dropout 없음). 화소 200·28·28 → 캠당 180 토큰.
     context = 현재 6 캠 1,080 + user 텍스트(B2D 809, 첫 표본에서 재고 전 표본 같아야 함). 정답 폴더 168/40/240/68/13/8(nuScenes 와 같음).
  2) 문법 = answer_grammar_b2d(B2D TRAIN GT 기본 집합, 행동 K 2040), 파서 = common_b2d.parse_answer(shim).
  3) 표본 = B2D 교사 정답의 모든 2 Hz 키프레임(train 47,346 + val 2,583 = 49,929). 학생 10 Hz 프레임은 로더가 키프레임(나이 ≤ 0.4 s)으로 사상.
     교사 배정 = 분할 파일 assignment(clip 단위): teacher_A → train.B + val.B, teacher_B → train.A + val.A (X.assigned_records: part 별
     표본 수 = 분할 파일 n_samples 정확히 일치).
  4) 파일 = <out_root>/<folder>/<clip>/<kf:05d>.pt (token '<clip>/<kf:05d>' 에 '/' 가 있어 clip 하위 폴더를 만든다).
  5) 샤드(--shard k --num_shards n): 배정 목록 [k::n]. 색인 = _index/teacher_<T>.jsonl(n=1) 또는 teacher_<T>.s<k>of<n>.jsonl.
출처 검사: QA → SFT(→ 병합) fold·분할 해시 = sft_common.verify_teacher_chain(B2D kf2hz 분할). meta pixels 200·28·28·use_prev·
  prev_first·mrope·qwen_default, meta.data_dir = B2D 정답 폴더, meta.b2d(train_sft_b2d 기록)·run_id 존재, b2d 출처(GT sha256·특수토큰)
  = 지금 파일. 누설 검사: 배정 clip ∩ train.<교사> = ∅, QA 실제 사용 장면 ⊆ train.<교사>·배정과 겹침 0, SFT hold-out 해시 = 분할 파일.
검증(--verify): 두 교사 배정 합이 7 폴더·색인에 정확히 한 번씩, layout_ok·제약 생성 하드 게이트, _meta 값. --infos 면 학생 config 의
  teacher_loader(LoadTeacherCacheB2D, cache_root 만 바꿈)를 실제로 만들어(색인 키프레임 격자·_meta 기대값 검사) b2d_infos_{train,val}.pkl 의
  모든 10 Hz 프레임이 키프레임(kf ≤ f, f − kf ≤ 4)에 사상되는지 + 교사별 표본 1 개를 로더로 실제 적재.
실행(GPU2 = B2D 전용, sft_common.require_gpu_env 가 GPU0 거부):
  CUDA_VISIBLE_DEVICES=2 python tools/vla_teacher/extract_teacher_cache_b2d.py --teacher A --model <b2d teacher_A 병합> [--shard 0 --num_shards 2]
  CUDA_VISIBLE_DEVICES="" python tools/vla_teacher/extract_teacher_cache_b2d.py --verify [--infos] [--deep]
"""
import argparse
import datetime
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b2d_sft_common as BS  # noqa: E402

B = BS.install_shim()   # 반드시 아래 import 보다 먼저(answer_grammar·laq_cache_layout·extract_teacher_cache_v2 가 common_b2d 를 보게)
import extract_teacher_cache_v2 as X  # noqa: E402  # 도우미(원자적 저장·파일 검사·색인·_meta 갱신·영상별 시각 인코더·체크포인트 해시)
import laq_cache_layout as LY  # noqa: E402
import answer_grammar_b2d as AGB  # noqa: E402
from sft_common import (REPO, build_messages, build_user_text, file_sha256, load_fold_split, load_images, load_meta,  # noqa: E402
                        lm_forward, make_processor, require_gpu_env, system_text, exact_fp32_attention, verify_teacher_chain)

if X.AG.V is not B or X.LY is not LY:
    raise RuntimeError("extract_teacher_cache_v2 가 B2D shim 없이 import 됐다")

TAG = "수정(2026-10-31): B2D 교사 체인"
PROMPT_MODE = X.PROMPT_MODE          # "B" — 프롬프트(=context)에 자차 상태 줄 포함(학생 config teacher_meta_expect)
OUT_ROOT = BS.B2D_CACHE_ROOT
STUDENT_CONFIG = REPO / "configs/b2d/v2_teacher/stage1/train.py"
# 프롬프트에 필요한 입력 필드만 남긴다(허용 목록) — 정답·정답 유도 필드(answer, targets, spans, copy_baseline, modes …)는 읽지 않는다.
KEEP_KEYS = ("token", "scene_token", "timestamp", "frame_idx", "town", "cams", "cams_prev", "prev_available", "prev_token",
             "command", "target_point", "ego_status", "ego_drop_ok", "command_drop_ok", "prompt", "prompt_parts")
N_IMG = 2 * BS.B2D_NUM_CAMS            # 직전 6 + 현재 6
CUR_BLOCK = range(6, 12)               # prev_first


def strip_to_input(r):
    out = {k: r[k] for k in KEEP_KEYS if k in r}
    out["prompt"] = {k: v for k, v in (r.get("prompt") or {}).items() if k in ("system", "user")}
    return out


def out_root_checked(p):
    root = X.out_root_checked(p)
    if "bench2drive" not in root.parts or any(s in ("nuscenes", "navsim") for s in root.parts):
        raise SystemExit(f"[거부] 출력 {root} 이 B2D(bench2drive) 경로가 아니다(nuScenes·NAVSIM 캐시와 섞지 않는다)")
    return root


def assigned(teacher, fold_file):
    """[(part, 입력 레코드)] — X.assigned_records(분할 파일 표본 수 정확히 대조) 뒤 허용 목록으로 줄인다. token 형식 검사."""
    items, d, fp = X.assigned_records(BS.B2D_GT_DIR, fold_file, teacher)
    out = []
    for part, r in items:
        clip, _, kf = r["token"].rpartition("/")
        if clip != r["scene_token"] or len(kf) != 5 or not kf.isdigit() or int(kf) != r["frame_idx"]:
            raise RuntimeError(f"token 형식 오류 {r['token']} (기대 '<clip>/<frame_idx:05d>', scene_token {r['scene_token']})")
        out.append((part, strip_to_input(r)))
    return out, d, fp


def check_meta(meta, chain, d, teacher, items):
    """교사 meta 의 B2D 설계값·출처 + 배정 clip 을 그 교사가 QA·SFT 에서 본 적이 없는지(구조·기록 양쪽)."""
    for k, v in (("pixels", BS.B2D_PIXELS), ("use_prev", True), ("frame_order", "prev_first"), ("rope", "mrope"),
                 ("system", "qwen_default")):
        if meta.get(k) != v:
            raise SystemExit(f"[거부] 교사 meta {k}={meta.get(k)!r} != {v!r} (B2D 교사 설계)")
    if not BS.same_path(meta.get("data_dir", ""), BS.B2D_GT_DIR):
        raise SystemExit(f"[거부] 교사 meta data_dir {meta.get('data_dir')} != {BS.B2D_GT_DIR}")
    rec = meta.get("b2d") or {}
    if rec.get("kind") != "b2d_teacher_gt" or not meta.get("run_id"):
        raise SystemExit("[거부] 교사 meta 에 b2d·run_id 기록 없음 — train_sft_b2d.py 로 학습한 B2D 교사가 아니다")
    try:
        src = BS.check_b2d_sources(BS.B2D_GT_DIR, BS.B2D_FOLD_FILE, special_tokens=meta.get("special_tokens"))
    except RuntimeError as e:
        raise SystemExit(f"[거부] B2D 출처: {e}")
    for k in ("gt_sha256", "fold_sha256", "special_tokens_sha256", "action_k"):
        if rec.get(k) != src[k]:
            raise SystemExit(f"[거부] 교사 SFT 때 b2d.{k}={rec.get(k)!r} != 지금 {src[k]!r} (SFT 뒤 정답·분할이 바뀜)")
    mine = set(d["train"][teacher]["scenes"])
    clips = {r["scene_token"] for _, r in items}
    if clips & mine:
        raise SystemExit(f"[거부] 교사 {teacher} 배정 clip {len(clips & mine)} 개가 train.{teacher}(교사 학습 반쪽)에 있다")
    qm = json.load(open(Path(chain["qa"]["path"]) / "qa_meta.json"))
    ss = qm["scene_sets"]
    if "train_scenes_used" not in ss or "holdout_scenes_used" not in ss:
        raise SystemExit("[거부] QA qa_meta.scene_sets 에 train_scenes_used/holdout_scenes_used 없음")
    qa_used = set(ss["train_scenes_used"]) | set(ss["holdout_scenes_used"])
    if not qa_used <= mine or qa_used & clips:
        raise SystemExit(f"[거부] QA 실제 사용 장면: train.{teacher} 밖 {len(qa_used - mine)} / 배정과 겹침 {len(qa_used & clips)}")
    if meta.get("holdout_sha256") != d["holdout"][teacher]["sha256"]:
        raise SystemExit(f"[거부] SFT hold-out sha256 {str(meta.get('holdout_sha256'))[:12]} != 분할 holdout.{teacher}")
    return dict(assigned_clips=len(clips), teacher_train_clips=len(mine), qa_scenes_used=len(qa_used), overlap=0, sources=src)


def analyse_answer(tok, gen_ids, template, grammar):
    """extract_teacher_cache_v2.analyse_answer 그대로(shim 으로 파서 = common_b2d.parse_answer, 위치·길이 = laq_cache_layout)."""
    return X.analyse_answer(tok, gen_ids, template, grammar)


def index_files(root, teacher):
    return sorted(glob.glob(str(root / "_index" / f"teacher_{teacher}.jsonl")) +
                  glob.glob(str(root / "_index" / f"teacher_{teacher}.s*of*.jsonl")))


def index_name(teacher, shard, num_shards):
    return f"teacher_{teacher}.jsonl" if num_shards == 1 else f"teacher_{teacher}.s{shard}of{num_shards}.jsonl"


def read_all_index(root):
    out = {}
    for p in sorted(glob.glob(str(root / "_index" / "teacher_*.jsonl"))):
        for t, e in X.read_index(Path(p)).items():
            if t in out:
                raise RuntimeError(f"색인 중복 {t} ({p})")
            out[t] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=["A", "B"])
    ap.add_argument("--model", help="B2D 병합 교사 모델 디렉터리(teacher_meta.json + processor)")
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--fold_file", default=str(BS.B2D_FOLD_FILE))
    ap.add_argument("--grammar", default=AGB.MASK_FILE)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fwd_batch", type=int, default=4)
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")   # nuScenes·NAVSIM 캐시와 같은 기본(bf16 생성·bf16 저장)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=0, help="스모크 전용: (샤드) 배정 목록 앞 N 개")
    ap.add_argument("--device", default=None, help="'cpu' = CPU 스모크(CUDA_VISIBLE_DEVICES='')")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--deep", action="store_true", help="--verify: 모든 파일을 실제로 읽어 모양·유한성 검사")
    ap.add_argument("--infos", action="store_true", help="--verify: 학생 로더로 b2d_infos 의 모든 10 Hz 프레임 키프레임 사상 검사")
    a = ap.parse_args()
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    root = out_root_checked(a.out_root)
    if not BS.same_path(a.fold_file, BS.B2D_FOLD_FILE):
        raise SystemExit(f"[거부] --fold_file {a.fold_file} != {BS.B2D_FOLD_FILE}")
    if not (0 <= a.shard < a.num_shards):
        raise SystemExit(f"--shard {a.shard} 범위 밖(0..{a.num_shards - 1})")
    if a.verify:
        sys.exit(verify(a, root))
    if a.teacher is None or a.model is None:
        raise SystemExit("--teacher 와 --model 필요(검증은 --verify)")
    run(a, root)


def run(a, root):
    dev = require_gpu_env(allow_cpu=(a.device == "cpu"))
    import torch
    from transformers import LogitsProcessorList, Qwen2VLForConditionalGeneration

    meta = load_meta(a.model)
    chain = verify_teacher_chain(meta, a.teacher, a.fold_file)   # QA → SFT(→ 병합) fold·분할 해시·base 출처(B2D 분할 파일)
    print(f"[chain] {json.dumps(chain, ensure_ascii=False)}", flush=True)
    items, fdoc, fold_path = assigned(a.teacher, a.fold_file)
    unseen = check_meta(meta, chain, fdoc, a.teacher, items)
    print(f"[assign] teacher_{a.teacher}: {len(items)} 표본, 미학습 검사 "
          f"{ {k: v for k, v in unseen.items() if k != 'sources'} }", flush=True)
    n_assigned = len(items)
    items = items[a.shard::a.num_shards]
    if a.max_samples:
        items = items[:a.max_samples]
    for _, r in items:   # B안(자차 상태 줄 포함) 프롬프트인지 표본마다 확인
        ego = (r.get("prompt_parts") or {}).get("ego")
        if not ego or ego not in build_user_text(r).split("\n"):
            raise SystemExit(f"[거부] {r['token']}: user 프롬프트에 자차 상태 줄이 없다 — 캐시 프롬프트 모드 {PROMPT_MODE} 위반")

    proc = make_processor(a.model, meta["pixels"])
    tok = proc.tokenizer
    tok.padding_side = "left"
    if len(tok) != meta["vocab_after"]:
        raise SystemExit(f"토크나이저 크기 {len(tok)} != 학습 {meta['vocab_after']}")
    template = LY.answer_template(tok)
    if {k: len(v) for k, v in template.items()} != LY.EXPECTED_COUNTS:
        raise SystemExit(f"정답 템플릿 폴더 길이 {({k: len(v) for k, v in template.items()})} != {LY.EXPECTED_COUNTS}")
    grammar = AGB.load(tok, a.grammar)
    print(f"[grammar] {grammar.info}", flush=True)
    ids_of = tok.convert_tokens_to_ids
    IMG, VEND, IMEND, EOT = ids_of("<|image_pad|>"), ids_of("<|vision_end|>"), ids_of("<|im_end|>"), ids_of("<|endoftext|>")

    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(a.model, torch_dtype=dt, attn_implementation=a.attn)
    model.to("cpu" if dev == "cpu" else "cuda").eval()
    if dev != "cpu":
        print(f"[attn] {exact_fp32_attention(dt)}", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    X.patch_visual_per_image(model)
    D = int(model.config.hidden_size)
    if not (1 <= a.layer <= model.config.num_hidden_layers):
        raise SystemExit(f"--layer {a.layer} 범위 밖(1..{model.config.num_hidden_layers})")
    rope = "mrope"

    r0 = items[0][1]
    sys0 = system_text(r0, meta["system"])
    probe = proc.apply_chat_template(build_messages(sys0, build_user_text(r0), N_IMG), tokenize=False, add_generation_prompt=True)
    if not probe.endswith("<|im_start|>assistant\n"):
        raise RuntimeError("프롬프트가 생성 프롬프트로 끝나지 않는다(정답 혼입 의심)")
    n_user = len(tok(build_user_text(r0), add_special_tokens=False)["input_ids"])
    L_ctx = LY.NUM_CAMS * LY.TOKENS_PER_CAM + n_user
    Ls = dict(context=L_ctx, **{k: len(template[k]) for k in LY.ANSWER_FOLDERS})
    ans_n = LY.ANSWER_TOKENS

    pos_doc = {k: template[k] for k in LY.ANSWER_FOLDERS}
    pos_sha = X.sha_text(json.dumps(pos_doc, sort_keys=True) + f"|ctx{L_ctx}|prev_first|b2d")
    instr_set = sorted({(r.get("prompt_parts") or {}).get("instruction", "") for _, r in items})
    tinfo = dict(
        checkpoint=str(Path(a.model).resolve()), checkpoint_sha=X.checkpoint_sha(a.model),
        teacher_meta_sha256=file_sha256(Path(a.model) / "teacher_meta.json"), run_id=meta.get("run_id"),
        fold=a.teacher, assign="fold", n_assigned=n_assigned,
        generate=fdoc["assignment"][f"teacher_{a.teacher}"]["generate"], fold_file=str(fold_path), fold_sha256=fdoc["sha256"],
        gt_dir=str(BS.B2D_GT_DIR.resolve()), sources=unseen["sources"],
        unseen_check={k: v for k, v in unseen.items() if k != "sources"},
        system_sha256=X.sha_text(sys0), instruction_sha256=[X.sha_text(s) for s in instr_set],
        grammar_sha256=file_sha256(BS.B2D_GT_DIR / "grammar.txt"),
        special_tokens_sha256=file_sha256(BS.B2D_GT_DIR / "special_tokens.txt"),
        script_sha256=file_sha256(__file__), layout_sha256=file_sha256(LY.__file__), common_b2d_sha256=file_sha256(B.__file__),
        chain=chain, prompt_mode=PROMPT_MODE, answer_grammar=grammar.info,
        generation=dict(do_sample=False, min_new_tokens=ans_n, max_new_tokens=ans_n, batch=a.batch, padding_side="left",
                        visual="per_image", attn=a.attn, dtype=a.dtype, rope=rope, frame_order="prev_first",
                        system=meta["system"], constrained_decoding=grammar.info["name"], num_cams=BS.B2D_NUM_CAMS,
                        pixels=meta["pixels"]),
        started=datetime.datetime.now().isoformat(timespec="seconds"))
    desc = {"context": f"현재 프레임 6 캠 영상 토큰 1,080(블록 {list(CUR_BLOCK)}) + user 텍스트 {n_user}(지시문·명령·목표점·자차 상태), 시퀀스 순서",
            "detection": "칸별 CLS·SEC·x·y·길이·폭·HEAD (21 × 8)", "tracking": "칸별 MOTION (5 × 8)",
            "motion": "칸별 1/2/3 s 변위 (30 × 8)", "map": "Map 줄 68 토큰(<free> 포함)",
            "plan": "칸별 RISK 8 + <free>DD.D 5 (지도 줄과 공유)", "action": "Intent 2 + Action 6 (3 s, B2D 코드북)"}
    for k in LY.FOLDERS:
        common = dict(folder=k, L=Ls[k], hidden_dim=D, dtype="bfloat16", layer=a.layer, answer_tokens=ans_n,
                      positions_sha256=pos_sha, description=desc[k], format="{'hidden': (L, D) bf16, 'attn_mask': (L,) uint8}",
                      answer_positions=(template[k] if k != "context" else None), source="teacher-generated answer (GT 아님)",
                      created_by=TAG, dataset="b2d", file_layout="<folder>/<clip>/<kf:05d>.pt", keyframe_stride=BS.KEYFRAME_STRIDE,
                      prompt_mode=PROMPT_MODE, ego_in_prompt=True, teacher_chain_verified=True,
                      constrained_decoding=grammar.info["name"], answer_grammar_sha256=grammar.info["sha256"])
        X.update_meta(root / k, common, a.teacher, tinfo)

    idx_dir = root / "_index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    my_name = index_name(a.teacher, a.shard, a.num_shards)
    others = [p for p in index_files(root, a.teacher) if Path(p).name != my_name
              and not Path(p).name.endswith(f"of{a.num_shards}.jsonl") and not (a.num_shards == 1 and Path(p).name == my_name)]
    if others:
        raise SystemExit(f"[거부] 교사 {a.teacher} 에 다른 샤드 수의 색인이 있다: {others} — 섞지 않는다")
    index = {}
    for p in index_files(root, a.teacher):
        for t, e in X.read_index(Path(p)).items():
            if t in index:
                raise SystemExit(f"[거부] 색인 중복 {t} ({p})")
            index[t] = e
    idx_path = idx_dir / my_name
    todo, n_skip = [], 0
    for part, r in items:
        t = r["token"]
        files = [root / k / f"{t}.pt" for k in LY.FOLDERS]
        if t in index:
            if index[t].get("teacher") != a.teacher:
                raise SystemExit(f"[거부] 색인 {t} 의 교사 {index[t].get('teacher')} != {a.teacher}")
            bad = [(f.parent.parent.name, X.check_file(f, Ls[f.parent.parent.name], D)) for f in files
                   if not f.exists() or X.check_file(f, Ls[f.parent.parent.name], D)]
            if bad:
                raise SystemExit(f"[실패] 색인에 완료로 기록된 {t} 의 파일 이상: {[(d, (m or '없음')) for d, m in bad]} — 수동 조치")
            n_skip += 1
            continue
        todo.append((part, r))
    print(f"[laq] teacher_{a.teacher} 샤드 {a.shard}/{a.num_shards} 배정 {n_assigned} (이번 {len(items)}) | 완료 건너뜀 {n_skip} | "
          f"할 일 {len(todo)} | L {Ls} | layer {a.layer} | D {D} | {dev} {a.dtype} {a.attn} | -> {root}", flush=True)

    fidx = open(idx_path, "a")
    t0, done, n_fail = time.time(), 0, 0
    gen_kw = dict(do_sample=False, min_new_tokens=ans_n, max_new_tokens=ans_n,
                  eos_token_id=[IMEND, EOT], pad_token_id=tok.pad_token_id, use_cache=True)
    for s in range(0, len(todo), a.batch):
        chunk = todo[s:s + a.batch]
        texts, imgs = [], []
        for _, r in chunk:
            sysm = system_text(r, meta["system"])
            if sysm != sys0:
                raise RuntimeError(f"{r['token']}: system 문장이 표본마다 다르다")
            texts.append(proc.apply_chat_template(build_messages(sysm, build_user_text(r), N_IMG), tokenize=False,
                                                  add_generation_prompt=True))
            imgs.extend(load_images(r, True, "prev_first", BS.B2D_NUM_CAMS))   # 영상 없으면 FileNotFoundError(대체 금지)
        enc = proc(text=texts, images=imgs, return_tensors="pt", padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        enc["pixel_values"] = enc["pixel_values"].to(dt)
        P = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, **gen_kw, logits_processor=LogitsProcessorList(
                [X.AG.make_logits_processor(grammar, P, len(chunk))]))
        gen = out[:, P:]
        if gen.shape[1] != ans_n:
            raise RuntimeError(f"생성 길이 {gen.shape[1]} != {ans_n} (강제 고정 길이 위반)")
        full = torch.cat([enc["input_ids"], gen], 1)
        am = torch.cat([enc["attention_mask"], torch.ones_like(gen)], 1)
        results = []
        g = enc["image_grid_thw"]
        per = [int(t * h * w) for t, h, w in g.tolist()]
        for b0 in range(0, len(chunk), a.fwd_batch):
            sl = slice(b0, b0 + a.fwd_batch)
            i0, i1 = N_IMG * b0, N_IMG * min(b0 + a.fwd_batch, len(chunk))
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
                ctx_pos, (ua, ub) = LY.context_positions(full[bi, :P].tolist(), IMG, VEND, IMEND, CUR_BLOCK)
                if len(ctx_pos) != L_ctx:
                    raise RuntimeError(f"{chunk[bi][1]['token']}: context {len(ctx_pos)} != {L_ctx} (프롬프트 길이 비고정)")
                if tok.decode(full[bi, ua:ub].tolist()) != build_user_text(chunk[bi][1]):
                    raise RuntimeError(f"{chunk[bi][1]['token']}: context 텍스트 구간이 user 텍스트와 다르다")
                with torch.no_grad():   # 순전파 ↔ 생성 일치(제약 안 argmax) · 547 뒤 종료 예측
                    lg = model.lm_head(last[j, P - 1:P - 1 + ans_n + 1])
                    am_ = lg.argmax(-1)
                    gi, gv = grammar.gather_index(grammar.replay_keys(gen[bi].tolist()), lg.device)
                    amc = gi.gather(1, lg[:ans_n].gather(1, gi).masked_fill(~gv, float("-inf")).argmax(-1, keepdim=True))[:, 0]
                agree = float((amc == gen[bi]).float().mean())
                free_agree = float((am_[:ans_n] == gen[bi]).float().mean())
                would_end = int(am_[-1]) in (IMEND, EOT)
                hs = {"context": H[j, ctx_pos]}
                for k in LY.ANSWER_FOLDERS:
                    hs[k] = H[j, [P + q for q in template[k]]]
                results.append((bi, {k: v.to("cpu", torch.bfloat16).contiguous() for k, v in hs.items()}, agree, would_end,
                                free_agree))
        for bi, hs, agree, would_end, free_agree in results:
            part, r = chunk[bi]
            e = dict(token=r["token"], teacher=a.teacher, part=part, scene_token=r["scene_token"], frame_idx=r["frame_idx"],
                     shard=f"{a.shard}/{a.num_shards}")
            e.update(analyse_answer(tok, gen[bi].tolist(), template, grammar))
            e.update(fwd_gen_agree=round(agree, 5), unconstrained_agree=round(free_agree, 5),
                     would_end_after=would_end, gen_ids=gen[bi].tolist())
            for k in LY.FOLDERS:
                if tuple(hs[k].shape) != (Ls[k], D):
                    raise RuntimeError(f"{r['token']} {k}: {tuple(hs[k].shape)} != ({Ls[k]}, {D})")
                f = root / k / f"{r['token']}.pt"
                f.parent.mkdir(parents=True, exist_ok=True)       # <folder>/<clip>/
                X.atomic_torch_save({"hidden": hs[k], "attn_mask": torch.ones(Ls[k], dtype=torch.uint8)}, f)
            fidx.write(json.dumps(e, ensure_ascii=False) + "\n")   # 7 파일 저장 뒤에만 색인 = 완료 표시
            fidx.flush(); os.fsync(fidx.fileno())
            done += 1; n_fail += int(e["parse_failure"])
            if not e["layout_ok"]:   # 제약 생성에서 위치 불일치 = 버그 → 즉시 중단(건너뛰지 않음)
                raise RuntimeError(f"{r['token']}: layout_bad (grammar_ok={e['grammar_ok']} pos={e['grammar_bad_pos']}, "
                                   f"parse={e['parse_error']}) — 문법 제약 생성인데 위치가 어긋났다. 원인 확인 전 진행 금지")
        el = time.time() - t0
        print(f"  [{s + len(chunk)}/{len(todo)}] {el / max(done, 1):.2f}s/표본 파싱실패 {n_fail} "
              f"ETA {(len(todo) - done) * el / max(done, 1) / 3600:.2f}h", flush=True)
    fidx.close()
    index = read_all_index(root)
    mine = [index[r["token"]] for _, r in items if r["token"] in index]
    summ = dict(n_done=len(mine), n_parse_failure=sum(e["parse_failure"] for e in mine),
                n_parse_error=sum(not e["parse_ok"] for e in mine), n_layout_bad=sum(not e.get("layout_ok", False) for e in mine),
                n_grammar_bad=sum(not e.get("grammar_ok", False) for e in mine),
                unconstrained_agree_mean=(sum(e.get("unconstrained_agree", 0.0) for e in mine) / max(len(mine), 1)),
                n_noncanonical=sum(not e["canonical"] for e in mine), n_would_end=sum(e["would_end_after"] for e in mine),
                fwd_gen_agree_min=min((e["fwd_gen_agree"] for e in mine), default=None),
                finished=datetime.datetime.now().isoformat(timespec="seconds"))
    for k in LY.FOLDERS:   # 샤드별 요약은 키를 나눠 기록(얕은 병합에서 다른 샤드 요약을 덮지 않게)
        X.update_meta(root / k, dict(folder=k, L=Ls[k], hidden_dim=D, layer=a.layer, answer_tokens=ans_n, positions_sha256=pos_sha,
                                     prompt_mode=PROMPT_MODE, constrained_decoding=grammar.info["name"],
                                     answer_grammar_sha256=grammar.info["sha256"]),
                      a.teacher, {**tinfo, f"summary_s{a.shard}of{a.num_shards}": summ})
    print(f"DONE teacher_{a.teacher} shard {a.shard}/{a.num_shards} 이번 저장 {done} ({time.time() - t0:.0f}s) | 요약 {summ}", flush=True)


def student_loader(root):
    """학생 config 의 teacher_loader(LoadTeacherCacheB2D) 를 cache_root 만 root 로 바꿔 만든다(색인·키프레임 격자·_meta 기대값 검사)."""
    sys.path.insert(0, str(REPO))
    from mmcv import Config
    import projects.mmdet3d_plugin  # noqa: F401  (레지스트리)
    import tools.b2d_teacher_register  # noqa: F401
    from projects.mmdet3d_plugin.models.encoder.vl_tokens_cache.loading_b2d import LoadTeacherCacheB2D, b2d_clip_name, b2d_keyframe_for
    cfg = Config.fromfile(str(STUDENT_CONFIG))
    tl = dict(cfg.teacher_loader)
    if tl.pop("type") != "LoadTeacherCacheB2D":
        raise RuntimeError(f"{STUDENT_CONFIG}: teacher_loader type 이 LoadTeacherCacheB2D 가 아니다")
    tl["cache_root"] = str(root)
    return LoadTeacherCacheB2D(**tl), b2d_clip_name, b2d_keyframe_for, tl


def verify_infos(root, exp, problems, load_samples=True):
    """학생이 실제로 읽는 10 Hz 프레임(b2d_infos) → 로더 규칙으로 키프레임 사상. 교사별 1 표본은 실제 적재(load_samples)."""
    import pickle
    ld, clip_of, kf_for, tl = student_loader(root)
    print(f"[verify] 학생 로더 생성 OK (meta_expect {tl['meta_expect']['context']}, 키프레임 클립 {len(ld.clip_keyframes)})", flush=True)
    n_frames, ages, sample = 0, {}, {}
    for split, p in BS.B2D_INFOS.items():
        d = pickle.load(open(p, "rb"))
        infos = d["infos"] if isinstance(d, dict) else d
        miss = []
        for info in infos:
            clip, f = clip_of(info["folder"]), int(info["frame_idx"])
            kfs = ld.clip_keyframes.get(clip)
            kf = None if kfs is None else kf_for(kfs, f, ld.keyframe_stride)
            if kf is None:
                miss.append(f"{clip}/{f}")
                continue
            key = f"{clip}/{kf:05d}"
            if key not in exp:
                miss.append(f"{clip}/{f}→{key}(배정 밖)")
                continue
            ages[f - kf] = ages.get(f - kf, 0) + 1
            sample.setdefault(exp[key], (info, key))
        n_frames += len(infos)
        if miss:
            problems.setdefault(f"infos_{split}_unmapped", []).extend(miss)
        print(f"[verify] info {split}: {len(infos)} 프레임, 키프레임 사상 실패 {len(miss)}", flush=True)
    print(f"[verify] 키프레임 나이 분포(프레임) {dict(sorted(ages.items()))}", flush=True)
    for T, (info, key) in (sorted(sample.items()) if load_samples else []):
        o = ld(dict(folder=info["folder"], frame_idx=info["frame_idx"], token="orig"))
        h = o["qwen_hidden"].data
        if o["teacher_key"] != key or o["token"] != "orig" or h.shape[1] != sum(ld.expect_lens.values()):
            problems.setdefault("loader_sample_bad", []).append(f"{T} {key} {tuple(h.shape)}")
        print(f"[verify] 로더 적재 teacher_{T} {key}: qwen_hidden {tuple(h.shape)} {h.dtype}, 나이 {o['teacher_age']} s", flush=True)
    return n_frames


def verify(a, root):
    """두 교사 배정 합(B2D train+val 키프레임 49,929)이 7 폴더·색인에 정확히 한 번씩 있는지. 반환 종료 코드."""
    exp, full = {}, {}
    n_assigned = 0  # 수정(2026-10-31): 독립 검토 지적 — 종전 중복 검사(len(full) != sum(1 for _ in full))는 항상 거짓 → 두 교사 배정 수 합과 비교
    for T in ("A", "B"):
        items, _, _ = assigned(T, a.fold_file)
        ts = [r["token"] for _, r in items]
        n_assigned += len(ts)
        full.update({t: T for t in ts})
        if a.teacher is not None and T != a.teacher:
            continue
        if a.max_samples:   # 스모크 전용: run 과 같은 부분집합([shard::num_shards][:max_samples])
            ts = ts[a.shard::a.num_shards][:a.max_samples]
        exp.update({t: T for t in ts})
    if len(full) != n_assigned:
        raise RuntimeError(f"배정 중복: 두 교사 배정 합 {n_assigned} ≠ 서로 다른 표본 {len(full)}")
    universe = set(exp) if (a.max_samples or a.teacher) else set(full)
    print(f"[verify] 기대 표본 {len(exp)} (A {sum(v == 'A' for v in exp.values())}, B {sum(v == 'B' for v in exp.values())}), "
          f"전체 배정 {len(full)}", flush=True)
    problems, metas = {}, {}
    for k in LY.FOLDERS:
        mp = root / k / "_meta.json"
        if not mp.exists():
            problems.setdefault("meta_missing", []).append(k); continue
        metas[k] = json.load(open(mp))
    try:
        idx = read_all_index(root)
    except RuntimeError as e:
        idx = {}
        problems.setdefault("index_dup", []).append(str(e))
    for t, T in exp.items():
        if t not in idx:
            problems.setdefault("index_missing", []).append(t)
        elif idx[t]["teacher"] != T:
            problems.setdefault("wrong_teacher", []).append(t)
    extra_idx = set(idx) - (universe if a.max_samples or a.teacher else set(full))
    if extra_idx and not a.teacher:
        problems.setdefault("index_extra", []).extend(sorted(extra_idx))
    sizes = {}
    for k in LY.FOLDERS:
        if k not in metas:
            continue
        L, D = metas[k]["L"], metas[k]["hidden_dim"]
        present = {str(p.relative_to(root / k))[:-3] for p in (root / k).glob("*/*.pt")}
        stray = [str(p) for p in (root / k).rglob("*.pt") if p.parent.parent != root / k]
        if stray:
            problems.setdefault(f"{k}_stray_depth", []).extend(stray)
        extra = present - set(full) if not a.max_samples else set()
        if extra:
            problems.setdefault(f"{k}_extra", []).extend(sorted(extra))
        for t in exp:
            f = root / k / f"{t}.pt"
            if t not in present:
                problems.setdefault(f"{k}_missing", []).append(t); continue
            if a.deep:
                m = X.check_file(f, L, D, deep=True)
            else:
                sz = f.stat().st_size
                if k not in sizes:
                    m = X.check_file(f, L, D)
                    sizes[k] = sz if m is None else None
                else:
                    m = None if sz == sizes[k] else (X.check_file(f, L, D) or None)
            if m:
                problems.setdefault(f"{k}_bad", []).append(f"{t}: {m}")
    pf = [e for t, e in idx.items() if t in exp]
    lay_bad = [e["token"] for e in pf if not e.get("layout_ok", False)]
    unconstrained = [e["token"] for e in pf if e.get("constrained_decoding") != X.AG.CONSTRAINT_NAME]
    st = dict(n_index=len(pf), parse_failure=sum(e["parse_failure"] for e in pf), parse_error=sum(not e["parse_ok"] for e in pf),
              layout_bad=len(lay_bad), grammar_bad=sum(not e.get("grammar_ok", False) for e in pf),
              not_constrained=len(unconstrained), noncanonical=sum(not e["canonical"] for e in pf),
              would_end=sum(e["would_end_after"] for e in pf))
    print(f"[verify] 색인 통계 {st}", flush=True)
    if lay_bad:
        problems["layout_bad"] = lay_bad
    if unconstrained:
        problems["index_not_constrained"] = unconstrained
    for k, m in metas.items():
        for kk, vv in (("prompt_mode", PROMPT_MODE), ("teacher_chain_verified", True), ("constrained_decoding", X.AG.CONSTRAINT_NAME),
                       ("dataset", "b2d"), ("answer_tokens", LY.ANSWER_TOKENS)):
            if m.get(kk) != vv:
                problems.setdefault("meta_key_mismatch", []).append(f"{k}.{kk}={m.get(kk)!r} (기대 {vv!r})")
        if k != "context" and m.get("L") != LY.EXPECTED_COUNTS[k]:
            problems.setdefault("meta_len_mismatch", []).append(f"{k}.L={m.get('L')} (기대 {LY.EXPECTED_COUNTS[k]})")
    if a.infos:
        if a.max_samples or a.teacher:
            raise SystemExit("--infos 는 전체 검증(--teacher·--max_samples 없이)에서만")
        verify_infos(root, exp, problems)
    if problems:
        for k, v in problems.items():
            print(f"[verify] 문제 {k}: {len(v)} (예: {v[:3]})", flush=True)
        print("[verify] FAIL", flush=True)
        return 1
    print(f"[verify] OK — {len(exp)} 표본 × 7 폴더 완전(deep={a.deep}, infos={a.infos})", flush=True)
    return 0


if __name__ == "__main__":
    main()
