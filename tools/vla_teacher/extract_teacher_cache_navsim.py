# 수정(2026-10-29): NAVSIM 교사 캐시 — extract_teacher_cache_v2.py(nuScenes, 수정 안 함)의 NAVSIM 판.
"""LaQ-AD NAVSIM 교사 캐시 추출(학생 config configs/navsim/v2_teacher/stage{1,2}/train.py 의 data/navsim/laq_cache).

흐름·저장 형식·색인·재개·원자적 쓰기는 extract_teacher_cache_v2.py 와 같다(그 파일의 도우미를 import 해 그대로 쓴다).
NAVSIM 에서 다른 것만:
  1) 입력 = 교사 학습과 같은 프롬프트: system(학습 meta, qwen_default) + 직전·현재 전방 3 캠(6 장, prev_first) + user 텍스트
     (지시문·명령·자차 상태 줄 = GT 생성기의 prompt.user 그대로, dropout 없음). 화소 128·28·28 → 캠당 120 토큰.
  2) 정답 = 549 토큰(행동 8) — 문법 제약 = answer_grammar_navsim(answer_grammar.py 규칙의 NAVSIM 표), 위치 = laq_cache_layout_navsim.
     context = 현재 프레임 3 캠 영상 토큰 360(블록 3·4·5) + user 텍스트 757 = 1,117. 정답 폴더 168/40/240/68/13/10.
  3) 표본 = 학생이 읽는 모든 프레임: navtrain GT 전 프레임(장면 프레임 103,288 + 과거 프레임 49,207 = 152,495, 교사 제외 6 로그
     포함 — 교사 학습에는 안 쓰였고 학생 학습에는 쓰인다) + navtest GT 전 프레임 17,872. 합 170,367.
  4) 교사 배정 = 분할 파일 assignment(로그 단위, nuScenes 와 같은 규칙): teacher_A → train.B + val.B, teacher_B → train.A + val.A.
     프레임의 로그(log_name)가 그 part 의 로그 집합에 있으면 그 교사 몫. 분할 파일의 n_samples 는 '장면 프레임' 수라서
     part 별 장면 프레임 수(is_scene_frame)를 그 값과 정확히 대조하고, 모든 GT 행이 정확히 한 part 에 속하는지 검사한다.
     추가 검사(누설 방지): 배정 로그 ∩ train.<교사> = ∅, QA 단계 qa_meta.scene_sets.train_scenes ⊆ train.<교사>,
     SFT 자료(meta.fold.navsim_sft) 학습·hold-out 로그 ⊆ train.<교사>.
  5) 샤드(--shard k --num_shards n): 같은 교사 배정 목록을 [k::n] 으로 나눠 여러 프로세스가 동시에 추출(nuScenes 실측: 추출은 CPU 한 코어
     병목). 색인 = _index/teacher_<T>.jsonl(n=1) 또는 teacher_<T>.s<k>of<n>.jsonl — 학생 로더는 _index/teacher_*.jsonl 을 모두 읽는다.
     건너뛰기 판단은 그 교사의 모든 색인 파일을 본다. 같은 교사에 다른 샤드 수의 색인이 섞이면 거부.
출처 검사: QA → SFT(→ 병합) fold·분할 해시 = sft_common.verify_teacher_chain(NAVSIM 분할 파일). meta 의 num_cams 3·pixels 128·28·28·
  use_prev·prev_first·mrope. 교사 SFT 자료 폴더의 special_tokens.txt·instruction.txt == GT 폴더의 것(바이트 단위), SFT manifest 의
  source_sha256 == GT navtrain 파일 sha256(SFT 자료가 이 GT 에서 만들어졌음).
실행(GPU3 = NAVSIM 전용, sft_common.require_gpu_env 가 GPU0 거부):
  CUDA_VISIBLE_DEVICES=3 python tools/vla_teacher/extract_teacher_cache_navsim.py --teacher A --model <navsim teacher_A 병합> [--shard 0 --num_shards 2]
  CUDA_VISIBLE_DEVICES="" python tools/vla_teacher/extract_teacher_cache_navsim.py --verify [--infos] [--deep]
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
import extract_teacher_cache_v2 as X  # noqa: E402  # 도우미(원자적 저장·파일 검사·색인·_meta 갱신·영상별 시각 인코더·체크포인트 해시)
import laq_cache_layout as LL  # noqa: E402
import laq_cache_layout_navsim as LN  # noqa: E402
import answer_grammar_navsim as AGN  # noqa: E402
import navsim_sft_common as NS  # noqa: E402
import navsim_teacher_common as NTC  # noqa: E402
from sft_common import (REPO, build_messages, build_user_text, file_sha256, load_fold_split, load_images, load_meta,  # noqa: E402
                        lm_forward, make_processor, require_gpu_env, system_text, exact_fp32_attention, verify_teacher_chain)

TAG = "수정(2026-10-29): NAVSIM 교사 캐시"
PROMPT_MODE = X.PROMPT_MODE          # "B" — 프롬프트(=context)에 자차 상태 줄 포함(학생 config teacher_meta_expect)
OUT_ROOT = "/home1/irteam/data-vol1/src/HiP-AD/data/navsim/laq_cache"   # 학생 config teacher_cache_root(data/navsim/laq_cache)
INFOS = {"train": "/home1/irteam/data-vol1/src/HiP-AD/data/infos/navsim_20261019/navsim_navtrain.pkl",
         "val": "/home1/irteam/data-vol1/src/HiP-AD/data/infos/navsim_20261019/navsim_navtest.pkl"}
# 프롬프트에 필요한 입력 필드만 남긴다(허용 목록) — 정답·정답 유도 필드(answer, targets, spans, copy_baseline, modes …)는 읽지 않는다.
KEEP_KEYS = ("token", "scene_token", "log_name", "timestamp", "is_scene_frame", "teacher_excluded_log", "cams", "cams_prev",
             "prev_available", "prev_token", "command", "ego_status", "command_drop_ok", "ego_drop_ok", "prompt", "prompt_parts")
N_IMG = 2 * LN.NUM_CAMS                # 직전 3 + 현재 3


def strip_to_input(r):
    out = {k: r[k] for k in KEEP_KEYS if k in r}
    out["prompt"] = {k: v for k, v in (r.get("prompt") or {}).items() if k in ("system", "user")}
    return out


def out_root_checked(p):
    root = X.out_root_checked(p)
    if "navsim" not in root.parts and not any("navsim" in s for s in root.parts):
        raise SystemExit(f"[거부] 출력 {root} 이 NAVSIM 경로가 아니다(nuScenes 캐시와 섞지 않는다)")
    return root


# ---------------------------------------------------------------- 배정
def scan_assignment(gt_dir, fold_file, need_records_for=None):
    """GT 두 파일 전 행 → {token: (교사, part, log)} + (요청한 교사의) [(part, 입력 레코드)] + 통계.
    모든 행이 정확히 한 part(train.A/B · val.A/B)에 속하고, part 별 장면 프레임 수 == 분할 파일 n_samples 여야 한다."""
    d, fp = load_fold_split(fold_file)
    owner = {}
    for T in ("A", "B"):
        for s in d["assignment"][f"teacher_{T}"]["generate"]:
            split, f = s.split(".")
            if split not in ("train", "val") or f not in ("A", "B"):
                raise ValueError(f"배정 항목 형식 오류: {s}")
            for log in d[split][f]["scenes"]:
                if log in owner:
                    raise RuntimeError(f"로그 {log} 가 두 part 에 배정됨")
                owner[log] = (T, s)
    assign, items, stats = {}, [], {}
    for split, fname in NS.NAVSIM_SPLIT_FILES.items():
        path = Path(gt_dir) / fname
        n_rows = 0
        with open(path) as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                n_rows += 1
                log = r["log_name"]
                if log not in owner or owner[log][1].split(".")[0] != split:
                    raise RuntimeError(f"{path}: 행 {r['token']} 의 로그 {log} 가 {split}.A/B 어디에도 없다")
                T, part = owner[log]
                if r["token"] in assign:
                    raise RuntimeError(f"표본 token 중복 {r['token']}")
                assign[r["token"]] = (T, part, log)
                st = stats.setdefault(part, dict(rows=0, scene_rows=0, logs=set(), excluded_rows=0))
                st["rows"] += 1; st["scene_rows"] += int(bool(r["is_scene_frame"])); st["logs"].add(log)
                st["excluded_rows"] += int(bool(r.get("teacher_excluded_log")))
                if need_records_for == T:
                    items.append((part, strip_to_input(r)))
        stats[f"{split}_rows"] = n_rows
    for part in ("train.A", "train.B", "val.A", "val.B"):
        split, f = part.split(".")
        st = stats.get(part)
        if st is None or st["scene_rows"] != d[split][f]["n_samples"] or st["logs"] != set(d[split][f]["scenes"]):
            raise RuntimeError(f"{part}: 장면 프레임 {None if st is None else st['scene_rows']} / 로그 수 "
                               f"{None if st is None else len(st['logs'])} != 분할 파일 {d[split][f]['n_samples']} / {d[split][f]['n_scenes']}")
        st["logs"] = len(st["logs"])
    return assign, items, d, fp, stats


def check_unseen(meta, chain, d, teacher, items):
    """배정 로그를 그 교사가 QA·SFT 에서 본 적이 없는지(구조·기록 양쪽)."""
    mine = set(d["train"][teacher]["scenes"])
    logs = {r["log_name"] for _, r in items}
    if logs & mine:
        raise SystemExit(f"[거부] 교사 {teacher} 배정 로그 {len(logs & mine)} 개가 train.{teacher}(교사 학습 반쪽)에 있다")
    qm = json.load(open(Path(chain["qa"]["path"]) / "qa_meta.json"))
    ss = qm["scene_sets"]
    qa_tr = set(ss["train_scenes"]) | set(ss.get("holdout_scenes", []))
    if not qa_tr <= mine or qa_tr & logs:
        raise SystemExit(f"[거부] QA 단계 장면 {len(qa_tr - mine)} 개가 train.{teacher} 밖 / 배정과 겹침 {len(qa_tr & logs)}")
    # 수정(2026-10-29): 검토 반영 — 선언 집합뿐 아니라 QA 가 실제로 쓴 장면(train/holdout_scenes_used)도 검사.
    #   실제 사용 ⊆ 선언 ⊆ train.T, 배정과 겹침 0, 교사 제외 6 로그(NuPlanQA-Eval)와 겹침 0(선언 집합엔 제외 로그가 들어 있을 수 있다 — 실제 사용이 기준).
    if "train_scenes_used" not in ss or "holdout_scenes_used" not in ss:
        raise SystemExit("[거부] QA qa_meta.scene_sets 에 train_scenes_used/holdout_scenes_used 없음 — 실제 사용 장면을 확인할 수 없다")
    qa_used = set(ss["train_scenes_used"]) | set(ss["holdout_scenes_used"])
    ex = NS.exclude_logs(d)
    if not qa_used <= qa_tr or qa_used & logs or qa_used & ex:
        raise SystemExit(f"[거부] QA 실제 사용 장면: 선언 밖 {len(qa_used - qa_tr)} / 배정과 겹침 {len(qa_used & logs)} / "
                         f"교사 제외 로그와 겹침 {sorted(qa_used & ex)}")
    ns = (meta.get("fold") or {}).get("navsim_sft")
    if not ns:
        raise SystemExit("[거부] 교사 meta.fold.navsim_sft 없음 — NAVSIM SFT(train_sft.py --navsim_sft_dir) 산출이 아니다")
    for part, fi in ns["files"].items():
        with open(fi["path"]) as fh:   # 로그 이름만(전 레코드를 메모리에 두지 않음)
            sl = {json.loads(ln)["scene_token"] for ln in fh if ln.strip()}
        if NS.sha256_file(fi["path"]) != fi["sha256"] or not sl <= mine or sl & logs or sl & ex:  # 수정(2026-10-29): 제외 로그 겹침도
            raise SystemExit(f"[거부] SFT {part} 자료 {fi['path']}: 해시 변경 또는 train.{teacher} 밖 로그/배정·제외 로그와 겹침")
    return dict(assigned_logs=len(logs), teacher_train_logs=len(mine), qa_scenes=len(qa_tr), qa_scenes_used=len(qa_used),  # 수정(2026-10-29): 검토 반영
                excluded_logs=len(ex), overlap=0)


def check_sources(meta, gt_dir):
    """교사 SFT 자료와 GT 의 토큰·지시문 파일이 바이트 단위로 같고, SFT 자료가 이 GT navtrain 에서 만들어졌는지."""
    sd = Path(meta["data_dir"])
    for n in ("special_tokens.txt", "instruction.txt"):
        if file_sha256(sd / n) != file_sha256(Path(gt_dir) / n):
            raise SystemExit(f"[거부] {n}: SFT 자료({sd}) ≠ GT({gt_dir})")
    man, _ = NS.load_manifest(sd)
    sha = NS.sha256_file(Path(gt_dir) / NS.NAVSIM_SPLIT_FILES["train"])
    if man.get("source_sha256") != sha:
        raise SystemExit(f"[거부] SFT manifest source_sha256 {str(man.get('source_sha256'))[:12]} != GT navtrain {sha[:12]}")
    toks, _ = __import__("sft_common").load_special_tokens(str(sd / "special_tokens.txt"))
    if toks != meta["special_tokens"]:
        raise SystemExit("[거부] 교사 meta 특수토큰 목록 ≠ special_tokens.txt")
    return dict(gt_navtrain_sha256=sha, sft_manifest_source_ok=True)


# ---------------------------------------------------------------- 표본 후처리
def analyse_answer(tok, gen_ids, template, grammar):
    """extract_teacher_cache_v2.analyse_answer 의 NAVSIM 판(파서·위치·길이만 NAVSIM)."""
    e = dict(n_new=len(gen_ids))
    g_ok, g_bad = grammar.accepts([int(i) for i in gen_ids])
    e["grammar_ok"], e["grammar_bad_pos"] = bool(g_ok), int(g_bad)
    text, offs = LL.token_char_offsets(tok, gen_ids)
    e["gen_text"] = text
    e["length_ok"] = len(gen_ids) == LN.ANSWER_TOKENS
    try:
        NTC.parse_answer(text)
        e["parse_ok"], e["parse_error"] = True, None
    except Exception as ex:   # 파싱 실패는 버리지 않고 기록(설계 결정 — nuScenes 와 같음)
        e["parse_ok"], e["parse_error"] = False, f"{type(ex).__name__}: {ex}"[:300]
    e["canonical"] = tok(text, add_special_tokens=False)["input_ids"] == [int(i) for i in gen_ids]
    lay = False
    if e["parse_ok"] and offs is not None and e["grammar_ok"] and e["length_ok"]:
        try:
            pos = LN.folder_token_positions_from_text(text, offs)
            lay = all(pos[k] == template[k] for k in LN.ANSWER_FOLDERS)
        except ValueError:
            lay = False
    e["layout_ok"] = lay
    e["parse_failure"] = not (e["parse_ok"] and e["layout_ok"] and e["length_ok"])
    e["constrained_decoding"] = grammar.info["name"]
    return e


def index_files(root, teacher):
    return sorted(glob.glob(str(root / "_index" / f"teacher_{teacher}.jsonl")) +
                  glob.glob(str(root / "_index" / f"teacher_{teacher}.s*of*.jsonl")))


def index_name(teacher, shard, num_shards):
    return f"teacher_{teacher}.jsonl" if num_shards == 1 else f"teacher_{teacher}.s{shard}of{num_shards}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=["A", "B"])
    ap.add_argument("--model", help="NAVSIM 병합 교사 모델 디렉터리(teacher_meta.json + processor)")
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--gt_dir", default=NS.NAVSIM_GT_DIR)
    ap.add_argument("--fold_file", default=NS.NAVSIM_FOLD_FILE)
    ap.add_argument("--grammar", default=AGN.MASK_FILE)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fwd_batch", type=int, default=4)
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")   # nuScenes 캐시와 같은 기본(bf16 생성·bf16 저장)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=0, help="스모크 전용: (샤드) 배정 목록 앞 N 개")
    ap.add_argument("--device", default=None, help="'cpu' = CPU 스모크(CUDA_VISIBLE_DEVICES='')")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--deep", action="store_true", help="--verify: 모든 파일을 실제로 읽어 모양·유한성 검사")
    ap.add_argument("--infos", action="store_true", help="--verify: 학생 info pkl(navtrain·navtest)의 모든 프레임 token 이 캐시와 같은 집합인지")
    a = ap.parse_args()
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    root = out_root_checked(a.out_root)
    if not (0 <= a.shard < a.num_shards):
        raise SystemExit(f"--shard {a.shard} 범위 밖(0..{a.num_shards - 1})")
    AGN.configure()
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
    chain = verify_teacher_chain(meta, a.teacher, a.fold_file)   # QA → SFT(→ 병합) fold·분할 해시·base 출처(NAVSIM 분할 파일)
    print(f"[chain] {json.dumps(chain, ensure_ascii=False)}", flush=True)
    for k, v in (("num_cams", LN.NUM_CAMS), ("pixels", NS.NAVSIM_PIXELS), ("use_prev", True), ("frame_order", "prev_first"),
                 ("rope", "mrope")):
        if meta.get(k) != v:
            raise SystemExit(f"[거부] 교사 meta {k}={meta.get(k)!r} != {v!r} (NAVSIM 교사 설계)")
    src = check_sources(meta, a.gt_dir)
    assign, items, fdoc, fold_path, stats = scan_assignment(a.gt_dir, a.fold_file, need_records_for=a.teacher)
    unseen = check_unseen(meta, chain, fdoc, a.teacher, items)
    print(f"[assign] teacher_{a.teacher}: {len(items)} 표본, 미학습 검사 {unseen}, part 통계 "
          f"{ {k: v for k, v in stats.items() if not k.endswith('_rows')} }", flush=True)
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
    template = LN.answer_template(tok)
    grammar = AGN.load(tok, a.grammar, a.gt_dir)
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
    L_ctx = LN.NUM_CAMS * LN.TOKENS_PER_CAM + n_user
    Ls = dict(context=L_ctx, **{k: len(template[k]) for k in LN.ANSWER_FOLDERS})
    ans_n = LN.ANSWER_TOKENS

    pos_doc = {k: template[k] for k in LN.ANSWER_FOLDERS}
    pos_sha = X.sha_text(json.dumps(pos_doc, sort_keys=True) + f"|ctx{L_ctx}|prev_first|navsim")
    instr_set = sorted({(r.get("prompt_parts") or {}).get("instruction", "") for _, r in items})
    tinfo = dict(
        checkpoint=str(Path(a.model).resolve()), checkpoint_sha=X.checkpoint_sha(a.model),
        teacher_meta_sha256=file_sha256(Path(a.model) / "teacher_meta.json"),
        fold=a.teacher, assign="fold", n_assigned=n_assigned,
        generate=fdoc["assignment"][f"teacher_{a.teacher}"]["generate"], fold_file=str(fold_path), fold_sha256=fdoc["sha256"],
        gt_dir=str(Path(a.gt_dir).resolve()), sources=src, unseen_check=unseen,
        system_sha256=X.sha_text(sys0), instruction_sha256=[X.sha_text(s) for s in instr_set],
        special_tokens_sha256=file_sha256(Path(a.gt_dir) / "special_tokens.txt"),
        script_sha256=file_sha256(__file__), layout_sha256=file_sha256(LN.__file__),
        chain=chain, prompt_mode=PROMPT_MODE, answer_grammar=grammar.info,
        generation=dict(do_sample=False, min_new_tokens=ans_n, max_new_tokens=ans_n, batch=a.batch, padding_side="left",
                        visual="per_image", attn=a.attn, dtype=a.dtype, rope=rope, frame_order="prev_first",
                        system=meta["system"], constrained_decoding=grammar.info["name"], num_cams=LN.NUM_CAMS,
                        pixels=meta["pixels"]),
        started=datetime.datetime.now().isoformat(timespec="seconds"))
    desc = {"context": f"현재 프레임 전방 3 캠 영상 토큰 360(블록 {list(LN.CUR_BLOCKS)}) + user 텍스트 {n_user}(지시문·명령·자차 상태), 시퀀스 순서",
            "detection": "칸별 CLS·SEC·x·y·길이·폭·HEAD (21 × 8)", "tracking": "칸별 MOTION (5 × 8)",
            "motion": "칸별 1/2/3 s 변위 (30 × 8)", "map": "Map 줄 68 토큰(<free> 포함)",
            "plan": "칸별 RISK 8 + <free>DD.D 5 (지도 줄과 공유)", "action": "Intent 2 + Action 8 (4 s)"}
    for k in LN.FOLDERS:
        common = dict(folder=k, L=Ls[k], hidden_dim=D, dtype="bfloat16", layer=a.layer, answer_tokens=ans_n,
                      positions_sha256=pos_sha, description=desc[k], format="{'hidden': (L, D) bf16, 'attn_mask': (L,) uint8}",
                      answer_positions=(template[k] if k != "context" else None), source="teacher-generated answer (GT 아님)",
                      created_by=TAG, dataset="navsim",
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
        files = [root / k / f"{t}.pt" for k in LN.FOLDERS]
        if t in index:
            if index[t].get("teacher") != a.teacher:
                raise SystemExit(f"[거부] 색인 {t} 의 교사 {index[t].get('teacher')} != {a.teacher}")
            bad = [(f.parent.name, X.check_file(f, Ls[f.parent.name], D)) for f in files
                   if not f.exists() or X.check_file(f, Ls[f.parent.name], D)]
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
    M = AGN.load_impl()
    for s in range(0, len(todo), a.batch):
        chunk = todo[s:s + a.batch]
        texts, imgs = [], []
        for _, r in chunk:
            sysm = system_text(r, meta["system"])
            if sysm != sys0:
                raise RuntimeError(f"{r['token']}: system 문장이 표본마다 다르다")
            texts.append(proc.apply_chat_template(build_messages(sysm, build_user_text(r), N_IMG), tokenize=False,
                                                  add_generation_prompt=True))
            imgs.extend(load_images(r, True, "prev_first", LN.NUM_CAMS))   # 영상 없으면 FileNotFoundError(대체 금지)
        enc = proc(text=texts, images=imgs, return_tensors="pt", padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        enc["pixel_values"] = enc["pixel_values"].to(dt)
        P = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, **gen_kw, logits_processor=LogitsProcessorList(
                [M.make_logits_processor(grammar, P, len(chunk))]))
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
                ctx_pos, (ua, ub) = LN.context_positions(full[bi, :P].tolist(), IMG, VEND, IMEND)
                if len(ctx_pos) != L_ctx:
                    raise RuntimeError(f"{chunk[bi][1]['token']}: context {len(ctx_pos)} != {L_ctx} (프롬프트 길이 비고정)")
                if tok.decode(full[bi, ua:ub].tolist()) != build_user_text(chunk[bi][1]):
                    raise RuntimeError(f"{chunk[bi][1]['token']}: context 텍스트 구간이 user 텍스트와 다르다")
                with torch.no_grad():   # 순전파 ↔ 생성 일치(제약 안 argmax) · 549 뒤 종료 예측
                    lg = model.lm_head(last[j, P - 1:P - 1 + ans_n + 1])
                    am_ = lg.argmax(-1)
                    gi, gv = grammar.gather_index(grammar.replay_keys(gen[bi].tolist()), lg.device)
                    amc = gi.gather(1, lg[:ans_n].gather(1, gi).masked_fill(~gv, float("-inf")).argmax(-1, keepdim=True))[:, 0]
                agree = float((amc == gen[bi]).float().mean())
                free_agree = float((am_[:ans_n] == gen[bi]).float().mean())
                would_end = int(am_[-1]) in (IMEND, EOT)
                hs = {"context": H[j, ctx_pos]}
                for k in LN.ANSWER_FOLDERS:
                    hs[k] = H[j, [P + q for q in template[k]]]
                results.append((bi, {k: v.to("cpu", torch.bfloat16).contiguous() for k, v in hs.items()}, agree, would_end,
                                free_agree))
        for bi, hs, agree, would_end, free_agree in results:
            part, r = chunk[bi]
            e = dict(token=r["token"], teacher=a.teacher, part=part, scene_token=r["log_name"], log_name=r["log_name"],
                     is_scene_frame=bool(r.get("is_scene_frame")), shard=f"{a.shard}/{a.num_shards}")
            e.update(analyse_answer(tok, gen[bi].tolist(), template, grammar))
            e.update(fwd_gen_agree=round(agree, 5), unconstrained_agree=round(free_agree, 5),
                     would_end_after=would_end, gen_ids=gen[bi].tolist())
            for k in LN.FOLDERS:
                if tuple(hs[k].shape) != (Ls[k], D):
                    raise RuntimeError(f"{r['token']} {k}: {tuple(hs[k].shape)} != ({Ls[k]}, {D})")
                X.atomic_torch_save({"hidden": hs[k], "attn_mask": torch.ones(Ls[k], dtype=torch.uint8)}, root / k / f"{r['token']}.pt")
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
    for k in LN.FOLDERS:   # 샤드별 요약은 키를 나눠 기록(얕은 병합에서 다른 샤드 요약을 덮지 않게)
        X.update_meta(root / k, dict(folder=k, L=Ls[k], hidden_dim=D, layer=a.layer, answer_tokens=ans_n, positions_sha256=pos_sha,
                                     prompt_mode=PROMPT_MODE, constrained_decoding=grammar.info["name"],
                                     answer_grammar_sha256=grammar.info["sha256"]),
                      a.teacher, {**tinfo, f"summary_s{a.shard}of{a.num_shards}": summ})
    print(f"DONE teacher_{a.teacher} shard {a.shard}/{a.num_shards} 이번 저장 {done} ({time.time() - t0:.0f}s) | 요약 {summ}", flush=True)


def read_all_index(root):
    out = {}
    for p in sorted(glob.glob(str(root / "_index" / "teacher_*.jsonl"))):
        for t, e in X.read_index(Path(p)).items():
            if t in out:
                raise RuntimeError(f"색인 중복 {t} ({p})")
            out[t] = e
    return out


def info_tokens(path):
    import pickle
    d = pickle.load(open(path, "rb"))
    return {i["token"] for i in (d["infos"] if isinstance(d, dict) else d)}


def verify(a, root):
    """두 교사 배정 합(navtrain+navtest GT 전 프레임)이 7 폴더·색인에 정확히 한 번씩 있는지. 반환 종료 코드."""
    assign, _, _, _, stats = scan_assignment(a.gt_dir, a.fold_file)
    exp = {t: v[0] for t, v in assign.items() if a.teacher is None or v[0] == a.teacher}
    if a.max_samples:   # 스모크 전용: 교사별 배정 목록(GT 파일 순서) → [shard::num_shards][:max_samples] — run 과 같은 부분집합
        exp = {}
        for T in (["A", "B"] if a.teacher is None else [a.teacher]):
            ts = [t for t, v in assign.items() if v[0] == T][a.shard::a.num_shards][:a.max_samples]
            exp.update({t: T for t in ts})
    print(f"[verify] 기대 표본 {len(exp)} (A {sum(v == 'A' for v in exp.values())}, B {sum(v == 'B' for v in exp.values())}) "
          f"part {stats}", flush=True)
    problems, metas = {}, {}
    if a.infos:   # 학생이 실제로 읽는 프레임 = info pkl 의 모든 token
        for split, p in INFOS.items():
            it = info_tokens(p)
            want = {t for t, v in assign.items() if v[1].startswith(split + ".")}
            if it != want:
                problems.setdefault(f"infos_{split}_mismatch", []).append(
                    f"info {len(it)} vs GT {len(want)} (info−GT {len(it - want)}, GT−info {len(want - it)})")
            print(f"[verify] info {split}: {len(it)} 프레임 token, GT 와 {'같음' if it == want else '다름'}", flush=True)
    for k in LN.FOLDERS:
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
    extra_idx = set(idx) - (set(exp) if a.max_samples else set(assign))
    if extra_idx:
        problems.setdefault("index_extra", []).extend(sorted(extra_idx))
    sizes = {}
    for k in LN.FOLDERS:
        if k not in metas:
            continue
        L, D = metas[k]["L"], metas[k]["hidden_dim"]
        present = {p.stem for p in (root / k).glob("*.pt")}
        extra = present - (set(exp) if a.max_samples else set(assign))
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
                       ("dataset", "navsim"), ("answer_tokens", LN.ANSWER_TOKENS)):
            if m.get(kk) != vv:
                problems.setdefault("meta_key_mismatch", []).append(f"{k}.{kk}={m.get(kk)!r} (기대 {vv!r})")
        if k != "context" and m.get("L") != LN.EXPECTED_COUNTS[k]:
            problems.setdefault("meta_len_mismatch", []).append(f"{k}.L={m.get('L')} (기대 {LN.EXPECTED_COUNTS[k]})")
    if problems:
        for k, v in problems.items():
            print(f"[verify] 문제 {k}: {len(v)} (예: {v[:3]})", flush=True)
        print("[verify] FAIL", flush=True)
        return 1
    print(f"[verify] OK — {len(exp)} 표본 × 7 폴더 완전(deep={a.deep}, infos={a.infos})", flush=True)
    return 0


if __name__ == "__main__":
    main()
