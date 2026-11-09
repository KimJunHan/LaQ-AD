# 수정(2026-10-15): 교사 VLM ② LoRA SFT — 공용 모듈(데이터·프롬프트·특수토큰·모델 구성·손실·GPU 규칙).
#   train_sft.py / merge_teacher_lora.py / eval_teacher.py / extract_teacher_cache.py / smoke_sft_cpu.py 가 공유한다.
#
#   형식 무관(format-agnostic) 설계 — v1(data/vla_teacher) 와 v2(data/vla_teacher/v2, 압축 문법 + 신규 특수토큰) 둘 다:
#     * 특수토큰: --special_tokens 파일(한 줄 한 토큰, '#' 주석·빈 줄 무시, 공백 뒤는 설명으로 무시). 없으면 v1 기본
#       <action_0>..<action_2040>.
#     * 프롬프트: 레코드에 prompt_parts 가 있으면 줄 단위로 조립 — 드롭된 명령/자차상태 줄은 '<줄>_dropped'
#       ('Command: unavailable.' / 'Ego state: unavailable.')로 교체(v2 common_v2.join_prompt 와 같은 규칙, 줄 수 고정).
#       없으면(v1) prompt.user 의 'Command:' / 'Ego state:' 줄을 같은 문장으로 교체. command_drop_ok / ego_drop_ok=False
#       표본은 드롭하지 않는다. dropout 이 일어나지 않은 표본은 prompt.user 를 글자 그대로 쓴다.
#     * 손실 가중: 레코드의 intent_weight(있으면) 를 Intent 구간 토큰 CE 에 곱한다. 구간은 answer_spans(있으면) 또는
#       줄 규칙(아래 answer_line_roles) 으로 찾는다. 못 찾으면 즉시 실패(조용한 skip 금지).
#     * future_complete=false 표본: Perception/Map 은 감독, Intent/Action 줄과 종료 토큰은 감독 제외(-100) —
#       장면 끝(미래 불완전)은 입력(현재 6 영상·명령·자차상태)에서 관측 불가라 'unavailable' 을 감독하면 예측 불가능한
#       잡음 타깃이 된다(README ① 의 '의도·행동 감독에서 제외' 규칙과 같다).
import ast
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]          # tools/vla_teacher/ → 저장소 루트
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

BASE_QWEN = "/home1/irteam/data-vol1/src/HiP-AD/ckpts/qwen2-vl-2b"
PIXELS = 200 * 28 * 28          # tools/qwen_cache.py --min/--max-pixels 기본값(280×504/캠 → 병합 후 180 토큰/캠)
TOKENS_PER_CAM = 180
NUM_CAMS = 6
QWEN_DEFAULT_SYSTEM = "You are a helpful assistant."   # Qwen2-VL chat_template 의 기본 system(캐시 생성 때 실제로 들어간 문장)
ACTION_K = 2041
CODEBOOK = "data/qwen3d/nusc_action_codebook_k2048.npz"
TAG = "# 수정(2026-10-15): 교사 VLM ② LoRA SFT"


# ---------------------------------------------------------------- GPU 규칙(GPU0 금지)
def require_gpu_env(allow_cpu=False):
    """CUDA_VISIBLE_DEVICES 검사. GPU0 이 보이면 거부. 반환 'cuda' | 'cpu'.
    미설정이면 모든 GPU(0 포함)가 보이므로 거부한다. 빈 문자열은 CPU 전용(allow_cpu 일 때만)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        raise SystemExit("[거부] CUDA_VISIBLE_DEVICES 미설정 — GPU0 이 보일 수 있다. GPU1 전용: CUDA_VISIBLE_DEVICES=1")
    ids = [s.strip() for s in cvd.split(",") if s.strip()]
    if not ids:
        if allow_cpu:
            return "cpu"
        raise SystemExit("[거부] CUDA_VISIBLE_DEVICES 가 비어 있다(이 스크립트는 GPU1 필요).")
    for s in ids:
        if not s.isdigit():
            raise SystemExit(f"[거부] CUDA_VISIBLE_DEVICES={cvd}: 정수 인덱스만 허용(UUID 는 GPU0 여부를 확인할 수 없다).")
        if int(s) == 0:
            raise SystemExit(f"[거부] CUDA_VISIBLE_DEVICES={cvd} 에 GPU0 포함 — GPU0 사용 금지(사용자 규칙).")
    return "cuda"


# ---------------------------------------------------------------- 데이터 위치(v2 우선, 없으면 v1)
def resolve_data_dir(arg=None):
    if arg:
        return Path(arg) if os.path.isabs(arg) else REPO / arg
    v2 = REPO / "data/vla_teacher/v2"
    if glob.glob(str(v2 / "teacher_gt*_train.jsonl")):
        return v2
    print(f"[data] v2({v2}) 정답 파일 없음 → v1(data/vla_teacher) 사용", flush=True)
    return REPO / "data/vla_teacher"


def split_file(data_dir, split):
    c = sorted(glob.glob(str(Path(data_dir) / f"teacher_gt*_{split}.jsonl")))
    if len(c) != 1:
        raise FileNotFoundError(f"{data_dir} 에서 teacher_gt*_{split}.jsonl 이 정확히 1개가 아니다: {c}")
    return c[0]


def load_records(data_dir, split, max_n=None):
    out = []
    with open(split_file(data_dir, split)) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if max_n and len(out) >= max_n:
                    break
    return out


def default_action_tokens():
    return [f"<action_{k}>" for k in range(ACTION_K)]


def load_special_tokens(path=None, data_dir=None):
    """반환 (토큰 리스트, 출처 문자열)."""
    if path is None and data_dir is not None and (Path(data_dir) / "special_tokens.txt").exists():
        path = str(Path(data_dir) / "special_tokens.txt")
    if path is None:
        return default_action_tokens(), "default:<action_0..2040>"
    toks = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        toks.append(s.split()[0])
    if len(set(toks)) != len(toks):
        raise ValueError(f"{path}: 중복 특수토큰")
    missing = [t for t in default_action_tokens() if t not in set(toks)]
    if missing:
        print(f"[tokens] 경고: {path} 에 action 토큰 {len(missing)}개 없음(예: {missing[:3]})", flush=True)
    return toks, path


_SPECIAL_RE = re.compile(r"<[A-Za-z][A-Za-z0-9_\-\.]*>")


def check_answers_use_known_tokens(records, special_tokens):
    """정답에 등장하는 <...> 꼴 토큰이 모두 등록 목록에 있는지. 없으면 실패(문자 조각으로 쪼개져 학습되는 것을 막는다)."""
    known = set(special_tokens)
    bad = {}
    for r in records:
        for t in _SPECIAL_RE.findall(r["answer"]):
            if t not in known:
                bad[t] = bad.get(t, 0) + 1
    if bad:
        raise ValueError(f"정답에 미등록 특수토큰 {len(bad)}종: {sorted(bad.items(), key=lambda x: -x[1])[:10]}")


# ---------------------------------------------------------------- 장면 단위 hold-out(train 의 ~3%)
HOLDOUT_FILE = "results/vla_teacher/holdout_train_scenes.json"


def scenes_hash(scenes):
    import hashlib
    return hashlib.sha256("\n".join(sorted(scenes)).encode()).hexdigest()


def holdout_scenes(records, n_scenes=50, seed=0, path=HOLDOUT_FILE):
    """체크포인트 선택용 고정 hold-out = TRAIN 장면 n_scenes 개(사용자 결정 2026-10-15: ~50 장면, 해시와 함께 저장).
    파일이 있으면 그것을 읽어 해시·train 소속을 검증(모든 run 이 같은 목록), 없으면 씨드 고정 표집으로 만들어 저장.
    val 장면은 절대 쓰지 않는다(train 레코드에서만 뽑고, 파일 검증 시에도 train 소속을 확인)."""
    p = Path(path) if os.path.isabs(path) else REPO / path
    train_scenes = sorted({r["scene_token"] for r in records})
    if p.exists():
        d = json.load(open(p))
        sc = d["scenes"]
        if scenes_hash(sc) != d["sha256"]:
            raise RuntimeError(f"{p}: 해시 불일치 — 목록이 수정되었다")
        missing = set(sc) - set(train_scenes)
        if missing:
            raise RuntimeError(f"{p}: train 에 없는 장면 {len(missing)}개(val 혼입 또는 다른 데이터)")
        return set(sc), d["sha256"], str(p)
    sc = sorted(random.Random(seed).sample(train_scenes, n_scenes))
    h = scenes_hash(sc)
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(scenes=sc, sha256=h, n=len(sc), seed=seed, source="train split scene_token (random.Random(seed).sample)",
                   n_train_scenes=len(train_scenes)), open(p, "w"), indent=1)
    return set(sc), h, str(p)


# ---------------------------------------------------------------- 교차 적합 분할
# 수정(2026-10-16): 2-fold 장면 분할
# 교차 적합(cross-fitting): TRAIN 장면을 서로 겹치지 않는 두 반쪽 A·B 로 나눠 교사 두 개를 따로 학습한다.
#   teacher_A = train A 로 학습 → train B 와 val B 의 정답 생성,  teacher_B = train B 로 학습 → train A 와 val A 생성.
#   그래서 학생이 받는 train 정답은 전부 '그 장면을 본 적 없는 교사'가 만든 것이 되고(교사가 외운 답이 섞이지 않음),
#   val 도 같은 방식으로 반씩 나눠 생성해 학생 학습 때와 같은 '두 교사 반반' 구성이 된다(val 은 두 교사 모두 미학습).
#   체크포인트 선택용 hold-out 은 각 반쪽 안의 train 장면에서만 뽑는다(다른 반쪽·val 금지).
FOLD_FILE = "data/vla_teacher/v2/fold_split.json"   # 수정(2026-10-16): 2-fold 장면 분할
FOLD_HOLDOUT_SCENES = 25                              # 수정(2026-10-16): 2-fold 장면 분할 — 기존 50/700 장면 비율 유지(25/350)
FOLDS = ("A", "B")


def file_sha256(path, chunk=1 << 22):  # 수정(2026-10-16): 2-fold 장면 분할
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def scene_sample_counts(records):  # 수정(2026-10-16): 2-fold 장면 분할
    """장면별 표본 수. 표본 token 중복이면 실패(한 표본이 두 번 배정되는 것을 원천 차단)."""
    toks = [r["token"] for r in records]
    if len(set(toks)) != len(toks):
        raise RuntimeError(f"표본 token 중복 {len(toks) - len(set(toks))}건 — 데이터 생성 단계 점검 필요")
    c = {}
    for r in records:
        c[r["scene_token"]] = c.get(r["scene_token"], 0) + 1
    return c


def balanced_scene_halves(counts, seed=0):  # 수정(2026-10-16): 2-fold 장면 분할
    """장면 → 두 반쪽(표본 수 균형, 결정적). 순서: 정렬한 장면 토큰을 seed 로 섞은 뒤(같은 크기끼리 순서가 토큰 사전순에
    치우치지 않게) 표본 수 내림차순 안정 정렬 → 하나씩 표본 합이 작은 쪽에 넣는다(같으면 장면 수가 적은 쪽, 그래도 같으면 A)."""
    order = sorted(counts)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda s: -counts[s])
    halves = {"A": [], "B": []}
    tot = {"A": 0, "B": 0}
    for s in order:
        k = min(FOLDS, key=lambda f: (tot[f], len(halves[f]), f))
        halves[k].append(s)
        tot[k] += counts[s]
    return {f: sorted(halves[f]) for f in FOLDS}


def _half_entry(scenes, counts):  # 수정(2026-10-16): 2-fold 장면 분할
    return dict(scenes=sorted(scenes), n_scenes=len(scenes), n_samples=int(sum(counts[s] for s in scenes)),
                sha256=scenes_hash(scenes))


def fold_doc_hash(d):  # 수정(2026-10-16): 2-fold 장면 분할 — 'sha256' 키를 뺀 전체 내용의 해시(수정 감지)
    import hashlib
    body = {k: v for k, v in d.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def make_fold_split(train_recs, val_recs, seed=0, n_holdout=FOLD_HOLDOUT_SCENES, sources=None):  # 수정(2026-10-16): 2-fold 장면 분할
    ctr, cva = scene_sample_counts(train_recs), scene_sample_counts(val_recs)
    if set(ctr) & set(cva):
        raise RuntimeError(f"train·val 장면 겹침 {len(set(ctr) & set(cva))}개")
    htr, hva = balanced_scene_halves(ctr, seed), balanced_scene_halves(cva, seed)
    hold = {}
    for f in FOLDS:
        if n_holdout >= len(htr[f]):
            raise ValueError(f"hold-out {n_holdout} 장면 ≥ fold {f} 장면 {len(htr[f])}")
        sc = sorted(random.Random(seed).sample(htr[f], n_holdout))     # 그 반쪽(정렬 목록) 안에서만 표집
        hold[f] = dict(_half_entry(sc, ctr), seed=seed, source=f"train.{f} 안에서 random.Random(seed).sample(sorted)")
    d = dict(
        version=1, created="2026-10-16", seed=seed,
        method="scene-level 2-fold, greedy sample-count balancing (seeded shuffle of sorted scene tokens, then stable "
               "sort by -count, assign to half with fewer samples; ties → fewer scenes → A)",
        sources=sources or {},
        train={f: _half_entry(htr[f], ctr) for f in FOLDS},
        val={f: _half_entry(hva[f], cva) for f in FOLDS},
        holdout=hold,
        assignment={
            "teacher_A": dict(sft_train="train.A − holdout.A", checkpoint_select="holdout.A",
                              generate=["train.B", "val.B"], out_default="results/vla_teacher/sft_foldA"),
            "teacher_B": dict(sft_train="train.B − holdout.B", checkpoint_select="holdout.B",
                              generate=["train.A", "val.A"], out_default="results/vla_teacher/sft_foldB"),
            "note": "train 정답은 그 장면을 학습하지 않은 교사가 생성. val 은 두 교사 모두 미학습이지만 학생 학습 때와 같은 "
                    "'반반' 구성이 되도록 반씩 생성(teacher_A→val.B, teacher_B→val.A). holdout.X 장면도 다른 교사가 생성한다."},
        balance=dict(train_sample_diff=abs(sum(ctr[s] for s in htr["A"]) - sum(ctr[s] for s in htr["B"])),
                     val_sample_diff=abs(sum(cva[s] for s in hva["A"]) - sum(cva[s] for s in hva["B"]))),
    )
    d["sha256"] = fold_doc_hash(d)
    return d


def load_fold_split(path=FOLD_FILE):  # 수정(2026-10-16): 2-fold 장면 분할
    p = Path(path) if os.path.isabs(path) else REPO / path
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음 — 먼저 tools/vla_teacher/make_fold_split.py 실행")
    d = json.load(open(p))
    if fold_doc_hash(d) != d.get("sha256"):
        raise RuntimeError(f"{p}: 전체 해시 불일치 — 분할 파일이 수정되었다")
    for part in ("train", "val", "holdout"):
        for f in FOLDS:
            if scenes_hash(d[part][f]["scenes"]) != d[part][f]["sha256"]:
                raise RuntimeError(f"{p}: {part}.{f} 장면 목록 해시 불일치")
    return d, str(p)


def fold_select(records, fold, path=FOLD_FILE):  # 수정(2026-10-16): 2-fold 장면 분할
    """train 레코드 → (fold 학습 표본, fold hold-out 표본, hold-out 장면 set, hold-out sha256, 분할 파일 경로, 분할 dict).
    분할 파일의 train 장면·장면별 표본 수가 지금 데이터와 정확히 같아야 한다(다르면 실패 — 조용히 맞추지 않음)."""
    d, p = load_fold_split(path)
    counts = scene_sample_counts(records)
    A, B = set(d["train"]["A"]["scenes"]), set(d["train"]["B"]["scenes"])
    if A & B or (A | B) != set(counts):
        raise RuntimeError(f"{p}: 분할 train 장면이 현재 데이터와 다르다(겹침 {len(A & B)}, "
                           f"누락 {len(set(counts) - (A | B))}, 초과 {len((A | B) - set(counts))})")
    for f in FOLDS:
        n = sum(counts[s] for s in d["train"][f]["scenes"])
        if n != d["train"][f]["n_samples"]:
            raise RuntimeError(f"{p}: train.{f} 표본 수 {d['train'][f]['n_samples']} ≠ 현재 데이터 {n}")
    mine = A if fold == "A" else B
    hold = set(d["holdout"][fold]["scenes"])
    if not hold <= mine:
        raise RuntimeError(f"{p}: holdout.{fold} 가 train.{fold} 밖의 장면을 포함")
    tr = [r for r in records if r["scene_token"] in mine and r["scene_token"] not in hold]
    ho = [r for r in records if r["scene_token"] in hold]
    return tr, ho, hold, d["holdout"][fold]["sha256"], p, d


# ---------------------------------------------------------------- 프롬프트(줄 dropout)
CMD_DROPPED_V1 = "Command: unavailable."     # v2 생성기(common_v2.CMD_DROPPED)와 같은 문장
EGO_DROPPED_V1 = "Ego state: unavailable."   # v1 의 CAN 결측 문장과 같다(common.ego_state_text)


def _part_kind(name, text):
    n = (name or "").lower()
    t = (text or "").lstrip()
    if n.endswith("_dropped"):
        return "dropped"
    if n == "system":
        return "system"
    if "cmd" in n or "command" in n or (not n and t.startswith("Command")):
        return "cmd"
    if "ego" in n or (not n and t.startswith("Ego state")):
        return "ego"
    return "other"


def prompt_parts_list(rec):
    """레코드 → [(kind, text, dropped_text)] (user 프롬프트 줄 단위).
    v2: prompt_parts = {instruction, command, ego, command_dropped, ego_dropped} — 드롭 시 '<줄>_dropped' 문장으로 교체
        (common_v2.join_prompt 와 같은 규칙: 줄 수·위치 고정).
    v1: prompt.user 를 줄로 나누고 'Command:' / 'Ego state:' 줄을 같은 방식('... unavailable.')으로 교체."""
    pp = rec.get("prompt_parts")
    out = []
    if pp is None:
        for line in rec["prompt"]["user"].split("\n"):
            k = _part_kind("", line)
            out.append((k, line, CMD_DROPPED_V1 if k == "cmd" else (EGO_DROPPED_V1 if k == "ego" else None)))
        return out
    if not isinstance(pp, dict):
        raise TypeError(f"prompt_parts 형식 미지원: {type(pp)} (dict 기대)")
    for name, text in pp.items():
        k = _part_kind(name, text)
        if k in ("dropped", "system") or text is None or text == "":
            continue
        if not isinstance(text, str):
            raise TypeError(f"prompt_parts[{name}] 가 문자열이 아니다: {type(text)}")
        out.append((k, text, pp.get(f"{name}_dropped")))
    return out


def build_user_text(rec, drop_cmd=False, drop_ego=False):
    if drop_cmd and rec.get("command_drop_ok") is False:
        drop_cmd = False                      # 생성기가 명령 제거 금지로 표시한 표본
    if drop_ego and rec.get("ego_drop_ok") is False:
        drop_ego = False                      # 생성기가 자차상태 제거 금지로 표시한 표본
    if not drop_cmd and not drop_ego and rec.get("prompt", {}).get("user") is not None:
        return rec["prompt"]["user"]          # dropout 없음 → 정답 생성기가 쓴 문장 그대로
    parts = prompt_parts_list(rec)
    kinds = [k for k, _, _ in parts]
    if drop_cmd and "cmd" not in kinds:
        raise ValueError(f"{rec['token']}: 명령 줄을 찾지 못해 dropout 불가")
    if drop_ego and "ego" not in kinds:
        raise ValueError(f"{rec['token']}: 자차상태 줄을 찾지 못해 dropout 불가")
    lines = []
    for k, t, dt in parts:
        if (k == "cmd" and drop_cmd) or (k == "ego" and drop_ego):
            if dt is None:
                raise ValueError(f"{rec['token']}: '{k}' 드롭 문장이 없다")
            lines.append(dt)
        else:
            lines.append(t)
    return "\n".join(lines)


def check_prompt_parts_roundtrip(records, n=1000):
    """dropout 없는 조립 = prompt.user 인지(조립 규칙이 생성기와 같은지) 검사. 반환 불일치 수."""
    bad = 0
    for r in records[:n]:
        if r.get("prompt", {}).get("user") is None:
            continue
        joined = "\n".join(t for _, t, _ in prompt_parts_list(r))
        if joined != r["prompt"]["user"]:
            bad += 1
    return bad


def system_text(rec, mode):
    if mode == "qwen_default":
        return QWEN_DEFAULT_SYSTEM
    if mode == "data":
        s = rec.get("prompt", {}).get("system")
        pp = rec.get("prompt_parts")
        if isinstance(pp, dict) and pp.get("system"):
            s = pp["system"]
        if not s:
            raise ValueError(f"{rec['token']}: system 프롬프트 없음(--system data)")
        return s
    raise ValueError(mode)


def build_messages(system, user_text, n_images):
    content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": user_text}]
    return [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content}]


# 수정(2026-10-26): 클라우드 이전 — 사내 서버에서 만든 QA·정답 jsonl 의 영상 경로가 /workspace/src/…, /mnt/nas2/… 절대경로다.
#   파일을 고치면 qa_manifest·분할 파일의 sha256 무결성 검사가 깨지므로, 자료는 그대로 두고 여는 순간에만 이 서버 경로로 옮긴다.
PATH_PREFIX_MAP = (("/workspace/src/", str(REPO.parent) + "/"), ("/mnt/nas2/", "/home1/irteam/data-vol1/nas_mirror/"))


def local_path(p):
    for old, new in PATH_PREFIX_MAP:
        if p.startswith(old):
            return new + p[len(old):]
    return p if os.path.isabs(p) else str(REPO / p)


def _abs_existing(paths, token, what, n_cams=NUM_CAMS):  # 수정(2026-10-29): NAVSIM 교사 SFT — n_cams 선택 인자(기본 6 = 종전 그대로)
    ps = []
    for p in paths:
        q = local_path(p)
        if not os.path.exists(q):
            raise FileNotFoundError(f"{what} 영상 없음: {q} (조용한 대체 금지)")
        ps.append(q)
    if len(ps) != n_cams:
        raise ValueError(f"{token}: {what} 카메라 {len(ps)}장 != {n_cams}")
    return ps


def has_prev(records):
    return any("cams_prev" in r or "prev_available" in r for r in records[:100])


def prev_available(rec):
    v = rec.get("prev_available", rec.get("targets", {}).get("prev_available"))
    return True if v is None else bool(v)


def cam_paths(rec, use_prev=False, order="prev_first", n_cams=NUM_CAMS):  # 수정(2026-10-29): NAVSIM 교사 SFT — n_cams(기본 6)
    """use_prev=False: 현재 6 캠(F,FL,FR,BL,B,BR). use_prev=True: 이전 키프레임(t−0.5 s) 6 캠 + 현재 6 캠 = 12 장.
    이전 프레임이 없는 표본(장면 첫 키프레임, prev_available=False)은 생성기가 준 cams_prev 를 쓰고, 없으면 현재 프레임을
    이전 자리에 반복한다(영상 수·토큰 배치 고정; 그 표본의 속도 값은 손실에서 제외 — tokenize_answer). 검은 영상 대체 금지."""
    cur = _abs_existing(rec["cams"], rec["token"], "현재", n_cams)
    if not use_prev:
        return cur
    pa = prev_available(rec)
    cp = rec.get("cams_prev")
    if cp:
        prev = _abs_existing(cp, rec["token"], "이전", n_cams)
    elif not pa:
        prev = cur
    else:
        raise ValueError(f"{rec['token']}: prev_available=True 인데 cams_prev 가 없다")
    if order == "prev_first":
        return prev + cur
    if order == "cur_first":
        return cur + prev
    raise ValueError(order)


def load_images(rec, use_prev=False, order="prev_first", n_cams=NUM_CAMS):  # 수정(2026-10-29): NAVSIM 교사 SFT — n_cams(기본 6)
    from PIL import Image
    return [Image.open(p).convert("RGB") for p in cam_paths(rec, use_prev, order, n_cams)]


# ---------------------------------------------------------------- 정답 구간(줄 역할) → 토큰 라벨/가중
def answer_spans(rec):
    for k in ("answer_spans", "spans", "loss_spans"):
        if rec.get(k):
            return rec[k]
        if rec.get("targets", {}).get(k):
            return rec["targets"][k]
    return None


def _as_span_list(v):
    if v is None:
        return []
    if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return [tuple(v)]
    return [tuple(x) for x in v]


_V2_SEC = r"(?:<FR>|<FL>|<BR>|<BL>|<F>|<B>|<L>|<R>)"
_V2_SPEED = re.compile(r"(?<=[0-9])" + _V2_SEC + r"((?:0|[1-9]\d*)\.\d)")   # SNUM 뒤 SEC UNUM = 이동방향·속력
_V1_SPEED = re.compile(r"moving \S+ (\d+\.\d)m/s")


def speed_spans(rec):
    """정답 안 '객체 속력 숫자' 문자 구간. 우선 GT 의 span 필드(answer_spans['speed'|'speeds']), 없으면 문법 정규식
    (v2: <cls><sec>±lat±lon<sec>speed 의 speed, v1: 'moving <dir> <v>m/s' 의 v). 객체 줄(첫 줄)만 본다."""
    sp = answer_spans(rec)
    if sp:
        for k in ("speed", "speeds", "obj_speed"):
            if k in sp:
                return _as_span_list(sp[k])
    first = rec["answer"].split("\n")[0]
    rx = _V2_SPEED if first.startswith("Objects:") else _V1_SPEED
    return [m.span(1) for m in rx.finditer(first)]


def answer_line_roles(rec):
    """정답 각 줄의 (start, end, role). role ∈ {'other','intent','action'}.
    우선순위: rec['answer_spans'] = {'intent':[s,e], 'action':[s,e]} (문자 오프셋) → 줄 규칙.
    줄 규칙: action = 'Action' 으로 시작하거나 '<action_' 을 담은 마지막 줄,
             intent = 'Intent' 로 시작하는 줄, 없으면 action 바로 앞 줄."""
    ans = rec["answer"]
    lines, pos = [], 0
    for ln in ans.split("\n"):
        lines.append([pos, pos + len(ln), "other", ln])
        pos += len(ln) + 1
    spans = answer_spans(rec)
    if spans and ("intent" in spans or "action" in spans):
        for role in ("intent", "action"):
            for s_, e_ in _as_span_list(spans.get(role)):
                hit = False
                for L in lines:          # span 이 줄 전체 또는 줄의 값 부분(접두 'Intent: ' 제외)일 수 있다 → 겹치는 줄
                    if L[0] < e_ and s_ < max(L[1], L[0] + 1):
                        L[2] = role; hit = True
                if not hit:
                    raise ValueError(f"{rec['token']}: '{role}' span {s_, e_} 이 정답 줄과 겹치지 않는다")
        return [(a, b, r) for a, b, r, _ in lines]
    ai = None
    for i, L in enumerate(lines):
        if L[3].startswith("Action") or "<action_" in L[3]:
            ai = i
    ii = None
    for i, L in enumerate(lines):
        if L[3].startswith("Intent"):
            ii = i
    if ai is None:
        raise ValueError(f"{rec['token']}: 정답에서 Action 줄을 찾지 못했다(answer_spans 를 주거나 규칙을 갱신할 것)")
    if ii is None:
        if ai == 0:
            raise ValueError(f"{rec['token']}: Intent 줄을 찾지 못했다")
        ii = ai - 1
    lines[ai][2] = "action"
    lines[ii][2] = "intent"
    return [(a, b, r) for a, b, r, _ in lines]


def future_complete(rec):
    t = rec.get("targets", {})
    if "future_complete" in t:
        return bool(t["future_complete"])
    return bool(rec["future_complete"])


def intent_weight(rec):
    w = rec.get("intent_weight", rec.get("targets", {}).get("intent_weight"))
    return 1.0 if w is None else float(w)


def tokenize_answer(tok, rec, end_token="<|im_end|>", mask_speed_without_prev=True):
    """반환 (ids, labels, weights) — 정답 + 종료 토큰. 라벨 -100 = 감독 제외.
    정답 토큰화는 프롬프트와 분리(생성 시 모델이 프롬프트 뒤에 내는 토큰열과 같은 정준 토큰화).
    마스크 규칙(사용자 결정 2026-10-15): future_complete=False → Intent/Action 줄·종료 토큰 제외,
    prev_available=False → 객체 속력 숫자 토큰 제외(단일 프레임으로는 관측 불가)."""
    ans = rec["answer"]
    enc = tok(ans + end_token, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    roles = answer_line_roles(rec)
    fc = future_complete(rec)
    iw = intent_weight(rec)
    sp = speed_spans(rec) if (mask_speed_without_prev and not prev_available(rec)) else []
    # 수정(2026-10-15): 생성기가 준 loss_mask_spans(v2-sensor: '<na>' 대체형 = GT 값 미상 등)도 감독 제외.
    #   v1 레코드에는 필드가 없어 동작 불변. 기존 v2 의 구간(Intent/Action 줄·속력 숫자)은 위 규칙과 같은 곳이라 결과 동일.
    sp = list(sp) + [tuple(x) for x in (rec.get("loss_mask_spans") or [])]
    labels, weights = [], []
    for tid, (s, e) in zip(ids, offs):
        if sp and any(s < b and a < max(e, s + 1) for a, b in sp):
            labels.append(-100); weights.append(0.0)
            continue
        if s >= len(ans):
            role = "end"
        else:
            role = "other"
            for a, b, r in roles:
                if a <= s <= b:          # 줄 끝 '\n' 위치(s==b)는 그 줄에 속한다
                    role = r
                    break
        if not fc and role in ("intent", "action", "end"):
            labels.append(-100); weights.append(0.0)
        else:
            labels.append(tid); weights.append(iw if role == "intent" else 1.0)
    return ids, labels, weights


# 수정(2026-10-16): SFT 발산 원인 진단 — 정답 토큰의 구간 번호(진단 로그 전용, 손실·라벨에는 영향 없음).
SEG_NAMES = ["prompt", "obj_tok", "obj_num", "map_tok", "map_num", "intent", "action", "end", "other"]
_NUM_CH = set("0123456789.+-")


def answer_segments(tok, rec, end_token="<|im_end|>"):
    """정답(+종료) 토큰별 구간 번호(SEG_NAMES 색인). 줄 구간은 answer_spans['lines'](v2) 또는 answer_line_roles.
    obj/map 줄 안에서는 토큰 글자가 전부 숫자·부호·점이면 *_num, 그 외(<car> 같은 특수토큰)는 *_tok."""
    ans = rec["answer"]
    enc = tok(ans + end_token, add_special_tokens=False, return_offsets_mapping=True)
    sp = answer_spans(rec) or {}
    lines = sp.get("lines") if isinstance(sp, dict) else None
    if not lines:
        lines = {}
        for a, b, r in answer_line_roles(rec):
            k = r if r in ("intent", "action") else ("objects" if ans[a:b].startswith("Objects") else
                                                    ("map" if ans[a:b].startswith("Map") else "other"))
            lines.setdefault(k, [a, b])
    out = []
    for (s, e) in enc["offset_mapping"]:
        if s >= len(ans):
            out.append(SEG_NAMES.index("end")); continue
        role = "other"
        for k, (a, b) in lines.items():
            if a <= s <= b:
                role = k; break
        txt = ans[s:e]
        isnum = bool(txt) and set(txt) <= _NUM_CH
        if role == "objects":
            out.append(SEG_NAMES.index("obj_num" if isnum else "obj_tok"))
        elif role == "map":
            out.append(SEG_NAMES.index("map_num" if isnum else "map_tok"))
        elif role in ("intent", "action"):
            out.append(SEG_NAMES.index(role))
        else:
            out.append(SEG_NAMES.index("other"))
    return out


# 수정(2026-10-16): SFT 발산 원인 진단 — fp32 어텐션은 정확한 커널(math)만 쓴다.
#   원인(실측, torch 2.7.1+cu128, RTX PRO 6000 sm_120): fp32 입력의 SDPA 는 flash·cuDNN 이 fp32 를 지원하지 않아
#   memory-efficient 커널로 가는데, 이 커널이 fp32 정밀도가 아니다(작은 로짓에서도 출력 상대오차 ~3e-3, 로짓이 수백이면
#   출력 오차 O(1), 역전파 기울기 오차 1e5~1e14 — float64 기준). Qwen2-VL-2B 0 층은 k_proj 편향이 최대 432 라
#   |q||k|/√d ≈ 2.9e4 로 로짓이 매우 커서, 이 커널의 역전파가 쓰레기 기울기(0층 q/v LoRA·입력 임베딩 행)를 내고
#   학습이 진행될수록 커져 비유한값이 된다. math 커널은 float64 기준 오차 1e-4 이하·결정적.
def exact_fp32_attention(dtype):
    """dtype 이 fp32 이고 CUDA 면 SDPA 를 math 커널로 고정(프로세스 전역 — 역전파 중 gradient checkpointing 재계산 포함).
    그 뒤 float64 기준 자기검사(로짓이 큰 합성 입력)로 오차를 확인하고 실패하면 즉시 중단. 반환: 설명 문자열."""
    import torch
    if dtype != torch.float32 or not torch.cuda.is_available():
        return "unchanged"
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return check_sdpa_fp32()


def check_sdpa_fp32(L=512, scale=100.0, tol=1e-2):
    """현재 활성 SDPA 커널의 fp32 정확도 검사(Qwen2-VL 0층처럼 키에 큰 상수 성분이 있는 경우). float64 기준."""
    import torch
    g = torch.Generator(device="cuda").manual_seed(0)
    B, H, D = 1, 4, 128
    q = torch.randn(B, H, L, D, device="cuda", generator=g) * scale / 10
    k = torch.randn(B, H, L, D, device="cuda", generator=g)
    k[..., :4] += scale
    v = torch.randn(B, H, L, D, device="cuda", generator=g)
    go = torch.randn(B, H, L, D, device="cuda", generator=g)
    errs = []
    for dt in (torch.float64, torch.float32):
        qq, kk, vv = [t.to(dt).clone().requires_grad_() for t in (q, k, v)]
        if dt == torch.float64:          # 기준: 명시적 softmax(어느 SDPA 커널도 쓰지 않음)
            s_ = (qq @ kk.transpose(-1, -2)) / D ** 0.5
            s_ = s_.masked_fill(torch.ones(L, L, device="cuda", dtype=torch.bool).triu(1), float("-inf"))
            o = torch.softmax(s_, -1) @ vv
        else:
            o = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
        o.backward(go.to(dt))
        errs.append((o.detach().double(), qq.grad.double(), kk.grad.double(), vv.grad.double()))
    rel = max(float((a - b).abs().max() / b.abs().max().clamp_min(1e-12)) for a, b in zip(errs[1], errs[0]))
    if not rel < tol:
        raise RuntimeError(f"fp32 SDPA 커널 정확도 검사 실패: float64 대비 최대 상대오차 {rel:.3g} ≥ {tol}")
    return f"sdpa=math(fp32 exact), self-check rel_err={rel:.2e}"


# ---------------------------------------------------------------- 프로세서·특수토큰·모델
def make_processor(path, pixels=PIXELS):
    from transformers import AutoProcessor
    p = AutoProcessor.from_pretrained(path, min_pixels=pixels, max_pixels=pixels)
    return p


def add_special_tokens(tok, toks):
    """반환 (새 id 리스트, 추가 전 길이). 이미 같은 토큰이 있으면 실패(의미 충돌)."""
    n0 = len(tok)
    vocab = tok.get_vocab()
    clash = [t for t in toks if t in vocab]
    if clash:
        raise ValueError(f"이미 존재하는 토큰 {len(clash)}개: {clash[:5]}")
    added = tok.add_tokens(toks, special_tokens=True)
    if added != len(toks):
        raise RuntimeError(f"토큰 추가 {added} != {len(toks)}")
    ids = tok.convert_tokens_to_ids(toks)
    assert ids == list(range(n0, n0 + len(toks))), "새 토큰 id 가 연속이 아니다"
    return ids, n0


DESC_ALIAS = {   # v2 축약 토큰의 뜻(초기화용 서술). 없는 토큰은 이름 그대로('_'→공백)
    "cv": "construction vehicle", "ped": "pedestrian", "moto": "motorcycle", "bike": "bicycle", "cone": "traffic cone",
    "F": "front", "FL": "front left", "FR": "front right", "L": "left", "R": "right", "BL": "back left", "B": "back",
    "BR": "back right", "pc": "pedestrian crossing", "pc_on": "on pedestrian crossing",
    "pc_none": "no pedestrian crossing", "div": "dividers", "bnd": "boundaries", "na": "unavailable",
    "none": "none", "static": "stationary",
    # 수정(2026-10-15): v2-sensor 신규 토큰 뜻
    "hF": "heading front", "hFL": "heading front left", "hFR": "heading front right", "hL": "heading left",
    "hR": "heading right", "hBL": "heading back left", "hB": "heading back", "hBR": "heading back right",
    "ttc_lt2": "collision risk within 2 seconds", "ttc_2_4": "collision risk in 2 to 4 seconds",
    "ttc_4_6": "collision risk in 4 to 6 seconds", "ttc_safe": "no collision risk",
    "at0": "at 0 m ahead", "at10": "at 10 m ahead", "at20": "at 20 m ahead", "free": "free distance ahead",
    "_": "empty"}


def _desc_text(t):
    k = t.strip("<>")
    if k in DESC_ALIAS:
        return DESC_ALIAS[k]
    return k.replace("_", " ").replace("-", " ")


def init_new_rows(model, tok, toks, new_ids, n_base, mode="desc", seed=0):
    """새 토큰 행 초기화(입력 임베딩; tie_word_embeddings=True 라 lm_head 행도 같은 값).
    desc : 토큰 이름(꺾쇠 제거, '_'→공백)의 기존 서브워드 임베딩 평균 + N(0, (0.1·σ_dim)²) 씨드 고정 잡음.
           평균은 순서 불변이라 <action_1035>/<action_1053> 처럼 숫자 순열이 같으면 같은 값이 되므로 잡음으로 대칭을 깬다.
    mean : 기존 어휘 임베딩 평균 + 같은 잡음(HF mean_resizing 과 같은 취지)."""
    import torch
    W = model.get_input_embeddings().weight
    with torch.no_grad():
        base = W[:n_base].float()
        mu, sd = base.mean(0), base.std(0)
        g = torch.Generator().manual_seed(seed)
        noise = torch.randn(len(new_ids), W.shape[1], generator=g) * (0.1 * sd.cpu())
        for j, (t, i) in enumerate(zip(toks, new_ids)):
            if mode == "desc":
                sub = tok(_desc_text(t), add_special_tokens=False)["input_ids"]
                sub = [s for s in sub if s < n_base]
                v = base[sub].mean(0) if sub else mu
            elif mode == "mean":
                v = mu
            else:
                raise ValueError(mode)
            W[i] = (v + noise[j].to(v.device)).to(W.dtype)


def resize_for_tokens(model, n_tok, multiple=64):
    cur = model.get_input_embeddings().weight.shape[0]
    if n_tok > cur:
        model.resize_token_embeddings(n_tok, pad_to_multiple_of=multiple, mean_resizing=False)
    if getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()
        assert model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr(), \
            "tie_word_embeddings=True 인데 lm_head 와 embed_tokens 가 공유되지 않는다"


LORA_ATTN = r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
LORA_MLP = r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)"


def lora_target_regex(targets):
    """언어모델(model.layers.*) 만 — 비전 인코더(visual.blocks.*: qkv/proj/fc1/fc2)·merger 는 이름이 달라 제외된다."""
    if targets == "attn":
        return LORA_ATTN
    if targets == "attn_mlp":
        return f"(?:{LORA_ATTN}|{LORA_MLP})"
    raise ValueError(targets)


def build_peft(model, new_ids, r=16, alpha=32, dropout=0.05, targets="attn_mlp"):
    import torch
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=lora_target_regex(targets),
                     trainable_token_indices={"embed_tokens": list(new_ids)})
    pm = get_peft_model(model, cfg)
    # 학습 파라미터(LoRA A/B + 새 토큰 행)는 fp32 마스터(동결 base 가 bf16 이어도). 순전파에서 peft 가 base dtype 으로 캐스팅.
    for n, p in pm.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return pm


def trainable_summary(pm):
    tot, tr, groups = 0, 0, {}
    for n, p in pm.named_parameters():
        tot += p.numel()
        if p.requires_grad:
            tr += p.numel()
            g = "lora" if "lora_" in n else ("tokens" if "trainable_tokens" in n else "other")
            groups[g] = groups.get(g, 0) + p.numel()
            if g == "other":
                raise RuntimeError(f"예상 외 학습 파라미터: {n}")
    return dict(total=tot, trainable=tr, **groups)


def base_of(m):
    """PeftModel → 내부 Qwen2VLForConditionalGeneration(LoRA 주입된 상태). 일반 모델이면 그대로."""
    return m.get_base_model() if hasattr(m, "get_base_model") else m


def embed_inputs(model, input_ids, pixel_values=None, image_grid_thw=None, vision_no_grad=True):
    """Qwen2VLForConditionalGeneration.forward 의 임베딩·영상 병합 부분과 같은 계산(transformers 4.46.3)."""
    import torch
    base = base_of(model)
    emb = base.model.embed_tokens(input_ids)
    if pixel_values is not None:
        ctx = torch.no_grad() if vision_no_grad else torch.enable_grad()
        with ctx:
            # 수정(2026-10-15 밤): 이미지 단위로 시각 인코더를 돌린다. 시각 어텐션은 이미지마다 독립(블록 대각 마스크)이라 결과는
            #   묶어서 돌린 것과 수학적으로 같다. 묶어 돌리면 SDPA 경로가 배치 전체 패치(예: 8 × 12장 × 720 = 69,120)의
            #   어텐션 행렬을 한 번에 만들어 fp32 에서 284 GiB 할당 → OOM(10-15 시험 실행 실측). 근본 원인 제거.
            pv = pixel_values.type(base.visual.get_dtype())
            sizes = [int(t * h * w) for t, h, w in image_grid_thw.tolist()]
            if sum(sizes) != pv.shape[0]:
                raise ValueError(f"패치 수 {sum(sizes)} != pixel_values {pv.shape[0]}")
            img = torch.cat([base.visual(p, grid_thw=g[None]) for p, g in zip(pv.split(sizes), image_grid_thw)], 0)
        m = input_ids == base.config.image_token_id
        if int(m.sum()) != img.shape[0]:
            raise ValueError(f"영상 토큰 {int(m.sum())} != 영상 특징 {img.shape[0]}")
        emb = emb.masked_scatter(m.unsqueeze(-1).expand_as(emb), img.to(emb.dtype))
    return emb


def position_ids_for(model, input_ids, image_grid_thw, attention_mask, rope):
    """rope='mrope': Qwen2-VL 공식 3D M-RoPE(generate 가 쓰는 것과 같음).
    rope='1d'  : position_ids=None → Qwen2VLModel 이 cache_position(0..L-1)을 3축에 복제. transformers 4.46.3 의
                 forward() 는 M-RoPE 위치를 스스로 만들지 않으므로 tools/qwen_cache.py·qwen_online.py 의 기존 캐시가 이 방식이다."""
    if rope == "1d":
        return None
    if rope != "mrope":
        raise ValueError(rope)
    pos, _ = base_of(model).get_rope_index(input_ids, image_grid_thw, None, attention_mask)
    return pos


def lm_forward(model, batch, rope="mrope", output_hidden_states=False):
    base = base_of(model)
    emb = embed_inputs(model, batch["input_ids"], batch.get("pixel_values"), batch.get("image_grid_thw"))
    pos = position_ids_for(model, batch["input_ids"], batch.get("image_grid_thw"), batch["attention_mask"], rope)
    return base.model(input_ids=None, inputs_embeds=emb, attention_mask=batch["attention_mask"], position_ids=pos,
                      use_cache=False, output_hidden_states=output_hidden_states, return_dict=True)


def sft_loss(model, batch, rope="mrope", diag_n_base=None):  # 수정(2026-10-16): SFT 발산 원인 진단 — diag_n_base 주면 로짓·구간 통계 추가
    """정답 위치만 lm_head 를 통과시킨다(전 위치 로짓 B×L×153k 를 만들지 않음 — 메모리 수 GB 절약).
    반환 (loss, n_tok, 통계 dict). loss = Σ w·CE / (감독 토큰 수) — intent_weight<1 은 실제 하향 가중."""
    import torch
    import torch.nn.functional as F
    out = lm_forward(model, batch, rope)
    h = out.last_hidden_state                               # 최종 norm 적용 후
    lab = batch["labels"][:, 1:]
    w = batch["weights"][:, 1:]
    m = lab != -100
    hs = h[:, :-1][m]
    logits = base_of(model).lm_head(hs).float()
    tgt = lab[m]
    ce = F.cross_entropy(logits, tgt, reduction="none")
    n = int(m.sum())
    loss = (ce * w[m].to(ce.dtype)).sum() / max(n, 1)
    with torch.no_grad():
        pred = logits.argmax(-1)
        acc = (pred == tgt).float().mean().item() if n else 0.0
        is_act = batch.get("is_action_label")
        st = dict(ce=ce.mean().item() if n else 0.0, acc=acc, n=n)
        if is_act is not None:
            a = is_act[:, 1:][m]
            if a.any():
                st["action_acc"] = (pred[a] == tgt[a]).float().mean().item()
                st["action_ce"] = ce[a].mean().item()
        if diag_n_base is not None and n:  # 수정(2026-10-16): SFT 발산 원인 진단(로그 전용, 기울기 경로 불변)
            lf = logits.detach()
            st["logit_absmax"] = lf.abs().max().item()
            st["logit_max_base"] = lf[:, :diag_n_base].max().item()
            st["logit_max_new"] = lf[:, diag_n_base:].max().item()
            st["logit_std"] = lf.std().item()
            st["h_norm_max"] = hs.detach().norm(dim=-1).max().item()
            st["h_norm_mean"] = hs.detach().norm(dim=-1).mean().item()
            st["ce_max"] = ce.detach().max().item()
            seg = batch.get("seg")
            if seg is not None:
                sg = seg[:, 1:][m]
                for k, name in enumerate(SEG_NAMES):
                    sel = sg == k
                    if sel.any():
                        st[f"ce_{name}"] = ce[sel].mean().item()
                        st[f"n_{name}"] = int(sel.sum())
    return loss, n, st


# ---------------------------------------------------------------- 데이터셋 / 콜레이트
class TeacherSFTDataset:
    """표본 = (프롬프트: system + 6 영상 + user) → (정답 + <|im_end|>). 영상 전처리는 tools/qwen_cache.py 와 같은
    processor(min=max=200·28·28)."""

    def __init__(self, records, processor, system_mode="qwen_default", drop_cmd_p=0.0, drop_ego_p=0.0,
                 seed=0, text_only=False, max_answer_tokens=768, action_ids=None, use_prev=False,  # 수정(2026-10-15): v2-sensor 정답 547(+종료 1) 토큰 > 512
                 frame_order="prev_first", with_segments=False,  # 수정(2026-10-16): SFT 발산 원인 진단 — 구간 번호(로그 전용)
                 n_cams=NUM_CAMS):  # 수정(2026-10-29): NAVSIM 교사 SFT — 프레임당 카메라 수(기본 6 = nuScenes, NAVSIM 전방 3)
        self.with_segments = with_segments
        self.n_cams = n_cams
        self.recs = records
        self.proc = processor
        self.tok = processor.tokenizer
        self.system_mode = system_mode
        self.drop_cmd_p, self.drop_ego_p = drop_cmd_p, drop_ego_p
        self.seed, self.epoch = seed, 0
        self.text_only = text_only
        self.max_answer_tokens = max_answer_tokens
        self.action_ids = set(action_ids or [])
        self.use_prev, self.frame_order = use_prev, frame_order
        self.n_images = 0 if text_only else n_cams * (2 if use_prev else 1)

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.recs)

    def drops(self, i):
        rng = random.Random(hash((self.seed, self.epoch, i)) & 0xFFFFFFFF)
        return rng.random() < self.drop_cmd_p, rng.random() < self.drop_ego_p

    def prompt(self, rec, drop_cmd=False, drop_ego=False):
        user = build_user_text(rec, drop_cmd, drop_ego)
        msgs = build_messages(system_text(rec, self.system_mode), user, self.n_images)
        return self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def __getitem__(self, i):
        import torch
        rec = self.recs[i]
        dc, de = self.drops(i)
        text = self.prompt(rec, dc, de)
        if self.text_only:
            enc = self.proc(text=[text], return_tensors="pt")
        else:
            enc = self.proc(text=[text], images=load_images(rec, self.use_prev, self.frame_order, self.n_cams), return_tensors="pt")
        p_ids = enc["input_ids"][0].tolist()
        a_ids, a_lab, a_w = tokenize_answer(self.tok, rec)
        if len(a_ids) > self.max_answer_tokens:
            raise ValueError(f"{rec['token']}: 정답 {len(a_ids)} 토큰 > max_answer_tokens {self.max_answer_tokens} (잘라내지 않음)")
        ids = p_ids + a_ids
        lab = [-100] * len(p_ids) + a_lab
        w = [0.0] * len(p_ids) + a_w
        item = dict(input_ids=torch.tensor(ids), labels=torch.tensor(lab), weights=torch.tensor(w),
                    is_action_label=torch.tensor([t in self.action_ids for t in lab]),
                    n_prompt=len(p_ids), token=rec["token"], drop_cmd=dc, drop_ego=de)
        if self.with_segments:  # 수정(2026-10-16): SFT 발산 원인 진단
            sg = answer_segments(self.tok, rec)
            assert len(sg) == len(a_ids)
            item["seg"] = torch.tensor([0] * len(p_ids) + sg)
        if not self.text_only:
            item["pixel_values"] = enc["pixel_values"]
            item["image_grid_thw"] = enc["image_grid_thw"]
        return item


def collate(items, pad_id):
    import torch
    L = max(len(x["input_ids"]) for x in items)
    B = len(items)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    lab = torch.full((B, L), -100, dtype=torch.long)
    w = torch.zeros((B, L))
    am = torch.zeros((B, L), dtype=torch.long)
    act = torch.zeros((B, L), dtype=torch.bool)
    for b, x in enumerate(items):                         # 오른쪽 패딩(학습)
        n = len(x["input_ids"])
        ids[b, :n] = x["input_ids"]; lab[b, :n] = x["labels"]; w[b, :n] = x["weights"]; am[b, :n] = 1
        act[b, :n] = x["is_action_label"]
    out = dict(input_ids=ids, labels=lab, weights=w, attention_mask=am, is_action_label=act,
               tokens=[x["token"] for x in items])
    if "seg" in items[0]:  # 수정(2026-10-16): SFT 발산 원인 진단
        sg = torch.zeros((B, L), dtype=torch.long)
        for b, x in enumerate(items):
            sg[b, :len(x["seg"])] = x["seg"]
        out["seg"] = sg
    if "pixel_values" in items[0]:
        out["pixel_values"] = torch.cat([x["pixel_values"] for x in items], 0)
        out["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in items], 0)
    return out


class LengthGroupedBatchSampler:
    """버킷팅: 무작위 순서를 mega 묶음(batch×mega)으로 자른 뒤 묶음 안에서 길이순 정렬 → 배치 → 배치 순서 섞기.
    프롬프트(영상 1080 + 텍스트 ~540 토큰)가 길이의 ~80% 로 거의 일정하므로 정답 길이만으로 묶어도 패딩 낭비가 작다."""

    def __init__(self, lengths, batch_size, seed=0, mega=50, drop_last=False):
        self.lengths = np.asarray(lengths)
        self.bs, self.seed, self.mega, self.drop_last = batch_size, seed, mega, drop_last
        self.epoch = 0
        self.skip = 0          # 재개용: 이번 에폭 앞쪽 배치 수(순서는 epoch 씨드로 결정적)

    def set_epoch(self, e):
        self.epoch = e

    def batches(self):
        rng = np.random.RandomState(self.seed + 1000 * self.epoch)
        idx = rng.permutation(len(self.lengths))
        M = self.bs * self.mega
        out = []
        for s in range(0, len(idx), M):
            chunk = idx[s:s + M]
            chunk = chunk[np.argsort(-self.lengths[chunk], kind="stable")]
            for t in range(0, len(chunk), self.bs):
                b = chunk[t:t + self.bs].tolist()
                if len(b) < self.bs and self.drop_last:
                    continue
                out.append(b)
        order = rng.permutation(len(out))
        return [out[i] for i in order]

    def __iter__(self):
        return iter(self.batches()[self.skip:])

    def __len__(self):
        return len(self.batches())


def answer_lengths(tok, records):
    return [len(tok(r["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]) for r in records]


# ---------------------------------------------------------------- 메타(병합·평가·캐시가 학습과 같은 설정을 쓰게)
def save_meta(path, **kw):
    with open(path, "w") as f:
        json.dump(kw, f, indent=1, ensure_ascii=False)


def load_meta(model_dir):
    p = Path(model_dir) / "teacher_meta.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음 — train_sft.py / merge_teacher_lora.py 산출물이 아니다")
    return json.load(open(p))


# 수정(2026-10-16): 설계 검토 반영 — 교사 단계 연결(QA 전체 학습 → SFT → GRPO → 병합 → 캐시)의 fold·분할 해시 검사.
#   QA 단계(train_qa_full.py) 산출 모델 폴더에는 qa_meta.json 이 있다: stage="teacher_qa_full", fold, fold_sha256
#   (fold_split_sha256 이라는 이름도 받는다), base(원본 Qwen 경로). SFT 는 --base 가 이 폴더면 fold·해시가 같아야 하고,
#   qa_meta.json 이 없는 base 는 원본 Qwen(BASE_QWEN) 일 때만 허용한다. SFT meta 의 base_chain 에 그 결과를 기록하고,
#   GRPO·병합·캐시 추출은 verify_teacher_chain 으로 전 단계를 다시 검사한다(어긋나면 즉시 중단).
QA_META = "qa_meta.json"
QA_STAGES = ("teacher_qa_full",)


def _realpath(p):
    return os.path.realpath(str(p))


def base_chain_info(base, fold, fold_file=FOLD_FILE):
    """train_sft --base 검사. 반환 dict(kind='original_qwen'|'qa_full', path, ...). 어긋나면 SystemExit."""
    qm = Path(base) / QA_META
    if qm.exists():
        m = json.load(open(qm))
        stage, f = m.get("stage"), m.get("fold")
        sha = m.get("fold_sha256", m.get("fold_split_sha256"))
        if stage not in QA_STAGES:
            raise SystemExit(f"[거부] {qm}: stage={stage!r} (허용 {QA_STAGES})")
        if fold in (None, "none"):
            raise SystemExit(f"[거부] base {base} 는 fold {f} 전용 QA 모델 — --fold none SFT 의 base 로 쓸 수 없다")
        if f != fold:
            raise SystemExit(f"[거부] base QA 모델 fold={f} != 이번 fold {fold} (교차 적합 깨짐)")
        d, _ = load_fold_split(fold_file)
        if sha != d["sha256"]:
            raise SystemExit(f"[거부] base QA 모델 분할 해시 {str(sha)[:12]} != 현재 fold_split.json {d['sha256'][:12]}")
        qb = m.get("base")
        if qb is None or _realpath(qb) != _realpath(BASE_QWEN):
            raise SystemExit(f"[거부] {qm}: QA 단계 base={qb} 가 원본 Qwen({BASE_QWEN}) 이 아니다")
        return dict(kind="qa_full", path=_realpath(base), qa_meta_sha256=file_sha256(qm), stage=stage, fold=f,
                    fold_sha256=sha, qa_base=_realpath(qb))
    if _realpath(base) != _realpath(BASE_QWEN):
        raise SystemExit(f"[거부] base {base} 에 {QA_META} 가 없고 원본 Qwen({BASE_QWEN}) 도 아니다 — 출처 미상 base 금지")
    return dict(kind="original_qwen", path=_realpath(base))


def verify_teacher_chain(meta, fold, fold_file=FOLD_FILE):
    """teacher_meta(SFT 체크포인트 / GRPO 체크포인트 / 병합 모델) → QA → SFT → GRPO 전 단계의 fold·분할 해시 검사.
    반환 요약 dict(캐시 _meta.json 에 기록). 어긋나면 SystemExit."""
    d, _ = load_fold_split(fold_file)
    fi = meta.get("fold") or {}
    if fi.get("fold") != fold:
        raise SystemExit(f"[거부] 교사 meta fold={fi.get('fold')} != {fold}")
    if fi.get("fold_sha256") != d["sha256"]:
        raise SystemExit(f"[거부] 교사 학습 때 분할 해시 {str(fi.get('fold_sha256'))[:12]} != 현재 {d['sha256'][:12]}")
    bc = base_chain_info(meta["base"], fold, fold_file)
    rec = meta.get("base_chain")
    if rec is not None:
        for k in ("kind", "path", "qa_meta_sha256"):
            if rec.get(k) != bc.get(k):
                raise SystemExit(f"[거부] SFT 때 기록한 base_chain.{k}={rec.get(k)} != 지금 {bc.get(k)} (base 가 SFT 뒤에 바뀜)")
    elif bc["kind"] != "original_qwen":
        raise SystemExit("[거부] SFT meta 에 base_chain 기록이 없는데 base 가 QA 모델이다 — 이 검사 도입 전 SFT 는 원본 Qwen base 만 인정")
    chain = dict(fold=fold, fold_sha256=d["sha256"], qa=bc, sft=dict(fold=fi["fold"], fold_sha256=fi["fold_sha256"]),
                 grpo=None)
    g = meta.get("grpo")
    if g is not None:
        sft_ck = g.get("sft_ckpt")
        if not sft_ck or not (Path(sft_ck) / "teacher_meta.json").exists():
            raise SystemExit(f"[거부] GRPO meta 의 SFT 체크포인트 {sft_ck} 에 teacher_meta.json 이 없다")
        sm = load_meta(sft_ck)
        sfi = sm.get("fold") or {}
        if sfi.get("fold") != fold or sfi.get("fold_sha256") != d["sha256"]:
            raise SystemExit(f"[거부] GRPO 의 SFT 체크포인트 fold={sfi.get('fold')}·해시가 이번 fold {fold}·현재 분할과 다르다")
        if _realpath(sm["base"]) != _realpath(meta["base"]) or sm.get("base_chain") != meta.get("base_chain"):
            raise SystemExit("[거부] GRPO 가 기록한 base(·base_chain) 와 그 SFT 체크포인트의 base 가 다르다")
        msha = file_sha256(Path(sft_ck) / "teacher_meta.json")
        if g.get("sft_meta_sha256") is not None and g["sft_meta_sha256"] != msha:
            raise SystemExit(f"[거부] GRPO 이후 SFT 체크포인트 {sft_ck} 의 teacher_meta.json 이 바뀌었다(덮어쓴 체크포인트)")
        chain["sft"].update(ckpt=_realpath(sft_ck), teacher_meta_sha256=msha)
        chain["grpo"] = dict(fold=fold, sft_ckpt=_realpath(sft_ck))
    return chain


def original_cache_instruction():
    """tools/qwen_cache.py 의 INSTRUCTION 문자열을 import 부작용 없이(AST) 읽는다. qwen_online.py 사본과 일치 검사."""
    src = (REPO / "tools/qwen_cache.py").read_text()
    val = None
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "INSTRUCTION" for t in node.targets):
            val = ast.literal_eval(node.value)
    if val is None:
        raise RuntimeError("tools/qwen_cache.py 에서 INSTRUCTION 을 찾지 못했다")
    on = REPO / "models/encoder/vlm_qwen2vl/qwen_online.py"
    if on.exists():
        for node in ast.parse(on.read_text()).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "INSTRUCTION" for t in node.targets):
                if ast.literal_eval(node.value) != val:
                    raise RuntimeError("qwen_cache.py 와 qwen_online.py 의 INSTRUCTION 이 다르다")
    return val
