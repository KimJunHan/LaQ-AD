# 수정(2026-11-09): 새 파일 — B2D 폐쇄루프 온라인 교사 빌더(LAQ_TEACHER_BUILDER="laq_teacher_builder_b2d:build").
#   hipad_b2d_laq_teacher_agent.py 의 계약:
#     build(device) → callable(sim_t, *, cams, cams_prev, prev_available, tick_data, ego, ego_line)
#       → dict(qwen_hidden=(1, L, D) bf16, qwen_mask=(1, L) uint8, qwen_type=(1, L) int64, ego_line=<프롬프트에 실제로 넣은 자차 줄>)
#     (TeacherRefreshScheduler.step 이 teacher_fn(t, **inputs) 로 부른다. 키워드 이름은 에이전트 tick() 의 호출과 같다.)
#   교사 캐시(tools/vla_teacher/extract_teacher_cache_b2d.py run())와 같은 경로를 표본 1 개에 대해 그대로 밟는다:
#     프롬프트   = system(qwen_default) + 12 영상(직전 6 → 현재 6, prev_first, 캠 순서 front, front_left, front_right, back_left,
#                  back, back_right) + user 텍스트 = common_b2d.join_prompt(build_prompt_parts(명령, 목표점, 자차 값))
#                  (교사 정답 생성기 gen_teacher_gt_b2d.build_record 와 같은 함수·같은 인자 형태 → record["prompt"]["user"] 와 같은 문자열)
#     전처리     = sft_common.make_processor(모델, meta.pixels = 200·28·28), apply_chat_template(add_generation_prompt), 왼쪽 패딩
#     생성       = HF generate(do_sample=False, min=max=547) + answer_grammar 문법 제약(answer_grammar_b2d.load, B2D 547)
#     은닉       = 생성된 전체 열을 sft_common.lm_forward(mrope)로 다시 순전파, layer-<L>(기본 18) forward hook 출력
#     자르기     = laq_cache_layout.context_positions(현재 블록 6..11 영상 1,080 + user 텍스트) + answer_template 고정 위치(6 폴더)
#     저장 형식  = 폴더별 bf16 → TEACHER_CACHE_FOLDERS 순서로 이어 붙임(LoadTeacherCacheNusc 와 같은 cat·mask 1·type 0..6)
#   도우미는 extract_teacher_cache_b2d 모듈(=캐시 생성 코드)이 import 한 같은 객체(X, LY, AGB, sft_common 함수)를 쓴다(복사 아님).
#   ego·ego_line: 에이전트가 넘긴 ego(자른 값)로 common_b2d.ego_line 을 다시 만들어 넘겨받은 ego_line 과 비교한다(다르면 즉시 실패).
#     그 줄이 프롬프트 자차 줄로 들어가고, 반환 dict 의 "ego_line" 으로 돌려준다(에이전트가 학생 E2 값을 이 줄에서 다시 읽음).
#   명령·목표점: tick_data 의 command(RoadOption, 학습 info command_near 와 같은 값)와 target_xy·pos·compass 로
#     hipad_b2d_agent.run_step 과 같은 식(학생이 받는 target_point = 데이터셋 target_point_far 규약)으로 만든다.
#   설정(환경변수):
#     LAQ_TEACHER_MODEL   병합 교사 폴더(teacher_meta.json + processor, 필수)
#     LAQ_TEACHER_FOLD    A|B (필수, meta.fold 와 같아야 함)
#     LAQ_TEACHER_LAYER   기본 18 (캐시 _meta.layer 와 같아야 함)
#     LAQ_TEACHER_DTYPE   bf16|fp32 기본 bf16 (캐시 generation.dtype 과 같아야 함)
#     LAQ_TEACHER_ATTN    sdpa|eager 기본 sdpa (캐시 generation.attn 과 같아야 함)
#     LAQ_TEACHER_CACHE_ROOT  캐시 루트(기본 data/bench2drive/laq_cache). 7 폴더 _meta.json 과 설정·체크포인트·문법·위치 해시를 대조한다.
#                         대조 없이 쓰려면 명시적으로 "none"(스모크 전용 — 학습/평가 불일치를 잡지 못함, 로그에 경고).
#     LAQ_TEACHER_GEN_BATCH / LAQ_TEACHER_FWD_BATCH  기본 = 캐시 기록 generation.batch / fwd_batch 가 없으면 1.
#                         같은 표본을 그 수만큼 복제해 배치로 돌리고 0 번 행을 쓴다(캐시 추출의 배치 크기와 같은 커널을 고르게 하려는
#                         선택 — 배치 1 과 배치 16 의 bf16 동률 뒤집힘이 토큰을 바꾸는 현상(docs/latency/navsim_idle_20261105.md §1b) 대응.
#                         효과는 GPU 에서 tools/verify_b2d/test_teacher_builder_b2d.py --gpu 로 확인해야 하는 가설).
#     LAQ_TEACHER_LOG     호출별 기록 jsonl(선택)
#   실패 처리: 문법 제약 생성인데 layout 이 어긋나거나, 문맥 길이·텍스트 구간이 다르면 즉시 예외(건너뛰지 않음).
import hashlib
import json
import os
import sys
import threading
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_VLA_DIR = os.path.join(_REPO, "tools", "vla_teacher")
TEACHER_CACHE_FOLDERS = ("context", "detection", "tracking", "motion", "map", "plan", "action")  # loading.TEACHER_CACHE_FOLDERS 와 같은 순서
N_CAMS = 6

__all__ = ["build", "OnlineTeacherB2D", "target_point_from_tick", "command_value", "user_text", "TEACHER_CACHE_FOLDERS"]

_E = None
_SINGLETON = {}
_LOCK = threading.Lock()


def _mods():
    """캐시 생성 모듈(extract_teacher_cache_b2d) 적재 — B2D shim 설치가 그 모듈 첫 줄에서 일어난다(common_v2 → common_b2d)."""
    global _E
    if _E is None:
        if _VLA_DIR not in sys.path:
            sys.path.insert(0, _VLA_DIR)
        import extract_teacher_cache_b2d as E  # noqa: E402
        if os.path.realpath(os.path.dirname(E.__file__)) != os.path.realpath(_VLA_DIR):
            raise ImportError(f"extract_teacher_cache_b2d 가 다른 경로에서 적재됐다: {E.__file__}")
        if E.X.AG.V is not E.B or sys.modules.get("common_v2") is not E.B:
            raise RuntimeError("B2D shim 미적용(common_v2 ≠ common_b2d) — 다른 모듈이 nuScenes common_v2 를 먼저 import 했다")
        _E = E
    return _E


def command_value(cmd):
    """RoadOption(IntEnum) 또는 정수 → int. 정답 생성기와 같은 raw 값(−1 → 4 변환은 common_b2d.norm_command 가 한다)."""
    v = getattr(cmd, "value", cmd)
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise TypeError(f"명령 값 형식 {type(cmd)} {cmd!r} (정수/RoadOption 기대)")
    return int(v)


def target_point_from_tick(tick_data):
    """hipad_b2d_agent.SparseAgent.run_step 의 target_point 식 그대로(학생 입력과 같은 값, float32 (2,)).
    = 데이터셋 get_data_info 의 target_point_far(라이다 좌표, +x 오른쪽 +y 앞) 규약."""
    pos, tgt = tick_data["pos"], tick_data["target_xy"]
    ego_pos = [pos[0], -pos[1]]
    target_xy = [tgt[0], -tgt[1]]
    raw_theta = tick_data["compass"] if not np.isnan(tick_data["compass"]) else 0
    target_xy = np.array([target_xy[0] - ego_pos[0], target_xy[1] - ego_pos[1]])
    rotation_matrix = np.array([[np.cos(raw_theta), -np.sin(raw_theta)],
                                [np.sin(raw_theta), np.cos(raw_theta)]])
    return np.array(rotation_matrix @ target_xy, dtype=np.float32)


def user_text(command, target_point, ego, ego_line=None):
    """(raw 명령, 목표점 (2,), 자차 (speed, accel, yaw)) → (user 텍스트, 자차 줄). 정답 생성기 build_record 와 같은 두 함수.
    ego_line 이 주어지면 다시 만든 줄과 같아야 한다(교사·학생 단일 출처 검사)."""
    B = _mods().B
    if ego is None or len(ego) != 3:
        raise ValueError(f"ego 는 (speed, accel, yaw) 3 값이어야 한다: {ego!r}")
    parts = B.build_prompt_parts(command, target_point, tuple(float(v) for v in ego))
    if ego_line is not None and parts["ego"] != ego_line:
        raise RuntimeError(f"자차 줄 불일치: 에이전트 {ego_line!r} != 교사 프롬프트 {parts['ego']!r}")
    return B.join_prompt(parts), parts["ego"]


def _sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


class OnlineTeacherB2D:
    def __init__(self, model_dir, fold, device="cuda", layer=18, dtype="bf16", attn="sdpa", cache_root=None,
                 gen_batch=None, fwd_batch=None, log_path=None):
        E = _mods()
        import torch
        from transformers import Qwen2VLForConditionalGeneration
        self.E, self.torch = E, torch
        S = sys.modules["sft_common"]
        self.S = S
        if fold not in ("A", "B"):
            raise ValueError(f"LAQ_TEACHER_FOLD {fold!r} (A|B)")
        self.model_dir, self.fold, self.layer = str(model_dir), fold, int(layer)
        meta = S.load_meta(model_dir)
        # 캐시 생성기 check_meta 와 같은 B2D 설계값(배정·누설 검사는 캐시 생성 쪽 일이라 여기서는 하지 않는다)
        for k, v in (("pixels", E.BS.B2D_PIXELS), ("use_prev", True), ("frame_order", "prev_first"), ("rope", "mrope"),
                     ("system", "qwen_default")):
            if meta.get(k) != v:
                raise RuntimeError(f"교사 meta {k}={meta.get(k)!r} != {v!r} (B2D 교사 설계)")
        if (meta.get("fold") or {}).get("fold") != fold:
            raise RuntimeError(f"교사 meta fold={(meta.get('fold') or {}).get('fold')} != LAQ_TEACHER_FOLD {fold}")
        if (meta.get("b2d") or {}).get("kind") != "b2d_teacher_gt":
            raise RuntimeError("교사 meta 에 b2d 기록 없음 — train_sft_b2d.py 로 학습한 B2D 교사가 아니다")
        self.meta = meta
        self.dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        self.dtype_name, self.attn = dtype, attn
        self.proc = S.make_processor(model_dir, meta["pixels"])
        tok = self.proc.tokenizer
        tok.padding_side = "left"
        if len(tok) != meta["vocab_after"]:
            raise RuntimeError(f"토크나이저 크기 {len(tok)} != 학습 {meta['vocab_after']}")
        self.tok = tok
        self.template = E.LY.answer_template(tok)
        if {k: len(v) for k, v in self.template.items()} != E.LY.EXPECTED_COUNTS:
            raise RuntimeError(f"정답 템플릿 길이 {({k: len(v) for k, v in self.template.items()})} != {E.LY.EXPECTED_COUNTS}")
        self.grammar = E.AGB.load(tok, E.AGB.MASK_FILE)
        ids_of = tok.convert_tokens_to_ids
        self.IMG, self.VEND, self.IMEND, self.EOT = (ids_of("<|image_pad|>"), ids_of("<|vision_end|>"), ids_of("<|im_end|>"),
                                                     ids_of("<|endoftext|>"))
        self.device = device
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=self.dt, attn_implementation=attn)
        model.to(device).eval()
        self.attn_note = S.exact_fp32_attention(self.dt) if device != "cpu" else "cpu"
        for p in model.parameters():
            p.requires_grad_(False)
        E.X.patch_visual_per_image(model)
        self.model = model
        self.D = int(model.config.hidden_size)
        if not (1 <= self.layer <= model.config.num_hidden_layers):
            raise RuntimeError(f"layer {self.layer} 범위 밖(1..{model.config.num_hidden_layers})")
        self.sys0 = S.system_text({}, meta["system"])
        # 문맥 길이: 명령·목표점·자차 줄이 고정 토큰 수(정답 생성기 설계) → 대표 값으로 잰다. 매 호출 실제 길이와 대조.
        probe, _ = user_text(4, (0.0, 10.0), (0.0, 0.0, 0.0))
        self.n_user = len(tok(probe, add_special_tokens=False)["input_ids"])
        self.L_ctx = E.LY.NUM_CAMS * E.LY.TOKENS_PER_CAM + self.n_user
        self.Ls = dict(context=self.L_ctx, **{k: len(self.template[k]) for k in E.LY.ANSWER_FOLDERS})
        pos_doc = {k: self.template[k] for k in E.LY.ANSWER_FOLDERS}
        self.pos_sha = E.X.sha_text(json.dumps(pos_doc, sort_keys=True) + f"|ctx{self.L_ctx}|prev_first|b2d")
        self.instr_sha = _sha(_mods().B.build_instruction())
        self.cache_check = self._check_cache_meta(cache_root)
        gen_rec = (self.cache_check or {}).get("generation") or {}
        self.gen_batch = int(gen_batch) if gen_batch else int(gen_rec.get("batch", 1))
        self.fwd_batch = int(fwd_batch) if fwd_batch else min(int(gen_rec.get("fwd_batch", 1)), self.gen_batch)
        if self.gen_batch < 1 or self.fwd_batch < 1 or self.fwd_batch > self.gen_batch:
            raise ValueError(f"gen_batch {self.gen_batch} / fwd_batch {self.fwd_batch}")
        self.log_path = log_path
        self.n_calls = 0
        self.call_log = []
        print(f"[LaQ-teacher-builder] {model_dir} fold {fold} layer {self.layer} {dtype} {attn} {device} D {self.D} "
              f"L {self.Ls} 합 {sum(self.Ls.values())} | 복제 배치 gen {self.gen_batch} fwd {self.fwd_batch} | "
              f"캐시 대조 {'없음(경고: none)' if self.cache_check is None else 'OK'} | attn {self.attn_note}", flush=True)

    # ------------------------------------------------------------------ 캐시 _meta 대조
    def _check_cache_meta(self, cache_root):
        if cache_root is None or str(cache_root).lower() == "none":
            print("[LaQ-teacher-builder][경고] LAQ_TEACHER_CACHE_ROOT=none — 캐시 _meta 대조 생략(스모크 전용)", flush=True)
            return None
        E = self.E
        exp_ck = E.X.checkpoint_sha(self.model_dir)
        tinfo0 = None
        for k in TEACHER_CACHE_FOLDERS:
            mp = os.path.join(str(cache_root), k, "_meta.json")
            if not os.path.exists(mp):
                raise FileNotFoundError(f"캐시 _meta 없음: {mp} (대조 없이 쓰려면 LAQ_TEACHER_CACHE_ROOT=none)")
            m = json.load(open(mp))
            want = dict(layer=self.layer, L=self.Ls[k], hidden_dim=self.D, answer_tokens=E.LY.ANSWER_TOKENS,
                        positions_sha256=self.pos_sha, prompt_mode=E.PROMPT_MODE, dataset="b2d",
                        constrained_decoding=E.X.AG.CONSTRAINT_NAME, answer_grammar_sha256=self.grammar.info["sha256"])
            bad = {kk: (m.get(kk), vv) for kk, vv in want.items() if m.get(kk) != vv}
            if bad:
                raise RuntimeError(f"캐시 {mp} 와 온라인 교사 설정 불일치 {bad} (캐시값, 온라인값)")
            t = (m.get("teachers") or {}).get(self.fold)
            if t is None:
                raise RuntimeError(f"캐시 {mp} 에 teacher_{self.fold} 기록이 없다")
            if t.get("checkpoint_sha") != exp_ck:
                raise RuntimeError(f"캐시 {k} teacher_{self.fold} checkpoint_sha {str(t.get('checkpoint_sha'))[:12]} != "
                                   f"LAQ_TEACHER_MODEL {exp_ck[:12]} (캐시를 만든 교사와 다른 가중치)")
            if t.get("system_sha256") != E.X.sha_text(self.sys0):
                raise RuntimeError(f"캐시 {k}: system 문장 해시 불일치")
            if self.instr_sha not in (t.get("instruction_sha256") or []):
                raise RuntimeError(f"캐시 {k}: 지시문 해시 {self.instr_sha[:12]} 가 캐시 기록 {t.get('instruction_sha256')} 에 없다")
            g = t.get("generation") or {}
            for kk, vv in (("dtype", self.dtype_name), ("attn", self.attn), ("pixels", self.meta["pixels"]),
                           ("frame_order", "prev_first"), ("rope", "mrope"), ("visual", "per_image"), ("do_sample", False)):
                if g.get(kk) != vv:
                    raise RuntimeError(f"캐시 {k} generation.{kk}={g.get(kk)!r} != 온라인 {vv!r}")
            tinfo0 = tinfo0 or t
        # 캐시 _meta generation 에는 batch 만 있고 fwd_batch(2 차 순전파 묶음)는 없다. 체인(chain_b2d_teacher_after_qa.sh)은
        #   --fwd_batch 를 넘기지 않으므로 extract_teacher_cache_b2d.py 의 argparse 기본값 4 가 쓰였다 — 그 값을 쓴다(체인 인자가 바뀌면 같이 바꿀 것).
        gen = dict(tinfo0.get("generation") or {})
        gen.setdefault("fwd_batch", min(4, int(gen.get("batch", 1))))
        return dict(checkpoint_sha=exp_ck, generation=gen)

    # ------------------------------------------------------------------ 호출
    def __call__(self, sim_t, *, cams, cams_prev, prev_available, tick_data, ego, ego_line=None):
        t0 = time.time()
        cmd = command_value(tick_data["command"])
        tp = target_point_from_tick(tick_data)
        text, line = user_text(cmd, tp, ego, ego_line)
        if len(cams) != N_CAMS or len(cams_prev) != N_CAMS:
            raise ValueError(f"영상 수 현재 {len(cams)} / 직전 {len(cams_prev)} != {N_CAMS}")
        imgs = [im.convert("RGB") for im in list(cams_prev) + list(cams)]     # prev_first(캐시 load_images 와 같은 순서·RGB)
        out = self.infer(imgs, text)
        out["ego_line"] = line
        rec = dict(sim_t=float(sim_t), command=cmd, target_point=[float(v) for v in tp], ego=[float(v) for v in ego],
                   prev_available=bool(prev_available), user_sha=_sha(text)[:16], sec=round(time.time() - t0, 3),
                   gen_sha=out["teacher_info"]["gen_sha"], layout_ok=out["teacher_info"]["layout_ok"],
                   replica_gen_diff=int(out["teacher_info"]["replica_gen_diff"]))   # 복제 배치 행끼리 생성이 달랐던 행 수(0 이어야 정상)
        self.n_calls += 1
        self.call_log = self.call_log[-999:] + [rec]   # 프로세스가 경로 여러 개를 돌므로 최근 1,000 개만(전체는 LAQ_TEACHER_LOG)
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return out

    def infer(self, imgs, text, return_folders=False):
        """12 영상(prev_first) + user 텍스트 → 교사 출력. extract_teacher_cache_b2d.run 의 배치 루프 본문(표본 1 개)과 같은 계산."""
        torch, E, S = self.torch, self.E, self.S
        G, F = self.gen_batch, self.fwd_batch
        prompt = self.proc.apply_chat_template(S.build_messages(self.sys0, text, E.N_IMG), tokenize=False,
                                               add_generation_prompt=True)
        if not prompt.endswith("<|im_start|>assistant\n"):
            raise RuntimeError("프롬프트가 생성 프롬프트로 끝나지 않는다")
        enc = self.proc(text=[prompt] * G, images=list(imgs) * G, return_tensors="pt", padding=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        enc["pixel_values"] = enc["pixel_values"].to(self.dt)
        P = enc["input_ids"].shape[1]
        ans_n = E.LY.ANSWER_TOKENS
        gen_kw = dict(do_sample=False, min_new_tokens=ans_n, max_new_tokens=ans_n,
                      eos_token_id=[self.IMEND, self.EOT], pad_token_id=self.tok.pad_token_id, use_cache=True)
        from transformers import LogitsProcessorList
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kw, logits_processor=LogitsProcessorList(
                [E.X.AG.make_logits_processor(self.grammar, P, G)]))
        gen = out[:, P:]
        if gen.shape[1] != ans_n:
            raise RuntimeError(f"생성 길이 {gen.shape[1]} != {ans_n}")
        n_rep_diff = int(sum(int(not torch.equal(gen[i], gen[0])) for i in range(1, G)))
        full = torch.cat([enc["input_ids"], gen], 1)
        am = torch.cat([enc["attention_mask"], torch.ones_like(gen)], 1)
        g = enc["image_grid_thw"]
        per = [int(a * b * c) for a, b, c in g.tolist()]
        i1 = E.N_IMG * F
        batch = dict(input_ids=full[:F], attention_mask=am[:F], pixel_values=enc["pixel_values"][:sum(per[:i1])],
                     image_grid_thw=g[:i1])
        cap = {}
        hk = self.model.model.layers[self.layer - 1].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0]))
        try:
            with torch.no_grad():
                S.lm_forward(self.model, batch, "mrope")
        finally:
            hk.remove()
        H = cap["h"]
        ids0 = full[0, :P].tolist()
        ctx_pos, (ua, ub) = E.LY.context_positions(ids0, self.IMG, self.VEND, self.IMEND, E.CUR_BLOCK)
        if len(ctx_pos) != self.L_ctx:
            raise RuntimeError(f"context {len(ctx_pos)} != {self.L_ctx} (프롬프트 길이 비고정)")
        if self.tok.decode(full[0, ua:ub].tolist()) != text:
            raise RuntimeError("context 텍스트 구간이 user 텍스트와 다르다")
        info = E.analyse_answer(self.tok, gen[0].tolist(), self.template, self.grammar)
        if not info["layout_ok"]:
            raise RuntimeError(f"layout_bad (grammar_ok={info['grammar_ok']} pos={info['grammar_bad_pos']}, "
                               f"parse={info['parse_error']}) — 문법 제약 생성인데 위치가 어긋났다(캐시 생성기와 같은 즉시 중단)")
        hs = {"context": H[0, ctx_pos]}
        for k in E.LY.ANSWER_FOLDERS:
            hs[k] = H[0, [P + q for q in self.template[k]]]
        hs = {k: v.to(torch.bfloat16).contiguous() for k, v in hs.items()}     # 캐시: .to("cpu", bfloat16) 저장 → 로더가 bf16 그대로 읽음
        hcat, mcat, tcat = [], [], []
        for tid, k in enumerate(TEACHER_CACHE_FOLDERS):
            if tuple(hs[k].shape) != (self.Ls[k], self.D):
                raise RuntimeError(f"{k}: {tuple(hs[k].shape)} != ({self.Ls[k]}, {self.D})")
            if not bool(torch.isfinite(hs[k]).all()):
                raise RuntimeError(f"{k}: 은닉에 NaN/Inf")
            hcat.append(hs[k])
            mcat.append(torch.ones(self.Ls[k], dtype=torch.uint8, device=hs[k].device))
            tcat.append(torch.full((self.Ls[k],), tid, dtype=torch.long, device=hs[k].device))
        res = dict(qwen_hidden=torch.cat(hcat, 0)[None], qwen_mask=torch.cat(mcat, 0)[None], qwen_type=torch.cat(tcat, 0)[None],
                   teacher_info=dict(gen_ids=gen[0].tolist(), gen_sha=_sha(json.dumps(gen[0].tolist()))[:16],
                                     layout_ok=info["layout_ok"], parse_ok=info["parse_ok"], replica_gen_diff=n_rep_diff,
                                     gen_text=info["gen_text"]))
        if return_folders:
            res["folders"] = {k: v.detach().to("cpu") for k, v in hs.items()}
        return res


def _env(name, default=None, required=False):
    v = os.environ.get(name, default)
    if required and not v:
        raise RuntimeError(f"환경변수 {name} 가 필요하다")
    return v


def build(device="cuda"):
    """에이전트 계약 진입점. 같은 프로세스에서 경로마다 setup 이 다시 불리므로 같은 설정이면 적재한 교사를 재사용한다."""
    cfg = dict(model_dir=_env("LAQ_TEACHER_MODEL", required=True), fold=_env("LAQ_TEACHER_FOLD", required=True),
               device=device, layer=int(_env("LAQ_TEACHER_LAYER", "18")), dtype=_env("LAQ_TEACHER_DTYPE", "bf16"),
               attn=_env("LAQ_TEACHER_ATTN", "sdpa"),
               cache_root=_env("LAQ_TEACHER_CACHE_ROOT", os.path.join(_REPO, "data", "bench2drive", "laq_cache")),
               gen_batch=_env("LAQ_TEACHER_GEN_BATCH") or None, fwd_batch=_env("LAQ_TEACHER_FWD_BATCH") or None,
               log_path=_env("LAQ_TEACHER_LOG") or None)
    key = json.dumps(cfg, sort_keys=True)
    with _LOCK:
        if key not in _SINGLETON:
            if _SINGLETON:
                raise RuntimeError("같은 프로세스에서 다른 교사 설정으로 다시 만들려 한다 — 하나만 허용")
            _SINGLETON[key] = OnlineTeacherB2D(**cfg)
        return _SINGLETON[key]
