# 수정(2026-11-05): NAVSIM 교사 온라인(배치 1) 지연 — teacher_latency.py(nuScenes) 의 NAVSIM 판, 새 파일(기존 소스 수정 없음).
#   차이: 영상 6 장(직전·현재 전방 3 캠, prev_first, 현재 블록 3·4·5), 정답 549 토큰(answer_grammar_navsim), context = 현재 3 캠 영상 360 +
#   user 텍스트(laq_cache_layout_navsim), 캐시 data/navsim/laq_cache, 표본 = --samples JSON(make_navsim_samples 로 만든 목록).
"""nuScenes v2 교사(Qwen2-VL-2B 병합, layer-18 은닉) 온라인 지연 측정.

캐시(data/nuscenes/laq_cache, tools/vla_teacher/extract_teacher_cache_v2.py) 경로를 배치 1 로 재현하고
단계별(CPU 전처리 / 시각 인코더 / prefill / decode / 은닉 수집) 시간을 CUDA event + NVTX 로 잰다.
모든 설정은 같은 동등성 검사(생성 토큰 == 캐시 색인 gen_ids, 폴더별 은닉 vs 캐시 텐서·vs 배치1 기준)를 거친다.

엔진
  --engine hf   : 추출 스크립트와 같은 경로(HF generate + answer_grammar LogitsProcessor + 2 차 전체 순전파 lm_forward).
  --engine fast : 직접 만든 디코드 루프(HF 모듈 그대로, KV 는 미리 잡은 버퍼의 '정확 길이' 슬라이스 → HF DynamicCache 와
                  같은 커널 입력). 옵션:
      --grammar cpu|gpu        : 토큰마다 파이썬 문법 상태(동기화) | GPU 텐서 상태기계(미리 계산한 허용 마스크, 동기화 없음)
      --hidden refwd|inloop    : 2 차 전체 순전파 | prefill·decode 중 layer-18 출력 바로 수집(+ 마지막 토큰 1 step)
      --decode eager|graph_step|graph_seg|graph_all : 파이썬 루프 | 위치별 CUDA graph | 구간 graph | 디코드 전체 1 graph
      --vision perimg|perimg_graph|fa2_graph : 영상 1 장씩(캐시와 같음) | 같은 커널을 CUDA graph 로 | flash-attn varlen 12 장 한 번
      --prefill eager|graph, --prep serial|threads, --reuse_prev 0|1(이전 프레임 6 장 전처리·시각 특징 재사용),
      --lm_head all|last(prefill 에서 lm_head 를 모든 위치 | 마지막 위치만)
실행(GPU0 공용 — 측정은 flock 으로 직렬화):
  export CUDA_VISIBLE_DEVICES=0 HIPAD_ALLOW_GPU0=1
  flock /tmp/claude-500/gpu0_timing.lock python tools/latency/teacher_latency.py --engine hf --tag hf_ref --save_ref
"""
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/vla_teacher"))
import sft_common as S  # noqa: E402
import answer_grammar as AG  # noqa: E402
import laq_cache_layout as LY  # noqa: E402
import extract_teacher_cache_v2 as E  # noqa: E402
import laq_cache_layout_navsim as LN  # noqa: E402
import answer_grammar_navsim as AGN  # noqa: E402
import navsim_sft_common as NS  # noqa: E402
from types import SimpleNamespace  # noqa: E402

NC = LN.NUM_CAMS              # 3
NI = 2 * NC                   # 6 장(직전 3 + 현재 3)
NA = LN.ANSWER_TOKENS         # 549
AGN.configure()
_M = AGN.load_impl()
# nuScenes 판 코드가 부르는 이름을 NAVSIM 구현으로(모듈 사본 — 원본 모듈은 건드리지 않음)
LY = SimpleNamespace(FOLDERS=LN.FOLDERS, ANSWER_FOLDERS=LN.ANSWER_FOLDERS, ANSWER_TOKENS=NA, answer_template=LN.answer_template,
                     context_positions=lambda ids, a, b, c, cur_block=None: LN.context_positions(ids, a, b, c))
AG = SimpleNamespace(make_logits_processor=_M.make_logits_processor, grammar_table=_M.grammar_table)
_S0 = S


class _SNav:
    def __getattr__(self, k):
        return getattr(_S0, k)

    @staticmethod
    def load_images(r, use_prev, order):
        return _S0.load_images(r, use_prev, order, NC)

    @staticmethod
    def cam_paths(r, use_prev, order):
        return _S0.cam_paths(r, use_prev, order, NC)


S = _SNav()

CACHE = REPO / "data/navsim/laq_cache"   # 읽기 전용(GPU3 추출이 쓰는 중 — 색인에 있는 완료 표본만 읽는다)
OUT_DIR = REPO / "docs/latency/navsim"
SCRATCH = Path(os.environ.get("LAT_SCRATCH", "/tmp/claude-500/-home1-irteam/296e6594-177a-44bc-af56-250390370217/scratchpad"))


def check_gpu_env():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd != "0" or os.environ.get("HIPAD_ALLOW_GPU0") != "1":
        raise SystemExit("[거부] 이 측정은 GPU0 전용: CUDA_VISIBLE_DEVICES=0 HIPAD_ALLOW_GPU0=1")


def teacher_ckpt(T):
    m = json.load(open(CACHE / "plan" / "_meta.json"))
    return m["teachers"][T]["checkpoint"], m["teachers"][T]["generation"]


def load_samples(path, n):
    """make_navsim_samples.py 가 만든 [{entry(색인 행 또는 token·part), rec(입력 필드만)}] 앞 n 개."""
    d = json.load(open(path))
    rows = d["samples"][:n]
    if len(rows) < n:
        raise SystemExit(f"{path}: 표본 {len(rows)} < {n}")
    return [(x["entry"], x["rec"]) for x in rows], d


def stats(xs):
    xs = sorted(xs)
    if not xs:
        return None
    p90 = xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))]
    return dict(median=round(statistics.median(xs), 3), p90=round(p90, 3), mean=round(sum(xs) / len(xs), 3),
                min=round(xs[0], 3), max=round(xs[-1], 3), n=len(xs))


# ---------------------------------------------------------------- GPU 문법 상태기계(answer_grammar.AnswerGrammar 와 같은 규칙)
class GpuGrammar:
    """AnswerGrammar.key/step 을 GPU 텐서로. 위치 t 는 디코드 루프에서 파이썬 정수(정적)이므로 위치별 분기는 파이썬,
    상태(force_until, empty, 최근 4 토큰)는 GPU 텐서 → 토큰마다 CPU 동기화 없음, CUDA graph 캡처 가능.
    허용 밖 = −inf 로 바꾼 뒤 argmax(= mask_scores 의 scatter 결과와 같은 값 → 같은 argmax)."""

    def __init__(self, g, V, dev):
        import torch
        self.g, self.V, self.dev = g, V, dev
        n = g.n_pos
        self.B = torch.zeros(n, V, dtype=torch.bool, device=dev)
        for t in range(n):
            if g.allow[t]:
                self.B[t, torch.tensor(g.allow[t], device=dev)] = True
        self.Z = {}
        for t, lst in g.allow_nz.items():
            z = torch.zeros(V, dtype=torch.bool, device=dev); z[torch.tensor(lst, device=dev)] = True
            self.Z[t] = z
        self.Frow = torch.zeros(V, dtype=torch.bool, device=dev); self.Frow[g.fill] = True
        self.Nrow = torch.zeros(V, dtype=torch.bool, device=dev); self.Nrow[g.none] = True
        self.pat = torch.tensor([g.minus, g.zero, g.zero, g.dot], device=dev)
        self.fu = torch.zeros((), dtype=torch.long, device=dev)
        self.empty = torch.zeros((), dtype=torch.bool, device=dev)
        self.hist = torch.full((4,), -1, dtype=torch.long, device=dev)
        self.ninf = torch.tensor(float("-inf"), device=dev)

    def reset(self):
        self.fu.zero_(); self.empty.zero_(); self.hist.fill_(-1)

    def step(self, scores, t, out):
        """scores (1,V) fp32, t 정수 위치, out (1,) long 에 선택 토큰을 쓴다."""
        import torch
        g = self.g
        isF = self.fu > t
        row = self.B[t]
        if t in self.Z:
            row = torch.where((self.hist == self.pat).all(), self.Z[t], row)
        if t in g.slot_heads:
            row = torch.where(self.empty, self.Nrow, row)
        row = torch.where(isF, self.Frow, row)
        i = torch.where(row, scores[0], self.ninf).argmax(-1, keepdim=True)
        out.copy_(i)
        heads = g.heads[t]
        if heads:
            n = sum((i[0] == tid).long() * int(nn) for tid, nn in heads.items())
            self.fu.copy_(torch.where((~isF) & (n > 0), t + 1 + n, self.fu))
        if t in g.slot_heads:
            self.empty.copy_(self.empty | ((~isF) & (i[0] == g.none)))
        self.hist.copy_(torch.cat([self.hist[1:], i]))


# ---------------------------------------------------------------- KV 캐시: 미리 잡은 버퍼, 반환은 '정확 길이' 슬라이스
def make_slice_cache(n_layers, n_kv, maxlen, hd, dtype, dev):
    import torch
    from transformers.cache_utils import Cache

    class SliceCache(Cache):
        def __init__(self):
            super().__init__()
            self.k = [torch.zeros(1, n_kv, maxlen, hd, dtype=dtype, device=dev) for _ in range(n_layers)]
            self.v = [torch.zeros(1, n_kv, maxlen, hd, dtype=dtype, device=dev) for _ in range(n_layers)]
            self.cur = 0

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            n = key_states.shape[-2]
            s = self.cur
            self.k[layer_idx][:, :, s:s + n].copy_(key_states)
            self.v[layer_idx][:, :, s:s + n].copy_(value_states)
            return self.k[layer_idx][:, :, :s + n], self.v[layer_idx][:, :, :s + n]

        def get_seq_length(self, layer_idx=0):
            return self.cur

        def get_usable_length(self, new_seq_length, layer_idx=0):
            return self.cur

        def get_max_length(self):
            return None

        def get_max_cache_shape(self):
            return None

    return SliceCache()


# ---------------------------------------------------------------- 공통 설정
class Setup:
    def __init__(self, a):
        import torch
        from transformers import Qwen2VLForConditionalGeneration
        self.a = a
        self.ckpt, self.gen_meta = teacher_ckpt(a.teacher)
        self.meta = S.load_meta(self.ckpt)
        self.fo = self.meta.get("frame_order", "prev_first")
        if self.fo != "prev_first" or self.meta.get("num_cams") != NC:
            raise SystemExit(f"교사 meta frame_order {self.fo}·num_cams {self.meta.get('num_cams')} != prev_first·{NC}")
        self.cur_block = LN.CUR_BLOCKS
        self.proc = S.make_processor(self.ckpt, self.meta["pixels"])
        self.tok = self.proc.tokenizer
        self.tok.padding_side = "left"
        self.template = LY.answer_template(self.tok)
        if len(self.tok) != self.meta["vocab_after"]:
            raise SystemExit(f"토크나이저 크기 {len(self.tok)} != 학습 {self.meta['vocab_after']}")
        self.grammar = AGN.load(self.tok, AGN.MASK_FILE, NS.NAVSIM_GT_DIR)
        ids_of = self.tok.convert_tokens_to_ids
        self.IMG, self.VEND, self.IMEND, self.EOT = (ids_of("<|image_pad|>"), ids_of("<|vision_end|>"),
                                                     ids_of("<|im_end|>"), ids_of("<|endoftext|>"))
        self.dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
        t0 = time.time()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.ckpt, torch_dtype=self.dt,
                                                                    attn_implementation=a.attn)
        self.model.to("cuda").eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.vis_attn_cls = type(self.model.visual.blocks[0].attn).__name__
        self.lm_attn_cls = type(self.model.model.layers[0].self_attn).__name__
        print(f"[setup] {self.ckpt} {a.dtype} attn={a.attn} vision={self.vis_attn_cls} lm={self.lm_attn_cls} "
              f"load {time.time() - t0:.1f}s", flush=True)
        self.samples, sdoc = load_samples(a.samples, a.n + a.warmup)
        if sdoc["teacher"] != a.teacher:
            raise SystemExit(f"표본 파일 교사 {sdoc['teacher']} != --teacher {a.teacher}")
        self.sample_set = sdoc["set"]
        self.sys0 = S.system_text(self.samples[0][1], self.meta["system"])

    def prompt_text(self, r):
        return self.proc.apply_chat_template(S.build_messages(self.sys0, S.build_user_text(r), NI), tokenize=False,
                                             add_generation_prompt=True)

    def load_cache(self, token):
        import torch
        if not all((CACHE / k / f"{token}.pt").exists() for k in LY.FOLDERS):
            return None   # navtest 는 아직 캐시가 없다(추출 중) — 캐시 비교 없음으로 기록
        return {k: torch.load(CACHE / k / f"{token}.pt", map_location="cpu", weights_only=True)["hidden"]
                for k in LY.FOLDERS}


def compare(hs, ref):
    """폴더별 {bitwise, maxabs, cos_mean, cos_min}. hs, ref: {folder: (L,D)} (CPU)."""
    import torch
    out = {}
    for k in LY.FOLDERS:
        a, b = hs[k].float(), ref[k].float()
        c = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        out[k] = dict(bitwise=bool(torch.equal(hs[k], ref[k])), maxabs=float((a - b).abs().max()),
                      cos_mean=float(c.mean()), cos_min=float(c.min()))
    return out


def tok_check(gen, ref_ids):
    mism = [i for i, (x, y) in enumerate(zip(gen, ref_ids)) if x != y]
    return dict(equal=(len(mism) == 0 and len(gen) == len(ref_ids)), n_mismatch=len(mism),
                first_mismatch=(mism[0] if mism else -1))


# ---------------------------------------------------------------- HF 경로(추출 스크립트와 같음, 배치 1)
class HFEngine:
    def __init__(self, su):
        import torch
        self.su, self.m = su, su.model
        E.patch_visual_per_image(self.m)
        vis = self.m.visual
        inner = vis.forward
        self.vis_ev = []

        def timed(hidden_states, grid_thw):
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push("vision"); e0.record()
            o = inner(hidden_states, grid_thw)
            e1.record(); torch.cuda.nvtx.range_pop()
            self.vis_ev.append((e0, e1))
            return o
        vis.forward = timed

    def run(self, ent, r):
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList
        su, m, a = self.su, self.m, self.su.a
        Ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        torch.cuda.nvtx.range_push("cpu_prep")
        text = su.prompt_text(r)
        imgs = S.load_images(r, True, su.fo)
        enc = su.proc(text=[text], images=imgs, return_tensors="pt", padding=True)
        torch.cuda.nvtx.range_pop()
        cpu_prep = (time.perf_counter() - w0) * 1e3
        e_h0, e_h1, e_g0, e_g1, e_f0, e_f1 = (Ev() for _ in range(6))
        e_h0.record()
        enc = {k: v.to("cuda", non_blocking=False) for k, v in enc.items()}
        enc["pixel_values"] = enc["pixel_values"].to(su.dt)
        e_h1.record()
        P = enc["input_ids"].shape[1]
        gp = AG.make_logits_processor(su.grammar, P, 1)
        ev_in, ev_out = [], []

        class Timed(LogitsProcessor):
            def __call__(self, input_ids, scores):
                e = Ev(); e.record(); ev_in.append(e)
                torch.cuda.nvtx.range_push("grammar")
                s = gp(input_ids, scores)
                torch.cuda.nvtx.range_pop()
                e = Ev(); e.record(); ev_out.append(e)
                return s
        gen_kw = dict(do_sample=False, min_new_tokens=NA, max_new_tokens=NA, eos_token_id=[su.IMEND, su.EOT],
                      pad_token_id=su.tok.pad_token_id, use_cache=True)
        self.vis_ev = []
        e_g0.record()
        torch.cuda.nvtx.range_push("generate")
        with torch.no_grad():
            out = m.generate(**enc, **gen_kw, logits_processor=LogitsProcessorList([Timed()]))
        torch.cuda.nvtx.range_pop()
        e_g1.record()
        vis_gen = list(self.vis_ev)
        gen = out[:, P:]
        # 은닉 수집 = 추출 스크립트와 같은 2 차 전체 순전파(시각 인코더 다시 포함)
        self.vis_ev = []
        e_f0.record()
        torch.cuda.nvtx.range_push("hidden_refwd")
        full = torch.cat([enc["input_ids"], gen], 1)
        batch = dict(input_ids=full, attention_mask=torch.ones_like(full), pixel_values=enc["pixel_values"],
                     image_grid_thw=enc["image_grid_thw"])
        cap = {}
        hk = m.model.layers[a.layer - 1].register_forward_hook(lambda mod, i, o: cap.__setitem__("h", o[0]))
        with torch.no_grad():
            S.lm_forward(m, batch, "mrope")
        hk.remove()
        H = cap["h"][0]
        ctx_pos, _ = LY.context_positions(full[0, :P].tolist(), su.IMG, su.VEND, su.IMEND, su.cur_block)
        hs = {"context": H[ctx_pos]}
        for k in LY.ANSWER_FOLDERS:
            hs[k] = H[[P + q for q in su.template[k]]]
        torch.cuda.nvtx.range_pop()
        e_f1.record()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - w0) * 1e3
        el = lambda x, y: x.elapsed_time(y)  # noqa: E731
        vis = sum(el(x, y) for x, y in vis_gen)
        n = len(ev_in)
        gram = [el(ev_in[k], ev_out[k]) for k in range(n)]
        fwd = [0.0] + [el(ev_out[k - 1], ev_in[k]) for k in range(1, n)]
        tm = dict(wall=wall, cpu_prep=cpu_prep, h2d=el(e_h0, e_h1), gen_pre_vision=el(e_g0, vis_gen[0][0]),
                  vision=vis, vision_span=el(vis_gen[0][0], vis_gen[-1][1]),
                  prefill=el(vis_gen[-1][1], ev_in[0]), decode_fwd=sum(fwd), grammar=sum(gram),
                  decode_total=el(ev_out[0], ev_out[-1]) + gram[0], gen_tail=el(ev_out[-1], e_g1),
                  hidden=el(e_f0, e_f1), hidden_vision=sum(el(x, y) for x, y in self.vis_ev))
        tm["per_pos"] = [fwd[k] + gram[k] for k in range(n)]
        tm["n_steps"] = n
        return gen[0].tolist(), {k: v.to("cpu", torch.bfloat16) for k, v in hs.items()}, tm, \
            dict(P=P, input_ids=enc["input_ids"][0].cpu())


# ---------------------------------------------------------------- 빠른 경로
class FastEngine:
    def __init__(self, su):
        import torch
        a = su.a
        self.su, self.a = su, a
        self.m = su.model
        self.lm = self.m.model
        self.vis = self.m.visual
        cfg = self.m.config
        self.dev = torch.device("cuda")
        self.N = LY.ANSWER_TOKENS
        self.g = su.grammar
        self.V = self.m.lm_head.out_features
        self.gg = GpuGrammar(self.g, self.V, self.dev) if a.grammar == "gpu" else None
        # 고정 프롬프트 모양(전 표본 동일해야 함 — 표본마다 검사)
        r0 = su.samples[0][1]
        enc0 = su.proc(text=[su.prompt_text(r0)], images=S.load_images(r0, True, su.fo), return_tensors="pt")
        self.grid1 = enc0["image_grid_thw"][:1].clone()          # (1,3) 영상 1 장
        self.grid12 = enc0["image_grid_thw"].clone()
        if not bool((self.grid12 == self.grid1).all()):
            raise RuntimeError("영상 격자가 영상마다 다르다")
        self.npatch = int(self.grid1.prod())
        self.ntok_img = self.npatch // 4
        ids0 = enc0["input_ids"]
        self.P = int(ids0.shape[1])
        pos, delta = self.lm_rope(ids0)
        self.pos_prefill_ref = pos
        self.img_pos = (ids0[0] == su.IMG).nonzero()[:, 0].to(self.dev)
        ctx_pos, _ = LY.context_positions(ids0[0].tolist(), su.IMG, su.VEND, su.IMEND, su.cur_block)
        self.ctx_idx = torch.tensor(ctx_pos, device=self.dev)
        self.L_ctx = len(ctx_pos)
        # 버퍼
        self.maxlen = self.P + self.N + 1
        self.cache = make_slice_cache(cfg.num_hidden_layers, cfg.num_key_value_heads, self.maxlen,
                                      cfg.hidden_size // cfg.num_attention_heads, su.dt, self.dev)
        self.ids_buf = torch.zeros(1, self.P, dtype=torch.long, device=self.dev)
        self.pos_prefill = torch.zeros(3, 1, self.P, dtype=torch.long, device=self.dev)
        self.pos_dec = torch.zeros(3, 1, self.N + 1, dtype=torch.long, device=self.dev)
        self.cp_dec = torch.arange(self.P, self.P + self.N + 1, device=self.dev)
        self.cp_pre = torch.arange(0, self.P, device=self.dev)
        self.gen = torch.zeros(self.N + 1, dtype=torch.long, device=self.dev)
        D = cfg.hidden_size
        self.hbuf = torch.zeros(self.N, D, dtype=su.dt, device=self.dev)
        self.ctx_buf = torch.zeros(self.L_ctx, D, dtype=su.dt, device=self.dev)
        self.pix_buf = torch.zeros(NI * self.npatch, enc0["pixel_values"].shape[1], dtype=su.dt, device=self.dev)
        self.img_emb = torch.zeros(NI * self.ntok_img, D, dtype=su.dt, device=self.dev)
        self.hook_mode = None
        self._hidx = 0
        self.lm.layers[a.layer - 1].register_forward_hook(self._hook)
        # 시각 인코더 상수(영상 1 장 격자 고정)
        self.rot1 = self.vis.rot_pos_emb(self.grid1.to(self.dev))
        self.mask1 = torch.ones(1, self.npatch, self.npatch, dtype=torch.bool, device=self.dev)
        self.cu1 = torch.tensor([0, self.npatch], dtype=torch.int32, device=self.dev)
        self.rot12 = self.vis.rot_pos_emb(self.grid12.to(self.dev))
        self.cu12 = torch.arange(0, NI + 1, dtype=torch.int32, device=self.dev) * self.npatch
        self.graphs = {}
        self.captured = False
        self.pool = None
        self.prev_cache = {}
        self.pool_exec = ThreadPoolExecutor(NI) if a.prep == "threads" else None

    def lm_rope(self, ids):
        import torch
        pos, delta = self.m.get_rope_index(ids.to(self.dev), self.grid12.to(self.dev), None,
                                           torch.ones_like(ids).to(self.dev))
        return pos, delta

    def _hook(self, mod, inp, out):
        if self.hook_mode == "prefill":
            self.ctx_buf.copy_(out[0][0].index_select(0, self.ctx_idx))
        elif self.hook_mode == "step":
            self.hbuf[self._hidx].copy_(out[0][0, -1])

    # ---- CPU 전처리
    def _img_one(self, path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        o = self.su.proc.image_processor(images=[img], return_tensors="pt")
        return o["pixel_values"]

    def prep(self, r, skip_paths=()):
        """반환 (ids (1,P) long cpu, pix 목록[12] (npatch, C) 또는 None(재사용)). 처리기와 같은 연산을 영상 1 장씩."""
        import torch
        su = self.su
        paths = S.cam_paths(r, True, su.fo)
        text = su.prompt_text(r)
        t = text
        while "<|image_pad|>" in t:
            t = t.replace("<|image_pad|>", "<|placeholder|>" * self.ntok_img, 1)
        t = t.replace("<|placeholder|>", "<|image_pad|>")
        todo = [(i, p) for i, p in enumerate(paths) if p not in skip_paths]
        if self.pool_exec is not None:
            futs = [(i, self.pool_exec.submit(self._img_one, p)) for i, p in todo]
            ids = su.tok([t], return_tensors="pt", padding=True)["input_ids"]
            pix = {i: f.result() for i, f in futs}
        else:
            ids = su.tok([t], return_tensors="pt", padding=True)["input_ids"]
            pix = {i: self._img_one(p) for i, p in todo}
        return ids, [pix.get(i) for i in range(NI)], paths

    # ---- 시각 인코더
    def _vis_attn_exact(self, attn, x, rot, mask):
        """VisionSdpaAttention.forward 와 같은 연산(마스크만 미리 계산 — 영상 1 장이면 전부 True)."""
        import torch
        from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision
        n = x.shape[0]
        q, k, v = attn.qkv(x).reshape(n, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rot).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rot).squeeze(0)
        o = torch.nn.functional.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                                                             mask, dropout_p=0.0)
        return attn.proj(o.transpose(0, 1).reshape(n, -1))

    def _vis_attn_fa2(self, attn, x, rot, cu, maxlen):
        from flash_attn import flash_attn_varlen_func
        from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision
        n = x.shape[0]
        q, k, v = attn.qkv(x).reshape(n, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rot).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rot).squeeze(0)
        o = flash_attn_varlen_func(q, k, v, cu, cu, maxlen, maxlen).reshape(n, -1)
        return attn.proj(o)

    def _vis_blocks(self, px, attn_fn):
        x = self.vis.patch_embed(px)
        for blk in self.vis.blocks:
            x = x + attn_fn(blk.attn, blk.norm1(x))
            x = x + blk.mlp(blk.norm2(x))
        return self.vis.merger(x)

    def vision_images(self, idxs):
        """pix_buf 의 영상 idxs → img_emb (해당 영상 칸). 모드별."""
        mode = self.a.vision
        if mode == "fa2_graph" and len(idxs) > 1 and idxs == list(range(idxs[0], idxs[0] + len(idxs))):
            i0, n = idxs[0], len(idxs)
            px = self.pix_buf[i0 * self.npatch:(i0 + n) * self.npatch]
            rot = self.rot12[:n * self.npatch]
            cu = self.cu12[:n + 1]
            o = self._vis_blocks(px, lambda at, x: self._vis_attn_fa2(at, x, rot, cu, self.npatch))
            self.img_emb[i0 * self.ntok_img:(i0 + n) * self.ntok_img].copy_(o)
            return
        for i in idxs:
            px = self.pix_buf[i * self.npatch:(i + 1) * self.npatch]
            if mode == "perimg":   # 캐시와 같은 경로: HF visual.forward(영상 1 장)
                o = self.vis(px, grid_thw=self.grid1.to(self.dev))
            else:
                o = self._vis_blocks(px, lambda at, x: self._vis_attn_exact(at, x, self.rot1, self.mask1))
            self.img_emb[i * self.ntok_img:(i + 1) * self.ntok_img].copy_(o)

    # ---- prefill / decode step
    def prefill(self):
        import torch
        emb = self.lm.embed_tokens(self.ids_buf)
        emb = emb.index_copy(1, self.img_pos, self.img_emb.unsqueeze(0).to(emb.dtype))
        self.cache.cur = 0
        self.hook_mode = "prefill" if self.a.hidden == "inloop" else None
        out = self.lm(input_ids=None, inputs_embeds=emb, position_ids=self.pos_prefill, attention_mask=None,
                      past_key_values=self.cache, cache_position=self.cp_pre, use_cache=True, return_dict=True)
        self.hook_mode = None
        hl = out.last_hidden_state
        if self.a.lm_head == "all":
            sc = self.m.lm_head(hl)[:, -1, :].float()
        else:
            sc = self.m.lm_head(hl[:, -1:, :])[:, -1, :].float()
        self.select(sc, 0)

    def select(self, sc, t):
        """위치 t 토큰 선택 → self.gen[t]."""
        import torch
        if self.gg is not None:
            self.gg.step(sc, t, self.gen[t:t + 1])
        else:   # HF 와 같은 파이썬 상태(토큰마다 동기화)
            if t > 0:
                self.g.step(self._st, int(self.gen[t - 1]))
            s = self.g.mask_scores(sc, [self.g.key(self._st)])
            self.gen[t:t + 1].copy_(torch.argmax(s, dim=-1))

    def step(self, j):
        """답 토큰 j 를 입력(캐시 위치 P+j). j < N-1 이면 토큰 j+1 선택. 은닉(inloop)은 hbuf[j]."""
        ids = self.gen[j:j + 1].view(1, 1)
        emb = self.lm.embed_tokens(ids)
        self.cache.cur = self.P + j
        self._hidx = j
        self.hook_mode = "step" if self.a.hidden == "inloop" else None
        out = self.lm(input_ids=None, inputs_embeds=emb, position_ids=self.pos_dec[:, :, j:j + 1], attention_mask=None,
                      past_key_values=self.cache, cache_position=self.cp_dec[j:j + 1], use_cache=True, return_dict=True)
        self.hook_mode = None
        if j < self.N - 1:
            sc = self.m.lm_head(out.last_hidden_state)[:, -1, :].float()
            self.select(sc, j + 1)

    def n_steps(self):
        return self.N if self.a.hidden == "inloop" else self.N - 1

    # ---- CUDA graph
    def capture(self):
        import torch
        a = self.a
        self.pool = torch.cuda.graph_pool_handle()
        t0 = time.time()
        if a.decode in ("graph_step", "graph_seg", "graph_all"):
            if a.decode == "graph_step":
                ranges = [(j, j + 1) for j in range(self.n_steps())]
            elif a.decode == "graph_seg":
                ranges = self.seg_ranges()
            else:
                ranges = [(0, self.n_steps())]
            self.dec_graphs = []
            for j0, j1 in ranges:
                gph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gph, pool=self.pool):
                    for j in range(j0, j1):
                        self.step(j)
                self.dec_graphs.append(((j0, j1), gph))
        if a.prefill == "graph":
            gph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gph, pool=self.pool):
                self.prefill()
            self.pre_graph = gph
        if a.vision in ("perimg_graph", "fa2_graph"):
            self.vis_graphs = {}
            for key, idxs in (("all", list(range(NI))), ("cur", list(self.su.cur_block))):
                gph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gph, pool=self.pool):
                    self.vision_images(idxs)
                self.vis_graphs[key] = gph
        torch.cuda.synchronize()
        self.captured = True
        self.capture_s = time.time() - t0
        print(f"[capture] {self.capture_s:.1f}s", flush=True)

    def seg_ranges(self):
        """폴더·머리말 경계로 나눈 구간(구간별 시간 → 폴더별 비용). 칸 하나 = 57 step."""
        cuts = sorted({0, self.n_steps()} | {3 + 57 * k for k in range(9)} | {459, 463, 531, 535, 537, 541})
        cuts = [c for c in cuts if c <= self.n_steps()]
        return list(zip(cuts[:-1], cuts[1:]))

    # ---- 표본 1 개
    def run(self, ent, r, prev_paths=()):
        import torch
        a, su = self.a, self.su
        Ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        torch.cuda.nvtx.range_push("cpu_prep")
        reuse = set(prev_paths) if a.reuse_prev else set()
        ids, pix, paths = self.prep(r, skip_paths=reuse)
        torch.cuda.nvtx.range_pop()
        cpu_prep = (time.perf_counter() - w0) * 1e3
        ev = {k: Ev() for k in ("h0", "h1", "v0", "v1", "p0", "p1", "d0", "d1", "f0", "f1")}
        ev["h0"].record()
        if ids.shape[1] != self.P:
            raise RuntimeError(f"프롬프트 길이 {ids.shape[1]} != {self.P} — 고정 길이 가정 위반")
        self.ids_buf.copy_(ids, non_blocking=False)
        need = []
        for i in range(NI):
            if pix[i] is not None:
                self.pix_buf[i * self.npatch:(i + 1) * self.npatch].copy_(pix[i])
                need.append(i)
        ev["h1"].record()
        # 위치(M-RoPE): 프롬프트 배치가 고정이면 표본마다 같다 — 검사용으로 매번 계산하지 않고 기준과 같은지 1 회 확인
        if not getattr(self, "_pos_ok", False):
            pos, delta = self.lm_rope(ids)
            if not torch.equal(pos, self.pos_prefill_ref):
                raise RuntimeError("M-RoPE 위치가 표본마다 다르다")
            self.pos_prefill.copy_(pos)
            self.pos_dec.copy_((self.cp_dec + delta.to(self.dev).view(-1)).view(1, 1, -1).expand(3, 1, -1))
            self._pos_ok = True
        # 이전 프레임 재사용: 앞 틱의 현재 6 장 특징(경로 키)을 이전 칸에 복사
        ev["v0"].record()
        torch.cuda.nvtx.range_push("vision")
        for i in range(NI):
            if pix[i] is None:
                self.img_emb[i * self.ntok_img:(i + 1) * self.ntok_img].copy_(self.prev_cache[paths[i]])
        if a.vision in ("perimg_graph", "fa2_graph") and self.captured and need:
            key = "all" if len(need) == NI else "cur"
            if key == "cur" and need != list(su.cur_block):
                raise RuntimeError(f"재사용 칸 {need} 이 현재 블록과 다르다")
            self.vis_graphs[key].replay()
        else:
            self.vision_images(need)
        torch.cuda.nvtx.range_pop()
        ev["v1"].record()
        # 다음 틱 재사용용: 현재 블록 특징 보관(복사 시간은 vision 밖, 무시할 수준)
        if a.reuse_prev:
            self.prev_cache = {paths[i]: self.img_emb[i * self.ntok_img:(i + 1) * self.ntok_img].clone()
                               for i in su.cur_block}
        if self.gg is not None:
            self.gg.reset()
        else:
            self._st = self.g.start()
        ev["p0"].record()
        torch.cuda.nvtx.range_push("prefill")
        if a.prefill == "graph" and self.captured:
            self.pre_graph.replay()
        else:
            self.prefill()
        torch.cuda.nvtx.range_pop()
        ev["p1"].record()
        torch.cuda.nvtx.range_push("decode")
        per = []
        if a.decode == "eager" or not self.captured:
            evs = [Ev() for _ in range(self.n_steps() + 1)]
            evs[0].record()
            for j in range(self.n_steps()):
                self.step(j)
                evs[j + 1].record()
            per = [(j, j + 1, evs[j], evs[j + 1]) for j in range(self.n_steps())]
        else:
            evs = [Ev() for _ in range(len(self.dec_graphs) + 1)]
            evs[0].record()
            for n, ((j0, j1), gph) in enumerate(self.dec_graphs):
                gph.replay()
                evs[n + 1].record()
            per = [(j0, j1, evs[n], evs[n + 1]) for n, ((j0, j1), _) in enumerate(self.dec_graphs)]
        torch.cuda.nvtx.range_pop()
        ev["d1"].record()
        ev["f0"].record()
        torch.cuda.nvtx.range_push("hidden")
        if a.hidden == "inloop":
            H_ctx, H_ans = self.ctx_buf, self.hbuf
        else:   # 추출과 같은 2 차 전체 순전파(시각 특징은 이미 계산한 것 재사용하지 않고 HF 경로 그대로)
            from types import SimpleNamespace  # noqa: F401
            full = torch.cat([self.ids_buf, self.gen[:self.N].view(1, -1)], 1)
            pv = self.pix_buf
            batch = dict(input_ids=full, attention_mask=torch.ones_like(full), pixel_values=pv,
                         image_grid_thw=self.grid12.to(self.dev))
            cap = {}
            hk = self.lm.layers[a.layer - 1].register_forward_hook(lambda mod, i, o: cap.__setitem__("h", o[0]))
            S.lm_forward(self.m, batch, "mrope")
            hk.remove()
            H = cap["h"][0]
            H_ctx, H_ans = H.index_select(0, self.ctx_idx), H[self.P:self.P + self.N]
        hs = {"context": H_ctx}
        for k in LY.ANSWER_FOLDERS:
            hs[k] = H_ans[su.template[k]]
        torch.cuda.nvtx.range_pop()
        ev["f1"].record()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - w0) * 1e3
        el = lambda x, y: x.elapsed_time(y)  # noqa: E731
        tm = dict(wall=wall, cpu_prep=cpu_prep, h2d=el(ev["h0"], ev["h1"]), vision=el(ev["v0"], ev["v1"]),
                  prefill=el(ev["p0"], ev["p1"]), decode_total=el(ev["p1"], ev["d1"]), hidden=el(ev["f0"], ev["f1"]),
                  n_steps=self.n_steps())
        tm["segments"] = [(j0, j1, el(x, y)) for j0, j1, x, y in per]
        if a.decode in ("eager", "graph_step") or not self.captured:
            tm["per_pos"] = self.per_pos_from_steps(tm["segments"])
        gen = self.gen[:self.N].tolist()
        return gen, {k: v.to("cpu", torch.bfloat16) for k, v in hs.items()}, tm, dict(P=self.P, input_ids=ids[0])

    def per_pos_from_steps(self, segs):
        """step j(입력 j) 가 토큰 j+1 을 만든다 → 위치 q 비용 = step q-1. 위치 0 은 prefill(context 로 셈) → 0.
        inloop 의 마지막 step(입력 546, 은닉만)은 위치 546 에 더한다(그 토큰 은닉을 얻는 비용)."""
        pp = [0.0] * self.N
        for j0, j1, ms in segs:
            q = j0 + 1
            if q < self.N:
                pp[q] += ms
            else:
                pp[self.N - 1] += ms
        return pp


# ---------------------------------------------------------------- 실행
def gpu_load_snapshot():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True, timeout=10)
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader", "-i", "0"],
                           capture_output=True, text=True, timeout=10)
        return dict(gpu0=out.stdout.strip(), apps=q.stdout.strip().splitlines())
    except Exception as ex:   # 진단용 스냅숏일 뿐(측정값에는 영향 없음)
        return dict(error=str(ex))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="A", choices=["A", "B"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--engine", default="hf", choices=["hf", "fast"])
    ap.add_argument("--grammar", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--hidden", default="inloop", choices=["refwd", "inloop"])
    ap.add_argument("--decode", default="eager", choices=["eager", "graph_step", "graph_seg", "graph_all"])
    ap.add_argument("--vision", default="perimg", choices=["perimg", "perimg_graph", "fa2_graph"])
    ap.add_argument("--prefill", default="eager", choices=["eager", "graph"])
    ap.add_argument("--prep", default="serial", choices=["serial", "threads"])
    ap.add_argument("--reuse_prev", type=int, default=0)
    ap.add_argument("--lm_head", default="all", choices=["all", "last"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", required=True, help="make_navsim_samples.py 표본 JSON")
    ap.add_argument("--save_ref", action="store_true", help="이 실행의 은닉을 배치1 기준으로 저장(hf 엔진)")
    ap.add_argument("--ref_tag", default="hf_ref")
    a = ap.parse_args()
    check_gpu_env()
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False   # 기본값 명시(수치 고정)
    su = Setup(a)
    if a.dtype == "fp32":
        print(f"[attn] {S.exact_fp32_attention(su.dt)}", flush=True)
    eng = HFEngine(su) if a.engine == "hf" else FastEngine(su)
    ref_dir = SCRATCH / f"ref_nav_{a.ref_tag}"
    if a.save_ref:
        ref_dir.mkdir(parents=True, exist_ok=True)
    load0 = gpu_load_snapshot()
    with torch.no_grad():
        if a.engine == "fast":
            # 첫 표본으로 eager 예열(cuBLAS 작업공간 등) 후 graph 캡처
            ent, r = su.samples[0]
            eng.run(ent, r)
            if any(x in (a.decode, a.vision, a.prefill) for x in ("graph_step", "graph_seg", "graph_all",
                                                                    "perimg_graph", "fa2_graph", "graph")):
                eng.capture()
        res = []
        for n, (ent, r) in enumerate(su.samples):
            prev = ()
            if a.engine == "fast" and a.reuse_prev:
                # 앞 틱 모사: 이전 프레임 6 장을 '현재'로 처리했던 결과를 미리 만든다(시간 밖)
                prev_paths = S.cam_paths(r, True, su.fo)[:NC]
                ids_, pix_, _ = eng.prep(r)
                for i in range(NC):
                    eng.pix_buf[i * eng.npatch:(i + 1) * eng.npatch].copy_(pix_[i])
                if a.vision in ("perimg_graph", "fa2_graph"):
                    eng.vis_graphs["all"].replay()
                else:
                    eng.vision_images(list(range(NC)))
                eng.prev_cache = {p: eng.img_emb[i * eng.ntok_img:(i + 1) * eng.ntok_img].clone()
                                  for i, p in enumerate(prev_paths)}
                prev = prev_paths
            gen, hs, tm, info = eng.run(ent, r, prev) if a.engine == "fast" else eng.run(ent, r)
            cached = su.load_cache(ent["token"])
            e = dict(token=ent["token"], warmup=n < a.warmup, tm=tm, P=info["P"])
            if cached is not None:
                e["tok_vs_cache"] = tok_check(gen, ent["gen_ids"])
                e["hid_vs_cache"] = compare(hs, cached)
            e["grammar_ok"] = su.grammar.accepts(gen)[0]
            rp = ref_dir / f"{ent['token']}.pt"
            if a.save_ref:
                torch.save(dict(hs=hs, gen=gen, input_ids=info["input_ids"]), rp)
            elif rp.exists():
                ref = torch.load(rp, weights_only=True)
                e["tok_vs_ref"] = tok_check(gen, ref["gen"])
                e["hid_vs_ref"] = compare(hs, ref["hs"])
                e["input_ids_vs_ref"] = bool(torch.equal(info["input_ids"].cpu(), ref["input_ids"].cpu()))
            res.append(e)
            print(f"  [{n}/{len(su.samples)}] wall {tm['wall']:.0f}ms prep {tm['cpu_prep']:.0f} vis {tm['vision']:.1f} "
                  f"pre {tm['prefill']:.1f} dec {tm['decode_total']:.0f} hid {tm['hidden']:.1f} | tok==cache "
                  f"{(str(e['tok_vs_cache']['equal']) + ' (' + str(e['tok_vs_cache']['n_mismatch']) + ')') if 'tok_vs_cache' in e else '-'} "
                  f"{'tok==ref ' + str(e['tok_vs_ref']['equal']) if 'tok_vs_ref' in e else ''}", flush=True)
    load1 = gpu_load_snapshot()
    meas = [e for e in res if not e["warmup"]]
    keys = [k for k, v in meas[0]["tm"].items() if isinstance(v, (int, float))]
    summ = {k: stats([e["tm"][k] for e in meas]) for k in keys}
    if "per_pos" in meas[0]["tm"]:
        import numpy as np
        pp = np.median(np.array([e["tm"]["per_pos"] for e in meas]), 0)
        summ["per_pos_median"] = [round(float(x), 4) for x in pp]
        summ["per_folder_ms"] = {k: round(float(pp[su.template[k]].sum()), 2) for k in LY.ANSWER_FOLDERS}
        hdr = AG.grammar_table(su.tok)["header"]
        summ["per_folder_ms"]["headers"] = round(float(pp[hdr].sum()), 2)
    if meas[0]["tm"].get("segments") and len(meas[0]["tm"]["segments"]) < 100:
        import numpy as np
        segs = meas[0]["tm"]["segments"]
        summ["segments_median"] = [(j0, j1, round(float(np.median([e["tm"]["segments"][i][2] for e in meas])), 3))
                                   for i, (j0, j1, _) in enumerate(segs)]

    def agg_cmp(field):
        if field not in meas[0]:
            return None
        out = {}
        for k in LY.FOLDERS:
            xs = [e[field][k] for e in meas]
            out[k] = dict(bitwise_all=all(x["bitwise"] for x in xs), n_bitwise=sum(x["bitwise"] for x in xs),
                          maxabs_max=max(x["maxabs"] for x in xs), maxabs_median=statistics.median(x["maxabs"] for x in xs),
                          cos_mean=sum(x["cos_mean"] for x in xs) / len(xs), cos_min=min(x["cos_min"] for x in xs))
        return out
    mc = [e for e in meas if "tok_vs_cache" in e]
    eq = dict(n=len(meas), n_cache=len(mc),
              tok_equal_cache=sum(e["tok_vs_cache"]["equal"] for e in mc),
              tok_mismatch_cache=[(e["token"][:8], e["tok_vs_cache"]["n_mismatch"], e["tok_vs_cache"]["first_mismatch"])
                                  for e in mc if not e["tok_vs_cache"]["equal"]],
              grammar_ok=sum(e["grammar_ok"] for e in meas),
              hid_vs_cache=agg_cmp("hid_vs_cache") if len(mc) == len(meas) and mc else None)
    mref = [e for e in meas if "tok_vs_ref" in e]
    if mref:
        meas_all, meas = meas, mref
        eq.update(n_ref=len(mref), tok_equal_ref=sum(e["tok_vs_ref"]["equal"] for e in meas),
                  tok_mismatch_ref=[(e["token"][:8], e["tok_vs_ref"]["n_mismatch"], e["tok_vs_ref"]["first_mismatch"])
                                    for e in meas if not e["tok_vs_ref"]["equal"]],
                  input_ids_equal_ref=sum(e["input_ids_vs_ref"] for e in meas),
                  hid_vs_ref=agg_cmp("hid_vs_ref"))
        meas = meas_all
    doc = dict(tag=a.tag, args=vars(a), sample_set=su.sample_set, ckpt=su.ckpt, cache_generation=su.gen_meta, vision_attn=su.vis_attn_cls,
               lm_attn=su.lm_attn_cls, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
               capture_s=getattr(eng, "capture_s", None), gpu_load_before=load0, gpu_load_after=load1,
               summary=summ, equivalence=eq, samples=[{k: v for k, v in e.items() if k != "tm"} | dict(
                   tm={k: v for k, v in e["tm"].items() if k not in ("per_pos",)}) for e in res])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    op = OUT_DIR / f"teacher_latency_{a.tag}.json"
    json.dump(doc, open(op, "w"), indent=1, ensure_ascii=False)
    brief = {k: (v["median"], v["p90"]) for k, v in summ.items() if isinstance(v, dict) and "median" in v}
    print(f"[summary] {a.tag} median/p90 ms: {brief}", flush=True)
    print(f"[equiv] tok==cache {eq['tok_equal_cache']}/{eq['n_cache']} grammar_ok {eq['grammar_ok']} "
          f"{('tok==ref ' + str(eq['tok_equal_ref'])) if 'tok_equal_ref' in eq else ''}", flush=True)
    for fld in ("hid_vs_cache", "hid_vs_ref"):
        if eq.get(fld):
            print(f"[{fld}] " + " | ".join(f"{k}: bit {v['n_bitwise']}/{eq['n']} max {v['maxabs_max']:.3g} "
                                           f"cos {v['cos_mean']:.6f}/{v['cos_min']:.4f}" for k, v in eq[fld].items()),
                  flush=True)
    if "per_folder_ms" in summ:
        print(f"[per_folder_ms] {summ['per_folder_ms']}", flush=True)
    print(f"-> {op}", flush=True)


if __name__ == "__main__":
    main()
