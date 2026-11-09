# 수정(2026-11-05): 교사 context-only 지연(1단계 추론용) — 새 파일(기존 소스 수정 없음, teacher_latency.py 재사용).
"""교사 A context-only 경로 = CPU 전처리 + 시각 인코더 + prefill(앞 18 층까지, layer-18 은닉 context 행만), 디코드 없음.

teacher_latency.FastEngine 의 전처리·시각 인코더(perimg | perimg_graph)·위치(M-RoPE)를 그대로 쓰고, prefill 만
LM 의 embed → rotary → layers[:18] 까지 돌린다(19–28 층·norm·lm_head·KV 저장 없음). 1단계 학생은 context 은닉만 쓴다.
동등성: context 은닉을 (a) 배치1 HF 기준(ref_hf_ref, 전체 28 층 prefill + 2 차 순전파 경로), (b) 캐시 텐서와 비교.
  --ctx_kv none  : use_cache=False(KV 안 씀)                 ← 기본(context-only 실사용)
  --ctx_kv slice : 기존 prefill 과 같은 SliceCache 에 KV 기록(동등성 대조용)
실행: CUDA_VISIBLE_DEVICES=0 HIPAD_ALLOW_GPU0=1 flock /tmp/claude-500/gpu0_timing.lock \
      python tools/latency/teacher_context_latency.py --tag ctx_graph_reuse --vision perimg_graph --prefill graph --prep threads --reuse_prev 1
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 수정(2026-11-05): --dataset navsim → teacher_latency_navsim(영상 6 장·NAVSIM 캐시·표본 파일). 클래스 정의 전에 골라야 해서 argv 를 먼저 본다.
DATASET = "navsim" if "--dataset" in sys.argv and sys.argv[sys.argv.index("--dataset") + 1] == "navsim" else "nuscenes"
if DATASET == "navsim":
    import teacher_latency_navsim as TL  # noqa: E402
    NCUR = TL.NC
    NIMG = TL.NI
else:
    import teacher_latency as TL  # noqa: E402
    NCUR = 6
    NIMG = 12

LY = TL.LY
S = TL.S


class CtxEngine(TL.FastEngine):
    def ctx_prefill(self):
        """embed(+영상 특징) → layers[:L] → context 행만 ctx_buf 로."""
        import torch
        emb = self.lm.embed_tokens(self.ids_buf)
        h = emb.index_copy(1, self.img_pos, self.img_emb.unsqueeze(0).to(emb.dtype))
        pe = self.lm.rotary_emb(h, self.pos_prefill)
        use_kv = self.a.ctx_kv == "slice"
        self.cache.cur = 0
        for layer in self.lm.layers[:self.a.layer]:
            h = layer(h, attention_mask=None, position_ids=self.pos_prefill,
                      past_key_value=self.cache if use_kv else None, output_attentions=False, use_cache=use_kv,
                      cache_position=self.cp_pre, position_embeddings=pe)[0]
        self.ctx_buf.copy_(h[0].index_select(0, self.ctx_idx))

    def capture(self):
        import torch
        a = self.a
        self.pool = torch.cuda.graph_pool_handle()
        t0 = time.time()
        if a.prefill == "graph":
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self.ctx_prefill()
            self.pre_graph = g
        if a.vision == "perimg_graph":
            self.vis_graphs = {}
            for key, idxs in (("all", list(range(NIMG))), ("cur", list(self.su.cur_block))):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    self.vision_images(idxs)
                self.vis_graphs[key] = g
        torch.cuda.synchronize()
        self.captured = True
        self.capture_s = time.time() - t0
        print(f"[capture] {self.capture_s:.1f}s", flush=True)

    def run(self, ent, r, prev_paths=()):
        import torch
        a, su = self.a, self.su
        Ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        reuse = set(prev_paths) if a.reuse_prev else set()
        ids, pix, paths = self.prep(r, skip_paths=reuse)
        cpu_prep = (time.perf_counter() - w0) * 1e3
        ev = {k: Ev() for k in ("h0", "h1", "v0", "v1", "p0", "p1")}
        ev["h0"].record()
        if ids.shape[1] != self.P:
            raise RuntimeError(f"프롬프트 길이 {ids.shape[1]} != {self.P}")
        self.ids_buf.copy_(ids)
        need = []
        for i in range(NIMG):
            if pix[i] is not None:
                self.pix_buf[i * self.npatch:(i + 1) * self.npatch].copy_(pix[i])
                need.append(i)
        ev["h1"].record()
        if not getattr(self, "_pos_ok", False):
            pos, delta = self.lm_rope(ids)
            if not torch.equal(pos, self.pos_prefill_ref):
                raise RuntimeError("M-RoPE 위치가 표본마다 다르다")
            self.pos_prefill.copy_(pos)
            self._pos_ok = True
        ev["v0"].record()
        for i in range(NIMG):
            if pix[i] is None:
                self.img_emb[i * self.ntok_img:(i + 1) * self.ntok_img].copy_(self.prev_cache[paths[i]])
        if a.vision == "perimg_graph" and self.captured and need:
            key = "all" if len(need) == NIMG else "cur"
            if key == "cur" and need != list(su.cur_block):
                raise RuntimeError(f"재사용 칸 {need} 이 현재 블록과 다르다")
            self.vis_graphs[key].replay()
        else:
            self.vision_images(need)
        ev["v1"].record()
        if a.reuse_prev:
            self.prev_cache = {paths[i]: self.img_emb[i * self.ntok_img:(i + 1) * self.ntok_img].clone()
                               for i in su.cur_block}
        ev["p0"].record()
        if a.prefill == "graph" and self.captured:
            self.pre_graph.replay()
        else:
            self.ctx_prefill()
        ev["p1"].record()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - w0) * 1e3
        el = lambda x, y: x.elapsed_time(y)  # noqa: E731
        tm = dict(wall=wall, cpu_prep=cpu_prep, h2d=el(ev["h0"], ev["h1"]), vision=el(ev["v0"], ev["v1"]),
                  prefill18=el(ev["p0"], ev["p1"]), gpu_total=el(ev["h0"], ev["p1"]))
        return self.ctx_buf.to("cpu", torch.bfloat16), tm, ids[0]


def cmp1(a, b):
    import torch
    x, y = a.float(), b.float()
    c = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    return dict(bitwise=bool(torch.equal(a, b)), maxabs=float((x - y).abs().max()), cos_mean=float(c.mean()),
                cos_min=float(c.min()))


def agg(xs):
    return dict(n=len(xs), n_bitwise=sum(x["bitwise"] for x in xs), maxabs_max=max(x["maxabs"] for x in xs),
                cos_mean=sum(x["cos_mean"] for x in xs) / len(xs), cos_min=min(x["cos_min"] for x in xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="A")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--vision", default="perimg", choices=["perimg", "perimg_graph"])
    ap.add_argument("--prefill", default="eager", choices=["eager", "graph"])
    ap.add_argument("--prep", default="serial", choices=["serial", "threads"])
    ap.add_argument("--reuse_prev", type=int, default=0)
    ap.add_argument("--ctx_kv", default="none", choices=["none", "slice"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ref_tag", default="hf_ref")
    ap.add_argument("--out_dir", default=str(TL.OUT_DIR))
    ap.add_argument("--dataset", default="nuscenes", choices=["nuscenes", "navsim"])
    ap.add_argument("--samples", default=None, help="navsim: make_navsim_samples.py 표본 JSON")
    a = ap.parse_args()
    # FastEngine 이 읽는 나머지 인자(디코드 관련은 쓰지 않음)
    a.grammar, a.hidden, a.decode, a.lm_head = "cpu", "inloop", "eager", "last"
    TL.check_gpu_env()
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    su = TL.Setup(a)
    eng = CtxEngine(su)
    ref_dir = TL.SCRATCH / (f"ref_nav_{a.ref_tag}" if DATASET == "navsim" else f"ref_{a.ref_tag}")
    load0 = TL.gpu_load_snapshot()
    res = []
    with torch.no_grad():
        eng.run(*su.samples[0])            # eager 예열
        if a.vision == "perimg_graph" or a.prefill == "graph":
            eng.capture()
        for n, (ent, r) in enumerate(su.samples):
            prev = ()
            if a.reuse_prev:   # 앞 틱 모사(teacher_latency.main 과 같음, 시간 밖)
                prev_paths = TL.S.cam_paths(r, True, su.fo)[:NCUR]
                _, pix_, _ = eng.prep(r)
                for i in range(NCUR):
                    eng.pix_buf[i * eng.npatch:(i + 1) * eng.npatch].copy_(pix_[i])
                if a.vision == "perimg_graph":
                    eng.vis_graphs["all"].replay()
                else:
                    eng.vision_images(list(range(NCUR)))
                eng.prev_cache = {p: eng.img_emb[i * eng.ntok_img:(i + 1) * eng.ntok_img].clone()
                                  for i, p in enumerate(prev_paths)}
                prev = prev_paths
            ctx, tm, ids = eng.run(ent, r, prev)
            e = dict(token=ent["token"], warmup=n < a.warmup, tm=tm)
            cp = TL.CACHE / "context" / f"{ent['token']}.pt"
            if cp.exists():
                e["ctx_vs_cache"] = cmp1(ctx, torch.load(cp, map_location="cpu", weights_only=True)["hidden"])
            rp = ref_dir / f"{ent['token']}.pt"
            if rp.exists():
                ref = torch.load(rp, weights_only=True)
                e["ctx_vs_ref"] = cmp1(ctx, ref["hs"]["context"])
                e["input_ids_vs_ref"] = bool(torch.equal(ids.cpu(), ref["input_ids"].cpu()))
            res.append(e)
            print(f"  [{n}/{len(su.samples)}] wall {tm['wall']:.1f} prep {tm['cpu_prep']:.1f} vis {tm['vision']:.2f} "
                  f"pre18 {tm['prefill18']:.2f} | ref bit {e.get('ctx_vs_ref', {}).get('bitwise')}", flush=True)
    load1 = TL.gpu_load_snapshot()
    meas = [e for e in res if not e["warmup"]]
    summ = {k: TL.stats([e["tm"][k] for e in meas]) for k in meas[0]["tm"]}
    eq = dict(ctx_vs_cache=agg([e["ctx_vs_cache"] for e in meas]) if all("ctx_vs_cache" in e for e in meas) else None)
    if all("ctx_vs_ref" in e for e in meas):
        eq["ctx_vs_ref"] = agg([e["ctx_vs_ref"] for e in meas])
        eq["input_ids_equal_ref"] = sum(e["input_ids_vs_ref"] for e in meas)
    doc = dict(tag=a.tag, args=vars(a), ckpt=su.ckpt, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
               L_ctx=eng.L_ctx, P=eng.P, capture_s=getattr(eng, "capture_s", None), gpu_load_before=load0,
               gpu_load_after=load1, summary=summ, equivalence=eq, samples=res)
    op = Path(a.out_dir) / f"teacher_latency_{a.tag}.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(op, "w"), indent=1, ensure_ascii=False)
    print(f"[summary] {a.tag} median/p90 ms: " + str({k: (v['median'], v['p90']) for k, v in summ.items()}), flush=True)
    print(f"[equiv] {json.dumps(eq)}", flush=True)
    print(f"-> {op}", flush=True)


if __name__ == "__main__":
    main()
