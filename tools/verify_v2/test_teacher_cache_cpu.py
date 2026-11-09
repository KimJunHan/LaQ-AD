# 수정(2026-10-16): 교사 캐시 7 폴더 · 토큰 유형 임베딩 — CPU 전용 시험(GPU·학습 없음).
#   사용(저장소 루트): OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=. CUDA_VISIBLE_DEVICES="" python tools/verify_v2/test_teacher_cache_cpu.py [--no-full]
#   (1) LoadTeacherCacheNusc: 가짜 캐시(실제 길이 1832/168/40/240/68/13/8 — 수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 4), 무작위 bf16)로 형상·dtype·유형 번호·부분집합·고정 순서,
#       누락 파일/잘못된 dtype·길이·NaN·_meta.json 불일치에서 즉시 예외, 길이 다른 표본 collate 실패.
#   (2) config: v2_teacher 4 개 Config.fromfile, v2 대비 바뀐 키가 의도한 것뿐인지.
#   (3) QwenTokenAttn(새 config 로 빌드한 모델의 실제 모듈): 게이트 무작위 + 유형 임베딩 0 → 종전 모듈(같은 가중치, qwen_type 없음)과
#       비트 단위 동일 / 유형 임베딩 ≠0 → 달라지고 기울기 흐름 / 게이트 0 → 출력 = 쿼리 / 경로 지정(전 폴더) = 경로 지정 OFF /
#       경로 지정(부분집합) = 블록 마스크 밀집 어텐션 기준값.
#   (4) 전체 모델 forward(평가 모드, val 표본 1 개, 실제 영상 + 가짜 교사 캐시): 새 config 모델(유형 임베딩 0, 게이트 무작위)과
#       v2 config 모델(같은 가중치, 같은 토큰을 qwen_type 없이) 의 검출·지도·계획 출력이 같다.
#   (3b) 수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 1·2) — 게이트는 언어 항에만: 출력 = query + g⊙A (g=0 → query, 1·2단계 gate_sep 경로 포함),
#       사영 1 회 계산: forward(kv=project_kv(...)) == forward(사영 매번 계산), 6 계층 연속 호출의 출력·기울기 비교.
#   flash-attn 은 CUDA 전용이라 FlashAttention.forward 를 같은 수식의 torch 기준 구현(fp32 softmax)으로, deformable aggregation 은
#   grid_sample 기준 구현으로 바꿔 돌린다(두 모델에 같은 대체 → 비교 공정). 커널 자체 검증은 GPU 시험 몫.
import os
import sys
import json
import shutil
import tempfile
import argparse

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch
import torch.nn.functional as F

_orig_load = torch.load
def _load(*a, **kw):  # tools/train.py 와 같은 호환 shim
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _load

import projects.mmdet3d_plugin  # noqa: F401  레지스트리
from mmcv import Config
from mmcv.parallel import collate, DataContainer as DC

from projects.mmdet3d_plugin.models.encoder.vl_tokens_cache.loading import LoadTeacherCacheNusc, TEACHER_CACHE_FOLDERS
from projects.mmdet3d_plugin.models.decoder.temporal_self_attn import flash_attention as fa

# 수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 4) — 실제 추출기 산출 길이(data/nuscenes/laq_cache/<폴더>/_meta.json 의 L)로 맞춤.
#   종전 계획 길이 context 1860 → 실제 1832 (합 2397 → 2369).
PLANNED = dict(context=1832, detection=168, tracking=40, motion=240, map=68, plan=13, action=8)
CFG_NEW = {"s1": "configs/nuscenes/v2_teacher/stage1/train.py", "s2": "configs/nuscenes/v2_teacher/stage2/train.py"}
CFG_OLD = {"s1": "configs/nuscenes/stage1/train_v2.py", "s2": "configs/nuscenes/stage2/train_v2.py"}
FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}", flush=True)
    if not cond:
        FAIL.append(name)


def expect_raise(name, exc, fn):
    try:
        fn()
    except exc as e:
        check(name, True, f"{type(e).__name__}: {str(e)[:100]}")
        return
    except Exception as e:  # 다른 예외 종류
        check(name, False, f"기대 {exc.__name__}, 실제 {type(e).__name__}: {e}")
        return
    check(name, False, "예외가 나지 않았다")


# ---------------- CPU 대체 구현 ----------------
def flash_ref_forward(self, q, kv, causal=False, key_padding_mask=None):
    """FlashAttention.forward 와 같은 수식(softmax(QK^T/sqrt(d))V, key_padding_mask True=유효). fp32 계산."""
    assert not causal
    k, v = kv.unbind(2)                                   # (B,S,H,D)
    q_, k_, v_ = (x.transpose(1, 2).float() for x in (q, k, v))  # (B,H,·,D)
    mask = None
    if key_padding_mask is not None:
        mask = key_padding_mask[:, None, None, :].bool()
    p = self.dropout_p if self.training else 0.0
    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask, dropout_p=p, scale=self.softmax_scale)
    return out.transpose(1, 2).to(q.dtype), None


def daf_torch(mc_ms_feat, spatial_shape, scale_start_index, loc, weights):
    """tools/verify_v2/bank_equiv.py 의 grid_sample 기준 deformable aggregation(같은 수식)."""
    bs, A, P, Ncam, _ = loc.shape
    L, G = weights.shape[-2:]
    C = mc_ms_feat.shape[-1]
    out = mc_ms_feat.new_zeros(bs, A, C)
    ss = spatial_shape.tolist(); st = scale_start_index.tolist()
    for cam in range(Ncam):
        lw, lh = loc[:, :, :, cam, 0], loc[:, :, :, cam, 1]
        valid = ((lw > 0) & (lw < 1) & (lh > 0) & (lh < 1)).to(mc_ms_feat.dtype)
        grid = torch.stack([lw * 2 - 1, lh * 2 - 1], -1)
        for l in range(L):
            h, w = ss[cam][l]; s0 = st[cam][l]
            feat = mc_ms_feat[:, s0:s0 + h * w].reshape(bs, h, w, C).permute(0, 3, 1, 2)
            smp = F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            wgt = weights[:, :, :, cam, l, :] * valid[..., None]
            wgt = wgt.repeat_interleave(C // G, dim=-1).permute(0, 3, 1, 2)
            out = out + (smp * wgt).sum(-1).permute(0, 2, 1)
    return out


fa.FlashAttention.forward = flash_ref_forward
agg = sys.modules["projects.mmdet3d_plugin.models.decoder.deformable_aggregation.aggregation"]
agg.DAF = daf_torch


META_EXPECT = dict(prompt_mode="B", teacher_chain_verified=True, constrained_decoding="answer_grammar_v1")  # 수정(2026-10-16): 설계 검토 반영


def write_index(root, entries, teacher="A"):
    """수정(2026-10-16): 설계 검토 반영 — extract_teacher_cache_v2.py 색인 형식(필요한 키만): token·teacher·layout_ok·constrained_decoding.
    수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 4) — 실제 색인 줄(data/nuscenes/laq_cache/_index/teacher_A.jsonl)이 가진 판정 키도 같은 값으로 넣는다
    (part·grammar_ok·length_ok·parse_ok·canonical·parse_failure·n_new). 로더가 읽는 키는 token·teacher·layout_ok·constrained_decoding."""
    os.makedirs(os.path.join(root, "_index"), exist_ok=True)
    with open(os.path.join(root, "_index", f"teacher_{teacher}.jsonl"), "a") as fh:
        for e in entries:
            base = dict(teacher=teacher, part="val.B" if teacher == "A" else "val.A", n_new=547, grammar_ok=True, grammar_bad_pos=-1,
                        length_ok=True, parse_ok=True, parse_error=None, canonical=True, layout_ok=True, parse_failure=False,
                        constrained_decoding="answer_grammar_v1")
            fh.write(json.dumps(dict(base, **e)) + "\n")


def make_cache(root, tokens, lens=PLANNED, seed=0, index=True, meta=False):
    g = torch.Generator().manual_seed(seed)
    for f, L in lens.items():
        os.makedirs(os.path.join(root, f), exist_ok=True)
        for t in tokens:
            h = torch.randn(L, 1536, generator=g).to(torch.bfloat16)
            torch.save({"hidden": h, "attn_mask": torch.ones(L, dtype=torch.uint8)}, os.path.join(root, f, f"{t}.pt"))
        if meta:   # 수정(2026-10-16): 설계 검토 반영 — 추출기가 쓰는 키(config teacher_meta_expect 가 요구)
            # 수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 4) — 실제 _meta.json(extract_teacher_cache_v2.py)의 기본 키를 같은 이름·형식으로
            with open(os.path.join(root, f, "_meta.json"), "w") as fh:
                json.dump(dict(folder=f, L=L, hidden_dim=1536, dtype="bfloat16", layer=18, answer_tokens=547,
                               format="{'hidden': (L, D) bf16, 'attn_mask': (L,) uint8}",
                               source="teacher-generated answer (GT 아님)", ego_in_prompt=True, **META_EXPECT), fh)
    if index:   # 수정(2026-10-16): 설계 검토 반영
        write_index(root, [dict(token=t) for t in tokens])


# ---------------- (1) 로더 ----------------
def test_loader(tmp):
    print("(1) LoadTeacherCacheNusc", flush=True)
    root = os.path.join(tmp, "c1")
    make_cache(root, ["tokA", "tokB"])
    ld = LoadTeacherCacheNusc(root)
    out = ld({"token": "tokA"})
    h, m, t = out["qwen_hidden"].data, out["qwen_mask"].data, out["qwen_type"].data
    Ltot = sum(PLANNED.values())
    check(f"형상 hidden (1, {Ltot}, 1536) bf16", h.shape == (1, Ltot, 1536) and h.dtype == torch.bfloat16, str((tuple(h.shape), h.dtype)))
    check("형상 mask (1, L) uint8 · type (1, L) int64", m.shape == (1, Ltot) and m.dtype == torch.uint8
          and t.shape == (1, Ltot) and t.dtype == torch.long)
    exp_t = torch.cat([torch.full((L,), i) for i, L in enumerate(PLANNED.values())])
    check("유형 번호 = 폴더 순서 0..6 블록", torch.equal(t[0], exp_t))
    raw = torch.load(os.path.join(root, "map", "tokA.pt"))["hidden"]
    off = sum(list(PLANNED.values())[:4])
    check("이어 붙인 값 = 원본 파일(map 구간)", torch.equal(h[0, off:off + PLANNED["map"]], raw))
    check("DC stack=True, pad_dims=None", all(isinstance(out[k], DC) and out[k].stack and out[k].pad_dims is None
                                              for k in ("qwen_hidden", "qwen_mask", "qwen_type")))
    sub = LoadTeacherCacheNusc(root, folders=["context", "map", "action"])({"token": "tokA"})
    st = sub["qwen_type"].data[0]
    check("부분집합: 길이 context+map+action, 유형 번호는 전체 기준(0,4,6)",
          st.numel() == PLANNED["context"] + PLANNED["map"] + PLANNED["action"] and torch.unique(st).tolist() == [0, 4, 6])
    Q = ("qwen_hidden", "qwen_mask", "qwen_type")
    qk = lambda d: {k: d[k] for k in Q}  # collate 는 DC 키만(문자열 token 제외)
    b = collate([qk(ld({"token": "tokA"})), qk(ld({"token": "tokB"}))], samples_per_gpu=2)
    check("collate 2 표본 → (2,1,L,1536)/(2,1,L)", b["qwen_hidden"].data[0].shape == (2, 1, Ltot, 1536)
          and b["qwen_type"].data[0].shape == (2, 1, Ltot))

    expect_raise("순서 어긋난 folders 거부", ValueError, lambda: LoadTeacherCacheNusc(root, folders=["map", "context"]))
    expect_raise("알 수 없는 폴더 거부", ValueError, lambda: LoadTeacherCacheNusc(root, folders=["context", "lidar"]))
    expect_raise("누락 토큰(색인에도 없음) → RuntimeError", RuntimeError, lambda: ld({"token": "tokZ"}))  # 수정(2026-10-16): 설계 검토 반영 — 색인 검사가 먼저
    write_index(root, [dict(token="tokZ")])
    expect_raise("색인엔 있으나 파일 없음 → FileNotFoundError", FileNotFoundError, lambda: LoadTeacherCacheNusc(root)({"token": "tokZ"}))
    os.remove(os.path.join(root, "plan", "tokB.pt"))
    expect_raise("한 폴더만 누락(plan/tokB) → FileNotFoundError", FileNotFoundError,
                 lambda: LoadTeacherCacheNusc(root)({"token": "tokB"}))
    torch.save({"hidden": torch.randn(13, 1536), "attn_mask": torch.ones(13, dtype=torch.uint8)},
               os.path.join(root, "plan", "tokB.pt"))
    expect_raise("fp32 hidden → ValueError", ValueError, lambda: LoadTeacherCacheNusc(root)({"token": "tokB"}))
    torch.save({"hidden": torch.randn(14, 1536).bfloat16(), "attn_mask": torch.ones(14, dtype=torch.uint8)},
               os.path.join(root, "plan", "tokB.pt"))
    ld2 = LoadTeacherCacheNusc(root); ld2({"token": "tokA"})
    expect_raise("폴더 길이 변동(13→14) → ValueError", ValueError, lambda: ld2({"token": "tokB"}))
    torch.save({"hidden": torch.randn(13, 1536).bfloat16(), "attn_mask": torch.ones(13, dtype=torch.int64)},
               os.path.join(root, "plan", "tokB.pt"))
    expect_raise("int64 attn_mask → ValueError", ValueError, lambda: LoadTeacherCacheNusc(root)({"token": "tokB"}))
    hn = torch.randn(13, 1536).bfloat16(); hn[3, 7] = float("nan")
    torch.save({"hidden": hn, "attn_mask": torch.ones(13, dtype=torch.uint8)}, os.path.join(root, "plan", "tokB.pt"))
    expect_raise("NaN hidden → ValueError", ValueError, lambda: LoadTeacherCacheNusc(root)({"token": "tokB"}))
    torch.save({"hid": hn}, os.path.join(root, "plan", "tokB.pt"))
    expect_raise("키 누락 → KeyError", KeyError, lambda: LoadTeacherCacheNusc(root)({"token": "tokB"}))
    with open(os.path.join(root, "plan", "_meta.json"), "w") as fh:
        json.dump({"L": 12, "prompt": "cache_ego"}, fh)
    expect_raise("_meta.json L=12 vs 파일 13 → ValueError", ValueError, lambda: LoadTeacherCacheNusc(root)({"token": "tokA"}))
    expect_raise("expect_lens 와 _meta.json 충돌 → ValueError", ValueError,
                 lambda: LoadTeacherCacheNusc(root, expect_lens={"plan": 13}))
    with open(os.path.join(root, "plan", "_meta.json"), "w") as fh:
        json.dump({"L": 13, "prompt": "cache_ego"}, fh)
    ok = LoadTeacherCacheNusc(root, meta_expect={"plan": {"prompt": "cache_ego"}})({"token": "tokA"})
    check("_meta.json L=13·prompt 일치 → 통과", ok["qwen_hidden"].data.shape[1] == Ltot)
    expect_raise("meta_expect 불일치(B안 검사) → ValueError", ValueError,
                 lambda: LoadTeacherCacheNusc(root, meta_expect={"plan": {"prompt": "cache"}}))
    expect_raise("meta_expect 인데 _meta.json 없음 → FileNotFoundError", FileNotFoundError,
                 lambda: LoadTeacherCacheNusc(root, meta_expect={"context": {"prompt": "cache_ego"}}))
    # 길이 다른 두 표본: 0 패딩 없이 collate 에서 실패해야 한다
    r2 = os.path.join(tmp, "c2"); make_cache(r2, ["a"], lens=dict(context=10)); make_cache(r2 + "x", ["b"], lens=dict(context=11))
    s_a = LoadTeacherCacheNusc(r2, folders=["context"])({"token": "a"})
    s_b = LoadTeacherCacheNusc(r2 + "x", folders=["context"])({"token": "b"})
    expect_raise("길이 다른 표본 collate → 실패(0 패딩 없음)", RuntimeError, lambda: collate([qk(s_a), qk(s_b)], samples_per_gpu=2))

    # 수정(2026-10-16): 설계 검토 반영 — 색인 플래그(layout_ok·constrained_decoding) 검사: 거부 = 예외
    r3 = os.path.join(tmp, "c3")
    make_cache(r3, ["good", "bad", "unc", "noidx"], lens=dict(context=4), index=False)
    expect_raise("색인 없음 → 초기화 FileNotFoundError", FileNotFoundError, lambda: LoadTeacherCacheNusc(r3, folders=["context"]))
    nr = LoadTeacherCacheNusc(r3, folders=["context"], require_index=False)({"token": "noidx"})
    check("require_index=False 면 색인 없이 읽음(시험 전용)", nr["qwen_hidden"].data.shape == (1, 4, 1536))
    write_index(r3, [dict(token="good"), dict(token="bad", layout_ok=False), dict(token="unc", constrained_decoding=None)])
    l3 = LoadTeacherCacheNusc(r3, folders=["context"])
    check("색인 layout_ok=True·제약 생성 표본 → 통과", l3({"token": "good"})["qwen_hidden"].data.shape == (1, 4, 1536))
    expect_raise("색인 layout_ok=False(layout_bad) → ValueError(거부)", ValueError, lambda: l3({"token": "bad"}))
    expect_raise("색인 constrained_decoding 없음(비제약 캐시) → ValueError", ValueError, lambda: l3({"token": "unc"}))
    expect_raise("파일은 있으나 색인에 없음(추출 미완료) → RuntimeError", RuntimeError, lambda: l3({"token": "noidx"}))
    write_index(r3, [dict(token="good")], teacher="B")
    expect_raise("한 표본이 두 교사 색인에 → ValueError", ValueError, lambda: LoadTeacherCacheNusc(r3, folders=["context"]))


# ---------------- (2) config ----------------
def test_configs():
    print("(2) config", flush=True)
    for p in ["configs/nuscenes/v2_teacher/stage1/train.py", "configs/nuscenes/v2_teacher/stage1/test.py",
              "configs/nuscenes/v2_teacher/stage2/train.py", "configs/nuscenes/v2_teacher/stage2/test.py"]:
        c = Config.fromfile(p)
        qa = c.model.head.onedecoder_head.qwen_token_attn
        pipes = [c.data.train.pipeline, c.data.val.pipeline, c.data.test.pipeline]
        ok = all(any(s["type"] == "LoadTeacherCacheNusc" for s in pl) and not any(s["type"] == "LoadQwenCacheNusc" for s in pl)
                 and "qwen_type" in [s for s in pl if s["type"] == "Collect"][0]["keys"] for pl in pipes)
        check(f"{p}: 로드·유형 임베딩 ON·경로 OFF·세 파이프라인 교체", ok and qa.num_token_types == 7 and qa.route_segs is None
              and c.model.qwen_cache_dir == "data/nuscenes/laq_cache")
        lds = [s for pl in pipes for s in pl if s["type"] == "LoadTeacherCacheNusc"]   # 수정(2026-10-16): 설계 검토 반영
        check(f"{p}: meta_expect = 7 폴더 모두 B안·교사 연결·제약 생성, 색인 검사 ON",
              len(lds) == 3 and all(s["meta_expect"] == {f: META_EXPECT for f in TEACHER_CACHE_FOLDERS}
                                    and s.get("require_index", True) is True for s in lds))
    for s in ("s1", "s2"):
        a, b = Config.fromfile(CFG_NEW[s]), Config.fromfile(CFG_OLD[s])
        for split in ("train", "test"):
            pa, pb = a.data[split].pipeline, b.data[split].pipeline
            same = len(pa) == len(pb)
            for x, y in zip(pa, pb):
                if x["type"] == "LoadTeacherCacheNusc":
                    same &= y["type"] == "LoadQwenCacheNusc"
                elif x["type"] == "Collect":
                    same &= [k for k in x["keys"] if k != "qwen_type"] == y["keys"] and x["meta_keys"] == y["meta_keys"]
                else:
                    same &= x == y
            check(f"{s} {split} 파이프라인: 로더·qwen_type 외 v2 와 동일", same)


# ---------------- (3)(4) 모델 ----------------
def build(cfg_path):
    from mmdet.models import build_detector
    cfg = Config.fromfile(cfg_path)
    torch.manual_seed(0)
    m = build_detector(cfg.model)
    m.init_weights()
    return cfg, m


def dense_masked_ref(qta, query, kv, qwen_type, allowed_per_seg):
    """블록 마스크 기준값: 전체 K/V 에 대해 쿼리 행별 허용 폴더 밖 키를 -inf 로 가린 밀집 어텐션(MultiheadFlashAttention 수식 재현)."""
    mha = qta.attn.attn
    E, H = mha.embed_dim, mha.num_heads
    wq, wk, wv = mha.in_proj_weight.chunk(3); bq, bk, bv = mha.in_proj_bias.chunk(3)
    q = F.linear(query, wq, bq); k = F.linear(kv, wk, bk); v = F.linear(kv, wv, bv)
    B, N, _ = q.shape; S = k.shape[1]
    q = q.view(B, N, H, -1).transpose(1, 2); k = k.view(B, S, H, -1).transpose(1, 2); v = v.view(B, S, H, -1).transpose(1, 2)
    allow = torch.zeros(N, S, dtype=torch.bool)
    prev = 0
    for si, end in enumerate(list(qta.seg_bounds) + [N]):
        end = min(end, N)
        if end > prev:
            allow[prev:end] = torch.isin(qwen_type[0], torch.tensor(allowed_per_seg[si]))[None]
            prev = end
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=allow[None, None])
    o = mha.out_proj(o.transpose(1, 2).reshape(B, N, E))
    return o   # 수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 1) — 교차 어텐션 출력 A 만(잔차 없음). 게이트 잔차는 호출부가 query + g⊙A 로


def test_module(model_new, cfg_old_path):
    print("(3) QwenTokenAttn", flush=True)
    from mmcv.utils import build_from_cfg
    from mmcv.cnn.bricks.registry import ATTENTION
    qta = model_new.head.onedecoder_head.qwen_token_attn
    check("token_type_embed (7,256) 이고 init_weights 뒤에도 전부 0",
          tuple(qta.token_type_embed.shape) == (7, 256) and bool((qta.token_type_embed == 0).all()))
    check("게이트 초기값 0", bool((qta.gate == 0).all()))
    old_cfg = Config.fromfile(cfg_old_path).model.head.onedecoder_head.qwen_token_attn
    old = build_from_cfg(dict(old_cfg), ATTENTION)
    sd = {k: v for k, v in qta.state_dict().items() if k != "token_type_embed"}
    old.load_state_dict(sd, strict=True)
    qta.eval(); old.eval()

    torch.manual_seed(1)
    B, N = 1, 1049
    Ls = list(PLANNED.values()); Ltot = sum(Ls)
    query = torch.randn(B, N, 256)
    hidden = torch.randn(B, Ltot, 1536).to(torch.bfloat16)
    mask = torch.ones(B, Ltot, dtype=torch.uint8)
    qtype = torch.cat([torch.full((L,), i) for i, L in enumerate(Ls)])[None].expand(B, -1).clone()
    with torch.no_grad():
        out0 = qta(query, hidden, mask, layer_idx=2, qwen_type=qtype)
        check("게이트 0 → 출력 == 쿼리(비트 단위)", torch.equal(out0, query))
        qta.gate.normal_(0, 0.5); old.gate.copy_(qta.gate)
        a = qta(query, hidden, mask, layer_idx=2, qwen_type=qtype)
        b = old(query, hidden, mask, layer_idx=2)
        check("게이트 무작위 + 유형 임베딩 0 → 종전 모듈과 비트 단위 동일", torch.equal(a, b),
              f"max|Δ|={float((a - b).abs().max()):.3e}, max|out-q|={float((a - query).abs().max()):.3e}")
        qta.token_type_embed.normal_(0, 0.5)
        c = qta(query, hidden, mask, layer_idx=2, qwen_type=qtype)
        check("유형 임베딩 ≠0 → 출력 달라짐", float((c - b).abs().max()) > 1e-3, f"max|Δ|={float((c - b).abs().max()):.3e}")
    # 기울기 흐름(유형 임베딩 ≠0 · 게이트 ≠0 상태)
    qta.zero_grad(set_to_none=True)
    qta(query, hidden, mask, layer_idx=2, qwen_type=qtype).pow(2).mean().backward()
    gt = qta.token_type_embed.grad
    check("token_type_embed 기울기 7 행 모두 ≠0", gt is not None and bool((gt.abs().sum(1) > 0).all()))
    expect_raise("유형 임베딩 ON 인데 qwen_type 없음 → RuntimeError", RuntimeError,
                 lambda: qta(query, hidden, mask, layer_idx=2))
    expect_raise("qwen_type 범위 밖 → ValueError", ValueError,
                 lambda: qta(query, hidden, mask, layer_idx=2, qwen_type=qtype + 1))

    # 경로 지정
    names = list(TEACHER_CACHE_FOLDERS)
    routes = [["context", "detection", "tracking", "motion"], ["context", "map"], ["context", "plan", "action", "motion", "map"]]
    with torch.no_grad():
        full = qta(query, hidden, mask, layer_idx=3, qwen_type=qtype)
        qta.route_type_ids = [list(range(7))] * 3
        r_all = qta(query, hidden, mask, layer_idx=3, qwen_type=qtype)
        check("경로 지정(전 구간 전 폴더) == 경로 OFF", torch.allclose(r_all, full, atol=1e-6, rtol=0),
              f"max|Δ|={float((r_all - full).abs().max()):.3e}")
        ids = [sorted(names.index(n) for n in r) for r in routes]
        qta.route_type_ids = ids
        r = qta(query, hidden, mask, layer_idx=3, qwen_type=qtype)
        # 기준값: 같은 kv(사영 + 유형 임베딩) · 블록 마스크 밀집 어텐션 → 게이트
        kv = qta.qwen_token_proj(hidden.float()) + qta.token_type_embed[qtype]
        ref_att = dense_masked_ref(qta, query, kv, qtype, ids)
        g = torch.zeros(N, 256); prev = 0
        for si, end in enumerate(list(qta.seg_bounds) + [N]):
            g[prev:end] = qta.gate[3, si]; prev = end
        ref = query + g * ref_att
        check("경로 지정(부분집합) == 블록 마스크 밀집 어텐션 기준값", torch.allclose(r, ref, atol=1e-5, rtol=0),
              f"max|Δ|={float((r - ref).abs().max()):.3e}")
        check("경로 지정(부분집합) ≠ 경로 OFF(실제로 가림)", float((r - full).abs().max()) > 1e-4)
        qta.route_type_ids = None
    # config 로 route_segs 를 켠 모듈 생성(이름 → 번호)
    rc = dict(old_cfg); rc.update(num_token_types=7, token_type_names=names, route_segs=routes)
    rmod = build_from_cfg(rc, ATTENTION)
    check("route_segs config → 번호 [[0,1,2,3],[0,4],[0,3,4,5,6]]", rmod.route_type_ids == [[0, 1, 2, 3], [0, 4], [0, 3, 4, 5, 6]])
    bad = dict(rc); bad["route_segs"] = routes[:2]
    expect_raise("route_segs 구간 수 불일치 → ValueError", ValueError, lambda: build_from_cfg(bad, ATTENTION))
    # state_dict 하위 호환: 종전 config 모듈에는 새 키가 없다
    check("종전 config 모듈 state_dict 에 token_type_embed 없음", "token_type_embed" not in old.state_dict())


# ---------------- (3b) 게이트 = 언어 항에만 · 사영 1 회 ----------------
def _gate_rows(qta, li, N):
    """forward 와 같은 규칙의 쿼리 행별 게이트 (N, d)."""
    g = torch.zeros(N, qta.embed_dims); prev = 0
    for si, end in enumerate(list(qta.seg_bounds) + [N]):
        end = min(int(end), N)
        if end > prev:
            g[prev:end] = (qta.gate_sep[li][qta.sep_gate_segs.index(si)] if si in qta.sep_gate_segs else qta.gate[li, si])
        prev = end
    return g


def test_gate_and_kv_cache():
    """수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 1·2)."""
    print("(3b) 게이트는 언어 항에만 · 교사 토큰 사영 1 회", flush=True)
    from mmcv.utils import build_from_cfg
    from mmcv.cnn.bricks.registry import ATTENTION
    torch.manual_seed(3)
    B, N = 2, 1049
    Ls = list(PLANNED.values()); Ltot = sum(Ls)
    query = torch.randn(B, N, 256)
    hidden = torch.randn(B, Ltot, 1536).to(torch.bfloat16)
    mask = torch.ones(B, Ltot, dtype=torch.uint8)
    qtype = torch.cat([torch.full((L,), i) for i, L in enumerate(Ls)])[None].expand(B, -1).clone()
    for st in ("s1", "s2"):
        cfg = Config.fromfile(CFG_NEW[st]).model.head.onedecoder_head.qwen_token_attn
        torch.manual_seed(0)
        qta = build_from_cfg(dict(cfg), ATTENTION)
        qta.eval()   # 어텐션 dropout 끔(결정적 비교)
        sep = list(qta.sep_gate_segs)
        with torch.no_grad():
            qta.token_type_embed.normal_(0, 0.5)
            out0 = qta(query, hidden, mask, layer_idx=1, qwen_type=qtype)
            check(f"{st}: 게이트 0 → 출력 == 쿼리(비트 단위, 기반 모델과 같은 시작)", torch.equal(out0, query))
            qta.gate.normal_(0, 0.5)
            if sep:
                qta.gate_sep.normal_(0, 0.5)
            kv, qt_l = qta.project_kv(hidden, qtype, query.dtype)
            A = qta._attn_only(query, kv)
            for li in (0, 4):
                out = qta(query, hidden, mask, layer_idx=li, qwen_type=qtype)
                g = _gate_rows(qta, li, N)
                d_new = float((out - (query + g * A)).abs().max())
                d_res = float(((out - query) - g * A).abs().max())
                d_old = float((out - (query + g * (query + A))).abs().max())
                check(f"{st} 계층 {li}: 출력 == query + g⊙A (비트 단위), 출력 − query = g⊙A, 종전 식 (1+g)⊙query + g⊙A 와는 다름",
                      torch.equal(out, query + g * A) and d_res < 1e-6 and d_old > 1e-2,
                      f"|Δ새식|={d_new:.1e}, |Δ잔차|={d_res:.1e}, |Δ종전식|={d_old:.2e}")
                out_kv = qta(query, hidden, mask, layer_idx=li, qwen_type=qt_l, kv=kv)
                check(f"{st} 계층 {li}: forward(kv=project_kv(...)) == 사영 매번 계산(비트 단위)", torch.equal(out_kv, out))
            if sep:   # 2단계: plan/ego 구간(gate_sep)만 켜고 det/map 게이트 0
                qta.gate.zero_()
                out = qta(query, hidden, mask, layer_idx=2, qwen_type=qtype)
                me = qta.seg_bounds[-1]
                check(f"{st} gate_sep 경로(sep_gate_segs={sep}): det/map 행 == 쿼리, plan/ego 행 == query + gate_sep⊙A",
                      torch.equal(out[:, :me], query[:, :me]) and
                      torch.equal(out[:, me:], query[:, me:] + qta.gate_sep[2][0] * A[:, me:]))
                qta.gate.normal_(0, 0.5)
        expect_raise(f"{st}: kv dtype 이 query 와 다르면 → ValueError", ValueError,
                     lambda: qta(query, hidden, mask, layer_idx=0, qwen_type=qt_l, kv=kv.double()))
        # 기울기: g ≠ 0 에서 사영·유형 임베딩·어텐션·게이트 모두 0 아닌 기울기
        qta.zero_grad(set_to_none=True)
        (qta(query, hidden, mask, layer_idx=3, qwen_type=qtype).pow(2).mean()).backward()
        names = ["qwen_token_proj.0.weight", "qwen_token_proj.2.weight", "qwen_token_proj.3.weight", "token_type_embed",
                 "attn.attn.in_proj_weight", "attn.attn.out_proj.weight", "gate"] + (["gate_sep"] if sep else [])
        gn = {n: float(p.grad.abs().sum()) if p.grad is not None else 0.0 for n, p in qta.named_parameters() if n in names}
        check(f"{st}: g≠0 → 사영·유형 임베딩·어텐션·게이트 기울기 모두 ≠0", all(v > 0 for v in gn.values()) and len(gn) == len(names),
              ", ".join(f"{k.split('.')[-2] if k.count('.') else k}:{v:.1e}" for k, v in gn.items()))
        # 6 계층 연속(디코더처럼 앞 계층 출력이 다음 계층 쿼리): 사영 6 회(종전) vs 1 회(새) — 출력·기울기
        res = {}
        for mode in ("each", "once"):
            qta.zero_grad(set_to_none=True)
            hq = hidden.clone().float().requires_grad_(True)
            x = query.clone().requires_grad_(True)
            y = x
            kvc = qta.project_kv(hq, qtype, x.dtype) if mode == "once" else None
            for li in range(6):
                y = qta(y, hq, mask, layer_idx=li, qwen_type=qtype if kvc is None else kvc[1], **({} if kvc is None else {"kv": kvc[0]}))
            y.pow(2).mean().backward()
            res[mode] = (y.detach(), x.grad.clone(), hq.grad.clone(), {n: p.grad.clone() for n, p in qta.named_parameters() if p.grad is not None})
        ya, xa, ha, pa = res["each"]; yb, xb, hb, pb = res["once"]
        dp = {n: float((pa[n] - pb[n]).abs().max() / pa[n].abs().max().clamp_min(1e-30)) for n in pa}
        check(f"{st} 6 계층 연속: 출력 비트 동일, 쿼리 기울기 비트 동일, 사영·입력 기울기는 합산 순서 차(상대 ≤ 1e-5)",
              torch.equal(ya, yb) and torch.equal(xa, xb) and set(pa) == set(pb) and max(dp.values()) <= 1e-5
              and float((ha - hb).abs().max() / ha.abs().max()) <= 1e-5,
              f"출력 max|Δ|={float((ya - yb).abs().max()):.1e}, 입력 기울기 상대 {float((ha - hb).abs().max() / ha.abs().max()):.1e}, "
              f"파라미터 기울기 상대 최대 {max(dp.values()):.1e} ({max(dp, key=dp.get)})")


def test_full_forward(tmp):
    print("(4) 전체 모델 forward(CPU, 평가 모드, val 표본 1)", flush=True)
    from mmdet.datasets import build_dataset
    cfg_new, m_new = build(CFG_NEW["s2"])
    cfg_old, m_old = build(CFG_OLD["s2"])
    sd = {k: v for k, v in m_new.state_dict().items() if "token_type_embed" not in k}
    missing, unexpected = m_old.load_state_dict(sd, strict=False)
    check("v2 모델에 새 모델 가중치 적재(누락·초과 키 없음)", not missing and not unexpected, f"{missing[:3]} {unexpected[:3]}")
    torch.manual_seed(5)
    with torch.no_grad():  # 융합이 실제로 출력에 영향을 주도록 게이트를 무작위로(두 모델 같은 값)
        g = torch.randn_like(m_new.head.onedecoder_head.qwen_token_attn.gate) * 0.5
        for mm in (m_new, m_old):
            mm.head.onedecoder_head.qwen_token_attn.gate.copy_(g)
            if hasattr(mm.head.onedecoder_head.qwen_token_attn, "gate_sep"):
                gs = g[:, mm.head.onedecoder_head.qwen_token_attn.sep_gate_segs]
                mm.head.onedecoder_head.qwen_token_attn.gate_sep.copy_(gs)
    m_new.eval(); m_old.eval()

    root = os.path.join(tmp, "laq_cache")
    ds_cfg = cfg_new.data.test
    ds_cfg.ann_file = "data/infos/nuscenes_infos_val.pkl"
    for s in ds_cfg.pipeline:
        if s["type"] == "LoadTeacherCacheNusc":
            s["cache_root"] = root
    # 수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 4) — 로더는 생성 시점(데이터셋 build)에 _meta.json·색인을 검사하므로 캐시를 먼저 만든다.
    #   표본 token 은 데이터셋과 같은 방법(pkl 의 infos 순서 그대로)으로 미리 읽는다.
    import pickle
    with open(ds_cfg.ann_file, "rb") as fh:
        _d = pickle.load(fh)
    tok = (_d["infos"] if isinstance(_d, dict) else _d)[0]["token"]
    del _d
    make_cache(root, [tok], seed=7, meta=True)   # 수정(2026-10-16): 설계 검토 반영 — config 의 meta_expect 를 만족하는 _meta.json
    ds = build_dataset(ds_cfg)
    assert ds.data_infos[0]["token"] == tok
    sample = ds[0]
    data = collate([sample], samples_per_gpu=1)
    def unwrap(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, DC):
                out[k] = v.data[0]
            elif isinstance(v, list) and v and isinstance(v[0], DC):
                out[k] = [x.data[0] for x in v]
            else:
                out[k] = v
        return out
    d = unwrap(data)
    Ltot = sum(PLANNED.values())
    check(f"배치 qwen_hidden (1,1,{Ltot},1536) / qwen_type (1,1,{Ltot})",
          tuple(d["qwen_hidden"].shape) == (1, 1, Ltot, 1536) and tuple(d["qwen_type"].shape) == (1, 1, Ltot))
    img = d.pop("img")
    if isinstance(img, list):
        img = img[0]
    d_old = {k: v for k, v in d.items() if k != "qwen_type"}

    calls = {"n": 0, "types": 0}
    qta_new = m_new.head.onedecoder_head.qwen_token_attn
    orig_fwd = qta_new.forward
    def spy(*a, **kw):
        calls["n"] += 1
        calls["types"] += int(kw.get("qwen_type") is not None)
        return orig_fwd(*a, **kw)
    qta_new.forward = spy
    with torch.no_grad():
        out_new = m_new.simple_test(img.clone(), **{k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()})
        out_old = m_old.simple_test(img.clone(), **{k: (v.clone() if torch.is_tensor(v) else v) for k, v in d_old.items()})
    check("디코더 6 계층 모두 qwen_type 을 받아 융합 호출", calls["n"] == 6 and calls["types"] == 6, str(calls))
    ra, rb = out_new[0]["img_bbox"], out_old[0]["img_bbox"]
    diffs = {}
    for k in ra:
        if torch.is_tensor(ra[k]) and ra[k].is_floating_point():
            diffs[k] = float((ra[k] - rb[k]).abs().max()) if ra[k].numel() else 0.0
        elif torch.is_tensor(ra[k]):
            diffs[k] = 0.0 if torch.equal(ra[k], rb[k]) else float("inf")
    check("새 config 모델(유형 임베딩 0) 출력 == v2 모델(같은 토큰, qwen_type 없음)", all(v == 0.0 for v in diffs.values()),
          ", ".join(f"{k}:{v:.1e}" for k, v in list(diffs.items())[:12]))
    # 융합이 실제로 출력에 기여하는지(게이트 무작위) — 유형 임베딩을 바꾸면 출력이 바뀌어야 한다
    with torch.no_grad():
        qta_new.token_type_embed.normal_(0, 1.0)
        m_new.head.onedecoder_head.det_instance_bank.reset()
        for n in ("map", "ego", "plan"):
            b = getattr(m_new.head.onedecoder_head, f"{n}_instance_bank", None)
            if b is not None:
                b.reset()
        out_t = m_new.simple_test(img.clone(), **{k: (v.clone() if torch.is_tensor(v) else v) for k, v in d.items()})
    k0 = [k for k in ra if torch.is_tensor(ra[k]) and ra[k].is_floating_point() and ra[k].numel()][0]
    dd = float((out_t[0]["img_bbox"][k0] - ra[k0]).abs().max())
    check(f"유형 임베딩 ≠0 → 전체 출력({k0}) 달라짐", dd > 0, f"max|Δ|={dd:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-full", action="store_true", help="(4) 전체 모델 forward 생략")
    a = ap.parse_args()
    assert not torch.cuda.is_available(), "CPU 전용 시험: CUDA_VISIBLE_DEVICES=\"\""
    tmp = tempfile.mkdtemp(prefix="teacher_cache_cpu_")
    try:
        test_loader(tmp)
        test_configs()
        _, m = build(CFG_NEW["s1"])
        test_module(m, CFG_OLD["s1"])
        del m
        test_gate_and_kv_cache()   # 수정(2026-10-19): 사용자 결정(무결점 검사 1차 지적 1·2)
        if not a.no_full:
            test_full_forward(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("결과:", "전부 통과" if not FAIL else f"실패 {len(FAIL)}: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
