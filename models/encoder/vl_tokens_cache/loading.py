# 수정(2026-10-15, 2차): 아키텍처 그림의 "VL tokens (offline cache)" 상자. 표본별 Qwen2-VL-2B 18층 은닉 토큰
#   (L=1159, 1536, bf16) 캐시를 데이터 워커에서 읽는 파이프라인 단계. 원래 mmdet3d_plugin/datasets/pipelines/transform.py.
#   레지스트리 type 이름 그대로("LoadQwenCacheNusc"); models/loader/pipelines/__init__.py 가 다시 내보내 config 의
#   test/train pipeline 에서 쓰인다. 같은 캐시를 모델 쪽에서 읽는 대체 경로는 SparseDetector._maybe_load_qwen_cache.
#   캐시 생성기는 tools/qwen_cache.py(nuScenes) / qwen_cache_b2d.py / qwen_cache_navsim.py(encoder/vlm_qwen2vl/qwen_online.py 사용).
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES

__all__ = ["LoadQwenCacheNusc", "LoadTeacherCacheNusc", "TEACHER_CACHE_FOLDERS"]  # 수정(2026-10-16): 교사 캐시 7 폴더 · 토큰 유형 임베딩


@PIPELINES.register_module()
class LoadQwenCacheNusc(object):
    """수정(2026-10-10): 학습 속도 근본 수정 — Qwen 캐시 로드를 모델 forward(메인 프로세스, NAS 순차 읽기)에서
    데이터 워커로 옮긴다. 워커가 병렬·선읽기하므로 NFS 지연이 학습 루프를 막지 않는다.
    값은 기존 SparseDetector._maybe_load_qwen_cache 와 같다(같은 파일, 같은 hidden/attn_mask, 배치 내 0 패딩).
    파일이 없으면 즉시 실패한다(조용한 skip 금지)."""

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    def __call__(self, input_dict):
        import os
        import torch
        tok = input_dict.get("token")
        if tok is None:
            raise RuntimeError("LoadQwenCacheNusc: input_dict 에 token 이 없다.")
        p = os.path.join(self.cache_dir, f"{tok}.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Qwen 캐시 누락: {p}")
        d = torch.load(p, map_location="cpu")
        # 배치 collate 에서 길이(L) 축을 0 으로 패딩(기존 경로의 zeros 패딩과 동일)
        #   mmcv collate 는 pad_dims < ndim 을 요구 → 앞에 크기 1 축을 두고(1,L,H)/(1,L), 검출기에서 제거한다.
        input_dict["qwen_hidden"] = DC(d["hidden"][None], stack=True, pad_dims=2)
        input_dict["qwen_mask"] = DC(d["attn_mask"].to(torch.uint8)[None], stack=True, pad_dims=1)
        return input_dict


# 수정(2026-10-16): 교사 캐시 7 폴더 · 토큰 유형 임베딩 — LaQ-AD 교사 VLM(SFT+GRPO) 토큰 캐시
#   data/nuscenes/laq_cache/{context,detection,tracking,motion,map,plan,action}/<sample_token>.pt
#   (각 {'hidden': (L_f, 1536) bf16, 'attn_mask': (L_f,) uint8}, 폴더마다 L_f 고정·패딩 없음)를 읽어
#   고정 순서(TEACHER_CACHE_FOLDERS)로 이어 붙인 K/V 한 줄과 토큰별 유형 번호(0..6)를 돌려준다.
#   유형 번호는 "선택한 부분집합 안의 위치"가 아니라 전체 7 폴더 목록에서의 위치다 — 폴더 일부만 읽어도
#   같은 폴더는 같은 번호(= 같은 유형 임베딩 행)를 쓴다.
TEACHER_CACHE_FOLDERS = ("context", "detection", "tracking", "motion", "map", "plan", "action")
_META_LEN_KEYS = ("L", "length", "seq_len", "num_tokens")
TEACHER_CONSTRAINT = "answer_grammar_v1"   # 수정(2026-10-16): 설계 검토 반영 — tools/vla_teacher/answer_grammar.CONSTRAINT_NAME 과 같은 값


def read_teacher_index(cache_root):
    """수정(2026-10-16): 설계 검토 반영 — <cache_root>/_index/teacher_*.jsonl(extract_teacher_cache_v2.py 색인: 표본 한 줄,
    키 token·teacher·layout_ok·constrained_decoding …) → {token: (teacher, layout_ok, constrained_decoding)}.
    한 토큰이 두 색인에 있으면 ValueError. 색인 디렉터리가 없으면 None."""
    import glob
    import json
    import os
    files = sorted(glob.glob(os.path.join(cache_root, "_index", "teacher_*.jsonl")))
    if not files:
        return None
    out = {}
    for fp in files:
        with open(fp) as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                e = json.loads(ln)
                t = e["token"]
                if t in out and out[t][0] != e.get("teacher"):
                    raise ValueError(f"교사 캐시 색인: {t} 가 두 교사 색인에 있다({fp})")
                out[t] = (e.get("teacher"), e.get("layout_ok"), e.get("constrained_decoding"))
    return out


@PIPELINES.register_module()
class LoadTeacherCacheNusc(object):
    """수정(2026-10-16): 교사 캐시 7 폴더 · 토큰 유형 임베딩.

    출력(input_dict):
      qwen_hidden : DC((1, L_tot, H) bf16)   — 폴더 순서대로 이어 붙인 은닉 토큰
      qwen_mask   : DC((1, L_tot) uint8)     — 1=유효
      qwen_type   : DC((1, L_tot) int64)     — 토큰별 폴더 번호(TEACHER_CACHE_FOLDERS 기준 0..6)
    셋 다 pad_dims=None 으로 묶는다: 표본마다 길이가 같아야 하며, 다르면 collate(torch.stack)에서 즉시 실패한다
    (종전 LoadQwenCacheNusc 의 0 패딩 경로를 쓰지 않는다 — 고정 길이 캐시에서 길이가 다르면 데이터 오류이고,
    패딩은 flash-attn varlen 역방향(Blackwell 실패) 경로를 부른다).

    엄격 검사(위반 시 즉시 예외, 조용한 skip 없음): 파일 존재, 키 'hidden'/'attn_mask', hidden (L,H) bf16·H=hidden_dim·
    유한값, attn_mask (L,) uint8·값 {0,1}, L 이 폴더 기대 길이와 같음.
    기대 길이 L_f 출처(우선순위): expect_lens[folder] → <folder>/_meta.json 의 길이 키(L/length/seq_len/num_tokens) →
    워커가 처음 읽은 파일의 L(이후 모든 표본이 같아야 한다).
    meta_expect={folder: {key: value}}: 해당 폴더 _meta.json 에 key 가 value 로 기록돼 있어야 한다(없거나 다르면 실패).
    수정(2026-10-16): 설계 검토 반영: require_index=True(기본)면 <cache_root>/_index/teacher_*.jsonl 색인이 있어야 하고(없으면 초기화 실패),
    표본마다 색인 줄이 있고 layout_ok=True·constrained_decoding=answer_grammar_v1 이어야 한다. 색인에 없는 표본
    (추출 미완료)·layout_ok 가 아닌 표본(고정 위치 자르기 불일치)은 예외 — 건너뛰지 않는다.
    """

    def __init__(self, cache_root, folders=None, hidden_dim=1536, expect_lens=None, meta_expect=None,
                 check_finite=True, require_index=True):
        import os
        self.cache_root = cache_root
        folders = list(TEACHER_CACHE_FOLDERS if folders is None else folders)
        unknown = [f for f in folders if f not in TEACHER_CACHE_FOLDERS]
        if unknown or not folders:
            raise ValueError(f"LoadTeacherCacheNusc: 알 수 없는/빈 폴더 {unknown or folders} (허용 {TEACHER_CACHE_FOLDERS})")
        order = [TEACHER_CACHE_FOLDERS.index(f) for f in folders]
        if len(set(order)) != len(order) or order != sorted(order):
            # 조용히 재정렬하지 않는다 — 순서가 곧 K/V 배치이고 유형 번호와 결합된다.
            raise ValueError(f"LoadTeacherCacheNusc: folders 는 중복 없이 고정 순서 {TEACHER_CACHE_FOLDERS} 를 따라야 한다: {folders}")
        self.folders = folders
        self.type_ids = order
        self.hidden_dim = int(hidden_dim)
        self.check_finite = bool(check_finite)
        self.expect_lens = dict(expect_lens or {})
        meta_expect = dict(meta_expect or {})
        for f in self.expect_lens:
            if f not in self.folders:
                raise ValueError(f"LoadTeacherCacheNusc: expect_lens 의 폴더 {f} 가 folders 에 없다")
        for f in self.folders:
            mp = os.path.join(cache_root, f, "_meta.json")
            meta = None
            if os.path.exists(mp):
                import json
                with open(mp) as fh:
                    meta = json.load(fh)
                mlen = next((meta[k] for k in _META_LEN_KEYS if k in meta), None)
                if mlen is not None:
                    mlen = int(mlen)
                    if f in self.expect_lens and self.expect_lens[f] != mlen:
                        raise ValueError(f"LoadTeacherCacheNusc: {f} 길이 불일치 — config {self.expect_lens[f]} vs {mp} {mlen}")
                    self.expect_lens[f] = mlen
            for k, v in meta_expect.get(f, {}).items():
                if meta is None:
                    raise FileNotFoundError(f"LoadTeacherCacheNusc: meta_expect[{f}] 검사에 필요한 {mp} 가 없다")
                if k not in meta or meta[k] != v:
                    raise ValueError(f"LoadTeacherCacheNusc: {mp} 의 {k}={meta.get(k, '<없음>')!r}, 기대 {v!r}")
        unknown_meta = [f for f in meta_expect if f not in self.folders]
        if unknown_meta:
            raise ValueError(f"LoadTeacherCacheNusc: meta_expect 의 폴더 {unknown_meta} 가 folders 에 없다")
        self.index = None   # 수정(2026-10-16): 설계 검토 반영
        if require_index:
            self.index = read_teacher_index(cache_root)
            if self.index is None:
                raise FileNotFoundError(f"LoadTeacherCacheNusc: 색인 {os.path.join(cache_root, '_index')}/teacher_*.jsonl 이 없다 "
                                        f"(extract_teacher_cache_v2.py 산출 — 표본별 layout_ok 검사에 필요)")

    def __call__(self, input_dict):
        import os
        import torch
        tok = input_dict.get("token")
        if tok is None:
            raise RuntimeError("LoadTeacherCacheNusc: input_dict 에 token 이 없다.")
        if self.index is not None:   # 수정(2026-10-16): 설계 검토 반영 — 색인 플래그 검사(거부 = 예외)
            e = self.index.get(tok)
            if e is None:
                raise RuntimeError(f"교사 캐시 색인에 {tok} 가 없다(추출 미완료 표본) — {self.cache_root}/_index")
            if e[1] is not True:
                raise ValueError(f"교사 캐시 {tok}: 색인 layout_ok={e[1]} (layout_bad — 고정 위치 자르기 불일치 표본, 교사 {e[0]})")
            if e[2] != TEACHER_CONSTRAINT:
                raise ValueError(f"교사 캐시 {tok}: 색인 constrained_decoding={e[2]!r} != {TEACHER_CONSTRAINT!r} (제약 없이 생성된 캐시)")
        hs, ms, ts = [], [], []
        for f, tid in zip(self.folders, self.type_ids):
            p = os.path.join(self.cache_root, f, f"{tok}.pt")
            if not os.path.exists(p):
                raise FileNotFoundError(f"교사 캐시 누락: {p}")
            d = torch.load(p, map_location="cpu")
            if not isinstance(d, dict) or "hidden" not in d or "attn_mask" not in d:
                raise KeyError(f"교사 캐시 형식 오류(키 hidden/attn_mask 필요): {p} → {list(d) if isinstance(d, dict) else type(d)}")
            h, m = d["hidden"], d["attn_mask"]
            if not (torch.is_tensor(h) and h.dim() == 2 and h.shape[1] == self.hidden_dim and h.dtype == torch.bfloat16):
                raise ValueError(f"교사 캐시 hidden 형식 오류: {p} → {getattr(h, 'shape', None)} {getattr(h, 'dtype', None)}"
                                 f" (기대 (L, {self.hidden_dim}) bfloat16)")
            L = h.shape[0]
            if not (torch.is_tensor(m) and m.shape == (L,) and m.dtype == torch.uint8):
                raise ValueError(f"교사 캐시 attn_mask 형식 오류: {p} → {getattr(m, 'shape', None)} {getattr(m, 'dtype', None)}"
                                 f" (기대 ({L},) uint8)")
            if bool((m > 1).any()):
                raise ValueError(f"교사 캐시 attn_mask 값이 0/1 이 아니다: {p}")
            if L == 0:
                raise ValueError(f"교사 캐시 길이 0: {p}")
            exp = self.expect_lens.setdefault(f, L)  # 기대 길이가 없으면 첫 파일 길이로 고정
            if L != exp:
                raise ValueError(f"교사 캐시 길이 불일치: {p} L={L}, 폴더 {f} 기대 {exp} (폴더별 고정 길이)")
            if self.check_finite and not bool(torch.isfinite(h).all()):
                raise ValueError(f"교사 캐시 hidden 에 NaN/Inf: {p}")
            hs.append(h)
            ms.append(m)
            ts.append(torch.full((L,), tid, dtype=torch.long))
        input_dict["qwen_hidden"] = DC(torch.cat(hs, 0)[None], stack=True, pad_dims=None)
        input_dict["qwen_mask"] = DC(torch.cat(ms, 0)[None], stack=True, pad_dims=None)
        input_dict["qwen_type"] = DC(torch.cat(ts, 0)[None], stack=True, pad_dims=None)
        return input_dict
