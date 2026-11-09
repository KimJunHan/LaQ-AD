# 수정(2026-10-16): 교사 QA 전체 학습 단계 — 공개 nuScenes QA(NuScenes-QA, DriveLM-nuScenes)로 Qwen2-VL-2B 전체 미세조정.
#   make_qa_data.py / check_qa_split.py / train_qa_full.py / smoke_qa_full_cpu.py 가 공유한다.
#   train_sft.py / sft_common.py 는 고치지 않고 import 만 한다(같은 processor·M-RoPE·정확한 fp32 어텐션·GPU 규칙).
#
#   표본 = 한 키프레임(한 데이터셋)의 QA 전부를 한 다중 턴 대화로 묶은 것:
#     system(Qwen 기본 문장) → user(현재 6 캠 영상 + Q1) → assistant(A1) → user(Q2) → assistant(A2) …
#   손실은 각 assistant 답 토큰 + 그 턴의 <|im_end|> 에만(질문·영상·system·턴 머리말은 -100).
#   너무 긴 프레임은 잘라내지 않고 QA 순서를 유지한 채 여러 대화(chunk)로 나눈다(모든 QA 를 그대로 사용).
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import sft_common as S  # noqa: E402

TAG = "# 수정(2026-10-16): 교사 QA 전체 학습 단계"
QA_DIR = "data/vla_teacher/qa"                    # 저장소 기준 산출 폴더
QA_RAW = "/home1/irteam/data-vol1/src/HiP-AD/data/nuscenes_qa"  # 내려받은 원본 주석(신뢰하지 않는 데이터 — python -I 로만 읽음)
CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]  # 교사 SFT 와 같은 순서
IMAGE_PAD_ID = 151655
DATASETS = ("nuscenes_qa", "drivelm_nus")


def qa_file(fold, part, qa_dir=QA_DIR):
    d = Path(qa_dir) if os.path.isabs(qa_dir) else S.REPO / qa_dir
    return d / f"fold{fold}_{part}.jsonl"


def load_qa(fold, part, qa_dir=QA_DIR):
    p = qa_file(fold, part, qa_dir)
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음 — 먼저 tools/vla_teacher/make_qa_data.py 실행")
    out = []
    with open(p) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out, str(p)


# ---------------------------------------------------------------- 대화 조립(문자열 조각 단위 — 답 위치를 정확히 알기 위해)
def conversation_pieces(rec, n_images=S.NUM_CAMS, system=S.QWEN_DEFAULT_SYSTEM):
    """반환 [(text, supervised: bool)]. 이어 붙이면 processor.apply_chat_template(전체 대화, add_generation_prompt=False)
    와 글자 단위로 같다(smoke 검사). 영상 자리표시 '<|vision_start|><|image_pad|><|vision_end|>' 는 processor 가 펼친다."""
    img = "<|vision_start|><|image_pad|><|vision_end|>" * n_images
    pcs = [(f"<|im_start|>system\n{system}<|im_end|>\n", False)]
    for i, t in enumerate(rec["turns"]):
        head = "<|im_start|>user\n" + (img if i == 0 else "") + t["q"] + "<|im_end|>\n<|im_start|>assistant\n"
        pcs.append((head, False))
        pcs.append((t["a"] + "<|im_end|>", True))
        pcs.append(("\n", False))
    return pcs


def chat_messages(rec, n_images=S.NUM_CAMS, system=S.QWEN_DEFAULT_SYSTEM):
    msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}]
    for i, t in enumerate(rec["turns"]):
        c = ([{"type": "image"} for _ in range(n_images)] if i == 0 else []) + [{"type": "text", "text": t["q"]}]
        msgs.append({"role": "user", "content": c})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": t["a"]}]})
    return msgs


def tokenize_pieces(tok, pcs, image_tokens_per_cam=S.TOKENS_PER_CAM):
    """영상 없이(텍스트만) 조각별 토큰화 — 길이 통계·마스크 검사용. image_pad 는 캠당 image_tokens_per_cam 개로 펼친다.
    반환 (ids, labels). labels 는 감독 조각만 id, 나머지 -100."""
    ids, lab = [], []
    for text, sup in pcs:
        text = text.replace("<|image_pad|>", "<|image_pad|>" * image_tokens_per_cam)
        t = tok(text, add_special_tokens=False)["input_ids"]
        ids += t
        lab += (t if sup else [-100] * len(t))
    return ids, lab


def cam_paths(rec, n_cams=S.NUM_CAMS):  # 수정(2026-10-16): NAVSIM 트랙 — 카메라 수 인자(기본 6 = nuScenes 그대로)
    ps = []
    for p in rec["cams"]:
        q = S.local_path(p)  # 수정(2026-10-26): 클라우드 이전 — 사내 서버 절대경로를 이 서버 경로로(sft_common.local_path)
        if not os.path.exists(q):
            raise FileNotFoundError(f"{rec['id']}: 영상 없음 {q} (대체 금지)")
        ps.append(q)
    if len(ps) != n_cams:
        raise ValueError(f"{rec['id']}: 카메라 {len(ps)}장 != {n_cams}")
    return ps


class QAFrameDataset:
    """한 표본 = 한 프레임 QA 묶음(다중 턴). sft_common.collate 와 같은 키(weights=감독 1.0, is_action_label=False)."""

    def __init__(self, records, processor, max_seq_tokens=None, text_only=False, n_cams=S.NUM_CAMS):
        self.recs, self.proc, self.tok = records, processor, processor.tokenizer
        self.n_cams = n_cams  # 수정(2026-10-16): NAVSIM 트랙 — 전방 3캠(기본 6 = nuScenes 그대로)
        self.max_seq_tokens = max_seq_tokens
        self.text_only = text_only

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        rec = self.recs[i]
        pcs = conversation_pieces(rec, 0 if self.text_only else self.n_cams)  # 수정(2026-10-16): NAVSIM 트랙
        first = pcs[0][0] + pcs[1][0]                         # system + 첫 user(영상 포함) — processor 가 image_pad 를 펼친다
        if self.text_only:
            enc = self.proc(text=[first], return_tensors="pt")
        else:
            enc = self.proc(text=[first], images=[Image.open(p).convert("RGB") for p in cam_paths(rec, self.n_cams)], return_tensors="pt")
        ids = enc["input_ids"][0].tolist()
        lab = [-100] * len(ids)
        for text, sup in pcs[2:]:
            t = self.tok(text, add_special_tokens=False)["input_ids"]
            ids += t
            lab += (t if sup else [-100] * len(t))
        if self.max_seq_tokens and len(ids) > self.max_seq_tokens:
            raise ValueError(f"{rec['id']}: 길이 {len(ids)} > max_seq_tokens {self.max_seq_tokens} (잘라내지 않음 — "
                             f"make_qa_data.py --max_seq_tokens 로 chunk 를 다시 만들 것)")
        n_sup = sum(1 for x in lab if x != -100)
        if n_sup == 0:
            raise ValueError(f"{rec['id']}: 감독 토큰 0")
        item = dict(input_ids=torch.tensor(ids), labels=torch.tensor(lab),
                    weights=torch.tensor([1.0 if x != -100 else 0.0 for x in lab]),
                    is_action_label=torch.zeros(len(ids), dtype=torch.bool), n_prompt=len(enc["input_ids"][0]),
                    token=rec["id"])
        if not self.text_only:
            item["pixel_values"] = enc["pixel_values"]
            item["image_grid_thw"] = enc["image_grid_thw"]
        return item


# ---------------------------------------------------------------- 순전파·손실(projector 까지 기울기)
def vision_features(base, pixel_values, image_grid_thw, train_merger=True):
    """시각 인코더 블록은 no_grad(동결), merger(projector)는 train_merger 면 기울기 경로 유지.
    transformers 4.46.3 Qwen2VisionTransformerPretrainedModel.forward 와 같은 계산(블록 → merger)을 이미지마다 수행
    (sft_common.embed_inputs 와 같은 이유: 묶어 돌리면 fp32 math SDPA 의 어텐션 행렬이 거대해진다)."""
    import torch
    import torch.nn.functional as F
    vis = base.visual
    pv = pixel_values.type(vis.get_dtype())
    sizes = [int(t * h * w) for t, h, w in image_grid_thw.tolist()]
    if sum(sizes) != pv.shape[0]:
        raise ValueError(f"패치 수 {sum(sizes)} != pixel_values {pv.shape[0]}")
    outs = []
    for p, g in zip(pv.split(sizes), image_grid_thw):
        g = g[None]
        with torch.no_grad():
            h = vis.patch_embed(p)
            rpe = vis.rot_pos_emb(g)
            cu = torch.repeat_interleave(g[:, 1] * g[:, 2], g[:, 0]).cumsum(dim=0, dtype=torch.int32)
            cu = F.pad(cu, (1, 0), value=0)
            for blk in vis.blocks:
                h = blk(h, cu_seqlens=cu, rotary_pos_emb=rpe)
        if train_merger:
            outs.append(vis.merger(h))
        else:
            with torch.no_grad():
                outs.append(vis.merger(h))
    return torch.cat(outs, 0)


def qa_forward_loss(model, batch, rope="mrope", train_merger=True):
    """반환 (CE 합(감독 토큰 전체), 감독 토큰 수, 맞힌 토큰 수). 정답 위치만 lm_head 를 통과(sft_common.sft_loss 와 같은 방식).
    정규화는 호출자가 한 optimizer step 의 전체 감독 토큰 수로 나눈다(누적 배치 간 토큰 평균이 정확하도록)."""
    import torch
    import torch.nn.functional as F
    base = S.base_of(model)
    ids = batch["input_ids"]
    emb = base.model.embed_tokens(ids)
    if batch.get("pixel_values") is not None:
        img = vision_features(base, batch["pixel_values"], batch["image_grid_thw"], train_merger)
        m = ids == base.config.image_token_id
        if int(m.sum()) != img.shape[0]:
            raise ValueError(f"영상 토큰 {int(m.sum())} != 영상 특징 {img.shape[0]}")
        emb = emb.masked_scatter(m.unsqueeze(-1).expand_as(emb), img.to(emb.dtype))
    pos = S.position_ids_for(model, ids, batch.get("image_grid_thw"), batch["attention_mask"], rope)
    out = base.model(input_ids=None, inputs_embeds=emb, attention_mask=batch["attention_mask"], position_ids=pos,
                     use_cache=False, return_dict=True)
    h = out.last_hidden_state
    lab = batch["labels"][:, 1:]
    m = lab != -100
    logits = base.lm_head(h[:, :-1][m]).float()
    tgt = lab[m]
    ce = F.cross_entropy(logits, tgt, reduction="sum")
    with torch.no_grad():
        correct = int((logits.argmax(-1) == tgt).sum())
    return ce, int(m.sum()), correct
