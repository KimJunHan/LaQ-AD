# 수정(2026-10-09): SFT(병합 모델) vs SFT+GRPO(KL 기준=SFT) 공정 비교 — compare_adapters.py 에서 파생.
# 원 설명(2026-09-29): B1 vs B2' 공정 비교 — 재현율이 다르면 매칭쌍 MAE 비교가 불공정하다
#   (더 많이 보고할수록 먼/어려운 객체가 매칭에 포함돼 MAE 가 기계적으로 커짐).
#   따라서 ① 두 모델이 "모두" 언급한 GT 부분집합에서의 MAE ② 거리대역별 MAE ③ 재현율을 함께 낸다.
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from spatial_common import SPATIAL_INSTRUCTION, load_six_cams, load_infos_by_token, parse_caption

def gen(model, processor, imgs):
    msgs = [{"role": "user", "content": [{"type": "image"}] * 6 +
             [{"type": "text", "text": SPATIAL_INSTRUCTION}]}]
    t = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[t], images=[imgs], return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = model.generate(**enc, do_sample=False, max_new_tokens=256,
                           pad_token_id=processor.tokenizer.pad_token_id)
    return processor.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]

def match(text, gt_objs):
    """GT index -> 거리오차 (매칭 실패 시 None). r_match 와 동일한 일대일 그리디."""
    _, preds = parse_caption(text); used=set(); out={}
    for gi, g in enumerate(gt_objs):
        best, bi = None, None
        for i, p in enumerate(preds):
            if i in used or p["name"] != g["name"].split(".")[-1]: continue
            e = abs(p["dist"] - g["dist"])
            if best is None or e < best: best, bi = e, i
        if bi is not None: used.add(bi); out[gi] = best
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--sft", default="ckpts/qwen2-vl-2b-3d")
ap.add_argument("--grpo-adapter", default="results/qwen3d_nusc_grpo_sftref/final")
ap.add_argument("--out", default="results/qwen3d_nusc_compare_sft_grpo.json")
a = ap.parse_args()
px = 200*28*28
proc = AutoProcessor.from_pretrained('ckpts/qwen2-vl-2b', min_pixels=px, max_pixels=px)
rows = [json.loads(l) for l in open('data/qwen3d/spatial_captions_val.jsonl')][:a.n]
infos = load_infos_by_token('data/infos/nuscenes_infos_val.pkl')
texts = {}
for tag, adp in zip(["B1","B2p"], [None, a.grpo_adapter]):   # B1 = SFT 병합 모델, B2p 자리 = SFT+GRPO(sftref)
    base = Qwen2VLForConditionalGeneration.from_pretrained(a.sft,
             torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda().eval()
    m = (PeftModel.from_pretrained(base, adp) if adp else base).eval()
    texts[tag] = [gen(m, proc, load_six_cams(infos[r["token"]], 'data/nuscenes')) for r in rows]
    del m, base; torch.cuda.empty_cache()
    print(f"{tag} 생성 완료", flush=True)

common, onlyA, onlyB, rec = {"B1":[], "B2p":[]}, [], [], {"B1":0, "B2p":0}
band = {"B1":{}, "B2p":{}}; ngt=0
for k, r in enumerate(rows):
    g = r["objects"]; ngt += len(g)
    mA, mB = match(texts["B1"][k], g), match(texts["B2p"][k], g)
    rec["B1"] += len(mA); rec["B2p"] += len(mB)
    for gi in set(mA) & set(mB):
        common["B1"].append(mA[gi]); common["B2p"].append(mB[gi])
    for tag, mm in (("B1",mA), ("B2p",mB)):
        for gi, e in mm.items():
            b = min(int(g[gi]["dist"]//15), 3)
            band[tag].setdefault(b, []).append(e)
print(f"\n표본 {len(rows)} · GT {ngt}개")
print(f"재현율   B1 {rec['B1']/ngt:.3f}   B2' {rec['B2p']/ngt:.3f}")
print(f"전체매칭 MAE  B1 {np.mean([e for v in band['B1'].values() for e in v]):.2f}m  "
      f"B2' {np.mean([e for v in band['B2p'].values() for e in v]):.2f}m   (재현율 다름 → 불공정)")
print(f"공통 GT {len(common['B1'])}개에서의 MAE  B1 {np.mean(common['B1']):.2f}m  "
      f"B2' {np.mean(common['B2p']):.2f}m   ← 공정 비교")
for b in sorted(set(band['B1']) | set(band['B2p'])):
    lo = b*15; hi = "55+" if b==3 else f"{lo+15}"
    f=lambda t: f"{np.mean(band[t][b]):.2f}m(n={len(band[t][b])})" if b in band[t] else "-"
    print(f"  거리 {lo}~{hi}m  B1 {f('B1'):18} B2' {f('B2p')}")

json.dump(dict(n=len(rows), n_gt=ngt, recall={k: v / ngt for k, v in rec.items()},
               mae_common={k: float(np.mean(v)) for k, v in common.items()}, n_common=len(common["B1"]),
               mae_band={t: {str(b): float(np.mean(e)) for b, e in band[t].items()} for t in band}),
          open(a.out, "w"), indent=1)
print("->", a.out, "  (B1=SFT, B2p=SFT+GRPO sftref)")
