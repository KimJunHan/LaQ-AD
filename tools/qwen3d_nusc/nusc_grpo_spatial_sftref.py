# 수정(2026-10-09): GRPO 재실험 — KL 기준을 SFT 모델로 고친 판(nuScenes 전용, 사용자 지시).
#   [근본 원인] 이전 grpo_spatial.py 는 SFT LoRA 를 이어 학습하면서 model.disable_adapter() 로 KL 기준을 잡아,
#   기준이 SFT 가 아니라 **원본 Qwen**(공간 서술 재현율 0)이었다 → KL 이 정책을 SFT 가 배운 행동에서 멀어지게 당겼다.
#   [수정] SFT 병합 모델(ckpts/qwen2-vl-2b-3d)을 base 로 불러 그 위에 **새 LoRA** 를 붙인다. disable_adapter() 순전파가
#   곧 SFT 모델이므로 KL 기준 = SFT 정책(표준 RLHF 구성). 새 LoRA 는 B=0 초기화라 step 0 정책 = SFT.
#   보상은 B2′ 수정본(spatial_common.smooth_reward: σ_d 2, σ_v 1, 일대일 정밀도항 0.30) 그대로, 다른 하이퍼파라미터도 B2′ 와 동일
#   (K=4, 2000 step, lr 2e-6, β 0.02) → 이전 B2′ 와의 차이는 KL 기준 하나.
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel, LoraConfig, get_peft_model
from spatial_common import SPATIAL_INSTRUCTION, load_six_cams, load_infos_by_token, smooth_reward

def seq_logprob(model, ids, prompt_len, pad_id, pixel_values, image_grid_thw):
    """생성 토큰 구간의 per-token logprob 합.
    수정(리뷰): ① Qwen2-VL 은 이미지 placeholder 임베딩 때문에 pixel_values 필수
    ② 우측 padding(pad_id) 은 mask 에서 제외 (K개 생성 시퀀스 길이가 달라 pad 존재)."""
    attn = (ids != pad_id).long()
    out = model(input_ids=ids, attention_mask=attn,   # 수정(리뷰2): enc 가 이미 K배 — repeat 제거
                pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    logits = out.logits[:, :-1]
    tgt = ids[:, 1:]
    # 수정(리뷰3): log_softmax(float()) 는 (K,L,151k) fp32 를 통째 실체화(~3.5GB)+backward 보존.
    #   F.cross_entropy(fused) 는 N×C 중간 텐서 없이 동일 값 → Run9 병행 시 OOM 여유 확보.
    lp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                          reduction="none").view_as(tgt).float()
    mask = torch.zeros_like(lp)
    mask[:, prompt_len - 1:] = 1.0
    mask = mask * (tgt != pad_id).float()   # padding 표적 제외
    return (lp * mask).sum(-1), mask.sum(-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default="ckpts/qwen2-vl-2b-3d", help="SFT(B1) 병합 모델 = 정책 초기값이자 KL 기준")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--jsonl", default="data/qwen3d/spatial_captions_train.jsonl")
    ap.add_argument("--infos", default="data/infos/nuscenes_infos_train.pkl")
    ap.add_argument("--data-root", default="data/nuscenes")
    ap.add_argument("--out", default="results/qwen3d_nusc_grpo_sftref")
    ap.add_argument("--group", type=int, default=4, help="GRPO 그룹 크기 K")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--kl-beta", type=float, default=0.02)
    ap.add_argument("--pixels", type=int, default=200 * 28 * 28)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--ap-alpha", type=float, default=1.0, help="AP-GRPO 절대 보상 지수 α (0=기존 GRPO)")
    ap.add_argument("--max-steps-smoke", type=int, default=0, help="VRAM 측정용 스모크(저장 없이 종료)")
    a = ap.parse_args()

    # 수정(2026-09-28): K=4 명시 배치에서 eager vision attention 이 (17,280패치)^2 softmax
    #   로 17.8GB 요구 → OOM x10. flash_attention_2(varlen) 는 이미지 블록별 선형 메모리.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        a.qwen, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    resume = os.path.join(a.out, "latest")   # 수정(리뷰2): crash 시 최신 저장분에서 warm-start
    if os.path.isdir(resume):
        model = PeftModel.from_pretrained(model, resume, is_trainable=True)
        print("adapter resume:", resume)
    else:
        # SFT 와 같은 LoRA 구성(r16, α32, dropout 0.05, 언어모델 어텐션 q/k/v/o)
        lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        print("fresh LoRA on SFT-merged base:", a.qwen)
    model.print_trainable_parameters()
    processor = AutoProcessor.from_pretrained(a.qwen, min_pixels=a.pixels, max_pixels=a.pixels)
    rows = [json.loads(l) for l in open(a.jsonl)]
    infos = load_infos_by_token(a.infos)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    rng = random.Random(0)
    os.makedirs(a.out, exist_ok=True)

    r_hist = []
    for step in range(1, a.steps + 1):
        gt = rng.choice(rows)
        imgs = load_six_cams(infos[gt["token"]], a.data_root)
        msgs = [{"role": "user", "content": [{"type": "image"}] * 6 +
                 [{"type": "text", "text": SPATIAL_INSTRUCTION}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # 수정(리뷰2): num_return_sequences 는 pixel_values(평탄 패치 텐서)를 패치 단위로
        # interleave 해 이미지 간 패치가 섞일 수 있다 → K배 배치를 명시적으로 구성.
        enc = processor(text=[text] * a.group, images=[imgs] * a.group,
                        return_tensors="pt", padding=True).to("cuda")
        plen = enc["input_ids"].shape[1]
        pad_id = processor.tokenizer.pad_token_id
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                                 max_new_tokens=a.max_new, pad_token_id=pad_id)
        texts = processor.batch_decode(gen[:, plen:], skip_special_tokens=True)
        rewards = torch.tensor([smooth_reward(t, gt) for t in texts], device="cuda")
        r_hist += rewards.tolist()
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)   # 그룹 정규화 (GRPO)
        # 수정(2026-10-09): AP-GRPO(Smooth Operator, arXiv 2601.07695 Eq.8) — 연속 수치 보상에서 그룹 정규화가
        #   거의 맞춘(near-miss) 표본의 advantage 를 뭉개는 문제를 절대 보상 배율 r̃^α 로 보정. 원 논문 구현값 α=1.0.
        #   보상은 smooth_reward 로 [0,1] 범위(가중합 0.05+0.10+0.55+0.30=1)라 r̃ = reward 를 그대로 쓴다. α=0 이면 기존 GRPO.
        adv = adv * rewards.clamp(min=0.0, max=1.0) ** a.ap_alpha
        if float(rewards.std()) < 1e-3:   # 그룹 내 차이 없으면 학습 신호 없음
            continue
        # policy logprob (어댑터 ON) vs reference (어댑터 OFF = SFT 병합 모델)
        lp_pol, ntok = seq_logprob(model, gen, plen, pad_id, enc["pixel_values"], enc["image_grid_thw"])
        with torch.no_grad(), model.disable_adapter():
            lp_ref, _ = seq_logprob(model, gen, plen, pad_id, enc["pixel_values"], enc["image_grid_thw"])
        kl = (lp_pol - lp_ref) / ntok.clamp(min=1)
        loss = -(adv * lp_pol / ntok.clamp(min=1)).mean() + a.kl_beta * kl.mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 20 == 0:
            print(f"step {step}/{a.steps} R(mean last100) {sum(r_hist[-100:])/min(100,len(r_hist)):.3f} "
                  f"loss {loss.item():.4f} kl {kl.mean().item():.4f} "
                  f"peak {torch.cuda.max_memory_allocated()//2**20}MB", flush=True)
        if step % 500 == 0 or step == a.steps:
            model.save_pretrained(f"{a.out}/step{step}")
            model.save_pretrained(f"{a.out}/latest")
        if a.max_steps_smoke and step >= a.max_steps_smoke:
            print(f"SMOKE done step {step} peak {torch.cuda.max_memory_allocated()//2**20}MB", flush=True); return
    model.save_pretrained(f"{a.out}/final")
    print("GRPO done ->", f"{a.out}/final")

if __name__ == "__main__":
    main()
