# 수정(2026-10-15): 경로 이동 models/qwen_online.py → models/encoder/vlm_qwen2vl/qwen_online.py (소스 정리).
# 수정(2026-10-09): LaQ-AD — Qwen2-VL-2B 은닉 토큰 "단일 인코더". [공용 모듈 — 벤치마크 무관, 이름에 태그 없음]
#   NAVSIM(8캠)은 지시문·시점 수만 tools/qwen_cache_navsim.py 에서 넘긴다(num_views, instruction).
#   nuScenes 캐시(tools/qwen_cache.py)와 같은 지시문·화소예산·계층으로 6방향 영상을 인코딩한다.
#   쓰임 두 곳이 같은 함수를 공유해야 학습(캐시)과 추론(온라인)이 같은 입력 분포를 본다:
#     (1) tools/qwen_cache_b2d.py      — B2D 학습/개방루프용 오프라인 캐시 생성
#     (2) team_code/hipad_b2d_laq_agent.py — CARLA 폐쇄루프에서 매 틱 온라인 인코딩(캐시 불가)
#   누락 영상은 검은 영상으로 대체하지 않고 즉시 실패한다(프로젝트 no-skip 규칙).
import torch

# tools/qwen_cache.py 의 INSTRUCTION 과 글자 단위로 같아야 한다(Run10 캐시 qwen_l18 생성 문구).
INSTRUCTION = (
    "These are 6 surround-view camera images of an autonomous vehicle (front, "
    "front-left, front-right, back-left, back, back-right). Describe the salient "
    "objects, lane geometry, traffic state, and any driving-relevant cues."
)
# 지시문이 말하는 순서. 데이터셋마다 카메라 이름이 다르므로 호출자가 이 순서로 정렬해 넘긴다.
INSTRUCTION_ORDER = ("front", "front_left", "front_right", "back_left", "back", "back_right")
# Bench2Drive(CARLA) 카메라 이름 -> 지시문 순서
B2D_CAM_ORDER = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                 "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")
PIXELS = 200 * 28 * 28   # tools/qwen_cache.py --min/--max-pixels 기본값과 동일
LAYER = 18               # Run10 캐시 계층(hidden_states[18]; [0]=임베딩)


class QwenOnlineEncoder:
    """동결 Qwen2-VL-2B 로 (1, L, 1536) bf16 은닉 토큰과 (1, L) uint8 마스크를 만든다."""

    def __init__(self, qwen_path, layer=LAYER, pixels=PIXELS, device="cuda",
                 instruction=INSTRUCTION, num_views=6):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        self.layer = int(layer)
        self.device = device
        self.instruction = instruction
        self.num_views = int(num_views)  #수정(2026-10-09): NAVSIM 8캠. 기본 6 = nuScenes/B2D 기존 동작
        # 캐시 생성과 같은 설정(bf16, eager attention)으로 적재한다.
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen_path, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(
            qwen_path, min_pixels=pixels, max_pixels=pixels)

    @torch.no_grad()
    def encode(self, images, ego_line=None):
        """images: 지시문이 말하는 순서의 PIL.Image num_views 장(기본 6 = INSTRUCTION_ORDER).
        ego_line: 수정(2026-10-15) B안 — 교사 캐시(tools/vla_teacher/extract_teacher_cache.py --prompt cache_ego)와 같은
            자차 상태 문장(tools/vla_teacher/common_v2.ego_line, 고정폭, 지시문 뒤 +41 토큰 → L 1,200). 주면 지시문 뒤에 '\n' 으로 붙인다.
            None 이면 종전 동작(지시문만) — 자차 상태 없는 캐시(qwen_l18 · cache 모드)와 같다."""
        if len(images) != self.num_views or any(im is None for im in images):
            raise ValueError(f"{self.num_views}방향 영상이 모두 필요하다(받은 수 {len(images)}).")
        msgs = [{"role": "user", "content":
                 [{"type": "image", "image": im} for im in images] +
                 [{"type": "text", "text": self.instruction if ego_line is None else self.instruction + "\n" + ego_line}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=[text], images=list(images), return_tensors="pt", padding=True)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        out = self.model(
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"],
            pixel_values=inp["pixel_values"].to(torch.bfloat16),
            image_grid_thw=inp.get("image_grid_thw"),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states[self.layer].to(torch.bfloat16)      # (1, L, 1536)
        mask = inp["attention_mask"].to(torch.uint8)                   # (1, L)
        return hidden, mask
