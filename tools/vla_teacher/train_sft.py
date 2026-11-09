# 수정(2026-10-15): 교사 VLM ② LoRA SFT — Qwen2-VL-2B 를 자동 생성 교사 정답(nuScenes)으로 지도 미세조정.
"""교사 VLM LoRA SFT.

설계(근거는 smoke_sft_cpu.py 검사·tools/vla_teacher/sft_common.py 주석):
  * LoRA r=16, alpha=32, dropout 0.05 — 언어모델 28층의 attention(q,k,v,o) + 기본으로 MLP(gate,up,down) 포함
    (--lora_targets attn 으로 attention 만). MLP 포함 근거: 이 과제는 (i) 6 영상 → 수치 사실(거리·속도·방위) 회귀형
    매핑과 (ii) 2,041(+v2 문법) 신규 토큰의 의미를 새로 배워야 한다. 지식/매핑은 주로 FFN 에 저장되고(Geva et al.
    2021, "Transformer Feed-Forward Layers Are Key-Value Memories"), 같은 파라미터 예산에서 모든 선형층 LoRA 가
    attention-only 보다 일관되게 낫다(Dettmers et al. 2023 QLoRA §4; Biderman et al. 2024 "LoRA Learns Less and
    Forgets Less"). 비용: 학습 파라미터 4.4M(attn) → 18.5M(attn+mlp), 2B 대비 0.8% — VRAM·속도 영향 미미.
    layer-18 캐시 관점: hidden_states[18] 는 0..17층만 지나므로 18..27층 LoRA 는 캐시에 영향 없고 SFT 손실만 돕는다.
  * 비전 인코더(visual.*, merger 포함) 동결 + no_grad(정규식이 model.layers.* 만 잡음 — 이름 검사로 확인).
  * 특수토큰: --special_tokens 파일(v2) 또는 v1 기본 <action_0..2040> 을 add_tokens(special=True) 로 등록 →
    resize_token_embeddings(64 배수). 새 행만 학습: peft trainable_token_indices(TrainableTokensLayer)가 해당 행의
    값을 별도 fp32 파라미터로 들고 순전파 때 index_copy 로 덮어쓴다 → 기존 151,657 행은 비트 단위로 불변(스모크 검사).
    Qwen2-VL-2B 는 tie_word_embeddings=True 라 lm_head 가 같은 가중치를 공유 → peft 가 lm_head 에 tied 어댑터를
    붙여 '새 입력 임베딩 행 = 새 출력 행' 을 함께 학습한다(별도 lm_head 행 파라미터 없음 — 원 모델의 묶음 구조 유지).
    초기화: 토큰 이름 서브워드 평균 + 씨드 고정 소잡음(sft_common.init_new_rows).
  * 손실: 정답(+<|im_end|>) 토큰만. 미래 불완전 표본은 Intent/Action 줄·종료 토큰 감독 제외. intent_weight(v2) 는
    Intent 구간 CE 에 곱한다. 정규화 = Σw·CE / 감독 토큰 수.
  * 위치: Qwen2-VL 공식 M-RoPE(--rope mrope, generate 와 같음). 기존 캐시 파이프라인(qwen_cache/qwen_online)은
    transformers 4.46.3 forward 가 M-RoPE 를 만들지 않아 사실상 1D 위치였다 — --rope 1d 로 그 방식도 선택 가능.
  * 정밀도: --dtype bf16(기본, 동결 base bf16 + 학습 파라미터 fp32 마스터) | fp32(전부 fp32). 실제 실행 전 사용자 확인 필요.
  * 입력(사용자 결정 2026-10-15): 이전 키프레임(t−0.5 s) 6 캠 + 현재 6 캠 = 12 장(각각 같은 processor, 200·28·28 화소
    예산 → 180 토큰/장 = 2,160 영상 토큰) + 명령 + 자차상태. 순서 기본 prev_first. Qwen2-VL 비디오 입력을 쓰지 않는
    이유: 비디오 경로는 temporal_patch_size=2 로 두 프레임을 한 패치로 합쳐 현재 프레임 토큰이 따로 존재하지 않는다 →
    (i) 현재 프레임만 넣는 'cache' 캐시 모드와 표현 단위가 달라지고, (ii) 영상 전처리가 기존 캐시와 달라진다.
    이미지 12 장이면 프레임별 토큰이 분리되고 전처리가 캐시와 비트 단위로 같다. 이전 프레임이 없는 표본은 현재 프레임을
    반복하고(cams_prev 가 없을 때) 그 표본의 객체 속력 숫자는 손실에서 제외한다.
  * 데이터: train split 만. 고정 hold-out = TRAIN 50 장면(results/vla_teacher/holdout_train_scenes.json, sha256 기록)으로
    체크포인트 선택(teacher-forced 가중 CE). val 은 선택에 절대 쓰지 않는다(eval_teacher.py 전용).
  * 2-fold 교차 적합(--fold A|B, 2026-10-16): data/vla_teacher/v2/fold_split.json(make_fold_split.py)의 train.<fold> 장면만
    학습하고, hold-out 은 그 fold 안의 25 장면(holdout.<fold>)을 쓴다. --fold none(기본) = 이전 동작 그대로.
  * 효율: gradient checkpointing(non-reentrant), 정답 길이 버킷팅(LengthGroupedBatchSampler), 정답 위치만 lm_head.
    시퀀스 packing 은 쓰지 않는다 — M-RoPE 위치가 표본별 3D 이고 varlen attention 이 필요해 복잡도 대비 이득이
    작다(프롬프트가 길이의 ~80% 로 거의 일정해 버킷팅만으로 패딩 낭비 ≈1–2%).
  * GPU0 금지: CUDA_VISIBLE_DEVICES 에 0 이 있거나 미설정이면 즉시 종료.
  * 수정(2026-10-29): NAVSIM 교사 SFT — --navsim_sft_dir 를 주면(기본 None = 위 nuScenes 동작 그대로) NAVSIM 자료
    (make_sft_navsim_scene.py 산출 fold{A,B}_{train,holdout}.jsonl)를 navsim_sft_common.navsim_fold_select 로 읽고(해시·로그 집합
    전수 검사), 전방 3 캠·화소 128·28·28(QA 단계 qa_meta 와 대조)로 학습한다. 학습 수식·하이퍼파라미터 기본값은 같다.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_common import (BASE_QWEN, PIXELS, REPO, TeacherSFTDataset, LengthGroupedBatchSampler, add_special_tokens,  # noqa: E402
                        answer_lengths, build_peft, check_answers_use_known_tokens, check_prompt_parts_roundtrip,
                        collate, has_prev, holdout_scenes, init_new_rows, load_records, load_special_tokens, make_processor,
                        require_gpu_env, resize_for_tokens, resolve_data_dir, save_meta, sft_loss, trainable_summary)
from sft_common import FOLD_FILE, fold_select  # noqa: E402  # 수정(2026-10-16): 2-fold 장면 분할
from sft_common import base_chain_info  # noqa: E402  # 수정(2026-10-16): 설계 검토 반영


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_QWEN)
    ap.add_argument("--data_dir", default=None, help="기본: data/vla_teacher/v2 (없으면 v1 data/vla_teacher)")
    ap.add_argument("--special_tokens", default=None, help="기본: <data_dir>/special_tokens.txt, 없으면 <action_0..2040>")
    ap.add_argument("--out", default=None,  # 수정(2026-10-16): 2-fold 장면 분할
                    help="기본: --fold none → results/vla_teacher/sft_default, --fold A|B → results/vla_teacher/sft_fold<A|B>")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")  # 수정(2026-10-15 밤): 프로젝트 규칙 — 모든 학습 fp32(bf16·fp16 미사용)
    ap.add_argument("--rope", choices=["mrope", "1d"], default="mrope")
    ap.add_argument("--system", choices=["qwen_default", "data"], default="qwen_default",
                    help="qwen_default: 캐시와 같은 system 문장 → 영상 토큰 앞 접두가 SFT·생성·캐시에서 동일(인과 마스크라 "
                         "영상 토큰 은닉상태는 그 뒤 지시문과 무관). data: 정답 생성기의 교사 system 문장")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", choices=["attn", "attn_mlp"], default="attn_mlp")
    ap.add_argument("--new_token_init", choices=["desc", "mean"], default="desc")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_tokens", type=float, default=None, help="새 토큰 행 lr(기본 = --lr)")
    ap.add_argument("--weight_decay", type=float, default=0.0, help="LoRA 에만. 토큰 행은 항상 0(값 자체라 0 으로 끌려감)")
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--drop_cmd_p", type=float, default=0.0)
    ap.add_argument("--drop_ego_p", type=float, default=0.0)
    ap.add_argument("--holdout_scenes", type=int, default=50, help="체크포인트 선택용 TRAIN 장면 수(고정 목록 파일이 없을 때만 사용)")
    ap.add_argument("--holdout_file", default=None,  # 수정(2026-10-16): 2-fold 장면 분할
                    help="고정 hold-out 목록(+sha256). 있으면 그대로 쓰고 해시 검증, 없으면 생성. 기본(--fold none): "
                         "results/vla_teacher/holdout_train_scenes.json. --fold A|B 에서는 쓰지 않는다(fold 파일의 holdout)")
    ap.add_argument("--fold", choices=["A", "B", "none"], default="none",  # 수정(2026-10-16): 2-fold 장면 분할
                    help="교차 적합: A|B = 그 반쪽 train 장면만 학습(hold-out 도 그 반쪽 안). none = 전체 train(이전 동작)")
    ap.add_argument("--fold_file", default=FOLD_FILE, help="장면 분할 파일(make_fold_split.py 산출, 해시 검증)")  # 수정(2026-10-16): 2-fold 장면 분할
    ap.add_argument("--dry_run", action="store_true",  # 수정(2026-10-16): 2-fold 장면 분할
                    help="데이터 구성(표본 수·장면 수·step 수)만 출력하고 종료. 모델·출력 폴더를 만들지 않는다(CPU 가능)")
    ap.add_argument("--prev", choices=["auto", "on", "off"], default="auto",
                    help="이전 키프레임(t−0.5 s) 6 캠 추가 입력. auto = 레코드에 cams_prev/prev_available 이 있으면 on")
    ap.add_argument("--frame_order", choices=["prev_first", "cur_first"], default="prev_first",
                    help="12 장 순서. prev_first = 사용자 지정(이전 6 → 현재 6). 정답 지시문의 순서 서술과 일치해야 한다")
    ap.add_argument("--holdout_every", type=int, default=400, help="optimizer step 간격(에폭 끝에도 평가)")
    ap.add_argument("--save_every", type=int, default=200, help="latest(재개용) 저장 간격")
    ap.add_argument("--max_answer_tokens", type=int, default=768)   # 수정(2026-10-15): v2-sensor 정답 547(+종료) 토큰
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0, help="스모크용(0 = 에폭 기준)")
    ap.add_argument("--max_train_samples", type=int, default=0, help="스모크 전용(0 = 전체). 실제 학습에서 쓰지 말 것")
    ap.add_argument("--max_holdout_samples", type=int, default=0, help="스모크 전용(0 = 전체)")
    ap.add_argument("--max_vram_gb", type=float, default=80.0, help="프로세스 VRAM 상한(torch 할당기 상한, GPU1 예산)")
    ap.add_argument("--smoke_fake_prev", action="store_true",
                    help="스모크 전용: cams_prev 가 아직 없는 정답으로 12 장 비용을 재기 위해 현재 프레임을 이전 자리에 복제")
    ap.add_argument("--no_grad_ckpt", action="store_true",  # 수정(2026-10-16): 사용자 지시 "손실 없이 시간 단축" — 재계산 생략
                    help="gradient checkpointing 끔(같은 배치·같은 수식, 활성값을 저장해 재계산 시간 절약; VRAM 증가)")
    ap.add_argument("--tf32", action="store_true", help="fp32 모드에서 TF32 matmul 허용(기본 끔 = 엄격 fp32)")
    ap.add_argument("--sdpa", choices=["math", "default"], default="math",  # 수정(2026-10-16): SFT 발산 원인 진단
                    help="fp32 어텐션 커널. math(기본) = 정확한 fp32. default = torch 자동 선택(fp32 에서 memory-efficient 커널 — "
                         "이 GPU 에서 부정확해 발산 원인이었다. 재현·진단용으로만)")
    ap.add_argument("--resume", action="store_true", help="<out>/latest 에서 재개")
    ap.add_argument("--device", default=None, help="스모크용 'cpu' (CUDA_VISIBLE_DEVICES='' 일 때만)")
    ap.add_argument("--diag", action="store_true",  # 수정(2026-10-16): SFT 발산 원인 진단
                    help="진단 로그: 그룹별 clip 전 기울기 노름(LoRA A/B·새 토큰 행), 정답 위치 최대 |로짓|, 새/기존 행 노름, "
                         "정답 구간별 CE, 그룹별 갱신 크기. 학습 계산은 바꾸지 않는다(로그 전용)")
    ap.add_argument("--stop_at_step", type=int, default=0,  # 수정(2026-10-16): SFT 발산 원인 진단
                    help="진단 전용: 이 optimizer step 에서 멈춘다(학습률 스케줄은 전체 길이 기준 그대로 — --max_steps 와 다름)")
    ap.add_argument("--log_every", type=int, default=10, help="로그 간격(optimizer step). --diag 면 기본 1")  # 수정(2026-10-16): SFT 발산 원인 진단
    # 수정(2026-10-29): NAVSIM 교사 SFT — 아래 3 개는 NAVSIM 전용 선택 인자. 주지 않으면 종전(nuScenes) 동작·메타 그대로.
    ap.add_argument("--navsim_sft_dir", default=None,
                    help="NAVSIM SFT 자료 폴더(make_sft_navsim_scene.py 산출). 주면 --fold A|B 필수, 분할 파일 기본 = NAVSIM 분할")
    ap.add_argument("--num_cams", type=int, default=None, help="프레임당 카메라 수. 기본 6(nuScenes) / --navsim_sft_dir 면 3")
    ap.add_argument("--pixels", type=int, default=None, help="영상 화소 예산. 기본 200·28·28 / --navsim_sft_dir 면 128·28·28")
    args = ap.parse_args(argv)
    resolve_navsim_args(args)  # 수정(2026-10-29): NAVSIM 교사 SFT
    if args.diag and "--log_every" not in (argv if argv is not None else sys.argv):  # 수정(2026-10-16): SFT 발산 원인 진단
        args.log_every = 1
    if args.fold != "none" and args.holdout_file is not None:  # 수정(2026-10-16): 2-fold 장면 분할
        raise SystemExit("--fold A|B 에서는 hold-out 이 fold 파일에 고정되어 있다 — --holdout_file 을 주지 말 것")
    if args.out is None:  # 수정(2026-10-16): 2-fold 장면 분할
        args.out = "results/vla_teacher/sft_default" if args.fold == "none" else f"results/vla_teacher/sft_fold{args.fold}"
    if args.holdout_file is None and args.fold == "none":  # 수정(2026-10-16): 2-fold 장면 분할
        args.holdout_file = "results/vla_teacher/holdout_train_scenes.json"
    return args


_NAVSIM_ARG_KEYS = ("navsim_sft_dir", "num_cams", "pixels")   # 수정(2026-10-29): NAVSIM 교사 SFT — nuScenes 메타에는 기록하지 않는다(종전 메타 그대로)


def resolve_navsim_args(args):  # 수정(2026-10-29): NAVSIM 교사 SFT
    """--navsim_sft_dir 없음: num_cams/pixels 는 종전 값(6, PIXELS)만 허용. 있음: fold A|B·NAVSIM 값(3, 128·28·28) 강제."""
    import navsim_sft_common as NS
    if args.navsim_sft_dir is None:
        if args.num_cams not in (None, 6) or args.pixels not in (None, PIXELS):
            raise SystemExit("--num_cams/--pixels 를 바꾸는 것은 --navsim_sft_dir 와 함께일 때만 허용(자료와 영상 수·화소가 어긋남)")
        args.num_cams, args.pixels = 6, PIXELS
        return
    if args.fold not in ("A", "B"):
        raise SystemExit("--navsim_sft_dir 는 --fold A|B 필요(교차 적합 교사)")
    if args.data_dir is not None:
        raise SystemExit("--navsim_sft_dir 와 --data_dir 를 함께 주지 말 것(자료 폴더는 --navsim_sft_dir 하나)")
    if args.fold_file == FOLD_FILE:
        args.fold_file = NS.NAVSIM_FOLD_FILE
    if args.num_cams not in (None, NS.NAVSIM_NUM_CAMS) or args.pixels not in (None, NS.NAVSIM_PIXELS):
        raise SystemExit(f"NAVSIM 은 전방 {NS.NAVSIM_NUM_CAMS} 캠·화소 {NS.NAVSIM_PIXELS} 만(QA 단계와 같은 값)")
    args.num_cams, args.pixels = NS.NAVSIM_NUM_CAMS, NS.NAVSIM_PIXELS


# 수정(2026-10-29): 저장·재개 안전 — 재개 시 이 인자들이 latest 의 메타 args 와 같아야 한다(학습 궤적·자료·평가 규칙을 정하는 값).
_RESUME_KEYS = ("base", "data_dir", "special_tokens", "dtype", "rope", "system", "lora_r", "lora_alpha", "lora_dropout",
                "lora_targets", "new_token_init", "epochs", "batch", "accum", "lr", "lr_tokens", "weight_decay", "warmup",
                "clip", "drop_cmd_p", "drop_ego_p", "holdout_scenes", "holdout_file", "fold", "fold_file", "prev",
                "frame_order", "holdout_every", "max_answer_tokens", "seed", "max_steps", "max_train_samples",
                "max_holdout_samples", "smoke_fake_prev", "tf32", "sdpa", "navsim_sft_dir", "num_cams", "pixels")
_RESUME_PATH_KEYS = ("base", "data_dir", "special_tokens", "holdout_file", "fold_file", "navsim_sft_dir")


def resume_arg_diff(saved, args):  # 수정(2026-10-29): 저장·재개 안전 — 체인 스크립트의 '끝난 SFT 건너뛰기' 출처 검사도 이 함수를 쓴다
    """저장된 메타 args(dict) vs 현재 args → (다른 항목 설명 목록, 저장본에 없는 키 목록). 경로 인자는 절대 경로로 비교."""
    norm = lambda k, v: os.path.abspath(v) if (k in _RESUME_PATH_KEYS and isinstance(v, str)) else v  # noqa: E731
    cur = vars(args)
    diff, missing = [], []
    for k in _RESUME_KEYS:
        if k not in saved:
            if not (args.navsim_sft_dir is None and k in _NAVSIM_ARG_KEYS):
                missing.append(k)
            continue
        if json.dumps(norm(k, saved[k])) != json.dumps(norm(k, cur.get(k))):
            diff.append(f"{k}: 저장 {saved[k]!r} != 현재 {cur.get(k)!r}")
    return diff, missing


def check_resume_state(args, out):  # 수정(2026-10-29): 저장·재개 안전(train_qa_full.py 시작 검사와 같은 규칙)
    """--resume: latest/trainer_state.pt 필수(없으면 처음부터 돌리지 않고 거부), latest 교체 중 중단 흔적(.tmp/.old 만 남음)
    이면 복구 방법을 알리고 거부, best 교체 중 중단(best 없음 + best.old)도 거부, latest 메타 args 와 현재 인자 비교(다르면 거부).
    --navsim_sft_dir 이고 --resume 이 아니면: 이전 산출물이 있으면 거부(덮어써서 기록이 섞이지 않게). nuScenes 의 비재개 실행은 종전 그대로."""
    lt = out / "latest"
    if args.resume:
        if not (lt / "trainer_state.pt").exists():
            left = [n for n in ("latest.tmp", "latest.old") if (out / n).exists()]
            how = (f" latest 교체 중 중단 흔적 {left}: latest.tmp 에 trainer_state.pt·adapter_model.safetensors·teacher_meta.json 이 "
                   f"모두 있으면 그것(더 최신)을, 아니면 latest.old 를 latest 로 이름 바꾼 뒤 --resume") if left and not lt.exists() else ""
            raise SystemExit(f"[거부] --resume 인데 {lt}/trainer_state.pt 없음 — 처음부터 돌리지 않는다.{how}")
        if (out / "best.old").exists() and not (out / "best").exists():
            raise SystemExit(f"[거부] {out}/best 없이 best.old 만 있다(best 교체 중 중단) — best.old 를 best 로 옮긴 뒤 재개")
        mp = lt / "teacher_meta.json"
        if not mp.exists():
            raise SystemExit(f"[거부] {mp} 없음 — 재개 인자 비교 불가")
        lm = json.load(open(mp))
        diff, missing = resume_arg_diff(lm.get("args", {}), args)
        if diff:
            raise SystemExit("[거부] 재개 인자가 latest 와 다르다 — 같은 인자로 재개하거나 새 출력 폴더로 시작:\n  " + "\n  ".join(diff))
        if missing:
            print(f"[resume] 경고: latest 메타에 없는 인자(비교 생략) {missing}", flush=True)
        if args.navsim_sft_dir is not None and not lm.get("run_id"):  # 수정(2026-10-29): 2차 검토 — NAVSIM 은 한 실행의 모든 저장에 run_id
            raise SystemExit(f"[거부] {mp} 에 run_id 없음 — 이 코드로 시작한 NAVSIM SFT 의 latest 가 아니다")
        return lm
    elif args.navsim_sft_dir is not None:
        prev = [n for n in ("latest", "latest.tmp", "latest.old", "best", "best.tmp", "best.old", "final", "final.tmp")
                if (out / n).exists()]
        if prev:
            raise SystemExit(f"[거부] {out} 에 이전 학습 산출물 {prev} — latest 가 없고 latest.tmp/latest.old 만 있으면 그것을 latest 로 "
                             f"옮긴 뒤 --resume, 아니면 --resume 으로 이어 가거나 폴더를 옮긴 뒤 새로 시작")
    return None


def navsim_select(args):  # 수정(2026-10-29): NAVSIM 교사 SFT — (학습, hold-out, hold-out 로그, 해시, 분할 파일, fold 정보)
    import navsim_sft_common as NS
    tr, ho, hold, hold_hash, fp, d, ninfo = NS.navsim_fold_select(args.navsim_sft_dir, args.fold, args.fold_file)
    e = d["train"][args.fold]
    finfo = dict(fold=args.fold, fold_file=fp, fold_sha256=d["sha256"], fold_scenes=e["n_scenes"],
                 fold_samples=e["n_samples"], generate=d["assignment"][f"teacher_{args.fold}"]["generate"], navsim_sft=ninfo)
    return tr, ho, hold, hold_hash, fp, finfo


def check_navsim_base(args, base_chain):  # 수정(2026-10-29): NAVSIM 교사 SFT — QA 단계와 영상 설정이 같아야 한다
    if base_chain["kind"] != "qa_full":
        raise SystemExit("[거부] NAVSIM SFT 의 --base 는 NAVSIM QA 전체 학습 체크포인트(qa_meta.json)여야 한다")
    qm = json.load(open(Path(args.base) / "qa_meta.json"))
    if qm.get("pixels") != args.pixels or qm.get("cams") != f"current {args.num_cams}":
        raise SystemExit(f"[거부] QA 단계 pixels={qm.get('pixels')}·cams={qm.get('cams')!r} != SFT {args.pixels}·current {args.num_cams}")


def select_train_holdout(args, recs, nav_sel=None):  # 수정(2026-10-16): 2-fold 장면 분할
    """train 레코드 → (학습 표본, hold-out 표본, hold-out 장면 set, 해시, 출처, fold 정보 dict|None).
    --fold none 은 이전 코드와 같은 식(전체 train − 고정 50 장면), A|B 는 fold 파일의 반쪽 − 그 반쪽 hold-out.
    수정(2026-10-29): NAVSIM 교사 SFT — nav_sel(navsim_select 결과)을 주면 그것을 쓴다(스모크 자르기만 같은 규칙)."""
    if nav_sel is not None:
        tr, ho, hold, hold_hash, hold_path, finfo = nav_sel
    elif args.fold == "none":
        hold, hold_hash, hold_path = holdout_scenes(recs, args.holdout_scenes, args.seed, args.holdout_file)
        tr = [r for r in recs if r["scene_token"] not in hold]
        ho = [r for r in recs if r["scene_token"] in hold]
        finfo = None
    else:
        tr, ho, hold, hold_hash, hold_path, d = fold_select(recs, args.fold, args.fold_file)
        e = d["train"][args.fold]
        finfo = dict(fold=args.fold, fold_file=hold_path, fold_sha256=d["sha256"], fold_scenes=e["n_scenes"],
                     fold_samples=e["n_samples"], generate=d["assignment"][f"teacher_{args.fold}"]["generate"])
    if args.max_train_samples:
        tr = tr[:args.max_train_samples]
    if args.max_holdout_samples:
        ho = ho[:args.max_holdout_samples]
    return tr, ho, hold, hold_hash, hold_path, finfo


def build_model(args, tok, toks, device):
    import torch
    from transformers import Qwen2VLForConditionalGeneration
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.base, torch_dtype=dt, attn_implementation="sdpa")
    new_ids, n_base = add_special_tokens(tok, toks)
    resize_for_tokens(model, len(tok))
    init_new_rows(model, tok, toks, new_ids, n_base, mode=args.new_token_init, seed=args.seed)
    for p in model.parameters():
        p.requires_grad_(False)
    if not args.no_grad_ckpt:  # 수정(2026-10-16): --no_grad_ckpt 면 재계산 없이(수식 동일) — 기본은 종전대로 켬
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    pm = build_peft(model, new_ids, args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_targets)
    for n, p in pm.named_parameters():
        if p.requires_grad and ".visual." in n:
            raise RuntimeError(f"비전 인코더 파라미터가 학습 대상이다: {n}")
    pm.to(device)
    return pm, new_ids, n_base


def diag_grads(lora_named, tok_p, tok, n_base):  # 수정(2026-10-16): SFT 발산 원인 진단
    """clip 전 그룹별 기울기 노름. LoRA A/B 분리, 가장 큰 LoRA 텐서, 새 토큰 행 중 가장 큰 행(토큰 이름)."""
    import torch
    with torch.no_grad():
        a2 = b2 = 0.0; top_n, top_v = None, -1.0
        for n, p in lora_named:
            if p.grad is None:
                continue
            v = float(p.grad.float().pow(2).sum())
            if "lora_A" in n:
                a2 += v
            else:
                b2 += v
            if v > top_v:
                top_v, top_n = v, n
        g = tok_p[0].grad
        if g is None:
            rows = torch.zeros(1)
        else:
            rows = g.float().norm(dim=-1)
        k = min(3, rows.numel())
        tv, ti = rows.topk(k)
    return dict(g_lora=math.sqrt(a2 + b2), g_lora_A=math.sqrt(a2), g_lora_B=math.sqrt(b2),
                g_lora_top=math.sqrt(max(top_v, 0.0)),
                g_lora_top_name=(top_n or "").replace("base_model.model.model.layers.", "L"),
                g_tok=float(rows.norm()), g_tok_rows_top=[(tok.convert_ids_to_tokens(n_base + int(i)), float(v))
                                                          for v, i in zip(tv, ti)],
                g_tok_rows_nonzero=int((rows > 0).sum()))


def evaluate_holdout(pm, dl, device, rope):
    import torch
    pm.eval()
    tot, n, acc_n, acc_s, act_s, act_n = 0.0, 0, 0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in dl:
            b = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            loss, nt, st = sft_loss(pm, b, rope)
            tot += loss.item() * nt; n += nt
            acc_s += st["acc"] * nt; acc_n += nt
            if "action_acc" in st:
                k = int(b["is_action_label"][:, 1:][b["labels"][:, 1:] != -100].sum())
                act_s += st["action_acc"] * k; act_n += k
    pm.train()
    return dict(loss=tot / max(n, 1), tok_acc=acc_s / max(acc_n, 1),
                action_tok_acc=(act_s / act_n) if act_n else None, n_tok=n)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("OPENBLAS_CORETYPE", "SkylakeX")
    dev_kind = require_gpu_env(allow_cpu=(args.device == "cpu" or args.dry_run))  # 수정(2026-10-16): 2-fold 장면 분할
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup
    device = torch.device("cpu" if dev_kind == "cpu" else "cuda")
    if args.dtype == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        if device.type == "cuda" and args.sdpa == "math":  # 수정(2026-10-16): SFT 발산 원인 진단 — 원인: fp32 memory-efficient SDPA 부정확
            from sft_common import exact_fp32_attention
            print(f"[attn] {exact_fp32_attention(torch.float32)}", flush=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "cuda":
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_vram_gb / tot), 0)

    out = Path(args.out) if os.path.isabs(args.out) else REPO / args.out
    resume_meta = None
    if not args.dry_run:  # 수정(2026-10-16): 2-fold 장면 분할 — dry run 은 출력 폴더를 만들지 않는다
        resume_meta = check_resume_state(args, out)  # 수정(2026-10-29): 저장·재개 안전 — 모델 적재 전에 거부
        out.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_data_dir(args.navsim_sft_dir or args.data_dir)  # 수정(2026-10-29): NAVSIM 교사 SFT
    toks, tok_src = load_special_tokens(args.special_tokens, data_dir)

    nav_sel = None  # 수정(2026-10-29): NAVSIM 교사 SFT — NAVSIM 은 fold 파일 2 개(학습·hold-out)를 직접 읽는다
    if args.navsim_sft_dir is not None:
        nav_sel = navsim_select(args)
        recs = nav_sel[0] + nav_sel[1]
    else:
        recs = load_records(data_dir, "train")
    if args.smoke_fake_prev:
        if not (args.max_steps and args.max_steps <= 20):
            raise SystemExit("--smoke_fake_prev 는 스모크 전용(--max_steps ≤ 20)")
        for r in recs:
            r.setdefault("cams_prev", r["cams"]); r.setdefault("prev_available", True)
    check_answers_use_known_tokens(recs, toks)
    if args.drop_cmd_p > 0 or args.drop_ego_p > 0:
        bad = check_prompt_parts_roundtrip(recs)
        if bad:
            raise SystemExit(f"prompt_parts 조립이 prompt.user 와 {bad}건 불일치 — dropout 프롬프트가 생성기와 다르게 "
                             f"조립된다. sft_common.prompt_parts_list 를 v2 형식에 맞출 것.")
    tr, ho, hold, hold_hash, hold_path, finfo = select_train_holdout(args, recs, nav_sel)  # 수정(2026-10-16): 2-fold 장면 분할
    # 수정(2026-10-16): 설계 검토 반영 — --base 출처 검사(QA 모델이면 fold·분할 해시 일치, qa_meta 없으면 원본 Qwen 만)
    base_chain = base_chain_info(args.base, args.fold, args.fold_file)
    print(f"[base] {base_chain}", flush=True)
    if args.navsim_sft_dir is not None:  # 수정(2026-10-29): NAVSIM 교사 SFT
        check_navsim_base(args, base_chain)
    use_prev = (args.prev == "on") or (args.prev == "auto" and has_prev(recs))
    if not args.dry_run:  # 수정(2026-10-16): 2-fold 장면 분할
        json.dump(dict(scenes=sorted(hold), sha256=hold_hash, source=hold_path, n_train=len(tr), n_holdout=len(ho),
                       **({} if finfo is None else finfo)), open(out / "holdout_scenes.json", "w"), indent=1)
    print(f"[data] {data_dir} train {len(tr)} 표본 / hold-out {len(ho)} 표본({len(hold)} train 장면, sha256 {hold_hash[:12]}), "
          f"특수토큰 {len(toks)} ({tok_src}), 이전 프레임 {'on' if use_prev else 'off'}({args.frame_order})", flush=True)
    if finfo is not None:  # 수정(2026-10-16): 2-fold 장면 분할
        print(f"[fold] {args.fold}: {finfo['fold_scenes']} 장면 / {finfo['fold_samples']} 표본 중 학습 "
              f"{len({r['scene_token'] for r in tr})} 장면·hold-out {len(hold)} 장면, 분할 sha256 {finfo['fold_sha256'][:12]}, "
              f"이 교사가 생성할 몫 {finfo['generate']}, 출력 {out}", flush=True)
    if args.dry_run:  # 수정(2026-10-16): 2-fold 장면 분할 — 토크나이저만 읽어 정답 길이·배치·step 수 계산 후 종료(모델 미적재)
        proc = make_processor(args.base, args.pixels)  # 수정(2026-10-29): NAVSIM 교사 SFT — 기본 args.pixels == PIXELS
        tok = proc.tokenizer
        add_special_tokens(tok, toks)
        lens = answer_lengths(tok, tr)
        n_b = len(LengthGroupedBatchSampler(lens, args.batch, seed=args.seed))
        spe = math.ceil(n_b / args.accum)
        total = args.max_steps or int(math.ceil(spe * args.epochs))
        print(json.dumps(dict(dry_run=True, fold=args.fold, out=str(out), n_train=len(tr), n_holdout=len(ho), base_chain=base_chain,  # 수정(2026-10-16): 설계 검토 반영
                              n_train_scenes=len({r["scene_token"] for r in tr}), n_holdout_scenes=len(hold),
                              answer_tokens_min=min(lens), answer_tokens_max=max(lens), batch=args.batch, accum=args.accum,
                              micro_batches_per_epoch=n_b, steps_per_epoch=spe, epochs=args.epochs, total_steps=total,
                              **({} if args.navsim_sft_dir is None else dict(num_cams=args.num_cams, pixels=args.pixels,  # 수정(2026-10-29): NAVSIM 교사 SFT
                                                                             holdout_logs=len(hold), navsim_sft=finfo["navsim_sft"]))),
                         ensure_ascii=False), flush=True)
        return

    proc = make_processor(args.base, args.pixels)  # 수정(2026-10-29): NAVSIM 교사 SFT — 기본 args.pixels == PIXELS
    tok = proc.tokenizer
    pm, new_ids, n_base = build_model(args, tok, toks, device)
    summ = trainable_summary(pm)
    print(f"[model] 학습 파라미터 {summ}", flush=True)
    action_ids = [i for t, i in zip(toks, new_ids) if t.startswith("<action_")]

    # 수정(2026-10-29): 2차 검토 — 실행 식별자(run_id): 새 실행이면 새로, 재개면 latest 의 것. trainer_state 에는 항상,
    #   메타(latest·best·final)에는 NAVSIM 에서만 기록(nuScenes 메타는 종전 그대로) → 체인이 final·best 가 한 실행인지 확인.
    import uuid
    run_id = (resume_meta or {}).get("run_id") or uuid.uuid4().hex
    # 처리기·토크나이저·메타(병합/평가/캐시가 같은 설정 사용)
    proc.save_pretrained(out / "processor")
    meta = dict(base=args.base, data_dir=str(data_dir), special_tokens=toks, special_tokens_src=tok_src,
                n_base_vocab=n_base, new_ids=[new_ids[0], new_ids[-1]], vocab_after=len(tok),
                pixels=args.pixels, rope=args.rope, system=args.system, dtype=args.dtype, use_prev=use_prev,  # 수정(2026-10-29): NAVSIM 교사 SFT(기본 = PIXELS)
                frame_order=args.frame_order, holdout_sha256=hold_hash,
                **({} if finfo is None else dict(fold=finfo)),  # 수정(2026-10-16): 2-fold 장면 분할
                base_chain=base_chain,  # 수정(2026-10-16): 설계 검토 반영 — QA → SFT 연결 기록(GRPO·캐시 추출이 다시 검사)
                lora=dict(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, targets=args.lora_targets),
                new_token_init=args.new_token_init,
                # 수정(2026-10-29): NAVSIM 교사 SFT — nuScenes 는 종전과 같은 args(NAVSIM 인자 키 제외)·키 순서, NAVSIM 은 num_cams 추가
                args={k: v for k, v in vars(args).items() if args.navsim_sft_dir is not None or k not in _NAVSIM_ARG_KEYS},
                trainable=summ, **({} if args.navsim_sft_dir is None else dict(num_cams=args.num_cams, run_id=run_id)))
    from sft_common import save_meta as _sm
    _sm(out / "teacher_meta.json", **meta)

    ds_tr = TeacherSFTDataset(tr, proc, args.system, args.drop_cmd_p, args.drop_ego_p, args.seed,
                              max_answer_tokens=args.max_answer_tokens, action_ids=action_ids, use_prev=use_prev,
                              frame_order=args.frame_order, with_segments=args.diag,  # 수정(2026-10-16): SFT 발산 원인 진단
                              n_cams=args.num_cams)  # 수정(2026-10-29): NAVSIM 교사 SFT(기본 6)
    ds_ho = TeacherSFTDataset(ho, proc, args.system, 0.0, 0.0, args.seed,
                              max_answer_tokens=args.max_answer_tokens, action_ids=action_ids, use_prev=use_prev,
                              frame_order=args.frame_order, n_cams=args.num_cams)  # 수정(2026-10-29): NAVSIM 교사 SFT(기본 6)
    lens = answer_lengths(tok, tr)
    if max(lens) > args.max_answer_tokens:
        raise SystemExit(f"정답 최대 {max(lens)} 토큰 > --max_answer_tokens {args.max_answer_tokens}")
    sampler = LengthGroupedBatchSampler(lens, args.batch, seed=args.seed)
    col = lambda b: collate(b, tok.pad_token_id)  # noqa: E731
    mk_dl = lambda: DataLoader(ds_tr, batch_sampler=sampler, num_workers=args.workers, collate_fn=col,  # noqa: E731
                               pin_memory=(device.type == "cuda"), persistent_workers=False)
    dl_ho = DataLoader(ds_ho, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=col)

    steps_per_epoch = math.ceil(len(sampler) / args.accum)
    total = args.max_steps or int(math.ceil(steps_per_epoch * args.epochs))
    lora_p = [p for n, p in pm.named_parameters() if p.requires_grad and "lora_" in n]
    tok_p = [p for n, p in pm.named_parameters() if p.requires_grad and "trainable_tokens" in n]
    opt = torch.optim.AdamW([dict(params=lora_p, weight_decay=args.weight_decay, lr=args.lr),
                             dict(params=tok_p, weight_decay=0.0, lr=args.lr_tokens or args.lr)],
                            betas=(0.9, 0.999), eps=1e-8)
    sch = get_cosine_schedule_with_warmup(opt, int(args.warmup * total), total)
    if args.diag:  # 수정(2026-10-16): SFT 발산 원인 진단
        lora_named = [(n, p) for n, p in pm.named_parameters() if p.requires_grad and "lora_" in n]
        base_m = pm.get_base_model()
        emb_l, head_l = base_m.model.embed_tokens, base_m.lm_head
        print(f"[diag] embed {type(emb_l).__name__} lm_head {type(head_l).__name__} "
              f"head_tied_to_embed={getattr(getattr(head_l, 'token_adapter', None), 'tied_adapter', None) is getattr(emb_l, 'token_adapter', 0)} "
              f"tie_word_embeddings={base_m.config.tie_word_embeddings} n_tok_param={len(tok_p)} "
              f"warmup_steps={int(args.warmup * total)} total={total}", flush=True)
        with torch.no_grad():
            W0 = emb_l.token_adapter.base_layer.weight
            base_row_norm = W0[:n_base].float().norm(dim=-1)
            tok_init = tok_p[0].detach().clone()
        diag_base = dict(base_row_norm_mean=base_row_norm.mean().item(), base_row_norm_max=base_row_norm.max().item(),
                         new_row_norm_init_mean=tok_init.norm(dim=-1).mean().item(),
                         base_elem_std=W0[:n_base].float().std().item())
        print("[diag]", json.dumps(diag_base), flush=True)
        log_f0 = open(out / "train_log.jsonl", "a"); log_f0.write(json.dumps(dict(kind="diag_init", **diag_base)) + "\n"); log_f0.close()

    step, epoch, best = 0, 0, float("inf")
    start_batch = 0
    best_ev, repair_best = None, False  # 수정(2026-10-29): 2차 검토 — 현재 best 의 hold-out 기록(재개 시 best 폴더 대조·복구용)
    if args.resume and (out / "latest" / "trainer_state.pt").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        st = torch.load(out / "latest" / "trainer_state.pt", map_location="cpu", weights_only=False)
        sd = load_file(str(out / "latest" / "adapter_model.safetensors"))
        set_peft_model_state_dict(pm, sd)
        for p in pm.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"])
        step, epoch, best, start_batch = st["step"], st["epoch"], st["best"], st["batch_in_epoch"]
        if "total" in st and st["total"] != total:  # 수정(2026-10-29): 저장·재개 안전(종전 trainer_state 엔 total 없음)
            raise SystemExit(f"[거부] 재개: 전체 step {st['total']} != 현재 설정 {total} — 같은 인자로 재개할 것")
        # 수정(2026-10-29): 2차 검토 — (i) RNG 복원(종전 상태엔 없음 → 그대로 적재). DataLoader 반복자를 재개 시점에 새로 만들므로
        #   중단 없는 실행과 비트 단위로 같지는 않다(LoRA dropout 등 torch RNG 흐름은 저장 시점 상태에서 이어짐).
        if "rng" in st:
            random.setstate(st["rng"]["py"]); torch.set_rng_state(st["rng"]["torch"])
            if device.type == "cuda" and st["rng"].get("cuda") is not None:
                torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        # (ii) best 기준 대조: best/ 가 latest 보다 앞서면 거부(저장 순서가 latest → best 라 생기지 않아야 함), 아니면
        #   best = min(상태 best, best/ 손실). 상태가 더 좋은 best 를 아는데 best/ 가 그보다 나쁘면 = latest 저장 직후·best 저장 전 중단
        #   → 그 best 가 바로 이 latest step 이면(best_ev.step == step) 적재한 가중치로 best 를 다시 저장, 아니면 거부.
        if args.navsim_sft_dir is None:
            run_id = st.get("run_id") or run_id          # nuScenes: 메타엔 없음 → 상태의 것을 이어 씀
        elif st.get("run_id") != run_id:
            raise SystemExit(f"[거부] latest trainer_state run_id {st.get('run_id')} != latest 메타 run_id {run_id}")
        best_ev = st.get("best_ev")
        bmp = out / "best" / "teacher_meta.json"
        bm = json.load(open(bmp)) if bmp.exists() else None
        b_step = bm.get("step") if bm else None
        b_loss = (bm.get("holdout") or {}).get("loss") if bm else None
        if bm is not None and args.navsim_sft_dir is not None and bm.get("run_id") != run_id:
            raise SystemExit(f"[거부] {bmp} run_id {bm.get('run_id')} != latest {run_id} — 다른 실행의 best")
        if b_step is not None and b_step > step:
            raise SystemExit(f"[거부] best step {b_step} > latest step {step} — best 가 latest 보다 앞선다(저장 순서상 생기지 않아야 함). "
                             f"원인을 확인한 뒤, 그 best 를 버리고 latest 에서 이어 가려면 best 를 best.ahead_{b_step} 로 옮긴 뒤 재개")
        if b_loss is not None and b_loss <= best:
            best = b_loss
        elif best < float("inf"):
            if best_ev is not None and best_ev.get("step") == step and best_ev.get("loss") == best:
                repair_best = True
            else:
                raise SystemExit(f"[거부] 상태의 best {best:.6f} 가 best/ 손실 {b_loss} 보다 좋은데 그 체크포인트를 복구할 수 없다(best_ev {best_ev})")
        print(f"[resume] step {step} epoch {epoch} batch {start_batch} best {best:.4f} (best/ step {b_step}) "
              f"rng {'복원' if 'rng' in st else '없음(종전 상태)'} run_id {run_id}", flush=True)

    def save(tag, extra=None, trainer_state=None):
        # 수정(2026-10-29): 저장·재개 안전 — 종전엔 d 를 제자리에서 덮어쓰고(save_pretrained) 그 뒤 trainer_state.pt 를 써서,
        #   중간 중단 시 어댑터·메타·상태가 서로 다른 step 인 latest 가 남을 수 있었다. 이제 tag.tmp 에 전부(trainer_state 포함)
        #   쓴 뒤 rename(d→d.old)·rename(tmp→d)·rmtree(d.old) — 어느 순간 중단돼도 d 는 완전하거나 없다(train_qa_full.save_full 과 같은 방식).
        import shutil
        d, tmp, old = out / tag, out / f"{tag}.tmp", out / f"{tag}.old"
        if tmp.exists():
            shutil.rmtree(tmp)
        pm.save_pretrained(tmp)
        tok.save_pretrained(tmp)
        _sm(tmp / "teacher_meta.json", **meta, step=step, **(extra or {}))
        if trainer_state is not None:
            torch.save(trainer_state, tmp / "trainer_state.pt")
        if old.exists():
            shutil.rmtree(old)
        if d.exists():
            d.rename(old)
        tmp.rename(d)
        if old.exists():
            shutil.rmtree(old)
        return d

    def tstate(ep, b_next):  # 수정(2026-10-29): 2차 검토 — 재개 상태(+total·run_id·best_ev·RNG)
        rng = dict(py=random.getstate(), torch=torch.get_rng_state(),
                   cuda=(torch.cuda.get_rng_state_all() if device.type == "cuda" else None))
        return dict(opt=opt.state_dict(), sch=sch.state_dict(), step=step, epoch=ep, best=best, batch_in_epoch=b_next,
                    total=total, run_id=run_id, best_ev=best_ev, rng=rng)

    if repair_best:  # 수정(2026-10-29): 2차 검토 — latest 저장 직후·best 저장 전 중단 복구(가중치 = 이 latest step = 그 best)
        print(f"[resume] best 복구: step {step} hold-out {best:.6f} 를 latest 가중치로 best 에 저장", flush=True)
        save("best", dict(holdout=best_ev))

    log_f = open(out / "train_log.jsonl", "a")
    if args.diag:  # 수정(2026-10-16): SFT 발산 원인 진단
        lora_mods = [m for n, m in pm.named_modules() if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})]
        lora_mod_names = [n.replace("base_model.model.model.layers.", "L") for n, m in pm.named_modules()
                          if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})]
    pm.train()
    base = pm.get_base_model()
    base.visual.eval()
    t0, tok_seen, done, step0 = time.time(), 0, step >= total, step  # 수정(2026-10-29): 2차 검토 — 이미 total 이면 추가 step 없이 final
    if done:
        print(f"[resume] step {step} ≥ total {total} — 학습 없이 final·요약만 저장", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    while not done:
        ds_tr.set_epoch(epoch); sampler.set_epoch(epoch)
        dl = mk_dl()
        opt.zero_grad(set_to_none=True)
        micro = 0
        micro_st = []  # 수정(2026-10-16): SFT 발산 원인 진단
        n_b = len(sampler)
        sampler.skip = start_batch          # 재개: 이미 끝낸 배치는 표집 단계에서 제외(영상 로드 없이)
        for bi, batch in enumerate(dl, start=start_batch):
            b = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}
            loss, nt, st = sft_loss(pm, b, args.rope, diag_n_base=(n_base if args.diag else None))  # 수정(2026-10-16): SFT 발산 원인 진단
            if args.diag:
                micro_st.append(dict(st, loss=loss.item()))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"비유한 손실 step {step} batch {bi} tokens {batch['tokens']} — 원인 진단 필요(건너뛰지 않음)")
            (loss / args.accum).backward()
            tok_seen += int(b["attention_mask"].sum())
            micro += 1
            last = (bi + 1 == n_b)
            if micro % args.accum and not last:
                continue
            if micro % args.accum:            # 에폭 끝 잔여 마이크로배치: 버리지 않고 accum 비율을 보정해 한 step
                for p_ in lora_p + tok_p:
                    if p_.grad is not None:
                        p_.grad.mul_(args.accum / (micro % args.accum))
            micro = 0
            if args.diag:  # 수정(2026-10-16): SFT 발산 원인 진단 — clip 전 그룹별 기울기·step 전 값
                dg = diag_grads(lora_named, tok_p, tok, n_base)
                prev_lora = [p.detach().clone() for p in lora_p]; prev_tok = tok_p[0].detach().clone()
                lr_now = [g["lr"] for g in opt.param_groups]
            gn = torch.nn.utils.clip_grad_norm_(lora_p + tok_p, args.clip)
            if not torch.isfinite(gn):
                if args.diag:  # 수정(2026-10-16): SFT 발산 원인 진단 — 실패 직전 상태를 남기고 그대로 실패
                    rec = dict(step=step + 1, kind="diag_nonfinite", gnorm=float(gn), **dg, micro=micro_st)
                    print(json.dumps(rec), flush=True); log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                raise FloatingPointError(f"비유한 기울기 노름 step {step} — 원인 진단 필요(건너뛰지 않음)")
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if args.diag:  # 수정(2026-10-16): SFT 발산 원인 진단
                with torch.no_grad():
                    upd_l = math.sqrt(sum(float((p.detach() - q).pow(2).sum()) for p, q in zip(lora_p, prev_lora)))
                    upd_rows = (tok_p[0].detach() - prev_tok).norm(dim=-1)
                    rn = tok_p[0].detach().norm(dim=-1)
                    drift = (tok_p[0].detach() - tok_init).norm(dim=-1)
                    ba = []
                    for mod in lora_mods:
                        A = mod.lora_A["default"].weight; B_ = mod.lora_B["default"].weight
                        ba.append(float((B_ @ A).norm()) * mod.scaling["default"])
                    j = int(torch.tensor(ba).argmax())
                rec = dict(step=step, kind="diag", lr_lora=lr_now[0], lr_tok=lr_now[1], gnorm=float(gn), **dg,
                           upd_lora=upd_l, upd_tok=float(upd_rows.norm()), upd_tok_row_max=float(upd_rows.max()),
                           new_row_norm_mean=float(rn.mean()), new_row_norm_max=float(rn.max()),
                           new_row_norm_argmax=tok.convert_ids_to_tokens(n_base + int(rn.argmax())),
                           new_row_drift_max=float(drift.max()), new_row_drift_mean=float(drift.mean()),
                           lora_delta_max=max(ba), lora_delta_max_mod=lora_mod_names[j], lora_delta_sum=sum(ba),
                           micro=micro_st)
                print(json.dumps(rec), flush=True); log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                micro_st = []
            if step % args.log_every == 0 or step == 1:  # 수정(2026-10-16): SFT 발산 원인 진단
                el = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                rec = dict(step=step, epoch=epoch, loss=loss.item(), ce=st["ce"], tok_acc=st["acc"],
                           action_acc=st.get("action_acc"), gnorm=float(gn), lr=sch.get_last_lr()[0],
                           s_per_step=el / max(step - step0, 1), tok_per_s=tok_seen / el,
                           peak_gb=round(mem, 2), eta_h=(total - step) * el / max(step - step0, 1) / 3600)
                print(json.dumps(rec), flush=True); log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            best_now = False  # 수정(2026-10-29): 2차 검토 — best 가 바뀐 step 은 latest 도 저장하고, 순서는 latest → best
            if step % args.holdout_every == 0 or step == total:
                ev = evaluate_holdout(pm, dl_ho, device, args.rope)
                ev.update(step=step, epoch=epoch, kind="holdout")
                print(json.dumps(ev), flush=True); log_f.write(json.dumps(ev) + "\n"); log_f.flush()
                if ev["loss"] < best:
                    best, best_ev, best_now = ev["loss"], ev, True
            if step % args.save_every == 0 or step == total or best_now:
                save("latest", trainer_state=tstate(epoch, bi + 1))  # 수정(2026-10-29): 저장·재개 안전 — tmp 안에서 함께 저장
            if best_now:
                save("best", dict(holdout=ev))
            if step >= total or (args.stop_at_step and step >= args.stop_at_step):  # 수정(2026-10-16): SFT 발산 원인 진단
                done = True
                break
        start_batch = 0
        sampler.skip = 0
        if not done:
            ev = evaluate_holdout(pm, dl_ho, device, args.rope)
            ev.update(step=step, epoch=epoch, kind="holdout_epoch_end")
            print(json.dumps(ev), flush=True); log_f.write(json.dumps(ev) + "\n"); log_f.flush()
            epoch += 1
            if ev["loss"] < best:  # 수정(2026-10-29): 2차 검토 — 에폭 끝 best 도 latest(다음 에폭 시작 상태)를 먼저 저장 → best 가 latest 를 앞서지 않음
                best, best_ev = ev["loss"], ev
                save("latest", trainer_state=tstate(epoch, 0))
                save("best", dict(holdout=ev))
    save("final")
    el = time.time() - t0
    summary = dict(steps=step, wall_s=el, s_per_step=el / max(step - step0, 1), tok_per_s=tok_seen / el, best_holdout_loss=best,
                   peak_gb=(torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None))
    json.dump(summary, open(out / "train_summary.json", "w"), indent=1)
    print("[done]", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
