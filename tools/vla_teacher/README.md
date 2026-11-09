# 수정(2026-10-15): 교사 VLM ① 정답 자동 생성

LaQ-AD 교사 VLM(nuScenes)의 감독 정답을 infos 에서 결정적으로 만든다. 교사는 학생과 독립이다.

- **입력(prompt)**: 6 카메라(Qwen 순서 F, FL, FR, BL, B, BR) + 주행 명령(nuScenes 표준) + 현재 자차 상태(CAN ego_status). 학생 예측·미래 정보(명령 제외) 없음.
- **출력(answer)**: 4줄 — 인지 → 판단 → 행동.
  1. `Perception:` 자차 속도 + 55 m 안 가장 가까운 객체 최대 12개 (구 공간 캡션 문법 + `(<lat>m left|right, <lon>m ahead|behind)`)
  2. `Map:` 20 m 안 횡단보도(거리·방위) / divider·boundary 폴리라인 수 (map_annos, ROI x±15 m·y±30 m)
  3. `Intent: <longitudinal>, <lateral>.`
  4. `Action:` 코드북(K=2041, 0.5 s/토큰) 토큰 6개 = 3 s 자차 미래. 미래 불완전 표본은 `Intent: unavailable.` / `Action: unavailable.`

## 파일

| 파일 | 역할 |
|---|---|
| `common.py` | 상수·임계값(근거 주석), 객체/지도/의도/코드북 토큰화, 프롬프트, 정답 조립, 엄격 파서 `parse_answer` |
| `gen_teacher_gt.py` | 생성(`data/vla_teacher/teacher_gt_{train,val}.jsonl`, `instruction.txt`, `stats.json`). `--explore` = 임계값 근거 통계(train 만) |
| `check_teacher_gt.py` | 검증: 파서 왕복, 누설, 투영 contact sheet(`check_{0..4}.png`), 장면 분리, 개수, 이미지 실재 → `check_report.json` |

재사용(수정 없음): `tools/qwen3d_nusc/spatial_common.py`(카메라 순서·`parse_caption`·`smooth_reward`), `tools/qwen3d_nusc/nusc_action_codebook.py`(`heading_series`; 토큰화 알고리즘은 같은 식을 `common.ActionCodebook` 에 옮김), `data/qwen3d/nusc_action_codebook_k2048.npz`.

## 실행 (CPU 전용, cwd = 저장소 루트)

```bash
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=.
PY=/opt/anaconda3/envs/hipad_bw/bin/python
$PY tools/vla_teacher/gen_teacher_gt.py --explore   # 임계값 근거
$PY tools/vla_teacher/gen_teacher_gt.py             # 생성(기존 산출물 있으면 중단; --force 로 data/vla_teacher 안만 덮어씀)
$PY tools/vla_teacher/check_teacher_gt.py           # 검증(이미지 실재 검사 때문에 NFS 에서 ~15 분)
```

## 규칙 요약 (상세·근거는 `common.py` 주석)

- 객체: `num_lidar_pts>0` & 10 검출 클래스 & lidar xy 거리 ≤ 55 m (v2 레시피와 동일), 원값 거리 정렬, 상위 12. 속도 ≥ 0.5 m/s 면 moving. 속도 NaN → 0(구 캡션과 동일, 개수 stats 기록).
- 자차 속도: CAN `|vel_xy|`. CAN 결측 장면(ego_status 전부 0, train 597 표본/15 장면)은 프롬프트에 `Ego state: unavailable.`, 정답 속도는 포즈 후진 차분(장면 첫 프레임은 전진 차분) — `targets.ego_speed_src` 에 출처 기록.
- 종방향: `v_end=|p6−p4|/1s`; `v_end<0.5` stop, `v_end−v0 ≥ +1.0` accelerate, `≤ −1.0` decelerate, 그 외 keep speed.
- 횡방향: 경로 < 2 m → go straight; 3 s 끝 진행방향 ±12.5° 이상 → turn; 그 외 미래 경로가 map divider 를 가로지르면 lane change(교차 방향); 그 외 go straight.
- 미래 불완전(`gt_ego_fut_masks` 미완, 장면 끝 6 프레임) 표본은 버리지 않고 JSONL 에 남기되 의도·행동을 `unavailable` 로 명시(계획 평가기와 같은 규칙으로 학습/평가 시 의도·행동 감독에서 제외).

---

# 수정(2026-10-15): 교사 VLM ① 정답 v2 (사용자 승인 4개 개선)

v1 코드·산출물은 그대로(재현 가능). v2 는 새 모듈 + `data/vla_teacher/v2/`.

| 파일 | 역할 |
|---|---|
| `common_v2.py` | 특수토큰·v2 문법·엄격 파서(`parse_answer`/`format_parsed` 왕복)·`answer_spans`, 중요도 객체 선택, 과거 포즈 자차 상태, copy-baseline, 분리 가능한 프롬프트(`prompt_parts`·`join_prompt`) |
| `lane_v2.py` | 차로변경 3 신호(지도 위상 / 구분선 교차 / 차로 기준 운동학) + 2-of-3 판정 |
| `gen_teacher_gt_v2.py` | 생성 → `v2/teacher_gt_{train,val}.jsonl`, `stats.json`, `instruction.txt`, `grammar.txt`, `special_tokens.txt` |
| `check_teacher_gt_v2.py` | 검증 → `v2/check_report.json`, `v2/check_{0..4}.png` |
| `audit_lc_v2.py` | 차로변경 감사 contact sheet → `v2/audit_lc_{lc,straight,uncertain}_NN.png`, `audit_lc_index.json` |

```bash
export CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=.
PY=/opt/anaconda3/envs/hipad_bw/bin/python
$PY tools/vla_teacher/gen_teacher_gt_v2.py --force [--ego_in_prompt current|past_pose|none] [--object_speed numeric|dir_only] [--out_dir ...]   # ~8 분
$PY tools/vla_teacher/check_teacher_gt_v2.py      # 이미지는 contact sheet 5장만 읽음
$PY tools/vla_teacher/audit_lc_v2.py              # 전방 3캠 × 175 표본만 읽음
```
`--map-root` 기본 `data/nuscenes`(maps/expansion/*.json, 도시당 1회 읽기). NAS 부하를 줄이려면 json 4개(48 MB)를 로컬에 복사해 그 루트를 넘긴다.

## 변경점
1. **차로변경 = 2-of-3 신호**(같은 방향): (a) 위상 — t0 차로 L0 의 도달 집합(L0, incoming, outgoing BFS 4단: 교차로 connector·분기 포함)에 없는 차로에 t5·t6 이 배정되고 체인 기준 횡오프셋 2–6 m(인접); (b) v1 구분선 교차; (c) 체인 기준 횡변위 ≥1.6 m + 그쪽으로 ≥3° 기운 진행방향 + 현재 yaw rate 가 반대로 >0.1 rad/s 아님(지도 차로 없으면 v1 원호 잔차 ≥1.0 m 대체). 1 신호(또는 방향 충돌) → `uncertain`: 정답은 `go straight`, `targets.intent.uncertain=true`, `targets.intent_weight=0.3`(완전 1.0, 미래 불완전 0.0). 회전·종방향 규칙은 v1 그대로.
2. **지름길 완화**: 정답에서 자차 속도 제거. `prompt_parts`(instruction / command / ego / *_dropped) — SFT 드롭아웃은 `join_prompt(parts, drop_command=, drop_ego=)`, 자차 상태 없는 캐시 프롬프트는 `join_prompt(parts, drop_ego=True, ego_mode="omit")`. `copy_baseline` = 명령만으로 의도(횡=명령, 종=keep speed/stop) + 현재 자차 상태 CTRV 3 s 궤적·행동 토큰·ADE.
3. **CAN 결측**: 과거 포즈(t−1, t−2 샘플)만으로 속도·종가속·yaw rate(`ego_status.src='pose_past'`, `n_past`), 장면 첫 프레임 → `unavailable`. `ego_drop_ok`.
4. **짧은 정답**: `grammar.txt` 문법(특수토큰, 단위 없는 1 소수 숫자, 부호 있는 횡/종 오프셋), 객체 최대 8개 = 중요도(경로 회랑 ±2 m → TTC≤6 s 접근 이동체 → 20 m 안 VRU → 최근접) 로 고른 뒤 거리순. 경로 회랑은 **라벨이므로 GT 미래 궤적**을 쓴다(미래 불완전 시 직진 경로 max(3 s·v0, 10 m)); 입력에는 영향 없음.
5. (검토 반영) **입력**: 이전 키프레임(t−0.5 s) 6 캠 `cams_prev` + 현재 6 캠(장면 첫 샘플은 현재 복제, `prev_available=false`). 플래그 `--ego_in_prompt`(기본 current), `--object_speed`(기본 numeric) — `stats.json.modes`·레코드 `modes` 에 기록. **손실 마스킹**: `answer_spans`(줄별 문자 구간, 객체 속력 숫자 구간), `loss_mask_spans`(미래 불완전 → Intent·Action 줄, 이전 프레임 없음 → 객체 속력 숫자), `loss_mask_token_spans`(answer 단독 토큰화 기준).

## 수정(2026-10-15): 정답 형식 `v2-sensor` (v2 에 통합 — 고정 길이·센서 대체 정보)

사용자 지시("v2 랑 통합")로 별도 v3 대신 v2 정답을 확장했다. **프롬프트(지시문·명령·B안 고정폭 자차 상태 줄, `EGO_DROPPED` 42 토큰, `EGO_LINE_TOKENS=41`, 캐시 L 1,200)는 그대로**이고 정답(answer)만 바뀐다. 이전(가변 길이) v2 산출물·코드는 `/tmp/claude-0/-workspace-src/868c57e8-6d51-4ca5-b2e3-25a0b7e11891/scratchpad/v2_preIntegrate_backup/`(세션 임시 — 필요하면 NAS 로 옮길 것)에 있다. 레코드·`stats.json` 의 `modes.answer_format = "v2-sensor"` 로 구분.

목표: 교사가 RGB 만으로 LiDAR/레이더 정보(위치·크기·진행방위·속도·미래 변위·충돌 위험·차선/경계 횡거리·자유거리)를 말하게 하고, 그 **생성 토큰**을 캐시해 학생 쿼리에 넣는다. 캐시 길이가 표본마다 같아야 하므로(가변 → 패딩 → sm_120 에서 실패하는 flash varlen 역방향) **정답 토큰 수를 전 표본에서 같게** 만든다 — 실측 **547 토큰**(ckpts/qwen2-vl-2b + `special_tokens.txt`, `<|im_end|>` 제외; train 28,130 + val 6,019 전수). 프롬프트 캐시(1,200) 뒤에 붙이면 1,747.

- 문법 전문: `data/vla_teacher/v2/grammar.txt`. 숫자는 고정폭(부호 `SDD.D` 5 토큰, 무부호 `DD.D` 4, 정수 `DD` 2 — Qwen2 는 숫자를 한 자리씩 자름). 대체형 = 머리 토큰 + `<_>` 채움(같은 토큰 수): `<none>…`/`<pc_on>…`/`<pc_none>…` = 구조적 없음(감독), `<na>…` = GT 미상(`loss_mask_spans` 로 손실 제외). 범위를 넘는 값은 자르지 않고 생성 실패(train 실측 최대: 길이 34.4 m, 속력 25.7 m/s, 3 s 변위 68.3 m, 지도 개수 16).
- **Objects**(항상 8 칸, 57 토큰/칸): 선택 = 기존 v2 중요도 규칙 → 거리순. `CLS SEC 횡x 종y 길이 폭 진행방위 운동 미래1s 미래2s 미래3s 위험`.
  - 진행방위 `<hF>…<hFL>` = 박스 yaw 의 자차 기준 8 방위(yaw↔속도 방향 p50 0.76°로 규약 확인). traffic_cone 은 `<na>`(방향 미정의).
  - 운동 = `<static>` 또는 이동방향 섹터 + 속력 `DD.D`(static 도 실제 속력, < 0.5). GT 속도 NaN → `<na>`.
  - 미래 = `gt_agent_fut_trajs`(lidar 프레임 **0.5 s 간격 증분** — 컨버터 `fut_traj_scene[1:]-fut_traj_scene[:-1]`, 첫 증분/0.5 와 gt_velocity 차 p50 0.25 m/s 로 확인) 누적의 1/2/3 s 값. 해당 시점까지 마스크 미완 → 그 시점만 `<na>`.
  - 위험 `<ttc_lt2|2_4|4_6|safe>`: t ∈ {0,…,6 s} 에서 자차 박스(4.084×1.85, 0.5 m 전방 오프셋, 평가기 `get_yaw` 규약; 3–6 s 는 마지막 0.5 s 등속 외삽)와 객체 박스(GT 미래, 마스크 끊기면 마지막 증분 등속, 증분 없으면 gt_velocity) 간격 ≤ `TTC_GAP` 인 첫 시각. `TTC_GAP=1.4 m` = train 선택 객체 0–3 s 최소 간격 p5(`--explore_ttc`, val 미사용; 0 m 순수 겹침은 99.6% safe 로 퇴화). 자차 미래 불완전 → `<na>`.
- **Map**(68 토큰): 기존 v2 의 횡단보도·divider/boundary 개수(2 자리) + 전방 0/10/20 m 단면(`<at0>` …)의 왼/오른 divider, 왼/오른 boundary 까지 |x| (없음 `<none>`) + `<free>` = 자차 폭 회랑(|x|≤0.925 m)에서 처음 만나는 boundary/객체 박스까지 y(상한 **30 m** = map_annos ROI 전방 한계; 사양 예시 50 m 에서 변경 — 30 m 너머 경계는 GT 에 없다). 회랑은 현재 진행축 직선(차로 중심선 아님).
- **Intent/Action**: v2 와 같은 값. 미래 불완전 시 `Intent: <na><_>`, `Action: <na><_>×5`(이전 `<na>` 1 토큰에서 고정 길이로 변경) — 줄 전체 손실 제외 그대로.
- `answer_spans` = {lines, object_speed(=speed), speed, na}, `loss_mask_spans` = na ∪ (미래 불완전 → Intent·Action 줄) ∪ (이전 프레임 없음 → 속력 숫자), 정렬·병합. `sft_common.tokenize_answer` 가 `loss_mask_spans` 를 그대로 존중하도록 최소 수정(v1 레코드는 필드가 없어 불변). `max_answer_tokens`/`max_new_tokens` 기본 512 → 768.
- 교사 SFT 지시문(`build_instruction`)은 v2-sensor 문법(8 칸 필드 순서·고정폭 숫자·`<none>/<na>/<_>` 채움·진행방위·1/2/3 s 변위·TTC·단면 0/10/20 m·`<free>`·Intent/Action `<na><_>`)을 설명하도록 교체(결정 1, 2026-10-15). 지시문 706 토큰. 교사 SFT 프롬프트 길이(system=qwen_default, 명령·자차 값/결측 무관 고정): **12 캠 2,955 / 6 캠 1,863**, + 정답 547 = **3,502 / 2,410**(+`<|im_end|>` 1). `--system data` 면 각 +76. 학생 캐시 지시문(`tools/qwen_cache.py` INSTRUCTION)·명령·자차 줄·캐시 L 1,200 은 무변경.

```bash
$PY tools/vla_teacher/gen_teacher_gt_v2.py --explore_ttc     # TTC_GAP 근거(train, 수 분, 8 프로세스)
$PY tools/vla_teacher/gen_teacher_gt_v2.py --force           # 재생성 ~13 분(train ~11 + val ~2.2)
$PY tools/vla_teacher/check_teacher_gt_v2.py                  # + 고정 길이 전수·수치 정합·TTC/지도 분포
```

## 수정(2026-10-16): 2-fold 장면 분할 — 교사 SFT 교차 적합

목적: 학생이 받는 교사 정답이 **그 장면을 학습한 교사**에게서 나오면, 교사가 외운 정답(실제보다 정확한 답)이 섞여 학생 학습 데이터와 평가 때 데이터의 질이 달라진다. 그래서 TRAIN 장면을 겹치지 않는 두 반쪽 A·B 로 나눠 교사를 두 개 학습하고, 각 교사는 **자기가 학습하지 않은 반쪽**의 정답만 만든다.

| 교사 | SFT 학습 | 체크포인트 선택 | 정답 생성 |
|---|---|---|---|
| teacher_A (`--fold A`) | train.A − holdout.A (325 장면 / 13,055 표본) | holdout.A (train.A 안 25 장면 / 1,007) | train.B + val.B |
| teacher_B (`--fold B`) | train.B − holdout.B (325 장면 / 13,064 표본) | holdout.B (train.B 안 25 장면 / 1,004) | train.A + val.A |

- 분할은 장면 단위(한 장면의 모든 프레임은 한쪽에만), 결정적(seed 0, 장면 토큰 정렬 → seed 섞기 → 표본 수 내림차순 → 표본이 적은 쪽에 배정). train A 350 장면/14,062 표본, B 350/14,068(차이 6); val A 75/3,010, B 75/3,009(차이 1).
- val 은 두 교사 모두 학습하지 않았지만, 학생 학습 때와 같은 '두 교사 반반' 구성이 되도록 반씩 생성한다(teacher_A → val.B, teacher_B → val.A).
- holdout.A/B 장면은 각 교사의 학습에서 빠지지만 정답은 다른 교사가 만든다(모든 train·val 장면이 정확히 한 교사에게 배정 — 검사 `generation_covers_all_once`).
- 분할 파일 `data/vla_teacher/v2/fold_split.json`: 장면 목록·표본 수·목록별 sha256·원본 jsonl sha256·배정·전체 sha256. 기존 파일은 `--force` 없이 덮어쓰지 않는다. `train_sft.py --fold A|B` 는 파일 해시와 현재 데이터의 장면·표본 수가 정확히 일치할 때만 진행한다.
- `--fold none`(기본) = 이전 동작 그대로(전체 train − 고정 50 장면, 출력 `sft_default`; `check_fold_split.py` 가 표본 목록·순서 동일 확인).

```bash
export CUDA_VISIBLE_DEVICES="" OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=.
PY=/opt/anaconda3/envs/hipad_bw/bin/python
$PY tools/vla_teacher/make_fold_split.py              # 분할 생성(이미 있으면 거부, --force 로 덮어씀)
$PY tools/vla_teacher/check_fold_split.py             # 24 검사 → results/vla_teacher/check_fold_split.json
$PY tools/vla_teacher/train_sft.py --fold A --dry_run # 데이터 구성·step 수만 출력(모델·출력 폴더 안 만듦)

# 실제 학습(GPU1 전용, 사용자 승인 후) — 출력 기본 results/vla_teacher/sft_fold{A,B}
CUDA_VISIBLE_DEVICES=1 $PY tools/vla_teacher/train_sft.py --fold A
CUDA_VISIBLE_DEVICES=1 $PY tools/vla_teacher/train_sft.py --fold B
```
batch 8 × accum 2, 2 에폭 기준 fold A 1,632 step / fold B 1,634 step(전체 train 학습 3,266 step 의 절반).

## 수정(2026-10-16): SFT 발산 원인 진단 — fp32 어텐션 커널을 math 로 고정

**원인**: fp32 학습에서 `scaled_dot_product_attention`(SDPA)은 flash·cuDNN 이 fp32 를 지원하지 않아 *memory-efficient* 커널로 간다. 이 GPU(RTX PRO 6000, sm_120)·torch 2.7.1+cu128 에서 그 커널은 fp32 정밀도가 아니다(float64 기준: 작은 로짓에서도 출력 상대오차 ~3e-3, 로짓이 수백이면 출력 오차 O(1)·역전파 기울기 오차 1e5~1e14). Qwen2-VL-2B 0층은 k_proj 편향이 최대 432(노름 1,112, 1층은 27)라 |q||k|/√d ≈ 29,190(1층 168)으로 로짓이 매우 크다 → 0층 어텐션의 역전파가 엉터리 기울기를 내고(0층 q/v LoRA·새 토큰 입력 임베딩 행), 학습으로 값이 조금씩 바뀔수록 커져 결국 비유한값이 된다. 같은 커널의 순전파도 틀려 **학습 전 모델의 손실부터 3.80(틀린 커널) vs 2.31(정확한 커널)** 이었다.

**수정**: `sft_common.exact_fp32_attention()` — fp32·CUDA 이면 SDPA 를 math 커널로 고정(프로세스 전역, gradient checkpointing 재계산 포함) + 시작 때 float64 기준 자기검사(큰 로짓 합성 입력, 상대오차 8.6e-5; 틀린 커널이면 2.65e3 으로 실패해 즉시 중단). `train_sft.py --sdpa math`(기본) / `default`(재현·진단용). `eval_teacher.py` 도 sdpa+fp32 이면 같은 고정(bf16 은 변화 없음). 학습률·LoRA·토큰 초기화·손실 등 다른 설정은 그대로(원인이 아니었음).

**검증(fold A, 원래 설정, 학습률 스케줄은 전체 1,632 step 기준 그대로, 110 step)**: clip 전 기울기 노름 step 1 14.6 → step 30 1.48 → step 41–110 0.6~1.5(최대 1.46), 손실 2.31 → 0.79(30) → 0.61(50) → 0.57(91–100 평균). 원래(틀린 커널)는 같은 구간 기울기 노름 200~800 이 반복되다 step 32 부터 4e3 → 1e14 → step 50(원 실행 64) 비유한. 비용: 최대 VRAM 22.3 → 32.5 GB, 속도 39 s/step(틀린 커널 33~45 s/step, GPU1 공유 상태라 차이 범위 안).

진단 도구(로그 전용, 학습 계산 불변):
- `train_sft.py --diag` — 매 step: LoRA A/B·새 토큰 행 그룹별 clip 전 기울기 노름, 가장 큰 LoRA 텐서·토큰 행, 정답 위치 최대 |로짓|(기존/새 토큰), 최종 은닉 노름, 새/기존 행 노름·초기값 대비 이동, LoRA ΔW 최대, 정답 구간별 CE(객체 토큰/숫자, 지도 토큰/숫자, 의도, 행동, 종료). `--stop_at_step N` = 스케줄을 바꾸지 않고 N step 에서 멈춤.
- `diag_grad_probe.py` — 저장 상태에서 특정 step 배치를 표본별로 커널(default/math)을 바꿔 통과시켜 층·위치별 기울기, 0층 로짓 상한, 유한차분 대조.

```bash
cd /workspace/src/HiP-AD_refactor
export OPENBLAS_CORETYPE=SkylakeX PYTHONPATH=. CUDA_VISIBLE_DEVICES=1
PY=/opt/anaconda3/envs/hipad_bw/bin/python
$PY tools/vla_teacher/train_sft.py --fold A     # 출력 results/vla_teacher/sft_foldA (--sdpa math 가 기본)
$PY tools/vla_teacher/train_sft.py --fold B
# 로그에 '[attn] sdpa=math(fp32 exact), self-check rel_err=…' 줄이 있어야 한다
```
주의: `results/vla_teacher/sft_foldA/` 에는 발산한 첫 실행(64 step)의 `train_log.jsonl` 이 남아 있고 로그는 이어 쓰기(append)라, 다시 돌리기 전에 그 폴더를 기록용 이름으로 옮기거나(사용자 판단) `--out` 으로 새 폴더를 준다. `run_sft_2fold.sh` 는 그대로 쓰면 된다(수정이 기본값).
