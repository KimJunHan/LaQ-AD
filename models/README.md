# models/ — 모델 소스

수정(2026-08-07): `mmdet3d_plugin/models/` 를 저장소 최상위 `models/` 로 올렸다.
학습·평가에 실제로 쓰이는 모델 코드를 한 곳에서 보게 하려는 목적.

## 구조와 import 규칙 (중요)

```
HiP-AD/
├── models/                        ← 실체 (여기를 편집한다)
└── mmdet3d_plugin/
    └── models -> ../models        ← 심볼릭 링크
```

**물리적으로 옮기되 패키지 경로는 그대로 두었다.** 이유:

- `models/blocks.py` 와 `models/sparse_detector.py` 가 `from ..ops import ...` 로 **상위 패키지
  (`mmdet3d_plugin.ops`, deformable_aggregation CUDA 커널)** 를 참조한다. `models` 를 독립
  최상위 패키지로 만들면 이 상대 import 가 전부 깨진다.
- config 의 `plugin_dir = "projects/mmdet3d_plugin/"` 와 `mmdet3d_plugin/__init__.py` 의
  `from .models import *` 가 mmcv 레지스트리 등록 경로다. 이 경로가 바뀌면 모든 config 가 죽는다.

심볼릭 링크는 저장소가 이미 쓰는 방식이다 (`projects/configs -> ../configs`,
`projects/mmdet3d_plugin -> ../mmdet3d_plugin`). Python 은 링크를 통해도 `mmdet3d_plugin.models`
를 하위 패키지로 인식하므로 `from ..ops` / `from .blocks` 가 그대로 동작한다.

→ **편집은 `models/` 에서 하고, import 경로는 계속 `projects.mmdet3d_plugin.models.*` 를 쓴다.**

## 현재 학습/평가에 쓰이는 소스 (nuScenes stage1/stage2)

진입점부터 따라가는 순서:

| 파일 | 역할 | config 의 `type` |
|---|---|---|
| `sparse_detector.py` | 최상위 모델 (backbone→FPN→head) | `SparseDetector` |
| `sparse_head.py` | 디코더 래핑 + 태스크별 loss 분배 | `SparseHead` |
| `sparse_onedecoder.py` | **단일 통합 디코더** — `operation_order` 로 동작 결정 | `SparseOneDecoder` |
| `instance_bank.py` | 프레임 간 anchor/query 시간적 메모리 | `InstanceBank` |
| `blocks.py` | `DeformableFeatureAggregation`, `AsymmetricFFN`, `DenseDepthNet` | 동명 |
| `attention.py` | flash attention. **B2D NaN 의 원인이었던 파일** (fp16 하드코딩 → bf16 수정) | `MultiheadFlashAttention` |
| `separate_attn.py` | 태스크 분리 어텐션 | `SeparateAttention`, `TemporalSeparateAttention` |
| `base_target.py` | denoising 포함 타깃 할당 베이스 | — |
| `grid_mask.py` | 입력 augmentation | — |
| `utils.py` | 공용 유틸 | — |

태스크별 서브패키지 (각각 `blocks.py`=인코더/refine, `decoder.py`=출력 디코딩, `target.py`=매칭, `loss*.py`):

| 폴더 | 사용 단계 |
|---|---|
| `det/` | stage1 ✅ / stage2 ✅ |
| `map/` | stage1 ✅ / stage2 ✅ |
| `motion/` | stage1 ❌ / **stage2 ✅** (`task_select` 에 motion 추가) |
| `plan/`, `ego/` | nuScenes stage1/2 에서는 **미사용** (`task_select` 에 없음). B2D config 는 사용 |

> tracking(AMOTA)은 별도 모델 코드가 없다 — `instance_bank.py` 가 유지하는 `instance_ids`
> (`sparse_onedecoder.with_instance_id=True`)를 track ID 로 쓰는 **후처리 평가**다.

## 현재 학습에 쓰이지 않는 소스

지우지 않고 남겨둔 것들. 지금 config 로는 **한 줄도 실행되지 않는다.**

| 파일/폴더 | 무엇 | 왜 비활성인가 |
|---|---|---|
| `qwen_token_attn.py` | Qwen2-VL 토큰 ↔ query cross-attn (VLA-v1) | `operation_order` 에 `qwen_attn` op 이 없음 |
| `qwen_traj_generator.py` | Qwen2-VL 궤적 생성 (VLA-v0) | config 에서 미참조 |
| `residual_rl/` | Residual RL 안전제어 (PPO, kinematic env) | 별도 트랙. `tools/residual_rl/` 로 직접 실행 |
| `optimizer_dedup.py` | VLA 학습용 파라미터 중복 제거 | VLA config 전용 |

즉 지금 도는 학습은 **VLA 없는 순수 HiP-AD 베이스라인**이며, HiP-AD 논문 Table 4 재현이 목적이다.

## 계보

원본 HiP-AD (ICCV 2025, `nullmax-vision/HiP-AD`) → 그 부모 SparseDrive (`swc-17/SparseDrive`).
저장소를 새로 파지 않고 **HiP-AD 를 in-place 수정**하는 방식이라, 로컬 변경에는 전부
`# 수정:` 주석이 붙어 있다.

```bash
grep -rn "수정:" models/ configs/     # 로컬 변경 전수 확인
```
