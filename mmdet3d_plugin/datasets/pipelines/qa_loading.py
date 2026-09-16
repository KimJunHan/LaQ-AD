#수정: VLA-v0 — DriveLM-nuScenes / Bench2Drive-VL Q/A 로딩 + Qwen2-VL 입력 구성 pipeline transforms.
#      두 데이터셋 schema 차이 (behavior/behaviour, 객체 ref 포맷, 단일 JSON vs route 디렉토리) 흡수.
#      LoadQwenInput 은 transformers >= 4.45 + Qwen2-VL 가중치 없으면 stub 모드로 placeholder 생성.
import json
import os
import os.path as osp
import re
import warnings

import numpy as np
from mmdet.datasets.builder import PIPELINES


# DriveLM-nuScenes 객체 ref: "<c1,CAM_FRONT,258.3,442.5>"
_DRIVELM_NUS_REF_RE = re.compile(r"<c\d+,CAM_[A-Z_]+,[\d.]+,[\d.]+>")
# Bench2Drive-VL 객체 ref: "<c45718<CAM_FRONT,799.8,544.5>>"
_B2D_VL_REF_RE = re.compile(r"<c\d+<CAM_[A-Z_]+,[\d.]+,[\d.]+>>")

_QA_CATEGORY_ALIASES = {"behavior": ("behavior", "behaviour"), "behaviour": ("behaviour", "behavior")}


def _resolve_qa_categories(qa_dict, requested):
    """B2D-VL 의 'behaviour' 와 DriveLM-nuScenes 의 'behavior' 처리."""
    resolved = {}
    for cat in requested:
        aliases = _QA_CATEGORY_ALIASES.get(cat, (cat,))
        for alias in aliases:
            if alias in qa_dict:
                resolved[cat] = qa_dict[alias]
                break
        else:
            resolved[cat] = []
    return resolved


@PIPELINES.register_module()
class LoadDriveLMQA(object):
    """frame_token 으로 QA chain 을 로드해 results 에 주입.

    nuScenes 와 Bench2Drive-VL 의 schema 차이를 흡수해 단일 dict 포맷으로 정규화한다:
      results['qa_chain'] = {
          'perception':  [{'Q': str, 'A': str, ...}, ...],
          'prediction':  [...],
          'behavior':    [...],
          'planning':    [...],
      }
      results['qa_meta'] = {'source': 'nuscenes'|'bench2drive', 'frame_token': str, 'scene_token': str|None}

    Args:
        qa_paths (dict): {'nuscenes': '/path/v1_0_train_nus.json',
                          'bench2drive': '/path/to/bench2drive_vl_base'}
        qa_categories (list[str]): 추출할 카테고리 — config 의 qwen_qa_categories.
        dataset_kind (str|None): 'nuscenes' 또는 'bench2drive'. None 이면 results 에서 추론.
        missing_ok (bool): QA 가 없는 frame 도 통과시킴 (빈 리스트). default True (부분 다운로드 대응).
    """

    def __init__(self, qa_paths, qa_categories, dataset_kind=None, missing_ok=True):
        self.qa_paths = dict(qa_paths)
        self.qa_categories = list(qa_categories)
        self.dataset_kind = dataset_kind
        self.missing_ok = missing_ok
        self._nus_cache = None  # 단일 JSON 통째 캐시 (193 MB → 메모리 1회 로드)

    def _load_nus(self):
        if self._nus_cache is not None:
            return self._nus_cache
        path = self.qa_paths.get("nuscenes")
        if not path or not osp.exists(path):
            warnings.warn(f"DriveLM-nuScenes QA 파일 없음: {path}")
            self._nus_cache = {}
            return self._nus_cache
        with open(path) as f:
            self._nus_cache = json.load(f)
        return self._nus_cache

    def _lookup_nus(self, scene_token, frame_token):
        data = self._load_nus()
        scene = data.get(scene_token)
        if scene is None:
            return None
        return scene.get("key_frames", {}).get(frame_token)

    def _lookup_b2d(self, scene_token, frame_idx):
        # B2D-VL 경로 규약: <qa_root>/<scene_token>/<frame_idx:05d>.json
        # scene_token = info['folder'] = e.g. "AccidentTwoWays_Town12_Route1444_Weather0"
        qa_root = self.qa_paths.get("bench2drive")
        if not qa_root:
            return None
        fpath = osp.join(qa_root, scene_token, f"{frame_idx:05d}.json")
        if not osp.exists(fpath):
            return None
        with open(fpath) as f:
            return json.load(f)

    def _infer_kind(self, results):
        if self.dataset_kind:
            return self.dataset_kind
        # nuScenes 는 scene_token 이 hex 32자 token, B2D 는 'Scenario_TownXX_RouteY_WeatherZ'.
        st = results.get("scene_token", "")
        return "bench2drive" if any(t in st for t in ("Town", "Route", "Weather")) else "nuscenes"

    def __call__(self, results):
        kind = self._infer_kind(results)
        scene_token = results.get("scene_token")
        frame_token = results.get("frame_token") or results.get("token")
        frame_idx = results.get("frame_idx")

        record = None
        if kind == "nuscenes":
            record = self._lookup_nus(scene_token, frame_token)
        elif kind == "bench2drive":
            record = self._lookup_b2d(scene_token, frame_idx)

        if record is None:
            if not self.missing_ok:
                raise FileNotFoundError(f"QA not found: kind={kind}, scene={scene_token}, "
                                        f"frame_token={frame_token}, frame_idx={frame_idx}")
            results["qa_chain"] = {cat: [] for cat in self.qa_categories}
            results["qa_meta"] = dict(source=kind, frame_token=frame_token,
                                      scene_token=scene_token, missing=True)
            return results

        qa = record.get("QA", {})
        results["qa_chain"] = _resolve_qa_categories(qa, self.qa_categories)
        results["qa_meta"] = dict(
            source=kind, frame_token=frame_token, scene_token=scene_token,
            scene_description=record.get("scene_description"),
            key_object_infos=record.get("key_object_infos", {}),
            image_paths=record.get("image_paths"),
            missing=False,
        )
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(categories={self.qa_categories})"


@PIPELINES.register_module()
class LoadQwenInput(object):
    """Qwen2-VL processor 로 6-cam 이미지 + DriveLM instruction → pixel_values/input_ids.

    transformers >= 4.45 + Qwen2-VL processor 가 import 가능하면 실제 토크나이즈.
    아니면 stub 모드: 형태(shape)만 맞는 placeholder tensor 를 생성해 파이프라인 검증만 가능.

    Args:
        qwen_model_id (str): HF model id, 예: 'Qwen/Qwen2-VL-2B-Instruct'.
        camera_order (list[str]): 6-cam 의 순서. config 의 qwen_camera_order.
        max_pixels_per_image (int): smart_resize 의 max_pixels.
        layout (str): 'sequential' 만 지원 (v0). grid 는 후속.
        system_prompt (str|None): 시스템 prompt. None 이면 기본값.
        instruction_template (str|None): user prompt 템플릿. None 이면 기본값.
    """

    DEFAULT_SYSTEM = (
        "You are an autonomous driving planner. Given 6 surround-view camera images and a "
        "question, reason about the scene and produce a future ego trajectory."
    )
    DEFAULT_INSTRUCTION_TAIL = (
        "\nNow plan the next 3 seconds. Output the ego trajectory as 6 waypoints "
        "(T=0.5s spacing) in ego coordinates (forward=+x, left=+y, meters):\n"
        "Trajectory: [(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6)]"
    )

    def __init__(self, qwen_model_id, camera_order, max_pixels_per_image,
                 layout="sequential", system_prompt=None, instruction_template=None,
                 min_pixels_per_image=None):
        assert layout == "sequential", "v0 는 sequential 만 지원."
        self.qwen_model_id = qwen_model_id
        self.camera_order = list(camera_order)
        self.max_pixels_per_image = int(max_pixels_per_image)
        self.min_pixels_per_image = int(min_pixels_per_image) if min_pixels_per_image else max(4 * 28 * 28, self.max_pixels_per_image // 4)
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM
        self.instruction_tail = instruction_template or self.DEFAULT_INSTRUCTION_TAIL
        self._processor = None
        self._stub = False
        self._try_load_processor()

    def _try_load_processor(self):
        try:
            from transformers import AutoProcessor  # type: ignore
            self._processor = AutoProcessor.from_pretrained(
                self.qwen_model_id,
                min_pixels=self.min_pixels_per_image,
                max_pixels=self.max_pixels_per_image,
            )
        except Exception as e:
            warnings.warn(f"LoadQwenInput: Qwen2-VL processor 로드 실패 ({type(e).__name__}: {e}). "
                          f"stub 모드로 전환 — placeholder tensor 생성.")
            self._processor = None
            self._stub = True

    @staticmethod
    def _smart_resize(h, w, factor=28, min_pixels=56 * 56, max_pixels=12845056):
        import math
        h_bar = round(h / factor) * factor
        w_bar = round(w / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((h * w) / max_pixels)
            h_bar = math.floor(h / beta / factor) * factor
            w_bar = math.floor(w / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (h * w))
            h_bar = math.ceil(h * beta / factor) * factor
            w_bar = math.ceil(w * beta / factor) * factor
        return h_bar, w_bar

    def _build_chain_text(self, qa_chain):
        """DriveLM QA chain 을 Qwen prompt 로 직렬화."""
        lines = []
        for cat in ("perception", "prediction", "behavior", "planning"):
            entries = qa_chain.get(cat, [])
            if not entries:
                continue
            lines.append(f"\n[{cat.upper()}]")
            for e in entries[:8]:  # 카테고리당 최대 8개 — 토큰 폭주 방지
                q = (e.get("Q") or "").strip()
                a = (e.get("A") or "").strip()
                if q:
                    lines.append(f"Q: {q}")
                if a:
                    lines.append(f"A: {a}")
        return "\n".join(lines)

    def _build_camera_header(self):
        lines = [
            f"The following {len(self.camera_order)} images are surround-view cameras of an ego vehicle, in this order:"
        ]
        for i, cam in enumerate(self.camera_order, 1):
            lines.append(f"{i}. {cam}")
        lines.append("Object references use the format <c, CAM_NAME, x, y>.")
        return "\n".join(lines)

    def __call__(self, results):
        # 1) img_filename(데이터셋 카메라 순서) → camera_order 순으로 재정렬
        img_filenames = results.get("img_filename") or []
        cams_in_results = results.get("camera_names")  # 데이터셋이 제공하면 사용
        if cams_in_results and len(cams_in_results) == len(img_filenames):
            cam_to_path = dict(zip(cams_in_results, img_filenames))
        else:
            # 데이터셋이 cam 이름 안 줬으면 camera_order 와 같은 순서라 가정
            cam_to_path = dict(zip(self.camera_order, img_filenames))
        ordered_paths = [cam_to_path.get(cam) for cam in self.camera_order]

        # 2) prompt 구성
        chain_text = self._build_chain_text(results.get("qa_chain") or {})
        prompt_text = (
            self._build_camera_header()
            + "\n\n" + chain_text
            + self.instruction_tail
        )

        if self._stub or self._processor is None:
            # stub 모드: shape 만 맞는 placeholder. Qwen 미설치 환경에서 파이프라인 import 검증용.
            n_imgs = len(self.camera_order)
            # smart_resize 기준 visual token 수 추정 (per image)
            h_bar, w_bar = self._smart_resize(900, 1600, max_pixels=self.max_pixels_per_image)
            results["qwen_pixel_values"] = np.zeros((n_imgs, 3, h_bar, w_bar), dtype=np.float32)
            results["qwen_image_grid_thw"] = np.array(
                [[1, h_bar // 14, w_bar // 14]] * n_imgs, dtype=np.int64
            )
            results["qwen_input_ids"] = np.zeros((1, 64), dtype=np.int64)  # placeholder
            results["qwen_attention_mask"] = np.ones((1, 64), dtype=np.int64)
            results["qwen_prompt_text"] = prompt_text
            results["qwen_stub"] = True
            return results

        # 실제 Qwen processor 경로 (transformers >= 4.45)
        from PIL import Image
        images = []
        for p in ordered_paths:
            if p is None or not osp.exists(p):
                images.append(Image.new("RGB", (1600, 900), (0, 0, 0)))
            else:
                images.append(Image.open(p).convert("RGB"))

        # Qwen chat template: 시스템 + user(이미지 6장 + prompt_text)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content":
                [{"type": "image", "image": im} for im in images]
                + [{"type": "text", "text": prompt_text}],
            },
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=images, return_tensors="np", padding=True)

        results["qwen_pixel_values"] = inputs["pixel_values"]
        results["qwen_image_grid_thw"] = inputs.get("image_grid_thw")
        results["qwen_input_ids"] = inputs["input_ids"]
        results["qwen_attention_mask"] = inputs["attention_mask"]
        results["qwen_prompt_text"] = text
        results["qwen_stub"] = False
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(model={self.qwen_model_id}, "
                f"max_pixels={self.max_pixels_per_image}, stub={self._stub})")
