# 수정(2026-11-04): 학생(LaQ-AD 2단계) 추론 지연 단축 — CUDA Graph / torch.compile 래퍼. 기존 파일은 한 줄도 바꾸지 않는다.
#
#   왜 느린가(tools/latency/find_syncs.py, results/latency/syncs_*.json):
#     프레임당 커널 ~4,700 개를 파이썬이 하나씩 띄우고(GPU 실행 38 ms vs 프레임 95 ms), 그 사이 호스트↔GPU 동기화가 ~171 회
#     (지도 후처리 벡터별 .cpu() 100 회, 뱅크의 numpy→GPU 복사, 파이썬 리스트 색인의 H2D 복사, .item()/bool() 검사 등) 있다.
#   무엇을 하나:
#     (1) deformable_aggregation 을 torch.library custom op(laq_latency::daf, fake 구현 포함)로 등록한다. 원본 확장은 커널을
#         legacy 기본 스트림에 올려 CUDA Graph 캡처(비기본 스트림)에 들어가지 않으므로, 같은 커널 소스를 #include 해 "현재 스트림"
#         에 올리는 작은 확장(tools/latency/daf_stream)을 JIT 빌드해 쓴다(커널 코드·격자 동일 → 결과 비트 동일).
#     (2) 첫 프레임(뱅크가 빈 상태)은 원본 그대로 eager 로 돌린다. 둘째 프레임부터는 모양이 고정이므로, 동기화를 일으키는 몇 곳만
#         값이 같은 동기화 없는 구현으로 바꿔(아래 _install_patches, 각 항목에 원본 위치·동치 근거) 프레임 전체
#         (백본+넥+통합 디코더 6 층+뱅크 갱신 [+후처리 텐서 연산])를 CUDA Graph 하나로 캡처하고 매 프레임 replay 한다.
#         시간 뱅크 상태는 고정 버퍼에 두고 그래프 끝에서 새 상태를 같은 버퍼에 복사한다(eager 와 같은 상태 전이).
#         매 프레임 바뀌는 호스트 값(T_global 행렬 곱, 장면 동일 여부, 타임스탬프 등)은 replay 전에 고정 입력 버퍼에 복사한다.
#     (3) 선택: torch.compile(mode="max-autotune-no-cudagraphs" 등)로 하위 모듈을 컴파일한 뒤 같은 그래프로 캡처한다
#         (커널 융합 → 부동소수 결과가 비트 동일하지 않을 수 있다; equiv.py 로 차이를 잰다).
#   동작 조건(어긋나면 조용히 넘기지 않고 즉시 실패): batch 1, qwen_mask 전부 1(패딩 없음), qwen_type 범위 안,
#     reset_bank_on_scene_change=False, 개방루프(with_close_loop=False), det/map 후처리 score_threshold=None.
#   사용: fs = FastStudent(model, post_graph=True); results = fs(data)   # data = DataLoader 표본(CPU), 반환 형식은 원본과 같다.
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

# ----------------------------------------------------------------------------------------------------------------------
# (1) deformable_aggregation: 현재 스트림 확장 + torch.library custom op
# ----------------------------------------------------------------------------------------------------------------------
_EXT = None


def _daf_ext():
    """tools/latency/daf_stream 에서 미리 빌드한 확장(빌드: cd tools/latency/daf_stream && python setup.py build_ext --inplace).
    원본 ops/setup.py 와 같은 nvcc 매크로, sm_90, conda x86_64-conda-linux-gnu 컴파일러(원본 build.ninja 의 -ccbin 과 같음)."""
    global _EXT
    if _EXT is None:
        d = os.path.join(_HERE, "daf_stream")
        if d not in sys.path:
            sys.path.insert(0, d)
        import laq_daf_stream_ext
        _EXT = laq_daf_stream_ext
    return _EXT


class _DetDAF:
    """동치 검사 전용: atomicAdd 없는 결정적 deformable aggregation(같은 수식: 위치 loc*size-0.5 의 쌍선형 표본, 영 패딩,
    loc∉(0,1) 이면 0, 채널 c 의 가중치 = 그룹 c//(C/G)). 원본 CUDA 커널은 출력에 atomicAdd 로 더해 합산 순서가 실행마다 달라
    eager 끼리도 비트 동일하지 않다(시간 뱅크 top-k 를 거쳐 커진다). 원본·빠른 경로 둘 다 이것으로 바꾸면 래퍼 자체가 값을
    바꾸는지 잡음 없이(비트 단위로) 확인할 수 있다. 지연 측정·전체 평가에는 쓰지 않는다."""
    on = False
    _cache = {}

    @classmethod
    def run(cls, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights):
        key = (spatial_shape.data_ptr(), scale_start_index.data_ptr())
        if key not in cls._cache:
            assert not torch.cuda.is_current_stream_capturing()
            cls._cache[key] = (spatial_shape.cpu().tolist(), scale_start_index.cpu().tolist(), spatial_shape, scale_start_index)
        ss, ssi, _, _ = cls._cache[key]
        mc_ms_feat = mc_ms_feat.float()
        loc = sampling_location.float()
        weights = weights.float()
        bs, nf, C = mc_ms_feat.shape
        G = weights.shape[-1]
        valid = (loc[..., 0] > 0) & (loc[..., 0] < 1) & (loc[..., 1] > 0) & (loc[..., 1] < 1)  # (bs,A,P,cams)
        out = None
        for cam in range(len(ss)):
            g = loc[:, :, :, cam] * 2 - 1
            for sc in range(len(ss[cam])):
                H, W = ss[cam][sc]
                st = ssi[cam][sc]
                f = mc_ms_feat[:, st:st + H * W].reshape(bs, H, W, C).permute(0, 3, 1, 2)
                smp = F.grid_sample(f, g, mode="bilinear", padding_mode="zeros", align_corners=False)  # (bs,C,A,P)
                w = (weights[:, :, :, cam, sc] * valid[..., cam, None]).repeat_interleave(C // G, dim=-1)  # (bs,A,P,C)
                term = (smp.permute(0, 2, 3, 1) * w).sum(2)
                out = term if out is None else out + term
        return out


@torch.library.custom_op("laq_latency::daf", mutates_args=())
def daf_op(mc_ms_feat: torch.Tensor, spatial_shape: torch.Tensor, scale_start_index: torch.Tensor,
           sampling_location: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # 원본 DeformableAggregationFunction.forward 와 같은 입력 정규화(contiguous·float/int)
    if _DetDAF.on:
        return _DetDAF.run(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights)
    return _daf_ext().forward(mc_ms_feat.contiguous().float(), spatial_shape.contiguous().int(),
                              scale_start_index.contiguous().int(), sampling_location.contiguous().float(),
                              weights.contiguous().float())


@daf_op.register_fake
def _daf_fake(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights):
    return mc_ms_feat.new_empty((mc_ms_feat.shape[0], sampling_location.shape[1], mc_ms_feat.shape[2]),
                                dtype=torch.float32)


# 패치 활성 플래그: 같은 프로세스의 원본(eager 기준) 모델은 패치 영향을 받지 않도록, 모듈 수준 패치는 이 플래그가 켜진
# 구간(빠른 경로의 예열·캡처·컴파일)에서만 새 구현을 쓴다(replay 중에는 파이썬이 돌지 않는다).
class _Active:
    on = False


def _module_of(obj):
    return sys.modules[type(obj).__module__]


class _FMF:
    """ops/__init__.py feature_maps_format 의 동기화 없는 판. 원본은 매 프레임 spatial_shape 를 torch.tensor 로 새로 만들고
    (H2D), inverse 에서 .cpu().numpy().tolist()·torch.all(...) 로 호스트 동기화를 3 회 한다. 모양은 상수이므로 처음 한 번
    원본과 같은 식으로 만든 텐서·리스트를 캐시한다. 텐서 연산(reshape/cat/permute/split/unflatten)은 원본과 같다."""

    def __init__(self, orig):
        self.orig = orig
        self.fwd_cache = {}
        self.inv_cache = {}

    def __call__(self, feature_maps, inverse=False):
        if not _Active.on:
            return self.orig(feature_maps, inverse)
        if inverse:
            return self._inverse(feature_maps)
        if isinstance(feature_maps[0], (list, tuple)):
            return self.orig(feature_maps, inverse)  # 이 모델에서 쓰지 않는 경로(카메라 그룹 여러 개) — 원본 그대로
        bs, num_cams = feature_maps[0].shape[:2]
        col_feats = [torch.reshape(feat, (bs, num_cams, feat.shape[2], -1)) for feat in feature_maps]
        col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
        key = (num_cams, tuple(tuple(f.shape[-2:]) for f in feature_maps), col_feats.device)
        if key not in self.fwd_cache:
            assert not torch.cuda.is_current_stream_capturing(), "feature_maps_format 상수는 캡처 전 예열에서 만들어야 한다"
            _, ss, ssi = self.orig([f[:1, :, :1] for f in feature_maps])  # 원본 식 그대로(작은 슬라이스로 모양만 사용)
            self.fwd_cache[key] = (ss, ssi)
        ss, ssi = self.fwd_cache[key]
        return [col_feats, ss, ssi]

    def _inverse(self, feature_maps):
        col_feats, spatial_shape, scale_start_index = feature_maps
        key = spatial_shape.data_ptr()
        if key not in self.inv_cache:
            assert not torch.cuda.is_current_stream_capturing()
            num_cams = spatial_shape.shape[0]
            split_size = (spatial_shape[..., 0] * spatial_shape[..., 1]).cpu().numpy().tolist()
            cam_split, cam_split_size = [1], [sum(split_size[0])]
            for i in range(num_cams - 1):
                if not torch.all(spatial_shape[i] == spatial_shape[i + 1]):
                    cam_split.append(0)
                    cam_split_size.append(0)
                cam_split[-1] += 1
                cam_split_size[-1] += sum(split_size[i + 1])
            self.inv_cache[key] = (split_size, cam_split, cam_split_size, spatial_shape.cpu().numpy().tolist(),
                                   spatial_shape)  # 마지막 원소: data_ptr 재사용 방지용 참조 유지
        split_size, cam_split, cam_split_size, ss_list, _ = self.inv_cache[key]
        mc_feat = [x.unflatten(1, (cam_split[i], -1)) for i, x in enumerate(col_feats.split(cam_split_size, dim=1))]
        mc_ms_feat = []
        shape_index = 0
        for i, feat in enumerate(mc_feat):
            feat = list(feat.split(split_size[shape_index], dim=2))
            for j, f in enumerate(feat):
                feat[j] = f.unflatten(2, ss_list[shape_index][j])
                feat[j] = feat[j].permute(0, 1, 4, 2, 3)
            mc_ms_feat.append(feat)
            shape_index += cam_split[i]
        return mc_ms_feat


class _DAFDispatch:
    # 진단용 강제 선택(student_test --daf-impl): None = 기본(래퍼 구간은 현재 스트림 확장, 원본 구간은 원본 확장),
    #   "orig" = 어디서나 원본 확장(그래프 캡처 불가 — use_graph=False 변형에서만), "mine" = 어디서나 현재 스트림 확장.
    force = None

    def __init__(self, orig):
        self.orig = orig

    def __call__(self, *args):
        if _DetDAF.on:
            return _DetDAF.run(*args) if not _Active.on else daf_op(*args)
        if _DAFDispatch.force == "orig":
            assert not torch.cuda.is_current_stream_capturing(), "원본 확장은 기본 스트림에 올라가 캡처할 수 없다"
            return self.orig(*args)
        if _DAFDispatch.force == "mine" or _Active.on:
            return daf_op(*args)
        return self.orig(*args)


_GLOBAL_PATCHED = False


def install_global_patches(det):
    """모듈 수준 이름(feature_maps_format, DAF)을 플래그 분기 래퍼로 바꾼다. 원본 모델에는 영향 없음(_Active.on=False)."""
    global _GLOBAL_PATCHED
    if _GLOBAL_PATCHED:
        return
    od = det.head.onedecoder_head
    for mod in (_module_of(det), _module_of(od.ego_instance_bank), _module_of(od.plan_instance_bank)):
        assert hasattr(mod, "feature_maps_format"), mod
        mod.feature_maps_format = _FMF(mod.feature_maps_format)
    agg = _module_of(od.det_deformable[0])
    assert agg.DAF is not None
    agg.DAF = _DAFDispatch(agg.DAF)
    _GLOBAL_PATCHED = True


# ----------------------------------------------------------------------------------------------------------------------
# (2) 인스턴스 패치 — 빠른 경로 모델 인스턴스에만 건다. 각 함수 = 원본 + 동기화 원인만 제거(값 동일).
# ----------------------------------------------------------------------------------------------------------------------
def _box_anchor_projection(anchor, T_src2dst_list, src_timestamp=None, dst_timestamps=None, time_intervals=None):
    """deformable_aggregation/aggregation.py SparseBox3DKeyPointsGenerator.anchor_projection 와 같다.
    차이: yaw[..., [1, 0]] (파이썬 리스트 색인 → 색인 텐서 H2D 복사·동기화) 를 yaw.flip(-1) 로(마지막 축 길이 2 → 같은 원소 순서)."""
    from_box = _box_anchor_projection
    dst_anchors = []
    for i in range(len(T_src2dst_list)):
        vel = anchor[..., 8:]
        vel_dim = vel.shape[-1]
        T_src2dst = torch.unsqueeze(T_src2dst_list[i].to(dtype=anchor.dtype), dim=1)
        center = anchor[..., 0:3]
        if time_intervals is not None:
            time_interval = time_intervals[i]
        elif src_timestamp is not None and dst_timestamps is not None:
            time_interval = (src_timestamp - dst_timestamps[i]).to(dtype=vel.dtype)
        else:
            time_interval = None
        if time_interval is not None:
            translation = vel.transpose(0, -1) * time_interval
            translation = translation.transpose(0, -1)
            center = center - translation
        center = torch.matmul(T_src2dst[..., :3, :3], center[..., None]).squeeze(dim=-1) + T_src2dst[..., :3, 3]
        size = anchor[..., 3:6]
        yaw = torch.matmul(T_src2dst[..., :2, :2], anchor[..., 6:8].flip(-1)[..., None]).squeeze(-1)
        yaw = yaw.flip(-1)
        vel = torch.matmul(T_src2dst[..., :vel_dim, :vel_dim], vel[..., None]).squeeze(-1)
        dst_anchors.append(torch.cat([center, size, yaw, vel], dim=-1))
    del from_box
    return dst_anchors


class FastStudent:
    """첫 프레임은 원본 eager, 이후 프레임은 CUDA Graph replay. 반환은 원본 model(return_loss=False, rescale=True, **data)
    와 같은 형식(list[dict(img_bbox=...)])."""

    def __init__(self, model, post_graph=True, compile_mode=None, compile_targets=("backbone", "neck"),
                 use_graph=True, verbose=True, warmup_stream=None):
        from mmcv.parallel import MMDataParallel
        self.dp = model
        self.det = model.module if isinstance(model, MMDataParallel) else model
        self.od = self.det.head.onedecoder_head
        self.post_graph = post_graph
        self.compile_mode = compile_mode
        self.compile_targets = tuple(compile_targets)
        self.use_graph = use_graph
        self.verbose = verbose
        # 진단용: 예열 스트림 "side"(곁 스트림) | "current"(현재 스트림). None = 캡처하면 side, 아니면 current
        self.warmup_stream = warmup_stream or ("side" if use_graph else "current")
        self.dev = torch.device("cuda", torch.cuda.current_device())
        self.frame = 0
        self.graph = None
        self._built = False
        self.timings = {}
        self._check_supported()
        install_global_patches(self.det)

    # ------------------------------------------------------------------ 조건 검사
    def _check_supported(self):
        od, det = self.od, self.det
        if det.reset_bank_on_scene_change:
            raise NotImplementedError("reset_bank_on_scene_change=True 는 지원하지 않는다")
        if od.with_close_loop or od.training:
            raise NotImplementedError("개방루프 추론만 지원(스트림 그룹은 학습 전용 — 추론에서는 _stream_group_enter 가 아무것도 안 한다)")
        if det.bev_encoders is not None:
            raise NotImplementedError("bev_encoders(B 융합) 미지원")
        if od.det_decoder.score_threshold is not None or od.map_decoder.score_threshold is not None:
            raise NotImplementedError("후처리 score_threshold 가 있으면 출력 개수가 데이터 의존이다")
        if od.qwen_token_attn is None or od.qwen_token_attn.route_type_ids is not None:
            raise NotImplementedError("route_segs 경로 미지원")
        if od.motion_decoder.score_threshold is not None:
            raise NotImplementedError("motion score_threshold 미지원")
        assert od.query_select == ["det", "map", "plan", "ego"] and od.task_select == ["det", "map", "plan", "ego", "motion"]
        assert od.plan_diffusion is not None and od.plan_diffusion_uses_map

    # ------------------------------------------------------------------ 호스트 측 입력 정리
    @staticmethod
    def _unwrap(data):
        """DataLoader 표본 → (img_metas list[dict], 텐서 dict). MMDataParallel.scatter 와 같은 값(DataContainer 풀기)."""
        from mmcv.parallel import DataContainer
        out = {}
        for k, v in data.items():
            if isinstance(v, DataContainer):
                v = v.data[0]
            out[k] = v
        metas = out.pop("img_metas")
        if isinstance(metas, list) and len(metas) == 1 and isinstance(metas[0], list):
            metas = metas[0]
        return metas, out

    def _host_checks(self, t):
        if t["img"].shape[0] != 1:
            raise NotImplementedError("batch 1 만 지원")
        qm = t["qwen_mask"]
        if not bool(qm.bool().all()):
            # 원본은 이때 key_padding_mask(unpad, 데이터 의존 모양) 경로로 간다 — 그래프로 잡을 수 없다. 조용히 넘기지 않는다.
            raise NotImplementedError("qwen_mask 에 패딩이 있다 — 고정 모양 그래프 경로 불가")
        qt = t["qwen_type"]
        n = self.od.qwen_token_attn.num_token_types
        if bool(((qt < 0) | (qt >= n)).any()):  # 원본 project_kv 와 같은 검사(CPU 에서)
            raise ValueError(f"QwenTokenAttn: qwen_type 범위 [{int(qt.min())}, {int(qt.max())}] 이 [0, {n}) 밖이다")

    # ------------------------------------------------------------------ 공개 호출
    def __call__(self, data):
        if self.frame == 0:
            out = self._eager_first(data)
        else:
            if not self._built:
                self._build(data)
                self._built = True
            out = self._run(data)
        self.frame += 1
        return out

    def _eager_first(self, data):
        with torch.no_grad():
            out = self.dp(return_loss=False, rescale=True, **data)
        metas, _ = self._unwrap(data)
        self.prev_metas = metas
        self.last_model_outs = None
        return out

    # ------------------------------------------------------------------ 패치 설치(빠른 경로 인스턴스)
    def _install_patches(self):
        od = self.od
        db, mb, eb, pb = od.det_instance_bank, od.map_instance_bank, od.ego_instance_bank, od.plan_instance_bank
        dev = self.dev
        S = self.S  # 상태·입력 고정 버퍼

        # --- det 뱅크 get: instance_bank.py:86-111. T_temp2cur(numpy→new_tensor, H2D 동기화)·new_tensor(0.5) 를 고정 버퍼로.
        dti_const = {}

        def det_get(bank, batch_size, metas=None):
            instance_feature = torch.tile(bank.instance_feature[None], (batch_size, 1, 1))
            anchor = torch.tile(bank.anchor[None], (batch_size, 1, 1))
            history_time = bank.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)
            bank.mask = torch.abs(time_interval) <= bank.max_time_interval
            bank.cached_anchor = bank.anchor_handler.anchor_projection(
                bank.cached_anchor, [S["T_det"]], time_intervals=[-time_interval])[0]
            key = time_interval.dtype
            if key not in dti_const:
                dti_const[key] = time_interval.new_tensor(bank.default_time_interval)
            time_interval = torch.where(torch.logical_and(time_interval != 0, bank.mask), time_interval, dti_const[key])
            return instance_feature, anchor, bank.cached_feature, bank.cached_anchor, time_interval

        # --- det 뱅크 update: instance_bank.py:113-139. new_tensor(0)/new_tensor(-1)(H2D) → 미리 만든 0 차원 상수(같은 dtype·값).
        upd_const = {}

        def det_update(bank, instance_feature, anchor, confidence):
            if bank.cached_feature is None:
                return instance_feature, anchor
            from_topk = _module_of(bank).topk
            N = bank.num_anchor - bank.num_temp_instances
            confidence = confidence.max(dim=-1).values
            _, (selected_feature, selected_anchor) = from_topk(confidence, N, instance_feature, anchor)
            selected_feature = torch.cat([bank.cached_feature, selected_feature], dim=1)
            selected_anchor = torch.cat([bank.cached_anchor, selected_anchor], dim=1)
            instance_feature = torch.where(bank.mask[:, None, None], selected_feature, instance_feature)
            anchor = torch.where(bank.mask[:, None, None], selected_anchor, anchor)
            if "c0" not in upd_const:
                upd_const["c0"] = bank.confidence.new_tensor(0)
                upd_const["im1"] = bank.instance_id.new_tensor(-1)
            bank.confidence = torch.where(bank.mask[:, None], bank.confidence, upd_const["c0"])
            if bank.instance_id is not None:
                bank.instance_id = torch.where(bank.mask[:, None], bank.instance_id, upd_const["im1"])
            return instance_feature, anchor

        # --- det 뱅크 get_instance_id: instance_bank.py:175-191. arange(num_new)(.item 동기화) + instance_id[where(mask)]=…(nonzero 동기화)
        #     → 행 우선 누적합: k 번째 True 위치에 prev_id+k (torch.where(mask) 순서 = 행 우선과 같다). 정수 연산이라 값 동일.
        def det_get_instance_id(bank, confidence, anchor=None, threshold=None):
            confidence = confidence.max(dim=-1).values.sigmoid()
            instance_id = confidence.new_full(confidence.shape, -1).long()
            if bank.instance_id is not None and bank.instance_id.shape[0] == instance_id.shape[0]:
                instance_id[:, : bank.instance_id.shape[1]] = bank.instance_id
            mask = instance_id < 0
            if threshold is not None:
                mask = mask & (confidence >= threshold)
            flat = mask.reshape(-1).long()
            new_ids = (torch.cumsum(flat, 0) - 1 + bank.prev_id).reshape(mask.shape)
            instance_id = torch.where(mask, new_ids, instance_id)
            bank.prev_id = bank.prev_id + flat.sum()
            bank.update_instance_id(instance_id, confidence)
            return instance_id

        db.get = types.MethodType(det_get, db)
        db.update = types.MethodType(det_update, db)
        db.get_instance_id = types.MethodType(det_get_instance_id, db)
        db.anchor_handler.anchor_projection = _box_anchor_projection

        # --- 지도 뱅크(PlanMapMemoryInstanceBank, num_temp_instances=0): plan_map_memory_bank.py:41-71 + instance_bank.py:107-109.
        #     temp 가 없어 매 프레임 reset 경로(time_interval = new_tensor([0.5]*bs), H2D) → 상수 버퍼. 지도 기억 투영의 T(numpy)·
        #     same_scene(파이썬 리스트→텐서) 은 호스트에서 같은 식으로 계산해 고정 버퍼로 넣는다.
        map_ti = {}

        def map_get(bank, batch_size, metas=None):
            instance_feature = torch.tile(bank.instance_feature[None], (batch_size, 1, 1))
            anchor = torch.tile(bank.anchor[None], (batch_size, 1, 1))
            key = (instance_feature.dtype, batch_size)
            if key not in map_ti:
                map_ti[key] = instance_feature.new_tensor([bank.default_time_interval] * batch_size)
            ts = metas["timestamp"]
            dt = (ts - bank.plan_map_ts).abs()
            valid = dt <= bank.max_time_interval
            valid = valid & S["same_scene"]
            pts = bank.anchor_handler.anchor_projection(bank.plan_map_pts, [S["T_map"]])[0]
            B, M, _ = pts.shape
            scores = bank.plan_map_scores * valid.to(bank.plan_map_scores.dtype)[:, None, None]
            bank.plan_map_memory = dict(pts=pts.reshape(B, M, -1, 2), scores=scores)
            return instance_feature, anchor, None, None, map_ti[key]

        def map_cache(bank, instance_feature, anchor, confidence, metas=None, feature_maps=None):
            # InstanceBank.cache 는 num_temp_instances<=0 이면 바로 반환(원본과 같음). numpy T_global·scene 은 호스트가 관리.
            bank.plan_map_pts = anchor.detach().float()
            bank.plan_map_scores = confidence.detach().float().sigmoid()
            bank.plan_map_ts = metas["timestamp"].detach()

        assert mb.num_temp_instances == 0
        mb.get = types.MethodType(map_get, mb)
        mb.cache = types.MethodType(map_cache, mb)

        # --- 계획 확산(MapAlignedPlanTruncatedDiffusion, plan_diffusion_map.py:203-236 추론 분기)
        #     (a) 고정 시드 생성기 난수 z·r 은 매 프레임 같은 값(B·모양 고정) → 처음 한 번 원본과 같은 호출로 만들어 상수로.
        #     (b) _segments(:102-120): n_keep = int(keep.sum(1).max().item())(동기화·데이터 의존 모양) → 모든 인스턴스를 그대로
        #         두고 keep 을 valid 에 넣는다. 원본은 통과 인스턴스를 원래 순서대로 앞으로 모아 n_keep 개만 남길 뿐이고, 나머지는
        #         valid=False 로 d2=inf / hit=False 가 되어 min·argmin(동률이면 낮은 색인) 결과가 같다. n_keep=0 이면 원본은 None
        #         (잡음 = z, 자르기 없음), 이 판은 has·crossed 가 모두 False 라 torch.where 로 같은 값이 나온다.
        #         sc[..., list(class_ids)](리스트 색인 H2D) → 장치 색인 텐서 index_select(같은 원소).
        pd = od.plan_diffusion
        pd_cache = {}
        cls_idx = {}

        def seg_static(map_memory, class_ids):
            pts = map_memory["pts"].float()
            sc = map_memory["scores"].float()
            ck = tuple(class_ids)
            if ck not in cls_idx:
                cls_idx[ck] = torch.tensor(list(class_ids), dtype=torch.long, device=sc.device)
            inst = sc.index_select(-1, cls_idx[ck]).max(-1).values
            keep = inst >= pd.score_thr
            A, E = pts[:, :, :-1], pts[:, :, 1:] - pts[:, :, :-1]
            valid = keep[..., None] & (E.norm(dim=-1) > 1e-6)
            B = pts.shape[0]
            conf = inst[..., None].expand(-1, -1, A.shape[2])
            return A.reshape(B, -1, 2), E.reshape(B, -1, 2), valid.reshape(B, -1), conf.reshape(B, -1)

        def pd_forward(mod, anchor, map_memory=None):
            assert not mod.training and map_memory is not None
            B, N, D = anchor.shape
            a = (anchor.reshape(B, N, mod.ego_fut_ts, 2) - mod.norm_offset) / mod.norm_scale
            key = (tuple(a.shape), a.dtype)
            if key not in pd_cache:
                assert not torch.cuda.is_current_stream_capturing()
                t = torch.full((B,), mod.infer_t, device=anchor.device, dtype=torch.long)
                g = torch.Generator(device=anchor.device); g.manual_seed(mod.infer_seed)
                z = torch.randn(a.shape, generator=g, device=anchor.device, dtype=a.dtype)
                r = None
                if mod.selective:
                    gr = torch.Generator(device=anchor.device); gr.manual_seed(mod.infer_seed + 1)
                    r = torch.rand(a.shape[:-1], generator=gr, device=anchor.device, dtype=a.dtype)
                pd_cache[key] = (t, z, r)
            t, z, r = pd_cache[key]
            sa = mod.sqrt_ab[t].view(B, 1, 1, 1)
            s1 = mod.sqrt_1mab[t].view(B, 1, 1, 1)
            mean_n = sa * a
            mean_m = mean_n * mod.norm_scale + mod.norm_offset
            with torch.no_grad():
                eps = mod._aniso_eps(z, mean_m.detach().float(), map_memory, r)
            x_t = (mean_n + s1 * eps) * mod.norm_scale + mod.norm_offset
            if mod.clip_to_boundary:
                with torch.no_grad():
                    clip = mod._clip_boundary(mean_m.detach().float(), x_t.detach().float(), map_memory)
                if clip is not None:
                    f, crossed = clip
                    clipped = mean_m + f[..., None].to(x_t.dtype) * (x_t - mean_m)
                    x_t = torch.where(crossed[..., None], clipped, x_t)
            x_t = x_t.reshape(B, N, D)
            t_emb = mod.t_mlp(_module_of(mod).timestep_embedding(t, mod.embed_dims)).unsqueeze(1)
            return x_t, t_emb

        pd._segments = seg_static
        pd.forward = types.MethodType(pd_forward, pd)

        # --- 교사 토큰 사영(qwen_token_attn.py:172-193): 범위 검사 bool(...any())(동기화)를 호스트(_host_checks, CPU 텐서)로 옮긴다.
        #     qwen_mask 는 전부 1 임을 호스트에서 확인하고 None 으로 넘긴다(원본도 전부 1 이면 마스크를 넘기지 않는다: :231).
        qa = od.qwen_token_attn

        def project_kv(mod, qwen_hidden, qwen_type=None, dtype=None):
            kv = mod.qwen_token_proj(qwen_hidden.to(mod.qwen_token_proj[0].weight.dtype))
            if dtype is not None:
                kv = kv.to(dtype)
            if mod.num_token_types > 0 or mod.route_type_ids is not None:
                if qwen_type is None or qwen_type.shape != kv.shape[:2]:
                    raise ValueError("QwenTokenAttn: qwen_type 모양 불일치")
                qwen_type = qwen_type.long()
            if mod.num_token_types > 0:
                kv = kv + mod.token_type_embed.to(kv.dtype)[qwen_type]
            return kv, qwen_type

        qa.project_kv = types.MethodType(project_kv, qa)

    # ------------------------------------------------------------------ 상태 버퍼
    def _state_attrs(self):
        od = self.od
        db, mb, eb, pb = od.det_instance_bank, od.map_instance_bank, od.ego_instance_bank, od.plan_instance_bank
        return [(db, "cached_feature"), (db, "cached_anchor"), (db, "confidence"), (db, "temp_confidence"),
                (db, "instance_id"), (db, "prev_id"),
                (mb, "plan_map_pts"), (mb, "plan_map_scores"), (mb, "plan_map_ts"),
                (eb, "cached_feature"), (eb, "cached_anchor"),
                (pb, "cached_feature"), (pb, "cached_anchor"), (pb, "confidence")]

    def _alloc(self, data):
        """고정 입력 버퍼·상태 버퍼를 만든다. 상태는 첫 eager 프레임이 남긴 뱅크 상태에서 복사."""
        metas, t = self._unwrap(data)
        self._host_checks(t)
        dev = self.dev
        S = {}
        for k in ("img", "timestamp", "projection_mat", "image_wh", "ego_status", "gt_ego_fut_cmd", "qwen_hidden", "qwen_type"):
            S[k] = torch.empty_like(t[k], device=dev)
        S["T_det"] = torch.empty((1, 4, 4), dtype=torch.float32, device=dev)
        S["T_map"] = torch.empty((1, 4, 4), dtype=torch.float32, device=dev)
        S["same_scene"] = torch.empty((1,), dtype=torch.bool, device=dev)
        S["prev_ts"] = self.od.det_instance_bank.metas["timestamp"].detach().clone()
        self.S = S
        self.state = []
        for obj, attr in self._state_attrs():
            v = getattr(obj, attr)
            if not torch.is_tensor(v):
                v = torch.tensor(v, device=dev, dtype=torch.long)  # prev_id 가 아직 파이썬 int 인 경우(원본 += 와 같은 dtype)
            buf = v.detach().clone()
            setattr(obj, attr, buf)
            self.state.append((obj, attr, buf))
        od = self.od
        for b in (od.det_instance_bank, od.ego_instance_bank, od.plan_instance_bank):
            b.metas = {"timestamp": S["prev_ts"]}
        # 고정 메타(디코더가 읽는 키). qwen_mask=None: 전부 1 임을 호스트에서 확인(위 주석).
        self.static_metas_keys = ("timestamp", "projection_mat", "image_wh", "ego_status", "gt_ego_fut_cmd")

    def _stage(self, data):
        """매 프레임 호스트 값 → 고정 입력 버퍼(replay 전). eager 와 같은 식·같은 dtype 변환."""
        metas, t = self._unwrap(data)
        self._host_checks(t)
        S = self.S
        for k in ("img", "timestamp", "projection_mat", "image_wh", "ego_status", "gt_ego_fut_cmd", "qwen_hidden", "qwen_type"):
            S[k].copy_(t[k], non_blocking=False)
        prev = self.prev_metas
        # det 뱅크 get(instance_bank.py:93-97): x["T_global_inv"] @ 이전 metas T_global → float32
        T_det = np.stack([x["T_global_inv"] @ prev[i]["T_global"] for i, x in enumerate(metas)])
        # 지도 기억(plan_map_memory_bank.py:67-68): 같은 식(이전 프레임 T_global = plan_map_T_global)
        T_map = np.stack([x["T_global_inv"] @ np.asarray(prev[i]["T_global"]) for i, x in enumerate(metas)])
        same = [(a is None or b is None or a == b) for a, b in
                zip([m.get("scene_token") for m in metas], [m.get("scene_token") for m in prev])]
        S["T_det"].copy_(torch.tensor(T_det, dtype=torch.float32))
        S["T_map"].copy_(torch.tensor(T_map, dtype=torch.float32))
        S["same_scene"].copy_(torch.tensor(same, dtype=torch.bool))
        self.cur_metas = metas
        return metas

    # ------------------------------------------------------------------ 캡처 대상
    def _core(self):
        """백본+넥+디코더(+후처리 텐서). 상태 버퍼를 읽고 끝에서 새 상태를 같은 버퍼에 쓴다."""
        S = self.S
        det = self.det
        img = S["img"]
        feature_maps = det.extract_feat(img)
        data = {"img_metas": self.cur_metas_static}
        for k in self.static_metas_keys:
            data[k] = S[k]
        data["qwen_hidden"] = S["qwen_hidden"].squeeze(1).to(dtype=torch.bfloat16)   # detector._maybe_load_qwen_cache 와 같음
        data["qwen_mask"] = None
        data["qwen_type"] = S["qwen_type"].squeeze(1).long()
        model_outs = det.head(img, feature_maps, data)
        post = self._post_tensors(model_outs, data) if self.post_graph else None
        # 상태 되쓰기
        for obj, attr, buf in self.state:
            new = getattr(obj, attr)
            if new is not buf:
                buf.copy_(new)
                setattr(obj, attr, buf)
        S["prev_ts"].copy_(S["timestamp"])
        for b in (self.od.det_instance_bank, self.od.ego_instance_bank, self.od.plan_instance_bank):
            b.metas = {"timestamp": S["prev_ts"]}
        return model_outs, post

    # ------------------------------------------------------------------ 후처리(텐서 부분) — 원본 decode 와 같은 연산, 고정 모양
    def _post_tensors(self, model_outs, data):
        det_output, map_output, ego_output, plan_output, motion_output = model_outs
        od = self.od
        out = {}
        # SparseBox3DDecoder.decode (detection_tracking/decoder.py:38-110), score_threshold=None, instance_id 있음(squeeze_cls)
        def det_part(cls_scores_l, box_preds_l, instance_id, quality_l, num_output, sorted_):
            cls_scores = cls_scores_l[-1].sigmoid()
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)
            box_preds = box_preds_l[-1]
            bs, num_pred, num_cls = cls_scores.shape
            cls_scores, indices = cls_scores.flatten(start_dim=1).topk(num_output, dim=1, sorted=sorted_)
            centerness = quality_l[-1][..., 0]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            indices = torch.gather(indices, 1, idx)
            category_ids = cls_ids[0][indices[0]]
            scores = cls_scores[0]
            box = box_preds[0, indices[0] // num_cls]
            scores_origin = cls_scores_origin[0]
            return category_ids, scores, box, scores_origin, indices, num_cls

        def decode_box(box):  # detection_tracking/decoder.py:9-20, 리스트 색인 [X,Y,Z]/[W,L,H] → 같은 원소 슬라이스
            yaw = torch.atan2(box[..., 6], box[..., 7])
            return torch.cat([box[..., 0:3], box[..., 3:6].clamp(max=5.0).exp(), yaw[..., None], box[..., 8:]], dim=-1)

        dd = od.det_decoder
        cat_ids, scores, box, scores_origin, indices, num_cls = det_part(
            det_output["classification"], det_output["prediction"], det_output["instance_id"],
            det_output["quality"], dd.num_output, dd.sorted)
        box_dec = decode_box(box)
        out["boxes_3d"] = box_dec
        out["scores_3d"] = scores
        out["labels_3d"] = cat_ids
        out["cls_scores"] = scores_origin
        out["instance_ids"] = det_output["instance_id"][0, indices[0]]
        # SparseMotionDecoder.decode (motion_planning/motion_decoder.py) — det 부분은 같은 연산을 한 번 더 한다(원본과 같음)
        md = od.motion_decoder
        m_cat, m_scores, m_box, _, m_indices, m_num_cls = det_part(
            det_output["classification"], det_output["prediction"], det_output["instance_id"],
            det_output["quality"], md.num_output, md.sorted)
        m_box = decode_box(m_box)
        trajs = motion_output["prediction"][-1]
        traj_cls = motion_output["classification"][-1].sigmoid()
        traj = trajs[0, m_indices[0] // m_num_cls]
        traj_cls = traj_cls[0, m_indices[0] // m_num_cls]
        out["trajs_3d"] = traj.cumsum(dim=-2) + m_box[:, None, None, :2]
        out["trajs_score"] = traj_cls
        # SparsePoint3DDecoder.decode (online_map/decoder.py:12-38), score_threshold=None
        mpd = od.map_decoder
        mcls = map_output["classification"][-1]
        bs, num_pred, mnum_cls = mcls.shape
        mcls = mcls.sigmoid()
        pts_preds = map_output["prediction"][-1].reshape(bs, num_pred, -1, mpd.coords_dim)
        mscores, mindices = mcls.flatten(start_dim=1).topk(num_pred, dim=1)
        out["map_labels"] = (mindices % mnum_cls)[0]
        out["map_scores"] = mscores[0]
        out["map_pts"] = pts_preds[0, mindices[0] // mnum_cls]
        # SparsePlanDecoder.decode (motion_planning/plan_decoder.py:99-…) — 고정 모양 부분
        out.update(self._plan_part(ego_output, det_output, motion_output, plan_output, data))
        return out

    def _plan_part(self, ego_output, det_output, motion_output, planning_output, data):
        pdec = self.od.plan_decoder
        assert pdec.num_group == 1 and pdec.anchor_types == [("temp", "2hz")] and pdec.with_rescore and not pdec.adapt_status
        prediction = planning_output["prediction"][-1]
        classification = planning_output["classification"][-1]
        batch_size = classification.shape[0]
        cls_p = classification.reshape(batch_size, pdec.ego_fut_cmd, -1)
        reg_p = prediction.reshape(batch_size, pdec.ego_fut_cmd, -1, pdec.ego_fut_ts, 2).cumsum(dim=-2)
        # select (plan_decoder.py select)
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        motion_cls = motion_output["classification"][-1].sigmoid()
        motion_reg = motion_output["prediction"][-1].cumsum(-2)
        bs = batch_size
        bs_indices = torch.arange(bs, device=cls_p.device)
        cmd = data["gt_ego_fut_cmd"].argmax(dim=-1)
        cls_p = cls_p[bs_indices, cmd]
        reg_p = reg_p[bs_indices, cmd]
        ego_fut_mode = reg_p.shape[1]
        cls_p, _ = self._rescore(pdec, cls_p, reg_p, motion_cls, motion_reg, det_anchors, det_confidence,
                                 ego_fut_mode=ego_fut_mode)
        vis_reg, vis_cls = reg_p.detach(), cls_p.detach()
        mode_idx = cls_p.argmax(dim=-1)
        reg_sel = reg_p[bs_indices, mode_idx]
        return {"plan_modes": vis_reg[0], "plan_mode_probs": vis_cls[0].float().softmax(-1), "plan_temp_2hz": reg_sel[0]}

    def _rescore(self, pdec, plan_cls, plan_reg, motion_cls, motion_reg, det_anchors, det_confidence,
                 score_thresh=0.15, static_dis_thresh=0.5, dim_scale=1.1, num_motion_mode=1, offset=None,  # 수정(2026-11-06): None → pdec.rescore_ego_offset. #이전 offset=0.5
                 ego_fut_ts=None, ego_fut_mode=None):
        """plan_decoder.py SparsePlanDecoder.rescore 와 같은 연산. 리스트 색인 쓰기([X,Y]→0:2, [W,L,H]→3:6, [YAW]→6:7)와
        new_tensor(ego_size)·box3d_to_corners_gpu 의 torch.tensor(corners_norm)(H2D)만 같은 값의 슬라이스·캐시 상수로 바꿨다."""
        if offset is None:  # 수정(2026-11-06): plan_decoder.py rescore 와 같은 기본값 규칙
            offset = pdec.rescore_ego_offset  # 수정(2026-11-06): 디코더 config 의 rescore_ego_offset 값을 쓴다
        if ego_fut_ts is None:
            ego_fut_ts = pdec.ego_fut_ts
        if ego_fut_mode is None:
            ego_fut_mode = pdec.ego_fut_mode
        C = self._consts

        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            return torch.cat([zeros, traj], dim=-2)

        def get_yaw(traj, start_yaw=np.pi / 2):
            yaw = traj.new_zeros(traj.shape[:-1])
            yaw[..., 1:-1] = torch.atan2(traj[..., 2:, 1] - traj[..., :-2, 1], traj[..., 2:, 0] - traj[..., :-2, 0])
            yaw[..., -1] = torch.atan2(traj[..., -1, 1] - traj[..., -2, 1], traj[..., -1, 0] - traj[..., -2, 0])
            yaw[..., 0] = start_yaw
            start = traj[..., 0, :]
            end = traj[..., -1, :]
            dist = torch.linalg.norm(end - start, dim=-1)
            mask = dist < static_dis_thresh
            start_yaw = yaw[..., 0].unsqueeze(-1)
            yaw = torch.where(mask.unsqueeze(-1), start_yaw, yaw)
            return yaw.unsqueeze(-1)

        bs = plan_reg.shape[0]
        plan_reg_cat = cat_with_zero(plan_reg)
        ego_box = det_anchors.new_zeros(bs, ego_fut_mode, ego_fut_ts + 1, 7)
        ego_box[..., 0:2] = plan_reg_cat
        ego_box[..., 3:6] = C["ego_size"] * dim_scale
        ego_box[..., 6:7] = get_yaw(plan_reg_cat, pdec.ego_start_yaw)  # 수정(2026-11-06): plan_decoder.py 와 같은 전진축 규약. #이전 get_yaw(plan_reg_cat)

        motion_reg = motion_reg[..., :ego_fut_ts, :]
        motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None, None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, 1, ego_fut_ts + 1, 2)
        motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)

        motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
        motion_box[..., 0:2] = motion_reg
        motion_box[..., 3:6] = det_anchors[..., None, None, 3:6].clamp(max=5.0).exp()
        box_yaw = torch.atan2(det_anchors[..., 6], det_anchors[..., 7])
        motion_box[..., 6:7] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))

        filter_mask = det_confidence < score_thresh
        motion_box[filter_mask] = 1e6

        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]

        bs, num_ego_mode, ts, _ = ego_box.shape
        bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape
        ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode, 1, 1, 1).flatten(0, -2)
        motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1, 1).flatten(0, -2)

        ego_box[:, 0] += offset * torch.cos(ego_box[:, 6])  # 수정(2026-11-06): (N,7) 열 색인(plan_decoder.py 와 같은 수정). #이전 ego_box[0] += ...
        ego_box[:, 1] += offset * torch.sin(ego_box[:, 6])  # 수정(2026-11-06): 같은 이유. #이전 ego_box[1] += ...
        col = self._check_collision(ego_box, motion_box)
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4)
        col = col.flatten(2, -1).any(dim=-1)
        all_col = col.all(dim=-1)
        col[all_col] = False
        score_offset = col.float() * -999
        plan_cls = plan_cls + score_offset
        return plan_cls, all_col

    def _check_collision(self, boxes1, boxes2):
        col_1 = self._corners_in_box(boxes1.clone(), boxes2.clone())
        col_2 = self._corners_in_box(boxes2.clone(), boxes1.clone())
        return torch.logical_or(col_1, col_2)

    def _corners_in_box(self, boxes1, boxes2):
        # plan_decoder.py corners_in_box 와 같다(모양 0 검사는 고정 모양이라 해당 없음). [:, [3]]·[:, [4]] → 3:4·4:5,
        # [:, [0,3,7,4], :2] → 캐시된 장치 색인 텐서.
        assert boxes1.shape[0] > 0 and boxes2.shape[0] > 0
        C = self._consts
        boxes1_yaw = boxes1[:, 6].clone()
        boxes1_loc = boxes1[:, :3].clone()
        cos_yaw = torch.cos(-boxes1_yaw)
        sin_yaw = torch.sin(-boxes1_yaw)
        rot_mat_T = torch.stack([torch.stack([cos_yaw, sin_yaw]), torch.stack([-sin_yaw, cos_yaw])])
        boxes1[:, :3] = boxes1[:, :3] - boxes1_loc
        boxes1[:, :2] = torch.einsum('ij,jki->ik', boxes1[:, :2], rot_mat_T)
        boxes1[:, 6] = boxes1[:, 6] - boxes1_yaw
        boxes2[:, :3] = boxes2[:, :3] - boxes1_loc
        boxes2[:, :2] = torch.einsum('ij,jki->ik', boxes2[:, :2], rot_mat_T)
        boxes2[:, 6] = boxes2[:, 6] - boxes1_yaw
        corners_box2 = self._box3d_to_corners_gpu(boxes2).index_select(1, C["c_idx"])[..., :2]
        H = boxes1[:, 3:4]
        W = boxes1[:, 4:5]
        collision = torch.logical_and(
            torch.logical_and(corners_box2[..., 0] <= H / 2, corners_box2[..., 0] >= -H / 2),
            torch.logical_and(corners_box2[..., 1] <= W / 2, corners_box2[..., 1] >= -W / 2))
        return collision.any(dim=-1)

    def _box3d_to_corners_gpu(self, box3d):
        # common/box_ops.py box3d_to_corners_gpu: corners_norm(torch.tensor H2D)→캐시 상수, box3d[:, None, [W,L,H]] → 3:6 슬라이스
        C = self._consts
        box3d = box3d.detach()
        corners = box3d[:, None, 3:6] * C["corners_norm"].reshape([1, 8, 3])
        rot_cos = torch.cos(box3d[:, 6])
        rot_sin = torch.sin(box3d[:, 6])
        rot_mat = torch.tile(C["eye3"][None], (box3d.shape[0], 1, 1))
        rot_mat[:, 0, 0] = rot_cos
        rot_mat[:, 0, 1] = -rot_sin
        rot_mat[:, 1, 0] = rot_sin
        rot_mat[:, 1, 1] = rot_cos
        corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
        corners += box3d[:, None, :3]
        return corners

    def _make_consts(self, det_dtype=torch.float32):
        dev = self.dev
        corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
        self._consts = {
            "corners_norm": torch.tensor(corners_norm, device=dev, dtype=torch.float32),
            "eye3": torch.eye(3, device=dev),
            "c_idx": torch.tensor([0, 3, 7, 4], device=dev, dtype=torch.long),
            "ego_size": torch.tensor(self.od.plan_decoder.ego_size, device=dev, dtype=det_dtype),
        }

    # ------------------------------------------------------------------ 후처리(호스트 부분) — 원본 출력 형식으로 조립
    def _assemble(self, post):
        """GPU 결과 → 원본 decode 와 같은 dict. CPU 복사는 고정(pinned) 버퍼로 비동기 → 한 번 동기화."""
        cpu = {}
        for k, v in post.items():
            if k == "instance_ids":
                continue
            p = self._pinned.get(k)
            if p is None or p.shape != v.shape or p.dtype != v.dtype:
                p = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
                self._pinned[k] = p
            p.copy_(v, non_blocking=True)
            cpu[k] = p
        inst = post["instance_ids"].clone()  # 원본 출력도 GPU 텐서(다음 replay 가 덮어쓰지 않게 복제)
        torch.cuda.current_stream().synchronize()
        r = {
            "boxes_3d": cpu["boxes_3d"].clone(), "scores_3d": cpu["scores_3d"].clone(),
            "labels_3d": cpu["labels_3d"].clone(), "cls_scores": cpu["cls_scores"].clone(), "instance_ids": inst,
        }
        mp = cpu["map_pts"].numpy().copy()
        r["vectors"] = [mp[i] for i in range(mp.shape[0])]  # 원본: [vec.cpu().numpy() for vec in pts]
        r["scores"] = cpu["map_scores"].numpy().copy()
        r["labels"] = cpu["map_labels"].numpy().copy()
        r["plan_modes"] = cpu["plan_modes"].clone()
        r["plan_mode_probs"] = cpu["plan_mode_probs"].clone()
        r["plan_temp_2hz"] = cpu["plan_temp_2hz"].clone()
        r["trajs_3d"] = cpu["trajs_3d"].clone()
        r["trajs_score"] = cpu["trajs_score"].clone()
        return [{"img_bbox": r}]

    def _assemble_eager_post(self, model_outs, metas):
        """후처리를 원본 함수 그대로 eager 로(그래프 출력 버퍼를 읽는다)."""
        data = {k: self.S[k] for k in self.static_metas_keys}
        data["img_metas"] = metas
        results = self.det.head.post_process(model_outs, data)
        output = []
        for result in results:
            out_dict = {}
            if "metric_results" in result:
                out_dict["metric_results"] = result.pop("metric_results")
            out_dict["img_bbox"] = result
            output.append(out_dict)
        return output

    # ------------------------------------------------------------------ 컴파일(선택)
    @staticmethod
    def _fix_torch_ops_namespace():
        """mmcv/device/npu/data_parallel.py(가져오기 시점)가 sys.modules 의 torch* 모듈마다 hasattr(m, '_check_balance') 를 하는데,
        torch.ops 의 __getattr__ 가 이 이름으로 빈 연산 이름공간을 만들어 torch.ops._dir 에 넣고, 이어서 setattr 로 함수가 덮인다.
        torch.compile(inductor)은 torch.ops._dir 의 모든 이름공간을 순회하다 'function' object is not iterable 로 죽는다.
        연산이 하나도 없는 이 가짜 이름공간 항목만 목록에서 뺀다(프로세스 메모리 안 정리, 파일 수정 없음)."""
        d = torch.ops._dir
        for name in list(d):
            if not isinstance(getattr(torch.ops, name, None), torch._ops._OpNamespace):
                d.remove(name)

    def _apply_compile(self):
        if not self.compile_mode:
            return
        self._fix_torch_ops_namespace()
        det = self.det
        t0 = time.perf_counter()
        opts = dict(mode=self.compile_mode, dynamic=False)
        if "backbone" in self.compile_targets:
            det.img_backbone.forward = torch.compile(det.img_backbone.forward, **opts)
        if "neck" in self.compile_targets:
            det.img_neck.forward = torch.compile(det.img_neck.forward, **opts)
        if "decoder_layers" in self.compile_targets:
            od = self.od
            for i, layer in enumerate(od.layers):
                if layer is None:
                    continue
                if isinstance(layer, torch.nn.ModuleDict):
                    for m in layer.values():
                        m.forward = torch.compile(m.forward, **opts)
                else:
                    layer.forward = torch.compile(layer.forward, **opts)
            for name in ("det_refine", "map_refine", "plan_refine", "ego_refine", "motion_refine"):
                for m in getattr(od, name):
                    m.forward = torch.compile(m.forward, **opts)
        if "head" in self.compile_targets:
            self._core_impl = torch.compile(self._core_impl, **opts)
        self.timings["compile_wrap_s"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ 빌드: 예열 → 상태 복원 → 캡처
    def _build(self, data):
        t0 = time.perf_counter()
        self._make_consts()
        self._pinned = {}
        self._alloc(data)
        self._install_patches()
        self._core_impl = self._core
        self._apply_compile()
        snap = [buf.clone() for _, _, buf in self.state] + [self.S["prev_ts"].clone()]

        def restore():
            for (obj, attr, buf), s in zip(self.state, snap[:-1]):
                buf.copy_(s)
                setattr(obj, attr, buf)
            self.S["prev_ts"].copy_(snap[-1])
            for b in (self.od.det_instance_bank, self.od.ego_instance_bank, self.od.plan_instance_bank):
                b.metas = {"timestamp": self.S["prev_ts"]}

        metas = self._stage(data)
        self.cur_metas_static = metas
        _Active.on = True
        try:
            with torch.no_grad():
                # 예열(eager, 패치 경로): 상수 캐시·cuBLAS·autocast·컴파일(있으면)을 캡처 전에 끝낸다
                tw = time.perf_counter()
                # 캡처할 때만 곁 스트림에서 예열(캡처 전 곁 스트림 예열이 PyTorch 권장). 캡처하지 않으면 현재 스트림 그대로
                #   (진단용 --daf-impl orig: 원본 확장은 legacy 기본 스트림에 올라가 곁 스트림 작업과 동기화되지 않는다).
                s = torch.cuda.Stream() if self.warmup_stream == "side" else torch.cuda.current_stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        restore()
                        self._mark_step()
                        self._core_impl()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                self.timings["warmup_s"] = time.perf_counter() - tw
                restore()
                torch.cuda.synchronize()
                if self.use_graph:
                    tc = time.perf_counter()
                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph):
                        self.g_out = self._core_impl()
                    torch.cuda.synchronize()
                    self.timings["capture_s"] = time.perf_counter() - tc
        finally:
            _Active.on = False
        restore()   # 캡처는 커널을 실행하지 않는다 — 상태는 첫 eager 프레임 직후 그대로
        torch.cuda.synchronize()
        self.timings["build_total_s"] = time.perf_counter() - t0
        if self.verbose:
            print(f"[FastStudent] build {self.timings}", flush=True)

    def _mark_step(self):
        # 수정: reduce-overhead(cudagraph trees) — 표시가 없으면 컴파일 함수에 들어갈 때마다 새 세대로 보아, 넥이 같은 프레임의
        #   백본 출력을 "덮어쓴 출력"으로 판정해 실패했다. 프레임 시작에서 한 번만 세대를 넘긴다(프레임 안 출력은 그 프레임 안에서만 쓴다).
        if self.compile_mode == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()

    def _run(self, data):
        metas = self._stage(data)
        self.cur_metas_static = metas
        if self.use_graph:
            self.graph.replay()
            model_outs, post = self.g_out
        else:
            _Active.on = True
            try:
                with torch.no_grad():
                    self._mark_step()
                    model_outs, post = self._core_impl()
            finally:
                _Active.on = False
        self.last_model_outs = model_outs  # 비교용(그래프 출력 버퍼 — 다음 replay 전에 읽을 것)
        if self.post_graph:
            out = self._assemble(post)
        else:
            out = self._assemble_eager_post(model_outs, metas)
        self.prev_metas = metas
        return out

    # ------------------------------------------------------------------ 비교용: 다른 모델(원본)의 시간 뱅크 상태를 그대로 가져온다
    def load_state_from(self, ref_det):
        rod = ref_det.head.onedecoder_head
        names = {id(self.od.det_instance_bank): "det_instance_bank", id(self.od.map_instance_bank): "map_instance_bank",
                 id(self.od.ego_instance_bank): "ego_instance_bank", id(self.od.plan_instance_bank): "plan_instance_bank"}
        with torch.no_grad():
            for obj, attr in self._state_attrs():
                src = getattr(getattr(rod, names[id(obj)]), attr)
                src = src if torch.is_tensor(src) else torch.tensor(src, device=self.dev, dtype=torch.long)
                if self._built:
                    buf = next(b for o, a, b in self.state if o is obj and a == attr)
                    buf.copy_(src)
                else:
                    setattr(obj, attr, src.detach().clone())
            ts = rod.det_instance_bank.metas["timestamp"]
            if self._built:
                self.S["prev_ts"].copy_(ts)
            else:
                for n in ("det_instance_bank", "ego_instance_bank", "plan_instance_bank"):
                    b = getattr(self.od, n)
                    b.metas = dict(b.metas, timestamp=ts.detach().clone())
                self.od.map_instance_bank.plan_map_ts = rod.map_instance_bank.plan_map_ts.detach().clone()
