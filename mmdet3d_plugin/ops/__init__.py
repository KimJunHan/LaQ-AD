import torch

from .deformable_aggregation import DeformableAggregationFunction


# 수정(2026-10-15): 소스 정리 — A800 전용 분기(deformable_aggregation_a800, 별도 .so)를 archive 로 이동.
#   A800 이 아닌 장치에서는 종전에도 항상 DeformableAggregationFunction 경로였으므로 동작 동일.
def deformable_aggregation_function(
    feature_maps,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    return DeformableAggregationFunction.apply(
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    )


# 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — feature_maps_format 모양 상수 캐시.
#   종전에는 매 호출 spatial_shape·scale_start_index 를 torch.tensor 로 새로 만들고(H2D 2 회), inverse 에서 .cpu() 2 회 +
#   torch.all(...) 비교(호스트 동기화)를 했다. 값은 특징 지도 모양(상수)으로만 정해지므로, 순전파 결과 텐서와 inverse 에 필요한
#   파이썬 리스트를 (카메라 수, 수준별 H·W, 장치) 키로 처음 한 번 종전과 같은 식으로 만들어 둔다.
#   inverse 는 "캐시가 만든 바로 그 텐서 객체"(is 비교)를 받았을 때만 캐시를 쓰고, 그 밖의 입력은 종전 경로 그대로(저장하지 않음
#   → 캐시가 커지지 않고 data_ptr 재사용 문제도 없다). 캐시 텐서는 읽기 전용으로만 쓰인다(DAF 는 .contiguous().int() 사본을 쓴다).
_FMF_CACHE = {}  # (num_cams, ((H, W), ...), device) -> (spatial_shape, scale_start_index, inverse 용 리스트)


def _fmf_inverse_lists(spatial_shape_list):
    """종전 inverse 의 split_size·cam_split·cam_split_size 계산을 파이썬 리스트(같은 정수)로 한다."""
    num_cams = len(spatial_shape_list)
    split_size = [[h * w for h, w in cam] for cam in spatial_shape_list]
    cam_split = [1]
    cam_split_size = [sum(split_size[0])]
    for i in range(num_cams - 1):
        if spatial_shape_list[i] != spatial_shape_list[i + 1]:
            cam_split.append(0)
            cam_split_size.append(0)
        cam_split[-1] += 1
        cam_split_size[-1] += sum(split_size[i + 1])
    return split_size, cam_split, cam_split_size


def _fmf_inverse_from_lists(col_feats, split_size, cam_split, cam_split_size, spatial_shape):
    mc_feat = [
        x.unflatten(1, (cam_split[i], -1))
        for i, x in enumerate(col_feats.split(cam_split_size, dim=1))
    ]
    mc_ms_feat = []
    shape_index = 0
    for i, feat in enumerate(mc_feat):
        feat = list(feat.split(split_size[shape_index], dim=2))
        for j, f in enumerate(feat):
            feat[j] = f.unflatten(2, spatial_shape[shape_index][j])
            feat[j] = feat[j].permute(0, 1, 4, 2, 3)
        mc_ms_feat.append(feat)
        shape_index += cam_split[i]
    return mc_ms_feat


def feature_maps_format(feature_maps, inverse=False):
    if inverse:
        col_feats, spatial_shape, scale_start_index = feature_maps
        for ent in _FMF_CACHE.values():  # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — 캐시가 만든 텐서 객체일 때만 동기화 없는 경로
            if spatial_shape is ent[0]:
                return _fmf_inverse_from_lists(col_feats, *ent[2])
        num_cams, num_levels = spatial_shape.shape[:2]

        split_size = spatial_shape[..., 0] * spatial_shape[..., 1]
        split_size = split_size.cpu().numpy().tolist()

        idx = 0
        cam_split = [1]
        cam_split_size = [sum(split_size[0])]
        for i in range(num_cams - 1):
            if not torch.all(spatial_shape[i] == spatial_shape[i + 1]):
                cam_split.append(0)
                cam_split_size.append(0)
            cam_split[-1] += 1
            cam_split_size[-1] += sum(split_size[i + 1])
        mc_feat = [
            x.unflatten(1, (cam_split[i], -1))
            for i, x in enumerate(col_feats.split(cam_split_size, dim=1))
        ]

        spatial_shape = spatial_shape.cpu().numpy().tolist()
        mc_ms_feat = []
        shape_index = 0
        for i, feat in enumerate(mc_feat):
            feat = list(feat.split(split_size[shape_index], dim=2))
            for j, f in enumerate(feat):
                feat[j] = f.unflatten(2, spatial_shape[shape_index][j])
                feat[j] = feat[j].permute(0, 1, 4, 2, 3)
            mc_ms_feat.append(feat)
            shape_index += cam_split[i]
        return mc_ms_feat

    if isinstance(feature_maps[0], (list, tuple)):
        formated = [feature_maps_format(x) for x in feature_maps]
        col_feats = torch.cat([x[0] for x in formated], dim=1)
        spatial_shape = torch.cat([x[1] for x in formated], dim=0)
        scale_start_index = torch.cat([x[2] for x in formated], dim=0)
        return [col_feats, spatial_shape, scale_start_index]

    bs, num_cams = feature_maps[0].shape[:2]
    spatial_shape = []

    col_feats = []
    for i, feat in enumerate(feature_maps):
        spatial_shape.append(feat.shape[-2:])
        col_feats.append(
            torch.reshape(feat, (bs, num_cams, feat.shape[2], -1))
        )

    col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
    key = (num_cams, tuple((int(s[0]), int(s[1])) for s in spatial_shape), col_feats.device)  # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — 모양 상수 캐시
    ent = _FMF_CACHE.get(key)
    if ent is not None:
        return [col_feats, ent[0], ent[1]]
    assert not (col_feats.is_cuda and torch.cuda.is_current_stream_capturing()), "feature_maps_format: 상수 캐시는 CUDA Graph 캡처 전에 만들어야 한다"  # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — 캡처 중 H2D 금지
    spatial_shape = [spatial_shape] * num_cams
    spatial_shape = torch.tensor(
        spatial_shape,
        dtype=torch.int64,
        device=col_feats.device,
    )
    scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
    scale_start_index = scale_start_index.flatten().cumsum(dim=0)
    scale_start_index = torch.cat(
        [torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]]
    )
    scale_start_index = scale_start_index.reshape(num_cams, -1)
    # 수정(2026-11-05): 사용자 결정 "검토해서 연산시간 줄이자" — inverse 용 리스트(종전 spatial_shape.cpu().numpy().tolist() 와 같은 정수)
    ss_list = [[[h, w] for h, w in key[1]] for _ in range(num_cams)]
    _FMF_CACHE[key] = (spatial_shape, scale_start_index, _fmf_inverse_lists(ss_list) + (ss_list,))

    feature_maps = [
        col_feats,
        spatial_shape,
        scale_start_index,
    ]
    return feature_maps
