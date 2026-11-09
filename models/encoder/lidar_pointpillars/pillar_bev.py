# 수정(2026-10-15): B 융합 — PointPillars 방식 BEV 인코더(새 파일). spconv/torch_scatter/mmdet3d 없이 순수 PyTorch.
#   점 → 기둥(pillar, z 방향 1칸) 동적 배정 → 점별 장식 특징(PointPillars: 원 특징 + 기둥 평균 대비 오프셋 xc,yc,zc
#   + 기둥 중심 대비 오프셋 xp,yp) → PFN(Linear→BN1d→ReLU) → 기둥별 max → BEV 캔버스 → 2D 합성곱 백본 → FPN 식 다중 해상도 출력.
#   * 동적 배정(기둥당 점 수 상한·기둥 수 상한 없음): hard voxelization 의 max_num_points / max_voxels 절단이 없다
#     (점·기둥을 버리지 않는다). max 집계는 torch.scatter_reduce('amax').
#   * 격자 규약: 캔버스[b, :, iy, ix], ix = floor((x - x0)/vx) 는 W 축, iy 는 H 축.
#     픽셀 (iy,ix) 중심 = (x0 + (ix+0.5)vx, y0 + (iy+0.5)vy) → F.grid_sample(align_corners=False) 정규화 좌표와 정확히 일치.
#   * 다운샘플은 kernel 2 / stride 2 합성곱: 출력 j 의 수용 중심이 입력 2j+1 경계 = 출력 픽셀 중심과 일치해
#     해상도별 특징 위치가 같은 격자 규약을 유지한다(3x3/stride2 는 단계마다 반 픽셀씩 어긋난다). 상향은 bilinear(align_corners=False).
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models import BACKBONES

__all__ = ["PillarBEVEncoder", "pillar_scatter"]


def _conv_bn_relu(cin, cout, k=3, s=1):
    p = (k - 1) // 2 if s == 1 else 0
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def pillar_scatter(points_list, voxel_size, pc_range, grid_hw):
    """점 리스트 → (점 묶음, 기둥 인덱스). 반환: pts(N,D), bidx(N,), ix, iy, pid(N,), uniq(U,), inv(N,)."""
    H, W = grid_hw
    pts, bidx = [], []
    for b, p in enumerate(points_list):
        pts.append(p)
        bidx.append(torch.full((p.shape[0],), b, dtype=torch.long, device=p.device))
    pts = torch.cat(pts, 0)
    bidx = torch.cat(bidx, 0)
    ix = torch.floor((pts[:, 0] - pc_range[0]) / voxel_size[0]).long()
    iy = torch.floor((pts[:, 1] - pc_range[1]) / voxel_size[1]).long()
    # 범위 밖 점은 파이프라인(PointsRangeFilterFusion)에서 이미 걸렀다. 남아 있으면 설정 불일치이므로 즉시 실패.
    if pts.shape[0] and (ix.min() < 0 or iy.min() < 0 or ix.max() >= W or iy.max() >= H):
        raise ValueError("BEV 격자 밖의 점이 있다: 파이프라인 point_cloud_range 와 인코더 pc_range 를 맞출 것.")
    pid = bidx * (H * W) + iy * W + ix
    uniq, inv = torch.unique(pid, return_inverse=True)
    return pts, bidx, ix, iy, uniq, inv


@BACKBONES.register_module()
class PillarBEVEncoder(BaseModule):
    """점 리스트(표본별 (N_i, in_channels), 앞 3열 = x,y,z LiDAR 키프레임 좌표) → BEV 특징 맵 리스트 [(B,out_ch,H_l,W_l)].

    Args:
        in_channels: 점 원 특징 수(xyz 포함).
        voxel_size: (vx, vy) 기둥 크기(m).
        pc_range: (x0, y0, z0, x1, y1, z1).
        pfn_channels: PFN 출력 채널.
        stage_channels / stage_layers / stage_strides: 2D 백본 단계별 채널·3x3 층 수·stride(1 또는 2).
        out_indices: FPN 출력으로 내보낼 단계 번호. out_channels: FPN 출력 채널(=디코더 embed_dims).
        points_key: 모델 입력 dict 에서 점을 읽는 키(파이프라인 out_key 와 같아야 함).
    """

    def __init__(self, in_channels, voxel_size, pc_range, pfn_channels=64,
                 stage_channels=(64, 128, 256), stage_layers=(3, 5, 5), stage_strides=(2, 2, 2),
                 out_indices=(1, 2), out_channels=256, points_key="lidar_points", init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.voxel_size = tuple(float(v) for v in voxel_size[:2])
        self.pc_range = tuple(float(v) for v in pc_range)
        self.W = int(round((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]))
        self.H = int(round((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]))
        self.points_key = points_key
        self.pfn_channels = pfn_channels
        self.pfn = nn.Sequential(nn.Linear(in_channels + 5, pfn_channels, bias=False),
                                 nn.BatchNorm1d(pfn_channels), nn.ReLU(inplace=True))
        stages, cin = [], pfn_channels
        for c, n, s in zip(stage_channels, stage_layers, stage_strides):
            layers = [_conv_bn_relu(cin, c, k=2 if s == 2 else 3, s=s)]
            layers += [_conv_bn_relu(c, c) for _ in range(n)]
            stages.append(nn.Sequential(*layers))
            cin = c
        self.stages = nn.ModuleList(stages)
        self.out_indices = tuple(out_indices)
        self.lateral = nn.ModuleList([nn.Conv2d(stage_channels[i], out_channels, 1) for i in self.out_indices])
        self.fpn_out = nn.ModuleList([_conv_bn_relu(out_channels, out_channels) for _ in self.out_indices])
        strides, s = [], 1
        for st in stage_strides:
            s *= st
            strides.append(s)
        self.out_strides = [strides[i] for i in self.out_indices]

    def bev_cell_size(self):
        return [(self.voxel_size[0] * s, self.voxel_size[1] * s) for s in self.out_strides]

    def scatter_canvas(self, points_list):
        B = len(points_list)
        pts, bidx, ix, iy, uniq, inv = pillar_scatter(points_list, self.voxel_size, self.pc_range, (self.H, self.W))
        canvas = pts.new_zeros(B * self.H * self.W, self.pfn_channels)
        if pts.shape[0] > 0:
            U = uniq.shape[0]
            xyz = pts[:, :3]
            cnt = torch.bincount(inv, minlength=U).clamp(min=1).to(pts.dtype)
            mean = pts.new_zeros(U, 3).index_add_(0, inv, xyz) / cnt[:, None]
            f_cluster = xyz - mean[inv]
            cx = self.pc_range[0] + (ix.to(pts.dtype) + 0.5) * self.voxel_size[0]
            cy = self.pc_range[1] + (iy.to(pts.dtype) + 0.5) * self.voxel_size[1]
            f_center = torch.stack([xyz[:, 0] - cx, xyz[:, 1] - cy], dim=1)
            feat = self.pfn(torch.cat([pts, f_cluster, f_center], dim=1))  # (N,C) >= 0 (ReLU)
            pillar = feat.new_zeros(U, self.pfn_channels).scatter_reduce(
                0, inv[:, None].expand(-1, self.pfn_channels), feat, reduce="amax", include_self=False)
            canvas[uniq] = pillar
        return canvas.view(B, self.H, self.W, self.pfn_channels).permute(0, 3, 1, 2).contiguous()

    def forward(self, points_list):
        x = self.scatter_canvas(points_list)
        feats = []
        for st in self.stages:
            x = st(x)
            feats.append(x)
        lat = [l(feats[i]) for l, i in zip(self.lateral, self.out_indices)]
        for k in range(len(lat) - 2, -1, -1):  # 위(거친) → 아래(세밀) top-down
            lat[k] = lat[k] + F.interpolate(lat[k + 1], size=lat[k].shape[-2:], mode="bilinear", align_corners=False)
        return [o(l) for o, l in zip(self.fpn_out, lat)]
