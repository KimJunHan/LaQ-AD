# 수정(2026-10-15): 경로 이동 models/separate_attn.py → models/decoder/temporal_self_attn/separate_attn.py 및 정리.
#   통합 쿼리 풀을 과제 구간별로 잘라 self/temporal/interactive 어텐션을 돌리는 래퍼.
#   제거(어떤 config 도 켜지 않음 — 동작 동일):
#     - attention mask 경로 전체(get_separate_attn_mask 등). 통합 디코더의 with_attn_mask 가 항상 False 라 mask 는
#       늘 None 이었고, MultiheadFlashAttention 은 mask 를 받으면 assert 로 실패한다.
#     - InteractiveAttention 의 거리/속도 기반 mask(with_distance_attn_mask / with_velocity_attn_mask) 와
#       attn_mask_ban_list / attn_mask_cancel_list.
#     - TemporalSeparateAttention 의 use_updated_query(기본 False).
import contextlib  # 수정(2026-11-09): C3 GRPO §9.3 계획 기울기 격리(아래 _group_ctx)

import numpy as np
import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import ATTENTION

__all__ = ["SeparateAttention", "TemporalSeparateAttention", "InteractiveAttention"]


def _build_attns(attn, num):
    if isinstance(attn, dict):
        attns = [build_from_cfg(attn, ATTENTION) for _ in range(num)]
    elif isinstance(attn, list):
        attns = [build_from_cfg(attn_, ATTENTION) for attn_ in attn]
    else:
        raise NotImplementedError
    return nn.Sequential(*attns)


def _group_ctx(grad_modalities, query_modalities):
    """수정(2026-11-09): C3 GRPO 설계 §9.3 계획 기울기 격리(RL 모드 전용).
    grad_modalities = None(기본) 이면 아무것도 하지 않는다(종전 동작·그래프 그대로).
    주면, 쿼리 모달리티에 grad_modalities 가 하나도 없는 그룹(예: det·map self-attention)을 torch.no_grad() 아래에서 계산한다.
    같은 연산·같은 입력이라 값은 그대로이고, 그 그룹의 계산 그래프(저장 활성)만 생기지 않는다. 계획 쿼리가 들어 있는 그룹은
    종전처럼 기울기를 켠 채 계산한다(키·값 det·map 행으로 흘러가는 기울기는 디코더가 split 에서 끊는다 — unified_decoder.py)."""
    if grad_modalities is None or any(m in grad_modalities for m in query_modalities):
        return contextlib.nullcontext()
    return torch.no_grad()


@ATTENTION.register_module()
class SeparateAttention(nn.Module):
    def __init__(self,
                 attn=None,
                 embed_dims=256,
                 query_select=None,
                 separate_list=None,
                 decouple_list=None,
                 **kwargs):
        super(SeparateAttention, self).__init__()
        self.query_select = query_select
        self.separate_list = separate_list
        self.decouple_list = decouple_list
        assert separate_list is not None
        assert decouple_list is not None
        assert len(separate_list) == len(decouple_list)

        self.attns = _build_attns(attn, len(separate_list))

    def forward(self,
                query, key=None, value=None,
                query_pos=None, key_pos=None,
                num_anchor_cumsum=None,
                num_temp_anchor_cumsum=None,
                fc_before=None, fc_after=None,
                grad_modalities=None,  # 수정(2026-11-09): C3 GRPO §9.3 계획 기울기 격리(_group_ctx 설명, None = 종전)
                **kwargs):
        output_instance = query.clone()
        if key is None:
            for sep, separate in enumerate(self.separate_list):
              with _group_ctx(grad_modalities, separate):  # 수정(2026-11-09): C3 GRPO §9.3 — 비계획 그룹은 no_grad(값 동일)
                sep_query = []
                sep_value = []
                sep_query_pos = []
                num_query_list = []
                for modality in separate:
                    index = self.query_select.index(modality)
                    start, end = num_anchor_cumsum[index], num_anchor_cumsum[index+1]
                    sep_query.append(query[:, start:end])
                    num_query_list.append(end - start)
                    if value is not None:
                        sep_value.append(value[:, start:end])
                    if query_pos is not None:
                        sep_query_pos.append(query_pos[:, start:end])
                sep_query = torch.cat(sep_query, dim=1)
                sep_value = torch.cat(sep_value, dim=1) if len(sep_value) else None
                sep_query_pos = torch.cat(sep_query_pos, dim=1) if len(sep_query_pos) else None

                if self.decouple_list[sep]:
                    sep_query = torch.cat([sep_query, sep_query_pos], dim=-1)
                    sep_query_pos = None

                    if sep_value is not None:
                        sep_value = fc_before(sep_value)

                    output = fc_after(self.attns[sep](query=sep_query, value=sep_value, query_pos=sep_query_pos))
                else:
                    output = self.attns[sep](query=sep_query, value=sep_value, query_pos=sep_query_pos)

                # update query
                num_query_cumsum = np.cumsum([0] + num_query_list)
                for index1, modality in enumerate(separate):
                    index2 = self.query_select.index(modality)
                    start1, end1 = num_query_cumsum[index1], num_query_cumsum[index1+1]
                    start2, end2 = num_anchor_cumsum[index2], num_anchor_cumsum[index2+1]
                    output_instance[:, start2:end2] = output[:, start1:end1]

        else:
            for sep, separate in enumerate(self.separate_list):
              with _group_ctx(grad_modalities, separate):  # 수정(2026-11-09): C3 GRPO §9.3 — 비계획 그룹은 no_grad(값 동일)
                sep_query = []
                sep_key = []
                sep_value = []
                sep_query_pos = []
                sep_key_pos = []
                num_query_list = []
                for modality in separate:
                    index = self.query_select.index(modality)
                    start1, end1 = num_anchor_cumsum[index], num_anchor_cumsum[index+1]
                    start2, end2 = num_temp_anchor_cumsum[index], num_temp_anchor_cumsum[index+1]
                    sep_query.append(query[:, start1:end1])
                    sep_key.append(key[:, start2:end2])
                    num_query_list.append(end1 - start1)

                    if value is not None:
                        sep_value.append(value[:, start2:end2])
                    if query_pos is not None:
                        sep_query_pos.append(query_pos[:, start1:end1])
                    if key_pos is not None:
                        sep_key_pos.append(key_pos[:, start2:end2])
                sep_query = torch.cat(sep_query, dim=1)
                sep_key = torch.cat(sep_key, dim=1)
                sep_value = torch.cat(sep_value, dim=1) if len(sep_value) else None
                sep_query_pos = torch.cat(sep_query_pos, dim=1) if len(sep_query_pos) else None
                sep_key_pos = torch.cat(sep_key_pos, dim=1) if len(sep_key_pos) else None

                # for none key / value
                if sep_key.size(1) == 0:
                    sep_key = None
                    sep_key_pos = None
                if sep_value.size(1) == 0:
                    sep_value = None

                if self.decouple_list[sep]:
                    sep_query = torch.cat([sep_query, sep_query_pos], dim=-1)
                    sep_key = torch.cat([sep_key, sep_key_pos], dim=-1)
                    sep_query_pos, sep_key_pos = None, None

                    if sep_value is not None:
                        sep_value = fc_before(sep_value)

                    output = fc_after(self.attns[sep](query=sep_query, key=sep_key, value=sep_value,
                                                      query_pos=sep_query_pos, key_pos=sep_key_pos))
                else:
                    output = self.attns[sep](query=sep_query, key=sep_key, value=sep_value,
                                             query_pos=sep_query_pos, key_pos=sep_key_pos)

                # update query
                num_query_cumsum = np.cumsum([0] + num_query_list)
                for index1, modality in enumerate(separate):
                    index2 = self.query_select.index(modality)
                    start1, end1 = num_query_cumsum[index1], num_query_cumsum[index1+1]
                    start2, end2 = num_anchor_cumsum[index2], num_anchor_cumsum[index2+1]
                    output_instance[:, start2:end2] = output[:, start1:end1]

        return output_instance


@ATTENTION.register_module()
class TemporalSeparateAttention(nn.Module):
    def __init__(self,
                 attn=None,
                 embed_dims=256,
                 query_select=None,
                 query_list=None,
                 key_list=None,
                 decouple_list=None,
                 **kwargs):
        super(TemporalSeparateAttention, self).__init__()
        self.query_select = query_select
        self.query_list = query_list
        self.key_list = key_list
        self.decouple_list = decouple_list
        assert query_list is not None
        assert decouple_list is not None
        assert len(query_list) == len(key_list) == len(decouple_list)

        self.attns = _build_attns(attn, len(query_list))

    def forward(self,
                query, key=None, value=None,
                query_pos=None, key_pos=None,
                num_anchor_cumsum=None,
                num_temp_anchor_cumsum=None,
                fc_before=None, fc_after=None,
                grad_modalities=None,  # 수정(2026-11-09): C3 GRPO §9.3 계획 기울기 격리(_group_ctx 설명, None = 종전)
                **kwargs):
        output_instance = query.clone()

        if key is None or num_temp_anchor_cumsum is None:
            key = query
            key_pos = query_pos
            num_temp_anchor_cumsum = num_anchor_cumsum

        for sep in range(len(self.decouple_list)):
          with _group_ctx(grad_modalities, self.query_list[sep]):  # 수정(2026-11-09): C3 GRPO §9.3 — 비계획 그룹은 no_grad(값 동일)
            sep_query = []
            sep_key = []
            sep_value = []
            sep_query_pos = []
            sep_key_pos = []
            num_query_list = []

            sub_query_list = self.query_list[sep]
            sub_key_list = self.key_list[sep]

            for query_modality in sub_query_list:
                query_index = self.query_select.index(query_modality)
                start1, end1 = num_anchor_cumsum[query_index], num_anchor_cumsum[query_index+1]
                sep_query.append(query[:, start1:end1])
                num_query_list.append(end1 - start1)

                if query_pos is not None:
                    sep_query_pos.append(query_pos[:, start1:end1])

            for key_modality in sub_key_list:
                key_index = self.query_select.index(key_modality)
                start2, end2 = num_temp_anchor_cumsum[key_index], num_temp_anchor_cumsum[key_index+1]
                sep_key.append(key[:, start2:end2])

                if key_pos is not None:
                    sep_key_pos.append(key_pos[:, start2:end2])
                if value is not None:
                    sep_value.append(value[:, start2:end2])

            sep_query = torch.cat(sep_query, dim=1)
            sep_key = torch.cat(sep_key, dim=1)
            sep_value = torch.cat(sep_value, dim=1) if len(sep_value) else None
            sep_query_pos = torch.cat(sep_query_pos, dim=1) if len(sep_query_pos) else None
            sep_key_pos = torch.cat(sep_key_pos, dim=1) if len(sep_key_pos) else None

            # for none key / value
            if sep_key.size(1) == 0:
                sep_key = None
                sep_key_pos = None
            if sep_value is not None and sep_value.size(1) == 0:
                sep_value = None

            if self.decouple_list[sep]:
                sep_query = torch.cat([sep_query, sep_query_pos], dim=-1)
                sep_query_pos = None
                if sep_key is not None:
                    sep_key = torch.cat([sep_key, sep_key_pos], dim=-1)
                    sep_key_pos = None
                if sep_value is not None:
                    sep_value = fc_before(sep_value)

                output = fc_after(self.attns[sep](query=sep_query, key=sep_key, value=sep_value,
                                                  query_pos=sep_query_pos, key_pos=sep_key_pos))
            else:
                output = self.attns[sep](query=sep_query, key=sep_key, value=sep_value,
                                         query_pos=sep_query_pos, key_pos=sep_key_pos)

            # update query
            num_query_cumsum = np.cumsum([0] + num_query_list)
            for index1, modality in enumerate(sub_query_list):
                index2 = self.query_select.index(modality)
                start1, end1 = num_query_cumsum[index1], num_query_cumsum[index1+1]
                start2, end2 = num_anchor_cumsum[index2], num_anchor_cumsum[index2+1]
                output_instance[:, start2:end2] = output[:, start1:end1]

        return output_instance


@ATTENTION.register_module()
class InteractiveAttention(nn.Module):
    def __init__(self,
                 attn=None,
                 embed_dims=256,
                 query_select=None,
                 query_list=None,
                 key_list=None,
                 decouple_list=None,
                 **kwargs):
        super(InteractiveAttention, self).__init__()
        self.query_select = query_select
        self.query_list = query_list
        self.key_list = key_list
        self.decouple_list = decouple_list

        assert query_list is not None
        assert decouple_list is not None
        assert len(query_list) == len(key_list) == len(decouple_list)

        self.attns = _build_attns(attn, len(query_list))

    def forward(self,
                query, key=None, value=None,
                query_pos=None, key_pos=None,
                num_anchor_cumsum=None,
                num_temp_anchor_cumsum=None,
                fc_before=None, fc_after=None,
                grad_modalities=None,  # 수정(2026-11-09): C3 GRPO §9.3 계획 기울기 격리(_group_ctx 설명, None = 종전)
                **kwargs):
        output_instance = query.clone()

        key = query
        key_pos = query_pos

        num_query_cumsum = num_anchor_cumsum
        num_key_cumsum = num_anchor_cumsum

        for sep in range(len(self.decouple_list)):
          with _group_ctx(grad_modalities, self.query_list[sep]):  # 수정(2026-11-09): C3 GRPO §9.3 — 비계획 그룹은 no_grad(값 동일)
            sep_query = []
            sep_key = []
            sep_value = []
            sep_query_pos = []
            sep_key_pos = []
            num_query_list = []

            sub_query_list = self.query_list[sep]
            sub_key_list = self.key_list[sep]

            for query_modality in sub_query_list:
                query_index = self.query_select.index(query_modality)
                start1, end1 = num_query_cumsum[query_index], num_query_cumsum[query_index+1]
                sep_query.append(query[:, start1:end1])
                num_query_list.append(end1 - start1)

                if query_pos is not None:
                    sep_query_pos.append(query_pos[:, start1:end1])

            for key_modality in sub_key_list:
                key_index = self.query_select.index(key_modality)
                start2, end2 = num_key_cumsum[key_index], num_key_cumsum[key_index+1]
                sep_key.append(key[:, start2:end2])

                if key_pos is not None:
                    sep_key_pos.append(key_pos[:, start2:end2])
                if value is not None:
                    sep_value.append(value[:, start2:end2])

            sep_query = torch.cat(sep_query, dim=1)
            sep_key = torch.cat(sep_key, dim=1)
            sep_value = torch.cat(sep_value, dim=1) if len(sep_value) else None
            sep_query_pos = torch.cat(sep_query_pos, dim=1) if len(sep_query_pos) else None
            sep_key_pos = torch.cat(sep_key_pos, dim=1) if len(sep_key_pos) else None

            # for none key / value
            if sep_key.size(1) == 0:
                sep_key = None
                sep_key_pos = None
            if sep_value is not None and sep_value.size(1) == 0:
                sep_value = None

            if self.decouple_list[sep]:
                sep_query = torch.cat([sep_query, sep_query_pos], dim=-1)
                sep_query_pos = None
                if sep_key is not None:
                    sep_key = torch.cat([sep_key, sep_key_pos], dim=-1)
                    sep_key_pos = None
                if sep_value is not None:
                    sep_value = fc_before(sep_value)

                output = fc_after(self.attns[sep](query=sep_query, key=sep_key, value=sep_value,
                                                  query_pos=sep_query_pos, key_pos=sep_key_pos))
            else:
                output = self.attns[sep](query=sep_query, key=sep_key, value=sep_value,
                                         query_pos=sep_query_pos, key_pos=sep_key_pos)

            # update query
            num_query_cumsum = np.cumsum([0] + num_query_list)
            for index1, modality in enumerate(sub_query_list):
                index2 = self.query_select.index(modality)
                start1, end1 = num_query_cumsum[index1], num_query_cumsum[index1+1]
                start2, end2 = num_anchor_cumsum[index2], num_anchor_cumsum[index2+1]
                output_instance[:, start2:end2] = output[:, start1:end1]

        return output_instance
