# https://github.com/Divadi/SOLOFusion/blob/main/mmdet3d/datasets/samplers/infinite_group_each_sample_in_batch_sampler.py
import itertools
import copy

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from torch.utils.data.sampler import Sampler


# https://github.com/open-mmlab/mmdetection/blob/3b72b12fe9b14de906d1363982b9fba05e7d47c1/mmdet/core/utils/dist_utils.py#L157
def sync_random_seed(seed=None, device="cuda"):
    """Make sure different ranks share the same seed.
    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.
    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


class GroupInBatchSampler(Sampler):
    """
    Pardon this horrendous name. Basically, we want every sample to be from its own group.
    If batch size is 4 and # of GPUs is 8, each sample of these 32 should be operating on
    its own group.

    Shuffling is only done for group order, not done within groups.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        world_size=None,
        rank=None,
        seed=0,
        skip_prob=0.,
        sequence_flip_prob=0.,
        num_stream_groups=1,  # 수정(2026-10-15): 실험 C — 1 GPU 에서 "GPU G 개 × batch_size" 의 독립 스트림 등가(아래 설명)
    ):
        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank

        # 수정(2026-10-15): 실험 C — 스트림 그룹. num_stream_groups=G 이면 이 rank 가 batch_size×G 개의 독립 시퀀스 스트림(슬롯)을
        #   유지하고, micro-step 마다 그룹 k(=micro-step 번호 mod G)의 슬롯 batch_size 개만 한 프레임씩 내보낸다.
        #   GradientCumulativeOptimizerHook(cumulative_iters=G) 와 맞추면 optimizer step 1회 = 스트림 batch_size×G 개 × 1 프레임
        #   = SparseDrive 의 "GPU G 개 × batch_size"(각 GPU 가 자기 스트림만 한 프레임씩) 와 같은 표본 구성이다.
        #   그룹 k 의 슬롯 = 전역 표본 위치 (rank×G + k)×batch_size + j — world_size×G 개 가상 GPU 의 rank (rank×G + k) 와 같은 그룹 배정.
        #   G=1 이면 슬롯·난수 호출 순서·내보내는 dict 가 종전과 비트 단위로 같다(stream_group 키도 붙이지 않는다).
        #   시간 인스턴스 뱅크 상태는 그룹별로 디코더가 저장·복원한다(models/decoder/temporal_bank/stream_state.py).
        self.num_stream_groups = int(num_stream_groups)
        assert self.num_stream_groups >= 1
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed(seed)

        self.size = len(self.dataset)

        assert hasattr(self.dataset, "flag")
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = batch_size * world_size * self.num_stream_groups  # 수정(2026-10-15): 실험 C — 전 슬롯 수(G=1 이면 종전 값)
        assert self.groups_num >= self.global_batch_size
        self.num_slots = batch_size * self.num_stream_groups  # 수정(2026-10-15): 실험 C — 이 rank 의 슬롯 수
        self._micro_step = 0  # 수정(2026-10-15): 실험 C — 다음에 내보낼 그룹 번호의 기준

        # Now, for efficiency, make a dict group_idx: List[dataset sample_idxs]
        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)
        }

        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator
        # 수정(2026-10-15): 실험 C — 슬롯 수 batch_size → num_slots(=batch_size×G). G=1 이면 종전과 같다.
        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(
                self.rank * self.num_slots + local_sample_idx
            )
            for local_sample_idx in range(self.num_slots)
        ]

        # Keep track of a buffer of dataset sample idxs for each local sample idx
        self.buffer_per_local_sample = [[] for _ in range(self.num_slots)]
        self.aug_per_local_sample = [None for _ in range(self.num_slots)]
        self.skip_prob = skip_prob
        self.sequence_flip_prob = sequence_flip_prob

    def _infinite_group_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.groups_num, generator=g).tolist()

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        yield from itertools.islice(
            self._infinite_group_indices(),
            global_sample_idx,
            None,
            self.global_batch_size,
        )

    def __iter__(self):
        while True:
            curr_batch = []
            # 수정(2026-10-15): 실험 C — 이번 micro-step 의 그룹 k 슬롯만 순회(G=1 이면 k=0, 슬롯 0..batch_size-1 = 종전).
            k = self._micro_step % self.num_stream_groups
            self._micro_step += 1
            for local_sample_idx in range(k * self.batch_size, (k + 1) * self.batch_size):
                skip = (
                    np.random.uniform() < self.skip_prob
                    and len(self.buffer_per_local_sample[local_sample_idx]) > 1
                )
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    # skip = False
                    new_group_idx = next(
                        self.group_indices_per_global_sample_idx[
                            local_sample_idx
                        ]
                    )
                    self.buffer_per_local_sample[
                        local_sample_idx
                    ] = copy.deepcopy(
                        self.group_idx_to_sample_idxs[new_group_idx]
                    )
                    if np.random.uniform() < self.sequence_flip_prob:
                        self.buffer_per_local_sample[
                            local_sample_idx
                        ] = self.buffer_per_local_sample[local_sample_idx][
                            ::-1
                        ]
                    if self.dataset.keep_consistent_seq_aug:
                        self.aug_per_local_sample[
                            local_sample_idx
                        ] = self.dataset.get_augmentation()

                if not self.dataset.keep_consistent_seq_aug:
                    self.aug_per_local_sample[
                        local_sample_idx
                    ] = self.dataset.get_augmentation()

                if skip:
                    self.buffer_per_local_sample[local_sample_idx].pop(0)
                item = dict(
                    idx=self.buffer_per_local_sample[local_sample_idx].pop(
                        0
                    ),
                    aug_config=self.aug_per_local_sample[local_sample_idx],
                )
                if self.num_stream_groups > 1:  # 수정(2026-10-15): 실험 C — 그룹 번호를 표본에 실어 디코더가 뱅크 상태를 고르게 한다
                    item["stream_group"] = k
                curr_batch.append(item)

            yield curr_batch

    def fast_forward(self, num_batches):
        # 수정(2026-10-15): 재개 시 샘플러 상태 복원 — runner 가 iter K 에서 재개하면 이 rank 의 배치 스트림을
        #   K 배치만큼 "인덱스만" 생성해 앞으로 감는다(데이터 로딩 없음). __iter__ 와 같은 코드 경로를 그대로 돌리므로
        #   _micro_step·슬롯 버퍼·장면 순열 생성기(torch.Generator, seed)·aug 호출 순서가 무중단 런의 K 배치 소비 직후와 같아지고,
        #   다음 배치 = 무중단 런의 iter K 배치(idx·stream_group). 건너뛰는 표본은 없다(무중단 런이 이미 학습한 K 배치를 재생성만).
        #   DataLoader 의 batch_sampler 는 항상 메인 프로세스에서 돈다 → IterLoader(iter(dataloader)) 가 worker prefetch 를
        #   시작하기 전에 호출해야 한다(mmdet_train.py: runner.resume 직후, runner.run 이전).
        num_batches = int(num_batches)
        assert num_batches >= 0
        assert self._micro_step == 0, "fast_forward 는 아직 한 배치도 내보내지 않은 새 샘플러에만 쓴다"
        it = self.__iter__()
        for _ in range(num_batches):
            next(it)
        return self

    def __len__(self):
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch):
        self.epoch = epoch
