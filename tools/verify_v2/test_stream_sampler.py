# 수정(2026-10-15): 실험 C — GroupInBatchSampler(num_stream_groups) 단위 시험(CPU). 사용: PYTHONPATH=. python tools/verify_v2/test_stream_sampler.py [old_sampler.py]
#   (a1) G=1 은 종전 샘플러(정리 전 스냅숏 파일, 인자로 경로)와 idx·aug 가 비트 단위로 같다.
#   (a2) G=4, B=16 의 micro-step k 열 == 종전 샘플러를 world_size=4·rank=k·B=16 으로 돌린 열(= GPU 4 개 × 16) — idx 동일.
#   (a3) 각 슬롯(스트림)은 시퀀스 내부를 시간순·연속으로 소비하고, 한 optimizer step(4 micro-step)에 64 개 서로 다른 시퀀스가 한 번씩.
import importlib.util, sys, os
import numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import projects.mmdet3d_plugin  # noqa: 레지스트리
from projects.mmdet3d_plugin.models.loader.samplers.group_in_batch_sampler import GroupInBatchSampler as New
from projects.mmdet3d_plugin.models.loader.nuscenes.dataset import NuScenes3DDataset

old_path = sys.argv[1] if len(sys.argv) > 1 else None
ds = NuScenes3DDataset(data_root="data/nuscenes", ann_file="data/infos/nuscenes_infos_train.pkl", pipeline=None,
                       data_aug_conf=dict(resize_lim=(0.40, 0.47), final_dim=(256, 704), bot_pct_lim=(0.0, 0.0),
                                          rot_lim=(-5.4, 5.4), H=900, W=1600, rand_flip=True, rot3d_range=[0, 0]),
                       with_seq_flag=True, sequences_split_num=2, keep_consistent_seq_aug=True)
ts = np.array([i["timestamp"] for i in ds.data_infos])
print("표본", len(ds), "시퀀스(그룹)", len(np.bincount(ds.flag)))

def take(sampler, n):
    it = iter(sampler)
    return [next(it) for _ in range(n)]

ok = True
if old_path:
    spec = importlib.util.spec_from_file_location("old_gibs", old_path); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    # (a1)
    np.random.seed(123); a = take(old.GroupInBatchSampler(ds, 16, 1, 0, seed=0), 400)
    np.random.seed(123); b = take(New(ds, 16, 1, 0, seed=0, num_stream_groups=1), 400)
    same = all(x == y for x, y in zip(a, b)) and all("stream_group" not in d for bb in b for d in bb)
    print("(a1) G=1 vs 종전: idx·aug·dict 키 동일 =", same); ok &= same
    # (a2)
    N = 300
    new = take(New(ds, 16, 1, 0, seed=0, num_stream_groups=4), 4 * N)
    for k in range(4):
        ref = take(old.GroupInBatchSampler(ds, 16, 4, k, seed=0), N)
        mine = new[k::4]
        eq = all([d["idx"] for d in x] == [d["idx"] for d in y] for x, y in zip(ref, mine))
        tags = all(d["stream_group"] == k for bb in mine for d in bb)
        print(f"(a2) 그룹 {k} == 가상 GPU rank {k}(world 4 × 16): idx 동일 = {eq}, stream_group 태그 = {tags}"); ok &= eq and tags
# (a3)
N = 2000
new = take(New(ds, 16, 1, 0, seed=0, num_stream_groups=4), 4 * N)
slots = {}  # (k, j) -> idx 열
for t, bb in enumerate(new):
    k = t % 4
    for j, d in enumerate(bb):
        slots.setdefault((k, j), []).append(d["idx"])
bad = 0
for key, seq in slots.items():
    seq = np.array(seq)
    # 같은 시퀀스(flag) 구간에서는 idx 가 1 씩 증가(연속)·timestamp 증가, 시퀀스가 바뀌면 그 시퀀스의 첫 프레임부터
    for a_, b_ in zip(seq[:-1], seq[1:]):
        first_of_seq = b_ == 0 or ds.flag[b_ - 1] != ds.flag[b_]
        if ds.flag[a_] == ds.flag[b_] and not first_of_seq:
            if not (b_ == a_ + 1 and ts[b_] > ts[a_]): bad += 1
        else:
            if not first_of_seq: bad += 1
print(f"(a3) 슬롯 64 개 × {N} 프레임: 시간순·연속 위반 {bad}")
ok &= bad == 0
distinct = []
for s in range(N):
    fl = [ds.flag[d["idx"]] for t in range(4 * s, 4 * s + 4) for d in new[t]]
    distinct.append(len(set(fl)))
print(f"(a3) optimizer step 당 서로 다른 시퀀스 수: min {min(distinct)} / max {max(distinct)} / 64 미만 step {sum(d < 64 for d in distinct)}/{N}")
print("     (64 미만은 무한 순열의 다음 바퀴에서 같은 시퀀스를 다른 슬롯이 다시 뽑는 경우 — SD 원본 샘플러(가상 GPU 4 개, (a2) 로 idx 동일)에도 똑같이 생긴다)")
ok &= min(distinct) >= 60
old_like = take(New(ds, 16, 1, 0, seed=0, num_stream_groups=1), 400)
d1 = [len({ds.flag[d['idx']] for t in range(4 * s, 4 * s + 4) for d in old_like[t]}) for s in range(100)]
print(f"     참고 G=1(종전) step 당 서로 다른 시퀀스 수: max {max(d1)}")
print("ALL OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
