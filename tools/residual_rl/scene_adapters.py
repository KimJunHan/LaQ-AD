# 수정: Residual RL kinematic env 용 통합 scene 어댑터.
#   nuScenes(미래궤적 precomputed)와 Bench2Drive(프레임 단위 → 윈도우+좌표변환) 를
#   동일한 scene-dict 포맷으로 변환 → KinematicSafetyEnv 가 둘 다 사용.
#   scene-dict 키: gt_ego_fut_trajs(H,2 델타), gt_boxes(N,7), gt_agent_fut_trajs(N,H,2 델타),
#                  gt_agent_fut_masks(N,H), ego_status.
import numpy as np
from collections import defaultdict
import mmcv


def load_scenes(dataset, infos_path, horizon=12, **kw):
    if dataset == 'nusc':
        d = mmcv.load(infos_path)
        return d['infos'] if isinstance(d, dict) and 'infos' in d else d
    elif dataset == 'b2d':
        d = mmcv.load(infos_path)
        infos = d['infos'] if isinstance(d, dict) and 'infos' in d else d
        return build_b2d_scenes(infos, horizon=horizon, **kw)
    raise ValueError(dataset)


def build_b2d_scenes(infos, horizon=12, stride=5):
    """Bench2Drive 프레임 단위 info → 윈도우별 scene-dict(nuScenes 포맷).
    ego/agent 미래를 frame-i 의 ego 프레임으로 변환. agent 는 gt_ids 로 추적."""
    g = defaultdict(list)
    for s in infos:
        g[s['folder']].append(s)
    scenes = []
    for folder, frames in g.items():
        frames.sort(key=lambda s: s['frame_idx'])
        for i in range(0, max(0, len(frames) - horizon), stride):
            f0 = frames[i]
            W0 = np.asarray(f0['world2ego'], dtype=np.float64)  # world->ego(frame i)

            # --- ego reference (frame-i 프레임, 델타) ---
            ego_i = []
            for k in range(horizon + 1):
                p = np.asarray(frames[i + k]['ego_translation'], dtype=np.float64)
                ego_i.append((W0 @ np.append(p, 1.0))[:2])
            ego_fut = np.diff(np.array(ego_i), axis=0).astype(np.float32)  # (H,2)

            # --- agent 추적: id -> {k: xy_in_frame_i} ---
            atraj = defaultdict(dict)
            for k in range(horizon + 1):
                fk = frames[i + k]
                Tk = W0 @ np.linalg.inv(np.asarray(fk['world2ego'], dtype=np.float64))  # frame-k ego -> frame-i ego
                bk = np.asarray(fk['gt_boxes'], dtype=np.float64)
                idk = np.asarray(fk['gt_ids'])
                for j, aid in enumerate(idk):
                    xyz = np.append(bk[j, :3], 1.0)
                    atraj[int(aid)][k] = (Tk @ xyz)[:2]

            boxes0 = np.asarray(f0['gt_boxes'], dtype=np.float64)
            ids0 = np.asarray(f0['gt_ids'])
            gt_boxes, fut, masks = [], [], []
            for j, aid in enumerate(ids0):
                aid = int(aid)
                start = atraj[aid][0]
                pos, m = [], []
                last = start
                for k in range(1, horizon + 1):
                    if k in atraj[aid]:
                        last = atraj[aid][k]; m.append(1.0)
                    else:
                        m.append(0.0)
                    pos.append(last)
                pos = np.array(pos)                        # (H,2) 절대(frame-i)
                prev = np.concatenate([start[None], pos[:-1]], axis=0)
                deltas = (pos - prev).astype(np.float32)   # (H,2) 델타
                w, l, h = boxes0[j, 3], boxes0[j, 4], boxes0[j, 5]
                yaw = boxes0[j, 6] if boxes0.shape[1] > 6 else 0.0
                gt_boxes.append([start[0], start[1], 0.0, w, l, h, yaw])
                fut.append(deltas); masks.append(m)

            scenes.append(dict(
                gt_ego_fut_trajs=ego_fut,
                gt_boxes=np.array(gt_boxes, np.float32) if gt_boxes else np.zeros((0, 7), np.float32),
                gt_agent_fut_trajs=np.array(fut, np.float32) if fut else np.zeros((0, horizon, 2), np.float32),
                gt_agent_fut_masks=np.array(masks, np.float32) if masks else np.zeros((0, horizon), np.float32),
                ego_status=np.asarray(f0['ego_vel'], np.float32),
            ))
    return scenes
