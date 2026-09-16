# 수정: Residual RL 안전제어 정량·정성 분석. base(residual OFF) vs base+residual(ON) 비교.
#   정량: 충돌률·안전마진위반·평균 표면거리·comfort(jerk)·추종오차·완주율.
#   정성: residual 이 충돌/근접을 회피하는 궤적 시각화(figure 저장).
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, 'projects/mmdet3d_plugin/models/residual_rl')
sys.path.insert(0, 'tools/residual_rl')
import mmcv
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from kinematic_safety_env import KinematicSafetyEnv
from residual_policy import ResidualActorCritic
from scene_adapters import load_scenes


def run_scene(env, sc, pi, lim, use_res):
    obs = env.reset(sc); hx = None
    traj = [env.ego[:2].copy()]
    surfs, jerks, devs = [], [], []
    coll = viol = comp = False
    done = False
    while not done:
        if use_res:
            with torch.no_grad():
                a, _, _, hx = pi.act(torch.tensor(obs)[None], hx, deterministic=True)
            act = a[0].numpy()
        else:
            act = np.zeros(2, np.float32)
        obs, info, done = env.step(act)
        traj.append(env.ego[:2].copy())
        if np.isfinite(info['min_surface']): surfs.append(info['min_surface'])
        jerks.append(info['jerk']); devs.append(info['lat_dev'])
        coll |= info['collision']; viol |= info['safety_violation']; comp = info['done_completed']
    return dict(traj=np.array(traj), coll=coll, viol=viol, comp=comp,
                min_surf=min(surfs) if surfs else np.inf,
                jerk=float(np.mean(jerks)), dev=float(np.mean(devs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='nusc', choices=['nusc', 'b2d'])
    ap.add_argument('--infos', default='data/infos/nuscenes_infos_val.pkl')
    ap.add_argument('--policy', default='work_dirs/residual_safety/policy.pt')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--out-dir', default='work_dirs/residual_safety')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ck = torch.load(args.policy, map_location='cpu')
    rb = ck.get('res_bound', [2.0, 0.2])  # 학습 시 residual bound 일치
    infos = load_scenes(args.dataset, args.infos, horizon=12)[:args.n]
    print(f"[{args.dataset}] scene {len(infos)}개 | res_bound={rb}")
    env = KinematicSafetyEnv(infos, horizon=12, res_bound=tuple(rb))
    lim = torch.tensor(rb)
    pi = ResidualActorCritic(obs_dim=ck['obs_dim'], act_dim=ck['act_dim'], act_limit=lim)
    pi.load_state_dict(ck['state']); pi.eval()

    res = {'OFF': [], 'ON': []}
    for sc in range(len(infos)):
        res['OFF'].append(run_scene(env, sc, pi, lim, False))
        res['ON'].append(run_scene(env, sc, pi, lim, True))

    def agg(rs):
        return dict(
            collision=100 * np.mean([r['coll'] for r in rs]),
            violation=100 * np.mean([r['viol'] for r in rs]),
            completion=100 * np.mean([r['comp'] for r in rs]),
            min_surf=np.mean([r['min_surf'] for r in rs if np.isfinite(r['min_surf'])]),
            jerk=np.mean([r['jerk'] for r in rs]),
            dev=np.mean([r['dev'] for r in rs]))

    a_off, a_on = agg(res['OFF']), agg(res['ON'])
    print(f"\n=== Residual RL 안전제어 정량 분석 ({args.dataset} val {len(infos)}씬) ===")
    print(f"{'지표':<22}{'Residual OFF':>14}{'Residual ON':>14}")
    rows = [('충돌률(%) ↓', 'collision'), ('안전마진위반(%) ↓', 'violation'),
            ('완주율(%) ↑', 'completion'), ('평균 표면거리(m) ↑', 'min_surf'),
            ('comfort jerk ↓', 'jerk'), ('추종 횡편차(m) ↓', 'dev')]
    for name, k in rows:
        print(f"{name:<22}{a_off[k]:>14.3f}{a_on[k]:>14.3f}")

    # ---- 정성: base 충돌 but residual 회피 씬 ----
    cand = [i for i in range(len(infos)) if res['OFF'][i]['viol'] and not res['ON'][i]['viol']]
    cand = cand or [int(np.argmin([r['min_surf'] for r in res['OFF']]))]
    sc = cand[0]
    env.reset(sc)
    fig, ax = plt.subplots(figsize=(7, 7))
    for a in env.agents:
        traj = a['traj']
        ax.plot(traj[:, 0], traj[:, 1], 'r.-', ms=3, alpha=0.5)
        ax.scatter(traj[0, 0], traj[0, 1], c='r', s=20)
    to = res['OFF'][sc]['traj']; tn = res['ON'][sc]['traj']
    ax.plot(to[:, 0], to[:, 1], 'b--o', ms=3, label='base (residual OFF)')
    ax.plot(tn[:, 0], tn[:, 1], 'g-o', ms=3, label='base+residual (ON)')
    ax.plot(env.ref[:, 0], env.ref[:, 1], 'k:', alpha=0.4, label='reference')
    ax.set_title(f'Residual safety control (scene {sc})\nOFF surf={res["OFF"][sc]["min_surf"]:.1f}m ON surf={res["ON"][sc]["min_surf"]:.1f}m')
    ax.legend(); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.axis('equal'); ax.grid(alpha=.3)
    fig.savefig(f'{args.out_dir}/qualitative.png', dpi=100, bbox_inches='tight')
    print(f"\n정성 시각화 저장: {args.out_dir}/qualitative.png (scene {sc})")
    # 결과 저장
    mmcv.dump(dict(off=a_off, on=a_on), f'{args.out_dir}/metrics.json')


if __name__ == '__main__':
    main()
