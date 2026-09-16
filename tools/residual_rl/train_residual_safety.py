# 수정: 차량 제어부 Residual RL(Safety Control) 학습 — 자체 구현 PPO(recurrent), sb3 불필요.
#   env: KinematicSafetyEnv(nuScenes 재생, CARLA 불필요). 정책: ResidualActorCritic(LSTM).
#   학습은 CPU 로 가능(정책 소형). 결과 정책 저장 → 분석 스크립트가 정량·정성 평가.
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, 'projects/mmdet3d_plugin/models/residual_rl')
sys.path.insert(0, 'tools/residual_rl')
import mmcv
from kinematic_safety_env import KinematicSafetyEnv
from residual_policy import ResidualActorCritic
from safety_reward import SafetyReward, REWARD_POLICIES
from scene_adapters import load_scenes


def collect_episode(env, pi, reward_fn, sc, lim, device):
    """한 씬 1 에피소드 rollout. 반환 seq dict."""
    obs = env.reset(sc); hx = None
    O, A, LP, V, R = [], [], [], [], []
    done = False
    while not done:
        ot = torch.tensor(obs, dtype=torch.float32, device=device)[None, None]  # (1,1,obs)
        with torch.no_grad():
            mu, std, val, hx = pi(ot, hx)
        dist = torch.distributions.Normal(mu, std)
        raw = dist.sample()
        logp = dist.log_prob(raw).sum(-1)
        act_env = torch.clamp(raw, -lim, lim)[0, 0].cpu().numpy()
        nobs, info, done = env.step(act_env)
        r = reward_fn.step(info)
        O.append(obs); A.append(raw[0, 0].cpu().numpy()); LP.append(float(logp))
        V.append(float(val)); R.append(r)
        obs = nobs
    return dict(obs=np.array(O, np.float32), act=np.array(A, np.float32),
                logp=np.array(LP, np.float32), val=np.array(V, np.float32),
                rew=np.array(R, np.float32))


def gae(rew, val, gamma=0.99, lam=0.95):
    T = len(rew); adv = np.zeros(T, np.float32); last = 0.0
    for t in reversed(range(T)):
        nv = val[t + 1] if t + 1 < T else 0.0
        delta = rew[t] + gamma * nv - val[t]
        adv[t] = last = delta + gamma * lam * last
    return adv, adv + val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='nusc', choices=['nusc', 'b2d'])
    ap.add_argument('--infos', default='data/infos/nuscenes_infos_train.pkl')
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--eps-per-iter', type=int, default=32)
    ap.add_argument('--ppo-epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--out', default=None, help='기본값: results/residual_safety/<policy>/policy.pt')
    ap.add_argument('--n-scenes', type=int, default=2000)
    # 수정: residual 강화용 — 보상 재가중 + residual 권한
    ap.add_argument('--res-accel', type=float, default=2.0)
    ap.add_argument('--res-steer', type=float, default=0.2)
    #수정(2026-08-11): 보상 가중치 정책 선택. 개별 --w-* 를 주면 그 항만 덮어쓴다.
    #   세 정책은 collision/progress 교환비(200/50/15)로 규정된다 — safety_reward.py 참조.
    #   퇴화 해(정지하면 충돌 0) 검증을 위해 세 정책을 모두 학습·보고한다.
    ap.add_argument('--policy', default='balanced',
                    choices=sorted(REWARD_POLICIES.keys()),
                    help='보상 가중치 정책: safety_first(비200) / balanced(비50) / progress_first(비15)')
    ap.add_argument('--w-collision', type=float, default=None)
    ap.add_argument('--w-safety', type=float, default=None)
    ap.add_argument('--w-tracking', type=float, default=None)
    args = ap.parse_args()
    device = 'cpu'
    #수정: 정책별로 출력 경로를 분리해 세 결과가 서로 덮어쓰지 않게 한다.
    if args.out is None:
        args.out = f'results/residual_safety/{args.policy}/policy.pt'
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    infos = load_scenes(args.dataset, args.infos, horizon=12)[:args.n_scenes]
    print(f"[{args.dataset}] scene {len(infos)}개 로드")
    env = KinematicSafetyEnv(infos, horizon=12, res_bound=(args.res_accel, args.res_steer))
    #수정: 정책에서 가중치를 가져오고, 명시된 --w-* 만 덮어쓴다.
    w = dict(REWARD_POLICIES[args.policy])
    for k, v in (('collision', args.w_collision), ('safety', args.w_safety),
                 ('tracking', args.w_tracking)):
        if v is not None:
            w[k] = v
    reward_fn = SafetyReward(w=w)
    print(f"[reward] policy={args.policy} 충돌/전진 교환비={w['collision']/w['progress']:.0f} "
          f"(충돌 1회 = 전진 {w['collision']/w['progress']:.0f} m 등가, 에피소드 전진량 ~60 m)")
    lim = torch.tensor([args.res_accel, args.res_steer])
    pi = ResidualActorCritic(obs_dim=env.obs_dim, act_dim=2, act_limit=lim).to(device)
    opt = torch.optim.Adam(pi.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)

    for it in range(args.iters):
        # ---- rollout ----
        eps = [collect_episode(env, pi, reward_fn, int(rng.integers(len(infos))), lim, device)
               for _ in range(args.eps_per_iter)]
        for e in eps:
            e['adv'], e['ret'] = gae(e['rew'], e['val'])
        # advantage 정규화
        alladv = np.concatenate([e['adv'] for e in eps])
        amu, astd = alladv.mean(), alladv.std() + 1e-6
        # ---- PPO 갱신 (에피소드=시퀀스, LSTM hx=None 으로 재처리) ----
        for _ in range(args.ppo_epochs):
            losses = []
            opt.zero_grad()
            for e in eps:
                ob = torch.tensor(e['obs'], device=device)[None]      # (1,T,obs)
                ac = torch.tensor(e['act'], device=device)[None]
                old = torch.tensor(e['logp'], device=device)
                adv = torch.tensor((e['adv'] - amu) / astd, device=device)
                ret = torch.tensor(e['ret'], device=device)
                logp, ent, val = pi.evaluate_actions(ob, ac)
                logp, val = logp[0], val[0]
                ratio = torch.exp(logp - old)
                pl = -torch.min(ratio * adv, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv).mean()
                vl = ((val - ret) ** 2).mean()
                loss = pl + 0.5 * vl - 0.01 * ent.mean()
                (loss / len(eps)).backward()
                losses.append(float(loss))
            torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
            opt.step()
        if it % 20 == 0 or it == args.iters - 1:
            coll = np.mean([e_['rew'][-1] < -40 for e_ in eps])  # 충돌 근사(큰 음수 종료)
            mret = np.mean([e['ret'][0] for e in eps])
            print(f"it {it:3d} | mean_return {mret:7.2f} | loss {np.mean(losses):6.3f}")
    torch.save(dict(state=pi.state_dict(), obs_dim=env.obs_dim, act_dim=2,
                    res_bound=[args.res_accel, args.res_steer]), args.out)
    print(f"저장: {args.out}")


if __name__ == '__main__':
    main()
