# 수정: End-to-End 모델 제어부에 안전제어(Safety Control)를 위한 PPO residual 정책망.
#   설계: docs/residual_ppo_design.md. IL base(HiP-AD planner+PID)는 freeze, 그 출력에
#   작고 bounded 한 보정(residual)만 학습 → a_final = base + clip(Δ_RL).
#   본 파일은 env-독립(순수 PyTorch) 핵심: actor-critic(LSTM) + bounded action.
#   실제 PPO 학습은 폐쇄루프 env(CARLA/Bench2Drive 또는 경량 kinematic sim) + RL 라이브러리 필요.
import torch
import torch.nn as nn


class ResidualActorCritic(nn.Module):
    """부분관측 시계열을 LSTM 으로 처리하는 residual actor-critic.

    Args:
        obs_dim: 관측 차원(ego state + HiP-AD sparse-query feature pooling + base 출력).
        act_dim: residual action 차원(Phase1 control=3 [Δsteer,Δthrottle,Δbrake] /
                 Phase2 planning=2N [Δwaypoints]).
        act_limit: action 별 bound(텐서 또는 스칼라). a = act_limit * tanh(raw).
        hidden: MLP/LSTM hidden 차원.
    """

    def __init__(self, obs_dim, act_dim, act_limit=0.1, hidden=256, lstm_layers=1):
        super().__init__()
        self.act_dim = act_dim
        self.register_buffer(
            "act_limit",
            torch.as_tensor(act_limit, dtype=torch.float32) if not torch.is_tensor(act_limit)
            else act_limit.float(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(hidden, hidden, num_layers=lstm_layers, batch_first=True)
        # actor: 평균(mu) + 상태독립 log_std
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        # critic
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, obs, hx=None):
        """obs: (B, T, obs_dim) 또는 (B, obs_dim). 반환 mu, std, value, hx."""
        single = obs.dim() == 2
        if single:
            obs = obs.unsqueeze(1)  # (B,1,obs_dim)
        feat = self.encoder(obs)
        out, hx = self.lstm(feat, hx)
        mu = self.act_limit * torch.tanh(self.mu_head(out))  # bounded
        std = self.log_std.exp().expand_as(mu)
        value = self.v_head(out).squeeze(-1)
        if single:
            mu, std, value = mu.squeeze(1), std.squeeze(1), value.squeeze(1)
        return mu, std, value, hx

    @torch.no_grad()
    def act(self, obs, hx=None, deterministic=False):
        """rollout 용: action, log_prob, value, hx."""
        mu, std, value, hx = self.forward(obs, hx)
        if deterministic:
            action = mu
            logp = None
        else:
            dist = torch.distributions.Normal(mu, std)
            raw = dist.sample()
            action = raw.clamp(-self.act_limit, self.act_limit)
            logp = dist.log_prob(raw).sum(-1)
        return action, logp, value, hx

    def evaluate_actions(self, obs, actions, hx=None):
        """PPO 갱신용: log_prob, entropy, value."""
        mu, std, value, _ = self.forward(obs, hx)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, entropy, value
