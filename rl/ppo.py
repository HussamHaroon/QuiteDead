"""Compact single-file PPO (clipped surrogate, GAE, parameter sharing).

Written from scratch on purpose: no gym/SB3 version coupling, and the whole
learning rule is inspectable for the course report.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.orthogonal_(self.v.weight)

    def forward(self, obs):
        h = self.trunk(obs)
        return Categorical(logits=self.pi(h)), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_np, deterministic=False):
        obs = torch.as_tensor(obs_np, dtype=torch.float32)
        dist, val = self.forward(obs)
        if deterministic:
            a = torch.argmax(dist.logits, dim=-1)
        else:
            a = dist.sample()
        return a.numpy(), val.numpy(), dist.log_prob(a).numpy()


def gae_per_trajectory(rewards, values, dones, last_value, gamma, lam):
    adv = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        nxt = last_value if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * nxt * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + values


class PPO:
    def __init__(self, obs_dim, act_dim, lr=3e-4, clip=0.2, epochs=4,
                 minibatch=1024, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 gamma=0.995, lam=0.95, device="cpu", seed=0):
        torch.manual_seed(seed)
        self.net = ActorCritic(obs_dim, act_dim).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.clip, self.epochs, self.minibatch = clip, epochs, minibatch
        self.ent_coef, self.vf_coef, self.max_grad_norm = ent_coef, vf_coef, max_grad_norm
        self.gamma, self.lam, self.device = gamma, lam, device

    def update(self, obs, act, adv, ret, logp_old):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.long, device=self.device)
        adv = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
        logp_old = torch.as_tensor(logp_old, dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(obs)
        stats = {"loss": 0.0, "pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0,
                 "clipfrac": 0.0, "kl": 0.0}
        nbatch = 0
        idx_all = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n, self.minibatch):
                idx = idx_all[start:start + self.minibatch]
                if len(idx) < 8:
                    continue
                dist, val = self.net(obs[idx])
                logp = dist.log_prob(act[idx])
                ratio = torch.exp(logp - logp_old[idx])
                clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                pi_loss = -torch.min(ratio * adv[idx], clipped * adv[idx]).mean()
                v_loss = ((val - ret[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    stats["clipfrac"] += ((ratio - 1).abs() > self.clip).float().mean().item()
                    stats["kl"] += (logp_old[idx] - logp).mean().item()
                stats["loss"] += loss.item()
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                nbatch += 1
        if nbatch:
            for k in stats:
                stats[k] /= nbatch
        return stats
