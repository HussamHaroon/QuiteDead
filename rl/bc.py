"""Behaviour cloning: warm-start the PPO policy by imitating the scripted ladder (DAgger-free, offline).

  python -m rl.bc --levels config/levels_ramp.yaml --out runs/bc/policy.pt

Demonstrations: run the scripted survivors on every level, record (egocentric obs, discrete action)
where the action is the 8-way direction of the survivor's actual displacement (or ENGAGE when the
ladder attacks, HOLD when it stands still). The network is then trained with cross-entropy.
The result is evaluated with the same harness as PPO and can seed `rl.train --init-from`.
"""
import argparse, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from rl.env import SurvivorEnv, obs_for, update_senses, auto_consume, OBS_DIM, ACT_DIM, DIRS
from rl.ppo import ActorCritic
from zombie_sim.core.levels import load_levels


def _dir_action(dx, dy):
    if dx * dx + dy * dy < 1e-4:
        return 0
    ang = math.atan2(dy, dx)
    best, ba = 1, 9.0
    for i, (ddx, ddy) in enumerate(DIRS):
        d = abs((ang - math.atan2(ddy, ddx) + math.pi) % (2 * math.pi) - math.pi)
        if d < ba:
            best, ba = i + 1, d
    return best


def collect(levels, eps_per_level, n_surv, horizon, seed):
    X, Y = [], []
    for lvl in levels:
        for ep in range(eps_per_level):
            env = SurvivorEnv(base_seed=seed + ep, env_index=lvl.idx, n_survivors=n_surv,
                              n_zombies=lvl.n_zombies, size=lvl.map.w, horizon=horizon, level=lvl)
            env.reset()
            w = env.world
            w.cfg.brain_survivors = "utility"
            for s in w.survivors:
                s.rl = False
            for t in range(horizon):
                update_senses(w)
                before = {s.id: (s.x, s.y, obs_for(w, s), s.cooldown) for s in w.survivors if s.alive}
                w.step()
                auto_consume(w)
                for s in w.survivors:
                    if s.id in before:
                        x0, y0, ob, cd0 = before[s.id]
                        a = 9 if s.cooldown > cd0 else _dir_action(s.x - x0, s.y - y0)
                        X.append(ob); Y.append(a)
                if w.counts()[0] == 0 or w.stockpile >= lvl.win_stock:
                    break
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)


def evaluate(policy, levels, n_surv, episodes, deterministic):
    out = {}
    for lvl in levels:
        wins = 0
        for ep in range(episodes):
            env = SurvivorEnv(base_seed=900 + ep, env_index=lvl.idx, n_survivors=n_surv,
                              n_zombies=lvl.n_zombies, size=lvl.map.w, horizon=4000, level=lvl)
            env.reset(); done = False
            while not done:
                w = env.world; acts = {}
                for s in w.survivors:
                    if s.alive:
                        a, _, _ = policy.act(obs_for(w, s)[None], deterministic=deterministic)
                        acts[s.id] = int(a[0])
                _, _, done, _ = env.step(acts)
            wins += env.last_stats["quota"]
        out[lvl.name] = wins / episodes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="config/levels_ramp.yaml")
    ap.add_argument("--out", default="runs/bc/policy.pt")
    ap.add_argument("--eps", type=int, default=12)
    ap.add_argument("--survivors", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-eps", type=int, default=10)
    a = ap.parse_args()
    levels = load_levels(a.levels)
    t0 = time.time()
    X, Y = collect(levels, a.eps, a.survivors, 3000, 5)
    print(f"collected {len(X)} samples in {time.time()-t0:.0f}s; action histogram {np.bincount(Y, minlength=ACT_DIM)}", flush=True)
    torch.manual_seed(0)
    net = ActorCritic(OBS_DIM, ACT_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    cnt = np.bincount(Y, minlength=ACT_DIM).astype(np.float32) + 1
    wts = torch.as_tensor((cnt.sum() / cnt) ** 0.5); wts = wts / wts.mean()
    n = len(X)
    for ep in range(a.epochs):
        perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, 512):
            idx = perm[i:i+512]
            logits = net.pi(net.trunk(Xt[idx]))
            loss = F.cross_entropy(logits, Yt[idx], weight=wts)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        with torch.no_grad():
            acc = (net.pi(net.trunk(Xt)).argmax(-1) == Yt).float().mean().item()
        print(f"epoch {ep+1} loss {tot/n:.3f} acc {acc:.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(net.state_dict(), a.out)
    for det in (False, True):
        print("eval", "deterministic" if det else "stochastic", evaluate(net, levels, a.survivors, a.eval_eps, det), flush=True)


if __name__ == "__main__":
    main()
