"""Evaluate a hierarchical policy: python rl/heval.py policy.pt [seeds]  (stochastic and deterministic)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from henv import HierEnv, obs_for, OBS_DIM
from options import N_OPT
from ppo import ActorCritic
from zombie_sim.core.levels import load_levels
from analysis.stats import wilson_ci

def run(policy, lvl, seeds, det):
    wins = 0; stock = []; surv = []
    for ep in range(seeds):
        env = HierEnv(base_seed=int(os.environ.get("SEED0", 900)) + ep, env_index=lvl.idx, n_survivors=6, n_zombies=lvl.n_zombies,
                      size=lvl.map.w, horizon=4000, level=lvl)
        env.reset(); done = False
        while not done:
            w = env.world; acts = {}
            for s in w.survivors:
                if s.alive:
                    a, _, _ = policy.act(obs_for(w, s)[None], deterministic=det); acts[s.id] = int(a[0])
            _, _, done, _ = env.step(acts)
        st = env.last_stats; wins += st["quota"]; stock.append(st["stockpile"]); surv.append(st["survivors"] / 6)
    lo, hi = wilson_ci(wins, seeds)
    return wins / seeds, lo, hi, float(np.mean(stock)), float(np.mean(surv))

if __name__ == "__main__":
    pol = ActorCritic(OBS_DIM, N_OPT); pol.load_state_dict(torch.load(sys.argv[1], map_location="cpu"))
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    lv = load_levels(os.environ.get("LEVELS", "config/levels_ramp.yaml"))
    for det in (False, True):
        for L in range(len(lv)):
            r = run(pol, lv[L], n, det)
            print(f"{'det ' if det else 'stoch'},{lv[L].name},{r[0]:.2f},{r[1]:.2f},{r[2]:.2f},{r[3]:.1f},{r[4]:.2f}", flush=True)
