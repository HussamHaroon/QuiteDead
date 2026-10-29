"""Hierarchical RL: option layer, observation shape, bug-fix regression, trained brain runs."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rl"))
from zombie_sim.core.levels import load_levels
from zombie_sim.core.config import Config
from zombie_sim.core.sim import World

LV = load_levels(os.path.join(os.path.dirname(__file__), "..", "config", "levels_ramp.yaml"))


def test_hier_env_obs_and_determinism():
    from henv import HierEnv, OBS_DIM, N_OPT

    def run():
        e = HierEnv(base_seed=3, n_survivors=6, n_zombies=3, size=40, horizon=400, level=LV[1])
        e.reset(); tot = 0.0
        for k in range(20):
            acts = {s.id: (k + s.id) % N_OPT for s in e.world.survivors if s.alive}
            ob, rew, done, _ = e.step(acts); tot += sum(rew.values())
            assert all(o.shape == (OBS_DIM,) for o in ob.values())
            if done:
                break
        return round(tot, 6), round(e.world.t)
    assert run() == run()


def test_taken_supplies_are_forgotten():
    """Regression: update_senses used a stale loop variable, so taken supplies stayed in memory."""
    from env import SurvivorEnv, update_senses
    e = SurvivorEnv(base_seed=1, n_survivors=2, n_zombies=1, size=40, horizon=100, level=LV[1]); e.reset()
    w = e.world; s = w.survivors[0]
    sid = next(iter(w.supplies)); s.memory_sup[sid] = (0.0, 0.0)
    w.supplies[sid].taken = True
    update_senses(w)
    assert sid not in s.memory_sup


def test_rl_hier_brain_runs():
    cfg = Config(**{**Config().to_dict(), "seed": 4, "n_survivors": 6, "n_zombies": LV[1].n_zombies,
                    "width": LV[1].map.w, "height": LV[1].map.h, "n_supply_outdoor": 0,
                    "supply_per_building": 0.0, "brain_survivors": "rl_hier"})
    w = World(cfg, level=LV[1])
    for _ in range(300):
        w.step()
    assert w.counts()[0] >= 1
