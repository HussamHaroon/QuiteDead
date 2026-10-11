"""D5 — zombie RL env + trainer smoke (DoD §D5)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl"))

import numpy as np


def test_zombie_env_obs_and_episodes():
    from zenv import ZombieEnv, OBS_DIM, ACT_DIM
    from zombie_sim.core.levels import load_levels
    lv = load_levels(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "config", "levels.yaml"))[1]
    env = ZombieEnv(n_survivors=6, n_zombies=6, horizon=600, level=lv)
    ob = env.reset()
    assert len(next(iter(ob.values()))) == OBS_DIM
    rng = np.random.default_rng(0)
    episodes = 0
    for _ in range(1500):
        for aid, o in ob.items():
            assert np.isfinite(o).all(), "zombie observation contains NaN/inf"
        ob, rew, done, info = env.step({aid: int(rng.integers(ACT_DIM)) for aid in ob})
        if done:
            episodes += 1
            ob = env.reset()
    assert episodes >= 2, "expected several episode cycles in 1500 ticks"


def test_zombie_trainer_sanity(tmp_path):
    import subprocess
    r = subprocess.run(
        [sys.executable, "ztrain.py", "--updates", "2",
         "--run-dir", os.path.join(str(tmp_path), "sanity_z")],
        cwd=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl"),
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    assert (tmp_path / "sanity_z" / "metrics.csv").exists()
