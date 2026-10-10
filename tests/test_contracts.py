"""Section B reward-attribution and curriculum-window invariants."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl"))

import numpy as np

from env import SurvivorEnv, apply_action, auto_consume, update_senses


def _levels():
    from zombie_sim.core.levels import load_levels
    return load_levels(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "config", "levels.yaml"))


def test_all_rewards_reference_start_ids():
    """Every reward key over an episode must be a survivor id present at reset."""
    env = SurvivorEnv(n_survivors=6, n_zombies=8, horizon=3000)
    ob = env.reset()
    start = set(env._start_ids)
    rng = np.random.default_rng(1)
    bad = []
    for _ in range(2000):
        keys_before = set()
        ob, rew, done, info = env.step({aid: int(rng.integers(10)) for aid in ob})
        for k in rew:
            keys_before.add(k)
        for k in keys_before:
            if k not in start:
                bad.append(k)
        if done:
            ob = env.reset()
    assert not bad, f"rewards referenced non-survivor (or post-reset) ids: {bad[:5]}"


def test_quota_win_pays_per_alive_survivor():
    from env import R_QUOTA, R_SURVIVE
    env = SurvivorEnv(n_survivors=6, n_zombies=0, horizon=4000, level=_levels()[0])
    ob = env.reset()
    w = env.world
    # teleport all survivors next to the food pile and let them deposit:
    # force the win and check the payout on a fully-alive squad
    from zombie_sim.core.config import Config  # noqa: F401  (import health)
    x0, y0, sw, sh = w.safehouse
    for s in w.survivors:
        s.x, s.y = x0 + sw / 2.0, y0 + sh / 2.0
    w.stockpile = w.level.win_stock  # force quota condition this step
    ob, rew, done, info = env.step({s.id: 0 for s in w.survivors if s.alive})
    assert done and env.last_stats["quota"] == 1
    alive = w.counts()[0]
    expected_bonus = alive * (R_QUOTA + R_SURVIVE)
    for s in w.survivors:
        if s.alive:
            assert rew.get(s.id, 0.0) >= R_QUOTA + R_SURVIVE - 1.0,                 f"win bonus missing for survivor {s.id}: {rew.get(s.id)}"


def test_promotion_resets_both_windows():
    """The trainer's advance branch must clear recent_surv AND recent_prog."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "rl", "train.py"), encoding="utf-8").read()
    assert "recent_surv.clear()" in src and "recent_prog.clear()" in src, \
        "promotion windows are not reset on advance"
    # and the gate requires consistent wins (>=4 entries, >=0.5 wins) OR survival gate
    assert "mean_prog >= 0.5 or mean_surv >= levels[cur_level].advance_at" in src


def test_eval_rows_carry_trained_labels():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "rl", "train.py")
    src = open(p, encoding="utf-8").read()
    assert 'int(lvl.idx <= cur_level)' in src, "eval rows do not label trained-on levels"
