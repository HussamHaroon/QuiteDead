"""QuiteDead test suite — enforces the Definition of Done (see DONE.md).

Run:  python -m pytest tests/ -x -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from zombie_sim.core.config import Config
from zombie_sim.core.levels import load_levels
from zombie_sim.core.sim import World


def make_world(**kw):
    cfg = Config(**kw)
    return World(cfg)


# --------------------------------------------------------------- determinism
def test_determinism_short(tmp_path):
    a = make_world(seed=42, n_survivors=6, n_zombies=3)
    b = make_world(seed=42, n_survivors=6, n_zombies=3)
    for _ in range(900):
        a.step()
        b.step()
    assert a.counts() == b.counts()
    assert len(a.log.events) == len(b.log.events)
    assert [(e["t"], e["type"]) for e in a.log.events] == \
           [(e["t"], e["type"]) for e in b.log.events]


# ------------------------------------------------------------ episode + win
def test_quota_win_is_terminal():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl"))
    from env import SurvivorEnv
    levels = load_levels(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "config", "levels.yaml"))
    env = SurvivorEnv(n_survivors=6, level=levels[0], horizon=3000)
    env.reset()
    # cheat the win: fill the stockpile directly
    env.world.stockpile = levels[0].win_stock
    ob, rew, done, info = env.step({})
    assert done, "quota met must end the episode"
    assert info.get("truncated") in (False,), "a quota win is a true terminal, not a truncation"


def test_extinction_is_loss():
    import sys
    rl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl")
    if rl_dir not in sys.path:
        sys.path.insert(0, rl_dir)
    from env import SurvivorEnv
    levels = load_levels(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "config", "levels.yaml"))
    env = SurvivorEnv(n_survivors=6, n_zombies=30, horizon=3000)
    ob = env.reset()
    rng = np.random.default_rng(0)
    done = False
    for _ in range(4000):
        ob, rew, done, info = env.step({aid: int(rng.integers(10)) for aid in ob})
        if done:
            break
    assert done
    assert env.world.counts()[0] == 0, "30 zombies should wipe 6 random survivors"


# --------------------------------------------------------------- observation
def test_obs_dim_and_finiteness():
    import sys
    rl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl")
    if rl_dir not in sys.path:
        sys.path.insert(0, rl_dir)
    from env import SurvivorEnv, OBS_DIM
    env = SurvivorEnv(n_survivors=4, n_zombies=4, horizon=500)
    ob = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(400):
        for aid, o in ob.items():
            assert np.isfinite(o).all(), "observation contains NaN/inf"
        ob, rew, done, info = env.step({aid: int(rng.integers(10)) for aid in ob})
        if done:
            ob = env.reset()
    assert len(next(iter(ob.values()))) == OBS_DIM


# ------------------------------------------------------------- supply kinds
SUPPLY_KINDS = ("food", "water", "meds", "gun", "ammo", "bat")


def test_all_supply_kinds_have_render_fallback():
    from render.viewer import SUPPLY_COLORS
    for kind in SUPPLY_KINDS:
        assert kind in SUPPLY_COLORS, f"SUPPLY_COLORS missing {kind}"


def test_all_supply_kinds_loot_path():
    from zombie_sim.core import systems
    src = open(systems.__file__, encoding="utf-8").read()
    for kind in ("gun", "ammo", "bat"):
        assert f'"{kind}"' in src, f"loot_tick has no branch for {kind}"


def test_levels_have_all_kinds_or_bats_guns_ammo():
    levels = load_levels(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "config", "levels.yaml"))
    kinds = {k for lv in levels for _, _, k in lv.supplies}
    for kind in SUPPLY_KINDS:
        assert kind in kinds, f"no {kind} placed on any level"


# ------------------------------------------------------------ weapons ladder
def test_unarmed_cannot_kill():
    import sys
    rl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl")
    if rl_dir not in sys.path:
        sys.path.insert(0, rl_dir)
    from env import apply_action
    import math
    lv = load_levels(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "config", "levels.yaml"))[0]
    w = make_world(width=lv.map.w, height=lv.map.h, n_survivors=1, n_zombies=1,
                   n_supply_outdoor=0, supply_per_building=0.0, seed=1)
    w.level = lv
    s = w.survivors[0]
    z = w.zombies[0]
    z.hp = 5.0
    s.x, s.y = z.x - 1.0, z.y  # in melee reach
    s.cooldown = 0
    apply_action(w, s, 9)  # engage
    for _ in range(20):
        w.step()
    assert not s.gun and not s.has_bat
    assert z.alive, "unarmed survivor killed a zombie — weapon ladder violated"


# --------------------------------------------------------------- infection
def test_bite_infects_and_turns():
    lv = load_levels(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "config", "levels.yaml"))[1]
    cfg = Config(width=lv.map.w, height=lv.map.h, n_survivors=1, n_zombies=1,
                 n_supply_outdoor=0, supply_per_building=0.0,
                 reanimation_enabled=True, seed=3)
    w = World(cfg, level=lv)
    s, z = w.survivors[0], w.zombies[0]
    s.x, s.y = 20.5, 20.5
    z.x, z.y = 21.5, 20.5
    z.heading = 3.14159
    bitten = turned = False
    for _ in range(2000):
        w.step()
        if w.log.count("bite"):
            bitten = True
        if w.log.count("turn"):
            turned = True
            break
    assert bitten and turned, "bite->infection->turn pipeline broken"


# ------------------------------------------------------------- reinforcement
def test_reinforcement_never_spawns_in_safehouse():
    cfg = Config(width=48, height=48, n_survivors=6, n_zombies=4,
                 reinforce_every_ticks=300, seed=9)
    w = World(cfg)
    x0, y0, sw, sh = w.safehouse
    w.t = 299
    for _ in range(60):
        w.step()
        for z in w.zombies:
            if z.alive and x0 <= int(z.x) < x0 + sw and y0 <= int(z.y) < y0 + sh:
                # zombies born at authored spawns far away can wander in later;
                # only *newly reinforced* ones materialize inside
                pass
    # direct check: every reinforcement event tile is outside the safehouse
    for e in w.log.events:
        if e["type"] == "reinforce":
            assert not (x0 <= int(float(e["x"])) < x0 + sw and
                        y0 <= int(float(e["y"])) < y0 + sh), \
                "reinforcement spawned inside the safe house"


# ------------------------------------------------------------- levels parse
def test_levels_parse_and_validate():
    levels = load_levels(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "config", "levels.yaml"))
    assert len(levels) >= 5
    for lv in levels:
        assert lv.win_stock > 0
        assert lv.safehouse is not None or lv.map is not None


# ------------------------------------------------------------- import health
def test_core_has_no_render_imports():
    import zombie_sim.core.sim
    src = open(zombie_sim.core.sim.__file__, encoding="utf-8").read()
    assert "import pygame" not in src and "from render" not in src
