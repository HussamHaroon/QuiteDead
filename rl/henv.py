"""Hierarchical survivor env: the policy chooses an option every K ticks; scripted code executes it."""
from env import *          # noqa: F401,F403
import math
import numpy as np
from env import SurvivorEnv, apply_action
from env import obs_for as _base_obs, OBS_DIM as _BASE_DIM
from zombie_sim.core.brains import utility as _lad
from options import run_option, N_OPT, NAMES   # noqa: F401

EXTRA = 7
OBS_DIM = _BASE_DIM + EXTRA   # the flat view lacks what an option-chooser needs: inventory, bank, home


def obs_for(world, s):
    base = _base_obs(world, s)
    lvl = world.level
    quota = float(lvl.win_stock) if lvl is not None else 10.0
    hx = hy = 0.0
    dist = 0.0
    if world.safehouse is not None:
        x0, y0, sw, sh = world.safehouse
        dist = math.hypot(s.x - (x0 + sw / 2.0), s.y - (y0 + sh / 2.0)) / 30.0
    ex = np.array([
        _lad._carried(s) / 6.0,
        1.0 if _lad._has_useful(world, s) else 0.0,
        1.0 if any(world.supplies[k].kind == "gun" for k in s.memory_sup) else 0.0,
        1.0 if (world.safehouse is not None and world.in_safehouse(s)) else 0.0,
        min(1.0, world.stockpile / quota),
        min(1.0, dist),
        min(1.0, len(s.memory_sup) / 8.0)], dtype=np.float32)
    return np.concatenate([base, ex])


K = 6
import env as _env
_env.R_LOOT = 2.0   # looting pays enough to bootstrap exploration (0.5 never learned)
_env.R_DEPOSIT = 5.0  # but banking pays more, so "loot then return" beats "loot forever"


class HierEnv(SurvivorEnv):
    def _apply(self, w, actions):
        self._opts = getattr(self, "_opts", {})
        for sid, a in actions.items():
            self._opts[sid] = int(a)
        for s in w.survivors:
            if s.alive:
                run_option(w, s, self._opts.get(s.id, 0), apply_action)

    def obs(self):
        return {s.id: obs_for(self.world, s) for s in self.world.survivors if s.alive}

    def reset(self):
        self._opts = {}
        return super().reset()

    def step(self, actions):
        self._opts = {}
        self._opts.update({sid: int(a) for sid, a in actions.items()})
        tot = {}
        first = True
        for _ in range(K):
            obs, rew, done, info = super().step({} if not first else actions)
            first = False
            for sid, r in rew.items():
                tot[sid] = tot.get(sid, 0.0) + r
            if done:
                break
        return obs, tot, done, info
