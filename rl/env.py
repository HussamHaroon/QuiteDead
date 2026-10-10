"""Survivor RL environment: N parameter-shared survivor agents vs scripted FSM zombies.

Own minimal API (no gym dependency): reset(seed) -> {agent_id: obs},
step({agent_id: action}) -> (obs, rewards, done, info). One policy controls
every survivor; each survivor's trajectory is tracked separately for GAE.
"""
import math

import numpy as np

from zombie_sim.core.config import Config
from zombie_sim.core.perception import can_see
from zombie_sim.core.sim import World

PATCH = 9                       # local wall patch (PATCH x PATCH)
SELF_DIM = 7
Z_VIS_DIM = 4                   # exists, dist, sin, cos
Z_MEM_DIM = 5                   # exists, dist, sin, cos, staleness
SUP_DIM = 4
PEER_DIM = 3
OBS_DIM = SELF_DIM + Z_VIS_DIM + Z_MEM_DIM + SUP_DIM + PEER_DIM + PATCH * PATCH  # 104
ACT_DIM = 10                    # 0 stay, 1-8 move dirs, 9 attack
# screen-consistent move directions (N, NE, E, SE, S, SW, W, NW) in tile coords
DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]

R_LOOT = 2.0
R_KILL = 3.0
R_BITE = -4.0
R_DEATH = -10.0
R_STEP = -0.002
R_NEED = -0.004
R_SURVIVE = 4.0


def obs_for(world, s):
    """Egocentric float32 observation for one survivor."""
    cfg, m = world.cfg, world.game_map
    vr = cfg.vision_range
    f = [s.hp / 100.0, s.hunger / 100.0, s.thirst / 100.0, s.fear,
         1.0 if s.infected else 0.0, s.infection / 100.0, s.cooldown / 30.0]

    def rel(dx, dy):
        d = math.hypot(dx, dy)
        a = math.atan2(dy, dx) - s.heading
        return d, math.sin(a), math.cos(a)

    vis = [z for z in world.spatial_z.query(s.x, s.y, vr)
           if z.alive and can_see(m, s, z.x, z.y, vr, cfg.vision_fov_deg)]
    if vis:
        z = min(vis, key=lambda z: (z.x - s.x) ** 2 + (z.y - s.y) ** 2)
        d, sn, cs = rel(z.x - s.x, z.y - s.y)
        f += [1.0, d / vr, sn, cs]
    else:
        f += [0.0, 0.0, 0.0, 0.0]

    if s.memory_z:
        zid, (zx, zy, tk) = min(s.memory_z.items(), key=lambda kv: -kv[1][2])
        d, sn, cs = rel(zx - s.x, zy - s.y)
        f += [1.0, d / vr, sn, cs, min(1.0, (world.t - tk) / 900.0)]
    else:
        f += [0.0, 0.0, 0.0, 0.0, 0.0]

    known = [(xy, sid) for sid, xy in s.memory_sup.items()
             if sid in world.supplies and not world.supplies[sid].taken]
    if known:
        (sx, sy), _ = min(known, key=lambda kv: (kv[0][0] - s.x) ** 2 + (kv[0][1] - s.y) ** 2)
        d, sn, cs = rel(sx - s.x, sy - s.y)
        f += [1.0, d / vr, sn, cs]
    else:
        f += [0.0, 0.0, 0.0, 0.0]

    peers = [o for o in world.spatial_s.query(s.x, s.y, vr) if o is not s and o.alive]
    if peers:
        p = min(peers, key=lambda o: (o.x - s.x) ** 2 + (o.y - s.y) ** 2)
        d, sn, cs = rel(p.x - s.x, p.y - s.y)
        f += [d / vr, sn, cs]
    else:
        f += [0.0, 0.0, 0.0]

    r = PATCH // 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            f.append(1.0 if m.blocked(int(s.x) + dx, int(s.y) + dy) else 0.0)
    return np.asarray(f, dtype=np.float32)


def apply_action(world, s, a):
    """Translate a discrete action into motor commands the systems layer executes."""
    cfg = world.cfg
    s.path = None
    s.goal = None
    s.fear = 0.6  # RL movers always travel at run speed -> run-level noise
    if a == 0:
        s.vel = (0.0, 0.0)
        s.action = None
    elif 1 <= a <= 8:
        dx, dy = DIRS[a - 1]
        s.vel = (dx * cfg.survivor_run, dy * cfg.survivor_run)
        s.heading = math.atan2(dy, dx)
        s.action = None
    else:  # attack nearest zombie in melee range
        z, best = None, 1e18
        for o in world.spatial_z.query(s.x, s.y, 1.3):
            if o.alive:
                d = (o.x - s.x) ** 2 + (o.y - s.y) ** 2
                if d < best:
                    best, z = d, o
        s.vel = (0.0, 0.0)
        if z is not None and s.cooldown <= 0:
            s.action = ("attack", z.id)
        else:
            s.action = None


class SurvivorEnv:
    """One parallel environment instance wrapping a fresh World per episode."""

    def __init__(self, env_index=0, base_seed=1, n_survivors=8, n_zombies=2,
                 size=32, horizon=1500, level=None):
        self.env_index = env_index
        self.base_seed = base_seed
        self.horizon = horizon
        self.base_cfg = Config(width=size, height=size,
                               n_survivors=n_survivors, n_zombies=n_zombies,
                               seed=base_seed)
        self.fixed_level = level   # authored Level or None (procedural map)
        self.next_level = None     # set by the curriculum controller
        self.current_level = level
        self.world = None
        self.episode = 0
        self.last_stats = None

    def reset(self):
        lvl = self.next_level if self.next_level is not None else self.fixed_level
        self.next_level = None
        self.current_level = lvl
        seed = self.base_seed * 100000 + self.episode * 7919
        overrides = {"seed": seed}
        if lvl is not None:
            overrides["n_zombies"] = lvl.n_zombies
            overrides["n_supply_outdoor"] = 0
            overrides["supply_per_building"] = 0.0
        cfg = Config(**{**self.base_cfg.to_dict(), **overrides})
        self.world = World(cfg, level=lvl)
        for s in self.world.survivors:
            s.rl = True
        self.episode += 1
        self._start_ids = [s.id for s in self.world.survivors]
        return self.obs()

    def obs(self):
        return {s.id: obs_for(self.world, s)
                for s in self.world.survivors if s.alive}

    def step(self, actions):
        w = self.world
        for s in w.survivors:
            if s.alive:
                apply_action(w, s, int(actions.get(s.id, 0)))

        log_i0 = len(w.log.events)
        w.step()

        rew = {sid: R_STEP for sid in self._start_ids if w.agents_by_id[sid].alive}
        for s in w.survivors:
            if s.alive:
                if s.hunger > 70:
                    rew[s.id] = rew.get(s.id, 0.0) + R_NEED
                if s.thirst > 70:
                    rew[s.id] = rew.get(s.id, 0.0) + R_NEED
        for e in w.log.events[log_i0:]:
            if e["type"] == "loot":
                rew[e["who"]] = rew.get(e["who"], 0.0) + R_LOOT
            elif e["type"] == "kill":
                rew[e["killer"]] = rew.get(e["killer"], 0.0) + R_KILL
            elif e["type"] == "bite":
                rew[e["who"]] = rew.get(e["who"], 0.0) + R_BITE
            elif e["type"] == "death" and e["kind"] == "survivor":
                rew[e["who"]] = rew.get(e["who"], 0.0) + R_DEATH

        survivors_alive, _, _ = w.counts()
        done = survivors_alive == 0 or w.t >= self.horizon
        if done:
            for s in w.survivors:
                if s.alive:
                    rew[s.id] = rew.get(s.id, 0.0) + R_SURVIVE
            self.last_stats = {
                "reward": float(sum(rew.values())),
                "survivors": survivors_alive,
                "supplies": w.log.count("loot"),
                "kills": w.log.count("kill"),
                "turns": w.log.count("turn"),
                "ticks": w.t,
            }
        dead = {sid for sid in self._start_ids if not w.agents_by_id[sid].alive}
        return self.obs(), rew, done, {"dead": dead}
