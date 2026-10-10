"""Zombie RL environment: parameter-shared zombie agents hunt FSM survivors.

Zombie senses are deliberately degraded (their real in-sim senses): a short
vision cone, the last-known survivor position, and the noise gradient. Movement
is Discrete(5): stay / 4 directions. Bites auto-resolve on contact through
combat_tick — the policy's job is to GET to survivors.
"""
import math

import numpy as np

from zombie_sim.core.config import Config
from zombie_sim.core.perception import can_see
from zombie_sim.core.sim import World

PATCH = 7
SELF_DIM = 2                    # hp, attack cooldown
ZVIS_DIM = 4                    # visible survivor: exists, dist, sin, cos
MEM_DIM = 5                     # last-known: exists, dist, sin, cos, staleness
NOISE_DIM = 3                   # heard something, strength, sin, cos
OBS_DIM = SELF_DIM + ZVIS_DIM + MEM_DIM + NOISE_DIM + PATCH * PATCH  # 63
ACT_DIM = 5                     # 0 stay, 1-4 move N/E/S/W
DIRS4 = [(0, -1), (1, 0), (0, 1), (-1, 0)]

R_BITE = 3.0
R_KILL = 6.0
R_STEP = -0.002
R_WIN = 5.0                     # all survivors dead
R_LOSE = -2.0                   # per survivor alive at the horizon
R_SHAPE = 0.02                  # approach the nearest visible/remembered survivor
PBRS_GAMMA = 0.999


def _nearest_survivor_info(world, z):
    """Nearest visible survivor, else the freshest memory. Returns
    (dx, dy, source) or (None,)."""
    cfg = world.cfg
    vr = cfg.zombie_vision_range
    best = None
    for s in world.spatial_s.query(z.x, z.y, vr):
        if s.alive and s.kind == "survivor" \
                and can_see(world.game_map, z, s.x, s.y, vr, cfg.zombie_vision_fov_deg):
            d = (s.x - z.x) ** 2 + (s.y - z.y) ** 2
            if best is None or d < best[0]:
                best = (d, s.x, s.y)
    if best is not None:
        return best[1], best[2], "vis"
    if z.last_known is not None:
        return z.last_known[0], z.last_known[1], "mem"
    return None, None, None


def obs_for_zombie(world, z):
    cfg = world.cfg
    vr = cfg.zombie_vision_range
    f = [z.hp / 100.0, z.attack_cd / 15.0]

    def ang(dx, dy):
        a = math.atan2(dy, dx) - z.heading
        return math.sin(a), math.cos(a)

    vis = []
    for s in world.spatial_s.query(z.x, z.y, vr):
        if s.alive and s.kind == "survivor" \
                and can_see(world.game_map, z, s.x, s.y, vr, cfg.zombie_vision_fov_deg):
            vis.append(((s.x - z.x) ** 2 + (s.y - z.y) ** 2, s))
    if vis:
        _, s = min(vis, key=lambda pair: pair[0])
        d = math.hypot(s.x - z.x, s.y - z.y)
        sn, cs = ang(s.x - z.x, s.y - z.y)
        f += [1.0, d / vr, sn, cs]
    else:
        f += [0.0, 0.0, 0.0, 0.0]

    if z.last_known is not None:
        lx, ly = z.last_known
        d = math.hypot(lx - z.x, ly - z.y)
        sn, cs = ang(lx - z.x, ly - z.y)
        f += [1.0, min(d / vr, 2.0), sn, cs, min(1.0, (world.t - z.last_seen) / 400.0)]
    else:
        f += [0.0, 0.0, 0.0, 0.0, 0.0]

    v, tile = world.noise.strongest(z.x, z.y, int(cfg.zombie_hearing_radius))
    if tile is not None and v > 0.5:
        sn, cs = ang(tile[0] + 0.5 - z.x, tile[1] + 0.5 - z.y)
        f += [min(v / 50.0, 1.0), sn, cs]
    else:
        f += [0.0, 0.0, 0.0]

    r = PATCH // 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            f.append(1.0 if world.game_map.blocked(int(z.x) + dx, int(z.y) + dy) else 0.0)
    return np.asarray(f, dtype=np.float32)


def apply_action_zombie(world, z, a):
    cfg = world.cfg
    if a == 0:
        z.vel = (0.0, 0.0)
    else:
        dx, dy = DIRS4[a - 1]
        z.vel = (dx * cfg.zombie_chase_speed, dy * cfg.zombie_chase_speed)
        z.heading = math.atan2(dy, dx)


def _threat_dist(world, z):
    """Distance to the nearest visible survivor; vision range when none visible."""
    cfg = world.cfg
    vr = cfg.zombie_vision_range
    best = None
    for s in world.spatial_s.query(z.x, z.y, vr):
        if s.alive and s.kind == "survivor" \
                and can_see(world.game_map, z, s.x, s.y, vr, cfg.zombie_vision_fov_deg):
            d = math.hypot(s.x - z.x, s.y - z.y)
            if best is None or d < best:
                best = d
    return vr if best is None else best


class ZombieEnv:
    """Zombies are the RL population; survivors run their scripted FSM brains."""

    def __init__(self, env_index=0, base_seed=1, n_survivors=6, n_zombies=6,
                 horizon=4000, level=None):
        self.env_index = env_index
        self.base_seed = base_seed
        self.horizon = horizon
        self.level = level
        self.current_level = level
        self.next_level = None
        self.pending_reset = False
        self.world = None
        self.episode = 0
        self.last_stats = None
        self._z_start = []
        self.base_nz = n_zombies
        self.base_ns = n_survivors

    def reset(self):
        lvl = self.next_level if self.next_level is not None else self.level
        self.next_level = None
        self.current_level = lvl
        seed = self.base_seed * 100000 + self.env_index * 4099 + self.episode * 7919
        overrides = {"seed": seed, "n_survivors": self.base_ns,
                     "n_zombies": self.base_nz,
                     "width": self.size_w, "height": self.size_h,
                     "n_supply_outdoor": 0, "supply_per_building": 0.0}
        if lvl is not None:
            overrides.update({"width": lvl.map.w, "height": lvl.map.h,
                              "n_zombies": lvl.n_zombies,
                              "n_survivors": self.base_ns})
        cfg = Config(**{**self.base_cfg_dict(), **overrides})
        self.world = World(cfg, level=lvl)
        for z in self.world.zombies:
            z.rl = True
        self.episode += 1
        self._z_start = [q.id for q in self.world.zombies]
        return self.obs()

    def base_cfg_dict(self):
        base = Config(width=32, height=32,
                      n_survivors=self.base_ns, n_zombies=self.base_nz,
                      seed=self.base_seed)
        return base.to_dict()

    @property
    def size_w(self):
        return self._lvl_wh[0] if self._lvl_wh else 32

    @property
    def size_h(self):
        return self._lvl_wh[1] if self._lvl_wh else 32

    @property
    def _lvl_wh(self):
        if self.current_level is not None:
            return (self.current_level.map.w, self.current_level.map.h)
        return None

    def obs(self):
        return {z.id: obs_for_zombie(self.world, z)
                for z in self.world.zombies if z.alive}

    def step(self, actions):
        w = self.world
        for z in w.zombies:
            if z.alive:
                apply_action_zombie(w, z, int(actions.get(z.id, 0)))

        log_i0 = len(w.log.events)
        thr_before = {z.id: _threat_dist(w, z) for z in w.zombies if z.alive}
        w.step()
        update_threat = {z.id: _threat_dist(w, z) for z in w.zombies if z.alive}

        rew = {z.id: R_STEP for z in w.zombies if z.alive}
        for e in w.log.events[log_i0:]:
            if e["type"] == "bite":
                rew[e["zombie"]] = rew.get(e["zombie"], 0.0) + R_BITE
            elif e["type"] == "killz":
                rew[e["attacker"]] = rew.get(e["attacker"], 0.0) + R_KILL

        survivors_alive = w.counts()[0]
        done = survivors_alive == 0 or w.t >= self.horizon
        if done:
            self.pending_reset = True
            if survivors_alive == 0:
                for z in w.zombies:
                    if z.alive:
                        rew[z.id] = rew.get(z.id, 0.0) + R_WIN
            else:
                for z in w.zombies:
                    if z.alive:
                        rew[z.id] = rew.get(z.id, 0.0) + R_LOSE * survivors_alive
            self.last_stats = {
                "reward": float(sum(rew.values())),
                "survivors": survivors_alive,
                "bites": w.log.count("bite"),
                "kills": w.log.count("killz"),
                "ticks": w.t,
            }

        # shaping: closing in on prey (visible or remembered)
        for z in w.zombies:
            if z.alive and z.id in thr_before:
                rew[z.id] = rew.get(z.id, 0.0) + R_SHAPE * (
                    PBRS_GAMMA * update_threat[z.id] - thr_before[z.id])

        dead = {zid for zid in self._z_start if not w.agents_by_id[zid].alive}
        return self.obs(), rew, done, {"dead": dead}
