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
SELF_DIM = 10                   # hp, hunger, thirst, fear, infected, infection, cooldown,
                                # gun, ammo, bat
Z_VIS_DIM = 4                   # exists, dist, sin, cos
Z_MEM_DIM = 5                   # exists, dist, sin, cos, staleness
SUP_DIM = 4
PEER_DIM = 3
OBS_DIM = SELF_DIM + Z_VIS_DIM + Z_MEM_DIM + SUP_DIM + PEER_DIM + PATCH * PATCH  # 107
ACT_DIM = 10                    # 0 stay, 1-8 move dirs, 9 attack
# screen-consistent move directions (N, NE, E, SE, S, SW, W, NW) in tile coords
DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]

R_LOOT = 2.0
R_KILL = 1.0      # combat pays, but the MISSION must pay more: a 3-item deposit
                  # run (+9) beats any kill; this kills the brawl-and-die local optimum
R_BITE = -6.0     # a bite is a near-certain death sentence — the immediate
                  # signal must reflect that or avoidance never wins the tug-of-war
R_DEATH = -10.0
R_STEP = -0.002
R_NEED = -0.004
R_SURVIVE = 4.0
R_DEPOSIT = 3.0   # per item deposited in the safe house ("deposit" event, n items)
R_GUN = 1.0       # bonus for picking up a gun (loot event with kind == "gun")
R_QUOTA = 6.0     # per living survivor when the level's stockpile quota is met
R_SHAPE = 0.02    # potential-based shaping coefficient (Ng et al. 1999: F = g*phi' - phi
                  # preserves the optimal policy while giving a dense gradient)
R_AVOID = 0.02    # threat-avoidance shaping: equal weight to objective pull, so
                  # "walk through the zombie to the supply" is never net-positive
PBRS_GAMMA = 0.999  # must match the PPO discount


def _threat_potential(world, s):
    """Distance to the nearest visible zombie — higher is safer. Full vision
    range when nothing is visible (treated as maximally safe)."""
    vr = world.cfg.vision_range
    best = None
    for z in world.spatial_z.query(s.x, s.y, vr):
        if z.alive and can_see(world.game_map, s, z.x, z.y, vr, world.cfg.vision_fov_deg):
            d = math.hypot(z.x - s.x, z.y - s.y)
            if best is None or d < best:
                best = d
    return vr if best is None else best


def _potential(world, s):
    """Negative distance to the current objective: the safe house when carrying
    supplies, else the nearest KNOWN supply (agent-private memory only)."""
    carried = sum(s.inventory[k] for k in ("food", "water", "meds"))
    if carried >= 1:
        if world.safehouse is not None:
            x0, y0, sw, sh = world.safehouse
            tx, ty = x0 + sw / 2.0, y0 + sh / 2.0
        else:
            return 0.0
    else:
        known = [xy for sid, xy in s.memory_sup.items()
                 if sid in world.supplies and not world.supplies[sid].taken]
        if not known:
            return 0.0
        tx, ty = min(known, key=lambda p: (p[0] - s.x) ** 2 + (p[1] - s.y) ** 2)
    return -math.hypot(tx - s.x, ty - s.y)


def auto_consume(world):
    """RL agents have no eat/drink/heal actions: consume at the same thresholds
    the scripted brain uses. Never overrides a policy-chosen engage/attack."""
    for s in world.survivors:
        if not s.alive or s.action is not None:
            continue
        if s.hunger > 70 and s.inventory["food"] > 0:
            s.action = ("eat", None)
        elif s.thirst > 70 and s.inventory["water"] > 0:
            s.action = ("drink", None)
        elif s.hp < 45 and s.inventory["meds"] > 0:
            s.action = ("heal", None)


def update_senses(world):
    """Keep the RL agents' memory/fear observation features alive (the scripted
    brain writes these for its own agents; RL agents bypass it)."""
    cfg = world.cfg
    vr = cfg.vision_range
    for s in world.survivors:
        if not s.alive:
            continue
        vis = [z for z in world.spatial_z.query(s.x, s.y, vr)
               if z.alive and can_see(world.game_map, s, z.x, z.y, vr, cfg.vision_fov_deg)]
        if vis:
            d = min(math.hypot(z.x - s.x, z.y - s.y) for z in vis)
            s.fear = min(1.0, max(s.fear - 0.004, 1.0 - d / vr))
            for z in vis:
                s.memory_z[z.id] = (z.x, z.y, world.t)
        else:
            s.fear = max(0.0, s.fear - 0.004)
        for sid, sup in world.supplies.items():
            if sup.taken or sid in s.memory_sup:
                continue
            if (sup.x - s.x) ** 2 + (sup.y - s.y) ** 2 <= vr * vr \
                    and can_see(world.game_map, s, sup.x, sup.y, vr, cfg.vision_fov_deg):
                s.memory_sup[sid] = (sup.x, sup.y)
        for sid in [k for k, (_, _, tk) in s.memory_z.items() if world.t - tk > 900]:
            del s.memory_z[sid]
        for sid in [k for k, xy in s.memory_sup.items()
                    if sid in world.supplies and world.supplies[sid].taken]:
            del s.memory_sup[sid]


def obs_for(world, s):
    """Egocentric float32 observation for one survivor."""
    cfg, m = world.cfg, world.game_map
    vr = cfg.vision_range
    f = [s.hp / 100.0, s.hunger / 100.0, s.thirst / 100.0, s.fear,
         1.0 if s.infected else 0.0, s.infection / 100.0, s.cooldown / 30.0,
         1.0 if s.gun else 0.0, s.ammo / 8.0, 1.0 if s.has_bat else 0.0]

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
    if a == 0:
        s.state_label = "HOLD"
        s.vel = (0.0, 0.0)
        s.action = None
    elif 1 <= a <= 8:
        s.state_label = "MOVE"
        dx, dy = DIRS[a - 1]
        s.vel = (dx * cfg.survivor_run, dy * cfg.survivor_run)
        s.heading = math.atan2(dy, dx)
        s.action = None
    else:  # action 9 "engage": attack the nearest visible zombie the survivor
        # can actually fight — gun range when armed, melee reach when not
        # (an unarmed whiff at 6 tiles would lock the agent in a wasted cooldown)
        vr = cfg.vision_range
        fov = cfg.vision_fov_deg
        armed = getattr(s, "gun", False) and s.ammo > 0
        reach = cfg.gun_range if armed else 1.45
        z, best = None, 1e18
        for o in world.spatial_z.query(s.x, s.y, reach):
            if o.alive and can_see(world.game_map, s, o.x, o.y, vr, fov):
                d = (o.x - s.x) ** 2 + (o.y - s.y) ** 2
                if d < best:
                    best, z = d, o
        s.vel = (0.0, 0.0)
        # spatial query over-approximates cell overlap — enforce true reach so
        # an out-of-range target can't burn the attack cooldown on a whiff
        if z is not None and best <= reach * reach and s.cooldown <= 0:
            s.state_label = "ENGAGE"
            s.action = ("attack", z.id)
        else:
            s.state_label = "HOLD"
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
        self.pending_reset = False
        self._ep_reward = 0.0

    def reset(self):
        # priority: controller override > stay on current level > static level
        lvl = self.next_level
        if lvl is None:
            lvl = self.current_level if self.current_level is not None else self.fixed_level
        self.next_level = None
        self.current_level = lvl
        self.pending_reset = False
        self._ep_reward = 0.0
        seed = self.base_seed * 100000 + self.env_index * 4099 + self.episode * 7919
        overrides = {"seed": seed}
        if lvl is not None:
            overrides.update({"n_zombies": lvl.n_zombies,
                              "n_supply_outdoor": 0, "supply_per_building": 0.0,
                              "width": lvl.map.w, "height": lvl.map.h})
            if lvl.reanimation_enabled is not None:
                overrides["reanimation_enabled"] = lvl.reanimation_enabled
        cfg = Config(**{**self.base_cfg.to_dict(), **overrides})
        if lvl is None or True:
            jitter = 0.9 + 0.2 * ((seed * 2654435761) % 1000 / 1000.0)
            cfg.thirst_rate = round(cfg.thirst_rate * jitter, 5)
            cfg.hunger_rate = round(cfg.hunger_rate * jitter, 5)
        self.world = World(cfg, level=lvl)
        # spatial hashes start empty (World.step rebuilds them, but the first
        # step's shaping would otherwise read the pre-reset state as empty)
        self.world.spatial_z.rebuild(self.world.zombies)
        self.world.spatial_s.rebuild(self.world.survivors)
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
        auto_consume(w)

        log_i0 = len(w.log.events)
        phi_before = {s.id: _potential(w, s) for s in w.survivors if s.alive}
        thr_before = {s.id: _threat_potential(w, s) for s in w.survivors if s.alive}
        w.step()
        update_senses(w)

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
                if e.get("kind") in ("gun", "bat"):
                    rew[e["who"]] = rew.get(e["who"], 0.0) + R_GUN
            elif e["type"] == "deposit":
                # safe-house deposit: ("deposit", who, n) — n items added to stockpile
                who = e.get("who")
                if who is not None:
                    rew[who] = rew.get(who, 0.0) + R_DEPOSIT * e.get("n", 0)
            elif e["type"] == "kill":
                rew[e["killer"]] = rew.get(e["killer"], 0.0) + R_KILL
            elif e["type"] == "bite":
                rew[e["who"]] = rew.get(e["who"], 0.0) + R_BITE
            elif e["type"] == "death" and e["kind"] == "survivor":
                rew[e["who"]] = rew.get(e["who"], 0.0) + R_DEATH

        survivors_alive, _, _ = w.counts()

        # dense shaping: reward every tick that closes in on the objective AND
        # every tick that keeps distance from visible threats
        for s in w.survivors:
            if s.alive and s.id in phi_before:
                phi_after = _potential(w, s)
                thr_after = _threat_potential(w, s)
                rew[s.id] = rew.get(s.id, 0.0) + R_SHAPE * (
                    PBRS_GAMMA * phi_after - phi_before[s.id]) \
                    + R_AVOID * (PBRS_GAMMA * thr_after - thr_before[s.id])

        # episode ends three ways: squad wiped (loss), stockpile quota met
        # (THE WIN — terminal, exactly like the viewer's banner), or safety cap
        won = w.level is not None and w.stockpile >= w.level.win_stock
        done = survivors_alive == 0 or won or w.t >= self.horizon
        if done:
            self.pending_reset = True
            for s in w.survivors:
                if s.alive:
                    rew[s.id] = rew.get(s.id, 0.0) - R_SHAPE * _potential(w, s)                         - R_AVOID * _threat_potential(w, s)
            quota_met = 1 if won else 0
            for s in w.survivors:
                if s.alive:
                    rew[s.id] = rew.get(s.id, 0.0) + R_QUOTA * quota_met + R_SURVIVE

        # episode reward accumulator: sum every component of every tick
        # (R_STEP, needs, events, shaping, terminal bonuses) so last_stats
        # carries the true episode return, not the final tick's bonuses alone
        self._ep_reward += sum(rew.values())

        if done:
            self.last_stats = {
                "reward": float(self._ep_reward),
                "survivors": survivors_alive,
                "supplies": w.log.count("loot"),
                "kills": w.log.count("kill"),
                "turns": w.log.count("turn"),
                "ticks": w.t,
                "quota": quota_met,
                "stockpile": w.stockpile,
            }
        dead = {sid for sid in self._start_ids if not w.agents_by_id[sid].alive}
        truncated = bool(done and survivors_alive > 0 and not won)
        return self.obs(), rew, done, {"dead": dead, "truncated": truncated}
