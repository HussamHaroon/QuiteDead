"""Survivor brain: utility scoring over needs, threats and known loot.

Priority ladder (evaluated every tick):
  flee visible threat > stand and fight (only if healthy) > eat/drink/heal
  > walk to nearest remembered supply > explore.
"""
import math

from ..perception import can_see
from ..pathfinding import astar

MEM_STALE_TICKS = 900
CAP = {"food": 4, "water": 4, "meds": 2}


def think(world, s):
    cfg = world.cfg
    m = world.game_map

    threats = []
    for z in world.spatial_z.query(s.x, s.y, cfg.vision_range):
        if z.alive and can_see(m, s, z.x, z.y, cfg.vision_range, cfg.vision_fov_deg):
            threats.append(z)
            s.memory_z[z.id] = (z.x, z.y, world.t)

    for sup in world.supplies.values():
        if sup.taken or sup.id in s.memory_sup:
            continue
        if (sup.x - s.x) ** 2 + (sup.y - s.y) ** 2 <= cfg.vision_range ** 2 \
                and can_see(m, s, sup.x, sup.y, cfg.vision_range, cfg.vision_fov_deg):
            s.memory_sup[sup.id] = (sup.x, sup.y)

    for sid in [sid for sid, (_, _, tk) in s.memory_z.items() if world.t - tk > MEM_STALE_TICKS]:
        del s.memory_z[sid]
    for sid in [sid for sid, xy in s.memory_sup.items() if world.supplies[sid].taken]:
        del s.memory_sup[sid]

    s.fear = max(0.0, s.fear - 0.004)
    s.fleeing = False

    if threats:
        nearest = min(threats, key=lambda z: (z.x - s.x) ** 2 + (z.y - s.y) ** 2)
        d = math.hypot(nearest.x - s.x, nearest.y - s.y)
        s.fear = min(1.0, s.fear + 0.08)
        if d < 1.15 and s.hp > cfg.fight_hp_threshold and s.cooldown <= 0:
            s.action = ("attack", nearest.id)
            s.path = None
            s.vel = (0.0, 0.0)
            s.heading = math.atan2(nearest.y - s.y, nearest.x - s.x)
            return
        if d < cfg.danger_radius:
            _flee(s, threats, cfg)
            return

    if s.hunger > 70 and s.inventory["food"] > 0:
        s.action = ("eat", None)
    elif s.thirst > 70 and s.inventory["water"] > 0:
        s.action = ("drink", None)
    elif s.hp < 45 and s.inventory["meds"] > 0:
        s.action = ("heal", None)
    else:
        s.action = None

    full = all(s.inventory[k] >= CAP[k] for k in CAP)
    if s.action is None and not full and s.memory_sup:
        sid, (sx, sy) = min(s.memory_sup.items(),
                            key=lambda kv: (kv[1][0] - s.x) ** 2 + (kv[1][1] - s.y) ** 2)
        if _goto(s, m, sx, sy):
            return
        s.memory_sup.pop(sid, None)  # unreachable — forget it

    if s.path is None:
        for _ in range(8):
            tx = int(s.x + world.rng.uniform(-14, 14))
            ty = int(s.y + world.rng.uniform(-14, 14))
            if m.walkable(tx + 0.5, ty + 0.5) and _goto(s, m, tx + 0.5, ty + 0.5):
                break


def _goto(s, m, tx, ty):
    if s.goal == (tx, ty) and s.path:
        return True
    p = astar(m, (s.x, s.y), (tx, ty))
    if p is None:
        s.goal = None
        return False
    s.path = p
    s.path_i = 0
    s.goal = (tx, ty)
    return True


def _flee(s, threats, cfg):
    vx = vy = 0.0
    for z in threats:
        dx, dy = s.x - z.x, s.y - z.y
        d = max(0.3, math.hypot(dx, dy))
        vx += dx / d / d
        vy += dy / d / d
    n = math.hypot(vx, vy) or 1.0
    s.path = None
    s.goal = None
    s.fleeing = True
    s.vel = (vx / n * cfg.survivor_run, vy / n * cfg.survivor_run)
    s.heading = math.atan2(vy, vx)
