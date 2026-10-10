"""Survivor brain: priority ladder over threats, needs, the safe house and loot.

Evaluated every tick:
  1. THREAT: visible zombie -> stand and shoot (gun + ammo + in range),
     else flee when it closes inside danger_radius, else point-blank melee.
  2. NEEDS: eat / drink / heal from inventory.
  3. RETURN: carrying enough supplies, or nothing useful left to find ->
     path to the safe house center (threats above still override mid-trip).
  4. OUTING: loot the nearest *useful* remembered supply
     (gun if unarmed > ammo if low > food/water/meds).
  5. EXPLORE: random walkable fallback.

The result is the visible mission loop: leave the safe house, loot, come
back, deposit, head out again.
"""
import math

from ..perception import can_see
from ..pathfinding import astar

MEM_STALE_TICKS = 900
CAP = {"food": 4, "water": 4, "meds": 2}
CARRIED_KINDS = ("food", "water", "meds")
RETURN_LOAD = 3   # carried supplies that trigger a run back home
AMMO_LOW = 4      # an armed survivor wants more ammo below this


def _carried(s):
    return sum(s.inventory[k] for k in CARRIED_KINDS)


def _useful(kind, s):
    """Would picking this supply kind up be worth the trip for survivor s?"""
    if kind == "gun":
        return not s.gun
    if kind == "bat":
        return not s.has_bat
    if kind == "ammo":
        return s.gun and s.ammo < AMMO_LOW
    return s.inventory[kind] < CAP[kind]


def _pick_target(world, s):
    """Nearest useful remembered supply; gun > bat > ammo > consumables, then distance."""
    best_sid, best_key = None, None
    for sid, (sx, sy) in s.memory_sup.items():
        kind = world.supplies[sid].kind
        if not _useful(kind, s):
            continue
        tier = 0 if kind == "gun" else (1 if kind == "bat" else (2 if kind == "ammo" else 3))
        key = (tier, (sx - s.x) ** 2 + (sy - s.y) ** 2)
        if best_key is None or key < best_key:
            best_sid, best_key = sid, key
    return best_sid


def _has_useful(world, s):
    return any(_useful(world.supplies[sid].kind, s) for sid in s.memory_sup)


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

    # 1. THREAT: shoot (armed) > flee > bat-fight (armed with bat, healthy,
    #    point-blank) — bare hands NEVER stand and fight; the cornered shove
    #    escape is resolved in move_agents
    if threats:
        nearest = min(threats, key=lambda z: (z.x - s.x) ** 2 + (z.y - s.y) ** 2)
        d = math.hypot(nearest.x - s.x, nearest.y - s.y)
        s.fear = min(1.0, s.fear + 0.08)
        if s.gun and s.ammo > 0 and d <= cfg.gun_range:
            s.state_label = "SHOOTING"
            s.action = ("attack", nearest.id)
            s.path = None
            s.goal = None
            s.vel = (0.0, 0.0)
            s.heading = math.atan2(nearest.y - s.y, nearest.x - s.x)
            return
        if d < cfg.danger_radius:
            if s.has_bat and d <= 1.15 and s.hp > cfg.fight_hp_threshold:
                s.state_label = "FIGHTING"
                s.action = ("attack", nearest.id)
                s.path = None
                s.vel = (0.0, 0.0)
                s.heading = math.atan2(nearest.y - s.y, nearest.x - s.x)
                return
            s.state_label = "FLEEING"
            _flee(s, threats, cfg)
            return

    # 2. NEEDS
    if s.hunger > 70 and s.inventory["food"] > 0:
        s.action = ("eat", None)
        s.state_label = "EATING"
    elif s.thirst > 70 and s.inventory["water"] > 0:
        s.action = ("drink", None)
        s.state_label = "DRINKING"
    elif s.hp < 45 and s.inventory["meds"] > 0:
        s.action = ("heal", None)
        s.state_label = "HEALING"
    else:
        s.action = None

    # 3. RETURN: load is full, or there is nothing useful left that we know of.
    #    (With nothing carried and nothing known, exploring beats sitting home.)
    if world.safehouse is not None:
        going_home = _carried(s) >= RETURN_LOAD \
            or (_carried(s) > 0 and not _has_useful(world, s))
        if going_home:
            s.state_label = "RETURNING"
            if world.in_safehouse(s):
                s.state_label = "DEPOSITING"
                s.path = None
                s.goal = None
                s.vel = (0.0, 0.0)
                return
            if _goto_safehouse(world, s):
                return

    # 4. OUTING: loot the nearest useful remembered supply
    if s.action is None and s.memory_sup:
        sid = _pick_target(world, s)
        if sid is not None:
            s.state_label = "LOOTING"
            sx, sy = s.memory_sup[sid]
            if _goto(s, m, sx, sy):
                return
            s.memory_sup.pop(sid, None)  # unreachable — forget it

    # 5. EXPLORE
    s.state_label = "EXPLORING"
    if s.path is None:
        for _ in range(8):
            tx = int(s.x + world.rng.uniform(-14, 14))
            ty = int(s.y + world.rng.uniform(-14, 14))
            if m.walkable(tx + 0.5, ty + 0.5) and _goto(s, m, tx + 0.5, ty + 0.5):
                break


def _goto_safehouse(world, s):
    x0, y0, w, h = world.safehouse
    return _goto(s, world.game_map, x0 + w // 2 + 0.5, y0 + h // 2 + 0.5)


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
