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
import os as _os
FLAGS = _os.environ.get('TAC', '').split(',')
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

    if 'intel' in FLAGS and world.t == 0:
        # pre-mission briefing: the squad knows where the armouries (guns) are, nothing else
        for sup in world.supplies.values():
            if sup.kind == "gun" and not sup.taken:
                s.memory_sup[sup.id] = (sup.x, sup.y)

    s.fear = max(0.0, s.fear - 0.004)
    s.fleeing = False

    # 1. THREAT: shoot (armed) > flee toward safe house > bat-fight (armed,
    #    healthy, point-blank) — bare hands NEVER stand and fight
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
        if 'hunker' in FLAGS and d < cfg.danger_radius and world.safehouse is not None and world.in_safehouse(s) \
                and not (s.gun and s.ammo > 0):
            # HUNKER: inside the sanctuary and unable to kill - running about only makes noise and
            # keeps the siege alive. Stand still (silent + invisible) and let them lose interest.
            s.state_label = "HUNKERING"
            s.action = None
            x0, y0, sw, sh = world.safehouse
            if math.hypot(s.x - (x0 + sw / 2.0), s.y - (y0 + sh / 2.0)) > 1.6:
                if _goto_safehouse(world, s):      # retreat from the door to the far interior
                    return
            s.path = None
            s.goal = None
            s.vel = (0.0, 0.0)
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
            _flee(world, s, threats, cfg)
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
    # GUN FIRST: an unarmed squad cannot kill anything, so until someone finds a weapon the
    # survivors search buildings instead of fetching consumables (unless a need is pressing).
    gun_known = any(world.supplies[k].kind == "gun" for k in s.memory_sup)
    armed_mate = any(o.alive and o.gun for o in world.survivors)
    pressing = s.hunger > 55 or s.thirst > 55
    gun_first = ('gun' in FLAGS) and (not s.gun) and (not armed_mate) and (not gun_known) and (not pressing) \
        and len(s.explored) < len(world.game_map.buildings) - 1
    if 'wait' in FLAGS and world.safehouse is not None and world.in_safehouse(s) and _siege(world, s):
        s.state_label = "WAITING"
        s.path = None; s.goal = None; s.vel = (0.0, 0.0); s.action = None
        return
    if s.action is None and s.memory_sup and not gun_first:
        sid = _pick_target(world, s)
        if sid is not None:
            s.state_label = "LOOTING"
            sx, sy = s.memory_sup[sid]
            if _goto(s, m, sx, sy):
                return
            s.memory_sup.pop(sid, None)  # unreachable — forget it

    # 5. EXPLORE (tactical): search *buildings* (weapons/supplies are inside), in pairs.
    #    Pair p = id // 2 shares one claimed target building, so partners travel together
    #    and the squad covers different buildings instead of overlapping.
    s.state_label = "SCOUTING"
    if s.path is None:
        _explore_pairs(world, s)
    return


def _siege(world, s):
    """Zombies seen close to the safe house recently -> stay in until they drift away."""
    x0, y0, sw, sh = world.safehouse
    hx, hy = x0 + sw / 2.0, y0 + sh / 2.0
    return any(world.t - tk < 120 and (zx - hx) ** 2 + (zy - hy) ** 2 < (max(sw, sh) / 2 + 5) ** 2
               for (zx, zy, tk) in s.memory_z.values())


def _mark_explored(world, s):
    for i, (bx, by, bw, bh) in enumerate(world.game_map.buildings):
        if i not in s.explored and (bx + bw / 2 - s.x) ** 2 + (by + bh / 2 - s.y) ** 2 < 12:
            s.explored.add(i)


def _risk(world, s, x, y):
    r = 0
    for (zx, zy, tk) in s.memory_z.values():
        if world.t - tk < 300 and (zx - x) ** 2 + (zy - y) ** 2 < 49:
            r += 1
    return r


def _explore_pairs(world, s):
    m = world.game_map
    _mark_explored(world, s)
    claims = world.__dict__.setdefault("_claims", {})
    p = s.id // 2
    c = claims.get(p)
    if c is not None and (c in s.explored or world.t - claims.get(("t", p), 0) > 700):
        c = None
    if c is None:
        taken = {v for k, v in claims.items() if not isinstance(k, tuple) and k != p}
        best, bc = None, 1e18
        hx = hy = None
        if world.safehouse is not None:
            x0, y0, sw, sh = world.safehouse
            hx, hy = x0 + sw / 2.0, y0 + sh / 2.0
        for i, (bx, by, bw, bh) in enumerate(m.buildings):
            if i in s.explored or i in taken:
                continue
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            if not m.walkable(cx, cy):
                continue
            ox, oy = (hx, hy) if hx is not None else (s.x, s.y)
            cost = math.hypot(cx - ox, cy - oy) * (1 + _risk(world, s, cx, cy))
            if cost < bc:
                best, bc = i, cost
        if best is not None:
            claims[p] = best
            claims[("t", p)] = world.t
            c = best
    if c is not None:
        bx, by, bw, bh = m.buildings[c]
        if _goto(s, m, bx + bw / 2.0, by + bh / 2.0):
            return
        s.explored.add(c)
    # fallback: ring around home like the ladder
    if world.safehouse is not None:
        x0, y0, sw_, sh_ = world.safehouse
        hx, hy = x0 + sw_ / 2.0, y0 + sh_ / 2.0
        angle = (s.id * 2.399) % (2 * math.pi) + 0.9 * (world.t // 400)
        for radius in (12, 18, 24, 10, 30):
            tx, ty = hx + math.cos(angle) * radius, hy + math.sin(angle) * radius
            if 1 < tx < m.w - 1 and 1 < ty < m.h - 1 and m.walkable(tx, ty) and _goto(s, m, tx, ty):
                return
    for _ in range(8):
        tx, ty = world.rng.uniform(2, m.w - 2), world.rng.uniform(2, m.h - 2)
        if m.walkable(tx, ty) and _goto(s, m, tx, ty):
            return


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


def _flee(world, s, threats, cfg):
    vx = vy = 0.0
    for z in threats:
        dx, dy = s.x - z.x, s.y - z.y
        d = max(0.3, math.hypot(dx, dy))
        vx += dx / d / d
        vy += dy / d / d
    # bias the escape toward the safe house: running "away" from one zombie
    # often runs into another — running home is both escape and progress
    if world is not None and world.safehouse is not None:
        x0, y0, sw, sh = world.safehouse
        hx, hy = x0 + sw / 2.0 - s.x, y0 + sh / 2.0 - s.y
        hd = max(0.3, math.hypot(hx, hy))
        vx += hx / hd * 0.6
        vy += hy / hd * 0.6
    n = math.hypot(vx, vy) or 1.0
    s.path = None
    s.goal = None
    s.fleeing = True
    s.vel = (vx / n * cfg.survivor_run, vy / n * cfg.survivor_run)
    s.heading = math.atan2(vy, vx)
