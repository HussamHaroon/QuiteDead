"""Survivor brain: TRUE utility AI (score every candidate action, execute the argmax).

Unlike the fixed-priority ladder in utility.py, every option gets a continuous score in
[0, 1] built from response curves over needs, threat proximity, load, ammo, distance and
*risk* (recently seen zombies near a target). Extra agent-level behaviours:

* HUNKER - when zombies are seen/remembered near the safe house the agent stays inside,
  silent and invisible to them, instead of dithering at the door; it leaves when the
  danger memory fades or hunger/thirst make waiting too costly.
* Risk-aware looting - targets near recently seen zombies are penalised.
* Commitment - a chosen non-emergency action is kept for a few ticks (hysteresis), which
  removes the flee/loot flip-flop of reactive brains.
* Squad comms (systems.comms_tick) pool supply and zombie knowledge between nearby agents.
"""
import math

from ..perception import can_see
from . import utility as lad

COMMIT_TICKS = 25
HUNKER_TICKS = 120
HUNKER_W = 0.0
RISK_RADIUS = 8.0
RISK_AGE = 200
HOME_RISK_AGE = 450
EMERGENCY = 0.80


def _clamp(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def _risk_at(world, s, x, y, age=RISK_AGE):
    """0..1: how many recently seen zombies lurk within RISK_RADIUS of (x, y)."""
    n = 0
    for zx, zy, tk in s.memory_z.values():
        if world.t - tk <= age and (zx - x) ** 2 + (zy - y) ** 2 <= RISK_RADIUS ** 2:
            n += 1
    return min(1.0, n / 2.0)


def _perceive(world, s):
    cfg, m = world.cfg, world.game_map
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
    for i, (bx, by, bw, bh) in enumerate(m.buildings):
        if i not in s.explored and (bx + bw / 2 - s.x) ** 2 + (by + bh / 2 - s.y) ** 2 < 16:
            s.explored.add(i)
    for zid in [k for k, (_, _, tk) in s.memory_z.items() if world.t - tk > lad.MEM_STALE_TICKS]:
        del s.memory_z[zid]
    for sid in [k for k in s.memory_sup if world.supplies[k].taken]:
        del s.memory_sup[sid]
    return threats


def _stop(s):
    s.path = None
    s.goal = None
    s.vel = (0.0, 0.0)


def think(world, s):
    cfg, m = world.cfg, world.game_map
    threats = _perceive(world, s)
    s.fear = max(0.0, s.fear - 0.004)
    s.fleeing = False
    home = world.safehouse is not None
    in_home = home and world.in_safehouse(s)
    armed = s.gun and s.ammo > 0
    carried = lad._carried(s)

    nearest, d = None, 1e9
    if threats:
        nearest = min(threats, key=lambda z: (z.x - s.x) ** 2 + (z.y - s.y) ** 2)
        d = math.hypot(nearest.x - s.x, nearest.y - s.y)
        s.fear = min(1.0, s.fear + 0.08)

    # ---- needs run in parallel with movement (they are instant actions) ----
    s.action = None
    if s.hunger > 70 and s.inventory["food"] > 0:
        s.action = ("eat", None); s.state_label = "EATING"
    elif s.thirst > 70 and s.inventory["water"] > 0:
        s.action = ("drink", None); s.state_label = "DRINKING"
    elif s.hp < 45 and s.inventory["meds"] > 0:
        s.action = ("heal", None); s.state_label = "HEALING"

    # ---- score every candidate ----
    urgency = _clamp(max(s.hunger, s.thirst) / 100.0 * 1.25 - 0.2)  # waiting gets costly late
    scores = {}
    if nearest is not None:
        if armed and d <= cfg.gun_range:
            scores["shoot"] = 0.90 + 0.05 * (1 - d / cfg.gun_range)
        if s.has_bat and d <= 1.15 and s.hp > cfg.fight_hp_threshold:
            scores["melee"] = 0.88
        if d < cfg.danger_radius and not in_home:
            closeness = 1 - d / cfg.danger_radius
            scores["flee"] = _clamp(0.55 + 0.45 * closeness - (0.15 if armed else 0.0))
    home_threat = 0.0
    if home:
        hx, hy = world._safehouse_center()
        home_threat = _risk_at(world, s, hx, hy, HOME_RISK_AGE)
        if nearest is not None and d < 12:
            home_threat = max(home_threat, 0.8)
        if home_threat > 0:
            scores["hunker"] = _clamp(HUNKER_W * home_threat * (1 - urgency))
    # return: full load, or carrying with nothing useful known
    if home and carried > 0:
        load = carried / lad.RETURN_LOAD
        none_left = not lad._has_useful(world, s)
        scores["return"] = _clamp(0.15 + 0.65 * min(1.0, load) ** 2 + (0.45 if none_left else 0.0))
    # loot: best (value x proximity x safety) among remembered supplies
    best_t, best_sc = None, 0.0
    for sid, (sx, sy) in s.memory_sup.items():
        kind = world.supplies[sid].kind
        if not lad._useful(kind, s):
            continue
        value = {"gun": 1.0, "bat": 0.8, "ammo": 0.75}.get(kind, 0.62)
        if kind in ("food", "water"):
            value += 0.2 * (1 - s.inventory[kind] / lad.CAP[kind])
        prox = 1.0 - min(1.0, math.hypot(sx - s.x, sy - s.y) / 60.0)
        sc = 0.75 * value * (0.35 + 0.65 * prox) * (1 - 0.8 * _risk_at(world, s, sx, sy))
        if sc > best_sc:
            best_t, best_sc = sid, sc
    if best_t is not None:
        scores["loot"] = best_sc
    scores["scout"] = 0.22 + (0.12 if not s.memory_sup else 0.0) - 0.15 * home_threat

    # ---- commitment (anti-dithering) ----
    choice = max(scores, key=scores.get)
    if scores[choice] < EMERGENCY and s.commit and s.commit in scores and world.t < s.commit_until \
            and s.commit not in ("shoot", "melee", "flee"):
        choice = s.commit
    elif choice != s.commit:
        s.commit, s.commit_until = choice, world.t + (HUNKER_TICKS if choice == "hunker" else COMMIT_TICKS)

    # ---- execute ----
    if choice == "shoot" or choice == "melee":
        s.state_label = "SHOOTING" if choice == "shoot" else "FIGHTING"
        s.action = ("attack", nearest.id)
        _stop(s)
        s.heading = math.atan2(nearest.y - s.y, nearest.x - s.x)
    elif choice == "flee":
        s.state_label = "FLEEING"
        lad._flee(world, s, threats, cfg)
    elif choice == "hunker":
        s.state_label = "HUNKERING"
        if in_home:
            _stop(s)
        elif not lad._goto_safehouse(world, s):
            lad._flee(world, s, threats, cfg) if threats else _stop(s)
    elif choice == "return":
        s.state_label = "DEPOSITING" if in_home else "RETURNING"
        if in_home:
            _stop(s)
        elif not lad._goto_safehouse(world, s):
            _stop(s)
    elif choice == "loot":
        sx, sy = s.memory_sup[best_t]
        s.state_label = "LOOTING"
        if not lad._goto(s, m, sx, sy):
            s.memory_sup.pop(best_t, None)
    else:
        s.state_label = "SCOUTING"
        if s.path is None:
            _scout_ring(world, s)


def _scout(world, s):
    """Systematic search: head for the nearest building nobody in the squad has searched
    (weapons and supplies are inside), weighting by distance and by zombie risk."""
    m = world.game_map
    best, best_c = None, 1e18
    for i, (bx, by, bw, bh) in enumerate(m.buildings):
        if i in s.explored:
            continue
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        if not m.walkable(cx, cy):
            continue
        c = math.hypot(cx - s.x, cy - s.y) * (1 + 2 * _risk_at(world, s, cx, cy)) + (s.id * 7 + i * 3) % 5
        if c < best_c:
            best, best_c = (cx, cy, i), c
    if best is not None:
        if lad._goto(s, m, best[0], best[1]):
            return
        s.explored.add(best[2])  # unreachable - skip it
    _scout_ring(world, s)


def _scout_ring(world, s):
    m = world.game_map
    if world.safehouse is not None:
        x0, y0, sw, sh = world.safehouse
        hx, hy = x0 + sw / 2.0, y0 + sh / 2.0
        base = (s.id * 2.399) % (2 * math.pi)
        # rotate the preferred bearing each outing so the squad sweeps new ground
        base += 0.9 * (world.t // 400)
        for radius in (12, 18, 24, 10, 30):
            tx, ty = hx + math.cos(base) * radius, hy + math.sin(base) * radius
            if 1 < tx < m.w - 1 and 1 < ty < m.h - 1 and m.walkable(tx, ty) \
                    and _risk_at(world, s, tx, ty) < 0.5 and lad._goto(s, m, tx, ty):
                return
    for _ in range(8):
        tx, ty = world.rng.uniform(2, m.w - 2), world.rng.uniform(2, m.h - 2)
        if m.walkable(tx, ty) and lad._goto(s, m, tx, ty):
            return
