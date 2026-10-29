"""Zombie brain: classic movie zombie behavior.

Zombies SHAMBLE aimlessly, INVESTIGATE noise, and CHASE visible prey.
They do NOT know where survivors are — they must see or hear them.
When prey escapes into the safe house, zombies lose interest and wander.

States:
- WANDER: shamble to a random patrol point (looks like aimless drifting)
- INVESTIGATE: move to a noise source, search briefly, resume patrol
- CHASE: full-speed pursuit of a VISIBLE survivor
"""
import math

from ..perception import can_see

PATROL_NEW_POINT_TICKS = 200   # pick a new patrol target this often
SEARCH_TICKS = 120             # how long to search a noise spot before giving up


def think(world, z):
    cfg = world.cfg
    m = world.game_map

    # ---- vision: is a survivor visible right now? ----
    # survivors inside the safe house are INVISIBLE to zombies — the walls
    # block sight and the zombie loses interest once prey enters sanctuary
    visible = None
    best_d = 1e18
    sh = world.safehouse  # (x0, y0, w, h) or None
    for s in world.spatial_s.query(z.x, z.y, cfg.zombie_vision_range):
        if s.alive and s.kind == "survivor":
            if sh is not None:
                sx0, sy0, sw_, shh = sh
                if sx0 <= s.x < sx0 + sw_ and sy0 <= s.y < sy0 + shh:
                    continue  # inside the safe house — invisible to zombies
            if can_see(m, z, s.x, s.y, cfg.zombie_vision_range, cfg.zombie_vision_fov_deg):
                d = math.hypot(s.x - z.x, s.y - z.y)
                p = cfg.zombie_detect_chance * (1.0 - 0.5 * d / cfg.zombie_vision_range)
                if world.rng.random() < p and d < best_d:
                    best_d, visible = d, s

    if visible is not None:
        z.state = "chase"
        z.last_known = (visible.x, visible.y)
        z.last_seen = world.t
        z.target = None  # clear patrol target while chasing
    elif z.state == "chase" and world.t - z.last_seen > cfg.zombie_memory_ticks:
        # lost sight of prey — go to last known position, then give up
        z.state = "investigate"
        z.target = z.last_known
        z.inv_since = world.t

    # ---- hearing: noise attracts zombies (when not already chasing) ----
    if z.state != "chase":
        v, tile = world.noise.strongest(z.x, z.y, int(cfg.zombie_hearing_radius))
        if tile is not None and v > cfg.noise_hear_threshold:
            z.state = "investigate"
            z.target = (tile[0] + 0.5, tile[1] + 0.5)
            z.inv_since = world.t

    # ---- act based on state ----
    if z.state == "chase" and z.last_known is not None:
        _move_toward(world, z, z.last_known, cfg.zombie_chase_speed)
        return

    if z.state == "investigate" and z.target is not None:
        dx = z.target[0] - z.x
        dy = z.target[1] - z.y
        d = math.hypot(dx, dy)
        # reached the noise spot or searched long enough -> resume wandering
        if d < 1.0 or world.t - z.inv_since > SEARCH_TICKS:
            z.state = "wander"
            z.target = None
        else:
            _move_toward(world, z, z.target, cfg.zombie_walk * 2.0)
            return

    # ---- wander: shamble toward a patrol point ----
    z.state = "wander"
    if z.target is None or _patrol_stale(world, z):
        z.target = _pick_patrol_point(world, z)
        z._patrol_set = world.t  # type: ignore[attr-defined]

    if z.target is not None:
        dx = z.target[0] - z.x
        dy = z.target[1] - z.y
        d = math.hypot(dx, dy)
        if d < 1.5:
            z.target = _pick_patrol_point(world, z)  # arrived — pick a new one
        else:
            # shamble: mostly straight with a wobble
            z.heading = math.atan2(dy, dx) + world.rng.uniform(-0.3, 0.3)
            z.vel = (math.cos(z.heading) * cfg.zombie_walk,
                     math.sin(z.heading) * cfg.zombie_walk)
            _separate(world, z)
            return

    # no target (edge case): drift
    z.heading += world.rng.uniform(-0.2, 0.2)
    z.vel = (math.cos(z.heading) * cfg.zombie_walk * 0.5,
             math.sin(z.heading) * cfg.zombie_walk * 0.5)


def _move_toward(world, z, target, speed):
    dx = target[0] - z.x
    dy = target[1] - z.y
    d = math.hypot(dx, dy)
    if d > 0.1:
        z.vel = (dx / d * speed, dy / d * speed)
        z.heading = math.atan2(dy, dx)
    else:
        z.vel = (0.0, 0.0)
    _separate(world, z)


def _patrol_stale(world, z):
    return world.t - getattr(z, "_patrol_set", 0) > PATROL_NEW_POINT_TICKS


def _pick_patrol_point(world, z):
    """Random walkable point on the map — zombies drift around aimlessly."""
    m = world.game_map
    for _ in range(20):
        tx = world.rng.uniform(2, m.w - 2)
        ty = world.rng.uniform(2, m.h - 2)
        if not m.blocked(int(tx), int(ty)):
            # don't patrol into the safe house — zombies can't enter it anyway
            if world.safehouse is not None:
                x0, y0, sw, sh = world.safehouse
                if x0 <= tx < x0 + sw and y0 <= ty < y0 + sh:
                    continue
            return (tx, ty)
    return (m.w / 2.0, m.h / 2.0)


def _separate(world, z):
    """Small repulsion so hordes don't stack into one pixel."""
    push_x = push_y = 0.0
    for o in world.spatial_z.query(z.x, z.y, 0.9):
        if o is z or not o.alive:
            continue
        dx, dy = z.x - o.x, z.y - o.y
        d = math.hypot(dx, dy)
        if 1e-6 < d < 0.9:
            push_x += dx / d * 0.04
            push_y += dy / d * 0.04
    if push_x or push_y:
        z.vel = (z.vel[0] + push_x, z.vel[1] + push_y)
