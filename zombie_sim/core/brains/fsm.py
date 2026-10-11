"""Zombie brain: RELENTLESS hunter.

Real zombies don't give up. They don't wander randomly. They SHAMBLE
tirelessly toward the nearest living flesh, and when they see one they
CHASE at full speed. They never forget. They never stop.

States (simplified from the old FSM):
- HUNT: drift toward the nearest survivor (zombies sense living flesh)
- CHASE: full-speed pursuit of a visible/remembered survivor

The key behavioral change: zombies ALWAYS move toward prey. No random
wandering. No giving up. This creates the classic zombie dynamic where
the survivors must constantly outmaneuver, not just outrun.
"""
import math

from ..perception import can_see


def think(world, z):
    cfg = world.cfg
    m = world.game_map

    # --- 1. VISION: scan for visible survivors (probabilistic + distance-scaled)
    visible = None
    best_d = 1e18
    for s in world.spatial_s.query(z.x, z.y, cfg.zombie_vision_range):
        if s.alive and s.kind == "survivor" \
                and can_see(m, z, s.x, s.y, cfg.zombie_vision_range, cfg.zombie_vision_fov_deg):
            d = math.hypot(s.x - z.x, s.y - z.y)
            p = cfg.zombie_detect_chance * (1.0 - 0.7 * d / cfg.zombie_vision_range)
            if world.rng.random() < p and d < best_d:
                best_d, visible = d, s

    if visible is not None:
        z.state = "chase"
        z.last_known = (visible.x, visible.y)
        z.last_seen = world.t

    # --- 2. HEARING: detect noise from any direction (no cone needed)
    if z.state != "chase":
        v, tile = world.noise.strongest(z.x, z.y, int(cfg.zombie_hearing_radius))
        if tile is not None and v > cfg.noise_hear_threshold:
            z.state = "chase"
            z.last_known = (tile[0] + 0.5, tile[1] + 0.5)
            z.last_seen = world.t

    # --- 3. MOVE: always toward the last-known survivor position
    #     Zombies NEVER stop. If they lose the trail, they shamble toward
    #     the general direction of the nearest survivor (they smell flesh).
    target = z.last_known
    if target is None:
        # no memory: shamble toward the map center (where the action is)
        cx, cy = world.game_map.w / 2.0, world.game_map.h / 2.0
        target = (cx, cy)

    dx, dy = target[0] - z.x, target[1] - z.y
    d = math.hypot(dx, dy)

    if d < 0.3:
        # reached the target but no visual contact: shamble in a small circle
        # (searching behavior)
        z.heading += world.rng.uniform(-0.8, 0.8)
        z.vel = (math.cos(z.heading) * cfg.zombie_walk * 0.7,
                 math.sin(z.heading) * cfg.zombie_walk * 0.7)
    else:
        speed = cfg.zombie_chase_speed if z.state == "chase" else cfg.zombie_walk
        z.vel = (dx / d * speed, dy / d * speed)
        z.heading = math.atan2(dy, dx)

    _separate(world, z)


def _separate(world, z):
    """Small repulsion so hordes don't stack into one pixel."""
    push_x = push_y = 0.0
    for o in world.spatial_z.query(z.x, z.y, 0.8):
        if o is z or not o.alive:
            continue
        dx, dy = z.x - o.x, z.y - o.y
        d = math.hypot(dx, dy)
        if 1e-6 < d < 0.8:
            push_x += dx / d * 0.03
            push_y += dy / d * 0.03
    if push_x or push_y:
        z.vel = (z.vel[0] + push_x, z.vel[1] + push_y)
