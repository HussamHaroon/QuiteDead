"""Zombie brain: a reactive FSM — wander / investigate / chase — with senses.

Vision is a probabilistic detection roll over a facing cone; hearing polls the
noise field. Both leave only a last-known position behind, so evasion and
decoys work the way they should.
"""
import math

from ..perception import can_see


def think(world, z):
    cfg = world.cfg
    m = world.game_map

    visible = None
    best_d = 1e18
    # spatial_s is the SURVIVOR hash — zombies hunt humans, not each other
    for s in world.spatial_s.query(z.x, z.y, cfg.zombie_vision_range):
        if s.alive and s.kind == "survivor" \
                and can_see(m, z, s.x, s.y, cfg.zombie_vision_range, cfg.zombie_vision_fov_deg):
            # detection is probabilistic AND distance-scaled: far survivors in
            # the cone are easy to miss, point-blank is nearly certain
            d = math.hypot(s.x - z.x, s.y - z.y)
            p = cfg.zombie_detect_chance * (1.0 - 0.7 * d / cfg.zombie_vision_range)
            if world.rng.random() < p and d < best_d:
                best_d, visible = d, s

    if visible is not None:
        z.state = "chase"
        z.last_known = (visible.x, visible.y)
        z.last_seen = world.t
    elif z.state == "chase" and world.t - z.last_seen > cfg.zombie_memory_ticks:
        z.state = "investigate"
        z.target = z.last_known
        z.inv_since = world.t

    if z.state != "chase":
        v, tile = world.noise.strongest(z.x, z.y, int(cfg.zombie_hearing_radius))
        if tile is not None and v > cfg.noise_hear_threshold:
            if z.state != "investigate":
                z.inv_since = world.t
            z.state = "investigate"
            z.target = (tile[0] + 0.5, tile[1] + 0.5)

    if z.state == "chase" and z.last_known is not None:
        dx, dy = z.last_known[0] - z.x, z.last_known[1] - z.y
        d = math.hypot(dx, dy)
        if d < 0.5:
            z.state = "investigate"
            z.target = z.last_known
            z.inv_since = world.t
        else:
            z.vel = (dx / d * cfg.zombie_chase_speed, dy / d * cfg.zombie_chase_speed)
            z.heading = math.atan2(dy, dx)
            _separate(world, z)
            return

    if z.state == "investigate" and z.target is not None:
        dx, dy = z.target[0] - z.x, z.target[1] - z.y
        d = math.hypot(dx, dy)
        # unreachable target (behind walls, noise faded): give up after a while
        # instead of grinding the same wall forever
        if d < 0.7 or world.t - z.inv_since > 240:
            z.state = "wander"
            z.target = None
        else:
            sp = cfg.zombie_walk * 1.5
            z.vel = (dx / d * sp, dy / d * sp)
            z.heading = math.atan2(dy, dx)
            _separate(world, z)
            return

    # wander: random drift, but shamble back toward the populated middle when
    # close to the border so zombies don't pile up in map corners
    w = world.game_map
    if z.x < 6 or z.x > w.w - 6 or z.y < 6 or z.y > w.h - 6:
        cx, cy = w.w / 2.0, w.h / 2.0
        center_head = math.atan2(cy - z.y, cx - z.x)
        diff = (center_head - z.heading + math.pi) % (2 * math.pi) - math.pi
        z.heading += diff * 0.08
    # DIRECTIONAL wander: drift toward the last investigation target or the
    # map center. Pure random walk makes zombies mill about uselessly.
    if z.target is not None:
        dx, dy = z.target[0] + 0.5 - z.x, z.target[1] + 0.5 - z.y
        d = math.hypot(dx, dy)
        if d > 0.5:
            z.heading = math.atan2(dy, dx)
    else:
        # slow drift toward map center (where most action is)
        cx, cy = world.game_map.w / 2.0, world.game_map.h / 2.0
        to_center = math.atan2(cy - z.y, cx - z.x)
        diff = (to_center - z.heading + math.pi) % (2 * math.pi) - math.pi
        z.heading += diff * 0.02
    z.heading += world.rng.uniform(-0.15, 0.15)
    z.vel = (math.cos(z.heading) * cfg.zombie_walk, math.sin(z.heading) * cfg.zombie_walk)
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
