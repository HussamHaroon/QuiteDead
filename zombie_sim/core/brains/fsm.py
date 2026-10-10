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
    for s in world.spatial_s.query(z.x, z.y, cfg.zombie_vision_range):
        if s.alive and can_see(m, z, s.x, s.y, cfg.zombie_vision_range, cfg.zombie_vision_fov_deg):
            if world.rng.random() < cfg.zombie_detect_chance:
                d = (s.x - z.x) ** 2 + (s.y - z.y) ** 2
                if d < best_d:
                    best_d, visible = d, s

    if visible is not None:
        z.state = "chase"
        z.last_known = (visible.x, visible.y)
        z.last_seen = world.t
    elif z.state == "chase" and world.t - z.last_seen > cfg.zombie_memory_ticks:
        z.state = "investigate"
        z.target = z.last_known

    if z.state != "chase":
        v, tile = world.noise.strongest(z.x, z.y, int(cfg.zombie_hearing_radius))
        if tile is not None and v > cfg.noise_hear_threshold:
            z.state = "investigate"
            z.target = tile

    if z.state == "chase" and z.last_known is not None:
        dx, dy = z.last_known[0] - z.x, z.last_known[1] - z.y
        d = math.hypot(dx, dy)
        if d < 0.5:
            z.state = "investigate"
            z.target = z.last_known
        else:
            z.vel = (dx / d * cfg.zombie_chase_speed, dy / d * cfg.zombie_chase_speed)
            z.heading = math.atan2(dy, dx)
            _separate(world, z)
            return

    if z.state == "investigate" and z.target is not None:
        dx, dy = z.target[0] + 0.5 - z.x, z.target[1] + 0.5 - z.y
        d = math.hypot(dx, dy)
        if d < 0.7:
            z.state = "wander"
            z.target = None
        else:
            sp = cfg.zombie_walk * 1.5
            z.vel = (dx / d * sp, dy / d * sp)
            z.heading = math.atan2(dy, dx)
            _separate(world, z)
            return

    z.heading += world.rng.uniform(-0.35, 0.35)
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
