"""Vision: cone + range + raycast line-of-sight."""
import math

_STEP = 0.35  # LOS march step in tiles


def line_of_sight(game_map, x0, y0, x1, y1):
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1e-6:
        return True
    steps = max(1, int(dist / _STEP))
    dx, dy = (x1 - x0) / steps, (y1 - y0) / steps
    x, y = x0, y0
    for _ in range(steps - 1):
        x += dx
        y += dy
        if game_map.blocked(int(x), int(y)):
            return False
    return True


def can_see(game_map, watcher, tx, ty, vrange, fov_deg):
    dx, dy = tx - watcher.x, ty - watcher.y
    d = math.hypot(dx, dy)
    if d > vrange:
        return False
    if d > 0.8:  # point blank: seen regardless of facing
        ang = math.degrees(math.atan2(dy, dx))
        diff = abs((ang - math.degrees(watcher.heading) + 180.0) % 360.0 - 180.0)
        if diff > fov_deg / 2.0:
            return False
    return line_of_sight(game_map, watcher.x, watcher.y, tx, ty)
