"""A* pathfinding on the tile grid (8-directional, no corner cutting)."""
import heapq

_DIAG = 2 ** 0.5


def astar(game_map, start, goal, max_expand=4000):
    """Returns a list of float waypoints (tile centers) from start to goal, or None."""
    sx, sy = int(start[0]), int(start[1])
    gx, gy = int(goal[0]), int(goal[1])
    if game_map.blocked(gx, gy):
        fixed = _nearby_walkable(game_map, gx, gy)
        if fixed is None:
            return None
        gx, gy = fixed
    if game_map.blocked(sx, sy):
        fixed = _nearby_walkable(game_map, sx, sy)
        if fixed is None:
            return None
        sx, sy = fixed
    if (sx, sy) == (gx, gy):
        return [(gx + 0.5, gy + 0.5)]

    open_heap = [(0.0, 0.0, sx, sy)]
    came = {}
    gscore = {(sx, sy): 0.0}
    closed = set()
    expansions = 0

    while open_heap:
        _, g, x, y = heapq.heappop(open_heap)
        if (x, y) in closed:
            continue
        closed.add((x, y))
        expansions += 1
        if expansions > max_expand:
            break
        if (x, y) == (gx, gy):
            path = [(x + 0.5, y + 0.5)]
            while (x, y) in came:
                x, y = came[(x, y)]
                path.append((x + 0.5, y + 0.5))
            path.reverse()
            return path
        for dx, dy, cost in ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                             (1, 1, _DIAG), (1, -1, _DIAG), (-1, 1, _DIAG), (-1, -1, _DIAG)):
            nx, ny = x + dx, y + dy
            if game_map.blocked(nx, ny) or (nx, ny) in closed:
                continue
            if dx and dy and (game_map.blocked(x + dx, y) or game_map.blocked(x, y + dy)):
                continue  # no squeezing through wall corners
            ng = g + cost
            if ng < gscore.get((nx, ny), 1e18):
                gscore[(nx, ny)] = ng
                came[(nx, ny)] = (x, y)
                hh = _octile(nx, ny, gx, gy)
                heapq.heappush(open_heap, (ng + hh, ng, nx, ny))
    return None


def _octile(x, y, gx, gy):
    dx, dy = abs(x - gx), abs(y - gy)
    return (dx + dy) + (_DIAG - 2.0) * min(dx, dy)


def _nearby_walkable(game_map, x, y, r=3):
    for radius in range(1, r + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                nx, ny = x + dx, y + dy
                if not game_map.blocked(nx, ny):
                    return (nx, ny)
    return None
