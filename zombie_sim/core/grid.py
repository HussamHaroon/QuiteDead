"""Tile grid and procedural town generation."""
import random

GRASS, ROAD, FLOOR, WALL, DOOR = 0, 1, 2, 3, 4
WALKABLE = {GRASS, ROAD, FLOOR, DOOR}


class GameMap:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.tiles = [[GRASS] * w for _ in range(h)]
        self.buildings = []  # (x0, y0, bw, bh)

    def walkable(self, x, y):
        """Float coords."""
        xi, yi = int(x), int(y)
        return 0 <= xi < self.w and 0 <= yi < self.h and self.tiles[yi][xi] in WALKABLE

    def blocked(self, xi, yi):
        """Int coords; outside the map counts as blocked."""
        if not (0 <= xi < self.w and 0 <= yi < self.h):
            return True
        return self.tiles[yi][xi] == WALL


def generate_map(rng: random.Random, w: int, h: int) -> GameMap:
    m = GameMap(w, h)
    t = m.tiles

    road_xs = list(range(8, w - 6, 16))
    road_ys = list(range(8, h - 6, 16))
    for x in road_xs:
        for y in range(h):
            t[y][x] = ROAD
            if x + 1 < w - 1:
                t[y][x + 1] = ROAD
    for y in road_ys:
        for x in range(w):
            t[y][x] = ROAD
            if y + 1 < h - 1:
                t[y + 1][x] = ROAD

    # borders BEFORE buildings: side_clear() must never mistake the edge for
    # open ground, or buildings get doors into the border wall (sealed pockets)
    for x in range(w):
        t[0][x] = t[h - 1][x] = WALL
    for y in range(h):
        t[y][0] = t[y][w - 1] = WALL

    bx = [0] + [x + 2 for x in road_xs] + [w]
    by = [0] + [y + 2 for y in road_ys] + [h]
    for i in range(len(bx) - 1):
        for j in range(len(by) - 1):
            x0, x1 = bx[i] + 1, bx[i + 1] - 2
            y0, y1 = by[j] + 1, by[j + 1] - 2
            if x1 - x0 < 7 or y1 - y0 < 6 or rng.random() < 0.15:
                continue  # block too small, or empty lot
            bw = rng.randint(6, min(10, x1 - x0))
            bh = rng.randint(5, min(8, y1 - y0))
            bxi = rng.randint(x0, x1 - bw)
            byi = rng.randint(y0, y1 - bh)
            _carve_building(m, rng, bxi, byi, bw, bh)
    return m


def _carve_building(m: GameMap, rng: random.Random, x0, y0, bw, bh):
    t = m.tiles
    for y in range(y0, y0 + bh):
        for x in range(x0, x0 + bw):
            edge = x in (x0, x0 + bw - 1) or y in (y0, y0 + bh - 1)
            t[y][x] = WALL if edge else FLOOR

    def side_clear(side):
        """Does a 2-tile door on this side reach open ground within a few tiles?"""
        if side == "N":
            sx, sy, dx, dy = x0 + bw // 2, y0 - 1, 0, -1
        elif side == "S":
            sx, sy, dx, dy = x0 + bw // 2, y0 + bh, 0, 1
        elif side == "W":
            sx, sy, dx, dy = x0 - 1, y0 + bh // 2, -1, 0
        else:
            sx, sy, dx, dy = x0 + bw, y0 + bh // 2, 1, 0
        for k in range(4):
            cx, cy = sx + dx * k, sy + dy * k
            if not (0 <= cx < m.w and 0 <= cy < m.h):
                return False
            if t[cy][cx] in WALKABLE and t[cy][cx] != FLOOR:
                return True
        return False

    sides = ["N", "S", "W", "E"]
    rng.shuffle(sides)
    door = next((s for s in sides if side_clear(s)), "S")
    if door == "N":
        t[y0][x0 + bw // 2] = DOOR
        t[y0][x0 + bw // 2 - 1] = DOOR
    elif door == "S":
        t[y0 + bh - 1][x0 + bw // 2] = DOOR
        t[y0 + bh - 1][x0 + bw // 2 - 1] = DOOR
    elif door == "W":
        t[y0 + bh // 2][x0] = DOOR
        t[y0 + bh // 2][x0 - 1 if x0 > 0 else x0] = DOOR
    else:
        t[y0 + bh // 2][x0 + bw - 1] = DOOR
        if x0 + bw < m.w:
            t[y0 + bh // 2][x0 + bw] = DOOR

    m.buildings.append((x0, y0, bw, bh))
