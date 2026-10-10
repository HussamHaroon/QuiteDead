"""Authored levels: YAML-defined maps with buildings, supplies, spawns and props.

Each level is a fixed hand-designed scenario; the training curriculum advances
through them by performance (see rl/train.py).
"""
import os

import yaml

from .grid import GameMap, GRASS, ROAD, FLOOR, WALL, DOOR, WALKABLE


class Level:
    def __init__(self, idx, name, game_map, supplies, survivor_spawns,
                 zombie_spawns, props, n_zombies, advance_at):
        self.idx = idx
        self.name = name
        self.map = game_map
        self.supplies = supplies          # [(x, y, kind)]
        self.survivor_spawns = survivor_spawns
        self.zombie_spawns = zombie_spawns
        self.props = props                # [(sprite_relpath, x, y)]
        self.n_zombies = n_zombies
        self.advance_at = advance_at


def load_levels(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [_build(i, d) for i, d in enumerate(data)]


def _build(idx, d):
    w = h = int(d.get("size", 32))
    m = GameMap(w, h)
    t = m.tiles

    for x in range(w):
        t[0][x] = t[h - 1][x] = WALL
    for y in range(h):
        t[y][0] = t[y][w - 1] = WALL

    for rx in d.get("roads_x", []):
        for y in range(1, h - 1):
            t[y][rx] = ROAD
            if rx + 1 < w - 1:
                t[y][rx + 1] = ROAD
    for ry in d.get("roads_y", []):
        for x in range(1, w - 1):
            t[ry][x] = ROAD
            if ry + 1 < h - 1:
                t[ry + 1][x] = ROAD
    for (px, py, pw, ph) in d.get("plazas", []):
        for y in range(py, min(py + ph, h - 1)):
            for x in range(px, min(px + pw, w - 1)):
                t[y][x] = ROAD

    for b in d.get("buildings", []):
        x0, y0, bw, bh = (int(v) for v in b[:4])
        door = b[4] if len(b) > 4 else "S"
        for y in range(y0, min(y0 + bh, h)):
            for x in range(x0, min(x0 + bw, w)):
                edge = x in (x0, x0 + bw - 1) or y in (y0, y0 + bh - 1)
                t[y][x] = WALL if edge else FLOOR
        cx, cy = x0 + bw // 2, y0 + bh // 2
        if door == "N":
            t[y0][cx] = t[y0][min(cx + 1, x0 + bw - 1)] = DOOR
        elif door == "S":
            t[y0 + bh - 1][cx] = t[y0 + bh - 1][min(cx + 1, x0 + bw - 1)] = DOOR
        elif door == "W":
            t[cy][x0] = t[min(cy + 1, y0 + bh - 1)][x0] = DOOR
        else:
            t[cy][x0 + bw - 1] = t[min(cy + 1, y0 + bh - 1)][x0 + bw - 1] = DOOR
        m.buildings.append((x0, y0, bw, bh))

    supplies = []
    for x, y in d.get("food", []):
        supplies.append((x + 0.5, y + 0.5, "food"))
    for x, y in d.get("water", []):
        supplies.append((x + 0.5, y + 0.5, "water"))
    for x, y in d.get("meds", []):
        supplies.append((x + 0.5, y + 0.5, "meds"))
    for x, y, kind in supplies:
        if m.blocked(int(x), int(y)):
            raise ValueError(f"level '{d.get('name', idx)}': supply {kind} at "
                             f"({x},{y}) is on a wall")
    for x, y in d.get("survivor_spawns", []) + d.get("zombie_spawns", []):
        if m.blocked(x, y):
            raise ValueError(f"level '{d.get('name', idx)}': spawn at ({x},{y}) is on a wall")

    props = [(p[0], int(p[1]), int(p[2])) for p in d.get("props", [])]
    return Level(idx, d.get("name", f"level_{idx}"), m, supplies,
                 [tuple(s) for s in d.get("survivor_spawns", [])],
                 [tuple(z) for z in d.get("zombie_spawns", [])],
                 props, int(d.get("n_zombies", 8)), float(d.get("advance_at", 0.6)))
