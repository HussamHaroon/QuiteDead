"""World: owns state and the deterministic tick loop.

Order per tick: spatial hash rebuild -> noise decay -> brains think
(survivors then zombies, id order) -> movement -> looting -> safe-house
deposits -> combat -> infection -> needs -> reanimation. One shared RNG,
stable iteration order: same seed + same config => identical run.
"""
import itertools
import random

from .agents import Survivor, Zombie, Supply
from .config import Config
from .events import EventLog
from .grid import generate_map, ROAD, DOOR, FLOOR, GRASS, WALKABLE
from .noise import NoiseField
from .spatial import SpatialHash
from .systems import (move_agents, loot_tick, deposit_tick, combat_tick,
                      survivor_attacks, infection_tick, needs_tick,
                      reanimate_tick)
from .brains import fsm, utility


def _nearest_building(buildings, points):
    """Building rect (x, y, w, h) whose center is closest to the points' centroid."""
    if not buildings or not points:
        return None
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    best, best_d = None, None
    for b in buildings:
        d = (b[0] + b[2] / 2.0 - cx) ** 2 + (b[1] + b[3] / 2.0 - cy) ** 2
        if best_d is None or d < best_d:
            best, best_d = (b[0], b[1], b[2], b[3]), d
    return best


class World:
    def __init__(self, cfg: Config, seed=None, level=None):
        self.cfg = cfg
        self.seed = cfg.seed if seed is None else seed
        self.rng = random.Random(self.seed)
        self.level = level
        self.game_map = level.map if level is not None else generate_map(self.rng, cfg.width, cfg.height)
        self.noise = NoiseField(cfg.width, cfg.height)
        self.spatial_s = SpatialHash()
        self.spatial_z = SpatialHash()

        self._next_id = 0
        self.survivors = []
        self.zombies = []
        self.supplies = {}
        self.agents_by_id = {}
        self.corpses = []  # (due_tick, x, y)
        self.history = []  # (tick, survivors, infected, zombies) every 30 ticks
        self.stockpile = 0  # food/water/meds deposited in the safe house
        self.safehouse = None  # (x, y, w, h) tile rect, resolved in _resolve_safehouse

        self._resolve_safehouse()
        self._spawn_supplies()
        self._spawn_agents()

        self.t = 0
        self.log = EventLog()
        self.log.add(0, "sim_start", seed=self.seed,
                     survivors=len(self.survivors), zombies=len(self.zombies),
                     supplies=len(self.supplies))

    # ------------------------------------------------------------------ setup
    def next_id(self):
        self._next_id += 1
        return self._next_id

    def _rand_open_tile(self, kinds=WALKABLE):
        for _ in range(500):
            x = self.rng.randrange(1, self.cfg.width - 1)
            y = self.rng.randrange(1, self.cfg.height - 1)
            if self.game_map.tiles[y][x] in kinds:
                return x + 0.5, y + 0.5
        return self.cfg.width / 2, self.cfg.height / 2

    def _spawn_supplies(self):
        kinds = ("food", "water", "meds")
        if self.level is not None:
            for x, y, kind in self.level.supplies:
                self._add_supply(x, y, kind)
            return
        for (bx, by, bw, bh) in self.game_map.buildings:
            if self.rng.random() < self.cfg.supply_per_building:
                x = bx + 1 + self.rng.randrange(max(1, bw - 2))
                y = by + 1 + self.rng.randrange(max(1, bh - 2))
                if self.game_map.tiles[y][x] == FLOOR:
                    self._add_supply(x + 0.5, y + 0.5, self.rng.choice(kinds))
        road = (ROAD, DOOR)
        for _ in range(self.cfg.n_supply_outdoor):
            x, y = self._rand_open_tile(road)
            self._add_supply(x, y, self.rng.choice(("food", "water")))
        # a rare few guns (mostly inside buildings) and some ammo crates outside —
        # no survivor spawns armed; finding them is part of the loop. Two gun +
        # ammo pairs sit in the safe house (the camp armory) so a couple of
        # survivors actually get armed early and the shooting loop can start.
        placed = 0
        if self.safehouse is not None:
            x0, y0, sw, sh = self.safehouse
            if sw > 2 and sh > 2:
                for _pair in range(2):
                    tx = self.rng.randrange(x0 + 1, x0 + sw - 1)
                    ty = self.rng.randrange(y0 + 1, y0 + sh - 1)
                    if self.game_map.blocked(tx, ty):
                        continue
                    self._add_supply(tx + 0.5, ty + 0.5, "gun")
                    self._add_supply(tx + 0.5, ty + 0.5, "ammo")
                    placed += 1
        for _ in range(max(0, self.cfg.n_guns_spawn - placed)):
            x, y = self._rand_open_tile((FLOOR, ROAD))
            self._add_supply(x, y, "gun")
        for _ in range(max(0, self.cfg.n_ammo_spawn - placed)):
            x, y = self._rand_open_tile(road)
            self._add_supply(x, y, "ammo")

    def _add_supply(self, x, y, kind):
        sid = self.next_id()
        self.supplies[sid] = Supply(sid, x, y, kind)

    # -------------------------------------------------------------- safe house
    def _resolve_safehouse(self):
        """Level-designated safe house, else the building nearest the intended
        survivor spawn centroid. Falls back to None when the map has no buildings."""
        rect = getattr(self.level, "safehouse", None) if self.level is not None else None
        if rect is None and self.level is not None and self.level.survivor_spawns:
            rect = _nearest_building(self.game_map.buildings,
                                     [(x + 0.5, y + 0.5) for x, y in self.level.survivor_spawns])
        if rect is None:
            centers = [self._rand_open_tile((ROAD, DOOR)) for _ in range(3)]
            rect = _nearest_building(self.game_map.buildings, centers)
        self.safehouse = rect

    def in_safehouse(self, agent):
        """True when the agent's tile lies inside the safe house rect."""
        r = self.safehouse
        if r is None:
            return False
        x0, y0, w, h = r
        return x0 <= int(agent.x) < x0 + w and y0 <= int(agent.y) < y0 + h

    def _safehouse_center(self):
        x0, y0, w, h = self.safehouse
        return x0 + w / 2.0, y0 + h / 2.0

    def _safehouse_spot(self, cx, cy):
        """Random interior tile of the safe house (center / open tile as fallbacks)."""
        x0, y0, w, h = self.safehouse
        if w > 2 and h > 2:
            for _ in range(50):
                tx = self.rng.randrange(x0 + 1, x0 + w - 1)
                ty = self.rng.randrange(y0 + 1, y0 + h - 1)
                if not self.game_map.blocked(tx, ty):
                    return tx + 0.5, ty + 0.5
        if not self.game_map.blocked(int(cx), int(cy)):
            return cx, cy
        return self._rand_open_tile((ROAD, DOOR))

    def _spawn_agents(self):
        # survivors start inside/near the safe house (mission loop home base);
        # zombies spawn scattered (or at authored level spots), never near it
        road = (ROAD, DOOR)
        if self.safehouse is not None:
            centers = [self._safehouse_center()]
        elif self.level is not None and self.level.survivor_spawns:
            centers = [(x + 0.5, y + 0.5) for x, y in self.level.survivor_spawns]
        else:
            centers = [self._rand_open_tile(road) for _ in range(3)]
        skins = itertools.cycle(range(6))
        for i in range(self.cfg.n_survivors):
            cx, cy = centers[i % len(centers)]
            if self.safehouse is not None:
                x, y = self._safehouse_spot(cx, cy)
            else:
                x = min(max(cx + self.rng.gauss(0, 2.5), 1.5), self.cfg.width - 1.5)
                y = min(max(cy + self.rng.gauss(0, 2.5), 1.5), self.cfg.height - 1.5)
                if self.game_map.blocked(int(x), int(y)):
                    x, y = self._rand_open_tile(road)
            s = Survivor(self.next_id(), x, y, hp=self.cfg.survivor_hp)
            s.skin = next(skins)
            self.survivors.append(s)
            self.agents_by_id[s.id] = s
        # zombies spawn scattered (or at authored level spots), never on survivors
        if self.level is not None and self.level.zombie_spawns:
            spots = list(self.level.zombie_spawns)
            self.rng.shuffle(spots)
            for x, y in spots[:self.cfg.n_zombies]:
                self._add_zombie(x + 0.5, y + 0.5)
            for _ in range(max(0, self.cfg.n_zombies - len(spots))):
                x, y = self._rand_open_tile()
                self._add_zombie(x, y)
            return
        for _ in range(self.cfg.n_zombies):
            x, y = self._rand_open_tile()
            for _try in range(200):
                cx = self.rng.uniform(1.5, self.cfg.width - 1.5)
                cy = self.rng.uniform(1.5, self.cfg.height - 1.5)
                if self.game_map.blocked(int(cx), int(cy)):
                    continue
                if all((cx - sx) ** 2 + (cy - sy) ** 2 > 64 for sx, sy in centers):
                    x, y = cx, cy
                    break
            self._add_zombie(x, y)

    def _add_zombie(self, x, y):
        z = Zombie(self.next_id(), x, y, hp=self.cfg.zombie_hp)
        self.zombies.append(z)
        self.agents_by_id[z.id] = z

    # ------------------------------------------------------------------- loop
    def all_agents(self):
        return itertools.chain(self.survivors, self.zombies)

    def step(self):
        cfg = self.cfg
        self.spatial_s.rebuild(self.survivors)
        self.spatial_z.rebuild(self.zombies)
        self.noise.decay(cfg.noise_decay)

        for s in self.survivors:
            if s.alive and not s.rl:
                utility.think(self, s)
        for z in self.zombies:
            if z.alive:
                fsm.think(self, z)

        move_agents(self)
        loot_tick(self)
        deposit_tick(self)
        combat_tick(self)
        survivor_attacks(self)
        infection_tick(self)
        needs_tick(self)
        reanimate_tick(self)
        self._reinforce_tick()

        self.t += 1
        if self.t % 30 == 0:
            self.history.append((self.t, *self.counts()))

    def _reinforce_tick(self):
        """Slow trickle of fresh wanderers: the horde grows over time, so the
        survivors' only real clock is how fast they finish the quota."""
        every = self.cfg.reinforce_every_ticks
        if not every or self.t == 0 or self.t % every != 0:
            return
        for _ in range(self.cfg.reinforce_count):
            for _try in range(60):
                x, y = self._rand_open_tile()
                if self.safehouse is not None:
                    x0, y0, sw, sh = self.safehouse
                    if x0 <= int(x) < x0 + sw and y0 <= int(y) < y0 + sh:
                        continue  # never materialize a zombie inside the base
                break
            self._add_zombie(x, y)
            self.log.add(self.t, "reinforce", x=round(x, 1), y=round(y, 1))

    def counts(self):
        surv = sum(1 for s in self.survivors if s.alive)
        infected = sum(1 for s in self.survivors if s.alive and s.infected)
        zomb = sum(1 for z in self.zombies if z.alive)
        return surv, infected, zomb

    def run(self, ticks, report_every=0):
        for _ in range(ticks):
            self.step()
            if report_every and self.t % report_every == 0:
                surv, inf, zomb = self.counts()
                print(f"  tick {self.t:>6} | survivors {surv:>3} | infected {inf:>2} "
                      f"| zombies {zomb:>3} | loot {self.log.count('loot'):>3} "
                      f"| stock {self.stockpile:>3} | shots {self.log.count('shot'):>3}")
