"""Agent state: survivors, zombies, supplies. Behaviour lives in brains/."""
import math
from dataclasses import dataclass


class Agent:
    __slots__ = ("id", "kind", "x", "y", "heading", "alive", "vel", "skin")

    def __init__(self, aid, kind, x, y):
        self.id = aid
        self.kind = kind
        self.x = x
        self.y = y
        self.heading = 0.0
        self.alive = True
        self.vel = (0.0, 0.0)
        self.skin = 0


class Survivor(Agent):
    __slots__ = ("hp", "hunger", "thirst", "inventory", "infected", "infection",
                 "fear", "memory_z", "memory_sup", "path", "path_i", "action",
                 "cooldown", "fleeing", "goal", "rl", "gun", "ammo", "state_label",
                 "has_bat")

    def __init__(self, aid, x, y, hp=100.0):
        super().__init__(aid, "survivor", x, y)
        self.hp = hp
        self.rl = False  # True when an RL policy drives this agent instead of a scripted brain
        self.gun = False
        self.ammo = 0
        self.has_bat = False
        self.hp = hp
        self.hunger = 0.0
        self.thirst = 0.0
        # survivors start with nothing: whatever they carry home is deposited
        # in the safe house, so spawning with supplies would be free stockpile
        self.inventory = {"food": 0, "water": 0, "meds": 0}
        self.infected = False
        self.infection = 0.0
        self.fear = 0.0
        self.memory_z = {}    # zombie_id -> (x, y, tick_seen)
        self.memory_sup = {}  # supply_id -> (x, y)
        self.path = None
        self.path_i = 0
        self.action = None    # ("eat"|"drink"|"heal"|"attack", target_id)
        self.cooldown = 0
        self.fleeing = False
        self.goal = None
        self.state_label = "idle"   # what the brain is doing (viewer inspector)


class Zombie(Agent):
    __slots__ = ("state", "target", "last_known", "last_seen", "attack_cd", "hp",
                 "inv_since", "rl")

    def __init__(self, aid, x, y, hp=100.0):
        super().__init__(aid, "zombie", x, y)
        self.hp = hp
        self.rl = False  # True when an RL policy drives this zombie instead of the FSM
        self.state = "wander"
        self.target = None
        self.last_known = None
        self.last_seen = -10**9
        self.attack_cd = 0
        self.inv_since = 0


@dataclass
class Supply:
    id: int
    x: float
    y: float
    kind: str  # food | water | meds | gun | ammo
    taken: bool = False


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)
