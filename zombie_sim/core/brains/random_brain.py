"""Uniform-random brains for both species (D1 baseline: the floor every
other brain is compared against)."""
import random as _random


class RandomSurvivorBrain:
    """Uniform-random movement + occasional item use. Cannot deposit
    deliberately — deposits happen whenever it wanders through the safe house
    carrying anything, exactly like any other brain's motor output."""
    kind = "random"

    def think(self, world, s):
        cfg = world.cfg
        s.state_label = "RANDOM"
        s.fleeing = False
        r = world.rng.random
        roll = r()
        if roll < 0.10 and _carries_usable(s):
            s.action = _pick_consumable(s)
        else:
            s.action = None
        # random 8-direction walk at run speed (same motor as RL movers)
        import math
        ang = r() * 2 * math.pi
        dx, dy = math.cos(ang), math.sin(ang)
        s.path = None
        s.goal = None
        s.vel = (dx * cfg.survivor_run, dy * cfg.survivor_run)
        s.heading = math.atan2(dy, dx)


def _pick_consumable(s):
    if s.hunger > 70 and s.inventory["food"] > 0:
        return ("eat", None)
    if s.thirst > 70 and s.inventory["water"] > 0:
        return ("drink", None)
    if s.hp < 45 and s.inventory["meds"] > 0:
        return ("heal", None)
    return None


def _carries_usable(s):
    return (s.inventory["food"] > 0 and s.hunger > 70) or \
           (s.inventory["water"] > 0 and s.thirst > 70) or \
           (s.inventory["meds"] > 0 and s.hp < 45)


class RandomZombieBrain:
    """Uniform-random zombie movement; bites auto-resolve on contact."""
    kind = "random"

    def think(self, world, z):
        z.last_known = None
        z.target = None
        import math
        ang = world.rng.random() * 2 * math.pi
        z.vel = (math.cos(ang) * world.cfg.zombie_walk,
                 math.sin(ang) * world.cfg.zombie_walk)
