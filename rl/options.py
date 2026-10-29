"""High-level options for hierarchical RL: the network picks WHAT to do, scripted code does HOW.

  0 HOLD    stand still (silent)          3 RETURN  walk to the safe house and deposit
  1 SCOUT   search the next building      4 FLEE    run from visible zombies (or home)
  2 LOOT    fetch best remembered supply  5 ENGAGE  shoot / fight the nearest reachable zombie
"""
import math
from zombie_sim.core.brains import utility as lad
from zombie_sim.core.brains import tactical as tac

N_OPT = 6
NAMES = ["HOLD", "SCOUT", "LOOT", "RETURN", "FLEE", "ENGAGE"]


def _stop(s):
    s.path = None; s.goal = None; s.vel = (0.0, 0.0); s.action = None


def _visible_threats(world, s):
    from zombie_sim.core.perception import can_see
    cfg = world.cfg
    return [z for z in world.spatial_z.query(s.x, s.y, cfg.vision_range)
            if z.alive and can_see(world.game_map, s, z.x, z.y, cfg.vision_range, cfg.vision_fov_deg)]


def run_option(world, s, opt, apply_action):
    m, cfg = world.game_map, world.cfg
    if opt == 0:
        s.state_label = "HOLD"; _stop(s)
    elif opt == 1:
        s.state_label = "SCOUTING"; s.action = None
        if s.path is None:
            tac._explore_pairs(world, s)
    elif opt == 2:
        s.action = None
        sid = lad._pick_target(world, s)
        if sid is None:
            s.state_label = "HOLD"; _stop(s)
        else:
            s.state_label = "LOOTING"
            sx, sy = s.memory_sup[sid]
            if not lad._goto(s, m, sx, sy):
                s.memory_sup.pop(sid, None)
    elif opt == 3:
        s.action = None
        if world.safehouse is not None and world.in_safehouse(s):
            s.state_label = "DEPOSITING"; _stop(s)
        else:
            s.state_label = "RETURNING"
            if world.safehouse is None or not lad._goto_safehouse(world, s):
                _stop(s)
    elif opt == 4:
        s.action = None
        th = _visible_threats(world, s)
        s.state_label = "FLEEING"
        if th:
            lad._flee(world, s, th, cfg)
        elif world.safehouse is not None and not world.in_safehouse(s):
            lad._goto_safehouse(world, s)
        else:
            _stop(s)
    else:
        apply_action(world, s, 9)   # ENGAGE (env's own attack action)
        # a gunner fires at range; an unarmed agent only swings at reach
