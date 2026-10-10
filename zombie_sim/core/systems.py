"""Per-tick systems: movement, looting, deposits, combat, infection, needs, reanimation."""
import itertools
import math

from .agents import Zombie
from .perception import line_of_sight

DEPOSIT_KINDS = ("food", "water", "meds")


def move_agents(world):
    cfg = world.cfg
    m = world.game_map
    for a in world.all_agents():
        if not a.alive:
            continue
        vx = vy = 0.0
        if a.kind == "survivor" and a.path is not None:
            if a.path_i >= len(a.path):
                a.path = None
                a.goal = None
            else:
                wp = a.path[a.path_i]
                dx, dy = wp[0] - a.x, wp[1] - a.y
                d = math.hypot(dx, dy)
                speed = cfg.survivor_run if a.fear > 0.55 else cfg.survivor_walk
                if d < 0.2:
                    a.path_i += 1
                    continue
                vx, vy = dx / d * speed, dy / d * speed
        else:
            vx, vy = a.vel

        if vx == 0.0 and vy == 0.0:
            continue
        old_x, old_y = a.x, a.y
        if m.walkable(a.x + vx, a.y):
            a.x += vx
        if m.walkable(a.x, a.y + vy):
            a.y += vy
        moved = math.hypot(a.x - old_x, a.y - old_y)
        if moved > 1e-4:
            a.heading = math.atan2(a.y - old_y, a.x - old_x)
            world.noise.emit(a.x, a.y, cfg.noise_run if _is_running(a) else cfg.noise_walk)
        elif a.kind == "zombie":
            # stuck against a wall: pick a new wander direction
            a.heading += world.rng.uniform(1.0, 2.5)
            a.vel = (math.cos(a.heading) * cfg.zombie_walk, math.sin(a.heading) * cfg.zombie_walk)
        elif a.kind == "survivor" and a.fleeing:
            # cornered — turn and fight whatever is closest
            near = None
            best = 1e18
            for z in world.spatial_z.query(a.x, a.y, 1.4):
                if not z.alive:
                    continue
                d = (z.x - a.x) ** 2 + (z.y - a.y) ** 2
                if d < best:
                    best, near = d, z
            if near is not None and best <= 1.4 ** 2 and a.cooldown <= 0:
                a.action = ("attack", near.id)


def _is_running(a):
    if getattr(a, "fleeing", False) or getattr(a, "fear", 0.0) > 0.55:
        return True
    # RL movers set velocities directly; they travel at run speed
    return a.kind == "survivor" and getattr(a, "rl", False) and a.vel != (0.0, 0.0)


def loot_tick(world):
    for sup in world.supplies.values():
        if sup.taken:
            continue
        # query radius must over-approximate the accept radius (0.7416)
        for s in world.spatial_s.query(sup.x, sup.y, 0.8):
            if s.alive and s.kind == "survivor" \
                    and (s.x - sup.x) ** 2 + (s.y - sup.y) ** 2 < 0.55:
                if sup.kind == "gun":
                    s.gun = True
                elif sup.kind == "ammo":
                    s.ammo += int(world.cfg.ammo_per_crate)
                elif sup.kind == "bat":
                    s.has_bat = True
                else:
                    s.inventory[sup.kind] += 1
                sup.taken = True
                world.log.add(world.t, "loot", who=s.id, kind=sup.kind,
                              x=round(sup.x, 1), y=round(sup.y, 1))
                break


def deposit_tick(world):
    """Standing in the safe house drops all carried food/water/meds into the
    stockpile — the payoff of the leave / loot / return mission loop."""
    if world.safehouse is None:
        return
    for s in world.survivors:
        if not s.alive or not world.in_safehouse(s):
            continue
        n = sum(s.inventory[k] for k in DEPOSIT_KINDS)
        if n <= 0:
            continue
        for k in DEPOSIT_KINDS:
            s.inventory[k] = 0
        world.stockpile += n
        world.log.add(world.t, "deposit", who=s.id, n=n,
                      x=round(s.x, 1), y=round(s.y, 1))


def combat_tick(world):
    cfg = world.cfg
    for z in world.zombies:
        if not z.alive:
            continue
        if z.attack_cd > 0:
            z.attack_cd -= 1
            continue
        target = None
        best = 1e18
        for s in world.spatial_s.query(z.x, z.y, 0.8):
            if s.alive and s.kind == "survivor":
                d = (s.x - z.x) ** 2 + (s.y - z.y) ** 2
                if d < best:
                    best, target = d, s
        if target is None or best > 0.8 ** 2:
            continue  # spatial query over-approximates cell overlap — enforce true reach
        z.attack_cd = cfg.zombie_attack_cooldown
        world.noise.emit(z.x, z.y, 5.0)
        if world.rng.random() < cfg.zombie_attack_chance:
            if not target.infected and world.rng.random() < cfg.bite_infection_chance:
                target.infected = True
                target.infection = 0.0
                world.log.add(world.t, "bite", zombie=z.id, who=target.id,
                              x=round(target.x, 1), y=round(target.y, 1))
            else:
                target.hp -= cfg.zombie_damage
                if target.hp <= 0:
                    die(world, target, "zombie")


MELEE_REACH = 1.45  # cornered-fight orders come from a 1.4-radius query


def survivor_attacks(world):
    """Weapon ladder: pistol (ranged, loud) > bat (quiet melee) > shove
    (unarmed escape tool — pushes the zombie back, cannot kill)."""
    cfg = world.cfg
    m = world.game_map
    for s in world.survivors:
        if not s.alive or s.cooldown > 0:
            s.cooldown = max(0, s.cooldown - 1)
            continue
        if s.action is None or s.action[0] != "attack":
            continue
        zid = s.action[1]
        z = world.agents_by_id.get(zid)
        s.action = None
        if z is None or not z.alive:
            s.cooldown = cfg.shove_cooldown
            continue
        d = math.hypot(z.x - s.x, z.y - s.y)

        if s.gun and s.ammo > 0 and d <= cfg.gun_range \
                and line_of_sight(m, s.x, s.y, z.x, z.y):
            # pistol shot: loud, single round, probabilistic hit
            s.ammo -= 1
            s.cooldown = cfg.gun_cooldown
            world.noise.emit(s.x, s.y, cfg.gun_noise)
            s.state_label = "SHOOTING"
            hit = world.rng.random() < cfg.gun_chance
            world.log.add(world.t, "shot", who=s.id, target=z.id, hit=bool(hit),
                          x=round(s.x, 1), y=round(s.y, 1))
            if hit:
                z.hp -= cfg.gun_damage
                if z.hp <= 0:
                    die(world, z, "killed_by_survivor")
                    world.log.add(world.t, "kill", killer=s.id, who=z.id)
        elif s.has_bat and d <= MELEE_REACH:
            # bat: quiet melee weapon — the only unarmed-adjacent way to kill
            s.cooldown = cfg.bat_cooldown
            world.noise.emit(s.x, s.y, cfg.bat_noise)
            s.state_label = "FIGHTING"
            if world.rng.random() < cfg.bat_chance:
                z.hp -= cfg.bat_damage
                if z.hp <= 0:
                    die(world, z, "killed_by_survivor")
                    world.log.add(world.t, "kill", killer=s.id, who=z.id)
        elif d <= MELEE_REACH:
            # shove: bare hands cannot kill — knock the zombie back to buy
            # escape room instead
            s.cooldown = cfg.shove_cooldown
            world.noise.emit(s.x, s.y, cfg.shove_noise)
            s.state_label = "SHOVING"
            dx, dy = z.x - s.x, z.y - s.y
            dist = max(0.3, math.hypot(dx, dy))
            nx, ny = z.x + dx / dist * cfg.shove_push, z.y + dy / dist * cfg.shove_push
            if m.walkable(nx, ny):
                z.x, z.y = nx, ny
        else:
            s.cooldown = cfg.shove_cooldown


def infection_tick(world):
    cfg = world.cfg
    for s in world.survivors:
        if s.alive and s.infected:
            s.infection += 100.0 / cfg.incubation_ticks
            if s.infection >= 100.0:
                die(world, s, "infection")


def needs_tick(world):
    cfg = world.cfg
    for s in world.survivors:
        if not s.alive:
            continue
        s.hunger = min(100.0, s.hunger + cfg.hunger_rate)
        s.thirst = min(100.0, s.thirst + cfg.thirst_rate)
        if s.hunger >= 100.0 or s.thirst >= 100.0:
            s.hp -= cfg.hunger_damage
            if s.hp <= 0:
                die(world, s, "starvation")
                continue
        act = s.action
        if act is None:
            continue
        if act[0] == "eat" and s.inventory["food"] > 0:
            s.inventory["food"] -= 1
            s.hunger = max(0.0, s.hunger - 55.0)
        elif act[0] == "drink" and s.inventory["water"] > 0:
            s.inventory["water"] -= 1
            s.thirst = max(0.0, s.thirst - 65.0)
        elif act[0] == "heal" and s.inventory["meds"] > 0:
            s.inventory["meds"] -= 1
            s.hp = min(100.0, s.hp + 40.0)
        s.action = None


def reanimate_tick(world):
    cfg = world.cfg
    if not cfg.reanimation_enabled:
        world.corpses.clear()
        return
    due = itertools.takewhile(lambda c: c[0] <= world.t,
                              sorted(world.corpses, key=lambda c: c[0]))
    for at, x, y in list(due):
        world.corpses.remove((at, x, y))
        spawn_zombie(world, x, y)


def die(world, agent, cause):
    agent.alive = False
    world.log.add(world.t, "death", who=agent.id, kind=agent.kind, cause=cause,
                  x=round(agent.x, 1), y=round(agent.y, 1), heading=agent.heading)
    if agent.kind == "survivor":
        world.corpses.append((world.t + world.cfg.reanimation_delay, agent.x, agent.y))


def spawn_zombie(world, x, y):
    z = Zombie(world.next_id(), x, y, hp=world.cfg.zombie_hp)
    world.zombies.append(z)
    world.agents_by_id[z.id] = z
    world.log.add(world.t, "turn", who=z.id, x=round(x, 1), y=round(y, 1))
