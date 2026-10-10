"""Pygame observer view: god-mode map, sprites, vision cone, noise overlay, HUD.

Controls: SPACE pause | 1-4 speed | TAB next survivor | V vision cone |
N noise heatmap | ESC quit.
"""
import math
import os

import pygame

TILE = 16
HUD_H = 46
SPEEDS = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 5, pygame.K_4: 15}

COLORS = {
    0: (38, 52, 36),    # grass
    1: (58, 58, 64),    # road
    2: (78, 66, 55),    # interior floor
    3: (112, 112, 124), # wall
    4: (146, 108, 56),  # door
}
SUPPLY_COLORS = {"food": (232, 146, 58), "water": (86, 156, 232), "meds": (226, 88, 96)}

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", "sprites", "top-down-shooter", "PNG")
SUPPLY_COLORS = {"food": (232, 146, 58), "water": (86, 156, 232), "meds": (226, 88, 96)}

try:
    from .sprites import load_bank
except ImportError:  # run as a plain script
    from sprites import load_bank


def _is_night(t):
    phase = (math.sin(2 * math.pi * t / 2400.0 - math.pi / 2) + 1) / 2
    return phase > 0.6


def _make_world(cfg, level=None):
    from zombie_sim.core.sim import World
    return World(cfg, level=level)


def _prerender_map(world, tiles=None):
    m = world.game_map
    surf = pygame.Surface((m.w * TILE, m.h * TILE))
    for y in range(m.h):
        for x in range(m.w):
            r = pygame.Rect(x * TILE, y * TILE, TILE, TILE)
            t = m.tiles[y][x]
            if tiles is not None and t in tiles:
                variants = tiles[t]
                surf.blit(variants[(x * 7 + y * 13) % len(variants)], r)
                if t == 3:  # wall: top highlight for depth
                    pygame.draw.line(surf, (90, 30, 40), r.topleft, (r.right - 1, r.top))
                continue
            pygame.draw.rect(surf, COLORS[t], r)
            if t == 3:
                pygame.draw.line(surf, (140, 140, 152), r.topleft, (r.right - 1, r.top))
                pygame.draw.line(surf, (66, 66, 74), (r.left, r.bottom - 1), (r.right - 1, r.bottom - 1))
    return surf


def _to_px(x, y):
    return x * TILE, y * TILE


def _rotated(cache, img, heading):
    dx, dy = math.cos(heading), math.sin(heading)
    deg = -math.degrees(math.atan2(dx, -dy))
    key = int(deg // 5) * 5
    s = cache.get((id(img), key))
    if s is None:
        s = pygame.transform.rotate(img, key)
        cache[(id(img), key)] = s
    return s


def _draw_vision_cone(surface, world, agent):
    import pygame.draw as draw
    cfg = world.cfg
    origin = _to_px(agent.x, agent.y)
    half = math.radians(cfg.vision_fov_deg / 2)
    n = 26
    pts = [origin]
    for i in range(n + 1):
        a = agent.heading - half + (2 * half) * i / n
        dx, dy = math.cos(a), math.sin(a)
        d = 0.0
        while d < cfg.vision_range:
            step = min(0.3, cfg.vision_range - d)
            nx, ny = agent.x + dx * (d + step), agent.y + dy * (d + step)
            if world.game_map.blocked(int(nx), int(ny)):
                break
            d += step
        pts.append(_to_px(agent.x + dx * d, agent.y + dy * d))
    if len(pts) > 2:
        draw.polygon(surface, (255, 230, 120, 40), pts)
        draw.polygon(surface, (255, 230, 120, 90), pts, 1)


def _draw_noise(surface, world):
    g = world.noise.grid
    for i, v in enumerate(g):
        if v > 0.4:
            a = min(150, int(v * 3))
            x, y = i % world.game_map.w, i // world.game_map.w
            pygame.draw.rect(surface, (255, 60, 30, a),
                             (x * TILE, y * TILE, TILE, TILE))


def run_viewer(cfg, screenshot=None, frames=240, level=None):
    if screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    m_px = cfg.width * TILE
    screen = pygame.display.set_mode((m_px, m_px + HUD_H))
    pygame.display.set_caption("QuiteDead")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 15)

    world = _make_world(cfg, level=level)
    bank = load_bank(level.theme if level is not None else None)
    background = _prerender_map(world, getattr(bank, "map_tiles", None))
    rot_cache = {}
    selected = next((s for s in world.survivors if s.alive), None)
    ticks_per_frame = 10 if screenshot else 1  # screenshot runs are for verification
    paused = False
    show_cone = True
    show_noise = False
    show_daynight = True
    night_overlay = pygame.Surface((m_px, m_px), pygame.SRCALPHA)
    extinct = False

    overlay = pygame.Surface((m_px, m_px), pygame.SRCALPHA)
    done = False
    frame = 0
    while not done:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                done = True
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    done = True
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key in SPEEDS:
                    ticks_per_frame = SPEEDS[ev.key]
                elif ev.key == pygame.K_v:
                    show_cone = not show_cone
                elif ev.key == pygame.K_n:
                    show_noise = not show_noise
                elif ev.key == pygame.K_t:
                    show_daynight = not show_daynight
                elif ev.key == pygame.K_r and extinct:
                    world = _make_world(cfg, level=level)
                    background = _prerender_map(world, getattr(bank, "map_tiles", None))
                    selected = next((s for s in world.survivors if s.alive), None)
                    extinct = False
                elif ev.key == pygame.K_TAB:
                    alive = [s for s in world.survivors if s.alive]
                    if alive:
                        idx = (alive.index(selected) + 1) % len(alive) if selected in alive else 0
                        selected = alive[idx]

        if not paused and not extinct:
            for _ in range(ticks_per_frame):
                world.step()
            frame += ticks_per_frame
            if world.counts()[0] == 0:
                extinct = True

        screen.fill((16, 16, 20))
        screen.blit(background, (0, 0))

        # cosmetic day/night cycle (viewer-only; does not affect simulation)
        if show_daynight:
            phase = (math.sin(2 * math.pi * world.t / 2400.0 - math.pi / 2) + 1) / 2  # 0 day .. 1 night
            a = int(phase * 120)
            if a > 4:
                night_overlay.fill((10, 14, 40, a))
                screen.blit(night_overlay, (0, 0))

        if level is not None:
            for relpath, px, py in level.props:
                img = bank.prop(relpath)
                if img is not None:
                    screen.blit(img, img.get_rect(center=((px + 0.5) * TILE, (py + 0.5) * TILE)))

        if show_noise:
            _draw_noise(overlay, world)
            screen.blit(overlay, (0, 0))
            overlay.fill((0, 0, 0, 0))

        for sup in world.supplies.values():
            if not sup.taken:
                img = bank.pickup(sup.kind)
                if img is not None:
                    screen.blit(img, img.get_rect(center=_to_px(sup.x, sup.y)))
                else:
                    pygame.draw.circle(screen, SUPPLY_COLORS[sup.kind], _to_px(sup.x, sup.y), 3)
                    pygame.draw.circle(screen, (235, 235, 235), _to_px(sup.x, sup.y), 3, 1)

        if show_cone and selected is not None and selected.alive:
            _draw_vision_cone(screen, world, selected)

        anim_frame = pygame.time.get_ticks() // 90
        for s in world.survivors:
            if s.alive:
                moving = s.vel != (0.0, 0.0) or s.path is not None
                img = bank.survivor(s.heading, moving, anim_frame, skin=s.skin, cache=rot_cache)
                r = img.get_rect(center=_to_px(s.x, s.y))
                screen.blit(img, r)
                if s.infected:
                    pygame.draw.circle(screen, (90, 230, 90), r.center, int(TILE * 0.7), 1)
        for z in world.zombies:
            if z.alive:
                moving = z.vel != (0.0, 0.0)
                img = bank.zombie(z.id % 3, z.heading, moving, anim_frame, cache=rot_cache)
                screen.blit(img, img.get_rect(center=_to_px(z.x, z.y)))

        if selected is not None and selected.alive:
            px, py = _to_px(selected.x, selected.y)
            pygame.draw.circle(screen, (255, 255, 255), (int(px), int(py)), TILE, 1)

        if extinct:
            box = pygame.Surface((m_px, 90), pygame.SRCALPHA)
            box.fill((10, 10, 14, 215))
            screen.blit(box, (0, m_px // 2 - 45))
            big = pygame.font.SysFont("consolas", 26, bold=True)
            t1 = big.render("ALL SURVIVORS LOST", True, (235, 90, 90))
            t2 = font.render("the outbreak is over  -  press R for a new outbreak, ESC to quit",
                             True, (220, 220, 220))
            screen.blit(t1, t1.get_rect(center=(m_px // 2, m_px // 2 - 14)))
            screen.blit(t2, t2.get_rect(center=(m_px // 2, m_px // 2 + 16)))

        pygame.draw.rect(screen, (24, 24, 30), (0, m_px, m_px, HUD_H))
        surv, inf, zomb = world.counts()
        loot = world.log.count("loot")
        line1 = (f"SURV {surv:>3}   INFECTED {inf:>2}   ZOMBIES {zomb:>3}   "
                 f"LOOT {loot:>3}   tick {world.t:>5}   x{ticks_per_frame if not paused else 0}"
                 + (f"   [{level.name}]" if level is not None else "")
                 + ("   NIGHT" if show_daynight and _is_night(world.t) else ""))
        sel = ""
        if selected is not None:
            sel = (f"[{selected.id}] hp {selected.hp:.0f} hunger {selected.hunger:.0f} "
                   f"thirst {selected.thirst:.0f} fear {selected.fear:.2f} "
                   f"food {selected.inventory['food']} water {selected.inventory['water']} "
                   f"meds {selected.inventory['meds']}"
                   + ("  INFECTED" if selected.infected else ""))
        screen.blit(font.render(line1, True, (225, 225, 225)), (10, m_px + 4))
        screen.blit(font.render(sel, True, (150, 220, 150)), (10, m_px + 24))
        screen.blit(font.render("SPACE pause | 1-4 speed | TAB agent | V cone | N noise | ESC quit",
                                True, (120, 120, 130)), (m_px - 430, m_px + 24))
        pygame.display.flip()

        if screenshot and (frame >= frames or extinct):
            pygame.image.save(screen, screenshot)
            done = True
        clock.tick(60)

    pygame.quit()
    if screenshot:
        print(f"screenshot -> {screenshot}")
