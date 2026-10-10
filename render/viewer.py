"""Pygame observer view: god-mode map, sprites, vision cone, noise overlay, HUD.

Controls: SPACE pause | 1-4 speed | TAB next survivor | V vision cone |
N noise heatmap | T day/night | R restart (after win/loss) | ESC quit.
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
SUPPLY_COLORS = {
    "food": (232, 146, 58), "water": (86, 156, 232), "meds": (226, 88, 96),
    "gun": (210, 210, 225), "ammo": (225, 95, 75), "bat": (205, 160, 90),
}

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", "sprites", "top-down-shooter", "PNG")

try:
    from . import sprites as sprites_mod
    from .sprites import load_bank
except ImportError:  # run as a plain script
    import sprites as sprites_mod
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
        # pygame.draw ignores the color's alpha component on surfaces without
        # per-pixel alpha (the display surface) — render the translucent cone
        # into a tight SRCALPHA patch and blit that instead
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bx, by = int(min(xs)) - 2, int(min(ys)) - 2
        bw, bh = int(max(xs)) - bx + 4, int(max(ys)) - by + 4
        if bw > 0 and bh > 0:
            patch = pygame.Surface((bw, bh), pygame.SRCALPHA)
            shifted = [(x - bx, y - by) for x, y in pts]
            draw.polygon(patch, (255, 230, 120, 40), shifted)
            draw.polygon(patch, (255, 230, 120, 90), shifted, 1)
            surface.blit(patch, (bx, by))


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
    font = pygame.font.SysFont("consolas", 14)
    small_font = pygame.font.SysFont("consolas", 11, bold=True)

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
    show_items = True          # item radar: pulsing rings on every untaken pickup
    item_flash = []            # gunshot shockwaves [(x, y, real_ms)]
    scanned_events = 0         # how many log events the flash scanner has seen
    decals = []                # death stains — persist for the whole episode
    dying = []                 # death animations in flight [(x,y,heading,kind,who,ms)]
    night_overlay = pygame.Surface((m_px, m_px), pygame.SRCALPHA)
    extinct = False
    extinct_at = 0
    won = False
    win_reason = None  # "stock": the camp's supply quota was met

    def _new_outbreak():
        nonlocal world, background, selected, extinct, won, win_reason, scanned_events
        world = _make_world(cfg, level=level)
        background = _prerender_map(world, getattr(bank, "map_tiles", None))
        selected = next((s for s in world.survivors if s.alive), None)
        extinct = False
        won = False
        win_reason = None
        decals.clear()
        item_flash.clear()
        dying.clear()
        scanned_events = 0

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
                elif ev.key == pygame.K_i:
                    show_items = not show_items
                elif ev.key == pygame.K_r and (extinct or won):
                    _new_outbreak()
                elif ev.key == pygame.K_TAB:
                    alive = [s for s in world.survivors if s.alive]
                    if alive:
                        idx = (alive.index(selected) + 1) % len(alive) if selected in alive else 0
                        selected = alive[idx]

        if not paused and not extinct and not won:
            for _ in range(ticks_per_frame):
                world.step()
            frame += ticks_per_frame
            win_stock = level.win_stock if level is not None else None
            if world.counts()[0] == 0:
                extinct = True
                extinct_at = pygame.time.get_ticks()
            elif win_stock and world.stockpile >= win_stock:
                won = True
                win_reason = "stock"
                extinct_at = pygame.time.get_ticks()

        # gunshot shockwaves: every shot is loud — make the noise mechanic visible
        # + persistent history: death stains + death animations (never despawn)
        for e in world.log.events[scanned_events:]:
            if e["type"] == "shot":
                item_flash.append((e["x"], e["y"], pygame.time.get_ticks()))
            elif e["type"] == "death":
                decals.append((e["x"], e["y"], e.get("kind", "?")))
                dying.append((e["x"], e["y"], e.get("heading", 0.0),
                              e.get("kind", "?"), e.get("who", 0),
                              pygame.time.get_ticks()))
        scanned_events = len(world.log.events)
        item_flash[:] = [f for f in item_flash if pygame.time.get_ticks() - f[2] < 450]
        dying[:] = [d for d in dying if pygame.time.get_ticks() - d[5] < 500]

        screen.fill((16, 16, 20))
        screen.blit(background, (0, 0))

        # task banner during the opening of the outbreak
        if world.t < 200:
            target = level.win_stock if level is not None else None
            task = font.render(
                (f"TASK: loot supplies and deposit them at the SAFE HOUSE  -  "
                 f"{target} needed  -  zombies hunt on sight")
                if target is not None else
                "SANDBOX: procedural outbreak - watch the agents",
                True, (255, 225, 70))
            tbox = pygame.Surface((task.get_width() + 16, 26), pygame.SRCALPHA)
            tbox.fill((10, 10, 14, 180))
            screen.blit(tbox, (m_px // 2 - task.get_width() // 2 - 8, 8))
            screen.blit(task, (m_px // 2 - task.get_width() // 2, 12))

        # cosmetic day/night cycle (viewer-only; does not affect simulation)
        if show_daynight:
            phase = (math.sin(2 * math.pi * world.t / 2400.0 - math.pi / 2) + 1) / 2  # 0 day .. 1 night
            a = int(phase * 120)
            if a > 4:
                night_overlay.fill((10, 14, 40, a))
                screen.blit(night_overlay, (0, 0))

        # safe house: translucent gold tint over its tiles + micro-label above
        if world.safehouse is not None:
            sx, sy, sw, sh = world.safehouse
            rect = pygame.Rect(sx * TILE, sy * TILE, sw * TILE, sh * TILE)
            gold = pygame.Surface(rect.size, pygame.SRCALPHA)
            gold.fill((255, 205, 40, 55))
            screen.blit(gold, rect.topleft)
            pygame.draw.rect(screen, (255, 210, 70), rect, 1)
            label = small_font.render("SAFE HOUSE", True, (255, 225, 70))
            screen.blit(label, label.get_rect(
                centerx=rect.centerx,
                bottom=max(label.get_height() + 2, rect.top - 2)))

        # expanding shockwave ring on every gunshot (the noise the zombies hear)
        for fx, fy, t0 in item_flash:
            k = (pygame.time.get_ticks() - t0) / 450.0
            r = int(6 + k * 42)
            a = int(180 * (1.0 - k))
            ring = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(ring, (255, 200, 90, a), (r + 2, r + 2), r, 2)
            screen.blit(ring, (fx * TILE - r - 2, fy * TILE - r - 2))

        if level is not None:
            for relpath, px, py in level.props:
                img = bank.prop(relpath)
                if img is not None:
                    screen.blit(img, img.get_rect(center=((px + 0.5) * TILE, (py + 0.5) * TILE)))

        # persistent history: death stains + picked-clean loot markers
        for dx_, dy_, kind in decals:
            col = (120, 22, 22) if kind == "survivor" else (70, 30, 30)
            pygame.draw.circle(screen, col, _to_px(dx_, dy_), 4)
            pygame.draw.circle(screen, (30, 10, 10), _to_px(dx_, dy_), 4, 1)
        for sup in world.supplies.values():
            if sup.taken:
                pygame.draw.circle(screen, (70, 70, 66), _to_px(sup.x, sup.y), 2, 1)

        # death animations: agents collapse visibly instead of popping away
        for dx_, dy_, head, kind, who, t0 in dying:
            k = min(1.0, (pygame.time.get_ticks() - t0) / 480.0)
            (dd,) = bank._dir_for(head)
            if kind == "survivor":
                dframes = bank.surv_death.get(dd)
            else:
                dframes = bank.zomb_death.get((sprites_mod.ZOMBIE_KINDS[who % 3], dd))
            if dframes:
                fi = min(int(k * len(dframes)), len(dframes) - 1)
                img = dframes[fi]
                screen.blit(img, img.get_rect(center=_to_px(dx_, dy_)))

        if show_noise:
            _draw_noise(overlay, world)
            screen.blit(overlay, (0, 0))
            overlay.fill((0, 0, 0, 0))

        # item radar: pulsing color-coded rings over every untaken pickup
        # (observer omniscience — the AGENTS still only know what they've seen)
        if show_items:
            pulse = 5 + int(2.5 * math.sin(pygame.time.get_ticks() * 0.006))
            ring_colors = {"food": (232, 146, 58), "water": (86, 156, 232),
                           "meds": (226, 88, 96), "gun": (80, 230, 255),
                           "ammo": (255, 220, 80), "bat": (205, 160, 90)}
            for sup in world.supplies.values():
                if not sup.taken:
                    c = ring_colors.get(sup.kind, (255, 255, 255))
                    pygame.draw.circle(screen, c, _to_px(sup.x, sup.y), pulse, 1)

        for sup in world.supplies.values():
            if not sup.taken:
                img = bank.pickup(sup.kind)
                if img is not None:
                    bob = int(1.5 * math.sin(pygame.time.get_ticks() * 0.004 + sup.id))
                    screen.blit(img, img.get_rect(
                        center=(_to_px(sup.x, sup.y)[0], _to_px(sup.x, sup.y)[1] - 2 + bob)))
                else:
                    pygame.draw.circle(screen, SUPPLY_COLORS.get(sup.kind, (255, 255, 255)), _to_px(sup.x, sup.y), 3)
                    pygame.draw.circle(screen, (235, 235, 235), _to_px(sup.x, sup.y), 3, 1)

        if show_cone and selected is not None and selected.alive:
            _draw_vision_cone(screen, world, selected)

        anim_frame = pygame.time.get_ticks() // 90
        for s in world.survivors:
            if s.alive:
                moving = s.vel != (0.0, 0.0) or s.path is not None
                img = bank.survivor(s.heading, moving, anim_frame, skin=s.skin, cache=rot_cache)
                if img is not None:  # TLS bank returns None if a sheet is missing
                    r = img.get_rect(center=_to_px(s.x, s.y))
                    screen.blit(img, r)
                    if s.infected:
                        pygame.draw.circle(screen, (90, 230, 90), r.center, int(TILE * 0.7), 1)
        for z in world.zombies:
            if z.alive:
                moving = z.vel != (0.0, 0.0)
                img = bank.zombie(z.id % 3, z.heading, moving, anim_frame, cache=rot_cache)
                if img is not None:
                    screen.blit(img, img.get_rect(center=_to_px(z.x, z.y)))

        if selected is not None and selected.alive:
            px, py = _to_px(selected.x, selected.y)
            pygame.draw.circle(screen, (255, 255, 255), (int(px), int(py)), TILE, 1)

        if extinct or won:
            box = pygame.Surface((m_px, 90), pygame.SRCALPHA)
            box.fill((10, 10, 14, 215))
            screen.blit(box, (0, m_px // 2 - 45))
            big = pygame.font.SysFont("consolas", 26, bold=True)
            if won and win_reason == "stock":
                head = pygame.font.SysFont("consolas", 22, bold=True)
                t1 = head.render("SUPPLIES SECURED — the camp eats another week",
                                 True, (120, 235, 120))
                alive = world.counts()[0]
                t2 = font.render(f"stockpile {world.stockpile}/{level.win_stock}  -  "
                                 f"{alive}/{cfg.n_survivors} alive  -  R = new outbreak, ESC = quit",
                                 True, (220, 220, 220))
            else:
                t1 = big.render("ALL SURVIVORS LOST", True, (235, 90, 90))
                t2 = font.render("the outbreak is over  -  press R for a new outbreak, ESC to quit",
                                 True, (220, 220, 220))
            screen.blit(t1, t1.get_rect(center=(m_px // 2, m_px // 2 - 14)))
            screen.blit(t2, t2.get_rect(center=(m_px // 2, m_px // 2 + 16)))

        pygame.draw.rect(screen, (24, 24, 30), (0, m_px, m_px, HUD_H))
        surv, inf, zomb = world.counts()
        loot = world.log.count("loot")
        win_stock = level.win_stock if level is not None else None
        line1 = (f"SURV {surv} INFECTED {inf} ZOMBIES {zomb} LOOT {loot} "
                 f"STOCK {world.stockpile}/{win_stock if win_stock is not None else '?'} "
                 f"tick {world.t} x{ticks_per_frame if not paused else 0}"
                 + (f" [{level.name}]" if level is not None else "")
                 + (" NIGHT" if show_daynight and _is_night(world.t) else ""))
        sel = ""
        if selected is not None:
            state = getattr(selected, "state_label", "?")
            gun = f"gun+{selected.ammo}r" if selected.gun else "unarmed"
            sel = (f"[{selected.id}] {state:<10} {gun} hp {selected.hp:.0f} "
                   f"hunger {selected.hunger:.0f} thirst {selected.thirst:.0f} "
                   f"fear {selected.fear:.2f} "
                   f"food {selected.inventory['food']} water {selected.inventory['water']} "
                   f"meds {selected.inventory['meds']}"
                   + ("  INFECTED" if selected.infected else ""))
        screen.blit(font.render(line1, True, (225, 225, 225)), (10, m_px + 4))
        hint = font.render("SPACE pause | 1-4 speed | TAB agent | V cone | I items | N noise | R restart | ESC quit",
                           True, (120, 120, 130))
        screen.blit(hint, hint.get_rect(topright=(m_px - 8, m_px + 24)))
        sel_surf = font.render(sel, True, (150, 220, 150))
        avail = m_px - 8 - hint.get_width() - 10 - 10  # stop before the hint
        if avail > 0:
            screen.blit(sel_surf, (10, m_px + 24),
                        area=pygame.Rect(0, 0, min(sel_surf.get_width(), avail), sel_surf.get_height()))
        else:
            screen.blit(sel_surf, (10, m_px + 24))
        pygame.display.flip()

        if screenshot and (frame >= frames or extinct or won):
            pygame.image.save(screen, screenshot)
            done = True
        clock.tick(60)

    pygame.quit()
    if screenshot:
        print(f"screenshot -> {screenshot}")
