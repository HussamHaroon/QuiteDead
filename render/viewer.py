"""Pygame observer view: god-mode map, sprites, vision cone, noise overlay, HUD.

Controls: SPACE pause | 1-4 speed | TAB next survivor | V vision cone |
N noise heatmap | T day/night | R restart (after win/loss) | ESC quit.
"""
import math
import os

import pygame

TILE = 16
HUD_H = 76
SPEEDS = {pygame.K_1: 0.5, pygame.K_2: 1, pygame.K_3: 3, pygame.K_4: 8}

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
    from .sound import SoundManager
except ImportError:  # run as a plain script
    import sprites as sprites_mod
    from sprites import load_bank
    from sound import SoundManager


def _is_night(t):
    phase = (math.sin(2 * math.pi * t / 2400.0 - math.pi / 2) + 1) / 2
    return phase > 0.6


def _make_world(cfg, level=None):
    """Fresh world each time: random seed unless the user pinned one with --seed."""
    from zombie_sim.core.sim import World
    import random as _r
    if cfg.seed == 42:  # the default sentinel — nobody explicitly chose it
        cfg = type(cfg)(**{**cfg.to_dict(), "seed": _r.randrange(1, 10**6)})
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


BRAIN_CYCLE = ["utility", "squad", "tactical", "rl_hier", "random"]
BRAIN_LABEL = {"utility": ("SCRIPTED (priority ladder)", (255, 200, 120)),
               "squad": ("SCRIPTED + squad comms", (255, 200, 120)),
               "tactical": ("SCRIPTED tactical", (255, 200, 120)),
               "utility_ai": ("SCRIPTED utility-AI", (255, 200, 120)),
               "rl_hier": ("RL (learned PPO policy)", (120, 235, 160)),
               "random": ("RANDOM baseline", (200, 140, 140))}


def run_viewer(cfg, screenshot=None, frames=240, level=None):
    """Returns "menu" (user asked to go back) or "quit"."""
    result = "quit"
    end_buttons = []   # [(rect, action)] rebuilt every frame while the end banner is up
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
    sfx = SoundManager(enabled=not screenshot)
    background = _prerender_map(world, getattr(bank, "map_tiles", None))
    rot_cache = {}
    selected = next((s for s in world.survivors if s.alive), None)
    ticks_per_frame = 10 if screenshot else 1  # screenshot runs are for verification
    tick_accum = 0.0  # fractional-speed accumulator
    paused = False
    show_cone = True
    show_noise = False
    show_daynight = True
    show_items = True          # item radar: pulsing rings on every untaken pickup
    item_flash = []            # gunshot shockwaves [(x, y, real_ms)]
    scanned_events = 0         # how many log events the flash scanner has seen
    decals = []                # death stains — persist for the whole episode
    dying = []                 # death animations in flight [(x,y,heading,kind,who,ms)]
    muzzle_flashes = []        # [(x, y, heading, ms)] — 3-frame muzzle burst
    bullets = []               # [(x0, y0, x1, y1, ms)] — bullet tracers
    night_overlay = pygame.Surface((m_px, m_px), pygame.SRCALPHA)
    extinct = False
    extinct_at = 0
    won = False
    win_reason = None  # "stock": the camp's supply quota was met

    def _new_outbreak(brain=None):
        nonlocal world, background, selected, extinct, won, win_reason, scanned_events, cfg
        if brain is not None:
            cfg = type(cfg)(**{**cfg.to_dict(), "brain_survivors": brain})
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
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and (extinct or won):
                for rect, act in end_buttons:
                    if rect.collidepoint(ev.pos):
                        if act == "restart":
                            _new_outbreak()
                        elif act == "menu":
                            result = "menu"; done = True
                        else:
                            done = True
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    done = True
                elif ev.key in (pygame.K_m, pygame.K_BACKSPACE):
                    result = "menu"; done = True
                elif ev.key == pygame.K_k:
                    cur = cfg.brain_survivors
                    nxt = BRAIN_CYCLE[(BRAIN_CYCLE.index(cur) + 1) % len(BRAIN_CYCLE)] if cur in BRAIN_CYCLE else BRAIN_CYCLE[0]
                    _new_outbreak(brain=nxt)
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
                elif ev.key == pygame.K_r:
                    _new_outbreak()
                elif ev.key == pygame.K_TAB:
                    alive = [s for s in world.survivors if s.alive]
                    if alive:
                        idx = (alive.index(selected) + 1) % len(alive) if selected in alive else 0
                        selected = alive[idx]

        if not paused and not extinct and not won:
            tick_accum += ticks_per_frame
            whole = int(tick_accum)
            tick_accum -= whole
            for _ in range(whole):
                world.step()
            frame += whole
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
            sfx.play(e["type"])
            if e["type"] == "shot":
                item_flash.append((e["x"], e["y"], pygame.time.get_ticks()))
                # muzzle flash + bullet tracer toward the target
                muzzle_flashes.append((e["x"], e["y"],
                                       e.get("heading", 0.0), pygame.time.get_ticks()))
                if "target" in e:
                    tz = world.agents_by_id.get(e["target"])
                    if tz is not None:
                        bullets.append((e["x"], e["y"], tz.x, tz.y,
                                        pygame.time.get_ticks()))
            elif e["type"] == "death":
                decals.append((e["x"], e["y"], e.get("kind", "?")))
                dying.append((e["x"], e["y"], e.get("heading", 0.0),
                              e.get("kind", "?"), e.get("who", 0),
                              pygame.time.get_ticks()))
        scanned_events = len(world.log.events)
        item_flash[:] = [f for f in item_flash if pygame.time.get_ticks() - f[2] < 450]
        dying[:] = [d for d in dying if pygame.time.get_ticks() - d[5] < 500]
        muzzle_flashes[:] = [f for f in muzzle_flashes
                             if pygame.time.get_ticks() - f[3] < 200]
        bullets[:] = [b for b in bullets if pygame.time.get_ticks() - b[4] < 150]

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

        # muzzle flash: brief 3-frame burst at the shooter's position
        for fx, fy, heading, t0 in muzzle_flashes:
            age = pygame.time.get_ticks() - t0
            fi = min(int(age / 66), 2)  # 3 frames over 200ms
            (fd,) = bank._dir_for(heading) if hasattr(bank, "_dir_for") else ("down",)
            mflash = getattr(bank, "flash", {}).get(fd)
            if mflash:
                img = mflash[fi]
                screen.blit(img, img.get_rect(center=_to_px(fx, fy)))

        # bullet tracers: line from shooter toward the target
        for bx0, by0, bx1, by1, t0 in bullets:
            k = (pygame.time.get_ticks() - t0) / 150.0
            cx = bx0 + (bx1 - bx0) * min(k, 1.0)
            cy = by0 + (by1 - by0) * min(k, 1.0)
            # draw a short bright line segment (tracer)
            pygame.draw.line(screen, (255, 240, 150),
                             _to_px(bx0 + (cx - bx0) * 0.7, by0 + (cy - by0) * 0.7),
                             _to_px(cx, cy), 2)

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
        state_colors = {"LOOTING": (120, 230, 120), "RETURNING": (100, 180, 255),
                        "FLEEING": (255, 160, 60), "SHOOTING": (255, 80, 80),
                        "FIGHTING": (255, 120, 60), "DEPOSITING": (255, 220, 80),
                        "EXPLORING": (180, 180, 180), "EATING": (200, 150, 100),
                        "DRINKING": (100, 200, 255), "HEALING": (255, 150, 150),
                        "SHOVING": (200, 200, 100), "HOLD": (150, 150, 150),
                        "MOVE": (150, 150, 150), "ENGAGE": (255, 100, 100)}
        for s in world.survivors:
            if s.alive:
                moving = s.vel != (0.0, 0.0) or s.path is not None
                img = bank.survivor(s.heading, moving, anim_frame, skin=s.skin,
                                    cache=rot_cache, gun=s.gun)
                if img is not None:  # TLS bank returns None if a sheet is missing
                    r = img.get_rect(center=_to_px(s.x, s.y))
                    screen.blit(img, r)
                    if s.infected:
                        pygame.draw.circle(screen, (90, 230, 90), r.center, int(TILE * 0.7), 1)
                    # state ring: colored circle showing what the brain is doing
                    sc = state_colors.get(getattr(s, "state_label", ""), (120, 120, 120))
                    pygame.draw.circle(screen, sc, (r.centerx, r.bottom + 3), 2)
                    # carried-item pips above the agent
                    carried = []
                    if s.gun: carried.append((80, 230, 255))
                    if s.has_bat: carried.append((205, 160, 90))
                    if s.inventory["food"] > 0: carried.append((232, 146, 58))
                    if s.inventory["water"] > 0: carried.append((86, 156, 232))
                    if s.inventory["meds"] > 0: carried.append((226, 88, 96))
                    for ci, c in enumerate(carried[:4]):
                        pygame.draw.circle(screen, c, (r.centerx - 7 + ci * 5, r.top - 3), 2)
        for z in world.zombies:
            if z.alive:
                moving = z.vel != (0.0, 0.0)
                img = bank.zombie(z.id % 3, z.heading, moving, anim_frame, cache=rot_cache)
                if img is not None:
                    r = img.get_rect(center=_to_px(z.x, z.y))
                    screen.blit(img, r)
                    # zombie state: red dot = chasing, gray = wandering
                    zc = (255, 60, 60) if z.state == "chase" else (120, 120, 120)
                    pygame.draw.circle(screen, zc, (r.centerx, r.bottom + 3), 2)

        if selected is not None and selected.alive:
            px, py = _to_px(selected.x, selected.y)
            pygame.draw.circle(screen, (255, 255, 255), (int(px), int(py)), TILE, 1)

        if extinct or won:
            box = pygame.Surface((m_px, 140), pygame.SRCALPHA)
            box.fill((10, 10, 14, 225))
            screen.blit(box, (0, m_px // 2 - 45))
            big = pygame.font.SysFont("consolas", 26, bold=True)
            if won and win_reason == "stock":
                head = pygame.font.SysFont("consolas", 22, bold=True)
                t1 = head.render("SUPPLIES SECURED — the camp eats another week",
                                 True, (120, 235, 120))
                alive = world.counts()[0]
                t2 = font.render(f"stockpile {world.stockpile}/{level.win_stock}  -  "
                                 f"{alive}/{cfg.n_survivors} alive  -  new outbreak or back to menu",
                                 True, (220, 220, 220))
            else:
                t1 = big.render("ALL SURVIVORS LOST", True, (235, 90, 90))
                t2 = font.render("the outbreak is over",
                                 True, (220, 220, 220))
            screen.blit(t1, t1.get_rect(center=(m_px // 2, m_px // 2 - 14)))
            screen.blit(t2, t2.get_rect(center=(m_px // 2, m_px // 2 + 16)))
            # clickable buttons (keys R / M / ESC do the same)
            end_buttons.clear()
            bf = pygame.font.SysFont("consolas", 16, bold=True)
            labels = [("R  Restart", "restart", (60, 120, 70)), ("M  Menu", "menu", (60, 90, 140)),
                      ("ESC  Quit", "quit", (130, 60, 60))]
            bw, bh, gap = 150, 34, 14
            x0 = m_px // 2 - (3 * bw + 2 * gap) // 2
            mx, my = pygame.mouse.get_pos()
            for i, (lab, act, col) in enumerate(labels):
                r = pygame.Rect(x0 + i * (bw + gap), m_px // 2 + 42, bw, bh)
                hot = r.collidepoint(mx, my)
                pygame.draw.rect(screen, tuple(min(255, c + 40) for c in col) if hot else col, r, border_radius=6)
                pygame.draw.rect(screen, (230, 230, 230), r, 1, border_radius=6)
                tt = bf.render(lab, True, (240, 240, 240))
                screen.blit(tt, tt.get_rect(center=r.center))
                end_buttons.append((r, act))

        surv, inf, zomb = world.counts()
        loot = world.log.count("loot")
        # ---- HUD: clean 2-row panel ----
        hud = pygame.Surface((m_px, HUD_H), pygame.SRCALPHA)
        hud.fill((16, 18, 24, 230))
        pygame.draw.line(hud, (80, 60, 50), (0, 0), (m_px, 0), 2)
        screen.blit(hud, (0, m_px))

        y1 = m_px + 6   # top row: game stats
        y2 = m_px + 30  # bottom row: selected agent

        # row 1: game stats (left) + level name (right)
        font_stats = pygame.font.SysFont("consolas", 14, bold=True)
        stats_left = (f"SURV {surv}  ZOMB {zomb}  INF {inf}  "
                      f"STOCK {world.stockpile}/{win_stock if win_stock is not None else '?'}"
                      f"  [tick {world.t}]")
        screen.blit(font_stats.render(stats_left, True, (200, 220, 200)), (12, y1))

        speed_label = f"{ticks_per_frame}x" + (" PAUSED" if paused else "")
        right_txt = (f"[{level.name}]  {speed_label}"
                     if level is not None else speed_label)
        if show_daynight and _is_night(world.t):
            right_txt += "  NIGHT"
        rt = font_stats.render(right_txt, True, (255, 200, 120))
        screen.blit(rt, (m_px - rt.get_width() - 12, y1))

        # row 2: selected agent info — compact, no overlap
        if selected is not None and selected.alive:
            state = getattr(selected, "state_label", "?")
            gun = f"gun({selected.ammo})" if selected.gun else ("bat" if selected.has_bat else "—")
            hp_col = (120, 230, 120) if selected.hp > 50 else (255, 150, 80) if selected.hp > 25 else (255, 80, 80)
            sel_txt = (f"#{selected.id} {state}  {gun}  "
                       f"HP {selected.hp:.0f}  HUN {selected.hunger:.0f}  "
                       f"THI {selected.thirst:.0f}")
            if selected.infected:
                sel_txt += "  [INFECTED]"
            st = font_stats.render(sel_txt, True, hp_col)
            screen.blit(st, (12, y2))

        blab, bcol = BRAIN_LABEL.get(cfg.brain_survivors, (cfg.brain_survivors, (200, 200, 200)))
        bt = font_stats.render(f"BRAIN: {blab}   [K] switch", True, bcol)
        if not paused:
            screen.blit(bt, (12, y2 + 22))

        # controls hint: only visible when paused (decluttered)
        if paused:
            help_txt = ("SPACE resume | 1=slow 2=norm 3=fast 4=turbo | TAB agent | "
                        "V cone I radar N noise | R restart K brain M menu | ESC quit")
            ht = pygame.font.SysFont("consolas", 12).render(help_txt, True, (150, 150, 160))
            screen.blit(ht, ht.get_rect(center=(m_px // 2, y2 + 30)))
        pygame.display.flip()

        if screenshot and (frame >= frames or extinct or won):
            pygame.image.save(screen, screenshot)
            done = True
        clock.tick(60)

    pygame.quit()
    if screenshot:
        print(f"screenshot -> {screenshot}")
    return result
