"""Watch the live training policy play — this IS the training loop, made visible.

Every episode uses the policy checkpoint currently on disk (hot-reloaded each
episode), so as the background trainer improves, this window visibly improves.
Episodes auto-restart; banners show the result of each attempt.

  python rl/trainview.py --run-dir runs/curriculum3 --levels config/levels.yaml
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import pygame
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo import ActorCritic                      # noqa: E402
from env import SurvivorEnv, OBS_DIM, ACT_DIM, obs_for, apply_action  # noqa: E402
from render.viewer import (_prerender_map, _to_px,               # noqa: E402
                           _draw_vision_cone, TILE, HUD_H, SUPPLY_COLORS)
from render import sprites as sprites_bank        # noqa: E402
from render.sprites import load_bank             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="training run dir with policy.pt")
    ap.add_argument("--levels", default=None)
    ap.add_argument("--survivors", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    torch.set_num_threads(2)
    policy = ActorCritic(OBS_DIM, ACT_DIM)
    ckpt_path = os.path.join(args.run_dir, "policy.pt")
    waited = 0.0
    while not os.path.exists(ckpt_path):
        if waited == 0.0:
            print(f"waiting for the first checkpoint in {args.run_dir} ...", flush=True)
        time.sleep(2.0)
        waited += 2.0
        if waited > 180:
            raise SystemExit(f"no checkpoint appeared at {ckpt_path}")
    policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    policy.eval()
    loaded_at = os.path.getmtime(ckpt_path)

    levels = None
    if args.levels:
        from zombie_sim.core.levels import load_levels
        levels = load_levels(args.levels)

    env = SurvivorEnv(base_seed=args.seed, n_survivors=args.survivors,
                      horizon=args.horizon)
    if levels:
        env.next_level = levels[0]
    env.reset()

    pygame.init()
    m_px = env.world.game_map.w * TILE
    screen = pygame.display.set_mode((m_px, m_px + HUD_H))
    pygame.display.set_caption("QuiteDead — LIVE TRAINING POLICY (hot-reloaded each episode)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 15)
    big = pygame.font.SysFont("consolas", 24, bold=True)

    episode = 1
    ep_kills0 = 0
    decals = []            # death stains — persist for the whole episode
    item_flash = []        # gunshot shockwaves
    scanned_events = 0
    dying = []             # death animations in flight
    first_lvl = env.world.level
    bank = load_bank(first_lvl.theme if first_lvl is not None else None)
    last_theme = first_lvl.theme if first_lvl is not None else None
    background = _prerender_map(env.world, getattr(bank, "map_tiles", None))
    rot_cache = {}
    banner = None          # (text1, text2, color, shown_at)
    BANNER_MS = 2600

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False

        if banner is None:
            w = env.world
            for s in w.survivors:
                if s.alive:
                    a, _, _ = policy.act(obs_for(w, s)[None], deterministic=True)
                    apply_action(w, s, int(a[0]))
            w.step()
            loot_now = w.log.count("loot")

            if w.counts()[0] == 0 or w.t >= env.horizon:
                surv, inf, zomb = w.counts()
                kills = w.log.count("kill") - ep_kills0
                quota = (w.level is not None and w.stockpile >= w.level.win_stock)
                if surv > 0 and quota:
                    banner = (f"EPISODE {episode}  -  QUOTA MET, CAMP SUPPLIED",
                              f"{surv}/{args.survivors} alive  -  stock {w.stockpile}/{w.level.win_stock}"
                              f"  -  reloading policy...",
                              (120, 235, 120))
                elif surv > 0:
                    banner = (f"EPISODE {episode}  -  EPISODE CAP REACHED",
                              f"{surv}/{args.survivors} alive  -  stock {w.stockpile}/{w.level.win_stock if w.level else '?'}"
                              f"  -  reloading policy...",
                              (235, 170, 70))
                else:
                    banner = (f"EPISODE {episode} FAILED  -  entire squad lost",
                              f"loot {loot_now}  -  reloading policy...",
                              (235, 90, 90))
                banner = (banner[0], banner[1], banner[2], pygame.time.get_ticks())
        elif pygame.time.get_ticks() - banner[3] > BANNER_MS:
            # reload the freshest checkpoint and start the next episode
            try:
                if os.path.getmtime(ckpt_path) != loaded_at:
                    policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                    loaded_at = os.path.getmtime(ckpt_path)
            except (FileNotFoundError, RuntimeError):
                pass
            episode += 1
            if levels:
                env.next_level = levels[episode % len(levels)]
            lvl = levels[episode % len(levels)] if levels else None
            env.reset()
            if env.world.game_map.w * TILE != m_px:
                m_px = env.world.game_map.w * TILE
                screen = pygame.display.set_mode((m_px, m_px + HUD_H))
            if lvl is not None and lvl.theme != last_theme:
                bank = load_bank(lvl.theme)
                last_theme = lvl.theme
            background = _prerender_map(env.world, getattr(bank, "map_tiles", None))
            rot_cache.clear()
            ep_kills0 = env.world.log.count("kill")
            decals.clear()
            item_flash.clear()
            dying.clear()
            scanned_events = 0
            banner = None

        w = env.world
        screen.fill((16, 16, 20))
        screen.blit(background, (0, 0))

        # task reminder during the opening of each episode
        if w.t < 200 and w.level is not None:
            task = font.render(
                f"TASK: loot supplies and deposit them at the SAFE HOUSE  -  "
                f"{w.level.win_stock} needed, no time limit, zombies hunt on sight",
                True, (255, 225, 70))
            tbox = pygame.Surface((task.get_width() + 16, 26), pygame.SRCALPHA)
            tbox.fill((10, 10, 14, 180))
            screen.blit(tbox, (m_px // 2 - task.get_width() // 2 - 8, 8))
            screen.blit(task, (m_px // 2 - task.get_width() // 2, 12))

        # persistent history: death stains and picked-clean markers never despawn
        for e in w.log.events[scanned_events:]:
            if e["type"] == "death":
                decals.append((e["x"], e["y"], e.get("kind", "?")))
                dying.append((e["x"], e["y"], e.get("heading", 0.0),
                              e.get("kind", "?"), e.get("who", 0),
                              pygame.time.get_ticks()))
            elif e["type"] == "shot":
                item_flash.append((e["x"], e["y"], pygame.time.get_ticks()))
        scanned_events = len(w.log.events)
        item_flash[:] = [f for f in item_flash if pygame.time.get_ticks() - f[2] < 450]
        dying[:] = [d for d in dying if pygame.time.get_ticks() - d[5] < 500]
        for dx_, dy_, kind in decals:
            col = (120, 22, 22) if kind == "survivor" else (70, 30, 30)
            pygame.draw.circle(screen, col, _to_px(dx_, dy_), 4)
            pygame.draw.circle(screen, (30, 10, 10), _to_px(dx_, dy_), 4, 1)
        for sup in w.supplies.values():
            if sup.taken:
                pygame.draw.circle(screen, (70, 70, 66), _to_px(sup.x, sup.y), 2, 1)

        # death animations: agents collapse visibly instead of popping away
        for dx_, dy_, head, kind, who, t0 in dying:
            k = min(1.0, (pygame.time.get_ticks() - t0) / 480.0)
            (d,) = bank._dir_for(head)
            if kind == "survivor":
                frames = bank.surv_death.get(d)
            else:
                frames = bank.zomb_death.get((sprites_bank.ZOMBIE_KINDS[who % 3], d))
            if frames:
                fi = min(int(k * len(frames)), len(frames) - 1)
                img = frames[fi]
                screen.blit(img, img.get_rect(center=_to_px(dx_, dy_)))

        # gunshot shockwaves
        for fx, fy, t0 in item_flash:
            k = (pygame.time.get_ticks() - t0) / 450.0
            r = int(6 + k * 42)
            a = int(180 * (1.0 - k))
            ring = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(ring, (255, 200, 90, a), (r + 2, r + 2), r, 2)
            screen.blit(ring, (fx * TILE - r - 2, fy * TILE - r - 2))

        # item radar: pulsing rings on every untaken pickup (observer cheat)
        pulse = 5 + int(2.5 * math.sin(pygame.time.get_ticks() * 0.006))
        ring_colors = {"food": (232, 146, 58), "water": (86, 156, 232),
                       "meds": (226, 88, 96), "gun": (80, 230, 255),
                       "ammo": (255, 220, 80)}
        for sup in w.supplies.values():
            if not sup.taken:
                pygame.draw.circle(screen, ring_colors.get(sup.kind, (255, 255, 255)),
                                   _to_px(sup.x, sup.y), pulse, 1)
        for sup in w.supplies.values():
            if not sup.taken:
                img = bank.pickup(sup.kind) if bank else None
                if img is not None:
                    bob = int(1.5 * math.sin(pygame.time.get_ticks() * 0.004 + sup.id))
                    screen.blit(img, img.get_rect(
                        center=(_to_px(sup.x, sup.y)[0], _to_px(sup.x, sup.y)[1] - 2 + bob)))
                else:
                    pygame.draw.circle(screen, SUPPLY_COLORS[sup.kind], _to_px(sup.x, sup.y), 3)

        af = pygame.time.get_ticks() // 90
        for s in w.survivors:
            if s.alive:
                moving = s.vel != (0.0, 0.0) or s.path is not None
                img = bank.survivor(s.heading, moving, af, skin=s.skin, cache=rot_cache)
                r = img.get_rect(center=_to_px(s.x, s.y))
                screen.blit(img, r)
                if s.infected:
                    pygame.draw.circle(screen, (90, 230, 90), r.center, 12, 1)
        for z in w.zombies:
            if z.alive:
                moving = z.vel != (0.0, 0.0)
                img = bank.zombie(z.id % 3, z.heading, moving, af, cache=rot_cache)
                screen.blit(img, img.get_rect(center=_to_px(z.x, z.y)))

        if banner is not None:
            box = pygame.Surface((m_px, 90), pygame.SRCALPHA)
            box.fill((10, 10, 14, 215))
            screen.blit(box, (0, m_px // 2 - 45))
            t1 = big.render(banner[0], True, banner[2])
            t2 = font.render(banner[1], True, (220, 220, 220))
            screen.blit(t1, t1.get_rect(center=(m_px // 2, m_px // 2 - 14)))
            screen.blit(t2, t2.get_rect(center=(m_px // 2, m_px // 2 + 16)))

        pygame.draw.rect(screen, (24, 24, 30), (0, m_px, m_px, HUD_H))
        surv, inf, zomb = w.counts()
        lvl_name = w.level.name if w.level is not None else "procedural"
        watcher = next((s for s in w.survivors if s.alive), None)
        state_txt = ""
        if watcher is not None:
            gun = f"gun+{watcher.ammo}r" if watcher.gun else "unarmed"
            state_txt = (f"[{watcher.id}] {getattr(watcher, 'state_label', '?'):<9} {gun}  "
                         f"hp {watcher.hp:.0f}  food {watcher.inventory['food']} "
                         f"water {watcher.inventory['water']}")
        stock = f"{w.stockpile}/{w.level.win_stock}" if w.level is not None else str(w.stockpile)
        line = (f"LIVE POLICY  EP {episode}  [{lvl_name}]   SURV {surv}/{args.survivors}   "
                f"ZOMB {zomb}   STOCK {stock}   tick {min(w.t, env.horizon)}/{env.horizon}")
        screen.blit(font.render(line, True, (150, 220, 150)), (10, m_px + 10))
        screen.blit(font.render(state_txt, True, (200, 200, 160)), (10, m_px + 26))
        screen.blit(font.render("network hot-reloads every episode",
                                True, (120, 120, 130)),
                    (m_px - 300, m_px + 26))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
