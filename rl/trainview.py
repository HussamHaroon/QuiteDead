"""Watch the live training policy play — this IS the training loop, made visible.

Every episode uses the policy checkpoint currently on disk (hot-reloaded each
episode), so as the background trainer improves, this window visibly improves.
Episodes auto-restart; banners show the result of each attempt.

  python rl/trainview.py --run-dir runs/curriculum3 --levels config/levels.yaml
"""
import argparse
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
from render.sprites import load_bank             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="training run dir with policy.pt")
    ap.add_argument("--levels", default=None)
    ap.add_argument("--survivors", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=1000)
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
                    banner = (f"EPISODE {episode}  -  DAY LIMIT, QUOTA MISSED",
                              f"{surv}/{args.survivors} alive  -  stock {w.stockpile}/{w.level.win_stock}"
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
            banner = None

        w = env.world
        screen.fill((16, 16, 20))
        screen.blit(background, (0, 0))
        for sup in w.supplies.values():
            if not sup.taken:
                img = bank.pickup(sup.kind) if bank else None
                if img is not None:
                    screen.blit(img, img.get_rect(center=_to_px(sup.x, sup.y)))
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
        line = (f"LIVE POLICY  EP {episode}  [{lvl_name}]   SURV {surv}/{args.survivors}   "
                f"ZOMB {zomb}   LOOT {w.log.count('loot')}   tick {min(w.t, env.horizon)}/{env.horizon}")
        screen.blit(font.render(line, True, (150, 220, 150)), (10, m_px + 10))
        screen.blit(font.render("the network reloads from the trainer's checkpoint after every episode",
                                True, (120, 120, 130)), (10, m_px + 26))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
