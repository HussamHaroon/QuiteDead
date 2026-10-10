"""Watch a trained survivor policy play in the pygame viewer.

  python rl/watch.py --ckpt runs/train_xxx/policy.pt [--size 32] [--det]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo import ActorCritic                     # noqa: E402
from env import (SurvivorEnv, OBS_DIM, ACT_DIM, obs_for, apply_action,
                 auto_consume, update_senses)  # noqa: E402

import pygame                                   # noqa: E402
from render.viewer import (_prerender_map, _to_px,
                           _draw_vision_cone, TILE, HUD_H, SUPPLY_COLORS)
from render.sprites import load_bank            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--survivors", type=int, default=8)
    ap.add_argument("--zombies", type=int, default=2)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--det", action="store_true", help="deterministic policy")
    ap.add_argument("--levels", default=None)
    ap.add_argument("--level", type=int, default=None)
    args = ap.parse_args()

    torch.set_num_threads(2)
    policy = ActorCritic(OBS_DIM, ACT_DIM)
    policy.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    policy.eval()

    level = None
    if args.levels is not None and args.level is not None:
        from zombie_sim.core.levels import load_levels
        level = load_levels(args.levels)[args.level]
    env = SurvivorEnv(base_seed=args.seed, n_survivors=args.survivors,
                      n_zombies=args.zombies, size=args.size, horizon=10 ** 9,
                      level=level)
    env.reset()

    os.environ.setdefault("SDL_VIDEODRIVER", "")
    pygame.init()
    m_px = env.world.game_map.w * TILE   # authored levels resize the world
    screen = pygame.display.set_mode((m_px, m_px + HUD_H))
    pygame.display.set_caption(f"QuiteDead — trained policy ({args.ckpt})")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 15)

    bank = load_bank(level.theme if level is not None else None)
    background = _prerender_map(env.world, getattr(bank, "map_tiles", None))
    rot_cache = {}

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False

        w = env.world
        acts = {}
        for s in w.survivors:
            if s.alive:
                a, _, _ = policy.act(obs_for(w, s)[None], deterministic=args.det)
                acts[s.id] = int(a[0])
        for s in w.survivors:
            if s.alive and s.id in acts:
                apply_action(w, s, acts[s.id])
        auto_consume(w)
        w.step()
        update_senses(w)
        if w.counts()[0] == 0:
            env.reset()
            background = _prerender_map(env.world, getattr(bank, "map_tiles", None))

        screen.fill((16, 16, 20))
        screen.blit(background, (0, 0))
        for sup in w.supplies.values():
            if not sup.taken:
                img = bank.pickup(sup.kind)
                if img is not None:
                    screen.blit(img, img.get_rect(center=_to_px(sup.x, sup.y)))
                else:
                    pygame.draw.circle(screen, SUPPLY_COLORS.get(sup.kind, (255, 255, 255)), _to_px(sup.x, sup.y), 3)
        sel = next((s for s in w.survivors if s.alive), None)
        if sel is not None:
            _draw_vision_cone(screen, w, sel)
        af = pygame.time.get_ticks() // 90
        for s in w.survivors:
            if s.alive:
                moving = s.vel != (0.0, 0.0) or s.path is not None
                img = bank.survivor(s.heading, moving, af, skin=s.skin, cache=rot_cache)
                if img is not None:  # TLS bank returns None if a sheet is missing
                    r = img.get_rect(center=_to_px(s.x, s.y))
                    screen.blit(img, r)
                    if s.infected:
                        pygame.draw.circle(screen, (90, 230, 90), r.center, 12, 1)
        for z in w.zombies:
            if z.alive:
                moving = z.vel != (0.0, 0.0)
                img = bank.zombie(z.id % 3, z.heading, moving, af, cache=rot_cache)
                if img is not None:
                    screen.blit(img, img.get_rect(center=_to_px(z.x, z.y)))

        pygame.draw.rect(screen, (24, 24, 30), (0, m_px, m_px, HUD_H))
        surv, inf, zomb = w.counts()
        loot = w.log.count("loot")
        line = (f"TRAINED POLICY {'(det)' if args.det else '(sampled)'}   "
                f"SURV {surv}   INF {inf}   ZOMB {zomb}   LOOT {loot}   tick {w.t}")
        screen.blit(font.render(line, True, (140, 230, 140)), (10, m_px + 10))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
