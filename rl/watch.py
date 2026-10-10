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
from env import SurvivorEnv, OBS_DIM, ACT_DIM, obs_for, apply_action  # noqa: E402

import pygame                                   # noqa: E402
from render.viewer import (_load_sprites, _prerender_map, _rotated, _to_px,
                           _draw_vision_cone, COLORS, TILE, HUD_H, SUPPLY_COLORS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--survivors", type=int, default=8)
    ap.add_argument("--zombies", type=int, default=2)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--det", action="store_true", help="deterministic policy")
    args = ap.parse_args()

    torch.set_num_threads(2)
    policy = ActorCritic(OBS_DIM, ACT_DIM)
    policy.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    policy.eval()

    env = SurvivorEnv(base_seed=args.seed, n_survivors=args.survivors,
                      n_zombies=args.zombies, size=args.size, horizon=10 ** 9)
    env.reset()

    os.environ.setdefault("SDL_VIDEODRIVER", "")
    pygame.init()
    m_px = args.size * TILE
    screen = pygame.display.set_mode((m_px, m_px + HUD_H))
    pygame.display.set_caption(f"QuiteDead — trained policy ({args.ckpt})")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 15)

    skins, zombie_img = _load_sprites()
    background = _prerender_map(env.world)
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
        w.step()
        if w.counts()[0] == 0:
            env.reset()
            background = _prerender_map(env.world)

        screen.fill((16, 16, 20))
        screen.blit(background, (0, 0))
        for sup in w.supplies.values():
            if not sup.taken:
                pygame.draw.circle(screen, SUPPLY_COLORS[sup.kind], _to_px(sup.x, sup.y), 3)
        sel = next((s for s in w.survivors if s.alive), None)
        if sel is not None:
            _draw_vision_cone(screen, w, sel)
        for s in w.survivors:
            if s.alive:
                pos = _rotated(rot_cache, skins[s.skin % len(skins)], s.heading)
                screen.blit(pos, pos.get_rect(center=_to_px(s.x, s.y)))
                if s.infected:
                    pygame.draw.circle(screen, (90, 230, 90),
                                       pos.get_rect(center=_to_px(s.x, s.y)).center, 12, 1)
        for z in w.zombies:
            if z.alive:
                pos = _rotated(rot_cache, zombie_img, z.heading)
                screen.blit(pos, pos.get_rect(center=_to_px(z.x, z.y)))

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
