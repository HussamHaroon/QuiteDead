"""D4 — brain comparison harness (the course's core deliverable).

Same levels, same seeds, different survivor brains -> comparable outcomes.

  python experiments/compare_brains.py --brains scripted,random --level 1 --seeds 10
  python experiments/compare_brains.py --brains scripted,rl:runs/curriculum21b/policy.pt ...
"""
import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zombie_sim.core.config import Config          # noqa: E402
from zombie_sim.core.levels import load_levels     # noqa: E402
from zombie_sim.core.sim import World              # noqa: E402


def run_episode(brain, level, seed, survivors, horizon, surv_policy=None):
    overrides = {"seed": seed * 977 + 13, "n_survivors": survivors,
                 "n_zombies": level.n_zombies if level else 6,
                 "width": level.map.w if level else 32,
                 "height": level.map.h if level else 32,
                 "n_supply_outdoor": 0, "supply_per_building": 0.0}
    if level is not None and level.reanimation_enabled is not None:
        overrides["reanimation_enabled"] = level.reanimation_enabled
    cfg = Config(**{**Config().to_dict(), **overrides})
    w = World(cfg, level=level)

    if brain == "rl":
        import torch
        from env import OBS_DIM, ACT_DIM
        assert surv_policy is not None
    if brain == "random":
        import random as _rnd_mod
    quota_met_tick = None
    for t in range(1, horizon + 1):
        if brain == "random":
            import math as _m
            rnd = _rnd_mod.Random(seed * 100 + t)
            for s in w.survivors:
                if s.alive:
                    ang = rnd.random() * 6.283
                    s.path = None
                    s.vel = (_m.cos(ang) * cfg.survivor_run,
                             _m.sin(ang) * cfg.survivor_run)
        elif brain == "rl":
            import numpy as _np
            from env import obs_for, apply_action  # noqa: E402
            for s in w.survivors:
                if s.alive:
                    a, _, _ = surv_policy.act(obs_for(w, s)[None], deterministic=True)
                    apply_action(w, s, int(a[0]))
        # scripted = default utility brain; World.step already drives it
        w.step()
        if quota_met_tick is None and w.stockpile >= (level.win_stock if level else 8):
            quota_met_tick = t
        if w.counts()[0] == 0:
            break
        if quota_met_tick is not None:
            break  # terminal win: stop at the moment the camp is supplied
    alive, _, _ = w.counts()
    return {
        "survival_rate": alive / cfg.n_survivors,
        "quota": 1 if (level is None or w.stockpile >= (level.win_stock if level else 8)) else 0,
        "quota_tick": quota_met_tick or 0,
        "stockpile": w.stockpile,
        "ticks": w.t,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brains", default="scripted,random",
                    help="comma list: scripted | random | rl:<policy.pt>")
    ap.add_argument("--levels", default="config/levels.yaml")
    ap.add_argument("--level", type=int, default=1, help="level index (1 = Parking Lot)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--survivors", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=4000)
    ap.add_argument("--out", default="analysis_out/compare.csv")
    args = ap.parse_args()

    brain_specs = [b.strip() for b in args.brains.split(",")]
    level = load_levels(args.levels)[args.level]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    all_rows = []
    for spec in brain_specs:
        surv_policy = None
        if spec.startswith("rl:"):
            ckpt = spec.split(":", 1)[1]
            import torch
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl"))
            from ppo import ActorCritic
            from env import OBS_DIM, ACT_DIM
            surv_policy = ActorCritic(OBS_DIM, ACT_DIM)
            surv_policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
            surv_policy.eval()
            brain = "rl"
        else:
            brain = spec
        results = []
        for seed in range(args.seeds):
            r = run_episode(brain, level, seed, args.survivors, args.horizon,
                            surv_policy=surv_policy)
            results.append(r)
            print(f"[{brain}] seed {seed}: surv {r['survival_rate']:.2f} "
                  f"quota {r['quota']} stock {r['stockpile']} ticks {r['ticks']}",
                  flush=True)
        srs = [r["survival_rate"] for r in results]
        qrs = [r["quota"] for r in results]
        ci = 1.96 * statistics.pstdev(srs) / max(1, len(srs)) ** 0.5 if len(srs) > 1 else 0.0
        qci = 1.96 * statistics.pstdev(qrs) / max(1, len(qrs)) ** 0.5 if len(qrs) > 1 else 0.0
        all_rows.append({"brain": spec,
                         "survival_mean": round(statistics.mean(srs), 3),
                         "survival_ci95": round(ci, 3),
                         "quota_rate": round(statistics.mean(qrs), 3),
                         "quota_ci95": round(qci, 3),
                         "episodes": len(results)})

    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["brain", "survival_mean", "survival_ci95",
                                          "quota_rate", "quota_ci95", "episodes"])
        if write_header:
            w.writeheader()
        w.writerows(all_rows)

    print(f"\n=== {args.seeds} seeds/brain on [{level.name}] ===")
    for r in all_rows:
        print(f"  {r['brain']:<28} survival {r['survival_mean']:.2f} ± {r['survival_ci95']:.2f}"
              f"  quota-rate {r['quota_rate']:.2f} ± {r['quota_ci95']:.2f}")
    print(f"table -> {args.out}")


if __name__ == "__main__":
    main()
