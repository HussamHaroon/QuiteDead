"""Run QuiteDead: headless simulation or pygame observer view.

Examples:
  python run_sim.py                          # 3000-tick headless run
  python run_sim.py --ticks 5000             # longer
  python run_sim.py --determinism            # verify same seed => same run
  python run_sim.py --view                   # interactive window
  python run_sim.py --view --screenshot shot.png --frames 240
"""
import argparse
import time

from zombie_sim.core.config import Config
from zombie_sim.core.sim import World


def main():
    ap = argparse.ArgumentParser(description="QuiteDead zombie apocalypse simulator")
    ap.add_argument("--config", default=None, help="scenario YAML path")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--survivors", type=int, default=None)
    ap.add_argument("--zombies", type=int, default=None)
    ap.add_argument("--view", action="store_true", help="open the pygame observer")
    ap.add_argument("--screenshot", default=None, help="save a frame and exit (with --view)")
    ap.add_argument("--frames", type=int, default=240, help="frames to run before screenshot")
    ap.add_argument("--determinism", action="store_true")
    ap.add_argument("--brain", choices=["scripted", "random"], default="scripted",
                    help="random = uniform-random brains for BOTH species (D1 floor)")
    ap.add_argument("--levels", default=None, help="levels YAML path")
    ap.add_argument("--level", type=int, default=None, help="level index to play")
    ap.add_argument("--save-events", default=None, help="write event log JSONL here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    overrides = {"seed": args.seed, "n_survivors": args.survivors, "n_zombies": args.zombies}
    if args.brain == "random":
        overrides["brain_survivors"] = "random"
        overrides["brain_zombies"] = "random"
    cfg = Config.load(args.config, overrides)

    level = None
    if args.levels is not None and args.level is not None:
        from zombie_sim.core.levels import load_levels
        level = load_levels(args.levels)[args.level]
        overrides = {"width": level.map.w, "height": level.map.h,
                     "n_zombies": level.n_zombies,
                     "n_supply_outdoor": 0, "supply_per_building": 0.0}
        if level.reanimation_enabled is not None:
            overrides["reanimation_enabled"] = level.reanimation_enabled
        cfg = Config(**{**cfg.to_dict(), **overrides})

    if args.view:
        from render.viewer import run_viewer
        run_viewer(cfg, screenshot=args.screenshot, frames=args.frames, level=level)
        return

    if args.determinism:
        # use the same merged cfg + level as the headless path, so the check
        # covers the scenario the user actually asked for (incl. --seed and
        # --survivors when a level is selected)
        a = World(cfg, level=level)
        b = World(cfg, level=level)
        t0 = time.time()
        a.run(args.ticks)
        b.run(args.ticks)
        same = (a.counts() == b.counts()
                and len(a.log.events) == len(b.log.events)
                and [e for e in a.log.events] == [e for e in b.log.events])
        print(f"determinism: {'PASS' if same else 'FAIL'} "
              f"({args.ticks} ticks x2 in {time.time() - t0:.1f}s)")
        return

    world = World(cfg, level=level)
    print(f"QuiteDead | seed {world.seed} | {cfg.width}x{cfg.height} map | "
          f"{cfg.n_survivors} survivors, {cfg.n_zombies} initial zombies")
    t0 = time.time()
    world.run(args.ticks, report_every=0 if args.quiet else 500)
    dt = time.time() - t0

    surv, inf, zomb = world.counts()
    deaths = world.log.tally("death", "cause")
    print("-" * 60)
    print(f"ran {world.t} ticks in {dt:.1f}s ({world.t / max(dt, 1e-9):.0f} ticks/s)")
    print(f"final: survivors {surv} | infected {inf} | zombies {zomb}")
    print(f"loot collected: {world.log.count('loot')} | bites: {world.log.count('bite')} "
          f"| turns: {world.log.count('turn')}")
    print(f"deposits: {world.log.count('deposit')} ({world.stockpile} items stockpiled) "
          f"| shots: {world.log.count('shot')} ({world.log.count('kill')} kills)")
    print(f"deaths by cause: {deaths or 'none'}")
    if args.save_events:
        world.log.save(args.save_events)
        print(f"events -> {args.save_events}")


if __name__ == "__main__":
    main()
