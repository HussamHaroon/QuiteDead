"""D3 — parameter sweep runner (DoD §D3).

  python experiments/sweep.py --param bite_infection_chance --values 0.3,0.5,0.65,0.8 --seeds 5
Produces sweep CSV + a heatmap PNG (metric = survivors alive at end).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from zombie_sim.core.config import Config  # noqa: E402
from zombie_sim.core.levels import load_levels  # noqa: E402
from zombie_sim.core.sim import World  # noqa: E402
from analysis.stats import mean_ci_t  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", required=True, help="Config field to sweep")
    ap.add_argument("--values", required=True, help="comma-separated values")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--level-index", type=int, default=None,
                    help="authored level index (default: procedural map)")
    ap.add_argument("--survivors", type=int, default=6)
    ap.add_argument("--zombies", type=int, default=6)
    ap.add_argument("--ticks", type=int, default=2500)
    ap.add_argument("--out", default="analysis_out/sweep")
    args = ap.parse_args()

    values = [float(v) if "." in v else int(v) for v in args.values.split(",")]
    level = None
    size = 40
    if args.level_index is not None:
        level = load_levels(os.path.join("config", "levels.yaml"))[args.level_index]
        size = level.map.w

    rows = []
    os.makedirs(args.out, exist_ok=True)
    for v in values:
        for seed in range(args.seeds):
            kw = dict(width=size, height=size, n_survivors=args.survivors,
                      n_zombies=args.zombies, n_supply_outdoor=0,
                      supply_per_building=0.0 if level else 30,
                      seed=seed * 31 + 7, **{args.param: v})
            if level is not None:
                kw.update({"n_zombies": level.n_zombies,
                           "n_supply_outdoor": 0, "supply_per_building": 0.0})
            w = World(Config(**kw), level=level)
            while w.counts()[0] > 0 and w.t < args.ticks:
                w.step()
            alive, infected, zombies = w.counts()
            rows.append({"param": v, "seed": seed, "survivors": alive,
                         "quota": w.stockpile, "bites": w.log.count("bite"),
                         "kills": w.log.count("kill"),
                         "ticks": w.t})
            print(f"{args.param}={v} seed={seed}: alive {alive}, quota {w.stockpile}",
                  flush=True)

    csv_path = os.path.join(args.out, f"sweep_{args.param}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["param", "value", "seed", "survivors", "quota", "bites",
                    "kills", "ticks"])
        for r in rows:
            w.writerow([args.param, r["param"], r["seed"], r["survivors"],
                        r["quota"], r["bites"], r["kills"], r["ticks"]])

    # figure: mean +- 95% t-CI of survivors alive and of items deposited
    vals = sorted({r["param"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, key, label, col in ((axes[0], "survivors", f"survivors alive @ end", "#1d7795"),
                                (axes[1], "quota", "items deposited (stockpile)", "#b0483f")):
        ms, hs = [], []
        for v in vals:
            m, h = mean_ci_t([r[key] for r in rows if r["param"] == v])
            ms.append(m); hs.append(h)
        ax.bar([str(v) for v in vals], ms, yerr=hs, capsize=4, color=col)
        ax.set_xlabel(args.param); ax.set_ylabel(label)
    fig.suptitle(f"Sweep of {args.param} ({args.seeds} seeds, mean +- 95% CI)")
    fig.tight_layout()
    png = os.path.join(args.out, f"sweep_{args.param}.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"outputs -> {csv_path} and {png}")


if __name__ == "__main__":
    main()
