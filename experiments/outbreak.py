"""Epidemic metrics over many seeds on one authored level (scripted brains).

  python experiments/outbreak.py --level 2 --survivors 20 --seeds 10 --ticks 5000 --out results/outbreak

Writes per-seed CSV (attack rate, R_eff, deaths by cause) and, for the median-attack-rate
seed, the event log so analysis/metrics.py can draw its epidemic and survival curves.
"""
import argparse, csv, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zombie_sim.core.config import Config  # noqa: E402
from zombie_sim.core.levels import load_levels  # noqa: E402
from zombie_sim.core.sim import World  # noqa: E402
from analysis.stats import mean_ci_t  # noqa: E402


def one(level, seed, survivors, ticks):
    cfg = Config(**{**Config().to_dict(), "seed": seed * 977 + 13, "n_survivors": survivors,
                    "n_zombies": level.n_zombies, "width": level.map.w, "height": level.map.h,
                    "n_supply_outdoor": 0, "supply_per_building": 0.0})
    w = World(cfg, level=level)
    for _ in range(ticks):
        w.step()
        if w.counts()[0] == 0:
            break
    ev = w.log.events
    bites = [e for e in ev if e["type"] == "bite"]
    turns = sum(1 for e in ev if e["type"] == "turn")
    zombies_ever = level.n_zombies + turns
    causes = Counter(e["cause"] for e in ev if e["type"] == "death" and e["kind"] == "survivor")
    return w, {"seed": seed, "ticks": w.t, "survivors_left": w.counts()[0],
               "infections": len(bites), "turns": turns,
               "attack_rate": len({e["who"] for e in bites}) / survivors,
               "R_eff": len(bites) / zombies_ever, "zombies_final": w.counts()[2],
               **{f"death_{k}": v for k, v in causes.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--survivors", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--ticks", type=int, default=5000)
    ap.add_argument("--out", default="results/outbreak")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    level = load_levels("config/levels.yaml")[a.level]
    rows, worlds = [], {}
    for s in range(a.seeds):
        w, r = one(level, s, a.survivors, a.ticks)
        rows.append(r); worlds[s] = w
        print(r, flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (not k[0].islower() or k.startswith("death"), k))
    with open(os.path.join(a.out, "outbreak_seeds.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys, restval=0)
        wr.writeheader(); wr.writerows(rows)
    med = sorted(rows, key=lambda r: r["attack_rate"])[len(rows) // 2]["seed"]
    worlds[med].log.save(os.path.join(a.out, "median_seed_events.jsonl"))
    ar = mean_ci_t([r["attack_rate"] for r in rows]); re = mean_ci_t([r["R_eff"] for r in rows])
    print(f"median seed {med}; attack rate {ar[0]:.3f}+-{ar[1]:.3f}; R_eff {re[0]:.3f}+-{re[1]:.3f}")


if __name__ == "__main__":
    main()
