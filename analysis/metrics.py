"""D2 — metrics from the event log (DoD §D2).

  python analysis/metrics.py logs/audit.jsonl --out analysis_out/
  python run_sim.py --level 2 --ticks 6000 --save-events logs/audit.jsonl

Produces: epidemic_curve.csv/png, survival_curve.csv/png, summary.json
(attack rate, R0 estimate from the transmission tree, deaths by cause).
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_events(path):
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def initial_counts(events):
    for e in events:
        if e["type"] == "sim_start":
            return e.get("survivors", 0), e.get("zombies", 0)
    return 0, 0


def series(events, n_survivors, n_zombies):
    """Reconstruct per-event-tick S / E(infected) / Z counts from the log."""
    dead = set()
    infected = set()
    zombie_ids = {f"z{i}" for i in range(n_zombies)} if False else set()
    # zombie ids in the log are ints at spawn/turn time
    out = []  # (tick, S, E, Z)
    si, zi = n_survivors, n_zombies
    ei = 0
    ticks = sorted({e["t"] for e in events})
    tick_idx = 0
    for e in events:
        t = e["t"]
        et = e["type"]
        if et == "bite":
            infected.add(e["who"])
        elif et == "turn":
            si -= 0  # survivor already counted dead via death event
            zi += 1
        elif et == "death":
            if e["kind"] == "survivor":
                si -= 1
                if e["who"] in infected:
                    infected.discard(e["who"])
            else:
                zi -= 1
        elif et == "reinforce":
            zi += 1
        # record on each distinct tick boundary we cross
        while tick_idx < len(ticks) and ticks[tick_idx] <= t:
            tt = ticks[tick_idx]
            out.append((tt, si, len(infected), zi))
            tick_idx += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events_jsonl")
    ap.add_argument("--out", default="analysis_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    events = load_events(args.events_jsonl)
    n_survivors, n_zombies = initial_counts(events)

    curve = series(events, n_survivors, n_zombies)
    with open(os.path.join(args.out, "epidemic_curve.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tick", "survivors", "infected", "zombies"])
        w.writerows(curve)

    # survival curve: first-death bookkeeping per survivor id
    death_t = {e["who"]: e["t"] for e in events if e["type"] == "death"
               and e["kind"] == "survivor"}
    last_t = curve[-1][0] if curve else 0
    with open(os.path.join(args.out, "survival_curve.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tick", "alive"])
        alive = n_survivors
        tprev = 0
        for who, td in sorted(death_t.items(), key=lambda kv: kv[1]):
            w.writerow([td, alive])
            alive -= 1
        w.writerow([last_t, alive])

    # attack rate + transmission tree
    bites = [e for e in events if e["type"] == "bite"]
    unique_bitten = {e["who"] for e in bites}
    attack_rate = len(unique_bitten) / n_survivors if n_survivors else 0.0
    per_zombie = Counter(e["zombie"] for e in bites)
    actives = [z for z, n in per_zombie.items() if n > 0]
    r0 = (sum(per_zombie[z] for z in actives) / len(actives)) if actives else 0.0

    deaths_by_cause = Counter(e.get("cause", "?") for e in events
                              if e["type"] == "death")
    summary = {
        "initial_survivors": n_survivors,
        "initial_zombies": n_zombies,
        "unique_bitten": len(unique_bitten),
        "attack_rate": round(attack_rate, 4),
        "R0_bites_per_active_zombie": round(r0, 3),
        "deaths_by_cause": dict(deaths_by_cause),
        "ticks": last_t,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # plots
    if curve:
        t = [r[0] for r in curve]
        plt.figure(figsize=(8, 4.5))
        plt.step([r[0] for r in curve], [r[1] for r in curve], label="Survivors")
        plt.step([r[0] for r in curve], [r[2] for r in curve], label="Infected")
        plt.step([r[0] for r in curve], [r[3] for r in curve], label="Zombies")
        plt.xlabel("tick")
        plt.ylabel("agents")
        plt.title("Epidemic curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "epidemic_curve.png"), dpi=130)
        plt.close()

    if death_t:
        ticks_sorted = sorted(death_t.values())
        ys, alive, prev = [], n_survivors, 0
        for td in ticks_sorted:
            ys.append((prev, td))
            alive -= 1
            prev = td
        plt.figure(figsize=(8, 4.5))
        plt.step([0] + [td for _, td in ys], [n_survivors] + [a for _, a in ys])
        plt.xlabel("tick")
        plt.ylabel("survivors alive")
        plt.title("Survival curve")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "survival_curve.png"), dpi=130)
        plt.close()

    print(json.dumps(summary, indent=2))
    print(f"outputs -> {args.out}/")


if __name__ == "__main__":
    main()
