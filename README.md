# QuiteDead — Zombie Apocalypse Agent Simulator

2D top-down zombie-apocalypse **agent simulation** for the Agent-Based
Intelligence course. Every survivor and zombie is an autonomous agent with
limited senses (vision cones, hearing, memory). Survivors loot supplies and
haul them back to the safe house; zombies hunt, bite, and turn their victims.
The RL question: can learned policies match or beat the hand-written brains?

> Asset credit: zombie/survivor/item/tile art in the local
> `assets/thelazystone_postapoc/` pack is by **TheLazyStone**
> (https://thelazystone.itch.io/post-apocalypse-pixel-art-asset-pack) — used
> under its free non-commercial license (not redistributed by this repo).
> The committed fallback art is Kenney's CC0 "Top-down Shooter" pack.

## Quickstart

```bash
pip install -r requirements.txt          # pygame-ce, PyYAML, numpy, torch

python run_sim.py --view                 # scripted agents, observer window
python run_sim.py --brain random --ticks 2000      # D1 random-brain floor
python run_sim.py --levels config/levels.yaml --level 1 --survivors 6 --view
python run_sim.py --determinism --ticks 1500       # same seed => same world
```

Viewer controls: `SPACE` pause · `1-4` speed · `TAB` next agent · `V` vision
cone · `I` item radar · `N` noise heatmap · `T` day/night · `R` restart ·
`ESC` quit.

## RL training

```bash
cd rl
python train.py  --updates 6000 --levels ../config/levels.yaml   # survivors
python ztrain.py --updates 4000 --levels ../config/levels.yaml   # zombies
python trainview.py --run-dir runs/curriculum21b --levels ../config/levels.yaml \
                    --zomb-ckpt runs/zcurriculum1b/policy.pt      # live demo
```

## Science (course core)

```bash
python run_sim.py --levels config/levels.yaml --level 2 --ticks 6000 --save-events logs/audit.jsonl
python analysis/metrics.py logs/audit.jsonl --out analysis_out/
python experiments/sweep.py --param bite_infection_chance --values 0.3,0.65 --seeds 5
python experiments/compare_brains.py --brains scripted,random,rl:runs/final_surv3/policy.pt --level 1 --seeds 20
python experiments/outbreak.py --level 2 --survivors 20 --seeds 10
```

## Tests (Definition of Done)

```bash
python -m pytest tests/ -q     # 26 checks: determinism, contracts, stats, config
```

## Controls & rules

Survivors win by stockpiling the level's quota at the SAFE HOUSE; they lose
when the whole squad dies. Unarmed survivors can only shove. Loot: food /
water / meds (consumables), bats (quiet melee), pistols + ammo (ranged, LOUD —
gunshots pull hordes). Zombies see a 130° cone, hear the noise field, and
remember where you were — bites infect, and the dead rise.

## Headline results (20 seeds, 6 survivors)
Scripted survivors win 100% of Training Grounds, 60% of Parking Lot and 0% of
Suburb Blocks; random never wins. Flat-action PPO never learned the mission (0%), but the
**hierarchical PPO** policy (`rl_hier`) wins 100% of the two levels it trained on and 75% of an
unseen Parking Lot. Details, CIs and caveats: `REPORT.md`.

## Survivor brains (`brain_survivors`)

| value | what it is | level-1 win rate (10 seeds) |
|---|---|---|
| `utility` | scripted priority ladder (default) | 0.6 |
| `squad` | ladder + **squad communication**: survivors within `comms_radius` (default 12) share supply and zombie memories | **0.7** |
| `tactical` | squad + pair-based building search (partners share a claimed building, squad shares searched-building memory); optional flags via `TAC=gun,hunker,wait,intel` | 0.7 |
| `rl_hier` | **learned**: a PPO network picks one of 6 goals every 6 ticks (hold/scout/loot/return/flee/engage), scripted code executes them. Trained only on Training Grounds + Warm-up Lot | 0.75 (stochastic) / 0.90 (deterministic) on original Parking Lot |
| `utility_ai` | experimental true utility-scoring brain (scored actions, risk-aware looting, commitment/hysteresis) | 0.4 |
| `random` | baseline | - |

`utility_ai` is documented as an honest negative result: scoring-based action selection was
more fragile than the hand-tuned ladder on these maps (see `zombie_sim/core/brains/utility_ai.py`).

## Level ramp and learned-agent status
`config/levels_ramp.yaml` adds a *Warm-up Lot* (3 zombies, quota 8; scripted and squad brains win 20/20) and
*Suburb Blocks (Easy)* (3 zombies, quota 12) between the original levels, for a gentler curriculum:
`python -m rl.train --levels config/levels_ramp.yaml`. `python -m rl.bc` clones the scripted ladder into the
PPO network. Honest status: PPO from scratch, with the reward fix, on the ramp, and from a behaviour-cloned
start all fail to win (logs in `results/ppo_rewardfix/` and `results/ppo_ramp_bc/`); see report Sec. 4.3b.

## Reinforcement learning that works: hierarchical PPO
Flat PPO (one move per tick) never learned the mission. `rl/htrain.py` trains a *hierarchical* policy instead:
the network chooses a goal (`rl/options.py`) and scripted code carries it out. Key ingredients: option layer,
7 extra observation features (carried items, stockpile, at-home, ...; `rl/henv.py`), loot reward 2 and deposit reward 5,
and a bug fix in `rl/env.py` (agents never forgot supplies that were already taken).

    python rl/htrain.py --levels config/levels_warm.yaml --updates 2500 --run-dir rl/runs/my_run   # ~35 min on 2 cores
    python rl/heval.py rl/runs/hier_final/policy.pt 20                                                # 20-seed evaluation
    # watch it in the viewer: set brain_survivors: rl_hier in a YAML and pass --config

## Viewer controls and brain switching
Start with `python run_sim.py` (menu). In **Settings** pick the **Brain** (RL learned / Scripted / Squad / Tactical / Random),
the **Level set** (original or gentle ramp) and the level. The HUD's bottom row always shows which brain is running
(`BRAIN: RL (learned PPO policy)` in green, `SCRIPTED ...` in amber).
In the game: `R` restart, `K` switch brain (cycles scripted -> squad -> tactical -> RL -> random and restarts),
`M` / Backspace back to the menu, `ESC` quit, plus `SPACE` pause, `1-4` speed, `TAB` agent, `V` cone, `I` radar, `N` noise.
When an episode ends, a banner shows **Restart / Menu / Quit** buttons (clickable, or press R / M / ESC).
From the command line: `python run_sim.py --view --survivor-brain rl_hier --levels config/levels.yaml --level 1`.
