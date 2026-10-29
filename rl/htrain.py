"""PPO training loop: N parallel envs, one shared survivor policy, checkpoints.

  python rl/train.py --updates 3000                 # full run
  python rl/train.py --updates 15                   # sanity run
Outputs under runs/<name>/: policy.pt (latest), ckpt_XXXX.pt, metrics.csv.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo import PPO, gae_per_trajectory
from henv import HierEnv as SurvivorEnv, OBS_DIM, obs_for
from henv import N_OPT as ACT_DIM


def rollout(envs, policy, steps_per_env, gamma, lam):
    """Collect steps_per_env world-steps from every env; returns flat PPO arrays."""
    keys, trajs = [], {}
    rew_all = []
    # finish episodes that ended during the previous rollout BEFORE building
    # trajectories, so recycled agent ids never mix into live ones
    for env in envs:
        if env.pending_reset:
            env.reset()

    for i, env in enumerate(envs):
        for aid in env.obs().keys():
            k = (i, aid)
            keys.append(k)
            trajs[k] = {"obs": [], "act": [], "rew": [], "done": [], "val": [],
                        "logp": [], "next_obs": None, "alive": True}

    # snapshot current observations
    cur = {i: env.obs() for i, env in enumerate(envs)}
    for _ in range(steps_per_env):
        # batch all alive agent obs
        batch_keys, batch_obs = [], []
        for i, ob in cur.items():
            for aid, o in ob.items():
                k = (i, aid)
                if k in trajs and trajs[k]["alive"]:
                    batch_keys.append(k)
                    batch_obs.append(o)
        if not batch_obs:
            break
        acts, vals, logps = {}, {}, {}
        for k, a, v, lp in zip(batch_keys, *policy.act(np.stack(batch_obs))):
            acts[k], vals[k], logps[k] = int(a), float(v), float(lp)

        next_cur = {}
        for i, env in enumerate(envs):
            if env.pending_reset:
                # episode already ended and its stats are waiting for the
                # controller — freeze until the next rollout boundary
                next_cur[i] = {}
                continue
            acts_i = {aid: acts[(i, aid)] for aid in cur[i]
                      if (i, aid) in acts and trajs[(i, aid)]["alive"]}
            ob2, rew, done, info = env.step(acts_i)
            next_cur[i] = ob2
            for aid in list(cur[i].keys()):
                k = (i, aid)
                if k not in trajs or not trajs[k]["alive"]:
                    continue
                t = trajs[k]
                t["obs"].append(cur[i][aid])
                t["act"].append(acts[k])
                t["val"].append(vals[k])
                t["logp"].append(logps[k])
                t["rew"].append(rew.get(aid, 0.0))
                rew_all.append(rew.get(aid, 0.0))
                agent_dead = aid in info["dead"]
                true_terminal = agent_dead or (done and not info.get("truncated", False))
                t["done"].append(1.0 if true_terminal else 0.0)
                if agent_dead:
                    t["next_obs"] = None
                    t["alive"] = False
                elif aid in ob2:
                    t["next_obs"] = ob2[aid]
                else:
                    t["next_obs"] = None
                    t["alive"] = False
        cur = next_cur

    obs_l, act_l, adv_l, ret_l, logp_l = [], [], [], [], []
    for k in keys:
        t = trajs[k]
        if not t["obs"]:
            continue
        if t["next_obs"] is not None:
            _, boot, _ = policy.act(t["next_obs"][None])
            last_v = float(boot[0])
        else:
            last_v = 0.0
        adv, ret = gae_per_trajectory(t["rew"], t["val"], t["done"], last_v, gamma, lam)
        obs_l += t["obs"]
        act_l += t["act"]
        adv_l += adv.tolist()
        ret_l += ret.tolist()
        logp_l += t["logp"]
    return (np.asarray(obs_l), np.asarray(act_l), np.asarray(adv_l, dtype=np.float32),
            np.asarray(ret_l, dtype=np.float32), np.asarray(logp_l),
            float(np.mean(rew_all)) if rew_all else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--steps-per-env", type=int, default=96)
    ap.add_argument("--survivors", type=int, default=8)
    ap.add_argument("--zombies", type=int, default=2)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=4000,
                    help="safety cap; episodes normally end by quota or extinction")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--levels", default=None,
                    help="YAML with authored levels -> performance-gated curriculum")
    ap.add_argument("--init-from", default=None,
                    help="policy checkpoint to warm-start from")
    ap.add_argument("--eval-every", type=int, default=100,
                    help="run held-out evaluation every N updates (0 = off)")
    ap.add_argument("--eval-eps", type=int, default=5,
                    help="max eval episodes per pass (one per held-out level)")
    ap.add_argument("--mix", action="store_true", help="multi-task: sample every level except the first uniformly, no gating")
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    run_dir = args.run_dir or time.strftime("runs/train_%Y%m%d_%H%M%S")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    envs = [SurvivorEnv(env_index=i, base_seed=args.seed, n_survivors=args.survivors,
                        n_zombies=args.zombies, size=args.size, horizon=args.horizon)
            for i in range(args.envs)]
    levels = None
    cur_level = 0
    recent_surv = deque(maxlen=6)   # sliding window: recent performance, not lifetime
    recent_prog = deque(maxlen=6)   # 1.0 when the episode ENDED with the quota met
    episode_count = 0
    if args.levels:
        from zombie_sim.core.levels import load_levels
        levels = load_levels(args.levels)
        for env in envs:
            env.next_level = levels[1] if args.mix else levels[0]
    for env in envs:
        env.reset()

    ppo = PPO(OBS_DIM, ACT_DIM, lr=args.lr, seed=args.seed)
    if args.init_from:
        ppo.net.load_state_dict(torch.load(args.init_from, map_location="cpu"))
        print(f"warm-started policy from {args.init_from}", flush=True)
    policy = ppo.net

    csv_path = os.path.join(run_dir, "metrics.csv")
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    wr = csv.writer(csv_f)
    wr.writerow(["update", "agent_steps", "level", "r_step", "ep_rew", "surv_rate", "quota",
                 "supplies", "kills", "turns", "loss", "entropy", "clipfrac", "steps_per_s"])

    eval_path = os.path.join(run_dir, "eval.csv")
    eval_f = open(eval_path, "a", newline="", encoding="utf-8")
    eval_wr = csv.writer(eval_f)
    if os.path.getsize(eval_path) == 0:
        eval_wr.writerow(["update", "level", "surv_rate", "quota", "loot",
                          "ticks", "stockpile", "trained"])

    print(f"QuiteDead RL training | {args.envs} envs x {args.survivors} survivors "
          f"| map {args.size}x{args.size} | {args.zombies} zombies | -> {run_dir}",
          flush=True)

    total_agent_steps = 0
    t0 = time.time()
    for upd in range(1, args.updates + 1):
        obs, act, adv, ret, logp, mean_rew = rollout(envs, policy, args.steps_per_env,
                                                     ppo.gamma, ppo.lam)
        total_agent_steps += len(obs)
        stats = ppo.update(obs, act, adv, ret, logp)

        recent = [e.last_stats for e in envs if e.last_stats]
        if recent:
            k = min(len(recent), args.envs)
            ep_rew = np.mean([s["reward"] for s in recent[-k:]])
            surv = np.mean([s["survivors"] / args.survivors for s in recent[-k:]])
            sup = np.mean([s["supplies"] for s in recent[-k:]])
            kil = np.mean([s["kills"] for s in recent[-k:]])
            trn = np.mean([s["turns"] for s in recent[-k:]])
            qta = np.mean([s.get("quota", 0) for s in recent[-k:]])
        else:
            ep_rew = surv = sup = kil = trn = qta = 0.0
        if levels is None:
            # without a curriculum no controller consumes last_stats — clear it
            # so metrics never re-report a stale episode
            for e in envs:
                e.last_stats = None

        sps = total_agent_steps / max(time.time() - t0, 1e-9)

        # ---- curriculum controller ------------------------------------
        if levels is not None:
            for env in envs:
                st = env.last_stats
                if st is None:
                    continue
                env.last_stats = None  # consume
                episode_count += 1
                if args.mix:
                    env.next_level = levels[1 + int(np.random.choice(len(levels) - 1, p=[0.5, 0.35, 0.15] if len(levels) == 4 else None))]
                    continue
                if env.current_level is not None and env.current_level.idx == cur_level:
                    recent_surv.append(st["survivors"] / args.survivors)
                    recent_prog.append(1.0 if st.get("quota", 0) else 0.0)
                # occasional replay of earlier levels so nothing is forgotten
                if episode_count % 7 == 0 and cur_level > 0:
                    env.next_level = levels[int(torch.randint(0, cur_level + 1, (1,)).item())]
            if (not args.mix) and len(recent_prog) >= 4 and cur_level < len(levels) - 1:
                mean_prog = sum(recent_prog) / len(recent_prog)
                mean_surv = sum(recent_surv) / max(1, len(recent_surv))
                # advance on CONSISTENT wins (half the recent episodes end with
                # the quota met) or the authored survival gate — near-misses
                # that die later no longer skip the education
                if mean_prog >= 0.4:  # quota wins only: survival alone let hiders advance
                    cur_level += 1
                    recent_surv.clear()   # keep the deque (a plain-list rebind
                                          # would silently unbound the window)
                    recent_prog.clear()   # stale wins from the old level must not
                                          # chain-advance the next one
                    for env in envs:
                        env.next_level = levels[cur_level]
                    print(f"*** ADVANCING to Level {cur_level + 1}: "
                          f"{levels[cur_level].name} "
                          f"(survival {mean_surv:.2f} >= {levels[cur_level - 1].advance_at:.2f}) ***",
                          flush=True)

        lvl_no = cur_level + 1 if levels else 0
        wr.writerow([upd, total_agent_steps, lvl_no, f"{mean_rew:.3f}", f"{ep_rew:.1f}", f"{surv:.3f}",
                     f"{qta:.2f}", f"{sup:.1f}", f"{kil:.1f}", f"{trn:.1f}", f"{stats['loss']:.4f}",
                     f"{stats['entropy']:.3f}", f"{stats['clipfrac']:.3f}", f"{sps:.0f}"])
        csv_f.flush()
        lvl_tag = f"L{lvl_no} " if levels else ""
        q_tag = " QUOTA" if qta > 0.5 else ""
        print(f"upd {upd:>5} {lvl_tag}| steps {total_agent_steps:>8} | r/step {mean_rew:>7.3f} "
              f"| ep_rew {ep_rew:>7.1f} | surv {surv:.2f} | supplies {sup:>5.1f} "
              f"| kills {kil:>4.1f} | turns {trn:>4.1f}{q_tag} | ent {stats['entropy']:.3f} "
              f"| {sps:.0f} a-steps/s",
              flush=True)

        if upd % args.ckpt_every == 0 or upd == args.updates:
            torch.save(policy.state_dict(), os.path.join(run_dir, "policy.pt"))
            torch.save(policy.state_dict(),
                       os.path.join(run_dir, f"ckpt_{upd:05d}.pt"))

        # ---- evaluation: generalization check ---------------------------
        # run the current policy on every level except the current curriculum
        # level, deterministic, no learning. The `trained` column marks levels
        # the curriculum already passed (still replayed) — those rows are NOT
        # held out. Train score up + eval flat = overfitting; both up = generalizing.
        if args.eval_every and upd % args.eval_every == 0 and levels and len(levels) > 1:
            for lvl in levels[1:][:max(0, args.eval_eps)]:
                eenv = SurvivorEnv(base_seed=777 + lvl.idx, n_survivors=args.survivors,
                                   n_zombies=lvl.n_zombies, size=lvl.map.w,
                                   horizon=4000, level=lvl)
                eenv.reset()
                done = False
                guard = 0
                while not done and guard < 20000:
                    guard += 1
                    w_ = eenv.world
                    acts = {}
                    for s in w_.survivors:
                        if s.alive:
                            a, _, _ = policy.act(obs_for(w_, s)[None], deterministic=True)
                            acts[s.id] = int(a[0])
                    _, _, done, _ = eenv.step(acts)
                if eenv.last_stats:
                    st = eenv.last_stats
                    surv_r = st["survivors"] / args.survivors
                    eval_wr.writerow([upd, lvl.name, f"{surv_r:.3f}", st["quota"],
                                      st["supplies"], st["ticks"], st["stockpile"],
                                      int(lvl.idx <= cur_level)])
                    eval_f.flush()
                    print(f"  EVAL upd {upd} [{lvl.name}]: surv {surv_r:.2f} "
                          f"quota {st['quota']} stock {st['stockpile']}/{lvl.win_stock} "
                          f"loot {st['supplies']}", flush=True)

    torch.save(policy.state_dict(), os.path.join(run_dir, "policy.pt"))
    csv_f.close()
    print(f"done: {total_agent_steps} agent steps in {time.time() - t0:.0f}s -> {run_dir}",
          flush=True)


if __name__ == "__main__":
    main()
