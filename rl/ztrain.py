"""Zombie policy training: PPO zombies hunt FSM survivors.

  python rl/ztrain.py --updates 3000 --run-dir runs/zcurriculum1
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo import PPO, gae_per_trajectory
from zenv import ZombieEnv, OBS_DIM, ACT_DIM


def rollout(envs, policy, steps_per_env, gamma, lam):
    if any(e.pending_reset for e in envs):
        for env in envs:
            if env.pending_reset:
                env.reset()

    keys, trajs = [], {}
    rew_all = []
    for i, env in enumerate(envs):
        for aid in env.obs().keys():
            k = (i, aid)
            keys.append(k)
            trajs[k] = {"obs": [], "act": [], "rew": [], "done": [], "val": [],
                        "logp": [], "next_obs": None, "alive": True}

    cur = {i: env.obs() for i, env in enumerate(envs)}
    for _ in range(steps_per_env):
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
                z_dead = aid in info["dead"]
                # only true terminals (zombie death, extinction) end the GAE
                # window — horizon truncations bootstrap from V(s_T)
                t["done"].append(1.0 if (z_dead or (done and not info.get("truncated", False))) else 0.0)
                if z_dead:
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
    ap.add_argument("--survivors", type=int, default=6)
    ap.add_argument("--zombies", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--levels", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "levels.yaml"))
    ap.add_argument("--level-index", type=int, default=1, help="training level (1 = Parking Lot)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--ckpt-every", type=int, default=50)
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    run_dir = args.run_dir or time.strftime("runs/ztrain_%Y%m%d_%H%M%S")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    from zombie_sim.core.levels import load_levels
    level = load_levels(os.path.abspath(args.levels))[args.level_index]

    envs = [ZombieEnv(env_index=i, base_seed=args.seed, n_survivors=args.survivors,
                      n_zombies=args.zombies, horizon=args.horizon, level=level)
            for i in range(args.envs)]
    for env in envs:
        env.reset()

    ppo = PPO(OBS_DIM, ACT_DIM, lr=args.lr, seed=args.seed)
    policy = ppo.net

    csv_path = os.path.join(run_dir, "metrics.csv")
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    wr = csv.writer(csv_f)
    wr.writerow(["update", "agent_steps", "ep_rew", "bites", "kills",
                 "loss", "entropy", "clipfrac", "steps_per_s"])

    print(f"QuiteDead ZOMBIE training | {args.envs} envs x {args.zombies} zombies "
          f"vs {args.survivors} FSM survivors on [{level.name}] -> {run_dir}", flush=True)

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
            bites = np.mean([s["bites"] for s in recent[-k:]])
            kills = np.mean([s["kills"] for s in recent[-k:]])
            # consume — never re-report a stale episode on later updates
            for e in envs:
                e.last_stats = None
        else:
            ep_rew = bites = kills = 0.0

        sps = total_agent_steps / max(time.time() - t0, 1e-9)
        wr.writerow([upd, total_agent_steps, f"{ep_rew:.1f}", f"{bites:.2f}",
                     f"{kills:.2f}", f"{stats['loss']:.4f}", f"{stats['entropy']:.3f}",
                     f"{stats['clipfrac']:.3f}", f"{sps:.0f}"])
        csv_f.flush()
        print(f"upd {upd:>5} | steps {total_agent_steps:>8} | r/step {mean_rew:>7.3f} "
              f"| ep_rew {ep_rew:>7.1f} | bites {bites:.2f} | kills {kills:.2f} "
              f"| ent {stats['entropy']:.3f} | {sps:.0f} a-steps/s",
              flush=True)

        if upd % args.ckpt_every == 0 or upd == args.updates:
            torch.save(policy.state_dict(), os.path.join(run_dir, "policy.pt"))
            torch.save(policy.state_dict(), os.path.join(run_dir, f"ckpt_{upd:05d}.pt"))

    torch.save(policy.state_dict(), os.path.join(run_dir, "policy.pt"))
    csv_f.close()
    print(f"done: {total_agent_steps} agent steps in {time.time() - t0:.0f}s -> {run_dir}",
          flush=True)


if __name__ == "__main__":
    main()
