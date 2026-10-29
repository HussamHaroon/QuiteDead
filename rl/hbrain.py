"""Run the trained hierarchical RL policy inside the normal simulator / viewer.

  brain_survivors: rl_hier   (policy path from env var QD_POLICY, default rl/runs/hier_final/policy.pt)
Every K=6 ticks the network picks an option per survivor; scripted code executes it each tick.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.dirname(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

_state = {"policy": None}


def _policy():
    if _state["policy"] is None:
        import torch
        from henv import OBS_DIM
        from options import N_OPT
        from ppo import ActorCritic
        path = os.environ.get("QD_POLICY", os.path.join(_HERE, "runs", "hier_final", "policy.pt"))
        net = ActorCritic(OBS_DIM, N_OPT)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        _state["policy"] = net
    return _state["policy"]


def think(world, s):
    from env import update_senses, auto_consume, apply_action
    from henv import obs_for, K
    from options import run_option
    if getattr(world, "_hb_tick", -1) != world.t:       # once per world tick
        world._hb_tick = world.t
        update_senses(world)
        auto_consume(world)
        if world.t % K == 0 or not hasattr(world, "_hb_opts"):
            ob = [(x.id, obs_for(world, x)) for x in world.survivors if x.alive]
            if ob:
                import numpy as np
                a, _, _ = _policy().act(np.stack([o for _, o in ob]), deterministic=True)
                world._hb_opts = {i: int(v) for (i, _), v in zip(ob, a)}
            else:
                world._hb_opts = {}
    run_option(world, s, world._hb_opts.get(s.id, 0), apply_action)
