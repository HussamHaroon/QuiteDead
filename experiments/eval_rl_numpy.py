"""Evaluate a PPO survivor checkpoint WITHOUT PyTorch (uses experiments/np_policy.py).

  python experiments/eval_rl_numpy.py runs/final_surv3/policy.pt <level_idx> <seeds>
Appends per-episode rows to results/rl_episodes.csv. Used for the report when torch was unavailable."""
import sys, os, csv
sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'rl'); sys.path.insert(0, 'experiments'); sys.path.insert(0, 'experiments')
from np_policy import NPActorCritic
from experiments.compare_brains import run_episode
from zombie_sim.core.levels import load_levels
from analysis.stats import wilson_ci, mean_ci_t
ck = sys.argv[1]; lv = int(sys.argv[2]); seeds = int(sys.argv[3])
level = load_levels('config/levels.yaml')[lv]
pol = NPActorCritic(ck, seed=0)
rows=[]
for sd in range(seeds):
    r = run_episode('rl', level, sd, 6, 4000, surv_policy=pol); rows.append(r)
    print(ck, lv, sd, r, flush=True)
q=[r['quota'] for r in rows]; s=[r['survival_rate'] for r in rows]
lo,hi=wilson_ci(sum(q),len(q)); m,h=mean_ci_t(s)
print('SUMMARY', ck, level.name, 'quota', sum(q)/len(q), (round(lo,3),round(hi,3)), 'surv', round(m,3), round(h,3),
      'stock', sum(r['stockpile'] for r in rows)/len(rows))
with open('results/rl_episodes.csv','a',newline='') as f:
    w=csv.writer(f)
    for sd,r in enumerate(rows): w.writerow([ck, level.name, sd, r['survival_rate'], r['quota'], r['stockpile'], r['ticks']])
