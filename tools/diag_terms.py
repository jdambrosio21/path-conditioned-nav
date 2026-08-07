"""Which reward term dominates, per path condition?"""
import torch, collections
from pcnav.algorithms import Runner
from pcnav.config import EnvConfig, ExperimentConfig, MapConfig, PPOConfig, TrainConfig
from pcnav.planning import PathQuality
from pcnav.utils import seed_everything

seed_everything(0)
cfg = ExperimentConfig(
    env=EnvConfig(num_envs=1024, device="mps", seed=0, maps=MapConfig(num_maps=30)),
    ppo=PPOConfig(rollout_steps=24),
    train=TrainConfig(total_iterations=80, run_dir="runs/diag", run_name="terms",
                      log_interval=40, checkpoint_interval=10**9))
r = Runner(cfg); r.train()

# Roll the trained policy and accumulate reward terms + speed by condition.
acc = collections.defaultdict(lambda: collections.defaultdict(float))
counts = collections.defaultdict(int)
obs = r.env.observe()
for _ in range(300):
    obs = r._attach_dropout_mask(obs)
    a = r.policy.act_deterministic(obs)
    obs, rew, done, info = r.env.step(a)
    q = info["path_quality"]
    for quality in PathQuality:
        m = (q == int(quality))
        if not bool(m.any()): continue
        n = int(m.sum()); counts[quality.name] += n
        for k, v in info["reward_terms"].items():
            acc[quality.name][k] += float(v[m].sum())
        acc[quality.name]["speed"] += float(info["forward_speed"][m].sum())

print(f"\n{'cond':11s}" + "".join(f"{k:>10s}" for k in ["progress","shortcut","clearance","time","goal","collision","speed"]))
for name, d in acc.items():
    n = counts[name]
    print(f"{name:11s}" + "".join(f"{d[k]/n:>10.4f}" for k in ["progress","shortcut","clearance","time","goal","collision","speed"]))
