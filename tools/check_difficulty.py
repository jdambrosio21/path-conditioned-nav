"""Does the reference path now actually help?

Two scripted controllers, no learning:
  greedy  -- drive at the goal, ignore the path entirely
  pursuit -- follow the reference path's waypoints

On an environment where local greedy navigation suffices, these score the same
and path conditioning has nothing to demonstrate. The redesign is only justified
if greedy now fails where pursuit succeeds.
"""
import sys, torch
from pcnav.config import EnvConfig, MapConfig
from pcnav.envs.torch_env import PathConditionedNavEnv

n_struct = (2, 5) if sys.argv[1] == "traps" else (0, 1)
detour = 1.25 if sys.argv[1] == "traps" else 1.0

for name in ["greedy", "pursuit"]:
    env = PathConditionedNavEnv(EnvConfig(
        num_envs=256, device="cpu", seed=42, fixed_path_quality="OPTIMAL",
        maps=MapConfig(num_maps=10, num_structures=n_struct, min_detour_ratio=detour)))
    o = env.observe(); succ = coll = to = 0
    for _ in range(1500):
        goal_dir = o["obs"][:, 67:69]                 # cos/sin of goal bearing
        wp, valid = o["path"][:, 2, :2], o["path"][:, 2, 2] > 0
        if name == "greedy":
            bx, by = goal_dir[:, 0], goal_dir[:, 1]
        else:
            bx = torch.where(valid, wp[:, 0], goal_dir[:, 0])
            by = torch.where(valid, wp[:, 1], goal_dir[:, 1])
        head = torch.atan2(by, bx)
        a = torch.stack([torch.cos(head).clamp(0.2, 1.0), (head / 1.2).clamp(-1, 1)], 1)
        o, r, d, info = env.step(a)
        succ += int(info["success"].sum()); coll += int(info["collision"].sum()); to += int(info["timeout"].sum())
    n = succ + coll + to
    print(f"  {name:8s} success={succ/max(n,1):5.1%}  collision={coll/max(n,1):5.1%}  timeout={to/max(n,1):5.1%}  (n={n})")
