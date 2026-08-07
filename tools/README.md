# Diagnostics

Throwaway-looking scripts that earned their place. Each answers one question that
training curves cannot.

| script | question it answers |
|---|---|
| `diagnose.py` | Does a *single* path condition train? Isolates the path pathway from the mixture. |
| `diag_mix.py` | Does the full mixture train, and how does each condition terminate? |
| `diag_terms.py` | **Which reward term dominates, per condition?** This one found the reward-hacking bug: `speed = -0.4967` with `shortcut = +0.0422` — the policy was driving backwards to farm a one-sided bonus. |
| `check_difficulty.py` | Does the reference path actually help? Compares a greedy controller against pure pursuit, with no learning involved. |

`check_difficulty.py` is the one to run before any training on a new environment.
It takes two minutes and answers "is this benchmark measuring the thing I care
about" — which on the first version of this environment it was not:

```
old (scattered circles):  greedy 10.4%   pursuit 45.4%    ratio  4.4x
new (traps + detour):     greedy  0.7%   pursuit 32.7%    ratio 47x
```
