# linear_cma: a linear policy found with CMA-ES

```powershell
python -m tools.use_policy linear_cma
python -m policies.linear_cma.train --run main --generations 200   # ~40 min, 14 JVMs
python -m policies.linear_cma.export --run main
python -m tools.evaluate
```

## Policy (`Greep.java`)

One forward pass per act round, `out = W · in + b` (19 × 24 weights):

| Inputs (23) | Outputs (19), decoded deterministically |
|---|---|
| carrying, at ship, at pile, at water, at edge | mode: argmax of keep heading / face home / face pile |
| paint under the greep: red, orange, purple | turn: round(z) × 10°, at most ±180° |
| 8 memory bits, 2 flags | move if z > 0 (the only "real" action) |
| heading (cos, sin) | 8 memory bits and 2 flags: set if z > 0 |
| bearing to the ship relative to heading (cos, sin) | spit: argmax of none / red / orange / purple |
| a uniform random number (symmetry breaking) | |

Dropping a tomato at the ship and loading other greeps are reflexes. All 20
greeps run the same weights and the weights are frozen for a whole episode.

## Training (`train.py`)

CMA-ES treats the 456 weights as one point and the game as a black box:
sample ~22 weight vectors from a Gaussian, play each for whole episodes in the
real Java scenario (the same 4 seeds for all, so they are compared fairly),
rank them by score, and move the Gaussian's mean and covariance towards the
winners. No gradients, no per-step rewards: only the team's final score,
which sidesteps the credit-assignment problem of 20 cooperating greeps.

It starts from a hand-written policy (`initialPolicyWeights`, the classic
heuristic: home when carrying, face piles, turn at water and edges) and uses
per-weight search scales (`policySearchScales`): rarely-active inputs can
change a lot, always-on inputs only a little.

## Results

First run: 150 generations, ~30 minutes. Real scenario, 64 seeds:
**59.2 / 63.4 / 22.5** on Maps 1 / 2 / 3 (Map 3 never used for training),
from 19.7 / 13.1 / 0 for the hand-written start; 23.8 per map on the
generated validation maps (see [../README.md](../README.md)). Map 3
plateaued early at ~22: the policy partly memorised Maps 1 and 2.

## What it learned

Ablations (switch one input off, 32 seeds, total over Maps 1-3; trained:
144.1): ignoring paint 144.5, memory 103.9, flags 134.9, the random input
133.9, heading 95.1, bearing to the ship 103.6.

* **Carrying:** face home plus a 20-40° clockwise offset and walk, so carriers
  slide along shorelines instead of hitting them head on.
* **Memory is a short timer, not a map.** Hitting water while carrying sets a
  2-tick countdown (memory 16 → 18 → 1 → 16) during which the home offset
  shrinks. Another short chain runs after a delivery.
* **Empty-handed:** walk straight with small heading-dependent turns (a
  compass bias fitted to Maps 1-2), bounce 80-90° at water and ~170° at the
  edge. On a pile, aim at it but keep walking: no greep waits as a loader, so
  tomatoes only move when two greeps happen to cross on a pile.
* **Paint is unused.** A trail only pays off if one greep lays it and others
  follow it, which no single weight change discovers, and the starting
  policy's wide "don't spit" margin meant spitting was never even tried.

These limits (no waiting loaders, no trails, no way around big lakes) are what
the later approaches try to fix.
