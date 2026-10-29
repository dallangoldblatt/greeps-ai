# mlp_ppo: a neural network trained with MAPPO on the GPU

```powershell
python -m tools.use_policy mlp_ppo
python -m policies.mlp_ppo.train --run main --minutes 90 --custom-maps maps/training
python -m policies.mlp_ppo.export --run main
python -m tools.evaluate
```

## Policy (`Greep.java`, `model.py`)

The same 22 legal observations as linear_cma (without the random input),
through `tanh(64) → tanh(64)` to 55 logits that are **sampled**, using
Greenfoot's (seedable) random numbers:

| Head | Distribution |
|---|---|
| mode: keep heading / face home / face pile | categorical (3) |
| turn: 10 × (bin − 18) degrees | categorical (37) |
| move | Bernoulli |
| 8 memory bits, 2 flags | Bernoulli each |
| spit: none / red / orange / purple | categorical (4) |

Sampling gives exploration for free (every action, including paint, has some
probability) and lets the identical greeps behave differently.

## Training (`train.py`): privileged training, legal execution

The deployed actor may only see what a greep legally sees; everything else
used in training is thrown away before export (centralised training,
decentralised execution):

* **Privileged critic.** The value function also sees walking distances and
  directions to the nearest pile and the ship (computed around water), pile
  sizes, team progress and time left. It only provides baselines, so it may
  use anything.
* **Per-greep credit.** Each delivered tomato pays half to the carrier and
  half to the greep that loaded it, so loaders are rewarded for their part.
* **Potential-based shaping** from the walking distances (towards a pile when
  empty, towards the ship when carrying), which provably does not change the
  optimal policy.
* **Auxiliary head.** The actor's hidden layer also predicts the privileged
  distances and directions (a training-only head).
* **Memory and flags are actions**, so training and deployment are identical.

Rollouts run on `greeps/torch_sim.py`: 2048 maps × 20 greeps at once, each
tick replayed as one CUDA graph (~12 ms per tick); PPO with GAE (γ = 0.995,
λ = 0.95). Training maps: Maps 1-2, 512 random maps and optional custom maps
(`--custom-maps`); Map 3 is held out.

## Results

First run (`out/ppo/run1`, 90 minutes before CUDA graphs, ~1000 updates,
no custom maps): **44.9 / 58.6 / 12.1** on Maps 1 / 2 / 3 in the real
scenario. Worse than linear_cma on the scenario maps but better on the
generated validation maps (39.0 per map vs 23.8), thanks to the random
training maps. Scores were still rising when it stopped, and the best
checkpoint was selected on Maps 1-2 only, which is why Map 3 slipped.

## Known gaps / next steps

* Train longer on the generated maps (`--custom-maps maps/training`), now
  that it is ~3x faster. This trainer still always plays Maps 1-2 and
  selects on them, unlike paint_comm (`--real-weight 0 --select-maps
  maps/validation`).
* Nothing yet learns to route around big lakes; the critic
  knows the walking directions but the actor cannot see them. Memory or
  paint would have to carry that, which is what teacher-student distillation
  or trainable memory (paint_comm's `--trainable-memory`) target.
