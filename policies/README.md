# Approaches

Each folder here is one way of solving Greeps: a policy representation plus a
way of training it. They all play the same game through the same interface,
so they can be swapped (`python -m tools.use_policy <name>`) and compared.

| Approach | Policy | Training | Signal | Simulator |
|---|---|---|---|---|
| [linear_cma](linear_cma/README.md) | linear, 456 weights, deterministic | CMA-ES (black-box evolution) | whole-episode score | Java |
| [mlp_ppo](mlp_ppo/README.md) | MLP 22-64-64-55, 9207 weights, stochastic | MAPPO (policy gradient) | per-step, per-greep rewards; privileged critic | GPU |
| [paint_comm](paint_comm/README.md) | same as mlp_ppo | MAPPO + communication incentives (influence, listening, signalling; annealed) | as mlp_ppo, plus credit for paint that changes others' moves | GPU |

## Communication

How much of each approach's score depends on the information in paint
(`python -m tools.paint_report`): the score with real paint minus the score
when each greep reads a random other greep's paint instead (same amount of
paint, no information). GPU simulator, the 315 validation maps, 8 paired
episodes each, mean per map.

*Gain from colour* instead gives each greep on paint the colours under a
random other greep on paint (whether there is paint, and the colour mix, are
kept). *I(colour; carrying)*: how much a drop's colour tells about whether
its writer carries a tomato.

| Approach | With paint | Scrambled | Gain from paint | Gain from colour | I(colour; carrying) |
|---|---|---|---|---|---|
| linear_cma | 23.8 | 22.2 | +1.6 ± 0.1 (spits rarely) | 0.0 | 0.00 bits |
| mlp_ppo | 39.0 | 34.1 | +4.8 ± 0.1 | +0.2 | 0.03 bits |
| paint_comm main/best.pt | 47.1 | 29.6 | +17.5 ± 0.2 | +12.3 | 0.08 bits |
| paint_comm main/best_comm.pt | 45.8 | 26.1 | +19.7 ± 0.2 | +16.4 | 0.08 bits |
| paint_comm colour_stage2/best.pt (embedded) | 47.9 | 23.0 | **+24.9 ± 0.3** | **+25.1** | 0.20 bits |
| paint_comm memory_st ([trained/](../trained/README.md)) | **50.7** | 30.2 | +20.6 ± 0.3 | +16.7 | 0.21 bits |

None of these trained on the validation maps. The colour
curriculum (stage 1 on communication maps, stage 2 on all training maps)
gave a two-colour-class protocol: carriers write orange, searchers red or
purple; see [paint_comm](paint_comm/README.md#colour-curriculum-stage-1--stage-2).

## Comparison

The scenario's Maps 1-3 in the real (Java) scenario, 64 seeds, mean score
per map, and the 315 validation maps (GPU simulator, 8 episodes each).

| Map | linear_cma | mlp_ppo | paint_comm colour_stage2 (embedded) | paint_comm memory_st ([trained/](../trained/README.md)) |
|---|---|---|---|---|
| 1 | 58.7 | 44.6 | 32.6 | 31.3 |
| 2 | 63.2 | 58.0 | 27.2 | 31.8 |
| 3 | 22.6 | 11.2 | 15.5 | 5.7 |
| **Maps 1-3** | **144.5** | 113.8 | 75.3 | 68.9 |
| **Validation, per map** | 23.8 | 39.0 | 47.9 | **50.7** |

linear_cma and mlp_ppo trained on Maps 1-2; paint_comm never saw Maps 1-3.
The two rows rank the approaches in opposite order, which is one of the
open problems in the [main README](../README.md#open-problems-help-welcome).

No approach has learned to route around large lakes: carriers head
straight for the ship and get stuck behind water. linear_cma is better
wherever "walk straight, bounce, head home" works.

## The policy interface

Greenfoot compiles exactly one `src/Greep.java`, and the competition receives
only that file. So each approach keeps its own `Greep.java`, and
`tools/use_policy.py` copies the chosen one into `src/`.

**Java (`policies/<name>/Greep.java`)**, a `Creature` subclass that also
provides, for the training tools (`headless/HeadlessGreeps.java`):

```java
public static final String POLICY_NAME;       // the folder name
public static final int NUM_PARAMS;           // size of the weight vector
public static double[] defaultPolicyWeights();// the embedded trained weights
public static double[] initialPolicyWeights();// a starting point for black-box search
public static double[] policySearchScales();  // per-weight search scale
public static void setPolicyWeights(double[]);// swap weights between episodes
// plus a "BEGIN/END GENERATED WEIGHTS" block holding EMBEDDED_WEIGHTS,
// and support for -Dgreeps.policy=<file> / policy.weights overrides
```

**Python (`policies/<name>/__init__.py`)**:

```python
NAME = "<name>"
def torch_policy(weights, device, generator=None):
    """The same policy for the GPU simulator: (obs, sim) -> actions."""
def fixture_weights():
    """A deterministic policy whose greeps never wait on piles (exact-parity tests)."""
```

`torch_policy` lets `tests/test_torch_sim.py` check that the Java and GPU
versions of an approach play the same game, and lets any approach be
evaluated or trained on the GPU simulator. `greeps.policies.embed_weights`
writes trained weights into a `Greep.java`.

## Adding an approach

1. Copy the closest folder to `policies/<new_name>/` and set `POLICY_NAME`
   in its `Greep.java` (observation code and the shared conventions can stay).
2. Change the policy: its representation in `Greep.java`, and the same thing
   as `torch_policy` in `__init__.py`.
3. Write `train.py` (and `export.py` calling `greeps.policies.embed_weights`).
   `greeps.torch_sim.GreepsSim` gives legal observations (`observe()`),
   privileged training-only features (`privileged()`), per-greep rewards and
   CUDA-graph-friendly stepping; `greeps.java_sim.HeadlessPool` scores weight
   vectors in the real scenario.
4. `python -m tools.use_policy <new_name>`, then run the tests and
   `python -m tools.evaluate` and
   `python -m tools.paint_report --maps maps/validation --episodes 8`, and
   add a column above.

Ideas that would each make a natural new folder: teacher-student
distillation from a full-observation teacher, relaxed (differentiable) paint
channels, or counterfactual credit for each paint drop. (Trainable memory, a
recurrent policy whose state is the 10 bits, is now an option of paint_comm:
`--trainable-memory`.)
