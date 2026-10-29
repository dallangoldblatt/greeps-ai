"""
Approach "paint_comm": greeps that learn to communicate with paint.

    Greep.java   the same network as mlp_ppo (sampled heads, 9207 weights)
    train.py     MAPPO plus communication incentives: influence reward for
                 writers (credited to the tick the drop was spat), positive
                 listening and signalling terms, a cost per drop; annealed
    export.py    embeds a checkpoint in Greep.java

Whether paint carries information is measured by tools/paint_report.py
(score with paint vs with a scrambled paint reading). See README.md.
"""

NAME = "paint_comm"
WEIGHTS_DTYPE = "<f4"


def fixture_weights():
    """For exact-parity tests: a hand-set network that never waits on piles."""
    from policies.mlp_ppo.model import heuristic_params
    return heuristic_params()


def torch_policy(weights, device, generator=None):
    """The same (stochastic) policy for the GPU simulator."""
    from policies.mlp_ppo import torch_policy as mlp_torch_policy
    return mlp_torch_policy(weights, device, generator)
