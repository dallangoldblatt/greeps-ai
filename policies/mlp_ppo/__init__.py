"""
Approach "mlp_ppo": a small neural network trained with MAPPO on the GPU.

    Greep.java   22 observations -> tanh(64) -> tanh(64) -> 55 logits, sampled
                 (categorical / Bernoulli heads) with Greenfoot's random numbers
    model.py     the same actor in PyTorch, plus sampling and log-probabilities
    train.py     MAPPO: legal-observation actor, privileged critic, per-greep
                 credit, potential-based shaping, auxiliary head; CUDA graphs
    export.py    embeds a checkpoint in Greep.java

See README.md for results and the design.
"""

NAME = "mlp_ppo"
WEIGHTS_DTYPE = "<f4"   # how Greep.java stores its embedded weights


def fixture_weights():
    """For exact-parity tests: a hand-set network that never waits on piles."""
    from .model import heuristic_params
    return heuristic_params()


def torch_policy(weights, device, generator=None):
    """The same (stochastic) policy for the GPU simulator."""
    from .model import Actor, sim_policy
    actor = Actor().to(device)
    actor.load_params(weights)
    actor.eval()
    return sim_policy(actor, generator)
