"""
Approach "linear_cma": a linear policy trained with CMA-ES.

    Greep.java   23 inputs (+ bias) -> 19 outputs, one forward pass per tick,
                 decoded deterministically (argmax / thresholds); a random
                 input gives the identical greeps a way to break symmetry
    train.py     CMA-ES over the 456 weights, scoring whole episodes in the
                 real Java scenario (black-box; no gradients)
    export.py    embeds the result in Greep.java

See README.md for results and what the trained policy does.
"""

import numpy as np
import torch


NAME = "linear_cma"
WEIGHTS_DTYPE = "<f8"   # how Greep.java stores its embedded weights
NUM_INPUTS = 23     # Greep.java's 22 observations + a uniform random input
NUM_OUTPUTS = 19
NUM_PARAMS = NUM_OUTPUTS * (NUM_INPUTS + 1)

# Output rows (Greep.java OUT_*).
_MODE, _TURN, _MOVE, _MEMORY, _FLAGS, _SPIT = 0, 3, 4, 5, 13, 15
# Inputs (Greep.java IN_*).
_CARRYING, _AT_FOOD, _AT_WATER, _AT_EDGE = 0, 2, 3, 4


def _set(w, output, column, value):
    w[output * (NUM_INPUTS + 1) + column] = value


def initial_weights():
    """Greep.initialPolicyWeights(): the hand-written CMA-ES starting point."""
    w = np.zeros(NUM_PARAMS)
    bias = NUM_INPUTS
    _set(w, _MODE, bias, 1.0)
    _set(w, _MODE + 1, _CARRYING, 3.0)
    _set(w, _MODE + 2, _AT_FOOD, 2.0)
    _set(w, _MODE + 2, _CARRYING, -3.0)
    _set(w, _TURN, _AT_WATER, 7.0)
    _set(w, _TURN, _AT_EDGE, 18.0)
    _set(w, _MOVE, bias, 2.0)
    for bit in range(8):
        _set(w, _MEMORY + bit, bias, -1.0)
    _set(w, _FLAGS, bias, -1.0)
    _set(w, _FLAGS + 1, bias, -1.0)
    _set(w, _SPIT, bias, 3.0)
    return w


def fixture_weights():
    """For exact-parity tests: the hand-written policy never waits on piles."""
    return initial_weights()


def torch_policy(weights, device, generator=None):
    """
    The same policy for the GPU simulator (greeps/torch_sim.py), decoded
    exactly as Greep.act() does. The random input is drawn from the
    simulator's generator (`generator` is unused: the policy is deterministic).
    """
    w = torch.tensor(np.asarray(weights), dtype=torch.float64,
                     device=device).view(NUM_OUTPUTS, NUM_INPUTS + 1)

    def policy(obs, sim):
        rnd = sim._randint(1000, obs.shape[:2]).double() / 1000.0
        x = torch.cat([obs.double(), rnd[..., None]], dim=-1)
        out = x @ w[:, :NUM_INPUTS].T + w[:, NUM_INPUTS]
        steps = out[..., _TURN].clamp(-18, 18)
        turn = torch.floor(steps + 0.5).long() + 18     # Math.round, as a bin
        bits = torch.arange(8, device=device)
        memory = ((out[..., _MEMORY:_MEMORY + 8] > 0).long() << bits).sum(-1)
        return {
            # argmax returns the first maximum, like Greep.argmax's strict >.
            "mode": out[..., _MODE:_MODE + 3].argmax(-1),
            "turn": turn,
            "move": out[..., _MOVE] > 0,
            "memory": memory,
            "flags": out[..., _FLAGS:_FLAGS + 2] > 0,
            "spit": out[..., _SPIT:_SPIT + 4].argmax(-1),
        }

    return policy
