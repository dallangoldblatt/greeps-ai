"""
The Greep actor network shared by the PPO trainer, the GPU simulator and the
Java Greep (policies/mlp_ppo/Greep.java implements the same forward pass and
sampling).

    obs (22) -> tanh(64) -> tanh(64) -> 55 logits:
      mode    3   categorical: keep heading, face home, face pile
      turn   37   categorical: turn 10 * (bin - 18) degrees
      move    1   Bernoulli
      memory  8   Bernoulli per bit
      flags   2   Bernoulli per flag
      spit    4   categorical: none, red, orange, purple

Only the actor is exported; the auxiliary head is training-only.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from greeps.torch_sim import MEMORY_BITS, NUM_FLAGS, NUM_MODES, NUM_OBS, NUM_SPITS, NUM_TURNS


HIDDEN = 64
HEADS = [("mode", NUM_MODES), ("turn", NUM_TURNS), ("move", 1),
         ("memory", MEMORY_BITS), ("flags", NUM_FLAGS), ("spit", NUM_SPITS)]
NUM_LOGITS = sum(n for _, n in HEADS)
_SLICES = {}
_start = 0
for _name, _n in HEADS:
    _SLICES[_name] = slice(_start, _start + _n)
    _start += _n

# Raw action layout: mode, turn, move, 8 memory bits, 2 flags, spit.
NUM_RAW = 3 + MEMORY_BITS + NUM_FLAGS + 1
# Training-only predictions of privileged features (indices into
# GreepsSim.privileged(): distance to pile, its direction, distance to ship).
AUX_TARGETS = [3, 4, 5, 2]


class Block(nn.Module):
    """Residual tanh block, h + B tanh(A h); B starts at zero (identity)."""

    def __init__(self, hidden):
        super().__init__()
        self.a = nn.Linear(hidden, hidden)
        self.b = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.b.bias)

    def forward(self, h):
        return h + self.b(torch.tanh(self.a(h)))


class Actor(nn.Module):
    """
    hidden=64, blocks=0 is the network Greep.java runs. Larger actors (wider,
    or with residual blocks) train and play on the GPU simulator only.
    """

    def __init__(self, hidden=HIDDEN, blocks=0):
        super().__init__()
        self.hidden, self.num_blocks = hidden, blocks
        self.l1 = nn.Linear(NUM_OBS, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.blocks = nn.ModuleList(Block(hidden) for _ in range(blocks))
        self.out = nn.Linear(hidden, NUM_LOGITS)
        self.aux = nn.Linear(hidden, len(AUX_TARGETS))

        # Start as an explorer that walks mostly straight, so greeps spread
        # over the map (uniform heads would make them jitter at the ship).
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        with torch.no_grad():
            b = torch.zeros(NUM_LOGITS)
            b[_SLICES["mode"].start] = 3.0                    # keep heading
            b[_SLICES["turn"].start + NUM_TURNS // 2] = 6.0   # no turn ~92%
            b[_SLICES["move"]] = 3.0
            b[_SLICES["spit"].start] = 3.0                    # spit ~13% of ticks
            self.out.bias.copy_(b)

    def forward(self, obs):
        h = torch.tanh(self.l2(torch.tanh(self.l1(obs))))
        for block in self.blocks:
            h = block(h)
        return self.out(h), self.aux(h)

    @property
    def java_compatible(self):
        return self.hidden == HIDDEN and self.num_blocks == 0

    def export_params(self):
        """Flat float32 vector in Greep.java's order: W1, b1, W2, b2, W3, b3."""
        assert self.java_compatible, "Greep.java runs only the 64-64 network"
        parts = [self.l1.weight, self.l1.bias, self.l2.weight, self.l2.bias,
                 self.out.weight, self.out.bias]
        return torch.cat([p.detach().flatten() for p in parts]).cpu().float().numpy()

    def load_params(self, flat):
        flat = torch.tensor(np.asarray(flat), dtype=torch.float32)
        i = 0
        for p in [self.l1.weight, self.l1.bias, self.l2.weight, self.l2.bias,
                  self.out.weight, self.out.bias]:
            n = p.numel()
            p.data.copy_(flat[i:i + n].view_as(p))
            i += n
        assert i == len(flat), f"expected {i} parameters, got {len(flat)}"


def heuristic_params():
    """
    Hand-set weights reproducing the original heuristic Greep: face home when
    carrying, face the pile when on one, turn 70 degrees at water and 180 at
    the world edge, always move, never spit, memory unused. Logit margins of
    at least 30 make sampling effectively deterministic.

    The exact-parity tests use it: its greeps never wait on piles, so Greenfoot
    rarely has to choose between several greeps touching a loader, a choice
    that depends on the JVM's identity hash codes (see test_headless_parity).
    """
    from greeps.torch_sim import OBS_AT_EDGE, OBS_AT_FOOD, OBS_AT_WATER, OBS_CARRYING

    actor = Actor()
    with torch.no_grad():
        for p in actor.parameters():
            p.zero_()
        # Hidden units 0..3 carry s = 2x - 1 (+-1) of four binary sensors.
        for unit, sensor in enumerate([OBS_CARRYING, OBS_AT_FOOD, OBS_AT_WATER, OBS_AT_EDGE]):
            actor.l1.weight[unit, sensor] = 10.0
            actor.l1.bias[unit] = -5.0
            actor.l2.weight[unit, unit] = 10.0
        carry, food, water, edge = 0, 1, 2, 3
        W, b = actor.out.weight, actor.out.bias
        mode, turn = _SLICES["mode"].start, _SLICES["turn"].start
        # mode: keep 0; home +-30 by carrying; pile +30 only on food and empty.
        W[mode + 1, carry] = 30.0
        W[mode + 2, food], W[mode + 2, carry], b[mode + 2] = 30.0, -30.0, -30.0
        # turn: straight (bin 18) unless at water (bin 25) or edge (bin 36).
        b[turn:turn + NUM_TURNS] = -100.0
        W[turn + 18, water], W[turn + 18, edge], b[turn + 18] = -30.0, -30.0, -30.0
        W[turn + 25, water], W[turn + 25, edge], b[turn + 25] = 30.0, -30.0, -30.0
        W[turn + 36, edge], b[turn + 36] = 30.0, 0.0
        b[_SLICES["move"]] = 30.0
        b[_SLICES["memory"]] = -30.0
        b[_SLICES["flags"]] = -30.0
        b[_SLICES["spit"]] = torch.tensor([30.0, -30.0, -30.0, -30.0])
    return actor.export_params()


def actor_shape(state_dict):
    """(hidden, blocks) of a saved actor."""
    blocks = len({k.split(".")[1] for k in state_dict if k.startswith("blocks.")})
    return state_dict["l1.weight"].shape[0], blocks


def load_actor(state_dict):
    """An Actor of the right size for a saved state dict."""
    actor = Actor(*actor_shape(state_dict))
    actor.load_state_dict(state_dict)
    return actor


def grow_actor(state_dict, hidden, blocks):
    """
    A larger Actor that computes exactly what the saved one does: new hidden
    units start with zero outgoing weights and new residual blocks as the
    identity, so a learned behaviour (and paint protocol) carries over and
    the extra capacity is free to learn on top.
    """
    small = load_actor(state_dict)
    big = Actor(hidden, blocks)
    n = small.hidden
    assert hidden >= n and blocks >= small.num_blocks
    with torch.no_grad():
        big.l1.weight[:n] = small.l1.weight
        big.l1.bias[:n] = small.l1.bias
        big.l2.weight[:n, :n] = small.l2.weight
        big.l2.weight[:n, n:] = 0
        big.l2.bias[:n] = small.l2.bias
        # Fresh units in copied blocks keep their random inputs (a) but are
        # silent (b is zero outside the copied part).
        for block, target in zip(small.blocks, big.blocks):
            target.a.weight[:n, :n] = block.a.weight
            target.a.weight[:n, n:] = 0
            target.a.bias[:n] = block.a.bias
            target.b.weight.zero_()
            target.b.weight[:n, :n] = block.b.weight
            target.b.bias.zero_()
            target.b.bias[:n] = block.b.bias
        for head in ("out", "aux"):
            src, dst = getattr(small, head), getattr(big, head)
            dst.weight.zero_()
            dst.weight[:, :n] = src.weight
            dst.bias.copy_(src.bias)
    return big


def num_params(hidden=HIDDEN):
    return (NUM_OBS + 1) * hidden + (hidden + 1) * hidden + (hidden + 1) * NUM_LOGITS


def _heads(logits):
    return {name: logits[..., s] for name, s in _SLICES.items()}


def split_heads(logits):
    """{head name: its logits}, e.g. split_heads(logits)["spit"] is (..., 4)."""
    return _heads(logits)


def sample(logits, generator=None):
    """Sample raw actions (..., NUM_RAW) long. Returns (raw, log_prob)."""
    h = _heads(logits)
    u = torch.rand(*logits.shape[:-1], NUM_RAW, device=logits.device, generator=generator)
    mode = _sample_categorical(h["mode"], u[..., 0])
    turn = _sample_categorical(h["turn"], u[..., 1])
    bern = torch.cat([h["move"], h["memory"], h["flags"]], dim=-1)
    bits = (u[..., 2:2 + bern.shape[-1]] < torch.sigmoid(bern)).long()
    spit = _sample_categorical(h["spit"], u[..., -1])
    raw = torch.cat([mode[..., None], turn[..., None], bits, spit[..., None]], dim=-1)
    return raw, log_prob(logits, raw)


def _sample_categorical(logits, u):
    """Inverse-CDF sampling, as Greep.java does it."""
    n = logits.shape[-1]
    # cumsum as a matmul with an upper-triangular matrix: much faster on the
    # GPU than a scan over many short rows.
    upper = torch.ones(n, n, device=logits.device).triu()
    cdf = torch.softmax(logits, dim=-1) @ upper
    return (cdf < u[..., None]).sum(-1).clamp(max=n - 1)


def log_prob(logits, raw, memory=True):
    """memory=False leaves out the memory and flag bits (trained as state, not actions)."""
    h = _heads(logits)
    lp = torch.log_softmax(h["mode"], -1).gather(-1, raw[..., 0:1]).squeeze(-1)
    lp = lp + torch.log_softmax(h["turn"], -1).gather(-1, raw[..., 1:2]).squeeze(-1)
    if memory:
        bern, bits = torch.cat([h["move"], h["memory"], h["flags"]], dim=-1), raw[..., 2:-1]
    else:
        bern, bits = h["move"], raw[..., 2:3]
    lp = lp - F.binary_cross_entropy_with_logits(bern, bits.float(), reduction="none").sum(-1)
    lp = lp + torch.log_softmax(h["spit"], -1).gather(-1, raw[..., -1:]).squeeze(-1)
    return lp


# The memory and flag bits: their logits, and their place in raw actions and
# in observations (memory then flags, contiguous in all three).
MEMORY_LOGITS = slice(_SLICES["memory"].start, _SLICES["flags"].stop)
MEMORY_RAW = slice(3, 3 + MEMORY_BITS + NUM_FLAGS)


def entropy(logits, memory=True):
    h = _heads(logits)

    def cat_entropy(l):
        logp = torch.log_softmax(l, -1)
        return -(logp.exp() * logp).sum(-1)

    bern = torch.cat([h["move"], h["memory"], h["flags"]], dim=-1) if memory else h["move"]
    p = torch.sigmoid(bern)
    bern_entropy = F.binary_cross_entropy_with_logits(bern, p, reduction="none").sum(-1)
    return cat_entropy(h["mode"]) + cat_entropy(h["turn"]) + bern_entropy + cat_entropy(h["spit"])


def to_sim_actions(raw):
    bits = torch.arange(MEMORY_BITS, device=raw.device)
    memory = (raw[..., 3:3 + MEMORY_BITS] << bits).sum(-1)
    return {
        "mode": raw[..., 0],
        "turn": raw[..., 1],
        "move": raw[..., 2].bool(),
        "memory": memory,
        "flags": raw[..., 3 + MEMORY_BITS:3 + MEMORY_BITS + NUM_FLAGS].bool(),
        "spit": raw[..., -1],
    }


def sim_policy(actor, generator=None):
    """A policy function for greeps.torch_sim.evaluate()."""
    def policy(obs, sim):
        logits, _ = actor(obs)
        raw, _ = sample(logits, generator)
        return to_sim_actions(raw)
    return policy
