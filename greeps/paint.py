"""
Paint ablations: what a policy plays like without the information in paint.

Wrap a GPU-simulator policy (obs, sim) -> actions:

    paint_scrambled(policy)  each greep reads the paint under a random other
                             greep: same paint statistics, no information about
                             its own surroundings. The clean test.
    paint_colours_shuffled(policy)
                             whether a greep is on paint is kept, but a greep
                             on paint reads the colours under a random other
                             greep that is also on paint: the information in
                             *which* colours are here is gone, the overall
                             colour mix and the presence of paint are kept.
                             (Relabelling colours instead would turn a mostly
                             orange world red, an input the policy never saw.)
    paint_blind(policy)      the paint sense is switched off. Can mislead: a
                             policy that is nearly always on paint never saw
                             "no paint" in training.
    memory_scrambled(policy) not paint: each greep reads the memory and flags
                             of a random other greep, so the score lost is
                             what the greep's own memory is worth.

All are CUDA-graph safe.
"""

import torch

from .torch_sim import MEMORY_BITS, NUM_FLAGS, OBS_MEMORY, PAINT_OBS


_MEMORY_OBS = slice(OBS_MEMORY, OBS_MEMORY + MEMORY_BITS + NUM_FLAGS)   # memory then flags


_CANDIDATES = 8     # random greeps tried when looking for one that is on paint


def shuffle_colours(obs, generator):
    """
    obs where each greep on paint reads the colours of a random other greep
    on paint (the first of a few random candidates that is; its own reading
    if none is). Fixed shapes, so CUDA-graph safe.
    """
    paint = obs[..., PAINT_OBS].reshape(-1, 3)
    on_paint = paint.sum(-1) > 0
    n = paint.shape[0]
    candidates = torch.randint(n, (n, _CANDIDATES), generator=generator, device=obs.device)
    usable = on_paint[candidates]                                    # (n, candidates)
    first = usable.float().argmax(-1)                                # first usable candidate
    donor = candidates.gather(1, first[:, None]).squeeze(1)
    donor = torch.where(usable.any(-1), donor, torch.arange(n, device=obs.device))
    new = torch.where(on_paint[:, None], paint[donor], paint)
    obs = obs.clone()
    obs[..., PAINT_OBS] = new.view(*obs.shape[:-1], 3)
    return obs


def paint_colours_shuffled(policy, seed=54321):
    generator = None

    def shuffled(obs, sim):
        nonlocal generator
        if generator is None:
            generator = torch.Generator(device=obs.device).manual_seed(seed)
        return policy(shuffle_colours(obs, generator), sim)
    return shuffled


def scramble_paint(obs, generator):
    """obs with each greep's paint reading replaced by a random greep's."""
    paint = obs[..., PAINT_OBS].reshape(-1, 3)
    pick = torch.randint(paint.shape[0], (paint.shape[0],), generator=generator,
                         device=obs.device)
    obs = obs.clone()
    obs[..., PAINT_OBS] = paint[pick].view(*obs.shape[:-1], 3)
    return obs


def paint_scrambled(policy, seed=12345):
    generator = None

    def scrambled(obs, sim):
        nonlocal generator
        if generator is None:
            generator = torch.Generator(device=obs.device).manual_seed(seed)
        return policy(scramble_paint(obs, generator), sim)
    return scrambled


def memory_scrambled(policy, seed=999):
    generator = None

    def scrambled(obs, sim):
        nonlocal generator
        if generator is None:
            generator = torch.Generator(device=obs.device).manual_seed(seed)
        memory = obs[..., _MEMORY_OBS].reshape(-1, MEMORY_BITS + NUM_FLAGS)
        pick = torch.randint(memory.shape[0], (memory.shape[0],), generator=generator,
                             device=obs.device)
        obs = obs.clone()
        obs[..., _MEMORY_OBS] = memory[pick].view(*obs.shape[:-1], -1)
        return policy(obs, sim)
    return scrambled


def paint_blind(policy):
    def blind(obs, sim):
        obs = obs.clone()
        obs[..., PAINT_OBS] = 0
        return policy(obs, sim)
    return blind
