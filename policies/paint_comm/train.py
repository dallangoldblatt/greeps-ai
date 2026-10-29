"""
Train greeps to communicate with paint: MAPPO (as in policies/mlp_ppo) plus
incentives that break the chicken-and-egg problem of emergent communication
(nobody reads paint because it means nothing; nobody writes meaning because
nobody reads it):

  * influence reward (writers): whenever a greep stands on paint spat by
    another greep, compare its movement decision (mode, turn, move) with what
    it would do with a scrambled paint reading (same statistics, no
    information; greeps.paint). The KL divergence is credited to the drop's
    author, retroactively at the tick the drop was spat if that is inside the
    current rollout, so the spitting action itself is reinforced
    (cf. Jaques et al. 2019, social influence);
  * positive listening (readers): a loss term that rewards readers whose
    movement depends on the paint under them (cf. Eccles et al. 2019);
  * positive signalling (writers): a loss term that rewards spit choices
    (none / red / orange / purple) that depend on the writer's situation,
    i.e. carry information (mutual information, batch estimate);
  * a small cost per drop, so paint stays sparse enough to mean something.

The first three fade out linearly over --anneal-updates, so in the end only
paint use that helps the team's score survives. Games also start longer than
the real 120 s (--game-seconds, default 360) and shrink to it over training:
on mazes and puzzle maps short games give too few successes, and too few
greeps reading each drop, to learn from. The actor never sees the clock, so
what it learns transfers to the real game length. Progress is measured with the
scrambled-paint ablation: "info gain" = score with paint - score with
scrambled paint. Checkpoints are chosen on --select-maps (default:
the training maps); --eval-maps are only reported, never trained or selected on.

--trainable-memory makes the 8 memory bits and 2 flags a recurrent state
trained by gradient instead of sampled actions trained by reward alone:

  * each tick the network outputs a probability s per bit; the stored value
    is h * b + (1 - h) * s, with b the sampled bit (as Greep.java samples it)
    and the hardness h rising from --memory-hardness-start to 1 over
    --memory-soft-updates (soft memory is easier to learn; hard bits are what
    the real game allows);
  * gradients pass through the stored bits as if they were s
    (straight-through), and PPO replays the rollout in chunks of --bptt ticks,
    so a later decision's loss teaches what to write earlier;
  * an optional warm-up (--memory-aux-coef, fading over --memory-aux-updates)
    asks the stored bits to predict privileged facts (pile and ship distance
    and direction) through a training-only linear read-out.

The memory bits leave the PPO ratio and entropy (they are state, no longer
actions). Evaluation, checkpoints and Greep.java are unchanged: they sample
hard bits exactly as before, so the 10-bit limit holds where it counts.

    python -m policies.paint_comm.train --run main --minutes 90 \
        --custom-maps maps/training --select-maps maps/validation
    python -m tools.paint_report --checkpoint out/paint/main/best_comm.pt --maps maps/validation
    python -m policies.paint_comm.export --run main
"""

import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from greeps.paint import paint_colours_shuffled, paint_scrambled, scramble_paint, shuffle_colours
from greeps.paths import OUT_DIR as REPO_OUT_DIR
from greeps.torch_sim import (
    MEMORY_BITS, NUM_FLAGS, NUM_OBS, NUM_PRIVILEGED, OBS_MEMORY,
    GreepsSim, MapBank, capture_graph, evaluate,
)
from policies.mlp_ppo import model as mp
from policies.mlp_ppo.train import Critic, build_bank


RUNS_DIR = REPO_OUT_DIR / "paint"
MEMORY_SIZE = MEMORY_BITS + NUM_FLAGS
MEMORY_OBS = slice(OBS_MEMORY, OBS_MEMORY + MEMORY_SIZE)    # memory then flags


def write_memory(logits, bits, hardness):
    """
    The memory value stored after a tick: h * b + (1 - h) * s, with gradients
    as if it were s (straight-through). bits are the sampled bits b.
    """
    s = torch.sigmoid(logits[..., mp.MEMORY_LOGITS])
    return s + (hardness * (bits - s)).detach()


def behaviour_kl(logits_p, logits_q):
    """
    KL(p || q) of the movement heads (mode, turn, move): how much a different
    paint reading changes where a greep goes. Memory, flags and spitting are
    left out, since changing them is cheap and says little.
    """
    hp, hq = mp.split_heads(logits_p), mp.split_heads(logits_q)
    kl = 0.0
    for head in ("mode", "turn"):
        lp, lq = F.log_softmax(hp[head], -1), F.log_softmax(hq[head], -1)
        kl = kl + (lp.exp() * (lp - lq)).sum(-1)
    zp, zq = hp["move"].squeeze(-1), hq["move"].squeeze(-1)
    p = torch.sigmoid(zp)
    kl = kl + p * (F.logsigmoid(zp) - F.logsigmoid(zq)) \
        + (1 - p) * (F.logsigmoid(-zp) - F.logsigmoid(-zq))
    return kl


def spit_information(logits):
    """
    Batch estimate of I(spit choice; situation): H(average spit distribution)
    minus the average entropy of the individual distributions.
    """
    p = torch.softmax(mp.split_heads(logits)["spit"], -1)
    mean = p.mean(0)
    marginal = -(mean * torch.log(mean + 1e-8)).sum()
    conditional = -(p * torch.log(p + 1e-8)).sum(-1).mean()
    return marginal - conditional


def colour_information(logits):
    """
    For the colour chosen when spitting (red / orange / purple): the batch
    estimate of I(colour; situation), and the entropy of the average colour
    distribution (how evenly the three colours are used overall).
    """
    q = torch.softmax(mp.split_heads(logits)["spit"][..., 1:], -1)
    mean = q.mean(0)
    marginal = -(mean * torch.log(mean + 1e-8)).sum()
    conditional = -(q * torch.log(q + 1e-8)).sum(-1).mean()
    return marginal - conditional, marginal


def evaluate_comm(actor, bank, device, episodes, seed):
    """
    Mean score per map with paint, with scrambled paint (no information) and
    with colours shuffled (no colour information), on paired seeds.
    """
    actor.eval()
    results = []
    for wrapper in (None, paint_scrambled, paint_colours_shuffled):
        gen = torch.Generator(device=device).manual_seed(seed)
        policy = mp.sim_policy(actor, gen)
        if wrapper is not None:
            policy = wrapper(policy, seed=seed + 7)
        scores = evaluate(policy, bank, list(range(len(bank))), episodes, device, seed=seed)
        results.append(scores.mean(axis=1))
    actor.train()
    return results


def train(args):
    run_dir = RUNS_DIR / args.run
    latest, best_file, best_comm_file = (run_dir / "latest.pt", run_dir / "best.pt",
                                         run_dir / "best_comm.pt")
    log_file = run_dir / "log.csv"
    if latest.exists() and not args.resume:
        raise SystemExit(f"{run_dir} already has a run; use --resume or another --run name")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    bank, real, custom, map_weights = build_bank(args, device)
    # Checkpoints are chosen on training maps; --eval-maps are only reported,
    # so they stay a held-out measure.
    if args.select_maps:      # e.g. only the hand-made originals, not every variant
        select_bank = MapBank.from_files(args.select_maps, device)
    else:
        select_bank = custom if custom else real
    report_bank = MapBank.from_files(args.eval_maps, device) if args.eval_maps else None
    sim = GreepsSim(bank, args.envs, device, seed=args.seed,
                    loader_share=args.loader_share, shaping=args.shaping,
                    gamma=args.gamma, map_weights=map_weights, team_share=args.team_share)
    sim.reset(start_ticks=torch.randint(1, 1100, (args.envs,), device=device))

    # The actor's size: --hidden/--blocks, or that of the run being resumed.
    # An --init-from actor that is smaller is grown to that size, behaving
    # exactly as before (mp.grow_actor).
    state = None
    if args.resume:
        state = torch.load(latest, map_location=device)
        actor = mp.load_actor(state["actor"])
    elif args.init_from:
        state = torch.load(args.init_from, map_location=device)
        actor = mp.grow_actor(state["actor"], args.hidden, args.blocks)
    else:
        actor = mp.Actor(args.hidden, args.blocks)
    actor, critic = actor.to(device), Critic().to(device)
    print(f"Actor: hidden {actor.hidden}, {actor.num_blocks} residual blocks, "
          f"{sum(p.numel() for p in actor.parameters()):,} parameters")
    # Training-only read-out of the stored memory (--memory-aux-coef warm-up).
    memory_aux = nn.Linear(MEMORY_SIZE, len(mp.AUX_TARGETS)).to(device)
    params = list(actor.parameters()) + list(critic.parameters())
    if args.trainable_memory:
        params += list(memory_aux.parameters())
        assert args.steps % args.bptt == 0, "--steps must be a multiple of --bptt"
    optimizer = torch.optim.Adam(params, lr=args.lr, eps=1e-5)
    # Adaptive coefficient of the colour-entropy floor (--colour-entropy-target).
    log_alpha = torch.tensor(np.log(0.01), device=device, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args.colour_entropy_lr)
    update, best, best_gain = 0, float("-inf"), float("-inf")
    if args.resume:
        if "memory_aux" in state:
            memory_aux.load_state_dict(state["memory_aux"])
        if "log_alpha" in state:
            with torch.no_grad():
                log_alpha.copy_(state["log_alpha"])
        critic.load_state_dict(state["critic"])
        optimizer.load_state_dict(state["optimizer"])
        update, best, best_gain = state["update"], state["best"], state["best_gain"]
        print(f"Resumed at update {update}")
    elif args.init_from:
        if "critic" in state:
            critic.load_state_dict(state["critic"])
        print(f"Actor initialised from {args.init_from}")

    gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    scramble_gen = torch.Generator(device=device).manual_seed(args.seed + 2)
    T, E, N = args.steps, args.envs, sim.N
    buf_obs = torch.zeros(T, E, N, NUM_OBS, device=device)
    buf_priv = torch.zeros(T, E, N, NUM_PRIVILEGED, device=device)
    buf_raw = torch.zeros(T, E, N, mp.NUM_RAW, dtype=torch.uint8, device=device)
    buf_logp = torch.zeros(T, E, N, device=device)
    buf_value = torch.zeros(T, E, N, device=device)
    buf_reward = torch.zeros(T, E, N, device=device)
    buf_bonus = torch.zeros(T, E, N, device=device)      # influence credit
    buf_drops = torch.zeros(T, E, N, device=device)
    buf_done = torch.zeros(T, E, device=device)
    buf_alive = torch.zeros(T, E, N, dtype=torch.bool, device=device)
    buf_on = torch.zeros(T, E, N, dtype=torch.bool, device=device)   # on another's paint

    new_log = not (args.resume and log_file.exists())
    log = open(log_file, "w" if new_log else "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["update", "samples", "train_episodes", "train_score", "entropy",
                         "value_loss", "drops_per_100", "on_others_paint", "influence",
                         "listen", "signal_mi", "incentive_scale", "train_maps_score",
                         "train_maps_scrambled", "train_maps_info_gain", "held_out_score",
                         "held_out_scrambled", "held_out_info_gain",
                         "train_maps_colour_gain", "held_out_colour_gain",
                         "memory_hardness", "memory_aux_loss", "memory_certainty",
                         "colour_entropy", "colour_alpha", "seconds"])

    obs, priv, alive = sim.observe(), sim.privileged(), sim.alive()
    # --trainable-memory: the stored (possibly soft) memory, which replaces
    # the simulator's integer memory in the observations.
    memory = obs[..., MEMORY_OBS].clone()
    hardness = torch.ones((), device=device)
    memory_certainty = torch.zeros(2, device=device)   # sum of |2s - 1|, count
    contacts = sim.paint_contacts()            # drops under each greep, matching obs
    rollout_start = torch.zeros((), dtype=torch.long, device=device)
    env_index = torch.arange(E, device=device)[:, None]
    finished = torch.zeros(2, device=device)   # score sum, episodes
    paint_stats = torch.zeros(4, device=device)  # greep ticks, drops, reader ticks, influence
    # The counterfactual reading that influence and listening are measured
    # against: scrambled paint (rewards any information in paint), or, with
    # --colour-incentives, the same paint with its colours relabelled
    # (rewards only the information in *which* colour).
    counterfactual = shuffle_colours if args.colour_incentives else scramble_paint

    def tick():
        logits, _ = actor(obs)
        raw, logp = mp.sample(logits, gen)
        if args.trainable_memory:
            logp = mp.log_prob(logits, raw, memory=False)
            if args.memory_noise > 0:
                # Exploration and robustness: flip some stored bits. The
                # flipped bits go into the buffer, so the replay matches.
                flip = torch.rand(raw[..., mp.MEMORY_RAW].shape, device=device,
                                  generator=gen) < args.memory_noise
                raw[..., mp.MEMORY_RAW] ^= flip.long()
        value = critic(obs, priv)

        # Influence of other greeps' paint on this decision.
        others = contacts & (sim.paint_author[None, None, :] != sim.greep_index[None, :, None])
        on = others.any(-1) & alive
        logits_scrambled, _ = actor(counterfactual(obs, scramble_gen))
        # Capped: otherwise greeps learn to spam paint that jolts readers.
        influence = behaviour_kl(logits, logits_scrambled).clamp(max=args.influence_clip) * on
        # Split each reader's influence over the drops it stands on, then
        # credit each drop's author: at the tick the drop was spat if that
        # is in this rollout (buf_bonus), otherwise now (returned).
        share = others.float() / others.sum(-1, keepdim=True).clamp(min=1)
        credit = (share * influence[..., None]).sum(1)                   # (E, N*K)
        when = sim.paint_step - rollout_start
        inside = (when >= 0) & (when < T)
        author = sim.paint_author[None, :].expand(E, -1)
        flat = (when.clamp(0, T - 1) * E + env_index) * N + author
        buf_bonus.view(-1).scatter_add_(0, flat.flatten(), (credit * inside).flatten())
        late = torch.zeros(E, N, device=device).scatter_add_(1, author, credit * ~inside)

        spits_before = sim.spits.clone()
        reward, done, scores = sim.step(mp.to_sim_actions(raw))
        drops = (sim.spits - spits_before).float()
        d = done.float()
        finished.add_(torch.stack([(scores * d).sum(), d.sum()]))
        paint_stats.add_(torch.stack([alive.sum().float(), drops.sum(),
                                      on.sum().float(), influence.sum()]))
        sim.reset(done)
        if args.trainable_memory:
            # As the simulator does with its integer memory: written while
            # alive, cleared when the game restarts.
            stored = write_memory(logits, raw[..., mp.MEMORY_RAW].float(), hardness)
            stored = torch.where(alive[..., None], stored, memory)
            memory.copy_(stored * (1 - d)[:, None, None])
            s = torch.sigmoid(logits[..., mp.MEMORY_LOGITS])
            memory_certainty.add_(torch.stack([((2 * s - 1).abs().mean(-1) * alive).sum(),
                                               alive.sum().float()]))
        obs.copy_(sim.observe())
        if args.trainable_memory:
            obs[..., MEMORY_OBS] = memory
        priv.copy_(sim.privileged())
        alive.copy_(sim.alive())
        contacts.copy_(sim.paint_contacts())
        return raw, logp, value, reward, d, drops, on, late

    with torch.no_grad():
        if args.eager:
            run = tick
        else:
            graph, outputs = capture_graph(tick)

            def run():
                graph.replay()
                return outputs

    def ppo_losses(logits, aux, obs_in, priv_in, raw, old_logp, a, ret, on, w, scale, memory):
        """
        The training loss for a batch of decisions (rows), weighted by w
        (0 for greeps not yet released). memory=False: the memory bits are
        not actions (--trainable-memory).
        """
        count = w.sum().clamp(min=1)
        ratio = (mp.log_prob(logits, raw, memory=memory) - old_logp).exp()
        pg_loss = -(torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a) * w).sum() / count
        ent = (mp.entropy(logits, memory=memory) * w).sum() / count
        v_loss = 0.5 * ((critic(obs_in.detach(), priv_in) - ret).pow(2) * w).sum() / count
        aux_loss = ((aux - priv_in[:, mp.AUX_TARGETS]).pow(2).mean(-1) * w).sum() / count

        # Positive listening: movement should depend on others' paint.
        on = on * w
        logits_scrambled, _ = actor(counterfactual(obs_in, gen))
        listen_kl = behaviour_kl(logits, logits_scrambled).clamp(max=args.listen_clip)
        listen = (listen_kl * on).sum() / on.sum().clamp(min=1)
        # Positive signalling: spit choices (or, with --colour-incentives,
        # colour choices) should carry information; plus a bonus for using
        # all three colours.
        released = logits[w > 0]
        if args.colour_incentives:
            signal, colour_spread = colour_information(released)
        else:
            signal, colour_spread = spit_information(released), torch.zeros((), device=device)
        # Entropy of the colour chosen when spitting, weighted by how likely
        # spitting is (kept above --colour-entropy-target by an adaptive
        # coefficient, so the greeps keep trying other colours).
        spit = torch.softmax(mp.split_heads(logits)["spit"], -1)
        colour = spit[..., 1:] / spit[..., 1:].sum(-1, keepdim=True).clamp(min=1e-8)
        spit_weight = (1 - spit[..., 0]).detach() * w
        colour_entropy = ((-(colour * torch.log(colour + 1e-8)).sum(-1) * spit_weight).sum()
                          / spit_weight.sum().clamp(min=1e-8))

        loss = (pg_loss - args.ent_coef * ent + args.vf_coef * v_loss
                + args.aux_coef * aux_loss
                - scale * args.listen_coef * listen
                - scale * args.signal_coef * signal
                - scale * args.colour_diversity_coef * colour_spread)
        if args.colour_entropy_target > 0:
            loss = loss - log_alpha.exp().detach() * colour_entropy
        return loss, [ent, v_loss, listen, signal, colour_entropy]

    def optimise(loss, colour_entropy):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        if args.colour_entropy_target > 0:
            # Raise the coefficient while the entropy is below the target.
            alpha_optimizer.zero_grad(set_to_none=True)
            (log_alpha * (colour_entropy.detach() - args.colour_entropy_target)).backward()
            alpha_optimizer.step()
            if args.colour_alpha_max > 0:
                with torch.no_grad():
                    log_alpha.clamp_(max=float(np.log(args.colour_alpha_max)))

    def flat_update(adv, returns, scale):
        """PPO on shuffled single decisions (memory bits are sampled actions)."""
        mask = buf_alive.flatten()
        b_obs = buf_obs.view(-1, NUM_OBS)[mask]
        b_priv = buf_priv.view(-1, NUM_PRIVILEGED)[mask]
        b_raw = buf_raw.view(-1, mp.NUM_RAW)[mask].long()
        b_logp = buf_logp.flatten()[mask]
        b_adv = adv.flatten()[mask]
        b_ret = returns.flatten()[mask]
        b_on = buf_on.flatten()[mask].float()
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n = b_obs.shape[0]
        batch = (n + args.minibatches - 1) // args.minibatches
        stats = []
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=device)
            for k in range(0, n, batch):
                idx = perm[k:k + batch]
                logits, aux = actor(b_obs[idx])
                loss, terms = ppo_losses(logits, aux, b_obs[idx], b_priv[idx], b_raw[idx],
                                         b_logp[idx], b_adv[idx], b_ret[idx], b_on[idx],
                                         torch.ones(len(idx), device=device), scale, memory=True)
                optimise(loss, terms[-1])
                stats.append(torch.stack([*terms, torch.zeros((), device=device)]).detach())
        return stats, n

    replay_checked = []

    def recurrent_update(adv, returns, scale, memory_aux_coef):
        """
        PPO on sequences of --bptt ticks per greep, replaying the stored
        memory so that gradients flow through it from later decisions to the
        tick it was written (straight-through bits).
        """
        L = args.bptt
        C = T // L

        def seq(x):          # (T, E, N, ...) -> (L, C * E * N, ...)
            x = x.reshape(C, L, *x.shape[1:]).transpose(0, 1)
            return x.reshape(L, C * E * N, *x.shape[4:])

        s_obs, s_priv, s_raw = seq(buf_obs), seq(buf_priv), seq(buf_raw).long()
        s_logp, s_ret, s_on = seq(buf_logp), seq(returns), seq(buf_on).float()
        s_alive = seq(buf_alive)
        s_done = seq(buf_done[..., None].expand(T, E, N))
        alive_adv = adv[buf_alive]
        s_adv = seq((adv - alive_adv.mean()) / (alive_adv.std() + 1e-8))
        n = int(buf_alive.sum())

        S = C * E * N
        batch = (S + args.minibatches - 1) // args.minibatches
        stats = []
        for _ in range(args.epochs):
            perm = torch.randperm(S, device=device)
            for k in range(0, S, batch):
                idx = perm[k:k + batch]
                o_seq, raw_seq = s_obs[:, idx], s_raw[:, idx]
                alive_seq, done_seq = s_alive[:, idx], s_done[:, idx]
                mem = o_seq[0, :, MEMORY_OBS]       # as stored at the chunk's start
                inputs, all_logits, all_aux, memories = [], [], [], []
                for l in range(L):
                    o = torch.cat([o_seq[l, :, :MEMORY_OBS.start], mem,
                                   o_seq[l, :, MEMORY_OBS.stop:]], -1)
                    logits, aux = actor(o)
                    inputs.append(o)
                    all_logits.append(logits)
                    all_aux.append(aux)
                    memories.append(mem)
                    stored = write_memory(logits, raw_seq[l, :, mp.MEMORY_RAW].float(), hardness)
                    mem = torch.where(alive_seq[l][:, None], stored, mem) * (1 - done_seq[l])[:, None]

                def rows(x):
                    return x.reshape(-1, *x.shape[2:])
                priv_rows = rows(s_priv[:, idx])
                w = rows(alive_seq).float()
                if not replay_checked:
                    # Before any optimiser step the replay must reproduce the
                    # rollout: same memory inputs, PPO ratio 1.
                    with torch.no_grad():
                        replayed = torch.stack(inputs)[..., MEMORY_OBS]
                        memory_error = (replayed - o_seq[..., MEMORY_OBS]).abs().max().item()
                        ratio = (mp.log_prob(torch.cat(all_logits), rows(raw_seq), memory=False)
                                 - rows(s_logp[:, idx])).exp()
                        ratio_error = ((ratio - 1).abs() * w).max().item()
                    print(f"Replay check: memory error {memory_error:.2e}, "
                          f"ratio error {ratio_error:.2e}", flush=True)
                    replay_checked.append(True)
                loss, terms = ppo_losses(
                    torch.cat(all_logits), torch.cat(all_aux), torch.cat(inputs), priv_rows,
                    rows(raw_seq), rows(s_logp[:, idx]), rows(s_adv[:, idx]), rows(s_ret[:, idx]),
                    rows(s_on[:, idx]), w, scale, memory=False)
                # Warm-up: the stored memory should predict privileged facts.
                predicted = memory_aux(torch.cat(memories))
                memory_aux_loss = ((predicted - priv_rows[:, mp.AUX_TARGETS]).pow(2).mean(-1)
                                   * w).sum() / w.sum().clamp(min=1)
                optimise(loss + memory_aux_coef * memory_aux_loss, terms[-1])
                stats.append(torch.stack([*terms, memory_aux_loss]).detach())
        return stats, n

    def save_latest():
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                    "memory_aux": memory_aux.state_dict(), "log_alpha": log_alpha.detach(),
                    "optimizer": optimizer.state_dict(), "update": update,
                    "best": best, "best_gain": best_gain}, latest)

    start = time.monotonic()
    deadline = start + args.minutes * 60
    try:
        while time.monotonic() < deadline and update < args.updates:
            # Incentive annealing, down to a floor that can stay on for good.
            scale = max(args.incentive_floor, 1 - update / args.anneal_updates)
            # Longer games early on (more chances to find far piles and to
            # read paint), shrinking to the real 120 s.
            game_seconds = round(args.game_seconds_final + (args.game_seconds - args.game_seconds_final)
                                 * max(0.0, 1 - update / args.game_anneal_updates))
            sim.game_seconds.fill_(game_seconds)
            # Memory: soft -> hard bits, and the warm-up read-out fading out.
            h = min(1.0, args.memory_hardness_start
                    + (1 - args.memory_hardness_start) * update / max(1, args.memory_soft_updates))
            h = h if args.trainable_memory else 1.0
            hardness.fill_(h)
            memory_aux_coef = args.memory_aux_coef * max(0.0, 1 - update / max(1, args.memory_aux_updates))
            phase_start = time.monotonic()
            with torch.no_grad():
                buf_bonus.zero_()
                rollout_start.copy_(sim.steps)
                for t in range(T):
                    buf_obs[t].copy_(obs)
                    buf_priv[t].copy_(priv)
                    buf_alive[t].copy_(alive)
                    raw, logp, value, reward, done, drops, on, late = run()
                    buf_raw[t].copy_(raw)
                    buf_logp[t].copy_(logp)
                    buf_value[t].copy_(value)
                    buf_reward[t].copy_(reward)
                    buf_bonus[t] += late
                    buf_drops[t].copy_(drops)
                    buf_done[t].copy_(done)
                    buf_on[t].copy_(on)
                next_value = critic(obs, priv)

                rewards = (buf_reward + args.influence_coef * scale * buf_bonus
                           - args.spit_cost * buf_drops)
                adv = torch.zeros_like(rewards)
                last = torch.zeros(E, N, device=device)
                for t in reversed(range(T)):
                    nonterminal = (1 - buf_done[t])[:, None]
                    nv = next_value if t == T - 1 else buf_value[t + 1]
                    delta = rewards[t] + args.gamma * nv * nonterminal - buf_value[t]
                    last = delta + args.gamma * args.lam * nonterminal * last
                    adv[t] = last
                returns = adv + buf_value
            torch.cuda.synchronize()
            rollout_seconds = time.monotonic() - phase_start
            phase_start = time.monotonic()

            if args.trainable_memory:
                stats, n = recurrent_update(adv, returns, scale, memory_aux_coef)
            else:
                stats, n = flat_update(adv, returns, scale)
            update += 1
            ent, v_loss, listen, signal, colour_entropy, memory_aux_loss = \
                torch.stack(stats).mean(0).tolist()
            colour_alpha = float(log_alpha.detach().exp())
            update_seconds = time.monotonic() - phase_start
            # A diverged run would otherwise train on NaN weights until time
            # runs out; stop before latest.pt is overwritten with them.
            if not all(np.isfinite([ent, v_loss, colour_entropy, colour_alpha])):
                raise SystemExit(f"update {update}: loss is not finite (entropy {ent}, "
                                 f"value loss {v_loss}); stopping, latest.pt is from "
                                 "before the divergence")

            total, episodes = finished.tolist()
            greep_ticks, drop_count, reader_ticks, influence_sum = paint_stats.tolist()
            train_score = total / episodes if episodes else float("nan")
            drops_per_100 = 100 * drop_count / max(1, greep_ticks)
            on_frac = reader_ticks / max(1, greep_ticks)
            influence = influence_sum / max(1, reader_ticks)
            paint_stats.zero_()
            if episodes:
                finished.zero_()
            row = [update, n, int(episodes), f"{train_score:.2f}", f"{ent:.3f}",
                   f"{v_loss:.4f}", f"{drops_per_100:.2f}", f"{on_frac:.3f}",
                   f"{influence:.4f}", f"{listen:.4f}", f"{signal:.4f}", f"{scale:.3f}",
                   "", "", "", "", "", "", "", ""]
            certainty_sum, certainty_count = memory_certainty.tolist()
            certainty = certainty_sum / max(1, certainty_count)   # 0: coin flips, 1: certain bits
            memory_certainty.zero_()
            row += [f"{h:.3f}", f"{memory_aux_loss:.4f}", f"{certainty:.3f}",
                    f"{colour_entropy:.4f}", f"{colour_alpha:.4f}"]
            message = (f"update {update:5d}  score {train_score:6.2f}  drops/100 {drops_per_100:5.2f}  "
                       f"on others' paint {on_frac:5.1%}  influence {influence:.4f}  "
                       f"listen {listen:.4f}  signal {signal:.4f}  incentives x{scale:.2f}  "
                       f"game {game_seconds}s  ")
            if args.trainable_memory:
                message += (f"memory: hard {h:.2f}  certain {certainty:.2f}  "
                            f"aux {memory_aux_loss:.4f} (x{memory_aux_coef:.3f})  ")
            message += f"colour entropy {colour_entropy:.3f}"
            if args.colour_entropy_target > 0:
                message += f" (x{colour_alpha:.3f})"
            message += "  "
            message += f"[rollout {rollout_seconds:.1f}s  update {update_seconds:.1f}s]"

            if update % args.eval_every == 0:
                with_paint, scrambled, recoloured = evaluate_comm(actor, select_bank, device,
                                                                  args.eval_episodes, seed=update)
                # Per map, so the numbers do not depend on how many maps there are.
                score = float(with_paint.mean())
                info_gain = float((with_paint - scrambled).mean())
                colour_gain = float((with_paint - recoloured).mean())
                # best_comm.pt tracks colour gain in colour mode, information gain otherwise.
                gain = colour_gain if args.colour_incentives else info_gain
                row[12:15] = [f"{score:.2f}", f"{scrambled.mean():.2f}", f"{info_gain:.2f}"]
                row[18] = f"{colour_gain:.2f}"
                message += (f"\n    selection maps ({len(select_bank)}, GPU sim): mean {score:.2f}  "
                            f"scrambled paint {scrambled.mean():.2f}  "
                            f"info gain per map {info_gain:+.2f}  colour gain {colour_gain:+.2f}")
                checkpoint = {"actor": actor.state_dict(), "update": update,
                              "eval": with_paint.tolist(), "eval_scrambled": scrambled.tolist(),
                              "eval_colours_shuffled": recoloured.tolist()}
                # Checkpoints are chosen on the training maps only.
                if score > best:
                    best = score
                    message += "  *best score*"
                    save_best = True
                else:
                    save_best = False
                if gain > best_gain:
                    best_gain = gain
                    message += "  *best colour gain*" if args.colour_incentives else "  *best info gain*"
                    save_comm = True
                else:
                    save_comm = False
                if report_bank is not None:
                    held, held_scrambled, held_recoloured = evaluate_comm(
                        actor, report_bank, device, args.eval_episodes, seed=update + 1)
                    held_gain = float((held - held_scrambled).mean())
                    held_colour_gain = float((held - held_recoloured).mean())
                    row[15:18] = [f"{held.mean():.2f}", f"{held_scrambled.mean():.2f}",
                                  f"{held_gain:.2f}"]
                    row[19] = f"{held_colour_gain:.2f}"
                    message += (f"\n    held-out maps ({len(report_bank)}): mean {held.mean():.2f}  "
                                f"scrambled paint {held_scrambled.mean():.2f}  "
                                f"info gain per map {held_gain:+.2f}  "
                                f"colour gain {held_colour_gain:+.2f}")
                    checkpoint["held_out"] = held.tolist()
                    checkpoint["held_out_scrambled"] = held_scrambled.tolist()
                    checkpoint["held_out_colours_shuffled"] = held_recoloured.tolist()
                if save_best:
                    torch.save(checkpoint, best_file)
                if save_comm:
                    torch.save(checkpoint, best_comm_file)
                save_latest()

            row.append(f"{time.monotonic() - start:.1f}")
            writer.writerow(row)
            log.flush()
            print(message, flush=True)
    finally:
        log.close()
    save_latest()
    print(f"Best eval score {best:.2f} -> {best_file}")
    print(f"Best info gain {best_gain:+.2f} -> {best_comm_file}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", default="main", help="run name: files go to out/paint/<run>/")
    p.add_argument("--minutes", type=float, default=60)
    p.add_argument("--updates", type=int, default=10**9)
    p.add_argument("--init-from", help="start the actor (and critic, if present) "
                                       "from a checkpoint, e.g. out/ppo/run1/best.pt")
    p.add_argument("--hidden", type=int, default=64,
                   help="actor width; only 64 (with no blocks) runs in Greep.java")
    p.add_argument("--blocks", type=int, default=0, help="residual blocks in the actor")
    # Trainable memory (see the module docstring).
    p.add_argument("--trainable-memory", action="store_true",
                   help="memory and flags as a recurrent state trained through time")
    p.add_argument("--bptt", type=int, default=32, help="ticks per training sequence")
    p.add_argument("--memory-hardness-start", type=float, default=0.0,
                   help="0: memory starts fully soft (continuous); 1: hard bits throughout")
    p.add_argument("--memory-soft-updates", type=int, default=300,
                   help="updates over which memory becomes hard bits")
    p.add_argument("--memory-aux-coef", type=float, default=0.0,
                   help="warm-up: stored memory predicts privileged facts")
    p.add_argument("--memory-aux-updates", type=int, default=300,
                   help="the warm-up fades out over this many updates")
    p.add_argument("--memory-noise", type=float, default=0.0,
                   help="chance per tick that a stored memory bit flips (training only)")
    # Exploration and credit.
    p.add_argument("--colour-entropy-target", type=float, default=0.0,
                   help="keep the entropy of the colour chosen when spitting above this "
                        "(nats, at most ln 3 = 1.1) with an adaptive coefficient")
    p.add_argument("--colour-entropy-lr", type=float, default=0.01)
    p.add_argument("--colour-alpha-max", type=float, default=0.0,
                   help="cap on the colour entropy coefficient (0: none)")
    p.add_argument("--team-share", type=float, default=0.0,
                   help="fraction of each delivery's reward split over all greeps")
    # Communication incentives.
    p.add_argument("--influence-coef", type=float, default=0.005,
                   help="reward per unit of KL a drop causes in another greep's movement")
    p.add_argument("--listen-coef", type=float, default=0.01)
    p.add_argument("--influence-clip", type=float, default=0.3,
                   help="cap on the influence counted per reader and tick")
    p.add_argument("--listen-clip", type=float, default=0.3)
    p.add_argument("--signal-coef", type=float, default=0.01)
    p.add_argument("--anneal-updates", type=int, default=800,
                   help="the three incentives above fade over this many updates ...")
    p.add_argument("--incentive-floor", type=float, default=0.0,
                   help="... down to this fraction (0: off at the end)")
    p.add_argument("--colour-incentives", action="store_true",
                   help="measure influence and listening against colour-shuffled paint, "
                        "and signalling on the colour choice: rewards only the information "
                        "in *which* colour is used")
    p.add_argument("--colour-diversity-coef", type=float, default=0.0,
                   help="bonus for the entropy of the average colour choice (all three used)")
    p.add_argument("--spit-cost", type=float, default=0.005,
                   help="per paint drop (0.02 drove spitting to zero within ~20 updates)")
    # Game-length curriculum (evaluation always plays the real 120 s).
    p.add_argument("--game-seconds", type=int, default=360,
                   help="game length at the start of training")
    p.add_argument("--game-seconds-final", type=int, default=120)
    p.add_argument("--game-anneal-updates", type=int, default=800,
                   help="shrink linearly to --game-seconds-final over this many updates")
    # As in policies/mlp_ppo/train.py.
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=128,
                   help="rollout length: drops read within it credit the spitting tick")
    p.add_argument("--random-maps", type=int, default=512)
    p.add_argument("--real-weight", type=float, default=0.2)
    p.add_argument("--train-maps", type=int, nargs="+", default=[1, 2])
    p.add_argument("--custom-maps", help="training maps, e.g. maps/training")
    p.add_argument("--custom-weight", type=float, default=0.4)
    p.add_argument("--eval-maps", help="held-out maps, only reported (never trained or selected on)")
    p.add_argument("--select-maps", help="maps used to choose checkpoints (default: all custom "
                                         "maps): a glob or folder, e.g. maps/validation")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--aux-coef", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--loader-share", type=float, default=0.5)
    p.add_argument("--shaping", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eager", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
