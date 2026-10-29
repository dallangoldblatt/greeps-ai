"""
Train the Greep MLP policy with MAPPO on the GPU simulator (greeps/torch_sim.py).

Centralized training, decentralized execution:
  * the actor (model.py) sees only what a greep may legally see; it is what
    gets exported to Greep.java;
  * the critic also sees privileged state (walking distances and directions
    to piles and the ship, team progress, time left); it is training-only;
  * rewards are per greep: a delivered tomato pays the carrier and the greep
    that loaded it, plus potential-based shaping from walking distances;
  * an auxiliary head trains the actor's hidden layer to predict privileged
    quantities (dropped on export).

Training maps: the real Maps 1 and 2, hand-made maps from tools/map_editor.py
(--custom-maps) and random maps. Map 3 is held out and only reported. Every
--eval-every updates the actor is scored in the GPU sim; the best one (mean
score per map over Maps 1, 2 and the custom maps) is saved to
out/ppo/<run>/best.pt. Each tick is replayed as a CUDA graph (--eager to
turn that off).

    python -m policies.mlp_ppo.train --minutes 60 --run main
    python -m policies.mlp_ppo.train --minutes 60 --run generated --custom-maps maps/training
    python -m policies.mlp_ppo.train --resume --minutes 60 --run main
    python -m policies.mlp_ppo.export --run main   # bake its best.pt into Greep.java
"""

import argparse
import csv
import time

import numpy as np
import torch
import torch.nn as nn

from greeps.paths import OUT_DIR as REPO_OUT_DIR
from greeps.torch_sim import NUM_OBS, NUM_PRIVILEGED, GreepsSim, MapBank, capture_graph, evaluate

from . import model as mp


RUNS_DIR = REPO_OUT_DIR / "ppo"
# Set per run in train(): out/ppo/<run>/{latest.pt, best.pt, log.csv}.
OUT_DIR = LATEST = BEST = LOG_FILE = None
CRITERION = "mean_per_map"     # how "best" is measured (see the eval block)


class Critic(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_OBS + NUM_PRIVILEGED, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, priv):
        return self.net(torch.cat([obs, priv], dim=-1)).squeeze(-1)


def build_bank(args, device):
    """
    Bank layout: the 3 real maps, then custom maps (tools/map_editor.py), then
    random maps. Episode shares: --real-weight for the real training maps,
    --custom-weight for custom maps, the rest for random maps.
    """
    real = MapBank.real(device)
    custom = MapBank.from_files(args.custom_maps, device) if args.custom_maps else None
    banks = [real] + ([custom] if custom else [])
    if args.random_maps:
        banks.append(MapBank.random(args.random_maps, device, seed=args.seed))
    bank = MapBank.concat(banks)

    n_custom = len(custom) if custom else 0
    shares = {"real": args.real_weight, "custom": args.custom_weight if custom else 0.0}
    shares["random"] = max(0.0, 1 - shares["real"] - shares["custom"]) if args.random_maps else 0.0
    total = sum(shares.values())
    weights = torch.zeros(len(bank))
    train_real = [m - 1 for m in args.train_maps]
    weights[train_real] = shares["real"] / total / len(train_real)
    if custom:
        weights[3:3 + n_custom] = shares["custom"] / total / n_custom
    if args.random_maps:
        weights[3 + n_custom:] = shares["random"] / total / args.random_maps
    return bank, real, custom, weights


def evaluate_actor(actor, bank, device, episodes, seed):
    """Mean score per map of `bank` in the GPU sim."""
    actor.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    scores = evaluate(mp.sim_policy(actor, gen), bank, list(range(len(bank))),
                      episodes, device, seed=seed)
    actor.train()
    return scores.mean(axis=1)


def train(args):
    global OUT_DIR, LATEST, BEST, LOG_FILE
    OUT_DIR = RUNS_DIR / args.run
    LATEST, BEST, LOG_FILE = OUT_DIR / "latest.pt", OUT_DIR / "best.pt", OUT_DIR / "log.csv"
    if LATEST.exists() and not args.resume:
        raise SystemExit(f"{OUT_DIR} already has a run; use --resume or another --run name")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bank, real, custom, map_weights = build_bank(args, device)
    sim = GreepsSim(bank, args.envs, device, seed=args.seed,
                    loader_share=args.loader_share, shaping=args.shaping,
                    gamma=args.gamma, map_weights=map_weights)
    # Stagger the first episodes so that batches mix all phases of a game.
    sim.reset(start_ticks=torch.randint(1, 1100, (args.envs,), device=device))

    actor, critic = mp.Actor().to(device), Critic().to(device)
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, eps=1e-5)
    update, best = 0, float("-inf")
    if args.resume:
        state = torch.load(LATEST, map_location=device)
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        optimizer.load_state_dict(state["optimizer"])
        update, best = state["update"], state["best"]
        if state.get("criterion") != CRITERION:
            best = float("-inf")    # older checkpoints selected on a different score
        print(f"Resumed at update {update}, best {best:.2f}")

    gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    T, E, N = args.steps, args.envs, sim.N
    buf_obs = torch.zeros(T, E, N, NUM_OBS, device=device)
    buf_priv = torch.zeros(T, E, N, NUM_PRIVILEGED, device=device)
    buf_raw = torch.zeros(T, E, N, mp.NUM_RAW, dtype=torch.uint8, device=device)  # all < 256
    buf_logp = torch.zeros(T, E, N, device=device)
    buf_value = torch.zeros(T, E, N, device=device)
    buf_reward = torch.zeros(T, E, N, device=device)
    buf_done = torch.zeros(T, E, device=device)
    buf_alive = torch.zeros(T, E, N, dtype=torch.bool, device=device)

    new_log = not (args.resume and LOG_FILE.exists())
    log = open(LOG_FILE, "w" if new_log else "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["update", "samples", "train_episodes", "train_score",
                         "real_map_score", "entropy", "value_loss", "aux_loss",
                         "eval_map1", "eval_map2", "eval_map3", "eval_custom", "seconds"])

    # The observation the next tick acts on (fixed tensors, updated in place).
    obs, priv, alive = sim.observe(), sim.privileged(), sim.alive()
    # Finished episodes: score sum and count, overall and on real Maps 1-2.
    finished = torch.zeros(4, device=device)

    def tick():
        """One tick: act, step, record finished games, restart them, observe."""
        logits, _ = actor(obs)
        raw, logp = mp.sample(logits, gen)
        value = critic(obs, priv)
        reward, done, scores = sim.step(mp.to_sim_actions(raw))
        d = done.float()
        real_done = d * (sim.map_id < 2).float()
        finished.add_(torch.stack([(scores * d).sum(), d.sum(),
                                   (scores * real_done).sum(), real_done.sum()]))
        sim.reset(done)
        obs.copy_(sim.observe())
        priv.copy_(sim.privileged())
        alive.copy_(sim.alive())
        return raw, logp, value, reward, d

    with torch.no_grad():
        if args.eager:
            run = tick
        else:
            # Record the tick once as a CUDA graph; each replay is one tick
            # with a single launch. The actor and critic are read in place,
            # so the graph keeps using the latest weights after each update.
            graph, outputs = capture_graph(tick)

            def run():
                graph.replay()
                return outputs

    start = time.monotonic()
    deadline = start + args.minutes * 60
    try:
        while time.monotonic() < deadline and update < args.updates:
            # ---- rollout ----
            phase_start = time.monotonic()
            with torch.no_grad():
                for t in range(T):
                    buf_obs[t].copy_(obs)
                    buf_priv[t].copy_(priv)
                    buf_alive[t].copy_(alive)
                    raw, logp, value, reward, done = run()
                    buf_raw[t].copy_(raw)
                    buf_logp[t].copy_(logp)
                    buf_value[t].copy_(value)
                    buf_reward[t].copy_(reward)
                    buf_done[t].copy_(done)
                next_value = critic(obs, priv)

                # ---- GAE (per greep; an episode end cuts every greep in that env) ----
                adv = torch.zeros_like(buf_reward)
                last = torch.zeros(E, N, device=device)
                for t in reversed(range(T)):
                    nonterminal = (1 - buf_done[t])[:, None]
                    nv = next_value if t == T - 1 else buf_value[t + 1]
                    delta = buf_reward[t] + args.gamma * nv * nonterminal - buf_value[t]
                    last = delta + args.gamma * args.lam * nonterminal * last
                    adv[t] = last
                returns = adv + buf_value
            torch.cuda.synchronize()
            rollout_seconds = time.monotonic() - phase_start
            phase_start = time.monotonic()

            # ---- PPO update on greeps that were out of the ship ----
            mask = buf_alive.flatten()
            b_obs = buf_obs.view(-1, NUM_OBS)[mask]
            b_priv = buf_priv.view(-1, NUM_PRIVILEGED)[mask]
            b_raw = buf_raw.view(-1, mp.NUM_RAW)[mask].long()
            b_logp = buf_logp.flatten()[mask]
            b_adv = adv.flatten()[mask]
            b_ret = returns.flatten()[mask]
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            aux_target = b_priv[:, mp.AUX_TARGETS]

            n = b_obs.shape[0]
            batch = (n + args.minibatches - 1) // args.minibatches
            stats = []
            for _ in range(args.epochs):
                perm = torch.randperm(n, device=device)
                for k in range(0, n, batch):
                    idx = perm[k:k + batch]
                    logits, aux = actor(b_obs[idx])
                    new_logp = mp.log_prob(logits, b_raw[idx])
                    ratio = (new_logp - b_logp[idx]).exp()
                    a = b_adv[idx]
                    pg_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                    ent = mp.entropy(logits).mean()
                    v_loss = 0.5 * (critic(b_obs[idx], b_priv[idx]) - b_ret[idx]).pow(2).mean()
                    aux_loss = (aux - aux_target[idx]).pow(2).mean()
                    loss = pg_loss - args.ent_coef * ent + args.vf_coef * v_loss + args.aux_coef * aux_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        log_ratio = new_logp - b_logp[idx]
                        kl = ((log_ratio.exp() - 1) - log_ratio).mean()
                        clipped = ((ratio - 1).abs() > args.clip).float().mean()
                    stats.append(torch.stack([ent, v_loss, aux_loss, kl, clipped]).detach())
            update += 1
            ent, v_loss, aux_loss, kl, clipped = torch.stack(stats).mean(0).tolist()
            update_seconds = time.monotonic() - phase_start

            # ---- logging / evaluation ----
            total, episodes, real_total, real_episodes = finished.tolist()
            train_score = total / episodes if episodes else float("nan")
            real_score = real_total / real_episodes if real_episodes else float("nan")
            row = [update, n, int(episodes), f"{train_score:.2f}", f"{real_score:.2f}",
                   f"{ent:.3f}", f"{v_loss:.4f}", f"{aux_loss:.4f}", "", "", "", ""]
            message = (f"update {update:5d}  samples {n / 1e6:5.2f}M  "
                       f"episodes {int(episodes):5d}  score {train_score:6.2f}  "
                       f"real 1+2 {real_score:6.2f}  entropy {ent:6.3f}  "
                       f"v_loss {v_loss:.4f}  aux {aux_loss:.4f}  kl {kl:.4f}  clip {clipped:.3f}  "
                       f"[rollout {rollout_seconds:.1f}s  update {update_seconds:.1f}s]")
            if episodes:
                finished.zero_()

            if update % args.eval_every == 0:
                means = evaluate_actor(actor, real, device, args.eval_episodes, seed=update)
                row[8:11] = [f"{m:.2f}" for m in means]
                selection = [means[m - 1] for m in args.train_maps]
                message += (f"\n    eval (GPU sim): map1 {means[0]:.2f}  map2 {means[1]:.2f}  "
                            f"map3 {means[2]:.2f} (held out)")
                if custom:
                    custom_means = evaluate_actor(actor, custom, device,
                                                  max(8, args.eval_episodes // 4), seed=update)
                    selection += list(custom_means)
                    row[11] = f"{custom_means.mean():.2f}"
                    message += f"  custom {custom_means.mean():.2f} ({len(custom)} maps)"
                # Select on the mean per-map score over training + custom maps.
                score = float(np.mean(selection))
                if score > best:
                    best = score
                    torch.save({"actor": actor.state_dict(), "update": update,
                                "eval": means.tolist()}, BEST)
                    message += "  *saved*"
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                            "optimizer": optimizer.state_dict(), "update": update,
                            "best": best, "criterion": CRITERION}, LATEST)

            row.append(f"{time.monotonic() - start:.1f}")
            writer.writerow(row)
            log.flush()
            print(message, flush=True)
    finally:
        log.close()
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(), "update": update, "best": best,
                "criterion": CRITERION}, LATEST)
    print(f"Best eval (mean per training/custom map): {best:.2f}  -> {BEST}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--minutes", type=float, default=60)
    p.add_argument("--updates", type=int, default=10**9)
    p.add_argument("--envs", type=int, default=2048)
    p.add_argument("--steps", type=int, default=64, help="rollout length (ticks)")
    p.add_argument("--random-maps", type=int, default=512)
    p.add_argument("--real-weight", type=float, default=0.3,
                   help="share of episodes on the real training maps")
    p.add_argument("--train-maps", type=int, nargs="+", default=[1, 2])
    p.add_argument("--custom-maps", help="directory of hand-made maps (tools/map_editor.py)")
    p.add_argument("--custom-weight", type=float, default=0.4,
                   help="share of episodes on custom maps")
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
    p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", default="main", help="run name: files go to out/ppo/<run>/")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eager", action="store_true",
                   help="run ticks without CUDA graphs (slower; for debugging)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
