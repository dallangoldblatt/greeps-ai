"""
Where do the real (Java) scenario and the GPU simulator differ for a policy?

    python -m tools.sim_gap --checkpoint out/paint/memory_st/latest.pt --maps maps/validation

Plays the paint_comm Greep in Java with an observation trace
(-Dgreeps.obsTrace; paint_comm is made active for the run, and the previous
policy restored), and the same weights on the GPU simulator, then compares:

  1. scores per map;
  2. observation and decision statistics per map, largest differences first;
  3. paint contact geometry: for every drop within 40 px of a greep in the
     Java trace, Greenfoot's "touching" against the GPU simulator's test for
     the same positions (greeps.torch_sim.touches_paint);
  4. policy parity: the log-likelihood of Java's sampled decisions under the
     PyTorch actor, against the actor's own entropy (equal if the two forward
     passes and samplers agree).

Results go to out/sim_gap/<checkpoint>/.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from greeps import policies
from greeps.java_sim import compile_scenario, custom_maps_prop, run_headless, write_policy
from greeps.maps import map_files, read_map_file
from greeps.paths import OUT_DIR
from greeps.torch_sim import (
    NUM_OBS, PAINT_SIZE, GreepsSim, MapBank, _RotationTables, box_shape, greep_shape,
    intersects, touches_paint,
)
from policies.mlp_ppo import model as mp


OBS_NAMES = (["carrying", "at_ship", "at_food", "at_water", "at_edge", "red", "orange", "purple"]
             + [f"memory{i}" for i in range(8)] + ["flag1", "flag2"])
# Decision statistics: name -> function of the raw decision columns
# (mode, turn bin, move, 8 memory bits, 2 flags, spit).
DECISIONS = {
    "mode=keep": lambda a: a[:, 0] == 0, "mode=home": lambda a: a[:, 0] == 1,
    "mode=pile": lambda a: a[:, 0] == 2, "turning": lambda a: a[:, 1] != 18,
    "move": lambda a: a[:, 2] == 1, "spit": lambda a: a[:, -1] > 0,
}


def parse_trace(path, ships):
    """Java trace -> per map: obs (n, 22), raw decisions (n, 13), and paint pairs."""
    obs, raw, maps, pairs = [], [], [], []
    for line in open(path, encoding="utf-8"):
        if not line.startswith("O "):
            continue
        head, rest = line.split(" I ", 1)
        values, rest = rest.split(" P", 1)
        paints, rest = rest.split(" A ", 1)
        decisions, _ = rest.split(" T ", 1)
        _, sx, sy, _, x, y, rot = head.split()
        m = ships.get((int(sx), int(sy)))
        if m is None:      # the map whose ship is nearest
            m = min(ships.items(), key=lambda kv: abs(kv[0][0] - int(sx)) + abs(kv[0][1] - int(sy)))[1]
            ships[(int(sx), int(sy))] = m
        maps.append(m)
        obs.append([float(v) for v in values.split()])
        mode, turn, memory, flags, spit, move = decisions.split()
        raw.append([int(mode), int(turn), int(move)]
                   + [(int(memory) >> b) & 1 for b in range(8)]
                   + [int(flags[0]), int(flags[1]), int(spit)])
        for p in paints.split():
            px, py, _, hit = map(int, p.split(","))
            pairs.append((int(x), int(y), int(rot), px, py, hit))
    return np.array(maps), np.array(obs, dtype=np.float32), np.array(raw), np.array(pairs)


def play_java(weights, maps, seeds, episodes, out):
    """Several JVMs, each with its own trace file. Returns scores (episodes, maps), trace paths."""
    write_policy(weights, out / "policy.weights")
    props = custom_maps_prop(maps)
    props["greeps.policy"] = str((out / "policy.weights").resolve())

    def one(seed):
        trace = out / f"java-trace-{seed}.txt"
        trace.unlink(missing_ok=True)
        scores = run_headless(seed=seed, episodes=episodes, timeout_seconds=3600,
                              props={**props, "greeps.obsTrace": str(trace.resolve())})
        return scores, trace

    with ThreadPoolExecutor(len(seeds)) as pool:
        results = list(pool.map(one, seeds))
    scores = np.array([s for r in results for s in r[0]], dtype=float)
    return scores, [r[1] for r in results]


def play_gpu(actor, bank, episodes, device):
    """Eager GPU episodes: scores (maps, episodes) and per-map sums of statistics."""
    ids = torch.arange(len(bank), device=device).repeat_interleave(episodes)
    sim = GreepsSim(bank, len(ids), device, seed=7)
    sim.reset(map_ids=ids)
    gen = torch.Generator(device=device).manual_seed(7)
    M = len(bank)
    obs_sum = torch.zeros(M, NUM_OBS, device=device, dtype=torch.float64)
    dec_sum = torch.zeros(M, len(DECISIONS), device=device, dtype=torch.float64)
    count = torch.zeros(M, device=device, dtype=torch.float64)
    scores = torch.zeros(len(ids), dtype=torch.long, device=device)
    with torch.no_grad():
        while bool(sim.active.any()):
            obs, alive = sim.observe(), sim.alive()
            logits, _ = actor(obs)
            raw, _ = mp.sample(logits, gen)
            m = sim.map_id[:, None].expand_as(alive)[alive]
            o, r = obs[alive].double(), raw[alive]
            obs_sum.index_add_(0, m, o)
            dec_sum.index_add_(0, m, torch.stack([f(r).double() for f in DECISIONS.values()], -1))
            count.index_add_(0, m, torch.ones_like(m, dtype=torch.float64))
            _, done, s = sim.step(mp.to_sim_actions(raw))
            scores = torch.where(done, s, scores)
    return (scores.view(M, episodes).cpu().numpy(),
            (obs_sum / count[:, None]).cpu().numpy(), (dec_sum / count[:, None]).cpu().numpy())


def geometry(pairs, device):
    """Confusion of Java's touching flag against the GPU tests, for the same positions."""
    t = torch.as_tensor(pairs, device=device)
    x, y, rot, px, py, java = t.unbind(-1)
    tables = _RotationTables(device)
    shape = greep_shape(tables, x, y, rot)
    result = {}
    for name, test in (("GPU sim test", lambda: touches_paint(tables, shape, x, y, rot, px, py)),
                       ("bounding box only", lambda: touches_paint(tables, shape, x, y, rot, px, py,
                                                                   exact=False)),
                       ("Actor.intersects", lambda: intersects(
                           shape, box_shape(px, py, PAINT_SIZE, PAINT_SIZE)))):
        gpu = test().long()
        result[name] = {
            "both": int(((gpu == 1) & (java == 1)).sum()),
            "java only": int(((gpu == 0) & (java == 1)).sum()),
            "gpu only": int(((gpu == 1) & (java == 0)).sum()),
            "neither": int(((gpu == 0) & (java == 0)).sum()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--maps", default="maps/validation")
    parser.add_argument("--java-jvms", type=int, default=4)
    parser.add_argument("--java-episodes", type=int, default=2, help="per JVM")
    parser.add_argument("--gpu-episodes", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda")
    out = OUT_DIR / "sim_gap" / Path(args.checkpoint).parent.name
    out.mkdir(parents=True, exist_ok=True)
    actor = mp.load_actor(torch.load(args.checkpoint, map_location="cpu")["actor"]).to(device).eval()
    weights = actor.export_params()
    files = map_files(args.maps)
    names = [f.stem for f in files]
    ships = {tuple(read_map_file(f)["ship"]): i for i, f in enumerate(files)}
    bank = MapBank.from_files(args.maps, device)

    previous = policies.active_name()
    policies.activate("paint_comm")
    try:
        compile_scenario()
        seeds = [9_000_000 + 1000 * j for j in range(args.java_jvms)]
        java_scores, traces = play_java(weights, args.maps, seeds, args.java_episodes, out)
    finally:
        policies.activate(previous)
        compile_scenario()
    gpu_scores, gpu_obs, gpu_dec = play_gpu(actor, bank, args.gpu_episodes, device)

    parsed = [parse_trace(p, ships) for p in traces]
    j_map = np.concatenate([p[0] for p in parsed])
    j_obs = np.concatenate([p[1] for p in parsed])
    j_raw = np.concatenate([p[2] for p in parsed])
    j_pairs = np.concatenate([p[3] for p in parsed])

    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say(f"{args.checkpoint} on {args.maps}: Java {len(java_scores)} episodes, "
        f"GPU {args.gpu_episodes} per map, {len(j_obs):,} Java decisions")

    say("\n1. Scores")
    say(f"   {'map':<14} {'Java':>7} {'GPU':>7} {'diff':>7}")
    for i, n in enumerate(names):
        say(f"   {n:<14} {java_scores[:, i].mean():7.2f} {gpu_scores[i].mean():7.2f} "
            f"{gpu_scores[i].mean() - java_scores[:, i].mean():+7.2f}")
    say(f"   {'Total':<14} {java_scores.sum(1).mean():7.2f} {gpu_scores.sum(0).mean():7.2f} "
        f"{gpu_scores.sum(0).mean() - java_scores.sum(1).mean():+7.2f}")

    say("\n2. Observation and decision rates, all maps (Java vs GPU), largest relative difference first")
    j_dec = np.stack([f(j_raw) for f in DECISIONS.values()], -1).astype(float)
    weights_per_map = np.bincount(j_map, minlength=len(names)) / len(j_map)
    rows = []
    for k, name in enumerate(OBS_NAMES):
        j, g = j_obs[:, k].mean(), (gpu_obs[:, k] * weights_per_map).sum()
        rows.append((name, j, g))
    for k, name in enumerate(DECISIONS):
        j, g = j_dec[:, k].mean(), (gpu_dec[:, k] * weights_per_map).sum()
        rows.append((name, j, g))
    rows.sort(key=lambda r: -abs(r[2] - r[1]) / max(1e-3, (r[1] + r[2]) / 2))
    say(f"   {'':<12} {'Java':>8} {'GPU':>8} {'GPU/Java':>9}")
    for name, j, g in rows:
        say(f"   {name:<12} {j:8.4f} {g:8.4f} {g / max(j, 1e-9):9.3f}")
    say("   (GPU rates are weighted by each map's share of the Java decisions)")

    say("\n3. Paint contact for the same greep and drop positions (Java trace)")
    for test, c in geometry(j_pairs, device).items():
        java_hits = c["both"] + c["java only"]
        say(f"   {test:<18} agree on {c['both']:,} of {java_hits:,} Java contacts; "
            f"missed {c['java only']:,} ({c['java only'] / max(1, java_hits):.1%}), "
            f"extra {c['gpu only']:,} ({c['gpu only'] / max(1, java_hits):.1%})")

    say("\n4. Policy parity: mean log-likelihood of Java's decisions under the PyTorch actor")
    with torch.no_grad():
        o = torch.as_tensor(j_obs, device=device)
        r = torch.as_tensor(j_raw, device=device)
        logits = torch.cat([actor(o[i:i + 65536])[0] for i in range(0, len(o), 65536)])
        heads = mp.split_heads(logits)
        for name, lp, ent in [
            ("mode", torch.log_softmax(heads["mode"], -1).gather(-1, r[:, 0:1]).squeeze(-1), None),
            ("turn", torch.log_softmax(heads["turn"], -1).gather(-1, r[:, 1:2]).squeeze(-1), None),
            ("spit", torch.log_softmax(heads["spit"], -1).gather(-1, r[:, -1:]).squeeze(-1), None),
        ]:
            p = torch.softmax(heads[name], -1)
            entropy = -(p * torch.log(p + 1e-12)).sum(-1)
            say(f"   {name:<5} log-likelihood {lp.mean():.4f}   expected (-entropy) {-entropy.mean():.4f}")
        z = torch.cat([heads["move"], heads["memory"], heads["flags"]], -1)
        bits = r[:, 2:-1].float()
        lp = -torch.nn.functional.binary_cross_entropy_with_logits(z, bits, reduction="none").mean(0)
        s = torch.sigmoid(z)
        ent = torch.nn.functional.binary_cross_entropy_with_logits(z, s, reduction="none").mean(0)
        labels = ["move"] + [f"m{i}" for i in range(8)] + ["f1", "f2"]
        say("   bits  " + "  ".join(f"{l} {a:.3f}/{-b:.3f}" for l, a, b in zip(labels, lp.tolist(), ent.tolist())))
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {out / 'report.txt'}")


if __name__ == "__main__":
    main()
