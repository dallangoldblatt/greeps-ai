"""
How does a policy use paint? Plays it on the GPU simulator and reports:

  1. Paint ablation (the decisive test): the score when each greep's paint
     reading is replaced by that of a random other greep ("scrambled": same
     statistics, no information), when only the colours are replaced by
     those under a random other greep on paint ("colours shuffled": paint
     stays where it is, the information in *which* colour is gone), and when
     paint is hidden ("blind"). Blind can mislead: a policy that is almost
     always standing on paint treats "no paint" as an input it has never seen
     and can collapse for reasons unrelated to communication.
  2. Writing: how often greeps spit, in which situations, which colours.
  3. Speaker consistency: how much a drop's colour tells about the greep that
     spat it (mutual information in bits, out of at most log2(3) = 1.58).
  4. Listener influence: how often paint spat by *another* greep changes what
     a greep does compared with a scrambled reading, and with shuffled
     colours (same random numbers).
  5. Heat maps of where each colour is spat (PNG).

    python -m tools.paint_report                          # active policy, Maps 1-3
    python -m tools.paint_report --policy mlp_ppo --maps maps/validation
    python -m tools.paint_report --checkpoint out/paint/main/best.pt

Results go to out/paint_report/<name>/ (report.json, heatmaps.png), or
out/paint_report/<name>/<run>/<checkpoint>/ for a checkpoint.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from greeps import policies
from greeps.paint import memory_scrambled, paint_blind, paint_colours_shuffled, paint_scrambled
from greeps.java_sim import read_policy
from greeps.paths import OUT_DIR
from greeps.torch_sim import (
    CELL, GRID_H, GRID_W, HEIGHT, PAINT_COLORS, PAINT_OBS, WIDTH,
    GreepsSim, MapBank, evaluate,
)


def mutual_information(a, b):
    """I(a; b) in bits for two integer arrays of equal length."""
    if len(a) == 0:
        return 0.0
    joint = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(joint, (a, b), 1)
    joint /= joint.sum()
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (pa @ pb)[nz])).sum())


def sector(angle_rad):
    """Absolute compass sector (0, 1, 2) of 120 degrees each."""
    return (torch.remainder(angle_rad, 2 * math.pi) / (2 * math.pi / 3)).long().clamp(max=2)


def instrumented_run(policy, gen, bank, episodes, device, seed):
    """Play eagerly, recording paint writing, reading and influence."""
    ids = torch.arange(len(bank), device=device).repeat_interleave(episodes)
    sim = GreepsSim(bank, len(ids), device, seed=seed)
    sim.reset(map_ids=ids)
    ablated = paint_scrambled(policy)
    recoloured = paint_colours_shuffled(policy)
    heads = ["mode", "turn", "move", "memory", "flags", "spit"]

    stats = {"greep_ticks": 0, "carry_ticks": 0, "attempts": 0, "drops": 0,
             "drops_carrying": 0, "on_paint": 0, "on_others_paint": 0,
             "changed_any": 0, "colour_changed_any": 0, "colour_changed_turn": 0,
             "colour_changed_mode": 0, **{f"changed_{h}": 0 for h in heads}}
    events = {k: [] for k in ("color", "carrying", "at_food", "near_pile", "near_ship",
                              "heading", "pile_dir", "ship_dir")}
    heat = torch.zeros(len(bank), 3, GRID_H, GRID_W, device=device)

    with torch.no_grad():
        while bool(sim.active.any()):
            obs, priv, alive = sim.observe(), sim.privileged(), sim.alive()
            contacts = sim.paint_contacts()
            others = contacts & (sim.paint_author[None, None, :] != sim.greep_index[None, :, None])
            on_others = others.any(-1) & alive

            # Same random numbers with the real and a scrambled paint reading:
            # any difference in the chosen actions is caused by the paint.
            generators = [g for g in (gen, sim.gen) if g is not None]
            states = [g.get_state() for g in generators]
            acts_blind = ablated(obs, sim)
            for g, s in zip(generators, states):
                g.set_state(s)
            acts_recoloured = recoloured(obs, sim)
            for g, s in zip(generators, states):
                g.set_state(s)
            acts = policy(obs, sim)
            # Same, but only the colours relabelled: changes caused by *which* colour.
            colour_changed = torch.zeros_like(on_others)
            for h in heads:
                diff = acts[h] != acts_recoloured[h]
                diff = diff.any(-1) if diff.dim() == 3 else diff
                if h in ("turn", "mode"):
                    stats[f"colour_changed_{h}"] += int((diff & on_others).sum())
                colour_changed |= diff
            stats["colour_changed_any"] += int((colour_changed & on_others).sum())
            changed = torch.zeros_like(on_others)
            for h in heads:
                diff = acts[h] != acts_blind[h]
                diff = diff.any(-1) if diff.dim() == 3 else diff
                stats[f"changed_{h}"] += int((diff & on_others).sum())
                changed |= diff
            stats["changed_any"] += int((changed & on_others).sum())

            spits_before = sim.spits.clone()
            x0, y0, rot = sim.x.clone(), sim.y.clone(), sim.rot.clone()
            sim.step(acts)
            dropped = (sim.spits > spits_before) & alive
            carrying = obs[..., 0] > 0

            stats["greep_ticks"] += int(alive.sum())
            stats["carry_ticks"] += int((carrying & alive).sum())
            stats["attempts"] += int(((acts["spit"] > 0) & alive).sum())
            stats["drops"] += int(dropped.sum())
            stats["drops_carrying"] += int((dropped & carrying).sum())
            stats["on_paint"] += int((contacts.any(-1) & alive).sum())
            stats["on_others_paint"] += int(on_others.sum())

            if bool(dropped.any()):
                heading = torch.deg2rad(rot.float())
                pile_abs = torch.atan2(priv[..., 5], priv[..., 4]) + heading
                ship_abs = torch.atan2(priv[..., 7], priv[..., 6]) + heading
                features = {
                    "color": acts["spit"] - 1,
                    "carrying": carrying.long(),
                    "at_food": (obs[..., 2] > 0).long(),
                    "near_pile": (priv[..., 3] * 1000 < 60).long(),
                    "near_ship": (priv[..., 2] * 1000 < 60).long(),
                    "heading": sector(heading),
                    "pile_dir": sector(pile_abs),
                    "ship_dir": sector(ship_abs),
                }
                for k, v in features.items():
                    events[k].append(v[dropped].cpu().numpy())
                m = sim.map_id[:, None].expand_as(x0)[dropped]
                c = (acts["spit"] - 1)[dropped]
                gy = (y0[dropped] // CELL).clamp(0, GRID_H - 1)
                gx = (x0[dropped] // CELL).clamp(0, GRID_W - 1)
                heat.index_put_((m, c, gy, gx), torch.ones_like(gx, dtype=torch.float),
                                accumulate=True)

    events = {k: (np.concatenate(v) if v else np.zeros(0, dtype=np.int64))
              for k, v in events.items()}
    return stats, events, heat.cpu().numpy()


def save_heatmaps(path, bank, heat):
    from PIL import Image

    tiles = []
    rgb = np.array([[232, 21, 27], [245, 131, 14], [115, 74, 153]], dtype=float)
    for i in range(len(bank)):
        water = bank.water[i].cpu().numpy()[::CELL, ::CELL]
        img = np.where(water[..., None], [70, 95, 150], [235, 222, 205]).astype(float)
        h = heat[i]                                            # (3, GH, GW)
        total = h.sum(0)
        if total.max() > 0:
            strength = np.clip(np.log1p(total) / np.log1p(total.max()), 0, 1)[..., None]
            colour = np.einsum("chw,cd->hwd", h, rgb) / np.maximum(total[..., None], 1e-9)
            img = img * (1 - strength) + colour * strength
        tile = Image.fromarray(img.astype(np.uint8)).resize((400, 300), Image.NEAREST)
        tiles.append(tile)
    cols = min(5, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    grid = Image.new("RGB", (cols * 400, rows * 300), "white")
    for k, tile in enumerate(tiles):
        grid.paste(tile, ((k % cols) * 400, (k // cols) * 300))
    grid.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy", help="approach name (default: the active one)")
    parser.add_argument("--weights", help="a weights text file instead of the embedded ones")
    parser.add_argument("--checkpoint", help="a training checkpoint (actor state) instead")
    parser.add_argument("--maps", help="directory of .map files (default: Maps 1-3)")
    parser.add_argument("--episodes", type=int, default=32, help="per map, paint-blind test")
    parser.add_argument("--detail-episodes", type=int, default=8, help="per map, detailed stats")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    name = args.policy or policies.active_name()
    package = policies.load(name)
    if args.checkpoint:
        # Any actor size: played directly, not through flat weights.
        from policies.mlp_ppo import model as mp
        actor = mp.load_actor(torch.load(args.checkpoint, map_location="cpu")["actor"])
        actor = actor.to(device).eval()
        source = args.checkpoint

        def make_policy(gen):
            return mp.sim_policy(actor, gen)
    else:
        if args.weights:
            weights, source = read_policy(args.weights), args.weights
        else:
            weights, source = policies.read_embedded_weights(name), "embedded weights"

        def make_policy(gen):
            return package.torch_policy(weights, device, gen)
    bank = MapBank.from_files(args.maps, device) if args.maps else MapBank.real(device)
    label = name if not args.checkpoint else f"{name}:{args.checkpoint}"
    out = OUT_DIR / "paint_report" / name
    if args.checkpoint:
        out = out / Path(args.checkpoint).parent.name / Path(args.checkpoint).stem
    out.mkdir(parents=True, exist_ok=True)
    print(f"{label} ({source}) on {len(bank)} maps")

    # 1. Paint ablation.
    def play(wrapper=None):
        gen = torch.Generator(device=device).manual_seed(args.seed)
        policy = make_policy(gen)
        policy = wrapper(policy) if wrapper else policy
        return evaluate(policy, bank, list(range(len(bank))), args.episodes, device,
                        seed=args.seed)
    normal, scrambled = play(), play(paint_scrambled)
    recoloured, blind = play(paint_colours_shuffled), play(paint_blind)
    forgetful = play(memory_scrambled)

    def paired_gain(other):
        diff = normal.sum(0) - other.sum(0)      # paired: same maps, same seeds
        return diff.mean(), diff.std(ddof=1) / math.sqrt(len(diff))
    gain, gain_se = paired_gain(scrambled)
    colour_gain, colour_gain_se = paired_gain(recoloured)
    memory_gain, memory_gain_se = paired_gain(forgetful)
    print(f"\n1. Paint ablation (GPU sim, {args.episodes} paired episodes per map)")
    print(f"   {'map':<14} {'paint':>7} {'scrambled':>10} {'colours shuffled':>17} {'blind':>7}")
    for i, n in enumerate(bank.names):
        print(f"   {n:<14} {normal[i].mean():7.2f} {scrambled[i].mean():10.2f} "
              f"{recoloured[i].mean():17.2f} {blind[i].mean():7.2f}")
    print(f"   {'Total':<14} {normal.sum(0).mean():7.2f} {scrambled.sum(0).mean():10.2f} "
          f"{recoloured.sum(0).mean():17.2f} {blind.sum(0).mean():7.2f}")
    print(f"   information gain from paint (paint - scrambled):         {gain:+.2f} +- {gain_se:.2f}")
    print(f"   of which from colour (paint - colours shuffled):         "
          f"{colour_gain:+.2f} +- {colour_gain_se:.2f}")
    print(f"   (for comparison) gain from memory (vs another greep's memory): "
          f"{memory_gain:+.2f} +- {memory_gain_se:.2f}")

    # 2-4. Detailed statistics.
    gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    policy = make_policy(gen)
    stats, events, heat = instrumented_run(policy, gen, bank, args.detail_episodes,
                                           device, args.seed + 1)
    s = stats
    per100 = 100 * s["drops"] / max(1, s["greep_ticks"])
    print("\n2. Writing")
    print(f"   drops per greep per 100 ticks: {per100:.2f} "
          f"(spit attempts {100 * s['attempts'] / max(1, s['greep_ticks']):.1f})")
    if s["drops"]:
        carrying_rate = s["drops_carrying"] / max(1, s["carry_ticks"])
        empty_rate = (s["drops"] - s["drops_carrying"]) / max(1, s["greep_ticks"] - s["carry_ticks"])
        counts = np.bincount(events["color"], minlength=3)
        print(f"   drop rate carrying vs empty-handed: {100 * carrying_rate:.2f} vs "
              f"{100 * empty_rate:.2f} per 100 ticks")
        print("   colours: " + ", ".join(f"{c} {n / counts.sum():.0%}"
                                          for c, n in zip(PAINT_COLORS, counts)))
    print(f"   greeps standing on paint: {s['on_paint'] / max(1, s['greep_ticks']):.0%} "
          f"(on another greep's paint: {s['on_others_paint'] / max(1, s['greep_ticks']):.0%})")

    mi = {k: mutual_information(events["color"], events[k])
          for k in events if k != "color"}
    print("\n3. Speaker consistency: I(colour; writer's situation) in bits")
    if s["drops"]:
        for k, v in sorted(mi.items(), key=lambda kv: -kv[1]):
            print(f"   {k:<10} {v:.3f}")
        # What the colours say: the colour mix in each of the most telling situations.
        print("   colour mix by situation (" + " / ".join(PAINT_COLORS) + "):")
        for k in sorted(mi, key=lambda k: -mi[k])[:3]:
            for value in range(int(events[k].max()) + 1):
                picked = events["color"][events[k] == value]
                if len(picked):
                    mix = np.bincount(picked, minlength=3) / len(picked)
                    print(f"     {k}={value:<3} " + " / ".join(f"{p:4.0%}" for p in mix)
                          + f"   ({len(picked)} drops)")
    else:
        print("   (no drops)")

    reads = max(1, s["on_others_paint"])
    print("\n4. Listener influence: decisions on another greep's paint that the paint changed")
    print(f"   any action {s['changed_any'] / reads:.1%}   " + "   ".join(
        f"{h} {s[f'changed_{h}'] / reads:.1%}" for h in ("mode", "turn", "move", "memory", "spit")))
    print(f"   by colour alone (colours shuffled): any action {s['colour_changed_any'] / reads:.1%}   "
          f"mode {s['colour_changed_mode'] / reads:.1%}   turn {s['colour_changed_turn'] / reads:.1%}")

    save_heatmaps(out / "heatmaps.png", bank, heat)
    report = {
        "policy": label, "maps": bank.names,
        "ablation": {"with_paint": normal.mean(1).tolist(),
                     "scrambled": scrambled.mean(1).tolist(),
                     "colours_shuffled": recoloured.mean(1).tolist(),
                     "blind": blind.mean(1).tolist(),
                     "memory_scrambled": forgetful.mean(1).tolist(),
                     "memory_gain_total": float(memory_gain),
                     "memory_gain_se": float(memory_gain_se),
                     "information_gain_total": float(gain), "gain_se": float(gain_se),
                     "colour_gain_total": float(colour_gain),
                     "colour_gain_se": float(colour_gain_se)},
        "writing": {"drops_per_100_ticks": per100, **stats},
        "speaker_mi_bits": mi,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n5. Heat maps: {out / 'heatmaps.png'}")


if __name__ == "__main__":
    main()
