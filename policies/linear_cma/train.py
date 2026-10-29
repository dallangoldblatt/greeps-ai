"""
Train the linear Greep (policies/linear_cma/Greep.java) with CMA-ES in the
real (Java) scenario.

Black-box optimization: each candidate is a whole weight vector, frozen for
an episode, and its fitness is simply the episode score. No gradients, no
per-step credit assignment.

Each generation every candidate plays the same freshly drawn seeds (common
random numbers, so candidates are compared on equal footing). Fitness is the
mean score over Maps 1 + 2. Map 3 is never used for selection; it is only
reported, as a held-out map.

Every --val-every generations the CMA-ES mean is evaluated on a fixed set of
validation seeds; the best one so far is saved to out/cma/<run>/best.weights.
Bake it into Greep.java with `python -m policies.linear_cma.export --run <run>`.

    python -m tools.use_policy linear_cma        # the trainer plays src/Greep.java
    python -m policies.linear_cma.train --run main --generations 200
    python -m policies.linear_cma.train --run main --resume --generations 400
"""

import argparse
import csv
import json
import pickle
import time

import cma
import numpy as np

from greeps import policies
from greeps.java_sim import HeadlessPool, compile_scenario, write_policy
from greeps.paths import OUT_DIR

from . import NAME


RUNS_DIR = OUT_DIR / "cma"
TRAIN_MAPS = slice(0, 2)    # Maps 1 and 2
HELD_OUT_MAP = 2            # Map 3
VALIDATION_SEED_BASE = 1_000_000   # far away from the training seeds


def train_fitness(scores):
    """Mean over seeds of the Maps 1 + 2 score. scores: (..., seeds, maps)."""
    return scores[..., TRAIN_MAPS].sum(axis=-1).mean(axis=-1)


def summarize(scores):
    """Per-map means and the training score for one policy's (seeds, maps)."""
    means = scores.mean(axis=0)
    return {
        "train_score": float(train_fitness(scores)),
        "map1_score": float(means[0]),
        "map2_score": float(means[1]),
        "map3_score": float(means[HELD_OUT_MAP]),
    }


def train(args):
    if policies.active_name() != NAME:
        raise SystemExit(f"src/Greep.java plays {policies.active_name()!r}; "
                         f"run `python -m tools.use_policy {NAME}` first.")
    run_dir = RUNS_DIR / args.run
    checkpoint, log_file = run_dir / "checkpoint.pkl", run_dir / "log.csv"
    if checkpoint.exists() and not args.resume:
        raise SystemExit(f"{run_dir} already has a run; use --resume or another --run name")
    run_dir.mkdir(parents=True, exist_ok=True)
    compile_scenario()

    with HeadlessPool(workers=args.workers) as pool:
        val_seeds = [VALIDATION_SEED_BASE + i for i in range(args.val_seeds)]

        if args.resume:
            with open(checkpoint, "rb") as f:
                state = pickle.load(f)
            es, best_val = state["es"], state["best_val"]
            print(f"Resumed at generation {es.countiter}, "
                  f"best validation {best_val:.2f}")
        else:
            # The hand-written policy, or the embedded trained one.
            x0 = pool.default_weights() if args.from_trained else pool.initial_weights()
            # Per-weight initial standard deviations: sigma * scale.
            options = {
                "seed": args.seed,
                "verbose": -9,
                "CMA_stds": pool.search_scales().tolist(),
            }
            if args.popsize:
                options["popsize"] = args.popsize
            es = cma.CMAEvolutionStrategy(x0, args.sigma, options)
            baseline = summarize(pool.evaluate([x0], val_seeds)[0])
            best_val = float("-inf")
            print(f"{pool.num_params} parameters, population {es.popsize}, "
                  f"{args.seeds} seeds per candidate")
            print("Starting point: "
                  f"train={baseline['train_score']:.2f} "
                  f"held-out Map 3={baseline['map3_score']:.2f}")

        new_log = not (args.resume and log_file.exists())
        log = open(log_file, "w" if new_log else "a", newline="")
        writer = csv.writer(log)
        if new_log:
            writer.writerow([
                "generation", "sigma", "pop_mean", "pop_best",
                "val_train", "val_map1", "val_map2", "val_map3", "seconds",
            ])

        start = time.monotonic()
        try:
            while es.countiter < args.generations:
                generation = es.countiter
                seeds = [generation * args.seeds + i for i in range(args.seeds)]

                candidates = es.ask()
                fitness = train_fitness(pool.evaluate(candidates, seeds))
                es.tell(candidates, (-fitness).tolist())   # CMA minimizes

                row = [
                    generation + 1, f"{es.sigma:.4f}",
                    f"{fitness.mean():.2f}", f"{fitness.max():.2f}",
                    "", "", "", "",
                ]
                message = (f"gen {generation + 1:4d}  sigma {es.sigma:.3f}  "
                           f"pop mean {fitness.mean():6.2f}  "
                           f"best {fitness.max():6.2f}")

                last = es.countiter >= args.generations
                if es.countiter % args.val_every == 0 or last:
                    mean = es.mean.copy()
                    val = summarize(pool.evaluate([mean], val_seeds)[0])
                    row[4:8] = [f"{val[k]:.2f}" for k in (
                        "train_score", "map1_score", "map2_score", "map3_score")]
                    message += (f"  | val train {val['train_score']:6.2f}  "
                                f"held-out Map 3 {val['map3_score']:5.2f}")

                    # Select on the training maps only.
                    if val["train_score"] > best_val:
                        best_val = val["train_score"]
                        save_best(run_dir, mean, val, es.countiter, args)
                        message += "  *saved*"

                    with open(checkpoint, "wb") as f:
                        pickle.dump({"es": es, "best_val": best_val}, f)

                row.append(f"{time.monotonic() - start:.1f}")
                writer.writerow(row)
                log.flush()
                print(message, flush=True)
        finally:
            log.close()

    print(f"\nBest validation training score (Maps 1 + 2): {best_val:.2f}")
    print(f"Saved: {run_dir / 'best.weights'}  "
          f"(embed with: python -m policies.linear_cma.export --run {args.run})")


def save_best(run_dir, weights, val, generation, args):
    write_policy(
        weights, run_dir / "best.weights",
        comment=(f"CMA-ES generation {generation}, validation train score "
                 f"{val['train_score']:.2f} over {args.val_seeds} seeds"),
    )
    (run_dir / "best.json").write_text(
        json.dumps({"generation": generation, "validation_seeds": args.val_seeds, **val},
                   indent=2),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", default="main", help="run name: files go to out/cma/<run>/")
    parser.add_argument("--generations", type=int, default=200,
                        help="total generations to run (including resumed ones)")
    parser.add_argument("--popsize", type=int, default=None,
                        help="CMA-ES population (default: 4 + 3 ln(n))")
    parser.add_argument("--seeds", type=int, default=4,
                        help="episodes per candidate per generation")
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="initial CMA-ES step size (times Greep's "
                             "per-weight search scales)")
    parser.add_argument("--from-trained", action="store_true",
                        help="start from the embedded trained weights instead of "
                             "the hand-written policy")
    parser.add_argument("--workers", type=int, default=14,
                        help="headless simulator processes")
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--val-seeds", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123,
                        help="CMA-ES random seed")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
