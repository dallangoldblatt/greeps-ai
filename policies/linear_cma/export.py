"""
Embed a CMA-ES result into policies/linear_cma/Greep.java (and src/Greep.java
if this approach is active), so the submitted Greep needs no extra files.

    python -m policies.linear_cma.export --run main      # out/cma/main/best.weights
    python -m policies.linear_cma.export --weights some.weights
"""

import argparse
import json

from greeps import policies
from greeps.java_sim import read_policy

from . import NAME, NUM_PARAMS
from .train import RUNS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", default="main")
    parser.add_argument("--weights", help="a weights file instead of the run's best")
    args = parser.parse_args()

    path = args.weights or RUNS_DIR / args.run / "best.weights"
    weights = read_policy(path)
    if len(weights) != NUM_PARAMS:
        raise SystemExit(f"{path} has {len(weights)} weights, expected {NUM_PARAMS}")
    note = f"from {path}"
    info = RUNS_DIR / args.run / "best.json"
    if not args.weights and info.exists():
        v = json.loads(info.read_text(encoding="utf-8"))
        note = (f"CMA-ES run {args.run}, generation {v['generation']}, validation "
                f"{v['map1_score']:.1f} / {v['map2_score']:.1f} / {v['map3_score']:.1f}")
    # Doubles: the policy's thresholds (z > 0) are exact as trained.
    print(policies.embed_weights(NAME, weights, "<f8", note))


if __name__ == "__main__":
    main()
