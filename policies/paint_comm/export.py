"""
Embed a trained actor into policies/paint_comm/Greep.java (and src/Greep.java
if this approach is active).

    python -m policies.paint_comm.export --run main              # best score
    python -m policies.paint_comm.export --run main --comm       # best info gain
    python -m policies.paint_comm.export --checkpoint out/paint/main/latest.pt
"""

import argparse
from pathlib import Path

import torch

from greeps import policies
from policies.mlp_ppo.model import load_actor

from . import NAME
from .train import RUNS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", default="main")
    parser.add_argument("--comm", action="store_true",
                        help="export best_comm.pt (largest information gain from paint)")
    parser.add_argument("--checkpoint", help="a specific checkpoint file instead")
    args = parser.parse_args()

    name = "best_comm.pt" if args.comm else "best.pt"
    checkpoint = Path(args.checkpoint) if args.checkpoint else RUNS_DIR / args.run / name
    state = torch.load(checkpoint, map_location="cpu")
    actor = load_actor(state["actor"])
    if not actor.java_compatible:
        raise SystemExit(f"{checkpoint} has a larger actor (hidden {actor.hidden}, "
                         f"{actor.num_blocks} blocks) than Greep.java runs (64, 0)")
    note = f"{checkpoint.parent.name}/{checkpoint.name}: update {state.get('update', '?')}"
    if "eval" in state and "eval_scrambled" in state:
        e, s = state["eval"], state["eval_scrambled"]
        note += f", GPU-sim total {sum(e):.1f} (scrambled paint {sum(s):.1f})"
    print(policies.embed_weights(NAME, actor.export_params(), "<f4", note))


if __name__ == "__main__":
    main()
