"""
Embed a trained actor into policies/mlp_ppo/Greep.java (and src/Greep.java if
this approach is active), so the submitted Greep needs no extra files.

    python -m policies.mlp_ppo.export --run main      # out/ppo/main/best.pt
    python -m policies.mlp_ppo.export --checkpoint out/ppo/main/latest.pt
"""

import argparse
from pathlib import Path

import torch

from greeps import policies

from . import NAME
from .model import Actor
from .train import RUNS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", default="main", help="export out/ppo/<run>/best.pt")
    parser.add_argument("--checkpoint", help="a specific checkpoint file instead")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else RUNS_DIR / args.run / "best.pt"
    state = torch.load(checkpoint, map_location="cpu")
    actor = Actor()
    actor.load_state_dict(state["actor"])

    note = f"{checkpoint.parent.name}/{checkpoint.name}: update {state.get('update', '?')}"
    if "eval" in state:
        note += ", GPU-sim eval " + " / ".join(f"{s:.1f}" for s in state["eval"])
    print(policies.embed_weights(NAME, actor.export_params(), "<f4", note))


if __name__ == "__main__":
    main()
