"""
Evaluate the active Greep (src/Greep.java) in the real (Java) scenario: its
embedded policy, or policy.weights if that file exists.

    python -m tools.evaluate                  # 64 headless seeds, scenario maps
    python -m tools.evaluate --maps maps/validation  # custom maps (tools/map_editor.py)
    python -m tools.evaluate --gui            # also watch one run in Greenfoot
"""

import argparse

import numpy as np

from greeps.java_sim import (
    HeadlessPool,
    compile_scenario,
    custom_maps_prop,
    read_policy,
    run_gui_episode,
)
from greeps.maps import map_files, read_map_file
from greeps.paths import POLICY_FILE


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--maps", help="a directory of .map files (or one .map file) "
                                       "to play instead of the scenario's maps")
    parser.add_argument("--paint-blind", action="store_true",
                        help="greeps ignore paint (does paint carry information?)")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    props = custom_maps_prop(args.maps) if args.maps else {}
    if args.paint_blind:
        props["greeps.paintBlind"] = "true"
    if args.maps:
        labels = [read_map_file(f)["name"] for f in map_files(args.maps)]
    else:
        labels = ["Map 1", "Map 2", "Map 3"]

    compile_scenario()
    with HeadlessPool(workers=args.workers, props=props) as pool:
        if POLICY_FILE.exists():
            weights, name = read_policy(), f"{pool.policy_name} with {POLICY_FILE.name}"
        else:
            weights, name = pool.default_weights(), f"{pool.policy_name} (embedded weights)"
        if args.paint_blind:
            name += ", paint-blind"
        # Seeds unused by any trainer's validation.
        seeds = [2_000_000 + i for i in range(args.seeds)]
        scores = pool.evaluate([weights], seeds)[0]

    means = scores.mean(axis=0)
    stderr = scores.std(axis=0, ddof=1) / np.sqrt(len(seeds))
    width = max(len(label) for label in labels + ["Total"])
    print(f"{name} over {len(seeds)} seeds (mean +- standard error):")
    for label, m, s in zip(labels, means, stderr):
        print(f"  {label:<{width}}  {m:6.2f} +- {s:.2f}")
    print(f"  {'Total':<{width}}  {scores.sum(axis=1).mean():6.2f}")

    if args.gui:
        print("GUI run:", run_gui_episode(props=props, expected_maps=len(labels)))


if __name__ == "__main__":
    main()
