"""
Hold out whole map templates as a validation set.

    python -m tools.split_validation            # 3 templates per family, seed 0
    python -m tools.split_validation --dry-run

Moves maps/training/<family>_<NN>.map (with its image) and all its variants
maps/training/generated/<family>_<NN>_vXXX.map to maps/validation/ (and
maps/validation/generated/). Variants share their template's layout, so
splitting by template keeps the validation maps genuinely unseen.

Validation maps choose checkpoints and compare training methods.
"""

import argparse
from collections import defaultdict
import random
import re
import shutil

from greeps.paths import SCENARIO_DIR


TRAINING = SCENARIO_DIR / "maps" / "training"
VALIDATION = SCENARIO_DIR / "maps" / "validation"


def move_map(map_file, target_dir, dry_run):
    """Move a .map file and the image it names."""
    image = next(line.split(None, 1)[1].strip() for line in map_file.read_text().splitlines()
                 if line.startswith("image "))
    for path in (map_file, map_file.parent / image):
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target_dir / path.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--per-family", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    families = defaultdict(list)
    for f in sorted(TRAINING.glob("*.map")):
        match = re.fullmatch(r"(.+)_(\d+)", f.stem)
        if match:
            families[match.group(1)].append(f)
    rng = random.Random(args.seed)
    for family, templates in sorted(families.items()):
        held = sorted(rng.sample(templates, args.per_family))
        for template in held:
            variants = sorted((TRAINING / "generated").glob(f"{template.stem}_v*.map"))
            print(f"{template.stem}: template + {len(variants)} variants")
            move_map(template, VALIDATION, args.dry_run)
            for v in variants:
                move_map(v, VALIDATION / "generated", args.dry_run)
    if args.dry_run:
        print("(dry run: nothing moved)")


if __name__ == "__main__":
    main()
