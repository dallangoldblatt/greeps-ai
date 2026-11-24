"""
Generate randomized variants of every training-map family and validate them.

Each script in maps/training/generators/ is a family of hand-designed map
ideas. Run without arguments it rewrites its original maps in maps/training/;
with --variants it writes that many randomized versions of each idea. This
tool runs all of them into maps/training/generated/ and checks the result.

    python -m tools.generate_training_maps --variants 20
    python -m tools.generate_training_maps --variants 20 --seed 1 --clean

Training picks the variants up automatically: --custom-maps maps/training
includes subfolders. Variants of the templates held out in maps/validation
are moved there.
"""

import argparse
import shutil
import subprocess
import sys

from greeps.paths import SCENARIO_DIR


GENERATORS = SCENARIO_DIR / "maps" / "training" / "generators"
OUT = SCENARIO_DIR / "maps" / "training" / "generated"
VALIDATION = SCENARIO_DIR / "maps" / "validation"


def move_map(map_file, target_dir):
    """Move a .map file and the image it names."""
    image = next(line.split(None, 1)[1].strip() for line in map_file.read_text().splitlines()
                 if line.startswith("image "))
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in (map_file, map_file.parent / image):
        shutil.move(str(path), str(target_dir / path.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variants", type=int, default=20, help="per map idea")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true", help="delete old variants first")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for f in list(OUT.glob("*.map")) + list(OUT.glob("*.png")):
            f.unlink()
    for script in sorted(GENERATORS.glob("*.py")):
        print(f"{script.name}: {args.variants} variants per idea ...", flush=True)
        subprocess.run([sys.executable, str(script), "--variants", str(args.variants),
                        "--seed", str(args.seed), "--out", str(OUT)],
                       cwd=SCENARIO_DIR, check=True)
    # Variants of held-out templates (the maps in maps/validation) belong to
    # the validation set, never to training.
    for template in sorted(VALIDATION.glob("*.map")):
        for variant in sorted(OUT.glob(f"{template.stem}_v*.map")):
            target = VALIDATION / "generated" / variant.name
            if target.exists():
                target.unlink()
                (target.parent / target.with_suffix(".png").name).unlink(missing_ok=True)
            move_map(variant, VALIDATION / "generated")
    print("Validating ...", flush=True)
    result = subprocess.run([sys.executable, "-m", "tools.check_maps", str(OUT)],
                            cwd=SCENARIO_DIR, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    for line in lines:
        if line.startswith("ERROR") or "error:" in line:
            print(line)
    print(lines[-1] if lines else result.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
