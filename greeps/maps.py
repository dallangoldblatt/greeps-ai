"""
Map definitions: the scenario's three maps and custom map files.

A custom map (written by tools/map_editor.py or tools/import_screenshots.py)
is a .map text file next to an 800x600 image:

    image <file>.png        # water where blue > 2 * red (Earth.isWater)
    ship <x> <y>            # where the ship lands
    pile <tomatoes> <x> <y>  # one or more
"""

from pathlib import Path

from .paths import SCENARIO_DIR


CUSTOM_MAP_DIR = SCENARIO_DIR / "maps"

# Earth.mapData: ship (x, landing y), then piles (x, y, tomatoes).
REAL_MAPS = [
    {"name": "map1", "ship": (100, 480),
     "piles": [(721, 532, 40), (400, 560, 12), (615, 400, 40),
               (642, 192, 40), (128, 113, 16), (400, 40, 30)]},
    {"name": "map2", "ship": (709, 496),
     "piles": [(322, 422, 10), (700, 241, 40), (681, 49, 40),
               (317, 54, 10), (90, 174, 50), (60, 339, 40)]},
    {"name": "map3", "ship": (394, 272),
     "piles": [(39, 30, 10), (71, 476, 30), (398, 520, 50), (655, 492, 40)]},
]


def read_map_file(path):
    """Parse a .map file. Returns {"name", "image" (absolute Path), "ship", "piles"}."""
    path = Path(path)
    result = {"name": path.stem, "image": None, "ship": None, "piles": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        if parts[0] == "image":
            result["image"] = path.parent / " ".join(parts[1:])
        elif parts[0] == "ship":
            result["ship"] = (int(parts[1]), int(parts[2]))
        elif parts[0] == "pile":
            n, x, y = map(int, parts[1:4])
            result["piles"].append((x, y, n))
        else:
            raise ValueError(f"{path}: unknown line {line!r}")
    if result["image"] is None or result["ship"] is None or not result["piles"]:
        raise ValueError(f"{path}: needs an image, a ship and at least one pile")
    return result


def write_map_file(path, image_name, ship, piles):
    path = Path(path)
    lines = ["# Greeps map (tools/map_editor.py)", f"image {image_name}",
             f"ship {int(ship[0])} {int(ship[1])}"]
    lines += [f"pile {int(n)} {int(x)} {int(y)}" for x, y, n in piles]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def map_files(spec):
    """
    A directory (all *.map in it and its subfolders, sorted), a glob pattern
    such as "maps/training/**/signal_*.map", or a list of .map files.
    """
    if isinstance(spec, str) and any(c in spec for c in "*?["):
        return sorted(Path().glob(spec))
    if isinstance(spec, (str, Path)) and Path(spec).is_dir():
        return sorted(Path(spec).rglob("*.map"))
    if isinstance(spec, (str, Path)):
        return [Path(spec)]
    return [Path(p) for p in spec]
