"""
Validate custom map files (CPU only) and optionally render a preview.

    python -m tools.check_maps maps/training
    python -m tools.check_maps maps/training --preview out/preview.png
    python -m tools.check_maps "maps/training/puzzle_*.map" --preview puzzle.png

Checks each .map: the image is 800x600, the greeps' release point (ship x,
ship y + 30) is on land, every pile can be reached from it through passages
at least --min-gap px wide (default 40, about 1.5 greep bodies: Greenfoot
only tests the pixel a greep's centre steps on, so narrower gaps are
technically passable but greeps rarely find their way through), at most 12
piles, and warns about water thin enough to step over (greeps step 5 px and
only test the pixel they land on). Exits with status 1 if any map has errors.
"""

import argparse
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from greeps.maps import map_files, read_map_file


W, H = 800, 600
MAX_PILES = 12
SPAWN_DY = 30
PILE, SHIP_W, SHIP_H = 33, 110, 56
REACH = PILE // 2 + 12      # a greep touches a pile from this far away (approx.)
MIN_GAP = 40                # narrowest passage a greep should have to use, px


def water_mask(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
    return rgb[..., 2] > 2 * rgb[..., 0], rgb.shape[:2]


def thin_water(water):
    """Water pixels in a horizontal or vertical run shorter than 6 px between land."""
    thin = np.zeros_like(water)
    for axis in (0, 1):
        w = water if axis == 1 else water.T
        t = thin if axis == 1 else thin.T
        for r in range(w.shape[0]):
            row = w[r]
            edges = np.flatnonzero(np.diff(np.r_[0, row.astype(np.int8), 0]))
            for start, end in zip(edges[::2], edges[1::2]):
                if end - start < 6 and start > 0 and end < len(row):
                    t[r, start:end] = True
    return thin


def roomy_reach(water, rx, ry, min_gap):
    """
    Land a greep can reach from (rx, ry) without passing through a gap
    narrower than min_gap: the region of points at least min_gap / 2 from
    water, connected to the release point's surroundings.
    """
    r = min_gap / 2
    roomy = ndimage.distance_transform_edt(~water) >= r
    labels, _ = ndimage.label(roomy)
    k = int(r) + 6
    near = labels[max(0, ry - k):ry + k + 1, max(0, rx - k):rx + k + 1]
    ids = np.unique(near[near > 0])
    return np.isin(labels, ids), len(ids) > 0


def check(path, min_gap=MIN_GAP):
    errors, warnings = [], []
    m = read_map_file(path)
    if not m["image"].exists():
        return m, None, [f"image {m['image'].name} missing"], []
    water, shape = water_mask(m["image"])
    if shape != (H, W):
        return m, None, [f"image is {shape[1]}x{shape[0]}, expected {W}x{H}"], []

    sx, sy = m["ship"]
    rx, ry = sx, sy + SPAWN_DY
    if not (3 <= rx <= W - 4 and 3 <= ry <= H - 4):
        errors.append(f"release point ({rx}, {ry}) is at or beyond the world edge")
        reach = None
    elif water[ry, rx]:
        errors.append(f"release point ({rx}, {ry}) is on water")
        reach = None
    else:
        labels, _ = ndimage.label(~water)          # 4-connected land regions
        reach = labels == labels[ry, rx]

    roomy = None
    if reach is not None and min_gap > 0:
        roomy, open_start = roomy_reach(water, rx, ry, min_gap)
        if not open_start:
            errors.append(f"release point ({rx}, {ry}) is in a pocket narrower than {min_gap} px")
            roomy = None

    if len(m["piles"]) > MAX_PILES:
        errors.append(f"{len(m['piles'])} piles (at most {MAX_PILES})")
    for x, y, n in m["piles"]:
        if not (0 <= x < W and 0 <= y < H):
            errors.append(f"pile at ({x}, {y}) is outside the world")
            continue
        if n < 1:
            errors.append(f"pile at ({x}, {y}) has {n} tomatoes")
        if reach is not None:
            box = reach[max(0, y - REACH):y + REACH + 1, max(0, x - REACH):x + REACH + 1]
            if not box.any():
                errors.append(f"pile at ({x}, {y}) cannot be reached on foot")
            elif roomy is not None:
                k = REACH + min_gap // 2
                if not roomy[max(0, y - k):y + k + 1, max(0, x - k):x + k + 1].any():
                    errors.append(f"pile at ({x}, {y}) is only reachable through a gap "
                                  f"narrower than {min_gap} px")
        if water[y, x]:
            warnings.append(f"pile centre ({x}, {y}) is on water")
    thin = thin_water(water)
    if thin.sum() > 50:
        warnings.append(f"{int(thin.sum())} px of water thin enough to step over")
    return m, water, errors, warnings


def render(path, results):
    tiles = []
    for m, water, errors, _ in results:
        if water is None:
            continue
        img = Image.fromarray(np.where(water[..., None], [40, 70, 160], [235, 215, 190])
                              .astype(np.uint8))
        d = ImageDraw.Draw(img)
        sx, sy = m["ship"]
        d.rectangle([sx - SHIP_W // 2, sy - SHIP_H // 2, sx + SHIP_W // 2, sy + SHIP_H // 2],
                    outline=(0, 0, 0), width=4)
        d.ellipse([sx - 7, sy + SPAWN_DY - 7, sx + 7, sy + SPAWN_DY + 7], fill=(255, 255, 255),
                  outline=(0, 0, 0), width=2)
        for x, y, n in m["piles"]:
            d.rectangle([x - 16, y - 16, x + 16, y + 16], fill=(210, 30, 30))
            d.text((x - 8, y - 6), str(n), fill=(255, 255, 255))
        d.text((8, 582), m["name"] + ("  ERRORS" if errors else ""), fill=(0, 0, 0))
        tiles.append(img.resize((400, 300)))
    if not tiles:
        return
    cols = min(4, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    grid = Image.new("RGB", (cols * 400, rows * 300), "white")
    for k, t in enumerate(tiles):
        grid.paste(t, ((k % cols) * 400, (k // cols) * 300))
    grid.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("maps", help="a directory of .map files, one .map file, "
                                     "or a glob pattern like maps/training/puzzle_*.map")
    parser.add_argument("--preview", help="write a PNG grid of the maps here")
    parser.add_argument("--min-gap", type=int, default=MIN_GAP,
                        help="narrowest passage allowed on the way to piles, px (0: off)")
    args = parser.parse_args()

    if any(c in args.maps for c in "*?["):
        files = sorted(Path().glob(args.maps))
    else:
        files = map_files(args.maps)
    results, bad = [], 0
    for f in files:
        m, water, errors, warnings = check(f, args.min_gap)
        results.append((m, water, errors, warnings))
        status = "ERROR" if errors else ("warn" if warnings else "ok")
        tomatoes = sum(n for _, _, n in m["piles"])
        water_pct = f"{water.mean():5.1%}" if water is not None else "  ?  "
        print(f"{status:5}  {m['name']:<28} water {water_pct}  piles {len(m['piles']):2d}  "
              f"tomatoes {tomatoes:4d}")
        for e in errors:
            print(f"       error: {e}")
        for w in warnings:
            print(f"       warning: {w}")
        bad += bool(errors)
    if args.preview:
        render(args.preview, results)
        print(f"preview: {args.preview}")
    print(f"{len(results)} maps, {bad} with errors")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
