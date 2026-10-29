"""
Turn screenshots of the Greenfoot window into map files for the editor, the
GPU simulator and the Java sim.

    python -m tools.import_screenshots "screenshots/*.jpg" --out maps/imported

For each screenshot the world (800x600, drawn at about half size) is located,
then:
  * water: blue > 1.5 * red (on Maps 1-3 this matches Greenfoot's own masks
    with 96-98% IoU); pixels hidden under the title, ship and tomatoes are
    filled in from their surroundings;
  * piles: clusters of red tomato pixels; the tomato count is estimated from
    the cluster's red area, calibrated on Maps 1-3 whose counts are known;
  * ship: its x from the ship image at the top. Its landing height is NOT
    visible (the screenshots show the ship before it descends), so it is
    estimated as the point below the ship with the most room around the
    greeps' release point. Check it in the editor.

Screenshots titled "Map 1".."Map 3" (the scenario's own maps) are written
with the exact scenario data instead.
"""

import argparse
from pathlib import Path
import re

import numpy as np
from PIL import Image
from scipy import ndimage

from greeps.java_sim import WATER_DIR
from greeps.maps import REAL_MAPS, write_map_file


W, H = 800, 600
# The world inside the screenshot: detected bounding box plus a sub-pixel
# offset and scale fitted against the exact masks of Maps 1-3.
OFFSET_X, OFFSET_Y, SCALE = 0.8, 0.3, 0.506
WATER_RGB, LAND_RGB = (40, 70, 160), (235, 215, 190)


def locate_world(rgb):
    """Top-left corner of the sand/water area in the screenshot."""
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    world = ((r > 200) & (g > 170) & (b > 140) & (r - b > 25)) | (b > r + 40)
    # Rows/columns that are mostly world, within the largest such block
    # (the slide around the window has other colours).
    rows = np.where(world.mean(1) > 0.3)[0]
    rows = max(np.split(rows, np.where(np.diff(rows) > 1)[0] + 1), key=len)
    cols = np.where(world[rows].mean(0) > 0.5)[0]
    cols = max(np.split(cols, np.where(np.diff(cols) > 1)[0] + 1), key=len)
    return cols.min() + OFFSET_X, rows.min() + OFFSET_Y


def world_image(path):
    """The world region resampled to 800x600 (RGB int array), and the box."""
    img = Image.open(path).convert("RGB")
    x0, y0 = locate_world(np.asarray(img))
    box = (x0, y0, x0 + W * SCALE, y0 + H * SCALE)
    world = img.transform((W, H), Image.EXTENT, box, Image.BICUBIC)
    return np.asarray(world).astype(int), np.asarray(img).astype(int), box


def classify_water(world, ship_x):
    r, g, b = world[..., 0], world[..., 1], world[..., 2]
    water = b > 1.5 * r
    sand = (r > 170) & (r - b > 15) & ~water
    unknown = ~water & ~sand            # text, tomatoes, frame
    # Overlays drawn on top of the map: the ship image (it has bluish
    # glass) at the top, and the "<author> - Map N" title around (200, 20).
    unknown[0:30, max(0, ship_x - 58):ship_x + 58] = True
    unknown[6:34, 95:310] = True
    # Each unknown pixel takes the value of the nearest known pixel.
    _, (iy, ix) = ndimage.distance_transform_edt(unknown, return_indices=True)
    water = water[iy, ix]
    # Remove specks: tiny water blobs and tiny land holes.
    labels, n = ndimage.label(water)
    sizes = ndimage.sum(water, labels, range(1, n + 1))
    water &= ~np.isin(labels, np.where(sizes < 60)[0] + 1)
    labels, n = ndimage.label(~water)
    sizes = ndimage.sum(~water, labels, range(1, n + 1))
    water |= np.isin(labels, np.where(sizes < 60)[0] + 1)
    return water


def find_piles(shot, box):
    """Red clusters -> [(x, y, red_pixels)] in world coordinates."""
    x0, y0, x1, y1 = box
    r, g, b = shot[..., 0], shot[..., 1], shot[..., 2]
    # Single tomatoes blur into pink against the sand at half size.
    red = (r > 100) & (r - g > 45) & (r - b > 45)
    region = np.zeros_like(red)
    region[int(y0) + 2:int(y1) - 1, int(x0) + 2:int(x1) - 1] = True
    red &= region
    labels, n = ndimage.label(ndimage.binary_dilation(red, iterations=4))
    piles = []
    for i in range(1, n + 1):
        ys, xs = np.where(red & (labels == i))
        if len(xs) < 3:
            continue
        wx = (xs.mean() - x0) / SCALE
        wy = (ys.mean() - y0) / SCALE
        piles.append((int(round(wx)), int(round(wy)), len(xs)))
    return piles


def find_ship_x(world):
    """Grey, unsaturated pixels of the ship image along the top of the world."""
    strip = world[:40]
    r, g, b = strip[..., 0], strip[..., 1], strip[..., 2]
    spread = np.max(strip, -1) - np.min(strip, -1)
    grey = (spread < 25) & (r > 90) & (r < 235)
    cols = grey.sum(0).astype(float)
    if cols.sum() == 0:
        raise ValueError("ship not found")
    # The ship image is 110 px wide: centre of the best-matching window.
    window = np.convolve(cols, np.ones(110), mode="same")
    return int(np.argmax(window))


def estimate_landing(water, x):
    """
    A guess at the landing y: the ship over land (so greeps can reach it from
    all sides), with room around the release point (x, y + 30), not too far
    from the middle.
    """
    clearance = ndimage.distance_transform_edt(~water)
    x0, x1 = max(0, x - 55), min(W, x + 55)
    best, best_y = -1e9, 300
    for y in range(60, H - 60):
        sy = y + 30
        if water[sy, x]:
            continue
        land = 1 - water[max(0, y - 28):y + 28, x0:x1].mean()
        score = min(clearance[sy, x], 60) + 60 * land - abs(y - 300) * 0.05
        if score > best:
            best, best_y = score, y
    return best_y


def map_number(path):
    m = re.search(r"(\d+)", Path(path).stem)
    return int(m.group(1)) if m else None


def calibrate(paths):
    """Fit red pixels per tomato on the scenario maps in `paths`."""
    pairs = []
    for p in paths:
        n = map_number(p)
        if n not in (1, 2, 3):
            continue
        world, shot, box = world_image(p)
        for x, y, red in find_piles(shot, box):
            true = min(REAL_MAPS[n - 1]["piles"], key=lambda q: (q[0] - x) ** 2 + (q[1] - y) ** 2)
            if (true[0] - x) ** 2 + (true[1] - y) ** 2 < 30 ** 2:
                pairs.append((red, true[2]))
    if not pairs:
        return 1.6, []
    red, count = np.array(pairs, dtype=float).T
    return float((red * count).sum() / (count * count).sum()), pairs


def save_png(path, water):
    img = np.where(water[..., None], WATER_RGB, LAND_RGB).astype(np.uint8)
    Image.fromarray(img).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("screenshots", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefix", default="map")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted({str(p) for s in args.screenshots for p in Path().glob(s)} or set(args.screenshots),
                   key=lambda p: map_number(p) or 0)

    per_tomato, pairs = calibrate(paths)
    print(f"Calibration: {per_tomato:.2f} red pixels per tomato from {len(pairs)} known piles")

    for p in paths:
        n = map_number(p)
        name = f"{args.prefix}{n:02d}"
        if n in (1, 2, 3):
            real = REAL_MAPS[n - 1]
            water = np.fromfile(WATER_DIR / f"water{n}.bin", dtype=np.uint8).reshape(H, W).astype(bool)
            save_png(out / f"{name}.png", water)
            write_map_file(out / f"{name}.map", f"{name}.png", real["ship"], real["piles"])
            print(f"{name}: scenario map {n} (exact data)")
            continue
        world, shot, box = world_image(p)
        ship_x = find_ship_x(world)
        water = classify_water(world, ship_x)
        ship_y = estimate_landing(water, ship_x)
        piles = [(x, y, max(1, int(round(red / per_tomato))))
                 for x, y, red in find_piles(shot, box)]
        save_png(out / f"{name}.png", water)
        write_map_file(out / f"{name}.map", f"{name}.png", (ship_x, ship_y), piles)
        text = (out / f"{name}.map").read_text(encoding="utf-8")
        (out / f"{name}.map").write_text(
            f"# Imported from {Path(p).name}. Estimated: ship landing y and tomato counts.\n" + text,
            encoding="utf-8")
        print(f"{name}: water {water.mean():.1%}, ship x {ship_x} (landing y {ship_y} estimated), "
              f"piles {[(x, y, c) for x, y, c in piles]}")


if __name__ == "__main__":
    main()
