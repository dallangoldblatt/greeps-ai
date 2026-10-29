"""Generate 'terrain_XX' training maps (natural terrain theme). Run from repo root:

    python maps/training/generators/terrain.py
        rewrites the 15 hand-designed originals maps/training/terrain_01..15

    python maps/training/generators/terrain.py --variants N --seed S --out DIR
        writes N randomized variants of each of the 15 ideas to
        DIR/terrain_NN_vKKK.png/.map (deterministic given seed, idea, variant);
        every variant is validated with tools.check_maps and resampled until clean.
"""
import argparse
import math
import os
import sys
from collections import deque
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

W, H = 800, 600
OUT = os.path.join("maps", "training")
WATER = (40, 70, 160)
LAND = (235, 215, 190)


class Reject(Exception):
    """A randomly drawn variant is unusable; resample."""


class Ctx:
    def __init__(self, nn, out, var=False, seed=0, k=0, attempt=0):
        self.nn, self.out, self.var = nn, out, var
        self.rng = np.random.default_rng([seed, nn, k, attempt])
        self.name = "terrain_%02d" % nn + ("_v%03d" % k if var else "")
        self.verbose = not var


# ---------------------------------------------------------------- helpers
class Canvas:
    def __init__(self, land=True):
        self.img = Image.new("L", (W, H), 255 if land else 0)
        self.d = ImageDraw.Draw(self.img)

    def line(self, pts, w, val):
        pts = [tuple(p) for p in pts]
        for a, b in zip(pts, pts[1:]):
            self.d.line([a, b], fill=val, width=int(w))
        r = w / 2
        for x, y in pts:
            self.d.ellipse([x - r, y - r, x + r, y + r], fill=val)

    def ell(self, cx, cy, rx, ry, val):
        self.d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=val)

    def rell(self, cx, cy, rx, ry, ang, val):
        t = np.linspace(0, 2 * np.pi, 90, endpoint=False)
        x = rx * np.cos(t)
        y = ry * np.sin(t)
        ca, sa = math.cos(ang), math.sin(ang)
        self.poly(list(zip(cx + x * ca - y * sa, cy + x * sa + y * ca)), val)

    def poly(self, pts, val):
        self.d.polygon([tuple(p) for p in pts], fill=val)

    def mask(self):
        return np.array(self.img) > 127


def smooth(pts, n=4):
    pts = np.array(pts, float)
    for _ in range(n):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        new = np.empty((len(q) * 2, 2))
        new[0::2] = q
        new[1::2] = r
        pts = np.vstack([pts[:1], new, pts[-1:]])
    return pts


def noise(seed, sigma):
    rng = np.random.default_rng(seed)
    n = ndi.gaussian_filter(rng.standard_normal((H, W)), sigma)
    return (n - n.mean()) / n.std()


def organic(land, seed, amp=0.18, blur=7, sigma=20):
    f = ndi.gaussian_filter(land.astype(float), blur) + amp * noise(seed, sigma)
    return f > 0.5


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def cleanup(land, land_r=18, water_r=9):
    land = ndi.binary_opening(land, disk(land_r))
    water = ~land
    water = ndi.binary_opening(water, disk(water_r))  # drop thin water slivers
    # thin water lines are removed from water, but thick ones stay
    return ~water


def component(land, pt):
    lab, _ = ndi.label(land)
    k = lab[int(pt[1]), int(pt[0])]
    return lab == k if k else np.zeros_like(land)


def snap(good, x, y):
    ys, xs = np.nonzero(good)
    if len(xs) == 0:
        raise Reject("no good land")
    i = np.argmin((xs - x) ** 2 + (ys - y) ** 2)
    return int(xs[i]), int(ys[i])


def geodesic(good, start, step=4):
    g = good[::step, ::step]
    h, w = g.shape
    dist = -np.ones((h, w), int)
    sx, sy = start[0] // step, start[1] // step
    # snap start to a good cell
    ys, xs = np.nonzero(g)
    i = np.argmin((xs - sx) ** 2 + (ys - sy) ** 2)
    sx, sy = xs[i], ys[i]
    dist[sy, sx] = 0
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and g[ny, nx] and dist[ny, nx] < 0:
                    dist[ny, nx] = dist[y, x] + 1
                    q.append((nx, ny))
    return dist


def auto_piles(good, rel, specs, seed, spacing=140, step=4):
    """specs: list of (fraction of max geodesic distance, tomatoes)."""
    rng = np.random.default_rng(seed)
    dist = geodesic(good, rel, step)
    mx = dist.max()
    chosen = []
    out = []
    for frac, tom in specs:
        target = frac * mx
        ys, xs = np.nonzero(dist >= 0)
        score = np.abs(dist[ys, xs] - target) + rng.random(len(xs)) * 3
        for j in np.argsort(score):
            px, py = xs[j] * step, ys[j] * step
            if all((px - cx) ** 2 + (py - cy) ** 2 > spacing ** 2 for cx, cy in chosen):
                chosen.append((px, py))
                out.append((tom, px, py))
                break
    return out


def fill_thin(land, minrun=7, iters=3):
    for _ in range(iters):
        for axis in (0, 1):
            w = ~land if axis == 1 else (~land).T
            l = land if axis == 1 else land.T
            for r in range(w.shape[0]):
                row = w[r]
                e = np.flatnonzero(np.diff(np.r_[0, row.astype(np.int8), 0]))
                for a, b in zip(e[::2], e[1::2]):
                    if b - a < minrun and a > 0 and b < len(row):
                        l[r, a:b] = True
    return land


def finish(c, desc, land, ship, piles, auto=None, seed=0, clean=False, margin=20, land_r=18):
    """ship=(x,y) wanted; piles=[(tom,x,y)] manual (snapped to land)."""
    rng = c.rng
    if clean:
        land = cleanup(land, land_r)
        lab, nl = ndi.label(land)
        if nl == 0:
            raise Reject("no land")
        big = np.argmax(ndi.sum(land, lab, range(1, nl + 1))) + 1
        land = lab == big
    land = fill_thin(land.copy())
    rel = (ship[0], ship[1] + 30)
    rel = (min(max(rel[0], 4), W - 5), min(max(rel[1], 4), H - 5))
    edt = ndi.distance_transform_edt(land)
    good0 = land & (edt >= 14)
    rx, ry = snap(good0, *rel)
    comp = component(land, (rx, ry))
    land = comp | (land & False)  # drop disconnected land
    edt = ndi.distance_transform_edt(land)
    good = land & (edt >= margin)
    plist = []
    for t, x, y in piles:
        if c.var:
            x = min(max(x + rng.uniform(-35, 35), 0), W - 1)
            y = min(max(y + rng.uniform(-35, 35), 0), H - 1)
        px, py = snap(good, x, y)
        plist.append((t, px, py))
    if auto:
        s = int(rng.integers(1, 10 ** 6)) if c.var else seed
        plist += auto_piles(good, (rx, ry), auto, s)

    fx = fy = False
    if c.var:
        # normalise the tomato total, keep it inside 5..60 per pile
        tot = sum(p[0] for p in plist)
        if tot > 210 or tot < 90:
            k = rng.uniform(115, 190) / tot
            plist = [(int(min(60, max(5, round(t * k)))), x, y) for t, x, y in plist]
        # piles must be apart from each other and from the ship
        for i, (_, x, y) in enumerate(plist):
            for _, x2, y2 in plist[:i]:
                if (x - x2) ** 2 + (y - y2) ** 2 < 70 ** 2:
                    raise Reject("piles too close")
            if abs(x - rx) < 85 and abs(y - (ry - 30)) < 60 and c.nn != 15:
                raise Reject("pile under ship")
        if c.nn in (4, 11, 15) and land.mean() < 0.42:
            raise Reject("too much water")
        if not 3 <= len(plist) <= 10 or not 80 <= sum(p[0] for p in plist) <= 220:
            raise Reject("pile counts")
        fx, fy = bool(rng.random() < 0.5), bool(rng.random() < 0.5)
        if fx:
            land = land[:, ::-1]
            rx = W - 1 - rx
            plist = [(t, W - 1 - x, y) for t, x, y in plist]
        if fy:
            land = land[::-1, :]
            ry = H - 1 - ry
            plist = [(t, x, H - 1 - y) for t, x, y in plist]
    ship = (rx, ry - 30)
    if c.var and not (58 <= ship[0] <= W - 58 and 30 <= ship[1] <= H - 30):
        raise Reject("ship outside")

    arr = np.zeros((H, W, 3), np.uint8)
    arr[:] = WATER
    arr[land] = LAND
    Image.fromarray(arr).save(os.path.join(c.out, c.name + ".png"))
    with open(os.path.join(c.out, c.name + ".map"), "w", encoding="utf-8") as f:
        f.write("# Greeps map: %s\n" % desc)
        f.write("image %s.png\nship %d %d\n" % (c.name, ship[0], ship[1]))
        for t, x, y in plist:
            f.write("pile %d %d %d\n" % (t, x, y))
    if c.verbose:
        print(c.name, "ship", ship, "piles", len(plist), "tomatoes", sum(p[0] for p in plist))


def wavy_v(x0, amp, per, ph, y0=-30, y1=630, step=30):
    ys = np.arange(y0, y1 + 1, step)
    return [(x0 + amp * np.sin(y / per + ph), y) for y in ys]


def wavy_h(y0, amp, per, ph, x0=-30, x1=830, step=30):
    xs = np.arange(x0, x1 + 1, step)
    return [(x, y0 + amp * np.sin(x / per + ph)) for x in xs]


def ford_h(cv, pts, y, half=75, w=55):
    """land strip crossing a (vertical-ish) river at height y"""
    pts = np.array(pts)
    x = np.interp(y, pts[:, 1], pts[:, 0])
    cv.line([(x - half, y), (x + half, y)], w, 255)


def ford_v(cv, pts, x, half=75, w=55):
    pts = np.array(pts)
    y = np.interp(x, pts[:, 0], pts[:, 1])
    cv.line([(x, y - half), (x, y + half)], w, 255)


def along(pts, t):
    """point and unit tangent at fraction t of a polyline's length"""
    p = np.array(pts, float)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    s = t * d[-1]
    x, y = np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])
    i = int(np.clip(np.searchsorted(d, s), 1, len(p) - 1))
    tg = p[i] - p[i - 1]
    tg = tg / (np.hypot(*tg) + 1e-9)
    return (float(x), float(y)), tg


def mindist(pts, x, y):
    p = np.array(pts, float)
    return float(np.hypot(p[:, 0] - x, p[:, 1] - y).min())


def rand_specs(r, n, hi=0.95, lo=0.2):
    fr = np.linspace(hi, lo, n) + r.uniform(-0.04, 0.04, n)
    tm = sorted((int(t) for t in r.integers(10, 56, n)), reverse=True)
    return [(float(min(max(f, 0.05), 0.98)), t) for f, t in zip(fr, tm)]


# ---------------------------------------------------------------- originals
def m01(c):
    cv = Canvas(True)
    r = wavy_v(400, 70, 95, 0.5)
    cv.line(smooth(r), 28, 0)
    ford_h(cv, r, 430)
    land = organic(cv.mask(), 1, 0.1)
    finish(c, "River splits the map; a single ford far to the south is the only crossing",
           land, (130, 160),
           [(40, 650, 90), (30, 700, 300), (30, 640, 500), (12, 110, 480)])


def m02(c):
    cv = Canvas(False)
    isl = [(130, 150, 100, 80), (390, 110, 80, 70), (630, 170, 110, 85),
           (250, 410, 110, 90), (540, 430, 100, 85), (730, 540, 55, 45)]
    for cx, cy, rx, ry in isl:
        cv.ell(cx, cy, rx, ry, 255)
    for a, b in [(0, 3), (3, 1), (1, 2), (4, 2), (3, 4), (4, 5)]:
        cv.line([isl[a][:2], isl[b][:2]], 60, 255)
    land = organic(cv.mask(), 2, 0.2)
    finish(c, "Archipelago of islands joined by isthmuses; piles on distant islands",
           land, (130, 150),
           [(40, 630, 170), (50, 540, 440), (20, 730, 540), (15, 390, 110)])


def m03(c):
    cv = Canvas(False)
    cv.ell(120, 300, 170, 230, 255)
    main = [(200, 300), (330, 390), (450, 330), (560, 210), (660, 150)]
    cv.line(smooth(main), 60, 255)
    cv.ell(700, 170, 70, 60, 255)
    br = [(450, 330), (540, 430), (630, 480)]
    cv.line(smooth(br), 55, 255)
    cv.ell(660, 490, 55, 50, 255)
    land = organic(cv.mask(), 3, 0.15)
    finish(c, "Long winding peninsula forking to two tips; the richest pile is at the far tip",
           land, (100, 300),
           [(60, 710, 165), (35, 660, 490), (10, 450, 335)])


def m04(c):
    n = noise(4, 30)
    xx, yy = np.meshgrid(np.arange(W) / W, np.arange(H) / H)
    f = n * 0.7 + 2.2 * (0.55 - 0.85 * xx - 0.25 * (1 - yy))
    land = f > 0
    finish(c, "Jagged coastline with bays and headlands; sea on the east, piles on far headlands",
           land, (130, 300), [], auto=[(0.95, 50), (0.75, 30), (0.5, 20), (0.3, 12)], seed=4, clean=True)


def m05(c):
    cv = Canvas(True)
    cs = [(405, 30), (390, 165), (420, 300), (395, 525), (410, 600)]
    for i, (cx, cy) in enumerate(cs):
        cv.ell(cx, cy, 42, 90 if i != 4 else 40, 0)
    cv.line([(405, 0), (405, 30)], 60, 0)
    land = organic(cv.mask(), 5, 0.12)
    finish(c, "Chain of lakes forms a wall; only one narrow gap between lakes lets greeps cross",
           land, (140, 300),
           [(40, 680, 120), (30, 690, 480), (25, 560, 300), (12, 180, 520)])


def m06(c):
    cv = Canvas(True)
    cv.poly([(0, 500), (800, 480), (800, 600), (0, 600)], 0)
    main = [(400, 110), (400, 200), (410, 290)]
    cv.line(smooth(main), 24, 0)
    brs = [[(410, 290), (300, 360), (230, 440), (200, 510)],
           [(410, 290), (370, 380), (350, 470), (340, 520)],
           [(410, 290), (460, 380), (500, 470), (510, 520)],
           [(410, 290), (540, 340), (640, 420), (700, 510)]]
    for b in brs:
        cv.line(smooth(b), 24, 0)
    cv.line([(0, 515), (800, 495)], 80, 0)
    # fords across some distributaries
    for j, tt in [(0, 0.66), (1, 0.7), (3, 0.66)]:
        p, tg = along(smooth(brs[j]), tt)
        nx, ny = -tg[1], tg[0]
        cv.line([(p[0] - 32 * nx, p[1] - 32 * ny), (p[0] + 32 * nx, p[1] + 32 * ny)], 60, 255)
    land = organic(cv.mask(), 6, 0.1)
    finish(c, "River delta: a river fans into four channels toward the sea; fords reach the fingers of land",
           land, (110, 180),
           [(40, 600, 230), (30, 270, 460), (30, 430, 440), (30, 680, 420), (12, 80, 420)])


def m07(c):
    cv = Canvas(False)
    pts = [(80, 80), (700, 80), (700, 230), (100, 230), (100, 380), (700, 380), (700, 510), (330, 510)]
    cv.line(smooth(pts, 3), 80, 255)
    cv.line([(420, 230), (420, 380)], 48, 255)
    cv.ell(300, 520, 70, 55, 255)
    land = organic(cv.mask(), 7, 0.1)
    finish(c, "Canyon: a serpentine plateau path across water cliffs with one shortcut bridge",
           land, (80, 70),
           [(30, 700, 100), (40, 100, 330), (50, 310, 515), (20, 600, 380)])


def m08(c):
    cv = Canvas(True)
    r = wavy_h(310, 80, 120, 0.3)
    cv.line(smooth(r), 26, 0)
    ford_v(cv, r, 220)
    ford_v(cv, r, 640)
    # oxbow lakes (crescents) that do not block the way
    for cx, cy, rad, a0, a1 in [(450, 150, 60, 20, 300), (140, 480, 60, 200, 480), (560, 500, 55, 100, 380)]:
        cv.d.arc([cx - rad, cy - rad, cx + rad, cy + rad], a0, a1, fill=0, width=22)
    cv.ell(330, 120, 40, 28, 0)
    land = organic(cv.mask(), 8, 0.1)
    finish(c, "Meandering river with oxbow lakes as distractors; two fords, piles hidden in bends",
           land, (100, 90),
           [(40, 700, 90), (30, 700, 540), (30, 440, 480), (12, 320, 200)])


def m09(c):
    cv = Canvas(True)
    cv.poly([(0, 0), (800, 0), (800, 90), (0, 90)], 0)
    for x, ln in [(140, 360), (330, 420), (520, 330), (690, 390)]:
        cv.poly([(x - 34, 0), (x + 34, 0), (x + 20, ln), (x, ln + 30), (x - 20, ln)], 0)
    land = organic(cv.mask(), 9, 0.2)
    finish(c, "Fjords: deep sea inlets from the north; piles on the tongues of land between them",
           land, (400, 520),
           [(50, 230, 150), (40, 610, 140), (30, 750, 160), (25, 60, 160), (12, 430, 160)])


def m10(c):
    cv = Canvas(False)
    cv.ell(400, 300, 330, 250, 255)
    cv.ell(400, 300, 220, 140, 0)
    cv.line([(580, 300), (500, 300)], 60, 255)
    cv.line([(400, 175), (400, 230)], 55, 255)
    cv.ell(380, 300, 70, 55, 255)
    cv.line([(440, 300), (500, 300)], 55, 255)
    cv.line([(400, 430), (400, 380)], 55, 255)
    land = organic(cv.mask(), 10, 0.15)
    finish(c, "Island ring around an inner lake with spokes and a central islet",
           land, (125, 300),
           [(40, 385, 300), (30, 400, 425), (40, 690, 300), (20, 400, 170)])


def m11(c):
    n = noise(11, 14)
    land = n < 0.8
    finish(c, "Marsh of ponds with winding dry paths",
           land, (110, 100), [], auto=[(0.95, 50), (0.75, 30), (0.55, 30), (0.35, 20), (0.2, 10)],
           seed=11, clean=True, margin=14)


def m12(c):
    cv = Canvas(True)
    cv.ell(400, 300, 270, 170, 0)
    cv.line([(600, 300), (860, 300)], 26, 0)
    cv.line([(0, 0), (800, 0)], 1, 255)
    land = organic(cv.mask(), 12, 0.2)
    finish(c, "Large lake with a creek cutting the east shore; the way round is the long way",
           land, (150, 520),
           [(40, 700, 150), (50, 700, 450), (25, 400, 60), (12, 70, 90)])


def m13(c):
    cv = Canvas(True)
    r1 = wavy_v(270, 60, 90, 0.0)
    r2 = wavy_v(560, 50, 110, 2.0)
    cv.line(smooth(r1), 26, 0)
    cv.line(smooth(r2), 26, 0)
    ford_h(cv, r1, 480)
    ford_h(cv, r2, 110)
    land = organic(cv.mask(), 13, 0.1)
    finish(c, "Two rivers with fords at opposite ends; piles in all three zones",
           land, (100, 100),
           [(30, 150, 480), (40, 400, 300), (50, 700, 470), (30, 690, 100)])


def m14(c):
    cv = Canvas(False)
    cv.ell(60, 300, 190, 260, 255)
    cv.ell(740, 320, 170, 230, 255)
    pts = [(200, 300), (300, 220), (400, 330), (500, 400), (600, 300)]
    cv.line(smooth(pts), 50, 255)
    cv.ell(400, 480, 50, 35, 255)
    cv.ell(380, 90, 55, 40, 255)
    land = organic(cv.mask(), 14, 0.15)
    finish(c, "Two landmasses joined by a long thin winding isthmus; richest piles on the far side",
           land, (60, 300),
           [(15, 120, 500), (50, 740, 200), (40, 740, 450), (25, 400, 480)])


def m15(c):
    n = noise(15, 30)
    land = n < 0.7
    finish(c, "Easy: broad headlands and bays, piles nearby around the ship",
           land, (400, 420), [], auto=[(0.9, 30), (0.6, 40), (0.4, 30), (0.2, 20), (0.1, 10)],
           seed=15, clean=True)


# ---------------------------------------------------------------- variants
def v01(c):
    r = c.rng
    cv = Canvas(True)
    horiz = r.random() < 0.4
    rw, amp, per, ph = r.uniform(26, 38), r.uniform(45, 90), r.uniform(75, 130), r.uniform(0, 6.28)
    fw = r.uniform(50, 64)
    if not horiz:
        x0 = r.uniform(340, 460)
        pts = wavy_v(x0, amp, per, ph)
        cv.line(smooth(pts), rw, 0)
        fy = r.uniform(330, 540)
        ford_h(cv, pts, fy, w=fw)
        xs = np.array(pts)[:, 0]
        lo, hi = xs.min() - rw / 2, xs.max() + rw / 2
        ship = (min(r.uniform(90, 200), lo - 70), r.uniform(90, 230))
        far = lambda: min(hi + r.uniform(110, 230), 735)
        piles = [(40, far(), r.uniform(60, 160)), (30, far(), r.uniform(250, 350)),
                 (30, far(), r.uniform(440, 540)), (12, ship[0], r.uniform(400, 520))]
    else:
        y0 = r.uniform(280, 330)
        amp = min(amp, 62)
        pts = wavy_h(y0, amp, per, ph)
        cv.line(smooth(pts), rw, 0)
        fx = r.uniform(420, 700)
        ford_v(cv, pts, fx, w=fw)
        ys = np.array(pts)[:, 1]
        lo, hi = ys.min() - rw / 2, ys.max() + rw / 2
        ship = (r.uniform(90, 250), min(r.uniform(70, 125), lo - 70))
        far = lambda: min(hi + r.uniform(75, 110), 555)
        piles = [(40, r.uniform(620, 740), far()), (30, r.uniform(350, 480), far()),
                 (30, r.uniform(90, 250), far()), (12, r.uniform(500, 700), ship[1])]
    land = organic(cv.mask(), int(r.integers(1, 10 ** 6)), 0.1)
    finish(c, "River splits the map; a single ford is the only crossing",
           land, ship, piles)


def v02(c):
    r = c.rng
    cv = Canvas(False)
    n = int(r.integers(5, 9))
    isl = []
    tries = 0
    while len(isl) < n and tries < 2000:
        tries += 1
        rx = r.uniform(55, 115)
        ry = rx * r.uniform(0.7, 0.95)
        cx, cy = r.uniform(rx + 15, W - rx - 15), r.uniform(ry + 15, H - ry - 15)
        if all(math.hypot(cx - o[0], cy - o[1]) > (rx + ry + o[2] + o[3]) / 2 + 60 for o in isl):
            isl.append((cx, cy, rx, ry))
    if len(isl) < 5:
        raise Reject("few islands")
    n = len(isl)
    iw = r.uniform(54, 66)
    # minimum spanning tree plus a couple of extra links
    D = [[math.hypot(a[0] - b[0], a[1] - b[1]) for b in isl] for a in isl]
    seen, edges = {0}, []
    while len(seen) < n:
        a, b = min(((a, b) for a in seen for b in range(n) if b not in seen), key=lambda e: D[e[0]][e[1]])
        edges.append((a, b))
        seen.add(b)
    extra = [(a, b) for a in range(n) for b in range(a + 1, n)
             if (a, b) not in edges and (b, a) not in edges and D[a][b] < 330]
    r.shuffle(extra)
    edges += extra[:int(r.integers(1, 3))]
    if max(D[a][b] for a, b in edges) > 400:
        raise Reject("link too long")
    for cx, cy, rx, ry in isl:
        cv.ell(cx, cy, rx, ry, 255)
    for a, b in edges:
        cv.line([isl[a][:2], isl[b][:2]], iw, 255)
    # graph distance from the ship island
    s = int(np.argmin([i[0] + r.uniform(0, 150) for i in isl]))
    dist = {s: 0.0}
    todo = [s]
    while todo:
        a = todo.pop(0)
        for e in edges:
            if a in e:
                b = e[1] if e[0] == a else e[0]
                if b not in dist or dist[b] > dist[a] + D[a][b] + 1e-9:
                    dist[b] = dist[a] + D[a][b]
                    todo.append(b)
    order = sorted((i for i in range(n) if i != s), key=lambda i: -dist[i])
    k = min(len(order), int(r.integers(3, 6)))
    tom = sorted((int(t) for t in r.integers(15, 55, k)), reverse=True)
    piles = [(t, isl[i][0], isl[i][1]) for t, i in zip(tom, order[:k])]
    land = organic(cv.mask(), int(r.integers(1, 10 ** 6)), 0.2)
    finish(c, "Archipelago of islands joined by isthmuses; piles on distant islands",
           land, (isl[s][0], isl[s][1]), piles)


def v03(c):
    r = c.rng
    cv = Canvas(False)
    cx, cy = r.uniform(100, 150), r.uniform(240, 360)
    rx, ry = r.uniform(150, 190), r.uniform(190, 250)
    cv.ell(cx, cy, rx, ry, 255)
    n = int(r.integers(4, 6))
    x0, xe = cx + rx * 0.7, r.uniform(630, 690)
    sign = 1 if r.random() < 0.5 else -1
    pts = [(x0, cy)]
    y = cy
    for i in range(1, n + 1):
        y = float(np.clip(y + sign * r.uniform(80, 150), 130, 470))
        sign = -sign
        pts.append((x0 + (xe - x0) * i / n, y))
    main = smooth(pts)
    cv.line(main, r.uniform(56, 66), 255)
    tip_r = r.uniform(55, 75)
    cv.ell(pts[-1][0], pts[-1][1], tip_r, tip_r * 0.85, 255)
    j = int(r.integers(1, n))
    px, py = pts[j]
    best = None
    for sg in (1, -1):
        e = (px + r.uniform(60, 130), py + sg * r.uniform(120, 190))
        if not (80 < e[1] < 520):
            continue
        dm = mindist(main[::3], *e)
        if best is None or dm > best[0]:
            best = (dm, e)
    if best is None or best[0] < 120:
        raise Reject("fork collides")
    e = best[1]
    cv.line(smooth([(px, py), ((px + e[0]) / 2 + r.uniform(-30, 30), (py + e[1]) / 2), e]), r.uniform(52, 60), 255)
    cv.ell(e[0], e[1], tip_r * 0.8, tip_r * 0.75, 255)
    land = organic(cv.mask(), int(r.integers(1, 10 ** 6)), 0.15)
    mp, _ = along(main, r.uniform(0.35, 0.55))
    piles = [(60, pts[-1][0], pts[-1][1]), (35, e[0], e[1]), (10, mp[0], mp[1])]
    finish(c, "Long winding peninsula forking to two tips; the richest pile is at the far tip",
           land, (r.uniform(70, 130), cy + r.uniform(-50, 50)), piles)


def v04(c):
    r = c.rng
    n = noise(int(r.integers(1, 10 ** 6)), r.uniform(24, 36))
    xx, yy = np.meshgrid(np.arange(W) / W, np.arange(H) / H)
    c0, a = r.uniform(0.5, 0.62), r.uniform(0.78, 0.95)
    tilt = r.uniform(-0.3, 0.35)
    if r.random() < 0.65:
        f = n * 0.7 + 2.2 * (c0 - a * xx - tilt * (1 - yy))
        ship = (r.uniform(100, 180), r.uniform(180, 420))
    else:
        f = n * 0.7 + 2.2 * (c0 + 0.08 - a * 1.15 * yy - tilt * (1 - xx))
        ship = (r.uniform(200, 600), r.uniform(70, 130))
    finish(c, "Jagged coastline with bays and headlands; piles on far headlands",
           f > 0, ship, [], auto=rand_specs(r, int(r.integers(4, 7))), clean=True, land_r=20)


def v05(c):
    r = c.rng
    cv = Canvas(True)
    horiz = r.random() < 0.4
    rl = r.uniform(80, 100)          # half length of a lake along the chain
    rw = r.uniform(38, 50)           # half width
    gap = r.uniform(50, 72)          # the only crossing
    base = r.uniform(380, 440) if not horiz else r.uniform(280, 330)
    ext = H if horiz else H
    L = W if horiz else H
    gpos = r.uniform(0.25, 0.7) * L
    cs = []
    p = -rl * r.uniform(0.2, 0.6)
    gap_done = False
    while p < L + rl:
        cs.append(p)
        if not gap_done and p > gpos - rl:
            p += 2 * rl + gap
            gap_done = True
        else:
            p += 1.5 * rl
    for i, p in enumerate(cs):
        w = base + r.uniform(-18, 18)
        if horiz:
            cv.ell(p, w, rl, rw, 0)
        else:
            cv.ell(w, p, rw, rl, 0)
    lenx = [p for p in cs]
    # also close the two ends of the chain at the map border
    if horiz:
        cv.line([(0, base), (rl, base)], 2 * rw * 0.8, 0)
        cv.line([(L - rl, base), (L, base)], 2 * rw * 0.8, 0)
    else:
        cv.line([(base, 0), (base, rl)], 2 * rw * 0.8, 0)
        cv.line([(base, L - rl), (base, L)], 2 * rw * 0.8, 0)
    land = organic(cv.mask(), int(r.integers(1, 10 ** 6)), 0.12)
    if not horiz:
        ship = (r.uniform(100, 200), r.uniform(150, 450))
        fx = lambda: min(base + r.uniform(170, 300), 740)
        piles = [(40, fx(), r.uniform(60, 180)), (30, fx(), r.uniform(420, 540)),
                 (25, min(base + r.uniform(110, 160), 740), r.uniform(240, 360)),
                 (12, r.uniform(120, 240), r.uniform(60, 540))]
    else:
        ship = (r.uniform(100, 700), r.uniform(60, 115))
        fy = lambda: min(base + r.uniform(130, 220), 570)
        piles = [(40, r.uniform(60, 200), fy()), (30, r.uniform(560, 740), fy()),
                 (25, r.uniform(300, 500), min(base + r.uniform(90, 140), 570)),
                 (12, r.uniform(60, 740), r.uniform(60, 110))]
    finish(c, "Chain of lakes forms a wall; only one narrow gap between lakes lets greeps cross",
           land, ship, piles)


def v06(c):
    r = c.rng
    cv = Canvas(True)
    sea = r.uniform(480, 520)
    slope = r.uniform(-25, 25)
    sea_y = lambda x: sea + slope * (x - 400) / 400
    cv.poly([(0, sea_y(0)), (800, sea_y(800)), (800, 600), (0, 600)], 0)
    ax, ay = r.uniform(340, 460), r.uniform(255, 310)
    top = r.uniform(90, 130)
    cw = r.uniform(26, 34)
    cv.line(smooth([(ax, top), (ax + r.uniform(-25, 25), (top + ay) / 2), (ax, ay)]), cw, 0)
    k = int(r.integers(3, 6))
    a, b = r.uniform(120, 200), r.uniform(640, 720)
    ex = np.sort(np.linspace(a, b, k) + r.uniform(-12, 12, k))
    chans = []
    for e in ex:
        ey = sea_y(e) + 6
        mid = (ax + (e - ax) * 0.55 + r.uniform(-25, 25), ay + (ey - ay) * r.uniform(0.4, 0.55))
        pts = smooth([(ax, ay), mid, (e, ey)])
        chans.append(pts)
        cv.line(pts, cw, 0)
    cv.line([(0, sea_y(0) + 15), (800, sea_y(800) + 15)], 80, 0)
    # choose fords so that every finger of land is reachable
    for _ in range(200):
        S = [j for j in range(k) if r.random() < 0.6]
        parent = list(range(k))          # nodes 0..k-2 fingers, k-1 = banks
        def fnd(x):
            while parent[x] != x:
                x = parent[x]
            return x
        for j in S:
            u = (j - 1) if j > 0 else k - 1
            v = j if j < k - 1 else k - 1
            parent[fnd(u)] = fnd(v)
        if len({fnd(i) for i in range(k)}) == 1 and len(S) >= 2:
            break
    else:
        raise Reject("fords")
    for j in S:
        for t in np.arange(r.uniform(0.55, 0.68), 0.93, 0.03):
            p, tg = along(chans[j], t)
            if all(mindist(chans[o], *p) > 66 + cw / 2 for o in range(k) if o != j):
                break
        else:
            raise Reject("ford too tight")
        nrm = np.array([-tg[1], tg[0]])
        cv.line([(p[0] - 32 * nrm[0], p[1] - 32 * nrm[1]), (p[0] + 32 * nrm[0], p[1] + 32 * nrm[1])],
                r.uniform(56, 64), 255)
    land = organic(cv.mask(), int(r.integers(1, 10 ** 6)), 0.1)
    piles = []
    tom = [int(t) for t in r.integers(15, 50, k + 2)]
    for i in range(k - 1):
        t = r.uniform(0.8, 0.9)
        p1, _ = along(chans[i], t)
        p2, _ = along(chans[i + 1], t)
        piles.append((tom[i], (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2))
    piles.append((tom[k - 1], min(ex[-1] + r.uniform(70, 100), 760), sea_y(700) - r.uniform(110, 140)))
    piles.append((tom[k], r.uniform(ax + 140, ax + 250), r.uniform(ay - 80, ay - 30)))
    if r.random() < 0.6:
        piles.append((12, max(ex[0] - r.uniform(80, 110), 40), sea_y(100) - r.uniform(100, 140)))
    finish(c, "River delta: a river fans into channels toward the sea; fords reach the fingers of land",
           land, (r.uniform(90, 200), r.uniform(130, 230)), piles)


def v07(c):
    r = c.rng
    cv = Canvas(False)
    nrow = int(r.integers(3, 5))
    ys = np.linspace(r.uniform(70, 95), r.uniform(500, 535), nrow) + r.uniform(-8, 8, nrow)
    xl, xr = r.uniform(65, 110), r.uniform(660, 725)
    pts = []
    for i, y in enumerate(ys):
        ltr = i % 2 == 0
        a, b = (xl, xr) if ltr else (xr, xl)
        if i == nrow - 1:
            b = r.uniform(280, 420) if ltr else r.uniform(380, 520)
        pts += [(a, y), (b, y)]
    sm = smooth(pts, 3)
    cv.line(sm, 76, 255)
    ex, ey = pts[-1]
    cv.ell(ex + (-1 if nrow % 2 == 0 else 1) * 20, ey + r.uniform(0, 12), r.uniform(60, 75), r.uniform(50, 58), 255)
    for _ in range(int(r.integers(1, 3))):
        i = int(r.integers(0, nrow - 1))
        x = r.uniform(250, 560)
        cv.line([(x, ys[i]), (x, ys[i + 1])], r.uniform(48, 60), 255)
    land = organic(cv.mask(), int(r.integers(1, 10 ** 6)), 0.1)
    k = int(r.integers(4, 6))
    fr = np.sort(np.linspace(0.18, 0.97, k) + r.uniform(-0.06, 0.06, k))
    tom = r.integers(15, 55, k)
    piles = [(int(t), *along(sm, float(f))[0]) for t, f in zip(tom, fr)]
    finish(c, "Canyon: a serpentine plateau path across water cliffs with shortcut bridges",
           land, (xl, ys[0] - 10), piles)


def v08(c):
    r = c.rng
    cv = Canvas(True)
    horiz = r.random() < 0.55
    per, ph, amp = r.uniform(100, 140), r.uniform(0, 6.28), r.uniform(60, 90)
    rw = r.uniform(24, 32)
    if horiz:
        y0 = r.uniform(290, 330)
        pts = wavy_h(y0, amp, per, ph)
        span = (110, 700)
    else:
        x0 = r.uniform(380, 420)
        amp = min(amp, 70)
        pts = wavy_v(x0, amp, per, ph)
        span = (90, 510)
    sm = smooth(pts)
    cv.line(sm, rw, 0)
    nf = int(r.integers(1, 4))
    fp = []
    for _ in range(100):
        v = r.uniform(*span)
        if all(abs(v - o) > 200 for o in fp):
            fp.append(v)
        if len(fp) == nf:
            break
    for v in fp:
        (ford_v if horiz else ford_h)(cv, pts, v, w=r.uniform(52, 64))
    ship = (r.uniform(80, 140), r.uniform(70, 110)) if horiz else (r.uniform(70, 120), r.uniform(90, 160))
    placed = []
    fordpts = []
    for v in fp:
        a = np.array(pts)
        fordpts.append(tuple(a[np.argmin(np.abs(a[:, 0 if horiz else 1] - v))]))
    for _ in range(600):
        if len(placed) >= int(r.integers(3, 6)):
            break
        rad = r.uniform(48, 64)
        cx, cy = r.uniform(rad + 20, W - rad - 20), r.uniform(rad + 20, H - rad - 20)
        if mindist(sm, cx, cy) < rad + 55 or math.hypot(cx - ship[0], cy - ship[1]) < 140:
            continue
        if any(math.hypot(cx - p[0], cy - p[1]) < 120 for p in fordpts):
            continue
        if any(math.hypot(cx - o[0], cy - o[1]) < 2.2 * rad + 40 for o in placed):
            continue
        placed.append((cx, cy, rad))
    for cx, cy, rad in placed:
        if r.random() < 0.8:
            a0 = r.uniform(0, 360)
            cv.d.arc([cx - rad, cy - rad, cx + rad, cy + rad], a0, a0 + r.uniform(240, 300), fill=0, width=22)
        else:
            cv.ell(cx, cy, rad * 0.7, rad * 0.5, 0)
    land = organic(cv.mask(), int(r.integers(1, 10 ** 6)), 0.1)
    piles = []
    tom = sorted((int(t) for t in r.integers(12, 52, 4)), reverse=True)
    sides = [1, -1, 1, -1]
    r.shuffle(sides)
    ts = np.sort(r.uniform(0.15, 0.9, 4))
    for t, s_, tm in zip(ts, sides, tom):
        p, tg = along(sm[15:-15], float(t))
        nrm = np.array([-tg[1], tg[0]]) * s_
        o = r.uniform(115, 150)
        piles.append((tm, p[0] + nrm[0] * o, p[1] + nrm[1] * o))
    finish(c, "Meandering river with oxbow lakes as distractors; fords cross it, piles hidden in bends",
           land, ship, piles)


def v09(c):
    r = c.rng
    cv = Canvas(True)
    sea = r.uniform(60, 105)
    cv.poly([(0, 0), (800, 0), (800, sea), (0, sea)], 0)
    nf = int(r.integers(3, 6))
    xs = np.sort(np.linspace(110, 690, nf) + r.uniform(-22, 22, nf))
    for x in xs:
        hw = r.uniform(26, 40)
        ln = r.uniform(280, 440)
        cv.poly([(x - hw, 0), (x + hw, 0), (x + hw * 0.6, ln), (x, ln + 30), (x - hw * 0.6, ln)], 0)
    land = organic(cv.mask(), int(r.integers(1, 10 ** 6)), 0.2)
    edges = np.r_[0, xs, W]
    mids = (edges[:-1] + edges[1:]) / 2
    idx = r.permutation(len(mids))[:int(r.integers(4, min(7, len(mids) + 1)))]
    tom = sorted((int(t) for t in r.integers(12, 55, len(idx))), reverse=True)
    far = sorted(idx, key=lambda i: -abs(mids[i] - 400))
    piles = [(t, min(max(mids[i], 45), 755), sea + r.uniform(55, 100)) for t, i in zip(tom, far)]
    finish(c, "Fjords: deep sea inlets from the north; piles on the tongues of land between them",
           land, (r.uniform(180, 620), r.uniform(500, 535)), piles)


def v10(c):
    r = c.rng
    cv = Canvas(False)
    cx, cy = 400 + r.uniform(-20, 20), 300 + r.uniform(-12, 12)
    rxo, ryo = r.uniform(300, 350), r.uniform(215, 255)
    rxi, ryi = rxo - r.uniform(95, 120), ryo - r.uniform(95, 115)
    cv.ell(cx, cy, rxo, ryo, 255)
    cv.ell(cx, cy, rxi, ryi, 0)
    P = lambda a, s: (cx + (rxi + s * (rxo - rxi)) * math.cos(a), cy + (ryi + s * (ryo - ryi)) * math.sin(a))
    angs = []

    def pick(sep):
        for _ in range(200):
            a = r.uniform(0, 2 * math.pi)
            if all(abs((a - o + math.pi) % (2 * math.pi) - math.pi) > sep for o in angs):
                angs.append(a)
                return a
        raise Reject("angles")
    a_ship = pick(0.6)
    spokes = [pick(0.7) for _ in range(int(r.integers(1, 3)))]
    stubs = [pick(0.7) for _ in range(int(r.integers(1, 3)))]
    ring_piles = [pick(0.7) for _ in range(int(r.integers(2, 4)))]
    ic = (cx + r.uniform(-30, 30), cy + r.uniform(-25, 25))
    for a in spokes:
        cv.line([P(a, 0.5), ic], r.uniform(52, 60), 255)
    for a in stubs:
        s0 = np.array(P(a, 0.5))
        d = np.array([cx, cy]) - s0
        d /= np.hypot(*d)
        cv.line([tuple(s0), tuple(s0 + d * r.uniform(110, 150))], r.uniform(50, 56), 255)
    cv.ell(ic[0], ic[1], r.uniform(62, 85), r.uniform(48, 60), 255)
    land = organic(cv.mask(), int(r.integers(1, 10 ** 6)), 0.15)
    tom = [int(t) for t in r.integers(15, 50, len(ring_piles) + 1)]
    piles = [(tom[0] + 5, ic[0], ic[1])] + [(t, *P(a, 0.5)) for t, a in zip(tom[1:], ring_piles)]
    finish(c, "Island ring around an inner lake with spokes and a central islet",
           land, P(a_ship, 0.5), piles)


def v11(c):
    r = c.rng
    n = noise(int(r.integers(1, 10 ** 6)), r.uniform(12, 18))
    land = n < r.uniform(0.6, 0.95)
    finish(c, "Marsh of ponds with winding dry paths", land,
           (r.uniform(90, 300), r.uniform(80, 180)), [], auto=rand_specs(r, int(r.integers(4, 7))),
           clean=True, margin=14, land_r=21)


def v12(c):
    r = c.rng
    cv = Canvas(True)
    cx, cy = r.uniform(380, 430), r.uniform(285, 320)
    rx, ry = r.uniform(225, 270), r.uniform(145, 185)
    cv.rell(cx, cy, rx, ry, r.uniform(-0.2, 0.2), 0)
    yc0 = cy + r.uniform(-50, 50)
    yce = float(np.clip(yc0 + r.uniform(-90, 90), 130, 470))
    xs = np.linspace(cx + rx * 0.6, 870, 7)
    ys = yc0 + (yce - yc0) * np.linspace(0, 1, 7) + r.uniform(-18, 18, 7)
    cr = smooth(list(zip(xs, ys)))
    cv.line(cr, r.uniform(24, 34), 0)
    land = organic(cv.mask(), int(r.integers(1, 10 ** 6)), 0.2)
    yat = lambda x: float(np.interp(x, cr[:, 0], cr[:, 1]))
    ex = min(cx + rx + r.uniform(50, 80), 740)
    piles = [(40, ex, yat(ex) - r.uniform(110, 150)), (50, ex + r.uniform(-20, 20), yat(ex) + r.uniform(110, 150)),
             (25, cx + r.uniform(-120, 120), cy - ry - r.uniform(50, 80)),
             (12, cx - rx - r.uniform(40, 80), r.uniform(70, 150))]
    finish(c, "Large lake with a creek cutting the shore; the way round is the long way",
           land, (r.uniform(100, 300), r.uniform(505, 540)), piles)


def v13(c):
    r = c.rng
    cv = Canvas(True)
    horiz = r.random() < 0.4
    if not horiz:
        x1 = r.uniform(190, 290)
        x2 = min(x1 + r.uniform(240, 320), 590)
        r1 = wavy_v(x1, r.uniform(40, 70), r.uniform(80, 120), r.uniform(0, 6.28))
        r2 = wavy_v(x2, r.uniform(40, 65), r.uniform(90, 125), r.uniform(0, 6.28))
        cv.line(smooth(r1), r.uniform(24, 34), 0)
        cv.line(smooth(r2), r.uniform(24, 34), 0)
        ford_h(cv, r1, r.uniform(380, 540), w=r.uniform(50, 64))
        ford_h(cv, r2, r.uniform(60, 230), w=r.uniform(50, 64))
        ship = (r.uniform(80, 150), r.uniform(70, 180))
        east = (x2 + 800) / 2
        piles = [(30, r.uniform(80, x1 - 70), r.uniform(380, 540)), (40, (x1 + x2) / 2, r.uniform(240, 360)),
                 (50, east, r.uniform(420, 520)), (30, east, r.uniform(60, 160))]
    else:
        y1 = r.uniform(170, 210)
        y2 = y1 + r.uniform(190, 215)
        r1 = wavy_h(y1, r.uniform(22, 42), r.uniform(80, 120), r.uniform(0, 6.28))
        r2 = wavy_h(y2, r.uniform(22, 42), r.uniform(90, 125), r.uniform(0, 6.28))
        cv.line(smooth(r1), r.uniform(24, 34), 0)
        cv.line(smooth(r2), r.uniform(24, 34), 0)
        ford_v(cv, r1, r.uniform(450, 720), w=r.uniform(50, 64))
        ford_v(cv, r2, r.uniform(80, 350), w=r.uniform(50, 64))
        ship = (r.uniform(80, 150), r.uniform(55, 85))
        piles = [(30, r.uniform(550, 740), (y1 - 40) / 2), (40, r.uniform(250, 550), (y1 + y2) / 2),
                 (50, r.uniform(620, 740), 555), (30, r.uniform(80, 250), 555)]
    land = organic(cv.mask(), int(r.integers(1, 10 ** 6)), 0.1)
    finish(c, "Two rivers with fords at opposite ends; piles in all three zones", land, ship, piles)


def v14(c):
    r = c.rng
    cv = Canvas(False)
    cxl, cyl, rxl, ryl = r.uniform(40, 90), r.uniform(280, 320), r.uniform(140, 170), r.uniform(220, 270)
    cxr, cyr, rxr, ryr = r.uniform(720, 770), r.uniform(300, 340), r.uniform(130, 160), r.uniform(200, 240)
    cv.ell(cxl, cyl, rxl, ryl, 255)
    cv.ell(cxr, cyr, rxr, ryr, 255)
    xa, xb = cxl + rxl - 15, cxr - rxr + 15
    n = int(r.integers(4, 7))
    sg = 1 if r.random() < 0.5 else -1
    pts = [(xa, cyl + r.uniform(-40, 40))]
    for i in range(1, n):
        pts.append((xa + (xb - xa) * i / n, float(np.clip(300 + sg * r.uniform(80, 170), 130, 470))))
        sg = -sg
    pts.append((xb, cyr + r.uniform(-40, 40)))
    iso = smooth(pts)
    cv.line(iso, r.uniform(48, 58), 255)
    bp, _ = along(iso, r.uniform(0.4, 0.6))
    cv.ell(bp[0], bp[1], r.uniform(55, 70), r.uniform(48, 60), 255)
    for _ in range(int(r.integers(0, 3))):
        p, tg = along(iso, r.uniform(0.2, 0.85))
        s_ = 1 if r.random() < 0.5 else -1
        nrm = np.array([-tg[1], tg[0]]) * s_
        e = (p[0] + nrm[0] * 90, p[1] + nrm[1] * 90)
        if 60 < e[1] < 540 and mindist(iso[::4], *e) > 80 and math.hypot(e[0] - bp[0], e[1] - bp[1]) > 120:
            cv.line([p, e], 48, 255)
            cv.ell(e[0], e[1], 42, 38, 255)
    land = organic(cv.mask(), int(r.integers(1, 10 ** 6)), 0.15)
    piles = [(15, cxl + r.uniform(-30, 80), cyl + r.uniform(110, 190)),
             (50, cxr + r.uniform(-30, 30), cyr - r.uniform(90, 150)),
             (40, cxr + r.uniform(-30, 30), cyr + r.uniform(90, 140)),
             (25, bp[0], bp[1])]
    finish(c, "Two landmasses joined by a long thin winding isthmus; richest piles on the far side",
           land, (r.uniform(70, 120), cyl + r.uniform(-50, 50)), piles)


def v15(c):
    r = c.rng
    n = noise(int(r.integers(1, 10 ** 6)), r.uniform(24, 36))
    land = n < r.uniform(0.6, 0.85)
    k = int(r.integers(4, 7))
    specs = rand_specs(r, k, hi=0.9, lo=0.1)
    finish(c, "Easy: broad headlands and bays, piles nearby around the ship", land,
           (r.uniform(300, 500), r.uniform(340, 480)), [], auto=specs, clean=True, land_r=20)


ORIG = [m01, m02, m03, m04, m05, m06, m07, m08, m09, m10, m11, m12, m13, m14, m15]
VAR = [v01, v02, v03, v04, v05, v06, v07, v08, v09, v10, v11, v12, v13, v14, v15]


def make_variant(nn, k, seed, out, tries=25):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools.check_maps import check
    last = None
    for attempt in range(tries):
        c = Ctx(nn, out, True, seed, k, attempt)
        try:
            VAR[nn - 1](c)
        except Reject as e:
            last = str(e)
            continue
        m, water, errors, warnings = check(os.path.join(out, c.name + ".map"))
        if not errors and not warnings:
            return attempt
        last = "; ".join(errors + warnings)
    raise SystemExit("FAILED terrain_%02d_v%03d after %d tries: %s" % (nn, k, tries, last))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.variants <= 0:
        os.makedirs(OUT, exist_ok=True)
        for i, f in enumerate(ORIG):
            f(Ctx(i + 1, OUT))
        return
    out = a.out or os.path.join("out", "terrain_variants")
    os.makedirs(out, exist_ok=True)
    for nn in range(1, 16):
        used = [make_variant(nn, k, a.seed, out) for k in range(1, a.variants + 1)]
        print("terrain_%02d: %d variants ok (resamples: %d)" % (nn, a.variants, sum(used)))


if __name__ == "__main__":
    main()




