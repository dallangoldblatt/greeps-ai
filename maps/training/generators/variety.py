"""Generate 'variety_XX' training maps: open search, pile dynamics and mixed-terrain worlds.

Run from the repo root:
    python maps/training/generators/variety.py                     # 15 base maps -> maps/training
    python maps/training/generators/variety.py --variants N --seed S --out DIR

Ideas 01-05 open search, 06-10 pile dynamics, 11-15 mixed terrain. Every written map is
validated with tools.check_maps.check and resampled (new internal seed) until it passes.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

W, H = 800, 600
WATER = (40, 70, 160)
LAND = (235, 215, 190)
GAP = 46          # clear space kept between obstacles (passages must be >= 40)
EDGE = 30         # obstacles keep this far from the world edge
MAX_ATTEMPTS = 40
BASE_SEED = 7


class Retry(Exception):
    pass


def rt(rng, a, b):
    return int(rng.integers(a, b + 1))


# ------------------------------------------------------------------ geometry helpers
def edt_land(land):
    if land.all():
        return np.full(land.shape, 999.0)
    return ndi.distance_transform_edt(land)


def rect_mask(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[max(0, int(y0)):max(0, int(y1)), max(0, int(x0)):max(0, int(x1))] = True
    return m


def open_edt(mask, r):
    """Morphological opening with a disk of radius r (via distance transforms)."""
    if mask.all() or not mask.any():
        return mask
    core = ndi.distance_transform_edt(mask) > r
    if not core.any():
        return core
    return ndi.distance_transform_edt(~core) <= r


def cleanup(land, land_r=22, water_r=8):
    land = open_edt(land, land_r)
    water = open_edt(~land, water_r)
    return ~water


def fill_thin(land, minrun=10, iters=3):
    land = land.copy()
    for _ in range(iters):
        for axis in (0, 1):
            water = ~land if axis == 1 else (~land).T
            l = land if axis == 1 else land.T
            for r in range(water.shape[0]):
                row = water[r]
                if not row.any():
                    continue
                e = np.flatnonzero(np.diff(np.r_[0, row.astype(np.int8), 0]))
                for a, b in zip(e[::2], e[1::2]):
                    if b - a < minrun and a > 0 and b < len(row):
                        l[r, a:b] = True
    return land


def noise(rng, sigma):
    n = ndi.gaussian_filter(rng.standard_normal((H, W)), sigma)
    return (n - n.mean()) / n.std()


def blob(x, y, r, rng, n=10):
    ang = np.sort(rng.uniform(0, 2 * math.pi, n))
    ang = (np.arange(n) + rng.uniform(-0.3, 0.3, n)) * 2 * math.pi / n
    return [(x + r * rng.uniform(0.85, 1.05) * math.cos(a), y + r * rng.uniform(0.85, 1.05) * math.sin(a))
            for a in ang]


def make(kind, x, y, rng):
    """Obstacle geometry: (drawkind, points, width, registered circles)."""
    if kind == "rock":
        r = rng.uniform(13, 26)
        return ("poly", blob(x, y, r, rng), 0, [(x, y, r * 1.06)])
    if kind == "pond":
        rx, ry, rot = rng.uniform(17, 36), rng.uniform(14, 28), rng.uniform(0, math.pi)
        pts = []
        for t in np.linspace(0, 2 * math.pi, 18, endpoint=False):
            ex, ey = rx * math.cos(t), ry * math.sin(t)
            pts.append((x + ex * math.cos(rot) - ey * math.sin(rot), y + ex * math.sin(rot) + ey * math.cos(rot)))
        return ("poly", pts, 0, [(x, y, max(rx, ry))])
    if kind == "stub":
        L, a, wd = rng.uniform(50, 115), rng.uniform(0, math.pi), rng.uniform(14, 18)
        dx, dy = math.cos(a) * L / 2, math.sin(a) * L / 2
        p0, p1 = (x - dx, y - dy), (x + dx, y + dy)
        n = int(L // 10) + 1
        cs = [(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t, wd / 2 + 1)
              for t in np.linspace(0, 1, n)]
        return ("line", [p0, p1], wd, cs)
    if kind == "block":
        bw, bh = rng.uniform(40, 70), rng.uniform(40, 60)
        pts = [(x - bw / 2, y - bh / 2), (x + bw / 2, y - bh / 2), (x + bw / 2, y + bh / 2), (x - bw / 2, y + bh / 2)]
        r = min(bw, bh) / 2
        long_ = max(bw, bh)
        offs = np.arange(-(long_ / 2 - r), long_ / 2 - r + 0.01, r) if long_ > 2 * r else [0.0]
        cs = [((x + o, y, r * 1.45) if bw >= bh else (x, y + o, r * 1.45)) for o in offs]
        return ("poly", pts, 0, cs)
    raise ValueError(kind)


class World:
    def __init__(self, land=True, mask=None):
        if mask is not None:
            self.img = Image.fromarray((mask * 255).astype(np.uint8), "L")
        else:
            self.img = Image.new("L", (W, H), 255 if land else 0)
        self.d = ImageDraw.Draw(self.img)
        self.circ = []

    def mask(self):
        return np.array(self.img) > 127

    def zone(self, x, y, r):
        self.circ.append((x, y, r))

    def fits(self, cs, gap=GAP, edge=EDGE):
        a = np.array(self.circ) if self.circ else None
        for x, y, r in cs:
            if x - r < edge or x + r > W - edge or y - r < edge or y + r > H - edge:
                return False
            if a is not None and (np.hypot(a[:, 0] - x, a[:, 1] - y) - a[:, 2] - r < gap).any():
                return False
        return True

    def draw(self, g):
        kind, pts, wd, cs = g
        if kind == "poly":
            self.d.polygon([tuple(p) for p in pts], fill=0)
        else:
            self.d.line([tuple(p) for p in pts], fill=0, width=int(wd))
            for px, py in pts:
                self.d.ellipse([px - wd / 2, py - wd / 2, px + wd / 2, py + wd / 2], fill=0)
        self.circ.extend(cs)

    def scatter(self, rng, n, kinds, box=(0, 0, W, H), tries=60):
        placed = 0
        for _ in range(n):
            for _ in range(tries):
                x, y = rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])
                g = make(kinds[int(rng.integers(len(kinds)))], x, y, rng)
                if self.fits(g[3]):
                    self.draw(g)
                    placed += 1
                    break
        return placed

    def wline(self, pts, wd, reg=True, val=0):
        pts = [tuple(p) for p in pts]
        for a, b in zip(pts, pts[1:]):
            self.d.line([a, b], fill=val, width=int(wd))
        r = wd / 2
        for x, y in pts:
            self.d.ellipse([x - r, y - r, x + r, y + r], fill=val)
        if reg and val == 0:
            for a, b in zip(pts, pts[1:]):
                n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) // 8))
                for t in np.linspace(0, 1, n + 1):
                    self.circ.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, r))

    def reg_rect(self, x0, y0, x1, y1):
        for x in np.arange(x0 + 15, x1 - 14, 30):
            for y in np.arange(y0 + 15, y1 - 14, 30):
                self.circ.append((x, y, 22))
        self.circ.append(((x0 + x1) / 2, (y0 + y1) / 2, 1))


# ------------------------------------------------------------------ ship / pile picking
def pick_ship(land, rng, region=None):
    edt = edt_land(land)
    foot = ndi.minimum_filter(land.astype(np.uint8), size=(57, 111), mode="constant", cval=0) > 0
    rel = np.zeros_like(foot)
    rel[:-30] = edt[30:] >= 26
    ok = foot & rel
    ok[:, :12] = False
    ok[:, -12:] = False
    ok[560:] = False
    if region is not None:
        ok &= region
    ys, xs = np.nonzero(ok)
    if len(xs) == 0:
        raise Retry("no ship position")
    i = int(rng.integers(len(xs)))
    return int(xs[i]), int(ys[i])


def cand_piles(land, ship):
    edt = edt_land(land)
    rx, ry = ship[0], ship[1] + 30
    lab, _ = ndi.label(edt >= 20)
    win = lab[max(0, ry - 26):ry + 27, max(0, rx - 26):rx + 27]
    ids = np.unique(win[win > 0])
    if len(ids) == 0:
        raise Retry("ship in pocket")
    cand = np.isin(lab, ids) & (edt >= 24)
    cand[:28] = cand[-28:] = False
    cand[:, :28] = cand[:, -28:] = False
    return cand


def sample(cand, rng, ship=None, dmin=0, dmax=1e9, others=(), spacing=100, disc=None, region=None):
    m = cand if region is None else cand & region
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    ok = np.ones(len(xs), bool)
    if ship is not None:
        d = np.hypot(xs - ship[0], ys - (ship[1] + 30))
        ok &= (d >= dmin) & (d <= dmax)
    for ox, oy in others:
        ok &= np.hypot(xs - ox, ys - oy) >= spacing
    if disc is not None:
        ok &= np.hypot(xs - disc[0], ys - disc[1]) <= disc[2]
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return None
    i = int(rng.choice(idx))
    return int(xs[i]), int(ys[i])


def place(cand, rng, ship, specs, regions=None, spacing=110, dmin=150, piles=None):
    """specs: (region name or None, count, (tmin, tmax)). Returns list of (tomatoes, x, y)."""
    piles = list(piles or [])
    for rname, n, (tmin, tmax) in specs:
        for _ in range(n):
            p = sample(cand, rng, ship, dmin, others=[(x, y) for _, x, y in piles], spacing=spacing,
                       region=None if rname is None else regions[rname])
            if p is None:
                raise Retry("no pile spot")
            piles.append((rt(rng, tmin, tmax), p[0], p[1]))
    return piles


def fit_total(piles, lo=60, hi=220):
    tot = sum(t for t, _, _ in piles)
    if tot > hi:
        s = hi / tot
        return [(max(5, int(round(t * s))), x, y) for t, x, y in piles]
    if tot < lo:
        s = lo / tot
        return [(min(80, int(round(t * s))), x, y) for t, x, y in piles]
    return piles


def flip_all(rng, land, regions):
    fx, fy = rng.random() < 0.5, rng.random() < 0.5
    def f(a):
        if fx:
            a = a[:, ::-1]
        if fy:
            a = a[::-1, :]
        return np.ascontiguousarray(a)
    return f(land), {k: f(v) for k, v in regions.items()}


def open_setup(rng, ship_box=None):
    w = World()
    ship = pick_ship(w.mask(), rng, rect_mask(*ship_box) if ship_box else None)
    w.zone(ship[0], ship[1] + 18, 70)
    return w, ship, cand_piles(w.mask(), ship)


def zone_piles(w, piles):
    for _, x, y in piles:
        w.zone(x, y, 17)


# ------------------------------------------------------------------ open search 01-05
def i01(rng):
    w, ship, cand = open_setup(rng, (60, 40, 740, 540))
    piles = place(cand, rng, ship, [(None, rt(rng, 6, 8), (12, 35))], spacing=150, dmin=200)
    zone_piles(w, piles)
    w.scatter(rng, rt(rng, 12, 18), ["rock", "rock", "pond", "stub"])
    return w.mask(), ship, piles


def i02(rng):
    w, ship, cand = open_setup(rng, (250, 180, 550, 420))
    rim = ~rect_mask(80, 80, 720, 520)
    piles = place(cand, rng, ship, [("rim", rt(rng, 6, 8), (15, 40))], {"rim": rim}, spacing=170, dmin=200)
    zone_piles(w, piles)
    w.scatter(rng, rt(rng, 8, 14), ["rock", "rock", "pond", "stub"], box=(100, 100, 700, 500))
    return w.mask(), ship, piles


def cluster(w, rng, px, py):
    k = rt(rng, 3, 4)
    R = 62 if k == 3 else 72
    r = rng.uniform(17, 21)
    step = 2 * math.asin(min(1.0, (2 * r * 1.06 + 50) / (2 * R)))
    opp = rng.uniform(0, 2 * math.pi)
    spots = [(px + R * math.cos(opp + (i - (k - 1) / 2) * step), py + R * math.sin(opp + (i - (k - 1) / 2) * step))
             for i in range(k)]
    cs = [(x, y, r * 1.06) for x, y in spots]
    if not w.fits(cs):
        raise Retry("cluster does not fit")
    for x, y in spots:
        w.draw(("poly", blob(x, y, r, rng), 0, [(x, y, r * 1.06)]))


def i03(rng):
    cx, cy = rng.random() < .5, rng.random() < .5
    box = (60 if cx else 540, 40 if cy else 360, 260 if cx else 740, 180 if cy else 540)
    w, ship, cand = open_setup(rng, box)
    nh, no = rt(rng, 4, 5), rt(rng, 1, 2)
    piles = place(cand, rng, ship, [("in", nh, (15, 45))], {"in": rect_mask(105, 105, 695, 495)},
                  spacing=190, dmin=260)
    piles = place(cand, rng, ship, [(None, no, (15, 45))], spacing=190, dmin=260, piles=piles)
    for _, x, y in piles[:nh]:
        cluster(w, rng, x, y)
    zone_piles(w, piles)
    w.scatter(rng, rt(rng, 4, 8), ["rock", "stub", "pond"])
    return w.mask(), ship, piles


def i04(rng):
    w, ship, cand = open_setup(rng, (60, 40, 740, 540))
    piles = place(cand, rng, ship, [(None, rt(rng, 5, 7), (15, 45))], spacing=150, dmin=200)
    zone_piles(w, piles)
    w.scatter(rng, rt(rng, 20, 30), ["pond"], tries=80)
    return w.mask(), ship, piles


def i05(rng):
    left, top = rng.random() < .5, rng.random() < .5
    box = (60 if left else 660, 30 if top else 480, 140 if left else 740, 90 if top else 540)
    w, ship, cand = open_setup(rng, box)
    far = rect_mask(400, 0, 800, 600) if left else rect_mask(0, 0, 400, 600)
    far = far & (rect_mask(0, 300, 800, 600) if top else rect_mask(0, 0, 800, 300))
    piles = place(cand, rng, ship, [("far", 1, (30, 50))], {"far": far}, spacing=150, dmin=420)
    piles = place(cand, rng, ship, [(None, rt(rng, 4, 6), (12, 40))], spacing=150, dmin=220, piles=piles)
    zone_piles(w, piles)
    w.scatter(rng, rt(rng, 8, 12), ["stub", "stub", "rock", "pond"])
    return w.mask(), ship, piles


# ------------------------------------------------------------------ pile dynamics 06-10
def i06(rng):
    w, ship, cand = open_setup(rng, (60, 40, 740, 540))
    piles = place(cand, rng, ship, [(None, rt(rng, 10, 12), (6, 10))], spacing=110, dmin=120)
    zone_piles(w, piles)
    w.scatter(rng, rt(rng, 6, 10), ["rock", "pond", "stub"])
    return w.mask(), ship, piles


def i07(rng):
    w, ship, cand = open_setup(rng, (60, 40, 740, 540))
    piles = place(cand, rng, ship, [(None, 1, (68, 80))], spacing=100, dmin=380)
    big = piles[0]
    others = []
    for _ in range(rt(rng, 6, 9)):
        p = sample(cand, rng, ship, 90, 340, others=[(x, y) for _, x, y in piles] + [(big[1], big[2])],
                   spacing=110)
        if p is None:
            raise Retry("decoy")
        piles.append((rt(rng, 5, 12), p[0], p[1]))
    zone_piles(w, piles)
    w.scatter(rng, rt(rng, 6, 10), ["rock", "pond", "stub"])
    return w.mask(), ship, piles


def i08(rng):
    w, ship, cand = open_setup(rng, (60, 40, 740, 540))
    piles = place(cand, rng, ship, [(None, rt(rng, 7, 9), (5, 5))], spacing=120, dmin=100)
    d = np.array([math.hypot(x - ship[0], y - ship[1] - 30) for _, x, y in piles])
    frac = np.clip((d - 100) / max(1.0, d.max() - 100), 0, 1)
    if rng.random() < 0.35:
        frac = 1 - frac
    out = []
    for f, (_, x, y) in zip(frac, piles):
        t = 5 + 75 * f ** 1.1
        t = rng.integers(5, 81) if rng.random() < 0.2 else t + rng.normal(0, 5)
        out.append((int(np.clip(round(t), 5, 80)), x, y))
    piles = fit_total(out)
    zone_piles(w, piles)
    w.scatter(rng, rt(rng, 8, 12), ["rock", "pond", "stub"])
    return w.mask(), ship, piles


def i09(rng):
    w, ship, cand = open_setup(rng, (60, 40, 740, 540))
    cc = sample(cand, rng, ship, 190, 330)
    if cc is None:
        raise Retry("cluster centre")
    piles = []
    for _ in range(rt(rng, 5, 7)):
        p = sample(cand, rng, ship, 100, others=[(x, y) for _, x, y in piles], spacing=65, disc=(cc[0], cc[1], 115))
        if p is None:
            raise Retry("cluster pile")
        piles.append((rt(rng, 8, 18), p[0], p[1]))
    rich = sample(cand, rng, ship, 420, others=[cc], spacing=300)
    if rich is None:
        raise Retry("rich pile")
    piles.append((rt(rng, 70, 80), rich[0], rich[1]))
    zone_piles(w, piles)
    w.scatter(rng, rt(rng, 6, 9), ["rock", "pond", "stub"])
    return w.mask(), ship, piles


def i10(rng):
    w, ship, cand = open_setup(rng, (330, 230, 470, 370))
    rx, ry = rng.uniform(290, 330), rng.uniform(200, 235)
    n = rt(rng, 9, 12)
    a0 = rng.uniform(0, 2 * math.pi)
    piles = []
    for i in range(n):
        a = a0 + 2 * math.pi * i / n + rng.uniform(-0.12, 0.12)
        tx, ty = 400 + rx * math.cos(a), 300 + ry * math.sin(a)
        p = sample(cand, rng, ship, 100, others=[(x, y) for _, x, y in piles], spacing=70, disc=(tx, ty, 45))
        if p is None:
            raise Retry("ring pile")
        piles.append((rt(rng, 8, 20), p[0], p[1]))
    zone_piles(w, piles)
    w.scatter(rng, rt(rng, 6, 10), ["rock", "pond", "stub"], box=(120, 100, 680, 500))
    return w.mask(), ship, piles


# ------------------------------------------------------------------ mixed 11-15
def maze(w, x0, y0, cols, rows, rng, exits):
    P, Wd, C = 64, 14, 50
    ex1, ey1 = x0 + cols * P + Wd, y0 + rows * P + Wd
    w.d.rectangle([x0, y0, ex1 - 1, ey1 - 1], fill=0)

    def org(i, j):
        return x0 + Wd + i * P, y0 + Wd + j * P

    def carve_cell(i, j):
        ox, oy = org(i, j)
        w.d.rectangle([ox, oy, ox + C - 1, oy + C - 1], fill=255)

    def carve_link(a, b):
        (ox, oy), (px, py) = org(*a), org(*b)
        w.d.rectangle([min(ox, px), min(oy, py), max(ox, px) + C - 1, max(oy, py) + C - 1], fill=255)

    for i in range(cols):
        for j in range(rows):
            carve_cell(i, j)
    seen = {(0, 0)}
    stack = [(0, 0)]
    links = set()
    while stack:
        i, j = stack[-1]
        nb = [(i + a, j + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= i + a < cols and 0 <= j + b < rows and (i + a, j + b) not in seen]
        if not nb:
            stack.pop()
            continue
        n = nb[int(rng.integers(len(nb)))]
        seen.add(n)
        links.add(((i, j), n))
        stack.append(n)
    for a, b in links:
        carve_link(a, b)
    for _ in range(rt(rng, 3, 6)):
        i, j = rt(rng, 0, cols - 2), rt(rng, 0, rows - 2)
        carve_link((i, j), (i + 1, j) if rng.random() < .5 else (i, j + 1))
    for side, k in exits:
        if side == "S":
            ox, oy = org(k, rows - 1)
            w.d.rectangle([ox, oy, ox + C - 1, ey1 - 1], fill=255)
        elif side == "E":
            ox, oy = org(cols - 1, k)
            w.d.rectangle([ox, oy, ex1 - 1, oy + C - 1], fill=255)
    return ex1, ey1


def river_pts(cx, amp, ph, per=95):
    return [(cx + amp * math.sin(y / per + ph), y) for y in range(-30, 640, 20)]


def bridge(w, pts, yb, half=85, wd=56):
    x = float(np.interp(yb, [p[1] for p in pts], [p[0] for p in pts]))
    w.d.line([(x - half, yb), (x + half, yb)], fill=255, width=wd)
    return x


def i11(rng):
    w = World()
    cols, rows = rt(rng, 5, 6), 4
    mx, my = maze(w, 0, 0, cols, rows, rng, [("S", rt(rng, 0, cols - 1)), ("E", rt(rng, 0, rows - 1))])
    w.reg_rect(0, 0, mx, my)
    cx, amp = rng.uniform(560, 600), rng.uniform(25, 45)
    pts = river_pts(cx, amp, rng.uniform(0, 6))
    w.wline(pts, 26)
    yb = rng.uniform(330, 500)
    xb = bridge(w, pts, yb)
    w.zone(xb, yb, 32)
    w.scatter(rng, rt(rng, 2, 5), ["rock", "pond", "stub"], box=(20, 20, cx - 90, 580))
    regions = {"maze": rect_mask(0, 0, mx, my), "east": rect_mask(cx + 60, 0, 800, 600),
               "field": rect_mask(0, 0, cx - 40, 600) & ~rect_mask(0, 0, mx, my),
               "ship": rect_mask(60, 330, cx - 120, 570)}
    land, regions = flip_all(rng, w.mask(), regions)
    land = fill_thin(land)
    ship = pick_ship(land, rng, regions["ship"])
    cand = cand_piles(land, ship)
    piles = place(cand, rng, ship, [("maze", 2, (30, 55)), ("field", 2, (10, 25)), ("east", rt(rng, 2, 3), (15, 40))],
                  regions, spacing=100, dmin=120)
    return land, ship, piles


def i12(rng):
    w = World()
    tx0, ty0 = 24, rng.uniform(100, 140)
    tx1, ty1 = tx0 + rng.uniform(190, 220), ty0 + rng.uniform(300, 340)
    lcx, lcy = rng.uniform(450, 480), rng.uniform(270, 330)
    rx, ry = rng.uniform(120, 150), rng.uniform(110, 150)
    w.d.ellipse([lcx - rx, lcy - ry, lcx + rx, lcy + ry], fill=0)
    for t in np.linspace(0, 2 * math.pi, 24, endpoint=False):
        w.zone(lcx + (rx - 25) * math.cos(t), lcy + (ry - 25) * math.sin(t), 28)
    w.zone(lcx, lcy, 70)
    # town walls
    t = 14
    gy = rng.uniform(ty0 + 50, ty1 - 110)
    walls = [(tx0, ty0, tx1, ty0 + t), (tx0, ty1 - t, tx1, ty1), (tx0, ty0, tx0 + t, ty1), (tx1 - t, ty0, tx1, ty1)]
    for r in walls:
        w.d.rectangle(list(r), fill=0)
    for r in walls:
        n = max(r[2] - r[0], r[3] - r[1])
        for s in np.arange(0, n + 1, 8):
            if r[2] - r[0] > r[3] - r[1]:
                w.zone(r[0] + s, (r[1] + r[3]) / 2, 8)
            else:
                w.zone((r[0] + r[2]) / 2, r[1] + s, 8)
    gates = [("E", gy)]
    other = ["N", "S"][int(rng.integers(2))]
    gates.append((other, rng.uniform(tx0 + 50, tx1 - 110)))
    for side, pos in gates:
        if side == "E":
            w.d.rectangle([tx1 - t - 1, pos, tx1, pos + 56], fill=255)
            w.zone(tx1 - t - 24, pos + 28, 30)
        elif side == "N":
            w.d.rectangle([pos, ty0 - 1, pos + 56, ty0 + t], fill=255)
            w.zone(pos + 28, ty0 + t + 24, 30)
        else:
            w.d.rectangle([pos, ty1 - t - 1, pos + 56, ty1], fill=255)
            w.zone(pos + 28, ty1 - t - 24, 30)
    w.scatter(rng, rt(rng, 3, 5), ["block"], box=(tx0 + t + 30, ty0 + t + 30, tx1 - t - 30, ty1 - t - 30), tries=100)
    lake = np.zeros((H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W]
    lake = ((xx - lcx) / rx) ** 2 + ((yy - lcy) / ry) ** 2 <= 1
    shore = (((xx - lcx) / (rx + 90)) ** 2 + ((yy - lcy) / (ry + 90)) ** 2 <= 1) & ~lake
    regions = {"town": rect_mask(tx0 + t, ty0 + t, tx1 - t, ty1 - t), "shore": shore,
               "east": rect_mask(lcx + rx + 40, 0, 800, 600), "ship": rect_mask(lcx + rx + 40, 40, 780, 560)}
    land, regions = flip_all(rng, w.mask(), regions)
    land = fill_thin(land)
    ship = pick_ship(land, rng, regions["ship"])
    cand = cand_piles(land, ship)
    piles = place(cand, rng, ship, [("town", rt(rng, 2, 3), (30, 60)), ("shore", rt(rng, 2, 3), (12, 25)),
                                    ("east", 1, (10, 20))], regions, spacing=100, dmin=120)
    return land, ship, piles


def bsp(x0, y0, x1, y1, rng, depth, walls):
    wd, ht = x1 - x0, y1 - y0
    can_v, can_h = wd >= 2 * 160 + 14, ht >= 2 * 160 + 14
    if depth == 0 or not (can_v or can_h) or (depth < 3 and rng.random() < 0.2):
        return
    vert = can_v and (not can_h or wd >= ht)
    if vert:
        c = int(rng.uniform(x0 + 160 + 7, x1 - 160 - 7))
        walls.append(["v", c, y0, y1])
        bsp(x0, y0, c, y1, rng, depth - 1, walls)
        bsp(c, y0, x1, y1, rng, depth - 1, walls)
    else:
        c = int(rng.uniform(y0 + 160 + 7, y1 - 160 - 7))
        walls.append(["h", c, x0, x1])
        bsp(x0, y0, x1, c, rng, depth - 1, walls)
        bsp(x0, c, x1, y1, rng, depth - 1, walls)


def i14(rng):
    xr = int(rng.uniform(380, 430))
    n = noise(rng, 28)
    xx = np.arange(W)[None, :]
    coast = (xx - xr) + 85 * n < rng.uniform(210, 260)
    land = cleanup(coast)
    w = World(mask=land)
    walls = []
    bsp(0, 0, xr, H, rng, 3, walls)
    walls.append(["v", xr, 0, H])
    for wl in walls:
        o, c, a, b = wl
        junc = [w2[1] for w2 in walls if w2[0] != o and a < w2[1] < b and (abs(w2[2] - c) < 2 or abs(w2[3] - c) < 2)]
        pos = None
        for _ in range(40):
            p = rng.uniform(a + 60, b - 60)
            if all(abs(p - j) > 62 for j in junc):
                pos = p
                break
        if pos is None:
            raise Retry("no door spot")
        wl.append(pos)
        wl.append(len(junc))
    doors = []
    for o, c, a, b, pos, _ in walls:
        if o == "v":
            w.d.rectangle([c - 7, a, c + 6, b], fill=0)
        else:
            w.d.rectangle([a, c - 7, b, c + 6], fill=0)
    for o, c, a, b, pos, _ in walls:
        if o == "v":
            w.d.rectangle([c - 8, pos - 28, c + 7, pos + 28], fill=255)
        else:
            w.d.rectangle([pos - 28, c - 8, pos + 28, c + 7], fill=255)
    # a second door in the east boundary wall
    if rng.random() < 0.7:
        p = rng.uniform(60, H - 60)
        w.d.rectangle([xr - 8, p - 28, xr + 7, p + 28], fill=255)
    regions = {"rooms": rect_mask(0, 0, xr - 10, H), "coast": rect_mask(xr + 50, 0, 800, 600),
               "ship": rect_mask(0, 0, xr - 10, H)}
    land, regions = flip_all(rng, w.mask(), regions)
    land = fill_thin(land)
    ship = pick_ship(land, rng, regions["ship"])
    cand = cand_piles(land, ship)
    piles = place(cand, rng, ship, [("rooms", 2, (30, 55)), ("coast", rt(rng, 3, 4), (12, 35))],
                  regions, spacing=100, dmin=200)
    return land, ship, piles


def i13(rng):
    w = World()
    cx, amp = rng.uniform(250, 320), rng.uniform(25, 40)
    pts = river_pts(cx, amp, rng.uniform(0, 6))
    w.wline(pts, 26)
    for _ in range(rt(rng, 1, 2)):
        yb = rng.uniform(130, 470)
        xb = bridge(w, pts, yb)
        w.zone(xb + 45, yb, 30)
        w.zone(xb - 45, yb, 30)
    ex = cx + amp + 70
    w.scatter(rng, rt(rng, 20, 28), ["pond", "pond", "rock"], box=(ex, 20, 780, 580), tries=80)
    regions = {"west": rect_mask(0, 0, cx - amp - 30, 600), "forest": rect_mask(ex, 0, 800, 600),
               "ship": rect_mask(30, 40, cx - amp - 80, 560)}
    land, regions = flip_all(rng, w.mask(), regions)
    land = fill_thin(land)
    ship = pick_ship(land, rng, regions["ship"])
    cand = cand_piles(land, ship)
    piles = place(cand, rng, ship, [("west", rt(rng, 1, 2), (10, 20)), ("forest", rt(rng, 4, 5), (20, 50))],
                  regions, spacing=100, dmin=100)
    return land, ship, piles


def i15(rng):
    yc = rng.uniform(130, 470)
    w = World(land=False)
    w.d.ellipse([10, yc - 70, 190, yc + 70], fill=255)
    p1x = rng.uniform(190, 240)
    y2 = yc + rng.choice([-1, 1]) * rng.uniform(120, 220)
    y2 = float(np.clip(y2, 90, 510))
    p3x = rng.uniform(290, 340)
    w.wline([(110, yc), (p1x, yc), (p1x, y2), (p3x, y2)], 56, reg=False, val=255)
    isl = []
    for _ in range(300):
        if len(isl) >= rt(rng, 5, 7):
            break
        x, y, r = rng.uniform(430, 730), rng.uniform(75, 525), rng.uniform(45, 80)
        if all(math.hypot(x - a, y - b) - r - c > 70 for a, b, c in isl):
            isl.append((x, y, r))
    if len(isl) < 4:
        raise Retry("islands")
    for x, y, r in isl:
        w.d.ellipse([x - r, y - r * rng.uniform(.75, 1), x + r, y + r * rng.uniform(.75, 1)], fill=255)
    nodes = [(p3x, y2)] + [(x, y) for x, y, _ in isl]
    done = {0}
    while len(done) < len(nodes):
        best = min(((math.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1]) * rng.uniform(.9, 1.2), i, j)
                    for i in done for j in range(len(nodes)) if j not in done))
        _, i, j = best
        w.wline([nodes[i], nodes[j]], 52, reg=False, val=255)
        done.add(j)
    if rng.random() < .6:
        i, j = rng.choice(len(isl), 2, replace=False) + 1
        w.wline([nodes[i], nodes[j]], 52, reg=False, val=255)
    land = w.mask()
    f = ndi.gaussian_filter(land.astype(float), 7) + 0.15 * noise(rng, 20)
    land = f > 0.5
    regions = {"ship": rect_mask(40, yc - 60, 170, yc + 60)}
    land, regions = flip_all(rng, land, regions)
    land = fill_thin(land)
    ship = pick_ship(land, rng, regions["ship"])
    cand = cand_piles(land, ship)
    piles = place(cand, rng, ship, [(None, rt(rng, 5, 7), (15, 50))], spacing=100, dmin=250)
    return land, ship, piles


IDEAS = {
    1: ("Wide open field: piles spread far apart around the whole map among a few small rocks and ponds", i01),
    2: ("Open field with the ship near the middle and the piles on a far rim around the edges", i02),
    3: ("Open field where most piles are hidden behind small rock clusters, plus a couple in the open", i03),
    4: ("Dotted field of many tiny ponds with a few far-apart piles", i04),
    5: ("Sparse rocks and wall stubs with the ship in a corner and piles scattered far away", i05),
    6: ("Many tiny piles (5-10 tomatoes) that empty fast, scattered widely over an open field", i06),
    7: ("One big far pile among many small decoy piles", i07),
    8: ("Piles of very different sizes at different distances (mostly richer the farther away)", i08),
    9: ("A cluster of small piles in one region versus a single rich pile far away", i09),
    10: ("A ring of small piles around the whole map with the ship in the middle", i10),
    11: ("Maze quarter, open field and a river with a bridge, piles in all three", i11),
    12: ("Walled town with gates on one side of a lake, the ship on the far shore", i12),
    13: ("River crossing by bridge into a forest of small ponds", i13),
    14: ("Rooms with doors on one half and a natural bay coastline on the other", i14),
    15: ("A corridor leading out into an archipelago of islands joined by isthmuses", i15),
}


# ------------------------------------------------------------------ output
def write_map(outdir, name, desc, land, ship, piles):
    arr = np.zeros((H, W, 3), np.uint8)
    arr[:] = WATER
    arr[land] = LAND
    Image.fromarray(arr).save(os.path.join(outdir, name + ".png"))
    with open(os.path.join(outdir, name + ".map"), "w", encoding="utf-8") as f:
        f.write("# Greeps map: %s\n" % desc)
        f.write("image %s.png\nship %d %d\n" % (name, ship[0], ship[1]))
        for t, x, y in piles:
            f.write("pile %d %d %d\n" % (t, x, y))


def generate(outdir, name, idea, key):
    from tools.check_maps import check
    desc, fn = IDEAS[idea]
    last = ""
    for attempt in range(MAX_ATTEMPTS):
        rng = np.random.default_rng([*key, attempt])
        try:
            land, ship, piles = fn(rng)
        except Retry as e:
            last = str(e)
            continue
        piles = fit_total(piles)
        land = fill_thin(land)
        if not 2 <= len(piles) <= 12:
            last = "pile count"
            continue
        write_map(outdir, name, desc, land, ship, piles)
        _, _, errors, warnings = check(os.path.join(outdir, name + ".map"))
        if not errors and not warnings:
            print("%s ship %s piles %d tomatoes %d (attempt %d)" % (name, ship, len(piles), sum(p[0] for p in piles), attempt))
            return
        last = "; ".join(errors + warnings)
    raise RuntimeError("could not generate %s: %s" % (name, last))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("maps", "training"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for idea in sorted(IDEAS):
        if a.variants <= 0:
            generate(a.out, "variety_%02d" % idea, idea, [BASE_SEED, idea, 0])
        else:
            for k in range(1, a.variants + 1):
                generate(a.out, "variety_%02d_v%03d" % (idea, k), idea, [a.seed, idea, k])


if __name__ == "__main__":
    main()
