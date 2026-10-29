"""Generates built_01 .. built_15 (built structures: mazes, rooms, walls). Run from repo root.

  python maps/training/generators/built.py
      rewrites the 15 hand-designed originals maps/training/built_01..15
  python maps/training/generators/built.py --variants N --seed S --out DIR
      writes N randomized variants of each of the 15 ideas as DIR/built_NN_vKKK.png/.map
      (deterministic given seed, idea and variant number; each is validated with
      tools.check_maps and resampled if it fails)
"""
import argparse
import os
import sys
from collections import deque
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join("maps", "training")
W, H = 800, 600
WATER = (40, 70, 160)
LAND = (235, 215, 190)
COUNT_OPTS = [10, 12, 16, 20, 30, 40, 50]


class Resample(Exception):
    """This random draw is unusable; try another seed."""


def ri(rng, a, b):
    return int(rng.integers(a, b + 1))


class Canvas:
    def __init__(self, land=False):
        self.m = np.full((H, W), land, dtype=bool)  # True = land

    def land(self, x0, y0, x1, y1):
        self.m[max(0, int(y0)):min(H, int(y1)), max(0, int(x0)):min(W, int(x1))] = True

    def water(self, x0, y0, x1, y1):
        self.m[max(0, int(y0)):min(H, int(y1)), max(0, int(x0)):min(W, int(x1))] = False

    def hwall(self, y, x0, x1, th=16, gaps=()):
        """Water wall centred on y from x0..x1 with gaps [(centre, width)]."""
        self.water(x0, y - th / 2, x1, y + th / 2)
        for c, w in gaps:
            self.land(c - w / 2, y - th / 2, c + w / 2, y + th / 2)

    def vwall(self, x, y0, y1, th=16, gaps=()):
        self.water(x - th / 2, y0, x + th / 2, y1)
        for c, w in gaps:
            self.land(x - th / 2, c - w / 2, x + th / 2, c + w / 2)

    def frame(self, x0, y0, x1, y1, th=16):
        self.water(x0, y0, x1, y0 + th)
        self.water(x0, y1 - th, x1, y1)
        self.water(x0, y0, x0 + th, y1)
        self.water(x1 - th, y0, x1, y1)


def maze(cv, cols, rows, pitch, wall, ox, oy, rng, braid=0.0):
    """DFS maze carved into water. Returns cell-centre function."""
    c = pitch - wall

    def cell(i, j):
        return ox + wall + i * pitch, oy + wall + j * pitch

    for j in range(rows):
        for i in range(cols):
            x, y = cell(i, j)
            cv.land(x, y, x + c, y + c)
    seen = {(0, 0)}
    stack = [(0, 0)]
    edges = set()
    while stack:
        i, j = stack[-1]
        nb = [(i + a, j + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= i + a < cols and 0 <= j + b < rows and (i + a, j + b) not in seen]
        if not nb:
            stack.pop()
            continue
        n = nb[rng.integers(len(nb))]
        seen.add(n)
        edges.add(((i, j), n))
        stack.append(n)
    allpairs = [((i, j), (i + 1, j)) for i in range(cols - 1) for j in range(rows)] + \
               [((i, j), (i, j + 1)) for i in range(cols) for j in range(rows - 1)]
    es = {frozenset(e) for e in edges}
    for p in allpairs:
        if frozenset(p) in es or rng.random() < braid:
            (a, b), (a2, b2) = p
            x, y = cell(a, b)
            if a2 > a:
                cv.land(x + c, y, x + pitch, y + c)
            else:
                cv.land(x, y + c, x + c, y + pitch)
    return cell


def bfs_dist(mask, start, erode):
    """Coarse BFS (5px grid) over land eroded by `erode` px radius."""
    r = erode
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disc = xx * xx + yy * yy <= r * r
    er = binary_erosion(mask, structure=disc, border_value=0)
    g = 5
    gh, gw = H // g, W // g
    free = er[g // 2::g, g // 2::g][:gh, :gw]
    sx, sy = start[0] // g, start[1] // g
    if not free[sy, sx]:
        best = None
        for y in range(gh):
            for x in range(gw):
                if free[y, x]:
                    d = (x - sx) ** 2 + (y - sy) ** 2
                    if best is None or d < best[0]:
                        best = (d, x, y)
        if best is None:
            raise Resample("no free cell")
        sx, sy = best[1], best[2]
    dist = -np.ones((gh, gw), dtype=int)
    dist[sy, sx] = 0
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + a, y + b
            if 0 <= nx < gw and 0 <= ny < gh and free[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = dist[y, x] + 1
                q.append((nx, ny))
    return dist, g


def auto_piles(mask, ship, quantiles, counts, rng, erode=10):
    start = (ship[0], ship[1] + 30)
    walk, g = bfs_dist(mask, start, erode)
    r = 17
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    er17 = binary_erosion(mask, structure=(xx * xx + yy * yy <= r * r), border_value=0)
    mx = walk.max()
    piles = []
    for q, n in zip(quantiles, counts):
        target = q * mx
        for tol in (0.03, 0.06, 0.1, 0.2):
            ys, xs = np.where((walk >= 0) & (np.abs(walk - target) <= tol * mx))
            cand = [(x * g + g // 2, y * g + g // 2) for x, y in zip(xs, ys)
                    if er17[y * g + g // 2, x * g + g // 2]]
            cand = [p for p in cand if np.hypot(p[0] - ship[0], p[1] - ship[1]) > 100 and
                    all(np.hypot(p[0] - o[1], p[1] - o[2]) > 110 for o in piles)]
            if cand:
                p = cand[rng.integers(len(cand))]
                piles.append((n, p[0], p[1]))
                break
    return piles


def scatter_piles(mask, ship, counts, rng, allow=None, sep=110, erode=20):
    """Random piles on walkable (eroded) land, optionally restricted to the `allow` mask."""
    walk, g = bfs_dist(mask, (ship[0], ship[1] + 30), erode)
    r = 17
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    er17 = binary_erosion(mask, structure=(xx * xx + yy * yy <= r * r), border_value=0)
    ys, xs = np.where(walk >= 0)
    cand = [(int(x * g + g // 2), int(y * g + g // 2)) for x, y in zip(xs, ys)]
    cand = [p for p in cand if er17[p[1], p[0]] and (allow is None or allow[p[1], p[0]])
            and np.hypot(p[0] - ship[0], p[1] - ship[1]) > 100]
    order = rng.permutation(len(cand))
    piles = []
    for idx in order:
        if len(piles) >= len(counts):
            break
        p = cand[idx]
        if all(np.hypot(p[0] - o[1], p[1] - o[2]) > sep for o in piles):
            piles.append((counts[len(piles)], p[0], p[1]))
    return piles


def counts_rand(n, rng, lo=95, hi=200):
    for _ in range(200):
        c = sorted((int(rng.choice(COUNT_OPTS)) for _ in range(n)), reverse=True)
        if lo <= sum(c) <= hi:
            return c
    return [40, 30, 30, 20, 16, 12][:n]


def qs(n, rng):
    q = np.linspace(0.97, 0.25, n) + rng.uniform(-0.04, 0.04, n)
    q[0] = min(q[0], 0.99)
    return [float(v) for v in np.clip(q, 0.15, 0.99)]


def rand_auto(mask, ship, rng, nmin=4, nmax=6):
    n = ri(rng, nmin, nmax)
    return auto_piles(mask, ship, qs(n, rng), counts_rand(n, rng), rng, erode=20)


def flip(m, ship, piles, h, v):
    rx, ry = ship[0], ship[1] + 30
    if h:
        m = m[:, ::-1]
        rx = W - 1 - rx
        piles = [(n, W - 1 - x, y) for n, x, y in piles]
    if v:
        m = m[::-1, :]
        ry = H - 1 - ry
        piles = [(n, x, H - 1 - y) for n, x, y in piles]
    return np.ascontiguousarray(m), (rx, ry - 30), piles


def write_map(outdir, name, m, desc, ship, piles):
    os.makedirs(outdir, exist_ok=True)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = WATER
    img[m] = LAND
    Image.fromarray(img).save(os.path.join(outdir, name + ".png"))
    with open(os.path.join(outdir, name + ".map"), "w", encoding="utf-8") as f:
        f.write(f"# Greeps map: {desc}\nimage {name}.png\nship {ship[0]} {ship[1]}\n")
        for n, x, y in piles:
            f.write(f"pile {n} {x} {y}\n")


# ---------------------------------------------------------------- the 15 ideas
# Each g(rng, var) returns (land mask, description, ship, piles). var=False gives the
# hand-designed original; var=True draws a randomized member of the same family.

def maze_dims(rng, c, wall, margin):
    pitch = c + wall
    cols = (W - 2 * margin - wall) // pitch
    rows = (H - 2 * margin - wall) // pitch
    ox = (W - cols * pitch - wall) // 2
    oy = (H - rows * pitch - wall) // 2
    return pitch, cols, rows, ox, oy


def g01(rng, var):
    cv = Canvas()
    desc = "perfect maze, ship in bottom-left corner, piles at dead ends far along the unique path"
    if not var:
        cell = maze(cv, 11, 8, 70, 16, 4, 16, rng)
        sx, sy = cell(0, 7)
        ship = (sx + 27, sy + 8)
        piles = auto_piles(cv.m, ship, [0.98, 0.85, 0.65, 0.45, 0.25], [50, 30, 30, 20, 20], rng)
        return cv.m, desc, ship, piles
    c, wall = ri(rng, 47, 58), ri(rng, 11, 18)
    pitch, cols, rows, ox, oy = maze_dims(rng, c, wall, ri(rng, 4, 12))
    cell = maze(cv, cols, rows, pitch, wall, ox, oy, rng)
    sx, sy = cell(0, rows - 1)
    ship = (sx + c // 2, sy + 8)
    return cv.m, desc, ship, rand_auto(cv.m, ship, rng, 4, 6)


def g02(rng, var):
    cv = Canvas()
    desc = "dense braided maze with loops, ship in the middle, many routes to scattered piles"
    if not var:
        cell = maze(cv, 13, 9, 60, 14, 3, 23, rng, braid=0.12)
        sx, sy = cell(6, 4)
        ship = (sx + 23, sy + 6)
        piles = auto_piles(cv.m, ship, [0.95, 0.8, 0.7, 0.55, 0.4, 0.3], [30, 16, 40, 12, 16, 10], rng)
        return cv.m, desc, ship, piles
    c, wall = ri(rng, 47, 54), ri(rng, 10, 16)
    pitch, cols, rows, ox, oy = maze_dims(rng, c, wall, ri(rng, 3, 10))
    cell = maze(cv, cols, rows, pitch, wall, ox, oy, rng, braid=float(rng.uniform(0.08, 0.26)))
    sx, sy = cell(cols // 2 + ri(rng, -1, 1), rows // 2 + ri(rng, -1, 1))
    ship = (sx + c // 2, sy + 6)
    return cv.m, desc, ship, rand_auto(cv.m, ship, rng, 5, 7)


def g03(rng, var):
    cv = Canvas()
    cv.land(20, 20, 780, 580)
    desc = "grid of rooms joined by random doorways, ship in a room"
    if not var:
        xs, ys = [273, 527], [207, 393]
        for x in xs:
            segs = [(20, 207), (207, 393), (393, 580)]
            cv.vwall(x, 20, 580, 16, gaps=[(int(rng.integers(a + 40, b - 40)), 56) for a, b in segs if rng.random() < 0.7])
        for y in ys:
            segs = [(20, 273), (273, 527), (527, 780)]
            cv.hwall(y, 20, 780, 16, gaps=[(int(rng.integers(a + 40, b - 40)), 56) for a, b in segs if rng.random() < 0.7])
        ship = (110, 110)
        piles = auto_piles(cv.m, ship, [0.97, 0.75, 0.55, 0.4, 0.25], [40, 20, 30, 12, 20], rng)
        return cv.m, "3x3 grid of rooms joined by random doorways, ship in a corner room", ship, piles
    nx, ny = ri(rng, 2, 4), ri(rng, 2, 4)
    th, gw = ri(rng, 10, 18), ri(rng, 50, 64)

    def bounds(lo, hi, n):
        out = [lo]
        for k in range(1, n):
            out.append(int(lo + (hi - lo) * k / n + rng.uniform(-0.12, 0.12) * (hi - lo) / n))
        return out + [hi]

    xb, yb = bounds(20, 780, nx), bounds(20, 580, ny)
    edges = [('v', a, j, (a - 1, j), (a, j)) for a in range(1, nx) for j in range(ny)] + \
            [('h', b, i, (i, b - 1), (i, b)) for b in range(1, ny) for i in range(nx)]
    parent = {(i, j): (i, j) for i in range(nx) for j in range(ny)}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    pd = float(rng.uniform(0.2, 0.5))
    doors = []
    for k in rng.permutation(len(edges)):
        e = edges[k]
        ra, rb = find(e[3]), find(e[4])
        if ra != rb:
            parent[ra] = rb
            doors.append(e)
        elif rng.random() < pd:
            doors.append(e)
    mg = gw // 2 + th // 2 + 14
    vg = {a: [] for a in range(1, nx)}
    hg = {b: [] for b in range(1, ny)}
    for kind, w, s, _, _ in doors:
        lo, hi = (yb[s], yb[s + 1]) if kind == 'v' else (xb[s], xb[s + 1])
        if hi - mg <= lo + mg:
            raise Resample("room too small for a door")
        (vg if kind == 'v' else hg)[w].append((ri(rng, lo + mg, hi - mg), gw))
    for a in range(1, nx):
        cv.vwall(xb[a], 20, 580, th, gaps=vg[a])
    for b in range(1, ny):
        cv.hwall(yb[b], 20, 780, th, gaps=hg[b])
    i, j = ri(rng, 0, nx - 1), ri(rng, 0, ny - 1)
    ship = ((xb[i] + xb[i + 1]) // 2, (yb[j] + yb[j + 1]) // 2 - 15)
    return cv.m, desc, ship, rand_auto(cv.m, ship, rng, 4, 6)


def g04(rng, var):
    cv = Canvas()
    cv.land(20, 20, 780, 580)
    desc = "serpentine switchback corridor, walls alternately open at each end, best pile at the far end"
    if not var:
        for k, y in enumerate([105, 200, 295, 390, 485]):
            gx = 730 if k % 2 == 0 else 70
            cv.hwall(y, 20, 780, 16, gaps=[(gx, 64)])
        ship = (90, 45)
        piles = [(60, 650, 540), (30, 130, 440), (20, 650, 340), (30, 130, 250), (20, 650, 150)]
        return cv.m, desc, ship, piles
    th, gw = ri(rng, 10, 16), ri(rng, 54, 80)
    horiz = rng.random() < 0.6
    n = ri(rng, 3, 6) if horiz else ri(rng, 4, 7)
    lo_, hi_ = (20, 580) if horiz else (20, 780)
    cuts = [lo_ + int(round((k + 1) * (hi_ - lo_) / (n + 1))) + ri(rng, -6, 6) for k in range(n)]
    bounds = [lo_] + cuts + [hi_]
    span_lo, span_hi = (20, 780) if horiz else (20, 580)
    hi_c = span_hi - gw // 2 - 14
    lo_c = span_lo + gw // 2 + 14
    for k, p in enumerate(cuts):
        c = ri(rng, hi_c - 60, hi_c) if k % 2 == 0 else ri(rng, lo_c, lo_c + 60)
        if horiz:
            cv.hwall(p, 20, 780, th, gaps=[(c, gw)])
        else:
            cv.vwall(p, 20, 580, th, gaps=[(c, gw)])
    mid = [(bounds[b] + bounds[b + 1]) // 2 for b in range(n + 1)]
    # ship in the first band, piles in the later ones (last band = best)
    if horiz:
        ship = (ri(rng, 70, 300), mid[0] - 30 + ri(rng, 0, 8))
    else:
        ship = (mid[0], ri(rng, 40, 250))
    nb = ri(rng, 3, min(6, n))
    pick = sorted(set([n] + [int(b) for b in rng.permutation(np.arange(1, n))[:nb - 1]]))
    cnt = counts_rand(len(pick), rng)
    piles = []
    for b, cn in zip(reversed(pick), cnt):
        if horiz:
            piles.append((cn, ri(rng, 110, 690), mid[b]))
        else:
            piles.append((cn, mid[b], ri(rng, 90, 510)))
    return cv.m, desc, ship, piles


def g05(rng, var):
    cv = Canvas()
    cv.land(10, 10, 790, 590)
    desc = "nested rectangular rings with gates on alternating sides, treasure in the innermost chamber"
    if not var:
        gates = ['top', 'right', 'bottom', 'left']
        for k, (hw, hh) in enumerate([(340, 250), (270, 190), (200, 130), (130, 70)]):
            x0, y0, x1, y1 = 400 - hw, 300 - hh, 400 + hw, 300 + hh
            cv.frame(x0, y0, x1, y1, 14)
            g = gates[k]
            off = int(rng.integers(-60, 60))
            if g == 'top':
                cv.land(400 + off - 30, y0, 400 + off + 30, y0 + 14)
            elif g == 'bottom':
                cv.land(400 + off - 30, y1 - 14, 400 + off + 30, y1)
            elif g == 'left':
                cv.land(x0, 300 + off // 3 - 30, x0 + 14, 300 + off // 3 + 30)
            else:
                cv.land(x1 - 14, 300 + off // 3 - 30, x1, 300 + off // 3 + 30)
        ship = (35, 300)
        piles = [(50, 400, 300), (30, 300, 300), (30, 500, 300), (20, 400, 140), (20, 130, 520)]
        return cv.m, desc, ship, piles
    for _ in range(500):
        n = ri(rng, 3, 4)
        hw0, hh0 = ri(rng, 310, 340), ri(rng, 225, 242)
        sx_, sy_ = ri(rng, 60, 80), ri(rng, 56, 70)
        th, gw = ri(rng, 10, 14), ri(rng, 50, 68)
        if sx_ - th >= 48 and sy_ - th >= 48 and hh0 - (n - 1) * sy_ >= 52 and hw0 - (n - 1) * sx_ >= 100:
            break
    else:
        raise Resample("rings")
    rings = [(hw0 - k * sx_, hh0 - k * sy_) for k in range(n)]
    sides = ['top', 'right', 'bottom', 'left']
    prev = None
    for hw, hh in rings:
        x0, y0, x1, y1 = 400 - hw, 300 - hh, 400 + hw, 300 + hh
        cv.frame(x0, y0, x1, y1, th)
        g = str(rng.choice([s for s in sides if s != prev]))
        prev = g
        if g in ('top', 'bottom'):
            m = max(0, hw - th - 20 - gw // 2)
            c = 400 + ri(rng, -min(m, 180), min(m, 180))
            yy = y0 if g == 'top' else y1 - th
            cv.land(c - gw // 2, yy, c + gw // 2, yy + th)
        else:
            m = max(0, hh - th - 10 - gw // 2)
            c = 300 + ri(rng, -m, m)
            xx = x0 if g == 'left' else x1 - th
            cv.land(xx, c - gw // 2, xx + th, c + gw // 2)
    ship = ((10 + 400 - hw0) // 2, ri(rng, 150, 450))
    cnt = counts_rand(ri(rng, 4, 5), rng)
    piles = [(cnt[0], 400, 300)]
    for cn in cnt[1:]:
        for _ in range(200):
            k = ri(rng, 0, n - 2)
            mw = (rings[k][0] + rings[k + 1][0]) / 2
            mh = (rings[k][1] + rings[k + 1][1]) / 2
            u = rng.uniform(0, 2 * (2 * mw + 2 * mh))
            if u < 2 * mw:
                p = (400 - mw + u, 300 - mh)
            elif u < 2 * mw + 2 * mh:
                p = (400 + mw, 300 - mh + u - 2 * mw)
            elif u < 4 * mw + 2 * mh:
                p = (400 + mw - (u - 2 * mw - 2 * mh), 300 + mh)
            else:
                p = (400 - mw, 300 + mh - (u - 4 * mw - 2 * mh))
            p = (int(p[0]), int(p[1]))
            if np.hypot(p[0] - ship[0], p[1] - ship[1]) > 100 and \
                    all(np.hypot(p[0] - o[1], p[1] - o[2]) > 100 for o in piles):
                piles.append((cn, p[0], p[1]))
                break
    return cv.m, desc, ship, piles


def g06(rng, var):
    cv = Canvas(land=True)
    desc = "walled castle: outer curtain wall with one gate, courtyards, towers and an inner keep"
    if not var:
        cv.frame(90, 80, 710, 520, 18)
        cv.land(370, 502, 430, 520)  # main gate bottom
        cv.water(80, 70, 140, 130); cv.water(660, 70, 720, 130)
        cv.water(80, 470, 140, 530); cv.water(660, 470, 720, 530)
        cv.frame(280, 200, 520, 380, 16)
        cv.land(300, 200, 360, 216)  # keep gate top
        cv.vwall(400, 98, 200, 16, gaps=[(150, 50)])
        cv.hwall(450, 108, 300, 16, gaps=[(200, 50)])
        ship = (150, 555)
        piles = [(60, 400, 290), (30, 160, 130), (30, 640, 130), (20, 640, 440), (20, 220, 400), (10, 400, 140)]
        return cv.m, desc, ship, piles
    x0, x1 = ri(rng, 75, 110), ri(rng, 690, 725)
    y0, y1 = ri(rng, 65, 100), ri(rng, 490, 520)
    th, ts = ri(rng, 14, 20), ri(rng, 48, 70)
    cv.frame(x0, y0, x1, y1, th)
    gw = ri(rng, 52, 70)
    gx = ri(rng, x0 + 130, x1 - 130)
    cv.land(gx - gw // 2, y1 - th, gx + gw // 2, y1)
    # towers: square blocks straddling each corner
    cv.water(x0 - 10, y0 - 10, x0 - 10 + ts, y0 - 10 + ts)
    cv.water(x1 + 10 - ts, y0 - 10, x1 + 10, y0 - 10 + ts)
    cv.water(x0 - 10, y1 + 10 - ts, x0 - 10 + ts, y1 + 10)
    cv.water(x1 + 10 - ts, y1 + 10 - ts, x1 + 10, y1 + 10)
    kw, kh = ri(rng, 200, 270), ri(rng, 130, 190)
    kcx, kcy = 400 + ri(rng, -40, 40), 300 + ri(rng, -30, 30)
    kx0, ky0, kx1, ky1 = kcx - kw // 2, kcy - kh // 2, kcx + kw // 2, kcy + kh // 2
    kth = ri(rng, 12, 18)
    kg = ri(rng, 52, 64)
    side = str(rng.choice(['top', 'bottom', 'left', 'right']))
    if side in ('top', 'bottom'):
        gc = kcx + ri(rng, -40, 40)
    else:
        gc = kcy + ri(rng, -25, 25)
    # courtyard dividers (drawn before the keep so the gate is cut cleanly)
    kinds = [str(k) for k in rng.permutation(['T', 'B', 'L', 'R'])[:ri(rng, 1, 3)]]
    dg = ri(rng, 50, 58)
    dth = ri(rng, 10, 16)
    for kind in kinds:
        if kind in 'TB':
            x = ri(rng, kx0 + 40, kx1 - 40)
            if side == ('top' if kind == 'T' else 'bottom') and abs(x - gc) < kg // 2 + 50:
                continue
            if kind == 'B' and abs(x - gx) < 100:
                continue
            a, b = (y0 + th // 2, ky0) if kind == 'T' else (ky1, y1 - th // 2)
            if b - a < dg + 70:
                continue
            cv.vwall(x, a, b, dth, gaps=[(ri(rng, a + dg // 2 + 25, b - dg // 2 - 25), dg)])
        else:
            y = ri(rng, ky0 + 40, ky1 - 40)
            if side == ('left' if kind == 'L' else 'right') and abs(y - gc) < kg // 2 + 50:
                continue
            a, b = (x0 + th // 2, kx0) if kind == 'L' else (kx1, x1 - th // 2)
            if b - a < dg + 70:
                continue
            cv.hwall(y, a, b, dth, gaps=[(ri(rng, a + dg // 2 + 25, b - dg // 2 - 25), dg)])
    cv.frame(kx0, ky0, kx1, ky1, kth)
    if side == 'top':
        cv.land(gc - kg // 2, ky0, gc + kg // 2, ky0 + kth)
    elif side == 'bottom':
        cv.land(gc - kg // 2, ky1 - kth, gc + kg // 2, ky1)
    elif side == 'left':
        cv.land(kx0, gc - kg // 2, kx0 + kth, gc + kg // 2)
    else:
        cv.land(kx1 - kth, gc - kg // 2, kx1, gc + kg // 2)
    ship = (ri(rng, 70, 730), ri(rng, 543, 556))
    allow = np.zeros((H, W), dtype=bool)
    allow[y0 + th + 10:y1 - th - 10, x0 + th + 10:x1 - th - 10] = True
    allow[ky0 - 10:ky1 + 10, kx0 - 10:kx1 + 10] = False
    cnt = counts_rand(ri(rng, 5, 7), rng)
    piles = [(cnt[0], kcx, kcy)] + scatter_piles(cv.m, ship, cnt[1:], rng, allow=allow)
    return cv.m, desc, ship, piles


def cuts_rand(rng, lo, hi, n, mn):
    for _ in range(300):
        c = sorted(int(v) for v in rng.uniform(lo + mn, hi - mn, n - 1))
        pts = [lo] + c + [hi]
        if all(b - a >= mn for a, b in zip(pts[:-1], pts[1:])):
            return pts
    raise Resample("cuts")


def g07(rng, var):
    cv = Canvas()
    cv.land(20, 20, 780, 580)
    desc = "office floor plan: long central corridor with closed rooms behind single doors"
    if not var:
        xs = [20 + 152 * k for k in range(1, 5)]
        bounds = [20] + xs + [780]
        for y in (250, 350):
            doors = [(int(rng.integers(a + 45, b - 45)), 52) for a, b in zip(bounds[:-1], bounds[1:])
                     if rng.random() < 0.85]
            cv.hwall(y, 20, 780, 16, gaps=doors)
        for x in xs:
            for (y0, y1) in ((20, 242), (358, 580)):
                gaps = [(int(rng.integers(y0 + 40, y1 - 40)), 50)] if rng.random() < 0.4 else []
                cv.vwall(x, y0, y1, 16, gaps=gaps)
        ship = (60, 300)
        piles = auto_piles(cv.m, ship, [0.97, 0.85, 0.7, 0.55, 0.4, 0.3], [30, 40, 16, 20, 30, 12], rng)
        return cv.m, desc, ship, piles
    th, gw = ri(rng, 10, 16), ri(rng, 50, 58)
    ccw = ri(rng, 90, 130)
    cy = 300 + ri(rng, -40, 40)
    yt, yb = cy - ccw // 2, cy + ccw // 2
    mg = gw // 2 + th // 2 + 14
    pd = float(rng.uniform(0.85, 1.0))
    parts = {}
    for side, y in (('top', yt), ('bot', yb)):
        pts = cuts_rand(rng, 20, 780, ri(rng, 4, 7), 112)
        parts[side] = pts
        doors = [(ri(rng, a + mg, b - mg), gw) for a, b in zip(pts[:-1], pts[1:]) if rng.random() < pd]
        cv.hwall(y, 20, 780, th, gaps=doors)
    for side, (y0, y1) in (('top', (20, yt)), ('bot', (yb, 580))):
        for x in parts[side][1:-1]:
            gaps = []
            if rng.random() < 0.4:
                gaps = [(ri(rng, y0 + mg, y1 - mg), 50)]
            cv.vwall(x, y0, y1, th, gaps=gaps)
    ship = (ri(rng, 50, 80), cy - 15)
    return cv.m, desc, ship, rand_auto(cv.m, ship, rng, 5, 7)


def g08(rng, var):
    cv = Canvas(land=True)
    desc = "easy open hall of scattered square pillars, piles hidden between them"
    if not var:
        ship = (250, 300)
        for gy in range(1, 6):
            for gx in range(1, 8):
                x = 100 * gx + int(rng.integers(-15, 16))
                y = 100 * gy + int(rng.integers(-15, 16))
                if np.hypot(x - ship[0], y - ship[1] - 30) < 90:
                    continue
                s = int(rng.integers(36, 56))
                cv.water(x - s // 2, y - s // 2, x + s // 2, y + s // 2)
        piles = auto_piles(cv.m, ship, [0.97, 0.8, 0.6, 0.4, 0.3], [30, 20, 40, 12, 16], rng)
        return cv.m, desc, ship, piles
    spx, spy = ri(rng, 90, 135), ri(rng, 85, 125)
    nxp, nyp = (W - 80) // spx + 1, (H - 80) // spy + 1
    ox, oy = (W - (nxp - 1) * spx) // 2, (H - (nyp - 1) * spy) // 2
    jit = ri(rng, 10, 24)
    smin = ri(rng, 28, 40)
    smax = smin + ri(rng, 10, 22)
    drop = float(rng.uniform(0.0, 0.15))
    ship = (ri(rng, 120, 680), ri(rng, 80, 450))
    for gy in range(nyp):
        for gx in range(nxp):
            if rng.random() < drop:
                continue
            x = ox + gx * spx + ri(rng, -jit, jit)
            y = oy + gy * spy + ri(rng, -jit, jit)
            if np.hypot(x - ship[0], y - ship[1] - 30) < 95:
                continue
            w, h = ri(rng, smin, smax), ri(rng, smin, smax)
            cv.water(x - w // 2, y - h // 2, x + w // 2, y + h // 2)
    return cv.m, desc, ship, rand_auto(cv.m, ship, rng, 4, 6)


def spiral_pts(L, T, R, B, p, k, frac):
    l, t, r, b = L, T, R, B
    pts = [(l, t), (r, t), (r, b), (l, b)]
    for _ in range(1, k):
        t += p; pts.append((l, t))
        r -= p; pts.append((r, t))
        b -= p; pts.append((r, b))
        l += p; pts.append((l, b))
    t += p; pts.append((l, t))
    r -= p; pts.append((r, t))
    b -= p; pts.append((r, b))
    pts.append((r - int(frac * (r - l)), b))
    return pts


def g09(rng, var):
    cv = Canvas()
    desc = "square spiral corridor from the outside inwards, richest pile at the very centre"
    if not var:
        pts = [(60, 60), (740, 60), (740, 540), (60, 540), (60, 126), (674, 126), (674, 474), (126, 474),
               (126, 192), (608, 192), (608, 408), (192, 408), (192, 258), (542, 258), (542, 342), (300, 342)]
        hw = 25
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            cv.land(min(ax, bx) - hw, min(ay, by) - hw, max(ax, bx) + hw, max(ay, by) + hw)
        ship = (90, 35)
        piles = [(60, 300, 342), (30, 450, 258), (20, 400, 192), (20, 674, 300), (20, 400, 540)]
        return cv.m, desc, ship, piles
    hw = ri(rng, 23, 30)
    p = 2 * hw + ri(rng, 12, 22)
    L, T = ri(rng, 55, 72), ri(rng, 55, 72)
    R, B = W - L, H - T
    k = 1
    while B - T - 2 * (k + 1) * p >= p:
        k += 1
    pts = spiral_pts(L, T, R, B, p, k, float(rng.uniform(0.55, 0.9)))
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        cv.land(min(ax, bx) - hw, min(ay, by) - hw, max(ax, bx) + hw, max(ay, by) + hw)
    ship = (L + 30, T - 25)
    cnt = counts_rand(ri(rng, 4, 6), rng)
    piles = [(cnt[0], pts[-1][0], pts[-1][1])]
    segs = list(zip(pts[:-1], pts[1:]))
    lens = np.array([abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in segs], dtype=float)
    for cn in cnt[1:]:
        for _ in range(300):
            (a, b) = segs[int(rng.choice(len(segs), p=lens / lens.sum()))]
            f = rng.uniform(0.25, 0.75)
            q = (int(a[0] + f * (b[0] - a[0])), int(a[1] + f * (b[1] - a[1])))
            if np.hypot(q[0] - ship[0], q[1] - ship[1]) > 100 and \
                    all(np.hypot(q[0] - o[1], q[1] - o[2]) > 110 for o in piles):
                piles.append((cn, q[0], q[1]))
                break
    return cv.m, desc, ship, piles


def g10(rng, var):
    cv = Canvas()
    desc = "wide-corridor labyrinth of large cells, ship in the middle, long winding routes"
    if not var:
        cell = maze(cv, 7, 5, 100, 16, 42, 42, rng, braid=0.1)
        sx, sy = cell(3, 2)
        ship = (sx + 42, sy + 22)
        piles = auto_piles(cv.m, ship, [0.95, 0.8, 0.6, 0.45, 0.3], [40, 30, 30, 20, 16], rng)
        return cv.m, desc, ship, piles
    wall = ri(rng, 12, 20)
    c = ri(rng, 62, 100)
    pitch, cols, rows, ox, oy = maze_dims(rng, c, wall, ri(rng, 25, 45))
    cell = maze(cv, cols, rows, pitch, wall, ox, oy, rng, braid=float(rng.uniform(0.04, 0.2)))
    sx, sy = cell(cols // 2 + ri(rng, -1, 1), rows // 2 + ri(rng, -1, 1))
    ship = (sx + c // 2, sy + c // 2 - 20)
    return cv.m, desc, ship, rand_auto(cv.m, ship, rng, 4, 6)


def g11(rng, var):
    cv = Canvas()
    desc = "two interlocking combs: one pile in a dead-end tooth, most teeth are empty"
    if not var:
        cv.land(30, 30, 780, 90)
        cv.land(30, 510, 780, 570)
        cv.land(735, 30, 780, 570)
        tips = []
        for k in range(5):
            x = 70 + 136 * k
            L = int(rng.integers(230, 400))
            cv.land(x - 27, 90, x + 27, 90 + L)
            tips.append((x, 90 + L - 25))
            x = 138 + 136 * k
            L = int(rng.integers(230, 400))
            cv.land(x - 27, 510 - L, x + 27, 510)
            tips.append((x, 510 - L + 25))
        ship = (90, 35)
        idx = rng.permutation(len(tips))[:6]
        cnt = [40, 30, 16, 20, 30, 12]
        piles = [(cnt[i], tips[j][0], tips[j][1]) for i, j in enumerate(idx)]
        return cv.m, desc, ship, piles
    tb, bb = ri(rng, 50, 66), ri(rng, 50, 66)
    sp = ri(rng, 46, 60)
    tw, wl = ri(rng, 46, 56), ri(rng, 12, 18)
    s = 2 * (tw + wl)
    ytop, ybot = 30 + tb, 570 - bb
    cv.land(30, 30, 780, ytop)
    cv.land(30, ybot, 780, 570)
    cv.land(780 - sp, 30, 780, 570)
    right = 780 - sp - wl
    first = 30 + wl + tw // 2
    n = int((right - (first + s // 2 + tw // 2)) // s) + 1
    if n < 3:
        raise Resample("comb too narrow")
    slack = right - (first + s // 2 + tw // 2 + (n - 1) * s)
    first += ri(rng, 0, max(0, int(slack)))
    lmax = ybot - ytop - 22
    tips = []
    for k in range(n):
        x = first + s * k
        L = ri(rng, 170, lmax)
        cv.land(x - tw // 2, ytop, x + tw // 2, ytop + L)
        tips.append((x, ytop + L - 25))
        x = first + s // 2 + s * k
        L = ri(rng, 170, lmax)
        cv.land(x - tw // 2, ybot - L, x + tw // 2, ybot)
        tips.append((x, ybot - L + 25))
    ship = (ri(rng, 70, 350), 30 + tb // 2 - 25)
    np_ = ri(rng, 3, 6)
    cnt = counts_rand(np_, rng)
    idx = rng.permutation(len(tips))[:np_]
    piles = [(cnt[i], tips[j][0], tips[j][1]) for i, j in enumerate(idx)]
    return cv.m, desc, ship, piles


def g12(rng, var):
    cv = Canvas(land=True)
    desc = "fenced compounds each with one gate; every pile is inside a pen"
    if not var:
        comps = [(50, 50, 250, 200, 'bottom'), (330, 40, 520, 180, 'left'), (600, 60, 760, 260, 'bottom'),
                 (60, 330, 260, 500, 'right'), (560, 360, 750, 540, 'top')]
        piles = []
        cnts = [30, 12, 40, 20, 50]
        for (x0, y0, x1, y1, g), n in zip(comps, cnts):
            cv.frame(x0, y0, x1, y1, 14)
            if g == 'bottom':
                c = (x0 + x1) // 2 + int(rng.integers(-40, 40)); cv.land(c - 30, y1 - 14, c + 30, y1)
            elif g == 'top':
                c = (x0 + x1) // 2 + int(rng.integers(-40, 40)); cv.land(c - 30, y0, c + 30, y0 + 14)
            elif g == 'left':
                c = (y0 + y1) // 2; cv.land(x0, c - 30, x0 + 14, c + 30)
            else:
                c = (y0 + y1) // 2; cv.land(x1 - 14, c - 30, x1, c + 30)
            piles.append((n, (x0 + x1) // 2, (y0 + y1) // 2))
        return cv.m, "five fenced compounds each with one gate; every pile is inside a pen", (400, 480), piles
    ship = (ri(rng, 100, 700), ri(rng, 60, 520))
    zone = (ship[0] - 85, ship[1] - 35, ship[0] + 85, ship[1] + 85)
    rects = []
    target = ri(rng, 4, 6)
    for _ in range(600):
        if len(rects) >= target:
            break
        w, h = ri(rng, 150, 230), ri(rng, 130, 190)
        x0, y0 = ri(rng, 25, W - 25 - w), ri(rng, 25, H - 25 - h)
        r = (x0, y0, x0 + w, y0 + h)
        infl = (x0 - 62, y0 - 62, x0 + w + 62, y0 + h + 62)

        def hit(a, b):
            return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
        if hit(infl, zone) or any(hit(infl, o) for o in rects):
            continue
        rects.append(r)
    if len(rects) < 4:
        raise Resample("pens")
    cnt = counts_rand(len(rects), rng, 95, 200)
    piles = []
    for (x0, y0, x1, y1), n in zip(rects, [int(c) for c in rng.permutation(cnt)]):
        th, gw = ri(rng, 10, 16), ri(rng, 50, 68)
        cv.frame(x0, y0, x1, y1, th)
        g = str(rng.choice(['top', 'bottom', 'left', 'right']))
        if g in ('top', 'bottom'):
            m = max(0, (x1 - x0) // 2 - gw // 2 - th - 25)
            c = (x0 + x1) // 2 + ri(rng, -m, m)
            yy = y0 if g == 'top' else y1 - th
            cv.land(c - gw // 2, yy, c + gw // 2, yy + th)
        else:
            m = max(0, (y1 - y0) // 2 - gw // 2 - th - 25)
            c = (y0 + y1) // 2 + ri(rng, -m, m)
            xx = x0 if g == 'left' else x1 - th
            cv.land(xx, c - gw // 2, xx + th, c + gw // 2)
        piles.append((n, (x0 + x1) // 2 + ri(rng, -25, 25), (y0 + y1) // 2 + ri(rng, -20, 20)))
    return cv.m, desc, ship, piles


def g13(rng, var):
    cv = Canvas()
    desc = "binary-tree corridors from the trunk; each branch splits and the piles sit at some leaf tips"
    if not var:
        cv.land(373, 330, 427, 575)
        cv.land(200 - 27, 330 - 27, 600 + 27, 330 + 27)
        for x in (200, 600):
            cv.land(x - 27, 200 - 0, x + 27, 330)
            cv.land(x - 127, 200 - 27, x + 127, 200 + 27)
        for x in (100, 300, 500, 700):
            cv.land(x - 27, 60, x + 27, 200)
            cv.land(x - 60, 60 - 27, x + 60, 60 + 27)
        ship = (400, 520)
        piles = [(40, 45, 62), (30, 355, 62), (50, 750, 62), (20, 545, 62), (12, 150, 62)]
        return cv.m, desc, ship, piles
    for _ in range(500):
        hw = ri(rng, 23, 28)
        y1, y2, y3 = ri(rng, 300, 365), ri(rng, 170, 230), ri(rng, 52, 85)
        d1 = [ri(rng, 175, 215), ri(rng, 175, 215)]
        d2 = [[ri(rng, 85, 115), ri(rng, 85, 115)], [ri(rng, 85, 115), ri(rng, 85, 115)]]
        d3 = ri(rng, 42, 66)
        leaves = []
        l2 = []
        for si, s in enumerate((-1, 1)):
            x1 = 400 + s * d1[si]
            for ti, t in enumerate((-1, 1)):
                x2 = x1 + t * d2[si][ti]
                l2.append((x2, x1))
                leaves.append(x2)
        bars2 = sorted((400 + s * d1[si] - d2[si][0] - hw, 400 + s * d1[si] + d2[si][1] + hw)
                       for si, s in enumerate((-1, 1)))
        lb = sorted((x - d3, x + d3) for x in leaves)
        ok = y1 - y2 >= 90 and y2 - y3 >= 2 * hw + 20 and lb[0][0] >= 22 and lb[-1][1] <= 778
        ok = ok and all(b[0] - a[1] >= 50 for a, b in zip(lb[:-1], lb[1:]))
        ok = ok and bars2[1][0] - bars2[0][1] >= 45 and bars2[0][0] >= 20 and bars2[1][1] <= 780
        # children of one parent: bar of x1 spans [x1-d2L-hw, x1+d2R+hw]
        if ok:
            break
    else:
        raise Resample("tree")
    sx_ = ri(rng, 500, 535)
    cv.land(400 - hw, y1, 400 + hw, 575)
    xl = 400 - d1[0] - hw
    xr = 400 + d1[1] + hw
    cv.land(xl, y1 - hw, xr, y1 + hw)
    keep = [True] * 4
    for _ in range(ri(rng, 0, 2)):
        keep[ri(rng, 0, 3)] = False
    if sum(keep) < 3:
        keep = [True] * 4
    ends = []
    li = 0
    for si, s in enumerate((-1, 1)):
        x1 = 400 + s * d1[si]
        cv.land(x1 - hw, y2, x1 + hw, y1)
        cv.land(x1 - d2[si][0] - hw, y2 - hw, x1 + d2[si][1] + hw, y2 + hw)
        for ti, t in enumerate((-1, 1)):
            x2 = x1 + t * d2[si][ti]
            if keep[li]:
                cv.land(x2 - hw, y3, x2 + hw, y2)
                cv.land(x2 - d3, y3 - hw, x2 + d3, y3 + hw)
                ends += [(x2 - d3 + 25, y3), (x2 + d3 - 25, y3)]
            li += 1
    cnt = counts_rand(ri(rng, 4, 6), rng)
    idx = rng.permutation(len(ends))[:len(cnt)]
    piles = [(c, ends[j][0], ends[j][1]) for c, j in zip(cnt, idx)]
    return cv.m, desc, (400, sx_), piles


def g14(rng, var):
    cv = Canvas()
    desc = "room-islands in a moat joined by narrow bridges, ship on one island"
    if not var:
        xs0, ys0 = [30, 305, 580], [40, 340]
        for x in xs0:
            for y in ys0:
                cv.land(x, y, x + 190, y + 220)
        hb = [(0, 0), (1, 0), (0, 1)]
        for i, j in hb:
            cv.land(xs0[i] + 190, ys0[j] + 87, xs0[i + 1], ys0[j] + 133)
        cv.land(xs0[1] + 70, ys0[0] + 220, xs0[1] + 120, ys0[1])
        cv.land(xs0[2] + 70, ys0[0] + 220, xs0[2] + 120, ys0[1])
        cv.land(xs0[0] + 70, ys0[0] + 220, xs0[0] + 120, ys0[1])
        cv.land(xs0[1] + 190, ys0[1] + 87, xs0[2], ys0[1] + 133)
        ship = (125, 120)
        piles = auto_piles(cv.m, ship, [0.97, 0.8, 0.6, 0.4, 0.3], [40, 30, 30, 12, 20], rng)
        return cv.m, "six room-islands in a moat joined by narrow bridges, ship in one corner island", ship, piles
    nx, ny = ri(rng, 2, 4), ri(rng, 2, 3)
    gx, gy = ri(rng, 46, 80), ri(rng, 46, 80)
    mx, my = ri(rng, 20, 40), ri(rng, 20, 40)
    cw = (W - 2 * mx - (nx - 1) * gx) / nx
    ch = (H - 2 * my - (ny - 1) * gy) / ny
    bw = ri(rng, 46, 56)
    rect = {}
    for i in range(nx):
        for j in range(ny):
            x0, y0 = mx + i * (cw + gx), my + j * (ch + gy)
            sh = [ri(rng, 0, 16) for _ in range(4)]
            rect[i, j] = (int(x0 + sh[0]), int(y0 + sh[1]), int(x0 + cw - sh[2]), int(y0 + ch - sh[3]))
            cv.land(*rect[i, j])
    edges = [((i, j), (i + 1, j)) for i in range(nx - 1) for j in range(ny)] + \
            [((i, j), (i, j + 1)) for i in range(nx) for j in range(ny - 1)]
    parent = {k: k for k in rect}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a
    extra = int(rng.integers(0, 3))
    chosen = []
    for k in rng.permutation(len(edges)):
        a, b = edges[k]
        if find(a) != find(b):
            parent[find(a)] = find(b)
            chosen.append((a, b))
        elif extra > 0:
            extra -= 1
            chosen.append((a, b))
    for a, b in chosen:
        A, B = rect[a], rect[b]
        if a[1] == b[1]:  # horizontal bridge
            lo, hi = max(A[1], B[1]) + 8, min(A[3], B[3]) - 8
            if hi - lo < bw:
                raise Resample("bridge")
            c = ri(rng, lo + bw // 2, hi - bw // 2)
            cv.land(A[2] - 1, c - bw // 2, B[0] + 1, c + bw // 2)
        else:
            lo, hi = max(A[0], B[0]) + 8, min(A[2], B[2]) - 8
            if hi - lo < bw:
                raise Resample("bridge")
            c = ri(rng, lo + bw // 2, hi - bw // 2)
            cv.land(c - bw // 2, A[3] - 1, c + bw // 2, B[1] + 1)
    R = rect[ri(rng, 0, nx - 1), ri(rng, 0, ny - 1)]
    ship = ((R[0] + R[2]) // 2, (R[1] + R[3]) // 2 - 15)
    return cv.m, desc, ship, rand_auto(cv.m, ship, rng, 4, 6)


def g15(rng, var):
    cv = Canvas(land=True)
    desc = "staggered brick-like wall segments forming a zigzag hall, ship at top centre"
    if not var:
        for r, y in enumerate([100, 185, 270, 355, 440, 525]):
            off = 0 if r % 2 == 0 else 125
            x = -100 + off
            while x < 800:
                cv.water(x, y - 8, x + 190, y + 8)
                x += 190 + 70
        ship = (400, 40)
        piles = auto_piles(cv.m, ship, [0.97, 0.8, 0.65, 0.5, 0.35], [30, 40, 20, 16, 30], rng)
        return cv.m, desc, ship, piles
    th, seg, gap = ri(rng, 10, 18), ri(rng, 150, 230), ri(rng, 52, 90)
    y, step = ri(rng, 90, 108), ri(rng, 76, 96)
    per = seg + gap
    shift = ri(rng, 0, per - 1)
    r = 0
    while y <= 545:
        off = 0 if r % 2 == 0 else per // 2 + ri(rng, -20, 20)
        x = -per + shift + off
        while x < W:
            cv.water(x, y - th // 2, x + seg, y + th // 2)
            x += per
        y += step
        r += 1
    ship = (ri(rng, 100, 700), 40)
    return cv.m, desc, ship, rand_auto(cv.m, ship, rng, 4, 6)


GENS = {1: g01, 2: g02, 3: g03, 4: g04, 5: g05, 6: g06, 7: g07, 8: g08, 9: g09, 10: g10,
        11: g11, 12: g12, 13: g13, 14: g14, 15: g15}


def sane(m, ship, piles):
    rx, ry = ship[0], ship[1] + 30
    if not (3 <= rx <= W - 4 and 3 <= ry <= H - 4) or not m[ry, rx]:
        return "release point"
    if not (2 <= len(piles) <= 10):
        return f"{len(piles)} piles"
    if any(not (5 <= n <= 60) for n, _, _ in piles):
        return "pile size"
    if not (80 <= sum(n for n, _, _ in piles) <= 220):
        return "tomato total"
    if any(not (0 <= x < W and 0 <= y < H) or not m[y, x] for _, x, y in piles):
        return "pile off land"
    return None


def main_originals():
    for nn in range(1, 16):
        rng = np.random.default_rng(100 + nn)
        m, desc, ship, piles = GENS[nn](rng, False)
        write_map(OUT, f"built_{nn:02d}", m, desc, ship, piles)


def main_variants(count, seed, outdir):
    sys.path.insert(0, ROOT)
    from tools.check_maps import check
    for nn in range(1, 16):
        for k in range(1, count + 1):
            name = f"built_{nn:02d}_v{k:03d}"
            last = None
            for attempt in range(30):
                rng = np.random.default_rng([seed, nn, k, attempt])
                try:
                    m, desc, ship, piles = GENS[nn](rng, True)
                    m, ship, piles = flip(m, ship, piles, rng.random() < 0.5, rng.random() < 0.5)
                except Resample as e:
                    last = str(e)
                    continue
                why = sane(m, ship, piles)
                if why:
                    last = why
                    continue
                write_map(outdir, name, m, desc, ship, piles)
                _, _, errors, warnings = check(os.path.join(outdir, name + ".map"))
                if not errors and not warnings:
                    break
                last = "; ".join(errors + warnings)
            else:
                raise SystemExit(f"{name}: could not make a valid variant ({last})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.variants > 0:
        if not a.out:
            ap.error("--out is required with --variants")
        main_variants(a.variants, a.seed, a.out)
    else:
        main_originals()
