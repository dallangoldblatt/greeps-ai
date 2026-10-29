"""Generates signal_01..signal_15 training maps: situations where the answer
(which arm / direction / side holds the tomatoes) changes per variant, so it
has to be communicated with paint. Run from repo root:

    python maps/training/generators/signal.py
    python maps/training/generators/signal.py --variants N --seed S --out DIR
"""
import math
import os
import random
import sys
from PIL import Image, ImageDraw

W, H = 800, 600
WATER = (40, 70, 160)
LAND = (235, 215, 190)
OUT = os.path.join("maps", "training")


class Retry(Exception):
    pass


class Canvas:
    def __init__(self, water=True):
        self.im = Image.new("RGB", (W, H), WATER if water else LAND)
        self.d = ImageDraw.Draw(self.im)

    def rect(self, x0, y0, x1, y1, c):
        self.d.rectangle([int(x0), int(y0), int(x1) - 1, int(y1) - 1], fill=c)

    def w(self, *a):
        self.rect(*a, WATER)

    def l(self, *a):
        self.rect(*a, LAND)

    def room(self, r):
        cx, cy, w, h = r
        self.l(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def hcor(self, x0, x1, y, t):
        self.l(min(x0, x1), y - t / 2, max(x0, x1), y + t / 2)

    def vcor(self, x, y0, y1, t):
        self.l(x - t / 2, min(y0, y1), x + t / 2, max(y0, y1))

    def ellipse(self, x, y, rx, ry, c):
        self.d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=c)

    def is_land(self, x, y):
        x, y = int(x), int(y)
        return 0 <= x < W and 0 <= y < H and self.im.getpixel((x, y)) == LAND

    def clear_near(self, x, y):
        return all(self.is_land(x + dx, y + dy) for dx in (-20, 0, 20) for dy in (-20, 0, 20))


def save(name, cv, desc, ship, piles, outdir):
    cv.im.save(os.path.join(outdir, name + ".png"))
    with open(os.path.join(outdir, name + ".map"), "w", encoding="utf-8") as f:
        f.write("# Greeps map: %s\nimage %s.png\nship %d %d\n" % (desc, name, ship[0], ship[1]))
        for n, x, y in piles:
            f.write("pile %d %d %d\n" % (n, x, y))


# ---------------------------------------------------------------- helpers

def ans(rng, n):
    """Answer index in range(n): stratified over variant number, shuffled per idea/seed."""
    return (rng.k + rng.off) % n


def split(rng, total, k, lo=12):
    w = [rng.uniform(0.6, 1.5) for _ in range(k)]
    s = sum(w)
    return [max(lo, min(80, int(round(total * x / s)))) for x in w]


def put(cv, rng, room, counts, mind=50, avoid=()):
    """Place piles with the given counts inside a room (cx, cy, w, h)."""
    cx, cy, w, h = room
    out = []
    for n in counts:
        for _ in range(400):
            x = rng.randint(int(cx - w / 2 + 25), int(cx + w / 2 - 25))
            y = rng.randint(int(cy - h / 2 + 25), int(cy + h / 2 - 25))
            if not cv.clear_near(x, y):
                continue
            if any(math.hypot(x - p[1], y - p[2]) < mind for p in out):
                continue
            if any(math.hypot(x - a, y - b) < 70 for a, b in avoid):
                continue
            out.append((n, x, y))
            break
        else:
            raise Retry("pile in room")
    return out


def fill_rooms(cv, rng, rooms, ship_xy, npile=2, second=0.4, decoy=0.4, lo=90, hi=150):
    """One rich room (random answer), maybe a second modest one, tiny decoys elsewhere."""
    a = ans(rng, len(rooms))
    avoid = [ship_xy]
    k = npile if rng.random() < 0.8 else 1
    total = rng.randint(lo, hi) if k > 1 else rng.randint(40, 75)
    piles = put(cv, rng, rooms[a], split(rng, total, k), avoid=avoid)
    others = [i for i in range(len(rooms)) if i != a]
    rng.shuffle(others)
    if others and rng.random() < second:
        i = others.pop()
        piles += put(cv, rng, rooms[i], [rng.randint(20, 40)], avoid=avoid)
    for i in others:
        if rng.random() < decoy:
            piles += put(cv, rng, rooms[i], [rng.randint(5, 9)], avoid=avoid)
    return piles, a


def jit(rng, v, d):
    return v + rng.randint(-d, d)


def star(rng, cv, dirs, t):
    """Hub (ship) with straight arms ending in rooms. Returns ship, hub, {dir: room}."""
    hx, hy = jit(rng, 400, 15), jit(rng, 300, 12)
    hub = (hx, hy, 150, 110)
    cv.room(hub)
    rooms = {}
    for d in dirs:
        o = rng.randint(-22, 22)
        if d == "N":
            r = (hx + o, jit(rng, 95, 10), 140, 130)
            cv.vcor(r[0], r[1], hy, t)
        elif d == "S":
            r = (hx + o, jit(rng, 505, 10), 140, 130)
            cv.vcor(r[0], r[1], hy, t)
        elif d == "W":
            r = (jit(rng, 105, 10), hy + o, 130, 140)
            cv.hcor(r[0], hx, r[1], t)
        else:
            r = (jit(rng, 695, 10), hy + o, 130, 140)
            cv.hcor(r[0], hx, r[1], t)
        cv.room(r)
        rooms[d] = r
    return (hx, hy - 10), hub, rooms


def tw(rng):
    return rng.randint(46, 62)


# ---------------------------------------------------------------- ideas

def v01(rng):
    cv = Canvas()
    dirs = rng.sample("NSEW", rng.choice([3, 4]))
    ship, hub, rooms = star(rng, cv, dirs, tw(rng))
    ds = sorted(rooms)
    piles, a = fill_rooms(cv, rng, [rooms[d] for d in ds], (ship[0], ship[1] + 30), npile=3, second=0.15, decoy=0.35)
    return cv, "hub with 3-4 arms, only one random arm holds the rich tomatoes", ship, piles


def v02(rng):
    cv = Canvas()
    t = tw(rng)
    sx, sy = jit(rng, 100, 15), jit(rng, 300, 15)
    cv.room((sx, sy, 150, 110))
    jx = jit(rng, 340, 25)
    cv.room((jx, sy, 110, 110))
    cv.hcor(sx, jx, sy, t)
    rooms = []
    n_up, n_dn = jit(rng, 90, 10), jit(rng, 510, 10)
    cv.vcor(jx, n_up, sy, t); r = (jx, n_up, 130, 130); cv.room(r); rooms.append(r)
    cv.vcor(jx, sy, n_dn, t); r = (jx, n_dn, 130, 130); cv.room(r); rooms.append(r)
    ex = jit(rng, 690, 10)
    if rng.random() < 0.5:
        r = (ex, sy, 130, 140)
        cv.hcor(jx, ex, sy, t); cv.room(r); rooms.append(r)
    else:
        bx = jit(rng, 560, 15)
        cv.hcor(jx, bx, sy, t)
        cv.vcor(bx, sy - 120, sy + 120, t)
        for yy in (sy - 120, sy + 120):
            r = (ex, yy, 130, 120)
            cv.hcor(bx, ex, yy, t); cv.room(r); rooms.append(r)
    ship = (sx, sy - 10)
    piles, a = fill_rooms(cv, rng, rooms, (sx, sy + 20), npile=2, second=0.2, decoy=0.4)
    return cv, "trunk to a fork of 3-4 branches; the rich branch is random", ship, piles


def v03(rng):
    cv = Canvas()
    t = tw(rng)
    ship = (90, 290)
    cv.room((90, 300, 120, 110))
    x1, x2 = jit(rng, 220, 10), jit(rng, 450, 12)
    cv.hcor(90, x1, 300, t)
    cv.vcor(x1, 150, 450, t)
    for y in (150, 450):
        cv.hcor(x1, x2, y, t)
    cv.vcor(x2, 80, 220, t)
    cv.vcor(x2, 380, 520, t)
    rooms = []
    for y in (80, 220, 380, 520):
        cv.hcor(x2, 640, y, t)
        r = (700, y, 110, 100)
        cv.room(r)
        rooms.append(r)
    piles, a = fill_rooms(cv, rng, rooms, (90, 320), npile=2, second=0.35, decoy=0.4, lo=90, hi=140)
    return cv, "binary tree of two turns: the rich leaf (1 of 4) needs the right turn at each junction", ship, piles


def v04(rng):
    cv = Canvas()
    t = tw(rng)
    sy = jit(rng, 300, 15)
    cv.room((85, sy, 120, 110))
    xs = [jit(rng, 250, 15), jit(rng, 430, 12), jit(rng, 610, 12)]
    cv.hcor(85, 710, sy, t)
    rooms = []
    for x in xs:
        for sgn in (-1, 1):
            ry = sy + sgn * rng.randint(160, 185)
            ry = max(75, min(525, ry))
            cv.vcor(x, sy, ry, t)
            r = (x, ry, 110, 110)
            cv.room(r)
            rooms.append(r)
    r = (710, sy, 110, 120)
    cv.room(r)
    rooms.append(r)
    piles, a = fill_rooms(cv, rng, rooms, (85, sy + 20), npile=2, second=0.3, decoy=0.35, lo=90, hi=140)
    return cv, "trunk with 3 successive T-junctions, stubs up and down; the rich stub (1 of 7) is random", (85, sy - 10), piles


def v05(rng):
    cv = Canvas()
    dirs = rng.sample("NSEW", 4)
    ship, hub, rooms = star(rng, cv, dirs, tw(rng))
    ds = sorted(rooms)
    a = ans(rng, 4)
    piles = []
    avoid = [(ship[0], ship[1] + 30)]
    for i, d in enumerate(ds):
        smalls = [rng.randint(5, 10) for _ in range(rng.randint(2, 3))]
        if i == a:
            piles += put(cv, rng, rooms[d], [rng.randint(38, 55)] + smalls[:1], avoid=avoid)
        else:
            piles += put(cv, rng, rooms[d], smalls[:2], avoid=avoid)
    return cv, "four arms with small piles that run out at different times plus one arm with a medium pile", ship, piles


def v06(rng):
    cv = Canvas()
    t = tw(rng)
    cx = jit(rng, 400, 15)
    sy = jit(rng, 505, 8)
    cv.room((cx, sy, 160, 100))
    jy = jit(rng, 330, 20)
    dx = rng.randint(220, 270)
    cv.vcor(cx, jy, sy, t)
    cv.hcor(cx - dx, cx + dx, jy, t)
    ry = jit(rng, 100, 10)
    rooms = []
    for sgn in (-1, 1):
        x = cx + sgn * dx
        cv.vcor(x, ry, jy, t)
        r = (x, ry, 140, 130)
        cv.room(r)
        rooms.append(r)
    third = rng.random() < 0.4
    if third:
        cv.vcor(cx, ry, jy, t)
        r = (cx, ry, 130, 120)
        cv.room(r)
        rooms.append(r)
    a = ans(rng, 2)
    avoid = [(cx, sy + 30)]
    piles = put(cv, rng, rooms[a], split(rng, rng.randint(100, 150), 3 if rng.random() < 0.5 else 2), avoid=avoid)
    if rng.random() < 0.5:
        piles += put(cv, rng, rooms[1 - a], [rng.randint(5, 9)], avoid=avoid)
    return cv, "symmetric Y (mirror twin): same layout, the rich side swaps", (cx, sy - 10), piles


def scatter_obstacles(cv, rng, keep_out, n, rmin=12, rmax=32, sep=120):
    placed = []
    for _ in range(n * 40):
        if len(placed) >= n:
            break
        x, y, r = rng.randint(30, 770), rng.randint(30, 570), rng.randint(rmin, rmax)
        if any(math.hypot(x - a, y - b) < r + k for a, b, k in keep_out):
            continue
        if any(math.hypot(x - a, y - b) < sep for a, b, _ in placed):
            continue
        cv.ellipse(x, y, r, int(r * rng.uniform(0.7, 1.1)), WATER)
        placed.append((x, y, r))


def compass_piles(cv, rng, ship, ang, nrich, dist=(320, 540)):
    piles = []
    for n in split(rng, rng.randint(90, 150), nrich):
        for _ in range(100):
            a = ang + math.radians(rng.uniform(-18, 18))
            d = rng.uniform(*dist)
            x, y = int(ship[0] + d * math.cos(a)), int(ship[1] + d * math.sin(a))
            if 45 <= x <= 755 and 45 <= y <= 555 and all(math.hypot(x - p[1], y - p[2]) > 70 for p in piles):
                piles.append((n, x, y))
                break
        else:
            raise Retry("compass pile")
    return piles


def v07(rng):
    cv = Canvas(water=False)
    ang = math.radians(((rng.k + rng.off) % 8) * 45 + rng.uniform(-15, 15))
    for _ in range(200):
        ship = (rng.randint(120, 680), rng.randint(90, 470))
        try:
            piles = compass_piles(cv, rng, ship, ang, rng.randint(2, 3))
            break
        except Retry:
            continue
    else:
        raise Retry("compass")
    keep = [(ship[0], ship[1] + 20, 90)] + [(p[1], p[2], 60) for p in piles]
    scatter_obstacles(cv, rng, keep, rng.randint(9, 15))
    return cv, "open ground with far piles in one random compass direction behind scattered obstacles", ship, piles


def v12(rng):
    cv = Canvas(water=False)
    base = ((rng.k + rng.off) % 8) * 45 + rng.uniform(-12, 12)
    ang = math.radians(base)
    for _ in range(300):
        ship = (rng.randint(150, 650), rng.randint(110, 490))
        try:
            piles = compass_piles(cv, rng, ship, ang, 2)
            others = [base + 90 + rng.uniform(-10, 10), base - 90 + rng.uniform(-10, 10), base + 180 + rng.uniform(-10, 10)]
            rng.shuffle(others)
            for oa in others[:rng.randint(2, 3)]:
                a = math.radians(oa)
                for _ in range(60):
                    d = rng.uniform(220, 400)
                    x, y = int(ship[0] + d * math.cos(a)), int(ship[1] + d * math.sin(a))
                    if 45 <= x <= 755 and 45 <= y <= 555 and all(math.hypot(x - p[1], y - p[2]) > 90 for p in piles):
                        piles.append((rng.randint(6, 12), x, y))
                        break
            break
        except Retry:
            continue
    else:
        raise Retry("compass2")
    keep = [(ship[0], ship[1] + 20, 90)] + [(p[1], p[2], 60) for p in piles]
    scatter_obstacles(cv, rng, keep, rng.randint(10, 16))
    return cv, "open ground: a rich far cluster in a random direction, small decoy piles in other directions", ship, piles


def ring_geometry(cv, t=70):
    cv.l(60, 50, 740, 550)
    cv.w(130, 120, 670, 480)


def loop_point(s):
    """Point at arclength s (clockwise from top-left) of the centreline rectangle x95..705, y85..515."""
    w, h = 610, 430
    s %= 2 * (w + h)
    if s < w:
        return 95 + s, 85
    s -= w
    if s < h:
        return 705, 85 + s
    s -= h
    if s < w:
        return 705 - s, 515
    s -= w
    return 95, 515 - s


def v08(rng):
    cv = Canvas()
    ring_geometry(cv)
    if rng.random() < 0.5:
        cv.hcor(125, 675, 300, 56)
    sx = rng.randint(130, 670)
    ship = (sx, 500)
    ss = 610 + 430 + (705 - sx)
    per = 2 * (610 + 430)
    total = rng.randint(100, 150)
    for _ in range(200):
        s0 = rng.uniform(0, per)
        dd = (s0 - ss) % per
        if min(dd, per - dd) > 280:
            break
    else:
        raise Retry("loop")
    piles = []
    for n, off in zip(split(rng, total, 2), (0, 60)):
        x, y = loop_point(s0 + off)
        piles.append((n, int(x), int(y)))
    if rng.random() < 0.5:
        x, y = loop_point(s0 + per / 2)
        piles.append((rng.randint(6, 10), int(x), int(y)))
    return cv, "ring road with a random pile site: go clockwise or anticlockwise, return may be shorter via a cross road", ship, piles


def v09(rng):
    cv = Canvas(water=False)
    horiz = rng.random() < 0.5  # bands separate left | middle | right (else top | middle | bottom)
    L = W if not horiz else H
    A = H if not horiz else W
    if not horiz:
        b = [(rng.randint(225, 255), 0), (rng.randint(545, 575), 0)]
    else:
        b = [(rng.randint(185, 205), 0), (rng.randint(390, 410), 0)]
    t = rng.randint(22, 30)
    regions = []

    def R(a0, c0, a1, c1):
        if horiz:
            cv.w(c0, a0, c1, a1)
        else:
            cv.w(a0, c0, a1, c1)

    for a0, _ in b:
        g = rng.randint(55, 75)
        gc = rng.randint(100 + g // 2, A - 100 - g // 2)
        R(a0, 0, a0 + t, gc - g // 2)
        R(a0, gc + g // 2, a0 + t, A)
    lo, hi = b[0][0], b[1][0] + t
    mid = (lo + t + b[1][0]) // 2
    ship = (400, 300) if not horiz else (rng.randint(300, 500), mid - 10)
    if not horiz:
        ship = (mid, rng.randint(250, 350))
        regions = [(0, 0, b[0][0], H), (b[1][0] + t, 0, W, H)]
    else:
        regions = [(0, 0, W, b[0][0]), (0, b[1][0] + t, W, H)]
    a = ans(rng, 2)
    x0, y0, x1, y1 = regions[a]
    room = ((x0 + x1) // 2, (y0 + y1) // 2, x1 - x0 - 30, y1 - y0 - 30)
    piles = put(cv, rng, room, split(rng, rng.randint(100, 150), rng.randint(2, 3)), avoid=[(ship[0], ship[1] + 30)])
    x0, y0, x1, y1 = regions[1 - a]
    if rng.random() < 0.5:
        room = ((x0 + x1) // 2, (y0 + y1) // 2, x1 - x0 - 30, y1 - y0 - 30)
        piles += put(cv, rng, room, [rng.randint(5, 9)], avoid=[(ship[0], ship[1] + 30)])
    return cv, "ship between two regions split by water walls with one gap each; tomatoes in a random region", ship, piles


def v10(rng):
    cv = Canvas()
    dirs = rng.sample("NSEW", 3)
    ship, hub, rooms = star(rng, cv, dirs, tw(rng))
    ds = sorted(rooms)
    perm = list(range(3))
    rng.shuffle(perm)
    i0 = ans(rng, 3)
    perm = [(i0 + j) % 3 for j in range(3)] if rng.random() < 0.5 else [(i0 - j) % 3 for j in range(3)]
    # perm[j] = rank of arm j; rank 0 rich, rank 1 medium, rank 2 poor
    piles = []
    avoid = [(ship[0], ship[1] + 30)]
    for j, d in enumerate(ds):
        rank = perm[j]
        if rank == 0:
            cnt = split(rng, rng.randint(90, 130), 2)
        elif rank == 1:
            cnt = [rng.randint(25, 40)]
        else:
            cnt = [rng.randint(5, 9)]
        piles += put(cv, rng, rooms[d], cnt, avoid=avoid)
    return cv, "three arms ranked rich / medium / poor in random order", ship, piles


def v11(rng):
    cv = Canvas()
    t = rng.randint(46, 56)
    hub = (400, 300, 150, 110)
    cv.room(hub)
    rooms = []
    dirs = rng.sample("NSEW", rng.choice([3, 4]))
    for d in sorted(dirs):
        if d in "NS":
            ys = -1 if d == "N" else 1
            by = 300 + ys * 160
            cv.vcor(400, 300, by, t)
            cv.hcor(275, 525, by, t)
            for x in (275, 525):
                r = (x, 300 + ys * 215, 110, 110)
                cv.vcor(x, by, r[1], t)
                cv.room(r)
                rooms.append(r)
        else:
            xs = -1 if d == "W" else 1
            bx = 400 + xs * 280
            cv.hcor(400, bx, 300, t)
            cv.vcor(bx, 175, 425, t)
            for y in (175, 425):
                r = (bx, y, 110, 110)
                cv.room(r)
                rooms.append(r)
    ship = (400, 290)
    piles, a = fill_rooms(cv, rng, rooms, (400, 320), npile=2, second=0.3, decoy=0.25, lo=90, hi=140)
    return cv, "cross of corridors that each end in a T with two rooms: the rich room is 1 of 6-8", ship, piles


def v13(rng):
    cv = Canvas()
    ring_geometry(cv)
    sites = [(x, s) for x in (200, 400, 600) for s in (-1, 1)]
    keep = rng.sample(sites, rng.randint(4, 6))
    t = rng.randint(50, 60)
    rooms = []
    for x, s in sorted(keep):
        x = x + rng.randint(-12, 12)
        yr = 225 if s < 0 else 375
        cv.vcor(x, 120 if s < 0 else 480, yr, t)
        r = (x, yr, 110, 100)
        cv.room(r)
        rooms.append(r)
    sx = rng.choice([rng.randint(130, 300), rng.randint(500, 670)])
    ship = (sx, 500)
    piles, a = fill_rooms(cv, rng, rooms, (sx, 530), npile=2, second=0.3, decoy=0.4, lo=90, hi=140)
    return cv, "ring road with 4-6 inward spur rooms; the rich spur and its side of the ring are random", ship, piles


def v14(rng):
    cv = Canvas(water=False)
    t = 24
    cv.w(388, 0, 412, 600)
    cv.w(0, 288, 800, 312)
    # gaps in the four arms (up, right, down, left), clearing them again
    g = lambda: rng.randint(56, 72)
    gy = rng.randint(60 + 40, 288 - 100)
    gg = g(); cv.l(388, gy, 412, gy + gg)               # up arm: TL-TR
    gx = rng.randint(450, 700 - 40)
    gg = g(); cv.l(gx, 288, gx + gg, 312)               # right arm: TR-BR
    gy = rng.randint(312 + 60, 540 - 60)
    gg = g(); cv.l(388, gy, 412, gy + gg)               # down arm: BR-BL
    gx = rng.randint(100, 330)
    gg = g(); cv.l(gx, 288, gx + gg, 312)               # left arm: BL-TL
    cv.l(388, 288, 412, 312)
    cv.w(388, 288, 412, 312)
    quads = [(0, 0, 388, 288), (412, 0, 800, 288), (412, 312, 800, 600), (0, 312, 388, 600)]
    sq = ans(rng, 4)
    qs = quads[sq]
    for _ in range(200):
        ship = (rng.randint(qs[0] + 80, qs[2] - 80), rng.randint(qs[1] + 60, qs[3] - 110))
        if cv.is_land(*ship) and cv.is_land(ship[0], ship[1] + 30):
            break
    else:
        raise Retry("ship")
    rest = [i for i in range(4) if i != sq]
    r = rng.choice(rest)
    q = quads[r]
    room = ((q[0] + q[2]) // 2, (q[1] + q[3]) // 2, q[2] - q[0] - 60, q[3] - q[1] - 60)
    piles = put(cv, rng, room, split(rng, rng.randint(100, 150), rng.randint(2, 3)))
    rest.remove(r)
    if rng.random() < 0.5:
        q = quads[rng.choice(rest)]
        room = ((q[0] + q[2]) // 2, (q[1] + q[3]) // 2, q[2] - q[0] - 60, q[3] - q[1] - 60)
        piles += put(cv, rng, room, [rng.randint(5, 9)])
    return cv, "four quadrants in a ring linked by one gap each; ship in one, tomatoes in a random other", ship, piles


def v15(rng):
    cv = Canvas()
    cols, rows = [140, 400, 660], [100, 300, 500]
    t = rng.randint(46, 56)
    cells = [(i, j) for i in range(3) for j in range(3)]
    start = rng.choice(cells)
    seen = {start}
    edges = []
    stack = [start]
    while stack:
        c = stack[-1]
        nb = [(c[0] + a, c[1] + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= c[0] + a < 3 and 0 <= c[1] + b < 3 and (c[0] + a, c[1] + b) not in seen]
        if not nb:
            stack.pop()
            continue
        n = rng.choice(nb)
        seen.add(n)
        edges.append((c, n))
        stack.append(n)
        if rng.random() < 0.25 and len(stack) > 1:
            stack.pop()
    if rng.random() < 0.4:
        c = rng.choice(cells)
        n = rng.choice([(c[0] + a, c[1] + b) for a, b in ((1, 0), (0, 1)) if c[0] + a < 3 and c[1] + b < 3] or [c])
        if n != c and (c, n) not in edges and (n, c) not in edges:
            edges.append((c, n))
    adj = {c: [] for c in cells}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    P = lambda c: (cols[c[0]], rows[c[1]])
    for a, b in edges:
        (x0, y0), (x1, y1) = P(a), P(b)
        cv.l(min(x0, x1) - t / 2, min(y0, y1) - t / 2, max(x0, x1) + t / 2, max(y0, y1) + t / 2)
    rooms = {}
    for c in cells:
        x, y = P(c)
        rooms[c] = (x, y, 120, 100)
        cv.room(rooms[c])
    sx, sy = P(start)
    ship = (sx, sy - 10)
    leaves = [c for c in cells if len(adj[c]) == 1 and c != start] or [c for c in cells if c != start]
    lr = [rooms[c] for c in leaves]
    piles, a = fill_rooms(cv, rng, lr, (sx, sy + 20), npile=2, second=0.3, decoy=0.4, lo=90, hi=140)
    return cv, "random 3x3 room maze; the rich dead-end leaf is random", ship, piles


GENS = [v01, v02, v03, v04, v05, v06, v07, v08, v09, v10, v11, v12, v13, v14, v15]
IDEAS = {}


def flip(cv, ship, piles, fx, fy):
    if fx:
        cv.im = cv.im.transpose(Image.FLIP_LEFT_RIGHT)
        ship = (W - 1 - ship[0], ship[1])
        piles = [(n, W - 1 - x, y) for n, x, y in piles]
    if fy:
        cv.im = cv.im.transpose(Image.FLIP_TOP_BOTTOM)
        ship = (ship[0], H - 1 - (ship[1] + 30) - 30)
        piles = [(n, x, H - 1 - y) for n, x, y in piles]
    return ship, piles


def despeckle(cv, thin_water):
    import numpy as np
    for _ in range(6):
        a = np.array(cv.im)
        water = (a[..., 0] == WATER[0]) & (a[..., 1] == WATER[1]) & (a[..., 2] == WATER[2])
        thin = thin_water(water)
        if not thin.any():
            return
        a[thin] = LAND
        cv.im = Image.fromarray(a)
    cv.d = ImageDraw.Draw(cv.im)


def build(nn, k, seed, outdir, name):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools.check_maps import check, thin_water
    gen = GENS[nn - 1]
    last = ""
    off = random.Random("off-%s-%d" % (seed, nn)).randrange(1000)
    for attempt in range(40):
        rng = random.Random("%s-%d-%d-%d" % (seed, nn, k, attempt))
        rng.k, rng.off = k, off
        try:
            cv, desc, ship, piles = gen(rng)
        except Retry as e:
            last = str(e)
            continue
        if not (1 <= len(piles) <= 12):
            last = "pile count"
            continue
        fx = rng.random() < 0.5
        fy = rng.random() < 0.5
        ship, piles = flip(cv, ship, piles, fx, fy)
        despeckle(cv, thin_water)
        save(name, cv, desc, ship, piles, outdir)
        try:
            _, _, errors, warnings = check(os.path.join(outdir, name + ".map"))
        except Exception as e:
            errors, warnings = [str(e)], []
        if not errors and not warnings:
            return sum(p[0] for p in piles)
        last = "; ".join(map(str, list(errors) + list(warnings)))[:200]
    raise SystemExit("FAILED to build valid %s after 40 attempts: %s" % (name, last))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=0)
    ap.add_argument("--seed", default="0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.variants:
        out = a.out or os.path.join("out", "signal_variants")
        os.makedirs(out, exist_ok=True)
        for nn in range(1, 16):
            for k in range(1, a.variants + 1):
                build(nn, k, a.seed, out, "signal_%02d_v%03d" % (nn, k))
            print("idea %02d: %d variants ok" % (nn, a.variants))
    else:
        os.makedirs(OUT, exist_ok=True)
        for nn in range(1, 16):
            t = build(nn, nn, "base", OUT, "signal_%02d" % nn)
            print("signal_%02d" % nn, t)
