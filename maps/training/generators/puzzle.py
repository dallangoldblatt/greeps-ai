"""Generates puzzle_01..puzzle_15 training maps. Run from repo root:
python maps/training/generators/puzzle.py
"""
import math
import os
import random
from PIL import Image, ImageDraw

W, H = 800, 600
WATER = (40, 70, 160)
LAND = (235, 215, 190)
OUT = os.path.join("maps", "training")


class Canvas:
    def __init__(self, water=False):
        self.im = Image.new("RGB", (W, H), WATER if water else LAND)
        self.d = ImageDraw.Draw(self.im)

    def rect(self, x0, y0, x1, y1, c):
        self.d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=c)

    def w(self, *a):
        self.rect(*a, WATER)

    def l(self, *a):
        self.rect(*a, LAND)

    def disk(self, x, y, r, c):
        self.d.ellipse([x - r, y - r, x + r, y + r], fill=c)

    def ellipse(self, x, y, rx, ry, c):
        self.d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=c)

    def ring(self, x, y, r, t, c, gap=None, gw=64):
        n = int(2 * math.pi * r / 2)
        for i in range(n):
            a = 2 * math.pi * i / n
            if gap is not None and abs((a - math.radians(gap) + math.pi) % (2 * math.pi) - math.pi) * r < gw / 2:
                continue
            self.disk(x + r * math.cos(a), y + r * math.sin(a), t / 2, c)

    def line(self, p, q, t, c):
        n = int(max(abs(q[0] - p[0]), abs(q[1] - p[1]), 1) / 2) + 1
        for i in range(n + 1):
            f = i / n
            self.disk(p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f, t / 2, c)

    def is_land(self, x, y):
        x, y = int(x), int(y)
        if not (0 <= x < W and 0 <= y < H):
            return False
        return self.im.getpixel((x, y)) == LAND

    def clear_near(self, x, y, r):
        """Land-ish check: all 4 corners + centre of a pile box are land."""
        return all(self.is_land(x + dx, y + dy) for dx in (-20, 0, 20) for dy in (-20, 0, 20))


def save(name, cv, desc, ship, piles, outdir=None, quiet=False):
    outdir = outdir or OUT
    cv.im.save(os.path.join(outdir, name + ".png"))
    with open(os.path.join(outdir, name + ".map"), "w", encoding="utf-8") as f:
        f.write("# Greeps map: %s\nimage %s.png\nship %d %d\n" % (desc, name, ship[0], ship[1]))
        for n, x, y in piles:
            f.write("pile %d %d %d\n" % (n, x, y))
    if not quiet:
        print(name, sum(p[0] for p in piles))


def sample_piles(cv, rng, counts, avoid, mind=70, xr=(40, 760), yr=(40, 560)):
    out = []
    for n in counts:
        for _ in range(5000):
            x, y = rng.randint(*xr), rng.randint(*yr)
            if not cv.clear_near(x, y, 20):
                continue
            if any(math.hypot(x - a, y - b) < mind for a, b in avoid + [(p[1], p[2]) for p in out]):
                continue
            out.append((n, x, y))
            break
    return out


def m01():
    cv = Canvas()
    ship = (110, 480)
    cx, cy, r = 110, 470, 105
    cv.ring(cx, cy, r, 16, WATER, gap=-35, gw=76)
    # single exit on the upper-right of the ring
    piles = [(40, 650, 90), (30, 700, 330), (30, 420, 540), (25, 330, 120)]
    save("puzzle_01", cv, "ship sealed in a ring pocket with a single exit; piles scattered far outside", ship, piles)


def m02():
    cv = Canvas()
    cv.w(392, 0, 410, 500)
    ship = (130, 90)
    piles = [(50, 700, 110), (40, 620, 300), (30, 560, 520)]
    # also a decoy pile on the near side
    piles.append((10, 250, 500))
    save("puzzle_02", cv, "long wall: piles all on far side, only gap is at the bottom edge (big detour)", ship, piles)


def m03():
    cv = Canvas()
    cx, cy = 400, 300
    pts = []
    th = 0.0
    while True:
        r = 52 + 64 * th / (2 * math.pi)
        if r > 285:
            break
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
        th += 3.0 / r
    for p in pts:
        cv.disk(p[0], p[1], 7, WATER)
    ship = (80, 60)
    piles = [(60, 392, 296), (40, 428, 310), (20, 640, 540)]
    save("puzzle_03", cv, "spiral corridor with the big piles at the centre; ship outside", ship, piles)


def m04():
    cv = Canvas(water=True)
    my = 300
    cv.l(20, my - 28, 780, my + 28)
    for i, x in enumerate((150, 290, 430, 570, 700)):
        if i % 2 == 0:
            cv.l(x - 28, 60, x + 28, my)
        else:
            cv.l(x - 28, my, x + 28, 545)
    ship = (60, my - 30)
    # real pile: end of 2 branches and the trunk end; others are decoys
    piles = [(50, 570, 545 - 30), (40, 700, 90 if False else 80), (30, 755, my)]
    cv.l(672, 60, 728, my)
    save("puzzle_04", cv, "decoy dead-end branches off a trunk; piles hide at the ends of only some", ship, piles)


def m05():
    rng = random.Random(5)
    cv = Canvas()
    for _ in range(7):
        cv.ellipse(rng.randint(180, 700), rng.randint(60, 540), rng.randint(30, 60), rng.randint(25, 50), WATER)
    cv.l(0, 0, 150, 150)
    ship = (70, 60)
    big = (80, 740, 555)
    cv.disk(740, 555, 40, LAND)
    tiny = sample_piles(cv, rng, [5] * 8, [(70, 90), (740, 555)], mind=90, xr=(60, 700), yr=(40, 560))
    save("puzzle_05", cv, "one huge far pile versus eight tiny piles nearby (greed vs distance)", ship, [big] + tiny)


def m06():
    cv = Canvas()
    cv.w(390, 0, 410, 270)
    cv.w(390, 330, 410, 600)
    ship = (170, 300)
    piles = [(40, 690, 120), (40, 690, 480), (30, 560, 300)]
    save("puzzle_06", cv, "bottleneck: every greep squeezes through one 60px gap in a wall", ship, piles)


def m07():
    rng = random.Random(7)
    cv = Canvas()
    for _ in range(8):
        x, y = rng.randint(50, 300), rng.randint(170, 560)
        rx, ry = rng.randint(25, 55), rng.randint(20, 50)
        cv.ellipse(x, y, rx, ry, WATER)
        cv.ellipse(800 - x, y, rx, ry, WATER)
    # clear central strip so the halves connect
    cv.l(350, 0, 450, 600)
    ship = (400, 60)
    left = sample_piles(cv, rng, [30, 20, 12], [(400, 60)], mind=110, xr=(50, 330), yr=(200, 560))
    piles = []
    for n, x, y in left:
        piles.append((n, x, y))
        piles.append((n, 800 - x, y))
    save("puzzle_07", cv, "mirror-symmetric lakes and piles around a top-centre ship", ship, piles)


def m08():
    cv = Canvas()
    cx, cy = 400, 300
    cv.ring(cx, cy, 160, 18, WATER, gap=150, gw=76)
    ship = (400, 290)
    piles = [(40, 690, 60), (30, 90, 80), (40, 700, 540), (20, 60, 300), (20, 400, 560)]
    save("puzzle_08", cv, "ship on a donut island; one hidden gap in the ring, piles outside", ship, piles)


def m09():
    cv = Canvas(water=True)
    cv.l(30, 20, 110, 580)
    ys = [50 + 85 * i for i in range(7)]
    for y in ys:
        cv.l(30, y - 25, 770, y + 25)
    ship = (70, 300)
    piles = [(40, 735, ys[4]), (30, 690, ys[4]), (30, 640, ys[4])]
    save("puzzle_09", cv, "comb of 7 parallel inlets, the piles are in only one tooth", ship, piles)


def m10():
    cv = Canvas()
    cv.ellipse(400, 330, 250, 270, WATER)
    cv.w(150, 330, 650, 600)
    ship = (100, 520)
    piles = [(50, 700, 520), (40, 680, 300), (30, 600, 80)]
    save("puzzle_10", cv, "lake touching the bottom edge: ship and piles on opposite shores, route only over the north", ship, piles)


def m11():
    cv = Canvas()
    x0, y0, x1, y1 = 500, 330, 740, 540
    t = 16
    cv.w(x0, y0, x1, y0 + t)
    cv.w(x0, y1 - t, x1, y1)
    cv.w(x0, y0, x0 + t, y1)
    cv.w(x1 - t, y0, x1, y1)
    cv.l(x0 + 40, y0 - 2, x0 + 90, y0 + t + 2)  # small entrance on top, left side
    ship = (140, 130)
    piles = [(60, 640, 440), (40, 560, 450), (10, 620, 130)]
    save("puzzle_11", cv, "hidden room: walled chamber with a 50px entrance holds the best piles", ship, piles)


def m12():
    cv = Canvas()
    t = 16

    def nook(fx, fy, opening):
        def X(v):
            return v if not fx else W - v

        def Y(v):
            return v if not fy else H - v

        def R(a, b, c, d):
            xs = sorted((X(a), X(c)))
            ys = sorted((Y(b), Y(d)))
            cv.w(xs[0], ys[0], xs[1], ys[1])
        # box 0..130 x 0..130, walls on inner sides
        if opening == "side":
            R(0, 130, 140, 130 + t)
            R(130, 0, 130 + t, 70)
        else:
            R(0, 130, 70, 130 + t)
            R(130, 0, 130 + t, 146)
    nook(False, False, "side")
    nook(True, False, "top")
    nook(False, True, "top")
    nook(True, True, "side")
    ship = (400, 300)
    piles = [(30, 50, 50), (30, 750, 50), (30, 50, 550), (30, 750, 550)]
    save("puzzle_12", cv, "open field but every pile hides in a corner nook with a narrow mouth", ship, piles)


def m13():
    cv = Canvas()
    ys = [120, 235, 350, 465]
    for i, y in enumerate(ys):
        if i % 2 == 0:
            cv.w(0, y, 735, y + 16)
        else:
            cv.w(65, y, 800, y + 16)
    ship = (80, 40)
    piles = [(60, 700, 550), (40, 100, 550), (20, 700, 300)]
    save("puzzle_13", cv, "serpentine switchbacks: ~3000px walk to piles at the bottom", ship, piles)


def m14():
    cv = Canvas(water=True)
    isl = {"A": (400, 300), "L": (110, 300), "R": (690, 300), "B": (110, 90), "C": (110, 510),
           "D": (690, 90), "E": (690, 510), "F": (400, 70), "G": (400, 530)}
    edges = [("A", "L"), ("A", "R"), ("L", "B"), ("L", "C"), ("R", "D"), ("R", "E"), ("A", "F"), ("A", "G")]
    for a, b in edges:
        (x0, y0), (x1, y1) = isl[a], isl[b]
        cv.l(min(x0, x1) - 25, min(y0, y1) - 25, max(x0, x1) + 25, max(y0, y1) + 25)
    for k, (x, y) in isl.items():
        cv.l(x - 65, y - 50, x + 65, y + 50)
    ship = (400, 290)
    piles = [(40, 110, 90), (50, 690, 510), (30, 690, 90), (30, 400, 530)]
    save("puzzle_14", cv, "archipelago of rectangular islands joined by narrow bridges; piles on remote leaf islands", ship, piles)

def m15():
    cv = Canvas()
    cx, cy = 400, 300
    for r, ang in ((270, 90), (190, 180), (110, 0)):
        cv.ring(cx, cy, r, 16, WATER)
        px, py = cx + r * math.cos(math.radians(ang)), cy + r * math.sin(math.radians(ang))
        if ang in (0, 180):
            cv.l(px - 24, py - 32, px + 24, py + 32)
        else:
            cv.l(px - 32, py - 24, px + 32, py + 24)
    ship = (400, 290)
    piles = [(40, 70, 60), (40, 740, 540), (30, 400, 40) if False else (30, 400, 135), (20, 400, 470)]
    save("puzzle_15", cv, "ship at centre of three concentric rings with staggered gaps; piles between and beyond", ship, piles)


# ---------------------------------------------------------------------------
# Randomized families: v01..v15 build one variant of idea NN from an rng.
# Each returns (canvas, description, ship, piles, allowed_flips).
# ---------------------------------------------------------------------------

class Retry(Exception):
    pass


def split_counts(rng, k, lo=110, hi=190):
    total = min(rng.randint(lo, hi), 80 * k)
    w = [rng.uniform(0.5, 1.6) for _ in range(k)]
    s = sum(w)
    return [max(5, min(80, int(round(total * x / s)))) for x in w]


def place(cv, rng, cnts, ok=lambda x, y: True, mind=70, avoid=(), ad=80, xr=(30, 770), yr=(30, 570),
          minimum=None, tries=2500):
    out = []
    for n in cnts:
        for _ in range(tries):
            x, y = rng.randint(*xr), rng.randint(*yr)
            if not ok(x, y) or not cv.clear_near(x, y, 20):
                continue
            if any(math.hypot(x - a, y - b) < ad for a, b in avoid):
                continue
            if any(math.hypot(x - p[1], y - p[2]) < mind for p in out):
                continue
            out.append((n, x, y))
            break
    if len(out) < (len(cnts) if minimum is None else minimum):
        raise Retry("could not place piles")
    return out


def release(ship):
    return (ship[0], ship[1] + 30)


def v01(rng):
    cv = Canvas()
    r = rng.randint(88, 125)
    cx = rng.randint(r + 30, 770 - r)
    cy = rng.randint(r + 30, 570 - r)
    cv.ring(cx, cy, r, 16, WATER, gap=rng.uniform(0, 360), gw=rng.randint(48, 70) + 18)
    ship = (cx + rng.randint(-15, 15), cy + rng.randint(-25, 5))
    k = rng.randint(3, 6)
    piles = place(cv, rng, split_counts(rng, k), ok=lambda x, y: math.hypot(x - cx, y - cy) > r + 60,
                  avoid=[(cx, cy)], ad=r + 60)
    return cv, "ship sealed in a ring pocket with a single exit; piles scattered far outside", ship, piles, "xy"


def v02(rng):
    cv = Canvas()
    t = rng.randint(12, 22)
    double = rng.random() < 0.35
    if not double:
        x = rng.randint(300, 500)
        mode = rng.choice(["bottom", "top", "mid"])
        if mode == "mid":
            g = rng.randint(50, 90)
            gy = rng.randint(120, 480)
            cv.w(x, 0, x + t, gy - g // 2)
            cv.w(x, gy + g // 2, x + t, H)
            sy = rng.randint(40, 520)
        else:
            g = rng.randint(55, 110)
            if mode == "bottom":
                cv.w(x, 0, x + t, H - g)
                sy = rng.randint(40, 230)
            else:
                cv.w(x, g, x + t, H)
                sy = rng.randint(330, 540)
        ship = (rng.randint(60, x - 90), sy)
        k = rng.randint(3, 5)
        piles = place(cv, rng, split_counts(rng, k, 100, 170), ok=lambda px, py: px > x + t + 50,
                      avoid=[release(ship)])
        piles += place(cv, rng, [rng.randint(8, 14)], ok=lambda px, py: px < x - 50, avoid=[release(ship)] + [(p[1], p[2]) for p in piles])
        desc = "long wall: piles on far side, the only gap is at one end or in the middle (detour)"
    else:
        x1 = rng.randint(230, 320)
        x2 = rng.randint(470, 560)
        g1, g2 = rng.randint(55, 100), rng.randint(55, 100)
        top_first = rng.random() < 0.5
        if top_first:
            cv.w(x1, g1, x1 + t, H)       # gap top
            cv.w(x2, 0, x2 + t, H - g2)   # gap bottom
            sy = rng.randint(340, 540)
        else:
            cv.w(x1, 0, x1 + t, H - g1)
            cv.w(x2, g2, x2 + t, H)
            sy = rng.randint(40, 240)
        ship = (rng.randint(50, x1 - 80), sy)
        piles = place(cv, rng, split_counts(rng, rng.randint(3, 4), 100, 160), ok=lambda px, py: px > x2 + t + 50,
                      avoid=[release(ship)])
        piles += place(cv, rng, [rng.randint(8, 14)], ok=lambda px, py: x1 + t + 50 < px < x2 - 50,
                       avoid=[release(ship)] + [(p[1], p[2]) for p in piles])
        desc = "two long walls with gaps at opposite ends: zig-zag detour to the far side"
    return cv, desc, ship, piles, "xy"


def v03(rng):
    cv = Canvas()
    cx, cy = 400 + rng.randint(-40, 40), 300 + rng.randint(-30, 30)
    r0 = rng.randint(50, 60)
    pitch = rng.randint(62, 74)
    wr = rng.choice([6, 7])
    phi = rng.uniform(0, 2 * math.pi)
    dirn = rng.choice([-1, 1])
    rmax = rng.randint(262, 285)
    th = 0.0
    while True:
        r = r0 + pitch * th / (2 * math.pi)
        if r > rmax:
            break
        cv.disk(cx + r * math.cos(dirn * th + phi), cy + r * math.sin(dirn * th + phi), wr, WATER)
        th += 3.0 / r
    ship = (rng.randint(50, 110), rng.randint(40, 80))
    piles = place(cv, rng, [rng.randint(35, 60), rng.randint(20, 45)] + ([rng.randint(15, 30)] if rng.random() < 0.4 else []),
                  ok=lambda x, y: math.hypot(x - cx, y - cy) < r0 + 35, mind=40, ad=0,
                  xr=(cx - 60, cx + 60), yr=(cy - 60, cy + 60), minimum=2)
    piles += place(cv, rng, [rng.randint(12, 30) for _ in range(rng.randint(1, 2))],
                   ok=lambda x, y: math.hypot(x - cx, y - cy) > rmax + 40, avoid=[release(ship)] + [(p[1], p[2]) for p in piles])
    return cv, "spiral corridor with the big piles at the centre; ship outside", ship, piles, "xy"


def v04(rng):
    cv = Canvas(water=True)
    my = rng.randint(190, 410)
    tw = rng.randint(50, 60)
    h = tw // 2
    bw = rng.randint(50, 60)
    cv.l(20, my - h, 780, my + h)
    n = rng.randint(4, 7)
    step = 600 / (n - 1)
    br = []
    for i in range(n):
        x = int(130 + i * step + rng.randint(-8, 8))
        up = rng.random() < 0.5
        if up:
            end = rng.randint(40, my - 110)
            cv.l(x - bw // 2, end, x + bw // 2, my)
        else:
            end = rng.randint(my + 110, 560)
            cv.l(x - bw // 2, my, x + bw // 2, end)
        br.append((x, end, up))
    ship = (60, my - 30)
    k = min(rng.randint(1, 3), n)
    real = rng.sample(range(n), k)
    pos = [(x, end + 30 if up else end - 30) for x, end, up in br]
    spots = [pos[i] for i in real]
    if rng.random() < 0.6 or k == 1:
        spots.append((752, my))
    cnts = split_counts(rng, len(spots), 100, 170)
    piles = [(c, x, y) for c, (x, y) in zip(cnts, spots)]
    rest = [i for i in range(n) if i not in real]
    if rest and rng.random() < 0.4:
        x, y = pos[rng.choice(rest)]
        piles.append((rng.randint(8, 12), x, y))
    return cv, "decoy dead-end branches off a trunk; piles hide at the ends of only some", ship, piles, "xy"


def v05(rng):
    cv = Canvas()
    for _ in range(rng.randint(5, 9)):
        cv.ellipse(rng.randint(180, 700), rng.randint(60, 540), rng.randint(30, 62), rng.randint(25, 50), WATER)
    cv.l(0, 0, 150, 150)
    ship = (70, 60)
    bx, by = rng.randint(610, 740), rng.randint(430, 558)
    cv.disk(bx, by, 40, LAND)
    big = (rng.randint(70, 80), bx, by)
    tiny = place(cv, rng, [rng.randint(5, 7)] * rng.randint(6, 10), ok=lambda x, y: x < 710,
                 avoid=[release(ship), (bx, by)], ad=90, mind=75, minimum=5)
    return cv, "one huge far pile versus several tiny piles nearby (greed vs distance)", ship, [big] + tiny, "xy"


def v06(rng):
    cv = Canvas()
    t = rng.randint(14, 26)
    double = rng.random() < 0.3
    gap = lambda: rng.randint(45, 85)
    if not double:
        x = rng.randint(330, 470)
        g = gap()
        gy = rng.randint(70 + g // 2, 530 - g // 2)
        cv.w(x, 0, x + t, gy - g // 2)
        cv.w(x, gy + g // 2, x + t, H)
        ship = (rng.randint(60, x - 100), rng.randint(60, 520))
        piles = place(cv, rng, split_counts(rng, rng.randint(3, 5), 100, 160), ok=lambda px, py: px > x + t + 60,
                      avoid=[release(ship)])
        desc = "bottleneck: every greep squeezes through one narrow gap in a wall"
    else:
        x1, x2 = rng.randint(220, 300), rng.randint(500, 580)
        g1, g2 = gap(), gap()
        y1 = rng.randint(70 + g1 // 2, 530 - g1 // 2)
        y2 = rng.randint(70 + g2 // 2, 530 - g2 // 2)
        cv.w(x1, 0, x1 + t, y1 - g1 // 2)
        cv.w(x1, y1 + g1 // 2, x1 + t, H)
        cv.w(x2, 0, x2 + t, y2 - g2 // 2)
        cv.w(x2, y2 + g2 // 2, x2 + t, H)
        ship = (rng.randint(50, x1 - 80), rng.randint(60, 520))
        piles = place(cv, rng, split_counts(rng, rng.randint(3, 4), 100, 160), ok=lambda px, py: px > x2 + t + 50,
                      avoid=[release(ship)])
        desc = "bottleneck: two walls in series, each with one narrow gap"
    return cv, desc, ship, piles, "xy"


def v07(rng):
    cv = Canvas()
    vertical = rng.random() < 0.6   # mirror axis: vertical line x=400 (else horizontal y=300)
    nl = rng.randint(6, 11)
    lakes = []
    for _ in range(nl):
        if vertical:
            lakes.append((rng.randint(50, 300), rng.randint(170, 560), rng.randint(25, 55), rng.randint(20, 50)))
        else:
            lakes.append((rng.randint(150, 750), rng.randint(40, 240), rng.randint(25, 55), rng.randint(20, 50)))
    sw = rng.randint(100, 140)
    for x, y, rx, ry in lakes:
        if vertical:
            cv.ellipse(x, y, rx, ry, WATER)
            cv.ellipse(799 - x, y, rx, ry, WATER)
        else:
            cv.ellipse(x, y, rx, ry, WATER)
            cv.ellipse(x, 599 - y, rx, ry, WATER)
    if vertical:
        cv.l(400 - sw // 2, 0, 400 + sw // 2, 600)
        ship = (400, rng.randint(40, 90))
        xr, yr = (50, 335), (150, 560)
    else:
        cv.l(0, 300 - sw // 2, 800, 300 + sw // 2)
        ship = (rng.randint(50, 100), 300 - 30)
        xr, yr = (150, 760), (40, 270)
    k = rng.randint(2, 4)
    half = place(cv, rng, split_counts(rng, k, 60, 100), mind=100, xr=xr, yr=yr, avoid=[release(ship)], ad=100)
    piles = []
    for n, x, y in half:
        piles.append((n, x, y))
        piles.append((n, 799 - x, y) if vertical else (n, x, 599 - y))
    return cv, "mirror-symmetric lakes and piles around the ship", ship, piles, "xy"


def v08(rng):
    cv = Canvas()
    r = rng.randint(130, 190)
    cx = rng.randint(max(r + 20, 330), min(780 - r, 470)) if r + 20 <= 470 else 400
    cx = max(r + 20, min(780 - r, cx))
    cy = rng.randint(max(r + 20, 270), min(580 - r, 330)) if max(r + 20, 270) <= min(580 - r, 330) else 300
    cv.ring(cx, cy, r, 18, WATER, gap=rng.uniform(0, 360), gw=rng.randint(48, 70) + 18)
    ship = (cx + rng.randint(-30, 30), cy - 10 + rng.randint(-30, 20))
    piles = place(cv, rng, split_counts(rng, rng.randint(4, 6), 110, 170),
                  ok=lambda x, y: math.hypot(x - cx, y - cy) > r + 60, avoid=[(cx, cy)], ad=r + 60, mind=90)
    return cv, "ship on a donut island; one hidden gap in the ring, piles outside", ship, piles, "xy"


def v09(rng):
    cv = Canvas(water=True)
    th = rng.randint(50, 58)
    n = rng.randint(5, 8)
    while True:
        hi = min(92, (560 - th) // (n - 1))
        if hi >= th + 16:
            break
        n -= 1
    pitch = rng.randint(th + 16, hi)
    span = (n - 1) * pitch
    y0 = (H - span) // 2 + rng.randint(-15, 15)
    ys = [y0 + i * pitch for i in range(n)]
    sw = rng.randint(60, 100)
    cv.l(30, ys[0] - th // 2 - 8, 30 + sw, ys[-1] + th // 2 + 8)
    sy = rng.randint(ys[0], ys[-1] - 10)
    far = [i for i in range(n) if abs(ys[i] - sy) >= 120] or list(range(n))
    tgt = rng.choice(far)
    xe_t = rng.randint(700, 775)
    for i, y in enumerate(ys):
        xe = xe_t if i == tgt else (rng.randint(700, 775) if rng.random() < 0.5 else rng.randint(420, 700))
        cv.l(30, y - th // 2, xe, y + th // 2)
    ship = (30 + sw // 2, sy)
    k = rng.randint(3, 4)
    cnts = split_counts(rng, k, 100, 150)
    piles = [(c, xe_t - 30 - i * rng.randint(45, 60), ys[tgt]) for i, c in enumerate(cnts)]
    return cv, "comb of parallel inlets, the piles are in only one tooth", ship, piles, "xy"


def v10(rng):
    cv = Canvas()
    w = rng.randint(200, 260)
    rx = w + rng.randint(0, 25)
    ry = rng.randint(215, 265)
    cy = ry + rng.randint(55, 90)
    cx = rng.randint(rx + 95, 800 - rx - 95)
    cv.ellipse(cx, cy, rx, ry, WATER)
    cv.w(cx - w, cy, cx + w, H)
    ship = (rng.randint(45, max(46, cx - rx - 45)), rng.randint(cy, 540))
    lake = lambda x, y: ((x - cx) / (rx + 25)) ** 2 + ((y - cy) / (ry + 25)) ** 2 > 1 and x > cx
    piles = place(cv, rng, split_counts(rng, rng.randint(3, 4), 100, 160), ok=lake, avoid=[release(ship)], ad=150, mind=80)
    return cv, "lake touching an edge: ship and piles on opposite shores, route only around the far end", ship, piles, "xy"


def v11(rng):
    cv = Canvas()
    t = rng.randint(14, 18)
    rw, rh = rng.randint(170, 250), rng.randint(150, 220)
    x0, y0 = rng.randint(40, 760 - rw), rng.randint(40, 560 - rh)
    x1, y1 = x0 + rw, y0 + rh
    cv.w(x0, y0, x1, y0 + t)
    cv.w(x0, y1 - t, x1, y1)
    cv.w(x0, y0, x0 + t, y1)
    cv.w(x1 - t, y0, x1, y1)
    ow = rng.randint(46, 70)
    side = rng.choice("TBLR")
    if side in "TB":
        off = rng.randint(t + 15, rw - t - 15 - ow)
        yy = y0 if side == "T" else y1 - t
        cv.l(x0 + off, yy - 2, x0 + off + ow, yy + t + 2)
    else:
        off = rng.randint(t + 15, rh - t - 15 - ow)
        xx = x0 if side == "L" else x1 - t
        cv.l(xx - 2, y0 + off, xx + t + 2, y0 + off + ow)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    for _ in range(500):
        ship = (rng.randint(40, 760), rng.randint(40, 540))
        if math.hypot(ship[0] - mx, ship[1] - my) > 330 and cv.is_land(*release(ship)):
            break
    else:
        raise Retry("ship")
    inside = lambda x, y: x0 + t + 25 < x < x1 - t - 25 and y0 + t + 25 < y < y1 - t - 25
    piles = place(cv, rng, split_counts(rng, rng.randint(2, 3), 90, 130), ok=inside, mind=55, ad=0,
                  xr=(x0, x1), yr=(y0, y1))
    piles += place(cv, rng, [rng.randint(8, 14)], ok=lambda x, y: not (x0 - 40 < x < x1 + 40 and y0 - 40 < y < y1 + 40),
                   avoid=[release(ship)] + [(p[1], p[2]) for p in piles])
    return cv, "hidden room: walled chamber with a narrow entrance holds the best piles", ship, piles, "xy"


def v12(rng):
    cv = Canvas()
    t = rng.randint(14, 18)
    s = rng.randint(110, 150)
    piles_at = []
    nk = rng.randint(2, 4)
    holders = set(rng.sample(range(4), nk))
    cnts = split_counts(rng, nk, 100, 170)
    ci = 0
    for idx, (fx, fy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        ow = rng.randint(46, 70)
        opening = rng.choice(["side", "top"])

        def X(v, fx=fx):
            return W - v if fx else v

        def Y(v, fy=fy):
            return H - v if fy else v

        def R(a, b, c, d, X=X, Y=Y):
            xs = sorted((X(a), X(c)))
            ys = sorted((Y(b), Y(d)))
            cv.w(xs[0], ys[0], xs[1], ys[1])
        if opening == "side":
            R(0, s, s + 10, s + t)
            R(s, 0, s + t, s - ow)
        else:
            R(0, s, s - ow, s + t)
            R(s, 0, s + t, s + 10)
        if idx in holders:
            px, py = rng.randint(32, s - 32), rng.randint(32, s - 32)
            piles_at.append((cnts[ci], X(px), Y(py)))
            ci += 1
    ship = (400 + rng.randint(-70, 70), 280 + rng.randint(-60, 40))
    for n, x, y in piles_at:
        if not cv.clear_near(x, y, 20):
            raise Retry("nook pile")
    return cv, "open field but the piles hide in corner nooks with narrow mouths (some nooks empty)", ship, piles_at, ""


def v13(rng):
    cv = Canvas()
    horiz = rng.random() < 0.6
    A, L = (H, W) if horiz else (W, H)    # across / along extents
    t = rng.randint(14, 18)
    n = rng.randint(3, 5) if horiz else rng.randint(3, 7)
    g = rng.randint(55, 90)
    lane = (A - n * t) // (n + 1)
    if lane < 70:
        raise Retry("lanes")
    lane_j = [lane] * (n + 1)
    pos = []
    a = lane_j[0]
    for i in range(n):
        pos.append(a)
        a += t + lane_j[i + 1]
    start_left = True
    for i, a in enumerate(pos):
        if i % 2 == 0:
            lo, hi = 0, L - g
        else:
            lo, hi = g, L
        if horiz:
            cv.w(lo, a, hi, a + t)
        else:
            cv.w(a, lo, a + t, hi)
    # lane i spans across [start_i, end_i)
    starts = [0] + [a + t for a in pos]
    ends = pos + [A]
    lane_mid = [(s + e) // 2 for s, e in zip(starts, ends)]
    if horiz:
        ship = (70, lane_mid[0] - 30)
    else:
        ship = (lane_mid[0], 40)
        # release is at y+30, along axis: fine
    lastlane = n
    def in_lane(i):
        s, e = starts[i], ends[i]
        return (lambda x, y: s + 25 < (y if horiz else x) < e - 25)
    piles = place(cv, rng, split_counts(rng, rng.randint(2, 3), 90, 150), ok=in_lane(lastlane), mind=90, avoid=[release(ship)])
    mid = rng.randint(1, max(1, n - 1))
    piles += place(cv, rng, [rng.randint(15, 30)], ok=in_lane(mid), avoid=[release(ship)] + [(p[1], p[2]) for p in piles], mind=90)
    return cv, "serpentine switchbacks: a very long walk to the piles in the last lane", ship, piles, "xy"


def v14(rng):
    cv = Canvas(water=True)
    cols = [110 + rng.randint(-25, 25), 400 + rng.randint(-25, 25), 690 + rng.randint(-25, 25)]
    rows = [90 + rng.randint(-20, 20), 300 + rng.randint(-20, 20), 510 + rng.randint(-20, 20)]
    hx, hy = rng.randint(55, 68), rng.randint(42, 52)
    bw = rng.randint(46, 58) // 2
    n = rng.randint(6, 9)
    start = (rng.randint(0, 2), rng.randint(0, 2))
    nodes = {start}
    edges = []
    while len(nodes) < n:
        cand = []
        for (i, j) in nodes:
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (i + di, j + dj)
                if 0 <= q[0] <= 2 and 0 <= q[1] <= 2 and q not in nodes:
                    cand.append(((i, j), q))
        e = rng.choice(cand)
        edges.append(e)
        nodes.add(e[1])
    P = lambda nd: (cols[nd[0]], rows[nd[1]])
    for a, b in edges:
        (x0, y0), (x1, y1) = P(a), P(b)
        cv.l(min(x0, x1) - bw, min(y0, y1) - bw, max(x0, x1) + bw, max(y0, y1) + bw)
    for nd in nodes:
        x, y = P(nd)
        cv.l(x - hx, y - hy, x + hx, y + hy)
    adj = {nd: [] for nd in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    shipn = rng.choice([nd for nd in nodes if len(adj[nd]) >= 2] or list(nodes))
    dist = {shipn: 0}
    q = [shipn]
    for u in q:
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    leaves = sorted([nd for nd in nodes if len(adj[nd]) == 1 and nd != shipn], key=lambda nd: -dist[nd] + rng.random())
    others = sorted([nd for nd in nodes if nd not in leaves and nd != shipn], key=lambda nd: -dist[nd] + rng.random())
    k = rng.randint(3, 5)
    chosen = (leaves + others)[:k]
    cnts = split_counts(rng, len(chosen), 110, 170)
    piles = []
    for c, nd in zip(cnts, chosen):
        x, y = P(nd)
        piles.append((c, x + rng.randint(-hx + 25, hx - 25), y + rng.randint(-hy + 22, hy - 22)))
    sx, sy = P(shipn)
    ship = (sx, sy - 10)
    for n_, x, y in piles:
        if not cv.clear_near(x, y, 20):
            raise Retry("pile")
    return cv, "archipelago of rectangular islands joined by narrow bridges; piles on remote islands", ship, piles, "xy"


def v15(rng):
    cv = Canvas()
    cx, cy = 400 + rng.randint(-15, 15), 300 + rng.randint(-15, 15)
    nr = 3
    r1 = rng.randint(90, 105)
    step = rng.randint(75, 85)
    radii = [r1 + i * step for i in range(nr)]
    ang = rng.uniform(0, 360)
    for r in radii:
        cv.ring(cx, cy, r, 16, WATER, gap=ang, gw=rng.randint(48, 70) + 18)
        ang += rng.uniform(100, 260)
    ship = (cx, cy - 10)
    bands = []
    for i in range(nr - 1):
        bands.append((radii[i] + 22, radii[i + 1] - 22))
    cnts = split_counts(rng, rng.randint(4, 6), 110, 170)
    piles = []
    targets = [("band", b) for b in bands] + [("out", None)]
    for c in cnts:
        kind, b = rng.choice(targets) if piles else targets[0]
        if kind == "band":
            ok = lambda x, y, b=b: b[0] < math.hypot(x - cx, y - cy) < b[1]
        else:
            ok = lambda x, y: math.hypot(x - cx, y - cy) > radii[-1] + 40
        piles += place(cv, rng, [c], ok=ok, mind=70, avoid=[(cx, cy)] + [(p[1], p[2]) for p in piles], ad=60)
    return cv, "ship at centre of concentric rings with staggered gaps; piles between and beyond", ship, piles, "xy"


GENS = [v01, v02, v03, v04, v05, v06, v07, v08, v09, v10, v11, v12, v13, v14, v15]


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
    """Turn rounded wall-tip slivers (water runs < 6 px between land) into land."""
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


def make_variants(nvar, seed, outdir):
    import sys
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, root)
    from tools.check_maps import check, thin_water
    os.makedirs(outdir, exist_ok=True)
    for nn, gen in enumerate(GENS, 1):
        for k in range(1, nvar + 1):
            name = "puzzle_%02d_v%03d" % (nn, k)
            last = ""
            for attempt in range(20):
                rng = random.Random("%s-%d-%d-%d" % (seed, nn, k, attempt))
                try:
                    cv, desc, ship, piles, flips = gen(rng)
                except Retry as e:
                    last = str(e)
                    continue
                fx = "x" in flips and rng.random() < 0.5
                fy = "y" in flips and rng.random() < 0.5
                ship, piles = flip(cv, ship, piles, fx, fy)
                despeckle(cv, thin_water)
                save(name, cv, desc, ship, piles, outdir=outdir, quiet=True)
                try:
                    _, _, errors, warnings = check(os.path.join(outdir, name + ".map"))
                except Exception as e:
                    errors, warnings = [str(e)], []
                if not errors and not warnings:
                    break
                last = "; ".join(map(str, list(errors) + list(warnings)))[:200]
            else:
                raise SystemExit("FAILED to build valid %s after 20 attempts: %s" % (name, last))
        print("idea %02d: %d variants ok" % (nn, nvar))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=0)
    ap.add_argument("--seed", default="0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.variants:
        make_variants(a.variants, a.seed, a.out or os.path.join("out", "puzzle_variants"))
    else:
        os.makedirs(OUT, exist_ok=True)
        for f in (m01, m02, m03, m04, m05, m06, m07, m08, m09, m10, m11, m12, m13, m14, m15):
            f()








