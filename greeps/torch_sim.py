"""
A vectorized GPU port of the Greeps scenario, for training only.

Thousands of maps run in parallel as tensors of shape (envs, greeps, ...).
The rules follow src/Creature.java, Ship.java, TomatoPile.java, Paint.java
and Timer.java; collisions use Greenfoot's own test (rotated image rectangles
with integer corners, see greenfoot.Actor.intersects), and the real maps'
water comes from Greenfoot itself (HeadlessGreeps --dump-water).

Known differences from the Java simulation, all small:
  * Greeps act simultaneously instead of one after another, so loading and
    pile depletion within one round are resolved afterwards in greep order.
  * When several greeps touch a loader, a random one is picked (Greenfoot
    picks whichever its collision tree finds first).
  * Paint written during a round is seen by the greeps acting later in that
    round in Java, but only from the next round here.
  * Episodes start when the first greep is released; the ship's landing is
    skipped (nothing else happens during it).

The Java headless simulator stays the ground truth for evaluation; see
tests/test_torch_sim.py for the statistical comparison (compared against a
policy's Java trace, paint contacts agree with Greenfoot on all but ~1 in
100,000 greep-drop pairs).

One episode is one map. A tick is one Greenfoot act round:

    sim = GreepsSim(MapBank.real(device), num_envs=256, device=device)
    obs = sim.observe()             # legal per-greep observations
    sim.step(actions)               # actions: dict of (envs, greeps) tensors

The observation and action layout below is shared by every policy; each
policy package provides a GPU twin of its Greep.java (see policies/README.md).
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from .java_sim import WATER_DIR, compile_scenario, dump_water
from .maps import REAL_MAPS, map_files, read_map_file


WIDTH, HEIGHT = 800, 600
NUM_GREEPS = 20
RELEASE_INTERVAL = 10           # ticks between greeps leaving the ship
GAME_SECONDS = 120
TICKS_PER_SECOND = 10
GAME_TICKS = GAME_SECONDS * TICKS_PER_SECOND
WALKING_SPEED = 5.0
DRIFT_PERCENT = 3
TIME_TO_SPIT = 10
PAINT_LIFETIME = 255
SPAWN_OFFSET_Y = 30

GREEP_W, GREEP_H = 30, 23
SHIP_W, SHIP_H = 110, 56
PILE_SIZE = 33                  # TomatoPile draws a (SIZE + 3)^2 image
PAINT_SIZE = 21

MAX_PILES = 12
# A greep spits at most every TIME_TO_SPIT moves; drops last PAINT_LIFETIME
# rounds, so this many slots per greep never overwrite a visible drop.
PAINT_SLOTS = PAINT_LIFETIME // TIME_TO_SPIT + 1
PAINT_COLORS = ("red", "orange", "purple")

# Distance fields are computed on a coarse grid of CELL x CELL pixels.
CELL = 4
GRID_W, GRID_H = WIDTH // CELL, HEIGHT // CELL

# Observation layout, identical to Greep.observe() in src/Greep.java.
OBS_CARRYING = 0
OBS_AT_SHIP = 1
OBS_AT_FOOD = 2
OBS_AT_WATER = 3
OBS_AT_EDGE = 4
OBS_PAINT = 5       # 3
PAINT_OBS = slice(OBS_PAINT, OBS_PAINT + 3)
OBS_MEMORY = 8      # 8
OBS_FLAGS = 16      # 2
OBS_HEADING = 18    # cos, sin
OBS_HOME = 20       # cos, sin of the ship's bearing relative to heading
NUM_OBS = 22

NUM_MODES = 3       # keep heading, face home, face pile
NUM_TURNS = 37      # turn by 10 * (bin - 18) degrees
NUM_SPITS = 4       # none, red, orange, purple
MEMORY_BITS = 8
NUM_FLAGS = 2

NUM_PRIVILEGED = 16


# ---------------------------------------------------------------------------
# Geometry (greenfoot.Actor.calcBounds / getRotatedCorners / intersects)
# ---------------------------------------------------------------------------

class _RotationTables:
    """Per-rotation (0..359) constants, as Greenfoot computes them."""

    def __init__(self, device):
        rot = np.arange(360)
        rad = np.radians(rot)
        cos, sin = np.cos(rad), np.sin(rad)

        # Creature.move: Math.round(x + cos * speed) == x + floor(cos * speed + .5)
        self.step_x = torch.tensor(np.floor(cos * WALKING_SPEED + 0.5),
                                   dtype=torch.long, device=device)
        self.step_y = torch.tensor(np.floor(sin * WALKING_SPEED + 0.5),
                                   dtype=torch.long, device=device)
        self.cos = torch.tensor(cos, dtype=torch.float32, device=device)
        self.sin = torch.tensor(sin, dtype=torch.float32, device=device)

        # Rotated corners: (int) (lx*cos - ly*sin + x + 0.5) etc.
        lx0 = int(-GREEP_W / 2)             # Java int division truncates
        ly0 = int(-GREEP_H / 2)
        lxs = np.array([lx0, lx0 + GREEP_W - 1, lx0 + GREEP_W - 1, lx0])
        lys = np.array([ly0, ly0, ly0 + GREEP_H - 1, ly0 + GREEP_H - 1])
        ox = lxs[None, :] * cos[:, None] - lys[None, :] * sin[:, None] + 0.5
        oy = lys[None, :] * cos[:, None] + lxs[None, :] * sin[:, None] + 0.5
        self.corner_dx = torch.tensor(ox, dtype=torch.float64, device=device)
        self.corner_dy = torch.tensor(oy, dtype=torch.float64, device=device)

        # Multiples of 90 degrees use the unrotated rectangle (with width and
        # height swapped at 90 and 270).
        self.axis = torch.tensor(rot % 90 == 0, device=device)
        width = np.where(rot % 180 == 0, GREEP_W, GREEP_H)
        height = np.where(rot % 180 == 0, GREEP_H, GREEP_W)
        self.axis_w = torch.tensor(width, dtype=torch.long, device=device)
        self.axis_h = torch.tensor(height, dtype=torch.long, device=device)
        self.axis_x0 = torch.tensor(np.trunc(-width / 2), dtype=torch.long, device=device)
        self.axis_y0 = torch.tensor(np.trunc(-height / 2), dtype=torch.long, device=device)


@dataclass
class Shape:
    """Integer corners (..., 4) and bounding rectangle of actors' images."""
    xs: torch.Tensor
    ys: torch.Tensor
    x0: torch.Tensor
    y0: torch.Tensor
    w: torch.Tensor
    h: torch.Tensor
    unrotated: torch.Tensor     # rotation == 0


def box_shape(cx, cy, w, h):
    """An unrotated image of w x h pixels centred on (cx, cy)."""
    x0 = cx + int(-w / 2)
    y0 = cy + int(-h / 2)
    x1 = x0 + w - 1
    y1 = y0 + h - 1
    xs = torch.stack([x0, x1, x1, x0], dim=-1)
    ys = torch.stack([y0, y0, y1, y1], dim=-1)
    full = lambda v: torch.full_like(x0, v, dtype=torch.int32)
    # int32 halves the memory traffic of the pairwise tests.
    return Shape(xs.int(), ys.int(), x0.int(), y0.int(), full(w), full(h),
                 torch.ones_like(x0, dtype=torch.bool))


def greep_shape(tables, x, y, rot):
    """Greenfoot's corners and bounding rectangle of greeps' images."""
    xs_rot = (x.unsqueeze(-1).double() + tables.corner_dx[rot]).trunc().long()
    ys_rot = (y.unsqueeze(-1).double() + tables.corner_dy[rot]).trunc().long()
    rx0 = xs_rot.min(-1).values - 1
    ry0 = ys_rot.min(-1).values - 1
    rw = xs_rot.max(-1).values + 1 - rx0 + 1
    rh = ys_rot.max(-1).values + 1 - ry0 + 1

    ax0 = x + tables.axis_x0[rot]
    ay0 = y + tables.axis_y0[rot]
    aw = tables.axis_w[rot]
    ah = tables.axis_h[rot]
    ax1 = ax0 + aw - 1
    ay1 = ay0 + ah - 1
    xs_axis = torch.stack([ax0, ax1, ax1, ax0], dim=-1)
    ys_axis = torch.stack([ay0, ay0, ay1, ay1], dim=-1)

    axis = tables.axis[rot]
    pick = lambda a, b: torch.where(axis, a, b)
    pick4 = lambda a, b: torch.where(axis.unsqueeze(-1), a, b)
    return Shape(pick4(xs_axis, xs_rot).int(), pick4(ys_axis, ys_rot).int(),
                 pick(ax0, rx0).int(), pick(ay0, ry0).int(),
                 pick(aw, rw).int(), pick(ah, rh).int(), rot == 0)


def _rects_overlap(a, b):
    # greenfoot.collision.ibsp.Rect.intersects (right/top edges exclusive)
    return ((b.x0 < a.x0 + a.w) & (a.x0 < b.x0 + b.w)
            & (b.y0 < a.y0 + a.h) & (a.y0 < b.y0 + b.h))


def _outside(axs, ays, bxs, bys):
    """Actor.checkOutside: some edge of polygon a has all of b beyond it."""
    # Edge i runs from corner i to corner i+1; all 4 edges x 4 corners at once.
    ax1, ay1 = axs.roll(-1, -1), ays.roll(-1, -1)
    nx = (ay1 - ays).unsqueeze(-1)
    ny = (axs - ax1).unsqueeze(-1)
    v = (nx * (bxs.unsqueeze(-2) - ax1.unsqueeze(-1))
         + ny * (bys.unsqueeze(-2) - ay1.unsqueeze(-1)))
    separates = (v > 0).all(-1) & ((nx != 0) | (ny != 0)).squeeze(-1)
    return separates.any(-1)


def intersects(a, b, exact=True):
    """Actor.intersects for broadcastable shapes a and b."""
    hit = _rects_overlap(a, b)
    if not exact:
        return hit
    sat = ~_outside(a.xs, a.ys, b.xs, b.ys) & ~_outside(b.xs, b.ys, a.xs, a.ys)
    return hit & ((a.unrotated & b.unrotated) | sat)


def touches_paint(tables, shape, x, y, rot, paint_x, paint_y, exact=True):
    """
    Whether greeps (their shape, centre x, y and rotation) touch paint drops
    centred at paint_x, paint_y; all arguments broadcast together.
    """
    # Actor.intersects, specialised to an unrotated square drop so each pair
    # costs a few integer comparisons. Real-valued half-sizes (used before)
    # counted 5% more contacts than Greenfoot.
    half = PAINT_SIZE // 2                  # drop corners at paint_x +- half
    paint = box_shape(paint_x, paint_y, PAINT_SIZE, PAINT_SIZE)
    hit = _rects_overlap(shape, paint)
    if not exact:
        return hit
    px, py = paint_x.int(), paint_y.int()
    # The drop's edges: all greep corners strictly beyond one of them.
    xs, ys = shape.xs, shape.ys
    separated = ((xs.max(-1).values < px - half) | (xs.min(-1).values > px + half)
                 | (ys.max(-1).values < py - half) | (ys.min(-1).values > py + half))
    # The greep's edges (corner i to i + 1, outward normal n): the drop's
    # nearest corner is beyond edge i when n.(p - corner_{i+1}) > half (|nx| + |ny|).
    nx = ys.roll(-1, -1) - ys
    ny = xs - xs.roll(-1, -1)
    threshold = nx * xs.roll(-1, -1) + ny * ys.roll(-1, -1) + half * (nx.abs() + ny.abs())
    separated |= (nx * px[..., None] + ny * py[..., None] > threshold).any(-1)
    # Greenfoot skips the polygon test when neither image is rotated.
    return hit & (shape.unrotated | ~separated)


def _expand(shape, dim):
    """Insert a broadcast dimension at dim (the corner axis stays last)."""
    return Shape(shape.xs.unsqueeze(dim), shape.ys.unsqueeze(dim),
                 shape.x0.unsqueeze(dim), shape.y0.unsqueeze(dim),
                 shape.w.unsqueeze(dim), shape.h.unsqueeze(dim),
                 shape.unrotated.unsqueeze(dim))


def bearing_degrees(dx, dy):
    """(int) (180 * Math.atan2(dy, dx) / Math.PI), normalized to 0..359."""
    deg = torch.rad2deg(torch.atan2(dy.double(), dx.double())).trunc().long()
    return deg % 360


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

def geodesic(free, sources, max_iter=4000):
    """
    Shortest walking distance (in grid cells, 8-connected) from any source
    cell to every free cell. free, sources: (B, GRID_H, GRID_W) bool.
    Water cells next to land get their land neighbours' distance + 1, so a
    greep standing at the shore still has a sensible value.
    """
    inf = torch.tensor(float("inf"), device=free.device)
    blocked = ~free & ~sources
    d = torch.where(sources, 0.0, inf)
    diag = math.sqrt(2.0)

    def neighbour_min(t):
        p = F.pad(t, (1, 1, 1, 1), value=float("inf"))
        orth = torch.minimum(torch.minimum(p[:, 1:-1, :-2], p[:, 1:-1, 2:]),
                             torch.minimum(p[:, :-2, 1:-1], p[:, 2:, 1:-1])) + 1
        corner = torch.minimum(torch.minimum(p[:, :-2, :-2], p[:, :-2, 2:]),
                               torch.minimum(p[:, 2:, :-2], p[:, 2:, 2:])) + diag
        return torch.minimum(orth, corner)

    for i in range(max_iter):
        new = torch.where(blocked, inf, torch.minimum(d, neighbour_min(d)))
        if i % 32 == 31 and torch.equal(new, d):
            break
        d = new
    for _ in range(3):
        d = torch.where(blocked, torch.minimum(d, neighbour_min(d)), d)
    return d


def _coarse_free(water):
    """(B, H, W) water -> (B, GRID_H, GRID_W) land at each cell's centre."""
    return ~water[:, CELL // 2::CELL, CELL // 2::CELL]


def _source_cells(free, cx, cy, half_w, half_h):
    """Free grid cells within a rectangle (pixels) around (cx, cy)."""
    gx = torch.arange(GRID_W, device=free.device) * CELL + CELL // 2
    gy = torch.arange(GRID_H, device=free.device) * CELL + CELL // 2
    inside = ((gx[None, None, :] - cx[:, None, None]).abs() <= half_w) & \
             ((gy[None, :, None] - cy[:, None, None]).abs() <= half_h)
    sources = inside & free
    # Fall back to the centre cell if the object sits on water.
    centre = torch.zeros_like(sources)
    b = torch.arange(free.shape[0], device=free.device)
    centre[b, (cy // CELL).clamp(0, GRID_H - 1), (cx // CELL).clamp(0, GRID_W - 1)] = True
    empty = ~sources.flatten(1).any(1)
    return torch.where(empty[:, None, None], centre, sources)


class MapBank:
    """
    A set of maps on the device: water masks, ship and pile positions, and
    walking-distance fields to the ship and to each pile.
    """

    def __init__(self, water, ship, piles, names):
        self.water = water                      # (M, H, W) bool
        self.ship = ship                        # (M, 2) long: x, y
        self.piles = piles                      # (M, MAX_PILES, 3): x, y, tomatoes
        self.names = names
        self.device = water.device
        self.dist = self._distance_fields()     # (M, 1 + MAX_PILES, GH, GW), pixels

    def __len__(self):
        return self.water.shape[0]

    @staticmethod
    def real(device):
        """The scenario's three maps, with water exactly as Greenfoot sees it."""
        if not all((WATER_DIR / f"water{m}.bin").exists() for m in (1, 2, 3)):
            compile_scenario()
            dump_water()
        water = np.stack([
            np.fromfile(WATER_DIR / f"water{m}.bin", dtype=np.uint8)
            .reshape(HEIGHT, WIDTH).astype(bool)
            for m in (1, 2, 3)
        ])
        ship = [m["ship"] for m in REAL_MAPS]
        piles = np.zeros((3, MAX_PILES, 3), dtype=np.int64)
        for i, m in enumerate(REAL_MAPS):
            piles[i, :len(m["piles"])] = m["piles"]
        return MapBank(
            torch.tensor(water, device=device),
            torch.tensor(ship, device=device),
            torch.tensor(piles, device=device),
            ["map1", "map2", "map3"],
        )

    @staticmethod
    def from_files(spec, device):
        """Custom maps (tools/map_editor.py): a directory or a list of .map files."""
        from PIL import Image

        files = map_files(spec)
        if not files:
            raise ValueError(f"No .map files in {spec}")
        water, ship, piles, names = [], [], [], []
        for f in files:
            m = read_map_file(f)
            rgb = np.asarray(Image.open(m["image"]).convert("RGB"), dtype=np.int32)
            if rgb.shape[:2] != (HEIGHT, WIDTH):
                raise ValueError(f"{m['image']} is {rgb.shape[1]}x{rgb.shape[0]}, "
                                 f"expected {WIDTH}x{HEIGHT}")
            if len(m["piles"]) > MAX_PILES:
                raise ValueError(f"{f}: at most {MAX_PILES} piles are supported")
            water.append(rgb[..., 2] > 2 * rgb[..., 0])       # Earth.isWater
            ship.append(m["ship"])
            p = np.zeros((MAX_PILES, 3), dtype=np.int64)
            p[:len(m["piles"])] = m["piles"]
            piles.append(p)
            names.append(m["name"])
        return MapBank(torch.tensor(np.stack(water), device=device),
                       torch.tensor(ship, device=device),
                       torch.tensor(np.stack(piles), device=device), names)

    @staticmethod
    def random(count, device, seed=0):
        return MapBank(*_random_maps(count, device, seed))

    @staticmethod
    def concat(banks):
        bank = MapBank.__new__(MapBank)
        bank.water = torch.cat([b.water for b in banks])
        bank.ship = torch.cat([b.ship for b in banks])
        bank.piles = torch.cat([b.piles for b in banks])
        bank.dist = torch.cat([b.dist for b in banks])
        bank.names = [n for b in banks for n in b.names]
        bank.device = banks[0].device
        return bank

    def _distance_fields(self, chunk=64):
        free = _coarse_free(self.water)
        fields = []
        reach = GREEP_W // 2
        for start in range(0, len(self), chunk):
            idx = slice(start, start + chunk)
            f = free[idx]
            sources = [_source_cells(f, self.ship[idx, 0], self.ship[idx, 1],
                                     SHIP_W // 2 + reach, SHIP_H // 2 + reach)]
            for p in range(MAX_PILES):
                sources.append(_source_cells(
                    f, self.piles[idx, p, 0], self.piles[idx, p, 1],
                    PILE_SIZE // 2 + reach, PILE_SIZE // 2 + reach))
            src = torch.stack(sources, dim=1)               # (b, S, GH, GW)
            b, s = src.shape[:2]
            ff = f[:, None].expand(b, s, GRID_H, GRID_W).reshape(b * s, GRID_H, GRID_W)
            d = geodesic(ff, src.reshape(b * s, GRID_H, GRID_W))
            fields.append(d.reshape(b, s, GRID_H, GRID_W) * CELL)
        dist = torch.cat(fields)
        # Unreachable (or absent) targets: a large but finite distance.
        return torch.nan_to_num(dist, posinf=4.0 * WIDTH).float()


def _random_maps(count, device, seed):
    """
    Random lakes and shorelines in the style of the real maps: smooth noise
    thresholded to 5-35% water, sometimes with a water border (like Map 3).
    The ship's release point and every pile are on land that can be reached
    on foot from the release point.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)

    noise = torch.zeros(count, 1, HEIGHT, WIDTH, device=device)
    for (h, w), amp in [((4, 5), 1.0), ((7, 9), 0.6), ((14, 18), 0.25), ((40, 52), 0.06)]:
        octave = torch.randn(count, 1, h, w, generator=gen, device=device)
        noise += amp * F.interpolate(octave, size=(HEIGHT, WIDTH),
                                     mode="bicubic", align_corners=True)
    noise = noise[:, 0]

    ys = torch.arange(HEIGHT, device=device).float()[:, None]
    xs = torch.arange(WIDTH, device=device).float()[None, :]
    edge = torch.minimum(torch.minimum(xs, WIDTH - 1 - xs),
                         torch.minimum(ys, HEIGHT - 1 - ys))
    border = torch.tensor(rng.random(count) < 0.35, device=device)
    border_width = torch.tensor(rng.uniform(20, 90, count), device=device).float()
    ramp = (1 - edge[None] / border_width[:, None, None]).clamp(min=0) * 3.0
    noise = noise + torch.where(border[:, None, None], ramp, torch.zeros_like(ramp))

    fraction = torch.tensor(rng.uniform(0.05, 0.35, count), device=device).float()
    coarse = noise[:, ::8, ::8].flatten(1).sort(dim=1).values
    rank = ((1 - fraction) * (coarse.shape[1] - 1)).long()
    threshold = coarse.gather(1, rank[:, None]).squeeze(1)
    water = noise > threshold[:, None, None]

    free = _coarse_free(water)
    # Land cells with land all around (about 12 px), for the release point.
    clear = -F.max_pool2d(-(free.float()[:, None]), 7, stride=1, padding=3)[:, 0] > 0.5

    ship = np.zeros((count, 2), dtype=np.int64)
    spawn_cells = np.zeros((count, 2), dtype=np.int64)
    clear_np = clear.cpu().numpy()
    margin_x, margin_y = 70 // CELL, 70 // CELL
    for i in range(count):
        cand = np.argwhere(clear_np[i, margin_y:-margin_y, margin_x:-margin_x])
        if len(cand) == 0:          # pathological map: make the middle land
            cand = np.array([[GRID_H // 2 - margin_y, GRID_W // 2 - margin_x]])
            water[i, HEIGHT // 2 - 40:HEIGHT // 2 + 40, WIDTH // 2 - 40:WIDTH // 2 + 40] = False
        cy, cx = cand[rng.integers(len(cand))] + [margin_y, margin_x]
        spawn_cells[i] = (cx, cy)
        sx = cx * CELL + CELL // 2
        sy = cy * CELL + CELL // 2 - SPAWN_OFFSET_Y
        ship[i] = (sx, sy)

    free = _coarse_free(water)
    sources = torch.zeros_like(free)
    sources[torch.arange(count), torch.tensor(spawn_cells[:, 1]), torch.tensor(spawn_cells[:, 0])] = True
    reach = torch.isfinite(geodesic(free, sources)).cpu().numpy()
    pile_ok = (clear.cpu().numpy() | free.cpu().numpy()) & reach

    piles = np.zeros((count, MAX_PILES, 3), dtype=np.int64)
    for i in range(count):
        n_piles = rng.integers(3, 7)
        cand = np.argwhere(pile_ok[i, 4:-4, 4:-4]) + 4
        rng.shuffle(cand)
        chosen = []
        sx, sy = ship[i]
        for cy, cx in cand:
            px, py = cx * CELL + CELL // 2, cy * CELL + CELL // 2
            if math.hypot(px - sx, py - sy) < 120:
                continue
            if any(math.hypot(px - qx, py - qy) < 70 for qx, qy, _ in chosen):
                continue
            chosen.append((px, py, int(rng.choice([10, 12, 16, 30, 40, 50]))))
            if len(chosen) == n_piles:
                break
        piles[i, :len(chosen)] = chosen

    names = [f"random{seed}_{i}" for i in range(count)]
    return (water, torch.tensor(ship, device=device),
            torch.tensor(piles, device=device), names)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _one_hot(index, n):
    """(..., n) bool. (F.one_hot can synchronize with the CPU to check its input.)"""
    return index.unsqueeze(-1) == torch.arange(n, device=index.device)


_CAPTURE_STREAMS = {}


def capture_graph(fn, warmup=2):
    """
    Record fn() as a CUDA graph and return (graph, fn's outputs). Replaying
    the graph reruns the same kernels on the same memory, without Python or
    launch overhead, so fn must only read and update tensors that persist
    (GreepsSim keeps its state in fixed tensors for this) and must not wait
    on GPU results (no .item(), bool(tensor), boolean-mask indexing).
    Random generators advance correctly on each replay (PyTorch tracks them).
    The warm-up calls really run.

    Warm-up always uses the same side stream: cuBLAS keeps a workspace for
    every stream it sees, and the allocator only reuses memory freed on the
    same stream, so a new stream per capture leaked ~200 MB per evaluation
    until training spilled into shared memory and slowed down 20-100x.
    """
    device = torch.cuda.current_device()
    if device not in _CAPTURE_STREAMS:
        _CAPTURE_STREAMS[device] = torch.cuda.Stream()
    stream = _CAPTURE_STREAMS[device]
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


class GreepsSim:
    """
    num_envs independent maps, each with NUM_GREEPS greeps.

    All state lives in fixed tensors that step() and reset() update in place
    (see _STATE), so a tick can be recorded once as a CUDA graph and replayed.

    Per-greep rewards (training only; the score is the plain tomato count):
      * a delivered tomato pays (1 - loader_share) to the greep that carried
        it and loader_share to the greep that loaded it onto the carrier;
        with team_share > 0, that fraction of it is instead split evenly over
        all released greeps (credit for helping, e.g. with paint);
      * the time bonus (seconds left when the last pile is emptied) is split
        evenly over all greeps;
      * optional potential-based shaping, shaping * (gamma * phi' - phi), with
        phi = -(walking distance to the nearest pile) / 1000 when empty-handed
        and 1 - (walking distance to the ship) / 1000 when carrying.
    """

    def __init__(self, bank, num_envs, device, seed=0, loader_share=0.5,
                 shaping=0.0, gamma=0.995, map_weights=None, exact_paint=True,
                 team_share=0.0):
        self.bank = bank
        self.E, self.N = num_envs, NUM_GREEPS
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.tables = _RotationTables(device)
        self.loader_share = loader_share
        self.team_share = team_share
        self.shaping = shaping
        self.gamma = gamma
        self.map_weights = (None if map_weights is None
                            else torch.as_tensor(map_weights, dtype=torch.float, device=device))

        E, N, K = self.E, self.N, PAINT_SLOTS
        z = lambda *s, dtype=torch.long: torch.zeros(*s, dtype=dtype, device=device)
        self.map_id = z(E)
        self.t = z(E)                   # next tick to play, from 1
        self.x, self.y, self.rot = z(E, N), z(E, N), z(E, N)
        self.carry = z(E, N, dtype=torch.bool)
        self.loader = z(E, N)           # who loaded the tomato we carry (-1: none)
        self.at_water = z(E, N, dtype=torch.bool)
        self.time_to_spit = z(E, N)
        self.memory = z(E, N)
        self.flags = z(E, N, 2, dtype=torch.bool)
        self.pile_x, self.pile_y, self.pile_n = z(E, MAX_PILES), z(E, MAX_PILES), z(E, MAX_PILES)
        self.pile_total = z(E)
        self.ship_x, self.ship_y = z(E), z(E)
        self.delivered = z(E)
        self.paint_x, self.paint_y = z(E, N * K), z(E, N * K)
        self.paint_color = z(E, N * K)
        self.paint_expire = z(E, N * K)  # visible while t < expire
        self.paint_step = z(E, N * K)    # value of self.steps when spat
        self.spits = z(E, N)
        self.steps = z(())               # ticks played by this sim (all envs at once)
        # Length of a game. A tensor (not a constant) so training can change it
        # between CUDA-graph replays; evaluation always uses the real 120 s.
        self.game_seconds = torch.full((), GAME_SECONDS, dtype=torch.long, device=device)
        self.active = z(E, dtype=torch.bool)
        self.phi = torch.zeros(E, N, device=device)
        self.greep_index = torch.arange(N, device=device)
        self.not_self = ~torch.eye(N, dtype=torch.bool, device=device)
        # Drop slot s belongs to greep s // PAINT_SLOTS.
        self.paint_author = torch.arange(N * K, device=device) // K
        self.exact_paint = exact_paint

    _STATE = ("map_id", "t", "x", "y", "rot", "carry", "loader", "at_water",
              "time_to_spit", "memory", "flags", "pile_x", "pile_y", "pile_n",
              "pile_total", "ship_x", "ship_y", "delivered", "paint_x", "paint_y",
              "paint_color", "paint_expire", "paint_step", "spits", "steps",
              "active", "phi")

    def _state(self):
        return {name: getattr(self, name) for name in self._STATE}

    def _commit(self, before):
        """Copy updated state back into the original tensors (fixed addresses)."""
        for name, original in before.items():
            current = getattr(self, name)
            if current is not original:
                original.copy_(current)
                setattr(self, name, original)

    # -- setup -------------------------------------------------------------

    def _randint(self, high, size):
        return torch.randint(high, size, generator=self.gen, device=self.device)

    def _rand(self, size):
        return torch.rand(size, generator=self.gen, device=self.device)

    def reset(self, envs=None, map_ids=None, start_ticks=None):
        """
        Start new episodes in the given envs (bool mask; default all).
        start_ticks (E,) starts episodes part-way through (greeps already
        released are still at the ship); training uses it once so that
        episodes do not all end at the same time.
        """
        before = self._state()
        if envs is None:
            envs = torch.ones(self.E, dtype=torch.bool, device=self.device)
        if map_ids is None:
            if self.map_weights is None:
                map_ids = self._randint(len(self.bank), (self.E,))
            else:
                map_ids = torch.multinomial(self.map_weights, self.E, replacement=True,
                                            generator=self.gen)
        map_ids = torch.as_tensor(map_ids, device=self.device).expand(self.E)
        e = envs
        b = self.bank
        self.map_id = torch.where(e, map_ids, self.map_id)
        m = self.map_id
        self.ship_x = torch.where(e, b.ship[m, 0], self.ship_x)
        self.ship_y = torch.where(e, b.ship[m, 1], self.ship_y)
        e2 = e[:, None]
        self.pile_x = torch.where(e2, b.piles[m, :, 0], self.pile_x)
        self.pile_y = torch.where(e2, b.piles[m, :, 1], self.pile_y)
        self.pile_n = torch.where(e2, b.piles[m, :, 2], self.pile_n)
        self.pile_total = torch.where(e, self.pile_n.sum(1), self.pile_total)

        self.t = torch.where(e, 1 if start_ticks is None else start_ticks, self.t)
        self.x = torch.where(e2, self.ship_x[:, None].expand(-1, self.N), self.x)
        self.y = torch.where(e2, (self.ship_y + SPAWN_OFFSET_Y)[:, None].expand(-1, self.N), self.y)
        self.rot = torch.where(e2, self._randint(360, (self.E, self.N)), self.rot)
        self.carry &= ~e2
        self.loader = torch.where(e2, -1, self.loader)
        self.at_water &= ~e2
        self.time_to_spit = torch.where(e2, 0, self.time_to_spit)
        self.memory = torch.where(e2, 0, self.memory)
        self.flags &= ~e2[..., None]
        self.delivered = torch.where(e, 0, self.delivered)
        self.paint_expire = torch.where(e2, 0, self.paint_expire)
        self.spits = torch.where(e2, 0, self.spits)
        self.active |= e
        self.phi = torch.where(e2, self._potential(), self.phi)
        self._commit(before)

    # -- queries -------------------------------------------------------------

    def alive(self):
        """Greeps that have left the ship (they act in tick self.t)."""
        return (self.greep_index[None, :] * RELEASE_INTERVAL + 1 <= self.t[:, None]) \
            & self.active[:, None]

    def _greep_shapes(self):
        return greep_shape(self.tables, self.x, self.y, self.rot)

    def _at_ship(self, shape):
        ship = box_shape(self.ship_x, self.ship_y, SHIP_W, SHIP_H)
        return intersects(shape, _expand(ship, 1))

    def _piles_touching(self, shape):
        """(E, N, P) greep touches pile (only piles with tomatoes exist)."""
        piles = box_shape(self.pile_x, self.pile_y, PILE_SIZE, PILE_SIZE)
        hit = intersects(_expand(shape, 2), _expand(piles, 1))
        return hit & (self.pile_n > 0)[:, None, :]

    def paint_contacts(self, shape=None):
        """
        (E, N, N * PAINT_SLOTS) bool: which visible drops each greep touches.
        Drop slot s was spat by greep paint_author[s] at self.paint_step[:, s].
        """
        if shape is None:
            shape = self._greep_shapes()
        visible = self.paint_expire > self.t[:, None]
        hit = touches_paint(self.tables, _expand(shape, 2), self.x[..., None], self.y[..., None],
                            self.rot[..., None], self.paint_x[:, None, :],
                            self.paint_y[:, None, :], self.exact_paint)
        return hit & visible[:, None, :]

    def _paint_under(self, shape):
        """(E, N, 3) which colours of visible paint each greep touches."""
        hit = self.paint_contacts(shape)
        colours = _one_hot(self.paint_color, 3)                 # (E, NK, 3)
        # (E, N, NK) x (E, NK, 3): any visible drop of each colour under us.
        return torch.bmm(hit.half(), colours.half()) > 0

    def _distances(self):
        """Walking distances (E, N): to the ship, and to the nearest pile."""
        gx = (self.x // CELL).clamp(0, GRID_W - 1)
        gy = (self.y // CELL).clamp(0, GRID_H - 1)
        m = self.map_id[:, None].expand(-1, self.N)
        d = self.bank.dist[m, :, gy, gx]                         # (E, N, 1+P)
        to_ship = d[..., 0]
        piles = d[..., 1:].masked_fill(~(self.pile_n > 0)[:, None, :], 4.0 * WIDTH)
        to_pile, nearest = piles.min(-1)
        return to_ship, to_pile, nearest, gx, gy

    def _potential(self):
        if self.shaping == 0:
            return torch.zeros(self.E, self.N, device=self.device)
        to_ship, to_pile, *_ = self._distances()
        return torch.where(self.carry, 1 - to_ship / 1000.0, -to_pile / 1000.0)

    def observe(self):
        """Legal observations (E, N, NUM_OBS), exactly as Greep.observe()."""
        shape = self._greep_shapes()
        obs = torch.zeros(self.E, self.N, NUM_OBS, device=self.device)
        obs[..., OBS_CARRYING] = self.carry.float()
        obs[..., OBS_AT_SHIP] = self._at_ship(shape).float()
        obs[..., OBS_AT_FOOD] = self._piles_touching(shape).any(-1).float()
        obs[..., OBS_AT_WATER] = self.at_water.float()
        obs[..., OBS_AT_EDGE] = ((self.x < 3) | (self.x > WIDTH - 3)
                                 | (self.y < 3) | (self.y > HEIGHT - 3)).float()
        obs[..., OBS_PAINT:OBS_PAINT + 3] = self._paint_under(shape).float()
        bits = torch.arange(MEMORY_BITS, device=self.device)
        obs[..., OBS_MEMORY:OBS_MEMORY + 8] = ((self.memory[..., None] >> bits) & 1).float()
        obs[..., OBS_FLAGS:OBS_FLAGS + 2] = self.flags.float()
        obs[..., OBS_HEADING] = self.tables.cos[self.rot]
        obs[..., OBS_HEADING + 1] = self.tables.sin[self.rot]
        home = bearing_degrees(self.ship_x[:, None] - self.x, self.ship_y[:, None] - self.y)
        rel = (home - self.rot) % 360
        obs[..., OBS_HOME] = self.tables.cos[rel]
        obs[..., OBS_HOME + 1] = self.tables.sin[rel]
        return obs

    def privileged(self):
        """Training-only features (E, N, NUM_PRIVILEGED) for the critic."""
        to_ship, to_pile, nearest, gx, gy = self._distances()
        m = self.map_id[:, None].expand(-1, self.N)

        # Walking direction to the nearest pile / ship (downhill on the
        # distance field), relative to the greep's heading.
        def gradient(channel):
            def at(dx, dy):
                return self.bank.dist[m, channel,
                                      (gy + dy).clamp(0, GRID_H - 1),
                                      (gx + dx).clamp(0, GRID_W - 1)]
            ddx = at(1, 0) - at(-1, 0)
            ddy = at(0, 1) - at(0, -1)
            angle = torch.atan2(-ddy, -ddx) - torch.deg2rad(self.rot.float())
            moving = (ddx.abs() + ddy.abs()) > 0
            return torch.where(moving, torch.cos(angle), 0.0), torch.where(moving, torch.sin(angle), 0.0)

        pile_cos, pile_sin = gradient(1 + nearest)
        ship_cos, ship_sin = gradient(torch.zeros_like(nearest))

        alive = self.alive()
        n_alive = alive.sum(1, keepdim=True).clamp(min=1)
        remaining = self.pile_n.sum(1, keepdim=True).float()
        nearest_left = self.pile_n.gather(1, nearest.flatten(1)).view(self.E, self.N)
        dx = self.x[:, :, None] - self.x[:, None, :]
        dy = self.y[:, :, None] - self.y[:, None, :]
        close = ((dx.abs() < 25) & (dy.abs() < 25) & alive[:, None, :]).sum(-1) - 1

        feats = [
            self.x / WIDTH, self.y / HEIGHT,
            to_ship / 1000.0, to_pile / 1000.0,
            pile_cos, pile_sin, ship_cos, ship_sin,
            nearest_left / 50.0,
            (remaining / self.pile_total[:, None].clamp(min=1)).expand(-1, self.N),
            (1 - self.t[:, None] / (self.game_seconds * TICKS_PER_SECOND)).expand(-1, self.N).float(),
            ((self.carry & alive).sum(1, keepdim=True) / n_alive).expand(-1, self.N),
            close.clamp(min=0) / 5.0,
            (self.delivered[:, None] / 100.0).expand(-1, self.N),
            (n_alive / self.N).expand(-1, self.N),
            self.loader.ge(0).float(),
        ]
        return torch.stack([f.float() for f in feats], dim=-1)

    # -- dynamics ------------------------------------------------------------

    def step(self, actions):
        """
        Play one tick. actions: dict of (E, N) tensors
          mode (0..2), turn (0..36), memory (0..255), flags (E, N, 2) bool,
          spit (0..3), move (bool).
        Returns (rewards (E, N), done (E,) bool, scores (E,) long; valid where done).
        Envs that are not active are left untouched.
        """
        E, N = self.E, self.N
        before = self._state()
        alive = self.alive()
        rewards = torch.zeros(E, N, device=self.device)

        # Heading: keep / face home / face the pile we stand on.
        shape = self._greep_shapes()
        touching = self._piles_touching(shape)
        on_pile = touching.any(-1)
        pile = touching.float().argmax(-1)
        mode = actions["mode"]
        home = bearing_degrees(self.ship_x[:, None] - self.x, self.ship_y[:, None] - self.y)
        to_pile = bearing_degrees(self.pile_x.gather(1, pile) - self.x,
                                  self.pile_y.gather(1, pile) - self.y)
        rot = torch.where(mode == 1, home, self.rot)
        rot = torch.where((mode == 2) & on_pile, to_pile, rot)
        rot = (rot + 10 * (actions["turn"] - (NUM_TURNS // 2))) % 360
        self.rot = torch.where(alive, rot, self.rot)

        self.memory = torch.where(alive, actions["memory"], self.memory)
        self.flags = torch.where(alive[..., None], actions["flags"], self.flags)

        # Spit (only every TIME_TO_SPIT + random(10) moves).
        spit = alive & (actions["spit"] > 0) & (self.time_to_spit == 0)
        slot = self.greep_index[None, :] * PAINT_SLOTS + self.spits % PAINT_SLOTS
        self.paint_x.scatter_(1, slot, torch.where(spit, self.x, self.paint_x.gather(1, slot)))
        self.paint_y.scatter_(1, slot, torch.where(spit, self.y, self.paint_y.gather(1, slot)))
        self.paint_color.scatter_(1, slot, torch.where(
            spit, actions["spit"] - 1, self.paint_color.gather(1, slot)))
        self.paint_expire.scatter_(1, slot, torch.where(
            spit, self.t[:, None] + PAINT_LIFETIME + 1, self.paint_expire.gather(1, slot)))
        self.paint_step.scatter_(1, slot, torch.where(
            spit, self.steps.expand(E, N), self.paint_step.gather(1, slot)))
        self.spits += spit.long()
        self.time_to_spit = torch.where(
            spit, TIME_TO_SPIT + self._randint(10, (E, N)), self.time_to_spit)

        # Move, with a 3% chance of drifting 10 degrees off course.
        move = alive & actions["move"].bool()
        drift = move & (self._randint(100, (E, N)) < DRIFT_PERCENT)
        self.rot = torch.where(drift, (self.rot + 10 * (self._randint(3, (E, N)) - 1)) % 360, self.rot)
        nx = (self.x + self.tables.step_x[self.rot]).clamp(0, WIDTH - 1)
        ny = (self.y + self.tables.step_y[self.rot]).clamp(0, HEIGHT - 1)
        flat = (self.map_id[:, None] * HEIGHT + ny) * WIDTH + nx
        water = self.bank.water.view(-1)[flat]
        self.at_water = torch.where(move, water, self.at_water)
        go = move & ~water
        self.x = torch.where(go, nx, self.x)
        self.y = torch.where(go, ny, self.y)
        self.time_to_spit = torch.where(move, (self.time_to_spit - 1).clamp(min=0), self.time_to_spit)

        # Load: a greep on a pile hands a tomato to another greep it touches.
        shape = self._greep_shapes()
        touching = self._piles_touching(shape) & alive[..., None]
        on_pile = touching.any(-1)
        pile = touching.float().argmax(-1)                      # (E, N)
        contact = intersects(_expand(shape, 2), _expand(shape, 1))
        contact &= alive[:, :, None] & alive[:, None, :]
        contact &= self.not_self
        pick = torch.where(contact, self._rand((E, N, N)), -1.0)
        target = pick.argmax(-1)                                # (E, N)
        load = on_pile & contact.any(-1) & ~self.carry.gather(1, target)
        # One load per receiver: the first loader in act order wins.
        targets = _one_hot(target, N)                           # (E, loader, receiver)
        claims = targets & load[..., None]
        first = claims & (claims.long().cumsum(1) == 1)
        load = first.any(-1)
        # Piles cannot hand out more tomatoes than they have.
        piles = _one_hot(pile, MAX_PILES)                       # (E, N, P)
        rank = (piles & load[..., None]).long().cumsum(1)
        load &= (rank <= self.pile_n[:, None, :]).gather(2, pile[..., None]).squeeze(-1)
        loads = (targets & load[..., None]).long()
        taken = (piles & load[..., None]).long().sum(1)
        self.pile_n = self.pile_n - taken
        received = loads.sum(1).bool()                          # (E, N)
        giver = (loads * self.greep_index[None, :, None]).sum(1)
        self.carry |= received
        self.loader = torch.where(received, giver, self.loader)

        # Drop at the ship (Greep.act's first reflex, done here at the end of
        # the previous tick, which is equivalent: nothing else moves us).
        at_ship = self._at_ship(shape)
        drop = self.carry & at_ship & alive
        self.delivered += drop.sum(1)
        delivery = drop.float() * (1 - self.loader_share)
        credit = drop & (self.loader >= 0)
        delivery = delivery.scatter_add(1, self.loader.clamp(min=0),
                                        credit.float() * self.loader_share)
        # team_share of every delivery is split over all released greeps.
        team = delivery.sum(1, keepdim=True) * alive / alive.sum(1, keepdim=True).clamp(min=1)
        rewards += (1 - self.team_share) * delivery + self.team_share * team
        self.carry &= ~drop
        self.loader = torch.where(drop, -1, self.loader)

        # Timer: every second, end the map if time is up or all piles are gone.
        second = self.t % TICKS_PER_SECOND == 0
        seconds = self.t // TICKS_PER_SECOND
        piles_gone = (self.pile_n == 0).all(1)
        done = self.active & second & ((seconds >= self.game_seconds) | piles_gone)
        # (Clamped: a training curriculum may shorten games that are already
        # past the new length.)
        time_bonus = torch.where(done, (self.game_seconds - seconds).clamp(min=0), 0)
        scores = self.delivered + time_bonus
        rewards += (time_bonus.float() / N)[:, None]

        if self.shaping:
            phi = self._potential()
            phi_next = torch.where(done[:, None], 0.0, phi)
            rewards += self.shaping * (self.gamma * phi_next - self.phi) * alive
            self.phi = phi

        rewards *= self.active[:, None]
        self.t = torch.where(self.active, self.t + 1, self.t)
        self.active &= ~done
        self.steps += 1
        self._commit(before)
        return rewards, done, scores


def evaluate(policy, bank, map_ids, episodes, device, seed=0, graph=True):
    """
    Play `episodes` episodes on each of `map_ids` (no auto-reset) and return
    scores of shape (len(map_ids), episodes). policy(obs, sim) -> actions.
    The tick is replayed as a CUDA graph unless graph=False.
    """
    ids = torch.tensor(map_ids, device=device).repeat_interleave(episodes)
    sim = GreepsSim(bank, len(ids), device, seed=seed)
    sim.reset(map_ids=ids)
    scores = torch.zeros(len(ids), dtype=torch.long, device=device)

    def tick():
        actions = policy(sim.observe(), sim)
        _, done, s = sim.step(actions)
        scores.copy_(torch.where(done, s, scores))

    with torch.no_grad():
        run = tick
        if graph and device.type == "cuda":
            g, _ = capture_graph(tick)
            run = g.replay
        # Finished envs stand still, so check for the end only now and then.
        while bool(sim.active.any()):
            for _ in range(50):
                run()
    return scores.view(len(map_ids), episodes).cpu().numpy()
