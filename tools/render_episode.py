"""
Render one episode of a policy as an animation (animated WebP or GIF), for
the README and for looking at what a policy does.

    python -m tools.render_episode --policy linear_cma --map maps/validation/generated/variety_08_v012.map
    python -m tools.render_episode --policy paint_comm --checkpoint trained/paint_comm/checkpoints/colour_stage2_best.pt \
        --map maps/validation/generated/variety_08_v012.map --view both --out docs/media/senses.webp

The episodes are played on the GPU simulator; of --episodes played at once,
the median by score is drawn. Views:
  * greenfoot: drawn with the scenario's own sprites and paint colours, as
    Greenfoot shows it (tomatoes a little larger);
  * senses: what a greep senses of the others: the paint under it (dot
    colour, grey for none) and whether it carries a tomato (white centre).
    The map is a faint outline for the viewer; greeps never see it.
  * both: the two side by side.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from greeps import policies
from greeps.maps import REAL_MAPS
from greeps.torch_sim import (GreepsSim, MapBank, OBS_CARRYING, OBS_PAINT, PAINT_LIFETIME,
                              PAINT_SIZE, TICKS_PER_SECOND, GAME_SECONDS)

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "src" / "images"
W, H = 800, 600
PAINT_RGB = [(232, 21, 27), (245, 131, 14), (115, 74, 153)]     # Paint.java
TOMATO_RGB = [(255, 100, 100), (227, 49, 49), (100, 20, 20)]     # TomatoPile.java
DARK, GHOST, INK = (13, 17, 23), (33, 45, 61), (201, 209, 217)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                   # Pillow < 10.1
        return ImageFont.load_default()


def load_map(spec, device):
    """(bank, background image) for 'real:N' or a .map file."""
    if spec.startswith("real:"):
        n = int(spec[5:])
        bank = MapBank.real(device)
        bank = MapBank(bank.water[n - 1:n], bank.ship[n - 1:n], bank.piles[n - 1:n],
                       [REAL_MAPS[n - 1]["name"]])
        return bank, Image.open(IMAGES / f"map{n - 1}.jpg").convert("RGB")
    path = Path(spec)
    image = next(line.split(None, 1)[1].strip() for line in path.read_text().splitlines()
                 if line.startswith("image "))
    return MapBank.from_files(str(path), device), Image.open(path.parent / image).convert("RGB")


def make_policy(args, device, gen):
    package = policies.load(args.policy)
    if args.checkpoint:
        from policies.mlp_ppo import model as mp
        actor = mp.load_actor(torch.load(args.checkpoint, map_location="cpu")["actor"])
        actor = actor.to(device).eval()
        return mp.sim_policy(actor, gen)
    return package.torch_policy(policies.read_embedded_weights(args.policy), device, gen)


def play(args, bank, device):
    """
    Play --episodes episodes at once and keep the median one (by score), so
    the animation shows a typical episode rather than a lucky one. Returns its
    recorded ticks (dicts of numpy arrays) and its score.
    """
    E = args.episodes
    sim = GreepsSim(bank, E, device, seed=args.seed)
    sim.game_seconds.fill_(args.seconds)
    sim.reset(map_ids=torch.zeros(E, dtype=torch.long, device=device))
    policy = make_policy(args, device, torch.Generator(device=device).manual_seed(args.seed))
    frames = []
    tick = 0                            # episodes that end early stop counting sim.t
    with torch.no_grad():
        while bool(sim.active.any()):
            obs = sim.observe()
            tick += 1
            if (tick - 1) % args.every == 0:
                frames.append({k: v.cpu().numpy() for k, v in {
                    "x": sim.x, "y": sim.y, "rot": sim.rot, "carry": sim.carry, "obs": obs,
                    "paint_x": sim.paint_x, "paint_y": sim.paint_y, "paint_color": sim.paint_color,
                    "paint_left": sim.paint_expire - sim.t[:, None], "pile_x": sim.pile_x,
                    "pile_y": sim.pile_y, "pile_n": sim.pile_n, "t": sim.t,
                    "delivered": sim.delivered, "ship_x": sim.ship_x, "ship_y": sim.ship_y}.items()})
            sim.step(policy(obs, sim))
    scores = sim.delivered.cpu().numpy()
    pick = int(np.argsort(scores, kind="stable")[E // 2])
    print(f"scores {sorted(scores.tolist())}; rendering episode {pick} ({scores[pick]})")
    return [{k: v[pick] for k, v in f.items()} for f in frames], int(scores[pick])


class Greenfoot:
    """The scenario's look: map image, paint, tomato piles, ship, greeps."""

    def __init__(self, background, piles, seconds):
        self.background = background
        self.seconds = seconds
        self.greep = Image.open(IMAGES / "greep.png").convert("RGBA")
        self.greep_food = Image.open(IMAGES / "greep-with-food.png").convert("RGBA")
        self.ship = Image.open(IMAGES / "spaceship.png").convert("RGBA")
        self.rotated = {}
        # Each pile scatters its tomatoes once, like TomatoPile; the first n are drawn.
        rng = np.random.default_rng(0)
        self.tomato_offsets = [np.clip(rng.normal(15, 7.5, (int(n), 2)), 0, 28).astype(int)
                               for n in piles]

    def sprite(self, carry, rot):
        key = (bool(carry), int(rot) // 5 * 5)
        if key not in self.rotated:
            base = self.greep_food if carry else self.greep
            self.rotated[key] = base.rotate(-key[1], resample=Image.BICUBIC, expand=True)
        return self.rotated[key]

    def draw(self, f):
        img = self.background.copy().convert("RGBA")
        paint = Image.new("RGBA", (W, H))
        d = ImageDraw.Draw(paint)
        for x, y, c, left in zip(f["paint_x"], f["paint_y"], f["paint_color"], f["paint_left"]):
            if 0 < left <= PAINT_LIFETIME:
                r = PAINT_SIZE // 2
                # Paint.java: alpha = intensity / 2, refreshed every 4 ticks.
                d.ellipse((x - r, y - r, x + r, y + r), fill=PAINT_RGB[c] + (int(left) // 4 * 4 // 2,))
        img = Image.alpha_composite(img, paint)
        d = ImageDraw.Draw(img)
        for p, (x, y, n) in enumerate(zip(f["pile_x"], f["pile_y"], f["pile_n"])):
            # Tomatoes a little larger than TomatoPile's 4 px, so piles stay
            # visible at half size.
            for ox, oy in self.tomato_offsets[p][:max(int(n), 0)]:
                tx, ty = x - 16 + ox, y - 16 + oy
                d.ellipse((tx - 3, ty - 3, tx + 3, ty + 3), fill=TOMATO_RGB[1], outline=TOMATO_RGB[2])
                d.point((tx - 1, ty - 1), fill=TOMATO_RGB[0])
        sx, sy = int(f["ship_x"]), int(f["ship_y"])
        img.alpha_composite(self.ship, (sx - self.ship.width // 2, sy - self.ship.height // 2))
        for x, y, rot, carry in zip(f["x"], f["y"], f["rot"], f["carry"]):
            s = self.sprite(carry, rot)
            img.alpha_composite(s, (int(x) - s.width // 2, int(y) - s.height // 2))
        d = ImageDraw.Draw(img)
        seconds = max(self.seconds - (int(f["t"]) - 1) // TICKS_PER_SECOND, 0)
        d.text((sx, sy - 45), f"Tomatoes: {int(f['delivered'])}", fill="white", font=font(18),
               anchor="mm", stroke_width=2, stroke_fill="black")
        d.text((700, 570), f"Time: {seconds}", fill="white", font=font(18), anchor="mm",
               stroke_width=2, stroke_fill="black")
        return img.convert("RGB")


class Senses:
    """
    What a greep senses of the others: the paint under it (dot colour, grey
    for none) and whether it carries a tomato (white centre). The map is only
    a faint outline for the viewer; greeps never see it.
    """

    def __init__(self, water):
        ghost = np.where(water[..., None], np.array(GHOST, np.uint8), np.array(DARK, np.uint8))
        self.background = Image.fromarray(ghost.astype(np.uint8))

    def draw(self, f):
        img = self.background.copy()
        d = ImageDraw.Draw(img)
        for o, x, y in zip(f["obs"], f["x"], f["y"]):
            x, y = int(x), int(y)
            under = [PAINT_RGB[c] for c in range(3) if o[OBS_PAINT + c] > 0.5]
            body = tuple(int(v) for v in np.mean(under, 0)) if under else (110, 118, 129)
            d.ellipse((x - 10, y - 10, x + 10, y + 10), fill=body)
            if o[OBS_CARRYING] > 0.5:
                d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 255))
        return img


def render(args):
    device = torch.device("cuda")
    bank, background = load_map(args.map, device)
    frames, score = play(args, bank, device)
    views = []
    if args.view in ("greenfoot", "both"):
        views.append((Greenfoot(background, frames[0]["pile_n"], args.seconds), "Greenfoot"))
    if args.view in ("senses", "both"):
        views.append((Senses(bank.water[0].cpu().numpy()), "What the greeps sense"))
    size = (int(W * args.scale), int(H * args.scale))
    images = []
    for f in frames:
        panels = []
        for view, label in views:
            panel = view.draw(f).resize(size, Image.LANCZOS)
            if args.border:     # GitHub strips CSS from READMEs, so the frame is drawn in
                ImageDraw.Draw(panel).rectangle((0, 0, size[0] - 1, size[1] - 1), outline="black",
                                                width=args.border)
            if args.view == "both":
                ImageDraw.Draw(panel).text((8, 6), label, fill="white", font=font(14),
                                           stroke_width=2, stroke_fill="black")
            panels.append(panel)
        frame = Image.new("RGB", (size[0] * len(panels) + 4 * (len(panels) - 1), size[1]), DARK)
        for k, p in enumerate(panels):
            frame.paste(p, (k * (size[0] + 4), 0))
        images.append(frame)
    images += [images[-1]] * int(args.fps * args.hold)     # pause on the final score
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = int(1000 / args.fps)
    if out.suffix == ".gif":
        palette = [im.quantize(colors=args.colors, method=Image.Quantize.FASTOCTREE) for im in images]
        palette[0].save(out, save_all=True, append_images=palette[1:], duration=duration,
                        loop=0, optimize=True, disposal=1)
    else:
        images[0].save(out, save_all=True, append_images=images[1:], duration=duration, loop=0,
                       quality=args.quality, method=6)
    print(f"{out}: {len(images)} frames, score {score}, {out.stat().st_size / 1024:.0f} KB")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--policy", default=None, help="approach name (default: the active one)")
    p.add_argument("--checkpoint", help="a training checkpoint instead of the embedded weights")
    p.add_argument("--map", default="real:1", help="a .map file, or real:1 .. real:3")
    p.add_argument("--view", choices=("greenfoot", "senses", "both"), default="greenfoot")
    p.add_argument("--out", default="out/render/episode.webp", help=".webp or .gif")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=8, help="played at once; the median is drawn")
    p.add_argument("--seconds", type=int, default=GAME_SECONDS, help="game length (real game: 120)")
    p.add_argument("--every", type=int, default=4, help="draw every n-th tick")
    p.add_argument("--fps", type=float, default=20)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--border", type=int, default=2, help="black frame around each panel, px (0: none)")
    p.add_argument("--hold", type=float, default=1.5, help="seconds to show the final frame")
    p.add_argument("--quality", type=int, default=60, help="WebP quality")
    p.add_argument("--colors", type=int, default=64, help="GIF palette size")
    args = p.parse_args()
    args.policy = args.policy or policies.active_name()
    render(args)


if __name__ == "__main__":
    main()
