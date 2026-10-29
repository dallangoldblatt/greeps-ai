"""
A browser-based editor for hand-made Greeps maps.

    python -m tools.map_editor            # opens http://127.0.0.1:8765

Maps are saved to maps/<name>.png (800x600; water is blue, as Earth.isWater
sees it) plus maps/<name>.map (ship and piles). They can then be used for
training and evaluation:

    python -m policies.mlp_ppo.train --custom-maps maps
    python -m tools.evaluate --maps maps          # real Java sim
"""

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse
import webbrowser

from greeps.maps import CUSTOM_MAP_DIR, REAL_MAPS, read_map_file, write_map_file
from greeps.paths import SCENARIO_DIR


EDITOR_HTML = Path(__file__).resolve().parent / "map_editor.html"
# Map names are paths under maps/ without the extension, e.g. "training/built_01".
NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}(/[A-Za-z0-9_-]{1,64}){0,3}$")


def _inside_maps(path):
    """The resolved path, if it lies inside maps/; otherwise None."""
    path = path.resolve()
    return path if path.is_relative_to(CUSTOM_MAP_DIR.resolve()) else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, body, content_type="application/json", status=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message, status=400):
        self._send({"error": message}, status=status)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self._send(EDITOR_HTML.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/maps":
            custom = sorted(p.relative_to(CUSTOM_MAP_DIR).with_suffix("").as_posix()
                            for p in CUSTOM_MAP_DIR.rglob("*.map"))
            self._send({"custom": custom, "real": [m["name"] for m in REAL_MAPS]})
        elif url.path == "/api/map":
            self._get_map(parse_qs(url.query).get("name", [""])[0])
        elif url.path.startswith("/image/"):
            self._get_image(url.path[len("/image/"):])
        else:
            self._error("not found", 404)

    def _get_map(self, name):
        for i, m in enumerate(REAL_MAPS):
            if m["name"] == name:
                return self._send({"name": name, "image": f"/image/real/{i}",
                                   "ship": m["ship"], "piles": m["piles"], "real": True})
        if NAME.match(name) and not (CUSTOM_MAP_DIR / f"{name}.map").exists() and "/" not in name:
            # A bare name like "built_01" finds maps/training/built_01.map if unique.
            matches = list(CUSTOM_MAP_DIR.rglob(f"{name}.map"))
            if len(matches) == 1:
                name = matches[0].relative_to(CUSTOM_MAP_DIR).with_suffix("").as_posix()
        if not NAME.match(name) or not (CUSTOM_MAP_DIR / f"{name}.map").exists():
            return self._error(f"No map named {name!r} in {CUSTOM_MAP_DIR}", 404)
        try:
            m = read_map_file(CUSTOM_MAP_DIR / f"{name}.map")
        except ValueError as e:
            return self._error(str(e))
        image = _inside_maps(m["image"])
        if image is None or not image.exists():
            return self._error(f"{name}.map refers to a missing image {m['image']}", 404)
        rel = image.relative_to(CUSTOM_MAP_DIR.resolve()).as_posix()
        self._send({"name": name, "image": f"/image/custom/{rel}",
                    "ship": m["ship"], "piles": m["piles"], "real": False})

    def _get_image(self, rest):
        kind, _, file = rest.partition("/")
        if kind == "real" and file.isdigit() and int(file) < len(REAL_MAPS):
            path = SCENARIO_DIR / "src" / "images" / f"map{int(file)}.jpg"
            return self._send(path.read_bytes(), "image/jpeg")
        if kind == "custom":
            path = _inside_maps(CUSTOM_MAP_DIR / file)
            if path and path.suffix.lower() in (".png", ".jpg", ".jpeg") and path.exists():
                kind = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
                return self._send(path.read_bytes(), kind)
        self._error("not found", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            return self._error("not found", 404)
        try:
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            name = data["name"]
            if not NAME.match(name):
                return self._error("name: letters, digits, - and _ (folders separated by /)")
            if any(m["name"] == name for m in REAL_MAPS):
                return self._error("that name is reserved for a scenario map")
            prefix = "data:image/png;base64,"
            if not data["png"].startswith(prefix):
                return self._error("expected a PNG data URL")
            ship = [int(v) for v in data["ship"]]
            piles = [(int(x), int(y), int(n)) for x, y, n in data["piles"]]
            if not piles:
                return self._error("a map needs at least one pile")
        except (KeyError, TypeError, ValueError) as e:
            return self._error(f"bad request: {e}")

        map_path = CUSTOM_MAP_DIR / f"{name}.map"
        stem = map_path.stem
        map_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep notes from an existing file (e.g. "Estimated: ..." from the importer).
        notes = []
        if map_path.exists():
            notes = [line for line in map_path.read_text(encoding="utf-8").splitlines()
                     if line.startswith("#") and "map_editor.py" not in line]
        (map_path.parent / f"{stem}.png").write_bytes(base64.b64decode(data["png"][len(prefix):]))
        write_map_file(map_path, f"{stem}.png", ship, piles)
        if notes:
            map_path.write_text("\n".join(notes) + "\n" + map_path.read_text(encoding="utf-8"),
                                encoding="utf-8")
        self._send({"saved": str(map_path)})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Greeps map editor at {url}  (maps are saved to {CUSTOM_MAP_DIR})")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
