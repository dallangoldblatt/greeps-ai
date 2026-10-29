"""
Run the real (Java) Greeps scenario from Python, headless or in the GUI.

The headless runner (headless/HeadlessGreeps.java) uses the real scenario and
Greenfoot classes with the GUI simulation thread replaced by a plain loop, so
with the same seed it reproduces the GUI run exactly, in a fraction of the
time. It plays whichever policy src/Greep.java currently implements (see
greeps/policies.py).
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import time

import numpy as np

from .maps import map_files
from .paths import OUT_DIR, POLICY_FILE, SCENARIO_DIR


SCORE_FILE = SCENARIO_DIR / "training_score.txt"
HEADLESS_SOURCE = SCENARIO_DIR / "headless" / "HeadlessGreeps.java"
HEADLESS_CLASSES = OUT_DIR / "headless"
WATER_DIR = OUT_DIR / "maps"

def _find_java_bin():
    """
    The JDK's bin directory: $JAVA_HOME/bin, else the one on PATH, else the
    JDK bundled with VS Code's Java extension (redhat.java), newest first.
    """
    if os.environ.get("JAVA_HOME"):
        return Path(os.environ["JAVA_HOME"]) / "bin"
    javac = shutil.which("javac")
    if javac:
        return Path(javac).parent
    bundled = sorted((Path.home() / ".vscode" / "extensions").glob("redhat.java-*/jre/*/bin"))
    if bundled:
        return bundled[-1]
    raise RuntimeError("No JDK found: set JAVA_HOME or put javac on PATH")


JAVA_BIN = _find_java_bin()
JAVA = str(JAVA_BIN / "java")
JAVAC = str(JAVA_BIN / "javac")

GREENFOOT_CLASSPATH = os.pathsep.join(["src", "lib/greenfoot-3.7.0.jar", "lib/bluejcore-3.7.0.jar"])
JAVAFX_ARGS = [
    "--module-path",
    "lib/javafx",
    "--add-modules",
    "javafx.controls,javafx.fxml",
]
# Headless JVMs only: without a cap each may grow its heap to a quarter of
# the machine's RAM, and a pool of a dozen ran the system out of memory.
HEADLESS_JVM_ARGS = ["-Xmx1g", *JAVAFX_ARGS]

JAVA_COMMAND = [
    JAVA,
    *JAVAFX_ARGS,
    "-Dgreeps.training=true",
    "-cp",
    GREENFOOT_CLASSPATH,
    "greenfoot.export.GreenfootScenarioApplication",
    "Greenfoot",
    "Earth",
]


def _jvm_props(props):
    return [f"-D{key}={value}" for key, value in (props or {}).items()]


def compile_scenario():
    """Compile src/*.java (in place, as Greenfoot does) and the runner."""
    sources = sorted(str(p) for p in (SCENARIO_DIR / "src").glob("*.java"))
    subprocess.run(
        [JAVAC, "-nowarn", "-cp", GREENFOOT_CLASSPATH, "-d", "src", *sources],
        cwd=SCENARIO_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            JAVAC, "-nowarn", "-cp", GREENFOOT_CLASSPATH,
            "-d", str(HEADLESS_CLASSES), str(HEADLESS_SOURCE),
        ],
        cwd=SCENARIO_DIR,
        check=True,
        capture_output=True,
        text=True,
    )


def custom_maps_prop(spec):
    """-D property that makes Earth play the given custom maps."""
    return {"greeps.maps": ",".join(str(p.resolve()) for p in map_files(spec))}


def run_headless(seed=None, episodes=1, trace_dir=None, timeout_seconds=600,
                 props=None):
    """
    Run `episodes` complete episodes in one headless JVM.

    Episode i uses seed `seed + i` (unseeded if `seed` is None). Returns a
    list of per-map score lists, one per episode. With `trace_dir`, each
    episode also writes trace-seed<N>.txt there. `props` are extra -D system
    properties, e.g. {"greeps.policy": path}.
    """
    command = [
        JAVA,
        *_jvm_props(props),
        *HEADLESS_JVM_ARGS,
        "-cp",
        f"{HEADLESS_CLASSES};{GREENFOOT_CLASSPATH}",
        "HeadlessGreeps",
        "--episodes",
        str(episodes),
    ]
    if seed is not None:
        command += ["--seed", str(seed)]
    if trace_dir is not None:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)
        command += ["--trace-dir", str(trace_dir)]

    result = subprocess.run(
        command,
        cwd=SCENARIO_DIR,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Headless run failed ({result.returncode}):\n{result.stderr}"
        )

    # The scenario itself prints to stdout too; only JSON lines are results.
    episodes_out = [
        json.loads(line)["scores"]
        for line in result.stdout.splitlines()
        if line.startswith("{")
    ]
    if len(episodes_out) != episodes:
        raise RuntimeError(
            f"Expected {episodes} results, got {len(episodes_out)}:\n"
            f"{result.stdout}"
        )
    return episodes_out


def dump_water(out_dir=WATER_DIR):
    """
    Export each map's water mask as Greenfoot sees it (Earth.isWater) to
    out_dir/water<m>.bin: 800*600 bytes, row-major, 1 = water.
    """
    subprocess.run(
        [
            JAVA,
            *HEADLESS_JVM_ARGS,
            "-cp",
            f"{HEADLESS_CLASSES};{GREENFOOT_CLASSPATH}",
            "HeadlessGreeps",
            "--dump-water",
            str(out_dir),
        ],
        cwd=SCENARIO_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(out_dir)


def write_policy(weights, path=POLICY_FILE, comment=None):
    """Write policy weights as text, one per line (Greep.java can read these)."""
    lines = [f"# {comment}"] if comment else []
    # repr() round-trips doubles exactly.
    lines += [repr(float(w)) for w in weights]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_policy(path=POLICY_FILE):
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            values += [float(token) for token in line.split()]
    return np.array(values)


class _ServerWorker:
    """One `HeadlessGreeps --server` JVM, talked to over stdin/stdout."""

    def __init__(self, props=None):
        self.process = subprocess.Popen(
            [
                JAVA,
                *_jvm_props(props),
                *HEADLESS_JVM_ARGS,
                "-cp",
                f"{HEADLESS_CLASSES};{GREENFOOT_CLASSPATH}",
                "HeadlessGreeps",
                "--server",
            ],
            cwd=SCENARIO_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def request(self, line):
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        reply = self.process.stdout.readline().strip()
        if not reply:
            raise RuntimeError("Headless server exited unexpectedly")
        if reply.startswith("ERROR"):
            raise RuntimeError(f"Headless server: {reply}")
        return reply

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write("QUIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        for pipe in (self.process.stdin, self.process.stdout):
            try:
                pipe.close()
            except OSError:
                pass


class HeadlessPool:
    """
    A pool of headless simulator processes for evaluating many weight vectors
    of the active Greep's policy.

        with HeadlessPool(workers=8) as pool:
            scores = pool.evaluate([w1, w2], seeds=[0, 1, 2])
            # scores.shape == (2 policies, 3 seeds, 3 maps)

    `props` are -D properties for every worker, e.g. custom_maps_prop("maps")
    to play custom maps instead of the scenario's three.
    """

    def __init__(self, workers=8, props=None):
        self._workers = [_ServerWorker(props) for _ in range(workers)]
        self._free = queue.Queue()
        for worker in self._workers:
            self._free.put(worker)
        self._executor = ThreadPoolExecutor(max_workers=workers)

        info = self._workers[0].request("INFO").split()
        self.num_params = int(info[1])
        self.policy_name = info[2]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._executor.shutdown(wait=True)
        for worker in self._workers:
            worker.close()

    def _vector(self, command):
        reply = self._workers[0].request(command)
        return np.array([float(v) for v in reply.split()[1:]])

    def default_weights(self):
        """The weights the Greep plays with (its embedded trained policy)."""
        return self._vector("DEFAULT")

    def initial_weights(self):
        """The Greep's suggested starting point for black-box training."""
        return self._vector("INITIAL")

    def search_scales(self):
        """The Greep's relative per-weight scales for black-box search."""
        return self._vector("SCALES")

    def _run(self, weights, seed):
        worker = self._free.get()
        try:
            reply = worker.request(
                f"EVAL {seed} 1 " + " ".join(repr(float(w)) for w in weights)
            )
        finally:
            self._free.put(worker)
        parts = reply.split()
        if len(parts) != 2 or parts[0] != "SCORES":
            raise RuntimeError(f"Unexpected reply for seed {seed}: {reply[:200]!r}")
        return [int(v) for v in parts[1].split(",")]

    def evaluate(self, policies, seeds):
        """
        Play every weight vector on every seed (one full episode each).

        Returns an int array of shape (len(policies), len(seeds), maps).
        """
        for weights in policies:
            if len(weights) != self.num_params:
                raise ValueError(
                    f"Expected {self.num_params} weights, got {len(weights)}"
                )
        futures = [
            [self._executor.submit(self._run, weights, seed) for seed in seeds]
            for weights in policies
        ]
        return np.array(
            [[future.result() for future in row] for row in futures]
        )


def read_score_file():
    """Read the per-map scores the GUI run writes to training_score.txt."""
    raw = SCORE_FILE.read_text(encoding="utf-8").strip()

    if not raw:
        raise ValueError("Score file is empty.")

    return [
        float(value)
        for value in raw.replace(",", "\n").split()
    ]


def run_gui_episode(timeout_seconds=240, props=None, expected_maps=3):
    """
    Run one episode in the real Greenfoot GUI and return its per-map scores.

    `props` are extra -D system properties, e.g. {"greeps.seed": 1}.
    """
    # Avoid accidentally reading the previous episode's score.
    SCORE_FILE.unlink(missing_ok=True)

    command = [JAVA_COMMAND[0], *_jvm_props(props), *JAVA_COMMAND[1:]]
    process = subprocess.Popen(
        command,
        cwd=str(SCENARIO_DIR),
    )

    start = time.monotonic()

    try:
        while time.monotonic() - start < timeout_seconds:
            if SCORE_FILE.exists():
                try:
                    scores = read_score_file()
                    if len(scores) == expected_maps:
                        return scores
                except (OSError, ValueError):
                    # The file may be incomplete while Java writes it.
                    pass

            if process.poll() is not None:
                raise RuntimeError(
                    "Greenfoot exited before publishing all "
                    f"{expected_maps} map scores. Check the Java command "
                    "and scenario output."
                )

            time.sleep(0.25)

        raise TimeoutError(
            f"No complete map scores after {timeout_seconds} seconds."
        )

    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
