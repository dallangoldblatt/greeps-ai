"""
Check that the headless simulator reproduces the real Greenfoot GUI run.

Both runs are seeded identically and write a trace with one line per act
round (every greep's position, rotation, cargo, memory and flags, plus ship,
pile and paint counts) and one END line per map with its score. The traces
must match line for line.

Run from the project root:

    python -m unittest discover -s tests -v

The GUI tests open a Greenfoot window and take a minute or two per seed. Set
GREEPS_SKIP_GUI=1 to run only the fast headless checks.

The tests use whichever approach src/Greep.java plays (tools/use_policy.py),
with that approach's fixture weights (policies/<name>/fixture_weights), not
the trained ones. Greenfoot itself is only reproducible within one JVM history:
its collision tree stores actors in HashMaps keyed by identity hash codes,
so when several greeps touch a loader on a pile, which one
getOneIntersectingObject returns depends on everything the JVM hashed
before. Policies whose greeps crowd piles therefore give slightly different
(equally valid) games in a fresh JVM, after other episodes, or in the GUI.
The fixture's greeps never wait on piles, which keeps such ties rare enough
for exact comparisons.
"""

import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from greeps import policies  # noqa: E402
from greeps.java_sim import (  # noqa: E402
    HeadlessPool,
    compile_scenario,
    read_policy,
    run_gui_episode,
    run_headless,
    write_policy,
)


GUI_PARITY_SEEDS = [1, 2]
SKIP_GUI = os.environ.get("GREEPS_SKIP_GUI") == "1"
NUM_MAPS = 3

FIXTURE = Path(tempfile.mkdtemp(prefix="greeps-policy-")) / "fixture.weights"
PROPS = {"greeps.policy": FIXTURE}


def setUpModule():
    compile_scenario()
    name = policies.active_name()
    write_policy(policies.load(name).fixture_weights(), FIXTURE,
                 comment=f"test fixture for {name}")


def read_trace(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise AssertionError(f"Trace {path} is empty")
    return lines


def first_difference(expected, actual):
    """Return (index, expected_line, actual_line) of the first mismatch."""
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e != a:
            return i, e, a
    if len(expected) != len(actual):
        i = min(len(expected), len(actual))
        return (
            i,
            expected[i] if i < len(expected) else "<end of trace>",
            actual[i] if i < len(actual) else "<end of trace>",
        )
    return None


def describe_mismatch(diff, expected_name, actual_name):
    index, expected, actual = diff
    return (
        f"Traces diverge at line {index + 1}:\n"
        f"  {expected_name}: {expected[:400]}\n"
        f"  {actual_name}: {actual[:400]}"
    )


class TraceTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="greeps-trace-"))

    def assertTracesEqual(self, expected, actual, expected_name, actual_name):
        diff = first_difference(expected, actual)
        if diff is not None:
            self.fail(describe_mismatch(diff, expected_name, actual_name))

    def assertWellFormed(self, trace):
        end_lines = [line for line in trace if line.startswith("END ")]
        self.assertEqual(len(end_lines), NUM_MAPS, "one END line per map")
        # Greeps actually got released and moved around.
        self.assertTrue(any(" G " in line for line in trace))


class HeadlessDeterminismTest(TraceTestCase):
    def test_same_seed_gives_identical_runs(self):
        first = run_headless(seed=7, trace_dir=self.tmp / "a", props=PROPS)
        second = run_headless(seed=7, trace_dir=self.tmp / "b", props=PROPS)

        self.assertEqual(first, second)
        a = read_trace(self.tmp / "a" / "trace-seed7.txt")
        b = read_trace(self.tmp / "b" / "trace-seed7.txt")
        self.assertWellFormed(a)
        self.assertTracesEqual(a, b, "run 1", "run 2")

    def test_episodes_in_one_jvm_match_fresh_jvms(self):
        # Greenfoot keeps some static state (actor sequence numbers, BSP node
        # cache); batching episodes in one JVM must not change the results.
        batched = run_headless(seed=10, episodes=3, trace_dir=self.tmp / "batch",
                               props=PROPS)

        for i, seed in enumerate([10, 11, 12]):
            fresh = run_headless(seed=seed, trace_dir=self.tmp / f"fresh{seed}",
                                 props=PROPS)
            self.assertEqual(batched[i], fresh[0], f"seed {seed}")
            self.assertTracesEqual(
                read_trace(self.tmp / f"fresh{seed}" / f"trace-seed{seed}.txt"),
                read_trace(self.tmp / "batch" / f"trace-seed{seed}.txt"),
                "fresh JVM",
                "batched JVM",
            )

    def test_training_server_matches_cli(self):
        # The trainer swaps weights into a long-lived server JVM; that must
        # play exactly like a fresh run (and hence the GUI) with that policy.
        seeds = [3, 4, 5]
        cli = run_headless(seed=3, episodes=3, props=PROPS)
        weights = read_policy(FIXTURE)
        with HeadlessPool(workers=2) as pool:
            # A different policy first, so stale state would show up.
            pool.evaluate([weights * 0.5], seeds)
            served = pool.evaluate([weights], seeds)[0]
        self.assertEqual(served.tolist(), cli)

    def test_different_seeds_differ(self):
        # Guards against the seed silently not being applied.
        a, b = run_headless(seed=100, episodes=2, trace_dir=self.tmp, props=PROPS)
        trace_a = read_trace(self.tmp / "trace-seed100.txt")
        trace_b = read_trace(self.tmp / "trace-seed101.txt")
        self.assertNotEqual(trace_a, trace_b)


@unittest.skipIf(SKIP_GUI, "GREEPS_SKIP_GUI=1")
class GuiParityTest(TraceTestCase):
    def test_headless_matches_gui(self):
        for seed in GUI_PARITY_SEEDS:
            with self.subTest(seed=seed):
                gui_trace_path = self.tmp / f"gui-seed{seed}.txt"
                gui_scores = run_gui_episode(
                    timeout_seconds=300,
                    props={
                        **PROPS,
                        "greeps.seed": seed,
                        "greeps.trace": gui_trace_path,
                    },
                )
                headless_scores = run_headless(
                    seed=seed, trace_dir=self.tmp / "headless", props=PROPS
                )[0]

                gui_trace = read_trace(gui_trace_path)
                headless_trace = read_trace(
                    self.tmp / "headless" / f"trace-seed{seed}.txt"
                )

                self.assertWellFormed(gui_trace)
                self.assertTracesEqual(
                    gui_trace, headless_trace, "GUI", "headless"
                )
                self.assertEqual(
                    [int(s) for s in gui_scores], headless_scores
                )


if __name__ == "__main__":
    unittest.main()
