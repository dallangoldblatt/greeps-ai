"""
Check that the GPU simulator (greeps/torch_sim.py) plays like the real one.

The GPU sim is not bit-identical to Greenfoot (greeps act simultaneously,
see its docstring), so the active Greep's embedded policy is played on all
three maps in both simulators (in the GPU sim through its policy package's
torch_policy twin), and the mean scores must agree within their sampling
error plus a small tolerance.

    python -m unittest tests.test_torch_sim -v

Needs a CUDA GPU; skipped otherwise.
"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from greeps import policies  # noqa: E402
from greeps.java_sim import HeadlessPool, compile_scenario  # noqa: E402
from greeps.paths import POLICY_FILE  # noqa: E402


JAVA_SEEDS = 64
GPU_EPISODES = 256


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class TorchSimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from greeps.torch_sim import MapBank

        if POLICY_FILE.exists():
            raise unittest.SkipTest(f"{POLICY_FILE.name} overrides the embedded policy")
        compile_scenario()
        cls.device = torch.device("cuda")
        cls.bank = MapBank.real(cls.device)
        cls.policy = policies.load()
        with HeadlessPool(workers=14) as pool:
            cls.weights = pool.default_weights()
            seeds = [3_000_000 + i for i in range(JAVA_SEEDS)]
            cls.java = pool.evaluate([cls.weights], seeds)[0]     # (seeds, maps)

    def play_gpu(self, seed):
        from greeps.torch_sim import evaluate

        gen = torch.Generator(device=self.device).manual_seed(seed)
        policy = self.policy.torch_policy(self.weights, self.device, gen)
        with torch.no_grad():
            return evaluate(policy, self.bank, [0, 1, 2], GPU_EPISODES,
                            self.device, seed=seed)                  # (maps, episodes)

    def test_scores_match_java(self):
        gpu = self.play_gpu(seed=11)
        for m in range(3):
            j, g = self.java[:, m], gpu[m]
            se = np.sqrt(j.var(ddof=1) / len(j) + g.var(ddof=1) / len(g))
            tolerance = 4 * se + 0.05 * j.mean() + 1.0
            with self.subTest(policy=self.policy.NAME, map=m + 1):
                self.assertLessEqual(
                    abs(g.mean() - j.mean()), tolerance,
                    f"{self.policy.NAME} Map {m + 1}: GPU {g.mean():.2f} vs "
                    f"Java {j.mean():.2f} (tolerance {tolerance:.2f})")

    def test_deterministic_for_seed(self):
        a = self.play_gpu(seed=5)
        b = self.play_gpu(seed=5)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
