"""Locations inside the repository."""

from pathlib import Path


SCENARIO_DIR = Path(__file__).resolve().parent.parent   # the Greenfoot scenario root
SRC_DIR = SCENARIO_DIR / "src"
GREEP_JAVA = SRC_DIR / "Greep.java"                     # the active policy's Greep
POLICIES_DIR = SCENARIO_DIR / "policies"
OUT_DIR = SCENARIO_DIR / "out"                          # build output, runs, caches
POLICY_FILE = SCENARIO_DIR / "policy.weights"           # optional override of a Greep's weights
