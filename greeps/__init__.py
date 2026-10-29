"""
Shared infrastructure for training Greeps policies.

    paths       repository locations
    maps        the scenario's maps and custom .map files
    java_sim    the real scenario: compile, headless runs, worker pool, GUI
    torch_sim   a vectorized GPU port of the scenario, for training
    policies    the approaches in policies/ and which one src/Greep.java plays

Run scripts as modules from the repository root, e.g.
`python -m policies.mlp_ppo.train`.
"""
