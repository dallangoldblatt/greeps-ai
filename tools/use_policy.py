"""
Choose which approach the scenario plays: copies policies/<name>/Greep.java
to src/Greep.java (the file Greenfoot compiles and the competition receives).

    python -m tools.use_policy              # show the active approach
    python -m tools.use_policy linear_cma
    python -m tools.use_policy mlp_ppo
"""

import argparse

from greeps import policies


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name", nargs="?", choices=policies.available())
    parser.add_argument("--force", action="store_true",
                        help="overwrite src/Greep.java even if it has local edits")
    args = parser.parse_args()

    if args.name:
        try:
            policies.activate(args.name, force=args.force)
        except RuntimeError as e:
            raise SystemExit(str(e))
    active = policies.active_name()
    state = "" if policies.src_in_sync() else " (src/Greep.java has local edits)"
    print(f"Active policy: {active}{state}")
    print(f"Available: {', '.join(policies.available())}")


if __name__ == "__main__":
    main()
