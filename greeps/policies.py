"""
Policies (approaches) and which one the scenario currently plays.

Each approach lives in policies/<name>/ with its own Greep.java (the policy
as the competition sees it) and its Python code (training, export, a GPU
twin for the torch simulator). Greenfoot compiles exactly one
src/Greep.java, so activate() copies the chosen approach's Greep.java there;
`python -m tools.use_policy <name>` does that from the command line.

A policy package (policies/<name>/__init__.py) provides:

    NAME                    the folder name; Greep.java's POLICY_NAME
    WEIGHTS_DTYPE           how Greep.java stores its embedded weights ("<f4", "<f8")
    torch_policy(weights, device, generator=None)
                            the same policy for the GPU simulator:
                            a function (obs, sim) -> actions
    fixture_weights()       a deterministic policy whose greeps never wait on
                            piles, for the exact-parity tests

Trained weights are embedded in Greep.java itself (embed_weights), so the
submitted file needs nothing else.
"""

import base64
import importlib
import re

import numpy as np

from .paths import GREEP_JAVA, POLICIES_DIR


_NAME = re.compile(r'POLICY_NAME\s*=\s*"([^"]+)"')
BEGIN = "    // BEGIN GENERATED WEIGHTS"
END = "    // END GENERATED WEIGHTS"
# Java string constants are limited to 65535 bytes.
_CHUNK = 60000


def available():
    return sorted(p.name for p in POLICIES_DIR.iterdir() if (p / "Greep.java").exists())


def greep_java(name):
    return POLICIES_DIR / name / "Greep.java"


def name_of(java_path):
    match = _NAME.search(java_path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"{java_path} has no POLICY_NAME")
    return match.group(1)


def active_name():
    """The approach src/Greep.java implements (None if it declares none)."""
    try:
        return name_of(GREEP_JAVA)
    except ValueError:
        return None


def load(name=None):
    """The policy package (default: the active one)."""
    name = name or active_name()
    if name is None:
        raise ValueError("src/Greep.java is not one of the approaches in policies/")
    return importlib.import_module(f"policies.{name}")


def src_in_sync():
    """True if src/Greep.java is an unmodified copy of its approach's Greep.java."""
    name = active_name()
    return (name in available()
            and GREEP_JAVA.read_text(encoding="utf-8")
            == greep_java(name).read_text(encoding="utf-8"))


def activate(name, force=False):
    """Copy policies/<name>/Greep.java to src/Greep.java."""
    if name not in available():
        raise ValueError(f"Unknown policy {name!r}; available: {', '.join(available())}")
    if GREEP_JAVA.exists() and not force and not src_in_sync():
        raise RuntimeError(
            "src/Greep.java has edits that are not in its policies/<name>/Greep.java. "
            "Copy them there first, or pass force=True (--force) to discard them.")
    GREEP_JAVA.write_text(greep_java(name).read_text(encoding="utf-8"),
                          encoding="utf-8", newline="\n")


def read_embedded_weights(name, dtype=None):
    """The weights embedded in policies/<name>/Greep.java (empty if none)."""
    dtype = dtype or load(name).WEIGHTS_DTYPE
    source = greep_java(name).read_text(encoding="utf-8")
    block = source[source.index(BEGIN):source.index(END)]
    block = block[block.index("{"):]
    encoded = "".join(re.findall(r'"([A-Za-z0-9+/=]*)"', block))
    return np.frombuffer(base64.b64decode(encoded), dtype=np.dtype(dtype).newbyteorder("<")).copy()


def embed_weights(name, weights, dtype, note):
    """
    Write weights into policies/<name>/Greep.java between the GENERATED
    WEIGHTS markers (base64 of little-endian `dtype`, which that Greep
    decodes), and refresh src/Greep.java if this approach is active.
    Returns a message describing what was updated.
    """
    java = greep_java(name)
    was_in_sync = active_name() == name and src_in_sync()
    encoded = base64.b64encode(np.asarray(weights, dtype=np.dtype(dtype).newbyteorder("<"))
                               .tobytes()).decode("ascii")
    chunks = [encoded[i:i + _CHUNK] for i in range(0, len(encoded), _CHUNK)]
    block = (f"{BEGIN} (greeps/policies.py)\n"
             f"    // {note}\n"
             "    private static final String[] EMBEDDED_WEIGHTS = {\n"
             + "".join(f'        "{c}",\n' for c in chunks)
             + "    };\n"
             + END)
    source = java.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(BEGIN) + ".*?" + re.escape(END), re.S)
    if not pattern.search(source):
        raise RuntimeError(f"No GENERATED WEIGHTS block in {java}")
    java.write_text(pattern.sub(lambda _: block, source), encoding="utf-8", newline="\n")

    message = f"Embedded {len(weights)} weights into {java}"
    if was_in_sync:
        activate(name, force=True)
        message += " and src/Greep.java"
    elif active_name() == name:
        message += " (src/Greep.java has local edits and was NOT updated)"
    else:
        message += f" (active policy is {active_name()}; "
        message += f"python -m tools.use_policy {name} to play it)"
    return message
