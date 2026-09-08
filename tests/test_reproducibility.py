"""Round-5 reproducibility/determinism audit tests.

Locks in guarantees that already hold today but had no regression coverage: cross-process
byte-identical output for a fixed seed, immunity to unrelated global numpy random state, and a
static guard against ever regressing to bare (non-reproducible) global `np.random.*` calls.
"""
import subprocess
import sys
import hashlib
import re
from pathlib import Path

import numpy as np

from selection_fragility.panel import LossPanel
from selection_fragility.report import report

_SRC = Path(__file__).resolve().parent.parent / "src" / "selection_fragility"

_REPRO_SNIPPET = """
import numpy as np
from selection_fragility.panel import LossPanel
from selection_fragility.report import report
rng = np.random.default_rng(42)
L = {'m0': rng.normal(1.0,0.3,30), 'm1': rng.normal(1.05,0.3,30), 'm2': rng.normal(1.2,0.3,30)}
p = LossPanel.from_losses(L)
out = report(p)
import hashlib
print(hashlib.sha256(out.encode()).hexdigest())
"""


def test_same_seed_byte_identical_across_fresh_process_invocations():
    """A fixed seed must give byte-identical report() output across two completely separate
    Python process invocations, not just within one session -- catches any dependency on
    leftover global random state, import order, or process-local caching."""
    hashes = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-c", _REPRO_SNIPPET],
            capture_output=True, text=True, check=True,
        )
        hashes.append(result.stdout.strip())
    assert len(hashes) == 2 and hashes[0] == hashes[1] and len(hashes[0]) == 64


def test_output_immune_to_unrelated_global_numpy_random_state():
    """report() must give the same result regardless of what the GLOBAL numpy random state
    happens to be beforehand -- every randomness-using function in this package takes an
    explicit seed via np.random.default_rng(seed), never the global np.random module, so
    perturbing global state first should have zero effect."""
    rng = np.random.default_rng(7)
    L = {"a": rng.normal(1.0, 0.2, 25), "b": rng.normal(1.1, 0.2, 25), "c": rng.normal(0.95, 0.2, 25)}
    panel = LossPanel.from_losses(L)

    np.random.seed(123)  # perturb GLOBAL state deliberately
    out1 = report(panel)

    np.random.seed(999999)  # perturb it again, differently
    out2 = report(panel)

    assert out1 == out2


def test_no_bare_global_numpy_random_calls_in_source():
    """Static guard: every randomness-using call in the shipped source must go through an
    explicit np.random.default_rng(...)-derived Generator, never the bare global np.random.X()
    API (e.g. np.random.normal(...), np.random.seed(...) used as the SOURCE of randomness
    rather than to perturb ambient state in a test). A regression here would silently
    reintroduce cross-call/import-order non-determinism. np.random.seed(...) calls inside
    THIS test file are exempt (they deliberately perturb global state to prove immunity above)."""
    offenders = []
    pattern = re.compile(r"np\.random\.(?!default_rng\b)(\w+)")
    for f in sorted(_SRC.glob("*.py")):
        text = f.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for m in pattern.finditer(line):
                # allow referencing np.random.Generator/np.random.default_rng in comments/type hints
                if m.group(1) in ("Generator",):
                    continue
                offenders.append(f"{f.name}:{lineno}: {stripped}")
    assert not offenders, "bare global np.random.* call(s) found:\n" + "\n".join(offenders)


def test_decimal_quantized_panel_deterministic_across_repeats():
    """Real-shaped decimal-quantized data (this project's actual data shape) run twice with the
    same seed must match exactly -- the jitter/tie-break machinery (identify.py) uses a fixed
    internal seed and must not introduce run-to-run drift on realistic inputs."""
    L = {
        "m0": np.array([0.12, 0.07, 1.98, 1.20, 0.36, 1.02, 0.32, 1.38]),
        "m1": np.array([0.33, 1.14, 1.24, 1.90, 0.88, 1.88, 0.00, 1.71]),
        "m2": np.array([0.50, 0.60, 1.10, 1.00, 0.70, 1.50, 0.40, 1.20]),
    }
    panel = LossPanel.from_losses(L)
    out1 = report(panel)
    out2 = report(panel)
    assert out1 == out2
