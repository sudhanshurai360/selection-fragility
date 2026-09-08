"""Round-8 concurrency/parallel-execution safety review -- a category no prior round (1-7) tested.
Round 6's demand-planning reviewer tested a SEQUENTIAL tight loop (1000+ calls, one after another)
and found no cross-call contamination; that says nothing about true concurrent execution, which is
how a real production pipeline would actually scale this (joblib/multiprocessing/ThreadPoolExecutor
fanning out across many series/SKUs/states at once).

Findings:
- multiprocessing: safe (processes have independent memory; confirmed empirically).
- threading, realistic use (concurrent calls on DIFFERENT panels, nobody touches module globals):
  safe (confirmed empirically, 600 concurrent calls across 200 panels via 16 threads).
- threading, the `elimination=None` -> module-level `ELIMINATION` global read-at-call-time pattern
  (mcs.py, intentional design, documented): inherently unsynchronized shared mutable state if two
  concurrent call sites want DIFFERENT elimination rules at the same time. 4,000 trials of adversarial
  concurrent mutation produced 0 observed mismatches -- but this is an empirical non-failure, not a
  proof of safety; the race window is real, just narrow. Not fixed here (find-only round) -- flagging
  as a real, if low-probability, design-level footgun for a caller who parallelizes across MULTIPLE
  elimination rules in the same process using the global-override pattern instead of the safer
  explicit `elimination=` kwarg.
- LossPanel.save()/.load() on a SHARED path under concurrent access: NOT safe, confirmed with a real,
  reproducible failure (299/800 concurrent reads hit JSONDecodeError from a partially-written file).
  Root cause: `save()` does `open(path, "w")` directly (panel.py:556) -- no write-to-temp-then-atomic-
  rename. A reader can observe a truncated/partial write mid-flight. This is a real, disclosed-here
  bug, not fixed in this find-only round.
"""
import json
import os
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pytest

from selection_fragility import decision_breakdown
from selection_fragility.panel import LossPanel


def _make_panel(seed):
    rng = np.random.default_rng(seed)
    K = int(rng.integers(2, 6))
    T = int(rng.integers(10, 40))
    return {f"m{i}": rng.normal(1.0 + 0.05 * i, 0.3, T) for i in range(K)}


def _run_one(seed):
    L = _make_panel(seed)
    k, opp, removed = decision_breakdown(L)
    return (seed, k, opp, tuple(sorted(removed)))


def test_multiprocessing_independent_panels_match_sequential():
    """50 distinct panels via 8 worker processes must match single-process sequential results
    exactly -- processes have independent memory, so this mainly guards against any hidden
    reliance on shared filesystem/OS state."""
    seeds = list(range(50))
    sequential = [_run_one(s) for s in seeds]
    with ProcessPoolExecutor(max_workers=8) as ex:
        parallel = list(ex.map(_run_one, seeds))
    assert sequential == parallel


def test_threading_concurrent_different_panels_match_sequential():
    """The realistic production pattern: many threads, each computing a DIFFERENT panel, nobody
    touching module-level config globals. 200 panels x 3 repeats = 600 concurrent calls across 16
    threads must all match their single-threaded sequential result exactly."""
    seeds = list(range(200))
    sequential = {s: _run_one(s)[1:] for s in seeds}

    lock = threading.Lock()
    mismatches = []

    def worker(s):
        got = _run_one(s)[1:]
        if got != sequential[s]:
            with lock:
                mismatches.append((s, got, sequential[s]))

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(worker, seeds * 3))

    assert mismatches == []


def test_concurrent_save_load_same_path_does_not_corrupt():
    """LossPanel.save() used to write directly to the target path (`open(path, "w")`) with no
    write-to-temp-then-atomic-rename -- a reader could observe a partially-written file mid-save.
    Reproduced directly by round-8's concurrency review: 3 writer threads + 5 reader threads, 100
    iterations each, all hitting the SAME path -- real JSONDecodeErrors from truncated reads, not a
    flake (hundreds of failures out of ~800 reads). FIXED 2026-08-27: save() now writes to a temp
    file in the same directory and atomically `os.replace()`s it into place, so a concurrent reader
    always sees either the complete old file or the complete new one. Flipped from the original
    KNOWN_BUG-documenting test (which asserted the corruption reproduced) to this one, which asserts
    it no longer does -- per that test's own explicit instructions."""
    rng = np.random.default_rng(0)
    L = {f"m{i}": rng.normal(1.0, 0.3, 20) for i in range(3)}
    panel = LossPanel.from_losses(L)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "shared.json")
        panel.save(path)

        errors = []
        lock = threading.Lock()

        def writer(n):
            for _ in range(n):
                panel.save(path)

        def reader(n):
            for _ in range(n):
                try:
                    LossPanel.load(path)
                except json.JSONDecodeError as e:
                    with lock:
                        errors.append(e)

        threads = (
            [threading.Thread(target=writer, args=(100,)) for _ in range(3)]
            + [threading.Thread(target=reader, args=(100,)) for _ in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], (
            f"concurrent save/load corruption reproduced ({len(errors)} JSONDecodeErrors) -- "
            "save() should be writing atomically (temp file + os.replace); this regressed."
        )
