"""Round-5 extensibility review (contributor/academic-building-on-top lens).

These tests lock in what a contributor extending this package can and cannot rely on -- distinct
from every other test file, which exercises the package as a black-box CONSUMER. Nothing here is a
bug fix; these document real behavior discovered while trying to build extensions on the public API.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

from selection_fragility import LossPanel, pooled_winner, breakdown_number, report


def pairwise_fragility_table(panel):
    """A REAL extension built entirely from the public API (LossPanel, pooled_winner,
    breakdown_number) with zero private imports -- for every non-champion model, its own
    breakdown_number k* against the current champion, sorted most- to least-fragile.
    Positive evidence for extensibility: this took no private/underscored access at all."""
    L, w = panel.losses, panel.weights
    champ = pooled_winner(L, w)
    rows = []
    for m in panel.models:
        if m == champ:
            continue
        k, removed = breakdown_number(L[champ], L[m], w)
        rows.append((m, k, len(removed)))
    return champ, sorted(rows, key=lambda r: r[1])


def test_pairwise_fragility_table_extension_builds_cleanly_on_public_api():
    rng = np.random.default_rng(0)
    L = {f"m{i}": rng.normal(1.0 + 0.05 * i, 0.3, 40) for i in range(5)}
    panel = LossPanel.from_losses(L)
    champ, table = pairwise_fragility_table(panel)
    assert champ == "m0"
    assert [m for m, k, n in table] == ["m2", "m3", "m1", "m4"]
    assert all(k >= 0 for m, k, n in table)


def test_mcs_submodule_shadowed_by_reexported_function_name():
    """KNOWN, non-obvious behavior for extenders, not a bug in normal (consumer) usage: __init__.py
    does `from .mcs import mcs, model_confidence_set`, which rebinds the `selection_fragility.mcs`
    ATTRIBUTE to the function `mcs`, shadowing the submodule at that same dotted path. A contributor
    who wants to reach mcs.py's internals (e.g. to monkeypatch `_block_idx` and try a different
    resampling scheme, exactly the kind of extension this round's lens is about) via the natural,
    idiomatic `import selection_fragility.mcs as mcs_mod; mcs_mod._block_idx` gets the FUNCTION, not
    the module, and a confusing AttributeError -- not a clear "use sys.modules instead" message.
    `from selection_fragility.mcs import _block_idx` (or any name) works fine, since that import form
    resolves the submodule directly rather than through the shadowed package attribute; that's the
    one reliable path documented here for future extenders."""
    import selection_fragility.mcs as shadowed
    assert callable(shadowed) and not hasattr(shadowed, "_block_idx"), (
        "if this assertion starts failing, the shadowing behavior changed (intentionally or not) -- "
        "update this test and consider whether README/CONTRIBUTING should now document the new state"
    )
    from selection_fragility.mcs import _block_idx  # the reliable path
    assert callable(_block_idx)
    real_module = sys.modules["selection_fragility.mcs"]
    assert hasattr(real_module, "_block_idx"), "sys.modules lookup is the reliable monkeypatch path"


def test_block_idx_resampling_scheme_is_swappable_via_sys_modules_monkeypatch():
    """A contributor CAN swap mcs()'s resampling scheme (e.g. stationary vs. circular block
    bootstrap) without forking the ~140-line mcs()/model_confidence_set() body, but only via the
    sys.modules workaround from the test above -- there is no injectable resampler parameter on
    mcs()/model_confidence_set() itself. This locks in that the workaround keeps working."""
    import selection_fragility.mcs as mcs_pkg_fn  # the shadowed function, unused here on purpose
    real_module = sys.modules["selection_fragility.mcs"]
    orig = real_module._block_idx
    calls = {"n": 0}

    def stationary_block_idx(T, block, rng, circular=True):
        calls["n"] += 1
        idx, i, p = [], rng.integers(0, T), 1.0 / block
        for _ in range(T):
            idx.append(i % T)
            i += 1
            if rng.random() < p:
                i = rng.integers(0, T)
        return np.array(idx)

    real_module._block_idx = stationary_block_idx
    try:
        from selection_fragility import model_confidence_set
        rng = np.random.default_rng(1)
        L = {f"m{i}": rng.normal(1.0 + 0.03 * i, 0.3, 30) for i in range(4)}
        surv, p = model_confidence_set(L, B=50, seed=0)
        assert calls["n"] == 50
        assert isinstance(surv, list) and len(surv) >= 1
    finally:
        real_module._block_idx = orig


def test_panel_strategy_reusable_by_new_contributor_tests_via_syspath():
    """Round-4's hypothesis strategy (_panel_strategy in test_property_based.py) IS reusable by a
    new contributor's own tests, but only after manually adding tests/ to sys.path -- tests/ has no
    __init__.py and isn't exposed as an importable fixture/plugin, so `from test_property_based
    import _panel_strategy` fails from a normal pytest collection context without this workaround.
    Documents the friction rather than fixing it (no conftest.py change made here)."""
    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from test_property_based import _panel_strategy
    from hypothesis import given, settings

    @given(L=_panel_strategy(k_max=3, t_max=10))
    @settings(max_examples=10, deadline=None)
    def a_new_contributors_test(L):
        r = report(LossPanel.from_losses(L))
        assert isinstance(r, str)

    a_new_contributors_test()


def test_internals_module_gives_stable_unshadowed_access():
    """FIX for the shadowing footgun documented in test_mcs_submodule_shadowed_by_reexported_
    function_name above -- rather than break v1.0's public API (`from selection_fragility import
    mcs` must keep returning the function for existing users), selection_fragility._internals
    gives contributors one documented, non-shadowed path to every real submodule, without relying
    on a raw sys.modules lookup or the easy-to-get-wrong `from X.mcs import _thing` per-symbol
    form. Locks in both halves: the public API is untouched, and _internals reaches the real
    modules (fragility/mcs/compare/report -- the four that actually collide with a re-exported
    name of the same string -- plus the five that don't, for one consistent import path)."""
    import types
    import selection_fragility as sf
    from selection_fragility import _internals

    # public API unchanged: the top-level names are still the re-exported functions/classes
    assert callable(sf.mcs) and not isinstance(sf.mcs, types.ModuleType)
    assert callable(sf.fragility) and not isinstance(sf.fragility, types.ModuleType)
    assert callable(sf.compare) and not isinstance(sf.compare, types.ModuleType)
    assert callable(sf.report) and not isinstance(sf.report, types.ModuleType)

    # _internals reaches the real modules for all nine submodules, shadowed or not
    for name in ("fragility", "mcs", "compare", "report", "panel", "identify", "resolution",
                 "prop22", "pivot"):
        mod = getattr(_internals, name)
        assert isinstance(mod, types.ModuleType), f"_internals.{name} should be the real module"
        assert mod is sys.modules[f"selection_fragility.{name}"]

    assert hasattr(_internals.mcs, "_block_idx")
    assert hasattr(_internals.fragility, "_MAX_ABS_LOSS")
