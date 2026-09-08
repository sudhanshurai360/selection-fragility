"""Round-8 fresh-eyes line-by-line code review finding.

Six public, prominently `__all__`-exported functions never received the `LossPanel`-acceptance fix
their sibling functions got in rounds 3 and 6: `minimum_detectable_edge`, `significance_boundary`,
`mcb_bound`, `selection_regret` (resolution.py) and `concentration_share`, `pivot_agreement`
(pivot.py). All six raised a confusing internal `TypeError: object of type 'LossPanel' has no
len()` when passed a `LossPanel` directly instead of a raw dict -- the exact error-quality class
already fixed for `resolution_report()` (round 3) and the raw-array tier (round 6). `prop22.py`'s
`certified_tied_subset` had the identical code shape and the same bug, confirmed by direct
execution, not just code-shape inspection.

Fixed via the existing `_unwrap_panel(L, w)` helper (round 6, `fragility.py`), reused here rather
than reinventing a parallel mechanism -- and `resolution_report()`'s own local duck-type-check
duplicate of the same logic was replaced with a call to the shared helper for consistency.
"""
import numpy as np
import pytest

from selection_fragility.panel import LossPanel
from selection_fragility.resolution import (
    minimum_detectable_edge, significance_boundary, mcb_bound, selection_regret, resolution_report,
)
from selection_fragility.pivot import concentration_share, pivot_agreement
from selection_fragility.prop22 import certified_tied_subset


def _panel_and_dict(seed=0, T=20):
    rng = np.random.default_rng(seed)
    L = {"a": rng.normal(1.0, 0.3, T), "b": rng.normal(1.1, 0.3, T)}
    return LossPanel.from_losses(L), L


@pytest.mark.parametrize("fn", [
    minimum_detectable_edge, significance_boundary, mcb_bound, selection_regret,
    concentration_share, pivot_agreement, certified_tied_subset,
])
def test_previously_broken_functions_now_accept_losspanel_directly(fn):
    panel, L = _panel_and_dict()
    from_panel = fn(panel)
    from_dict = fn(L)
    assert from_panel == from_dict, (
        f"{fn.__name__}(panel) != {fn.__name__}(dict): {from_panel!r} vs {from_dict!r}"
    )


def test_selection_regret_losspanel_weights_respected():
    """selection_regret's parameter is named `weights`, not `w` -- verify the LossPanel's own
    weights are actually threaded through under that name, not silently dropped."""
    rng = np.random.default_rng(1)
    T = 10
    L = {"a": rng.normal(1.0, 0.2, T), "b": rng.normal(1.05, 0.2, T)}
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    panel = LossPanel.from_losses(L, weights=w)
    assert selection_regret(panel) == pytest.approx(selection_regret(L, weights=w))
    # sanity: a genuinely different weighting must not happen to give the same answer by luck
    assert selection_regret(panel) != pytest.approx(selection_regret(L))


def test_resolution_report_still_works_after_unwrap_panel_refactor():
    """resolution_report()'s own local duck-type check was replaced with a call to the shared
    _unwrap_panel() helper as part of this fix -- lock in it still behaves identically."""
    rng = np.random.default_rng(2)
    T = 15
    L = {"a": rng.normal(1.0, 0.3, T), "b": rng.normal(1.1, 0.3, T)}
    panel = LossPanel.from_losses(L)
    r_panel = resolution_report(panel)
    r_dict = resolution_report(L)
    assert r_panel == r_dict
