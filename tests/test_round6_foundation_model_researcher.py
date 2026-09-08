"""Round 6 - foundation-model/zero-shot forecasting researcher persona.

Distinct from rounds 1-5: tests whether the tool behaves sensibly when candidate "models" are
really the same pretrained foundation model at different context lengths/prompts/fine-tunes, i.e.
far more correlated than the "genuinely different architectures" framing the rest of the suite
assumes. Find/test-only per round-6 instruction -- the B-mismatch finding below is deliberately
NOT fixed here.
"""
import numpy as np
import pytest

from selection_fragility.mcs import model_confidence_set
from selection_fragility.identify import _run_mcs, mcs_size, identified
from selection_fragility.fragility import decision_breakdown, pooled_winner
from selection_fragility.panel import LossPanel
from selection_fragility.report import report


def _foundation_model_variants(T=200, seed=0):
    """5 'models' = the same pretrained model at 5 context lengths -- shared per-period difficulty
    plus small variant-specific noise, so pairwise correlation is ~0.998-0.999 (far tighter than any
    fixture elsewhere in this suite, which assumes genuinely different architectures)."""
    rng = np.random.default_rng(seed)
    base_difficulty = rng.gamma(2.0, 1.0, T)
    offsets = {"ctx256": 0.30, "ctx512": 0.10, "ctx1024": 0.02, "ctx2048": 0.00, "ctx4096": -0.01}
    return {
        name: np.clip(base_difficulty + off + rng.normal(0, 0.05, T), 0.01, None)
        for name, off in offsets.items()
    }


def test_no_crash_on_highly_correlated_foundation_model_variants():
    """The whole public surface should tolerate near-collinear candidates without raising."""
    L = _foundation_model_variants()
    panel = LossPanel.from_losses(L)
    surv, p = model_confidence_set(L, alpha=0.10, seed=0)
    assert len(surv) >= 1
    k, opp, removed = decision_breakdown(L, w=None)
    assert 0 <= k <= len(next(iter(L.values())))
    report(panel)  # must not raise


def test_report_and_direct_mcs_call_can_disagree_on_default_B_alone():
    """FINDING (round 6, foundation-model persona): report()'s internal MCS path (identify.py's
    _run_mcs, arch-based, default reps=500) and calling model_confidence_set() directly (mcs.py's
    own implementation, default B=2000) can report DIFFERENT identification verdicts for the exact
    same panel/alpha, purely because their default bootstrap resample counts differ -- not because
    the two engines disagree algorithmically. Confirmed by matching B: at B=500 both paths agree
    exactly (both give {'ctx4096'}); at mcs.py's own default B=2000, the direct call additionally
    admits 'ctx2048'. This is a real, reproducible discrepancy a user would hit by computing MCS
    both ways on the same highly-correlated panel (exactly what comparing context-length variants
    of one foundation model produces) and getting contradicting "identified" statuses. Locked in as
    a documented finding, not a bug in either individual algorithm -- NOT fixed this round (whether
    to unify the defaults, or have report() warn when its B differs from a value the caller may
    have used elsewhere, is a design decision for a future fix pass)."""
    L = _foundation_model_variants()

    surv_b500, _ = model_confidence_set(L, alpha=0.10, B=500, block=3, seed=0)
    arch_run = _run_mcs(L, alpha=0.10, reps=500, block_size=3, seed=0)
    assert sorted(surv_b500) == sorted(arch_run.included), (
        "with B/reps matched, the two MCS engines should agree on this panel -- if this starts "
        "failing, the finding has changed from 'default mismatch' to a real algorithmic divergence "
        "and needs re-diagnosing, not just re-asserting"
    )

    surv_b2000, _ = model_confidence_set(L, alpha=0.10, B=2000, block=3, seed=0)
    # This asserts the CURRENT (undesirable but real) behavior so a silent regression -- or a
    # silent fix that changes the story without anyone noticing -- gets caught either way.
    assert len(surv_b2000) != len(arch_run.included), (
        "expected the documented default-B discrepancy to still reproduce on this fixture; if it "
        "no longer does, either the panel/seed needs updating or the discrepancy was fixed -- in "
        "the latter case, update this test's docstring and assertion rather than deleting it"
    )


def test_foundation_model_scale_T_thousands_no_crash_reasonable_time():
    """Foundation models are routinely zero-shot evaluated over thousands of windows, far beyond
    the T~30-40 scale the rest of this suite's fixtures assume. K stays small (comparing a handful
    of real candidates), matching a realistic foundation-model-vs-baselines comparison."""
    rng = np.random.default_rng(1)
    T, K = 5000, 3
    L = {f"m{i}": np.clip(rng.gamma(2.0, 1.0, T) + rng.normal(0, 0.02, T) + 0.01 * i, 0.001, None)
         for i in range(K)}
    surv, p = model_confidence_set(L, alpha=0.10, B=2000, block=3, seed=0)
    assert len(surv) >= 1
    k, opp, removed = decision_breakdown(L, w=None)
    assert 0 <= k <= T


def test_pooled_winner_exact_tie_identity_is_weight_scale_invariant():
    """FIXED (same round-6 pass, by a sibling fork owning fragility.py's weighted-average tie-
    break): originally found here as a NEW bug -- `tests/test_property_based.py::
    test_decision_breakdown_weight_scale_invariance` failed on this minimized panel:
    L={'m0': [0,0,2,2.125], 'm1': [0,0,0,4.125]} -- both models have an EXACTLY tied raw mean
    (1.03125), and `np.average(m0's values, weights=np.ones(4)*1e-6)` returned
    1.0312500000000002, not exactly 1.03125 -- float64 residue from the weighted-average
    computation at extreme small weight scale, flipping `pooled_winner`'s exact-tie result from
    'm0' to 'm1' at w-scale 1e-6 only (k*=0 stayed correct throughout -- the COUNT was never
    wrong, per round 4's earlier fix; only the WINNER/OPPONENT identity flipped). Same general
    defect class (float64 precision near an exact tie) as several other fixes this session, in a
    previously-uncovered location: `pooled_winner`'s weighted-average tie-break, not
    `breakdown_number`'s degenerate-margin floor. Now fixed at the source; this test verifies the
    fix rather than documenting the gap, per its own original instruction to invert rather than
    delete when that happens."""
    L = {"m0": np.array([0.0, 0.0, 2.0, 2.125]), "m1": np.array([0.0, 0.0, 0.0, 4.125])}
    assert L["m0"].mean() == L["m1"].mean(), "fixture must be an EXACT tie for this to be the right repro"
    winners = {c: pooled_winner(L, np.ones(4) * c) for c in (1.0, 1e-3, 1.0, 1e3, 1e6, 1e-6)}
    assert len(set(winners.values())) == 1, (
        f"pooled_winner should be invariant to positive weight rescaling on an exact tie; got "
        f"{winners}"
    )


def test_identified_wrapper_is_internally_consistent_with_its_own_mcs_size():
    """Sanity companion to the B-mismatch finding: identify.py's OWN two functions (identified,
    mcs_size), which share the same defaults, must never disagree with each other on the same
    panel -- this isolates "the two entry points disagree" (real, documented above) from "a single
    entry point is internally inconsistent" (would be a much more serious bug; confirmed NOT the
    case here)."""
    L = _foundation_model_variants()
    assert identified(L, alpha=0.10, seed=0) == (mcs_size(L, alpha=0.10, seed=0) == 1)
