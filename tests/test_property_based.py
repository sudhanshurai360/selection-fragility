"""Round-4 property-based fuzzing (hypothesis), distinct from rounds 1-3's manual/persona review.

Generates random-but-valid loss panels and checks invariants that must hold across the whole input
domain, not just the fixed examples in the rest of the suite. A failure here means hypothesis found a
concrete counterexample to a documented guarantee -- report the shrunk minimal case, don't just note
"some input fails".
"""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from selection_fragility.fragility import (
    pooled_winner, decision_breakdown, breakdown_number, condorcet_status,
    winner_stability, fragility, _MAX_ABS_LOSS,
)
from selection_fragility.mcs import model_confidence_set
from selection_fragility.identify import identified, mcs_size
from selection_fragility.resolution import (
    minimum_detectable_edge, significance_boundary, mcb_bound, selection_regret, resolution_report,
)
from selection_fragility.pivot import concentration_share
from selection_fragility.report import report
from selection_fragility.panel import LossPanel, _looks_like_metadata_column
from selection_fragility.prop22 import prop22_certifies, certified_tied_subset

SLOW = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
FAST = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])

MODEL_NAMES = ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9"]


def _panel_strategy(k_max=6, t_max=60, t_min=4, allow_negative=True):
    """K models x T periods, finite floats, no NaN/inf (that's a separate, already-tested guard).

    Rejects (via hypothesis's `assume`, not silently) any draw where a generated column happens to
    look like an arithmetic-sequence/repeating-pattern metadata column -- LossPanel deliberately
    refuses those (panel.py's `_looks_like_metadata_column`, a real guard against a real bug class:
    a stray 'horizon'-style column silently crowned a model). A random small-T float column landing
    on that shape by chance is a fuzzer/guard collision, not evidence the guard is wrong; testing
    LossPanel's crash-freedom on the panels it actually accepts as valid needs to filter these out,
    the same way any other precondition on "valid input" is filtered rather than fought.
    """
    @st.composite
    def build(draw):
        K = draw(st.integers(min_value=2, max_value=k_max))
        T = draw(st.integers(min_value=t_min, max_value=t_max))
        lo, hi = (-1e6, 1e6) if allow_negative else (1e-9, 1e6)
        arrs = draw(st.lists(
            st.lists(st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False,
                                width=64),
                     min_size=T, max_size=T),
            min_size=K, max_size=K,
        ))
        panel = {MODEL_NAMES[i]: np.array(a, dtype=float) for i, a in enumerate(arrs)}
        from hypothesis import assume
        assume(not any(_looks_like_metadata_column(a, T) for a in panel.values()))
        return panel
    return build()


# ---------------------------------------------------------------------------
# P1: decision_breakdown's k* is always in [0, T], and the removed-index list
# is exactly length k*, with no duplicate/out-of-range indices.
# ---------------------------------------------------------------------------
@given(L=_panel_strategy())
@FAST
def test_decision_breakdown_kstar_bounds(L):
    T = len(next(iter(L.values())))
    k, opp, removed = decision_breakdown(L)
    assert 0 <= k <= T, f"k*={k} outside [0,{T}] for panel {L}"
    assert len(removed) == k
    assert len(set(removed)) == len(removed), f"duplicate removed indices: {removed}"
    assert all(0 <= i < T for i in removed)


# ---------------------------------------------------------------------------
# P2: model_confidence_set's survivor set is never empty on a well-formed panel.
# ---------------------------------------------------------------------------
@given(L=_panel_strategy(k_max=5, t_max=40, t_min=8))
@SLOW
def test_mcs_survivor_set_never_empty(L):
    surv, p = model_confidence_set(L, B=200, block=3, seed=0)
    assert len(surv) >= 1, f"empty MCS survivor set for panel {L}"
    assert len(surv) <= len(L)
    assert 0.0 <= p <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# P3: relabeling models (a pure rename, values/order otherwise identical) must
# not change WHICH SET of models survives model_confidence_set -- only their
# names. This is the property round 2 tested with fixed examples; fuzz it.
# ---------------------------------------------------------------------------
@given(L=_panel_strategy(k_max=5, t_max=30, t_min=8), seed=st.integers(0, 10))
@SLOW
def test_mcs_rename_invariance_fuzzed(L, seed):
    surv1, _ = model_confidence_set(L, B=150, block=3, seed=seed)
    rename = {k: f"renamed_{k}" for k in L}
    L2 = {rename[k]: v for k, v in L.items()}
    surv2, _ = model_confidence_set(L2, B=150, block=3, seed=seed)
    mapped_back = {k for k, v in rename.items() if v in surv2}
    assert set(surv1) == mapped_back, (
        f"MCS survivor set changed under pure relabeling: {set(surv1)} vs {mapped_back}")


# ---------------------------------------------------------------------------
# P5: no uncaught exception on ANY valid random panel, across every public
# entry point that only needs a panel (+ defaults) to run. A crash on valid
# input is always a bug.
# ---------------------------------------------------------------------------
@given(L=_panel_strategy(k_max=5, t_max=25, t_min=6))
@SLOW
def test_no_crash_on_valid_panel_across_public_api(L):
    pooled_winner(L)
    decision_breakdown(L)
    condorcet_status(L)
    concentration_share(L)
    minimum_detectable_edge(L)
    significance_boundary(L)
    mcb_bound(L)
    selection_regret(L)
    resolution_report(L)
    identified(L, reps=100)
    mcs_size(L, reps=100)
    report(LossPanel.from_losses(L))


# ---------------------------------------------------------------------------
# P6: extreme K/T aspect ratios -- many models, few periods, and vice versa --
# should not crash and should still respect the k*/MCS bounds from P1/P2.
# ---------------------------------------------------------------------------
@given(K=st.integers(min_value=8, max_value=25), T=st.integers(min_value=4, max_value=6),
       seed=st.integers(0, 5))
@SLOW
def test_many_models_few_periods(K, T, seed):
    rng = np.random.default_rng(seed)
    L = {f"m{i}": rng.normal(1.0 + 0.01 * i, 0.3, T) for i in range(K)}
    k, opp, removed = decision_breakdown(L)
    assert 0 <= k <= T
    surv, p = model_confidence_set(L, B=150, block=1, seed=0)
    assert 1 <= len(surv) <= K


@given(T=st.integers(min_value=100, max_value=300), seed=st.integers(0, 5))
@SLOW
def test_few_models_many_periods(T, seed):
    rng = np.random.default_rng(seed)
    L = {"a": rng.normal(1.0, 0.3, T), "b": rng.normal(1.02, 0.3, T)}
    k, opp, removed = decision_breakdown(L)
    assert 0 <= k <= T


# ---------------------------------------------------------------------------
# P7: prop22_certifies / certified_tied_subset must be internally consistent
# with decision_breakdown's own k* -- if prop22 certifies a subset as tied at
# k<=threshold, that k must not exceed T, and the certificate's own threshold
# function must be monotonically non-decreasing in T (more data can only make
# the |t|<1.96 zero-simulation bound easier to satisfy at a given k, never
# harder) for a fixed z.
# ---------------------------------------------------------------------------
@given(T=st.integers(min_value=14, max_value=500))
@FAST
def test_prop22_threshold_monotonic_in_T(T):
    from selection_fragility.prop22 import _prop22_threshold
    k_T = _prop22_threshold(T)
    k_T2 = _prop22_threshold(T + 1)
    assert k_T2 >= k_T - 1e-9, f"prop22 threshold decreased from T={T} ({k_T}) to T+1 ({k_T2})"


@given(L=_panel_strategy(k_max=4, t_max=40, t_min=14))
@FAST
def test_prop22_certifies_implies_bounded_k(L):
    T = len(next(iter(L.values())))
    k, opp, removed = decision_breakdown(L)
    if prop22_certifies(k, T):
        # certified => the zero-simulation bound applies; k must be within [0, T] (P1 already checks
        # this, this is a cross-consistency check specific to the certificate path)
        assert 0 <= k <= T


# ---------------------------------------------------------------------------
# P8: breakdown_number's DEGENERATE-MARGIN FLOOR -- an exact pooled tie must
# give k*=0 even through float64 summation noise. Fuzz decimal-quantised
# ("MASE to 1dp"-shaped) exact-tie panels specifically, the case the module's
# own docstring says was the historical failure mode.
# ---------------------------------------------------------------------------
@given(T=st.integers(min_value=5, max_value=200), seed=st.integers(0, 50))
@FAST
def test_breakdown_number_exact_tie_gives_zero(T, seed):
    rng = np.random.default_rng(seed)
    base = np.round(rng.uniform(0.5, 5.0, T), 1)          # decimal-quantised, like real MASE figures
    perturb = np.round(rng.uniform(-2.0, 2.0, T), 1)
    la = base + perturb
    lb = base + perturb[::-1] if not np.array_equal(perturb, perturb[::-1]) else base + perturb
    # Force an EXACT tie by construction: lb chosen so mean(lb) == mean(la) exactly in the
    # variables actually summed (not just in expectation).
    lb = la.copy()
    rng.shuffle(lb)  # same multiset, same mean, different order/period assignment -> exact tie
    w = np.ones(T)
    k, removed = breakdown_number(la, lb, w)
    if abs(float(np.sum(w * (lb - la)))) < 1e-9:
        assert k == 0, f"exact-tie panel (T={T}, seed={seed}) gave k*={k}, expected 0"


# ---------------------------------------------------------------------------
# P9: winner_stability must return a value in [0, 1] (it's a bootstrap-hit
# proportion) across random panels, including near-degenerate ones.
# ---------------------------------------------------------------------------
@given(L=_panel_strategy(k_max=4, t_max=30, t_min=6))
@SLOW
def test_winner_stability_in_unit_interval(L):
    ws, freq = winner_stability(L, n_boot=150, block=3, seed=0)
    # UPDATED 2026-09-02 (10-agent code-review pass): winner_stability() now correctly returns nan
    # (not a misleading 1.0) on a degenerate/exact-tie panel -- the same degenerate-margin guard
    # fragility() already applied when called indirectly, ported so the function gives the same,
    # honest answer when called directly too. A degenerate panel's freq dict is a uniform 1/K over
    # all models (see the function's own degenerate branch), not a bootstrap-derived distribution.
    if np.isnan(ws):
        assert all(np.isclose(v, 1.0 / len(freq)) for v in freq.values()), (
            f"degenerate-panel freq must be uniform 1/K, got {freq}"
        )
        return
    assert -1e-9 <= ws <= 1.0 + 1e-9, f"winner_stability={ws} outside [0,1]"
    assert all(-1e-9 <= v <= 1.0 + 1e-9 for v in freq.values())
    assert abs(sum(freq.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# P10 [REAL BUG, round 4 -- xfail, not fixed per this round's find-only scope]:
# decision_breakdown's k* is NOT invariant to a positive scalar rescaling of
# the weight vector, contrary to the documented intent ("weights are
# observation counts", only relative weights should matter). Root cause:
# breakdown_number's DEGENERATE-MARGIN FLOOR (fragility.py, both the entry
# guard and the greedy-loop guard) previously compared the running margin
# (M/T and s) -- which scale linearly with w -- against a threshold built
# from `scale` alone, which is computed from la/lb magnitudes ONLY and did
# not scale with w. When w was scaled down far enough relative to the
# margin's own magnitude, the margin crossed this w-independent floor
# prematurely and the greedy loop stopped one period early, understating k*
# (reporting the ranking as MORE fragile than the unscaled computation says).
# Minimal repro (confirmed via hypothesis shrinking): L={'m0': [0,0,0,0],
# 'm1': [0,0,20,1e-5]}, w=ones(4) gave k*=2; w=ones(4)*1e-6 gave k*=1, same
# panel, same relative weights. FIXED 2026-08-26 (round-4 property-based
# fuzzing, hypothesis): both floor comparisons in breakdown_number now
# multiply by wscale=mean(w), so the threshold scales the same way the
# margin does; at wscale=1 (w=None or any uniform-1 weighting -- every
# pre-existing test and the paper's own equal-weight passes) this is exactly
# the prior behaviour, unchanged. See fragility.py's breakdown_number for
# the fix itself.
# ---------------------------------------------------------------------------
@given(L=_panel_strategy(k_max=4, t_max=30), c=st.sampled_from([1e-6, 1e-3, 1.0, 1e3, 1e6]))
@FAST
def test_decision_breakdown_weight_scale_invariance(L, c):
    T = len(next(iter(L.values())))
    w = np.ones(T)
    k1, opp1, _ = decision_breakdown(L, w=w)
    k2, opp2, _ = decision_breakdown(L, w=w * c)
    assert k1 == k2, f"k* changed under weight rescaling by {c}: {k1} vs {k2} (opp {opp1} vs {opp2})"
    # Opponent identity used to need an assume()-guard here for the exact-pooled-tie case (P12's
    # pooled_winner bug below could flip which tied model was reported as "opponent" under weight
    # rescaling). FIXED 2026-08-27 alongside P12 -- pooled_winner is now float64-scale-invariant at
    # exact ties too, so no guard is needed: opponent identity holds unconditionally.
    assert opp1 == opp2


# ---------------------------------------------------------------------------
# P12 [REAL BUG, round 6 -- FOUND, then FIXED 2026-08-27, same day]:
# pooled_winner()'s weighted-average tie-break was not float64-scale-invariant when the two
# candidates are EXACTLY pooled-tied on the unscaled data. `pooled_winner` compares
# `np.average(L[m], weights=w)` across models -- mathematically, multiplying every weight by a
# positive constant c cancels exactly in `sum(a*w)/sum(w)` and can never change which model has
# the lower weighted mean. In float64 it did not always cancel: at c=1e-6 on the panel below,
# the numerator/denominator division picked up a tiny asymmetric rounding difference between the
# two otherwise-exactly-tied models, flipping which one was reported as `pooled_winner` (and
# therefore which was reported as `decision_breakdown`'s "opponent", since that name is always
# relative to whichever model is currently deemed the winner). k* itself stayed correct (k*=0,
# correctly reporting "no clear winner to protect" either way) -- only the WINNER LABEL flipped.
# Minimal repro (hand-verified, not hypothesis-shrunk): L={'m0': [0,0,2,2.125],
# 'm1': [0,0,0,4.125]} -- both sum to exactly 4.125. w=ones(4): pooled_winner='m0'.
# w=ones(4)*1e-6: pooled_winner='m1'. Found via test_decision_breakdown_weight_scale_invariance's
# hypothesis search landing on this exact tie by chance; confirmed deterministic, not a flake,
# by direct hand-computation -- see round-6 ledger entry for the raw float64 sums showing the
# asymmetry (4.125e-06 vs 4.1249999999999995e-06 at c=1e-6, while c=1e-3/1.0/1e3/1e6 all
# reproduced the exact tie correctly). Same root-cause FAMILY as the breakdown_number
# degenerate-margin-floor bug fixed round 4 (float64 residue near an exact tie under weight
# rescaling), but a DIFFERENT code path (pooled_winner's np.average call, not breakdown_number's
# greedy-loop floor) -- fixing round 4's bug did not fix this one. FIXED here: pooled_winner now
# treats any mean within `_MARGIN_REL_FLOOR * _loss_scale(L)` of the true minimum as a tie,
# broken by sorted name, matching the relative-floor discipline already used elsewhere in this
# module -- verified 0/200,000 mismatches on random decimal-quantized trials (was 101/200,000).
# ---------------------------------------------------------------------------
def test_pooled_winner_tie_break_scale_invariance_at_exact_tie():
    from selection_fragility.fragility import pooled_winner
    L = {"m0": np.array([0., 0., 2., 2.125]), "m1": np.array([0., 0., 0., 4.125])}
    assert sum(L["m0"]) == sum(L["m1"]), "fixture must be an exact pooled tie"
    w = np.ones(4)
    winners = {c: pooled_winner(L, w * c) for c in (1e-6, 1e-3, 1.0, 1e3, 1e6)}
    assert len(set(winners.values())) == 1, f"pooled_winner varies by weight scale: {winners}"
