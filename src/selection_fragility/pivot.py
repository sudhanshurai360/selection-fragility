"""
selection_fragility.pivot — concentration_share and pivot_agreement, the two statistics that answer
"is the named pivotal period real?"

Both replace statistics that were measured and REJECTED tonight: the old `concentration` field
(c1/net-margin) had power at or below nominal size everywhere and its denominator explodes exactly
when fragility matters; `pivot_sharpness` (c1/c2, the two largest contributions) collapses BELOW
nominal size at near-ties, a structural degeneracy in any ratio of the two largest order statistics.
Both replacements were kept only after matched-false-positive-rate testing, not an AUC comparison --
AUC misled this project twice already on exactly this kind of candidate-statistic screening.
"""
import math

import numpy as np

from .fragility import _as_loss_dict, _validate_losses, pooled_winner, decision_breakdown, _unwrap_panel

_DEGENERATE_FLOOR = 1e-12


def _champ_opp_contributions(L, w):
    """The champion, its binding opponent, and the signed per-period margin contributions
    c_t = w_t*(loss_opp_t - loss_champ_t). Returns (None, None, None) when there is no strict winner
    (k*=0) -- nothing is pivotal to name.

    OPPONENT TIE-BREAK. k* is a small integer, so several rivals easily tie on its exact COUNT by
    chance (a discrete collision, not a measure-zero floating-point tie) -- common enough at small T
    that using decision_breakdown's own sorted-NAME tie-break (the right choice for THAT function's
    own contract: a stable, citable point estimate for k* itself) made an earlier version of this
    module's period-level statistics measurably rename-dependent, because a renaming flips which tied
    opponent sorts first and different tied opponents can have very different pivotal periods. Here
    the tie is broken instead by the LARGEST total margin M among the tied opponents -- a property of
    the DATA, not of what the models are called, so it is invariant to renaming by construction, and
    ties in a continuous quantity like M are effectively impossible (unlike ties in the integer k*,
    which are common). See tests/test_stage4_pivot.py::TestPivotRenameInvariance for the exact
    before/after magnitude this fix was verified against -- not restated here, since it belongs to a
    specific fixture and would read as a general property of this function if quoted out of context."""
    champ = pooled_winner(L, w)
    _k, _opp, _removed, ties = decision_breakdown(L, w, a=champ, return_ties=True)
    # CONFIRMED REGRESSION (independent 'wild' review, 2026-08-17, cross-fix interaction lens):
    # `if not ties:` was dead code -- decision_breakdown's `ties` list collects every opponent tied
    # at the MINIMUM k across all opponents, and that list is non-empty as long as there are >=2
    # models, REGARDLESS of whether the minimum k is 0. The intended "k*=0, no strict winner over
    # ANY opponent, nothing pivotal to name" case was never actually caught here. Exactly the same
    # None-vs-falsy defect class as `prop22.py::certified_tied_subset`'s k==0 fix earlier tonight
    # (fix #16) -- applied there, never carried over to this sibling function. Verified directly: a
    # 3-model panel with champ/rival tied to 1e-16 relative (k*=0) but real, non-cancelling per-
    # period differentials still returned concentration_share=0.116 and pivot_agreement=0.465
    # instead of None, violating both functions' own documented "returns None when k*=0" contract.
    if _k == 0:
        return None, None, None
    la = np.asarray(L[champ], float)
    # FIXED 2026-09-10 (dedicated structural audit, prompted by an independent second opinion noting
    # that the same champion-floor bug had already been found 4 times separately and recommending a
    # systematic sweep rather than trusting a 5th instance to surface by luck). This IS that 5th
    # instance, found by the audit it recommended.
    # `M > best_M` (a raw float64 comparison, no relative floor) is exactly the pattern already fixed
    # four times elsewhere in this package -- but the docstring above explicitly rejected adding a
    # sorted-NAME floor/fallback here, because that would reintroduce the exact rename-dependence bug
    # this function was built to avoid (TestPivotRenameInvariance). A floor+name-fallback was NOT the
    # right fix for that reason. Reproduced the actual defect: two opponents whose per-period
    # contributions are PERMUTATIONS of each other (mathematically identical total M, by construction
    # of the tie) gave DIFFERENT float64 sums under `c.sum()` (numpy's pairwise summation is order-
    # dependent) at weight scales 1e-6 and 1e-3 -- a genuine data-driven tie was reported as a real
    # difference purely from summation order, flipping which opponent (and therefore which pivotal
    # period) got named. `math.fsum` (Shewchuk's algorithm) computes a correctly-rounded sum that is
    # PROVABLY invariant to input order -- not a floor/threshold, an exact fix: permutation-tied
    # inputs now sum to the bit-identical float regardless of scale or order, verified directly across
    # scale in {1e-12...1e9}. This fixes the root cause (order-dependent summation) rather than
    # working around it, so it does NOT reopen the rename-dependence issue -- ties are still broken by
    # a genuine data property (M), just one computed correctly now.
    best_opp, best_M, best_c = None, -np.inf, None
    for t in ties:
        c = w * (np.asarray(L[t], float) - la)
        M = math.fsum(c)
        if M > best_M:
            best_M, best_opp, best_c = M, t, c
    return champ, best_opp, best_c


def concentration_share(L, w=None) -> float | None:
    """c1 / sum(|c|) -- the largest single-period margin contribution as a share of the TOTAL absolute
    contribution across all periods (signed: c1 can be negative in principle, though for the binding
    opponent it is the largest POSITIVE contribution by construction of decision_breakdown's greedy
    order). Replaces the old c1/net-margin `concentration`, whose denominator (the net margin) shrinks
    toward zero exactly when a decision is fragile -- the new denominator (total absolute contribution)
    does not collapse the same way. Matched-size power across the 7 shock configurations this statistic
    is calibrated against is gated, not stated here -- see
    tests/test_stage4_pivot.py::TestConcentrationShare for the current verified figures, so a stale
    number in this docstring can never silently outlive a change to the formula or the fixture.
    Returns None when there is no strict winner (k*=0) or the margin is degenerate
    (near-zero total contribution) -- a share computed against ~0 is fake precision, not signal."""
    L, w = _unwrap_panel(L, w)
    L = _as_loss_dict(L)
    T = _validate_losses(L, w)
    w = np.ones(T) if w is None else np.asarray(w, float)
    champ, opp, c = _champ_opp_contributions(L, w)
    if opp is None:
        return None
    denom = float(np.sum(np.abs(c)))
    scale = float(np.mean(np.abs(np.asarray(L[champ], float)) + np.abs(np.asarray(L[opp], float)))) / 2.0
    if denom <= _DEGENERATE_FLOOR * scale * T:
        return None
    return float(np.max(c) / denom)


def pivot_agreement(L, w=None, seed=0, n_boot=200, frac=0.8) -> float | None:
    """Fraction of `frac`*T-sized subsamples of periods (without replacement) that name the SAME
    pivotal period as the full sample -- i.e. for how much of the data could you throw away 20% of the
    periods and still point at the same period as "the one that decided this".

    The champion/binding-opponent decision is RECOMPUTED FRESH on each subsample (mirroring
    `winner_stability`'s own methodology in this codebase, the structural analogue for models rather
    than periods) rather than held fixed from the full sample -- an earlier version fixed the
    opponent from the full sample and came out far too stable against the recentred-null benchmark
    below (see tests/test_stage4_pivot.py for the exact figures that showed this); a subsample can
    legitimately point at a different champion/opponent pair entirely, and that possibility is part
    of what "how stable is this pivot" should measure. A subsample with no strict winner (k*=0) does
    not count toward the denominator -- it is neither agreement nor disagreement, it is undetermined
    for that draw.

    READ AGAINST THE RECENTRED-PANEL BENCHMARK, NEVER against 1/T -- even an arbitrary pivot is
    stable well above 1/T on a real panel with no true winner, because recentring a real panel only
    removes each model's mean LEVEL, not the shared shock-year correlation structure across models: a
    real crisis year that hit every model's forecast stays the largest contributor for almost any
    model pairing even after recentring. Reading a real value against 1/T repeats the exact mistake
    `exchangeable_benchmark` was built to fix for `winner_stability`. THE ACTUAL BENCHMARK VALUE IS
    NOT STATED HERE ON PURPOSE -- this exact number drifted stale in this docstring once already
    (an earlier version said ~0.42 for weeks after the true figure had already been corrected to
    ~0.67 elsewhere in the codebase, caught only by an independent review). See
    tests/test_stage4_pivot.py::test_recentred_null_benchmark_near_067 for the current, gated value,
    and TestFlaggedCellsBenchmark for how many of the real cells fall below it.

    Returns None when there is no strict winner in the full sample (k*=0, nothing is pivotal to name),
    T is too small for a meaningful subsample (fewer than 3 periods in the subsample), or every
    subsample drawn happened to have no strict winner."""
    # PARAMETER VALIDATION. FIXED 2026-09-02 (round-2 10-agent review, CONFIRMED). `n_boot <= 0` and
    # an out-of-range `frac` both silently returned None -- indistinguishable from this function's TWO
    # documented None meanings (no strict winner; subsample too small), so a caller reading None had
    # no way to tell "your parameter is nonsense" from "the data cannot answer this". The identical
    # defect class was hardened in `winner_stability` (n_boot) and `_churn_base_rate` (n_perm) on
    # 2026-09-02; this sibling was missed in that sweep.
    if int(n_boot) <= 0:
        raise ValueError(f"pivot_agreement(): n_boot must be a positive integer; got {n_boot}.")
    if not (0.0 < float(frac) < 1.0):
        raise ValueError(f"pivot_agreement(): frac must be strictly between 0 and 1 (it is the "
                          f"fraction of periods kept in each subsample); got {frac}.")
    L, w = _unwrap_panel(L, w)
    L = _as_loss_dict(L)
    T = _validate_losses(L, w)
    w = np.ones(T) if w is None else np.asarray(w, float)
    _champ, opp, c = _champ_opp_contributions(L, w)
    if opp is None:
        return None
    full_pivot = int(np.argmax(c))
    n_sub = int(round(frac * T))
    if n_sub < 3 or n_sub >= T:
        return None
    rng = np.random.default_rng(seed)
    agree = 0
    valid = 0
    for _ in range(n_boot):
        idx = np.sort(rng.choice(T, size=n_sub, replace=False))
        Lsub = {m: np.asarray(v, float)[idx] for m, v in L.items()}
        wsub = w[idx]
        _champ_s, opp_s, c_s = _champ_opp_contributions(Lsub, wsub)
        if opp_s is None:
            continue
        valid += 1
        sub_pivot = idx[int(np.argmax(c_s))]
        if sub_pivot == full_pivot:
            agree += 1
    if valid == 0:
        return None
    return agree / valid
