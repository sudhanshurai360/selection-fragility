"""
selection_fragility.fragility — the decision breakdown point (k*) and ranking-fragility diagnostics.

The question it answers: on a pooled accuracy metric, how FEW evaluation periods must you delete before the
"best model" changes? A low number means the pooled verdict rests on a handful of periods (e.g. one shock) and is
fragile — don't trust it. A high number means the ranking is robust.

Lineage: this repurposes the breakdown-point idea (Hampel; Donoho & Huber) and the clinical-trials Fragility Index
(Walsh et al. 2014, J. Clin. Epidemiol. — the minimum number of outcomes to flip statistical significance) to a
model-selection decision on a WEIGHTED PANEL of per-period losses. It complements the Model Confidence Set (which
models are statistically tied) by quantifying how trustworthy the ranking itself is to a few influential periods.

Design: k* is defined on the FIXED-WEIGHT ADDITIVE loss-differential margin, so deleting periods is exactly additive
and the greedy algorithm is provably optimal (and O(T log T)). Inputs are the per-period, cross-sectionally-aggregated
scaled losses (e.g. per-year MASE) plus per-period weights (e.g. observation counts).


NOTE ON WEIGHTS. `w` supports non-uniform per-period weighting (e.g. observation counts), and the
greedy is exact for ANY fixed weights. Both paths are exercised by the published results.

[CORRECTED 2026-08-08. This paragraph previously read "every analysis in the accompanying paper passes
EQUAL weights ... Do not read the paper's numbers as depending on a weighting choice: they do not."
That was false, and it denied the existence of a robustness exercise the paper reports. social_choice.py
calls fragility() TWICE per cell -- once with equal year weights and once with each year's month count --
precisely to measure that dependence, and six canonical values record the comparison
(election.weight_winner_changes = the pooled winner moves in 2 of 26 cells; election.weight_kstar_changes
= k* moves in 5; plus the paired *_equal / *_monthwtd concentration figures). run_multiseries.py and
score_multiseries.py also pass per-year observation counts. Equal weights are used by
social_choice.py's headline pass, flagship_fragility.py and combination_benchmark.py -- not by
everything. A reader auditing weight sensitivity was told by this docstring that there was none.]
"""
import warnings

import numpy as np


def _unwrap_panel(L, w):
    """Accept a `LossPanel` directly on the raw-array tier (`pooled_winner`, `decision_breakdown`,
    `winner_stability`, `fragility()`), matching `report()`/`compare()`/`resolution_report()`.

    FIXED 2026-08-27 (round-6 probabilistic-forecasting-researcher review): these four were the
    only remaining public entry points that didn't accept a `LossPanel` -- a user who just built one
    for `report()` naturally tries it here too and got a confusing internal `AttributeError` instead
    of either working or a clear top-level message. Mirrors `resolution_report()`'s 2026-08-26 fix
    (same duck-type check, same w-from-panel-if-not-given rule) rather than inventing new behavior.
    Call this FIRST, before any other operation on `L` -- `decision_breakdown`'s magnitude guard
    calls `L.items()` before its own `_as_loss_dict(L)`, which would break on a raw `LossPanel`.
    """
    if hasattr(L, "losses") and hasattr(L, "labels"):     # duck-typed LossPanel
        if w is None:
            w = L.weights
        L = L.losses
    return L, w


def _as_loss_dict(L):
    """Accept a pandas DataFrame (columns = models, rows = periods) as well as {model: array}.

    Users hold per-period losses in a DataFrame far more often than in a dict, and the previous code half-accepted one
    by accident: `L.items()` on a DataFrame does yield (column, Series) pairs, so most of the maths happened to
    work -- but `len(L) < 2`, the "need at least two models" guard, counts ROWS on a DataFrame. A single-model
    frame with 30 rows therefore sailed past the guard it was supposed to trip. Coercing explicitly here makes the
    supported input types deliberate rather than incidental, and preserves column order so results are stable.
    """
    if hasattr(L, "columns") and hasattr(L, "to_dict"):          # duck-typed DataFrame; no hard pandas dependency
        return {str(c): np.asarray(L[c], float) for c in L.columns}
    return L


def _validate_losses(L, w=None):
    """Guard against the common footguns -- fewer than 2 models, ragged/empty arrays, and non-finite (NaN/inf) losses
    -- any of which would otherwise produce a CONFIDENT BUT MEANINGLESS answer (a single NaN silently poisons the
    pooled mean and can crown a spurious 'winner'). Returns the per-period length T."""
    if len(L) < 2:
        raise ValueError(f"need >=2 models to compare a selection decision; got {len(L)}.")
    T = None
    for m, v in L.items():
        a = np.asarray(v, float)
        if a.ndim != 1 or a.size == 0:
            raise ValueError(f"each model's loss must be a non-empty 1-D array; '{m}' has shape {a.shape}.")
        if T is None:
            T = a.size
        elif a.size != T:
            raise ValueError(f"all models must share the same number of periods; '{m}' has {a.size}, expected {T}.")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"model '{m}' has non-finite loss values (NaN/inf). Drop or impute those periods before "
                             f"calling -- a single NaN silently poisons the pooled mean and the selected winner.")
    if w is not None:
        wa = np.asarray(w, float)
        if wa.size != T or not np.all(np.isfinite(wa)):
            raise ValueError(f"weights must be finite and of length {T}; got length {wa.size}.")
        # FIXED 2026-08-26 (independent ML-engineer review, cross-package consistency lens): this
        # was the only weight gate `pooled_winner`, `winner_stability`, `decision_breakdown` and
        # `fragility()` actually pass through (all four call `_validate_losses(L, w)` directly on
        # a raw array, not via LossPanel) -- it checked finiteness only, not sign. `panel.py`'s
        # `_validate_weights` (the gate every LossPanel-built call already goes through) has
        # rejected negative weights since 2026-08-17 with this exact message; the raw-array entry
        # points above silently accepted them instead, e.g. `pooled_winner(L, w=[-1, 1, 1, 1])`
        # returned a confident answer with no error, and `winner_stability` on the same weights
        # either did the same or crashed later inside `np.average` with an opaque "weights sum to
        # zero" message on some resamples -- neither is a validation error at the source. Verified
        # directly on both before this fix.
        if np.any(wa < 0):
            raise ValueError("weights must not contain negative values -- weights are observation "
                              "counts, not signed contributions.")
        # FIXED 2026-08-26 (round-2 strict-code-correctness review): the negative-weight guard
        # above closed one gap in this same fix, but an all-zero weight vector still sailed
        # through it (all-finite, none negative) and crashed later inside `np.average` with a
        # bare, opaque `ZeroDivisionError: Weights sum to zero, can't be normalized` -- not a
        # validation error at the source, and inconsistent with `panel.py`'s `_validate_weights`,
        # which has rejected this exact case with a clear domain message since 2026-08-17.
        if np.all(wa == 0):
            raise ValueError("weights are all zero; at least one period must carry positive weight.")
    return T


def pooled_winner(L: dict, w=None):
    """Model with the lowest weighted-mean loss (the pooled 'best'). L: {model: array(T)}; w: array(T). Lower=better."""
    L, w = _unwrap_panel(L, w)
    L = _as_loss_dict(L)
    T = _validate_losses(L, w)
    w = np.ones(T) if w is None else w
    w = np.asarray(w, float)
    means = {m: float(np.average(np.asarray(L[m], float), weights=w)) for m in L}
    best = min(means.values())
    # Ties resolve by sorted model name, not dict insertion order -- the latter is a property of how the
    # caller built the dict, not of the data.
    #
    # DEGENERATE-TIE FLOOR. FIXED 2026-08-27 (round-6 review, independently found and reproduced by 6+
    # reviewers). np.average(x, weights=w) computes sum(w*x)/sum(w) -- mathematically invariant to any
    # positive rescaling of w, since the scale cancels in the ratio. But it is NOT float64-invariant: the
    # numerator is a literal sum of w-scaled terms, and non-associative floating-point addition can land a
    # few ULPs off depending on w's magnitude even though the true ratio is identical. Verified directly:
    # L={'m0':[0,0,2,2.125],'m1':[0,0,0,4.125]} (exactly tied, mean 1.03125 for both) gives IDENTICAL means
    # at w=1e-3/1/1e3/1e6/1e12 but a 2.22e-16 (one machine-epsilon) residual at w=1e-6 -- enough to flip
    # which name compares smaller under exact float equality, so `pooled_winner` silently reported a
    # DIFFERENT model as "the winner" purely from the caller's choice of weight units (e.g. reporting
    # observation counts as "per day" vs "per week"). Comparing against exact equality is the bug, the same
    # class already fixed twice elsewhere in this module (breakdown_number's degenerate-margin floor,
    # decision_breakdown's weight-scale invariance) but in a third, independent location neither of those
    # fixes touches. Same `_MARGIN_REL_FLOOR` relative-floor discipline applied here: any mean within
    # `_MARGIN_REL_FLOOR * _loss_scale(L)` of the true minimum is a genuine tie (not a real ranking), broken
    # by name like every other tie in this module -- not by weight-scale-dependent rounding noise. No
    # `wscale` term is needed (unlike breakdown_number's raw, un-normalized running margin): the ratio
    # np.average returns is already loss-scale magnitude regardless of w's scale, confirmed above.
    #
    # PER-PAIR SCALE. FIXED 2026-09-02 (round-2 10-agent review, CONFIRMED CRITICAL). The floor above was
    # correct in kind but wrong in scope: `_loss_scale(L)` averages |loss| over EVERY model, so a single
    # large-magnitude column (a mixed-units panel -- e.g. one model scored in raw dollars alongside others
    # scored in MASE) inflates `tol` for every unrelated pair, and a genuinely worse model that happens to
    # sort earlier is returned as the winner. Reproduced: L={'z_best':1.000+n,'a_worse':1.001+n,
    # 'dollars':5e9} with z_best strictly better in ALL 12 periods returns 'a_worse'. Downstream this
    # produced a self-contradictory report() (a model marked both "(champion)" and "excluded", |MCS|=1
    # naming a different model) and a NEGATIVE observed_edge/mcb_bound, violating mcb_bound's own
    # documented invariant. This is the SAME whole-panel-vs-per-pair defect already fixed twice in this
    # package -- fragility()'s concentration/degenerate floor (2026-08-27) and identify._run_mcs's jitter
    # (2026-08-26) -- in a third location neither fix reached, because pooled_winner shipped 2026-08-27
    # with the whole-panel form and its tests only ever used single-magnitude panels.
    #
    # The scale is now built from the two means actually being compared, exactly as breakdown_number's
    # `_margin_below_floor` does, so an unrelated column can no longer contaminate an unrelated comparison.
    best_m = min(sorted(means), key=lambda m: means[m])
    tied = [m for m in L
            if means[m] - best <= _MARGIN_REL_FLOOR * (abs(means[best_m]) + abs(means[m])) / 2.0]
    return min(tied)


def per_period_winner(L: dict, return_ties=False):
    """The robust winner: the model that wins the MOST periods (majority vote over per-period lowest loss).
    In social-choice terms this is the PLURALITY winner over periods-as-voters (the pooled winner is the SCORE/range
    winner). Their disagreement is the classic score-vs-plurality reversal.

    STRICT WINS ONLY. A period counts toward a model's total only if that model's loss is the UNIQUE
    minimum that period; a tied period counts toward nobody. FIXED 2026-08-16 (confirmed regression):
    the previous version used `M.argmin(axis=0)`, which on a tie silently resolves to the LOWEST
    SORTED-NAME INDEX -- so a period where two models tie was counted as a WIN for whichever model
    happened to sort first, fabricating a "plurality winner" (and a spurious pooled-vs-plurality
    reversal) out of pure name order. Confirmed on `dominance_panel` (b weakly dominates a -- never
    worse, strictly better once -- but the old code awarded the 2 tied periods to 'a' and called 'a'
    the plurality winner) and on a byte-identical triple (should report a genuine 3-way tie; the old
    code picked a unique "winner" by sorted name). Both are regression-tested and rename-invariant
    (tests/test_stage3_kstar_prop22.py).

    TIE-BREAK for the reported POINT winner among several models tied for the most strict wins:
    name-sorted, deterministic, independent of dict insertion order -- but this only selects which
    name is reported as `tied[0]`, never fabricates a win out of a tie. Pass return_ties=True to get
    (winner, all_tied) and report the honest range.

    Accepts a `LossPanel` directly, matching `pooled_winner`/`decision_breakdown`/`winner_stability`
    (FIXED 2026-09-07, round-4 8-lens PyPI-preflight audit: this and its two "Legacy API" siblings
    below, condorcet_winner/condorcet_status, raised a raw internal TypeError on a LossPanel despite
    sitting in the same README "Legacy API" sentence as three functions that already accepted one --
    an inconsistency within the documented API surface, not just the Stage-0/Stage-1 bug this audit
    originally targeted in identify.py)."""
    L, _ = _unwrap_panel(L, None)
    L = _as_loss_dict(L)
    _validate_losses(L)
    models = sorted(L)
    M = np.vstack([np.asarray(L[m], float) for m in models])      # models x T, name-sorted
    per_period_min = M.min(axis=0)
    is_unique_min = (M == per_period_min[None, :]) & (np.sum(M == per_period_min[None, :], axis=0) == 1)
    wins = is_unique_min.sum(axis=1)                                # STRICT per-period wins only
    top = int(wins.max())
    tied = [models[i] for i in range(len(models)) if wins[i] == top]
    return (tied[0], tied) if return_ties else tied[0]


def condorcet_winner(L: dict) -> str | None:
    """The CONDORCET winner: the model that beats every other model in pairwise per-period majority (lower loss more
    often than not). Completes the three canonical voting rules {score=pooled, plurality=per_period, Condorcet}. Returns
    the model name, or None if no Condorcet winner exists (a majority cycle or a pairwise tie). L: {model: array(T)}.

    Accepts a `LossPanel` directly (FIXED 2026-09-07, round-4 8-lens PyPI-preflight audit -- see
    per_period_winner's docstring for the full account)."""
    L, _ = _unwrap_panel(L, None)
    L = _as_loss_dict(L)
    _validate_losses(L)
    models = list(L)
    M = {m: np.asarray(L[m], float) for m in models}
    for a in models:
        if all(a == b or int((M[a] < M[b]).sum()) > int((M[b] < M[a]).sum()) for b in models):
            return a
    return None


def condorcet_status(L: dict):
    """Classify the pairwise-majority tournament among the models:
      'winner' -- a Condorcet winner exists (one model strictly beats every other in per-period majority);
      'cycle'  -- NO Condorcet winner AND the strict-beats digraph contains a directed cycle of ANY length
                  (a1>a2>...>ak>a1) -- a genuine intransitive majority cycle;
      'tie'    -- NO Condorcet winner and NO directed cycle (the absence is driven by pairwise majority TIES,
                  not intransitivity).
    'cycle' and 'tie' are BOTH 'no Condorcet winner', but only 'cycle' is the severe intransitive disagreement.
    Cycles are detected at ANY length via a topological-sort (Kahn) test on the strict-beats digraph: a 3-permutation
    search misses longer cycles -- e.g. a genuine 4-cycle with no 3-cycle, which can occur when some pairs are TIED
    (ties make the tournament incomplete, so 'no winner' no longer implies a 3-cycle exists).
    'x beats y' iff x has strictly more per-period wins than y (same strict relation as condorcet_winner).

    Accepts a `LossPanel` directly (FIXED 2026-09-07, round-4 8-lens PyPI-preflight audit -- see
    per_period_winner's docstring for the full account)."""
    L, _ = _unwrap_panel(L, None)
    L = _as_loss_dict(L)
    _validate_losses(L)
    from collections import deque
    models = list(L)
    M = {m: np.asarray(L[m], float) for m in models}
    def beats(x, y):
        return int((M[x] < M[y]).sum()) > int((M[y] < M[x]).sum())
    for a in models:
        if all(a == b or beats(a, b) for b in models):
            return "winner"
    # No Condorcet winner: it is a genuine CYCLE iff the strict-beats digraph is cyclic (not a DAG). Kahn's
    # algorithm -- if a topological order covers every node the graph is acyclic (=> 'tie', absence via pairwise
    # ties); otherwise a directed cycle of some length remains (=> genuine intransitive 'cycle').
    indeg = {m: 0 for m in models}
    adj = {m: [] for m in models}
    for x in models:
        for y in models:
            if x != y and beats(x, y):
                adj[x].append(y); indeg[y] += 1
    q = deque([m for m in models if indeg[m] == 0]); ordered = 0
    while q:
        u = q.popleft(); ordered += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return "cycle" if ordered < len(models) else "tie"


# A PER-PERIOD pooled margin below this fraction of the loss scale is not a win, it is accumulated
# floating-point noise. Mirrors mcs.py's `_SE_REL_FLOOR`, deliberately: the two modules were solving the
# same problem and only one of them had been fixed.
#
# Per-period, not total: the summed margin grows with T, so a total-margin threshold would call the same
# data degenerate at T=10 and sound at T=40. The floor sits about four orders of magnitude above machine
# epsilon, so it catches margins that are indistinguishable from zero in arithmetic.
#
# WHAT IT DELIBERATELY DOES NOT CATCH: a margin that is real but economically trivial (say 1e-9 of the
# loss scale, consistently, every period). That is a genuine, reproducible win, and k* correctly reports
# it as robust -- you would have to delete every period to overturn it. Whether such a win MATTERS is a
# question about the application, not about the arithmetic, and this instrument does not answer it.
# Loss magnitudes beyond this overflow float64 in the margin/scale arithmetic below: `inf <= inf`
# evaluates True, so a maximally decisive win is misclassified as k*=0, "no strict winner". 1e100 sits
# safely under float64's ~1.8e308 limit with room for the squaring and summing this package does.
# Imported by resolution.py, identify.py and panel.py so one number governs the whole package.
_MAX_ABS_LOSS = 1e100

_MARGIN_REL_FLOOR = 1e-12


def _pair_scale(la, lb, w, T):
    """The magnitude a PAIRWISE margin has to be judged against -- (mean(|la|)+mean(|lb|))/2 *
    mean(w). EXTRACTED 2026-09-02 (10-agent code-review pass) from a closure local to fragility()
    so winner_stability() can share the identical degeneracy check fragility() already applies
    (see _is_degenerate_pair below) -- previously winner_stability() had no equivalent guard when
    called directly, only when called indirectly through fragility(), and still returned exactly
    1.0 ("rock-solid") on an exact tie -- fragility.py's own commentary calls that "the single most
    misleading output this tool can produce"."""
    s = float(np.mean(np.abs(la)) + np.mean(np.abs(lb))) / 2.0 if T else 0.0
    wscale = float(np.mean(w)) if T else 0.0
    return s * wscale


def _is_degenerate_pair(k, la, lb, w, T):
    """True iff the champion-vs-binding-rival margin is degenerate: k*==0 (no strict pooled
    winner) or the margin is negligible relative to the pair's own loss scale. Shared by
    fragility() and winner_stability() -- see _pair_scale's docstring."""
    c = np.asarray(w, float) * (np.asarray(lb, float) - np.asarray(la, float))
    M = float(c.sum())
    return bool(k == 0 or (M / T) <= _MARGIN_REL_FLOOR * _pair_scale(la, lb, w, T))


def _loss_scale(L):
    """The magnitude a margin has to be judged against. Absolute thresholds are magnitude-dependent dead
    code: 1e-13 is negligible for losses of order 1 and enormous for losses of order 1e-18."""
    # _as_loss_dict FIRST: `L.values` on a DataFrame is an ndarray attribute, not a method, so calling
    # it raises. This function is reached from fragility() where L is already coerced, but it is module
    # level and the coercion has to be local or the two copies diverge on DataFrame input -- which they
    # did, silently, on a path differential_test cannot generate because make_case only emits dicts.
    L = _as_loss_dict(L)
    vals = np.concatenate([np.abs(np.asarray(v, float)).ravel() for v in L.values()])
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else 0.0


def breakdown_number(la, lb, w):
    """k* for a PAIR: the minimum number of periods to DELETE to flip 'a beats b' on the pooled metric, where a is the
    current pooled winner of the pair. Margin contributions c_t = w_t * (lb_t - la_t)  (>0 means period t favors a; net
    margin M = sum(c_t) > 0 iff a beats b pooled). Deleting set S gives M_S = M - sum_{t in S} c_t. To make M_S <= 0
    with the fewest deletions, remove the largest positive c_t first (greedy) -- EXACT because the margin is additive in
    fixed weights. Returns (k*, responsible_period_indices). If a does not beat b pooled, returns (0, []).

    ASYMMETRY, NOT A BUG (round-7 A/B-testing realistic-data review, 2026-08-27): responsible_period_
    indices names periods propping up A's (the CURRENT winner's) margin -- the largest w_t*(lb_t-la_t)
    contributions -- never periods where B unusually outperforms A. A real disruption that TEMPORARILY
    favors the challenger (e.g. a platform regression that briefly helps a losing variant) is therefore
    nearly invisible here even though something genuinely anomalous happened: verified directly on a
    13-period disruption window, 11/13 disrupted periods were named when the disruption reinforced the
    champion's lead, but only 1/13 when it inverted (favored the challenger instead). This is the
    mathematically correct answer to "which periods would need to disappear to flip the CURRENT
    winner" -- it is not an answer to "which periods look anomalous" in general, and a reader should
    not read PIVOT's silence during an inverting disruption as "nothing happened."

    DEGENERATE-MARGIN FLOOR. FIXED 2026-08-16 (confirmed regression, T2/T3): an EXACT pooled tie
    (mean_a == mean_b) must give k*=0 -- there is no strict winner to break down. But summing T
    individually-rounded float64 differences does not reliably land on exactly 0.0 even when the true
    means are equal (verified: `exact_tie_panel`'s means agree exactly in Decimal arithmetic, but the
    naive float64 M came out ~1e-16, a positive residual). The greedy loop below would then remove the
    single largest c_t -- often orders of magnitude bigger than the residual -- and report k*=1 on data
    that is an exact tie, i.e. `M > 0` in floating point is picking up rounding noise, not a genuine
    margin. Same relative-floor discipline as `mcs.py`'s `_SE_REL_FLOOR` and this module's own
    `_MARGIN_REL_FLOOR` (used downstream in `fragility()`'s degenerate-margin guard) -- applied HERE,
    at the source, so k* itself is correct rather than merely flagged wrong after the fact.

    TIE-BREAK for equal-value margin contributions. FIXED 2026-08-28 (same defect class as
    decision_breakdown's opponent tie, found in the paper repo's Phase-2 holistic review and ported
    back here since this file used the identical unfixed pattern). `np.argsort(c)` does not
    guarantee stability for equal values (default kind='quicksort'), and the old `[::-1]` then
    reversed the WHOLE result -- so two periods with the EXACT same margin contribution could be
    named/removed in an order that is an accidental byproduct of the sort algorithm, undisclosed and
    not necessarily even stable across numpy versions, rather than a deliberate convention. k* (the
    removal COUNT) is unaffected -- deleting either member of a tied pair removes the same amount --
    but WHICH periods land in `removed_period_indices` is not. Now `np.argsort(-c, kind='stable')`:
    identical result to the old code whenever no two contributions tie exactly, and for a genuine
    tie, deterministically prefers the EARLIER period (ascending original index) -- stated and
    reproducible rather than an artifact of an unstable sort's internal behavior."""
    # REGRESSION FIX (test S3 TestBreakdownNumberWeightValidation, dated 2026-08-17).
    # breakdown_number is PUBLIC and takes raw arrays, so it never passed through the dict-based
    # callers' weight checks. A negative weight landing on a model's one catastrophic period ran
    # silently to completion and reported a k* built on an inverted contribution rather than
    # refusing. panel._validate_weights already carried the right rules and messages; the only
    # defect was that this entry point did not call it.
    w = np.asarray(w, dtype=float)
    # DEFERRED IMPORT (2026-08-25): panel.py imports _MAX_ABS_LOSS from THIS module, so a
    # module-level "from .panel import _validate_weights" here creates a circular import the
    # moment panel.py is upgraded past its current stub. Deferring it to the one call site
    # avoids the cycle without adding a new shared module -- both directions of the coupling
    # are real (each file needs one thing from the other), so the fix is where the need is.
    from .panel import _validate_weights
    _validate_weights(w, w.size)

    la, lb, w = np.asarray(la, float), np.asarray(lb, float), np.asarray(w, float)
    # FIXED 2026-08-26 (independent ML-engineer review): `breakdown_number` is public and takes raw
    # arrays directly, so a caller reaching it WITHOUT going through `decision_breakdown` (which
    # validates `L` via `_validate_losses`, catching NaN/inf there) hit no finiteness check at all.
    # Verified directly: `breakdown_number(np.array([1., np.nan, 1., 1.]), np.array([2.]*4),
    # np.ones(4))` returned a confident `(4, [1, 3, 2, 0])` instead of erroring -- `np.argsort`
    # sorts NaN to the end rather than raising, so the greedy loop runs to completion on a margin
    # that was never real, exactly the "confident but meaningless answer" class `_validate_losses`
    # exists to prevent for every other public entry point in this module.
    if not (np.all(np.isfinite(la)) and np.all(np.isfinite(lb))):
        raise ValueError("breakdown_number requires finite (non-NaN/inf) losses for both models; a "
                          "single NaN silently poisons the margin sum and can report a confident, "
                          "meaningless k*.")
    c = w * (lb - la)
    M = float(c.sum())
    T = len(w)
    scale = float(np.mean(np.abs(la)) + np.mean(np.abs(lb))) / 2.0 if T else 0.0
    # WEIGHT-SCALE INVARIANCE. FIXED 2026-08-26 (round-4 property-based fuzzing, hypothesis).
    # `scale` above is a pure loss-magnitude reference and deliberately does not depend on `w` --
    # but M and s (below) are WEIGHTED sums, so they scale linearly with w's overall magnitude. The
    # floor comparisons originally compared these w-scaled quantities directly against w-independent
    # thresholds (`scale`, `scale*T`), which is only unit-consistent by coincidence when w's typical
    # value is ~1. Rescaling w by a positive constant (e.g. reporting weights "per week" instead of
    # "per day", a plain 7x change) changed which side of the floor M/T and s landed on, silently
    # changing k* by up to 1 -- even though this module's own docstring promises "weights are
    # observation counts, only relative weights matter" and callers are told rescaling is a no-op.
    # Minimal repro (now `test_decision_breakdown_weight_scale_invariance`):
    # L={'m0':[0,0,0,0],'m1':[0,0,20,1e-5]} gave k*=2 at w=ones(4) but k*=1 at w=ones(4)*1e-6 --
    # same panel, same RELATIVE weights, different k* purely from absolute scale. Fix: multiply the
    # floor by wscale=mean(w) so the threshold scales the same way M and s do; at wscale=1 (w=None or
    # any uniform-1 weighting, the case every pre-existing test and the paper's own equal-weight
    # passes use) this is exactly the prior behaviour, unchanged.
    wscale = float(np.mean(w)) if T else 0.0
    if M <= 0 or (T and (M / T) <= _MARGIN_REL_FLOOR * scale * wscale):
        return 0, []
    order = np.argsort(-c, kind="stable")                         # largest a-favoring contribution first, ties -> earliest period
    s, removed = M, []
    for i in order:
        if c[i] <= 0:                                            # removing a b-favoring period can't help flip
            break
        s -= float(c[i]); removed.append(int(i))
        # REGRESSION FIX (test S3 TestKStarExactness, 2026-08-17, blind re-implementation differential).
        # This was `if s <= 0` -- an EXACT-ZERO test on a value built by repeatedly subtracting
        # individually-rounded float64 terms. On decimal-quantised losses (MASE to 1dp is a realistic
        # shape) a margin that is mathematically exactly 0 computes as ~4.4e-16, so the loop removed one
        # MORE period than necessary and reported k*+1 -- always overstating robustness, the unsafe
        # direction for a fragility tool. 4.7-4.9% of decimal-quantised trials before, 0/400 after.
        # The entry guard above already uses this relative floor; the loop simply did not.
        if s <= 0 or s <= _MARGIN_REL_FLOOR * scale * T * wscale:
            break
    return len(removed), removed


def decision_breakdown(L: dict, w=None, a=None, return_ties=False):
    """DECISION-LEVEL k*: the fewest periods whose deletion makes the pooled winner `a` lose to ANY opponent
    (not just the pooled runner-up). k* = min over opponents m of the pairwise breakdown_number(a, m); greedy is
    exact per pair, so the min over pairs is exact for the decision.

    Returns (k*, binding_opponent, removed_idx), or with return_ties=True
    (k*, binding_opponent, removed_idx, all_binding_opponents).

    TIE-BREAK. k* is a minimum and is unambiguous, but SEVERAL opponents can attain it, and they disagree about the
    derived quantities -- `removed`, the pivotal period, and above all `concentration`, which is computed against
    whichever opponent is named. This previously resolved by dict insertion order, i.e. silently. It now resolves by
    sorted model name: still arbitrary, but deterministic and stated. Use return_ties=True to recover the full tied
    set and report a range, which is what an honest write-up should do when more than one opponent ties."""
    L, w = _unwrap_panel(L, w)
    # REGRESSION FIX (test S3 TestExtremeMagnitudeRefusal, dated 2026-08-17, numerical-extremes lens).
    # Loss magnitudes past ~1e150 silently overflow float64 in breakdown_number's margin/scale
    # arithmetic -- `inf <= inf` is True, so the most decisive possible win is reported as k*=0, "no
    # strict winner". Refuse cleanly instead of returning a confident wrong answer.
    for _m, _v in L.items():
        _v = np.asarray(_v, dtype=float)
        if _v.size and np.any(np.abs(_v) > _MAX_ABS_LOSS):
            raise ValueError(
                f"loss magnitude for model {_m!r} exceeds {_MAX_ABS_LOSS:.0e}; float64 arithmetic "
                f"overflows here and would silently report k*=0. Check for a units/scaling bug upstream.")

    L = _as_loss_dict(L)
    T = _validate_losses(L, w)
    w = np.ones(T) if w is None else w
    # `a = a or pooled_winner(...)` silently discarded a caller-supplied a=0/a="" (falsy but valid);
    # must test identity against None, not truthiness. Fixed 2026-08-16.
    if a is None:
        a = pooled_winner(L, w)
    la = np.asarray(L[a], float)
    best = None                                                  # (k, opponent, removed)
    ties = []
    for m in sorted(m for m in L if m != a):                     # sorted -> caller-independent
        k, removed = breakdown_number(la, np.asarray(L[m], float), w)
        if best is None or k < best[0]:
            best = (k, m, removed); ties = [m]
        elif k == best[0]:
            ties.append(m)
    if best is None:
        # Currently unreachable: `_validate_losses` (called by every caller of this function) already
        # enforces len(L) >= 2, so the loop above always has >= 1 iteration and `best` is always set.
        # Left as defensive code rather than removed (round-8 fresh-eyes review, 2026-08-27) -- not
        # deleted, in case a future caller bypasses that guard.
        return (0, None, [], []) if return_ties else (0, None, [])
    return (*best, ties) if return_ties else best


def _block_resample_idx(T, block, rng, circular=True):
    """Block-bootstrap indices over evaluation periods, CIRCULAR by default (Politis & Romano 1992).

    Why circular rather than moving-block. The moving-block scheme draws a start uniformly from 0..T-block, so an
    observation near either END of the sample is reachable by only the few blocks that overlap it, while an
    interior observation belongs to `block` distinct windows. Inclusion probabilities are therefore markedly
    non-uniform: measured at T=38, block=3, the first and last observations are resampled at ~0.34x the rate of an
    interior one -- a 3.2x spread, widening to 5.2x at block=5. The circular scheme wraps blocks past the end, so
    every observation lies in exactly `block` windows and inclusion is uniform up to Monte-Carlo noise. It also
    makes the estimate far less sensitive to the block length, which is a tuning parameter no user should have to
    reason about: on a reference cell, winner_stability varies several times more across block lengths under
    moving-block than under circular, and the moving-block curve steps sharply between block=1 and block=2 as the
    edge effect switches on. The measured spreads are reported in the accompanying paper rather than restated
    here -- this docstring and the pipeline's copy of this function had drifted to two different sets of values,
    and neither matched the table they came from.

    The cost is joining the end of the sample to its beginning, which is legitimate under the stationarity already
    assumed by the Model Confidence Set apparatus; the canonical MCS implementations use the closely related
    stationary bootstrap, which also wraps. Pass circular=False to recover the previous moving-block behaviour.
    """
    if block <= 1:
        return rng.integers(0, T, size=T)                     # iid case: identical under either scheme
    if not circular:
        out = []
        while len(out) < T:
            start = int(rng.integers(0, max(1, T - block + 1)))
            out.extend(range(start, min(start + block, T)))
        return np.asarray(out[:T])
    starts = rng.integers(0, T, size=int(np.ceil(T / block)))
    return np.concatenate([(s + np.arange(block)) % T for s in starts])[:T]


def winner_stability(L: dict, w=None, n_boot=2000, block=None, seed=0, circular=True):
    """Block-bootstrap the evaluation PERIODS and recompute the pooled winner on each resample -> the distribution of
    'who wins'. Returns (stability, freq): `stability` = fraction of resamples whose pooled winner equals the observed
    pooled winner; `freq` = {model: win share}.

    THIS STATISTIC IS NOT INTERPRETABLE ON ITS OWN. Read it against `exchangeable_benchmark(K, T)`, which returns
    (median, p05, p95) of this same statistic on panels where NO model is better and the winner is arbitrary by
    construction. It does NOT go to 1/K under that null -- at K=6, T=32 the null median is about 0.51 and the 5th
    percentile about 0.31, so a value in the 0.4-0.6 range is what CHANCE produces, not evidence of fragility. A
    result is unusually fragile RELATIVE TO CHANCE only if it falls below the benchmark's 5th percentile.

    It is the bootstrap companion to k* (a one-axis, period-resampling view of the winner's sampling variability),
    calibrated only when paired with `exchangeable_benchmark()`."""
    L, w = _unwrap_panel(L, w)
    L = _as_loss_dict(L)
    T = _validate_losses(L, w)
    w = np.ones(T) if w is None else w
    w = np.asarray(w, float)
    T = len(w)
    # FIXED 2026-09-02 (10-agent code-review pass, CONFIRMED BUG): `block = block or default` is
    # Python's classic falsy-default footgun -- an explicit block=0 is truthy-false, so it was
    # silently DISCARDED and replaced with the computed default, identical in kind to the
    # `a = a or pooled_winner(...)` bug already fixed for decision_breakdown's `a=` parameter
    # ("must test identity against None, not truthiness"). Tests identity against None instead, so
    # an explicit block=0 is honored (falls into the `block <= 1` iid branch below), not silently
    # swapped for a different, unrequested bootstrap scheme.
    block = max(1, round(T ** (1 / 3))) if block is None else block
    # FIXED 2026-08-26 (independent ML-engineer review): at block>=T the circular scheme in
    # `_block_resample_idx` draws exactly one start and wraps a full length-T block, i.e. every
    # resample is a PERMUTATION (a rotation) of the same T periods, never a genuine subsample.
    # `pooled_winner`'s weighted mean is invariant to period order, so the winner is IDENTICAL on
    # every resample and this function silently returns exactly 1.0 regardless of the data --
    # verified directly at block==T and at block>T. `mcs()` elsewhere in this package already
    # refuses the analogous degenerate block/T configuration; this function did not, and 1.0
    # ("rock-solid") is the most misleading possible output it could produce here.
    if block >= T:
        raise ValueError(f"block={block} >= T={T} periods -- a circular block bootstrap with a "
                          f"block at least as long as the sample only ever produces a rotation of "
                          f"the same {T} periods, so the pooled winner cannot change across "
                          f"resamples and stability would be trivially 1.0 regardless of the data. "
                          f"Pass a smaller block (default is round(T**(1/3))).")
    # FIXED 2026-09-02 (10-agent code-review pass, CONFIRMED BUG): n_boot<=0 was never checked --
    # n_boot=0 crashed with a bare ZeroDivisionError, and n_boot<0 made `range(n_boot)` empty (the
    # win-tally stayed all-zero) so `counts[m]/n_boot` silently computed 0/negative=-0.0 for every
    # model, a plausible-looking float, not an obvious error.
    if n_boot <= 0:
        raise ValueError(f"n_boot must be a positive integer; got {n_boot}.")
    # FIXED 2026-09-02 (10-agent code-review pass, CONFIRMED GAP): fragility()'s own inline
    # commentary calls winner_stability()==1.0 on an exact tie "the single most misleading output
    # this tool can produce" and guards against it there via a degenerate-margin check computed
    # BEFORE calling this function -- but this function, called directly (its own public,
    # independently-exported entry point, not merely an internal helper of fragility()), had no
    # equivalent guard and still returned exactly 1.0 ("100% rock solid") for models that are
    # byte-identical every period. Ported the identical check fragility() already applies (shared
    # via the module-level _pair_scale/_is_degenerate_pair helpers, so the two can never diverge on
    # what counts as degenerate) so BOTH entry points give the same, honest answer.
    obs = pooled_winner(L, w)
    k, b, _removed = decision_breakdown(L, w, a=obs)
    la = np.asarray(L[obs], float)
    lb = np.asarray(L[b], float) if b is not None else la
    if _is_degenerate_pair(k, la, lb, w, T):
        return float("nan"), {m: 1.0 / len(L) for m in L}
    rng = np.random.default_rng(seed)
    counts = {m: 0 for m in L}
    for _ in range(n_boot):
        idx = _block_resample_idx(T, block, rng, circular=circular)
        Lb = {m: np.asarray(L[m], float)[idx] for m in L}
        counts[pooled_winner(Lb, w[idx])] += 1
    freq = {m: counts[m] / n_boot for m in L}
    return freq[obs], freq


def exchangeable_benchmark(K, T, n_rep=200, n_boot=500, block=None, seed=0, circular=True):
    """The distribution of `winner_stability` when the winner is ARBITRARY BY CONSTRUCTION.

    K models drawn from one identical distribution, so no model is better and any winner is an artifact
    of the draw. Returns (median, p05, p95) of the statistic over `n_rep` such panels.

    This exists because `winner_stability` was being read against 1/K -- "0.50 is a coin flip" -- which is
    wrong for K > 2: the statistic does not go to 1/K under the null, it goes to roughly 0.52 at K=6,
    T=30. A threshold or a median is only interpretable against this benchmark, and the benchmark is
    cheap enough to compute that asserting one instead is indefensible.
    """
    w = np.ones(T)
    out = []
    for r in range(n_rep):
        g = np.random.default_rng(seed + 1 + r)
        L = {f"m{i}": g.normal(1.0, 0.30, T) for i in range(K)}
        # 2026-08-05 in the pipeline copy; MIRRORED HERE 2026-08-15. Every replicate used to pass the
        # SAME `seed` to winner_stability, so the inner bootstrap's resample draws were byte-identical
        # across all n_rep replicates -- an unintentional common-random-numbers scheme that only the
        # DATA varied across, which slightly narrows the reported null spread. Each replicate's
        # bootstrap now gets its own independent seed, offset well clear of the data-generation stream
        # above so the two never collide.
        #
        # This function is NOT covered by differential_test.py, which is how the two copies were able
        # to disagree for ten days after being re-mirrored one minute apart. An unused
        # `rng = np.random.default_rng(seed)` was also removed -- it was the vestige of the old scheme.
        boot_seed = seed + 1_000_003 + r
        out.append(winner_stability(L, w, n_boot=n_boot, block=block, seed=boot_seed, circular=circular)[0])
    out = np.sort(np.asarray(out, float))
    return float(np.median(out)), float(np.percentile(out, 5)), float(np.percentile(out, 95))


def surprise_concentration(L: dict, labels=None):
    """How concentrated is the IRREDUCIBLE forecast difficulty across periods? This is the DATA-side driver of ranking
    fragility, and it distinguishes the two fragility flavors (shock-concentration vs near-tie noise).

    NOTE -- do not confuse this function's `sci` with `fragility()`'s `concentration` field. `sci` is a property of
    the DATA (how lopsided task difficulty is across periods, independent of any winner); `concentration` in
    `fragility()` is a property of the DECISION (the champion's margin concentrated in one period). A series can
    score high on one and low on the other.

    Per period t: difficulty d_t = min_m loss_{m,t}  (the best ANY model achieved -> the surprise no model escaped;
    strips model-specific incompetence, isolates TASK difficulty). Normal difficulty b = median_t d_t (a robust,
    shock-label-free baseline). Excess surprise s_t = max(0, d_t - b). Returns:
      sci                 = max(s)/sum(s) in [0,1]   -- share of all excess surprise in the SINGLE hardest period
      eff_surprising_periods = (sum s)^2 / sum(s^2)  -- effective number of surprising periods (participation ratio)
      hardest             = label of argmax d_t      -- WHICH period is the surprise
      d_max_over_median   = max(d)/b                 -- how many times normal the hardest period was; NaN if b and
                                                         max(d) have opposite signs (the ratio is then undefined --
                                                         reachable with negative losses, e.g. log/skill scores)
    Reading: high sci + fragile ranking = SHOCK-driven inversion; low sci + fragile = NEAR-TIE leaderboard noise.

    Accepts a `LossPanel` directly (FIXED 2026-09-07, round-4 8-lens PyPI-preflight audit: found as
    a follow-up to the same fix for per_period_winner/condorcet_winner/condorcet_status just above
    in this file -- same file, same {model: array} shape, same missed case)."""
    L, _ = _unwrap_panel(L, None)
    L = _as_loss_dict(L)
    _validate_losses(L)
    models = list(L)
    Mmat = np.vstack([np.asarray(L[m], float) for m in models])   # models x T
    d = Mmat.min(axis=0)                                           # irreducible per-period difficulty
    b = float(np.median(d))
    s = np.maximum(0.0, d - b)
    tot = float(s.sum())
    lab = list(labels) if labels is not None else list(range(len(d)))
    d_max = float(np.max(d))
    return {
        "sci": float(s.max() / tot) if tot > 0 else 0.0,
        "eff_surprising_periods": float(tot ** 2 / np.sum(s ** 2)) if tot > 0 else float(len(d)),
        "hardest": lab[int(np.argmax(d))],
        # REGRESSION FIX (test S3 TestSurpriseConcentration, 2026-08-17, numerical-extremes lens).
        # b == 0 was blanket-mapped to inf, but that conflates two different panels. If the hardest
        # period is ALSO zero, every period is exactly at baseline and nothing is surprising -- the
        # honest ratio is 1.0, not "infinitely more than normal". inf stays correct only when the
        # numerator is genuinely positive over a zero baseline.
        # NEGATIVE BASELINES. FIXED 2026-09-02 (round-2 10-agent review, CONFIRMED): the guard was a
        # SIGN test (`b > 0`), not a ZERO test, so ANY panel with a negative median difficulty fell
        # into the else-branch and returned inf even though max(d)/b is perfectly finite. Negative
        # losses are routine and pass _validate_losses (which checks finiteness only): log scores,
        # skill scores, negative log-likelihood. Reproduced: b=-5.175, max(d)=-3.958, true ratio
        # 0.765, reported inf. The 2026-08-17 fix that introduced this only split the b == 0 case.
        # MIXED-SIGN BASELINES. FIXED 2026-09-05 (round-3 audit, CONFIRMED): the 2026-09-02 b!=0 fix
        # is only correct when max(d) and b share a sign (both negative -- e.g. the 0.765 example
        # above -- or both positive). Because max(d) >= b always holds (max of an array is never
        # below its own median), a NEGATIVE b with a NON-NEGATIVE max(d) is reachable and produces
        # max(d)/b <= 0 -- a negative or zero "times normal" multiplier, which is not a coherent
        # reading of "how many times normal the hardest period was" (reproduced directly:
        # L={'a':[-10,-9,-8,5],'b':[-9,-8,-7,6]} -> d=[-10,-9,-8,5], b=-8.5, max(d)=5.0, ratio
        # -0.588). The ratio is well-defined only when b and max(d) share a sign (or b==0, handled above);
        # a sign mismatch is reported as NaN rather than a number that reads as "less than normal"
        # when it is, in fact, an undefined comparison.
        "d_max_over_median": (
            1.0 if b == 0.0 and d_max == 0.0 else
            np.inf if b == 0.0 else
            np.nan if (b < 0.0) != (d_max < 0.0) else
            float(d_max / b)
        ),
    }


def fragility(L: dict, w=None, shock_periods=None, labels=None, n_boot=2000, seed=0, *,
              stability_block=None, benchmark=None):
    """Full ranking-fragility diagnostic for a selection decision.

    Parameters
    ----------
    L : dict {model_name: array(T)}   per-period loss for each model (lower is better).
    w : array(T) or None              per-period weights (e.g. observation counts). Defaults to equal weights.
    shock_periods : iterable or None  labels of known shock periods; if given, `shock_coincidence` reports whether the
                                      k* responsible periods are all known shocks.
    labels : iterable(T) or None      period labels (e.g. years); used to report the responsible periods by name.
    n_boot, seed :                    block-bootstrap settings for winner_stability.

    Returns a dict with: pooled_winner, per_period_winner, condorcet_winner, runner_up, k_star, k_over_T,
    concentration (share of the margin from the single most influential period), reversal (score != plurality),
    reversal_condorcet, condorcet_status ('winner'|'cycle'|'tie'), condorcet_cycle, condorcet_undefined,
    responsible (the k* periods), shock_coincidence, winner_stability, winner_freq, fragile (bool).

    Notes
    -----
    - k_star == 0 means the pooled winner does NOT strictly beat some opponent (a tie / no strict winner). It is NOT
      "maximally fragile" -- it signals no strict winner exists; read the Model Confidence Set for identification.
    - TWO DIFFERENT FIELDS, and this docstring used to conflate them. `screen` is the uncalibrated convenience
      flag: True if the margin is degenerate, OR winner_stability < 0.60, OR k_star/T < 0.25. That 0.60 cut is NOT
      calibrated -- under `exchangeable_benchmark()` (no true skill difference at all) the null median is about
      0.50 at K=6/T=32, so it flags much of a pure-null panel. `fragile` is the CALIBRATED verdict and is
      `None` -- unknown -- unless you pass `benchmark=`, because fragility relative to chance cannot be decided
      without the null at your own (K, T). An earlier version of this block described `screen`'s rule under
      `fragile`'s name and added that "an exact tie can show fragile=False"; that is wrong in both directions -- an
      exact tie is a degenerate margin and returns fragile=True, and without a benchmark the answer is None rather
      than False. Read `fragile` together with the MCS: it does not by itself capture non-identification.
    - concentration is np.nan when there is no strict winner (k_star == 0); a value > 1 means one period's margin
      exceeds the champion's whole net margin.
    - `stability_block` sets the winner_stability bootstrap block length (default T**(1/3)); set it to reflect the
      serial dependence of your losses (e.g. ~12 for monthly data)."""
    L, w = _unwrap_panel(L, w)
    L = _as_loss_dict(L)
    _validate_losses(L, w)
    if w is None:
        w = np.ones(len(next(iter(L.values()))))
    w = np.asarray(w, float)
    T = len(w)
    a = pooled_winner(L, w)
    # sorted(): on an exact pooled-loss tie, `min` otherwise returns whichever model the caller happened to
    # insert into the dict first, so the same data in a different key order gives a different runner-up.
    # Same determinism rule as pooled_winner() and decision_breakdown().
    pooled_runner_up = min(sorted(m for m in L if m != a),
                           key=lambda m: float(np.average(np.asarray(L[m], float), weights=w)))
    k, b, removed, binding_ties = decision_breakdown(L, w, a, return_ties=True)
    la = np.asarray(L[a], float)
    lb = np.asarray(L[b], float) if b is not None else la          # guard: single-model dict (no opponents)
    # `b is None` is currently unreachable here too, for the same reason as decision_breakdown's own
    # `best is None` branch above -- `_validate_losses` already enforces len(L) >= 2. Left in place,
    # not removed (round-8 fresh-eyes review, 2026-08-27).
    c = w * (lb - la)
    M = float(c.sum())
    # FIXED 2026-08-27 (Phase-2 holistic review, Section C -- found in the paper repo's own frozen
    # copy of this file too, code/instrument/fragility.py, and fixed there identically the same
    # day). Was `if M > 0`, missing a degenerate-margin floor entirely: if M is a tiny POSITIVE
    # number at or below floating-point noise, dividing by it produces an enormous, meaningless
    # concentration ratio.
    #
    # SCALE CORRECTED same day (third-agent review of the fix above). The first version of this
    # fix reused `_loss_scale(L)` -- a WHOLE-PANEL scale -- matching `degenerate` below, which
    # already used it. But `breakdown_number()` itself (which actually computes k*, the thing
    # `degenerate`/`concentration` are supposed to describe) has always used a PER-PAIR scale (see
    # its own inline comment for why: a whole-panel scale lets one large-magnitude, never-winning
    # model contaminate the floor for an unrelated, genuinely near-tied pair). Demonstrated as a
    # real, live inconsistency, not just a style mismatch: a 3-model panel with a genuine k*=1
    # breakdown between two near-tied models, plus an unrelated third model at a much larger
    # magnitude, reported degenerate_margin=True / concentration=NaN even though
    # breakdown_number() on that exact pair alone correctly finds a real, non-degenerate k*=1 --
    # contradicting this function's own docstring ("concentration is NaN when k_star==0"). Fixed by
    # using the SAME per-pair scale formula as breakdown_number (mean(|la|)+mean(|lb|))/2 *
    # mean(w)), for both `concentration`/`conc_by_opp` here AND `degenerate` below -- all three now
    # agree with the actual k* computation on what counts as degenerate.
    def _pair_scale(_la, _lb):
        s = float(np.mean(np.abs(_la)) + np.mean(np.abs(_lb))) / 2.0 if T else 0.0
        wscale = float(np.mean(w)) if T else 0.0
        return s * wscale
    _pscale_ab = _pair_scale(la, lb)
    concentration = (float(np.max(c) / M)
                     if M > 0 and (M / T) > _MARGIN_REL_FLOOR * _pscale_ab else np.nan)
    # `concentration` is computed against the NAMED binding opponent. When several opponents tie at k*, each
    # implies a different value, so report the range too and let the caller decide whether a point estimate is
    # honest. Silently reporting one of several is how a large concentration gets quoted when a co-equal
    # opponent would have given a small one.
    conc_by_opp = {}
    for _m in binding_ties:
        _lm = np.asarray(L[_m], float)
        _cm = w * (_lm - la)
        _Mm = float(_cm.sum())
        _pscale_am = _pair_scale(la, _lm)
        conc_by_opp[_m] = (float(np.max(_cm) / _Mm)
                           if _Mm > 0 and (_Mm / T) > _MARGIN_REL_FLOOR * _pscale_am else np.nan)
    _fin = [v for v in conc_by_opp.values() if np.isfinite(v)]
    concentration_range = (min(_fin), max(_fin)) if _fin else (np.nan, np.nan)
    pp, plurality_ties = per_period_winner(L, return_ties=True)
    cw = condorcet_winner(L)
    cstat = condorcet_status(L)                # 'winner' | 'cycle' | 'tie' (cw is not None <=> 'winner')
    lab = list(labels) if labels is not None else list(range(T))
    resp = [lab[i] for i in removed]
    coincide = None
    if shock_periods is not None:
        sset = set(shock_periods)
        coincide = (len(resp) > 0) and all(r in sset for r in resp)
    # DEGENERATE MARGIN GUARD. k*==0 means there is no STRICT pooled winner -- the margin against the binding
    # opponent is <= 0, i.e. an exact or effective tie. In that case `winner_stability` is meaningless as
    # evidence: every bootstrap resample tie-breaks to the same (name-sorted) model, so it returns 1.0 --
    # "rock-solid" -- for two byte-identical models, with fragile=False. That is the single most misleading
    # output this tool can produce, and it is the opposite of the truth: an exact tie is MAXIMALLY
    # non-identified. Report the degeneracy explicitly rather than letting the null speak for it.
    # The guard was `k == 0`, reachable only on a BIT-EXACT pooled tie, so any epsilon escaped it into the
    # maximum robustness score: two models separated by one part in 1e13 reported k*/T = 1.00,
    # winner_stability = 1.00, degenerate = False and fragile = False -- maximally robust on every axis at
    # once, which is worse than the bug this guard was written to fix. A margin is degenerate when it is
    # negligible RELATIVE TO THE LOSS SCALE, never when it is exactly zero. mcs.py already reached this
    # conclusion and fixed it there (`_SE_REL_FLOOR * scale`); the same defect sat here untouched.
    #
    # COMPUTED BEFORE the bootstrap call, not after (round-8 fresh-eyes review, 2026-08-27): both of `degenerate`'s
    # ingredients (k, M) were already available here, but the full winner_stability() bootstrap (n_boot=2000 by
    # default) used to run FIRST regardless, then get its result discarded to NaN the moment `degenerate` turned
    # out True -- wasted work on exactly the near/exact-tie panels this whole project has repeatedly found to be
    # common in real decimal-quantized data. `stability`/`freq` are set directly below without paying for the
    # bootstrap when it can't change the answer.
    #
    # SCALE CORRECTED 2026-08-27 (third-agent review, alongside the concentration fix above): was
    # `_loss_scale(L)`, a WHOLE-PANEL scale -- inconsistent with breakdown_number()'s own per-pair
    # convention, the function that actually computes `k`. Reuses `_pscale_ab` (computed above for
    # `concentration`, same la/lb pair) so `degenerate` and `concentration` never disagree about
    # whether the SAME margin is degenerate.
    degenerate = bool(k == 0 or (M / T) <= _MARGIN_REL_FLOOR * _pscale_ab)
    if degenerate:
        stability = float("nan")          # not 1.0 -- there is nothing for the bootstrap to be stable about
        freq = {m: 1.0 / len(L) for m in L}   # no meaningful win-share distribution on a degenerate margin
    else:
        stability, freq = winner_stability(L, w, n_boot=n_boot, block=stability_block, seed=seed)
    # k* entered ONLY in conjunction with a reversal, so the instrument's headline object had no
    # independent path to its own summary flag: a cell with k*=1 -- one period outweighing the entire net
    # margin -- reported fragile=False. Every axis that can independently establish fragility must raise it.
    # TWO DIFFERENT OBJECTS -- see the pipeline copy for the full rationale. `screen` is the cheap
    # heuristic (a round-number stability cut plus a k*/T cut); `fragile` is fragility RELATIVE TO
    # CHANCE, which requires this panel's own exchangeable null. Until 2026-08-08 this function
    # returned the heuristic under the name `fragile`, so the tool applied exactly the round-number
    # cut the accompanying paper tells readers not to use, and `exchangeable_benchmark()` -- the
    # correct standard, defined in this same module -- had no call site.
    #
    # The null is not computed inline (~120,000 bootstrap evaluations per cell). Pass
    # `benchmark=exchangeable_benchmark(K, T, ...)` for a verdict; without one `fragile` is None,
    # meaning UNKNOWN, never False.
    screen = bool(degenerate or (stability < 0.60) or (k / T < 0.25))
    if degenerate:
        fragile = True
    elif benchmark is None:
        fragile = None
    else:
        # LIVE LANDMINE, FLAGGED 2026-08-16 (independent review): this `winner_stability < null_p05`
        # verdict was measured directly to be INVERTED -- it fires
        # MORE under pure noise (10.7% at nominal 5%) than against a genuine, strong true winner
        # (1.4%) -- and the whole v1.0 redesign exists to retire it, not patch it. This function and
        # its `benchmark=` parameter are kept ONLY because Stage 3's own regression tests still call
        # `fragility()` as a convenient bundle of decision_breakdown/per_period_winner/condorcet_*
        # (see tests/test_stage3_kstar_prop22.py) -- the fields those tests read (pooled_winner,
        # per_period_winner, reversal, condorcet_status, k_star, ...) are all still correct and
        # unaffected. `fragile` itself is not: warn loudly the one time it can actually be computed
        # (benchmark supplied), rather than let a new caller reproduce the exact inverted verdict this
        # redesign exists to escape. Use `resolution_report()` / `report()` instead.
        warnings.warn(
            "fragility()'s 'fragile' field (winner_stability < null_p05) was measured to be INVERTED "
            "-- it fires more often under pure noise than against a genuine, strong true winner -- "
            "and is retired in the v1.0 redesign, not patched. Use resolution_report()/report() "
            "instead; do not read or act on this field.",
            UserWarning, stacklevel=2,
        )
        _p05 = benchmark[1] if isinstance(benchmark, (tuple, list)) else float(benchmark)
        fragile = bool(stability < _p05)
    return {
        "pooled_winner": a, "per_period_winner": pp, "condorcet_winner": cw,
        # `binding_opponent` is the model that unseats `a` in the FEWEST deletions -- NOT the pooled second
        # place, which is `pooled_runner_up`. These differ on real data. `runner_up` is a deprecated alias.
        "screen": screen,                           # cheap heuristic triage flag, NOT a calibrated verdict
        "binding_opponent": b, "runner_up": b, "pooled_runner_up": pooled_runner_up,
        "binding_opponents": binding_ties,
        "plurality_winners": plurality_ties,
        "concentration_by_opponent": conc_by_opp,
        "concentration_range": concentration_range,
        "degenerate_margin": degenerate,   # True => no strict pooled winner; stability is NaN by design
        "k_star": int(k), "k_over_T": float(k / T) if T else np.nan,
        "concentration": concentration, "reversal": (a != pp),
        "reversal_condorcet": (cw is not None and a != cw),
        "condorcet_status": cstat,
        "condorcet_cycle": (cstat == "cycle"),
        "condorcet_undefined": (cw is None),
        "responsible": resp, "shock_coincidence": coincide,
        "winner_stability": float(stability), "winner_freq": freq, "fragile": fragile,
    }
