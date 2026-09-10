"""
selection_fragility.resolution — the resolution report: minimum detectable edge (MDE), selection_regret,
the MCB simultaneous bound, and the R1/R2 refusal rules.

No null distribution is needed anywhere in this module -- every statistic here is either a closed-form
power calculation (MDE, MCB) or a deterministic leave-one-out empirical quantity (selection_regret).
That absence of a null is itself the point: most of this design needs no simulated null at all, unlike
the retired `fragile`/`screen` verdict it replaces.
"""
import numpy as np
from scipy.stats import t as tdist

from .fragility import (_as_loss_dict, _validate_losses, pooled_winner, _MAX_ABS_LOSS, _unwrap_panel,
                         _MARGIN_REL_FLOOR)
from .identify import _validate_alpha


def _validate_one_sided(alpha, power=None):
    # CONFIRMED REGRESSION (independent 'wild' review, 2026-08-16): `_validate_alpha` only checks
    # 0<x<1, which is the right (and only) contract for identify.py's MCS alpha, but MDE/MCB use
    # `z = norm.ppf(1-alpha) + norm.ppf(power)` -- a genuinely one-sided construction that only makes
    # sense for alpha<0.5 (a "more than half the time" significance level is not a significance
    # level) and power>0.5 (a "less likely than a coin flip" power target is not a power target).
    # Verified directly: alpha=0.7 made mcb_bound() produce a bound TIGHTER than the observed point
    # estimate it's supposed to bound (violating the function's own documented "never tighter than
    # the observed edge" invariant), and power=0.01 made minimum_detectable_edge() go NEGATIVE,
    # which then defeated resolution_report()'s own R1 refusal gate on real project data (a
    # nonsensical negative MDE trivially clears any observed edge, reporting "resolved" when it
    # should refuse). Both are real, one-line-reachable, and neither was caught by the generic
    # 0<alpha<1 check used elsewhere in the package.
    _validate_alpha(alpha)
    if alpha >= 0.5:
        raise ValueError(f"alpha must be < 0.5 for a one-sided bound to be meaningful (alpha={alpha} "
                          f"would make the critical value non-positive, so the reported bound could "
                          f"come out TIGHTER than the observed edge it is supposed to bound); got {alpha}.")
    if power is not None:
        _validate_alpha(power)
        if power <= 0.5:
            raise ValueError(f"power must be > 0.5 (a power target at or below a coin flip is not a "
                              f"real power target, and makes the minimum detectable edge go "
                              f"negative); got {power}.")


def _effective_n(w):
    # Kish's (1965) effective sample size for a weighted mean: n_eff = (sum w)^2 / sum(w^2).
    # Reduces to exactly T when `w` is uniform (n_eff = (T*c)^2/(T*c^2) = T for any constant c);
    # falls toward 1 as weight concentrates onto a single period. See `_weighted_se`'s comment for
    # why this alone is not sufficient -- the variance ESTIMATE also has to stay unweighted.
    #
    # CONFIRMED REGRESSION (unconstrained re-audit, 2026-08-17): fragility.py's `_MAX_ABS_LOSS` caps
    # LOSS magnitude everywhere (panel.py, identify.py, fragility.py itself), but nothing capped
    # WEIGHT magnitude anywhere in the package -- `_validate_weights`/`_validate_losses`'s weight
    # checks only require finite/non-negative/not-all-zero. Squaring a weight here overflows float64
    # (~1.8e308) once weights reach ~1e155, silently returning NaN with zero exception and zero
    # visible warning at the call site. Verified directly: uniform w=full(30, 1e160) -- passes every
    # existing weight-validity check, including the MCS uniformity gate -- made resolution_report()
    # return a fully-formed dict with mde=nan, significance_boundary=nan, resolved=False, no error at
    # all. Same defect class already fixed twice on the loss side; applying the identical guard here.
    w = np.asarray(w, float)
    if np.any(np.abs(w) > _MAX_ABS_LOSS):
        raise ValueError(f"weight magnitude above {_MAX_ABS_LOSS:.0e}, far beyond any real observation-"
                          f"count or frequency weight -- squaring it overflows float64 silently. Check "
                          f"for a units/scaling bug upstream.")
    # FIXED 2026-09-02 (10-agent code-review pass, CONFIRMED SEV-1 finding): the check above guards
    # the OVERFLOW direction only. A weight vector at denormal magnitude (individually finite,
    # non-negative, non-zero -- passes every existing validation gate, including this function's
    # own upper-bound check and panel.py's _validate_weights) makes `w**2` UNDERFLOW to exactly 0.0
    # once |w| drops below ~1.5e-154, producing 0.0/0.0=nan here with only an easily-missed
    # RuntimeWarning, not an exception. That NaN then propagates through _weighted_se into
    # mde/significance_boundary/mcb_bound, and because NaN comparisons are always False in Python,
    # `resolved = bool(observed_edge >= sig_boundary)` silently flips to False -- even for a
    # maximally decisive true edge. Demonstrated: w=full(T, 1e-200) (all-uniform, so this is
    # nominally a documented no-op rescaling, exactly the case _effective_n's own docstring says
    # must be scale-invariant) flipped a 99-point, every-period-decisive edge from resolved=True to
    # resolved=False with zero error. Symmetric with the upper bound above (1e-100 is
    # _MAX_ABS_LOSS's reciprocal): well outside any real observation-count or frequency weight's
    # range, with a huge safety margin before squaring risks underflow.
    if np.any((w != 0) & (np.abs(w) < 1e-100)):
        raise ValueError(f"weight magnitude below 1e-100 (and nonzero) -- squaring it underflows "
                          f"float64 silently, corrupting the effective-sample-size computation with "
                          f"a NaN that then propagates through every resolution-report statistic. "
                          f"Check for a units/scaling bug upstream.")
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def _weighted_se(diff, w):
    # CONFIRMED REGRESSION (independent 'wild' review, 2026-08-17, numerical-extremes lens): both
    # minimum_detectable_edge and mcb_bound computed SE as `sqrt(weighted_variance(diff, w)) /
    # sqrt(T)` -- dividing by the RAW period count while also estimating the variance ITSELF with a
    # weighted formula. As weight concentrates onto one period (fully realistic for this package's
    # own documented use case, observation-count weights -- e.g. a state-population-weighted panel
    # where one state dwarfs the rest), the WEIGHTED VARIANCE collapses toward the single dominant
    # point's own (zero) deviation from the weighted mean, while T stays fixed -- so SE (and
    # therefore MDE) shrinks toward zero instead of growing to reflect that only about ONE period of
    # real information remains. Verified directly at REALISTIC magnitudes: with weight[0] going from
    # 1 to 1e8 on an otherwise-uniform T=30 weight vector, the WEIGHTED variance fell from 0.0262 to
    # ~0.0 while the UNWEIGHTED variance of the same `diff` array stayed exactly 0.0262 throughout --
    # confirming the collapse is an artifact of using a weighted variance estimator under skew, not a
    # real change in the underlying noise level. A first fix attempt corrected only the denominator
    # (T -> n_eff) and left the collapsing weighted-variance numerator in place; MDE still shrank
    # toward zero at extreme weight[0], just less steeply, with a spurious non-monotonic hump at
    # moderate skew -- caught by testing the fix against the same extreme-weight sweep before
    # trusting it. The correct fix (standard survey-statistics practice, Kish's design-effect
    # formulation `Var(weighted mean) = sigma^2 / n_eff` under a homoscedastic assumption): estimate
    # sigma^2 -- the noise level of an INDIVIDUAL period's differential -- from the plain, UNWEIGHTED
    # sample variance (a property of the data, not of how it happens to be weighted), and divide by
    # n_eff, not T, for the SE of the weighted MEAN specifically. Under uniform weights this is
    # numerically identical to the previous formula (weighted variance with uniform weights IS the
    # unweighted variance, and n_eff==T), so every existing uniform-weight result is unchanged.
    # CONFIRMED REGRESSION (control-flow/static-logic audit, 2026-08-17): this function's own comment
    # above says to estimate sigma^2 from the "plain, UNWEIGHTED sample variance", but `np.var`
    # defaults to ddof=0 (population variance, dividing by T), not the ddof=1 sample variance the
    # comment describes and the standard SE-of-the-mean formula requires. This understates SE by
    # exactly sqrt((T-1)/T) -- worst at small T, which is exactly this package's own worst case.
    # Verified directly at T=2 (documented minimum panel size): champ=[10,10], rival=[10.4,12.4] --
    # shipped ddof=0 SE=0.707 gave resolved=True; correct ddof=1 SE=1.0 gives resolved=False. Same
    # "overstated precision" direction as the earlier R1 fix this session (resolved compared against
    # MDE instead of significance_boundary) -- exactly the unsafe direction this codebase's other
    # fixes explicitly guard against, resurfacing here in the variance estimator itself.
    # CONFIRMED REGRESSION (unconstrained re-audit, 2026-08-17): `np.var(diff, ddof=1)` is
    # mathematically undefined for T=1 (divides by T-1=0). `selection_regret` in this same module
    # already guards its own minimum T explicitly (T<3 raises) -- this function, reached by
    # minimum_detectable_edge/significance_boundary/mcb_bound/resolution_report, had no such guard.
    # Verified directly: L={'champ':[1.0],'rival':[2.0]} silently returned mde=nan,
    # significance_boundary=nan, mcb_bound=nan (RuntimeWarning only, no exception); downstream in
    # resolution_report, NaN comparisons are always False in Python, so `resolved` silently comes out
    # False -- a plausible-looking verdict built on undefined arithmetic. These public functions are
    # callable directly on a raw dict without going through LossPanel's own T>=2 check, so this must
    # be enforced here too, matching identify.py's independent validation for the same reason.
    T = len(diff)
    if T < 2:
        raise ValueError(f"need at least 2 periods to estimate a standard error; got T={T}.")
    n_eff = _effective_n(w)
    var = float(np.var(diff, ddof=1))
    return float(np.sqrt(var) / np.sqrt(n_eff))


def _binding_rival(L, w, champ):
    """The rival CLOSEST to the champion on pooled weighted mean -- the one whose comparison actually
    defines how resolved this decision is. MDE must be computed against this rival, never averaged
    across all rivals: a distant rival cannot make a close call look more resolved than it is.

    NOT THE SAME MODEL AS fragility.py's "binding_opponent". ADDED 2026-09-10 (round-6 stress-review,
    api_consistency lens): fragility.py's `decision_breakdown`/`fragility()` pick `binding_opponent`
    by the FEWEST-DELETIONS criterion (k*); this function picks by CLOSEST POOLED MEAN. Different
    criteria, can name different models on the same panel (18.25% disagreement measured across
    20,000 random panels) -- see fragility.py's `decision_breakdown` docstring for the full account.

    CONFIRMED REGRESSION (independent 'wild' review, 2026-08-17, cross-fix interaction lens): every
    OTHER tie-break in this package was deliberately hardened tonight to be independent of dict/
    column insertion order (`pooled_winner`, `decision_breakdown`, `per_period_winner`, and
    `identify.py`'s own column-sort fix) -- this one was missed. `min(rivals, key=...)` on an exact
    tie for "closest mean to champion" silently resolves to whichever tied rival happens to sort
    first by dict order, not by name. Verified directly through the full public API: identical data
    built with dict keys in a different order flipped MDE from 12.7% to 14.7% and could flip the
    RESOLUTION verdict outright ("resolved" vs "cannot determine") -- the flagship report field,
    driven purely by Python's dict iteration order. `sorted(rivals)` makes the tie-break depend only
    on model IDENTITY, matching every sibling tie-break in the package.

    FIXED 2026-09-09 (round-5 stress-review, champion_pattern_hunt lens): the sorted-name tie-break
    above only guards an EXACT float tie of `abs(means[m]-means[champ])` -- it does not float this
    comparison to a relative floor for NEAR-ties, the same degenerate-tie-floor gap already fixed in
    pooled_winner() (2026-08-27/09-02) and in this module's own selection_regret() (2026-09-09).
    Reproduced directly: with two rivals tied at the champion's pooled mean to within float64 noise,
    a uniform weight rescaling (w -> w*1e-6, mathematically a no-op for np.average's ratio) flipped
    which rival compared closer, changing binding_rival, significance_boundary by 73%, and the
    headline `resolved` verdict itself -- on data whose only change was the unit the weights happen
    to be expressed in. Now floored the same way pooled_winner() floors its own tie: `means` here are
    already weight-scale-invariant RATIOS (np.average divides by sum(w)), so the tolerance must be too
    -- NO `wscale`/`_pair_scale` term, exactly per pooled_winner's own comment on this point ("the
    ratio np.average returns is already loss-scale magnitude regardless of w's scale"). A first
    version of this fix reused `_pair_scale` (which DOES carry a `mean(w)` factor, appropriate for
    RAW un-normalized margins like `_edge_components`'s own `M = sum(w*diff)`, not for a ratio) and
    the w*1e-6 repro above still flipped `binding_rival` -- caught by re-running that exact repro
    against the fix before trusting it."""
    means = {m: float(np.average(np.asarray(L[m], float), weights=w)) for m in L}
    rivals = [m for m in L if m != champ]
    dists = {m: abs(means[m] - means[champ]) for m in rivals}
    best_dist = min(dists.values())
    tied = [m for m in rivals
            if dists[m] - best_dist <= _MARGIN_REL_FLOOR * (abs(means[m]) + abs(means[champ])) / 2.0]
    return min(tied)


def _edge_components(L, w):
    """Shared setup for `minimum_detectable_edge`/`significance_boundary`/`mcb_bound`/
    `resolution_report`: the champion, its binding rival, the per-period (rival - champion)
    differential, its SE (Kish-corrected under weights), and both models' mean loss levels.

    EXTRACTED 2026-08-17 (own-review pass, architecture lens): `significance_boundary`
    (2026-08-17) duplicated `minimum_detectable_edge`'s setup near-verbatim, and `mcb_bound`
    duplicated it a third time -- exactly the "same computation needed in several places, only some
    of which get a fix when one of them changes" shape that caused `_binding_rival`'s own order-
    dependence bug earlier tonight (fixed in `identify.py`, missed in this module's copy of the same
    tie-break logic for two more review rounds). One shared computation means a future fix to the
    tie-break, the SE formula, or the scale only has to happen once, here."""
    champ = pooled_winner(L, w)
    rival = _binding_rival(L, w, champ)
    la = np.asarray(L[champ], float)
    lb = np.asarray(L[rival], float)
    diff = lb - la
    se = _weighted_se(diff, w)
    mean_champ = float(np.average(la, weights=w))
    mean_rival = float(np.average(lb, weights=w))
    return champ, rival, diff, se, mean_champ, mean_rival


def _scale(mean_champ):
    # abs(): a "loss level" used as a denominator must be a magnitude, not a signed value -- on a
    # synthetic panel noisy enough to put the champion's mean below zero, a signed scale flips the
    # sign of every downstream fraction (an MDE computed as strictly non-negative numerator/negative
    # scale came out negative, which the S2.1 extreme-variance test caught directly).
    return abs(mean_champ) if abs(mean_champ) > 1e-12 else 1.0


def minimum_detectable_edge(L, w=None, power=0.80, alpha=0.05):
    """The smallest true edge (as a fraction of the champion's loss level) this panel could detect at
    the given power and one-sided significance level, against the BINDING (closest) rival -- not an
    average across rivals, since a distant rival cannot make a close call look resolved. One-sided
    MDE: `(t_alpha + t_power) * SE`, `SE = std(diff, ddof=1)/sqrt(n_eff)`, `diff` the per-period
    (rival - champion) loss differential. Zero-variance (deterministic) panels give SE=0 and
    therefore MDE=0 -- any nonzero edge is detectable with certainty. No clipping: an MDE exceeding
    100% of the loss scale is a real, correctly-reported "you cannot resolve any practical edge
    here" result, not an error.

    `alpha` HERE IS INDEPENDENT OF `mcs()`/`identified()`/`mcs_size()`'s OWN `alpha` (NOTED
    2026-09-10, round-6 stress-review, api_consistency lens): this module's default (0.05) is a
    conventional one-sided significance level for a power calculation; the MCS family's default
    (0.10) is Hansen-Lunde-Nason's own convention for their equal-predictive-ability test. Both are
    individually standard for what they each control -- they are not meant to be the same number,
    and passing one where the other is expected is a real mistake this docstring exists to head off.

    CONFIRMED REGRESSION (real end-to-end practitioner workflow audit, 2026-08-17): `se` is
    ESTIMATED from the same small sample (ddof=1 sample variance over T periods), which is exactly
    the textbook case for a Student-t critical value with df=T-1, not a normal (z) one -- yet this
    used `norm.ppf`. t-critical > z-critical always, growing sharply as T shrinks (df=1: t=6.31 vs
    z=1.645, a 283% gap), so the z-approximation systematically UNDERSTATES the boundary needed --
    the same "overstated precision" direction as this session's earlier ddof=0-vs-1 fix in
    `_weighted_se`, just manifesting in the choice of DISTRIBUTION rather than the choice of
    variance divisor. Verified directly: T=2 (this package's own documented floor), two models whose
    losses move together (a routine real pattern -- shared macro shocks), gave `mde ~ 1.5e-16` under
    z -- the tool claiming it could detect an arbitrarily tiny true edge with 80% power from two data
    points. `(t_alpha+t_power)*SE` with df=T-1 is the standard t-approximation to the exact MDE (the
    fully exact planning quantity requires the noncentral t-distribution's inverse; this t-quantile
    sum is the widely-used applied approximation to it, exact only in the normal/known-variance
    limit as T grows -- consistent with how this module already documents MDE as an approximation,
    not a promise of exactness)."""
    _validate_one_sided(alpha, power)
    L, w = _unwrap_panel(L, w)
    L = _as_loss_dict(L)
    T = _validate_losses(L, w)
    w = np.ones(T) if w is None else np.asarray(w, float)
    _champ, _rival, _diff, se, mean_champ, _mean_rival = _edge_components(L, w)
    crit = tdist.ppf(1 - alpha, df=T - 1) + tdist.ppf(power, df=T - 1)
    return float(crit * se) / _scale(mean_champ)


def significance_boundary(L, w=None, alpha=0.05):
    """The one-sided significance threshold against the BINDING (closest) rival: `t_alpha * SE`, as a
    fraction of the champion's loss level. An observed edge clearing this bound is statistically
    significant at level `alpha` (one-sided) -- this is the correct decision threshold for whether an
    OBSERVED edge is real, distinct from `minimum_detectable_edge` (which additionally bakes in a
    power target and answers a different, PLANNING question: "what is the smallest true effect this
    sample size could reliably detect", not "is the effect I actually observed real").

    FIXED 2026-08-17 (independent 'wild' review, statistical-calibration lens): `resolution_report`'s
    R1 rule used to compare the observed edge directly against `minimum_detectable_edge`
    ((z_alpha+z_power)*SE) instead of against this boundary. Verified directly by simulation: at
    true_edge == the reported MDE, "observed_edge >= MDE" fired only ~52% of the time -- not the
    claimed 80% power -- because MDE bakes in a SECOND critical value (z_power) on top of the
    significance threshold, so requiring the OBSERVED value to also clear that inflated bar collapses
    the test back to roughly a coin flip at its own advertised detection point. This function's
    boundary, compared against the same construction, hits ~79.4% -- essentially exactly the claimed
    power, because it IS the standard one-sided significance test MDE's own derivation assumes.
    `minimum_detectable_edge` is unchanged and still reported in `resolution_report` as honest
    planning context; it must never be used as the significance decision threshold itself.

    CONFIRMED REGRESSION (real end-to-end practitioner workflow audit, 2026-08-17): this is the
    EXACT decision threshold R1 gates `resolved` on, and `se` is estimated from the same small
    sample -- textbook grounds for a Student-t critical value (df=T-1), not a normal one, exactly
    the same fix already applied to `minimum_detectable_edge` above (see its docstring for the T=2
    repro). This one is exact, not an approximation: a boundary built from an ESTIMATED variance is
    precisely the one-sample/paired t-test's own critical value, no noncentral-distribution
    approximation involved."""
    _validate_one_sided(alpha)
    L, w = _unwrap_panel(L, w)
    L = _as_loss_dict(L)
    T = _validate_losses(L, w)
    w = np.ones(T) if w is None else np.asarray(w, float)
    _champ, _rival, _diff, se, mean_champ, _mean_rival = _edge_components(L, w)
    crit = tdist.ppf(1 - alpha, df=T - 1)
    return float(crit * se) / _scale(mean_champ)


def mcb_bound(L, w=None, alpha=0.05):
    """MCB-style simultaneous bound: "the champion's true edge over the best rival could be as large
    as X%, at 1-alpha confidence" -- no test, so no power ceiling. A one-sided (1-alpha) UPPER
    confidence bound on the edge against the BINDING (closest) rival specifically -- "the best rival"
    in the design doc's own phrasing is singular, the strongest real competitor, not a worst case
    across every rival.

    CONFIRMED REGRESSION (textbook-fidelity audit, 2026-08-17): the docstring/report()/README text
    used to say "may be up to X% WORSE than the best rival" -- but `bound = point + t_crit*SE` with
    `point` already the champion's positive observed edge means `bound` is by construction ALWAYS
    >= the observed edge (see the "by construction never tighter" note below): this is an upper
    bound on how much BETTER the champion's true advantage could be, never a bound on how much worse
    it could be (that would require `point - t_crit*SE`, a DIFFERENT quantity this function does not
    compute, and would need its own from-scratch coverage validation -- tracked as a possible future
    addition, e.g. a separately-named function, not conflated with this one). Verified directly
    against the README's own shipped example: edge=6.8%, bound=44.8%, a value only possible as an
    upper bound on "better," not "worse." Math unchanged (already coverage-tested, see
    test_matched_coverage_across_tied_rival_counts below); only the English description was wrong.

    SELECTION-EFFECT CORRECTION. FIXED 2026-08-16 (independent review): `_binding_rival` picks
    whichever rival's SAMPLE mean looks closest to the champion, and an EARLIER version of this
    function then built the bound from that rival's own SE with a plain z_alpha critical value, with
    no correction for having picked it post-hoc out of several candidates -- a coverage simulation
    (K-1 genuinely tied rivals) showed actual coverage of the claimed 95% bound degrading well below
    nominal as the number of tied rivals grew, worsening monotonically with more tied rivals -- a
    real, measurable overclaim, the classic winner's-curse/selective-inference problem. A FIRST
    attempt at fixing this took the max over ALL rivals' own point differences with an
    alpha/(K-1)-corrected z, and came out drastically too conservative on the real 26-cell panel (most
    series carry several distant, obviously-uncompetitive rivals whose own point difference alone
    dominates that max, which is the wrong question -- "how bad is my worst rival" instead of "how
    much worse might my real competitor be"). The fix that actually restores calibrated coverage
    applies the Bonferroni correction to the CRITICAL VALUE only, still evaluated against the single
    BINDING rival's own point and SE -- correcting for the act of selecting among K-1 candidates
    without changing which comparison is being bounded. `point + t_{alpha/(K-1), df=T-1} * SE`, reported as a
    fraction of the champion's loss level -- by construction never tighter than the observed
    point-estimate edge, since the critical value and SE are both non-negative and the champion's mean
    is the global minimum by definition. The current coverage numbers are gated, not stated here --
    see tests/test_stage2_resolution.py::test_matched_coverage_across_tied_rival_counts for the
    verified figures; a number in this docstring would drift the next time this formula changes and
    nothing would catch it."""
    _validate_one_sided(alpha)
    L, w = _unwrap_panel(L, w)
    L = _as_loss_dict(L)
    # FIXED 2026-09-10 (round-6 stress-review, error_message_quality lens): a dedicated
    # "need at least 2 models for an MCB bound" message used to sit here, but _validate_losses()
    # below already raises its own (differently-worded) ">=2 models" ValueError on the exact same
    # condition, on the exact same data, and runs first -- this line was genuinely unreachable dead
    # code, confirmed by inspection and by calling mcb_bound() on a 1-model dict (it raises
    # _validate_losses's generic message, never this one). Removed; matches how every sibling
    # function in this module relies solely on _validate_losses for this case.
    T = _validate_losses(L, w)
    w = np.ones(T) if w is None else np.asarray(w, float)
    models = list(L)
    _champ, _rival, diff, se, mean_champ, _mean_rival = _edge_components(L, w)
    n_rivals = len(models) - 1
    point = float(np.average(diff, weights=w))   # kept as np.average(diff, ...), not mean_rival -
    # mean_champ, to preserve the exact prior floating-point computation path bit-for-bit (pure
    # refactor, zero behavior change) rather than a mathematically-equivalent but not bit-identical
    # alternative.
    #
    # CONFIRMED REGRESSION (real end-to-end practitioner workflow audit, 2026-08-17): same z-vs-t
    # issue as `significance_boundary` -- `se` is estimated from the same small sample, so the
    # Bonferroni-corrected critical value must come from a Student-t distribution (df=T-1), not a
    # normal one, for the same reason and with the same exact (non-approximate) justification.
    crit = tdist.ppf(1 - alpha / n_rivals, df=T - 1)
    bound = point + crit * se
    return bound / _scale(mean_champ)


def selection_regret(L, weights=None, *, w=None):
    """Leave-one-period-out (LOO) cost of the picking RULE, not of any single model: for each held-out
    period t, recompute the pooled champion from the remaining T-1 periods (under the SAME weights as
    the pooled estimate -- a skewed weight vector must actually change which fold-champion gets
    picked, not be silently dropped in the refit loop), then measure that fold-champion's realised
    regret at period t against whichever model was actually best AT that period. The weighted average
    over all T folds (weighted by each held-out period's own weight) is `selection_regret` -- an
    empirical, backward-looking answer to "what has this picking rule actually cost, historically",
    complementing MDE's forward-looking "what precision does my sample size buy me". No RNG anywhere
    in this statistic -- fully deterministic given the data.

    `w=` is accepted as an alias for `weights=` (FIXED 2026-09-10, round-6 stress-review,
    api_consistency lens): every sibling function in this module (minimum_detectable_edge,
    significance_boundary, mcb_bound, resolution_report) names this parameter `w`, so a caller who
    learned that convention from any one of them got a raw TypeError the first time they called
    this function the same way. `weights` is kept as the positional/primary name -- it shipped in
    the published v1.0.0 and a semver patch release must not break it -- `w` is keyword-only and
    purely additive."""
    if w is not None:
        if weights is not None:
            raise TypeError("selection_regret() got both 'weights' and 'w' -- pass only one; they "
                             "are the same parameter (w is an alias for backward-compatible weights).")
        weights = w
    L, weights = _unwrap_panel(L, weights)
    L = _as_loss_dict(L)
    T = _validate_losses(L, weights)
    w = np.ones(T) if weights is None else np.asarray(weights, float)
    if T < 3:
        raise ValueError(f"need at least 3 periods for leave-one-period-out selection_regret; got T={T}.")
    models = list(L)
    M = {m: np.asarray(L[m], float) for m in models}
    regrets = np.empty(T)
    for t in range(T):
        keep = np.arange(T) != t
        # FIXED 2026-09-09 (round-3 stress-review, tool_dormant_bugs lens): this used to pick the
        # fold champion via a raw, un-floored `min(sorted(models), key=...)` on the fold means --
        # the same class of bug already found and fixed in compare.py's _churn_base_rate (round 2):
        # on a near-tied fold, this can silently disagree with pooled_winner()'s documented
        # _MARGIN_REL_FLOOR degenerate-tie logic, which every OTHER champion pick in this package
        # (including the whole-panel champion selection_regret is itself benchmarked against
        # implicitly) goes through. Routed through pooled_winner() so a fold's champion always
        # agrees with what the rest of the package would call the champion on that same fold.
        fold_champ = pooled_winner({m: M[m][keep] for m in models}, w[keep])
        best_at_t = min(M[m][t] for m in models)
        regrets[t] = M[fold_champ][t] - best_at_t
    return float(np.average(regrets, weights=w))


def resolution_report(L, w=None, alpha=0.05, power=0.80, mcs_alpha=0.10):
    """R1/R2 refusal rules: a resolution-dependent read must refuse rather than print a
    confidently-looking number when the data cannot support it.

    R1 -- resolved: the observed edge against the binding rival is statistically significant at
    level `alpha` (one-sided) -- clears `significance_boundary`, NOT the (larger) `minimum_
    detectable_edge`. Below this, `resolved=False` and downstream resolution-dependent fields must
    be read as "cannot determine at this T", not as a small-but-precise-looking number. `mde` is
    still reported alongside `observed_edge` as honest planning context ("the smallest effect this
    sample size could reliably detect at `power`"), but is no longer the decision threshold itself
    -- see `significance_boundary`'s docstring for why comparing the observed edge against the full
    MDE only delivered ~50% power instead of the claimed `power` (FIXED 2026-08-17, independent
    'wild' review, statistical-calibration lens).

    R2 -- fragility_read: any fragility-adjacent read requires the champion to be point-identified
    (|MCS|==1) first; otherwise it is refused (`"undetermined"`) rather than answered on top of a
    decision that was never identified in the first place.

    Accepts a `LossPanel` directly, matching `report()`/`compare()` (FIXED 2026-08-26, independent
    applied-practitioner review): previously only a raw `{model: array}` dict or DataFrame was
    accepted, so passing the same `LossPanel` already built for `report()` raised a confusing
    internal `TypeError: object of type 'LossPanel' has no len()` from deep inside
    `_validate_losses` instead of working directly or failing with a clear top-level message.
    """
    from .identify import identified as _identified

    L, w = _unwrap_panel(L, w)
    L = _as_loss_dict(L)
    T = _validate_losses(L, w)
    ww = np.ones(T) if w is None else np.asarray(w, float)
    champ, rival, _diff, _se, mean_champ, mean_rival = _edge_components(L, ww)
    observed_edge = (mean_rival - mean_champ) / _scale(mean_champ)
    # mde/sig_boundary still go through the PUBLIC functions (not `_edge_components` inline) so this
    # dict's fields are always identical to what a caller gets calling minimum_detectable_edge()/
    # significance_boundary() directly on the same data -- a guaranteed consistency, at the cost of
    # recomputing champ/rival/SE a second time internally. T is small enough that this is free.
    mde = minimum_detectable_edge(L, ww, power=power, alpha=alpha)
    sig_boundary = significance_boundary(L, ww, alpha=alpha)
    resolved = bool(observed_edge >= sig_boundary)

    # w THREADED THROUGH, FIXED 2026-08-16 (independent 'wild' review): this used to call
    # `_identified(L, alpha=mcs_alpha)` with no weights, silently computing identification on the
    # UNWEIGHTED panel while `champ`/`observed_edge` above are computed WITH `ww` -- a mismatch that
    # let `report()` mark a model both "(champion)" and "excluded" from a supposedly-identified MCS
    # in the same breath when custom weights were used. `identified()` now refuses outright on a
    # genuinely non-uniform `ww` (see `identify._validate_mcs_weights`) rather than silently dropping
    # it; uniform weights (the common case) are unaffected.
    # ROUND-7 FIX (2026-08-27, clinical-trials realistic-data review): a genuinely non-uniform `ww`
    # makes `_identified()` refuse outright (see the comment above) -- report() already catches this
    # ValueError and falls back to an "undetermined" read (report.py, R2's fallback dict), but
    # resolution_report() itself had no try/except here, so calling it directly on the exact same
    # realistic weighted panel (e.g. patient-count-weighted multi-site registry data) crashed raw
    # and uncaught instead of degrading gracefully like report(). CORRECTED (round-8 fresh-eyes
    # review): this comment previously also claimed compare() already degraded gracefully the same
    # way -- untrue, and never actually tested when originally written; compare() had the identical
    # raw-crash gap on non-uniform weights, found and fixed separately in round 8 (compare.py).
    # Unlike report.py's external fallback (which has to NaN out resolved/observed_edge/mde too,
    # since it doesn't have them in scope), this can preserve the REAL already-computed values above
    # -- only is_identified/fragility_read actually depend on the failing MCS call.
    try:
        is_identified = bool(_identified(L, alpha=mcs_alpha, w=ww))
    except ValueError:
        is_identified = None
        fragility_read = "undetermined"
    else:
        if not is_identified:
            fragility_read = "undetermined"
        elif not resolved:
            fragility_read = "undetermined"
        else:
            fragility_read = "resolved"

    return {
        # ADDED 2026-09-10 (round-6 stress-review, api_consistency lens): `champ` was already
        # computed locally above (via _edge_components) but only `binding_rival` was ever placed in
        # this dict -- a caller reading resolution_report()'s output had to separately call
        # pooled_winner() to learn WHICH model the resolution/rival/edge fields are even about.
        # Purely additive (a new key), so this changes nothing for existing callers.
        "champion": champ,
        "resolved": resolved,
        "observed_edge": observed_edge,
        "mde": mde,
        # NEW 2026-08-17: the ACTUAL decision boundary `resolved` is computed against -- exposed
        # so the R1 decision is auditable rather than implicit. `mde` remains separate, unchanged
        # planning context; the two are equal only when power=0.5 (z_power=0).
        "significance_boundary": sig_boundary,
        "binding_rival": rival,
        "identified": is_identified,
        "fragility_read": fragility_read,
    }
