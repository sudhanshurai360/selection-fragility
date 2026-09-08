"""Round-6 cross-domain review: A/B-testing (experimentation platform) persona.

Verifies the tool's core abstractions (models->variants, periods->daily cohorts, loss->1-conversion)
genuinely generalize outside forecasting, and locks in one real finding: weighted-MCS is unavailable
(honestly refused, not silently wrong) for the realistic-for-this-domain non-uniform-traffic-weight
case, plus a real, narrow pooled_winner weight-scale tie-break bug found by the existing hypothesis
suite while reproducing this scenario (see test_property_based.py::
test_decision_breakdown_weight_scale_invariance -- FAILED on this run, not previously known-broken).
"""
import numpy as np
import pytest

from selection_fragility.panel import LossPanel
from selection_fragility.report import report
from selection_fragility.fragility import decision_breakdown, pooled_winner


def _ab_experiment(seed=42):
    """4-variant A/B experiment, 30 daily cohorts, conversion-rate loss, day-of-week seasonality,
    unequal daily traffic per arm (realistic: a full-size control/treatment pair plus a smaller
    holdout pair) -- the shape a real experimentation platform's data actually has."""
    rng = np.random.default_rng(seed)
    days = np.arange(30)
    weekend = np.isin(days % 7, [5, 6])
    base_conv = 0.12 - 0.02 * weekend
    true_lift = {"A_control": 0.0, "B_treatment": 0.015, "C_treatment": 0.003, "D_treatment": -0.005}
    daily_users = {
        "A_control": rng.integers(800, 1200, 30),
        "B_treatment": rng.integers(800, 1200, 30),
        "C_treatment": rng.integers(300, 500, 30),
        "D_treatment": rng.integers(300, 500, 30),
    }
    losses = {}
    for arm, lift in true_lift.items():
        n = daily_users[arm]
        p = np.clip(base_conv + lift, 0.001, 0.5)
        conversions = rng.binomial(n, p)
        losses[arm] = 1.0 - conversions / n
    return losses, daily_users


def test_ab_experiment_workflow_end_to_end_with_uniform_weights():
    """The core toolkit genuinely generalizes to A/B testing when periods are equally weighted
    (treat each day as one unit regardless of traffic volume): the correct, higher-true-conversion
    variant (B_treatment, +1.5pp true lift) is both the pooled winner and the sole MCS survivor."""
    losses, _ = _ab_experiment()
    panel = LossPanel.from_losses(losses, labels=[f"day{d}" for d in range(30)])
    assert pooled_winner(losses) == "B_treatment"
    out = report(panel)
    assert "VERDICT: identified (MCS size 1 of 4)" in out
    assert "B_treatment" in out and "(champion)" in out


def test_ab_experiment_with_realistic_unequal_traffic_weights_refuses_mcs_honestly():
    """The realistic-for-this-domain case -- daily period weights proportional to actual traffic,
    which varies day to day for every real experimentation platform -- makes MCS UNAVAILABLE, not
    silently wrong. This is a genuine, previously-undocumented limitation for the A/B-testing
    persona specifically: volume-weighted panels are the norm there (unlike much of forecasting,
    where equal-period evaluation is more common), so this refusal would fire on most realistic A/B
    analyses run through this tool as-is. Locking in the HONEST-REFUSAL behavior (not a crash, not a
    silent wrong answer) as the documented current behavior, not proposing a fix."""
    losses, daily_users = _ab_experiment()
    w = sum(daily_users.values()).astype(float)
    panel = LossPanel.from_losses(losses, weights=w, labels=[f"day{d}" for d in range(30)])
    out = report(panel)
    assert "VERDICT: not available for this panel" in out
    assert "non-uniform period weights" in out
    # decision_breakdown (pooled-winner-based, no bootstrap) still works under real weights --
    # only the MCS/bootstrap layer is the part that refuses.
    k, opponent, _ = decision_breakdown(losses, w=w)
    assert isinstance(k, int) and 0 <= k <= 30


def test_pooled_winner_weight_scale_tie_break_not_invariant_KNOWN_BUG():
    """FOUND round 6, FIXED same day (2026-08-27): pooled_winner's tie-break was NOT invariant to
    weight-vector rescaling when two models are in an EXACT weighted-average tie, because
    `np.average(x, weights=w)` computes sum(w*x)/sum(w) and the two terms' floating-point rounding
    did not cancel identically at every scale. Minimal repro: m0=[0,0,2,2.125], m1=[0,0,0,4.125]
    (both average to exactly 1.03125 under uniform weights). At w=ones(4)*1, 1e-3, 1e3, 1e6 the tie
    resolved to 'm0' (sorted-name order, as documented); at w=ones(4)*1e-6 specifically it flipped
    to 'm1', because np.average(m0, weights=w*1e-6) evaluated to 1.0312500000000002 (not exactly
    1.03125) while np.average(m1, ...) still landed on exactly 1.03125 at that scale. This was
    DIFFERENT from the already-fixed decision_breakdown/breakdown_number weight-scale bug (round
    4) -- that fix made the k* VALUE scale-invariant; this was a separate scale-sensitivity in
    pooled_winner's OWN tie-break, one function earlier in the pipeline. Fixed by treating any mean
    within a relative floor of the true minimum as a tie (see fragility.py::pooled_winner)."""
    m0 = np.array([0., 0., 2., 2.125])
    m1 = np.array([0., 0., 0., 4.125])
    L = {"m0": m0, "m1": m1}
    winners = {c: pooled_winner(L, w=np.ones(4) * c) for c in (1.0, 1e-3, 1e3, 1e6, 1e-6)}
    assert len(set(winners.values())) == 1, (
        f"pooled_winner's tie-break is scale-dependent: {winners} "
        f"(np.average rounding differs by weight scale on an exact tie)"
    )
