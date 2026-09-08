"""Round-8 epidemiology/disease-surveillance domain deep-dive -- a genuinely new domain, not
covered by any prior round. Public-health surveillance forecasting (flu/COVID-style case
forecasting, CDC FluSight/COVID Forecast Hub-style model comparison) has one methodological
challenge with no analog in any domain tested so far: reporting-delay / backfill revision, where
a case count "reported" for a given week is heavily revised upward over subsequent weeks as
delayed reports arrive. This means the SAME historical period can have different "true" loss
values depending on WHEN, relative to real time, you compute the evaluation.

Every scenario here is a permanent regression test with a fixed seed -- each asserts something
specific about correct behavior on realistic epi-surveillance data shapes, not just "doesn't crash".
"""
import numpy as np
import pytest

from selection_fragility.fragility import pooled_winner, decision_breakdown
from selection_fragility.mcs import model_confidence_set
from selection_fragility.panel import LossPanel


# ---------------------------------------------------------------------------
# Shared realistic backfill machinery
# ---------------------------------------------------------------------------
# Fraction of the eventual final case count visible at lag 0..6+ weeks after the target week --
# a realistic backfill/reporting-completeness curve for weekly disease surveillance data (loosely
# modeled on the shape reported for real syndromic/case surveillance systems: roughly half-complete
# at report time, ~85% after 2 weeks, effectively final by 6 weeks).
BACKFILL_CURVE = np.array([0.55, 0.72, 0.85, 0.92, 0.96, 0.98, 1.00])


def _as_reported(true_final, as_of_week, target_week):
    """What the surveillance system would show for `target_week`'s count, if queried at
    `as_of_week` -- None if the target week hasn't happened yet as of that query."""
    lag = as_of_week - target_week
    if lag < 0:
        return None
    frac = BACKFILL_CURVE[min(lag, len(BACKFILL_CURVE) - 1)]
    return true_final[target_week] * frac


def _epi_curve(T, seed):
    """A realistic two-wave seasonal epidemic curve (e.g. two flu-season-shaped waves over 3
    years) plus a low, slowly-varying endemic baseline -- not modeling a specific real disease,
    but the real SHAPE (baseline + sharp Gaussian-ish growth/peak/decline waves) that any real
    surveillance forecaster has to handle."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    baseline = 60 + 15 * np.sin(2 * np.pi * t / 52)
    wave1 = 350 * np.exp(-0.5 * ((t - 40) / 6) ** 2)
    wave2 = 500 * np.exp(-0.5 * ((t - 110) / 5) ** 2)
    return np.clip(baseline + wave1 + wave2 + rng.normal(0, 8, T), 5, None)


# ---------------------------------------------------------------------------
# The most important finding of this round: evaluating against the WRONG ground-truth basis
# (real-time-premature data instead of properly-settled data) can flip which model is declared
# the winner. This is not a tool bug -- the tool has no way to know which basis is "correct" for
# your domain, it just scores whatever losses you feed it -- but it is a real, high-stakes
# methodological trap specific to any domain with reporting-delay/backfill (epi surveillance,
# also relevant to e.g. revised economic statistics), and worth a permanent, explicit regression
# lock so this project's own understanding of the risk doesn't silently erode.
# ---------------------------------------------------------------------------
def test_premature_vs_settled_ground_truth_can_flip_the_declared_winner():
    """Two models forecasting weekly case counts under realistic backfill: `backfill_naive` uses
    the raw as-reported count for the most recent known week (systematically LOW during growth
    phases, since recent weeks are always undercounted at report time); `backfill_aware` inflates
    the raw count by the known backfill factor (the methodologically correct real-world practice,
    often called 'nowcasting'). Evaluated against the fully-settled truth, the backfill-aware
    model is clearly better (it is, after all, correcting for a real, known bias). Evaluated
    against a PREMATURE ground truth (grabbing the still-incomplete database value 1 week after
    the target week, instead of waiting for full settlement -- a real, documented analyst mistake
    in this field), the naive model wins instead, because the premature truth is itself
    undercounted in the same direction the naive model is. The tool correctly reports a different
    winner under the two loss bases (it has no way to detect which is right) -- an epi analyst
    MUST evaluate against a properly-settled ground truth, never a same-day or near-term database
    snapshot, or their model comparison is silently measuring the wrong thing."""
    T = 156
    true_final = _epi_curve(T, seed=11)

    def backfill_naive_forecast(wk):
        if wk == 0:
            return 50.0
        return _as_reported(true_final, wk - 1, wk - 1)

    def backfill_aware_forecast(wk):
        if wk == 0:
            return 50.0
        raw = _as_reported(true_final, wk - 1, wk - 1)
        return raw / BACKFILL_CURVE[0]

    fc_naive = np.array([backfill_naive_forecast(wk) for wk in range(T)])
    fc_aware = np.array([backfill_aware_forecast(wk) for wk in range(T)])

    settled_loss = {
        "backfill_naive": np.abs(fc_naive - true_final),
        "backfill_aware": np.abs(fc_aware - true_final),
    }
    premature_truth = np.array([_as_reported(true_final, wk + 1, wk) for wk in range(T)])
    premature_loss = {
        "backfill_naive": np.abs(fc_naive - premature_truth),
        "backfill_aware": np.abs(fc_aware - premature_truth),
    }

    winner_correct = pooled_winner(dict(settled_loss))
    winner_wrong = pooled_winner(dict(premature_loss))

    assert winner_correct == "backfill_aware", (
        f"against properly-settled truth, the methodologically correct model should win; got {winner_correct}")
    assert winner_wrong == "backfill_naive", (
        f"against a premature ground truth, the naive model should spuriously win; got {winner_wrong}")
    assert winner_correct != winner_wrong, (
        "ground-truth timing basis should be able to flip the declared winner -- if this stops "
        "reproducing, the synthetic scenario's backfill asymmetry may have been weakened, not that "
        "the underlying methodological risk has gone away")


# ---------------------------------------------------------------------------
# Peak-attribution: does the pivot/responsible-periods diagnostic correctly identify epidemic-wave
# peak weeks as the driver of a real, peak-concentrated performance gap?
# ---------------------------------------------------------------------------
def test_epidemic_peak_weeks_correctly_identified_as_responsible_periods():
    """robust_A performs identically to peak_blind_B off-peak, but peak_blind_B's error blows up
    specifically during the two epidemic peak windows (a realistic failure mode: a method that
    assumes smooth local trends breaks exactly when case counts are growing/declining fastest).
    decision_breakdown's responsible-periods diagnostic must attribute robust_A's win entirely to
    the peak windows, the same way this project's own regime-robustness analysis attributes its
    real forecasting-model results to the pandemic period."""
    T = 156
    peak_idx = set(range(36, 46)) | set(range(104, 116))
    rng = np.random.default_rng(21)
    loss_A = np.abs(rng.normal(2.0, 0.4, T))
    loss_B = np.abs(rng.normal(2.0, 0.4, T))
    for i in peak_idx:
        loss_B[i] += rng.normal(15.0, 2.0)

    L = {"robust_A": loss_A, "peak_blind_B": loss_B}
    winner = pooled_winner(dict(L))
    k, opp, removed = decision_breakdown(dict(L))

    assert winner == "robust_A"
    assert k > 0
    n_in_peaks = sum(1 for i in removed if i in peak_idx)
    assert n_in_peaks / len(removed) >= 0.90, (
        f"expected the responsible periods to be overwhelmingly peak weeks, got "
        f"{n_in_peaks}/{len(removed)}")


# ---------------------------------------------------------------------------
# Low-volatility endemic baseline: must not manufacture spurious fragility when the data
# genuinely supports high confidence.
# ---------------------------------------------------------------------------
def test_stable_endemic_period_reports_high_confidence_not_spurious_fragility():
    """A stable, low-transmission (endemic) 2-year period with a consistently-better model and
    low week-to-week noise -- the tool must correctly report near-maximal robustness, not
    manufacture fragility just because the data is disease-surveillance-shaped."""
    rng = np.random.default_rng(31)
    T = 104
    L = {
        "good": np.abs(rng.normal(2.0, 0.3, T)),
        "worse1": np.abs(rng.normal(3.5, 0.3, T)),
        "worse2": np.abs(rng.normal(4.0, 0.3, T)),
    }
    winner = pooled_winner(dict(L))
    k, opp, removed = decision_breakdown(dict(L))
    surv, p = model_confidence_set(dict(L), B=1000, seed=0)

    assert winner == "good"
    assert k / T >= 0.85, f"expected near-maximal robustness (k*/T>=0.85), got k*={k}/{T}"
    assert sorted(surv) == ["good"]


# ---------------------------------------------------------------------------
# Forecast-hub-style small-K weekly comparison (the real CDC FluSight/COVID Forecast Hub shape:
# a handful of independently-submitted models compared weekly).
# ---------------------------------------------------------------------------
def test_forecast_hub_style_six_team_weekly_panel_gives_sensible_mcs():
    """K=6 independently-submitted models (a realistic FluSight/Forecast-Hub-style roster size),
    T=52 weekly evaluations, with a real but modest performance gradient across teams -- the MCS
    must not spuriously collapse to a singleton when several teams are close, nor to the full set
    when there's a real gap; the pooled winner must be the genuinely best-performing team."""
    rng = np.random.default_rng(5)
    T = 52
    L = {f"team_{i}": np.abs(rng.normal(2.0 + 0.15 * i, 0.8, T)) for i in range(6)}
    surv, p = model_confidence_set(dict(L), B=1000, seed=0)
    winner = pooled_winner(dict(L))

    assert winner == "team_0"
    assert 1 <= len(surv) < 6, f"expected a genuine partial MCS (some but not all survive), got {sorted(surv)}"
    assert "team_0" in surv
