"""Round-9 fraud-detection/cybersecurity domain deep-dive -- a genuinely new domain. Two
pathologies with no exact analog in any prior round:

1. Label-delay/revision: a transaction/event initially labeled "not fraud" can later be confirmed
   fraudulent (a chargeback, a delayed investigation outcome) -- structurally similar to round 8's
   epidemiology backfill-revision finding (data revision), but here it is the OUTCOME LABEL itself
   that is revised, not an intermediate observed count, and the real-world stakes are financial
   loss rather than public health.

2. Adversarial adaptation: sophisticated adversaries actively adapt their behavior specifically to
   evade whichever detector is currently winning -- a moving target where the CAUSE of a regime
   change is the model's own success, not an exogenous shock. Genuinely different from round 5's
   competition-integrity tests (gaming the STATISTICAL METHODOLOGY once known) and round 6's
   financial regime-change test (an exogenous shock, not adaptation triggered by the model itself).

Every scenario here is a permanent regression test with a fixed seed.
"""
import numpy as np
import pytest

from selection_fragility.fragility import pooled_winner, decision_breakdown
from selection_fragility.panel import LossPanel


# ---------------------------------------------------------------------------
# 1. Label-delay/revision: does the epidemiology backfill finding (round 8) transfer to fraud
#    detection's label-confirmation delay, or is there a domain-specific nuance?
# ---------------------------------------------------------------------------
# Fraction of fraud cases confirmed (via chargeback/investigation) by lag 0..8+ weeks after the
# transaction period -- realistic shape: most confirmed fraud is NOT yet known at evaluation time
# (chargebacks take 30-90+ days in real card-network practice), rising slowly, not fully settled
# even at 8 weeks (a real, harder-than-epi-surveillance tail, since some fraud is never formally
# confirmed at all within a reporting window).
CONFIRMATION_CURVE = np.array([0.10, 0.22, 0.38, 0.52, 0.64, 0.74, 0.81, 0.86, 0.90])


def _confirmed_as_of(true_final_rate, as_of_period, target_period):
    """Fraud rate visible for `target_period` if queried at `as_of_period` -- None if the future."""
    lag = as_of_period - target_period
    if lag < 0:
        return None
    frac = CONFIRMATION_CURVE[min(lag, len(CONFIRMATION_CURVE) - 1)]
    return true_final_rate[target_period] * frac


def _fraud_rate_curve(T, seed):
    """A realistic fraud-rate series: a low baseline plus periodic elevated-risk windows (e.g.
    holiday shopping surges, a known real pattern of seasonal fraud-rate spikes)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    baseline = 0.8 + 0.15 * np.sin(2 * np.pi * t / 26)
    surge = 1.2 * np.exp(-0.5 * ((t % 52 - 46) / 3) ** 2)  # a yearly holiday-season surge
    return np.clip(baseline + surge + rng.normal(0, 0.05, T), 0.1, None)


def test_premature_vs_settled_fraud_labels_can_flip_the_declared_winner():
    """The epidemiology finding (round 8) transfers here, with a domain-specific nuance: the
    confirmation delay is longer and the eventual-completeness ceiling is lower (0.90 vs epi's
    ~1.00), so the premature-evaluation trap is at least as severe, arguably worse, for fraud
    detection. `label_naive` scores against the raw as-confirmed rate at report time (systematically
    LOW, since most fraud isn't confirmed yet); `label_aware` inflates by the known confirmation
    curve (the correct real-world practice). Evaluated against fully-settled labels, label_aware
    wins; evaluated against a premature snapshot grabbed just 2 weeks after the target period (a
    real, documented analyst mistake -- reporting fraud-model performance before chargebacks have
    had time to land), label_naive spuriously wins instead."""
    T = 130
    true_final = _fraud_rate_curve(T, seed=31)

    def naive_forecast(wk):
        if wk == 0:
            return 0.8
        return _confirmed_as_of(true_final, wk - 1, wk - 1)

    def aware_forecast(wk):
        if wk == 0:
            return 0.8
        raw = _confirmed_as_of(true_final, wk - 1, wk - 1)
        return raw / CONFIRMATION_CURVE[0]

    fc_naive = np.array([naive_forecast(wk) for wk in range(T)])
    fc_aware = np.array([aware_forecast(wk) for wk in range(T)])

    settled_loss = {
        "label_naive": np.abs(fc_naive - true_final),
        "label_aware": np.abs(fc_aware - true_final),
    }
    premature_truth = np.array([_confirmed_as_of(true_final, wk + 2, wk) for wk in range(T)])
    premature_loss = {
        "label_naive": np.abs(fc_naive - premature_truth),
        "label_aware": np.abs(fc_aware - premature_truth),
    }

    winner_correct = pooled_winner(dict(settled_loss))
    winner_premature = pooled_winner(dict(premature_loss))

    assert winner_correct == "label_aware", (
        f"against properly-settled fraud labels, the confirmation-aware model should win; got {winner_correct}")
    assert winner_premature == "label_naive", (
        f"against premature (2-week) labels, the naive model should spuriously win; got {winner_premature}")
    assert winner_correct != winner_premature, (
        "cross-domain corroboration of the round-8 epidemiology finding: the premature-vs-settled "
        "ground-truth trap is not epi-specific -- it recurs in fraud detection's label-confirmation "
        "delay, with a longer lag and a lower completeness ceiling, making it if anything a sharper "
        "risk here than in the epi case this was first found in")


# ---------------------------------------------------------------------------
# 2. Adversarial adaptation: a moving-target dynamic where the model's own success causes the
#    regime change, not an exogenous shock. Does pooling the whole panel understate how quickly a
#    "winner" model's advantage erodes once adversaries adapt to it?
# ---------------------------------------------------------------------------
def test_success_triggered_adaptation_can_leave_pooled_winner_driven_by_stale_history():
    """model_A has a LARGE, genuine advantage for periods 0-39. Exactly because it was winning and
    got deployed as the primary detector, adversaries adapt specifically to evade it starting
    period 40 -- a realistic "success invites evasion" dynamic, distinct from an exogenous shock
    (round 6's financial regime-change test) because the CAUSE is the model's own prior success,
    not an external event. model_B, using a structurally different approach adversaries haven't
    adapted to, becomes relatively better post-adaptation, but by a SMALLER margin than A's original
    pre-adaptation edge -- a realistic pattern (an established detector's historical edge is often
    larger than a fresh challenger's early advantage). Empirically verified before asserting: under
    this construction, the pooled winner is still model_A, and decision_breakdown's k*
    responsible-periods concentrate ENTIRELY in the pre-adaptation window -- i.e. the "responsible"
    periods a pooled comparison points to are exactly the STALE, no-longer-representative history,
    not the current (already-reversed) reality. This is not a tool bug (the greedy algorithm is
    correctly identifying which periods have the largest raw margin, which is a well-defined,
    correct answer to "which periods, if removed, would flip the pooled winner") -- but it is a
    real, important risk for this domain: a fraud team trusting a pooled "still winning" verdict
    could be leaning on an advantage that has already eroded post-adaptation, and k*'s own removed-
    period list, read carelessly, would not obviously flag that the periods it names are all old."""
    T = 80
    rng = np.random.default_rng(41)
    loss_A = np.empty(T)
    loss_B = np.empty(T)
    # pre-adaptation: A's LARGE genuine advantage
    loss_A[:40] = np.abs(rng.normal(1.0, 0.2, 40))
    loss_B[:40] = np.abs(rng.normal(3.0, 0.3, 40))
    # post-adaptation: adversaries evade A; B becomes relatively better, but by a smaller margin
    loss_A[40:] = np.abs(rng.normal(2.0, 0.3, 40))
    loss_B[40:] = np.abs(rng.normal(1.5, 0.25, 40))

    L = {"model_A": loss_A, "model_B": loss_B}
    winner_pre = pooled_winner({"model_A": loss_A[:40], "model_B": loss_B[:40]})
    winner_post = pooled_winner({"model_A": loss_A[40:], "model_B": loss_B[40:]})
    winner_full = pooled_winner(dict(L))

    assert winner_pre == "model_A", "pre-adaptation, A's large genuine advantage should dominate"
    assert winner_post == "model_B", "post-adaptation, B should be the relatively better detector"
    assert winner_full == "model_A", (
        "the pooled winner should still be A -- its larger pre-adaptation edge outweighs B's "
        "smaller post-adaptation reversal in the raw pooled comparison")

    k, opp, removed = decision_breakdown(dict(L))
    frac_pre = sum(1 for i in removed if i < 40) / max(1, len(removed))

    assert frac_pre == 1.0, (
        f"pooled panel's k*={k} responsible periods should concentrate ENTIRELY in the "
        f"pre-adaptation (stale) window when the historical edge outweighs the recent reversal -- "
        f"got {frac_pre:.0%} in the pre-adaptation window. If this changes, the point being locked "
        f"in (a pooled 'winner' can be justified entirely by outdated, already-reversed history) "
        f"may no longer reproduce with these parameters and the scenario needs re-tuning, not the "
        f"assertion silently loosened")


def test_stable_threat_baseline_reads_normally_no_adaptation_dynamic():
    """A contrasting baseline: fraud patterns NOT adaptively evolving, a genuinely comparable
    detection environment throughout. The tool should read this as an ordinary stable comparison,
    with no artificial regime-driven fragility -- a control confirming the adaptation test above
    is detecting a real dynamic, not an artifact of the panel-construction method itself."""
    T = 80
    rng = np.random.default_rng(43)
    loss_stable_winner = np.abs(rng.normal(1.0, 0.25, T))
    loss_stable_loser = np.abs(rng.normal(2.0, 0.3, T))

    L = {"model_A": loss_stable_winner, "model_B": loss_stable_loser}
    winner = pooled_winner(dict(L))
    k, opp, removed = decision_breakdown(dict(L))

    assert winner == "model_A"
    # no reason for responsible periods to cluster in either half under a stable, non-adaptive DGP
    frac_second_half = sum(1 for i in removed if i >= 40) / max(1, len(removed))
    assert 0.15 <= frac_second_half <= 0.85, (
        f"under a stable (non-adaptive) threat environment, responsible periods should not "
        f"artificially concentrate in either half; got {frac_second_half:.0%} in the second half")


def test_full_toolkit_runs_end_to_end_on_realistic_fraud_panel_via_losspanel():
    """Sanity check the whole realistic panel (both regimes, real LossPanel construction with
    labels) runs cleanly through the public API end to end -- no crash on this domain's data shape."""
    T = 80
    rng = np.random.default_rng(41)
    loss_A = np.abs(rng.normal(1.0, 0.2, 40))
    loss_A = np.concatenate([loss_A, np.abs(rng.normal(3.0, 0.4, 40))])
    loss_B = np.abs(rng.normal(2.2, 0.3, 40))
    loss_B = np.concatenate([loss_B, np.abs(rng.normal(1.8, 0.3, 40))])
    labels = [f"period_{i:03d}" for i in range(T)]
    panel = LossPanel.from_losses({"model_A": loss_A, "model_B": loss_B}, labels=labels)
    k, opp, removed = decision_breakdown(panel)
    assert 0 <= k <= T
