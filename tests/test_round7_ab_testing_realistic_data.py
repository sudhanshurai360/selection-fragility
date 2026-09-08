"""Round 7 -- A/B-testing / experimentation-platform persona, REALISTIC long-tenure data.

Round 6's A/B-testing review built one illustrative 30-day, 4-variant example and confirmed the core
math generalizes outside forecasting, but flagged that MCS becomes unavailable under realistic
traffic-weighted panels (arch has no native non-uniform-weight support). This round goes deeper: real,
multi-year-shaped synthetic data (T=110-140 weekly periods, not 30), varying volatility regimes, growing
traffic, and a precise characterization of what PIVOT's "responsible periods" diagnostic actually means
under a disruption -- which turned out to be more subtle than "flags anomalous weeks."

All losses are 1-conversion_rate (lower-is-better, matching this package's convention).
"""
import numpy as np
import pytest

from selection_fragility.panel import LossPanel
from selection_fragility.report import report
from selection_fragility.fragility import decision_breakdown, pooled_winner, fragility


def _long_running_program(seed=7, T=130):
    """~2.5 years of weekly A/B analysis, 4 variants, platform traffic growth + holiday-week
    volatility bumps (weeks 50-52 and 102-104, a two-holiday-season span)."""
    rng = np.random.default_rng(seed)
    weeks = np.arange(T)
    traffic = np.clip(1000 + 50 * weeks - 0.15 * weeks ** 2, 500, None)
    traffic = traffic / traffic.mean()
    holiday_mask = np.isin(weeks % 52, [50, 51, 0, 1])
    base_conv = {"control": 0.0420, "variant_b": 0.0415, "variant_c": 0.0455, "variant_d": 0.0440}
    losses = {}
    for name, base in base_conv.items():
        noise_sd = np.where(holiday_mask, 0.006, 0.0025)
        conv = base + rng.normal(0, 1, T) * noise_sd
        losses[name] = 1.0 - conv
    labels = [f"W{w + 1}" for w in weeks]
    return losses, traffic, labels


def test_long_running_program_uniform_weights_real_run():
    """A real ~2.5-year, 4-variant program: report() must actually run (not raise) and identify the
    true-best variant as champion with a real, non-trivial k* -- this is a genuine execution check,
    not a smoke test, so it asserts on the SPECIFIC winner and a real k* bound, not just 'ran fine'."""
    losses, traffic, labels = _long_running_program()
    panel = LossPanel.from_losses(losses, labels=labels)
    text = report(panel)
    assert "VERDICT: identified" in text
    assert "variant_c" in text.split("LEADERBOARD")[1].split("\n")[1]  # variant_c is the true best
    k, opp, removed = decision_breakdown(losses)
    T = len(labels)
    assert 1 <= k <= T  # genuine fragility read, not degenerate
    assert opp is not None


def test_long_running_program_traffic_weighted_mcs_unavailable_but_raw_tier_works():
    """Characterizes PRECISELY what is and isn't available to a real long-tenure, growing-traffic
    weighted A/B program today: MCS-dependent VERDICT/RESOLUTION correctly and honestly refuse
    (arch has no non-uniform-weight support), but the raw-array tier (decision_breakdown,
    pooled_winner, fragility()) -- which doesn't depend on the MCS bootstrap -- still computes a real,
    usable, weight-aware answer. A program that can't get MCS identification can still get a
    weighted pooled winner and a weighted k*."""
    losses, traffic, labels = _long_running_program()
    # weights are genuinely non-uniform (real traffic growth), not accidentally near-uniform
    assert traffic.std() / traffic.mean() > 0.15

    panel_w = LossPanel.from_losses(losses, weights=traffic, labels=labels)
    text = report(panel_w)
    assert "VERDICT: not available for this panel" in text
    assert "non-uniform period weights" in text

    # raw-array tier: genuinely runs, genuinely weight-aware (different from the unweighted answer)
    k_w, opp_w, removed_w = decision_breakdown(losses, w=traffic)
    k_uniform, _, _ = decision_breakdown(losses)
    assert isinstance(k_w, (int, np.integer)) and 0 <= k_w <= len(labels)
    assert k_w != k_uniform, (
        "weighted k* happened to exactly match the unweighted k* -- re-check the fixture uses "
        "genuinely informative (non-degenerate) weights"
    )
    winner_w = pooled_winner(losses, w=traffic)
    assert winner_w in losses

    frag_w = fragility(losses, w=traffic, labels=labels)
    assert frag_w["k_star"] == k_w
    assert frag_w["pooled_winner"] == winner_w


def test_low_volatility_mature_program_reports_high_confidence_not_spurious_uncertainty():
    """A mature, stable surface (T=110, low period-to-period noise, a real 1.5pp true edge --
    realistic for e.g. a checkout-flow test on a high-volume, low-variance property) must be
    reported as maximally robust: MCS size 1, and k* at or near T (almost every period would need
    to be deleted to flip the winner), not spurious/muted confidence."""
    rng = np.random.default_rng(11)
    T = 110
    losses = {}
    for name, b in {"control": 0.030, "new_layout": 0.045}.items():
        losses[name] = 1.0 - (b + rng.normal(0, 1, T) * 0.0015)
    labels = [f"W{i + 1}" for i in range(T)]
    panel = LossPanel.from_losses(losses, labels=labels)
    text = report(panel)
    assert "VERDICT: identified (MCS size 1 of 2)" in text
    assert "new_layout" in text.split("LEADERBOARD")[1].split("\n")[1]
    k, _, _ = decision_breakdown(losses)
    assert k / T >= 0.9, f"expected near-maximal robustness on a clean low-volatility edge, got k*/T={k / T:.2f}"


def test_disruption_period_pivot_semantics_precisely_characterized():
    """A real, non-obvious finding from round 7: PIVOT's 'responsible periods' identify what is
    PROPPING UP the current pooled winner's margin -- NOT 'the most anomalous/volatile periods' in
    general. Two disruption scenarios with IDENTICAL volatility, differing only in DIRECTION:

    (a) the disruption REINFORCES the champion's advantage (widens it) -> disruption weeks DO
        dominate the k*-removed ("responsible") set, since they're propping the margin up.
    (b) the disruption FIGHTS the champion (temporarily favors the challenger, a real inversion,
        e.g. a redesign that briefly regresses the new variant) -> disruption weeks are largely
        ABSENT from the removed set, because they're already reducing the champion's margin, so
        deleting them does not help flip the result toward the challenger -- they're irrelevant to
        "why does the champion currently win," even though a human reading the raw data would call
        that period the single most important event in the whole series.

    This is mathematically correct (greedy removal targets periods that most FAVOR the current
    winner) but is exactly the kind of thing a real analyst could misread PIVOT's silence on a real
    disruption as "nothing anomalous happened" -- worth a permanent regression lock since it governs
    how this tool's output should be described to a non-technical audience.
    """
    rng = np.random.default_rng(11)
    T = 140
    weeks = np.arange(T)
    shock = (weeks >= 60) & (weeks < 73)  # a real 13-week disruption block
    labels = [f"W{i + 1}" for i in range(T)]

    def _build(reinforcing):
        losses = {}
        for name, base0 in {"control": 0.040, "experiment": 0.043}.items():
            if reinforcing:
                shock_level = 0.040 if name == "control" else 0.063  # widens experiment's lead
            else:
                shock_level = 0.046 if name == "control" else 0.030  # inverts: control wins the shock
            level = np.where(shock, shock_level, base0)
            noise = np.where(shock, rng.normal(0, 1, T) * 0.014, rng.normal(0, 1, T) * 0.0025)
            losses[name] = 1.0 - (level + noise)
        return losses

    shock_labels = {labels[i] for i in range(T) if shock[i]}

    losses_reinforcing = _build(reinforcing=True)
    k_r, _, removed_r = decision_breakdown(losses_reinforcing)
    overlap_r = shock_labels & {labels[i] for i in removed_r}
    assert len(overlap_r) >= 9, (
        f"reinforcing disruption should dominate the responsible-periods set, got only "
        f"{len(overlap_r)}/13 shock weeks in a k*={k_r} removed-set"
    )

    losses_inverting = _build(reinforcing=False)
    k_i, _, removed_i = decision_breakdown(losses_inverting)
    overlap_i = shock_labels & {labels[i] for i in removed_i}
    assert len(overlap_i) <= 3, (
        f"an inverting disruption should be largely ABSENT from the responsible-periods set (it "
        f"already favors the challenger, not the champion), got {len(overlap_i)}/13 shock weeks in "
        f"a k*={k_i} removed-set -- if this now fires, the underlying mechanism characterized above "
        f"has changed and the documentation/framing implications need re-examining"
    )


def test_growing_traffic_weights_are_genuinely_informative_not_a_degenerate_fixture():
    """Sanity check on the long-running-program fixture itself: traffic must actually grow/vary
    meaningfully over the program's life (not be accidentally near-constant), so the weighted-vs-
    unweighted tests above are exercising a real weighting difference, not a no-op."""
    _, traffic, _ = _long_running_program()
    assert traffic.max() / traffic.min() > 2.0, "fixture traffic growth is too flat to be realistic"
    assert traffic[-1] > traffic[0] * 1.3, "fixture should show real platform growth over its life"
