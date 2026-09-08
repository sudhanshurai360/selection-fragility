"""Round 9 -- real-estate/AVM (automated valuation model) domain deep-dive.

Persona: a real-estate analytics researcher comparing candidate home-price forecasting/AVM
models across metro markets and valuation cycles (a realistic analog to the Zillow Prize-style
AVM comparison culture). Genuinely new domain for this project. All panels are synthetic but
built from real domain knowledge of AVM error behavior, not toy examples.

Baseline verified before writing these: `pytest tests/` -> 430 passed, 1 xfailed (2026-08-27,
pre-round-9). No code bugs found in this domain -- every scenario below behaved correctly and is
locked in as a permanent regression guard; the one real, field-specific finding is the mean-vs-
median aggregation choice test (#3), which is not a tool defect but a genuine methodological
decision an AVM team must make BEFORE the panel-loss framing this tool assumes is even valid.
"""
import numpy as np
import pytest

from selection_fragility.panel import LossPanel
from selection_fragility.report import report
from selection_fragility.fragility import pooled_winner, decision_breakdown


def _quarterly_labels(T, start_year=2018):
    return [f"{start_year + i // 4}Q{(i % 4) + 1}" for i in range(T)]


# ---------------------------------------------------------------------------
# 1. Stable, low-volatility market -- confirm no manufactured fragility.
# ---------------------------------------------------------------------------
def test_stable_suburb_market_identifies_confidently():
    rng = np.random.default_rng(42)
    T = 32
    models = {
        "hedonic_regression": np.abs(rng.normal(3.0, 0.6, T)),
        "gbm_avm": np.abs(rng.normal(2.2, 0.5, T)),
        "comp_based": np.abs(rng.normal(3.5, 0.7, T)),
        "neural_avm": np.abs(rng.normal(2.6, 0.6, T)),
    }
    panel = LossPanel.from_losses(models, labels=_quarterly_labels(T))
    out = report(panel)
    assert "VERDICT: identified" in out
    assert "gbm_avm" in out and "champion" in out
    k, opp, removed = decision_breakdown(models)
    assert k >= 8, f"stable market should not be maximally fragile, got k*={k}"


# ---------------------------------------------------------------------------
# 2. Boom-town market with a genuine bubble-and-correction cycle -- pivot
#    should meaningfully over-represent the correction window among the
#    periods responsible for the champion's margin (enrichment, not
#    necessarily exclusivity -- a broadly-superior model wins on more than
#    just the correction quarters too, which is itself correct behavior).
# ---------------------------------------------------------------------------
def test_boom_town_correction_periods_enriched_in_pivot():
    rng = np.random.default_rng(7)
    T = 32
    correction = set(range(20, 24))  # a ~1yr sharp correction after a run-up
    base = {
        "hedonic_regression": np.abs(rng.normal(3.0, 0.6, T)),
        "gbm_avm": np.abs(rng.normal(2.2, 0.5, T)),
        "comp_based": np.abs(rng.normal(3.5, 0.7, T)),
        "neural_avm": np.abs(rng.normal(2.6, 0.6, T)),
    }
    for i in correction:
        base["comp_based"][i] *= 6.0          # stale comps blow up during a correction
        base["hedonic_regression"][i] *= 3.0
        base["neural_avm"][i] *= 2.0
        base["gbm_avm"][i] *= 1.3              # champion degrades least

    k, opp, removed = decision_breakdown(base)
    frac_removed_in_correction = sum(1 for i in removed if i in correction) / len(removed)
    frac_correction_of_all = len(correction) / T
    assert frac_removed_in_correction > 2 * frac_correction_of_all, (
        f"correction-window periods should be enriched among responsible periods: "
        f"{frac_removed_in_correction:.2%} of removed vs {frac_correction_of_all:.2%} base rate"
    )


# ---------------------------------------------------------------------------
# 3. THE FIELD-SPECIFIC FINDING: real-estate valuation error has outlier
#    PROPERTIES (a mansion/unique architectural property, genuinely hard to
#    value, in an otherwise-normal quarter), not just outlier PERIODS like
#    every prior round's single-time-series framing assumed. This tool has
#    no native concept of "property" -- an AVM team MUST pre-aggregate
#    per-property errors into a per-period loss before this tool is usable
#    at all, and the aggregation choice (mean vs median) is not a detail:
#    it can flip the declared winner. Locked in with a constructed case
#    where it demonstrably does.
# ---------------------------------------------------------------------------
def test_mean_vs_median_aggregation_can_flip_winner_under_outlier_properties():
    rng = np.random.default_rng(23)
    T = 24
    N_PROPS = 200
    modelA_periods, modelB_periods = [], []
    for _ in range(T):
        a = np.abs(rng.normal(2.5, 0.5, N_PROPS))  # A: better on typical properties
        b = np.abs(rng.normal(2.7, 0.5, N_PROPS))  # B: worse on typical properties
        outlier_idx = rng.choice(N_PROPS, size=2, replace=False)
        a[outlier_idx] *= rng.uniform(40, 60, size=2)  # A: terrible on rare hard-to-value mansions
        b[outlier_idx] *= rng.uniform(2, 4, size=2)    # B: reasonable on the same properties
        modelA_periods.append(a)
        modelB_periods.append(b)

    mean_L = {
        "A": np.array([p.mean() for p in modelA_periods]),
        "B": np.array([p.mean() for p in modelB_periods]),
    }
    median_L = {
        "A": np.array([np.median(p) for p in modelA_periods]),
        "B": np.array([np.median(p) for p in modelB_periods]),
    }

    assert pooled_winner(mean_L) == "B", (
        "mean aggregation should crown B: A's mean is dragged up by rare catastrophic "
        "outlier-property errors even though A is typically better"
    )
    assert pooled_winner(median_L) == "A", (
        "median aggregation should crown A: robust to the 2-of-200 outlier properties, "
        "correctly reflects A's genuinely-better typical performance"
    )


def test_mean_vs_median_agree_when_no_outlier_properties_present():
    """Contrast case: when there ARE no per-property outliers (or the model ranking is
    consistent under both central tendencies), the aggregation choice should NOT matter --
    confirms the divergence above is specifically about outlier properties, not a general
    mean-vs-median instability in the tool."""
    rng = np.random.default_rng(11)
    T = 24
    N_PROPS = 200
    models_raw = {}
    for name, base_err in [("hedonic_regression", 3.0), ("gbm_avm", 2.2), ("comp_based", 3.5)]:
        per_period = []
        for _ in range(T):
            errs = np.abs(rng.normal(base_err, 0.8, N_PROPS))
            outlier_idx = rng.choice(N_PROPS, size=2, replace=False)
            errs[outlier_idx] *= rng.uniform(15, 30, size=2)
            per_period.append(errs)
        models_raw[name] = per_period

    mean_L = {m: np.array([p.mean() for p in per]) for m, per in models_raw.items()}
    median_L = {m: np.array([np.median(p) for p in per]) for m, per in models_raw.items()}
    assert pooled_winner(mean_L) == pooled_winner(median_L) == "gbm_avm", (
        "when the true ranking is consistent under both central tendencies, "
        "the aggregation choice should not flip the winner"
    )


# ---------------------------------------------------------------------------
# 4. Multi-market contrast: a stable market and a volatile market evaluated
#    with the same candidate models should NOT necessarily agree on
#    robustness -- confirms the tool discriminates per-market fragility
#    rather than reporting a fixed answer regardless of input.
# ---------------------------------------------------------------------------
def test_stable_and_volatile_markets_give_different_robustness_reads():
    rng = np.random.default_rng(99)
    T = 28
    stable = {
        "gbm_avm": np.abs(rng.normal(2.2, 0.3, T)),
        "hedonic_regression": np.abs(rng.normal(3.0, 0.35, T)),
    }
    volatile = {
        "gbm_avm": np.abs(rng.normal(2.2, 1.8, T)),
        "hedonic_regression": np.abs(rng.normal(3.0, 2.0, T)),
    }
    k_stable, _, _ = decision_breakdown(stable)
    k_volatile, _, _ = decision_breakdown(volatile)
    assert k_stable > k_volatile, (
        f"a low-volatility market (k*={k_stable}) should read as more robust than a "
        f"high-volatility market with the same mean gap (k*={k_volatile})"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
