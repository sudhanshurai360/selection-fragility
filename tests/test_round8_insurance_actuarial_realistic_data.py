"""Round 8 (2026-08-27) -- actuarial/insurance-analytics deep-dive with REALISTIC severity data.

A genuinely new domain, not tested in any prior round. Insurance claim SEVERITY forecasting error
is heavy-tailed in a way that's more extreme than round 6's financial GARCH test: a small number of
catastrophic claims 10-100x+ the typical claim is normal, expected, well-understood behavior in
this field, not an anomaly. The actuarial question this file characterizes precisely: how easily
can ONE catastrophic claim, in ONE period, flip which model is declared the pooled winner -- and
does the tool's own k*/PIVOT diagnostic correctly flag that fragility, or does it silently mislead?

Finding (see test_single_catastrophic_claim_flips_pooled_winner_and_crashes_kstar below): a single
claim just ~10.5x the typical claim size, in 1 of 24 quarters, is enough to flip the reported
winner AND crash k* from 14 (robust) to 1 (maximally fragile). This is mathematically correct
behavior (an unweighted arithmetic-mean comparison is inherently sensitive to one huge value), not
a bug -- and importantly, the tool's own k*/PIVOT/concentration_share diagnostics correctly and
precisely name the exact catastrophic period as responsible. The real risk is an actuary who reads
only "the winner is X" without checking k*/RESOLUTION, since the underlying LEADERBOARD looks
identical to a genuinely robust win. Recommended (not implemented here, out of this file's
find/test-only scope): README could use an explicit note for actuarial-style extreme-severity use
cases, in the spirit of the existing loss/accuracy sign-confusion warning, pointing readers at k*
specifically for this reason.
"""
import numpy as np
import pytest

from selection_fragility.panel import LossPanel
from selection_fragility.report import report
from selection_fragility.fragility import decision_breakdown, pooled_winner
from selection_fragility.pivot import concentration_share
from selection_fragility.mcs import model_confidence_set


# ---------------------------------------------------------------------------------------------
# P1: a single catastrophic claim, well within realistic actuarial severity-error magnitude, can
# flip the pooled winner and crash k* from robust to maximally fragile. Precisely characterized,
# not just observed: locks in the exact before/after and the flip threshold's order of magnitude.
# ---------------------------------------------------------------------------------------------
def test_single_catastrophic_claim_flips_pooled_winner_and_crashes_kstar():
    rng = np.random.default_rng(7)
    T = 24
    a = rng.lognormal(mean=np.log(8_000), sigma=0.4, size=T)
    b = rng.lognormal(mean=np.log(11_000), sigma=0.3, size=T)

    # Without any catastrophe: A is the genuinely, robustly better model.
    assert pooled_winner({"A": a.copy(), "B": b}) == "A"
    k_before, opp_before, _ = decision_breakdown({"A": a.copy(), "B": b})
    assert k_before >= 10, f"expected a robust win (k* well above trivial), got k*={k_before}"

    # Inject ONE catastrophic claim at period 5, ~25x the typical claim -- realistic for a real CAT
    # loss landing in an otherwise-ordinary quarter.
    a_cat = a.copy()
    a_cat[5] = 200_000  # typical ~8,000; this is ~25x
    assert pooled_winner({"A": a_cat, "B": b}) == "B", (
        "a single realistic-scale catastrophic claim did not flip the pooled winner -- "
        "either the scenario needs adjusting or pooled_winner's sensitivity changed"
    )
    k_after, opp_after, removed_after = decision_breakdown({"A": a_cat, "B": b})
    assert k_after == 1, f"expected k*=1 (maximally fragile) after the catastrophe, got {k_after}"
    assert removed_after == [5], (
        f"PIVOT should name exactly period 5 (the injected catastrophe) as responsible, got {removed_after}"
    )

    # concentration_share should correctly attribute the large majority of the margin to that one period.
    cs = concentration_share({"A": a_cat, "B": b})
    assert cs > 0.5, f"expected the single catastrophic period to dominate the margin, concentration={cs}"


def test_catastrophic_claim_flip_threshold_is_a_realistic_order_of_magnitude():
    """The flip point itself, not just one worked example -- locks in that it takes roughly an
    order of magnitude above the typical claim, not something wildly larger, so a reader can
    calibrate how easily this class of fragility can occur in practice."""
    rng = np.random.default_rng(7)
    T = 24
    a = rng.lognormal(mean=np.log(8_000), sigma=0.4, size=T)
    b = rng.lognormal(mean=np.log(11_000), sigma=0.3, size=T)
    typical = np.median(a)

    lo, hi = 0.0, 500_000.0
    for _ in range(40):
        mid = (lo + hi) / 2
        a2 = a.copy()
        a2[5] = mid
        if pooled_winner({"A": a2, "B": b}) == "A":
            lo = mid
        else:
            hi = mid
    flip_multiple = hi / typical
    assert 5 < flip_multiple < 20, (
        f"flip threshold was {flip_multiple:.1f}x the typical claim -- expected roughly one order "
        "of magnitude, characterizing how easily a single real catastrophic claim can dominate"
    )


# ---------------------------------------------------------------------------------------------
# P2: real actuarial dynamic range (thousands to hundreds of millions within the SAME panel) is
# handled with full precision -- no overflow-guard false trigger (well under _MAX_ABS_LOSS=1e100),
# no loss of correctness in pooled_winner's ranking.
# ---------------------------------------------------------------------------------------------
def test_real_actuarial_dynamic_range_no_precision_loss():
    rng = np.random.default_rng(3)
    T = 32
    models = ["catastrophe_model", "attritional_model", "blended_model"]
    L = {}
    for i, m in enumerate(models):
        typical = rng.lognormal(mean=np.log(5_000 * (1 + 0.1 * i)), sigma=0.5, size=T)
        cat_idx = rng.choice(T, size=3, replace=False)
        typical[cat_idx] = rng.uniform(5e7, 5e8, size=3) * (1 + 0.05 * i)
        L[m] = typical

    # A real actuarial range: min claim ~$1-3k, max claim ~$200M-$400M within a single model.
    for m in L:
        assert L[m].max() / L[m].min() > 1e4, "fixture should exercise a genuinely wide dynamic range"
        assert L[m].max() < 1e9, "keep this in real CAT-loss territory, not an artificial overflow probe"

    means = {m: float(np.mean(L[m])) for m in L}
    assert pooled_winner(L) == min(means, key=means.get), (
        "pooled_winner disagreed with the true arithmetic-mean ranking at real actuarial scale -- "
        "possible precision loss"
    )

    # Full pipeline should run cleanly end to end at this scale, no crash, no NaN/inf.
    panel = LossPanel.from_losses(L)
    out = report(panel)
    assert "VERDICT" in out and "nan" not in out.lower() and "inf" not in out.lower()

    surv, p = model_confidence_set(L, B=300, block=3, seed=0)
    assert 1 <= len(surv) <= len(models)
    assert np.isfinite(p)


# ---------------------------------------------------------------------------------------------
# P3: claim FREQUENCY (count-based) forecasting is genuinely more stable than severity -- a
# contrasting low-volatility scenario where a real, decisive edge should read as robustly
# identified, not fragile. Confirms the tool discriminates real robustness, doesn't just always
# hedge.
# ---------------------------------------------------------------------------------------------
def test_claim_frequency_forecasting_low_volatility_reads_as_robust():
    rng = np.random.default_rng(11)
    T = 36  # 3 years, monthly
    true_rate = 500
    L = {
        "glm_poisson": np.abs(rng.poisson(true_rate, T) - rng.poisson(true_rate * 0.99, T)).astype(float) + 1,
        "naive_seasonal": np.abs(rng.poisson(true_rate, T) - rng.poisson(true_rate * 1.30, T)).astype(float) + 1,
        "arima_count": np.abs(rng.poisson(true_rate, T) - rng.poisson(true_rate * 1.18, T)).astype(float) + 1,
    }
    panel = LossPanel.from_losses(L)
    out = report(panel)
    assert "VERDICT: identified" in out, "a genuinely decisive frequency-forecast edge should be identified"

    k, opp, removed = decision_breakdown(L)
    assert k / T > 0.5, (
        f"expected a robust win (k*/T > 0.5) on stable, Poisson-like frequency data, got k*={k} of {T}"
    )


# ---------------------------------------------------------------------------------------------
# P4: sanity check that the severity scenario (P1) and the frequency scenario (P3) are genuinely
# contrasting, not both incidentally landing in the same fragility regime -- the actuarial point
# this whole file makes only holds if severity is fragile AND frequency is robust.
# ---------------------------------------------------------------------------------------------
def test_severity_and_frequency_scenarios_are_genuinely_contrasting():
    rng = np.random.default_rng(7)
    T = 24
    a = rng.lognormal(mean=np.log(8_000), sigma=0.4, size=T)
    a[5] = 200_000
    b = rng.lognormal(mean=np.log(11_000), sigma=0.3, size=T)
    k_severity, _, _ = decision_breakdown({"A": a, "B": b})

    rng2 = np.random.default_rng(11)
    T2 = 36
    true_rate = 500
    L = {
        "glm_poisson": np.abs(rng2.poisson(true_rate, T2) - rng2.poisson(true_rate * 0.99, T2)).astype(float) + 1,
        "naive_seasonal": np.abs(rng2.poisson(true_rate, T2) - rng2.poisson(true_rate * 1.30, T2)).astype(float) + 1,
        "arima_count": np.abs(rng2.poisson(true_rate, T2) - rng2.poisson(true_rate * 1.18, T2)).astype(float) + 1,
    }
    k_frequency, _, _ = decision_breakdown(L)

    assert k_severity / T < 0.1, "severity scenario should read as maximally fragile"
    assert k_frequency / T2 > 0.5, "frequency scenario should read as robust"
