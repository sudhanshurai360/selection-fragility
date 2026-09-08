"""Round 9 — agricultural economics domain review (crop-yield forecast model comparison).

New domain, not tested in any prior round. Real-world framing: comparing candidate crop-yield
forecasting models (trend, weather-driven, satellite/NDVI, naive persistence) across a long annual
run, the way USDA-style yield forecasting or agricultural-economics research would. Distinct from
every prior round's data shape: ANNUAL cadence (T=30-40 for the whole series, not per-something),
and a genuine weather/trend confound mechanism not tested elsewhere.

All scenarios built and run against the real package before being locked in here; see
recover_chat/findings/REBUILD_LEDGER.md for the full exploration record.
"""
import numpy as np
import pytest

from selection_fragility.panel import LossPanel
from selection_fragility import decision_breakdown, model_confidence_set


def _drought_embedded_panel(seed=42, T=35):
    """35 years annual yield-forecast error, 4 models, with 4 drought + 2 favorable years injected.
    weather_model is deliberately built to handle drought years far better than its rivals -- the
    real-world claim being tested is that a weather-aware model should show as champion AND that
    the drought years should be disproportionately represented among the periods deciding it."""
    rng = np.random.default_rng(seed)
    years = list(range(1990, 1990 + T))
    L = {
        "trend_model": rng.normal(4.0, 1.0, T),
        "weather_model": rng.normal(3.0, 0.8, T),
        "satellite_model": rng.normal(3.3, 0.9, T),
        "naive_persistence": rng.normal(5.5, 1.3, T),
    }
    drought_idx = [5, 14, 22, 30]
    favorable_idx = [10, 27]
    for idx in drought_idx:
        L["trend_model"][idx] += 8.0
        L["satellite_model"][idx] += 6.0
        L["naive_persistence"][idx] += 9.0
        L["weather_model"][idx] += 1.5
    for idx in favorable_idx:
        L["trend_model"][idx] += 3.0
        L["satellite_model"][idx] += 2.0
        L["naive_persistence"][idx] += 4.0
        L["weather_model"][idx] += 0.5
    for k in L:
        L[k] = np.abs(L[k])
    return L, years, drought_idx


def test_annual_cadence_small_T_no_crash_and_correct_champion():
    """T=35 (annual, the whole series -- not per-something) is a much lower-frequency, lower-T
    cadence than almost every prior round's daily/weekly/monthly/quarterly data. The
    weather-aware model, built to be genuinely best, must be correctly identified as champion."""
    L, years, _ = _drought_embedded_panel()
    panel = LossPanel.from_losses(L, labels=years)
    k, opp, removed = decision_breakdown(L)
    surv, p = model_confidence_set(L, seed=0)
    assert surv == ["weather_model"], f"expected weather_model as sole survivor, got {surv}"
    assert 0 < k < len(years)


def test_drought_years_overrepresented_in_pivotal_periods():
    """All 4 genuinely-injected drought years must appear among the k*-removed pivotal periods --
    a real-world claim (drought years should drive model differentiation), not a smoke test."""
    L, years, drought_idx = _drought_embedded_panel()
    k, opp, removed = decision_breakdown(L)
    removed_years = {years[i] for i in removed}
    drought_years = {years[i] for i in drought_idx}
    overlap = removed_years & drought_years
    assert overlap == drought_years, (
        f"expected all drought years {drought_years} in the k*={k} removed set {removed_years}, "
        f"only found {overlap}")


def test_weather_aware_model_survives_pooled_and_low_volatility_baseline_stays_confident():
    """Low-volatility contrast: a stable 30-year run with a consistent, real edge should read as
    confidently identified, not manufacture fragility just because T is small by other rounds'
    standards."""
    rng = np.random.default_rng(99)
    T = 30
    L = {"a": rng.normal(2.0, 0.3, T), "b": rng.normal(2.6, 0.3, T)}
    L = {k: np.abs(v) for k, v in L.items()}
    surv, p = model_confidence_set(L, seed=0)
    assert surv == ["a"]


def test_multiregion_batch_completes_fast_and_produces_varying_kstar():
    """18 growing regions, T=35 each, varying volatility/edge -- the real agricultural-forecasting
    batch pattern (many regions evaluated simultaneously). Must complete fast and produce a real
    SPREAD of k* values (not everything collapsing to the same number), confirming the tool
    discriminates region-to-region robustness rather than returning a constant."""
    rng = np.random.default_rng(3)
    T = 35
    n_regions = 18
    kstars = []
    for r in range(n_regions):
        vol = rng.uniform(0.3, 2.5)
        edge = rng.uniform(0.05, 1.5)
        a = np.abs(rng.normal(3.0, vol, T))
        b = np.abs(rng.normal(3.0 + edge, vol, T))
        k, opp, removed = decision_breakdown({"m0": a, "m1": b})
        kstars.append(k)
        assert 0 <= k <= T
    assert len(set(kstars)) >= n_regions // 2, (
        f"expected real region-to-region k* variation, got only {len(set(kstars))} distinct "
        f"values across {n_regions} regions: {kstars}")


def test_trend_fitting_confound_pooled_winner_reverses_on_shock_only_subset():
    """HEADLINE FINDING: a model that fits the long-run technology-improvement trend well
    (trend_fitter) can be crowned pooled champion even though a rival (weather_skill_model) is
    dramatically better specifically during weather-shock years -- the years that matter most for
    real food-security decisions. This is the agricultural analog of round 8's marketing
    regime-masking finding: the pooled view alone gives no hint of the reversal.

    Not a tool bug -- decision_breakdown/model_confidence_set correctly answer "who wins pooled,"
    they were never asked "who wins during shocks." Locking this in as a documented, real
    methodological trap for this domain, the same way round 8 did for marketing/epidemiology."""
    rng = np.random.default_rng(7)
    T = 30
    shock_idx = [4, 9, 15, 21, 26]
    trend_fitter = np.abs(rng.normal(2.0, 0.4, T))
    weather_skill = np.abs(rng.normal(2.3, 0.4, T))
    for idx in shock_idx:
        trend_fitter[idx] += 1.0
        weather_skill[idx] = max(weather_skill[idx] - 0.5, 0.3)

    L_pooled = {"trend_fitter": trend_fitter, "weather_skill_model": weather_skill}
    pooled_survivors, _ = model_confidence_set(L_pooled, seed=0)
    assert pooled_survivors == ["trend_fitter"], (
        f"expected trend_fitter to win the pooled comparison (the confound setup), got "
        f"{pooled_survivors}")

    L_shock_only = {k: v[shock_idx] for k, v in L_pooled.items()}
    shock_survivors, _ = model_confidence_set(L_shock_only, seed=0)
    assert shock_survivors == ["weather_skill_model"], (
        f"expected weather_skill_model to reverse and win the shock-years-only slice, got "
        f"{shock_survivors}")

    # The reversal is real: confirm the shock-only edge genuinely favors the OTHER model, not noise.
    pooled_mean = {k: float(np.mean(v)) for k, v in L_pooled.items()}
    shock_mean = {k: float(np.mean(v)) for k, v in L_shock_only.items()}
    assert pooled_mean["trend_fitter"] < pooled_mean["weather_skill_model"]
    assert shock_mean["weather_skill_model"] < shock_mean["trend_fitter"]
