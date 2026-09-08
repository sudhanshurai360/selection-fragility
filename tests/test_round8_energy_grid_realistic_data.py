"""Round-8 energy/electricity-grid load-forecasting domain deep-dive -- a genuinely new domain,
not covered by any prior round. Grid load data has a distinct pathology from what earlier rounds
tested: STRONG, predictable, multi-scale seasonality (daily/weekly/annual all stacked) combined with
rare but genuinely catastrophic-magnitude single events (a realistic analog to the 2021 Texas grid
freeze), not just "a regime change" (round 6, WHICH model wins changes) or generic fat tails (round 6
financial/GARCH). Every scenario here is a permanent regression test with a fixed seed and a real,
meaningful assertion -- not a smoke test.
"""
import numpy as np
import pytest

from selection_fragility.fragility import decision_breakdown, pooled_winner, _MAX_ABS_LOSS
from selection_fragility.mcs import model_confidence_set
from selection_fragility.panel import LossPanel
from selection_fragility.report import report


def _seasonal_base(T, annual_amp=0.5, weekly_amp=0.15, phase=-np.pi / 2):
    t = np.arange(T)
    annual = 1.0 + annual_amp * np.sin(2 * np.pi * t / 365.25 + phase)
    weekly = 1.0 + weekly_amp * np.sin(2 * np.pi * t / 7)
    return annual * weekly


def _five_model_grid_panel(rng, T, seasonal):
    """5 candidate load-forecasting models with different skill/noise profiles, seasonally scaled."""
    def build(rel_skill, noise_scale):
        return np.abs(rng.normal(0, noise_scale, T)) * seasonal * rel_skill
    return {
        "persistence": build(1.4, 0.5),
        "ARIMA_seasonal": build(1.0, 0.35),
        "gradient_boost": build(0.9, 0.32),
        "neural_lstm": build(0.95, 0.40),
        "ensemble": build(0.85, 0.30),
    }


def test_multiyear_multiscale_seasonal_panel_no_crash_sane_bounds():
    """T=1460 (4 years daily), stacked annual+weekly seasonality, 5 candidate models -- this
    combination of horizon length and multi-scale seasonal structure is untested by any prior
    round. Must not crash; k*/MCS must respect their documented bounds."""
    rng = np.random.default_rng(42)
    T = 1460
    L = _five_model_grid_panel(rng, T, _seasonal_base(T))
    panel = LossPanel.from_losses(dict(L))
    assert len(panel.labels) == T
    assert set(panel.models) == set(L)

    k, opp, removed = decision_breakdown(dict(L))
    assert 0 <= k <= T
    assert opp in L
    assert len(removed) == k
    assert len(set(removed)) == len(removed)

    surv, p = model_confidence_set(dict(L), B=300, block=7, seed=0)
    assert 1 <= len(surv) <= len(L)
    assert 0.0 <= p <= 1.0 + 1e-9

    out = report(panel)
    assert isinstance(out, str) and len(out) > 0


def test_catastrophic_event_is_the_dominant_pivotal_driver_not_a_false_positive_on_max_abs_loss():
    """A realistic Texas-freeze-style event: forecast-error magnitude spikes 15-40x its normal
    seasonal level for a 5-day window (days 800-804 of a 4-year panel), then reverts. Two things
    must both hold: (1) _MAX_ABS_LOSS's overflow guard (round-4 fix, cap=1e100) must NOT false-
    positive on this data -- the spike is large for THIS domain but nowhere near the guard's
    threshold, and a guard that fires on legitimate extreme-but-real data would be a real usability
    bug; (2) decision_breakdown's greedy removal order must correctly and precisely surface the
    catastrophic days as the single most decisive periods in the entire panel -- not merely present
    somewhere in a large removed set (with only 5 catastrophic days out of T=1460, "majority of
    removed periods" is not even a coherent bar the way it was for round 7's 50-period shock
    windows), but disproportionately represented at the very front of the removal order, which is
    the correct, domain-relevant reading: "these 5 days are what actually decided this comparison."
    """
    rng = np.random.default_rng(42)
    T = 1460
    L = _five_model_grid_panel(rng, T, _seasonal_base(T))
    catastrophic_days = list(range(800, 805))
    for m in L:
        L[m] = L[m].copy()
        L[m][catastrophic_days] *= rng.uniform(15, 40)

    max_loss = max(v.max() for v in L.values())
    assert max_loss < 1e-6 * _MAX_ABS_LOSS, (
        f"sanity check on the test's own construction: max loss {max_loss:.3f} must be nowhere "
        f"near the {_MAX_ABS_LOSS:.0e} guard threshold, or this isn't testing 'realistic extreme'")

    # (1) the guard does not false-positive -- LossPanel construction and decision_breakdown must
    # both succeed cleanly on this realistic-but-large data.
    panel = LossPanel.from_losses(dict(L))
    k, opp, removed = decision_breakdown(dict(L))
    assert k > 0

    # (2) catastrophic days dominate the FRONT of the removal order (rank = position in `removed`,
    # 0 = single most decisive period in the whole 1460-day panel).
    ranks = [removed.index(d) for d in catastrophic_days if d in removed]
    assert len(ranks) >= 3, (
        f"expected at least 3 of the 5 catastrophic days to appear among the removed/pivotal "
        f"periods at all; only {len(ranks)} did (removed={removed[:20]}...)")
    assert min(ranks) <= 2, (
        f"expected at least one catastrophic day to be among the top-3 MOST decisive periods "
        f"(rank<=2) in the entire {T}-day panel; best rank found was {min(ranks)}")
    # base-rate check: 5 catastrophic days out of 1460 -- if periods were picked at random, the
    # expected count of catastrophic days in the first 5 removed positions would be ~5*5/1460 =
    # 0.017. Getting >=2 is a >100x enrichment over chance, not a coincidence.
    in_first_five = sum(1 for d in catastrophic_days if d in removed[:5])
    assert in_first_five >= 2, (
        f"expected the catastrophic event to visibly dominate the first 5 removed periods "
        f"(>>100x enrichment over the ~0.017 expected-by-chance rate); got {in_first_five}/5")


def test_stable_mild_climate_baseline_reports_high_confidence_not_manufactured_fragility():
    """A mild-climate, low-weather-variance grid region with a consistently-better model over 3
    years (T=1095) -- the tool must NOT manufacture false fragility on genuinely stable data. MCS
    must converge to the true best model alone, with a strong (not borderline) significance level,
    and k* must reflect real robustness (a large fraction of T needed to flip the winner), not a
    fragile few-period result."""
    rng = np.random.default_rng(7)
    T = 1095
    seasonal = _seasonal_base(T, annual_amp=0.15, weekly_amp=0.05)
    L = {
        "best_model": np.abs(rng.normal(0, 0.10, T)) * seasonal,
        "rival_a": np.abs(rng.normal(0.08, 0.12, T)) * seasonal,
        "rival_b": np.abs(rng.normal(0.12, 0.14, T)) * seasonal,
    }
    k, opp, removed = decision_breakdown(dict(L))
    assert k / T > 0.15, (
        f"a low-volatility, consistently-best-model panel should require removing a substantial "
        f"fraction of periods to flip the winner; got k*={k} of T={T} ({k/T:.1%})")

    surv, p = model_confidence_set(dict(L), B=500, block=7, seed=0)
    assert surv == ["best_model"], f"expected sole survivor 'best_model', got {surv}"
    assert p < 0.01, f"expected strong (not borderline) significance on genuinely stable data, got p={p}"


def test_recency_weighted_scheme_shifts_kstar_sensibly_without_crash():
    """Grid forecasting commonly weights recent data more heavily (evolving topology/demand
    patterns) via exponential decay. On the seasonal+catastrophic-event panel, an exponential-decay
    recency weighting (half-life=365 days, most recent day weight=1) must run cleanly through both
    the raw-array tier (decision_breakdown/pooled_winner) and produce a sensible, non-degenerate
    result -- distinct from, but not wildly inconsistent with, the uniform-weight reading."""
    rng = np.random.default_rng(42)
    T = 1460
    L = _five_model_grid_panel(rng, T, _seasonal_base(T))
    catastrophic_days = list(range(800, 805))
    for m in L:
        L[m][catastrophic_days] *= rng.uniform(15, 40)

    halflife = 365.0
    decay = np.log(2) / halflife
    w_recency = np.exp(-decay * (T - 1 - np.arange(T)))
    assert w_recency[-1] == pytest.approx(1.0)
    assert 0 < w_recency.min() < w_recency.max()

    k_uniform, opp_u, _ = decision_breakdown(dict(L), w=np.ones(T))
    k_recency, opp_r, _ = decision_breakdown(dict(L), w=w_recency)
    assert 0 <= k_recency <= T
    assert opp_r in L
    # recency weighting should not swing the breakdown count wildly (>3x) relative to uniform on
    # the same underlying data -- a sanity bound on "sensible", not an exact-match requirement,
    # since the two ARE genuinely different weightings and are allowed to disagree somewhat.
    assert k_recency <= 3 * k_uniform and k_uniform <= 3 * k_recency, (
        f"recency-weighted k*={k_recency} vs uniform k*={k_uniform} diverge by more than 3x -- "
        f"investigate whether the weighting is being applied sensibly")

    pw_uniform = pooled_winner(dict(L), w=np.ones(T))
    pw_recency = pooled_winner(dict(L), w=w_recency)
    assert pw_uniform in L and pw_recency in L
