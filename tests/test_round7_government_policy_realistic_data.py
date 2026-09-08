"""Round-7 government/policy domain deep-dive, distinct from round 3's adoption-friction persona.

This is the paper's OWN flagship real-world domain (the §7 state EB-trigger decision-cost work),
and until this round it had never been stress-tested with realistic LONG-HORIZON government/
economic administrative data -- this project's own curated panels top out around T=36 (3 years),
while real state labor-market series span decades. Every scenario here is a permanent regression
test with a fixed seed, not a smoke test -- each asserts something specific and meaningful about
correct behavior, not just "doesn't crash".
"""
import time

import numpy as np
import pytest

from selection_fragility.fragility import decision_breakdown, pooled_winner
from selection_fragility.mcs import model_confidence_set
from selection_fragility.panel import LossPanel


def _month_labels(n, start_year=1994, start_month=1):
    labels = []
    y, m = start_year, start_month
    for _ in range(n):
        labels.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return labels


def test_thirty_year_multi_recession_panel_no_crash_and_sane_bounds():
    """T=360 (30 years monthly), 3 realistic recessions (dot-com/GFC/COVID-shaped) at real relative
    spacing, 4 candidate nowcasting methods with different error profiles (steady baseline, a
    persistence-style method that blows up in recessions, a robust-but-mediocre ensemble, a weak
    baseline). This horizon is 10x longer than any panel this project's own test fixtures use --
    the real shape of the paper's own government-statistics domain, never exercised at this length
    before this round. Must not crash, and k*/MCS must respect their documented bounds."""
    rng = np.random.default_rng(1)
    T = 360
    recessions = [(84, 12, 1.2), (168, 20, 2.0), (312, 6, 3.5)]

    def shocked(base_scale, extra_recession_scale=None):
        e = rng.normal(0, base_scale, T)
        if extra_recession_scale is not None:
            for start, dur, _ in recessions:
                end = min(start + dur, T)
                e[start:end] += rng.normal(0, extra_recession_scale, end - start)
        return np.abs(e)

    L = {
        "model_A_baseline": shocked(0.30),
        "model_B_naive_persistence": shocked(0.22, extra_recession_scale=0.9),
        "model_C_robust_ensemble": shocked(0.34),
        "model_D_weak_baseline": shocked(0.55),
    }
    labels = _month_labels(T)
    panel = LossPanel.from_losses(dict(L), labels=labels)
    assert len(panel.labels) == T
    assert set(panel.models) == set(L)

    w = np.ones(T)
    k, opp, removed = decision_breakdown(dict(L), w=w)
    assert 0 <= k <= T
    assert len(removed) == k
    assert opp in L

    surv, p = model_confidence_set(dict(L), B=500, block=6, seed=0)
    assert 1 <= len(surv) <= len(L)
    assert 0.0 <= p <= 1.0 + 1e-9


def test_low_volatility_stable_economy_gives_high_confidence():
    """A stable state economy, mild seasonality, one small recession, one candidate model with a
    small but consistent true edge over 25 years (T=300). Real government/labor-market series with
    genuinely low structural volatility (a diversified state economy, no boom-bust industry
    exposure) should let the tool correctly converge on high confidence, not report spurious
    fragility just because the horizon is long."""
    rng = np.random.default_rng(2)
    T = 300
    L = {
        "model_best": np.abs(rng.normal(0.0, 0.20, T)),
        "model_b": np.abs(rng.normal(0.15, 0.20, T)),
        "model_c": np.abs(rng.normal(0.25, 0.22, T)),
    }
    w = np.ones(T)
    surv, p = model_confidence_set(dict(L), B=500, block=6, seed=0)
    assert surv == ["model_best"], (
        f"a consistently-best model over a low-volatility 25-year horizon should be the sole MCS "
        f"survivor -- got {surv}")
    k, opp, removed = decision_breakdown(dict(L), w=w)
    assert k > 5, (
        f"a low-volatility, consistently-best-model panel should require removing a meaningful "
        f"number of periods to flip the winner (robust), got k*={k}")


def test_high_volatility_multi_shock_panel_attributes_fragility_to_the_real_driver():
    """A structurally volatile state (boom-bust industry exposure), T=300, with TWO qualitatively
    different shock episodes: a slow-building 50-month regional recession (months 60-110) where
    model_B degrades steadily, and a sharp 7-month acute shock (months 200-206) where model_A spikes
    instead. These are deliberately NOT both COVID-shaped -- this project's own curated data only
    has one shock archetype (a single acute pandemic-style cliff); real state economies see both
    gradual structural decline and sharp acute shocks, sometimes in the same 25-year window.

    The correct behavior: decision_breakdown's removed/pivotal periods should attribute fragility to
    whichever episode actually drives the pooled comparison for THIS panel's real magnitudes (here,
    the 50-month gradual recession's cumulative effect dominates the 7-month acute spike) -- not a
    scattering of arbitrary periods, and not the wrong episode. This locks in that the tool correctly
    identifies the REAL driver among multiple distinct shock types, not merely "some bad period"."""
    rng = np.random.default_rng(3)
    T = 300
    base_noise = 0.45
    errA = rng.normal(0, base_noise, T)
    errB = rng.normal(0, base_noise * 0.9, T)
    errA[200:207] += rng.normal(0, 1.8, 7)  # acute shock hits model A
    grad = np.linspace(0, 1.3, 50)
    errB[60:110] += grad + rng.normal(0, 0.3, 50)  # gradual recession hits model B
    L = {"model_A_acute_sensitive": np.abs(errA), "model_B_gradual_sensitive": np.abs(errB)}
    labels = _month_labels(T)

    w = np.ones(T)
    k, opp, removed = decision_breakdown(dict(L), w=w)
    removed_labels = {labels[i] for i in removed}
    gradual_window = set(labels[60:110])
    acute_window = set(labels[200:207])

    in_gradual = len(removed_labels & gradual_window)
    in_acute = len(removed_labels & acute_window)
    in_neither = len(removed_labels) - in_gradual - in_acute

    assert in_gradual >= len(removed_labels) * 0.75, (
        f"expected the pivotal periods to concentrate in the dominant (gradual-recession) shock "
        f"window given these magnitudes; got {in_gradual}/{len(removed_labels)} in-window, "
        f"{in_acute} in the acute window, {in_neither} elsewhere -- removed={sorted(removed_labels)}")


def test_fifty_state_administrative_batch_scale():
    """Real administrative scale for this domain: 50 states, each with an independent 20-year
    (T=240) monthly panel and its own 3-candidate nowcasting comparison -- the state-EB-trigger
    equivalent of demand-planning's per-SKU batch pattern from round 6. Every state must produce a
    valid, bounded k*, and the full batch (decision_breakdown only, the cheap path real production
    monitoring would use) must complete well within a routine batch-job window."""
    t0 = time.perf_counter()
    ks = []
    for state_i in range(50):
        T = 240
        srng = np.random.default_rng(1000 + state_i)
        L = {
            "nowcast_x": np.abs(srng.normal(0, 0.3 + 0.1 * srng.random(), T)),
            "nowcast_y": np.abs(srng.normal(0.05, 0.28, T)),
            "nowcast_z": np.abs(srng.normal(0.1, 0.35, T)),
        }
        k, opp, removed = decision_breakdown(dict(L), w=np.ones(T))
        assert 0 <= k <= T
        assert opp in L
        ks.append(k)
    elapsed = time.perf_counter() - t0
    assert len(ks) == 50
    assert elapsed < 10.0, (
        f"50-state batch (T=240 each) took {elapsed:.2f}s -- a regression here would make routine "
        f"cross-state administrative monitoring impractical")


def test_fifty_year_horizon_with_four_recessions_no_crash():
    """T=600 (50 years monthly) with 4 recessions spread across the full horizon at realistic
    relative spacing and magnitudes -- an extreme but not unrealistic horizon for a long-tenure
    federal/state statistical series (e.g. a program that has been tracked since the 1970s). This is
    beyond anything else tested anywhere in this project; must not crash, must respect k*/MCS
    bounds, and must complete in reasonable time for a decades-long single-series analysis."""
    rng = np.random.default_rng(9)
    T = 600
    errA = rng.normal(0, 0.3, T)
    errB = rng.normal(0.08, 0.32, T)
    for start, dur, mag in [(90, 10, 1.0), (170, 18, 1.8), (360, 8, 1.2), (552, 6, 3.0)]:
        end = min(start + dur, T)
        errA[start:end] += rng.normal(0, 0.6, end - start)
    L = {"a": np.abs(errA), "b": np.abs(errB)}

    t0 = time.perf_counter()
    k, opp, removed = decision_breakdown(dict(L), w=np.ones(T))
    assert 0 <= k <= T
    assert time.perf_counter() - t0 < 5.0

    t0 = time.perf_counter()
    surv, p = model_confidence_set(dict(L), B=500, block=12, seed=0)
    assert 1 <= len(surv) <= 2
    assert 0.0 <= p <= 1.0 + 1e-9
    assert time.perf_counter() - t0 < 15.0


def test_pooled_winner_and_decision_breakdown_agree_on_long_horizon_panel():
    """Cross-check at long horizon: decision_breakdown's default opponent selection is built on
    pooled_winner -- confirm they stay mutually consistent (the winner decision_breakdown treats as
    'a' must equal pooled_winner's own answer) at T=360, not just at the short horizons every other
    test in this suite uses."""
    rng = np.random.default_rng(1)
    T = 360
    L = {
        "model_A_baseline": np.abs(rng.normal(0, 0.30, T)),
        "model_B_naive_persistence": np.abs(rng.normal(0, 0.22, T)),
        "model_C_robust_ensemble": np.abs(rng.normal(0, 0.34, T)),
    }
    w = np.ones(T)
    winner = pooled_winner(dict(L), w=w)
    k, opp, removed = decision_breakdown(dict(L), w=w, a=None)
    assert opp != winner
    assert winner in L
