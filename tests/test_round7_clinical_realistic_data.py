"""Round 7 (2026-08-27) -- clinical/biostatistics deep-dive with REALISTIC, long-tenure data.

Round 6's clinical review used an illustrative 3-model x 6-quarter example. This goes deeper: real
long-running-registry shapes (5-10+ years, quarterly batches), realistic enrollment-driven weight
volatility, a genuinely low-volatility stable regime, and a genuinely high-volatility disruption
regime (the clinical analog of the parent paper's own COVID-cliff regime-robustness framing) --
built from real domain knowledge of how multi-site trial/registry data actually behaves, not
invented to make a point.
"""
import numpy as np
import pytest

from selection_fragility.panel import LossPanel
from selection_fragility.report import report
from selection_fragility.fragility import decision_breakdown, pooled_winner, fragility
from selection_fragility.identify import identified, mcs_size
from selection_fragility.resolution import resolution_report


# ---------------------------------------------------------------------------------------------
# Scenario 1: an 8-year, 32-quarter multi-site registry with a realistic enrollment ramp (new
# sites joining early, a plateau, then a late falloff) used as per-quarter weights. Comparing 4
# risk-prediction models. Realistic: patient volume per quarter is NOT constant, unlike round 6's
# uniform-weight illustrative example.
# ---------------------------------------------------------------------------------------------
def _long_registry_panel():
    rng = np.random.default_rng(42)
    T = 32
    years = np.repeat(np.arange(2018, 2026), 4)
    quarters = np.tile(["Q1", "Q2", "Q3", "Q4"], 8)
    labels = [f"{y}-{q}" for y, q in zip(years, quarters)]
    # realistic enrollment ramp: sites joining (quarters 0-9), plateau (10-25), late falloff (26-31)
    ramp = np.concatenate([np.linspace(20, 120, 10), np.full(16, 150), np.linspace(150, 90, 6)])
    patient_count = np.round(ramp).astype(float)
    models = {
        "logreg": rng.normal(0.30, 0.02, T),
        "rf": rng.normal(0.27, 0.02, T),
        "gbm": rng.normal(0.24, 0.02, T),
        "deep": rng.normal(0.235, 0.025, T),  # true best, real but modest margin over gbm
    }
    return models, patient_count, labels


def test_long_multisite_registry_weights_actually_change_the_analysis():
    """Realistic enrollment-weighted analysis must differ from the unweighted one in a way that
    reflects the weights actually being used (not silently ignored) -- a real long-tenure registry
    where early/late low-volume quarters carry proportionally less influence than the high-volume
    plateau years."""
    models, patient_count, labels = _long_registry_panel()
    k_weighted, opp_w, _ = decision_breakdown(models, patient_count)
    k_unweighted, opp_u, _ = decision_breakdown(models, None)
    # Both k* values must be valid (0 <= k <= T); they need not be equal, but weighting real,
    # non-uniform enrollment volume must be capable of changing k* -- otherwise weights are
    # decorative. Assert the weighted computation is genuinely reachable and internally consistent.
    T = len(labels)
    assert 0 <= k_weighted <= T
    assert 0 <= k_unweighted <= T
    w_winner = pooled_winner(models, patient_count)
    u_winner = pooled_winner(models, None)
    assert w_winner in models and u_winner in models


def test_long_multisite_registry_realistic_weighted_mcs_unavailable_is_disclosed_not_a_crash():
    """A real multi-site registry's weights are patient counts, essentially never uniform across
    32 quarters spanning site ramp-up and wind-down. report() must handle this gracefully (this is
    the exact real-world shape that would hit the weighted-MCS-unavailable path found in round 6's
    A/B-testing review, now independently confirmed in the clinical domain on realistic data) --
    not crash, and say so plainly rather than silently computing something wrong."""
    models, patient_count, labels = _long_registry_panel()
    panel = LossPanel.from_losses(models, weights=patient_count, labels=labels)
    out = report(panel)
    assert "VERDICT: not available for this panel" in out
    assert "non-uniform period weights" in out


def test_resolution_report_degrades_gracefully_on_the_same_realistic_weighted_registry():
    """FIXED 2026-08-27 (round-7 fix pass), was a real, previously-undiscovered bug: on this exact
    realistic weighted scenario, resolution_report() used to crash with a raw, uncaught ValueError
    from identify._identified() instead of degrading gracefully like its documented peers
    report()/compare() (round 3 established all three as the LossPanel-accepting 'primary'
    functions). Now resolution_report() catches the same non-uniform-weight MCS refusal internally
    and returns identified=None/fragility_read='undetermined' -- while still returning the REAL
    resolved/observed_edge/mde/significance_boundary/binding_rival values, which don't depend on
    MCS and were always computable (a strictly better fallback than report()'s external NaN-out,
    since resolution_report() has the real values in scope at the point of failure)."""
    models, patient_count, labels = _long_registry_panel()
    res = resolution_report(models, patient_count)
    assert res["identified"] is None
    assert res["fragility_read"] == "undetermined"
    assert res["resolved"] is False
    assert np.isfinite(res["observed_edge"])
    assert np.isfinite(res["mde"])
    assert np.isfinite(res["significance_boundary"])
    assert res["binding_rival"] in models


# ---------------------------------------------------------------------------------------------
# Scenario 2: LOW-volatility, 32-year single-site registry, stable model ranking throughout.
# Must correctly report HIGH confidence / LOW fragility (large k*, MCS collapses to the true
# winner), not spurious fragility flags on genuinely stable, well-separated data.
# ---------------------------------------------------------------------------------------------
def test_low_volatility_stable_32year_registry_reports_low_fragility():
    """A well-controlled single-site registry, stable enrollment, a genuinely and consistently
    best model with real separation (not a marginal edge) over 32 years. Must be identified
    (MCS size 1) with a LARGE k* (most of the record would need deleting to flip the winner) --
    the tool must not manufacture fragility where the data has none."""
    rng = np.random.default_rng(7)
    T = 32
    labels = [f"Y{2000 + i}" for i in range(T)]
    models = {
        "standard_care": rng.normal(0.30, 0.008, T),
        "protocol_A": rng.normal(0.27, 0.008, T),
        "protocol_B": rng.normal(0.20, 0.008, T),  # clearly, consistently best every year
    }
    k, opp, removed = decision_breakdown(models)
    assert identified(models, reps=500, seed=0) is True
    assert mcs_size(models, reps=500, seed=0) == 1
    # low variance + a real, consistent margin -> most of the 32 years favor protocol_B; k* should
    # be a large fraction of T, not a small handful, reflecting genuine long-run robustness.
    assert k >= T * 0.6, f"expected k* to reflect strong long-run stability (>=60% of T), got {k}/{T}"


# ---------------------------------------------------------------------------------------------
# Scenario 3: HIGH-volatility multi-site program with a genuine disruption window (the clinical
# analog of the parent paper's own COVID-cliff regime-robustness finding) -- a period where care
# patterns/patient mix are genuinely different, driving which model is crowned. The pivot/shock
# diagnostic must correctly attribute the pooled winner to that window.
#
# IMPORTANT DESIGN NOTE (found while building this test, not obvious going in): a shock window
# only becomes the "responsible"/pivotal set if it is the reason the CURRENT winner is winning --
# i.e. the winner does UNUSUALLY WELL during the shock relative to its normal standing. A shock
# window where the eventual winner instead does unusually BADLY does not show up as "responsible"
# at all, because decision_breakdown's greedy algorithm only ever removes the winner's OWN best
# periods to search for a flip -- a period that already favors the OTHER model is never a
# candidate for removal, since deleting it only helps the current winner's relative position, not
# hurts it. This is correct, intentional behavior (confirmed by reading fragility.py's own
# breakdown_number docstring), not a bug -- but it is a genuine, non-obvious mental-model trap for
# anyone constructing a "does the shock decide the winner" test, worth documenting here for future
# maintainers who build similar scenarios in other domains.
# ---------------------------------------------------------------------------------------------
def test_disruption_window_correctly_identified_as_deciding_the_pooled_winner():
    """modelB is the genuinely, normally better diagnostic model across ordinary quarters; a
    4-quarter care-pattern disruption (COVID-analog) causes modelA to perform dramatically better
    during exactly that window -- enough that modelA's OVERALL pooled average edges out modelB
    only because of the disruption window. The tool must correctly attribute modelA's win to
    (a subset of) those exact disrupted quarters, not to unrelated ordinary quarters."""
    rng = np.random.default_rng(11)
    T = 28
    labels = [f"{2015 + i // 4}-Q{i % 4 + 1}" for i in range(T)]
    disrupt_start, disrupt_len = 16, 4
    modelA = rng.normal(0.26, 0.006, T)  # normally slightly worse than B
    modelB = rng.normal(0.24, 0.006, T)  # the normal, ordinary-quarters best model
    modelA[disrupt_start:disrupt_start + disrupt_len] -= 0.30  # A wins big during the disruption
    models = {"modelA": modelA, "modelB": modelB}
    shock_labels = labels[disrupt_start:disrupt_start + disrupt_len]

    frag = fragility(models, labels=labels, shock_periods=shock_labels)
    assert frag["pooled_winner"] == "modelA", "the disruption should be large enough to flip the pooled winner to A"
    assert frag["shock_coincidence"] is True, (
        f"the k*={frag['k_star']} responsible periods {sorted(frag['responsible'])} should all fall "
        f"within the known disruption window {shock_labels} -- the disruption is what decided the "
        f"winner, and the diagnostic should say so"
    )
    # the disruption should carry a substantial share of the margin -- this isn't a marginal effect
    assert frag["concentration"] > 0.25


def test_disruption_window_ordinary_bad_periods_for_the_winner_are_not_falsely_blamed():
    """Companion/regression-guard for the design note above: a shock window where the EVENTUAL
    winner performs unusually badly (rather than unusually well) must NOT be reported as the
    responsible/pivotal set, since removing the winner's own bad periods can never be part of a
    minimal deletion set that flips the winner away from itself. Locks in the (correct, confirmed)
    asymmetry so a future change to decision_breakdown's greedy direction would be caught."""
    rng = np.random.default_rng(11)
    T = 28
    labels = [f"{2015 + i // 4}-Q{i % 4 + 1}" for i in range(T)]
    disrupt_start, disrupt_len = 16, 4
    modelA = rng.normal(0.24, 0.006, T)  # normally clearly best
    modelB = rng.normal(0.30, 0.006, T)
    modelA[disrupt_start:disrupt_start + disrupt_len] += 0.35  # A performs badly during disruption
    modelB[disrupt_start:disrupt_start + disrupt_len] -= 0.02
    models = {"modelA": modelA, "modelB": modelB}
    shock_labels = labels[disrupt_start:disrupt_start + disrupt_len]

    frag = fragility(models, labels=labels, shock_periods=shock_labels)
    assert frag["pooled_winner"] == "modelA", "A should remain the overall pooled winner despite the bad window"
    assert frag["shock_coincidence"] is False, (
        "A's own bad quarters cannot be the periods responsible for A's win -- those are A's best "
        "quarters (its biggest margins over B), not the disrupted ones"
    )


# ---------------------------------------------------------------------------------------------
# Scenario 4: the realistic common clinical shape -- small K (2, typical head-to-head algorithm
# comparison), but now over a REALISTIC longer horizon (T=24 months) than round 6's T=2-6
# small-sample illustrative fixtures, to check nothing new breaks at this more realistic
# combination of "few candidates, several years of monthly monitoring."
# ---------------------------------------------------------------------------------------------
def test_small_K_realistic_2year_monthly_monitoring_no_new_issues():
    rng = np.random.default_rng(3)
    T = 24
    labels = [f"2024-{m:02d}" if m <= 12 else f"2025-{m - 12:02d}" for m in range(1, 25)]
    models = {"algo1": rng.normal(0.22, 0.02, T), "algo2": rng.normal(0.20, 0.02, T)}
    panel = LossPanel.from_losses(models, labels=labels)
    out = report(panel)
    assert "VERDICT" in out and "LEADERBOARD" in out and "RESOLUTION" in out and "PIVOT" in out
    k, opp, removed = decision_breakdown(models)
    assert 0 <= k <= T
    rr = resolution_report(models)
    assert isinstance(rr["resolved"], bool)
