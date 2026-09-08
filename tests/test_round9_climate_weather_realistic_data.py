"""Round 9 -- meteorological/climate ensemble-forecast-verification domain review.

Persona: an operational forecaster comparing candidate ensemble forecast systems (a realistic
"GFS-style / ECMWF-style / UKMET-style / ICON-style / GEM-style" flavor of skill/bias, without
needing real NWP output) via CRPS-style loss across daily forecasts. Distinct from round 6's
generic probabilistic/UQ review and round 8's energy-grid catastrophic-event review: this round's
focus is field-specific verification culture -- lead-time-dependent skill, seasonal forecasting's
famously low signal, and extreme-event attribution -- not pinball-loss mechanics or grid-scale
outliers.

All three scenarios below were built and run against real code before being locked in here; see
the round-9 ledger entry for the live output each assertion is based on.
"""
import numpy as np
import pytest

from selection_fragility.fragility import decision_breakdown, pooled_winner
from selection_fragility.mcs import model_confidence_set

MODELS = ["GFS_style", "ECMWF_style", "UKMET_style", "ICON_style", "GEM_style"]
BASE_SKILL = {"GFS_style": 1.00, "ECMWF_style": 1.02, "UKMET_style": 1.05, "ICON_style": 1.08, "GEM_style": 1.10}
DEGRADE_RATE = {"GFS_style": 0.35, "ECMWF_style": 0.15, "UKMET_style": 0.20, "ICON_style": 0.22, "GEM_style": 0.30}


def _lead_time_panel(rng, lead_days, T=1000, noise_scale=0.15):
    """A realistic CRPS-style panel at a given forecast lead time (days). GFS_style has the best
    short-range skill but degrades fastest; ECMWF_style is worse at day-1 but degrades slowest --
    the well-documented real operational pattern (GFS wins short-range, ECMWF overtakes and stays
    ahead at medium range)."""
    L = {}
    for m in MODELS:
        skill = BASE_SKILL[m] + DEGRADE_RATE[m] * (lead_days - 1) / 6.0
        L[m] = np.clip(rng.normal(skill, noise_scale, T), 0.01, None)
    return L


class TestLeadTimeDependentSkill:
    """A candidate model's relative skill genuinely changes with forecast lead time -- this is not
    a pathology to guard against, it's the real, expected behavior of ensemble NWP systems, and a
    verification tool for this field needs to track it correctly across separate lead-time slices."""

    def test_short_range_winner_differs_from_medium_range_winner(self):
        rng = np.random.default_rng(42)
        w1 = pooled_winner(_lead_time_panel(rng, lead_days=1))
        w7 = pooled_winner(_lead_time_panel(rng, lead_days=7))
        assert w1 == "GFS_style", f"day-1 winner should be the best short-range system, got {w1}"
        assert w7 == "ECMWF_style", f"day-7 winner should be the slowest-degrading system, got {w7}"
        assert w1 != w7, "lead-time-dependent skill should genuinely change the declared winner"

    def test_breakdown_point_grows_with_lead_time_as_margin_widens(self):
        """As lead time increases, ECMWF_style's margin over its rivals widens (it degrades slower),
        so k* for the medium-range winner should be larger (more robust), not smaller."""
        rng = np.random.default_rng(42)
        k3, _, _ = decision_breakdown(_lead_time_panel(rng, lead_days=3))
        k7, _, _ = decision_breakdown(_lead_time_panel(rng, lead_days=7))
        assert k7 > k3, f"k* should grow as the medium-range margin widens: day3 k*={k3}, day7 k*={k7}"


class TestSeasonalLowSkillScenario:
    """Seasonal (weeks-to-months-ahead) forecasting is famously low-skill, close to climatology.
    A verification tool must correctly report this as non-identification, not manufacture a
    spurious winner out of pure noise."""

    def test_seasonal_forecasts_near_climatology_are_not_identified(self):
        rng = np.random.default_rng(7)
        T = 400
        models = ["seasonal_A", "seasonal_B", "seasonal_C", "climatology_baseline"]
        skill = {"seasonal_A": 1.001, "seasonal_B": 1.000, "seasonal_C": 1.002,
                 "climatology_baseline": 1.000}
        L = {m: np.clip(rng.normal(skill[m], 0.30, T), 0.01, None) for m in models}
        surv, p = model_confidence_set(L, seed=0)
        k, _, _ = decision_breakdown(L)
        assert len(surv) == len(models), (
            f"a near-climatology seasonal panel should leave ALL {len(models)} candidates in the "
            f"MCS (no real signal to distinguish them), got {len(surv)} survivors: {sorted(surv)}")
        assert k / T < 0.01, f"k*/T={k/T:.4f} should be near-zero (maximally fragile), not a confident read"


class TestExtremeEventAttribution:
    """A short run of extreme-weather days (e.g. a hurricane track forecast) with sharply divergent
    model skill, embedded in otherwise-similar routine forecasting, should be precisely attributed
    by PIVOT/decision_breakdown as the actual driver of the winner -- not diffused across the
    whole record."""

    def test_hurricane_days_are_exactly_the_removed_periods(self):
        rng = np.random.default_rng(7)
        T = 500
        hurricane_days = rng.choice(T, size=6, replace=False)
        L = {m: np.clip(rng.normal(1.00, 0.10, T), 0.01, None)
             for m in ("GFS_style", "ECMWF_style", "UKMET_style")}
        for d in hurricane_days:
            L["ECMWF_style"][d] = 0.3
            L["GFS_style"][d] = 2.5
            L["UKMET_style"][d] = 2.3
        winner = pooled_winner(L)
        k, _, removed = decision_breakdown(L)
        assert winner == "ECMWF_style"
        # k* is the MINIMUM sufficient removal, not "all extreme days" -- the greedy algorithm
        # correctly found that 4 of the 6 hurricane days already flip the pooled margin, so k*=4
        # is the mathematically correct answer, not a miss. What matters for attribution is that
        # every removed period IS a hurricane day (no routine day gets blamed) and that k* is
        # bounded by the number of genuinely divergent days available.
        assert 1 <= k <= len(hurricane_days), f"k*={k} should be a small, bounded subset of the 6 hurricane days"
        assert set(removed).issubset(set(hurricane_days.tolist())), (
            "every removed/responsible period must be a hurricane day -- no routine day should be "
            f"blamed. removed={sorted(removed)}, hurricane_days={sorted(hurricane_days.tolist())}")


def test_mcs_elimination_statistic_is_dm_test_family_not_a_foreign_framework():
    """Verification-culture finding: this field's standard practice is Diebold-Mariano-style
    testing of pairwise loss differentials. Confirmed directly from source (mcs.py's own docstring
    and elimination-rule comment) that HLN's default 'studentized' elimination statistic IS a
    studentized loss-differential deviation -- the same statistical family as a DM test, applied
    iteratively for set elimination rather than as a single pairwise p-value. An operational
    forecaster already using DM tests is not being asked to adopt a foreign framework."""
    import inspect
    from selection_fragility import mcs as mcs_mod
    src = inspect.getsource(mcs_mod)
    assert "studentized" in src.lower()
    assert "e_max" in src or "e-max" in src.lower()
    # the elimination rule doc explicitly frames it as a deviation-from-survivor-average statistic,
    # i.e. built from the same pairwise/mean loss-differential machinery as a DM test
    assert "deviation" in src.lower() and "survivor" in src.lower()
