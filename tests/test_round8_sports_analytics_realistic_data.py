"""Round 8 -- sports analytics domain deep-dive (release/selection-fragility).

Persona: a sports analytics researcher comparing candidate models for team/player performance
forecasting across an NBA-length season (T~82 games) and multi-season horizons. Sports data has a
distinctive non-stationarity signature not tested by any prior round: TRUE quality genuinely shifts
WITHIN a season on a timescale of weeks (trades, injury returns), seasons are short (real small-T even
at "full season" scale), and the field has a well-known, actively-debated pathology -- the "hot hand"
question: is a short streak of good results genuine signal or pure variance? That question is exactly
what this tool's k*/MCS/winner_stability diagnostics claim to answer, so it is the sharpest possible
test of the tool's core value proposition for this domain.

HEADLINE FINDING (see test_lucky_short_streak_can_still_report_identified_like_genuine_signal below):
the tool does NOT reliably distinguish a genuine, sustained ~40-game improvement from an 8-game lucky
streak embedded in an otherwise perfectly null season -- at the default alpha=0.10, BOTH report a
single MCS survivor (i.e. "identified"). The two cases ARE distinguishable on close inspection (the
streak case has a visibly weaker MCS p-value, 0.089 vs 0.023, and only 2 of its k*=8 "responsible"
periods actually fall inside the true streak window, vs 100% for the genuine-shift case) -- but a
user reading only the headline VERDICT, the way `report()` presents it, would see "identified" for
both. This is a real, disclosed interpretive risk for exactly the sports-analytics "hot hand" use
case, not a code bug: MCS's job is to test whether the OBSERVED pooled ranking is statistically
distinguishable from a tie, and a short lucky streak genuinely does make it so at this sample size --
the tool is behaving correctly given what it's asked, but a naive reading of "identified = real signal"
would be wrong here. Not fixed in this round (find/test-only); worth a documented caveat for this
audience specifically, similar in spirit to round 6's MCS-block-length-under-serial-dependence
disclosure.
"""
import numpy as np
import pytest

from selection_fragility.fragility import fragility, exchangeable_benchmark
from selection_fragility.mcs import model_confidence_set

T_SEASON = 82  # realistic NBA regular-season length


def _labels(T):
    return [f"g{i}" for i in range(T)]


# ---------------------------------------------------------------------------
# 1. Genuine in-season roster change: pivot correctly attributes responsibility
#    to the post-change games, and winner_stability reads high (real, robust signal).
# ---------------------------------------------------------------------------
def test_genuine_roster_change_pivot_attributes_to_postchange_games():
    rng = np.random.default_rng(42)
    change_at = 40
    A = np.concatenate([
        rng.normal(1.0, 0.3, change_at),
        rng.normal(0.65, 0.3, T_SEASON - change_at),  # genuine, sustained improvement (e.g. a trade)
    ])
    B = rng.normal(1.0, 0.3, T_SEASON)
    L = {"model_A": A, "model_B": B}
    labels = _labels(T_SEASON)
    shock = [f"g{i}" for i in range(change_at, T_SEASON)]

    res = fragility(L, labels=labels, shock_periods=shock)

    assert res["pooled_winner"] == "model_A"
    # every single responsible (k*) period must fall in the post-change window -- the pooled
    # comparison really is being decided by the changed period, and PIVOT correctly says so.
    assert res["shock_coincidence"] is True
    assert all(int(r[1:]) >= change_at for r in res["responsible"])
    # a real, sustained 42-game shift should read as a high-confidence, robust result
    assert res["winner_stability"] > 0.95

    surv, p = model_confidence_set(L, seed=0)
    assert surv == ["model_A"]
    assert p < 0.05


# ---------------------------------------------------------------------------
# 2. THE HEADLINE FINDING: a lucky short streak, embedded in an otherwise perfectly
#    null (no true skill difference) season, can still make MCS report "identified" --
#    just as confidently, by winner_stability alone, as a genuine sustained shift.
# ---------------------------------------------------------------------------
def test_lucky_short_streak_can_still_report_identified_like_genuine_signal():
    rng = np.random.default_rng(7)
    streak_start, streak_len = 40, 8
    A = rng.normal(1.0, 0.3, T_SEASON)
    B = rng.normal(1.0, 0.3, T_SEASON)
    A[streak_start:streak_start + streak_len] = rng.normal(0.5, 0.1, streak_len)  # pure luck, no true skill change
    L = {"model_A": A, "model_B": B}
    labels = _labels(T_SEASON)
    shock = [f"g{i}" for i in range(streak_start, streak_start + streak_len)]

    res = fragility(L, labels=labels, shock_periods=shock)
    surv, p = model_confidence_set(L, seed=0)

    # The dangerous part: MCS reports a single, "identified" survivor here too, exactly like the
    # genuine-signal case above -- despite there being NO true skill difference anywhere except an
    # 8-game fluke. A user reading only report()'s headline VERDICT would see "identified" for both.
    assert surv == ["model_A"]

    # It is NOT indistinguishable on close inspection, though -- this is what a careful reader (or a
    # future doc improvement) should point people at instead of the bare VERDICT line:
    # (a) the p-value is visibly weaker than the genuine-signal case's 0.023 (still identified, but
    #     closer to the alpha=0.10 boundary)
    assert p > 0.05
    # (b) PIVOT's "responsible" periods are NOT cleanly confined to the true streak window -- unlike
    #     the genuine-shift case's 100% shock-coincidence, here most of the k* responsible periods are
    #     ordinary within-season noise elsewhere, not the actual streak (a real, useful tell that
    #     something is off, even though shock_coincidence alone doesn't block the headline VERDICT).
    assert res["shock_coincidence"] is False
    in_streak = sum(1 for r in res["responsible"] if streak_start <= int(r[1:]) < streak_start + streak_len)
    assert in_streak < len(res["responsible"])


# ---------------------------------------------------------------------------
# 3. Clean, stable-roster baseline (no true difference all season): correctly NOT identified.
# ---------------------------------------------------------------------------
def test_stable_roster_no_true_difference_correctly_not_identified():
    rng = np.random.default_rng(3)
    A = rng.normal(1.0, 0.3, T_SEASON)
    B = rng.normal(1.0, 0.3, T_SEASON)
    surv, p = model_confidence_set({"model_A": A, "model_B": B}, seed=0)
    assert set(surv) == {"model_A", "model_B"}
    assert p > 0.10


# ---------------------------------------------------------------------------
# 4. Multi-season (T~230), realistic small-T-per-season but larger pooled T, genuine sustained
#    edge -- confirms the tool scales down cleanly to realistic sports-analytics sample sizes at
#    the opposite end from round 7's decades-long government/policy horizons.
# ---------------------------------------------------------------------------
def test_multiseason_realistic_horizon_genuine_edge_identified():
    rng = np.random.default_rng(11)
    T = 230  # ~3 NBA seasons pooled
    A = rng.normal(0.85, 0.3, T)
    B = rng.normal(1.0, 0.3, T)
    L = {"model_A": A, "model_B": B}
    surv, p = model_confidence_set(L, seed=0)
    res = fragility(L)
    assert surv == ["model_A"]
    assert res["k_star"] > 0
    assert res["winner_stability"] > 0.95
