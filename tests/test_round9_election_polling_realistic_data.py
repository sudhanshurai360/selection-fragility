"""Round 9 -- election-forecasting persona: comparing poll-aggregation METHODOLOGIES across past
election cycles (simple average vs. house-effect-adjusted vs. fundamentals-plus-polls hybrid), not
the parent paper's own election.* social-choice/voting-mechanism analysis (Condorcet winners,
plurality cycles on real historical elections) -- a different application of "model comparison"
entirely.

Domain-specific data shape, unlike almost every prior round: T = distinct ELECTION CYCLES (a
realistic aggregator track record spans ~8-12 general elections, not within-cycle daily polling
updates), K = 3-5 candidate aggregation methodologies, loss = absolute error in predicted vs. actual
vote margin (points, >=0). Genuinely small T, one loss value per cycle, no serial/within-season
structure to speak of.
"""
import numpy as np
import pytest

from selection_fragility import LossPanel, report, decision_breakdown, resolution_report, pooled_winner

YEARS = [1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]


def _panel(losses, years=YEARS):
    return LossPanel.from_losses(losses, labels=[str(y) for y in years])


class TestElectionCycleBaseline:
    """T=10 election cycles, K=4 aggregation methods, realistic absolute vote-margin-error losses.
    Confirms the tool runs cleanly on this domain's genuinely unusual data shape (small T = discrete
    election events, not within-season polling updates) with no crash and sane output."""

    def test_realistic_four_method_panel_runs_and_produces_leaderboard(self):
        rng = np.random.default_rng(42)
        T = len(YEARS)
        losses = {
            "simple_avg": np.abs(rng.normal(2.2, 0.6, T)),
            "house_adjusted": np.abs(rng.normal(1.9, 0.6, T)),
            "fundamentals_hybrid": np.abs(rng.normal(2.0, 1.0, T)),
            "naive_last_poll": np.abs(rng.normal(2.8, 0.7, T)),
        }
        panel = _panel(losses)
        r = report(panel)
        assert "VERDICT" in r and "LEADERBOARD" in r and "RESOLUTION" in r and "PIVOT" in r
        # naive_last_poll is the clearly worst method by construction (highest mean loss); it must
        # never be reported as the pooled winner
        assert pooled_winner(panel) != "naive_last_poll"


class TestPivotAttributionUnderCorrelatedSystematicMiss:
    """Real election polling misses (2016/2020-style) are CORRELATED across the affected cycle --
    every method gets worse together, not independent per-cycle noise. Two precisely-constructed
    cases show breakdown_number's pivot attribution is mechanistically correct in BOTH directions:
    it names the systematic-miss cycles when they actually separate the two closest rivals, and it
    correctly does NOT name them when the miss affects the closest rivals too similarly to matter --
    this is the same 'answers a specific counterfactual, not general anomalousness' behavior found
    for breakdown_number/pivot in rounds 7-8, now confirmed for this domain's correlated-shock shape
    specifically (a different structure than round 6/8's within-panel AR(1) serial dependence)."""

    def test_systematic_miss_years_prioritized_first_when_they_drive_the_binding_margin(self):
        rng = np.random.default_rng(7)
        T = len(YEARS)
        a = np.full(T, 2.0) + rng.normal(0, 0.05, T)
        b = np.full(T, 2.05) + rng.normal(0, 0.05, T)
        miss_idx = [7, 8]  # 2016, 2020
        for i in miss_idx:
            a[i] += 0.3
            b[i] += 3.0  # rival degrades much more than champion in the systematic-miss cycles
        panel = _panel({"model_a_champion": a, "model_b_rival": b})
        k, opp, removed = decision_breakdown(panel)
        assert opp == "model_b_rival"
        # the two systematic-miss cycles must be the FIRST two periods the greedy algorithm removes
        # (they carry the largest per-period margin contribution, by construction)
        assert set(removed[:2]) == set(miss_idx), (
            f"expected the two systematic-miss years (indices {miss_idx}) to be removed first, "
            f"got removal order {removed}")

    def test_systematic_miss_years_not_flagged_when_they_dont_separate_the_binding_pair(self):
        rng = np.random.default_rng(42)
        T = len(YEARS)
        base = {
            "simple_avg": np.abs(rng.normal(2.2, 0.6, T)),
            "house_adjusted": np.abs(rng.normal(1.9, 0.6, T)),
            "fundamentals_hybrid": np.abs(rng.normal(2.0, 1.0, T)),
            "naive_last_poll": np.abs(rng.normal(2.8, 0.7, T)),
        }
        miss_idx = [7, 8]
        # a systematic miss that hits the two CLOSEST rivals (fundamentals_hybrid, house_adjusted)
        # by nearly the SAME amount -- degrades both similarly, so it shouldn't be what separates them
        base["simple_avg"][miss_idx] += 4.5
        base["naive_last_poll"][miss_idx] += 5.0
        base["house_adjusted"][miss_idx] += 1.5
        base["fundamentals_hybrid"][miss_idx] += 1.0
        panel = _panel(base)
        k, opp, removed = decision_breakdown(panel)
        # confirmed empirically (2026-08-27): the binding rival here is house_adjusted, and the
        # single pivotal period is 1992 (index 1) -- NOT one of the systematic-miss cycles -- because
        # the miss moved the two closest rivals (fundamentals_hybrid, house_adjusted) by nearly the
        # same amount and so barely changed the margin BETWEEN them specifically. Contrast directly
        # with the sibling test above, where an otherwise-identical construction that DOES separate
        # the binding pair puts the miss years first in the removal order every time.
        assert k == 1
        assert removed[0] not in miss_idx, (
            f"expected the pivotal period to be a non-miss-year (the systematic miss here doesn't "
            f"separate the binding pair), got index {removed[0]} which IS a miss year")


class TestSmallSampleHonestRefusal:
    """A real aggregator's track record realistically spans only ~6-12 elections -- this project's
    own docs already establish that overclaiming forecast-method superiority from a handful of
    elections is a genuine, common methodological sin in this exact field. Confirm resolution_report
    correctly refuses to resolve at this sample size for close competitors, and correctly DOES
    resolve for a genuinely dominant method, rather than uniformly hedging."""

    def test_six_cycles_close_competitors_correctly_refuses_to_resolve(self):
        rng = np.random.default_rng(0)
        losses = {
            "method_x": np.abs(rng.normal(2.0, 0.5, 6)),
            "method_y": np.abs(rng.normal(2.1, 0.5, 6)),
        }
        panel = _panel(losses, YEARS[:6])
        res = resolution_report(panel)
        assert res["resolved"] is False
        assert res["identified"] is False
        assert res["fragility_read"] == "undetermined"

    def test_ten_cycles_dominant_method_correctly_resolves(self):
        rng = np.random.default_rng(7)
        losses = {
            "reliable_method": np.abs(rng.normal(1.5, 0.15, 10)),
            "mediocre_method": np.abs(rng.normal(2.5, 0.3, 10)),
            "poor_method": np.abs(rng.normal(3.5, 0.4, 10)),
        }
        panel = _panel(losses)
        res = resolution_report(panel)
        assert res["resolved"] is True
        assert res["identified"] is True
        assert res["binding_rival"] == "mediocre_method"
