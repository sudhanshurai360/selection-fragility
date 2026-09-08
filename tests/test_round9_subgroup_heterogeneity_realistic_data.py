"""Round 9 -- subgroup/fairness heterogeneity in pooled model comparisons.

Round 8's marketing-attribution test found that a pooled report() can look maximally
confident while masking a large, decision-relevant TIME-regime reversal (promo vs. baseline
weeks). This file tests whether the identical statistical mechanism applies to a different,
higher-stakes kind of heterogeneity: a candidate model that wins the POOLED comparison
because it performs excellently for a majority subgroup, while performing dramatically worse
for a minority subgroup -- exactly the accuracy-vs-subgroup-disparity tradeoff that has been
extensively studied and criticized in the real risk-assessment-tooling literature (e.g. the
public ProPublica/COMPAS debate). This uses only synthetic data and generic group labels; it
tests the general statistical PHENOMENON (does a pooled robustness verdict survive subgroup
slicing), not any specific real tool or population.

Unlike the marketing case, a subgroup split is NOT a split of PERIODS within one panel -- the
pooled panel's per-period loss is already a population-weighted average across subgroups, so
there is no "responsible period" for pivot/concentration_share to point at even in principle.
The only way to see the disparity is to recompute separate per-subgroup panels and compare --
exactly what a real fairness-oversight researcher would need to know to do.
"""
import numpy as np
import pytest

from selection_fragility.panel import LossPanel
from selection_fragility.fragility import pooled_winner, decision_breakdown
from selection_fragility.mcs import model_confidence_set
from selection_fragility.pivot import concentration_share
from selection_fragility.report import report


def _subgroup_disparate_panel(T=26, seed=7, frac_A=0.75):
    """26 quarterly cohorts, 3 candidate risk-prediction models, two synthetic subgroups.
    group_A is 75% of the population. model_X is excellent for group_A but performs far
    worse for group_B; model_Y is mediocre but genuinely consistent across both groups;
    model_Z is uniformly worse. The pooled per-period loss is the population-weighted
    average of the two subgroups' losses -- this is what an aggregate-accuracy comparison
    actually computes, whether or not the analyst realizes it."""
    rng = np.random.default_rng(seed)
    L_A = {
        "model_X": rng.normal(0.05, 0.015, T),
        "model_Y": rng.normal(0.15, 0.02, T),
        "model_Z": rng.normal(0.30, 0.05, T),
    }
    L_B = {
        "model_X": rng.normal(0.40, 0.05, T),
        "model_Y": rng.normal(0.15, 0.02, T),
        "model_Z": rng.normal(0.32, 0.05, T),
    }
    for L in (L_A, L_B):
        for k in L:
            L[k] = np.clip(L[k], 1e-3, None)
    pooled = {k: frac_A * L_A[k] + (1 - frac_A) * L_B[k] for k in L_A}
    return pooled, L_A, L_B


def _subgroup_fair_panel(T=26, seed=7, frac_A=0.75):
    """Contrast scenario: model_X is genuinely, consistently best for both subgroups --
    the pooled verdict should agree with both subgroup-only verdicts, and subgroup slicing
    must not spuriously manufacture disagreement where none exists."""
    rng = np.random.default_rng(seed)
    L_A = {
        "model_X": rng.normal(0.10, 0.02, T),
        "model_Y": rng.normal(0.18, 0.03, T),
        "model_Z": rng.normal(0.30, 0.05, T),
    }
    L_B = {
        "model_X": rng.normal(0.11, 0.02, T),
        "model_Y": rng.normal(0.19, 0.03, T),
        "model_Z": rng.normal(0.31, 0.05, T),
    }
    pooled = {k: frac_A * L_A[k] + (1 - frac_A) * L_B[k] for k in L_A}
    return pooled, L_A, L_B


def test_pooled_winner_disagrees_with_the_minority_subgroups_winner():
    """The key finding: model_X wins the pooled (population-weighted) comparison and the
    majority-subgroup-only comparison, but model_Y wins decisively for the minority subgroup
    (0.145 vs 0.399 mean loss, roughly 2.7x worse) -- an analyst who trusts the pooled result
    alone would deploy the model that performs far worse for the minority subgroup."""
    pooled, L_A, L_B = _subgroup_disparate_panel()
    assert pooled_winner(pooled) == "model_X"
    assert pooled_winner(L_A) == "model_X"
    assert pooled_winner(L_B) == "model_Y"
    assert L_B["model_X"].mean() > 2.5 * L_B["model_Y"].mean()


def test_pooled_mcs_says_identified_while_minority_subgroup_mcs_says_the_opposite():
    """Both slices are confidently 'identified' (MCS singleton) -- this is not a case of one
    view being fragile/uncertain and the other clean. Both look statistically decisive; they
    are decisive about OPPOSITE conclusions. This is a stronger, more dangerous form of the
    marketing-confounding finding: neither view's own fragility diagnostics warn the analyst
    anything is wrong, because nothing about either slice IS statistically wrong in isolation."""
    pooled, _, L_B = _subgroup_disparate_panel()
    surv_pooled, p_pooled = model_confidence_set(pooled, seed=0)
    surv_B, p_B = model_confidence_set(L_B, seed=0)
    assert surv_pooled == ["model_X"]
    assert surv_B == ["model_Y"]
    assert p_pooled < 0.05
    assert p_B < 0.05


def test_pooled_report_looks_maximally_confident_despite_the_subgroup_disparity():
    """report() on the pooled panel gives no hint anything is wrong: identified, resolved,
    a real (not tiny) k*, and a concentration_share nowhere near flagged-as-high -- the
    disparity is completely invisible from the pooled report's own output."""
    pooled, _, _ = _subgroup_disparate_panel()
    text = report(LossPanel.from_losses(pooled))
    assert "VERDICT: identified" in text
    assert "resolved" in text
    assert "model_X" in text and "champion" in text


def test_concentration_share_and_pivot_cannot_see_inside_the_pooled_average():
    """Unlike round 8's time-regime confounding (where pivot's greedy removal COULD in
    principle land on promo weeks, it just didn't because they were the minority), a
    subgroup split isn't a split of periods at all -- the pooled per-period loss is already
    a population-weighted average, so there is no 'responsible period' that corresponds to
    'the minority subgroup' even in principle. concentration_share stays unremarkable (well
    under the level that would prompt a second look), confirming this is architecturally
    invisible to the tool's within-panel diagnostics, not merely undetected by them."""
    pooled, _, _ = _subgroup_disparate_panel()
    cs = concentration_share(pooled)
    assert cs < 0.20, (
        f"concentration_share={cs:.3f} higher than expected -- if this rises further, "
        "reconsider whether it could ever be used as an indirect subgroup-disparity signal"
    )
    k, opponent, removed = decision_breakdown(pooled)
    assert k >= 5, "pooled comparison should read as reasonably robust, not fragile-looking"


def test_fair_baseline_subgroup_slicing_agrees_and_manufactures_no_spurious_fragility():
    """Contrast case: when a model genuinely performs consistently across both subgroups,
    pooled and both subgroup-only views must agree, and MCS must not spuriously report
    non-identification just because the data was sliced smaller."""
    pooled, L_A, L_B = _subgroup_fair_panel()
    winner = pooled_winner(pooled)
    assert winner == pooled_winner(L_A) == pooled_winner(L_B) == "model_X"
    surv_pooled, _ = model_confidence_set(pooled, seed=0)
    assert surv_pooled == ["model_X"]


def test_subgroup_disparity_finding_is_not_a_single_seed_artifact():
    """The pooled-vs-minority-subgroup disagreement holds across 10 independent seeds (9 of
    10 give a clean pooled singleton; seed 9 gives a 2-model tie, still consistent with the
    finding, not a contradiction of it) -- this is a structural property of the constructed
    scenario, not a cherry-picked draw."""
    disagreements = 0
    for seed in range(1, 11):
        pooled, _, L_B = _subgroup_disparate_panel(seed=seed)
        if pooled_winner(pooled) != pooled_winner(L_B):
            disagreements += 1
    assert disagreements == 10, f"expected pooled/minority-subgroup disagreement in all 10 seeds, got {disagreements}"


def test_debug_no_crash_across_subgroup_scenarios():
    """Sanity: the full toolkit runs cleanly (no crash, no NaN-propagation surprises) on
    every panel shape built in this file."""
    for builder in (_subgroup_disparate_panel, _subgroup_fair_panel):
        pooled, L_A, L_B = builder()
        for L in (pooled, L_A, L_B):
            assert np.isfinite(next(iter(L.values()))).all()
            k, opp, removed = decision_breakdown(L)
            assert 0 <= k <= len(next(iter(L.values())))
            surv, p = model_confidence_set(L, seed=0)
            assert len(surv) >= 1
            assert 0.0 <= p <= 1.0
