"""Round 6 (2026-08-27) -- biostatistician/clinical-trial-design persona review.

Cross-domain stress test: repurposing the loss-panel/MCS/k* abstraction for diagnostic-model
selection across trial sites or enrollment quarters, a genuinely different corner of (K, T) space
than this project's own forecasting-shaped fixtures (small K -- 2-3 candidate algorithms is typical
in clinical comparisons -- crossed with small T -- few sites/quarters, often single-digit).
"""
import numpy as np
import pytest

from selection_fragility.panel import LossPanel
from selection_fragility.report import report
from selection_fragility.fragility import decision_breakdown, pooled_winner
from selection_fragility.identify import identified


def test_clinical_diagnostic_model_panel_end_to_end_no_crash():
    """3 risk-prediction algorithms x 6 enrollment-quarter loss(=1-AUC) values -- the panel/loss
    abstraction should map cleanly onto a non-forecasting domain with zero special-casing needed."""
    algos = {
        "logreg_baseline": np.array([0.28, 0.31, 0.29, 0.33, 0.27, 0.30]),
        "rf_enhanced": np.array([0.24, 0.26, 0.25, 0.29, 0.23, 0.27]),
        "deep_model": np.array([0.26, 0.22, 0.30, 0.20, 0.31, 0.19]),
    }
    panel = LossPanel.from_losses(algos, labels=["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"])
    out = report(panel)
    assert "VERDICT" in out and "LEADERBOARD" in out and "RESOLUTION" in out and "PIVOT" in out
    # deep_model has the lowest mean but the highest per-quarter variance -- MCS should NOT
    # collapse to a confident singleton given that instability (real clinical stakes: don't let a
    # high-variance model's lucky mean crown it without disclosure).
    assert "not identified" in out


@pytest.mark.parametrize("T", [2, 3])
def test_small_K_small_T_clinical_partial_tie_no_crash(T):
    """K=2 (typical head-to-head clinical comparison), T=2 or 3 (few trial sites), with a genuine
    exact tie on some but not all sites -- must not crash, and the RESOLUTION line must disclose
    the underpowered read rather than let 'VERDICT: identified' stand alone as if it settled
    anything (a real clinical-audience risk: MCS 'identified' reads like a significance claim to
    someone used to p<0.05 conventions, but it is a point-estimate-selection statement, not a
    power statement -- the tool already separates these; this test locks in that the RESOLUTION
    disclosure actually fires for a realistic small-sample clinical panel, not just large-T ones)."""
    L = {"algo_A": np.array([0.20, 0.25, 0.25])[:T], "algo_B": np.array([0.22, 0.25, 0.25])[:T]}
    panel = LossPanel.from_losses(L, labels=[f"Site{i}" for i in range(T)])
    out = report(panel)
    assert "cannot determine at this sample size" in out, (
        "a small-T clinical panel with a thin observed edge must carry the underpowered "
        "disclosure alongside any 'identified' verdict, not report identification bare"
    )


def test_fully_tied_small_clinical_panel_gives_zero_kstar_and_no_edge():
    """K=2, T=3, every site EXACTLY tied -- a real scenario when two algorithms agree on every
    case in a small held-out cohort. Must report k*=0 and an honest 'no edge exists', not spurious
    fragility language or a crash on the degenerate all-tied case."""
    L = {"algo_A": np.array([0.25, 0.25, 0.25]), "algo_B": np.array([0.25, 0.25, 0.25])}
    panel = LossPanel.from_losses(L, labels=["SiteA", "SiteB", "SiteC"])
    out = report(panel)
    assert "not identified" in out
    assert "no edge exists" in out
    assert "k*=0" in out


def test_pooled_winner_deterministic_on_exact_tie_regardless_of_dict_order():
    """K=2 exact pooled tie -- pooled_winner must give the same answer regardless of which model
    is inserted into the dict first (name-order tie-break, not insertion-order), independent of
    the separate, NOT-yet-fixed weight-rescaling instability documented in test_property_based.py."""
    a = {"algo_A": np.array([0.25, 0.25, 0.25]), "algo_B": np.array([0.25, 0.25, 0.25])}
    b = {"algo_B": np.array([0.25, 0.25, 0.25]), "algo_A": np.array([0.25, 0.25, 0.25])}
    assert pooled_winner(a) == pooled_winner(b) == "algo_A"  # sorted-name tie-break, documented
