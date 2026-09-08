"""Round 8 -- marketing-analytics/retail-promotions domain deep-dive.

Retail/marketing demand-forecasting data is heavily confounded by promotional activity: a
model's error during a promo week (price cuts, campaigns, holiday events) reflects both
genuine forecasting skill AND how well it happens to handle promo-driven demand spikes, a
qualitatively different signal from baseline demand. This file locks in a realistic,
reproducible example where the POOLED (all-weeks) comparison names a different winner than
either the promo-only or baseline-only view -- and demonstrates that none of the tool's
native diagnostics (concentration_share, PIVOT's "responsible periods") surface this
confounding on their own. This is not a bug: it's a real, disclosed capability boundary an
analyst in this field needs to know about before trusting a pooled `report()` at face value.
"""
import numpy as np
import pytest

from selection_fragility.panel import LossPanel
from selection_fragility.fragility import pooled_winner, decision_breakdown
from selection_fragility.pivot import concentration_share
from selection_fragility.report import report


def _promo_confounded_panel(T=130, promo_frac=0.18, seed=42):
    """2.5 years weekly, 4 candidate demand-forecast models. model_A dominates baseline
    weeks but is weak during promos; model_B is the reverse (built for promo uplift)."""
    rng = np.random.default_rng(seed)
    is_promo = np.zeros(T, dtype=bool)
    is_promo[rng.choice(T, size=int(promo_frac * T), replace=False)] = True
    L = {
        "model_A": np.where(is_promo, rng.normal(0.55, 0.15, T), rng.normal(0.10, 0.04, T)),
        "model_B": np.where(is_promo, rng.normal(0.15, 0.05, T), rng.normal(0.28, 0.08, T)),
        "model_C": np.where(is_promo, rng.normal(0.30, 0.10, T), rng.normal(0.20, 0.06, T)),
        "model_D": np.where(is_promo, rng.normal(0.60, 0.15, T), rng.normal(0.35, 0.08, T)),
    }
    for k in L:
        L[k] = np.clip(L[k], 1e-3, None)
    return L, is_promo


def test_pooled_view_names_a_different_winner_than_the_promo_only_view():
    """The key finding: model_A wins pooled and baseline-only, but model_B wins promo-only
    by a huge margin (0.15 vs 0.53 mean loss) -- an analyst who only reads the pooled report
    would pick the wrong model specifically for promotional-campaign planning."""
    L, is_promo = _promo_confounded_panel()
    pooled = pooled_winner(L)
    promo_only = pooled_winner({k: v[is_promo] for k, v in L.items()})
    base_only = pooled_winner({k: v[~is_promo] for k, v in L.items()})
    assert pooled == "model_A"
    assert base_only == "model_A"
    assert promo_only == "model_B"
    assert promo_only != pooled, "confounding scenario collapsed -- promo/pooled should disagree"


def test_pooled_report_looks_maximally_confident_despite_the_confound():
    """report() on the pooled (unsliced) panel gives no hint anything is wrong: identified,
    resolved, low concentration -- this is the real risk, not a crash or an error message."""
    L, is_promo = _promo_confounded_panel()
    labels = [f"2024-W{i:03d}" for i in range(len(is_promo))]
    text = report(LossPanel.from_losses(L, labels=labels))
    assert "VERDICT: identified" in text
    assert "resolved" in text
    assert "model_A" in text and "champion" in text


def test_concentration_share_does_not_flag_the_promo_confound():
    """concentration_share stays low (~1-2%) on the confounded panel -- it measures margin
    concentration in a single period, not regime-dependent performance, so it does NOT
    surface promo-confounding even though a large, decision-relevant effect is hiding
    underneath. Documents a real capability boundary, not a bug."""
    L, _ = _promo_confounded_panel()
    cs = concentration_share(L)
    assert cs < 0.05, (
        f"concentration_share={cs:.3f} unexpectedly high -- if this ever rises, the promo "
        "confound may have started leaking into the diagnostic and this test's premise "
        "(that concentration_share is NOT a confound detector) should be re-examined")


def test_pivot_responsible_periods_do_not_cluster_on_promo_weeks():
    """k*'s greedy removal targets the CHAMPION's own best periods to check if the runner-up
    could overtake it. Since model_A dominates on baseline weeks (the majority, ~82% of the
    panel), the responsible periods are drawn from baseline weeks, not the minority promo
    weeks that actually explain the promo-only reversal -- PIVOT's own output gives an
    attentive reader no clue to go looking for a promo-specific effect."""
    L, is_promo = _promo_confounded_panel()
    _, _, removed = decision_breakdown(L)
    promo_in_removed = sum(1 for i in removed if is_promo[i])
    assert promo_in_removed == 0, (
        f"{promo_in_removed}/{len(removed)} responsible periods were promo weeks -- if this "
        "becomes nonzero, PIVOT may now be surfacing regime-confounding after a code change; "
        "worth re-examining whether that's an improvement or a coincidence of this fixture")


def test_weighting_lets_an_analyst_who_already_suspects_confounding_investigate_it():
    """The tool won't proactively flag the confound, but an analyst who already suspects one
    (e.g. domain knowledge: 'promos behave differently') CAN investigate correctly via `w=`:
    downweighting promo weeks preserves the baseline-quality read; upweighting them recovers
    the promo-specialist winner. This is a real, working escape hatch -- just not automatic."""
    L, is_promo = _promo_confounded_panel()
    T = len(is_promo)
    downweighted = pooled_winner(L, w=np.where(is_promo, 0.05, 1.0))
    upweighted = pooled_winner(L, w=np.where(is_promo, 5.0, 1.0))
    assert downweighted == "model_A"
    assert upweighted == "model_B"


def test_clean_unconfounded_panel_behaves_normally_as_a_contrast():
    """No promotional activity, one consistently-best model, large margins throughout --
    confirms the tool's ordinary behavior is unaffected; the confounding risk above is a
    property of promo-structured data, not a general defect."""
    rng = np.random.default_rng(42)
    T = 130
    L = {
        "model_A": np.clip(rng.normal(0.15, 0.05, T), 1e-3, None),
        "model_B": np.clip(rng.normal(0.25, 0.05, T), 1e-3, None),
        "model_C": np.clip(rng.normal(0.30, 0.05, T), 1e-3, None),
    }
    winner = pooled_winner(L)
    k, _, removed = decision_breakdown(L)
    assert winner == "model_A"
    assert k > T * 0.5, f"k*={k} of T={T} unexpectedly low for a clean, decisive panel"
