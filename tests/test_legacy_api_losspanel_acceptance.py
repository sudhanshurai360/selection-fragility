"""Legacy-API LossPanel acceptance, FIXED 2026-09-07 (round-4 8-lens PyPI-preflight audit).

THE GAP: README.md's "Legacy API" section lists seven functions in one sentence -- fragility,
decision_breakdown, breakdown_number, winner_stability, exchangeable_benchmark, pooled_winner,
per_period_winner, condorcet_winner/condorcet_status, mcs/model_confidence_set -- as one family of
"original, lower-level diagnostics". An independent review (fixing identify.py's identified()/
mcs_size(), which sit in the separate "Staged v1.0 API" table) found that within THIS Legacy API
sentence itself, support for LossPanel input was inconsistent: pooled_winner/decision_breakdown/
winner_stability already accepted a LossPanel directly (via the shared `_unwrap_panel` helper), but
per_period_winner, condorcet_winner, condorcet_status (fragility.py), and model_confidence_set
(mcs.py) did not -- each raised a raw, confusing internal TypeError instead. `mcs()` itself (the raw
T x K matrix engine model_confidence_set wraps) is a genuinely different input contract and is
intentionally excluded from this fix -- it was never documented as accepting a LossPanel.

These tests lock in that all four now accept a LossPanel and agree with the equivalent dict input.
"""
import numpy as np
import pytest

from selection_fragility import (
    LossPanel,
    per_period_winner,
    condorcet_winner,
    condorcet_status,
    model_confidence_set,
    surprise_concentration,
)


@pytest.fixture
def panel_and_dict():
    rng = np.random.default_rng(11)
    data = {
        "a": rng.normal(1.0, 0.2, 18).tolist(),
        "b": rng.normal(1.1, 0.2, 18).tolist(),
        "c": rng.normal(2.0, 0.2, 18).tolist(),
    }
    return LossPanel.from_losses(data), data


def test_per_period_winner_accepts_losspanel(panel_and_dict):
    panel, data = panel_and_dict
    assert per_period_winner(panel) == per_period_winner(data)


def test_per_period_winner_return_ties_matches_too(panel_and_dict):
    panel, data = panel_and_dict
    assert per_period_winner(panel, return_ties=True) == per_period_winner(data, return_ties=True)


def test_condorcet_winner_accepts_losspanel(panel_and_dict):
    panel, data = panel_and_dict
    assert condorcet_winner(panel) == condorcet_winner(data)


def test_condorcet_status_accepts_losspanel(panel_and_dict):
    panel, data = panel_and_dict
    assert condorcet_status(panel) == condorcet_status(data)


def test_model_confidence_set_accepts_losspanel(panel_and_dict):
    panel, data = panel_and_dict
    surv_panel, p_panel = model_confidence_set(panel)
    surv_dict, p_dict = model_confidence_set(data)
    assert surv_panel == surv_dict
    assert p_panel == p_dict


def test_readme_legacy_api_sentence_functions_all_accept_losspanel_now(panel_and_dict):
    """The specific inconsistency the audit found: pooled_winner/decision_breakdown/winner_stability
    already worked; the other four in the SAME README sentence didn't. This is the completeness
    check -- every function in that sentence must now behave the same way on LossPanel input."""
    from selection_fragility import pooled_winner, decision_breakdown, winner_stability

    panel, data = panel_and_dict
    w = np.ones(len(data["a"]))
    assert pooled_winner(panel) == pooled_winner(data)
    assert decision_breakdown(data, w, pooled_winner(data))[0] == \
           decision_breakdown(panel, w, pooled_winner(panel))[0]
    # winner_stability is stochastic (bootstrap) but seeded -- must match exactly with the same seed
    assert winner_stability(panel, w, n_boot=200, seed=0) == winner_stability(data, w, n_boot=200, seed=0)
    assert per_period_winner(panel) == per_period_winner(data)
    assert condorcet_winner(panel) == condorcet_winner(data)
    assert condorcet_status(panel) == condorcet_status(data)
    assert model_confidence_set(panel)[0] == model_confidence_set(data)[0]


def test_surprise_concentration_accepts_losspanel(panel_and_dict):
    """FOUND AS A FOLLOW-UP (round-4 audit's own reviewer): surprise_concentration() sits in the
    same file, takes the identical {model: array} shape, and was missed by the first pass of this
    fix -- same bug class, same file, immediately after the three now-fixed sibling functions."""
    panel, data = panel_and_dict
    assert surprise_concentration(panel) == surprise_concentration(data)


def test_mcs_raw_matrix_engine_unaffected_by_this_fix():
    """mcs() (the raw T x K matrix engine model_confidence_set wraps) was never meant to accept a
    LossPanel -- confirm it still requires a matrix and this fix didn't change that contract."""
    from selection_fragility import mcs

    rng = np.random.default_rng(12)
    M = rng.normal(1.0, 0.2, (18, 3))
    surv, p = mcs(M)   # must still work on a raw matrix, unchanged
    assert len(surv) >= 1
