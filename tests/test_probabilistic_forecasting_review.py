"""Round-6 review: probabilistic forecasting / uncertainty-quantification researcher persona.

Locks in behavior verified during a round-6 review: pinball-loss-scale point loss data works
end to end through report(); an averaged-across-quantiles CRPS-style aggregation (the standard way
a probabilistic forecaster reduces a multi-quantile evaluation to one scalar per period) composes
cleanly with the existing API; winner_stability behaves sensibly (lower stability) on fat-tailed
(lognormal) loss data versus a same-mean Gaussian panel, rather than silently mis-calibrating on
non-Gaussian-shaped errors.
"""
import numpy as np
import pytest

from selection_fragility.panel import LossPanel
from selection_fragility.report import report
from selection_fragility.fragility import winner_stability


def _pinball_loss(y, q_pred, tau):
    e = y - q_pred
    return np.maximum(tau * e, (tau - 1) * e)


def test_pinball_loss_scale_data_runs_end_to_end_through_report():
    rng = np.random.default_rng(0)
    T = 36
    y = rng.normal(100, 10, T)
    tau = 0.1
    models_q = {
        "model_a": y - rng.normal(2, 5, T),
        "model_b": y - rng.normal(0, 3, T),
        "model_c": y - rng.normal(5, 8, T),
    }
    L = {m: _pinball_loss(y, q, tau) for m, q in models_q.items()}
    assert all((v >= 0).all() for v in L.values()), "pinball loss must be non-negative by construction"
    panel = LossPanel.from_losses(L)
    out = report(panel)
    assert "VERDICT" in out and "LEADERBOARD" in out


def test_averaged_multiquantile_pinball_aggregation_composes_cleanly():
    """The standard probabilistic-forecasting reduction (mean pinball loss across several quantile
    levels, approximating CRPS) is just another per-period scalar -- should need zero special
    handling."""
    rng = np.random.default_rng(1)
    T = 30
    y = rng.normal(100, 10, T)
    taus = [0.1, 0.5, 0.9]
    models_q = {
        "model_a": {tau: y - rng.normal(0, 5, T) for tau in taus},
        "model_b": {tau: y - rng.normal(1, 4, T) for tau in taus},
    }
    L_avg = {
        m: np.mean([_pinball_loss(y, models_q[m][tau], tau) for tau in taus], axis=0)
        for m in models_q
    }
    panel = LossPanel.from_losses(L_avg)
    out = report(panel)
    assert "VERDICT" in out


def test_winner_stability_lower_on_fat_tailed_than_gaussian_same_mean():
    """winner_stability should read as LESS stable on fat-tailed (lognormal, right-skewed --
    the empirical shape of pinball-loss panels with occasional large quantile misses) loss data
    than on a Gaussian panel sharing the identical per-model means. A tool whose calibration were
    silently tuned only to point-forecast-shaped (roughly symmetric) errors could plausibly get
    this backwards or fail to discriminate; verified directly it does not."""
    rng = np.random.default_rng(3)
    T = 40
    L_fat = {
        "a": rng.lognormal(0.0, 1.2, T),
        "b": rng.lognormal(0.05, 1.2, T),
        "c": rng.lognormal(0.1, 1.3, T),
    }
    means = {k: v.mean() for k, v in L_fat.items()}
    rng2 = np.random.default_rng(3)
    L_gauss = {k: np.abs(rng2.normal(means[k], means[k] * 0.3, T)) for k in L_fat}

    ws_fat, _ = winner_stability(L_fat, n_boot=1000, seed=0)
    ws_gauss, _ = winner_stability(L_gauss, n_boot=1000, seed=0)
    assert ws_fat <= ws_gauss, (
        f"expected fat-tailed panel to be at least as unstable as a same-mean Gaussian panel, "
        f"got fat={ws_fat} > gauss={ws_gauss}"
    )


def test_raw_array_entry_points_now_accept_losspanel_directly():
    """FIXED 2026-08-27 (round-6 probabilistic-forecasting-researcher finding, closed same round):
    decision_breakdown/pooled_winner/winner_stability/fragility() are documented dict-only
    ('raw-array tier', distinct from report()/compare()/resolution_report()'s LossPanel-accepting
    'primary' tier by design -- NOT itself the bug). Passing a LossPanel to any of them used to
    give a confusing INTERNAL error instead of a clear top-level guard -- the same error-quality
    class already fixed for resolution_report() in round 3. Rather than add a guard error, all
    four were fixed to accept a LossPanel directly (via the new `_unwrap_panel` helper), matching
    resolution_report()'s own fix pattern -- so the correct behavior now is that all four just
    work, identically to passing the panel's own .losses/.weights by hand."""
    from selection_fragility.fragility import decision_breakdown, pooled_winner, winner_stability, fragility

    rng = np.random.default_rng(2)
    L = {"a": rng.normal(1, 0.3, 20), "b": rng.normal(1.2, 0.3, 20)}
    panel = LossPanel.from_losses(L)

    assert pooled_winner(panel) == pooled_winner(L)
    assert decision_breakdown(panel)[:2] == decision_breakdown(L)[:2]
    ws_panel, freq_panel = winner_stability(panel, n_boot=50, seed=0)
    ws_dict, freq_dict = winner_stability(L, n_boot=50, seed=0)
    assert ws_panel == ws_dict and freq_panel == freq_dict
    frag_panel = fragility(panel, n_boot=50, seed=0)
    frag_dict = fragility(L, n_boot=50, seed=0)
    assert frag_panel["k_star"] == frag_dict["k_star"]
    assert frag_panel["pooled_winner"] == frag_dict["pooled_winner"]
